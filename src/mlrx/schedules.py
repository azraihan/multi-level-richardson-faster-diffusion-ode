"""EDM time-step schedules and extrapolation block structure.

The schedule is the source of the non-uniformity that makes this project
interesting.  EDM (Karras et al., NeurIPS 2022, Eq. 5) places its noise levels
at

    sigma_i = ( sigma_max^(1/rho)
                + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)) )^rho

for ``i = 0 .. N-1``, followed by ``sigma_N = 0``.  The exponent ``rho``
controls how sharply the steps bunch towards the clean end: ``rho = 1`` is
uniform in sigma, larger ``rho`` shortens the steps near ``sigma_min`` and
stretches those near ``sigma_max``.  Karras et al. settle on ``rho = 7``
empirically, noting that ``rho ~ 3`` would roughly equalise truncation error
per step but that concentrating resolution at low noise gives better samples.

That choice was tuned for Euler and Heun.  Our extrapolation weights are
computed *from* the step widths, so ``rho`` also sets the conditioning of the
weight system -- which is why :mod:`mlrx.experiments.rho_search` treats it as a
free parameter and searches over it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "edm_schedule",
    "block_lambdas",
    "partition_blocks",
    "Block",
    "BlockPlan",
]


def edm_schedule(num_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0,
                 dtype=np.float64):
    """EDM noise schedule, returning ``num_steps + 1`` levels ending at zero.

    Matches ``edm_sampler`` in the reference implementation exactly, including
    the trailing ``sigma = 0``.  Note the returned array has length
    ``num_steps + 1`` while the number of solver *steps* is ``num_steps``.
    """
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    i = np.arange(num_steps, dtype=np.float64)
    a, b = sigma_max ** (1.0 / rho), sigma_min ** (1.0 / rho)
    t = (a + i / (num_steps - 1) * (b - a)) ** rho
    return np.concatenate([t, [0.0]]).astype(dtype)


def block_lambdas(t_block, dtype=np.float64):
    """Normalised sub-step widths of one extrapolation block.

    ``t_block`` holds the ``K + 1`` schedule values bounding the block's ``K``
    sub-steps, in descending (denoising) order.  Returns ``K`` positive values
    summing to one.
    """
    # float() per element rather than np.asarray: the block may hold 0-d torch
    # tensors on a GPU, which numpy cannot convert directly.
    t = np.array([float(v) for v in t_block], dtype=np.float64)
    if t.size < 2:
        raise ValueError("a block needs at least two time points")
    widths = t[:-1] - t[1:]
    total = t[0] - t[-1]
    if total <= 0:
        raise ValueError("block must have positive width and descend in time")
    if np.any(widths <= 0):
        raise ValueError("schedule must be strictly decreasing within a block")
    return (widths / total).astype(dtype)


@dataclass(frozen=True)
class Block:
    """One extrapolation block: a contiguous run of base-grid steps."""

    start: int
    """Index into the schedule where the block begins."""

    n_steps: int
    """Number of base-grid steps the block spans."""

    extrapolate: bool
    """Whether extrapolation is applied at the end of this block.  The final
    block is run plainly when it is too short to support the scheme."""

    @property
    def stop(self):
        return self.start + self.n_steps


@dataclass(frozen=True)
class BlockPlan:
    """How a schedule of ``N`` steps is carved into extrapolation blocks."""

    blocks: tuple
    num_steps: int
    frequency: int
    skip_last: bool

    @property
    def nfe(self):
        """Network evaluations for an Euler base solver.

        Every base-grid step costs one evaluation and the coarse estimates are
        reused rather than recomputed, so this is simply ``num_steps`` -- the
        property that makes the method free.
        """
        return self.num_steps

    def __iter__(self):
        return iter(self.blocks)

    def __len__(self):
        return len(self.blocks)


def partition_blocks(num_steps, frequency, skip_last=True):
    """Carve ``num_steps`` steps into blocks of ``frequency``, handling remainder.

    This reproduces the remainder handling of the reference implementation,
    which is subtle enough to be worth stating plainly.  When ``frequency``
    does not divide ``num_steps`` there is a short block left over, and there
    are two ways to deal with it:

    ``skip_last=True``
        Keep the short block as its own block but do not extrapolate over it
        (a short block would otherwise be extrapolated with a different, and
        much worse conditioned, width ratio than every other block).  This is
        the reference default.

    ``skip_last=False``
        Absorb the remainder into the final full block, making one longer
        block and extrapolating over all of it.

    Parameters
    ----------
    num_steps : int
    frequency : int
        Block length ``K``.  ``frequency = 1`` disables extrapolation entirely.
    skip_last : bool

    Returns
    -------
    BlockPlan
    """
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if frequency < 1:
        raise ValueError("frequency must be positive")

    if frequency == 1:
        blocks = (Block(i, 1, False) for i in range(num_steps))
        return BlockPlan(tuple(blocks), num_steps, frequency, skip_last)

    full, rem = divmod(num_steps, frequency)
    blocks = []
    pos = 0

    if rem == 0:
        for _ in range(full):
            blocks.append(Block(pos, frequency, True))
            pos += frequency
    elif skip_last:
        for _ in range(full):
            blocks.append(Block(pos, frequency, True))
            pos += frequency
        blocks.append(Block(pos, rem, False))   # trailing short block, plain
        pos += rem
    else:
        for _ in range(full - 1):
            blocks.append(Block(pos, frequency, True))
            pos += frequency
        blocks.append(Block(pos, frequency + rem, True))  # absorb remainder
        pos += frequency + rem

    assert pos == num_steps, (pos, num_steps)
    return BlockPlan(tuple(blocks), num_steps, frequency, skip_last)


def nested_steps(frequency, n_levels):
    """Step counts for ``n_levels`` nested levels spanning a block of ``K``.

    Returns ascending counts beginning at 1 and ending at ``frequency``, each
    dividing the next.  Nestedness is what keeps the coarse estimates free, so
    a requested level count that cannot be realised nestedly is rejected rather
    than silently approximated.

    ``nested_steps(8, 4) -> [1, 2, 4, 8]``; ``nested_steps(4, 3) -> [1, 2, 4]``.
    """
    if n_levels < 2:
        raise ValueError("extrapolation needs at least two levels")
    if frequency < 2:
        raise ValueError("frequency must be at least 2 to extrapolate")

    # Powers of two are the natural nested chain; fall back to any divisor
    # chain of the right length when frequency is not a power of two.
    chain = [1]
    while chain[-1] * 2 < frequency:
        chain.append(chain[-1] * 2)
    chain.append(frequency)
    chain = sorted(set(chain))

    if len(chain) < n_levels:
        raise ValueError(
            f"frequency={frequency} admits at most {len(chain)} nested levels, "
            f"{n_levels} requested (use a power-of-two frequency such as "
            f"{2 ** (n_levels - 1)})"
        )

    # Keep the coarsest, the finest, and spread the rest between them.
    if len(chain) == n_levels:
        return chain
    idx = np.unique(np.round(np.linspace(0, len(chain) - 1, n_levels)).astype(int))
    while idx.size < n_levels:                      # pragma: no cover - safety
        missing = set(range(len(chain))) - set(idx.tolist())
        idx = np.sort(np.append(idx, sorted(missing)[0]))
    return [chain[i] for i in idx]
