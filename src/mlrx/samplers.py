"""ODE samplers: Euler, Heun, RX-DPM, and the multi-level extension.

Every sampler here is written against a *denoiser callable*

    denoiser(x, sigma) -> D(x; sigma)

and uses nothing but arithmetic on ``x``, so the same code runs unmodified on
numpy arrays (the analytic toy problem of :mod:`mlrx.toy`) and on torch tensors
(the real EDM network).  That is deliberate: the order-verification results and
the image results are then produced by literally the same sampler, so a bug
cannot hide in the gap between a "test" implementation and a "real" one.

NFE accounting
--------------
``SamplerResult.nfe`` is incremented on every denoiser call and is the honest
cost of the sample.  For the record:

* Euler over ``N`` steps costs ``N``.
* Heun costs ``2N - 1`` (EDM skips the correction on the final step).
* **RX with reuse costs ``N`` at any number of levels** -- the coarse
  estimates reuse evaluations the fine trajectory already made.  Preserving
  this is the whole point of the reuse-approximate scheme; see
  :func:`rx_sampler`.
* RX with ``reuse_mode="exact"`` costs ``N + sum_n (n - 1)`` per block over the
  coarse levels, and is the honest-but-not-free comparison point.

The reuse approximation
-----------------------
At two levels there is no approximation at all.  The coarse estimate is a
single step spanning the block, so the only evaluation it needs is at the
block's start -- where the coarse and fine trajectories coincide exactly.  This
is why RX-DPM is genuinely free.

At three or more levels an intermediate level needs a denoiser value at a node
the fine trajectory also visits, but *at a different state*: after the first
sub-step the two trajectories have separated.  Recomputing would cost extra
NFE and forfeit the method's selling point, so we reuse the fine trajectory's
value and recompute only the drift from the coarse level's own state:

    d = (x_coarse - D(x_fine, t)) / t          rather than  d = (x_fine - D(x_fine, t)) / t

This keeps the dominant, linear-in-``x`` part of the drift exact and
approximates only the slowly varying denoiser output.  It is the same trade the
paper itself makes for RX-Runge-Kutta, where ``z_{i-delta'}`` is approximated by
``z_{i-1}`` or ``z_{i-1-delta}`` (Sec. 4.3).  ``reuse_mode="derivative"``
selects the cruder alternative of reusing the drift wholesale, for ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from . import extrapolation as extrap
from . import schedules

__all__ = [
    "SamplerResult",
    "euler_sampler",
    "heun_sampler",
    "rx_sampler",
    "rx_edm_sampler",
    "SAMPLERS",
]


@dataclass
class SamplerResult:
    """Sample plus the bookkeeping needed to report and audit it."""

    x: object
    nfe: int
    weights: list = field(default_factory=list, repr=False)
    """One :class:`~mlrx.extrapolation.Weights` per extrapolating block, kept so
    conditioning can be reported per run rather than recomputed."""

    @property
    def max_cond(self):
        return max((w.cond for w in self.weights), default=float("nan"))

    @property
    def max_amplification(self):
        return max((w.amplification for w in self.weights), default=float("nan"))

    @property
    def mean_amplification(self):
        vals = [w.amplification for w in self.weights]
        return float(np.mean(vals)) if vals else float("nan")


class _Counter:
    """Wraps a denoiser so NFE is counted where it is actually spent."""

    __slots__ = ("fn", "n")

    def __init__(self, fn):
        self.fn = fn
        self.n = 0

    def __call__(self, x, sigma):
        self.n += 1
        return self.fn(x, sigma)


def _cast(x, dtype):
    """Cast an array/tensor to a numpy dtype, for numpy and torch alike."""
    if dtype is None:
        return x
    if hasattr(x, "to") and hasattr(x, "dtype") and not isinstance(x, np.ndarray):
        import torch                                    # local: torch optional
        return x.to(getattr(torch, np.dtype(dtype).name))
    return np.asarray(x, dtype=dtype)


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def euler_sampler(denoiser, x_init, t_steps, **_):
    """First-order Euler on the probability-flow ODE.  ``NFE = N``."""
    d = _Counter(denoiser)
    x = x_init
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        x = x + (t_next - t_cur) * ((x - d(x, t_cur)) / t_cur)
    return SamplerResult(x=x, nfe=d.n)


def heun_sampler(denoiser, x_init, t_steps, **_):
    """EDM's second-order Heun corrector (Karras et al. Alg. 2).  ``NFE = 2N-1``.

    The correction is skipped on the final step because ``t_next = 0`` there and
    the corrector's ``(x - D)/t_next`` would divide by zero -- EDM's own
    convention, reproduced here so NFE and quality both match the baseline.
    """
    d = _Counter(denoiser)
    x = x_init
    n = len(t_steps) - 1
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        d_cur = (x - d(x, t_cur)) / t_cur
        x_euler = x + (t_next - t_cur) * d_cur
        if i < n - 1:
            d_next = (x_euler - d(x_euler, t_next)) / t_next
            x = x + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_next)
        else:
            x = x_euler
    return SamplerResult(x=x, nfe=d.n)


# ---------------------------------------------------------------------------
# RX-DPM and the multi-level extension
# ---------------------------------------------------------------------------

def rx_sampler(
    denoiser,
    x_init,
    t_steps,
    frequency=2,
    n_levels=2,
    p=2,
    skip_last=True,
    coefficients="grid_aware",
    reuse_mode="denoised",
    work_dtype=None,
    collect_weights=True,
    **_,
):
    """RX-DPM with an arbitrary number of nested extrapolation levels.

    ``n_levels=2`` reproduces the published method exactly (and, with
    ``coefficients="grid_aware"``, reproduces the reference implementation bit
    for bit).  ``n_levels>2`` is this project's extension.

    Parameters
    ----------
    denoiser : callable
    x_init : array or tensor
        Initial state at ``t_steps[0]``, i.e. ``latents * t_steps[0]``.
    t_steps : sequence
        ``N + 1`` descending noise levels ending at zero.
    frequency : int
        Block length ``K``.  Must be a power of two to support more than two
        levels nestedly.
    n_levels : int
        Number of levels ``L``.  Cancels ``L - 1`` error terms.
    p : int
        Local order of the base solver; ``2`` for Euler.
    skip_last : bool
        Remainder handling -- see :func:`mlrx.schedules.partition_blocks`.
    coefficients : {"grid_aware", "naive"}
        ``"grid_aware"`` solves the moment system on the actual step widths
        (the paper's contribution).  ``"naive"`` uses the fixed uniform-grid
        coefficients of classical Richardson extrapolation (their Fig. 2
        ablation); only meaningful at two levels.
    reuse_mode : {"denoised", "derivative"}
        How coarse levels reuse fine-trajectory evaluations; see module
        docstring.  Irrelevant at two levels, where nothing is approximated.
    work_dtype : numpy dtype or None
        Precision for the weight solve *and* the extrapolation combination.
        ``None`` leaves both at the state's native precision.  This is the knob
        the round-off study turns.

    Returns
    -------
    SamplerResult
    """
    if n_levels < 2:
        raise ValueError("n_levels must be at least 2; use euler_sampler for 1")
    if coefficients not in ("grid_aware", "naive"):
        raise ValueError(f"unknown coefficients scheme {coefficients!r}")
    if reuse_mode not in ("denoised", "derivative", "exact"):
        raise ValueError(f"unknown reuse_mode {reuse_mode!r}")
    if coefficients == "naive" and n_levels != 2:
        raise ValueError("the naive uniform-grid baseline is defined for two levels")

    d = _Counter(denoiser)
    num_steps = len(t_steps) - 1
    plan = schedules.partition_blocks(num_steps, frequency, skip_last=skip_last)
    weights_log = []
    x = x_init

    for block in plan:
        t_block = [t_steps[i] for i in range(block.start, block.stop + 1)]

        # A block shorter than the level structure cannot support the scheme.
        do_extrap = block.extrapolate and block.n_steps >= 2
        if do_extrap and n_levels > 2:
            try:
                level_steps = schedules.nested_steps(block.n_steps, n_levels)
            except ValueError:
                do_extrap = False
        else:
            level_steps = [1, block.n_steps] if do_extrap else None

        if not do_extrap:
            for t_cur, t_next in zip(t_block[:-1], t_block[1:]):
                x = x + (t_next - t_cur) * ((x - d(x, t_cur)) / t_cur)
            continue

        x_block = x

        # -- finest level: the only one that spends evaluations -------------
        # Both the denoiser outputs and the resulting drifts are stored, since
        # the two reuse modes need different ones.
        fine = x_block
        denoised_at, drift_at = {}, {}
        for j in range(block.n_steps):
            t_cur, t_next = t_block[j], t_block[j + 1]
            dn = d(fine, t_cur)
            denoised_at[j] = dn
            drift_at[j] = (fine - dn) / t_cur
            fine = fine + (t_next - t_cur) * drift_at[j]

        # -- coarser levels ------------------------------------------------
        # In the two reuse modes these cost nothing.  In "exact" mode each
        # level integrates its own trajectory honestly, which costs extra
        # evaluations everywhere except the block's first node, where the
        # coarse and fine states still coincide.
        estimates = []
        for n in level_steps[:-1]:
            g = block.n_steps // n
            xc = x_block
            for m in range(n):
                j = m * g
                t_cur, t_next = t_block[j], t_block[j + g]
                if reuse_mode == "exact":
                    dn = denoised_at[j] if m == 0 else d(xc, t_cur)
                    drift = (xc - dn) / t_cur
                elif reuse_mode == "derivative":
                    drift = drift_at[j]
                else:
                    drift = (xc - denoised_at[j]) / t_cur
                xc = xc + (t_next - t_cur) * drift
            estimates.append(xc)
        estimates.append(fine)

        # -- combine --------------------------------------------------------
        base_lambdas = schedules.block_lambdas(t_block)
        wdt = work_dtype if work_dtype is not None else np.float64
        if coefficients == "naive":
            W = extrap.naive_richardson_weights(block.n_steps, p=p, dtype=wdt)
        else:
            lv = extrap.nested_level_lambdas(base_lambdas, level_steps)
            W = extrap.extrapolation_weights(lv, p=p, dtype=wdt)
        if collect_weights:
            weights_log.append(W)

        if work_dtype is not None:
            estimates = [_cast(e, work_dtype) for e in estimates]
        x = W.apply(estimates)
        if work_dtype is not None:
            x = _cast(x, _native_dtype(x_block))

    return SamplerResult(x=x, nfe=d.n, weights=weights_log)


def _native_dtype(x):
    if isinstance(x, np.ndarray):
        return x.dtype
    return np.dtype(str(x.dtype).replace("torch.", ""))


def rx_edm_sampler(
    denoiser,
    x_init,
    t_steps,
    frequency=2,
    n_levels=2,
    n_heun_steps=None,
    heun_fraction=0.5,
    **kwargs,
):
    """The RX+EDM hybrid of the paper's Figure 3.

    RX-Euler is strongest at low NFE and Heun catches up at higher NFE, which
    the authors read as the two methods suiting different parts of the
    trajectory: interpolation (Heun) is safer early, where predictions are
    close to noise and less accurate, while extrapolation (RX) pays off later.
    They therefore run Heun on the early, high-noise steps and RX-Euler on the
    remaining low-noise steps.

    ``n_heun_steps`` sets the split explicitly; otherwise ``heun_fraction`` of
    the steps go to Heun.  Note the NFE is *not* ``N``: Heun steps cost two
    evaluations each, so comparisons must be made against NFE, not step count.
    """
    num_steps = len(t_steps) - 1
    if n_heun_steps is None:
        n_heun_steps = int(round(heun_fraction * num_steps))
    n_heun_steps = max(0, min(num_steps, n_heun_steps))

    d = _Counter(denoiser)
    x = x_init

    for i in range(n_heun_steps):
        t_cur, t_next = t_steps[i], t_steps[i + 1]
        d_cur = (x - d(x, t_cur)) / t_cur
        x_euler = x + (t_next - t_cur) * d_cur
        if t_next != 0:
            d_next = (x_euler - d(x_euler, t_next)) / t_next
            x = x + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_next)
        else:
            x = x_euler

    tail = list(t_steps[n_heun_steps:])
    if len(tail) >= 2:
        res = rx_sampler(
            d, x, tail, frequency=frequency, n_levels=n_levels, **kwargs
        )
        x = res.x
        return SamplerResult(x=x, nfe=d.n, weights=res.weights)
    return SamplerResult(x=x, nfe=d.n)


SAMPLERS = {
    "euler": euler_sampler,
    "heun": heun_sampler,
    "rx": rx_sampler,
    "rx_edm": rx_edm_sampler,
}
