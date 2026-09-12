"""Configuration builders for the GPU (CIFAR-10 / FID) experiments.

Each builder returns a list of :class:`mlrx.runner.Config`.  Keeping them
declarative and separate from execution means the full cost of a sweep can be
priced before a single image is generated -- see :func:`estimate_cost` -- which
matters when the budget is a fixed Kaggle session.

Budget discipline
-----------------
``n_images`` defaults to 10,000 rather than the paper's 50,000.  That is a
deliberate, documented trade: FID is a biased estimator whose bias grows as the
sample count falls, so our absolute numbers read high relative to published
ones.  They remain internally comparable because every configuration uses the
same count and the same latents.  :func:`build_sample_size_study` measures the
bias so the offset can be quantified rather than assumed.
"""

from __future__ import annotations

from ..runner import Config

__all__ = [
    "build_validity_sweep",
    "build_main_panel",
    "build_multilevel_sweep",
    "build_reuse_sweep",
    "build_seed_blocks",
    "build_all",
    "estimate_cost",
]

DEFAULT_NFE_GRID = (6, 8, 10, 12, 16, 20)
GATE_NFE = 10


def build_validity_sweep(nfe_grid=DEFAULT_NFE_GRID, n_images=10_000):
    """The paper's Figure 2: RX-Euler at several ``k`` against Euler and naive.

    Every method is run at matched *NFE*, not matched step count, since RX is
    free and Heun is not; comparing at equal steps would flatter RX.
    """
    cfgs = []
    for nfe in nfe_grid:
        cfgs.append(Config(method="euler", num_steps=nfe,
                           n_images=n_images, tag="validity"))
        for k in (2, 3, 4):
            cfgs.append(Config(method="rx", num_steps=nfe, frequency=k,
                               n_levels=2, n_images=n_images, tag="validity"))
        cfgs.append(Config(method="rx", num_steps=nfe, frequency=2, n_levels=2,
                           coefficients="naive", n_images=n_images,
                           tag="validity"))
    return cfgs


def build_main_panel(nfe_grid=DEFAULT_NFE_GRID, n_images=10_000):
    """The paper's Figure 3 panel: Euler, Heun (EDM), RX-Euler, RX+EDM.

    Heun's step count is chosen so its NFE matches the grid: ``2N - 1 = nfe``.
    Odd NFE values are therefore the ones Heun can hit exactly, which is why
    the paper's Heun points sit at NFE 9, 11, ... rather than 10, 12.
    """
    cfgs = []
    for nfe in nfe_grid:
        cfgs.append(Config(method="euler", num_steps=nfe,
                           n_images=n_images, tag="panel"))
        cfgs.append(Config(method="rx", num_steps=nfe, frequency=2, n_levels=2,
                           n_images=n_images, tag="panel"))
        n_heun = (nfe + 1) // 2
        cfgs.append(Config(method="heun", num_steps=n_heun,
                           n_images=n_images, tag="panel"))
        cfgs.append(Config(method="rx_edm", num_steps=nfe, frequency=2,
                           n_levels=2, heun_fraction=0.5,
                           n_images=n_images, tag="panel"))
    return cfgs


def build_multilevel_sweep(level_counts=(2, 3, 4), nfe_grid=(8, 10, 16),
                           precisions=("float64", "float32", "float16"),
                           n_images=10_000):
    """The extension: level count against working precision.

    ``frequency`` is pinned to ``2^(L-1)``, the narrowest block supporting ``L``
    nested levels.  Only the extrapolation arithmetic varies with ``precision``;
    the ODE state and the network stay in their defaults so that any measured
    effect is attributable to the weight solve and combination alone.
    """
    cfgs = []
    for nfe in nfe_grid:
        for L in level_counts:
            K = 2 ** (L - 1)
            if K > nfe:
                continue
            for prec in precisions:
                cfgs.append(Config(
                    method="rx", num_steps=nfe, frequency=K, n_levels=L,
                    work_dtype=prec, n_images=n_images, tag="multilevel",
                ))
    return cfgs


def build_reuse_sweep(level_counts=(2, 3, 4), nfe_grid=(10, 16),
                      n_images=10_000):
    """Reuse-approximate versus exact level recomputation.

    On the analytic problem this comparison is decisive: reuse caps the local
    order near 3.2 for every ``L``, while exact recomputation reaches the
    predicted ``p + L - 1``.  This sweep asks whether that gap survives contact
    with a real network and a perceptual metric, where the extra evaluations
    exact mode spends might have bought a better sample simply by being extra
    evaluations.  Note the NFE differs between arms by construction -- these
    points must be plotted against NFE, never against step count.
    """
    cfgs = []
    for nfe in nfe_grid:
        for L in level_counts:
            K = 2 ** (L - 1)
            if K > nfe:
                continue
            for mode in ("denoised", "exact"):
                cfgs.append(Config(
                    method="rx", num_steps=nfe, frequency=K, n_levels=L,
                    reuse_mode=mode, n_images=n_images, tag="reuse",
                ))
    return cfgs


def build_seed_blocks(n_blocks=3, nfe=GATE_NFE, n_images=10_000):
    """Independent seed blocks at the validation gate, for error bars.

    A single FID is a point estimate with no stated uncertainty, which makes
    small differences impossible to interpret.  Repeating over disjoint seed
    blocks gives a spread; combined with common random numbers across methods
    within a block, it is what licenses claims about gaps of less than a point.
    """
    cfgs = []
    for b in range(n_blocks):
        cfgs.append(Config(method="euler", num_steps=nfe, seed_offset=b,
                           n_images=n_images, tag="seedblock"))
        cfgs.append(Config(method="rx", num_steps=nfe, frequency=2, n_levels=2,
                           seed_offset=b, n_images=n_images, tag="seedblock"))
        cfgs.append(Config(method="heun", num_steps=(nfe + 1) // 2,
                           seed_offset=b, n_images=n_images, tag="seedblock"))
        cfgs.append(Config(method="rx_edm", num_steps=nfe, frequency=2,
                           n_levels=2, seed_offset=b, n_images=n_images,
                           tag="seedblock"))
    return cfgs


def build_all(n_images=10_000, nfe_grid=DEFAULT_NFE_GRID, include_reuse=True):
    """Every GPU configuration, de-duplicated by key."""
    cfgs = []
    cfgs += build_validity_sweep(nfe_grid, n_images)
    cfgs += build_main_panel(nfe_grid, n_images)
    cfgs += build_multilevel_sweep(n_images=n_images)
    if include_reuse:
        cfgs += build_reuse_sweep(n_images=n_images)
    cfgs += build_seed_blocks(n_images=n_images)

    seen, out = set(), []
    for c in cfgs:
        if c.key not in seen:
            seen.add(c.key)
            out.append(c)
    return out


def estimate_cost(configs, throughput_img_nfe_per_s=157.0, n_gpus=2,
                  fid_overhead_s=60.0):
    """Price a sweep before running it.

    ``throughput_img_nfe_per_s`` defaults to a T4 estimate derived by scaling
    the paper's own timing table (their Table 6: A6000, batch 128, 10 steps =
    1.737 s, i.e. ~737 image-NFE/s) by the roughly 4.7x fp32 gap between an
    A6000 and a T4.  Override it with a measured value once one run has
    completed -- :func:`mlrx.runner.run_config` records ``wall_seconds`` for
    exactly this purpose.
    """
    units = sum(c.expected_nfe * c.n_images for c in configs)
    gen_s = units / max(throughput_img_nfe_per_s, 1e-9)
    total_s = gen_s + fid_overhead_s * len(configs)
    return {
        "n_configs": len(configs),
        "image_nfe": units,
        "gpu_seconds": total_s,
        "gpu_hours": total_s / 3600.0,
        "wall_hours": total_s / 3600.0 / max(n_gpus, 1),
    }
