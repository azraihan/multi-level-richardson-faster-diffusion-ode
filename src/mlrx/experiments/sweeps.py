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

# The NFE values we test across all sweeps (Number of Function Evaluations,
# i.e. how many times the neural network is called per image).
DEFAULT_NFE_GRID = (6, 8, 10, 12, 16, 20)

# NFE used for the single "gate" comparison (the headline number in the paper).
GATE_NFE = 10


def rx_edm_config(nfe, **kw):
    """An RX+EDM configuration whose cost is exactly ``nfe``.

    Heun steps cost two evaluations and RX steps one, so ``N`` total steps of
    which ``h`` are Heun cost ``N + h``.  Giving Heun the first third of the
    budget (``h = round(nfe / 3)``, ``N = nfe - h``) hits the target exactly and
    leaves RX the low-noise tail, which is where the paper found the hybrid
    works best.  Specifying ``heun_fraction`` instead would silently overspend:
    half of ten steps as Heun costs fifteen evaluations, not ten.
    """
    # Assign the first third of the budget to Heun (2nd-order, higher quality
    # early in the trajectory) and the rest to RX-Euler (cheap extrapolation
    # that works well at low noise).
    h = int(round(nfe / 3))
    cfg = Config(method="rx_edm", num_steps=nfe - h, n_heun_steps=h,
                 frequency=2, n_levels=2, **kw)
    # Sanity check: the Config must agree that its own NFE matches what we asked.
    assert cfg.expected_nfe == nfe, (nfe, cfg.expected_nfe)
    return cfg


def heun_steps_for(nfe):
    """Heun step count whose cost ``2N - 1`` is closest to ``nfe`` from above.

    Heun can only realise odd NFE.  Rounding up matches the paper's own
    comparison, which sets Heun at NFE 11 against RX-Euler at NFE 10 (Fig. 6).
    """
    # Heun costs 2N-1 evaluations. Given a target NFE, solve for N (rounded up).
    return nfe // 2 + 1


def build_validity_sweep(nfe_grid=DEFAULT_NFE_GRID, n_images=10_000):
    """The paper's Figure 2: RX-Euler at several ``k`` against Euler and naive.

    Every method is run at matched *NFE*, not matched step count, since RX is
    free and Heun is not; comparing at equal steps would flatter RX.
    """
    cfgs = []
    for nfe in nfe_grid:
        # Plain Euler: the cheapest baseline, order 1.
        cfgs.append(Config(method="euler", num_steps=nfe,
                           n_images=n_images, tag="validity"))

        # RX-Euler with different block sizes k=2,3,4 — tests whether the
        # frequency (how many fine steps per block) matters.
        for k in (2, 3, 4):
            cfgs.append(Config(method="rx", num_steps=nfe, frequency=k,
                               n_levels=2, n_images=n_images, tag="validity"))

        # Naïve Richardson: uses fixed uniform-grid coefficients instead of the
        # grid-aware ones. This is the ablation that shows WHY the paper's
        # non-uniform correction matters.
        cfgs.append(Config(method="rx", num_steps=nfe, frequency=2, n_levels=2,
                           coefficients="naive", n_images=n_images,
                           tag="validity"))
    return cfgs


def build_main_panel(nfe_grid=DEFAULT_NFE_GRID, n_images=10_000):
    """The paper's Figure 3 panel: Euler, Heun (EDM), RX-Euler, RX+EDM.

    Every arm is built to cost the grid NFE.  Heun can only realise odd NFE, so
    its points sit one above the even grid values -- the same convention as the
    paper's Fig. 6, which compares Heun at NFE 11 with RX-Euler at NFE 10.
    """
    cfgs = []
    for nfe in nfe_grid:
        # Euler: 1st-order baseline. Simple but cheapest.
        cfgs.append(Config(method="euler", num_steps=nfe,
                           n_images=n_images, tag="panel"))

        # RX-Euler (k=2): the paper's main proposed method.
        cfgs.append(Config(method="rx", num_steps=nfe, frequency=2, n_levels=2,
                           n_images=n_images, tag="panel"))

        # Heun (EDM): 2nd-order corrector, costs one extra eval per step.
        # Step count is chosen so the total NFE just meets or exceeds the target.
        cfgs.append(Config(method="heun", num_steps=heun_steps_for(nfe),
                           n_images=n_images, tag="panel"))

        # RX+EDM: hybrid that uses Heun for the first third of steps and
        # RX-Euler for the remainder.
        cfgs.append(rx_edm_config(nfe, n_images=n_images, tag="panel"))
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
            # The minimum block size that supports L nested levels is 2^(L-1).
            # For example: L=2 → K=2, L=3 → K=4, L=4 → K=8.
            K = 2 ** (L - 1)

            # Skip combinations where the block is larger than the total budget.
            if K > nfe:
                continue

            for prec in precisions:
                # work_dtype controls the precision of the weight solve and the
                # final weighted combination — the ODE state stays in float64.
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
                # "denoised" reuses the neural network output from the fine pass
                # — free but approximate at L≥3.
                # "exact" re-evaluates the network for each coarse level
                # — honest but costs extra NFE.
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
        # Each block b uses a different starting seed, so the 10k latents are
        # completely independent from block to block → real statistical spread.
        # Within a block, all methods share the same latents → fair comparison.
        cfgs.append(Config(method="euler", num_steps=nfe, seed_offset=b,
                           n_images=n_images, tag="seedblock"))
        cfgs.append(Config(method="rx", num_steps=nfe, frequency=2, n_levels=2,
                           seed_offset=b, n_images=n_images, tag="seedblock"))
        cfgs.append(Config(method="heun", num_steps=heun_steps_for(nfe),
                           seed_offset=b, n_images=n_images, tag="seedblock"))
        cfgs.append(rx_edm_config(nfe, seed_offset=b, n_images=n_images,
                                  tag="seedblock"))
    return cfgs


def build_all(n_images=10_000, nfe_grid=DEFAULT_NFE_GRID, include_reuse=True):
    """Every GPU configuration, de-duplicated by key."""
    cfgs = []
    # Collect configs from every individual sweep.
    cfgs += build_validity_sweep(nfe_grid, n_images)
    cfgs += build_main_panel(nfe_grid, n_images)
    cfgs += build_multilevel_sweep(n_images=n_images)
    if include_reuse:
        cfgs += build_reuse_sweep(n_images=n_images)
    cfgs += build_seed_blocks(n_images=n_images)

    # Deduplicate: the same physical config (e.g. Euler at NFE 10) appears in
    # multiple sweeps but should only be run once. The key ignores the "tag"
    # field so these overlaps are detected correctly.
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
    # Total work = sum over all configs of (NFE per image × number of images).
    # This is the same unit the throughput estimate is expressed in.
    units = sum(c.expected_nfe * c.n_images for c in configs)

    # Time to generate all images, ignoring FID computation overhead.
    gen_s = units / max(throughput_img_nfe_per_s, 1e-9)

    # Add a fixed per-config overhead for computing the FID score itself
    # (Inception feature extraction over 10k images takes ~1 minute).
    total_s = gen_s + fid_overhead_s * len(configs)

    return {
        "n_configs": len(configs),
        "image_nfe": units,          # total network calls across all images
        "gpu_seconds": total_s,      # total GPU-seconds if running on 1 GPU
        "gpu_hours": total_s / 3600.0,
        "wall_hours": total_s / 3600.0 / max(n_gpus, 1),  # actual clock time
    }
