"""Empirical order of accuracy on the analytic diffusion ODE.

This is the experiment that decides whether the extension's error model is
sound.  :mod:`mlrx.extrapolation` *posits* that an ``L``-level combination
cancels ``L-1`` error terms and therefore converges at order ``p + L - 1``.
That claim rests on a linear error-accumulation assumption which the original
paper adopts for two levels and explicitly flags as not holding in general.
Here we simply measure the achieved order and report it.

Because the toy problem's exact solution is known, "error" here is the real
thing -- a distance to ground truth -- not a perceptual proxy like FID.  A
result showing the predicted order *failing* to materialise is as valuable as
one showing it holding, and is reported either way.
"""

from __future__ import annotations

import numpy as np

from .. import regression, samplers, schedules, toy

__all__ = ["run_order_study", "run_vcurve_study", "run_reuse_ablation",
           "measure_error"]

SIGMA_MIN, SIGMA_MAX = 0.002, 80.0


def _ground_truth(problem, x_init, sigma_max, sigma_min, cache={}):
    key = (problem.name, float(sigma_max), float(sigma_min),
           x_init.shape, float(x_init.sum()))
    if key not in cache:
        cache[key] = problem.ground_truth(x_init, sigma_max, sigma_min)
    return cache[key]


def measure_error(problem, sampler_fn, x_init, t_steps, truth, **kw):
    """RMS error against ground truth, plus realised NFE."""
    res = sampler_fn(problem.denoise, x_init, t_steps, **kw)
    x = np.asarray(res.x, dtype=np.float64)
    err = float(np.sqrt(np.mean((x - truth) ** 2)))
    return err, res


def run_order_study(
    problems=None,
    step_counts=None,
    level_counts=(2, 3, 4),
    frequency=8,
    rho=7.0,
    n_samples=64,
    seed=0,
    dtype=np.float64,
    asymptotic_from=None,
):
    """Global error versus step count, and the order fitted from it.

    Methodological choices, each of which materially affects the answer:

    *Step counts are multiples of* ``frequency``.  Otherwise the remainder
    handling changes the block structure from one ``N`` to the next, and the
    resulting sawtooth in the error swamps the trend being fitted.

    *The order is fitted against* ``N``, *not against a step width*.  The
    paper states its convergence results globally -- ``O(N^-1)`` for Euler,
    ``O(N^-2)`` for RX-Euler (their Eqs. 28-29) -- and on a non-uniform
    schedule no single ``h`` is a faithful abscissa anyway.  Fitting
    ``log E`` against ``log N`` gives a slope of ``-order`` directly
    comparable to those statements.

    *Only the asymptotic tail is fitted.*  Extrapolation is a statement about
    the leading term of a Taylor expansion and means nothing until ``h`` is
    small enough for that term to dominate.  At small ``N`` with a wide block
    the method is not merely inaccurate but actively divergent, and including
    those points would misreport the order.  ``asymptotic_from`` sets the
    cutoff (default: the upper half of the sweep); the discarded points are
    retained in the returned rows and are themselves a reported result.

    ``frequency`` is held fixed across ``level_counts`` so that block geometry
    is identical and the only thing varying is the number of levels.  Note that
    ``L`` levels need ``K >= 2^(L-1)``, so studying more levels forces a wider
    block -- a coupling explored in
    :func:`mlrx.experiments.conditioning.run_block_width_study`.
    """
    if problems is None:
        problems = [toy.single_gaussian(dim=1, std=0.5), toy.bimodal(4.0, 0.5, 1)]
    if step_counts is None:
        step_counts = tuple(frequency * m for m in (1, 2, 3, 4, 6, 8, 12, 16))

    bad = [N for N in step_counts if N % frequency]
    if bad:
        raise ValueError(
            f"step_counts must be multiples of frequency={frequency}; "
            f"offenders: {bad}"
        )
    if asymptotic_from is None:
        asymptotic_from = sorted(step_counts)[len(step_counts) // 2]

    rng = np.random.default_rng(seed)
    rows = []

    for problem in problems:
        x0 = rng.normal(size=(n_samples, problem.dim)) * SIGMA_MAX
        truth = _ground_truth(problem, x0, SIGMA_MAX, SIGMA_MIN)

        for N in step_counts:
            t = schedules.edm_schedule(N, SIGMA_MIN, SIGMA_MAX, rho, dtype=dtype)

            variants = [
                ("euler", samplers.euler_sampler, {}),
                ("heun", samplers.heun_sampler, {}),
            ]
            for L in level_counts:
                try:
                    schedules.nested_steps(frequency, L)
                except ValueError:
                    continue
                variants.append((
                    f"rx_L{L}", samplers.rx_sampler,
                    dict(frequency=frequency, n_levels=L, p=2,
                         work_dtype=dtype, collect_weights=True),
                ))
            variants.append((
                "rx_naive", samplers.rx_sampler,
                dict(frequency=2, n_levels=2, p=2, coefficients="naive",
                     work_dtype=dtype),
            ))

            for name, fn, kw in variants:
                problem.reset_nfe()
                try:
                    err, res = measure_error(problem, fn, x0, t, truth, **kw)
                except Exception as exc:
                    print(f"  ! {problem.name} N={N} {name}: {exc!r}")
                    continue
                rows.append({
                    "problem": problem.name,
                    "is_linear": problem.is_linear,
                    "method": name,
                    "num_steps": N,
                    "nfe": res.nfe,
                    "n_blocks": N // frequency if name.startswith("rx_L") else "",
                    "frequency": frequency if name.startswith("rx_L") else "",
                    "h_max": float(np.max(t[:-1] - t[1:])),
                    "rms_error": err,
                    "max_cond": res.max_cond,
                    "mean_amplification": res.mean_amplification,
                    "rho": rho,
                    "dtype": np.dtype(dtype).name,
                    "asymptotic": N >= asymptotic_from,
                })

    order_rows = []
    for problem in sorted({r["problem"] for r in rows}):
        for method in sorted({r["method"] for r in rows if r["problem"] == problem}):
            sub = [r for r in rows
                   if r["problem"] == problem and r["method"] == method
                   and r["asymptotic"]]
            sub.sort(key=lambda r: r["num_steps"])
            if len(sub) < 3:
                continue
            # slope of log E vs log N is -order
            slope, fit = regression.fit_order([1.0 / r["num_steps"] for r in sub],
                                              [r["rms_error"] for r in sub])
            order_rows.append({
                "problem": problem,
                "method": method,
                "fitted_order": slope,
                "r2": fit.r2 if fit else float("nan"),
                "n_points": fit.n if fit else 0,
                "fit_from_N": asymptotic_from,
                "predicted_order": _predicted_order(method),
            })
    return rows, order_rows


def _predicted_order(method):
    """Order the error model predicts, for comparison with what is measured."""
    if method == "euler":
        return 1.0            # global order; local is 2
    if method == "heun":
        return 2.0
    if method == "rx_naive":
        return 2.0
    if method.startswith("rx_L"):
        return float(int(method[4:]))     # L levels -> global order L
    return float("nan")


def run_vcurve_study(
    problem=None,
    step_counts=(16, 32, 64, 128, 256, 512, 1024, 2048),
    level_counts=(2, 3),
    precisions=("float64", "float32", "float16"),
    rho=7.0,
    n_samples=64,
    seed=0,
):
    """Total error against step count, per working precision -- the V-curve.

    Swept along ``N`` rather than along the level count, because that is where
    the classical trade-off actually lives.  Refining the grid drives
    truncation error down; round-off, amplified by the extrapolation weights,
    does not shrink, so past some ``N`` the total error stops improving and
    turns back up.  This is the textbook picture (Chapra & Canale, Ch. 3) and
    it appears here in float64 at ``L = 2`` around ``N ~ 256``.

    Two things worth stating plainly, because both were measured rather than
    assumed:

    * **Sweeping the level count instead does not produce a V.**  Error rises
      monotonically with ``L`` in every precision tested.  That is a direct
      consequence of the reuse ceiling documented in
      :func:`run_reuse_ablation`: without extra evaluations the extra levels
      buy no order, so all they contribute is conditioning cost.
    * **Reduced precision shows up as a floor, not as a shifted minimum.**
      float32 tracks float64 almost exactly, while float16 saturates near
      ``3e-3`` and no amount of refinement gets below it.

    The smooth single-Gaussian problem is the default because its truncation
    error falls far enough for round-off to become visible at all; on the
    nonlinear problem, model error dominates everywhere and the floor is never
    reached.
    """
    if problem is None:
        problem = toy.single_gaussian(dim=1, mean=0.0, std=0.5)

    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(n_samples, problem.dim)) * SIGMA_MAX
    truth = _ground_truth(problem, x0, SIGMA_MAX, SIGMA_MIN)

    rows = []
    for L in level_counts:
        K = 2 ** (L - 1)
        for N in step_counts:
            if N % K:
                continue
            t = schedules.edm_schedule(N, SIGMA_MIN, SIGMA_MAX, rho)
            for prec in precisions:
                dt = np.dtype(prec)
                try:
                    err, res = measure_error(
                        problem, samplers.rx_sampler, x0, t, truth,
                        frequency=K, n_levels=L, p=2,
                        work_dtype=dt, collect_weights=True,
                    )
                    failed = ""
                except Exception as exc:
                    err, res, failed = float("nan"), None, repr(exc)
                rows.append({
                    "problem": problem.name,
                    "num_steps": N,
                    "n_levels": L,
                    "frequency": K,
                    "precision": prec,
                    "eps": float(np.finfo(dt).eps),
                    "rms_error": err,
                    "nfe": res.nfe if res else 0,
                    "max_cond": res.max_cond if res else float("nan"),
                    "mean_amplification": (res.mean_amplification if res
                                           else float("nan")),
                    "failed": failed,
                })
    return rows


def run_reuse_ablation(
    problems=None,
    level_counts=(2, 3, 4, 5),
    modes=("denoised", "derivative", "exact"),
    block_hi=2.0,
    widths=(0.8, 0.4, 0.2, 0.1, 0.05, 0.025),
    n_samples=256,
    seed=0,
    p=2,
):
    """Local order of a single extrapolation block, by level count and reuse mode.

    The experiment that determines what the multi-level extension is actually
    worth, and the most important measurement in the project.

    A *single* block is integrated from an exact initial condition, so what is
    measured is local truncation error uncontaminated by accumulation across
    blocks, and the block is then shrunk to extract an order.  Levels are
    nested with ``K = 2^(L-1)``.

    Measured result, consistent across both the linear and the nonlinear
    problem: ``"denoised"`` and ``"exact"`` agree *exactly* at ``L = 2`` --
    there is nothing to approximate there, since the coarse estimate's only
    evaluation is at the block start where both trajectories coincide -- but
    for ``L >= 3`` the reuse modes saturate near order ``3.2`` while exact
    recomputation tracks the predicted ``p + L - 1``.  The free-of-charge
    property of RX-DPM is therefore specific to two levels and does not
    survive generalisation: buying extra orders means buying extra
    evaluations.

    Returns ``(rows, order_rows)``.
    """
    if problems is None:
        problems = [toy.single_gaussian(dim=1, mean=0.0, std=0.5),
                    toy.bimodal(4.0, 0.5, 1)]

    rng = np.random.default_rng(seed)
    rows = []

    for problem in problems:
        x0 = rng.normal(size=(n_samples, problem.dim)) * block_hi

        # Ground truth depends only on the block, not on the sampler settings,
        # so it is computed once per width rather than once per
        # (L, mode, width) -- the nonlinear reference solve is by far the most
        # expensive step here and recomputing it would dominate the runtime.
        truth_for = {}
        for h in widths:
            lo = block_hi - h
            if lo <= 0:
                continue
            truth_for[h] = (
                problem.exact(x0, block_hi, lo) if problem.is_linear
                else problem.reference(x0, block_hi, lo, n_steps=4000)
            )

        for L in level_counts:
            K = 2 ** (L - 1)
            for mode in modes:
                for h in widths:
                    lo = block_hi - h
                    if lo <= 0:
                        continue
                    t = np.geomspace(block_hi, lo, K + 1)
                    problem.reset_nfe()
                    res = samplers.rx_sampler(
                        problem.denoise, x0, t, frequency=K, n_levels=L, p=p,
                        reuse_mode=mode, skip_last=False,
                    )
                    err = float(np.sqrt(np.mean(
                        (np.asarray(res.x) - truth_for[h]) ** 2)))
                    rows.append({
                        "problem": problem.name,
                        "is_linear": problem.is_linear,
                        "n_levels": L,
                        "frequency": K,
                        "reuse_mode": mode,
                        "block_width": h,
                        "sigma_hi": block_hi,
                        "sigma_lo": lo,
                        "rms_error": err,
                        "nfe": res.nfe,
                        "nfe_overhead": res.nfe - K,
                        "max_cond": res.max_cond,
                        "mean_amplification": res.mean_amplification,
                    })

    order_rows = []
    for problem in sorted({r["problem"] for r in rows}):
        for L in sorted({r["n_levels"] for r in rows}):
            for mode in modes:
                sub = [r for r in rows if r["problem"] == problem
                       and r["n_levels"] == L and r["reuse_mode"] == mode]
                if len(sub) < 3:
                    continue
                sub.sort(key=lambda r: r["block_width"])
                order, fit = regression.fit_order(
                    [r["block_width"] for r in sub],
                    [r["rms_error"] for r in sub],
                )
                order_rows.append({
                    "problem": problem,
                    "n_levels": L,
                    "frequency": 2 ** (L - 1),
                    "reuse_mode": mode,
                    "fitted_local_order": order,
                    "predicted_local_order": p + L - 1,
                    "r2": fit.r2 if fit else float("nan"),
                    "mean_nfe_overhead": float(np.mean(
                        [r["nfe_overhead"] for r in sub])),
                })
    return rows, order_rows
