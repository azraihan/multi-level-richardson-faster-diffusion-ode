"""Optimising the schedule exponent ``rho`` by golden-section search.

EDM fixes ``rho = 7`` empirically, tuned for plain Euler and Heun sampling.
Our extrapolation weights are derived from the step widths, so ``rho`` controls
not only where the solver spends its resolution but also how well conditioned
the weight system is.  There is no reason the exponent that suits one sampler
should suit another, which makes this a genuine one-dimensional optimisation
rather than a hyperparameter shrug.

Two variants are provided:

:func:`search_rho_toy`
    Objective is true RMS error on the analytic problem.  Free, exact, and
    runs on a CPU in seconds; use it to establish the shape of the objective
    and to sanity-check the bracket before spending GPU time.
:func:`search_rho_fid`
    Objective is FID on CIFAR-10.  Expensive and noisy.  Each evaluation is
    cached through the result store, so an interrupted search resumes.
"""

from __future__ import annotations

import numpy as np

from .. import optimize, samplers, schedules, toy

__all__ = ["search_rho_toy", "search_rho_fid", "scan_rho_toy"]

SIGMA_MIN, SIGMA_MAX = 0.002, 80.0


def _toy_objective(problem, x0, truth, num_steps, frequency, n_levels,
                   method="rx", reuse_mode="denoised"):
    def objective(rho):
        t = schedules.edm_schedule(num_steps, SIGMA_MIN, SIGMA_MAX, float(rho))
        if method == "euler":
            res = samplers.euler_sampler(problem.denoise, x0, t)
        elif method == "heun":
            res = samplers.heun_sampler(problem.denoise, x0, t)
        else:
            res = samplers.rx_sampler(
                problem.denoise, x0, t, frequency=frequency,
                n_levels=n_levels, p=2, reuse_mode=reuse_mode,
            )
        return float(np.sqrt(np.mean((np.asarray(res.x) - truth) ** 2)))
    return objective


def scan_rho_toy(problem=None, rhos=None, num_steps=32, frequency=2,
                 n_levels=2, n_samples=128, seed=0, methods=("euler", "heun", "rx")):
    """Dense scan of the objective, to check unimodality before searching.

    Golden-section search assumes a unimodal objective.  Rather than assume it,
    scan first: the scan is cheap on the toy problem and either justifies the
    search or reveals that it cannot be trusted.  The scan is also what gets
    plotted, with the search's evaluations overlaid.
    """
    if problem is None:
        problem = toy.bimodal(4.0, 0.5, 1)
    if rhos is None:
        rhos = np.round(np.arange(1.0, 15.01, 0.25), 4)

    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(n_samples, problem.dim)) * SIGMA_MAX
    truth = problem.ground_truth(x0, SIGMA_MAX, SIGMA_MIN)

    rows = []
    for method in methods:
        obj = _toy_objective(problem, x0, truth, num_steps, frequency,
                             n_levels, method=method)
        for rho in rhos:
            try:
                val = obj(rho)
            except Exception:
                val = float("nan")
            rows.append({
                "problem": problem.name, "method": method, "rho": float(rho),
                "num_steps": num_steps, "frequency": frequency,
                "n_levels": n_levels, "rms_error": val,
            })
    return rows


def search_rho_toy(problem=None, bracket=(1.0, 15.0), num_steps=32,
                   frequency=2, n_levels=2, n_samples=128, seed=0,
                   tol=0.05, max_eval=30, method="rx"):
    """Golden-section search for the best ``rho`` on the analytic problem."""
    if problem is None:
        problem = toy.bimodal(4.0, 0.5, 1)
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(n_samples, problem.dim)) * SIGMA_MAX
    truth = problem.ground_truth(x0, SIGMA_MAX, SIGMA_MIN)

    obj = _toy_objective(problem, x0, truth, num_steps, frequency,
                         n_levels, method=method)
    res = optimize.golden_section_search(obj, *bracket, tol=tol,
                                         max_eval=max_eval)
    rows = res.as_rows()
    for r in rows:
        r.update({"problem": problem.name, "method": method,
                  "num_steps": num_steps, "frequency": frequency,
                  "n_levels": n_levels, "objective": "rms_error"})
    return res, rows


def search_rho_fid(ctx, store, method="rx", num_steps=10, frequency=2,
                   n_levels=2, n_images=10_000, bracket=(1.0, 15.0),
                   tol=0.25, max_eval=12, verbose=True):
    """Golden-section search for the best ``rho`` under FID on CIFAR-10.

    Each objective evaluation is a full generate-and-score run, recorded in
    ``store`` so a killed session resumes rather than repeats.  ``rho`` is
    rounded to two decimals before evaluation so that the cache keys on a
    stable value.

    The returned history is as much a deliverable as the optimum: with a noisy
    objective and a dozen evaluations, the honest claim is about the region the
    search settled into, not a certified minimiser.
    """
    from ..runner import Config, run_config

    def objective(rho):
        cfg = Config(
            method=method, num_steps=num_steps, frequency=frequency,
            n_levels=n_levels, rho=round(float(rho), 2), n_images=n_images,
            tag="rho_search",
        )
        for row in store.rows():
            if row["key"] == cfg.key and row.get("fid"):
                return float(row["fid"])
        metrics = run_config(cfg, ctx)
        store.append(cfg, **metrics)
        if verbose:
            print(f"  rho={rho:6.3f} -> FID {metrics['fid']:.3f} "
                  f"({metrics['wall_seconds']:.0f}s)", flush=True)
        return float(metrics["fid"])

    res = optimize.golden_section_search(objective, *bracket, tol=tol,
                                         max_eval=max_eval, verbose=False)
    rows = res.as_rows()
    for r in rows:
        r.update({"method": method, "num_steps": num_steps,
                  "frequency": frequency, "n_levels": n_levels,
                  "n_images": n_images, "objective": "fid"})
    return res, rows
