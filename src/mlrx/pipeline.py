"""End-to-end stage drivers.

Two stages, deliberately separated by cost and by hardware:

:func:`run_cpu_stage`
    The analytic and pure-numerics studies.  No GPU, no network, no downloads;
    a couple of minutes on any laptop.  Produces the order verification, the
    reuse ablation, the conditioning and precision studies, the V-curve and the
    ``rho`` scan -- which is to say, most of the project's actual findings.

:func:`run_gpu_stage`
    The CIFAR-10 FID sweeps.  Hours on a GPU, resumable, and dependent on the
    vendored assets.

Both write a CSV per study under ``results/`` and a PDF per figure under
``figures/``, with the plotted numbers mirrored into ``figures/data/``.
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd

from . import toy
from .experiments import conditioning, rho_search, sweeps, toy_order
from .plotting import figures as F

__all__ = ["run_cpu_stage", "run_gpu_stage", "save_table"]


def save_table(rows, results_dir, name):
    os.makedirs(results_dir, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    path = os.path.join(results_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cpu_stage(outdir="outputs", quick=False, verbose=True):
    """Run every experiment that needs neither a GPU nor the vendored assets."""
    results = os.path.join(outdir, "results")
    figdir = os.path.join(outdir, "figures")
    os.makedirs(results, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    written = {"tables": [], "figures": []}

    n_samples = 32 if quick else 128
    widths = ((0.8, 0.4, 0.2, 0.1) if quick
              else (0.8, 0.4, 0.2, 0.1, 0.05, 0.025))
    levels = (2, 3, 4) if quick else (2, 3, 4, 5)

    # -- 0. sanity: is the reference solution good enough to be ground truth?
    _log("reference self-test")
    selftest = toy.reference_selftest(n_steps=5_000 if quick else 20_000)
    written["tables"].append(save_table(
        [{"check": "rk4_reference_vs_closed_form", "max_rel_error": selftest,
          "threshold": 1e-10, "passed": bool(selftest < 1e-10)}],
        results, "reference_selftest"))
    if verbose:
        _log(f"  RK4 reference vs closed form: {selftest:.2e}")

    # -- 1. global convergence order -------------------------------------
    _log("convergence order study")
    rows, orders = toy_order.run_order_study(
        frequency=8, level_counts=(2, 3, 4), n_samples=n_samples,
        step_counts=(8, 16, 24, 32) if quick else None)
    written["tables"] += [save_table(rows, results, "order_study_raw"),
                          save_table(orders, results, "order_study_fitted")]
    for prob in sorted({r["problem"] for r in rows}):
        written["figures"] += F.fig_convergence_order(rows, figdir, problem=prob)

    # -- 2. reuse ablation (the headline result) --------------------------
    _log("reuse ablation -- local order by level count and reuse mode")
    ab_rows, ab_orders = toy_order.run_reuse_ablation(
        level_counts=levels, widths=widths, n_samples=2 * n_samples)
    written["tables"] += [save_table(ab_rows, results, "reuse_ablation_raw"),
                          save_table(ab_orders, results, "reuse_ablation_order")]
    for prob in sorted({r["problem"] for r in ab_orders}):
        written["figures"] += F.fig_reuse_ablation(ab_orders, figdir, problem=prob)

    # -- 3. conditioning ---------------------------------------------------
    _log("conditioning studies")
    lvl = conditioning.run_level_study(level_counts=(2, 3, 4, 5, 6, 7))
    prec = conditioning.run_precision_study(level_counts=(2, 3, 4, 5, 6, 7))
    blk = conditioning.run_block_width_study(
        frequencies=(2, 4, 8, 16) if quick else (2, 4, 8, 16, 32))
    rho_cond = conditioning.run_rho_study()
    written["tables"] += [
        save_table(lvl, results, "conditioning_levels"),
        save_table(prec, results, "conditioning_precision"),
        save_table(blk, results, "conditioning_block_width"),
        save_table(rho_cond, results, "conditioning_rho"),
    ]
    written["figures"] += F.fig_conditioning_levels(lvl, figdir)
    written["figures"] += F.fig_weight_precision(prec, figdir)
    written["figures"] += F.fig_block_width(blk, figdir)
    written["figures"] += F.fig_weight_values(figdir)

    # -- 4. the V-curve ----------------------------------------------------
    _log("V-curve: error against level count, per precision")
    v = toy_order.run_vcurve_study(
        level_counts=(2, 3),
        step_counts=(16, 32, 64, 128, 256) if quick
        else (16, 32, 64, 128, 256, 512, 1024, 2048),
        n_samples=n_samples)
    written["tables"].append(save_table(v, results, "vcurve"))
    for L in sorted({r["n_levels"] for r in v}):
        written["figures"] += F.fig_vcurve(v, figdir, n_levels=L)
    written["figures"] += F.fig_error_vs_levels(v, figdir)

    # -- 5. rho ------------------------------------------------------------
    _log("rho scan and golden-section search")
    scan = rho_search.scan_rho_toy(
        num_steps=32, n_samples=n_samples,
        rhos=None if not quick else [1, 3, 5, 7, 9, 11, 13, 15],
        methods=("euler", "heun", "rx"))
    res, srows = rho_search.search_rho_toy(
        num_steps=32, n_samples=n_samples, tol=0.05)
    written["tables"] += [save_table(scan, results, "rho_scan"),
                          save_table(srows, results, "rho_search_toy")]
    written["figures"] += F.fig_rho_scan(scan, figdir, search_rows=srows)

    summary = {
        "reference_selftest": selftest,
        "rho_search_best": res.x,
        "rho_search_evals": res.n_eval,
        "n_tables": len(written["tables"]),
        "n_figures": len([p for p in written["figures"] if p.endswith(".pdf")]),
    }
    with open(os.path.join(results, "cpu_stage_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"CPU stage done: {summary['n_tables']} tables, "
         f"{summary['n_figures']} figures")
    return written, summary


def run_gpu_stage(outdir, ctx_kwargs, n_images=10_000, nfe_grid=None,
                  n_gpus=None, include_reuse=True, do_rho_search=True,
                  verbose=True):
    """Run the CIFAR-10 FID sweeps and build their figures.

    Safe to call repeatedly -- :class:`mlrx.runner.ResultStore` skips
    configurations already recorded, so an interrupted session resumes.
    """
    from .runner import GPUContext, ResultStore, run_sweep

    results = os.path.join(outdir, "results")
    figdir = os.path.join(outdir, "figures")
    os.makedirs(results, exist_ok=True)
    store_path = os.path.join(results, "fid_results.csv")

    nfe_grid = tuple(nfe_grid) if nfe_grid else sweeps.DEFAULT_NFE_GRID
    cfgs = sweeps.build_all(n_images=n_images, nfe_grid=nfe_grid,
                            include_reuse=include_reuse)
    est = sweeps.estimate_cost(cfgs, n_gpus=n_gpus or 1)
    _log(f"{est['n_configs']} configs, {est['image_nfe']/1e6:.1f}M image-NFE, "
         f"~{est['wall_hours']:.1f}h wall (pre-run estimate)")

    store = run_sweep(cfgs, store_path, ctx_kwargs, n_gpus=n_gpus,
                      verbose=verbose)

    if do_rho_search:
        _log("rho search under FID")
        try:
            ctx = GPUContext(device="cuda:0", **ctx_kwargs)
            res, rows = rho_search.search_rho_fid(
                ctx, store, n_images=n_images, max_eval=10)
            save_table(rows, results, "rho_search_fid")
            _log(f"  best rho = {res.x:.3f} (FID {res.f:.3f})")
        except Exception as exc:
            # The sweep's results are already on disk; do not let this
            # optional step prevent the figures from being built.
            import traceback
            traceback.print_exc()
            _log(f"  rho search failed ({exc!r}); continuing to figures")

    df = store.dataframe()
    save_table(df, results, "fid_results_merged")

    # Configurations shared between studies are run once and stored once
    # (their key ignores the tag), so each figure selects its rows by the keys
    # its own builder produces rather than by tag.
    def rows_for(cfg_list):
        if df.empty:
            return df
        keys = {c.key for c in cfg_list}
        sub = df[df["key"].isin(keys)].copy()
        return sub

    groups = {
        "validity": sweeps.build_validity_sweep(nfe_grid, n_images),
        "panel": sweeps.build_main_panel(nfe_grid, n_images),
        "multilevel": sweeps.build_multilevel_sweep(n_images=n_images),
        "reuse": sweeps.build_reuse_sweep(n_images=n_images),
        "seedblock": sweeps.build_seed_blocks(n_images=n_images),
    }
    written = []
    sel = {name: rows_for(cfgs_) for name, cfgs_ in groups.items()}
    for name, sub in sel.items():
        if not sub.empty:
            save_table(sub, results, f"fid__{name}")
    if not sel["validity"].empty:
        written += F.fig_fid_vs_nfe(sel["validity"], figdir, tag=None,
                                    name="fid_vs_nfe__validity")
    if not sel["panel"].empty:
        written += F.fig_fid_vs_nfe(sel["panel"], figdir, tag=None,
                                    name="fid_vs_nfe__panel")
    if not sel["multilevel"].empty:
        written += F.fig_multilevel_fid(sel["multilevel"], figdir)
    if not sel["reuse"].empty:
        written += F.fig_reuse_fid(sel["reuse"], figdir)
    if not sel["seedblock"].empty:
        written += F.fig_seed_blocks(sel["seedblock"], figdir)
    rho_rows = df[df["tag"] == "rho_search"] if not df.empty else df
    if len(rho_rows):
        written += F.fig_rho_fid(rho_rows, figdir)

    _log(f"GPU stage done: {len(df)} rows, "
         f"{len([p for p in written if p.endswith('.pdf')])} figures")
    return written, df
