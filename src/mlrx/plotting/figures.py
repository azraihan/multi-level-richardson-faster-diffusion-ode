"""Figure builders.

One function per figure, each returning the list of paths written.  Per the
house rules in :mod:`mlrx.plotting.style`: no titles, no in-figure captions,
vector PDF, and a CSV of the plotted numbers written beside every figure so any
of them can be restyled later without re-running an experiment.

Functions accept plain row lists (as the experiment drivers return) or
DataFrames interchangeably.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

from .style import apply_style, figsize, save_figure, series_style, PALETTE

__all__ = [
    "fig_convergence_order", "fig_reuse_ablation", "fig_conditioning_levels",
    "fig_weight_precision", "fig_vcurve", "fig_error_vs_levels", "fig_rho_scan", "fig_block_width",
    "fig_weight_values", "fig_fid_vs_nfe", "fig_multilevel_fid",
    "fig_reuse_fid", "fig_seed_blocks", "fig_rho_fid",
    "fig_multilevel_precision", "fig_fid_sample_size",
]

# Human-readable labels for each sampler method, used in legends.
METHOD_LABEL = {
    "euler": "Euler",
    "heun": "Heun (EDM)",
    "rx": "RX-Euler",
    "rx_edm": "RX+EDM",
    "rx_naive": r"Naïve Richardson",
    "rx_L2": "RX, $L=2$",
    "rx_L3": "RX, $L=3$",
    "rx_L4": "RX, $L=4$",
    "rx_L5": "RX, $L=5$",
}
# Labels for the three reuse modes.
MODE_LABEL = {
    "denoised": "reuse (denoised)",
    "derivative": "reuse (derivative)",
    "exact": "exact (extra NFE)",
}
# Labels for numerical precision levels.
PREC_LABEL = {"float64": "float64", "float32": "float32", "float16": "float16"}


def _df(x):
    """Coerce a list of dicts or an existing DataFrame into a DataFrame."""
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)


def _legend_below(ax, ncol=3, y=-0.30, fontsize=7.4):
    """Put the legend under the axes.

    Several of these plots have curves crossing the whole panel, leaving no
    interior region a legend can occupy without covering data.  Moving it out
    is more reliable than hunting for a free corner.
    """
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
              fontsize=fontsize, frameon=False)


def _loglog(ax):
    """Set both axes to log scale and add a fine grid for readability."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.8)
    ax.grid(True, which="minor", alpha=0.3, linewidth=0.4)
    # Minor tick marks at every 0.2 decade (i.e. 2, 3, 4, …, 9 × 10^n).
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())


# ---------------------------------------------------------------------------
# toy / CPU figures
# ---------------------------------------------------------------------------

def fig_convergence_order(rows, outdir, problem=None, name=None):
    """Global RMS error against step count, log-log, one line per method.

    A straight line on a log-log plot means the error scales as a power of N.
    The slope equals (minus) the convergence order: slope -1 → order 1 (Euler),
    slope -2 → order 2 (Heun, RX-Euler).
    """
    apply_style()
    df = _df(rows)
    # Filter to a specific toy problem if requested.
    if problem:
        df = df[df["problem"] == problem]
    prob = problem or (df["problem"].iloc[0] if len(df) else "toy")
    name = name or f"convergence_order__{prob}"

    fig, ax = plt.subplots(figsize=figsize("single"))
    # Draw methods in a consistent order so the same colour always means the
    # same method across all figures in the paper.
    order = ["euler", "heun", "rx_naive", "rx_L2", "rx_L3", "rx_L4", "rx_L5"]
    methods = [m for m in order if m in set(df["method"])]

    for i, m in enumerate(methods):
        sub = df[df["method"] == m].sort_values("num_steps")
        ax.plot(sub["num_steps"], sub["rms_error"],
                label=METHOD_LABEL.get(m, m), **series_style(i))

    ax.set_xlabel("number of solver steps $N$")
    ax.set_ylabel("RMS error")
    _loglog(ax)
    _legend_below(ax, ncol=3)
    # save_figure writes both the PDF and a companion CSV of the plotted data.
    return save_figure(fig, outdir, name, data=df)


def fig_reuse_ablation(order_rows, outdir, problem=None,
                       name="reuse_ablation_order"):
    """Fitted local order against level count, one line per reuse mode.

    The project's headline figure.  The dotted reference line is the order the
    error model predicts; the gap between it and the reuse curves is the cost
    of keeping the method free.

    Key insight to read from this figure:
      - At L=2, the "reuse" curve touches the predicted line → the free-lunch
        trick introduces zero approximation error.
      - At L≥3, the reuse curve stays flat near 3.2 while the "exact" curve
        keeps rising → reuse caps the achievable accuracy.
    """
    apply_style()
    df = _df(order_rows)
    if problem:
        df = df[df["problem"] == problem]
        name = f"{name}__{problem}"

    fig, ax = plt.subplots(figsize=figsize("single"))
    Ls = sorted(df["n_levels"].unique())

    # The dotted line shows the theoretical prediction: order = p + L - 1 = 2 + L - 1.
    ax.plot(Ls, [2 + L - 1 for L in Ls], color="0.45", linestyle=":",
            linewidth=1.2, marker=None, label="predicted $p+L-1$", zorder=1)

    # Draw one line per reuse mode. "denoised" is the free approximation,
    # "exact" pays extra evaluations for the honest answer.
    for i, mode in enumerate(["denoised", "derivative", "exact"]):
        sub = df[df["reuse_mode"] == mode].sort_values("n_levels")
        if sub.empty:
            continue
        ax.plot(sub["n_levels"], sub["fitted_local_order"],
                label=MODE_LABEL.get(mode, mode), **series_style(i))

    ax.set_xlabel("extrapolation levels $L$")
    ax.set_ylabel("fitted local order")
    ax.set_xticks(Ls)
    ax.grid(True, alpha=0.75)
    ax.legend(loc="upper left")
    return save_figure(fig, outdir, name, data=df)


def fig_conditioning_levels(rows, outdir, name="conditioning_vs_levels"):
    """Condition number and cancellation factor against level count.

    The condition number κ∞ measures how much rounding errors in the input
    get amplified in the output weights. Σ|w| is the "cancellation factor":
    weights that alternate in sign with large magnitudes cancel each other,
    so small errors in individual estimates produce large errors in the result.
    Both grow rapidly with L — confirming that the weight system becomes
    dangerously ill-conditioned beyond L=2.
    """
    apply_style()
    df = _df(rows).sort_values("n_levels")

    fig, ax = plt.subplots(figsize=figsize("single"))
    # Left y-axis: condition number (how much the system amplifies errors).
    ax.plot(df["n_levels"], df["cond"], label=r"$\kappa_\infty(M)$",
            **series_style(0))
    ax.set_xlabel("extrapolation levels $L$")
    ax.set_ylabel(r"$\kappa_\infty(M)$")
    ax.set_yscale("log")
    ax.set_xticks(df["n_levels"].tolist())
    ax.grid(True, which="major", alpha=0.8)

    # Right y-axis: sum of absolute weight values (cancellation factor).
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(df["n_levels"], df["amplification"],
             label=r"$\sum_n |w_n|$", **series_style(1))
    ax2.set_ylabel(r"$\sum_n |w_n|$")
    ax2.grid(False)

    # Merge the two legends into one.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    return save_figure(fig, outdir, name, data=df)


def fig_weight_precision(rows, outdir, name="weight_error_vs_precision"):
    """Error in the computed weights against level count, per precision.

    Shows how much the weights differ from the float64 reference when computed
    in float32 or float16. Also plots the theoretical bound ε·κ where ε is
    the machine epsilon and κ is the condition number. The key result is that
    at float16 the weights become essentially wrong by L=5.
    """
    apply_style()
    df = _df(rows)

    fig, ax = plt.subplots(figsize=figsize("single"))
    # float64 is the reference, so its error is identically zero — cannot be
    # shown on a log scale. We omit it silently.
    for i, prec in enumerate(["float32", "float16"]):
        sub = df[df["precision"] == prec].sort_values("n_levels")
        if sub.empty:
            continue
        # Solid line: actual measured weight error.
        ax.plot(sub["n_levels"], sub["weight_rel_error"],
                label=PREC_LABEL.get(prec, prec), **series_style(i + 1))

    for i, prec in enumerate(["float32", "float16"]):
        sub = df[df["precision"] == prec].sort_values("n_levels")
        if sub.empty:
            continue
        # Dotted line: theoretical bound predicted by condition number theory.
        ax.plot(sub["n_levels"], sub["predicted_error"],
                color=PALETTE[(i + 1) % len(PALETTE)],
                linestyle=":", linewidth=1.0, marker=None, alpha=0.8,
                label=rf"$\varepsilon\kappa$ bound ({PREC_LABEL[prec]})")

    ax.set_xlabel("extrapolation levels $L$")
    ax.set_ylabel("relative error in weights")
    ax.set_yscale("log")
    ax.set_xticks(sorted(df["n_levels"].unique()))
    ax.grid(True, which="major", alpha=0.8)
    _legend_below(ax, ncol=2)
    return save_figure(fig, outdir, name, data=df)


def fig_vcurve(rows, outdir, n_levels=2, name="vcurve_error_vs_steps"):
    """RMS error against step count, one line per working precision.

    The classical truncation-versus-round-off picture.  Open circles mark each
    series' minimum: to their left the error is truncation-limited and falls
    with refinement, to their right it is round-off-limited and does not.

    The key result: the V bottom is at N≈256 for float64 and float32.
    Since real diffusion samplers use N=10-20, we are always on the
    left arm → round-off never limits practical accuracy.
    """
    apply_style()
    df = _df(rows)
    df = df[df["rms_error"].notna()]
    if n_levels is not None and "n_levels" in df:
        df = df[df["n_levels"] == n_levels]
        name = f"{name}__L{n_levels}"

    fig, ax = plt.subplots(figsize=figsize("single"))
    for i, prec in enumerate(["float64", "float32", "float16"]):
        sub = df[df["precision"] == prec].sort_values("num_steps")
        if sub.empty:
            continue
        # float32 overlaps float64 for most of the range — that overlap IS the
        # result (round-off only diverges at very large N). Draw float64 wider
        # and semi-transparent so both lines are visible.
        st = series_style(i)
        if prec == "float64":
            st.update(linewidth=3.0, alpha=0.45, markersize=0.0)
        ax.plot(sub["num_steps"], sub["rms_error"],
                label=PREC_LABEL.get(prec, prec), **st)
        # Mark the minimum with an open circle — that is the V's bottom.
        j = int(np.nanargmin(sub["rms_error"].values))
        ax.plot([sub["num_steps"].values[j]], [sub["rms_error"].values[j]],
                marker="o", markersize=8.5, markerfacecolor="none",
                markeredgewidth=1.2, color=PALETTE[i % len(PALETTE)],
                linestyle="none", zorder=5)

    ax.set_xlabel("number of solver steps $N$")
    ax.set_ylabel("RMS error")
    _loglog(ax)
    ax.legend(loc="lower left")
    return save_figure(fig, outdir, name, data=df)


def fig_error_vs_levels(rows, outdir, num_steps=None,
                        name="error_vs_levels"):
    """RMS error against level count, per precision.

    The companion to :func:`fig_vcurve`, and a negative result: error rises
    monotonically with ``L``, because without extra evaluations the added
    levels buy no order while still costing conditioning.
    """
    apply_style()
    df = _df(rows)
    df = df[df["rms_error"].notna()]
    if num_steps is not None:
        df = df[df["num_steps"] == num_steps]
        name = f"{name}__N{num_steps}"

    fig, ax = plt.subplots(figsize=figsize("single"))
    for i, prec in enumerate(["float64", "float32", "float16"]):
        # Average error over all step counts at each level count to give one
        # summary line per precision.
        sub = (df[df["precision"] == prec]
               .groupby("n_levels", as_index=False)["rms_error"].mean()
               .sort_values("n_levels"))
        if sub.empty:
            continue
        ax.plot(sub["n_levels"], sub["rms_error"],
                label=PREC_LABEL.get(prec, prec), **series_style(i))

    ax.set_xlabel("extrapolation levels $L$")
    ax.set_ylabel("RMS error")
    ax.set_yscale("log")
    ax.set_xticks(sorted(df["n_levels"].unique()))
    ax.grid(True, which="major", alpha=0.8)
    ax.legend(loc="upper left")
    return save_figure(fig, outdir, name, data=df)


def fig_rho_scan(scan_rows, outdir, search_rows=None, name="rho_scan"):
    """Objective against the schedule exponent, with search evaluations marked.

    Shows RMS error (on the toy ODE) as a function of ρ for three methods.
    The dotted vertical lines mark each method's optimal ρ in the scan.
    The dashed vertical line at ρ=7 shows the EDM default.
    The 'x' markers show where the golden-section search placed its evaluations.
    """
    apply_style()
    df = _df(scan_rows)

    fig, ax = plt.subplots(figsize=figsize("single"))
    for i, m in enumerate(sorted(df["method"].unique())):
        sub = df[df["method"] == m].sort_values("rho")
        # Solid line: error at each scanned ρ.
        ax.plot(sub["rho"], sub["rms_error"], label=METHOD_LABEL.get(m, m),
                **dict(series_style(i), marker=None))
        # Dotted vertical: where this method's best ρ lies.
        j = sub["rms_error"].values.argmin()
        ax.axvline(sub["rho"].values[j], color=PALETTE[i % len(PALETTE)],
                   linestyle=":", linewidth=0.9, alpha=0.8)

    if search_rows is not None and len(search_rows):
        # Overlay the golden-section search evaluation points with 'x' markers.
        s = _df(search_rows)
        ax.plot(s["x"], s["f"], linestyle="none", marker="x", markersize=5,
                markeredgewidth=1.1, color="0.25",
                label="golden-section evaluations")

    # Dashed vertical at the EDM default ρ=7 for reference.
    ax.axvline(7.0, color="0.6", linewidth=0.9, linestyle="--", alpha=0.9)
    ax.set_xlabel(r"schedule exponent $\rho$")
    ax.set_ylabel("RMS error")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.8)
    _legend_below(ax, ncol=2)
    return save_figure(fig, outdir, name, data={"scan": df,
                                                "search": _df(search_rows)
                                                if search_rows is not None else None})


def fig_block_width(rows, outdir, name="conditioning_vs_block"):
    """Conditioning against block width, separating it from level count.

    Each line is a fixed L; the x-axis is the block length k (how many fine
    steps per extrapolation block). This isolates the effect of block geometry
    from the effect of adding more levels.
    """
    apply_style()
    df = _df(rows)

    fig, ax = plt.subplots(figsize=figsize("single"))
    for i, L in enumerate(sorted(df["n_levels"].unique())):
        # Average condition number over any repeated measurements at each k.
        sub = (df[df["n_levels"] == L]
               .groupby("frequency", as_index=False)["cond"].mean()
               .sort_values("frequency"))
        if sub.empty:
            continue
        ax.plot(sub["frequency"], sub["cond"], label=rf"$L={L}$",
                **series_style(i))

    ax.set_xlabel("block length $k$ (base steps)")
    ax.set_ylabel(r"$\kappa_\infty(M)$")
    ax.set_xscale("log", base=2)   # block lengths are powers of 2
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.8)
    ax.legend(loc="upper left")
    return save_figure(fig, outdir, name, data=df)


def fig_weight_values(outdir, level_counts=(2, 3, 4, 5), rho=7.0,
                      name="weight_values"):
    """The weights themselves, showing sign alternation and growth.

    At L=2 the weights are modest (e.g., -3 and +4).
    At L=3+ they start alternating in sign with increasing magnitude —
    the visual signature of a poorly-conditioned linear system where large
    positive and negative terms nearly cancel.
    """
    from .. import extrapolation as ex, schedules

    apply_style()
    fig, ax = plt.subplots(figsize=figsize("single"))
    records = []

    for i, L in enumerate(level_counts):
        # Use the minimum block size that supports L nested levels.
        K = 2 ** (L - 1)
        # Build an EDM schedule, extract the first block's sub-step widths.
        t = schedules.edm_schedule(max(K * 4, 32), rho=rho)
        base = schedules.block_lambdas(t[0:K + 1])
        lv = ex.nested_level_lambdas(base, schedules.nested_steps(K, L))
        # Solve the extrapolation weight system for this block geometry.
        W = ex.extrapolation_weights(lv, p=2)
        xs = np.arange(W.w.size)  # level indices 0, 1, ..., L-1
        ax.plot(xs, W.w, label=rf"$L={L}$", **series_style(i))
        # Collect the raw weights for the companion CSV.
        for j, v in enumerate(W.w):
            records.append({"n_levels": L, "frequency": K, "level_index": j,
                            "level_steps": schedules.nested_steps(K, L)[j],
                            "weight": float(v), "cond": W.cond,
                            "amplification": W.amplification})

    # Draw a horizontal line at zero to highlight sign changes.
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("level index (coarsest to finest)")
    ax.set_ylabel("weight $w_n$")
    ax.set_xticks(range(max(level_counts)))
    ax.grid(True, alpha=0.75)
    _legend_below(ax, ncol=4)
    return save_figure(fig, outdir, name, data=records)


# ---------------------------------------------------------------------------
# FID / GPU figures
# ---------------------------------------------------------------------------

def _fid_series(ax, df, key, labeller, i0=0):
    """Helper: plot one FID-vs-NFE line for each unique value of df[key]."""
    for i, val in enumerate(sorted(df[key].dropna().unique())):
        # Average FID across seed blocks (if any) at each NFE point.
        sub = (df[df[key] == val]
               .groupby("nfe", as_index=False)["fid"].mean()
               .sort_values("nfe"))
        if sub.empty:
            continue
        ax.plot(sub["nfe"], sub["fid"], label=labeller(val),
                **series_style(i + i0))


def fig_fid_vs_nfe(df, outdir, tag="panel", name=None):
    """FID against NFE.  Used for both the validity test and the main panel.

    Lower FID = better image quality. The x-axis is NFE (neural network calls
    per image), which is the true computational cost being compared.
    The key result: RX-Euler at NFE 10 beats Heun at NFE 11.
    """
    apply_style()
    df = _df(df)
    if tag is not None and "tag" in df:
        df = df[df["tag"] == tag]
    name = name or f"fid_vs_nfe__{tag}"

    fig, ax = plt.subplots(figsize=figsize("single"))

    def label(row):
        """Build a human-readable series label for each (method, settings) row."""
        if row["method"] == "rx" and row.get("coefficients") == "naive":
            return "Naïve Richardson"
        if row["method"] == "rx":
            return f"RX-Euler ($k={int(row['frequency'])}$)"
        return METHOD_LABEL.get(row["method"], row["method"])

    df = df.copy()
    df["series"] = df.apply(label, axis=1)

    for i, s in enumerate(sorted(df["series"].unique())):
        sub = (df[df["series"] == s]
               .groupby("nfe", as_index=False)["fid"].mean()
               .sort_values("nfe"))
        ax.plot(sub["nfe"], sub["fid"], label=s, **series_style(i))

    ax.set_xlabel("NFE")
    ax.set_ylabel("FID")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.75)
    ax.legend(loc="upper right")
    return save_figure(fig, outdir, name, data=df)


def fig_multilevel_fid(df, outdir, name="fid_multilevel_levels", tag=None):
    """FID against level count, one line per NFE budget.

    Plotted per NFE rather than averaged over NFE: at eight evaluations the
    three-level sampler scores 243 and at sixteen it scores 65, so a mean over
    budgets would describe no configuration that was actually run.  The y-axis
    is logarithmic because the spread across ``L`` is two orders of magnitude.

    Precision is held at float64 here; :func:`fig_multilevel_precision`
    reports the (negligible) effect of varying it.

    The key result: FID skyrockets from L=2 to L=3 — confirming that more
    levels hurt rather than help on real images.
    """
    apply_style()
    df = _df(df)
    if tag is not None and "tag" in df:
        df = df[df["tag"] == tag]
    # Only show the "reuse" mode (not exact) for this figure.
    if "reuse_mode" in df:
        df = df[df["reuse_mode"] == "denoised"]
    # Use float64 as the reference precision for this figure.
    ref = df[df["work_dtype"] == "float64"] if "work_dtype" in df else df

    fig, ax = plt.subplots(figsize=figsize("single"))
    for i, nfe in enumerate(sorted(ref["nfe"].dropna().unique())):
        sub = (ref[ref["nfe"] == nfe]
               .groupby("n_levels", as_index=False)["fid"].mean()
               .sort_values("n_levels"))
        if sub.empty:
            continue
        ax.plot(sub["n_levels"], sub["fid"], label=f"NFE {int(nfe)}",
                **series_style(i))

    ax.set_xlabel("extrapolation levels $L$")
    ax.set_ylabel("FID")
    ax.set_yscale("log")   # log scale because values span 2 orders of magnitude
    ax.set_xticks(sorted(ref["n_levels"].dropna().unique().astype(int)))
    ax.grid(True, which="major", alpha=0.8)
    ax.legend(loc="lower right")
    return save_figure(fig, outdir, name, data=ref)


def fig_multilevel_precision(df, outdir, name="fid_multilevel_precision",
                             tag=None):
    """FID change from reducing the extrapolation arithmetic precision.

    Reported as a percentage difference from float64 so that the comparison is
    readable despite FID itself spanning two orders of magnitude across ``L``.
    The result is a null: the bars are a couple of percent at most, confirming
    on real data what the analytic study predicted -- at realistic step counts
    truncation dominates and round-off in the weights never becomes the
    limiting error.
    """
    apply_style()
    df = _df(df)
    if tag is not None and "tag" in df:
        df = df[df["tag"] == tag]
    if "reuse_mode" in df:
        df = df[df["reuse_mode"] == "denoised"]

    # Pivot so each row is a (nfe, L) combination and each column is a precision.
    piv = df.pivot_table(index=["nfe", "n_levels"], columns="work_dtype",
                         values="fid")
    piv = piv.dropna(subset=["float64"])
    rows, labels = [], []
    for (nfe, L), r in piv.iterrows():
        for prec in ("float32", "float16"):
            if prec in r and np.isfinite(r[prec]):
                # Compute percentage change vs the float64 reference.
                rows.append({"nfe": int(nfe), "n_levels": int(L),
                             "precision": prec, "fid": r[prec],
                             "fid_float64": r["float64"],
                             "pct_change": 100 * (r[prec] - r["float64"])
                             / r["float64"]})
        labels.append(f"{int(nfe)}/{int(L)}")

    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=figsize("single"))
    xs = np.arange(len(labels))
    width = 0.38
    # Draw one bar group per (NFE, L) configuration, two bars per group.
    for i, prec in enumerate(("float32", "float16")):
        sub = d[d["precision"] == prec]
        y = [float(sub[(sub.nfe == int(l.split("/")[0]))
                       & (sub.n_levels == int(l.split("/")[1]))]["pct_change"]
                   .squeeze() or 0.0) if len(sub) else 0.0 for l in labels]
        ax.bar(xs + (i - 0.5) * width, y, width,
               label=PREC_LABEL[prec], color=PALETTE[i + 1])

    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel("NFE / levels $L$")
    ax.set_ylabel(r"change in FID vs float64 (\%)")
    ax.grid(True, axis="y", alpha=0.75)
    ax.legend(loc="best")
    return save_figure(fig, outdir, name, data=d)


def fig_reuse_fid(df, outdir, name="fid_reuse_vs_exact", tag=None):
    """FID against NFE for reuse versus exact recomputation, by level count.

    The "reuse" arm is free (same NFE as Euler). The "exact" arm pays extra
    evaluations to get the honest coarse estimates.
    Key result: even paying for exact evaluations, L=3 exact at NFE 20
    (FID 6.35) loses to plain L=2 at NFE 20 (FID 4.57).
    """
    apply_style()
    df = _df(df)
    if tag is not None and "tag" in df:
        df = df[df["tag"] == tag]

    fig, ax = plt.subplots(figsize=figsize("single"))
    i = 0
    # Plot all (L, mode) combinations sorted so colours are consistent.
    for mode in ("denoised", "exact"):
        for L in sorted(df["n_levels"].dropna().unique()):
            sub = (df[(df["reuse_mode"] == mode) & (df["n_levels"] == L)]
                   .groupby("nfe", as_index=False)["fid"].mean()
                   .sort_values("nfe"))
            if sub.empty:
                continue
            ax.plot(sub["nfe"], sub["fid"],
                    label=rf"$L={int(L)}$, {MODE_LABEL[mode]}",
                    **series_style(i))
            i += 1

    ax.set_xlabel("NFE")
    ax.set_ylabel("FID")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
              fontsize=6.8)
    return save_figure(fig, outdir, name, data=df)


def fig_seed_blocks(df, outdir, name="fid_seed_blocks", tag=None):
    """FID at the validation gate with across-block spread as error bars.

    Each method is run on 3 independent seed blocks of 10k images.
    The bar height is the mean FID; the error bar is the standard error of
    the mean (SEM). Methods with non-overlapping SEM bars are genuinely
    different — this is what licenses claims like "RX-Euler beats Heun".
    """
    apply_style()
    df = _df(df)
    if tag is not None and "tag" in df:
        df = df[df["tag"] == tag]

    # Create a readable series label that includes both method name and NFE.
    df = df.copy()
    df["series"] = df.apply(
        lambda r: f"{METHOD_LABEL.get(r['method'], r['method'])}\n"
                  f"(NFE {int(r['nfe'])})", axis=1)
    # Compute mean, std, and SEM across the seed blocks.
    g = df.groupby("series")["fid"].agg(["mean", "std", "count"]).reset_index()
    g = g.sort_values("mean")
    # SEM = std / sqrt(n) — the uncertainty of the mean, not of a single run.
    g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))

    fig, ax = plt.subplots(figsize=figsize("single"))
    xs = np.arange(len(g))
    ax.bar(xs, g["mean"], yerr=g["sem"], width=0.62,
           color=[PALETTE[i % len(PALETTE)] for i in range(len(g))],
           capsize=3, error_kw=dict(lw=0.9, capthick=0.9, ecolor="0.25"))
    ax.set_xticks(xs)
    ax.set_xticklabels(g["series"])
    ax.set_ylabel("FID")
    ax.grid(True, axis="y", alpha=0.75)
    return save_figure(fig, outdir, name, data={"summary": g, "raw": df})


def fig_rho_fid(df, outdir, name="rho_search_fid"):
    """FID at each schedule exponent the golden-section search evaluated.

    Shows the FID as a scatter plot of ρ vs FID (only the evaluations made
    by the golden-section search, not a dense scan). The circle marks the
    best point found; the dashed line marks the EDM default of ρ=7.
    """
    apply_style()
    df = _df(df).sort_values("rho")

    fig, ax = plt.subplots(figsize=figsize("single"))
    # Scatter plot — no line, because the search doesn't visit ρ in order.
    ax.plot(df["rho"], df["fid"], linestyle="none", **series_style(0))
    # Mark the best evaluated point with an open circle.
    j = int(df["fid"].values.argmin())
    ax.plot([df["rho"].values[j]], [df["fid"].values[j]], marker="o",
            markersize=9, markerfacecolor="none", markeredgewidth=1.2,
            color=PALETTE[0], linestyle="none")
    # Reference line at the EDM default ρ=7.
    ax.axvline(7.0, color="0.6", linewidth=0.9, linestyle="--")
    ax.set_xlabel(r"schedule exponent $\rho$")
    ax.set_ylabel("FID")
    ax.grid(True, alpha=0.75)
    return save_figure(fig, outdir, name, data=df)


def fig_fid_sample_size(rows, outdir, name="fid_sample_size_bias"):
    """FID against the number of samples used to estimate it.

    FID is a biased estimator: it reads artificially high when computed from
    fewer images, and the bias grows nonlinearly. This figure quantifies that
    bias so we can state exactly how much our 10k-image numbers differ from
    the paper's 50k-image numbers — and confirm the ranking is preserved.
    """
    apply_style()
    df = _df(rows)
    # Average and spread of FID at each sample count.
    g = df.groupby("n_samples")["fid"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=figsize("single"))
    ax.errorbar(g["n_samples"], g["mean"], yerr=g["std"].fillna(0.0),
                capsize=3, elinewidth=0.9, capthick=0.9, **series_style(0))
    ax.set_xlabel("images used for FID")
    ax.set_ylabel("FID")
    ax.set_xscale("log")
    ax.grid(True, which="major", alpha=0.8)
    return save_figure(fig, outdir, name, data={"summary": g, "raw": df})
