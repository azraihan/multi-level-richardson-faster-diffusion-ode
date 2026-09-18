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
MODE_LABEL = {
    "denoised": "reuse (denoised)",
    "derivative": "reuse (derivative)",
    "exact": "exact (extra NFE)",
}
PREC_LABEL = {"float64": "float64", "float32": "float32", "float16": "float16"}


def _df(x):
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
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.8)
    ax.grid(True, which="minor", alpha=0.3, linewidth=0.4)
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())


# ---------------------------------------------------------------------------
# toy / CPU figures
# ---------------------------------------------------------------------------

def fig_convergence_order(rows, outdir, problem=None, name=None):
    """Global RMS error against step count, log-log, one line per method."""
    apply_style()
    df = _df(rows)
    if problem:
        df = df[df["problem"] == problem]
    prob = problem or (df["problem"].iloc[0] if len(df) else "toy")
    name = name or f"convergence_order__{prob}"

    fig, ax = plt.subplots(figsize=figsize("single"))
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
    return save_figure(fig, outdir, name, data=df)


def fig_reuse_ablation(order_rows, outdir, problem=None,
                       name="reuse_ablation_order"):
    """Fitted local order against level count, one line per reuse mode.

    The project's headline figure.  The dotted reference line is the order the
    error model predicts; the gap between it and the reuse curves is the cost
    of keeping the method free.
    """
    apply_style()
    df = _df(order_rows)
    if problem:
        df = df[df["problem"] == problem]
        name = f"{name}__{problem}"

    fig, ax = plt.subplots(figsize=figsize("single"))
    Ls = sorted(df["n_levels"].unique())

    ax.plot(Ls, [2 + L - 1 for L in Ls], color="0.45", linestyle=":",
            linewidth=1.2, marker=None, label="predicted $p+L-1$", zorder=1)

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
    """Condition number and cancellation factor against level count."""
    apply_style()
    df = _df(rows).sort_values("n_levels")

    fig, ax = plt.subplots(figsize=figsize("single"))
    ax.plot(df["n_levels"], df["cond"], label=r"$\kappa_\infty(M)$",
            **series_style(0))
    ax.set_xlabel("extrapolation levels $L$")
    ax.set_ylabel(r"$\kappa_\infty(M)$")
    ax.set_yscale("log")
    ax.set_xticks(df["n_levels"].tolist())
    ax.grid(True, which="major", alpha=0.8)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(df["n_levels"], df["amplification"],
             label=r"$\sum_n |w_n|$", **series_style(1))
    ax2.set_ylabel(r"$\sum_n |w_n|$")
    ax2.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    return save_figure(fig, outdir, name, data=df)


def fig_weight_precision(rows, outdir, name="weight_error_vs_precision"):
    """Error in the computed weights against level count, per precision."""
    apply_style()
    df = _df(rows)

    fig, ax = plt.subplots(figsize=figsize("single"))
    # float64 is the reference the others are measured against, so its error is
    # identically zero and cannot be drawn on a log axis.  It is omitted rather
    # than shown as an empty legend entry.
    for i, prec in enumerate(["float32", "float16"]):
        sub = df[df["precision"] == prec].sort_values("n_levels")
        if sub.empty:
            continue
        ax.plot(sub["n_levels"], sub["weight_rel_error"],
                label=PREC_LABEL.get(prec, prec), **series_style(i + 1))

    for i, prec in enumerate(["float32", "float16"]):
        sub = df[df["precision"] == prec].sort_values("n_levels")
        if sub.empty:
            continue
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
        # float32 lies exactly on float64 for most of the range -- that
        # coincidence is itself the result, so the lower series is drawn
        # wider and semi-transparent rather than being hidden underneath.
        st = series_style(i)
        if prec == "float64":
            st.update(linewidth=3.0, alpha=0.45, markersize=0.0)
        ax.plot(sub["num_steps"], sub["rms_error"],
                label=PREC_LABEL.get(prec, prec), **st)
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
    """Objective against the schedule exponent, with search evaluations marked."""
    apply_style()
    df = _df(scan_rows)

    fig, ax = plt.subplots(figsize=figsize("single"))
    for i, m in enumerate(sorted(df["method"].unique())):
        sub = df[df["method"] == m].sort_values("rho")
        ax.plot(sub["rho"], sub["rms_error"], label=METHOD_LABEL.get(m, m),
                **dict(series_style(i), marker=None))
        j = sub["rms_error"].values.argmin()
        ax.axvline(sub["rho"].values[j], color=PALETTE[i % len(PALETTE)],
                   linestyle=":", linewidth=0.9, alpha=0.8)

    if search_rows is not None and len(search_rows):
        s = _df(search_rows)
        ax.plot(s["x"], s["f"], linestyle="none", marker="x", markersize=5,
                markeredgewidth=1.1, color="0.25",
                label="golden-section evaluations")

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
    """Conditioning against block width, separating it from level count."""
    apply_style()
    df = _df(rows)

    fig, ax = plt.subplots(figsize=figsize("single"))
    for i, L in enumerate(sorted(df["n_levels"].unique())):
        sub = (df[df["n_levels"] == L]
               .groupby("frequency", as_index=False)["cond"].mean()
               .sort_values("frequency"))
        if sub.empty:
            continue
        ax.plot(sub["frequency"], sub["cond"], label=rf"$L={L}$",
                **series_style(i))

    ax.set_xlabel("block length $K$ (base steps)")
    ax.set_ylabel(r"$\kappa_\infty(M)$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.8)
    ax.legend(loc="upper left")
    return save_figure(fig, outdir, name, data=df)


def fig_weight_values(outdir, level_counts=(2, 3, 4, 5), rho=7.0,
                      name="weight_values"):
    """The weights themselves, showing sign alternation and growth."""
    from .. import extrapolation as ex, schedules

    apply_style()
    fig, ax = plt.subplots(figsize=figsize("single"))
    records = []

    for i, L in enumerate(level_counts):
        K = 2 ** (L - 1)
        t = schedules.edm_schedule(max(K * 4, 32), rho=rho)
        base = schedules.block_lambdas(t[0:K + 1])
        lv = ex.nested_level_lambdas(base, schedules.nested_steps(K, L))
        W = ex.extrapolation_weights(lv, p=2)
        xs = np.arange(W.w.size)
        ax.plot(xs, W.w, label=rf"$L={L}$", **series_style(i))
        for j, v in enumerate(W.w):
            records.append({"n_levels": L, "frequency": K, "level_index": j,
                            "level_steps": schedules.nested_steps(K, L)[j],
                            "weight": float(v), "cond": W.cond,
                            "amplification": W.amplification})

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
    for i, val in enumerate(sorted(df[key].dropna().unique())):
        sub = (df[df[key] == val]
               .groupby("nfe", as_index=False)["fid"].mean()
               .sort_values("nfe"))
        if sub.empty:
            continue
        ax.plot(sub["nfe"], sub["fid"], label=labeller(val),
                **series_style(i + i0))


def fig_fid_vs_nfe(df, outdir, tag="panel", name=None):
    """FID against NFE.  Used for both the validity test and the main panel."""
    apply_style()
    df = _df(df)
    if tag is not None and "tag" in df:
        df = df[df["tag"] == tag]
    name = name or f"fid_vs_nfe__{tag}"

    fig, ax = plt.subplots(figsize=figsize("single"))

    def label(row):
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
    """
    apply_style()
    df = _df(df)
    if tag is not None and "tag" in df:
        df = df[df["tag"] == tag]
    if "reuse_mode" in df:
        df = df[df["reuse_mode"] == "denoised"]
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
    ax.set_yscale("log")
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

    piv = df.pivot_table(index=["nfe", "n_levels"], columns="work_dtype",
                         values="fid")
    piv = piv.dropna(subset=["float64"])
    rows, labels = [], []
    for (nfe, L), r in piv.iterrows():
        for prec in ("float32", "float16"):
            if prec in r and np.isfinite(r[prec]):
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
    """FID against NFE for reuse versus exact recomputation, by level count."""
    apply_style()
    df = _df(df)
    if tag is not None and "tag" in df:
        df = df[df["tag"] == tag]

    fig, ax = plt.subplots(figsize=figsize("single"))
    i = 0
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
    """FID at the validation gate with across-block spread as error bars."""
    apply_style()
    df = _df(df)
    if tag is not None and "tag" in df:
        df = df[df["tag"] == tag]

    df = df.copy()
    df["series"] = df.apply(
        lambda r: f"{METHOD_LABEL.get(r['method'], r['method'])}\n"
                  f"(NFE {int(r['nfe'])})", axis=1)
    g = df.groupby("series")["fid"].agg(["mean", "std", "count"]).reset_index()
    g = g.sort_values("mean")
    # standard error of the mean across independent seed blocks
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
    """FID at each schedule exponent the golden-section search evaluated."""
    apply_style()
    df = _df(df).sort_values("rho")

    fig, ax = plt.subplots(figsize=figsize("single"))
    ax.plot(df["rho"], df["fid"], linestyle="none", **series_style(0))
    j = int(df["fid"].values.argmin())
    ax.plot([df["rho"].values[j]], [df["fid"].values[j]], marker="o",
            markersize=9, markerfacecolor="none", markeredgewidth=1.2,
            color=PALETTE[0], linestyle="none")
    ax.axvline(7.0, color="0.6", linewidth=0.9, linestyle="--")
    ax.set_xlabel(r"schedule exponent $\rho$")
    ax.set_ylabel("FID")
    ax.grid(True, alpha=0.75)
    return save_figure(fig, outdir, name, data=df)


def fig_fid_sample_size(rows, outdir, name="fid_sample_size_bias"):
    """FID against the number of samples used to estimate it."""
    apply_style()
    df = _df(rows)
    g = df.groupby("n_samples")["fid"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=figsize("single"))
    ax.errorbar(g["n_samples"], g["mean"], yerr=g["std"].fillna(0.0),
                capsize=3, elinewidth=0.9, capthick=0.9, **series_style(0))
    ax.set_xlabel("images used for FID")
    ax.set_ylabel("FID")
    ax.set_xscale("log")
    ax.grid(True, which="major", alpha=0.8)
    return save_figure(fig, outdir, name, data={"summary": g, "raw": df})
