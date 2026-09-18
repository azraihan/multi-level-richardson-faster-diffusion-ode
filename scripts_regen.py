"""Rebuild every figure from the saved CSVs -- no GPU, no experiments re-run.

Usage:  python scripts_regen.py results/outputs
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import matplotlib
matplotlib.use("Agg")
from mlrx.plotting import figures as F

out = sys.argv[1] if len(sys.argv) > 1 else "results/outputs"
res, fig = f"{out}/results", f"{out}/figures"


def load(name):
    p = f"{res}/{name}.csv"
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


written = []

# ---- CPU / analytic ------------------------------------------------------
d = load("order_study_raw")
for prob in sorted(d["problem"].unique()) if not d.empty else []:
    written += F.fig_convergence_order(d, fig, problem=prob)

d = load("reuse_ablation_order")
for prob in sorted(d["problem"].unique()) if not d.empty else []:
    written += F.fig_reuse_ablation(d, fig, problem=prob)

for name, fn in [("conditioning_levels", F.fig_conditioning_levels),
                 ("conditioning_precision", F.fig_weight_precision),
                 ("conditioning_block_width", F.fig_block_width)]:
    d = load(name)
    if not d.empty:
        written += fn(d, fig)

written += F.fig_weight_values(fig)

d = load("vcurve")
if not d.empty:
    for L in sorted(d["n_levels"].unique()):
        written += F.fig_vcurve(d, fig, n_levels=int(L))
    written += F.fig_error_vs_levels(d, fig)

scan, search = load("rho_scan"), load("rho_search_toy")
if not scan.empty:
    written += F.fig_rho_scan(scan, fig,
                              search_rows=None if search.empty else search)

# ---- GPU / FID -----------------------------------------------------------
for name, fn, kw in [
    ("fid__validity", F.fig_fid_vs_nfe, dict(tag=None, name="fid_vs_nfe__validity")),
    ("fid__panel", F.fig_fid_vs_nfe, dict(tag=None, name="fid_vs_nfe__panel")),
    ("fid__multilevel", F.fig_multilevel_fid, {}),
    ("fid__multilevel", F.fig_multilevel_precision, {}),
    ("fid__reuse", F.fig_reuse_fid, {}),
    ("fid__seedblock", F.fig_seed_blocks, {}),
]:
    d = load(name)
    if not d.empty:
        written += fn(d, fig, **kw)

allr = load("fid_results")
if not allr.empty and "tag" in allr:
    r = allr[allr["tag"] == "rho_search"]
    if len(r):
        written += F.fig_rho_fid(r, fig)

pdfs = [p for p in written if p.endswith(".pdf")]
print(f"{len(pdfs)} figures rebuilt")

# refresh the PNG previews the README embeds
try:
    import pymupdf, glob
    os.makedirs(f"{fig}/png", exist_ok=True)
    for f in sorted(glob.glob(f"{fig}/*.pdf")):
        n = os.path.splitext(os.path.basename(f))[0]
        pymupdf.open(f)[0].get_pixmap(dpi=160).save(f"{fig}/png/{n}.png")
    print(f"{len(glob.glob(f'{fig}/png/*.png'))} PNG previews refreshed")
except ImportError:
    print("pymupdf not installed; PNG previews not refreshed")
