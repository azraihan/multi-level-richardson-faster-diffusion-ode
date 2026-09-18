"""Rebuild the FID figures from saved CSVs -- no GPU, no re-running."""
import sys, os, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import matplotlib; matplotlib.use("Agg")
from mlrx.plotting import figures as F

out = sys.argv[1]
res, fig = f"{out}/results", f"{out}/figures"
w = []
def load(n):
    p = f"{res}/{n}.csv"
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

for name, fn, kw in [
    ("fid__validity", F.fig_fid_vs_nfe, dict(tag=None, name="fid_vs_nfe__validity")),
    ("fid__panel",    F.fig_fid_vs_nfe, dict(tag=None, name="fid_vs_nfe__panel")),
    ("fid__multilevel", F.fig_multilevel_fid, {}),
    ("fid__multilevel", F.fig_multilevel_precision, {}),
    ("fid__reuse",    F.fig_reuse_fid, {}),
    ("fid__seedblock", F.fig_seed_blocks, {}),
]:
    d = load(name)
    if not d.empty:
        w += fn(d, fig, **kw)
# the rho figure wants the tagged FID rows (rho, fid), not the search log (x, f)
allr = load("fid_results")
if not allr.empty and "tag" in allr:
    r = allr[allr["tag"] == "rho_search"]
    if len(r):
        w += F.fig_rho_fid(r, fig)
print("\n".join(p for p in w if p.endswith(".pdf")))
