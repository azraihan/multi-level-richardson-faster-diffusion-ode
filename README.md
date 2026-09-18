# Multi-Level Richardson Extrapolation for Diffusion ODE Sampling

Reproduction and extension of **RX-DPM** — Choi, Kang & Han, *Enhanced Diffusion
Sampling via Extrapolation with Multiple ODE Solutions*, ICLR 2025
([arXiv:2504.01855](https://arxiv.org/abs/2504.01855)).

CSE 402 — Numerical Analysis, Simulation and Modeling Sessional, BUET.

---

## What this is

Diffusion models generate images by numerically solving an ODE. Each solver step
costs one neural-network evaluation, so sample quality at a fixed budget is a
question about truncation error — squarely a numerical-methods problem.

RX-DPM applies **Richardson extrapolation** to that ODE: it combines a coarse
one-step estimate with a fine multi-step estimate of the same interval to cancel
the leading error term, generalising the classical uniform-grid coefficients to
the non-uniform schedules diffusion models actually use. Crucially it is *free* —
the coarse estimate reuses an evaluation the fine solver already made.

This repository reproduces that, then asks the obvious next question: **why stop
at two levels?** Classical Romberg integration stacks extrapolations into a
tableau, each column cancelling one more error term. We build the same tableau on
non-uniform grids, solve the resulting generalised Vandermonde system with our own
LU factorisation, and measure what actually happens.

## The main finding

**The free-of-charge property is specific to two levels and does not generalise.**

At two levels nothing is approximated: the coarse estimate's only evaluation sits
at the block's start, where the coarse and fine trajectories still coincide. At
three or more levels the intermediate estimates need evaluations at points the
fine trajectory visits *in a different state*. Reusing them anyway keeps the cost
at zero — and caps the achievable local order at ≈3.2 no matter how many levels
are added. Paying for exact recomputation recovers the predicted order.

Measured on an analytic problem with a known exact solution (`mlrx.toy`), fitting
local order against block width, r² ≈ 0.999:

| L | reuse (0 extra NFE) | exact (extra NFE) | predicted `p+L-1` | exact's extra NFE/block |
|---:|---:|---:|---:|---:|
| 2 | 3.22 | 3.22 | 3 | 0 |
| 3 | 3.21 | 4.28 | 4 | 1 |
| 4 | 3.22 | 5.36 | 5 | 4 |
| 5 | 3.22 | 5.90 | 6 | 11 |

Identical at L=2, as theory demands; divergent thereafter. The smooth
single-Gaussian problem shows the same pattern (reuse flat at 3.31, exact
reaching 4.36 / 5.67). Supporting results:

* **Conditioning** of the weight system grows roughly an order of magnitude per
  level — κ∞ = 8.0, 60, 579, 7.4e3, 9.8e4, 6.1e5 for L = 2…7 — while the LU
  **growth factor stays at 1.00**, so the ill-conditioning is intrinsic to the
  problem rather than an artefact of our elimination.
* **In float16 the weights themselves** are already ~4% wrong at L=4 and useless
  at L=6, so reduced precision imposes an error *floor* rather than shifting an
  optimum.
* **The truncation/round-off V-curve** appears along the step count. In float64
  at L=2 the error bottoms out at N = 256 (1.53e-6) and climbs back to 3.81e-6 by
  N = 2048; float32 tracks float64 almost exactly, while float16 never gets below
  ≈1.4e-3 however fine the grid.
* **ρ = 7**, EDM's hand-picked schedule exponent, is near-optimal for Euler
  (dense scan: 6.25) but not for RX (8.0) — though the objective is not cleanly
  unimodal, so golden-section search must be read with care.

## Results on CIFAR-10

Measured on an RTX PRO 6000, 10,000 images per FID, conditional CIFAR-10, EDM
backbone, common random numbers across methods. ~2h20m for 80 configurations.

**Validation gate** — mean ± std over 3 independent seed blocks:

| Method | NFE | ours (10k imgs) | paper (50k imgs) |
|---|---:|---:|---:|
| Euler | 10 | 17.48 ± 0.30 | 15.88 |
| Heun (EDM) | 11 | 16.45 ± 0.15 | 14.46 |
| **RX-Euler (k=2)** | 10 | **6.15 ± 0.04** | 4.35 |
| RX+EDM | 10 | 10.72 ± 0.07 | 4.26 |

The reproduction holds: RX-Euler cuts FID by **2.8×** against Euler at equal
NFE, and beats Heun even though Heun is given an extra evaluation. Absolute
values sit above the published ones for the sample-size reason below, and the
ordering is preserved. RX+EDM is the one disagreement — we find it *worse* than
plain RX-Euler, where the paper finds it slightly better; our Heun/RX split is a
reimplementation choice the paper does not fully specify.

**Grid-aware coefficients beat the classical fixed ones** at every budget,
reproducing the paper's Fig. 2 ablation (FID, k=2):

| NFE | 6 | 8 | 10 | 12 | 16 | 20 |
|---|---:|---:|---:|---:|---:|---:|
| Naïve Richardson | 30.21 | 18.30 | 13.41 | 10.65 | 7.91 | 6.60 |
| Grid-aware (RX) | 33.27 | **8.21** | **6.12** | **5.47** | **4.86** | **4.57** |

**The extension fails on real data too, and by a wide margin.** FID at NFE 10:
L=2 → 6.12, L=3 → 66.3, L=4 → 113.2. Exact recomputation helps but never
recovers the cost: L=3 exact reaches 6.35 at NFE 20, still worse than L=2 reuse
at 4.57 for the same budget. At L=2, reuse and exact agree to the last digit
(6.12/6.12 at NFE 10, 4.86/4.86 at NFE 16) — the same identity the analytic
study predicts, now confirmed on a real network.

**Precision is a null result.** Dropping the extrapolation arithmetic to
float32 changes FID by under 0.1%, and float16 by at most a couple of percent
at the only usable setting (L=2). The analytic study explains why: at realistic
step counts truncation dominates and round-off in the weights never becomes the
limiting error. Round-off only matters once the grid is refined far past
anything a sampler would use.

**ρ = 7 is not optimal for RX.** Golden-section search over ρ ∈ [1, 15] under
FID settles at **ρ ≈ 6.16 (FID 5.95)** against 6.12 at ρ = 7 — a real but small
2.7% gain, consistent with the analytic scan. The objective is not cleanly
unimodal, so this is best read as "the good region is near 6", not as a
certified minimiser.

## Correctness

`tests/test_reference_equivalence.py` transcribes the authors' published sampler
and `get_coeff` verbatim and asserts our rewrite matches it **to < 1e-12 across
eight step/frequency combinations**, with identical NFE. Our general L-level
solver reduces to their closed form exactly at L=2.

```
PYTHONPATH=src python -m pytest tests -q
```

## Layout

```
src/mlrx/
  linalg.py         LU with partial pivoting, condition number, growth factor
  regression.py     least squares via normal equations, for order fitting
  optimize.py       golden-section search
  extrapolation.py  the weight system — the numerical core
  schedules.py      EDM schedules, block partitioning, nested levels
  samplers.py       Euler, Heun, RX-DPM, multi-level RX
  toy.py            analytic Gaussian-mixture diffusion ODE with exact solution
  edm_model.py      pretrained EDM network loading and adaptation
  fid.py            offline, single-process FID
  runner.py         resumable, multi-GPU sweep execution
  experiments/      the studies
  plotting/         house style and figure builders
  pipeline.py       CPU and GPU stage drivers
notebooks/
  01_prefetch_online.ipynb      run once, Internet ON — vendors ~340 MB of assets
  02_run_pipeline_offline.ipynb the full pipeline, no network required
```

## Running it on Kaggle

1. Run **`01_prefetch_online`** with Internet **ON** (any accelerator). It clones
   this repo and NVlabs/edm, downloads the EDM CIFAR-10 checkpoint, the StyleGAN3
   Inception-v3 pickle and the FID reference statistics, vendors pip wheels, and
   writes `manifest.json`. Save & Run All.
2. In **`02_run_pipeline_offline`**, add that notebook's output via
   **+ Add Input → Your Work → Notebooks**, set `PREFETCH_NAME` to its slug (or
   leave `None` to auto-discover), and run. No network needed.

The GPU stage is **resumable** — completed configurations are appended to
`results/fid_results.csv` and skipped on re-run, so a session that times out just
needs running again. A throughput probe measures the real rate on your hardware
and projects the total before the sweep starts.

## Outputs

```
outputs/results/*.csv       every number, one table per study
outputs/figures/*.pdf       one caption-less vector figure per result
outputs/figures/data/*.csv  the exact numbers behind each figure
```

Figures carry **no titles or captions** — captions belong in the report — and are
vector PDFs with embedded editable text (`pdf.fonttype=42`). Every figure ships
the CSV it was drawn from, so any of them can be restyled without re-running an
experiment.

## A note on FID numbers

We use **10,000 images per FID**, not the paper's 50,000. FID is a biased
estimator and the bias grows as the sample count falls, so our absolute values
read **higher** than published ones. They remain internally comparable — every
configuration uses the same count and the same latents (common random numbers) —
but should not be set directly against the paper's 15.88 / 14.46 / 4.35 / 4.26.
`mlrx.fid.fid_sample_size_curve` measures the bias directly.

## Attribution

`refcode/` and `tests/test_reference_equivalence.py` contain code from the
official RX-DPM release (`github.com/jin01020/rx-dpm`) and from
[NVlabs/edm](https://github.com/NVlabs/edm), which is licensed CC BY-NC-SA 4.0.
Pretrained checkpoints and FID statistics are NVIDIA's, under the same licence.

```bibtex
@inproceedings{choi2025rxdpm,
  author    = {Choi, Jinyoung and Kang, Junoh and Han, Bohyung},
  title     = {Enhanced Diffusion Sampling via Extrapolation with Multiple ODE Solutions},
  booktitle = {ICLR}, year = {2025}
}
@inproceedings{karras2022edm,
  author    = {Karras, Tero and Aittala, Miika and Aila, Timo and Laine, Samuli},
  title     = {Elucidating the Design Space of Diffusion-Based Generative Models},
  booktitle = {NeurIPS}, year = {2022}
}
```
