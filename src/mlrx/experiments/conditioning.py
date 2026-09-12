"""Conditioning of the extrapolation weight system.

Pure numerics -- no network, no GPU, runs in seconds.  These are the
measurements that explain *why* the multi-level extension behaves as it does,
and they are independent of any particular dataset or model, which makes them
the most portable results in the project.

Three quantities are tracked throughout:

``cond``
    ``kappa_inf`` of the coefficient matrix.  Bounds how much a relative
    perturbation of the inputs can be amplified in the solution.
``amplification``
    ``sum_n |w_n|``.  Since the weights sum to one by construction, this is the
    cancellation factor directly: a perturbation ``eps`` in the level estimates
    can emerge as ``amplification * eps`` in the combination.  It is the more
    interpretable of the two and the one to quote.
``growth``
    Pivot growth of the LU factorisation.  Included to demonstrate that the
    ill-conditioning is a property of the *problem* and not an artefact of our
    elimination -- it stays at 1.0 throughout, so partial pivoting is doing its
    job and the conditioning reported is genuine.
"""

from __future__ import annotations

import numpy as np

from .. import extrapolation as ex
from .. import linalg, schedules

__all__ = [
    "run_level_study",
    "run_rho_study",
    "run_precision_study",
    "run_block_width_study",
]


def _block_levels(K, L, rho=7.0, num_steps=None, block_index=0):
    """Level widths for block ``block_index`` of an EDM schedule."""
    num_steps = num_steps or max(K * 4, 32)
    t = schedules.edm_schedule(num_steps, rho=rho)
    s = block_index * K
    blk = t[s:s + K + 1]
    base = schedules.block_lambdas(blk)
    return ex.nested_level_lambdas(base, schedules.nested_steps(K, L)), blk


def run_level_study(level_counts=(2, 3, 4, 5, 6, 7), rho=7.0, p=2,
                    num_steps=128, block_index=0):
    """Conditioning versus number of levels.

    ``K`` is set to ``2^(L-1)``, the smallest block admitting ``L`` nested
    levels.  This coupling is unavoidable and is itself part of the story: more
    levels *require* wider blocks.
    """
    rows = []
    for L in level_counts:
        K = 2 ** (L - 1)
        if K * (block_index + 1) > num_steps:
            continue
        lv, blk = _block_levels(K, L, rho, num_steps, block_index)
        W = ex.extrapolation_weights(lv, p=p)
        rows.append({
            "n_levels": L,
            "frequency": K,
            "rho": rho,
            "p": p,
            "cond": W.cond,
            "amplification": W.amplification,
            "growth": W.growth,
            "max_abs_weight": float(np.max(np.abs(W.w))),
            "sum_weights": float(W.w.sum()),
            "sigma_hi": float(blk[0]),
            "sigma_lo": float(blk[-1]),
            "weights": ";".join(f"{v:.6g}" for v in W.w),
        })
    return rows


def run_rho_study(rhos=None, level_counts=(2, 3, 4, 5), p=2, num_steps=128,
                  block_index=0):
    """Conditioning versus the schedule exponent ``rho``.

    Motivates searching over ``rho``: the weights are computed from the step
    widths, so ``rho`` -- chosen by EDM for plain Euler and Heun -- also sets
    how well conditioned our weight system is.  There is no reason the value
    that suits one should suit the other.
    """
    if rhos is None:
        rhos = np.round(np.arange(1.0, 15.01, 0.5), 3)
    rows = []
    for rho in rhos:
        for L in level_counts:
            K = 2 ** (L - 1)
            if K * (block_index + 1) > num_steps:
                continue
            try:
                lv, blk = _block_levels(K, L, float(rho), num_steps, block_index)
                W = ex.extrapolation_weights(lv, p=p)
            except Exception:
                continue
            lam = lv[-1]
            rows.append({
                "rho": float(rho),
                "n_levels": L,
                "frequency": K,
                "cond": W.cond,
                "amplification": W.amplification,
                "lambda_spread": float(lam.max() / lam.min()),
                "sigma_hi": float(blk[0]),
                "sigma_lo": float(blk[-1]),
            })
    return rows


def run_precision_study(level_counts=(2, 3, 4, 5, 6, 7),
                        precisions=("float64", "float32", "float16"),
                        rho=7.0, p=2, num_steps=128):
    """Error in the *weights themselves* at reduced precision.

    Isolates one link in the chain: before any question of how the weights are
    applied, are they even computed correctly?  Each precision's weights are
    compared against the float64 result for the same system.  This is where the
    round-off story starts, and it needs no ODE solve at all.
    """
    rows = []
    for L in level_counts:
        K = 2 ** (L - 1)
        if K > num_steps:
            continue
        lv, _ = _block_levels(K, L, rho, num_steps, 0)
        ref = ex.extrapolation_weights(lv, p=p, dtype=np.float64)
        for prec in precisions:
            dt = np.dtype(prec)
            try:
                W = ex.extrapolation_weights(lv, p=p, dtype=dt)
                err = float(np.max(np.abs(W.w.astype(np.float64) - ref.w)))
                rel = err / max(float(np.max(np.abs(ref.w))), 1e-300)
                failed = ""
            except Exception as exc:
                err = rel = float("nan")
                W = None
                failed = repr(exc)
            rows.append({
                "n_levels": L,
                "frequency": K,
                "precision": prec,
                "eps": float(np.finfo(dt).eps),
                "weight_abs_error": err,
                "weight_rel_error": rel,
                "cond": ref.cond,
                "amplification": ref.amplification,
                "predicted_error": float(np.finfo(dt).eps) * ref.cond,
                "failed": failed,
            })
    return rows


def run_block_width_study(frequencies=(2, 4, 8, 16, 32), rho=7.0, p=2,
                          num_steps=128):
    """Conditioning versus block width and position along the trajectory.

    Answers a question the level study cannot: is the conditioning driven by
    the *number of levels*, or by the extreme spread of step widths that a wide
    block on a geometric schedule produces?  Reporting both separates the two
    and prevents attributing one effect to the other.
    """
    rows = []
    for K in frequencies:
        n_blocks = num_steps // K
        for b in range(n_blocks):
            t = schedules.edm_schedule(num_steps, rho=rho)
            blk = t[b * K:b * K + K + 1]
            base = schedules.block_lambdas(blk)
            for L in range(2, int(np.log2(K)) + 2):
                try:
                    lv = ex.nested_level_lambdas(
                        base, schedules.nested_steps(K, L))
                    W = ex.extrapolation_weights(lv, p=p)
                except Exception:
                    continue
                rows.append({
                    "frequency": K,
                    "block_index": b,
                    "n_blocks": n_blocks,
                    "n_levels": L,
                    "rho": rho,
                    "sigma_hi": float(blk[0]),
                    "sigma_lo": float(blk[-1]),
                    "lambda_spread": float(base.max() / base.min()),
                    "cond": W.cond,
                    "amplification": W.amplification,
                    "growth": W.growth,
                })
    return rows
