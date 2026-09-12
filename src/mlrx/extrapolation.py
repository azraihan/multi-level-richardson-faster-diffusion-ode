"""Richardson extrapolation coefficients on non-uniform grids.

This is the numerical core of the project.  Everything else -- the samplers,
the sweeps, the figures -- exists to exercise or measure what happens here.

Background
----------
Consider one *extrapolation block* covering the time interval
``[t_i, t_{i-K}]`` of total width ``h = t_i - t_{i-K}``, subdivided by the
base grid into ``K`` sub-steps.  A *level* is any subdivision of that block
into ``n`` sub-steps whose normalised widths ``lambda_j`` sum to one.

Following RX-DPM (Choi, Kang & Han, ICLR 2025, Eqs. 16-17 and 20-21), the
numerical solution produced by running the base solver across one level obeys

    X(Lambda) = x* + c * S_p(Lambda) * h^p + O(h^q),          (*)

where ``S_r(Lambda) = sum_j lambda_j ** r`` is the ``r``-th moment of the
sub-step widths and ``p`` is the solver's local order.  The single-step level
has ``Lambda = {1}`` and hence ``S_p = 1``; finer levels have ``S_p < 1``.
The leading error of a level is therefore controlled entirely by ``S_p``, and
the classical uniform-grid ratio ``k^-p`` is just the special case
``S_p = n^(1-p)``.

Two levels let you eliminate ``c`` and recover ``x*`` to ``O(h^q)`` -- that is
exactly RX-DPM's Eq. 18/22, and :func:`two_level_weights` reproduces the
reference implementation's ``get_coeff`` bit for bit.

The extension
-------------
Nothing restricts (*) to two levels.  With ``L`` levels we may posit

    X_n = x* + sum_{m=1..L-1} b_m * S_{p+m-1}(Lambda_n)                (**)

and solve the resulting ``L x L`` system for ``x*``, cancelling ``L-1`` error
terms instead of one.  On a uniform grid the coefficient matrix collapses to a
Vandermonde matrix in the nodes ``1/n``, and the scheme becomes the Romberg
tableau; on the non-uniform grids diffusion models actually use, it does not,
and the moments ``S_r`` must be formed explicitly.

Two things deserve emphasis, because they are assumptions rather than
theorems:

1. Equation (**) extends RX-DPM's own simplification (their Eq. 21, justified
   in their Appendix A) to higher order.  The authors are explicit that the
   linear error-accumulation form "does not hold in general" -- error
   propagates through the Jacobian of the drift, contributing cross terms such
   as ``sum_{j<l} lambda_j^2 lambda_l`` that (**) omits.  We adopt the same
   simplification and then *measure* whether the predicted order materialises,
   using the toy harness in :mod:`mlrx.toy`.  Reporting where it fails is a
   result, not a defect.

2. The coefficient matrix is a generalised Vandermonde matrix, and those are
   notoriously ill-conditioned.  As ``L`` grows the weights stop being small
   and start alternating in sign with large magnitude, so the combination
   ``sum_n w_n X_n`` becomes a difference of nearly-equal large quantities.
   Truncation error falls while round-off rises; total error traces a V.
   Locating the bottom of that V, and showing it moves with working precision,
   is the point of :mod:`mlrx.experiments.multilevel`.

Every routine here therefore reports conditioning alongside its weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import linalg

__all__ = [
    "moment",
    "nested_level_lambdas",
    "build_system",
    "extrapolation_weights",
    "two_level_weights",
    "naive_richardson_weights",
    "Weights",
]


# ---------------------------------------------------------------------------
# moments
# ---------------------------------------------------------------------------

def moment(lambdas, r, dtype=np.float64):
    """``S_r = sum_j lambda_j ** r``, the ``r``-th moment of the sub-step widths.

    ``lambdas`` are normalised widths summing to one, so ``S_1 == 1`` always and
    ``S_r`` decreases as ``r`` grows or as the level is refined.
    """
    dtype = np.dtype(dtype)
    lam = np.asarray(lambdas, dtype=dtype)
    return dtype.type(np.sum(lam.astype(dtype) ** dtype.type(r)))


def nested_level_lambdas(base_lambdas, level_steps):
    """Normalised sub-step widths for each level of a *nested* subdivision.

    The levels must be nested -- each level's breakpoints a subset of the next
    finer level's -- because that is what makes the coarse estimates free: a
    coarse sub-step is a union of consecutive fine sub-steps, so the solver
    never has to visit a time point the fine trajectory did not already visit.

    Parameters
    ----------
    base_lambdas : array_like, shape (K,)
        Normalised widths of the finest (base) subdivision; must sum to one.
    level_steps : sequence of int
        Step counts per level, ascending, each dividing ``K`` and each dividing
        the next.  Typically ``[1, 2, 4, ..., K]``.

    Returns
    -------
    list of ndarray
        ``lambdas[i]`` has length ``level_steps[i]`` and sums to one.
    """
    base = np.asarray(base_lambdas, dtype=np.float64)
    K = base.size
    steps = list(level_steps)

    if not steps:
        raise ValueError("need at least one level")
    if steps != sorted(steps):
        raise ValueError(f"level_steps must be ascending, got {steps}")
    if steps[-1] != K:
        raise ValueError(
            f"finest level has {steps[-1]} steps but the base grid has {K}"
        )
    for n in steps:
        if K % n:
            raise ValueError(f"level with {n} steps does not divide K={K}")
    for a, b in zip(steps, steps[1:]):
        if b % a:
            raise ValueError(f"levels {a} and {b} are not nested")

    out = []
    for n in steps:
        g = K // n
        out.append(np.array([base[j * g:(j + 1) * g].sum() for j in range(n)]))
    return out


# ---------------------------------------------------------------------------
# the weight system
# ---------------------------------------------------------------------------

@dataclass
class Weights:
    """Extrapolation weights plus the diagnostics needed to trust them."""

    w: np.ndarray
    """Weights, ordered coarsest level first.  Sums to one by construction."""

    system: np.ndarray
    """The coefficient matrix that produced them."""

    cond: float
    """``kappa_inf`` of the system -- the round-off amplification factor."""

    growth: float
    """Pivot growth of the LU factorisation."""

    amplification: float
    """``sum_n |w_n|``.  The weights sum to 1, so this is the cancellation
    factor: an input perturbation of size ``eps`` can emerge as
    ``amplification * eps``.  It is the single most directly interpretable
    number in the round-off study."""

    moments: np.ndarray = field(repr=False, default=None)
    """Moment matrix rows actually used, kept for the figures."""

    dtype: np.dtype = np.dtype(np.float64)

    @property
    def n_levels(self):
        return int(self.w.size)

    def apply(self, estimates):
        """Combine per-level estimates, coarsest first.

        Accepts numpy arrays or torch tensors; the weights are cast to the
        estimates' dtype and device so the combination happens in whatever
        precision the caller is studying.
        """
        if len(estimates) != self.w.size:
            raise ValueError(
                f"expected {self.w.size} estimates, got {len(estimates)}"
            )
        out = None
        for coeff, est in zip(self.w, estimates):
            term = est * type(est)(coeff) if np.isscalar(est) else est * coeff
            out = term if out is None else out + term
        return out


def build_system(level_lambdas: Sequence[np.ndarray], p, dtype=np.float64):
    """Coefficient matrix of the extrapolation system, Eq. (**).

    Row ``n`` corresponds to one level; column ``0`` is the constant ``1``
    multiplying ``x*`` and column ``m >= 1`` holds ``S_{p+m-1}`` for that level.

    With ``L`` levels the matrix is ``L x L``: one unknown for the answer and
    ``L-1`` unknowns for the error coefficients being cancelled.
    """
    dtype = np.dtype(dtype)
    L = len(level_lambdas)
    M = np.empty((L, L), dtype=dtype)
    M[:, 0] = dtype.type(1.0)
    for n, lam in enumerate(level_lambdas):
        for m in range(1, L):
            M[n, m] = moment(lam, p + m - 1, dtype=dtype)
    return M


def extrapolation_weights(level_lambdas, p, dtype=np.float64):
    """Solve for the multi-level extrapolation weights.

    The weights are the first row of the inverse of :func:`build_system`:
    writing ``X = M @ [x*, b_1, ..., b_{L-1}]^T``, the answer is
    ``x* = (M^-1 X)_0``, so the coefficients applied to the estimates are
    ``(M^-1)_{0,:}``.

    Parameters
    ----------
    level_lambdas : sequence of array_like
        Normalised sub-step widths per level, coarsest first.  The coarsest
        level is normally ``[1.0]`` (a single step spanning the block).
    p : int
        Local order of the base solver.  ``p = 2`` for Euler / DDIM,
        ``p = n + 1`` for a single-step DPM-Solver-``n``, ``p = 3`` and ``p = 5``
        for S-PNDM and F-PNDM respectively (RX-DPM Sec. 5.5).
    dtype : numpy dtype
        Working precision for building *and* solving the system.

    Returns
    -------
    Weights
    """
    dtype = np.dtype(dtype)
    M = build_system(level_lambdas, p, dtype=dtype)
    L = M.shape[0]

    rhs = np.zeros(L, dtype=dtype)
    rhs[0] = dtype.type(1.0)
    # First row of M^-1 is the solution of M^T y = e_0.
    w = linalg.solve(M.T, rhs, dtype=dtype)

    return Weights(
        w=w,
        system=M,
        cond=linalg.cond(M, dtype=dtype),
        growth=linalg.growth_factor(M, dtype=dtype),
        amplification=float(np.sum(np.abs(w.astype(np.float64)))),
        moments=M[:, 1:].copy(),
        dtype=dtype,
    )


def two_level_weights(sub_lambdas, p=2, dtype=np.float64):
    """RX-DPM's original two-level rule, Eq. 18 (``p=2``) and Eq. 22 (general).

    Equivalent to ``extrapolation_weights([[1.0], sub_lambdas], p)`` but written
    in closed form, both as documentation and as an independent check on the
    general solver.  ``sub_lambdas`` are the normalised widths of the ``K``
    base sub-steps spanning the block.

    Returns weights ordered ``[coarse (1 step), fine (K steps)]``, matching

        x_tilde = (X_fine - S_p * X_coarse) / (1 - S_p).
    """
    dtype = np.dtype(dtype)
    S = moment(sub_lambdas, p, dtype=dtype)
    one = dtype.type(1.0)
    denom = one - S
    w = np.array([-S / denom, one / denom], dtype=dtype)
    M = np.array([[one, one], [one, S]], dtype=dtype)
    return Weights(
        w=w,
        system=M,
        cond=linalg.cond(M, dtype=dtype),
        growth=linalg.growth_factor(M, dtype=dtype),
        amplification=float(np.sum(np.abs(w.astype(np.float64)))),
        moments=M[:, 1:].copy(),
        dtype=dtype,
    )


def naive_richardson_weights(k, p=2, dtype=np.float64):
    """Classical uniform-grid Richardson extrapolation, RX-DPM Eq. 8.

    This is the ``Naive`` baseline of the paper's Figure 2.  It assumes the
    fine estimate was obtained with uniform step ``h/k`` and therefore carries
    error ``c (h/k)^p``, giving the fixed coefficients

        V_tilde = (k^p V(h/k) - V(h)) / (k^p - 1).

    Note that this does *not* agree with :func:`two_level_weights` even on a
    uniform grid: there, ``S_p = k^(1-p)``, so the grid-aware rule for ``k=2,
    p=2`` weights the pair ``[-1, 2]`` whereas this one gives ``[-1/3, 4/3]``.
    The discrepancy is the difference between modelling the fine estimate's
    error as a single step of width ``h/k`` and modelling it as ``k``
    accumulated local errors -- and it is precisely what the paper's Figure 2
    ablation is measuring.

    Returns weights ordered ``[coarse, fine]``.
    """
    dtype = np.dtype(dtype)
    kp = dtype.type(float(k) ** p)
    one = dtype.type(1.0)
    denom = kp - one
    w = np.array([-one / denom, kp / denom], dtype=dtype)
    M = np.array([[one, one], [one, one / kp]], dtype=dtype)
    return Weights(
        w=w,
        system=M,
        cond=linalg.cond(M, dtype=dtype),
        growth=linalg.growth_factor(M, dtype=dtype),
        amplification=float(np.sum(np.abs(w.astype(np.float64)))),
        moments=M[:, 1:].copy(),
        dtype=dtype,
    )
