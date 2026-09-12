"""Dense linear algebra written from first principles.

The extrapolation weight system (see :mod:`mlrx.extrapolation`) is a small
generalised Vandermonde system.  We deliberately solve it with our own
Gaussian elimination / LU factorisation instead of calling out to LAPACK:

* the pivoting strategy is explicit and auditable,
* the growth factor and condition number fall out of the same factorisation,
* the whole thing can be re-run at a chosen working precision, which is what
  the round-off study needs.

Everything here operates on small matrices (order <= 8 in practice), so the
straightforward triple-loop implementations are entirely adequate; clarity is
worth more than speed at this size.

References
----------
Golub & Van Loan, *Matrix Computations*, 4th ed., Sections 3.2 (LU),
3.4 (pivoting, growth factor) and 2.7.2 (condition estimation).
Chapra & Canale, *Numerical Methods for Engineers*, Chapters 9-10.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "lu_factor",
    "lu_solve",
    "solve",
    "inv",
    "cond",
    "growth_factor",
    "LUFactorization",
]


class LUFactorization:
    """Result of :func:`lu_factor`: ``P @ A == L @ U`` in the working dtype.

    The factors are held in the usual compact form -- the strict lower triangle
    of :attr:`lu` stores the multipliers of ``L`` (whose diagonal is implicitly
    one) and the upper triangle stores ``U``.
    """

    __slots__ = ("lu", "piv", "swaps", "dtype", "_growth")

    def __init__(self, lu, piv, swaps, dtype, growth):
        self.lu = lu
        self.piv = piv
        self.swaps = swaps
        self.dtype = dtype
        self._growth = growth

    @property
    def L(self):
        n = self.lu.shape[0]
        out = np.eye(n, dtype=self.dtype)
        for i in range(1, n):
            out[i, :i] = self.lu[i, :i]
        return out

    @property
    def U(self):
        return np.triu(self.lu)

    @property
    def P(self):
        n = self.lu.shape[0]
        return np.eye(n, dtype=self.dtype)[self.piv]

    @property
    def growth(self):
        """Pivot growth factor ``max|U| / max|A|`` -- see :func:`growth_factor`."""
        return self._growth

    def det(self):
        d = self.dtype.type(-1.0) ** self.swaps
        for i in range(self.lu.shape[0]):
            d = d * self.lu[i, i]
        return d


def _as_working(a, dtype):
    return np.asarray(a, dtype=dtype)


def lu_factor(A, dtype=np.float64):
    """LU factorisation with partial (row) pivoting.

    Partial pivoting is not optional here.  The Vandermonde-like systems we
    build become badly scaled as the number of extrapolation levels grows, and
    unpivoted elimination on them loses digits catastrophically -- the very
    effect the round-off study is meant to measure, so it must not be
    contaminated by an avoidable implementation artefact.

    Parameters
    ----------
    A : array_like, shape (n, n)
    dtype : numpy dtype
        Working precision.  The round-off study drives this with
        ``float64`` / ``float32`` / ``float16``.

    Returns
    -------
    LUFactorization

    Raises
    ------
    ValueError
        If ``A`` is not square, or is exactly singular to working precision.
    """
    dtype = np.dtype(dtype)
    a = _as_working(A, dtype).copy()
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {a.shape}")
    n = a.shape[0]

    max_a = float(np.max(np.abs(a))) if n else 0.0
    piv = np.arange(n)
    swaps = 0

    for k in range(n - 1):
        # --- pivot search -------------------------------------------------
        p = k + int(np.argmax(np.abs(a[k:, k])))
        if a[p, k] == 0:
            raise ValueError(
                f"matrix is exactly singular at column {k} in {dtype.name}"
            )
        if p != k:
            a[[k, p], :] = a[[p, k], :]
            piv[[k, p]] = piv[[p, k]]
            swaps += 1

        # --- elimination --------------------------------------------------
        for i in range(k + 1, n):
            m = a[i, k] / a[k, k]
            a[i, k] = m                      # store the multiplier in place
            if m != 0:
                a[i, k + 1:] -= m * a[k, k + 1:]

    if n and a[n - 1, n - 1] == 0:
        raise ValueError(f"matrix is exactly singular in {dtype.name}")

    max_u = float(np.max(np.abs(np.triu(a)))) if n else 0.0
    growth = (max_u / max_a) if max_a > 0 else float("inf")
    return LUFactorization(a, piv, swaps, dtype, growth)


def lu_solve(fac: LUFactorization, b):
    """Solve ``A x = b`` from a precomputed factorisation."""
    dtype = fac.dtype
    rhs = _as_working(b, dtype)
    vector = rhs.ndim == 1
    if vector:
        rhs = rhs.reshape(-1, 1)
    n = fac.lu.shape[0]
    if rhs.shape[0] != n:
        raise ValueError(f"rhs has {rhs.shape[0]} rows, expected {n}")

    x = rhs[fac.piv].copy()

    # forward substitution, L has unit diagonal
    for i in range(1, n):
        x[i] -= fac.lu[i, :i] @ x[:i]

    # back substitution
    for i in range(n - 1, -1, -1):
        if i + 1 < n:
            x[i] -= fac.lu[i, i + 1:] @ x[i + 1:]
        x[i] /= fac.lu[i, i]

    return x[:, 0] if vector else x


def solve(A, b, dtype=np.float64):
    """Solve ``A x = b`` by LU with partial pivoting, at the given precision."""
    return lu_solve(lu_factor(A, dtype=dtype), b)


def inv(A, dtype=np.float64):
    """Explicit inverse, via ``n`` triangular solves against the identity.

    Only used for small matrices and for condition estimation.  Forming an
    inverse is normally poor practice, but here ``n <= 8`` and we genuinely
    want the inverse itself: its first row *is* the vector of extrapolation
    weights.
    """
    dtype = np.dtype(dtype)
    fac = lu_factor(A, dtype=dtype)
    n = fac.lu.shape[0]
    return lu_solve(fac, np.eye(n, dtype=dtype))


def cond(A, dtype=np.float64, p=np.inf):
    """Condition number ``kappa_p(A) = ||A||_p * ||A^-1||_p``.

    Computed from an explicit inverse rather than estimated.  At these sizes
    that is affordable and exact up to round-off, which matters because the
    condition number is a reported quantity in this project, not just an
    internal guard.
    """
    dtype = np.dtype(dtype)
    a = _as_working(A, dtype)
    try:
        ainv = inv(a, dtype=dtype)
    except ValueError:
        return float("inf")
    return float(_norm(a, p) * _norm(ainv, p))


def _norm(a, p):
    a64 = np.asarray(a, dtype=np.float64)
    if p == np.inf:
        return np.max(np.sum(np.abs(a64), axis=1)) if a64.size else 0.0
    if p == 1:
        return np.max(np.sum(np.abs(a64), axis=0)) if a64.size else 0.0
    if p == 2:
        return np.linalg.norm(a64, 2)
    if p == "fro":
        return np.linalg.norm(a64, "fro")
    raise ValueError(f"unsupported norm {p!r}")


def growth_factor(A, dtype=np.float64):
    """Pivot growth ``max|U| / max|A|`` for partial-pivoted LU.

    A growth factor far above one is the fingerprint of an elimination that has
    amplified its inputs, and therefore of an answer with fewer correct digits
    than the residual alone would suggest.
    """
    return lu_factor(A, dtype=dtype).growth
