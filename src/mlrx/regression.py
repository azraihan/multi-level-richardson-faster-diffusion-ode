"""Least-squares fitting, via normal equations solved with our own LU.

Used to extract empirical convergence orders: if the error behaves as
``E(h) = C h^p`` then ``log E = log C + p log h`` is linear, and the fitted
slope is the observed order of accuracy.

Solving the normal equations ``(A^T A) c = A^T y`` is the textbook approach and
is what the course covers, so that is what is implemented.  It is worth being
honest that it is *not* the most numerically sound choice in general: forming
``A^T A`` squares the condition number, and a QR or SVD factorisation of ``A``
avoids that.  For the low-degree fits here (degree 1, a handful of points,
well-scaled ``log h``) the difference is immaterial, and :func:`polyfit`
reports the condition number of ``A^T A`` so a reader can confirm that for
themselves rather than take it on trust.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import linalg

__all__ = ["polyfit", "PolyFit", "fit_order"]


@dataclass
class PolyFit:
    coeffs: np.ndarray
    """Ascending powers: ``coeffs[j]`` multiplies ``x**j``."""
    r2: float
    residual_norm: float
    cond_normal: float
    n: int

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        return sum(c * x ** j for j, c in enumerate(self.coeffs))

    @property
    def slope(self):
        """Degree-1 slope, i.e. the fitted exponent in a log-log fit."""
        if self.coeffs.size < 2:
            raise ValueError("slope is defined for degree >= 1")
        return float(self.coeffs[1])


def polyfit(x, y, degree=1):
    """Least-squares polynomial fit through the normal equations."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size < degree + 1:
        raise ValueError(f"need at least {degree + 1} points for degree {degree}")

    A = np.vander(x, degree + 1, increasing=True)
    ATA, ATy = A.T @ A, A.T @ y

    coeffs = linalg.solve(ATA, ATy, dtype=np.float64)
    resid = y - A @ coeffs
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())

    return PolyFit(
        coeffs=coeffs,
        r2=(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        residual_norm=float(np.sqrt(ss_res)),
        cond_normal=linalg.cond(ATA),
        n=int(x.size),
    )


def fit_order(h, err, drop_nonfinite=True, min_points=3):
    """Fit ``err ~ C h^p`` and return ``(p, PolyFit)``.

    Non-positive or non-finite errors are dropped: they arise when a method
    reaches round-off saturation and the measured error stops being a function
    of ``h`` at all.  Including those points would drag the fitted slope toward
    zero and misreport the order, so they are excluded and the caller is left
    to notice that fewer points were used.
    """
    h = np.asarray(h, dtype=np.float64).ravel()
    err = np.asarray(err, dtype=np.float64).ravel()

    mask = np.ones(h.shape, dtype=bool)
    if drop_nonfinite:
        mask &= np.isfinite(err) & (err > 0) & np.isfinite(h) & (h > 0)
    if mask.sum() < min_points:
        return float("nan"), None

    fit = polyfit(np.log(h[mask]), np.log(err[mask]), degree=1)
    return fit.slope, fit
