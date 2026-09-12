"""Derivative-free one-dimensional minimisation.

Used to tune the schedule exponent ``rho``.  Golden-section search is the right
tool here for reasons worth stating, since the choice is part of the course
content: the objective is FID, which has no usable derivative, costs minutes per
evaluation, and carries Monte-Carlo noise.  Golden-section needs no derivative,
makes the smallest number of evaluations of any bracketing method for a given
final interval, and -- unlike a parabolic or Newton method -- cannot be thrown
off a cliff by curvature estimated from noisy samples.

Every objective evaluation is logged, so the search history itself can be
plotted rather than just its answer.

Reference: Chapra & Canale, *Numerical Methods for Engineers*, Sec. 13.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["golden_section_search", "GoldenSectionResult"]

INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0          # 0.6180339887...
INV_PHI2 = (3.0 - math.sqrt(5.0)) / 2.0         # 1 - INV_PHI


@dataclass
class GoldenSectionResult:
    x: float
    f: float
    n_eval: int
    bracket: tuple
    history: list = field(default_factory=list)
    converged: bool = False

    def as_rows(self):
        """Evaluation log, ready for a CSV."""
        return [
            {"iteration": i, "x": x, "f": f, "a": a, "b": b}
            for i, (x, f, a, b) in enumerate(self.history)
        ]


def golden_section_search(f, a, b, tol=1e-2, max_eval=40, cache=True,
                          verbose=False):
    """Minimise a unimodal ``f`` on ``[a, b]``.

    The interval shrinks by a factor ``0.618`` per evaluation after the first
    two, so the evaluation count for a target tolerance is known in advance --
    which matters when each evaluation costs a FID run and the total budget has
    to be quoted up front.

    Parameters
    ----------
    f : callable
        Objective.  Assumed unimodal on ``[a, b]``; see the caveat below.
    a, b : float
    tol : float
        Stop once the bracket is narrower than this.
    max_eval : int
        Hard cap on objective evaluations.
    cache : bool
        Memoise on the rounded argument, so repeated probes of the same point
        cost nothing.
    verbose : bool

    Returns
    -------
    GoldenSectionResult

    Notes
    -----
    Unimodality is an assumption, not a guarantee.  With a noisy objective the
    method converges to *a* local minimum of the sampled surface, and the
    returned bracket should be read as "the best region found", not as a
    certified global optimum.  The logged history lets a reader judge that for
    themselves, which is why it is reported alongside the answer.
    """
    if a > b:
        a, b = b, a
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("bracket must be finite")

    history = []
    memo = {}

    def evaluate(x):
        key = round(float(x), 10)
        if cache and key in memo:
            return memo[key]
        val = float(f(x))
        memo[key] = val
        history.append((float(x), val, float(a), float(b)))
        if verbose:
            print(f"  [golden] f({x:.6g}) = {val:.6g}   bracket=[{a:.4g}, {b:.4g}]")
        return val

    lo, hi = a, b
    x1 = lo + INV_PHI2 * (hi - lo)
    x2 = lo + INV_PHI * (hi - lo)
    f1, f2 = evaluate(x1), evaluate(x2)
    converged = False

    while len(memo) < max_eval:
        if abs(hi - lo) < tol:
            converged = True
            break
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = lo + INV_PHI2 * (hi - lo)
            f1 = evaluate(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + INV_PHI * (hi - lo)
            f2 = evaluate(x2)

    best_x, best_f = min(((x, v) for x, v, _, _ in history), key=lambda t: t[1])
    return GoldenSectionResult(
        x=best_x, f=best_f, n_eval=len(memo),
        bracket=(lo, hi), history=history, converged=converged,
    )
