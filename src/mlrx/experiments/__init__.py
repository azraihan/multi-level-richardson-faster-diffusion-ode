"""Experiment drivers.

Split by cost, because the two halves have very different characters:

CPU-only, seconds to minutes -- :mod:`~mlrx.experiments.toy_order`,
:mod:`~mlrx.experiments.conditioning`.  These use the analytic problem of
:mod:`mlrx.toy`, where the exact solution is known, so error is *measured*
rather than proxied.  They carry the scientific core of the extension.

GPU, hours -- :mod:`~mlrx.experiments.sweeps`, :mod:`~mlrx.experiments.rho_search`.
These produce FID on CIFAR-10 and connect the analysis to the published result.
"""

__all__ = ["conditioning", "sweeps", "toy_order", "rho_search"]
