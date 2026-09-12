"""Multi-level Richardson extrapolation for diffusion ODE sampling.

Reproduction of RX-DPM (Choi, Kang & Han, ICLR 2025) and an extension to
multiple nested extrapolation levels, with an accompanying conditioning and
round-off analysis.

CSE 402 -- Numerical Analysis, Simulation and Modeling Sessional, BUET.

Layout
------
``linalg``          LU with partial pivoting, condition numbers (ours, not LAPACK's)
``regression``      least squares via normal equations, for order fitting
``optimize``        golden-section search
``extrapolation``   the weight system -- the numerical core
``schedules``       EDM time-step schedules and block partitioning
``samplers``        Euler, Heun, RX-DPM, multi-level RX
``toy``             analytic Gaussian-mixture diffusion ODE with known solution
``edm_model``       pretrained EDM network loading and adaptation
``fid``             offline, single-process FID
``runner``          resumable, multi-GPU sweep execution
``experiments``     the studies themselves
``plotting``        house style and figure builders
``pipeline``        end-to-end stage drivers
"""

__version__ = "1.0.0"

__all__ = [
    "linalg", "regression", "optimize", "extrapolation", "schedules",
    "samplers", "toy", "edm_model", "fid", "runner", "experiments",
    "plotting", "pipeline",
]
