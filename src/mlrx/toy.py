"""An analytically solvable diffusion ODE, for order verification.

Measuring the empirical order of accuracy of a sampler requires knowing the
exact solution, which a neural denoiser never gives you.  The usual dodge is to
verify the solver on some unrelated test ODE and hope the conclusion transfers.
We can do better: if the data distribution is a Gaussian mixture, the exact
denoiser is available in closed form, and the probability-flow ODE being solved
is then *literally* the one EDM solves -- same equation, same schedule, same
sampler code -- only with an analytic ``D`` substituted for the network.

Derivation
----------
For data ``p_data = sum_c pi_c N(mu_c, s_c^2 I)``, convolving with noise of
scale ``sigma`` keeps the mixture Gaussian:

    p_sigma = sum_c pi_c N(mu_c, (s_c^2 + sigma^2) I).

Its score is a responsibility-weighted average of the per-component scores,

    grad log p_sigma(x) = sum_c gamma_c(x) (mu_c - x) / (s_c^2 + sigma^2),
    gamma_c(x)          = pi_c N(x; mu_c, (s_c^2+sigma^2) I) / p_sigma(x),

and EDM's denoiser and probability-flow ODE (Karras et al. Eqs. 3 and 4, with
``s(t) = 1``, ``sigma(t) = t``) follow immediately:

    D(x; sigma) = x + sigma^2 grad log p_sigma(x),
    dx/dsigma   = (x - D(x; sigma)) / sigma = -sigma grad log p_sigma(x).

For a *single* component the responsibilities are identically one and the ODE
separates, giving a closed-form solution

    x(sigma) - mu = (x(sigma_0) - mu) sqrt( (s^2 + sigma^2) / (s^2 + sigma_0^2) ),

which is what :class:`ToyProblem.exact` returns.  Genuine mixtures have no such
form; there we integrate to high accuracy with a small-step RK4 reference
instead (:meth:`ToyProblem.reference`).

Mixtures matter because the single-Gaussian case is *linear* in ``x``, and a
linear ODE flatters extrapolation: the higher-order error terms that
:mod:`mlrx.extrapolation`'s model omits are exactly the ones a nonlinear drift
produces.  Both cases are therefore exercised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ToyProblem", "single_gaussian", "bimodal", "spiral_2d"]


@dataclass
class ToyProblem:
    """A Gaussian-mixture probability-flow ODE with a known solution.

    Attributes
    ----------
    means : ndarray, shape (C, d)
    stds : ndarray, shape (C,)
        Per-component isotropic data standard deviations.
    weights : ndarray, shape (C,)
        Mixture weights, normalised on construction.
    name : str
    """

    means: np.ndarray
    stds: np.ndarray
    weights: np.ndarray
    name: str = "toy"
    _nfe: int = field(default=0, repr=False)

    def __post_init__(self):
        self.means = np.atleast_2d(np.asarray(self.means, dtype=np.float64))
        self.stds = np.atleast_1d(np.asarray(self.stds, dtype=np.float64))
        w = np.atleast_1d(np.asarray(self.weights, dtype=np.float64))
        if not (self.means.shape[0] == self.stds.size == w.size):
            raise ValueError("means, stds and weights disagree on component count")
        if np.any(self.stds <= 0):
            raise ValueError("component standard deviations must be positive")
        self.weights = w / w.sum()

    # -- geometry ---------------------------------------------------------
    @property
    def dim(self):
        return int(self.means.shape[1])

    @property
    def n_components(self):
        return int(self.means.shape[0])

    @property
    def is_linear(self):
        """Single component => drift linear in ``x`` => extrapolation flattered."""
        return self.n_components == 1

    # -- instrumentation --------------------------------------------------
    @property
    def nfe(self):
        """Denoiser calls so far, mirroring how NFE is counted for a network."""
        return self._nfe

    def reset_nfe(self):
        self._nfe = 0

    # -- the model --------------------------------------------------------
    def denoise(self, x, sigma):
        """Exact EDM denoiser ``D(x; sigma)``.

        ``x`` has shape ``(..., d)``.  Responsibilities are evaluated through a
        log-sum-exp so that well-separated components at small ``sigma`` do not
        underflow -- at ``sigma = 2e-3`` the raw densities are far outside
        float64's range.
        """
        x = np.asarray(x, dtype=np.float64)
        self._nfe += 1

        if sigma <= 0:
            return x

        var = self.stds ** 2 + float(sigma) ** 2          # (C,)
        diff = x[..., None, :] - self.means               # (..., C, d)
        sq = np.sum(diff * diff, axis=-1)                 # (..., C)

        log_w = (
            np.log(self.weights)
            - 0.5 * self.dim * np.log(var)
            - 0.5 * sq / var
        )
        log_w -= log_w.max(axis=-1, keepdims=True)
        gamma = np.exp(log_w)
        gamma /= gamma.sum(axis=-1, keepdims=True)        # (..., C)

        score = np.sum(
            (gamma / var)[..., None] * (self.means - x[..., None, :]), axis=-2
        )
        return x + float(sigma) ** 2 * score

    def drift(self, x, sigma):
        """``dx/dsigma = (x - D(x; sigma)) / sigma``."""
        if sigma <= 0:
            return np.zeros_like(x)
        return (np.asarray(x, dtype=np.float64) - self.denoise(x, sigma)) / float(sigma)

    # -- ground truth -----------------------------------------------------
    def exact(self, x0, sigma0, sigma1):
        """Closed-form flow map, single-component problems only."""
        if not self.is_linear:
            raise NotImplementedError(
                "closed form exists only for a single Gaussian; "
                "use reference() for mixtures"
            )
        mu, s2 = self.means[0], self.stds[0] ** 2
        ratio = np.sqrt((s2 + float(sigma1) ** 2) / (s2 + float(sigma0) ** 2))
        return mu + (np.asarray(x0, dtype=np.float64) - mu) * ratio

    def reference(self, x0, sigma0, sigma1, n_steps=200_000):
        """High-accuracy reference flow map by fixed-step RK4.

        Steps are laid out uniformly in ``log sigma`` rather than in ``sigma``:
        the drift ``(x - D)/sigma`` stiffens as ``sigma -> 0``, and uniform
        spacing would spend nearly all its effort where the solution barely
        moves.  With the default step count the reference is converged far
        below any error we are trying to resolve; :func:`reference_selftest`
        checks that against the closed form.
        """
        x = np.array(x0, dtype=np.float64, copy=True)
        lo = max(float(sigma1), 1e-8)
        grid = np.geomspace(float(sigma0), lo, n_steps + 1)
        if sigma1 <= 0:
            grid = np.append(grid, 0.0)

        saved, self._nfe = self._nfe, 0
        for a, b in zip(grid[:-1], grid[1:]):
            h = b - a
            k1 = self.drift(x, a)
            k2 = self.drift(x + 0.5 * h * k1, a + 0.5 * h)
            k3 = self.drift(x + 0.5 * h * k2, a + 0.5 * h)
            k4 = self.drift(x + h * k3, b)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._nfe = saved
        return x

    def ground_truth(self, x0, sigma0, sigma1, **kw):
        """Closed form when available, otherwise the RK4 reference."""
        if self.is_linear:
            return self.exact(x0, sigma0, sigma1)
        return self.reference(x0, sigma0, sigma1, **kw)


# ---------------------------------------------------------------------------
# standard instances
# ---------------------------------------------------------------------------

def single_gaussian(dim=1, mean=0.0, std=0.5):
    """One isotropic Gaussian: linear drift, exact solution available."""
    return ToyProblem(
        means=np.full((1, dim), float(mean)),
        stds=np.array([float(std)]),
        weights=np.array([1.0]),
        name=f"single_gaussian_d{dim}",
    )


def bimodal(separation=4.0, std=0.5, dim=1):
    """Two well-separated modes.

    The nonlinearity here is sharply localised: responsibilities switch over a
    narrow band of ``sigma`` near the separation scale, which is exactly where
    a fixed error model based on step widths alone should struggle.  This is
    the problem that stress-tests the linear error-accumulation assumption.
    """
    m = np.zeros((2, dim))
    m[0, 0] = -0.5 * separation
    m[1, 0] = +0.5 * separation
    return ToyProblem(
        means=m,
        stds=np.full(2, float(std)),
        weights=np.array([0.5, 0.5]),
        name=f"bimodal_sep{separation:g}_d{dim}",
    )


def spiral_2d(n_components=8, radius=3.0, std=0.35):
    """Eight modes on a circle -- a 2-D problem with structure worth plotting."""
    ang = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
    means = radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return ToyProblem(
        means=means,
        stds=np.full(n_components, float(std)),
        weights=np.ones(n_components),
        name=f"ring{n_components}_r{radius:g}",
    )


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

def reference_selftest(n_steps=20_000, sigma_max=80.0, sigma_min=0.002, seed=0):
    """Validate the RK4 reference against the closed form.

    Run on the single-Gaussian problem, where both are available.  Returns the
    max relative error; anything above roughly ``1e-10`` means the reference is
    not accurate enough to serve as ground truth for the order study.
    """
    prob = single_gaussian(dim=1, std=0.5)
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(16, 1)) * sigma_max
    exact = prob.exact(x0, sigma_max, sigma_min)
    ref = prob.reference(x0, sigma_max, sigma_min, n_steps=n_steps)
    scale = np.maximum(np.abs(exact), 1e-12)
    return float(np.max(np.abs(ref - exact) / scale))
