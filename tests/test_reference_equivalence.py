"""Equivalence against the authors' reference implementation.

The single most important test in the repository.  Our sampler is a rewrite --
backend-agnostic, multi-level, no ``torch.distributed`` -- and a rewrite is only
trustworthy if it reproduces the original bit for bit on the original's own
settings.  ``rx_euler_sampler`` and ``get_coeff`` below are transcribed verbatim
from ``refcode/generate.py`` (RX-DPM, ICLR 2025) with only the distributed and
image-saving scaffolding removed.

The analytic Gaussian-mixture denoiser of :mod:`mlrx.toy` stands in for the
network: it satisfies the same interface, is deterministic, and needs no GPU,
so the comparison is exact rather than statistical.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlrx import extrapolation as extrap
from mlrx import samplers, schedules, toy


# ---------------------------------------------------------------------------
# reference implementation, transcribed from refcode/generate.py
# ---------------------------------------------------------------------------

def get_coeff(frequency, grids):
    A = torch.ones(2, 2).to(grids[0][0].device).double()
    K = (grids[1][0] - grids[1][-1])
    A[1, 1] = 0.0
    for m in range(frequency):
        k = (grids[1][m] - grids[1][m + 1])
        A[1, 1] += (k / K) ** 2
    A_inv = torch.inverse(A)
    return A_inv[0, :]


def rx_euler_sampler_reference(
    net, latents, num_steps=18, frequency=2,
    sigma_min=0.002, sigma_max=80, rho=7, skip_last=True,
):
    t_steps_list = []
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
               * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    ext_steps = t_steps[::frequency]

    adj_freq = frequency
    if num_steps % frequency > 0:
        if skip_last:
            ext_steps = torch.cat([ext_steps, torch.zeros_like(t_steps[:1])])
            adj_freq = num_steps % frequency
        else:
            ext_steps = torch.cat([ext_steps[:-1], torch.zeros_like(t_steps[:1])])
            adj_freq = adj_freq + num_steps % frequency

    num_ext = len(ext_steps) - 1
    t_steps_list.append(t_steps)
    t_steps_list.append(ext_steps)

    recent_idx = [0] * 2
    x_next = latents.to(torch.float64) * t_steps[0]
    cnt = 0

    for i in range(num_ext):
        outs, grids = [], []
        denoised_save = 0.0
        x_cur_save = x_next
        apply_ext = not (i == num_ext - 1 and skip_last and adj_freq < frequency)

        for lev in range(2):
            local_steps = (frequency if i < num_ext - 1 else adj_freq) if lev == 0 else 1
            _t_steps = t_steps_list[lev]
            x_next = x_cur_save
            t_cur = _t_steps[recent_idx[lev]]
            t_next = _t_steps[recent_idx[lev] + 1]

            for j in range(local_steps):
                if lev == 0 or apply_ext:
                    x_cur = x_next
                    t_hat = net.round_sigma(t_cur)
                    x_hat = x_cur
                    if not apply_ext:
                        denoised = net(x_hat, t_hat).to(torch.float64); cnt += 1
                    elif not lev == 1:
                        denoised = net(x_hat, t_hat).to(torch.float64); cnt += 1
                        if lev == 0 and j == 0:
                            denoised_save = denoised
                    else:
                        denoised = denoised_save
                    d_cur = (x_hat - denoised) / t_hat
                    x_next = x_hat + (t_next - t_hat) * d_cur
                recent_idx[lev] += 1
                t_cur = _t_steps[recent_idx[lev]]
                t_next = (_t_steps[recent_idx[lev] + 1]
                          if recent_idx[lev] + 1 < len(_t_steps) else torch.zeros_like(t_cur))

            grids.append(_t_steps[recent_idx[lev] - local_steps:recent_idx[lev] + 1])
            outs.append(x_next)

        grids.reverse()
        outs.reverse()

        if apply_ext:
            coeff = get_coeff(frequency if i < num_ext - 1 else adj_freq, grids)
            x_next = 0.0
            for j in range(len(coeff)):
                x_next = x_next + coeff[j] * outs[j]
        else:
            x_next = outs[1]

    return x_next, cnt


class _ToyNet:
    """Adapts the analytic denoiser to the reference sampler's net interface."""

    def __init__(self, problem):
        self.problem = problem

    @staticmethod
    def round_sigma(s):
        return s

    def __call__(self, x, sigma):
        out = self.problem.denoise(x.cpu().numpy(), float(sigma))
        return torch.as_tensor(out, dtype=torch.float64)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_steps,frequency", [
    (18, 2), (18, 3), (10, 2), (12, 4), (9, 2), (11, 3), (20, 5), (16, 4),
])
def test_two_level_matches_reference(num_steps, frequency):
    """Our rx_sampler at L=2 must equal the published sampler exactly."""
    problem = toy.bimodal(separation=4.0, std=0.5, dim=1)
    net = _ToyNet(problem)

    rng = np.random.default_rng(0)
    latents = torch.as_tensor(rng.normal(size=(8, 1)), dtype=torch.float64)

    ref_x, ref_nfe = rx_euler_sampler_reference(
        net, latents, num_steps=num_steps, frequency=frequency, skip_last=True
    )

    t = schedules.edm_schedule(num_steps, rho=7.0)
    x_init = latents.numpy() * t[0]
    got = samplers.rx_sampler(
        problem.denoise, x_init, t,
        frequency=frequency, n_levels=2, p=2, skip_last=True,
    )

    err = np.max(np.abs(got.x - ref_x.numpy()))
    assert err < 1e-12, f"trajectory mismatch: {err:.3e}"
    assert got.nfe == ref_nfe, f"NFE mismatch: {got.nfe} vs {ref_nfe}"


@pytest.mark.parametrize("num_steps,frequency", [(18, 2), (12, 4), (16, 4)])
def test_nfe_is_free_at_every_level_count(num_steps, frequency):
    """Multi-level extrapolation must not cost a single extra evaluation."""
    problem = toy.bimodal()
    t = schedules.edm_schedule(num_steps)
    x_init = np.random.default_rng(1).normal(size=(4, 1)) * t[0]

    baseline = samplers.euler_sampler(problem.denoise, x_init, t).nfe
    assert baseline == num_steps

    max_levels = len(schedules.nested_steps(frequency, 2)) if frequency == 2 else None
    for L in (2, 3):
        try:
            schedules.nested_steps(frequency, L)
        except ValueError:
            continue
        res = samplers.rx_sampler(
            problem.denoise, x_init, t, frequency=frequency, n_levels=L
        )
        assert res.nfe == num_steps, f"L={L} spent {res.nfe} NFE, expected {num_steps}"


def test_weights_always_sum_to_one():
    """Consistency: an extrapolation that does not preserve constants is wrong."""
    t = schedules.edm_schedule(32, rho=7.0)
    base = schedules.block_lambdas(t[0:17])
    for L in range(2, 6):
        lv = extrap.nested_level_lambdas(base, schedules.nested_steps(16, L))
        W = extrap.extrapolation_weights(lv, p=2)
        assert abs(float(W.w.sum()) - 1.0) < 1e-12, f"L={L}: sum={W.w.sum()}"


def test_two_level_closed_form_matches_general_solver():
    """The closed form and the L-level solver must agree where both apply."""
    t = schedules.edm_schedule(18, rho=7.0)
    for K in (2, 3, 4, 5):
        lam = schedules.block_lambdas(t[0:K + 1])
        a = extrap.two_level_weights(lam, p=2).w
        b = extrap.extrapolation_weights([np.array([1.0]), lam], p=2).w
        assert np.allclose(a, b, atol=1e-13), f"K={K}: {a} vs {b}"


def test_naive_differs_from_grid_aware_on_edm_schedule():
    """The paper's Fig. 2 ablation only means something if these really differ."""
    t = schedules.edm_schedule(18, rho=7.0)
    lam = schedules.block_lambdas(t[0:3])
    grid_aware = extrap.two_level_weights(lam, p=2).w
    naive = extrap.naive_richardson_weights(2, p=2).w
    assert not np.allclose(grid_aware, naive, atol=1e-3)
    assert abs(float(naive.sum()) - 1.0) < 1e-13
