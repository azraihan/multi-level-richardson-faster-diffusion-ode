"""Combined regression tests for samplers, schedules, and device paths.

This file was added after the initial submission to consolidate tests that
previously lived in separate files. It covers:

  - Bit-exact equivalence with the original RX-DPM reference implementation.
  - NFE accounting: confirming the "free lunch" claim (RX costs no more than
    plain Euler in reuse mode).
  - Weight correctness: weights must always sum to 1.
  - Schedule correctness: nested_steps must return valid nested chains or raise.
  - Device safety: the sampler works with torch tensors that cannot be
    directly converted to numpy (simulating a GPU tensor).
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the entire module if torch is not installed (e.g. on a CPU-only machine).
torch = pytest.importorskip("torch")

from mlrx import extrapolation as extrap
from mlrx import samplers, schedules, toy
from mlrx.runner import Config


# ---------------------------------------------------------------------------
# Reference implementation (copied verbatim from refcode/generate.py)
# ---------------------------------------------------------------------------
# This is the ORIGINAL code from the ICLR 2025 paper authors, used as the
# ground truth in the bit-exact equivalence tests below.

def get_coeff(frequency, grids):
    """Original paper's coefficient computation (verbatim from refcode).

    Builds a 2x2 system where:
      - Row 0: coarse level (1 step) — error coefficient = 1 (all of h^p)
      - Row 1: fine level (K steps) — error coefficient = Σ(λ_j^2)
    Inverts the system and returns the first row of the inverse as weights.
    """
    A = torch.ones(2, 2).to(grids[0][0].device).double()
    K = grids[1][0] - grids[1][-1]   # total block width
    A[1, 1] = 0.0
    for m in range(frequency):
        # Compute the sum of squared normalised sub-step widths (the S_2 moment).
        k = grids[1][m] - grids[1][m + 1]
        A[1, 1] += (k / K) ** 2
    A_inv = torch.inverse(A)
    return A_inv[0, :]   # first row = weights for [coarse, fine]


def rx_euler_sampler_reference(
    net, latents, num_steps=18, frequency=2,
    sigma_min=0.002, sigma_max=80, rho=7, skip_last=True,
):
    """Full reference RX-DPM sampler, transcribed from the paper's refcode.

    This is used ONLY for the bit-exact equivalence tests. It is the authors'
    original implementation, kept unmodified so we can verify our rewrite
    matches it to < 1e-12 across eight (num_steps, frequency) combinations.
    """
    # Build the EDM noise schedule: N descending sigma values + 0.
    t_steps_list = []
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
               * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # Build the coarse schedule: one time point every `frequency` fine steps.
    ext_steps = t_steps[::frequency]

    # Handle the case where num_steps is not divisible by frequency.
    adj_freq = frequency
    if num_steps % frequency > 0:
        if skip_last:
            # Keep the remainder block but don't extrapolate over it.
            ext_steps = torch.cat([ext_steps, torch.zeros_like(t_steps[:1])])
            adj_freq = num_steps % frequency
        else:
            # Absorb the remainder into the last full block.
            ext_steps = torch.cat([ext_steps[:-1], torch.zeros_like(t_steps[:1])])
            adj_freq = adj_freq + num_steps % frequency

    num_ext = len(ext_steps) - 1
    t_steps_list.append(t_steps)     # level 0: fine (K steps per block)
    t_steps_list.append(ext_steps)   # level 1: coarse (1 step per block)

    recent_idx = [0] * 2    # current position in each level's schedule
    x_next = latents.to(torch.float64) * t_steps[0]
    cnt = 0   # NFE counter

    for i in range(num_ext):
        outs, grids = [], []
        denoised_save = 0.0
        x_cur_save = x_next
        # Only apply extrapolation if this is not the short remainder block
        # (skip_last convention: the last block runs plain Euler).
        apply_ext = not (i == num_ext - 1 and skip_last and adj_freq < frequency)

        for lev in range(2):
            # Level 0 = fine (K sub-steps); Level 1 = coarse (1 step).
            local_steps = (frequency if i < num_ext - 1 else adj_freq) if lev == 0 else 1
            _t_steps = t_steps_list[lev]
            x_next = x_cur_save   # both levels start from the same state
            t_cur = _t_steps[recent_idx[lev]]
            t_next = _t_steps[recent_idx[lev] + 1]

            for j in range(local_steps):
                if lev == 0 or apply_ext:
                    x_cur = x_next
                    t_hat = net.round_sigma(t_cur)
                    x_hat = x_cur
                    if not apply_ext:
                        # No extrapolation: evaluate network normally.
                        denoised = net(x_hat, t_hat).to(torch.float64)
                        cnt += 1
                    elif not lev == 1:
                        # Fine level: always evaluate; save the first one for reuse.
                        denoised = net(x_hat, t_hat).to(torch.float64)
                        cnt += 1
                        if lev == 0 and j == 0:
                            denoised_save = denoised   # save for the coarse reuse
                    else:
                        # Coarse level: REUSE the fine trajectory's first evaluation.
                        # This is the key trick — zero extra NFE.
                        denoised = denoised_save
                    d_cur = (x_hat - denoised) / t_hat   # drift
                    x_next = x_hat + (t_next - t_hat) * d_cur   # Euler step
                recent_idx[lev] += 1
                t_cur = _t_steps[recent_idx[lev]]
                t_next = (_t_steps[recent_idx[lev] + 1]
                          if recent_idx[lev] + 1 < len(_t_steps)
                          else torch.zeros_like(t_cur))

            grids.append(_t_steps[recent_idx[lev] - local_steps:recent_idx[lev] + 1])
            outs.append(x_next)

        # The reference code orders [fine, coarse]; reverse to [coarse, fine]
        # to match get_coeff's convention (coarsest level first).
        grids.reverse()
        outs.reverse()

        if apply_ext:
            # Combine coarse and fine estimates with the paper's coefficients.
            coeff = get_coeff(frequency if i < num_ext - 1 else adj_freq, grids)
            x_next = 0.0
            for j in range(len(coeff)):
                x_next = x_next + coeff[j] * outs[j]
        else:
            # No extrapolation: just use the fine (Euler) estimate.
            x_next = outs[1]

    return x_next, cnt


class _ToyNet:
    """Adapts the analytic denoiser to the reference sampler's net interface.

    The reference sampler calls net(x, sigma) expecting a torch.Tensor. Our
    analytic toy denoiser takes numpy arrays. This shim does the conversion.
    """

    def __init__(self, problem):
        self.problem = problem

    @staticmethod
    def round_sigma(s):
        """EDM's round_sigma is a no-op for the analytic problem."""
        return s

    def __call__(self, x, sigma):
        out = self.problem.denoise(x.cpu().numpy(), float(sigma))
        return torch.as_tensor(out, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Equivalence tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_steps,frequency", [
    (18, 2), (18, 3), (10, 2), (12, 4), (9, 2), (11, 3), (20, 5), (16, 4),
])
def test_two_level_matches_reference(num_steps, frequency):
    """Our rx_sampler must match the reference implementation to < 1e-12.

    This is the most important test: it proves our rewrite is not just
    approximately correct but bit-for-bit identical to the authors' code
    across 8 different (num_steps, frequency) combinations.
    """
    problem = toy.bimodal(separation=4.0, std=0.5, dim=1)
    net = _ToyNet(problem)
    rng = np.random.default_rng(0)
    latents = torch.as_tensor(rng.normal(size=(8, 1)), dtype=torch.float64)

    # Run the authors' reference sampler.
    ref_x, ref_nfe = rx_euler_sampler_reference(
        net, latents, num_steps=num_steps, frequency=frequency, skip_last=True
    )
    # Run our reimplementation on the same schedule and initial state.
    t = schedules.edm_schedule(num_steps, rho=7.0)
    x_init = latents.numpy() * t[0]
    got = samplers.rx_sampler(
        problem.denoise, x_init, t,
        frequency=frequency, n_levels=2, p=2, skip_last=True,
    )

    # Both the output and NFE must match to numerical precision.
    assert np.max(np.abs(got.x - ref_x.numpy())) < 1e-12
    assert got.nfe == ref_nfe


# ---------------------------------------------------------------------------
# NFE accounting tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_steps,frequency", [(18, 2), (12, 4), (16, 4)])
def test_nfe_is_free_at_every_level_count(num_steps, frequency):
    """RX with reuse mode must cost exactly num_steps evaluations (same as Euler).

    This is the "free lunch" property: Richardson extrapolation at any number
    of levels costs zero extra neural network evaluations in reuse mode.
    """
    problem = toy.bimodal()
    t = schedules.edm_schedule(num_steps)
    x_init = np.random.default_rng(1).normal(size=(4, 1)) * t[0]
    # Euler is the reference: always costs exactly num_steps.
    assert samplers.euler_sampler(problem.denoise, x_init, t).nfe == num_steps

    for levels in (2, 3):
        try:
            schedules.nested_steps(frequency, levels)
        except ValueError:
            continue   # this (frequency, levels) combination is not nestable
        result = samplers.rx_sampler(
            problem.denoise, x_init, t, frequency=frequency, n_levels=levels
        )
        # The free lunch: RX should cost the same as Euler.
        assert result.nfe == num_steps


# ---------------------------------------------------------------------------
# Weight correctness tests
# ---------------------------------------------------------------------------

def test_weights_always_sum_to_one():
    """Extrapolation weights must sum to exactly 1 for all level counts.

    Weights summing to 1 means: if all levels produce the exact same estimate
    (zero error), the combination also returns that exact estimate. This is
    the basic consistency check for any linear combination scheme.
    """
    t = schedules.edm_schedule(32, rho=7.0)
    base = schedules.block_lambdas(t[0:17])
    for levels in range(2, 6):
        level_lambdas = extrap.nested_level_lambdas(
            base, schedules.nested_steps(16, levels)
        )
        weights = extrap.extrapolation_weights(level_lambdas, p=2)
        assert abs(float(weights.w.sum()) - 1.0) < 1e-12


def test_two_level_closed_form_matches_general_solver():
    """The closed-form 2-level formula must match the general L-level solver.

    two_level_weights() implements the paper's Eq. 18 directly in closed form.
    extrapolation_weights() is our general solver for any L.
    At L=2 they must give identical results — this is a key internal check.
    """
    t = schedules.edm_schedule(18, rho=7.0)
    for block_size in (2, 3, 4, 5):
        lambdas = schedules.block_lambdas(t[0:block_size + 1])
        closed_form = extrap.two_level_weights(lambdas, p=2).w
        general = extrap.extrapolation_weights([np.array([1.0]), lambdas], p=2).w
        assert np.allclose(closed_form, general, atol=1e-13)


def test_naive_differs_from_grid_aware_on_edm_schedule():
    """Grid-aware and naïve (uniform-grid) weights must differ on the EDM schedule.

    The whole point of the paper's contribution is that the EDM schedule is
    non-uniform, so the classical Richardson formula (which assumes uniform
    steps) gives the wrong coefficients. This test confirms they actually differ.
    """
    t = schedules.edm_schedule(18, rho=7.0)
    lambdas = schedules.block_lambdas(t[0:3])
    grid_aware = extrap.two_level_weights(lambdas, p=2).w
    naive = extrap.naive_richardson_weights(2, p=2).w
    assert not np.allclose(grid_aware, naive, atol=1e-3)  # they must differ
    assert abs(float(naive.sum()) - 1.0) < 1e-13          # but still sum to 1


# ---------------------------------------------------------------------------
# Schedule / nested_steps tests
# ---------------------------------------------------------------------------

def _legacy_nested_steps(frequency, n_levels):
    """Old version of nested_steps, kept here to check the new one is backward-compatible."""
    chain = [1]
    while chain[-1] * 2 < frequency:
        chain.append(chain[-1] * 2)
    chain.append(frequency)
    chain = sorted(set(chain))
    if len(chain) < n_levels:
        return None   # not enough divisors to form a nested chain
    if len(chain) == n_levels:
        return chain
    indices = np.unique(np.round(np.linspace(0, len(chain) - 1, n_levels)).astype(int))
    return [chain[index] for index in indices]


def _omega(number):
    """Count the number of prime factors of `number` (with multiplicity).

    For example: omega(8) = 3 (2×2×2), omega(12) = 3 (2×2×3).
    This equals the maximum number of divisors in a nested chain starting
    at 1 and ending at `number`, minus 1.
    """
    count, factor = 0, 2
    while number > 1:
        while number % factor == 0:
            number //= factor
            count += 1
        factor += 1
    return count


def _is_nested(chain, block_size, levels):
    """Check that a chain is a valid nested level sequence.

    Valid means: starts at 1, ends at block_size, has exactly `levels`
    elements, and each element divides the next (so that coarser steps
    are exact unions of finer steps).
    """
    return (len(chain) == levels and chain[0] == 1 and chain[-1] == block_size
            and all(b > a and b % a == 0 for a, b in zip(chain, chain[1:])))


@pytest.mark.parametrize("block_size", range(2, 97))
def test_every_chain_is_nested_or_rejected(block_size):
    """nested_steps must return a valid nested chain or raise ValueError.

    For every block_size from 2 to 96, and every level count from 2 to 9,
    the function must either return a chain that satisfies _is_nested OR
    raise a ValueError. It must never silently return an invalid chain.
    """
    for levels in range(2, 10):
        # A nested chain of length `levels` exists iff the block has at least
        # `levels - 1` prime factors (omega(block_size) + 1 >= levels).
        exists = _omega(block_size) + 1 >= levels
        if exists:
            chain = schedules.nested_steps(block_size, levels)
            assert _is_nested(chain, block_size, levels)
            assert all(type(number) is int for number in chain)
        else:
            with pytest.raises(ValueError):
                schedules.nested_steps(block_size, levels)


def test_unchanged_wherever_the_old_chain_was_valid():
    """The new nested_steps must agree with the old version on all valid cases.

    When the old implementation would have returned a valid nested chain, the
    new implementation must return the exact same chain (backward compatibility).
    """
    checked = 0
    for block_size in range(2, 257):
        for levels in range(2, 10):
            old = _legacy_nested_steps(block_size, levels)
            if old is not None and _is_nested(old, block_size, levels):
                assert schedules.nested_steps(block_size, levels) == old
                checked += 1
    # Make sure we actually checked a meaningful number of cases.
    assert checked > 200


@pytest.mark.parametrize("block_size,levels,expected", [
    (8, 4, [1, 2, 4, 8]), (4, 3, [1, 2, 4]), (16, 3, [1, 4, 16]),
    (16, 4, [1, 2, 8, 16]), (12, 3, [1, 4, 12]), (6, 3, [1, 2, 6]),
    (12, 4, [1, 2, 6, 12]), (18, 3, [1, 3, 18]),
])
def test_known_chains(block_size, levels, expected):
    """Spot-check a handful of known (block_size, levels) → chain mappings."""
    assert schedules.nested_steps(block_size, levels) == expected


@pytest.mark.parametrize("block_size,levels", [(3, 3), (5, 3), (7, 3), (6, 4), (12, 5)])
def test_impossible_requests_raise_with_true_limit(block_size, levels):
    """Impossible requests must raise ValueError with the correct max level count.

    The error message should state the true maximum number of nested levels
    so the caller knows what is actually achievable.
    """
    with pytest.raises(ValueError, match=f"at most {_omega(block_size) + 1} nested"):
        schedules.nested_steps(block_size, levels)


def test_integer_like_inputs_are_accepted():
    """nested_steps must accept numpy int64 and float inputs (common in practice).

    Callers often pass schedule lengths derived from numpy operations, which
    produce numpy integers rather than Python ints. The function must handle
    these without raising a TypeError.
    """
    assert schedules.nested_steps(np.int64(6), np.int64(3)) == [1, 2, 6]
    assert schedules.nested_steps(8.0, 4) == [1, 2, 4, 8]


@pytest.mark.parametrize("block_size,levels", [(1, 2), (0, 2), (4, 1), (6.5, 2), (6, 2.5)])
def test_bad_inputs_raise_value_error(block_size, levels):
    """Clearly invalid inputs (block_size < 2, levels < 2, non-integer) must raise."""
    with pytest.raises(ValueError):
        schedules.nested_steps(block_size, levels)


def test_deterministic():
    """nested_steps must return the same chain on repeated calls (no randomness)."""
    for block_size in (6, 12, 18, 24, 30, 36, 48, 60):
        for levels in (3, 4):
            try:
                first = schedules.nested_steps(block_size, levels)
            except ValueError:
                continue
            assert all(schedules.nested_steps(block_size, levels) == first for _ in range(3))


# ---------------------------------------------------------------------------
# rx_sampler robustness tests
# ---------------------------------------------------------------------------

# A representative set of (num_steps, frequency, n_levels, skip_last)
# configurations covering both power-of-two and non-power-of-two block sizes.
CASES = [
    (12, 6, 3, True), (12, 6, 4, True), (12, 3, 3, True),
    (10, 4, 3, False), (11, 4, 3, False), (13, 4, 3, True),
    (24, 12, 4, True), (16, 8, 4, True),
]


@pytest.mark.parametrize("reuse_mode", ["denoised", "derivative", "exact"])
@pytest.mark.parametrize("num_steps,frequency,n_levels,skip_last", CASES)
def test_rx_sampler_never_crashes(num_steps, frequency, n_levels, skip_last, reuse_mode):
    """rx_sampler must produce finite outputs and the correct NFE for all cases.

    Tests all three reuse modes across a grid of schedule / block / level
    combinations to catch any configuration that causes NaN, inf, or an
    incorrect NFE count.
    """
    problem = toy.bimodal()
    t = schedules.edm_schedule(num_steps)
    x_init = np.random.default_rng(0).normal(size=(8, 1)) * t[0]
    result = samplers.rx_sampler(
        problem.denoise, x_init, t, frequency=frequency, n_levels=n_levels,
        skip_last=skip_last, reuse_mode=reuse_mode,
    )
    # All output values must be finite (no NaN, no inf).
    assert np.all(np.isfinite(result.x))

    # Compute expected NFE independently and compare.
    extra = 0
    plan = schedules.partition_blocks(num_steps, frequency, skip_last=skip_last)
    for block in plan:
        if not block.extrapolate or block.n_steps < 2:
            continue
        try:
            levels = schedules.nested_steps(block.n_steps, n_levels)
        except ValueError:
            continue
        if reuse_mode == "exact":
            # Each coarse level of n sub-steps costs n-1 extra evaluations.
            extra += sum(number - 1 for number in levels[:-1])
    assert result.nfe == num_steps + extra

    # Also verify that the Config.expected_nfe property agrees (it is used for
    # budgeting and must match the sampler's actual NFE).
    if skip_last:
        config = Config(method="rx", num_steps=num_steps, frequency=frequency,
                        n_levels=n_levels, reuse_mode=reuse_mode)
        assert config.expected_nfe == result.nfe


def test_non_power_of_two_block_actually_extrapolates():
    """Non-power-of-two blocks (e.g. K=6) must actually extrapolate and give order > 2.

    This is a functional correctness test: we run the sampler at multiple step
    counts and fit the convergence order. If extrapolation is working, the
    order should exceed the Euler baseline of 1.
    """
    problem = toy.single_gaussian()
    x_init = np.random.default_rng(0).normal(size=(256, 1)) * 80.0
    errors = []
    for num_steps in (96, 192, 384):
        t = schedules.edm_schedule(num_steps)
        truth = problem.exact(x_init, t[0], t[-1])
        result = samplers.rx_sampler(
            problem.denoise, x_init, t, frequency=6, n_levels=3, reuse_mode="exact"
        )
        # Verify the expected structure: one Weights object per block.
        assert len(result.weights) == num_steps // 6
        assert all(weights.w.size == 3 for weights in result.weights)
        errors.append(np.sqrt(np.mean((result.x - truth) ** 2)))

        # Sanity check: the extrapolated result must beat plain Euler.
        euler = samplers.euler_sampler(problem.denoise, x_init, t)
        assert errors[-1] < np.sqrt(np.mean((euler.x - truth) ** 2))

    # Fit the empirical convergence order (doubling N should reduce error by 2^order).
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 2.7)   # order > 2 confirms extrapolation is working


# ---------------------------------------------------------------------------
# Device safety tests
# ---------------------------------------------------------------------------

class _DeviceLike(torch.Tensor):
    """Simulates a GPU tensor that cannot be directly converted to numpy.

    On a real GPU, calling .numpy() on a CUDA tensor raises a TypeError.
    This subclass simulates that so we can test the code path without
    needing an actual GPU.
    """
    def __array__(self, *args, **kwargs):
        raise TypeError("can't convert device tensor to numpy (simulated)")


def _setup(num_steps=16):
    """Build a standard toy problem and two versions of the schedule:
    one as a numpy array (CPU) and one as a _DeviceLike tensor (simulated GPU).
    """
    problem = toy.bimodal()
    t_np = schedules.edm_schedule(num_steps)
    # Wrap the numpy schedule in a tensor subclass that refuses numpy conversion.
    t_dev = torch.as_tensor(t_np).as_subclass(_DeviceLike)
    x0 = torch.randn(4, 1, dtype=torch.float64,
                     generator=torch.Generator().manual_seed(0)) * float(t_np[0])

    def den(x, sigma):
        """Denoiser that accepts torch tensors and returns torch tensors."""
        return torch.as_tensor(problem.denoise(x.numpy(), float(sigma)))

    return problem, t_np, t_dev, x0, den


@pytest.mark.parametrize("n_levels,frequency", [(2, 2), (2, 3), (3, 4)])
@pytest.mark.parametrize("reuse_mode", ["denoised", "exact"])
@pytest.mark.parametrize("work_dtype", ["float64", "float32", "float16"])
def test_rx_on_device_schedule_matches_numpy(n_levels, frequency, reuse_mode, work_dtype):
    """The sampler must give identical results regardless of whether the schedule
    is a numpy array or a torch tensor that refuses numpy conversion.

    This tests the block_lambdas() code path that extracts step widths from
    the schedule: it must work with torch tensors, not just numpy arrays.
    """
    problem, t_np, t_dev, x0, den = _setup()
    kwargs = dict(frequency=frequency, n_levels=n_levels, reuse_mode=reuse_mode,
                  work_dtype=np.dtype(work_dtype))
    # Run with the device-like tensor schedule.
    got = samplers.rx_sampler(den, x0, t_dev, **kwargs)
    # Run with the plain numpy schedule.
    reference = samplers.rx_sampler(problem.denoise, x0.numpy(), t_np, **kwargs)
    assert got.nfe == reference.nfe
    assert got.x.dtype == torch.float64
    assert np.array_equal(got.x.numpy(), reference.x)


@pytest.mark.parametrize("coefficients", ["grid_aware", "naive"])
def test_rx_edm_and_naive_on_device_schedule(coefficients):
    """rx_edm_sampler must produce finite output with device schedules."""
    _, _, t_dev, x0, den = _setup()
    result = samplers.rx_edm_sampler(
        den, x0, t_dev, frequency=2, n_levels=2, coefficients=coefficients
    )
    assert torch.isfinite(result.x).all()
