"""Nested level chains, and the samplers on block lengths that are not powers
of two.

``nested_steps`` used to fill non-power-of-two blocks with a chain that was not
nested (``K=6, L=3 -> [1, 4, 6]``).  The sampler then integrated a coarse level
that stopped short of the block end and crashed on the weight solve.  These
tests pin the contract every caller relies on: a genuine nested chain, or
``ValueError`` -- and the chain is unchanged wherever the old one was valid.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlrx import samplers, schedules, toy
from mlrx.runner import Config


def _legacy_nested_steps(frequency, n_levels):
    """The pre-fix algorithm, verbatim, for the backward-compatibility check."""
    chain = [1]
    while chain[-1] * 2 < frequency:
        chain.append(chain[-1] * 2)
    chain.append(frequency)
    chain = sorted(set(chain))
    if len(chain) < n_levels:
        return None
    if len(chain) == n_levels:
        return chain
    idx = np.unique(np.round(np.linspace(0, len(chain) - 1, n_levels)).astype(int))
    return [chain[i] for i in idx]


def _omega(n):
    """Number of prime factors of ``n`` counted with multiplicity."""
    count, f = 0, 2
    while n > 1:
        while n % f == 0:
            n //= f
            count += 1
        f += 1
    return count


def _is_nested(chain, K, L):
    return (len(chain) == L and chain[0] == 1 and chain[-1] == K
            and all(b > a and b % a == 0 for a, b in zip(chain, chain[1:])))


# ---------------------------------------------------------------------------
# nested_steps itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", range(2, 97))
def test_every_chain_is_nested_or_rejected(K):
    """Exhaustive over block lengths: valid chain exactly when one exists."""
    for L in range(2, 10):
        exists = _omega(K) + 1 >= L
        if exists:
            chain = schedules.nested_steps(K, L)
            assert _is_nested(chain, K, L), (K, L, chain)
            assert all(type(n) is int for n in chain)
        else:
            with pytest.raises(ValueError):
                schedules.nested_steps(K, L)


def test_unchanged_wherever_the_old_chain_was_valid():
    """Every published result used a chain the old code got right.  Those
    must come out identical, so no stored number can shift."""
    checked = 0
    for K in range(2, 257):
        for L in range(2, 10):
            old = _legacy_nested_steps(K, L)
            if old is not None and _is_nested(old, K, L):
                assert schedules.nested_steps(K, L) == old, (K, L)
                checked += 1
    assert checked > 200


@pytest.mark.parametrize("K,L,expected", [
    (8, 4, [1, 2, 4, 8]),
    (4, 3, [1, 2, 4]),
    (16, 3, [1, 4, 16]),
    (16, 4, [1, 2, 8, 16]),
    (12, 3, [1, 4, 12]),           # old chain was already valid: kept
    (6, 3, [1, 2, 6]),             # was [1, 4, 6]
    (12, 4, [1, 2, 6, 12]),        # was [1, 2, 8, 12]
    (18, 3, [1, 3, 18]),           # was [1, 4, 18]
])
def test_known_chains(K, L, expected):
    assert schedules.nested_steps(K, L) == expected


@pytest.mark.parametrize("K,L", [(3, 3), (5, 3), (7, 3), (6, 4), (12, 5)])
def test_impossible_requests_raise_with_true_limit(K, L):
    with pytest.raises(ValueError, match=f"at most {_omega(K) + 1} nested"):
        schedules.nested_steps(K, L)


def test_integer_like_inputs_are_accepted():
    assert schedules.nested_steps(np.int64(6), np.int64(3)) == [1, 2, 6]
    assert schedules.nested_steps(8.0, 4) == [1, 2, 4, 8]


@pytest.mark.parametrize("K,L", [(1, 2), (0, 2), (4, 1), (6.5, 2), (6, 2.5)])
def test_bad_inputs_raise_value_error(K, L):
    with pytest.raises(ValueError):
        schedules.nested_steps(K, L)


def test_deterministic():
    for K in (6, 12, 18, 24, 30, 36, 48, 60):
        for L in (3, 4):
            try:
                first = schedules.nested_steps(K, L)
            except ValueError:
                continue
            assert all(schedules.nested_steps(K, L) == first for _ in range(3))


# ---------------------------------------------------------------------------
# the samplers on non-power-of-two blocks
# ---------------------------------------------------------------------------

CASES = [
    # (num_steps, frequency, n_levels, skip_last)
    (12, 6, 3, True),      # crashed before: [1, 4, 6]
    (12, 6, 4, True),      # impossible for K=6: must fall back, not crash
    (12, 3, 3, True),      # impossible for K=3: must fall back, not crash
    (10, 4, 3, False),     # remainder absorbed -> a 6-step block; crashed before
    (11, 4, 3, False),     # remainder absorbed -> a 7-step block (prime)
    (13, 4, 3, True),
    (24, 12, 4, True),     # crashed before: [1, 2, 8, 12]
    (16, 8, 4, True),      # the power-of-two path, for comparison
]


@pytest.mark.parametrize("reuse_mode", ["denoised", "derivative", "exact"])
@pytest.mark.parametrize("num_steps,frequency,n_levels,skip_last", CASES)
def test_rx_sampler_never_crashes(num_steps, frequency, n_levels, skip_last,
                                  reuse_mode):
    problem = toy.bimodal()
    t = schedules.edm_schedule(num_steps)
    x_init = np.random.default_rng(0).normal(size=(8, 1)) * t[0]

    res = samplers.rx_sampler(problem.denoise, x_init, t, frequency=frequency,
                              n_levels=n_levels, skip_last=skip_last,
                              reuse_mode=reuse_mode)
    assert np.all(np.isfinite(res.x))

    # NFE must match what the sweep's cost model predicts before running.
    extra = 0
    plan = schedules.partition_blocks(num_steps, frequency, skip_last=skip_last)
    for b in plan:
        if not b.extrapolate or b.n_steps < 2:
            continue
        try:
            levels = schedules.nested_steps(b.n_steps, n_levels)
        except ValueError:
            continue
        if reuse_mode == "exact":
            extra += sum(n - 1 for n in levels[:-1])
    assert res.nfe == num_steps + extra

    if skip_last:           # the runner's cost model assumes the default
        cfg = Config(method="rx", num_steps=num_steps, frequency=frequency,
                     n_levels=n_levels, reuse_mode=reuse_mode)
        assert cfg.expected_nfe == res.nfe


def test_non_power_of_two_block_actually_extrapolates():
    """K=6, L=3 must run the three-level scheme, not quietly fall back to
    Euler, and must converge at third order on a problem with a known exact
    answer -- the order a power-of-two three-level chain reaches.  Checked in
    the asymptotic range: a 6-step block on a coarse EDM grid spans so much of
    sigma that at N ~ 12 even two-level RX is pre-asymptotic."""
    problem = toy.single_gaussian()
    x_init = np.random.default_rng(0).normal(size=(256, 1)) * 80.0

    errors = []
    for N in (96, 192, 384):
        t = schedules.edm_schedule(N)
        truth = problem.exact(x_init, t[0], t[-1])
        res = samplers.rx_sampler(problem.denoise, x_init, t, frequency=6,
                                  n_levels=3, reuse_mode="exact")
        assert len(res.weights) == N // 6               # every block extrapolated
        assert all(w.w.size == 3 for w in res.weights)  # with three levels each
        errors.append(np.sqrt(np.mean((res.x - truth) ** 2)))

        eul = samplers.euler_sampler(problem.denoise, x_init, t)
        assert errors[-1] < np.sqrt(np.mean((eul.x - truth) ** 2))

    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 2.7), orders
