"""Combined regression tests for samplers, schedules, and device paths."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlrx import extrapolation as extrap
from mlrx import samplers, schedules, toy
from mlrx.runner import Config


# Reference implementation, transcribed from refcode/generate.py.
def get_coeff(frequency, grids):
	A = torch.ones(2, 2).to(grids[0][0].device).double()
	K = grids[1][0] - grids[1][-1]
	A[1, 1] = 0.0
	for m in range(frequency):
		k = grids[1][m] - grids[1][m + 1]
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
						denoised = net(x_hat, t_hat).to(torch.float64)
						cnt += 1
					elif not lev == 1:
						denoised = net(x_hat, t_hat).to(torch.float64)
						cnt += 1
						if lev == 0 and j == 0:
							denoised_save = denoised
					else:
						denoised = denoised_save
					d_cur = (x_hat - denoised) / t_hat
					x_next = x_hat + (t_next - t_hat) * d_cur
				recent_idx[lev] += 1
				t_cur = _t_steps[recent_idx[lev]]
				t_next = (_t_steps[recent_idx[lev] + 1]
						   if recent_idx[lev] + 1 < len(_t_steps)
						   else torch.zeros_like(t_cur))

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


@pytest.mark.parametrize("num_steps,frequency", [
	(18, 2), (18, 3), (10, 2), (12, 4), (9, 2), (11, 3), (20, 5), (16, 4),
])
def test_two_level_matches_reference(num_steps, frequency):
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

	assert np.max(np.abs(got.x - ref_x.numpy())) < 1e-12
	assert got.nfe == ref_nfe


@pytest.mark.parametrize("num_steps,frequency", [(18, 2), (12, 4), (16, 4)])
def test_nfe_is_free_at_every_level_count(num_steps, frequency):
	problem = toy.bimodal()
	t = schedules.edm_schedule(num_steps)
	x_init = np.random.default_rng(1).normal(size=(4, 1)) * t[0]
	assert samplers.euler_sampler(problem.denoise, x_init, t).nfe == num_steps

	for levels in (2, 3):
		try:
			schedules.nested_steps(frequency, levels)
		except ValueError:
			continue
		result = samplers.rx_sampler(
			problem.denoise, x_init, t, frequency=frequency, n_levels=levels
		)
		assert result.nfe == num_steps


def test_weights_always_sum_to_one():
	t = schedules.edm_schedule(32, rho=7.0)
	base = schedules.block_lambdas(t[0:17])
	for levels in range(2, 6):
		level_lambdas = extrap.nested_level_lambdas(
			base, schedules.nested_steps(16, levels)
		)
		weights = extrap.extrapolation_weights(level_lambdas, p=2)
		assert abs(float(weights.w.sum()) - 1.0) < 1e-12


def test_two_level_closed_form_matches_general_solver():
	t = schedules.edm_schedule(18, rho=7.0)
	for block_size in (2, 3, 4, 5):
		lambdas = schedules.block_lambdas(t[0:block_size + 1])
		closed_form = extrap.two_level_weights(lambdas, p=2).w
		general = extrap.extrapolation_weights([np.array([1.0]), lambdas], p=2).w
		assert np.allclose(closed_form, general, atol=1e-13)


def test_naive_differs_from_grid_aware_on_edm_schedule():
	t = schedules.edm_schedule(18, rho=7.0)
	lambdas = schedules.block_lambdas(t[0:3])
	grid_aware = extrap.two_level_weights(lambdas, p=2).w
	naive = extrap.naive_richardson_weights(2, p=2).w
	assert not np.allclose(grid_aware, naive, atol=1e-3)
	assert abs(float(naive.sum()) - 1.0) < 1e-13


def _legacy_nested_steps(frequency, n_levels):
	chain = [1]
	while chain[-1] * 2 < frequency:
		chain.append(chain[-1] * 2)
	chain.append(frequency)
	chain = sorted(set(chain))
	if len(chain) < n_levels:
		return None
	if len(chain) == n_levels:
		return chain
	indices = np.unique(np.round(np.linspace(0, len(chain) - 1, n_levels)).astype(int))
	return [chain[index] for index in indices]


def _omega(number):
	count, factor = 0, 2
	while number > 1:
		while number % factor == 0:
			number //= factor
			count += 1
		factor += 1
	return count


def _is_nested(chain, block_size, levels):
	return (len(chain) == levels and chain[0] == 1 and chain[-1] == block_size
			and all(b > a and b % a == 0 for a, b in zip(chain, chain[1:])))


@pytest.mark.parametrize("block_size", range(2, 97))
def test_every_chain_is_nested_or_rejected(block_size):
	for levels in range(2, 10):
		exists = _omega(block_size) + 1 >= levels
		if exists:
			chain = schedules.nested_steps(block_size, levels)
			assert _is_nested(chain, block_size, levels)
			assert all(type(number) is int for number in chain)
		else:
			with pytest.raises(ValueError):
				schedules.nested_steps(block_size, levels)


def test_unchanged_wherever_the_old_chain_was_valid():
	checked = 0
	for block_size in range(2, 257):
		for levels in range(2, 10):
			old = _legacy_nested_steps(block_size, levels)
			if old is not None and _is_nested(old, block_size, levels):
				assert schedules.nested_steps(block_size, levels) == old
				checked += 1
	assert checked > 200


@pytest.mark.parametrize("block_size,levels,expected", [
	(8, 4, [1, 2, 4, 8]), (4, 3, [1, 2, 4]), (16, 3, [1, 4, 16]),
	(16, 4, [1, 2, 8, 16]), (12, 3, [1, 4, 12]), (6, 3, [1, 2, 6]),
	(12, 4, [1, 2, 6, 12]), (18, 3, [1, 3, 18]),
])
def test_known_chains(block_size, levels, expected):
	assert schedules.nested_steps(block_size, levels) == expected


@pytest.mark.parametrize("block_size,levels", [(3, 3), (5, 3), (7, 3), (6, 4), (12, 5)])
def test_impossible_requests_raise_with_true_limit(block_size, levels):
	with pytest.raises(ValueError, match=f"at most {_omega(block_size) + 1} nested"):
		schedules.nested_steps(block_size, levels)


def test_integer_like_inputs_are_accepted():
	assert schedules.nested_steps(np.int64(6), np.int64(3)) == [1, 2, 6]
	assert schedules.nested_steps(8.0, 4) == [1, 2, 4, 8]


@pytest.mark.parametrize("block_size,levels", [(1, 2), (0, 2), (4, 1), (6.5, 2), (6, 2.5)])
def test_bad_inputs_raise_value_error(block_size, levels):
	with pytest.raises(ValueError):
		schedules.nested_steps(block_size, levels)


def test_deterministic():
	for block_size in (6, 12, 18, 24, 30, 36, 48, 60):
		for levels in (3, 4):
			try:
				first = schedules.nested_steps(block_size, levels)
			except ValueError:
				continue
			assert all(schedules.nested_steps(block_size, levels) == first for _ in range(3))


CASES = [
	(12, 6, 3, True), (12, 6, 4, True), (12, 3, 3, True),
	(10, 4, 3, False), (11, 4, 3, False), (13, 4, 3, True),
	(24, 12, 4, True), (16, 8, 4, True),
]


@pytest.mark.parametrize("reuse_mode", ["denoised", "derivative", "exact"])
@pytest.mark.parametrize("num_steps,frequency,n_levels,skip_last", CASES)
def test_rx_sampler_never_crashes(num_steps, frequency, n_levels, skip_last, reuse_mode):
	problem = toy.bimodal()
	t = schedules.edm_schedule(num_steps)
	x_init = np.random.default_rng(0).normal(size=(8, 1)) * t[0]
	result = samplers.rx_sampler(
		problem.denoise, x_init, t, frequency=frequency, n_levels=n_levels,
		skip_last=skip_last, reuse_mode=reuse_mode,
	)
	assert np.all(np.isfinite(result.x))

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
			extra += sum(number - 1 for number in levels[:-1])
	assert result.nfe == num_steps + extra

	if skip_last:
		config = Config(method="rx", num_steps=num_steps, frequency=frequency,
						n_levels=n_levels, reuse_mode=reuse_mode)
		assert config.expected_nfe == result.nfe


def test_non_power_of_two_block_actually_extrapolates():
	problem = toy.single_gaussian()
	x_init = np.random.default_rng(0).normal(size=(256, 1)) * 80.0
	errors = []
	for num_steps in (96, 192, 384):
		t = schedules.edm_schedule(num_steps)
		truth = problem.exact(x_init, t[0], t[-1])
		result = samplers.rx_sampler(
			problem.denoise, x_init, t, frequency=6, n_levels=3, reuse_mode="exact"
		)
		assert len(result.weights) == num_steps // 6
		assert all(weights.w.size == 3 for weights in result.weights)
		errors.append(np.sqrt(np.mean((result.x - truth) ** 2)))
		euler = samplers.euler_sampler(problem.denoise, x_init, t)
		assert errors[-1] < np.sqrt(np.mean((euler.x - truth) ** 2))

	orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
	assert np.all(orders > 2.7)


class _DeviceLike(torch.Tensor):
	def __array__(self, *args, **kwargs):
		raise TypeError("can't convert device tensor to numpy (simulated)")


def _setup(num_steps=16):
	problem = toy.bimodal()
	t_np = schedules.edm_schedule(num_steps)
	t_dev = torch.as_tensor(t_np).as_subclass(_DeviceLike)
	x0 = torch.randn(4, 1, dtype=torch.float64,
					 generator=torch.Generator().manual_seed(0)) * float(t_np[0])

	def den(x, sigma):
		return torch.as_tensor(problem.denoise(x.numpy(), float(sigma)))

	return problem, t_np, t_dev, x0, den


@pytest.mark.parametrize("n_levels,frequency", [(2, 2), (2, 3), (3, 4)])
@pytest.mark.parametrize("reuse_mode", ["denoised", "exact"])
@pytest.mark.parametrize("work_dtype", ["float64", "float32", "float16"])
def test_rx_on_device_schedule_matches_numpy(n_levels, frequency, reuse_mode, work_dtype):
	problem, t_np, t_dev, x0, den = _setup()
	kwargs = dict(frequency=frequency, n_levels=n_levels, reuse_mode=reuse_mode,
				  work_dtype=np.dtype(work_dtype))
	got = samplers.rx_sampler(den, x0, t_dev, **kwargs)
	reference = samplers.rx_sampler(problem.denoise, x0.numpy(), t_np, **kwargs)
	assert got.nfe == reference.nfe
	assert got.x.dtype == torch.float64
	assert np.array_equal(got.x.numpy(), reference.x)


@pytest.mark.parametrize("coefficients", ["grid_aware", "naive"])
def test_rx_edm_and_naive_on_device_schedule(coefficients):
	_, _, t_dev, x0, den = _setup()
	result = samplers.rx_edm_sampler(
		den, x0, t_dev, frequency=2, n_levels=2, coefficients=coefficients
	)
	assert torch.isfinite(result.x).all()
