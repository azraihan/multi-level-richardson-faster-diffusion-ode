"""The samplers must work when the schedule lives on a GPU.

On CUDA, ``np.asarray`` of a tensor raises, whereas on CPU it silently
succeeds -- so a CPU-only test suite cannot catch code that converts schedule
values carelessly.  ``_DeviceLike`` reproduces the CUDA behaviour on CPU, and
the sampler's output is required to match the pure-numpy path exactly.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlrx import samplers, schedules, toy


class _DeviceLike(torch.Tensor):
    def __array__(self, *args, **kwargs):
        raise TypeError("can't convert device tensor to numpy (simulated)")


def _setup(num_steps=16):
    prob = toy.bimodal()
    t_np = schedules.edm_schedule(num_steps)
    t_dev = torch.as_tensor(t_np).as_subclass(_DeviceLike)
    x0 = torch.randn(4, 1, dtype=torch.float64,
                     generator=torch.Generator().manual_seed(0)) * float(t_np[0])

    def den(x, s):
        return torch.as_tensor(prob.denoise(x.numpy(), float(s)))

    return prob, t_np, t_dev, x0, den


@pytest.mark.parametrize("n_levels,frequency", [(2, 2), (2, 3), (3, 4)])
@pytest.mark.parametrize("reuse_mode", ["denoised", "exact"])
@pytest.mark.parametrize("work_dtype", ["float64", "float32", "float16"])
def test_rx_on_device_schedule_matches_numpy(n_levels, frequency, reuse_mode,
                                             work_dtype):
    prob, t_np, t_dev, x0, den = _setup()
    kw = dict(frequency=frequency, n_levels=n_levels, reuse_mode=reuse_mode,
              work_dtype=np.dtype(work_dtype))
    got = samplers.rx_sampler(den, x0, t_dev, **kw)
    ref = samplers.rx_sampler(prob.denoise, x0.numpy(), t_np, **kw)
    assert got.nfe == ref.nfe
    assert got.x.dtype == torch.float64
    assert np.array_equal(got.x.numpy(), ref.x)


@pytest.mark.parametrize("coefficients", ["grid_aware", "naive"])
def test_rx_edm_and_naive_on_device_schedule(coefficients):
    _, _, t_dev, x0, den = _setup()
    res = samplers.rx_edm_sampler(den, x0, t_dev, frequency=2, n_levels=2,
                                  coefficients=coefficients)
    assert torch.isfinite(res.x).all()
