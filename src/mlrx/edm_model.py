"""Loading the pretrained EDM network and adapting it to our sampler interface.

Three concerns are separated here that the reference implementation mixes
together, because the round-off study needs to vary them independently:

``net_dtype``
    Precision of the *network forward pass*.  EDM's networks carry a
    ``use_fp16`` flag and a ``force_fp32`` argument.
``ode_dtype``
    Precision of the *ODE state and arithmetic*.  EDM runs the sampler in
    float64 regardless of the network's precision, and we keep that default so
    the baselines match published numbers.
``work_dtype``
    Precision of the *extrapolation weight solve and combination*, passed
    through to :func:`mlrx.samplers.rx_sampler`.

Conflating these would make the precision study meaningless: a change in FID
could then be attributed to the extrapolation arithmetic when it actually came
from the network forward pass.

Loading the ``.pkl`` requires NVlabs/edm on ``sys.path`` -- the pickle refers to
classes in ``training.networks`` and helpers in ``torch_utils`` / ``dnnlib``.
:func:`ensure_edm_on_path` handles that and fails with an actionable message
rather than a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "ensure_edm_on_path",
    "load_edm_network",
    "StackedRandomGenerator",
    "EDMDenoiser",
    "make_latents",
]


# ---------------------------------------------------------------------------
# import plumbing
# ---------------------------------------------------------------------------

def ensure_edm_on_path(edm_root=None):
    """Put the NVlabs/edm checkout on ``sys.path``.

    Searched in order: the explicit argument, ``$EDM_ROOT``, then a few
    conventional locations relative to this repository and to a Kaggle input
    mount.
    """
    candidates = []
    if edm_root:
        candidates.append(edm_root)
    if os.environ.get("EDM_ROOT"):
        candidates.append(os.environ["EDM_ROOT"])
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    candidates += [
        os.path.join(repo, "third_party", "edm"),
        os.path.join(repo, "..", "edm"),
        "/kaggle/working/third_party/edm",
    ]

    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "torch_utils")):
            c = os.path.abspath(c)
            if c not in sys.path:
                sys.path.insert(0, c)
            return c

    raise RuntimeError(
        "Could not locate the NVlabs/edm checkout, which is required to "
        "unpickle the pretrained network (it references training.networks, "
        "torch_utils and dnnlib).\n"
        "Set EDM_ROOT, or pass edm_root=..., pointing at a directory that "
        "contains a 'torch_utils' folder.\n"
        f"Searched: {[c for c in candidates if c]}"
    )


def load_edm_network(pkl_path, device="cuda", edm_root=None, net_dtype=None):
    """Load the EMA network from an EDM ``.pkl``.

    ``net_dtype`` of ``float16`` flips the network's internal ``use_fp16``; the
    sampler still holds state in ``ode_dtype``.  Anything else leaves the
    network in float32, which is EDM's default and what published FIDs use.
    """
    ensure_edm_on_path(edm_root)
    with open(pkl_path, "rb") as f:
        net = pickle.load(f)["ema"]
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    if net_dtype is not None and np.dtype(net_dtype) == np.float16:
        if hasattr(net, "use_fp16"):
            net.use_fp16 = True
    elif hasattr(net, "use_fp16"):
        net.use_fp16 = False
    return net


# ---------------------------------------------------------------------------
# reproducible latents (verbatim from EDM / refcode)
# ---------------------------------------------------------------------------

class StackedRandomGenerator:
    """Per-sample seeded RNG, so a given seed yields the same latent always.

    Transcribed from the reference implementation.  This is what makes the
    "common random numbers" protocol work: every method is evaluated on the
    *same* set of initial latents, so differences in FID are attributable to
    the solver rather than to sampling noise.  Variance reduction by common
    random numbers is standard Monte-Carlo practice and is what lets us resolve
    FID gaps smaller than the raw seed-to-seed spread.
    """

    def __init__(self, device, seeds):
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32))
            for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=g, **kwargs) for g in self.generators]
        )

    def randn_like(self, x):
        return self.randn(x.shape, dtype=x.dtype, layout=x.layout, device=x.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randint(*args, size=size[1:], generator=g, **kwargs)
             for g in self.generators]
        )


def make_latents(net, seeds, device="cuda", ode_dtype=torch.float64,
                 class_idx=None):
    """Latents and class labels for a batch of seeds.

    Returns ``(latents, class_labels)``.  The latents are *unscaled*; the
    sampler multiplies by ``t_steps[0]``, matching EDM.
    """
    rnd = StackedRandomGenerator(device, seeds)
    latents = rnd.randn(
        [len(seeds), net.img_channels, net.img_resolution, net.img_resolution],
        device=device,
    ).to(ode_dtype)

    class_labels = None
    if getattr(net, "label_dim", 0):
        idx = rnd.randint(net.label_dim, size=[len(seeds)], device=device)
        class_labels = torch.eye(net.label_dim, device=device)[idx]
        if class_idx is not None:
            class_labels[:] = 0
            class_labels[:, class_idx] = 1
    return latents, class_labels


# ---------------------------------------------------------------------------
# the denoiser adapter
# ---------------------------------------------------------------------------

@dataclass
class EDMDenoiser:
    """Adapts an EDM network to the ``denoiser(x, sigma)`` interface.

    The sampler is written to be backend-agnostic and knows nothing about class
    conditioning, sigma rounding, or dtype juggling; all of that lives here.
    """

    net: object
    class_labels: object = None
    ode_dtype: object = torch.float64
    net_dtype: object = None

    def round_sigma(self, sigma):
        """Snap a noise level to one the network was trained to accept."""
        return self.net.round_sigma(sigma)

    @torch.no_grad()
    def __call__(self, x, sigma):
        sigma_t = torch.as_tensor(sigma, device=x.device, dtype=self.ode_dtype)
        net_in = x.to(torch.float32 if self.net_dtype is None else self.net_dtype)
        out = self.net(net_in, sigma_t, self.class_labels)
        return out.to(self.ode_dtype)

    def schedule(self, num_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0,
                 device="cuda"):
        """EDM schedule clipped to the network's supported range and rounded.

        The clipping and rounding are the network's business, not the
        schedule's, which is why this lives on the denoiser rather than in
        :mod:`mlrx.schedules`.
        """
        smin = max(sigma_min, float(self.net.sigma_min))
        smax = min(sigma_max, float(self.net.sigma_max))
        i = torch.arange(num_steps, dtype=self.ode_dtype, device=device)
        a, b = smax ** (1 / rho), smin ** (1 / rho)
        t = (a + i / (num_steps - 1) * (b - a)) ** rho
        return torch.cat([self.net.round_sigma(t), torch.zeros_like(t[:1])])
