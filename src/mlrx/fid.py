"""Frechet Inception Distance, single-process and offline.

The reference implementation computes FID through ``torch.distributed``, which
needs a process group even on one GPU.  This is the same computation with the
distributed scaffolding removed and statistics accumulated incrementally, so a
run never holds more than one batch of images in memory.

The Inception network is the StyleGAN3 port of the original TensorFlow
``inception-2015-12-05`` graph, which is what EDM's published FIDs use.  Using
``torchvision``'s Inception instead would shift every number by a
non-negligible amount and make comparison against the paper meaningless, so the
exact pickle is vendored by the prefetch notebook.

A note on sample size
---------------------
FID is a *biased* estimator, and the bias grows as the sample count falls: the
covariance of a 2048-dimensional feature is estimated from ``n`` samples, and
at ``n = 10,000`` that estimate is noticeably worse than at ``n = 50,000``.
The bias is positive -- small-sample FID reads high -- and it is substantial,
typically a point or more on CIFAR-10 at these sizes.

Consequences, which the reports must respect:

* Our 10k-image FIDs are **not** directly comparable to the paper's 50k values.
* They *are* comparable to each other, since every method here uses the same
  sample count and the same latents.
* :func:`fid_sample_size_curve` measures the bias explicitly so the gap between
  our numbers and the published ones can be quantified rather than waved at.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
import torch

__all__ = [
    "InceptionFeatures",
    "FIDAccumulator",
    "load_detector",
    "frechet_distance",
    "load_reference_stats",
    "fid_sample_size_curve",
]

FEATURE_DIM = 2048


def load_detector(pkl_path, device="cuda", edm_root=None):
    """Load the StyleGAN3 Inception-v3 feature extractor."""
    from .edm_model import ensure_edm_on_path
    ensure_edm_on_path(edm_root)
    with open(pkl_path, "rb") as f:
        detector = pickle.load(f)
    return detector.to(device).eval()


@dataclass
class FIDAccumulator:
    """Streaming accumulator for Inception feature mean and covariance.

    Holds the raw sums rather than the normalised statistics so that batches
    can arrive in any order and from any number of workers, and the final
    statistics are formed once at the end.  Everything is float64: the
    covariance is a sum of ~1e4 outer products of features with magnitudes
    around unity, and float32 loses meaningful precision in the accumulation.
    """

    device: str = "cuda"
    n: int = 0
    _sum: torch.Tensor = field(default=None, repr=False)
    _sqsum: torch.Tensor = field(default=None, repr=False)

    def __post_init__(self):
        self._sum = torch.zeros([FEATURE_DIM], dtype=torch.float64, device=self.device)
        self._sqsum = torch.zeros([FEATURE_DIM, FEATURE_DIM], dtype=torch.float64,
                                  device=self.device)

    @torch.no_grad()
    def update(self, detector, images_uint8):
        """Accumulate one batch of ``uint8`` images, shape ``(B, C, H, W)``."""
        if images_uint8.shape[0] == 0:
            return
        img = images_uint8
        if img.shape[1] == 1:
            img = img.repeat([1, 3, 1, 1])
        feats = detector(img.to(self.device), return_features=True).to(torch.float64)
        self._sum += feats.sum(0)
        self._sqsum += feats.T @ feats
        self.n += int(img.shape[0])

    def merge(self, other):
        """Combine another accumulator into this one (for multi-GPU shards)."""
        self._sum += other._sum.to(self.device)
        self._sqsum += other._sqsum.to(self.device)
        self.n += other.n
        return self

    def finalize(self):
        """Return ``(mu, sigma)`` as numpy float64 arrays."""
        if self.n < 2:
            raise ValueError(f"need at least 2 samples for a covariance, got {self.n}")
        mu = self._sum / self.n
        sigma = (self._sqsum - torch.outer(mu, mu) * self.n) / (self.n - 1)
        return mu.cpu().numpy(), sigma.cpu().numpy()


@dataclass
class InceptionFeatures:
    mu: np.ndarray
    sigma: np.ndarray
    n: int = 0


def frechet_distance(mu, sigma, mu_ref, sigma_ref):
    """``||mu - mu_ref||^2 + tr(S + S_ref - 2 (S S_ref)^(1/2))``.

    Uses the same ``scipy.linalg.sqrtm`` path as the reference implementation,
    including taking the real part: the matrix square root of a product of two
    PSD matrices is real in exact arithmetic, but round-off leaves a small
    imaginary component that must be discarded rather than propagated.
    """
    diff = np.square(mu - mu_ref).sum()
    covmean, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    return float(np.real(diff + np.trace(sigma + sigma_ref - covmean * 2)))


def load_reference_stats(npz_path):
    """Load dataset reference statistics from an EDM ``fid-refs`` ``.npz``."""
    with np.load(npz_path) as data:
        return InceptionFeatures(mu=data["mu"], sigma=data["sigma"])


def fid_sample_size_curve(features, ref, sizes=(1000, 2000, 5000, 10000, 20000),
                          n_repeats=3, seed=0):
    """FID as a function of sample count, to quantify small-sample bias.

    ``features`` is an ``(N, 2048)`` array of per-image Inception features.
    Subsamples without replacement and recomputes FID, repeating to get a
    spread.  The resulting curve is what justifies (or refutes) comparing our
    10k numbers against the paper's 50k ones.

    Returns a list of dicts, ready for a CSV.
    """
    rng = np.random.default_rng(seed)
    feats = np.asarray(features, dtype=np.float64)
    rows = []
    for n in sizes:
        if n > feats.shape[0]:
            continue
        for rep in range(n_repeats):
            idx = rng.choice(feats.shape[0], size=n, replace=False)
            sub = feats[idx]
            mu = sub.mean(0)
            sigma = np.cov(sub, rowvar=False)
            rows.append({
                "n_samples": int(n),
                "repeat": int(rep),
                "fid": frechet_distance(mu, sigma, ref.mu, ref.sigma),
            })
    return rows


def to_uint8(x):
    """EDM's canonical conversion from ODE state to image bytes."""
    return (x * 127.5 + 128).clip(0, 255).to(torch.uint8)
