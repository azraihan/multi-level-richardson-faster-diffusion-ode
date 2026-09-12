"""Experiment orchestration: resumable, parallel over GPUs, offline-safe.

Design notes
------------
**Resumability is not optional.** A full sweep is hours of GPU time and Kaggle
sessions die; every completed configuration is appended to a CSV immediately
and skipped on a later run.  Interrupting and restarting is therefore always
safe and never repeats work.

**Parallelism is over configurations, not within them.** Sharding one FID run
across GPUs means merging Inception statistics between processes; parallelising
over whole configurations instead means each worker owns its results end to
end and writes its own rows.  Throughput is the same, the code is far simpler,
and a crash in one worker costs one configuration rather than the batch.

**The work queue is ordered cheapest-first** so that a session that runs out of
time has still produced the widest possible coverage.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = ["Config", "ResultStore", "run_sweep", "GPUContext"]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """One fully specified image-generation + FID measurement."""

    method: str = "euler"                 # euler | heun | rx | rx_edm
    num_steps: int = 10
    frequency: int = 2
    n_levels: int = 2
    coefficients: str = "grid_aware"      # grid_aware | naive
    reuse_mode: str = "denoised"          # denoised | derivative
    rho: float = 7.0
    work_dtype: str = "float64"           # extrapolation arithmetic
    ode_dtype: str = "float64"            # ODE state arithmetic
    net_dtype: str = "float32"            # network forward
    n_images: int = 10_000
    seed_offset: int = 0                  # distinct block => independent estimate
    n_heun_steps: int = -1                # rx_edm only; -1 => use heun_fraction
    heun_fraction: float = 0.5
    tag: str = ""                         # free-form grouping label

    @property
    def key(self):
        """Stable identity used for resume.  Order-independent and readable."""
        d = {k: v for k, v in sorted(asdict(self).items()) if k != "tag"}
        blob = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    @property
    def expected_nfe(self):
        """Cost, known before running.  Used to order and to budget the sweep."""
        if self.method == "euler":
            return self.num_steps
        if self.method == "heun":
            return 2 * self.num_steps - 1
        if self.method == "rx":
            return self.num_steps
        if self.method == "rx_edm":
            nh = (self.n_heun_steps if self.n_heun_steps >= 0
                  else int(round(self.heun_fraction * self.num_steps)))
            nh = max(0, min(self.num_steps, nh))
            tail = self.num_steps - nh
            return 2 * nh - (1 if tail == 0 else 0) + tail
        raise ValueError(f"unknown method {self.method!r}")

    def describe(self):
        if self.method == "rx":
            extra = f"K={self.frequency},L={self.n_levels},{self.coefficients}"
            if self.work_dtype != "float64":
                extra += f",{self.work_dtype}"
        elif self.method == "rx_edm":
            extra = f"K={self.frequency},L={self.n_levels},heun={self.heun_fraction:g}"
        else:
            extra = ""
        rho = "" if self.rho == 7.0 else f",rho={self.rho:g}"
        return f"{self.method}[{extra}{rho}] N={self.num_steps} nfe={self.expected_nfe}"


# ---------------------------------------------------------------------------
# result storage
# ---------------------------------------------------------------------------

FIELDS = [
    "key", "tag", "method", "num_steps", "frequency", "n_levels",
    "coefficients", "reuse_mode", "rho", "work_dtype", "ode_dtype", "net_dtype",
    "n_images", "seed_offset", "n_heun_steps", "heun_fraction",
    "nfe", "fid", "max_cond", "mean_amplification", "max_amplification",
    "wall_seconds", "gpu",
]


class ResultStore:
    """Append-only CSV of completed configurations, with resume support."""

    def __init__(self, path):
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._done = self._load_keys()

    def _load_keys(self):
        import csv
        keys = set()
        if os.path.exists(self.path):
            with open(self.path, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("key"):
                        keys.add(row["key"])
        return keys

    def __contains__(self, cfg):
        return cfg.key in self._done

    def append(self, cfg, **metrics):
        import csv
        new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        row = {f: "" for f in FIELDS}
        row.update({k: v for k, v in asdict(cfg).items() if k in row})
        row["key"] = cfg.key
        row.update({k: v for k, v in metrics.items() if k in row})
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        self._done.add(cfg.key)

    def rows(self):
        import csv
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline="") as f:
            return list(csv.DictReader(f))

    def dataframe(self):
        import pandas as pd
        rows = self.rows()
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        num = ["num_steps", "frequency", "n_levels", "rho", "n_images",
               "seed_offset", "nfe", "fid", "max_cond", "mean_amplification",
               "max_amplification", "wall_seconds", "heun_fraction"]
        for c in num:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df


# ---------------------------------------------------------------------------
# GPU-side execution
# ---------------------------------------------------------------------------

@dataclass
class GPUContext:
    """Per-worker state: the heavy objects loaded once and reused."""

    device: str
    network_pkl: str
    detector_pkl: str
    ref_npz: str
    edm_root: str = None
    batch_size: int = 128
    net: object = field(default=None, repr=False)
    detector: object = field(default=None, repr=False)
    ref: object = field(default=None, repr=False)
    _net_dtype: str = field(default=None, repr=False)

    def ensure_loaded(self, net_dtype="float32"):
        from . import edm_model, fid as fid_mod
        if self.net is None or self._net_dtype != net_dtype:
            self.net = edm_model.load_edm_network(
                self.network_pkl, device=self.device,
                edm_root=self.edm_root, net_dtype=net_dtype,
            )
            self._net_dtype = net_dtype
        if self.detector is None:
            self.detector = fid_mod.load_detector(
                self.detector_pkl, device=self.device, edm_root=self.edm_root
            )
        if self.ref is None:
            self.ref = fid_mod.load_reference_stats(self.ref_npz)


def run_config(cfg: Config, ctx: GPUContext, progress=None):
    """Generate ``cfg.n_images`` samples and return their FID plus diagnostics."""
    import torch
    from . import edm_model, fid as fid_mod, samplers

    ctx.ensure_loaded(cfg.net_dtype)
    t0 = time.time()

    ode_dtype = getattr(torch, cfg.ode_dtype)
    work_dtype = np.dtype(cfg.work_dtype)

    denoiser_obj = edm_model.EDMDenoiser(
        net=ctx.net, class_labels=None, ode_dtype=ode_dtype,
        net_dtype=getattr(torch, cfg.net_dtype) if cfg.net_dtype != "float32" else None,
    )
    t_steps = denoiser_obj.schedule(
        cfg.num_steps, rho=cfg.rho, device=ctx.device
    )

    acc = fid_mod.FIDAccumulator(device=ctx.device)
    weights_seen, nfe_seen = [], None

    base_seed = cfg.seed_offset * 1_000_000
    seeds_all = range(base_seed, base_seed + cfg.n_images)
    seeds_all = list(seeds_all)

    for i in range(0, len(seeds_all), ctx.batch_size):
        seeds = seeds_all[i:i + ctx.batch_size]
        latents, class_labels = edm_model.make_latents(
            ctx.net, seeds, device=ctx.device, ode_dtype=ode_dtype
        )
        denoiser_obj.class_labels = class_labels
        x_init = latents * t_steps[0]

        kw = dict(
            frequency=cfg.frequency, n_levels=cfg.n_levels, p=2,
            coefficients=cfg.coefficients, reuse_mode=cfg.reuse_mode,
            work_dtype=work_dtype,
            collect_weights=(i == 0),
        )
        if cfg.method == "euler":
            res = samplers.euler_sampler(denoiser_obj, x_init, t_steps)
        elif cfg.method == "heun":
            res = samplers.heun_sampler(denoiser_obj, x_init, t_steps)
        elif cfg.method == "rx":
            res = samplers.rx_sampler(denoiser_obj, x_init, t_steps, **kw)
        elif cfg.method == "rx_edm":
            nh = cfg.n_heun_steps if cfg.n_heun_steps >= 0 else None
            res = samplers.rx_edm_sampler(
                denoiser_obj, x_init, t_steps, n_heun_steps=nh,
                heun_fraction=cfg.heun_fraction, **kw
            )
        else:
            raise ValueError(f"unknown method {cfg.method!r}")

        if res.weights:
            weights_seen = res.weights
        nfe_seen = res.nfe
        acc.update(ctx.detector, fid_mod.to_uint8(res.x))
        if progress is not None:
            progress(min(i + ctx.batch_size, len(seeds_all)), len(seeds_all))

    mu, sigma = acc.finalize()
    score = fid_mod.frechet_distance(mu, sigma, ctx.ref.mu, ctx.ref.sigma)

    conds = [w.cond for w in weights_seen]
    amps = [w.amplification for w in weights_seen]
    return dict(
        fid=score,
        nfe=nfe_seen,
        max_cond=max(conds) if conds else "",
        mean_amplification=float(np.mean(amps)) if amps else "",
        max_amplification=max(amps) if amps else "",
        wall_seconds=round(time.time() - t0, 2),
        gpu=ctx.device,
    )


# ---------------------------------------------------------------------------
# sweep driver
# ---------------------------------------------------------------------------

def _worker(rank, world_size, configs, store_path, ctx_kwargs, verbose):
    import torch
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    ctx = GPUContext(device=device, **ctx_kwargs)
    store = ResultStore(store_path.replace(".csv", f".rank{rank}.csv"))

    mine = [c for i, c in enumerate(configs) if i % world_size == rank]
    for j, cfg in enumerate(mine):
        if cfg in store:
            continue
        try:
            metrics = run_config(cfg, ctx)
        except Exception as exc:                      # keep the sweep alive
            print(f"[rank {rank}] FAILED {cfg.describe()}: {exc!r}", flush=True)
            continue
        store.append(cfg, **metrics)
        if verbose:
            print(f"[rank {rank}] {j + 1}/{len(mine)}  {cfg.describe()}  "
                  f"FID={metrics['fid']:.3f}  ({metrics['wall_seconds']:.0f}s)",
                  flush=True)


def run_sweep(configs, store_path, ctx_kwargs, n_gpus=None, verbose=True):
    """Run every configuration not already present in the store.

    Returns the merged :class:`ResultStore`.  Safe to call repeatedly: work
    already done is skipped, so a killed session resumes where it stopped.
    """
    import torch

    configs = sorted(configs, key=lambda c: (c.expected_nfe * c.n_images))
    if n_gpus is None:
        n_gpus = max(1, torch.cuda.device_count())

    pending = [c for c in configs if c not in ResultStore(store_path)]
    if verbose:
        total_units = sum(c.expected_nfe * c.n_images for c in pending)
        print(f"{len(pending)}/{len(configs)} configurations pending "
              f"({total_units / 1e6:.1f}M image-NFE) across {n_gpus} GPU(s)")

    if not pending:
        return ResultStore(store_path)

    if n_gpus == 1:
        _worker(0, 1, pending, store_path, ctx_kwargs, verbose)
    else:
        import torch.multiprocessing as mp
        mp.spawn(_worker, nprocs=n_gpus,
                 args=(n_gpus, pending, store_path, ctx_kwargs, verbose),
                 join=True)

    return _merge_shards(store_path, n_gpus)


def _merge_shards(store_path, n_gpus):
    """Fold per-rank CSVs into the canonical store."""
    import csv
    main = ResultStore(store_path)
    for rank in range(max(n_gpus, 1)):
        shard = store_path.replace(".csv", f".rank{rank}.csv")
        if not os.path.exists(shard):
            continue
        with open(shard, newline="") as f:
            rows = list(csv.DictReader(f))
        new = not os.path.exists(store_path) or os.path.getsize(store_path) == 0
        with open(store_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
                new = False
            for row in rows:
                if row["key"] in main._done:
                    continue
                w.writerow({k: row.get(k, "") for k in FIELDS})
                main._done.add(row["key"])
        os.remove(shard)
    return ResultStore(store_path)
