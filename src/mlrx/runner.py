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
    """One fully specified image-generation + FID measurement.

    A Config is just a bundle of hyperparameters. It is frozen (immutable) so
    it can be hashed and deduplicated. The `key` property is a short SHA-1
    digest of all fields except `tag`, so two configs that differ only in tag
    (e.g. the same Euler run appearing in both the "validity" and "panel"
    sweeps) are treated as one piece of work and run only once.
    """

    method: str = "euler"                 # euler | heun | rx | rx_edm
    num_steps: int = 10
    frequency: int = 2                    # block size K (fine steps per block)
    n_levels: int = 2                     # extrapolation levels L
    coefficients: str = "grid_aware"      # grid_aware | naive
    reuse_mode: str = "denoised"          # denoised | derivative | exact
    rho: float = 7.0                      # noise schedule exponent
    work_dtype: str = "float64"           # precision of extrapolation weights
    ode_dtype: str = "float64"            # precision of ODE state tensors
    net_dtype: str = "float32"            # precision of neural network forward pass
    n_images: int = 10_000               # number of images to generate for FID
    seed_offset: int = 0                  # distinct block => independent FID estimate
    n_heun_steps: int = -1                # rx_edm only; -1 => use heun_fraction
    heun_fraction: float = 0.5
    tag: str = ""                         # free-form grouping label (excluded from key)

    @property
    def key(self):
        """Stable identity used for resume.  Order-independent and readable.

        We hash all fields except 'tag' so that the same physical config
        requested by two different studies (e.g. Euler at NFE 10 in both
        'validity' and 'panel') is run once and reused for both figures.
        """
        d = {k: v for k, v in sorted(asdict(self).items()) if k != "tag"}
        blob = json.dumps(d, sort_keys=True, default=str)
        # Take the first 16 characters of the SHA-1 hex digest as the key.
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    @property
    def expected_nfe(self):
        """Cost in neural-network evaluations, known before running.

        This is used to order the sweep cheapest-first and to price it with
        estimate_cost() before a single image is generated.
        """
        if self.method == "euler":
            # Euler: one evaluation per step, simple.
            return self.num_steps
        if self.method == "heun":
            # Heun: two evaluations per step, minus one because the last step
            # skips the corrector (EDM convention, avoids dividing by zero).
            return 2 * self.num_steps - 1
        if self.method == "rx":
            # RX: same as Euler in reuse mode; "exact" mode costs extra
            # evaluations for the coarse levels.
            return self.num_steps + self._exact_overhead(self.num_steps)
        if self.method == "rx_edm":
            # RX+EDM: Heun for the first nh steps, then RX for the rest.
            nh = (self.n_heun_steps if self.n_heun_steps >= 0
                  else int(round(self.heun_fraction * self.num_steps)))
            nh = max(0, min(self.num_steps, nh))
            tail = self.num_steps - nh
            return (2 * nh - (1 if tail == 0 else 0) + tail
                    + self._exact_overhead(tail))
        raise ValueError(f"unknown method {self.method!r}")

    def _exact_overhead(self, steps):
        """Extra evaluations spent by ``reuse_mode="exact"`` over ``steps``.

        In reuse mode every coarse level recycles evaluations from the fine
        trajectory — zero extra cost. In exact mode each coarse level must
        evaluate the network at every intermediate node except the block's
        first (which is shared with the fine trajectory).
        """
        if self.reuse_mode != "exact" or steps < 2:
            return 0
        from . import schedules
        extra = 0
        plan = schedules.partition_blocks(steps, self.frequency)
        for b in plan:
            if not b.extrapolate or b.n_steps < 2:
                continue
            if self.n_levels > 2:
                try:
                    levels = schedules.nested_steps(b.n_steps, self.n_levels)
                except ValueError:
                    continue
            else:
                levels = [1, b.n_steps]
            # Each coarse level of n sub-steps needs n-1 extra evaluations
            # (the first node is shared with the fine trajectory for free).
            extra += sum(n - 1 for n in levels[:-1])
        return extra

    def describe(self):
        """Short human-readable description for progress logging."""
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

# All column names written to the results CSV. Adding a column here and in
# run_config() is the only change needed to record a new metric.
FIELDS = [
    "key", "tag", "method", "num_steps", "frequency", "n_levels",
    "coefficients", "reuse_mode", "rho", "work_dtype", "ode_dtype", "net_dtype",
    "n_images", "seed_offset", "n_heun_steps", "heun_fraction",
    "nfe", "fid", "max_cond", "mean_amplification", "max_amplification",
    "wall_seconds", "gpu",
]


class ResultStore:
    """Append-only CSV of completed configurations, with resume support.

    Each completed Config gets one row appended immediately after it finishes.
    The file is fsynced after each write so a killed session cannot lose the
    row. On restart, already-done keys are loaded from the CSV and skipped.
    """

    def __init__(self, path):
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        # Load the set of already-completed config keys from disk at startup.
        self._done = self._load_keys()

    def _load_keys(self):
        """Read the CSV and collect all config keys that are already done."""
        import csv
        keys = set()
        if os.path.exists(self.path):
            with open(self.path, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("key"):
                        keys.add(row["key"])
        return keys

    def __contains__(self, cfg):
        """Allow `if cfg in store:` to check whether a config is done."""
        return cfg.key in self._done

    def append(self, cfg, **metrics):
        """Write one row to the CSV, immediately flushing to disk.

        The fsync() call is what makes this crash-safe: without it, the OS
        might buffer the write and lose it if the process is killed.
        """
        import csv
        new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        # Start from empty values, then fill in config fields and metrics.
        row = {f: "" for f in FIELDS}
        row.update({k: v for k, v in asdict(cfg).items() if k in row})
        row["key"] = cfg.key
        row.update({k: v for k, v in metrics.items() if k in row})
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()   # write column names for a brand-new file
            w.writerow(row)
            # flush() empties Python's buffer; fsync() makes the OS write it to
            # disk now, so a session killed a moment later still keeps the row.
            f.flush()
            os.fsync(f.fileno())
        self._done.add(cfg.key)

    def rows(self):
        """Return all stored rows as a list of dicts."""
        import csv
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline="") as f:
            return list(csv.DictReader(f))

    def dataframe(self):
        """Return all stored rows as a pandas DataFrame with numeric columns cast."""
        import pandas as pd
        rows = self.rows()
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # CSV stores everything as strings; cast known numeric columns back.
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
    """Per-worker state: the heavy objects loaded once and reused.

    Loading the neural network and FID detector takes several seconds and
    gigabytes of GPU memory. We load them once per worker process and reuse
    them across all configs assigned to that worker.
    """

    device: str                               # e.g. "cuda:0"
    network_pkl: str                          # path to pretrained EDM network
    detector_pkl: str                         # path to Inception-v3 for FID
    ref_npz: str                              # path to CIFAR-10 FID reference stats
    edm_root: str = None                      # path to NVlabs/edm repo (for imports)
    batch_size: int = 128                     # images per GPU forward pass
    net: object = field(default=None, repr=False)
    detector: object = field(default=None, repr=False)
    ref: object = field(default=None, repr=False)
    _net_dtype: str = field(default=None, repr=False)

    def ensure_loaded(self, net_dtype="float32"):
        """Load network and FID detector if not already in memory.

        Called at the start of each config. If the requested network dtype
        changed (e.g. the previous config used float32 and this one needs
        float16) the network is reloaded; otherwise the cached copy is reused.
        """
        from . import edm_model, fid as fid_mod
        if self.net is None or self._net_dtype != net_dtype:
            # Load the pretrained CIFAR-10 EDM network from its pickle file.
            self.net = edm_model.load_edm_network(
                self.network_pkl, device=self.device,
                edm_root=self.edm_root, net_dtype=net_dtype,
            )
            self._net_dtype = net_dtype
        if self.detector is None:
            # Load the Inception-v3 feature extractor used to compute FID.
            self.detector = fid_mod.load_detector(
                self.detector_pkl, device=self.device, edm_root=self.edm_root
            )
        if self.ref is None:
            # Load the precomputed real-image Inception statistics for CIFAR-10.
            self.ref = fid_mod.load_reference_stats(self.ref_npz)


def run_config(cfg: Config, ctx: GPUContext, progress=None):
    """Generate ``cfg.n_images`` samples and return their FID plus diagnostics.

    This is the innermost function: one Config goes in, one row of results
    comes out. Everything here runs on the GPU.
    """
    import torch
    from . import edm_model, fid as fid_mod, samplers

    # Make sure the network and FID tools are loaded before we start the clock.
    ctx.ensure_loaded(cfg.net_dtype)
    t0 = time.time()

    ode_dtype = getattr(torch, cfg.ode_dtype)
    work_dtype = np.dtype(cfg.work_dtype)

    # Wrap the network in a denoiser object that handles dtype conversion and
    # exposes the same interface as the toy-problem denoiser.
    denoiser_obj = edm_model.EDMDenoiser(
        net=ctx.net, class_labels=None, ode_dtype=ode_dtype,
        net_dtype=getattr(torch, cfg.net_dtype) if cfg.net_dtype != "float32" else None,
    )
    # Build the noise-level schedule [σ_max, ..., σ_min, 0] for this config.
    t_steps = denoiser_obj.schedule(
        cfg.num_steps, rho=cfg.rho, device=ctx.device
    )

    # Accumulator that collects Inception features batch-by-batch so we never
    # need all 10k generated images in GPU memory at once.
    acc = fid_mod.FIDAccumulator(device=ctx.device)
    weights_seen, nfe_seen = [], None

    # Each seed block uses a distinct range of seeds so independent blocks give
    # genuinely independent FID estimates. Within a block, all methods share
    # the same seeds → fair apple-to-apple comparison.
    base_seed = cfg.seed_offset * 1_000_000
    seeds_all = list(range(base_seed, base_seed + cfg.n_images))

    # Process images in batches of ctx.batch_size to fit GPU memory.
    for i in range(0, len(seeds_all), ctx.batch_size):
        seeds = seeds_all[i:i + ctx.batch_size]

        # Draw fixed latents (random noise tensors) from the seeds.
        latents, class_labels = edm_model.make_latents(
            ctx.net, seeds, device=ctx.device, ode_dtype=ode_dtype
        )
        denoiser_obj.class_labels = class_labels

        # Start at the highest noise level: x_init = latents * σ_max.
        x_init = latents * t_steps[0]

        # Keyword arguments for the RX-based samplers.
        kw = dict(
            frequency=cfg.frequency, n_levels=cfg.n_levels, p=2,
            coefficients=cfg.coefficients, reuse_mode=cfg.reuse_mode,
            work_dtype=work_dtype,
            collect_weights=(i == 0),   # only collect weights for the first batch
        )

        # Dispatch to the appropriate sampler based on the method name.
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

        # Save conditioning diagnostics from the first batch (they don't change
        # across batches since the schedule and block structure are fixed).
        if res.weights:
            weights_seen = res.weights
        nfe_seen = res.nfe

        # Convert generated float images → uint8 → feed into Inception-v3.
        # The FIDAccumulator keeps running sums of μ and Σ so we never store
        # all images at once.
        acc.update(ctx.detector, fid_mod.to_uint8(res.x))
        if progress is not None:
            progress(min(i + ctx.batch_size, len(seeds_all)), len(seeds_all))

    # Finalise the Inception statistics and compute the FID score.
    mu, sigma = acc.finalize()
    score = fid_mod.frechet_distance(mu, sigma, ctx.ref.mu, ctx.ref.sigma)

    # Collect conditioning diagnostics across all extrapolation blocks.
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

def _worker(rank, world_size, configs, store_path, ctx_kwargs, verbose,
            max_consecutive_failures=3):
    """One GPU worker: runs its assigned configs and appends results to disk."""
    import torch
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    ctx = GPUContext(device=device, **ctx_kwargs)
    # Each rank writes to its own per-rank CSV to avoid concurrent writes to
    # the same file. Shards are merged into the canonical CSV by _merge_shards.
    store = ResultStore(store_path.replace(".csv", f".rank{rank}.csv"))

    import traceback

    # Round-robin split: with two GPUs rank 0 takes configs 0, 2, 4, ... and
    # rank 1 takes 1, 3, 5, ...  Because the list is sorted by cost, this also
    # keeps the ranks' workloads roughly balanced.  With one GPU, rank 0 takes all.
    mine = [c for i, c in enumerate(configs) if i % world_size == rank]
    consecutive = 0
    for j, cfg in enumerate(mine):
        # Skip configs already recorded from a previous session.
        if cfg in store:
            continue
        try:
            metrics = run_config(cfg, ctx)
        except Exception as exc:
            # One failure may be transient (e.g. out of memory); keep going.
            # Repeated failures are a bug, and continuing would silently burn
            # GPU time on configurations that can never succeed.
            consecutive += 1
            print(f"[rank {rank}] FAILED {cfg.describe()}: {exc!r}", flush=True)
            if consecutive == 1:
                traceback.print_exc()
            if consecutive >= max_consecutive_failures:
                raise RuntimeError(
                    f"{consecutive} consecutive configurations failed on rank "
                    f"{rank}; aborting the sweep.  Completed results are kept "
                    f"and a re-run resumes from them."
                ) from exc
            continue
        consecutive = 0
        # Immediately persist the result so it survives a crash.
        store.append(cfg, **metrics)
        if verbose:
            print(f"[rank {rank}] {j + 1}/{len(mine)}  {cfg.describe()}  "
                  f"FID={metrics['fid']:.3f}  ({metrics['wall_seconds']:.0f}s)",
                  flush=True)


def run_sweep(configs, store_path, ctx_kwargs, n_gpus=None, verbose=True,
              max_consecutive_failures=3):
    """Run every configuration not already present in the store.

    Returns the merged :class:`ResultStore`.  Safe to call repeatedly: work
    already done is skipped, so a killed session resumes where it stopped.
    """
    import torch

    # Cheapest first (cost = network evaluations = NFE x images): a session that
    # runs out of time has then finished as many configs as possible, and a
    # broken setup fails within the first minute rather than an hour in.
    configs = sorted(configs, key=lambda c: (c.expected_nfe * c.n_images))
    if n_gpus is None:
        n_gpus = max(1, torch.cuda.device_count())

    # Fold in any per-rank shard CSVs left behind by an earlier interrupted run.
    _merge_shards(store_path)
    pending = [c for c in configs if c not in ResultStore(store_path)]
    if verbose:
        total_units = sum(c.expected_nfe * c.n_images for c in pending)
        print(f"{len(pending)}/{len(configs)} configurations pending "
              f"({total_units / 1e6:.1f}M image-NFE) across {n_gpus} GPU(s)")

    if not pending:
        # All configs already done — nothing to run.
        return ResultStore(store_path)

    try:
        if n_gpus == 1:
            # Single GPU: run synchronously in this process (simpler, easier to debug).
            _worker(0, 1, pending, store_path, ctx_kwargs, verbose,
                    max_consecutive_failures)
        else:
            # Multiple GPUs: spawn one worker process per GPU using PyTorch's
            # multiprocessing utilities. Each worker gets a disjoint slice of
            # the config list via round-robin assignment inside _worker.
            import torch.multiprocessing as mp
            mp.spawn(_worker, nprocs=n_gpus,
                     args=(n_gpus, pending, store_path, ctx_kwargs, verbose,
                           max_consecutive_failures),
                     join=True)
    finally:
        # Even on abort, completed work is merged into the canonical CSV so
        # the next run can skip already-done configs.
        _merge_shards(store_path)

    return ResultStore(store_path)


def _merge_shards(store_path, n_gpus=None):
    """Fold every per-rank CSV into the canonical store, then delete it.

    After a multi-GPU run (or after a crash partway through), each GPU rank
    has its own '.rankN.csv' file. This function reads them all, appends any
    new rows to the main CSV, and deletes the shards.
    """
    import csv
    import glob
    main = ResultStore(store_path)
    for shard in sorted(glob.glob(store_path.replace(".csv", ".rank*.csv"))):
        with open(shard, newline="") as f:
            rows = list(csv.DictReader(f))
        new = not os.path.exists(store_path) or os.path.getsize(store_path) == 0
        with open(store_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
                new = False
            for row in rows:
                # Skip rows already in the main store (from a partial merge).
                if row["key"] in main._done:
                    continue
                w.writerow({k: row.get(k, "") for k in FIELDS})
                main._done.add(row["key"])
        # Delete the shard after its rows have been incorporated.
        os.remove(shard)
    return ResultStore(store_path)
