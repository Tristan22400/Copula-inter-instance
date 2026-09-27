"""Generate GP episodes to disk as shards.

Each shard_XXXXXX.pt is a list of episode dicts from one generate_gp_batch
call with keys x_norm_train, x_norm_test, y_train, y_test, z_train, z_test,
log_pdf_test, R_star, mu_star, sigma_star, n_train, n_test. R_prior and
Sigma_star are not stored; CopulaDataset reconstructs them. meta.pt holds
{"n_total", "shard_size"} for the contiguous prefix of finished shards.

Usage:
    python -m copula_inter.generate_pit_dataset data.n_tasks=5000
    python -m copula_inter.generate_pit_dataset data.n_tasks=5000000 data.shard_size=512
    python -m copula_inter.generate_pit_dataset data.n_tasks=5000 data.z_train_source=tabicl
    python -m copula_inter.generate_pit_dataset data.n_tasks=5000 data.z_train_source=tabicl_split
    python -m copula_inter.generate_pit_dataset data.n_tasks=500 data.z_train_source=tabldm
"""

from __future__ import annotations

import fcntl
import gc
import os
import time
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from copula_inter.pit import TabICLLike

# Set before torch initializes CUDA (reduces allocator fragmentation).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from copula_inter.artifacts import atomic_json_save, atomic_torch_save, file_digest
from copula_inter.backend_registry import GENERIC_MARGINAL_BACKENDS, TABICL_Z_TRAIN_SOURCES, validate_z_train_source
from copula_inter.backend_registry import z_train_source as z_train_source_of
from copula_inter.config_path import config_dir
from copula_inter.data_gen import generate_gp_batch
from copula_inter.dataset_manifest import (
    contiguous_shard_counts,
    ensure_manifest,
    generation_spec,
    shard_count_path,
    stat_identity,
    verified_shard_digest,
)
from copula_inter.episode_contracts import validate_episode

_MAX_CUSOLVER_RETRIES = 8


def _is_transient_cusolver_error(exc: BaseException) -> bool:
    """True for transient CUDA library/allocation errors seen when several generation workers share a GPU.

    Matches cusolver/cublas internal errors, TabICL's pinned-memory fallback
    failure, and the mixed-device error that follows it. These are retried after
    a delay.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return False
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return (
        "cusolver" in msg
        or "cublas" in msg
        or "cpu memory allocation failed" in msg
        or "found at least two devices" in msg
    )


def _generate_shard_with_oom_retry(
    cfg: DictConfig,
    n_this: int,
    device: str,
    *,
    tabicl_model: TabICLLike | None,
    tabicl_k_folds: int,
    tabicl_split_calib_frac: float = 0.0,
    marginal_backend: str | None = None,
    marginal_regressor: Any = None,
    marginal_probs_n: int = 99,
) -> list:
    """Generate n_this episodes for one shard, halving the chunk size on CUDA OOM.

    Transient errors (_is_transient_cusolver_error) are retried unchanged after a
    delay. Each chunk uses its own seed offset and the first chunk's d_features.
    Runs gc.collect() before empty_cache() after an OOM.
    """
    base_seed = getattr(cfg, "seed", None)
    episodes: list = []
    remaining = n_this
    chunk = n_this
    chunk_idx = 0
    cusolver_retries = 0
    d_fixed = None
    while remaining > 0:
        this_chunk = min(chunk, remaining)
        if base_seed is not None:
            cfg.seed = base_seed + chunk_idx * 900_001
        try:
            new_episodes = generate_gp_batch(
                cfg,
                this_chunk,
                device,
                d_override=d_fixed,
                tabicl_model=tabicl_model,
                tabicl_k_folds=tabicl_k_folds,
                tabicl_split_calib_frac=tabicl_split_calib_frac,
                marginal_backend=marginal_backend,
                marginal_regressor=marginal_regressor,
                marginal_probs_n=marginal_probs_n,
            )
            if d_fixed is None and new_episodes:
                d_fixed = int(new_episodes[0]["x_norm_train"].shape[-1])
            episodes += new_episodes
            remaining -= this_chunk
            chunk_idx += 1
        except torch.cuda.OutOfMemoryError:
            if this_chunk == 1:
                raise  # nothing smaller left to try -- a genuine failure
            gc.collect()
            torch.cuda.empty_cache()
            chunk = max(1, this_chunk // 2)
            warnings.warn(
                f"generate_pit_dataset: CUDA OOM generating a {this_chunk}-episode "
                f"chunk; retrying at chunk size {chunk}.",
                RuntimeWarning,
            )
        except RuntimeError as e:
            if not _is_transient_cusolver_error(e):
                raise
            cusolver_retries += 1
            if cusolver_retries > _MAX_CUSOLVER_RETRIES:
                raise
            gc.collect()
            torch.cuda.empty_cache()
            delay = min(1.0 * cusolver_retries, 10.0)
            warnings.warn(
                f"generate_pit_dataset: transient cusolver/cublas error on a "
                f"{this_chunk}-episode chunk (retry {cusolver_retries}/"
                f"{_MAX_CUSOLVER_RETRIES}, likely concurrent-worker CUDA "
                f"context contention); retrying unchanged after {delay:.0f}s.",
                RuntimeWarning,
            )
            time.sleep(delay)
    if base_seed is not None:
        cfg.seed = base_seed
    return episodes


def _write_meta(pit_dir: str, n_total: int, shard_size: int, manifest_digest: str) -> None:
    atomic_torch_save(
        {"n_total": n_total, "shard_size": shard_size, "manifest_digest": manifest_digest},
        os.path.join(pit_dir, "meta.pt"),
    )


def _save_shard_atomic(episodes: list, out_path: str) -> None:
    """Write a shard atomically, then its episode-count sidecar."""
    for episode in episodes:
        validate_episode(episode)
    atomic_torch_save(episodes, out_path)
    atomic_json_save(
        {"count": len(episodes), "sha256": file_digest(out_path), **stat_identity(out_path)},
        shard_count_path(out_path),
    )


def _shard_is_resumable(out_path: str, expected_count: int) -> bool:
    """Return whether a shard and its sidecar form a completed, matching pair."""
    sidecar = shard_count_path(out_path)
    if not sidecar.is_file():
        print(f"  [resume] {out_path} has no completed sidecar; regenerating it")
        return False
    try:
        count, _ = verified_shard_digest(out_path, require_match=True)
    except ValueError as exc:
        if not str(exc).startswith("shard content differs from its sidecar:"):
            raise
        print(f"  [resume] {out_path} has a stale sidecar; regenerating it")
        return False
    if count != expected_count:
        raise ValueError(f"cannot resume {out_path}: expected {expected_count} episodes, found {count}")
    return True


def _scan_meta_total(pit_dir: str, n_tasks: int, n_shards: int, shard_size: int, manifest_digest: str) -> int:
    """Episode count of the contiguous prefix of finished shards, written to meta.pt.

    Resumes from the full shards the current meta.pt already published (shards are never
    removed during generation), so each refresh reads only the new sidecars.
    """
    start = 0
    meta_path = os.path.join(pit_dir, "meta.pt")
    if os.path.isfile(meta_path):
        meta = torch.load(meta_path, map_location="cpu", weights_only=True)
        if meta.get("manifest_digest") == manifest_digest and meta.get("shard_size") == shard_size:
            start = min(int(meta["n_total"]) // shard_size, n_shards)
    counts = contiguous_shard_counts(pit_dir, n_shards, shard_size, start=start)
    total = start * shard_size + sum(counts)
    if start + len(counts) == n_shards and total != n_tasks:
        raise ValueError("final shard count does not match requested n_tasks")
    return total


def _refresh_meta(pit_dir: str, n_tasks: int, n_shards: int, shard_size: int, digest: str) -> None:
    # Scan and publish under one lock so a newer count is never overwritten by an older one.
    with open(os.path.join(pit_dir, "meta.lock"), "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            _write_meta(
                pit_dir,
                _scan_meta_total(pit_dir, n_tasks, n_shards, shard_size, digest),
                shard_size,
                digest,
            )
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _reject_disk_unsupported_z_train_source(z_train_source: str) -> None:
    """Raise for z_train_source values with no on-disk implementation ("y_train")."""
    if z_train_source == "y_train":
        raise ValueError(
            "data.z_train_source=y_train is only supported under "
            "training.live_generation=true (src/copula_inter/live_dataset.py) -- no "
            "on-disk generate_pit_dataset.py implementation exists."
        )


@hydra.main(config_path=config_dir(__file__), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pit_dir = cfg.data.pit_dir
    os.makedirs(pit_dir, exist_ok=True)

    n_tasks = cfg.data.n_tasks
    B = int(cfg.data.get("shard_size", 256))
    n_shards = (n_tasks + B - 1) // B
    base_seed = getattr(cfg, "seed", None)

    # Parallel generation: worker worker_id handles shards with shard_idx % num_workers == worker_id.
    worker_id = int(getattr(cfg, "worker_id", 0))
    num_workers = int(getattr(cfg, "num_workers", 1))
    if not (0 <= worker_id < num_workers):
        raise ValueError(f"worker_id={worker_id} must be in [0, num_workers={num_workers})")

    # Load the TabICL marginal once for tabicl / tabicl_split.
    z_train_source = z_train_source_of(cfg)
    validate_z_train_source(z_train_source)
    _reject_disk_unsupported_z_train_source(z_train_source)
    tabicl_model = None
    marginal_backend = z_train_source if z_train_source in GENERIC_MARGINAL_BACKENDS else None
    marginal_regressor = None
    marginal_probs_n = int(cfg.data.get("z_train_marginal_probs_n", 99))
    tabicl_k_folds = int(cfg.data.get("z_train_tabicl_k_folds", 10))
    tabicl_split_calib_frac = (
        float(cfg.data.get("z_train_split_calib_frac", 1.0)) if z_train_source == "tabicl_split" else 0.0
    )
    ckpt = None
    if z_train_source in TABICL_Z_TRAIN_SOURCES:
        from copula_inter.pit import load_tabicl, resolve_pit_ckpt

        ckpt = resolve_pit_ckpt(cfg)
        if ckpt is None:
            raise ValueError(
                f"data.z_train_source={z_train_source} requires a resolvable TabICL "
                "checkpoint -- set tabicl.ckpt (with tabicl.pretrained=true) or "
                "tabicl.pit_ckpt."
            )
        print(f"Loading frozen TabICL marginal for data.z_train_source={z_train_source}: {ckpt}")
        tabicl_model = load_tabicl(ckpt, device)
    elif marginal_backend is not None:
        # Build one backend regressor for all shards (CPU is allowed here, only slower).
        from eval.spatial.marginal_backends import make_regressor

        print(f"Building {z_train_source} marginal for data.z_train_source={z_train_source} on {device}")
        marginal_regressor = make_regressor(marginal_backend, device=device)

    manifest = ensure_manifest(pit_dir, generation_spec(cfg, ckpt))
    for name in os.listdir(pit_dir):
        if name.startswith("shard_") and name.endswith(".pt"):
            try:
                index = int(name[6:-3])
            except ValueError:
                continue
            if index < 0 or index >= n_shards:
                raise ValueError(f"out-of-range shard in {pit_dir}: {name}")

    worker_shard_idxs = range(worker_id, n_shards, num_workers)
    n_tasks_this_worker = sum(min(B, n_tasks - i * B) for i in worker_shard_idxs)

    print(
        f"Generating {n_tasks} episodes → {pit_dir}"
        + (f"  |  worker {worker_id}/{num_workers} owns {n_tasks_this_worker} of them" if num_workers > 1 else "")
    )
    print(
        f"Batch/shard size: {B}  |  Total shards: {n_shards}  |  Device: {device}  |  z_train_source: {z_train_source}"
    )

    # Initialize meta.pt from the shards already on disk.
    _refresh_meta(pit_dir, n_tasks, n_shards, B, manifest["digest"])

    with tqdm(total=n_tasks_this_worker, desc=f"episodes[w{worker_id}]", unit="ep") as pbar:
        for shard_idx in worker_shard_idxs:
            out_path = os.path.join(pit_dir, f"shard_{shard_idx:06d}.pt")

            # Episode count from shard_idx (a worker's shards are strided).
            n_this = min(B, n_tasks - shard_idx * B)

            if cfg.data.resume and os.path.exists(out_path):
                if _shard_is_resumable(out_path, n_this):
                    pbar.update(n_this)
                    continue

            # Per-shard seed from the global shard index.
            if base_seed is not None:
                cfg.seed = base_seed + shard_idx
            episodes = _generate_shard_with_oom_retry(
                cfg,
                n_this,
                device,
                tabicl_model=tabicl_model,
                tabicl_k_folds=tabicl_k_folds,
                tabicl_split_calib_frac=tabicl_split_calib_frac,
                marginal_backend=marginal_backend,
                marginal_regressor=marginal_regressor,
                marginal_probs_n=marginal_probs_n,
            )
            # Don't store R_prior and Sigma_star (reconstructed at load).
            for ep in episodes:
                ep.pop("R_prior", None)
                ep.pop("Sigma_star", None)
            if len(episodes) != n_this:
                raise ValueError(f"generator returned {len(episodes)} episodes; expected {n_this}")
            _save_shard_atomic(episodes, out_path)

            pbar.update(n_this)
            # Update meta.pt after the shard is written, from a disk scan.
            _refresh_meta(pit_dir, n_tasks, n_shards, B, manifest["digest"])

            # Periodically release cached CUDA blocks (gc.collect() first) so fragmentation
            # doesn't shrink the free memory _max_batch_for_context sees.
            if device == "cuda" and shard_idx % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    print(
        f"Done. Worker {worker_id}/{num_workers} wrote {len(worker_shard_idxs)} of {n_shards} total shards to {pit_dir}"
    )


if __name__ == "__main__":
    main()
