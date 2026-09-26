"""
generate_pit_dataset.py — Fast single-stage episode generation.

Each call to generate_gp_batch() produces B episodes in one vectorised pass
(batched kernel construction, batched Cholesky, batched LOO PIT) and writes
them as a shard file.  This replaces both the two-stage TabICL pipeline and
the per-episode Python loop.

Shard format
------------
Each shard_XXXXXX.pt is a list of B episode dicts with the schema:

    x_norm_train, x_norm_test, y_train, y_test  — raw features / targets
    z_train, z_test, log_pdf_test               — standardised PIT + marginals
    R_star, mu_star, sigma_star                 — oracle (prior, per cfg.data.oracle_mode)
    n_train, n_test                             — episode sizes

R_prior and Sigma_star are NOT stored: with oracle_mode="prior" (the only
supported mode) they're exact functions of R_star/sigma_star already in the
dict (R_prior == R_star; Sigma_star == R_star * outer(sigma_star, sigma_star)
— see data_gen.py's oracle_mode="prior" branch), and together they were 2/3
of on-disk shard size for no new information. dataset.py's CopulaDataset
reconstructs both transparently at load time (see _add_derived_fields), so
this is invisible to every downstream consumer (collate_fn, train.py,
loss.py). Older shards that DO have these keys stored are left untouched
and loaded as-is.

A meta.pt file records {"n_total": int, "shard_size": int} so CopulaDataset
can build the episode index without loading any shard. It is (re)written
after every shard with n_total = episodes completed *so far*, not the final
target — so a training run started mid-generation only ever sees indices
backed by shards that actually exist on disk (no clamping to stale shards,
see CopulaDataset._get_sharded).

Usage
-----
    python -m copula_inter.generate_pit_dataset data.n_tasks=5000
    python -m copula_inter.generate_pit_dataset data.n_tasks=5000000 data.shard_size=512

    # z_train from the real frozen TabICL marginal's K-fold PIT instead of
    # the exact analytic GP-LOO residual (see data.z_train_source in
    # conf/data/gp_tasks.yaml) — substantially slower, pilot on a small
    # n_tasks first:
    python -m copula_inter.generate_pit_dataset data.n_tasks=5000 data.z_train_source=tabicl

    # Same, but via a one-pass calibration split instead of K-fold rotation
    # (data.z_train_split_calib_frac controls the calibration pool size) —
    # ~(z_train_tabicl_k_folds + 1)x fewer TabICL forward passes than
    # z_train_source=tabicl, see run_pit_calib_split_batched's docstring in
    # pit.py for the cost/quality trade-off:
    python -m copula_inter.generate_pit_dataset data.n_tasks=5000 data.z_train_source=tabicl_split

    # ... or any other tabular-foundation-model marginal backend
    # (eval/spatial/marginal_backends.py). Unlike training.live_generation
    # these do not require a GPU here -- offline generation may simply take
    # much longer on CPU. Pilot on a small n_tasks first.
    python -m copula_inter.generate_pit_dataset data.n_tasks=500 data.z_train_source=tabldm
"""

from __future__ import annotations

import gc
import fcntl
import os
import time
import warnings

# Must be set before any CUDA call (i.e. before `import torch`) -- see the
# identical setdefault in train.py. expandable_segments avoids OOMs caused by
# allocator fragmentation (a request failing despite enough total free memory
# because it's split across pieces too small individually), which compounds
# the risk _generate_shard_with_oom_retry below is a safety net for.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm


from copula_inter.data_gen import generate_gp_batch
from copula_inter.artifacts import atomic_json_save, atomic_torch_save, file_digest
from copula_inter.config_path import config_dir
from copula_inter.episode_contracts import validate_episode
from copula_inter.dataset_manifest import (
    contiguous_shard_counts,
    ensure_manifest,
    generation_spec,
    shard_count_path,
    verified_shard_digest,
)
from copula_inter.live_dataset import _GENERIC_MARGINAL_BACKENDS, _validate_z_train_source
from copula_inter.backend_registry import TABICL_Z_TRAIN_SOURCES
from copula_inter.backend_registry import z_train_source as z_train_source_of


_MAX_CUSOLVER_RETRIES = 8


def _is_transient_cusolver_error(exc: BaseException) -> bool:
    """True for the cusolver/cublas/pinned-allocation contention races seen
    when several generate_pit_dataset.py workers (scripts/generate_dataset.sh's
    GEN_WORKERS) call into CUDA driver/context APIs at close to the same
    instant -- e.g. `cusolverDnCreate` returning CUSOLVER_STATUS_INTERNAL_ERROR
    with no OOM involved (mem_get_info showed plenty of free VRAM when this
    was observed empirically running N concurrent workers on one GPU).
    Unlike torch.cuda.OutOfMemoryError, it isn't fixed by a smaller batch --
    it's a transient contention error that clears itself on a short delay and
    retry -- so it must not be confused with a genuine bug's RuntimeError,
    which should still propagate and kill the run.

    Same contention window also hits data.z_train_source="tabicl" runs via a
    different code path: tabicl's InferenceManager._allocate_output_buffer
    (tabicl_upstream/src/tabicl/_model/inference.py) tries a GPU alloc, falls
    back to a *pinned* CPU alloc (cudaHostAlloc) on failure, and that pinned
    alloc itself competes for the same limited GPU-managed pinned-memory pool
    across GEN_WORKERS -- observed raising "CPU memory allocation failed
    (CUDA error: invalid argument...) and disk offload is not available" from
    that fallback (job 3000709, worker 2, 4 consecutive occurrences). The
    same contention window was also seen manifesting one attempt later as
    "Expected all tensors to be on the same device, but found at least two
    devices, cuda:0 and cpu" out of tabicl's quantile_dist.cdf -- a mixed-
    device tensor left over from a GPU/CPU offload-mode fallback that raced
    with another worker's own allocation. Neither is a genuine bug in our
    code (tabicl_upstream is vendored, kept pristine -- see its own retry
    convention here rather than patching it in place), so both are treated
    as transient and retried the same way as the cusolver races above.
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
    cfg, n_this: int, device: str, *, tabicl_model, tabicl_k_folds: int,
    tabicl_split_calib_frac: float = 0.0,
    marginal_backend=None, marginal_regressor=None, marginal_probs_n: int = 99,
) -> list:
    """Generate n_this episodes for one shard, halving the chunk size and
    retrying on CUDA OOM (or backing off and retrying unchanged on a
    transient cusolver/cublas contention error -- see
    _is_transient_cusolver_error) instead of killing a multi-day generation
    run.

    data_gen.py::_max_batch_for_context already estimates a safe per-call
    batch size up front from live free VRAM, so the OOM branch should rarely
    fire -- it's a safety net for when that estimate is wrong (e.g. another
    process sharing the GPU, or the frozen TabICL marginal's own memory
    footprint when cfg.data.z_train_source="tabicl" varying with P in a way
    the estimate doesn't fully capture).

    On OOM: gc.collect() BEFORE empty_cache(). A CUDA OOM's traceback keeps
    the failed batch's tensors alive via a reference cycle (exception ->
    traceback -> frame -> locals -> ... ); plain refcounting doesn't free
    cycles, only the cyclic GC does, so empty_cache() alone would see those
    blocks as still "in use" and reclaim nothing (see the identical fix,
    and the three prior attempts that didn't work, for train.py's OOM
    handler in feedback memory / git history). Chunk calls use a distinct
    cfg.seed offset per chunk so a halved retry doesn't just redraw the
    identical (still-too-large) batch from the same RNG state -- same
    "offset by a large prime" convention generate_gp_batch's own top-up
    loop uses, kept in a different range so the two don't collide.

    Every chunk is pinned to the first chunk's d_features (via
    generate_gp_batch's d_override) for the same reason generate_gp_batch
    pins it across its own top-up rounds: an OOM/cusolver retry here splits
    one shard's episodes across multiple generate_gp_batch calls, and
    without the pin each call would independently sample its own d and the
    shard could come out with internally-mixed feature counts --
    ShardHomogeneousBatchSampler/collate_fn assume that can't happen.
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
                cfg, this_chunk, device, d_override=d_fixed,
                tabicl_model=tabicl_model, tabicl_k_folds=tabicl_k_folds,
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
    """Publish a shard, then its count sidecar for concurrent readers."""
    for episode in episodes:
        validate_episode(episode)
    atomic_torch_save(episodes, out_path)
    stat = os.stat(out_path)
    atomic_json_save(
        {"count": len(episodes), "sha256": file_digest(out_path),
         "size": stat.st_size, "ctime_ns": stat.st_ctime_ns,
         "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino},
        shard_count_path(out_path),
    )


def _scan_meta_total(pit_dir: str, n_tasks: int, n_shards: int, shard_size: int) -> int:
    """Expose only a contiguous prefix, regardless of worker completion order."""
    counts = contiguous_shard_counts(pit_dir, n_shards, shard_size)
    if len(counts) == n_shards and sum(counts) != n_tasks:
        raise ValueError("final shard count does not match requested n_tasks")
    return sum(counts)


def _refresh_meta(pit_dir: str, n_tasks: int, n_shards: int, shard_size: int, digest: str) -> None:
    # Scan and publish under one lock: concurrent writers must not replace a
    # newer contiguous count with an older one. The lock file is persistent.
    with open(os.path.join(pit_dir, "meta.lock"), "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            _write_meta(
                pit_dir, _scan_meta_total(pit_dir, n_tasks, n_shards, shard_size),
                shard_size, digest,
            )
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _reject_disk_unsupported_z_train_source(z_train_source: str) -> None:
    """Raise for any data.z_train_source with no on-disk implementation,
    instead of silently falling through to plain analytic generation --
    exactly the silent-no-op bug class _validate_z_train_source itself was
    built to catch (see its docstring's "tabicl-split" typo root-cause).
    Factored out of main() (which is @hydra.main-wrapped and awkward to
    exercise directly in a test) so this one check stays independently
    testable.

    Currently only "y_train" (see data_gen.py::_generate_gp_batch_raw's
    raw_y_override): threaded through the live-generation path only
    (live_dataset.py) -- no on-disk implementation exists.
    """
    if z_train_source == "y_train":
        raise ValueError(
            "data.z_train_source=y_train is only supported under "
            "training.live_generation=true (src/copula_inter/live_dataset.py) -- no "
            "on-disk generate_pit_dataset.py implementation exists."
        )


@hydra.main(config_path=config_dir(__file__), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    pit_dir = cfg.data.pit_dir
    os.makedirs(pit_dir, exist_ok=True)

    n_tasks    = cfg.data.n_tasks
    B          = int(cfg.data.get("shard_size", 256))
    n_shards   = (n_tasks + B - 1) // B
    base_seed  = getattr(cfg, "seed", None)

    # Multi-process parallel generation (scripts/generate_dataset.sh's
    # GEN_WORKERS): worker_id/num_workers default to 0/1, i.e. the original
    # single-process behaviour. Each worker only ever handles shard indices
    # `shard_idx % num_workers == worker_id`, so N workers pointed at the
    # same data.dataset_dir partition the work with no overlap and no
    # coordination beyond the shared pit_dir -- see _scan_meta_total for how
    # meta.pt stays correct despite each worker only knowing about the
    # shards it personally wrote.
    worker_id    = int(getattr(cfg, "worker_id", 0))
    num_workers  = int(getattr(cfg, "num_workers", 1))
    if not (0 <= worker_id < num_workers):
        raise ValueError(f"worker_id={worker_id} must be in [0, num_workers={num_workers})")

    # z_train source override (see data.z_train_source's docstring in
    # conf/data/gp_tasks.yaml): load the frozen TabICL marginal once, up
    # front, and thread it through every generate_gp_batch call below rather
    # than reloading per shard. "tabicl" and "tabicl_split" share the same
    # checkpoint -- they only differ in how pit.py scores the train set
    # against it (K-fold rotation vs. a one-pass calibration split; see
    # tabicl_split_calib_frac below).
    z_train_source = z_train_source_of(cfg)
    _validate_z_train_source(z_train_source)
    _reject_disk_unsupported_z_train_source(z_train_source)
    tabicl_model = None
    marginal_backend = z_train_source if z_train_source in _GENERIC_MARGINAL_BACKENDS else None
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
        # "exaone"/"tabpfn"/"tabldm": one regressor built here and reused for
        # every shard, exactly like tabicl_model above (avoid reloading
        # backbone weights per .fit() call -- see
        # eval/spatial/marginal_backends.py::make_regressor).
        #
        # Unlike training.live_generation, this pipeline does NOT require
        # device='cuda' for these backends. The live path rejects CPU because
        # a slow per-episode PIT stalls the training loop itself
        # (live_dataset.py::build_live_train_loader); offline dataset
        # generation just takes longer, which is a cost the caller can choose
        # to pay -- and this pipeline already supports --num_workers sharding
        # to spread it. It is still slow enough to warrant a pilot run: see
        # data.z_train_source's docstring in conf/data/gp_tasks.yaml for
        # measured per-episode numbers per backend.
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

    print(f"Generating {n_tasks} episodes → {pit_dir}"
          + (f"  |  worker {worker_id}/{num_workers} owns {n_tasks_this_worker} of them"
             if num_workers > 1 else ""))
    print(f"Batch/shard size: {B}  |  Total shards: {n_shards}  |  Device: {device}  |  "
          f"z_train_source: {z_train_source}")

    # meta.pt reflects whatever shard_*.pt files are actually already on disk
    # (0 on a fresh dataset_dir, >0 if this is a parallel worker joining a
    # run another worker already started, or a restart after data.resume=true)
    # rather than unconditionally resetting to 0 -- the old hardcoded-0 write
    # was fine for a lone process starting fresh, but here it would stomp a
    # sibling worker's already-accurate count of shards it wrote before this
    # process (re)started. See _scan_meta_total.
    _refresh_meta(pit_dir, n_tasks, n_shards, B, manifest["digest"])

    with tqdm(total=n_tasks_this_worker, desc=f"episodes[w{worker_id}]", unit="ep") as pbar:
        for shard_idx in worker_shard_idxs:
            out_path = os.path.join(pit_dir, f"shard_{shard_idx:06d}.pt")

            # Computed from shard_idx directly (not a running counter) since
            # a worker's shard indices are a strided subset of range(n_shards),
            # not a contiguous prefix -- see worker_shard_idxs above.
            n_this = min(B, n_tasks - shard_idx * B)

            if cfg.data.resume and os.path.exists(out_path):
                if shard_count_path(out_path).is_file():
                    count, _ = verified_shard_digest(out_path, require_match=True)
                    if count != n_this:
                        raise ValueError(f"cannot resume {out_path}: expected {n_this} episodes, found {count}")
                    pbar.update(n_this)
                    continue
                print(f"  [resume] {out_path} has no completed sidecar; regenerating it")

            # generate_gp_batch reads cfg.seed to seed torch's RNG; vary it per
            # shard so shards don't restart from the identical RNG state.
            # Global (not per-worker-local) shard_idx keeps every worker's
            # seed stream disjoint from every other worker's, same as the
            # single-process case.
            if base_seed is not None:
                cfg.seed = base_seed + shard_idx
            episodes = _generate_shard_with_oom_retry(
                cfg, n_this, device,
                tabicl_model=tabicl_model, tabicl_k_folds=tabicl_k_folds,
                tabicl_split_calib_frac=tabicl_split_calib_frac,
                marginal_backend=marginal_backend,
                marginal_regressor=marginal_regressor,
                marginal_probs_n=marginal_probs_n,
            )
            # Drop the two fields reconstructible from R_star/sigma_star at
            # load time (see module docstring) -- cuts on-disk shard size by
            # ~2/3 for free. Left untouched in the in-memory dict returned by
            # generate_gp_batch/_generate_shard_with_oom_retry, so live-
            # generation training and any other direct caller keep seeing
            # the full schema.
            for ep in episodes:
                ep.pop("R_prior", None)
                ep.pop("Sigma_star", None)
            if len(episodes) != n_this:
                raise ValueError(f"generator returned {len(episodes)} episodes; expected {n_this}")
            _save_shard_atomic(episodes, out_path)

            pbar.update(n_this)
            # Update after the shard write completes, never before — meta.pt's
            # n_total must never claim a shard that isn't fully on disk yet.
            # Rescanned from disk (not this worker's local n_this sum) so
            # sibling workers' concurrently-written shards are reflected too.
            _refresh_meta(pit_dir, n_tasks, n_shards, B, manifest["digest"])

            # Periodic cache trim: P/N (context length T) are resampled per
            # shard from wide, independent ranges, so the CUDA allocator sees
            # a different (B,T,T) shape almost every call. Without ever
            # trimming, reserved-but-fragmented blocks from earlier (large-T)
            # shards accumulate across a multi-day, 100k+-shard run and are
            # never returned to the driver -- torch.cuda.mem_get_info (which
            # _max_batch_for_context uses to size each call) sees less and
            # less "free" memory over time even though little is genuinely
            # live, so the per-call batch size ratchets down and per-episode
            # overhead (esp. the tabicl_model K-fold forward under
            # z_train_source="tabicl") dominates. gc.collect() must run
            # before empty_cache() -- same reference-cycle reasoning as
            # _generate_shard_with_oom_retry's OOM handler above and
            # train.py's OOM handler (a bare empty_cache() does not reclaim
            # tensors still held alive by a cycle).
            if device == "cuda" and shard_idx % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    print(f"Done. Worker {worker_id}/{num_workers} wrote {len(worker_shard_idxs)} of "
          f"{n_shards} total shards to {pit_dir}")


if __name__ == "__main__":
    main()
