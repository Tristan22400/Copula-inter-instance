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
    python src/generate_pit_dataset.py data.n_tasks=5000
    python src/generate_pit_dataset.py data.n_tasks=5000000 data.shard_size=512

    # z_train from the real frozen TabICL marginal's K-fold PIT instead of
    # the exact analytic GP-LOO residual (see data.z_train_source in
    # conf/data/gp_tasks.yaml) — substantially slower, pilot on a small
    # n_tasks first:
    python src/generate_pit_dataset.py data.n_tasks=5000 data.z_train_source=tabicl
"""

from __future__ import annotations

import gc
import os
import sys
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

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_gen import generate_gp_batch


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


def _write_meta(pit_dir: str, n_total: int, shard_size: int) -> None:
    """Atomically (write-temp + rename) refresh meta.pt so a concurrent
    reader (e.g. train.py starting mid-generation) never observes a torn
    file or an n_total ahead of the shards actually on disk."""
    meta_path = os.path.join(pit_dir, "meta.pt")
    tmp_path  = meta_path + f".tmp{os.getpid()}"
    torch.save({"n_total": n_total, "shard_size": shard_size}, tmp_path)
    os.replace(tmp_path, meta_path)


def _save_shard_atomic(episodes: list, out_path: str) -> None:
    """torch.save then os.replace, never a direct save to out_path.

    With a single writer this was already safe (nothing reads a shard until
    meta.pt claims it exists). With multiple GEN_WORKERS processes writing
    into the same pit_dir, another worker's meta.pt refresh (_scan_meta_total
    below) globs shard_*.pt directly, and a torch.save() in progress on this
    path is not atomic from a glob'ing reader's point of view -- rename on
    the same filesystem is, so a reader only ever sees a fully-written file
    or none at all.
    """
    tmp_path = out_path + f".tmp{os.getpid()}"
    torch.save(episodes, tmp_path)
    os.replace(tmp_path, out_path)


def _scan_meta_total(pit_dir: str, n_tasks: int, n_shards: int, shard_size: int) -> int:
    """Recompute n_total from the shard_*.pt files actually present on disk,
    rather than a per-process running counter -- the only option once
    multiple GEN_WORKERS processes are writing disjoint shard indices into
    the same pit_dir (each process's own counter only knows about the shards
    *it* wrote, not its siblings').

    Every shard has exactly shard_size episodes except the single highest-
    numbered one (index n_shards-1), which gets whatever remainder n_tasks
    doesn't evenly divide by shard_size -- both worker and non-worker runs
    compute n_this from shard_idx the same deterministic way (see main), so
    this can size each present file from its filename alone without loading
    it. Safe to call after any shard write regardless of which worker's turn
    it is: it only ever reports shards that actually exist right now (same
    "never claim a shard that isn't fully on disk yet" invariant as before),
    and _save_shard_atomic's rename means a shard file is never counted
    half-written.
    """
    last_shard_size = n_tasks - shard_size * (n_shards - 1)
    total = 0
    for fname in os.listdir(pit_dir):
        if not (fname.startswith("shard_") and fname.endswith(".pt")):
            continue
        try:
            idx = int(fname[len("shard_"):-len(".pt")])
        except ValueError:
            continue
        total += shard_size if idx < n_shards - 1 else last_shard_size
    return total


@hydra.main(config_path="../conf", config_name="config", version_base=None)
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
    # than reloading per shard.
    z_train_source = cfg.data.get("z_train_source", "analytic")
    tabicl_model = None
    tabicl_k_folds = int(cfg.data.get("z_train_tabicl_k_folds", 10))
    if z_train_source == "tabicl":
        from pit import load_tabicl
        from train import _resolve_pit_ckpt

        ckpt = _resolve_pit_ckpt(cfg)
        if ckpt is None:
            raise ValueError(
                "data.z_train_source=tabicl requires a resolvable TabICL checkpoint "
                "-- set tabicl.ckpt (with tabicl.pretrained=true) or tabicl.pit_ckpt."
            )
        print(f"Loading frozen TabICL marginal for data.z_train_source=tabicl: {ckpt}")
        tabicl_model = load_tabicl(ckpt, device)
    elif z_train_source != "analytic":
        raise ValueError(
            f"Unknown data.z_train_source {z_train_source!r}; expected 'analytic' or 'tabicl'."
        )

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
    _write_meta(pit_dir, _scan_meta_total(pit_dir, n_tasks, n_shards, B), B)

    with tqdm(total=n_tasks_this_worker, desc=f"episodes[w{worker_id}]", unit="ep") as pbar:
        for shard_idx in worker_shard_idxs:
            out_path = os.path.join(pit_dir, f"shard_{shard_idx:06d}.pt")

            # Computed from shard_idx directly (not a running counter) since
            # a worker's shard indices are a strided subset of range(n_shards),
            # not a contiguous prefix -- see worker_shard_idxs above.
            n_this = min(B, n_tasks - shard_idx * B)

            if cfg.data.resume and os.path.exists(out_path):
                pbar.update(n_this)
                continue

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
            _save_shard_atomic(episodes, out_path)

            pbar.update(n_this)
            # Update after the shard write completes, never before — meta.pt's
            # n_total must never claim a shard that isn't fully on disk yet.
            # Rescanned from disk (not this worker's local n_this sum) so
            # sibling workers' concurrently-written shards are reflected too.
            _write_meta(pit_dir, _scan_meta_total(pit_dir, n_tasks, n_shards, B), B)

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
