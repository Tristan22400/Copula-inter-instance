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


def _generate_shard_with_oom_retry(
    cfg, n_this: int, device: str, *, tabicl_model, tabicl_k_folds: int,
) -> list:
    """Generate n_this episodes for one shard, halving the chunk size and
    retrying on CUDA OOM instead of killing a multi-day generation run.

    data_gen.py::_max_batch_for_context already estimates a safe per-call
    batch size up front from live free VRAM, so this should rarely fire --
    it's a safety net for when that estimate is wrong (e.g. another process
    sharing the GPU, or the frozen TabICL marginal's own memory footprint
    when cfg.data.z_train_source="tabicl" varying with P in a way the
    estimate doesn't fully capture).

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
    """
    base_seed = getattr(cfg, "seed", None)
    episodes: list = []
    remaining = n_this
    chunk = n_this
    chunk_idx = 0
    while remaining > 0:
        this_chunk = min(chunk, remaining)
        if base_seed is not None:
            cfg.seed = base_seed + chunk_idx * 900_001
        try:
            episodes += generate_gp_batch(
                cfg, this_chunk, device,
                tabicl_model=tabicl_model, tabicl_k_folds=tabicl_k_folds,
            )
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


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    pit_dir = cfg.data.pit_dir
    os.makedirs(pit_dir, exist_ok=True)

    n_tasks    = cfg.data.n_tasks
    B          = int(cfg.data.get("shard_size", 256))
    n_shards   = (n_tasks + B - 1) // B
    base_seed  = getattr(cfg, "seed", None)

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

    print(f"Generating {n_tasks} episodes → {pit_dir}")
    print(f"Batch/shard size: {B}  |  Total shards: {n_shards}  |  Device: {device}  |  "
          f"z_train_source: {z_train_source}")

    # meta.pt from the start (n_total=0) so CopulaDataset never sees a
    # shard_*.pt without a matching meta.pt during the very first shard.
    _write_meta(pit_dir, 0, B)

    n_generated = 0
    with tqdm(total=n_tasks, desc="episodes", unit="ep") as pbar:
        for shard_idx in range(n_shards):
            out_path = os.path.join(pit_dir, f"shard_{shard_idx:06d}.pt")

            n_this = min(B, n_tasks - n_generated)

            if cfg.data.resume and os.path.exists(out_path):
                n_generated += n_this
                pbar.update(n_this)
                _write_meta(pit_dir, n_generated, B)
                continue

            # generate_gp_batch reads cfg.seed to seed torch's RNG; vary it per
            # shard so shards don't restart from the identical RNG state.
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
            torch.save(episodes, out_path)

            n_generated += n_this
            pbar.update(n_this)
            # Update after the shard write completes, never before — meta.pt's
            # n_total must never claim a shard that isn't fully on disk yet.
            _write_meta(pit_dir, n_generated, B)

    print(f"Done. {n_shards} shards written to {pit_dir}")


if __name__ == "__main__":
    main()
