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
    R_star, Sigma_star, mu_star, sigma_star     — posterior oracle
    n_train, n_test                             — episode sizes

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
"""

from __future__ import annotations

import os
import sys

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_gen import generate_gp_batch


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

    print(f"Generating {n_tasks} episodes → {pit_dir}")
    print(f"Batch/shard size: {B}  |  Total shards: {n_shards}  |  Device: {device}")

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
            episodes = generate_gp_batch(cfg, n_this, device)
            torch.save(episodes, out_path)

            n_generated += n_this
            pbar.update(n_this)
            # Update after the shard write completes, never before — meta.pt's
            # n_total must never claim a shard that isn't fully on disk yet.
            _write_meta(pit_dir, n_generated, B)

    print(f"Done. {n_shards} shards written to {pit_dir}")


if __name__ == "__main__":
    main()
