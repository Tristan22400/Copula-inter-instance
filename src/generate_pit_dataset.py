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
can build the episode index without loading any shard.

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


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    pit_dir = cfg.data.pit_dir
    os.makedirs(pit_dir, exist_ok=True)

    n_tasks    = cfg.data.n_tasks
    B          = int(cfg.data.get("shard_size", 256))
    n_shards   = (n_tasks + B - 1) // B

    print(f"Generating {n_tasks} episodes → {pit_dir}")
    print(f"Batch/shard size: {B}  |  Total shards: {n_shards}  |  Device: {device}")

    n_generated = 0
    with tqdm(total=n_tasks, desc="episodes", unit="ep") as pbar:
        for shard_idx in range(n_shards):
            out_path = os.path.join(pit_dir, f"shard_{shard_idx:06d}.pt")

            n_this = min(B, n_tasks - n_generated)

            if cfg.data.resume and os.path.exists(out_path):
                n_generated += n_this
                pbar.update(n_this)
                continue

            episodes = generate_gp_batch(cfg, n_this, device)
            torch.save(episodes, out_path)

            n_generated += n_this
            pbar.update(n_this)

    # Write index so CopulaDataset can load without scanning shards
    torch.save({"n_total": n_tasks, "shard_size": B}, os.path.join(pit_dir, "meta.pt"))
    print(f"Done. {n_shards} shards written to {pit_dir}")


if __name__ == "__main__":
    main()
