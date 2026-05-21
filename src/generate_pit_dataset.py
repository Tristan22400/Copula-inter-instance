"""
generate_pit_dataset.py — Offline episode generation (Stage A + Stage B).

Stage A: sample GP tasks and save raw (X_norm, Y, R_star) to raw_dir.
Stage B: load frozen TabICLv2, run PIT to transform Y → Z, save latent
         episodes to latent_dir.

Usage:
    python src/generate_pit_dataset.py data.n_tasks=5000
    python src/generate_pit_dataset.py data.n_tasks=100 data.raw_dir=./data/debug_raw \
        data.latent_dir=./data/debug_latent
"""

from __future__ import annotations

import os
import sys
import warnings

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_gen import generate_gp_task
from pit import load_tabicl, run_pit


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------ #
    # Stage A: GP task generation                                          #
    # ------------------------------------------------------------------ #
    raw_dir = cfg.data.raw_dir
    os.makedirs(raw_dir, exist_ok=True)
    n_tasks = cfg.data.n_tasks

    print(f"[Stage A] Generating {n_tasks} GP tasks → {raw_dir}")
    for i in tqdm(range(n_tasks), desc="GP tasks"):
        path = os.path.join(raw_dir, f"task_{i:06d}.pt")
        if cfg.data.resume and os.path.exists(path):
            continue
        task = generate_gp_task(cfg)
        torch.save(task, path)

    # ------------------------------------------------------------------ #
    # Stage B: PIT projection via frozen TabICLv2                          #
    # ------------------------------------------------------------------ #
    latent_dir = cfg.data.latent_dir
    os.makedirs(latent_dir, exist_ok=True)

    print(f"[Stage B] Running TabICLv2 PIT → {latent_dir}")
    tabicl = load_tabicl(cfg.tabicl.ckpt, device)

    raw_paths = sorted(
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".pt")
    )

    n_warn = 0
    for i, raw_path in enumerate(tqdm(raw_paths, desc="PIT")):
        out_path = os.path.join(latent_dir, os.path.basename(raw_path))
        if cfg.data.resume and os.path.exists(out_path):
            continue

        task = torch.load(raw_path, map_location=device)

        X_train = task["x_norm_train"].to(device)  # (P, d_x)
        Y_train = task["y_train"].unsqueeze(-1).to(device)  # (P, 1)
        X_test = task["x_norm_test"].to(device)  # (N, d_x)
        Y_test = task["y_test"].unsqueeze(-1).to(device)  # (N, 1)

        Z_train, Z_test, _ = run_pit(
            tabicl,
            X_train,
            Y_train,
            X_test,
            Y_test,
            pit_batch_size=cfg.tabicl.pit_batch_size,
            eps=cfg.tabicl.pit_eps,
        )
        # Z_train: (P, 1), Z_test: (N, 1)
        z_train = Z_train[:, 0].cpu()  # (P,)
        z_test = Z_test[:, 0].cpu()  # (N,)

        # Calibration sanity check
        z_all = torch.cat([z_train, z_test])
        mean_z, std_z = z_all.mean().item(), z_all.std().item()
        if abs(mean_z) > 0.5 or abs(std_z - 1.0) > 0.5:
            n_warn += 1
            warnings.warn(
                f"Task {i}: z-score calibration suspect "
                f"(mean={mean_z:.2f}, std={std_z:.2f}). "
                "Check TabICLv2 marginal calibration.",
                RuntimeWarning,
            )

        latent = {
            "x_norm_train": task["x_norm_train"].cpu(),  # (P, d_x)
            "z_train": z_train,  # (P,)
            "x_norm_test": task["x_norm_test"].cpu(),  # (N, d_x)
            "z_test": z_test,  # (N,)
            "R_star": task["R_star"].cpu(),  # (N, N)
            "mu_star": task["mu_star"].cpu(),  # (N,)
            "sigma_star": task["sigma_star"].cpu(),  # (N,)
            "n_train": task["n_train"],
            "n_test": task["n_test"],
        }
        torch.save(latent, out_path)

    if n_warn > 0:
        print(f"[Stage B] {n_warn}/{len(raw_paths)} tasks had calibration warnings.")
    print("[Stage B] Done.")


if __name__ == "__main__":
    main()
