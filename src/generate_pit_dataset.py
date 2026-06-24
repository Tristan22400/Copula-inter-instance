"""
generate_pit_dataset.py — Offline episode generation (Stage A + Stage B).

Stage A: sample GP tasks; save raw (X_norm, Y, R_star, mu_star, sigma_star) to raw_dir.
Stage B: load frozen TabICL; run PIT (K-fold for train, single pass for test);
         save latent episodes carrying both Y- and Z-space tensors plus the
         per-instance marginal log-PDF (used by the Y-space copula loss).

Usage:
    python src/generate_pit_dataset.py data.n_tasks=5000
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
from pit import DEFAULT_K_FOLDS, load_tabicl, run_pit


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- Stage A: GP tasks -----
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

    # ----- Stage B: PIT projection -----
    latent_dir = cfg.data.latent_dir
    os.makedirs(latent_dir, exist_ok=True)
    print(f"[Stage B] Running TabICL PIT → {latent_dir}")
    tabicl = load_tabicl(cfg.tabicl.ckpt, device)

    k_folds = int(cfg.tabicl.get("k_folds", DEFAULT_K_FOLDS))
    eps = float(cfg.tabicl.get("pit_eps", 1.0e-6))

    raw_paths = sorted(
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".pt")
    )

    n_warn = 0
    for i, raw_path in enumerate(tqdm(raw_paths, desc="PIT")):
        out_path = os.path.join(latent_dir, os.path.basename(raw_path))
        if cfg.data.resume and os.path.exists(out_path):
            continue

        task = torch.load(raw_path, map_location=device)

        X_train = task["x_norm_train"].to(device)                  # (P, d_x)
        Y_train = task["y_train"].unsqueeze(-1).to(device)         # (P, 1)
        X_test = task["x_norm_test"].to(device)                    # (N, d_x)
        Y_test = task["y_test"].unsqueeze(-1).to(device)           # (N, 1)

        out = run_pit(tabicl, X_train, Y_train, X_test, Y_test,
                      k_folds=k_folds, eps=eps)

        z_train = out["z_train"][:, 0].cpu()                       # (P,)
        z_test = out["z_test"][:, 0].cpu()                         # (N,)
        log_pdf_test = out["log_pdf_test"][:, 0].cpu()             # (N,)

        # Standardize z using train statistics so marginals are ~N(0,1)
        # regardless of TabICL miscalibration.
        #
        # The log-density correction follows from the change of variables
        # z'_i = (z_i - μ) / σ.  By Sklar, the marginal of y_i in z'-space is:
        #
        #   log p'(y_i) = log p_TabICL(y_i) + log φ(z'_i) − log φ(z_i) − log σ
        #               = log_pdf_test_i + 0.5·(z_i² − z'_i²) − log σ
        #
        # (the 0.5·log(2π) terms cancel).  This keeps y_space_nll correct.
        mu_z = z_train.mean()
        sig_z = z_train.std().clamp(min=1e-4)
        z_test_orig = z_test.clone()
        z_train = (z_train - mu_z) / sig_z
        z_test = (z_test - mu_z) / sig_z
        log_pdf_test = log_pdf_test + 0.5 * (z_test_orig**2 - z_test**2) - sig_z.log()

        # Calibration sanity check on standardized z (train and test separately)
        m_tr, s_tr = z_train.mean().item(), z_train.std().item()
        m_te, s_te = z_test.mean().item(), z_test.std().item()
        degenerate = sig_z.item() < 0.1 or sig_z.item() > 3.0
        if degenerate:
            n_warn += 1
            warnings.warn(
                f"Task {i}: degenerate z (sig_z={sig_z.item():.3f} before standardization).",
                RuntimeWarning,
            )

        # Recover Sigma_star (posterior covariance at test points) from
        # R_star and sigma_star — used as oracle for Y-space NLL benchmark.
        R_star = task["R_star"].cpu()                              # (N, N)
        sigma_star = task["sigma_star"].cpu()                      # (N,)
        Sigma_star = R_star * sigma_star.unsqueeze(0) * sigma_star.unsqueeze(1)

        latent = {
            "x_norm_train": task["x_norm_train"].cpu(),            # (P, d_x)
            "x_norm_test": task["x_norm_test"].cpu(),              # (N, d_x)
            "y_train": task["y_train"].cpu(),                      # (P,)
            "y_test": task["y_test"].cpu(),                        # (N,)
            "z_train": z_train,                                    # (P,)
            "z_test": z_test,                                      # (N,)
            "log_pdf_test": log_pdf_test,                          # (N,)
            "R_star": R_star,                                      # (N, N)
            "Sigma_star": Sigma_star,                              # (N, N)
            "mu_star": task["mu_star"].cpu(),                      # (N,)
            "sigma_star": sigma_star,                              # (N,)
            "n_train": task["n_train"],
            "n_test": task["n_test"],
        }
        torch.save(latent, out_path)

    if n_warn > 0:
        print(f"[Stage B] {n_warn}/{len(raw_paths)} tasks had calibration warnings.")
    print("[Stage B] Done.")


if __name__ == "__main__":
    main()
