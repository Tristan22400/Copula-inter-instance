"""
plot_covariance_comparison.py — visualize predicted vs. oracle correlation matrices.

For a handful of synthetic GP draws (same sampling logic as
``experiment_b_quantitative.py``), plots side-by-side heatmaps of the copula
model's predicted test-test correlation matrix (``R_test``) against the true
kernel correlation (``R_true``), plus their difference — to see *where*
(which distances / which functions) the model over- or under-estimates
correlation, rather than just the aggregate Frobenius-norm summary.

Usage:
    python experiments/plot_covariance_comparison.py \\
        [--copula-ckpt ./checkpoints/systematic-composition/step_0180000.pt] \\
        [--tabicl-ckpt tabicl-regressor-v2-20260212.ckpt] \\
        [--kernels rbf,matern32] [--seeds 0,1,2,3] \\
        [--n-test 40] [--n-train-min 5] [--n-train-max 15] \\
        [--out-dir ./results/figures] [--device auto]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pit import run_pit  # noqa: E402

from experiments.experiment_b_quantitative import _compute_r_true, _sample_one_function  # noqa: E402
from inference.copula_inference import (  # noqa: E402
    get_test_correlation,
    load_copula_model,
    load_tabicl_marginal,
    normalize_features,
)


@torch.no_grad()
def plot_one(seed: int, tabicl_model, copula_model, oracle_mode: str, args, out_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng_np = np.random.default_rng(seed)
    rng_torch = torch.Generator().manual_seed(seed)
    kernel_name, lengthscale, n_train, X_train_t, y_train_t, X_test_t, y_test_t, kernel_fn = _sample_one_function(
        rng_np, rng_torch, args.n_test, (args.n_train_min, args.n_train_max), args.kernels
    )
    X_train, y_train = X_train_t.numpy(), y_train_t.numpy()
    X_test = X_test_t.numpy()
    n_test = X_test.shape[0]

    R_true = _compute_r_true(oracle_mode, X_train_t, y_train_t, X_test_t, kernel_fn)

    X_train_norm, X_test_norm = normalize_features(X_train, X_test)
    X_train_norm_t = torch.as_tensor(X_train_norm, dtype=X_train_t.dtype)
    X_test_norm_t = torch.as_tensor(X_test_norm, dtype=X_test_t.dtype)

    tabicl_device = next(tabicl_model.parameters()).device
    with torch.no_grad():
        pit_out = run_pit(
            tabicl_model,
            X_train_norm_t.to(tabicl_device), y_train_t.to(tabicl_device).unsqueeze(-1),
            X_test_norm_t.to(tabicl_device), y_test_t.to(tabicl_device).unsqueeze(-1),
            k_folds=min(10, len(X_train)),
        )
        Z_train = pit_out["z_train"].squeeze(-1).cpu().numpy()
        R_test = get_test_correlation(copula_model, X_train_norm, Z_train, X_test_norm)

    eigvals = np.linalg.eigvalsh(R_test)
    corr_frob = float(np.linalg.norm(R_test - R_true, "fro") / n_test)

    diff = R_test - R_true
    vmax_diff = max(abs(diff.min()), abs(diff.max()), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, mat, title, cmap, vmin, vmax in [
        (axes[0], R_test, "Predicted R_test (copula model)", "RdBu_r", -1, 1),
        (axes[1], R_true, "Oracle R_true (kernel)", "RdBu_r", -1, 1),
        (axes[2], diff, "R_test - R_true", "RdBu_r", -vmax_diff, vmax_diff),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("test index")
        ax.set_ylabel("test index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"seed={seed}  kernel={kernel_name}  l={lengthscale:.3f}  n_train={n_train}  "
        f"||diff||_F/N={corr_frob:.4f}  R_test cond={eigvals.max() / max(eigvals.min(), 1e-8):.1f}",
        fontsize=10,
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"covariance_comparison_seed{seed}_{kernel_name}.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(
        f"seed={seed:3d} [{kernel_name:<9} l={lengthscale:.3f} n_train={n_train:>2}] "
        f"corr_frob={corr_frob:.4f}  R_test min_eig={eigvals.min():.4f}  -> {out_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot predicted vs. oracle correlation matrices")
    parser.add_argument("--copula-ckpt", default="./checkpoints/systematic-composition/step_0180000.pt")
    parser.add_argument("--tabicl-ckpt", default="tabicl-regressor-v2-20260212.ckpt")
    parser.add_argument("--kernels", default="rbf,matern32", help="Comma-separated kernel names to sample from")
    parser.add_argument("--seeds", default="0,1,2,3", help="Comma-separated function seeds to plot")
    parser.add_argument("--n-test", type=int, default=40)
    parser.add_argument("--n-train-min", type=int, default=5)
    parser.add_argument("--n-train-max", type=int, default=15)
    parser.add_argument("--out-dir", default="./results/figures")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    print(f"Loading TabICL marginal model: {args.tabicl_ckpt}")
    tabicl_model = load_tabicl_marginal(args.tabicl_ckpt, device)

    print(f"Loading copula model: {args.copula_ckpt}")
    copula_model, copula_cfg = load_copula_model(args.copula_ckpt, device=device)
    oracle_mode = OmegaConf.select(copula_cfg, "data.oracle_mode", default="prior")
    print(f"Copula checkpoint's oracle_mode: {oracle_mode}")

    os.makedirs(args.out_dir, exist_ok=True)
    for seed in seeds:
        plot_one(seed, tabicl_model, copula_model, oracle_mode, args, args.out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
