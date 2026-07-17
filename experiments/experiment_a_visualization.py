"""
experiment_a_visualization.py — Experiment A: qualitative 1D sample plots.

For a handful of 1D synthetic GP test functions (RBF at a few lengthscales,
one Matern-3/2), draws sparse training points, queries TabICL's marginal
quantiles at a dense test grid, runs loo_pit + the copula model to get
R_test, and samples trajectories under three/four conditions:

  1. "your model"        — TabICL quantiles + copula R_test
  2. "PFN4BO (R=I)"       — same TabICL quantiles, independence assumption
  3. "PFN4BO (own marg.)" — PFN4BO's own marginal quantiles, R=I
  4. "reference GP"       — exact GP posterior (known kernel), if available

All non-trivial inference logic (PIT, correlation query, sampling) is
imported from inference/copula_inference.py — this script is thin CLI +
plotting only.

Usage:
    python experiments/experiment_a_visualization.py \\
        [--copula-ckpt ./checkpoints/systematic-composition/step_0180000.pt] \\
        [--tabicl-ckpt tabicl-regressor-v2-20260212.ckpt] \\
        [--out-dir ./results/figures] [--device auto] [--seed 0] \\
        [--n-samples 8] [--n-train 7] [--n-test 60]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gen import _safe_cholesky, gp_posterior  # noqa: E402

from experiments._synthetic import OBS_NOISE_STD, pick_train_indices, sample_gp_function  # noqa: E402
from inference.copula_inference import (  # noqa: E402
    get_marginal_quantiles,
    get_marginal_quantiles_pfn4bo,
    get_test_correlation,
    load_copula_model,
    load_pfn4bo,
    load_tabicl_marginal,
    loo_pit,
    normalize_features,
    sample_trajectories,
)

# ---------------------------------------------------------------------------
# Synthetic test functions
# ---------------------------------------------------------------------------

TEST_FUNCTIONS = [
    ("rbf", 0.5, "RBF (l=0.5)"),
    ("rbf", 0.15, "RBF (l=0.15)"),
    ("rbf", 1.0, "RBF (l=1.0)"),
    ("matern32", 0.3, "Matern-3/2 (l=0.3)"),
]

def _locality_check(R_test: np.ndarray, X_test: np.ndarray, label: str) -> None:
    n = R_test.shape[0]
    i = n // 2
    r_adjacent = R_test[i, i + 1]
    dx = abs(X_test[i + 1, 0] - X_test[i, 0])
    flag = "" if r_adjacent > 0.7 else "  <-- FLAG: low adjacent correlation, local smoothness not learned?"
    print(f"  [{label}] R_test[{i},{i+1}] (dx={dx:.4f}) = {r_adjacent:.4f}{flag}")


@torch.no_grad()
def run_one_function(
    kernel_name: str,
    lengthscale: float,
    label: str,
    tabicl_model,
    copula_model,
    pfn4bo_model,
    args,
    seed: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch_gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)

    X_dense, true_f, kernel_fn = sample_gp_function(kernel_name, lengthscale, args.n_test, torch_gen)
    train_idx = pick_train_indices(args.n_test, args.n_train, rng)

    X_train_t = X_dense[train_idx]
    y_train_t = true_f[train_idx] + OBS_NOISE_STD * torch.randn(len(train_idx), generator=torch_gen)
    X_test_t = X_dense

    X_train = X_train_t.numpy()
    y_train = y_train_t.numpy()
    X_test = X_test_t.numpy()
    true_f_np = true_f.numpy()

    # Models expect zero-mean/unit-std features (data_gen.py's x_norm
    # convention, computed jointly over train+test) — the raw [0, 1] grid
    # used for the reference GP / plotting below is NOT on that scale.
    X_train_norm, X_test_norm = normalize_features(X_train, X_test)

    # --- Reference exact GP posterior (known kernel) ---
    mu_star, Sigma_star = gp_posterior(
        X_train_t, y_train_t, X_test_t, kernel_fn, noise=OBS_NOISE_STD**2, latent=False
    )
    mu_star_np = mu_star.numpy()
    sigma_star_np = Sigma_star.diagonal().clamp(min=1e-12).sqrt().numpy()
    L_star = _safe_cholesky(Sigma_star)
    gp_samples = (
        mu_star.unsqueeze(0) + torch.randn(args.n_samples, len(X_test), generator=torch_gen) @ L_star.T
    ).numpy()

    # --- TabICL marginals + copula correlation ("your model") ---
    quantile_grid, probs = get_marginal_quantiles(tabicl_model, X_train_norm, y_train, X_test_norm)
    Z_train = loo_pit(tabicl_model, X_train_norm, y_train, k_folds=min(10, len(X_train)))
    R_test = get_test_correlation(copula_model, X_train_norm, Z_train, X_test_norm)
    _locality_check(R_test, X_test, label)

    samples_ours, n_clipped_ours = sample_trajectories(
        quantile_grid, probs, R_test, args.n_samples, rng=np.random.default_rng(seed + 1)
    )

    # --- PFN4BO (R=I), reusing the SAME TabICL quantiles ---
    R_I = np.eye(len(X_test))
    samples_pfn_RI, n_clipped_pfn_RI = sample_trajectories(
        quantile_grid, probs, R_I, args.n_samples, rng=np.random.default_rng(seed + 2)
    )

    panels = [
        ("Reference GP posterior (known kernel)", mu_star_np, sigma_star_np, gp_samples),
        ("Your model (copula R_test)", quantile_grid.mean(axis=1), quantile_grid.std(axis=1), samples_ours),
        ("PFN4BO (R=I, TabICL marginals)", quantile_grid.mean(axis=1), quantile_grid.std(axis=1), samples_pfn_RI),
    ]

    # --- PFN4BO's own marginals + R=I ---
    try:
        quantile_grid_pfn, probs_pfn = get_marginal_quantiles_pfn4bo(
            pfn4bo_model, X_train_norm, y_train, X_test_norm
        )
        samples_pfn_own, n_clipped_pfn_own = sample_trajectories(
            quantile_grid_pfn, probs_pfn, R_I, args.n_samples, rng=np.random.default_rng(seed + 3)
        )
        panels.append(
            (
                "PFN4BO (R=I, own marginals)",
                quantile_grid_pfn.mean(axis=1),
                quantile_grid_pfn.std(axis=1),
                samples_pfn_own,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [{label}] PFN4BO own-marginals panel failed, skipping: {exc}")

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    x_flat = X_test[:, 0]
    for ax, (title, mean, std, samples) in zip(axes, panels):
        for s in range(samples.shape[0]):
            ax.plot(x_flat, samples[s], color="tab:blue", alpha=0.25, lw=0.8)
        ax.plot(x_flat, mean, color="tab:orange", lw=1.5, label="mean")
        ax.fill_between(x_flat, mean - 2 * std, mean + 2 * std, color="tab:orange", alpha=0.15, label="±2σ")
        ax.plot(x_flat, true_f_np, "k--", lw=1.2, label="true f")
        ax.scatter(X_train[:, 0], y_train, color="black", zorder=5, s=25, label="train pts")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"Experiment A — {label}", fontsize=12)
    plt.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    fname = f"experiment_a_{kernel_name}_l{lengthscale}.png".replace(".", "p", 1)
    out_path = os.path.join(args.out_dir, fname)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment A — qualitative 1D sample visualization")
    parser.add_argument("--copula-ckpt", default="./checkpoints/systematic-composition/step_0180000.pt")
    parser.add_argument("--tabicl-ckpt", default="tabicl-regressor-v2-20260212.ckpt")
    parser.add_argument("--pfn4bo-model", default="hebo_plus_model")
    parser.add_argument("--out-dir", default="./results/figures")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=7)
    parser.add_argument("--n-test", type=int, default=60)
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    print(f"Loading TabICL marginal model: {args.tabicl_ckpt}")
    tabicl_model = load_tabicl_marginal(args.tabicl_ckpt, device)

    print(f"Loading copula model: {args.copula_ckpt}")
    copula_model, copula_cfg = load_copula_model(args.copula_ckpt, device=device)

    print(f"Loading PFN4BO model: {args.pfn4bo_model}")
    try:
        pfn4bo_model = load_pfn4bo(args.pfn4bo_model, device=device)
    except Exception as exc:  # noqa: BLE001
        print(f"PFN4BO failed to load ({exc}); the 'own marginals' panel will be skipped for all functions.")
        pfn4bo_model = None

    for i, (kernel_name, lengthscale, label) in enumerate(TEST_FUNCTIONS):
        print(f"\n=== {label} ===")
        run_one_function(
            kernel_name, lengthscale, label,
            tabicl_model, copula_model, pfn4bo_model, args,
            seed=args.seed + i,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
