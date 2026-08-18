"""compare_marginal_backbones.py — z_train gap comparison across tabular
foundation models, on top of the SAME trained copula head.

Per synthetic (kernel, grid_size) task (constants.SYNTHETIC_SWEEP_PROFILES,
same profile spatial_correlation_eval.py's `sweep --mode synthetic` uses),
the true generating GP kernel is known exactly, so two things are
computable in closed form and used as a fixed reference point:
  1. the EXACT GP leave-one-out z_train (eval.spatial.diagnostics.
     _exact_gp_loo_z_train — Rasmussen & Williams Eq. 5.12), and
  2. the true spatial correlation matrix R_true.

For each marginal backend (eval.spatial.marginal_backends: tabicl, tabpfn,
exaone, tabm), this script:
  a) estimates z_train via that backend's own K-fold leave-fold-out PIT on
     the SAME context points, and scores the gap against the exact z_train
     (Pearson corr / RMSE / MAE / calibration mean+std);
  b) feeds that estimated z_train through the SAME frozen, trained copula
     head (--ckpt, default kernel-sweep-classic-prod) to get a predicted
     spatial correlation matrix, and scores it against R_true the same way
     eval/spatial/sweep_core.py::run_synthetic_config does (shape_corr,
     rmse, bias, model_r2) — i.e. how much a worse z_train estimate
     actually degrades the DOWNSTREAM spatial-correlation recovery task,
     not just the z_train numbers in isolation.

TabPFN requires a one-time license acceptance + `TABPFN_TOKEN` env var (see
eval.spatial.marginal_backends._require_tabpfn_token) — omit "tabpfn" from
--backends to skip it without that.

Usage:
    python eval/runners/compare_marginal_backbones.py \
        --ckpt kernel-sweep-classic-prod --backends tabicl,tabpfn,exaone,tabm
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.configs import constants  # noqa: E402
from eval.configs.checkpoints import resolve_checkpoint  # noqa: E402
from eval.spatial.diagnostics import (  # noqa: E402
    _exact_gp_loo_z_train,
    _forward_correlation,
    bin_correlation_by_distance,
    build_synthetic_grid_task,
    load_copula_model,
)
from eval.spatial.marginal_backends import BACKEND_NAMES, loo_pit, make_regressor  # noqa: E402
from eval.spatial.sweep_core import _weighted_corr, _weighted_r2, _weighted_rmse_bias  # noqa: E402
from inference.copula_inference import normalize_features  # noqa: E402

_RESULTS_DIR = os.path.join(_REPO_ROOT, "eval", "results")
_FIGURES_DIR = os.path.join(_REPO_ROOT, "eval", "reports", "figures")

# Coarser than run_benchmarks.py's 999-point DEFAULT_PROBS: this script pays
# one quantile-grid forward pass per K-fold per backend per task, so a
# 49-point grid keeps compute_pit's linear interpolation accurate to well
# under the noise floor of a 25-30-point PIT estimate without paying for
# needless quantile-query resolution.
PROBS = np.linspace(0.02, 0.98, 49)


def _z_train_gap(z_true: np.ndarray, z_hat: np.ndarray) -> dict:
    mask = np.isfinite(z_true) & np.isfinite(z_hat)
    zt, zh = z_true[mask], z_hat[mask]
    corr = (
        float(np.corrcoef(zt, zh)[0, 1]) if len(zt) > 2 and zt.std() > 0 and zh.std() > 0 else float("nan")
    )
    return {
        "z_corr": corr,
        "z_rmse": float(np.sqrt(np.mean((zt - zh) ** 2))) if len(zt) else float("nan"),
        "z_mae": float(np.mean(np.abs(zt - zh))) if len(zt) else float("nan"),
        "z_hat_mean": float(zh.mean()) if len(zh) else float("nan"),
        "z_hat_std": float(zh.std()) if len(zh) else float("nan"),
    }


def run_task(
    model, cfg, device, backends: list, regressors: dict,
    kernel_name: str, grid_size: int, n_context: int, seed: int,
) -> dict:
    task = build_synthetic_grid_task(cfg, kernel_name, grid_size, n_context, constants.N_BINS, seed, min_context=4)
    coords, D = task["coords"], task["D"]
    true_cov = task["true_cov"]
    dist, bin_edges, pair_counts = task["dist"], task["bin_edges"], task["pair_counts"]
    rho_true = bin_correlation_by_distance(task["R_true"], dist, bin_edges)

    context_idx, context_coords = task["context_idx"], task["context_coords"]
    K_ff_context = true_cov[np.ix_(context_idx, context_idx)]

    z_true = task["L"] @ task["rng"].standard_normal(D)
    context_values = z_true[context_idx]

    z_train_true = _exact_gp_loo_z_train(K_ff_context, context_values)
    x_train_norm, x_test_norm = normalize_features(context_coords, coords)

    R_pred_true = _forward_correlation(model, device, x_train_norm, z_train_true, x_test_norm)
    rho_pred_true = bin_correlation_by_distance(R_pred_true, dist, bin_edges)

    out = {
        "kernel": kernel_name, "grid_size": grid_size, "seed": seed,
        "ground_truth_spatial_model_r2": _weighted_r2(rho_pred_true, rho_true, pair_counts),
    }
    for name in backends:
        z_hat = loo_pit(
            name, regressors[name], x_train_norm, context_values, PROBS,
            k_folds=constants.PIT_K_FOLDS, seed=seed,
        )
        gap = _z_train_gap(z_train_true, z_hat)

        R_pred = _forward_correlation(model, device, x_train_norm, z_hat, x_test_norm)
        rho_pred = bin_correlation_by_distance(R_pred, dist, bin_edges)
        gap["spatial_model_r2"] = _weighted_r2(rho_pred, rho_true, pair_counts)
        gap["spatial_shape_corr"] = _weighted_corr(rho_pred, rho_true, pair_counts)
        rmse, bias = _weighted_rmse_bias(rho_pred, rho_true, pair_counts)
        gap["spatial_rmse"], gap["spatial_bias"] = rmse, bias
        out[name] = gap
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=str, default="kernel-sweep-classic-prod")
    parser.add_argument("--backends", type=str, default="tabicl,exaone,tabm",
                         help=f"Comma-separated subset of {BACKEND_NAMES}. 'tabpfn' needs TABPFN_TOKEN.")
    parser.add_argument("--profile", type=str, default="low_context_7config",
                         choices=list(constants.SYNTHETIC_SWEEP_PROFILES))
    parser.add_argument("--n-draws", type=int, default=5)
    parser.add_argument("--n-context", type=int, default=constants.N_CONTEXT)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=constants.SEED)
    parser.add_argument("--out", type=str, default=os.path.join(_RESULTS_DIR, "marginal_backbone_compare.json"))
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for b in backends:
        if b not in BACKEND_NAMES:
            raise ValueError(f"Unknown backend '{b}', choose from {BACKEND_NAMES}.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = resolve_checkpoint(args.ckpt)
    print(f"Loading checkpoint '{args.ckpt}' -> {ckpt} on {device} ...")
    model, cfg, resolved_device = load_copula_model(ckpt, device=device)

    print(f"Building regressors for backends: {backends}")
    regressors = {name: make_regressor(name, device=resolved_device) for name in backends}

    profile_configs = constants.SYNTHETIC_SWEEP_PROFILES[args.profile]
    results = []
    for config_name, kernel_name, grid_size in profile_configs:
        for draw in range(args.n_draws):
            seed = args.seed * 10_000 + hash((config_name, draw)) % 10_000
            r = run_task(
                model, cfg, resolved_device, backends, regressors,
                kernel_name, grid_size, args.n_context, seed,
            )
            r["config"] = config_name
            results.append(r)
            print(f"[{config_name} draw {draw}] gt_r2={r['ground_truth_spatial_model_r2']:.3f} "
                  + " ".join(f"{b}: z_corr={r[b]['z_corr']:.3f} r2={r[b]['spatial_model_r2']:.3f}" for b in backends))
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} task results to {args.out}")
    _print_summary(results, backends)
    _plot_summary(results, backends, os.path.join(_FIGURES_DIR, "marginal_backbone_compare.png"))


def _print_summary(results: list, backends: list) -> None:
    print("\n=== Aggregate summary (mean over all tasks/draws) ===")
    gt_r2 = np.mean([r["ground_truth_spatial_model_r2"] for r in results])
    print(f"Ground-truth-z_train spatial model_r2 (upper bound): {gt_r2:.3f}")
    header = f"{'backend':10s} {'z_corr':>8s} {'z_rmse':>8s} {'z_hat_std':>10s} {'spatial_r2':>11s} {'shape_corr':>11s}"
    print(header)
    for b in backends:
        z_corr = np.nanmean([r[b]["z_corr"] for r in results])
        z_rmse = np.nanmean([r[b]["z_rmse"] for r in results])
        z_std = np.nanmean([r[b]["z_hat_std"] for r in results])
        s_r2 = np.nanmean([r[b]["spatial_model_r2"] for r in results])
        s_corr = np.nanmean([r[b]["spatial_shape_corr"] for r in results])
        print(f"{b:10s} {z_corr:8.3f} {z_rmse:8.3f} {z_std:10.3f} {s_r2:11.3f} {s_corr:11.3f}")


def _plot_summary(results: list, backends: list, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    metrics = [("z_corr", "z_train corr vs. ground truth"), ("spatial_model_r2", "downstream spatial model_r2")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(backends)))
    for ax, (key, title) in zip(axes, metrics):
        means = [np.nanmean([r[b][key] for r in results]) for b in backends]
        stds = [np.nanstd([r[b][key] for r in results]) for b in backends]
        ax.bar(backends, means, yerr=stds, color=colors, capsize=4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(key)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
