"""s7_backbone.py — z_train gap comparison across tabular foundation models,
on top of the SAME trained copula head. Debug pipeline stage S7(a); see
debug/README.md. Moved from eval/runners/compare_marginal_backbones.py
(2026-08-26) — nothing outside this file imports it, only doc comments
referenced its old path (eval/spatial/marginal_backends.py,
eval/spatial/diagnostics.py, eval/spatial/sweep_core.py,
eval/configs/constants.py, all updated to point here).

Per synthetic (kernel, grid_size) task (constants.SYNTHETIC_SWEEP_PROFILES,
same profile spatial_correlation_eval.py's `sweep --mode synthetic` uses),
the true generating GP kernel is known exactly, so two things are
computable in closed form and used as a fixed reference point:
  1. the EXACT GP leave-one-out z_train (eval.spatial.diagnostics.
     _exact_gp_loo_z_train — Rasmussen & Williams Eq. 5.12), and
  2. the true spatial correlation matrix R_true.

For each marginal backend (eval.spatial.marginal_backends: tabicl, tabpfn,
exaone, tabfm, tabm), this script:
  a) estimates z_train via that backend's own K-fold leave-fold-out PIT on
     the SAME context points, and scores the gap against the exact z_train
     (Pearson corr / RMSE / MAE / calibration mean+std);
  b) feeds that estimated z_train through the SAME frozen, trained copula
     head (--ckpt, default kernel-sweep-classic-prod) to get a predicted
     spatial correlation matrix, and scores it against R_true the same way
     eval/spatial/sweep_core.py::run_synthetic_config does (shape_corr,
     rmse, bias, model_r2) — i.e. how much a worse z_train estimate
     actually degrades the DOWNSTREAM spatial-correlation recovery task,
     not just the z_train numbers in isolation;
  c) additionally scores a genuine total (marginal+copula) Y-space NLL, in
     nats/point, on a small held-out point set that was never in context —
     this backend's own one-shot (non-K-fold) quantile grid there supplies
     the marginal, this task's predicted R supplies the copula, combined
     via eval/metrics/joint_nll.py::compute_joint_nll (Sklar decomposition).
     spatial_model_r2 (b) is a curve-shape diagnostic on BINNED, distance-
     averaged correlations — it never sees marginal calibration and isn't a
     proper scoring rule, so it can't answer "how many nats worse is the
     real predictive density with this backend"; nll_total here does.

TabPFN requires a one-time license acceptance + `TABPFN_TOKEN` env var (see
eval.spatial.marginal_backends._require_tabpfn_token) — omit "tabpfn" from
--backends to skip it without that.

Usage:
    python debug/stages/s7_backbone.py \
        --ckpt kernel-sweep-classic-prod --backends tabicl,tabpfn,exaone,tabfm,tabm
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
from eval.metrics.joint_nll import compute_joint_nll  # noqa: E402
from eval.spatial.diagnostics import (  # noqa: E402
    _exact_gp_loo_z_train,
    _forward_correlation,
    bin_correlation_by_distance,
    build_synthetic_grid_task,
    load_copula_model,
)
from eval.spatial.marginal_backends import BACKEND_NAMES, loo_pit, make_regressor, quantiles  # noqa: E402
from eval.spatial.sweep_core import weighted_corr, weighted_r2, weighted_rmse_bias  # noqa: E402
from inference.copula_inference import normalize_features  # noqa: E402

_RESULTS_DIR = os.path.join(_REPO_ROOT, "eval", "results")
_FIGURES_DIR = os.path.join(_REPO_ROOT, "eval", "reports", "figures")

# Coarser than run_benchmarks.py's 999-point DEFAULT_PROBS: this script pays
# one quantile-grid forward pass per K-fold per backend per task, so a
# 49-point grid keeps compute_pit's linear interpolation accurate to well
# under the noise floor of a 25-30-point PIT estimate without paying for
# needless quantile-query resolution.
PROBS = np.linspace(0.02, 0.98, 49)

# Size of the held-out (never-in-context) point set the joint-NLL score (c)
# is computed on — see eval/configs/constants.py::N_NLL_TEST (shared with
# sweep_core.py::run_real_config's real-ERA5 analogue).
N_NLL_TEST = constants.N_NLL_TEST


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
    n_nll_test: int = N_NLL_TEST,
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

    # Held-out points (never in context) for the joint-NLL score below — see
    # N_NLL_TEST. Drawn from the SAME seeded rng as context_idx, so the task
    # stays fully deterministic per `seed`.
    remaining_idx = np.setdiff1d(np.arange(D), context_idx)
    nll_test_idx = task["rng"].choice(remaining_idx, size=min(n_nll_test, len(remaining_idx)), replace=False)
    y_nll_test = z_true[nll_test_idx]

    z_train_true = _exact_gp_loo_z_train(K_ff_context, context_values)
    x_train_norm, x_test_norm = normalize_features(context_coords, coords)
    x_nll_test_norm = x_test_norm[nll_test_idx]

    R_pred_true = _forward_correlation(model, device, x_train_norm, z_train_true, x_test_norm)
    rho_pred_true = bin_correlation_by_distance(R_pred_true, dist, bin_edges)

    # Ground-truth-marginal reference NLL (upper bound), the NLL analogue of
    # ground_truth_spatial_model_r2 below: z_true's per-point marginal is
    # exactly N(0, true_cov_ii) by construction (z_true = L @ N(0,I)), no
    # fitting needed, so this isolates how much of any backend's total-NLL
    # gap is really a copula (R_pred_true vs. R_pred) effect vs. a marginal
    # (fitted quantile grid) effect.
    from scipy.stats import norm as _norm
    exact_std = np.sqrt(np.clip(np.diag(true_cov)[nll_test_idx], 1e-12, None))
    qgrid_exact = exact_std[:, None] * _norm.ppf(PROBS)[None, :]
    R_nll_true = R_pred_true[np.ix_(nll_test_idx, nll_test_idx)]
    nll_true = compute_joint_nll(qgrid_exact, PROBS, R_nll_true, y_nll_test)

    out = {
        "kernel": kernel_name, "grid_size": grid_size, "seed": seed,
        "ground_truth_spatial_model_r2": weighted_r2(rho_pred_true, rho_true, pair_counts),
        "ground_truth_nll_total": nll_true["total"],
        "ground_truth_nll_marginal": nll_true["marginal"],
        "ground_truth_nll_copula": nll_true["copula"],
    }
    for name in backends:
        z_hat = loo_pit(
            name, regressors[name], x_train_norm, context_values, PROBS,
            k_folds=constants.PIT_K_FOLDS, seed=seed,
        )
        gap = _z_train_gap(z_train_true, z_hat)

        R_pred = _forward_correlation(model, device, x_train_norm, z_hat, x_test_norm)
        rho_pred = bin_correlation_by_distance(R_pred, dist, bin_edges)
        gap["spatial_model_r2"] = weighted_r2(rho_pred, rho_true, pair_counts)
        gap["spatial_shape_corr"] = weighted_corr(rho_pred, rho_true, pair_counts)
        rmse, bias = weighted_rmse_bias(rho_pred, rho_true, pair_counts)
        gap["spatial_rmse"], gap["spatial_bias"] = rmse, bias

        # Total (marginal+copula) Y-space NLL on the held-out points: this
        # backend's own one-shot (non-K-fold — these points were never in
        # context) quantile grid supplies the marginal, R_pred restricted to
        # the same points supplies the copula. See module docstring (c).
        try:
            qgrid = quantiles(name, regressors[name], x_train_norm, context_values, x_nll_test_norm, PROBS, seed=seed)
            R_nll = R_pred[np.ix_(nll_test_idx, nll_test_idx)]
            nll = compute_joint_nll(qgrid, PROBS, R_nll, y_nll_test)
            gap["nll_total"], gap["nll_marginal"], gap["nll_copula"] = nll["total"], nll["marginal"], nll["copula"]
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name}] joint-NLL scoring failed: {exc}")
            gap["nll_total"] = gap["nll_marginal"] = gap["nll_copula"] = float("nan")

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
                  f"gt_nll={r['ground_truth_nll_total']:.3f} "
                  + " ".join(f"{b}: z_corr={r[b]['z_corr']:.3f} r2={r[b]['spatial_model_r2']:.3f} "
                             f"nll={r[b]['nll_total']:.3f}" for b in backends))
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} task results to {args.out}")
    _print_summary(results, backends)
    _plot_summary(results, backends, os.path.join(_FIGURES_DIR, "marginal_backbone_compare.png"))


def _print_summary(results: list, backends: list) -> None:
    print("\n=== Aggregate summary (mean over all tasks/draws) ===")
    gt_r2 = np.mean([r["ground_truth_spatial_model_r2"] for r in results])
    gt_nll = np.nanmean([r["ground_truth_nll_total"] for r in results])
    print(f"Ground-truth-z_train spatial model_r2 (upper bound): {gt_r2:.3f}")
    print(f"Ground-truth-marginal total NLL (upper bound, nats/point): {gt_nll:.3f}")
    header = (f"{'backend':10s} {'z_corr':>8s} {'z_rmse':>8s} {'z_hat_std':>10s} {'spatial_r2':>11s} "
              f"{'shape_corr':>11s} {'nll_total':>10s} {'nll_marg':>10s} {'nll_cop':>10s}")
    print(header)
    for b in backends:
        z_corr = np.nanmean([r[b]["z_corr"] for r in results])
        z_rmse = np.nanmean([r[b]["z_rmse"] for r in results])
        z_std = np.nanmean([r[b]["z_hat_std"] for r in results])
        s_r2 = np.nanmean([r[b]["spatial_model_r2"] for r in results])
        s_corr = np.nanmean([r[b]["spatial_shape_corr"] for r in results])
        n_tot = np.nanmean([r[b]["nll_total"] for r in results])
        n_marg = np.nanmean([r[b]["nll_marginal"] for r in results])
        n_cop = np.nanmean([r[b]["nll_copula"] for r in results])
        print(f"{b:10s} {z_corr:8.3f} {z_rmse:8.3f} {z_std:10.3f} {s_r2:11.3f} {s_corr:11.3f} "
              f"{n_tot:10.3f} {n_marg:10.3f} {n_cop:10.3f}")


def _plot_summary(results: list, backends: list, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    metrics = [
        ("z_corr", "z_train corr vs. ground truth"),
        ("spatial_model_r2", "downstream spatial model_r2"),
        ("nll_total", "total NLL (marginal+copula, nats/pt)"),
    ]
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
