"""
experiment_b_quantitative.py — Experiment B: quantitative joint comparison.

Across many synthetic 1D GP draws (varied kernel, lengthscale, train-set size
/ placement), compares "your model" (TabICL marginals + copula R_test)
against PFN4BO (own marginals, R=I — the independence assumption baked into
PFN4BO's use in this comparison) on:

  1. Joint NLPD — see the "NLPD proxy" note below for exactly what's computed.
  2. Correlation recovery — ||R_test - R_true||_F / N, where R_true follows
     whatever ``oracle_mode`` (prior vs. posterior) the loaded copula
     checkpoint was actually trained under (read from its own saved cfg).
  3. Locality-vs-distance aggregate — binned predicted correlation vs. true
     kernel correlation, by |x_i - x_j|.

All non-trivial inference logic (marginal quantiles, PIT, correlation query)
is imported from inference/copula_inference.py.

NLPD proxy note: a true joint density under quantile-grid (non-Gaussian)
marginals has no closed form. Both models' marginal PIT (Z-values and
log-densities) are computed by interpolating their OWN returned
``(quantile_grid, probs)`` pair (linear interpolation for Z, local
finite-difference slope of the quantile function for log-density — the same
"f(z) = 1/Q'(F(z))" identity TabICL's own ``QuantileDistribution.log_prob``
uses, just via finite differences on the grid instead of the exact
spline/tail machinery) — this keeps the two models' marginal-density
treatment symmetric/fair rather than giving one an exact method and the other
an approximation. The joint density is then the Gaussian-copula-in-Z-space
proxy (``loss.y_space_nll``): copula term + marginal term, exactly 0 copula
term for the PFN4BO (R=I) baseline.

Usage:
    python experiments/experiment_b_quantitative.py \\
        [--copula-ckpt ./checkpoints/systematic-composition/step_0180000.pt] \\
        [--tabicl-ckpt tabicl-regressor-v2-20260212.ckpt] \\
        [--n-functions 60] [--n-test 40] [--n-train-min 5] [--n-train-max 15] \\
        [--out-csv ./results/quantitative_comparison.csv] \\
        [--out-dir ./results/figures] [--device auto] [--seed 0]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import norm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gen import gp_posterior, sigma_to_correlation  # noqa: E402
from loss import y_space_nll  # noqa: E402
from pit import run_pit  # noqa: E402

from experiments._synthetic import OBS_NOISE_STD, pick_train_indices, sample_gp_function  # noqa: E402
from inference.copula_inference import (  # noqa: E402
    get_marginal_quantiles_pfn4bo,
    get_test_correlation,
    load_copula_model,
    load_pfn4bo,
    load_tabicl_marginal,
    normalize_features,
)

KERNELS = ["rbf", "matern32"]
LENGTHSCALE_LOG_RANGE = (np.log(0.05), np.log(1.2))


def _quantile_grid_pit(
    quantile_grid: np.ndarray, probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """PIT (Z-values) and log-density for true targets from an ALREADY-BUILT
    (quantile_grid, probs) pair (whichever backend produced it), via linear
    interpolation for u = F(y) and a local finite-difference slope of the
    quantile function for the density (f(z) = 1/Q'(F(z))). Applied
    identically to TabICL's and PFN4BO's own quantile grids for a fair
    marginal-density comparison (see module docstring)."""
    n = quantile_grid.shape[0]
    u = np.empty(n)
    log_pdf = np.empty(n)
    for i in range(n):
        u[i] = np.interp(y_true[i], quantile_grid[i], probs)
        j = int(np.clip(np.searchsorted(probs, u[i]), 1, len(probs) - 1))
        dQ = quantile_grid[i, j] - quantile_grid[i, j - 1]
        dP = probs[j] - probs[j - 1]
        slope = dQ / max(dP, eps)
        log_pdf[i] = -np.log(max(slope, eps))
    u_clamped = np.clip(u, eps, 1 - eps)
    z = norm.ppf(u_clamped)
    return z, log_pdf


def _sample_one_function(
    rng_np: np.random.Generator, rng_torch: torch.Generator, n_test: int, n_train_range: tuple, kernels: list[str]
):
    kernel_name = rng_np.choice(kernels)
    lengthscale = float(np.exp(rng_np.uniform(*LENGTHSCALE_LOG_RANGE)))
    n_train = int(rng_np.integers(n_train_range[0], n_train_range[1] + 1))

    X_dense, true_f, kernel_fn = sample_gp_function(kernel_name, lengthscale, n_test, rng_torch)
    train_idx = pick_train_indices(n_test, n_train, rng_np)

    X_train_t = X_dense[train_idx]
    y_train_t = true_f[train_idx] + OBS_NOISE_STD * torch.randn(len(train_idx), generator=rng_torch)
    X_test_t = X_dense
    y_test_t = true_f
    return kernel_name, lengthscale, n_train, X_train_t, y_train_t, X_test_t, y_test_t, kernel_fn


def _compute_r_true(oracle_mode: str, X_train_t, y_train_t, X_test_t, kernel_fn) -> np.ndarray:
    if oracle_mode == "posterior":
        _, Sigma_star = gp_posterior(X_train_t, y_train_t, X_test_t, kernel_fn, noise=OBS_NOISE_STD**2, latent=False)
    else:  # "prior": raw kernel structure among test points, ignoring train conditioning
        Sigma_star = kernel_fn(X_test_t, X_test_t)
    R_true, _ = sigma_to_correlation(Sigma_star)
    return R_true.numpy()


@torch.no_grad()
def run_one_function(
    seed: int, tabicl_model, copula_model, pfn4bo_model, oracle_mode: str, args
) -> tuple[dict, tuple]:
    rng_np = np.random.default_rng(seed)
    rng_torch = torch.Generator().manual_seed(seed)

    kernel_name, lengthscale, n_train, X_train_t, y_train_t, X_test_t, y_test_t, kernel_fn = _sample_one_function(
        rng_np, rng_torch, args.n_test, (args.n_train_min, args.n_train_max), args.kernels
    )
    X_train, y_train = X_train_t.numpy(), y_train_t.numpy()
    X_test, y_test = X_test_t.numpy(), y_test_t.numpy()
    n_test = X_test.shape[0]

    # R_true and the locality distances below use the RAW [0, 1] grid — the
    # true kernel correlation depends on real distance in that domain, not
    # the standardized one below. Only the features fed to the models get
    # normalized (see normalize_features's docstring: data_gen.py z-scores
    # x jointly over train+test before ever calling TabICL/CopulaTabICL;
    # neither model normalizes internally).
    R_true = _compute_r_true(oracle_mode, X_train_t, y_train_t, X_test_t, kernel_fn)

    X_train_norm, X_test_norm = normalize_features(X_train, X_test)
    X_train_norm_t = torch.as_tensor(X_train_norm, dtype=X_train_t.dtype)
    X_test_norm_t = torch.as_tensor(X_test_norm, dtype=X_test_t.dtype)

    # Use TabICL's own exact PIT (pit.run_pit — QuantileDistribution.cdf/log_prob,
    # not the coarser quantile-grid-interpolation approximation) for "ours",
    # since this is the exact convention Z_train/Z_test were computed with when
    # this copula checkpoint was trained — anything cruder would make R_test's
    # evaluation unfair to the model. PFN4BO has no such training-time coupling,
    # so its simpler self-consistent approximation (_quantile_grid_pit) is fine.
    tabicl_device = next(tabicl_model.parameters()).device
    pit_out = run_pit(
        tabicl_model,
        X_train_norm_t.to(tabicl_device), y_train_t.to(tabicl_device).unsqueeze(-1),
        X_test_norm_t.to(tabicl_device), y_test_t.to(tabicl_device).unsqueeze(-1),
        k_folds=min(10, len(X_train)),
    )
    Z_train = pit_out["z_train"].squeeze(-1).cpu().numpy()
    z_ours = pit_out["z_test"].squeeze(-1).cpu().numpy()
    log_pdf_ours = pit_out["log_pdf_test"].squeeze(-1).cpu().numpy()

    R_test = get_test_correlation(copula_model, X_train_norm, Z_train, X_test_norm)

    mask = torch.ones(1, n_test, dtype=torch.bool)
    nlpd_ours = y_space_nll(
        torch.as_tensor(R_test, dtype=torch.float32).unsqueeze(0),
        torch.as_tensor(z_ours, dtype=torch.float32).unsqueeze(0),
        torch.as_tensor(log_pdf_ours, dtype=torch.float32).unsqueeze(0),
        mask,
    )

    corr_frob = float(np.linalg.norm(R_test - R_true, "fro") / n_test)

    dists = np.abs(X_test[:, 0][:, None] - X_test[:, 0][None, :])
    iu = np.triu_indices(n_test, k=1)
    locality_pairs = (dists[iu], R_test[iu], R_true[iu])

    mid = n_test // 2
    result = {
        "function_idx": seed,
        "kernel": kernel_name,
        "lengthscale": lengthscale,
        "n_train": n_train,
        "oracle_mode": oracle_mode,
        "corr_frob_norm": corr_frob,
        "nlpd_ours_total": float(nlpd_ours["total"]),
        "nlpd_ours_copula": float(nlpd_ours["copula"]),
        "nlpd_ours_marginal": float(nlpd_ours["marginal"]),
        "adjacent_R_test": float(R_test[mid, mid + 1]),
        "adjacent_R_true": float(R_true[mid, mid + 1]),
        "nlpd_pfn4bo_total": float("nan"),
        "nlpd_pfn4bo_marginal": float("nan"),
    }

    if pfn4bo_model is not None:
        try:
            quantile_grid_pfn, probs_pfn = get_marginal_quantiles_pfn4bo(
                pfn4bo_model, X_train_norm, y_train, X_test_norm
            )
            z_pfn, log_pdf_pfn = _quantile_grid_pit(quantile_grid_pfn, probs_pfn, y_test)
            nlpd_pfn = y_space_nll(
                torch.eye(n_test, dtype=torch.float32).unsqueeze(0),
                torch.as_tensor(z_pfn, dtype=torch.float32).unsqueeze(0),
                torch.as_tensor(log_pdf_pfn, dtype=torch.float32).unsqueeze(0),
                mask,
            )
            result["nlpd_pfn4bo_total"] = float(nlpd_pfn["total"])
            result["nlpd_pfn4bo_marginal"] = float(nlpd_pfn["marginal"])
        except Exception as exc:  # noqa: BLE001
            print(f"  [fn {seed}] PFN4BO failed, recording NaN: {exc}")

    return result, locality_pairs


def _print_summary(results: list[dict]) -> None:
    keys = [
        "corr_frob_norm", "nlpd_ours_total", "nlpd_ours_copula", "nlpd_ours_marginal",
        "nlpd_pfn4bo_total", "nlpd_pfn4bo_marginal",
    ]
    print(f"\n{'-'*70}")
    print(f"Summary over {len(results)} synthetic functions (mean +/- std)")
    print(f"{'-'*70}")
    for k in keys:
        vals = np.array([r[k] for r in results], dtype=float)
        print(f"  {k:<22} {np.nanmean(vals):>10.4f} +/- {np.nanstd(vals):>8.4f}  (n_valid={np.isfinite(vals).sum()})")
    print(f"{'-'*70}\n")


def _plot_locality_aggregate(all_dists, all_r_test, all_r_true, out_path: str, n_bins: int = 15) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.linspace(0.0, all_dists.max() + 1e-9, n_bins + 1)
    bin_idx = np.clip(np.digitize(all_dists, bins) - 1, 0, n_bins - 1)

    centers, mean_test, mean_true = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        centers.append(0.5 * (bins[b] + bins[b + 1]))
        mean_test.append(all_r_test[mask].mean())
        mean_true.append(all_r_true[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(all_dists, all_r_test, s=2, alpha=0.05, color="tab:blue", label="R_test (raw pairs)")
    ax.plot(centers, mean_test, "o-", color="tab:blue", label="R_test (binned mean)")
    ax.plot(centers, mean_true, "s--", color="black", label="true kernel correlation (binned mean)")
    ax.set_xlabel("|x_i - x_j|")
    ax.set_ylabel("correlation")
    ax.set_title("Locality: predicted vs. true correlation-vs-distance")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved locality aggregate plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment B — quantitative joint comparison")
    parser.add_argument("--copula-ckpt", default="./checkpoints/systematic-composition/step_0180000.pt")
    parser.add_argument("--tabicl-ckpt", default="tabicl-regressor-v2-20260212.ckpt")
    parser.add_argument("--pfn4bo-model", default="hebo_plus_model")
    parser.add_argument("--kernels", default="rbf,matern32", help="Comma-separated kernel names to sample from")
    parser.add_argument("--n-functions", type=int, default=60)
    parser.add_argument("--n-test", type=int, default=40)
    parser.add_argument("--n-train-min", type=int, default=5)
    parser.add_argument("--n-train-max", type=int, default=15)
    parser.add_argument("--out-csv", default="./results/quantitative_comparison.csv")
    parser.add_argument("--out-dir", default="./results/figures")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    assert all(k in KERNELS for k in args.kernels), f"Unknown kernel(s) in {args.kernels}, must be subset of {KERNELS}"

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    print(f"Loading TabICL marginal model: {args.tabicl_ckpt}")
    tabicl_model = load_tabicl_marginal(args.tabicl_ckpt, device)

    print(f"Loading copula model: {args.copula_ckpt}")
    copula_model, copula_cfg = load_copula_model(args.copula_ckpt, device=device)
    oracle_mode = OmegaConf.select(copula_cfg, "data.oracle_mode", default="prior")
    print(f"Copula checkpoint's oracle_mode: {oracle_mode} (determines the R_true convention used below)")

    print(f"Loading PFN4BO model: {args.pfn4bo_model}")
    try:
        pfn4bo_model = load_pfn4bo(args.pfn4bo_model, device=device)
    except Exception as exc:  # noqa: BLE001
        print(f"PFN4BO failed to load ({exc}); its NLPD column will be all-NaN.")
        pfn4bo_model = None

    results = []
    all_dists, all_r_test, all_r_true = [], [], []

    for i in range(args.n_functions):
        seed = args.seed + i
        result, (dists, r_test, r_true) = run_one_function(
            seed, tabicl_model, copula_model, pfn4bo_model, oracle_mode, args
        )
        results.append(result)
        all_dists.append(dists)
        all_r_test.append(r_test)
        all_r_true.append(r_true)
        print(
            f"  fn {i:03d} [{result['kernel']:<9} l={result['lengthscale']:.3f} n_train={result['n_train']:>2}] "
            f"corr_frob={result['corr_frob_norm']:.4f}  "
            f"nlpd_ours={result['nlpd_ours_total']:.4f}  nlpd_pfn4bo={result['nlpd_pfn4bo_total']:.4f}"
        )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {args.out_csv}")

    _print_summary(results)

    os.makedirs(args.out_dir, exist_ok=True)
    _plot_locality_aggregate(
        np.concatenate(all_dists), np.concatenate(all_r_test), np.concatenate(all_r_true),
        os.path.join(args.out_dir, "experiment_b_locality_aggregate.png"),
    )

    print("Done.")


if __name__ == "__main__":
    main()
