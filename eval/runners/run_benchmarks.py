"""run_benchmarks.py — CLI entry point for the TabICLv2 + Copula inter-instance
joint-distribution validation suite.

For each of three benchmarks (spatial interpolation, sensor in-painting,
synthetic BBO surrogates), compares three methods that all produce a
(quantile_grid, probs, R) predictive representation:

  independent  — TabICL marginals, R = I
  copula       — TabICL marginals, R = CopulaTabICL's predicted correlation
  standard_gp  — a fully independent end-to-end GaussianProcessRegressor fit
                 in raw y-space (own mean/covariance, not tied to TabICL)

All non-trivial inference logic (marginal quantiles, PIT, correlation query,
trajectory sampling) is imported from inference/copula_inference.py; the
joint-NLL/energy-score math is imported from src/loss.py via eval/metrics/.
This script is orchestration only.

Usage:
    python eval/runners/run_benchmarks.py \\
        --copula_ckpt ./checkpoints/systematic-composition-k5/step_0045000.pt \\
        --tabicl_ckpt tabicl-regressor-v2-20260212.ckpt \\
        --device auto --num_episodes 50 --n_samples 1000
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference.copula_inference import (  # noqa: E402
    get_test_correlation,
    load_copula_model,
    normalize_features,
    sample_trajectories,
)

from eval.baselines import independent, standard_gp  # noqa: E402
from eval.datasets import sensor_imputation, spatial_housing, synthetic_bbo  # noqa: E402
from eval.metrics.energy_score import compute_energy_score  # noqa: E402
from eval.metrics.joint_nll import compute_joint_nll, compute_pit  # noqa: E402
from eval.tabicl_utils import make_tabicl_regressor, tabicl_loo_pit, tabicl_quantiles  # noqa: E402
from eval.viz.correlation_plots import (  # noqa: E402
    collect_pair_distances_and_values,
    plot_correlation_heatmaps,
    plot_correlation_vs_distance,
)
from eval.io import gp_to_quantile_and_R, print_markdown_summary, save_results_json  # noqa: E402

BENCHMARK_NAMES = ["spatial_housing", "sensor_imputation", "synthetic_bbo"]
# TabICL's own native quantile-grid density (p_j = j/1001), used as the
# probability levels queried from TabICLRegressor.
DEFAULT_PROBS = np.arange(1, 1000) / 1001


def _load_episode(benchmark_name: str, seed: int, synthetic_bbo_cfg=None):
    """Returns (X_train, y_train, X_test, y_test, R_true_or_None)."""
    if benchmark_name == "spatial_housing":
        X_train, y_train, X_test, y_test = spatial_housing.load_split(n_ctx=64, n_test=32, seed=seed)
        return X_train, y_train, X_test, y_test, None
    if benchmark_name == "sensor_imputation":
        X_train, y_train, X_test, y_test = sensor_imputation.load_split(n_ctx=100, n_test=30, seed=seed)
        return X_train, y_train, X_test, y_test, None
    if benchmark_name == "synthetic_bbo":
        return synthetic_bbo.load_split(d=4, n_ctx=50, n_test=20, seed=seed, cfg=synthetic_bbo_cfg)
    raise ValueError(f"Unknown benchmark: {benchmark_name}")


def run_episode(
    benchmark_name: str,
    seed: int,
    tabicl_reg,
    copula_model,
    n_samples: int,
    rng: np.random.Generator,
    synthetic_bbo_cfg=None,
) -> tuple[list[dict], dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
    X_train, y_train, X_test, y_test, R_true = _load_episode(benchmark_name, seed, synthetic_bbo_cfg)
    n_test = X_test.shape[0]

    X_train_n, X_test_n = normalize_features(X_train, X_test)

    # Marginals and PIT come from the public tabicl.TabICLRegressor, not the
    # low-level TabICL model class inference/copula_inference.py wraps.
    # TabICLRegressor.fit() standardizes y internally (its own StandardScaler)
    # and predict() inverse-transforms back — this is the documented,
    # canonical way TabICL expects to be used (see
    # tabicl_upstream/.../regressor.py), so y_train is passed in RAW, no
    # manual normalization on our end.
    probs = DEFAULT_PROBS
    quantile_grid = tabicl_quantiles(tabicl_reg, X_train_n, y_train, X_test_n, probs)
    Z_train = tabicl_loo_pit(tabicl_reg, X_train_n, y_train, probs, k_folds=10, seed=seed)

    R_independent = independent.get_correlation(n_test)
    R_copula = get_test_correlation(copula_model, X_train_n, Z_train, X_test_n)

    gp_mean, gp_cov = standard_gp.fit_predict(X_train, y_train, X_test, seed=seed)
    qgrid_gp, probs_gp, R_gp = gp_to_quantile_and_R(gp_mean, gp_cov, probs)

    methods = {
        "independent": (quantile_grid, probs, R_independent),
        "copula": (quantile_grid, probs, R_copula),
        "standard_gp": (qgrid_gp, probs_gp, R_gp),
    }

    # Ground truth: for synthetic_bbo the analytical posterior correlation is
    # known exactly. For real datasets (spatial_housing, sensor_imputation)
    # there is no known generative kernel, so we fall back to an EMPIRICAL
    # proxy built from the PIT z-residuals of the true y_test under TabICL's
    # own marginals: z_i * z_j is an unbiased (but single-episode-noisy)
    # estimator of Corr(z_i, z_j), since PIT z-values are ~N(0,1) marginally
    # by construction. This single-episode outer(z, z) is rank-1 and noisy —
    # not a real correlation matrix — so it is not scored (no corr_frob for
    # real datasets), only shown for visual reference; the statistically
    # meaningful signal is the binned correlation-vs-distance plot pooled
    # across all episodes (see plot_correlation_vs_distance in main()).
    if R_true is not None:
        gt_name, R_gt = "ground_truth", R_true
    else:
        z_true, _ = compute_pit(quantile_grid, probs, y_test)
        R_gt = np.clip(np.outer(z_true, z_true), -1.0, 1.0)
        np.fill_diagonal(R_gt, 1.0)
        gt_name = "empirical_ground_truth"

    records = []
    R_by_method = {}
    pair_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method_name, (qgrid, method_probs, R) in methods.items():
        samples, n_clipped = sample_trajectories(qgrid, method_probs, R, n_samples, rng=rng)
        es = compute_energy_score(samples, y_test)
        nll = compute_joint_nll(qgrid, method_probs, R, y_test)
        corr_frob = float(np.linalg.norm(R - R_true, "fro") / n_test) if R_true is not None else None
        records.append(
            {
                "benchmark": benchmark_name,
                "method": method_name,
                "seed": seed,
                "energy_score": es,
                "nll_total": nll["total"],
                "nll_copula": nll["copula"],
                "nll_marginal": nll["marginal"],
                "n_clipped": int(n_clipped),
                "corr_frob": corr_frob,
            }
        )
        R_by_method[method_name] = R
        pair_series[method_name] = collect_pair_distances_and_values(X_test_n, R)

    R_by_method[gt_name] = R_gt
    pair_series[gt_name] = collect_pair_distances_and_values(X_test_n, R_gt)
    return records, R_by_method, pair_series


def main() -> None:
    parser = argparse.ArgumentParser(description="TabICLv2 + Copula inter-instance benchmark suite")
    parser.add_argument("--copula_ckpt", default="./checkpoints/systematic-composition-k5/step_0045000.pt")
    parser.add_argument("--tabicl_ckpt", default="tabicl-regressor-v2-20260212.ckpt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--benchmarks", default=",".join(BENCHMARK_NAMES))
    parser.add_argument("--out_dir", default=os.path.join(_REPO_ROOT, "eval", "results"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    benchmark_names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    unknown = set(benchmark_names) - set(BENCHMARK_NAMES)
    assert not unknown, f"Unknown benchmark(s): {unknown}, must be a subset of {BENCHMARK_NAMES}"

    print(f"Loading TabICL marginal model (TabICLRegressor): {args.tabicl_ckpt}")
    tabicl_reg = make_tabicl_regressor(checkpoint=args.tabicl_ckpt, device=device)

    print(f"Loading copula model: {args.copula_ckpt}")
    copula_model, cfg = load_copula_model(args.copula_ckpt, device=device)

    # synthetic_bbo's ground-truth kernel is sampled with this SAME cfg (see
    # synthetic_bbo.load_split/_sample_episode_kernel_fn), so its
    # kernel family, chain length, and lengthscale/noise priors always match
    # whatever this checkpoint actually trained on, instead of a hardcoded
    # guess that silently drifts if a different checkpoint's cfg differs.
    exclude_kernels = OmegaConf.select(cfg, "data.composite_exclude_kernels", default=None)
    print(f"synthetic_bbo composite_exclude_kernels (from checkpoint cfg): {exclude_kernels}")

    all_records: list[dict] = []
    heatmap_saved: set[str] = set()
    results_path = os.path.join(args.out_dir, "benchmark_results.json")

    for benchmark_name in benchmark_names:
        print(f"\n=== {benchmark_name} ===", flush=True)
        pooled_series: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        for ep in range(args.num_episodes):
            episode_seed = args.seed + ep
            rng = np.random.default_rng(episode_seed)
            try:
                records, R_by_method, pair_series = run_episode(
                    benchmark_name, episode_seed, tabicl_reg, copula_model, args.n_samples, rng,
                    cfg,
                )
            except Exception:  # noqa: BLE001
                print(f"  [episode {ep}] FAILED:", flush=True)
                traceback.print_exc()
                continue

            all_records.extend(records)
            # Saved after every episode (not just at the end) so a run in
            # progress can be inspected on disk — this is a slow run (TabICL
            # is fit ~11x per episode for K-fold PIT + marginals), and the
            # JSON write is negligible next to that cost.
            save_results_json(all_records, results_path)
            for name, dv in pair_series.items():
                pooled_series[name].append(dv)
            if benchmark_name not in heatmap_saved:
                heatmap_path = os.path.join(args.out_dir, "plots", f"{benchmark_name}_correlation.png")
                plot_correlation_heatmaps(R_by_method, heatmap_path)
                heatmap_saved.add(benchmark_name)
            print(f"  episode {ep + 1}/{args.num_episodes} done", flush=True)

        if pooled_series:
            merged_series = {
                name: (np.concatenate([d for d, _ in items]), np.concatenate([v for _, v in items]))
                for name, items in pooled_series.items()
            }
            gt_key = "ground_truth" if "ground_truth" in merged_series else "empirical_ground_truth"
            dist_plot_path = os.path.join(args.out_dir, "plots", f"{benchmark_name}_corr_vs_distance.png")
            plot_correlation_vs_distance(merged_series, dist_plot_path, scatter_series=gt_key)
            print(f"  Saved correlation-vs-distance plot: {dist_plot_path}")

    save_results_json(all_records, results_path)
    print(f"\nSaved {len(all_records)} records to {results_path}")

    print()
    print_markdown_summary(all_records)


if __name__ == "__main__":
    main()
