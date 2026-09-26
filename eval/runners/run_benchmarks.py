"""Joint-distribution benchmarks: spatial interpolation, sensor in-painting and synthetic BBO.

Methods (all give (quantile_grid, probs, R)): independent (TabICL marginals,
R = I), copula (TabICL marginals, the copula model's R) and standard_gp (an
end-to-end sklearn GP on raw y).

Usage:
    python eval/runners/run_benchmarks.py \
        --copula_ckpt <copula checkpoint> \
        --tabicl_ckpt tabicl-regressor-v2-20260212.ckpt \
        --device auto --num_episodes 50 --n_samples 1000
"""

from __future__ import annotations

import argparse
import os
import traceback
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

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
from eval.results import require_coverage  # noqa: E402

BENCHMARK_NAMES = ["spatial_housing", "sensor_imputation", "synthetic_bbo"]
# TabICL's native quantile levels (p_j = j/1001).
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

    # TabICLRegressor standardizes y itself, so y_train is passed raw.
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

    # Ground truth: exact for synthetic_bbo; for real datasets a single-episode
    # outer(z, z) proxy, shown but not scored (the pooled correlation-vs-distance
    # plot is the real signal).
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
    parser.add_argument("--tabicl_ckpt", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--benchmarks", default=",".join(BENCHMARK_NAMES))
    parser.add_argument("--out_dir", default=os.path.join(_REPO_ROOT, "eval", "results"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_failed_fraction", type=float, default=0.0,
        help="Maximum failed-episode fraction before exiting nonzero (1.0 allows exploratory partial runs).",
    )
    args = parser.parse_args()
    if not 0 <= args.max_failed_fraction <= 1:
        parser.error("--max_failed_fraction must lie in [0, 1]")

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

    # synthetic_bbo samples its kernels with this checkpoint's cfg.
    exclude_kernels = OmegaConf.select(cfg, "data.composite_exclude_kernels", default=None)
    print(f"synthetic_bbo composite_exclude_kernels (from checkpoint cfg): {exclude_kernels}")

    all_records: list[dict] = []
    failed_episodes = 0
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
            except Exception as exc:  # noqa: BLE001
                print(f"  [episode {ep}] FAILED:", flush=True)
                traceback.print_exc()
                failed_episodes += 1
                all_records.append({
                    "benchmark": benchmark_name, "seed": episode_seed,
                    "method": "__episode__", "status": "failed",
                    "error": repr(exc),
                })
                save_results_json(all_records, results_path)
                continue

            all_records.extend(records)
            # Save results after every episode.
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
    attempted = len(benchmark_names) * args.num_episodes
    print(f"Episode completion: {attempted - failed_episodes}/{attempted}")
    require_coverage(attempted - failed_episodes, attempted, 1 - args.max_failed_fraction)


if __name__ == "__main__":
    main()
