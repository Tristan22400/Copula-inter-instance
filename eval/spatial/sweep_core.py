"""sweep_core.py — batch driver behind the `sweep`/`diagnose` subcommands:
for one (checkpoint, region-or-kernel, grid_size) config, fetch/sample the
ground truth, run the model's context-conditioned forward pass, and score
how well the model's binned correlogram matches it. Promoted from
plots/spatial_correlation_sweep.py (real-ERA5 path) and
plots/run_synthetic_checkpoint_comparison.py (synthetic path), unified
behind one weighted-scoring convention.

Keeps a process-wide model cache (_MODEL_CACHE) so a `sweep`/`diagnose`
call that loops over many configs for the same checkpoint loads its weights
once, not once per config — the in-process-loop, no-subprocess design the
plan's automation goal relies on.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from eval.configs.constants import MAX_DIST_PERCENTILE, N_BINS, N_CONTEXT, N_DAYS, PIT_K_FOLDS, SEED
from eval.configs.regions import REGIONS
from eval.data.era5_io import haversine_distance_km, load_era5_data
from eval.data.fetch_era5 import fetch as fetch_era5
from eval.spatial.diagnostics import (
    bin_correlation_by_distance,
    build_synthetic_grid_task,
    empirical_spatial_correlation,
    extract_model_context_correlation,
    extract_model_dummy_context_correlation,
    fit_theoretical_law,
    get_ground_truth_observations,
    load_copula_model,
    load_marginal_tabicl,
    pair_counts_by_distance,
)

__all__ = ["get_model", "run_real_config", "run_synthetic_config"]

_MODEL_CACHE: dict = {}


def get_model(ckpt: str, device: "str | None" = None):
    if ckpt not in _MODEL_CACHE:
        print(f"Loading checkpoint '{ckpt}'...")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, cfg, resolved_device = load_copula_model(ckpt, device=device)
        marginal = load_marginal_tabicl(cfg, resolved_device)
        _MODEL_CACHE[ckpt] = (model, cfg, resolved_device, marginal)
    return _MODEL_CACHE[ckpt]


def _weighted_corr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b) & (w > 0)
    if mask.sum() < 3:
        return float("nan")
    a, b, w = a[mask], b[mask], w[mask]
    aw = np.average(a, weights=w)
    bw = np.average(b, weights=w)
    cov = np.average((a - aw) * (b - bw), weights=w)
    var_a = np.average((a - aw) ** 2, weights=w)
    var_b = np.average((b - bw) ** 2, weights=w)
    if var_a <= 0 or var_b <= 0:
        return float("nan")
    return float(cov / np.sqrt(var_a * var_b))


def _weighted_rmse_bias(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple:
    mask = np.isfinite(a) & np.isfinite(b) & (w > 0)
    if mask.sum() < 1:
        return float("nan"), float("nan")
    a, b, w = a[mask], b[mask], w[mask]
    diff = a - b
    rmse = float(np.sqrt(np.average(diff ** 2, weights=w)))
    bias = float(np.average(diff, weights=w))
    return rmse, bias


def _weighted_r2(pred: np.ndarray, target: np.ndarray, n: np.ndarray) -> float:
    """Same weighted-R^2 convention as fit_theoretical_law (sigma =
    1/sqrt(n), i.e. bin-mean SEM weighting), applied to a model curve
    instead of a fitted theoretical law, so the two R^2 numbers are directly
    comparable: how much of the ground-truth curve's own weighted variance
    does each explain."""
    mask = np.isfinite(pred) & np.isfinite(target) & (n > 0)
    if mask.sum() < 3:
        return float("nan")
    pred, target, n = pred[mask], target[mask], n[mask]
    sigma = 1.0 / np.sqrt(n)
    resid = pred - target
    t_bar = float(np.average(target, weights=1.0 / sigma ** 2))
    weighted_ss_res = float(np.sum((resid / sigma) ** 2))
    weighted_ss_tot = float(np.sum(((target - t_bar) / sigma) ** 2))
    return 1.0 - weighted_ss_res / weighted_ss_tot if weighted_ss_tot > 0 else float("nan")


def run_real_config(
    ckpt: str, config_name: str, region_name: str, grid_size: int,
    n_days: int = N_DAYS, device: "str | None" = None, seed: int = SEED,
    n_context: int = N_CONTEXT, n_bins: int = N_BINS,
) -> dict:
    """Real-ERA5 config: fetch (region_name, grid_size, n_days) from the
    on-disk-cached ARCO-ERA5 fetcher, then score the checkpoint's
    context-conditioned correlogram against the empirical ground truth."""
    rng = np.random.default_rng(seed)
    model, cfg, resolved_device, marginal = get_model(ckpt, device)
    lat_bounds, lon_bounds = REGIONS[region_name]

    nc_path = fetch_era5(region_name, lat_bounds, lon_bounds, grid_size, n_days)
    data = load_era5_data(nc_path)
    lat, lon = data["latitude"], data["longitude"]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    D = coords.shape[0]

    R_emp = empirical_spatial_correlation(data, target="raw")
    R_dummy = extract_model_dummy_context_correlation(model, resolved_device, coords)

    dist = haversine_distance_km(coords)
    dist_iu = dist[np.triu_indices_from(dist, k=1)]
    max_dist = np.percentile(dist_iu, MAX_DIST_PERCENTILE)
    bin_edges = np.linspace(0.0, max_dist, n_bins + 1)
    dist_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pair_counts = pair_counts_by_distance(dist, bin_edges).astype(float)

    n_time = data["t2m"].shape[0]
    n_pick = min(6, n_time)
    days = sorted(set(np.linspace(0, n_time - 1, n_pick).round().astype(int).tolist()))

    n_context_eff = max(1, min(n_context, D - 1))
    context_idx = rng.choice(D, size=n_context_eff, replace=False)
    context_coords = coords[context_idx]

    rho_context_per_day = []
    for d in days:
        day_values = data["t2m"][d].ravel()
        context_values = day_values[context_idx]
        R_context = extract_model_context_correlation(
            model, resolved_device, marginal, context_coords, context_values, coords, k_folds=PIT_K_FOLDS,
        )
        rho_context_per_day.append(bin_correlation_by_distance(R_context, dist, bin_edges))
    rho_context_mean = np.nanmean(np.array(rho_context_per_day), axis=0)

    rho_emp = bin_correlation_by_distance(R_emp, dist, bin_edges)
    rho_dummy = bin_correlation_by_distance(R_dummy, dist, bin_edges)

    shape_corr = _weighted_corr(rho_context_mean, rho_emp, pair_counts)
    rmse, bias = _weighted_rmse_bias(rho_context_mean, rho_emp, pair_counts)
    fro_ratio = float(np.sqrt(np.nanmean((rho_context_mean - rho_emp) ** 2)) /
                       max(np.sqrt(np.nanmean(rho_emp ** 2)), 1e-8))

    gt_fit = fit_theoretical_law(dist_centers, rho_emp, pair_counts.astype(int), "matern")
    model_r2 = _weighted_r2(rho_context_mean, rho_emp, pair_counts)

    result = {
        "ckpt": ckpt,
        "config": config_name,
        "region": region_name,
        "grid_size_requested": grid_size,
        "D": int(D),
        "n_context": int(n_context_eff),
        "n_days": int(n_time),
        "shape_corr": shape_corr,
        "rmse": rmse,
        "bias": bias,
        "fro_ratio_rms": fro_ratio,
        "model_r2": model_r2,
        "gt_matern_r2": gt_fit["r_squared"] if gt_fit else None,
        "gt_matern_L_km": gt_fit["params"]["L"] if gt_fit else None,
        "gt_matern_nu": gt_fit["params"].get("nu") if gt_fit else None,
        "dist_centers": dist_centers.tolist(),
        "pair_counts": pair_counts.tolist(),
        "rho_emp": rho_emp.tolist(),
        "rho_context_mean": rho_context_mean.tolist(),
        "rho_dummy": rho_dummy.tolist(),
    }
    print(f"[{config_name} | {os.path.basename(ckpt)}] shape_corr={shape_corr:.3f} "
          f"rmse={rmse:.3f} bias={bias:+.3f} model_r2={model_r2:.3f}")
    return result


def run_synthetic_config(
    ckpt: str, config_name: str, kernel_name: str, grid_size: int,
    n_context: int = N_CONTEXT, n_draws: int = 20, seed: int = SEED,
    device: "str | None" = None, n_bins: int = N_BINS,
) -> dict:
    """Synthetic config: sample a KNOWN ground-truth covariance directly
    from one src/data_gen.py kernel (exact, zero estimation noise) and score
    how well the checkpoint's context-conditioned forward pass recovers it.
    """
    model, cfg, resolved_device, marginal = get_model(ckpt, device)

    task = build_synthetic_grid_task(cfg, kernel_name, grid_size, n_context, n_bins, seed, min_context=1)
    coords, D = task["coords"], task["D"]
    R_true, dist, bin_edges, pair_counts = task["R_true"], task["dist"], task["bin_edges"], task["pair_counts"]
    context_idx, context_coords = task["context_idx"], task["context_coords"]
    n_context_eff, rng, L = task["n_context_eff"], task["rng"], task["L"]
    dist_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    R_pred_draws = []
    for _ in range(n_draws):
        z_true = L @ rng.standard_normal(D)
        context_values = z_true[context_idx]
        R_pred = extract_model_context_correlation(
            model, resolved_device, marginal, context_coords, context_values, coords, k_folds=PIT_K_FOLDS,
        )
        R_pred_draws.append(R_pred)
    R_pred_mean = np.mean(R_pred_draws, axis=0)

    rho_true = bin_correlation_by_distance(R_true, dist, bin_edges)
    rho_pred = bin_correlation_by_distance(R_pred_mean, dist, bin_edges)

    shape_corr = _weighted_corr(rho_pred, rho_true, pair_counts)
    rmse, bias = _weighted_rmse_bias(rho_pred, rho_true, pair_counts)
    model_r2 = _weighted_r2(rho_pred, rho_true, pair_counts)
    gt_fit = fit_theoretical_law(dist_centers, rho_true, pair_counts.astype(int), "matern")

    result = {
        "ckpt": ckpt, "config": config_name, "true_kernel": kernel_name,
        "grid_size": grid_size, "D": int(D), "n_context": int(n_context_eff), "n_draws": n_draws,
        "shape_corr": shape_corr, "rmse": rmse, "bias": bias, "model_r2": model_r2,
        "gt_matern_r2": gt_fit["r_squared"] if gt_fit else None,
        "dist_centers": dist_centers.tolist(),
        "pair_counts": pair_counts.tolist(),
        "rho_true": rho_true.tolist(),
        "rho_pred": rho_pred.tolist(),
    }
    print(f"[{config_name} | {os.path.basename(ckpt)}] shape_corr={shape_corr:.3f} "
          f"model_r2={model_r2:.3f} bias={bias:+.3f}")
    return result
