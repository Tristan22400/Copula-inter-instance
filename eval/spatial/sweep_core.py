"""Driver behind the sweep/diagnose subcommands: for one (checkpoint, region or kernel, grid_size) config, build the ground truth, run the model and score its correlogram.

Loaded models are cached per process (_MODEL_CACHE).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from copula_inter.baselines.gp_mle import fit_gp_baseline_nll
from copula_inter.era5.fetch import fetch as fetch_era5
from copula_inter.era5.io import haversine_distance_km, load_era5_data
from copula_inter.era5.regions import REGIONS
from copula_inter.era5.spatial_stats import (
    bin_correlation_by_distance,
    empirical_spatial_correlation,
    pair_counts_by_distance,
)
from copula_inter.marginal.pit import normalize_features, resolve_pit_ckpt
from copula_inter.marginal.tabicl_regressor import make_tabicl_regressor, tabicl_quantiles
from eval.configs.constants import (
    GP_BASELINE_KERNELS,
    GP_LR_MLE,
    GP_N_RESTARTS_MLE,
    GP_N_STEPS_MLE,
    MAX_DIST_PERCENTILE,
    N_BINS,
    N_CONTEXT,
    N_DAYS,
    N_NLL_TEST,
    N_YSPACE_MC_SAMPLES,
    NLL_PROBS,
    PIT_K_FOLDS,
    SEED,
)
from eval.metrics.joint_nll import compute_joint_nll
from eval.spatial.diagnostics import (
    build_synthetic_grid_task,
    extract_model_context_correlation,
    extract_model_dummy_context_correlation,
    fit_theoretical_law,
    load_marginal_tabicl,
    pool_yspace_samples_and_correlate,
    sample_copula_residual_fields,
)
from inference.copula_inference import load_copula_model

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from copula_inter.copula.model import CopulaTabICL
    from tabicl import TabICLRegressor
    from tabicl._model.tabicl import TabICL

__all__ = [
    "get_model",
    "run_real_config",
    "run_synthetic_config",
    "weighted_corr",
    "weighted_rmse_bias",
    "weighted_r2",
]

_MODEL_CACHE: dict = {}

# Sklearn TabICLRegressor for one-shot quantile grids at held-out points.
_TABICL_REGRESSOR_CACHE: dict = {}


def get_model(ckpt: str, device: "str | None" = None) -> tuple[CopulaTabICL, DictConfig, str, TabICL | None]:
    if ckpt not in _MODEL_CACHE:
        print(f"Loading checkpoint '{ckpt}'...")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, cfg = load_copula_model(ckpt, device=device)
        resolved_device = device
        marginal = load_marginal_tabicl(cfg, resolved_device)
        _MODEL_CACHE[ckpt] = (model, cfg, resolved_device, marginal)
    return _MODEL_CACHE[ckpt]


def _get_tabicl_regressor(source: "str | None", device: str) -> TabICLRegressor:
    """TabICLRegressor for source (pit.resolve_pit_ckpt(cfg) of the checkpoint it is paired with), cached per (source, device)."""
    key = (source, device)
    if key not in _TABICL_REGRESSOR_CACHE:
        _TABICL_REGRESSOR_CACHE[key] = make_tabicl_regressor(checkpoint=source, device=device)
    return _TABICL_REGRESSOR_CACHE[key]


def weighted_corr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
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


def weighted_rmse_bias(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple:
    mask = np.isfinite(a) & np.isfinite(b) & (w > 0)
    if mask.sum() < 1:
        return float("nan"), float("nan")
    a, b, w = a[mask], b[mask], w[mask]
    diff = a - b
    rmse = float(np.sqrt(np.average(diff**2, weights=w)))
    bias = float(np.average(diff, weights=w))
    return rmse, bias


def weighted_r2(pred: np.ndarray, target: np.ndarray, n: np.ndarray) -> float:
    """Weighted R^2 (sigma = 1/sqrt(n) per bin) of a model curve against the binned ground-truth curve.

    A curve-shape diagnostic, not a proper scoring rule (see nll_total).
    """
    mask = np.isfinite(pred) & np.isfinite(target) & (n > 0)
    if mask.sum() < 3:
        return float("nan")
    pred, target, n = pred[mask], target[mask], n[mask]
    sigma = 1.0 / np.sqrt(n)
    resid = pred - target
    t_bar = float(np.average(target, weights=1.0 / sigma**2))
    weighted_ss_res = float(np.sum((resid / sigma) ** 2))
    weighted_ss_tot = float(np.sum(((target - t_bar) / sigma) ** 2))
    return 1.0 - weighted_ss_res / weighted_ss_tot if weighted_ss_tot > 0 else float("nan")


def run_real_config(
    ckpt: str,
    config_name: str,
    region_name: str,
    grid_size: int,
    n_days: int = N_DAYS,
    device: "str | None" = None,
    seed: int = SEED,
    n_context: int = N_CONTEXT,
    n_bins: int = N_BINS,
    compute_gp_baseline: bool = True,
    gp_baseline_kernels: "list | None" = None,
    gp_n_steps_mle: int = GP_N_STEPS_MLE,
    gp_lr_mle: float = GP_LR_MLE,
    gp_n_restarts_mle: int = GP_N_RESTARTS_MLE,
) -> dict:
    """Real-ERA5 config: fetch the region's days and score the checkpoint's context-conditioned correlogram against the empirical one."""
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

    # Held-out points, drawn from the same rng right after the context.
    remaining_idx = np.setdiff1d(np.arange(D), context_idx)
    nll_test_idx = rng.choice(remaining_idx, size=min(N_NLL_TEST, len(remaining_idx)), replace=False)
    x_train_norm, x_test_norm = normalize_features(context_coords, coords)
    x_nll_test_norm = x_test_norm[nll_test_idx]
    # Same marginal as `marginal` (see _get_tabicl_regressor).
    tabicl_reg = _get_tabicl_regressor(resolve_pit_ckpt(cfg), resolved_device)

    gp_kernels = gp_baseline_kernels if gp_baseline_kernels is not None else GP_BASELINE_KERNELS

    model_yspace_samples = []  # (n_days * N_YSPACE_MC_SAMPLES, D) once stacked below
    nll_total_per_day, nll_marginal_per_day, nll_copula_per_day = [], [], []
    gp_nll_per_day: dict = {k: {"total": [], "marginal": [], "copula": []} for k in gp_kernels}
    for d in days:
        day_values = data["t2m"][d].ravel()
        context_values = day_values[context_idx]
        R_context = extract_model_context_correlation(
            model,
            resolved_device,
            marginal,
            context_coords,
            context_values,
            coords,
            k_folds=PIT_K_FOLDS,
        )

        # Y-space correlation curve from pooled model samples.
        z_batch = rng.standard_normal((N_YSPACE_MC_SAMPLES, D))
        model_yspace_samples.append(
            sample_copula_residual_fields(
                marginal,
                context_coords,
                context_values,
                coords,
                R_context,
                resolved_device,
                z_batch,
            )
        )

        # Y-space NLL on the held-out points: one-shot TabICL marginal plus R_context restricted to them.
        try:
            y_nll_test = day_values[nll_test_idx]
            qgrid = tabicl_quantiles(tabicl_reg, x_train_norm, context_values, x_nll_test_norm, NLL_PROBS)
            R_nll = R_context[np.ix_(nll_test_idx, nll_test_idx)]
            nll = compute_joint_nll(qgrid, NLL_PROBS, R_nll, y_nll_test)
            nll_total_per_day.append(nll["total"])
            nll_marginal_per_day.append(nll["marginal"])
            nll_copula_per_day.append(nll["copula"])
        except Exception as exc:  # noqa: BLE001
            print(f"  [day {d}] joint-NLL scoring failed: {exc}")

        if compute_gp_baseline:
            gp_day = fit_gp_baseline_nll(
                cache_key=(config_name, d, seed, n_context_eff),
                x_train_norm=x_train_norm,
                context_values=context_values,
                x_test_norm=x_nll_test_norm,
                y_test=day_values[nll_test_idx],
                kernel_names=gp_kernels,
                n_steps=gp_n_steps_mle,
                lr=gp_lr_mle,
                n_restarts=gp_n_restarts_mle,
                device=resolved_device,
            )
            for kname, parts in gp_day.items():
                for comp in ("total", "marginal", "copula"):
                    gp_nll_per_day[kname][comp].append(parts[comp])
    R_model_yspace = pool_yspace_samples_and_correlate(model_yspace_samples)
    rho_model_yspace = bin_correlation_by_distance(R_model_yspace, dist, bin_edges)
    nll_total = float(np.nanmean(nll_total_per_day)) if nll_total_per_day else float("nan")
    nll_marginal = float(np.nanmean(nll_marginal_per_day)) if nll_marginal_per_day else float("nan")
    nll_copula = float(np.nanmean(nll_copula_per_day)) if nll_copula_per_day else float("nan")
    gp_baseline_nll = (
        {
            kname: {comp: float(np.nanmean(vals)) if vals else float("nan") for comp, vals in parts.items()}
            for kname, parts in gp_nll_per_day.items()
        }
        if compute_gp_baseline
        else {}
    )

    rho_emp = bin_correlation_by_distance(R_emp, dist, bin_edges)
    rho_dummy = bin_correlation_by_distance(R_dummy, dist, bin_edges)

    shape_corr = weighted_corr(rho_model_yspace, rho_emp, pair_counts)
    rmse, bias = weighted_rmse_bias(rho_model_yspace, rho_emp, pair_counts)
    fro_ratio = float(
        np.sqrt(np.nanmean((rho_model_yspace - rho_emp) ** 2)) / max(np.sqrt(np.nanmean(rho_emp**2)), 1e-8)
    )

    gt_fit = fit_theoretical_law(dist_centers, rho_emp, pair_counts.astype(int), "matern")
    model_r2 = weighted_r2(rho_model_yspace, rho_emp, pair_counts)

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
        "nll_total": nll_total,
        "nll_marginal": nll_marginal,
        "nll_copula": nll_copula,
        "gp_baseline_nll": gp_baseline_nll,
        "gt_matern_r2": gt_fit["r_squared"] if gt_fit else None,
        "gt_matern_L_km": gt_fit["params"]["L"] if gt_fit else None,
        "gt_matern_nu": gt_fit["params"].get("nu") if gt_fit else None,
        "dist_centers": dist_centers.tolist(),
        "pair_counts": pair_counts.tolist(),
        "rho_emp": rho_emp.tolist(),
        "rho_model_yspace": rho_model_yspace.tolist(),
        "rho_dummy": rho_dummy.tolist(),
    }
    print(
        f"[{config_name} | {os.path.basename(ckpt)}] shape_corr={shape_corr:.3f} "
        f"rmse={rmse:.3f} bias={bias:+.3f} model_r2={model_r2:.3f} nll_total={nll_total:.3f}"
    )
    return result


def run_synthetic_config(
    ckpt: str,
    config_name: str,
    kernel_name: str,
    grid_size: int,
    n_context: int = N_CONTEXT,
    n_draws: int = 20,
    seed: int = SEED,
    device: "str | None" = None,
    n_bins: int = N_BINS,
) -> dict:
    """Synthetic config: sample a known covariance from one gp_episodes kernel and score how well the checkpoint recovers it."""
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
            model,
            resolved_device,
            marginal,
            context_coords,
            context_values,
            coords,
            k_folds=PIT_K_FOLDS,
        )
        R_pred_draws.append(R_pred)
    R_pred_mean = np.mean(R_pred_draws, axis=0)

    rho_true = bin_correlation_by_distance(R_true, dist, bin_edges)
    rho_pred = bin_correlation_by_distance(R_pred_mean, dist, bin_edges)

    shape_corr = weighted_corr(rho_pred, rho_true, pair_counts)
    rmse, bias = weighted_rmse_bias(rho_pred, rho_true, pair_counts)
    model_r2 = weighted_r2(rho_pred, rho_true, pair_counts)
    gt_fit = fit_theoretical_law(dist_centers, rho_true, pair_counts.astype(int), "matern")

    result = {
        "ckpt": ckpt,
        "config": config_name,
        "true_kernel": kernel_name,
        "grid_size": grid_size,
        "D": int(D),
        "n_context": int(n_context_eff),
        "n_draws": n_draws,
        "shape_corr": shape_corr,
        "rmse": rmse,
        "bias": bias,
        "model_r2": model_r2,
        "gt_matern_r2": gt_fit["r_squared"] if gt_fit else None,
        "dist_centers": dist_centers.tolist(),
        "pair_counts": pair_counts.tolist(),
        "rho_true": rho_true.tolist(),
        "rho_pred": rho_pred.tolist(),
    }
    print(
        f"[{config_name} | {os.path.basename(ckpt)}] shape_corr={shape_corr:.3f} "
        f"model_r2={model_r2:.3f} bias={bias:+.3f}"
    )
    return result
