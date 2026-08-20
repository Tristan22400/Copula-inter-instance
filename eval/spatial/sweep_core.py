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

from eval.configs.constants import (
    MAX_DIST_PERCENTILE, N_BINS, N_CONTEXT, N_DAYS, N_NLL_TEST, NLL_PROBS, PIT_K_FOLDS, SEED,
)
from eval.configs.regions import REGIONS
from eval.data.era5_io import haversine_distance_km, load_era5_data
from eval.data.fetch_era5 import fetch as fetch_era5
from eval.metrics.joint_nll import compute_joint_nll
from eval.spatial.diagnostics import (
    bin_correlation_by_distance,
    build_synthetic_grid_task,
    compute_context_z_train,
    empirical_spatial_correlation,
    extract_model_context_correlation,
    extract_model_dummy_context_correlation,
    fit_theoretical_law,
    get_ground_truth_observations,
    load_copula_model,
    load_marginal_tabicl,
    pair_counts_by_distance,
)
from eval.tabicl_utils import make_tabicl_regressor, tabicl_quantiles
from inference.copula_inference import normalize_features

__all__ = [
    "get_model", "run_real_config", "run_synthetic_config", "build_era5_probe",
    "weighted_corr", "weighted_rmse_bias", "weighted_r2",
]

_MODEL_CACHE: dict = {}
# Separate from _MODEL_CACHE: a public sklearn-style TabICLRegressor used
# ONLY for one-shot (non-K-fold) marginal quantile grids at held-out
# joint-NLL test points in run_real_config -- a different interface from
# `marginal` (src/pit.py::load_tabicl, K-fold PIT for context z_train), same
# tabicl_quantiles helper eval/runners/run_benchmarks.py already uses.
_TABICL_REGRESSOR_CACHE: dict = {}


def get_model(ckpt: str, device: "str | None" = None):
    if ckpt not in _MODEL_CACHE:
        print(f"Loading checkpoint '{ckpt}'...")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, cfg, resolved_device = load_copula_model(ckpt, device=device)
        marginal = load_marginal_tabicl(cfg, resolved_device)
        _MODEL_CACHE[ckpt] = (model, cfg, resolved_device, marginal)
    return _MODEL_CACHE[ckpt]


def _get_tabicl_regressor(device: str):
    if device not in _TABICL_REGRESSOR_CACHE:
        _TABICL_REGRESSOR_CACHE[device] = make_tabicl_regressor(device=device)
    return _TABICL_REGRESSOR_CACHE[device]


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
    rmse = float(np.sqrt(np.average(diff ** 2, weights=w)))
    bias = float(np.average(diff, weights=w))
    return rmse, bias


def weighted_r2(pred: np.ndarray, target: np.ndarray, n: np.ndarray) -> float:
    """Same weighted-R^2 convention as fit_theoretical_law (sigma =
    1/sqrt(n), i.e. bin-mean SEM weighting), applied to a model curve
    instead of a fitted theoretical law, so the two R^2 numbers are directly
    comparable: how much of the ground-truth curve's own weighted variance
    does each explain.

    CAVEAT: this is a curve-shape diagnostic on the binned, distance-averaged
    correlation curve, NOT a proper scoring rule -- it never sees marginal
    calibration or density sharpness, and per-bin averaging can mask
    per-instance miscalibration. It is complementary to, and must never
    substitute for, the likelihood-based `nll_total`
    (eval/metrics/joint_nll.py::compute_joint_nll) for cross-method/
    cross-checkpoint comparisons -- use this to localize *where* (which
    distance range) a curve mismatches, and nll_total to say *how much
    worse* the actual predictive density is."""
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

    # Held-out points (never in context) for the joint-NLL score below —
    # drawn from the SAME rng stream right after context_idx, matching
    # eval/runners/compare_marginal_backbones.py::run_task's synthetic-mode
    # analogue. x_train_norm/x_test_norm recomputes what
    # extract_model_context_correlation does internally per call, needed
    # here directly for tabicl_quantiles's one-shot marginal fit.
    remaining_idx = np.setdiff1d(np.arange(D), context_idx)
    nll_test_idx = rng.choice(remaining_idx, size=min(N_NLL_TEST, len(remaining_idx)), replace=False)
    x_train_norm, x_test_norm = normalize_features(context_coords, coords)
    x_nll_test_norm = x_test_norm[nll_test_idx]
    tabicl_reg = _get_tabicl_regressor(resolved_device)

    rho_context_per_day = []
    nll_total_per_day, nll_marginal_per_day, nll_copula_per_day = [], [], []
    for d in days:
        day_values = data["t2m"][d].ravel()
        context_values = day_values[context_idx]
        R_context = extract_model_context_correlation(
            model, resolved_device, marginal, context_coords, context_values, coords, k_folds=PIT_K_FOLDS,
        )
        rho_context_per_day.append(bin_correlation_by_distance(R_context, dist, bin_edges))

        # Total (marginal+copula) Y-space NLL on the held-out points: a
        # one-shot (non-K-fold — never in context) TabICL quantile grid
        # supplies the marginal, R_context restricted to the same points
        # supplies the copula. Neither model_r2 (a binned correlation-curve-
        # shape diagnostic, not a proper scoring rule) nor any other real-
        # ERA5 script in eval/ currently reports a likelihood-based number.
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
    rho_context_mean = np.nanmean(np.array(rho_context_per_day), axis=0)
    nll_total = float(np.nanmean(nll_total_per_day)) if nll_total_per_day else float("nan")
    nll_marginal = float(np.nanmean(nll_marginal_per_day)) if nll_marginal_per_day else float("nan")
    nll_copula = float(np.nanmean(nll_copula_per_day)) if nll_copula_per_day else float("nan")

    rho_emp = bin_correlation_by_distance(R_emp, dist, bin_edges)
    rho_dummy = bin_correlation_by_distance(R_dummy, dist, bin_edges)

    shape_corr = weighted_corr(rho_context_mean, rho_emp, pair_counts)
    rmse, bias = weighted_rmse_bias(rho_context_mean, rho_emp, pair_counts)
    fro_ratio = float(np.sqrt(np.nanmean((rho_context_mean - rho_emp) ** 2)) /
                       max(np.sqrt(np.nanmean(rho_emp ** 2)), 1e-8))

    gt_fit = fit_theoretical_law(dist_centers, rho_emp, pair_counts.astype(int), "matern")
    model_r2 = weighted_r2(rho_context_mean, rho_emp, pair_counts)

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
          f"rmse={rmse:.3f} bias={bias:+.3f} model_r2={model_r2:.3f} nll_total={nll_total:.3f}")
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

    shape_corr = weighted_corr(rho_pred, rho_true, pair_counts)
    rmse, bias = weighted_rmse_bias(rho_pred, rho_true, pair_counts)
    model_r2 = weighted_r2(rho_pred, rho_true, pair_counts)
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


def build_era5_probe(
    region_name: str, grid_size: int, n_days_fetch: int, n_days_probe: int,
    n_context: int, n_bins: int, tabicl_marginal, device: str, seed: int = SEED,
) -> dict:
    """Frozen real-ERA5 (context, target-grid, ground-truth correlogram)
    probe for `region_name` — the training-loop analogue of `run_real_config`
    above, split into a model-INDEPENDENT precompute step. Meant to be called
    ONCE (see src/train.py::_build_era5_val_batches), not per validate() call.

    Unlike a GP-generated episode, real ERA5 has no known oracle Sigma_star /
    R_star, so what's frozen here is the ground-truth EMPIRICAL correlogram
    (rho_emp, from Pearson correlation across days — same as run_real_config's
    R_emp) plus the PIT z_train for a fixed context sample on a handful of
    fixed days. Everything downstream that depends on the (still-training,
    changing every call) copula model — the context-conditioned forward pass
    and its scoring against rho_emp — is deliberately NOT done here; that
    happens in train.py::validate() every call, on the cheap frozen inputs
    this function returns.

    z_train is computed via eval.spatial.diagnostics.compute_context_z_train —
    the same K-fold LOO PIT step extract_model_context_correlation's
    real-context z_train always uses, but only once per probe day here
    instead of once per validate() call — `tabicl_marginal` never changes
    during training, so re-running PIT on the same (context_coords,
    context_values) every call would just recompute an identical result
    (mirrors train.py::_build_tabicl_val_z's same precompute-once rationale).
    """
    from inference.copula_inference import normalize_features

    rng = np.random.default_rng(seed)
    lat_bounds, lon_bounds = REGIONS[region_name]
    nc_path = fetch_era5(region_name, lat_bounds, lon_bounds, grid_size, n_days_fetch)
    data = load_era5_data(nc_path)
    lat, lon = data["latitude"], data["longitude"]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    D = coords.shape[0]

    R_emp = empirical_spatial_correlation(data, target="raw")
    dist = haversine_distance_km(coords)
    dist_iu = dist[np.triu_indices_from(dist, k=1)]
    max_dist = np.percentile(dist_iu, MAX_DIST_PERCENTILE)
    bin_edges = np.linspace(0.0, max_dist, n_bins + 1)
    dist_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pair_counts = pair_counts_by_distance(dist, bin_edges).astype(float)
    rho_emp = bin_correlation_by_distance(R_emp, dist, bin_edges)

    n_time = data["t2m"].shape[0]
    n_pick = min(n_days_probe, n_time)
    days = sorted(set(np.linspace(0, n_time - 1, n_pick).round().astype(int).tolist()))

    n_context_eff = max(1, min(n_context, D - 1))
    context_idx = rng.choice(D, size=n_context_eff, replace=False)
    context_coords = coords[context_idx]

    x_train_norm, x_test_norm = normalize_features(context_coords, coords)

    z_train_per_day = []
    for d in days:
        day_values = data["t2m"][d].ravel()
        context_values = day_values[context_idx]
        z_train = compute_context_z_train(
            x_train_norm, context_values, tabicl_marginal, device, k_folds=PIT_K_FOLDS,
        )
        z_train_per_day.append(z_train)

    return {
        "region": region_name,
        "x_train_norm": x_train_norm,                       # (P, 2)
        "x_test_norm": x_test_norm,                          # (D, 2)
        "z_train_per_day": np.stack(z_train_per_day, axis=0),  # (n_days_probe, P)
        "dist": dist,                                        # (D, D)
        "bin_edges": bin_edges,
        "dist_centers": dist_centers,
        "pair_counts": pair_counts,
        "rho_emp": rho_emp,
        "D": int(D),
        "n_context": int(n_context_eff),
    }
