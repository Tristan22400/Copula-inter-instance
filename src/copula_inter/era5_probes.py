"""Fixed real-ERA5 validation batches and the figures train.py logs from them."""

from __future__ import annotations

import matplotlib

from copula_inter.probe_batches import _name_seed, _tabicl_pit_batch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from copula_inter.model import build_sigma
from copula_inter.pit import (
    DEFAULT_K_FOLDS,
    normalize_targets,
    tabicl_forward,
)
from eval.configs.constants import GP_LR_MLE
from eval.configs.regions import REGIONS as ERA5_REGIONS
from eval.data.era5_io import load_era5_data, safe_cholesky
from eval.data.fetch_era5 import fetch as fetch_era5
from eval.spatial.diagnostics import compute_context_z_train
from eval.spatial.sweep_core import _fit_gp_baseline_nll, build_era5_probe
from eval.viz.correlation_plots import (
    plot_marginal_variance_grid,
    plot_mean_removed_grid,
    plot_residual_grid,
    plot_z_predictor_samples,
)
from inference.copula_inference import normalize_features


def _build_era5_val_batches(cfg: DictConfig, tabicl_marginal, device: str) -> dict[str, dict]:
    """Fixed real-ERA5 probes per region for the era5_fit/<region> metrics.

    For each region, eval.spatial.sweep_core.build_era5_probe freezes a context
    sample (PIT'd with tabicl_marginal, or standardized when it is None) on a few
    days. With tabicl_marginal, the held-out points are PIT'd too
    (nll_test_z / nll_test_log_pdf) so validate() can score a Y-space NLL. A
    classical GP-MLE baseline NLL is fitted once per region with the
    baselines.era5_gp_* settings.
    """
    ecfg = cfg.get("baselines", {}) or {}
    region_names = list(ecfg.get("era5_regions") or list(ERA5_REGIONS.keys()))
    grid_size = int(ecfg.get("era5_grid_size", 10))
    n_days_fetch = int(ecfg.get("era5_n_days_fetch", 60))
    n_days_probe = int(ecfg.get("era5_n_days_probe", 3))
    n_context = int(ecfg.get("era5_n_context", 30))
    n_bins = int(ecfg.get("era5_n_bins", 12))
    base_seed = int(ecfg.get("era5_seed", 20260818))
    pit_k_folds = int(cfg.tabicl.get("pit_k_folds", DEFAULT_K_FOLDS))

    # Classical GP-MLE baseline per region, at lighter settings than the spatial
    # sweep's (baselines.era5_gp_baseline_kernels / _n_restarts_mle / _n_steps_mle).
    gp_baseline_enabled = bool(ecfg.get("era5_gp_baseline", True))
    gp_baseline_kernels = list(ecfg.get("era5_gp_baseline_kernels") or ["matern32", "rational_quadratic"])
    gp_baseline_n_days = int(ecfg.get("era5_gp_baseline_n_days", 1))
    gp_n_steps_mle = int(ecfg.get("era5_gp_n_steps_mle", 300))
    gp_lr_mle = float(ecfg.get("era5_gp_lr_mle", GP_LR_MLE))
    gp_n_restarts_mle = int(ecfg.get("era5_gp_n_restarts_mle", 1))

    batches: dict[str, dict] = {}
    for region_name in region_names:
        if region_name not in ERA5_REGIONS:
            continue  # not a registered eval/configs/regions.py entry
        region_seed = _name_seed(base_seed, region_name)
        probe = build_era5_probe(
            region_name, grid_size, n_days_fetch, n_days_probe, n_context, n_bins,
            tabicl_marginal, device, seed=region_seed,
        )
        n_days_p = probe["z_train_per_day"].shape[0]
        x_train = torch.as_tensor(probe["x_train_norm"], dtype=torch.float32, device=device)
        x_test = torch.as_tensor(probe["x_test_norm"], dtype=torch.float32, device=device)
        z_train = torch.as_tensor(probe["z_train_per_day"], dtype=torch.float32, device=device)
        model_batch = {
            "x_train": x_train.unsqueeze(0).expand(n_days_p, -1, -1).contiguous(),
            "x_test": x_test.unsqueeze(0).expand(n_days_p, -1, -1).contiguous(),
            "z_train": z_train,
            "test_mask": torch.ones(n_days_p, probe["D"], dtype=torch.bool, device=device),
        }
        region_batch = {
            "batch": model_batch,
            "dist": probe["dist"],
            "bin_edges": probe["bin_edges"],
            "pair_counts": probe["pair_counts"],
            "rho_emp": probe["rho_emp"],
        }
        if tabicl_marginal is not None:
            n_nll = probe["x_nll_test_norm"].shape[0]
            x_nll_test = torch.as_tensor(probe["x_nll_test_norm"], dtype=torch.float32, device=device)
            nll_pit_batch = {
                "x_train": model_batch["x_train"],
                "y_train": torch.as_tensor(probe["context_values_per_day"], dtype=torch.float32, device=device),
                "train_mask": torch.ones(n_days_p, probe["n_context"], dtype=torch.bool, device=device),
                "x_test": x_nll_test.unsqueeze(0).expand(n_days_p, -1, -1).contiguous(),
                "y_test": torch.as_tensor(probe["nll_test_values_per_day"], dtype=torch.float32, device=device),
                "test_mask": torch.ones(n_days_p, n_nll, dtype=torch.bool, device=device),
            }
            nll_pit = _tabicl_pit_batch(nll_pit_batch, tabicl_marginal, pit_k_folds, device)
            region_batch["nll_test_idx"] = probe["nll_test_idx"]
            region_batch["nll_test_z"] = nll_pit["z_test"].to(device)
            region_batch["nll_test_log_pdf"] = nll_pit["log_pdf_test"].to(device)

        if gp_baseline_enabled:
            n_gp_days = max(1, min(gp_baseline_n_days, n_days_p))
            gp_nll_per_day: dict = {k: {"total": [], "marginal": [], "copula": []} for k in gp_baseline_kernels}
            for d in range(n_gp_days):
                gp_day = _fit_gp_baseline_nll(
                    cache_key=(region_name, d, region_seed, n_context),
                    x_train_norm=probe["x_train_norm"],
                    context_values=probe["context_values_per_day"][d],
                    x_test_norm=probe["x_nll_test_norm"],
                    y_test=probe["nll_test_values_per_day"][d],
                    kernel_names=gp_baseline_kernels,
                    n_steps=gp_n_steps_mle, lr=gp_lr_mle, n_restarts=gp_n_restarts_mle,
                    device=device,
                )
                for kname, parts in gp_day.items():
                    for comp in ("total", "marginal", "copula"):
                        gp_nll_per_day[kname][comp].append(parts[comp])
            region_batch["gp_baseline_nll"] = {
                kname: {comp: float(np.nanmean(vals)) for comp, vals in parts.items()}
                for kname, parts in gp_nll_per_day.items()
            }
        batches[region_name] = region_batch
    return batches


def _build_era5_viz_batch(cfg: DictConfig, tabicl_marginal, device: str) -> "dict | None":
    """Fixed sparse-context ERA5 probe for the val/era5_* figures.

    One region, fixed days, and a context of fewer than
    baselines.era5_viz_context_frac of the grid points. Precomputes, per day, the
    context z_train, the TabICL marginal distribution at every grid point, its
    mean and variance in Kelvin, and the fitted-GP references, so validate()
    only reruns the copula model.
    """
    ecfg = cfg.get("baselines", {}) or {}
    region_pool = list(ecfg.get("era5_regions") or ERA5_REGIONS.keys())
    if not region_pool:
        return None
    region_name = str(ecfg.get("era5_viz_region") or region_pool[0])
    if region_name not in ERA5_REGIONS:
        return None
    grid_size = int(ecfg.get("era5_viz_grid_size", 24))
    n_days_fetch = int(ecfg.get("era5_n_days_fetch", 60))
    n_days_viz = int(ecfg.get("era5_viz_n_days", 4))
    context_frac = float(ecfg.get("era5_viz_context_frac", 0.05))
    seed = int(ecfg.get("era5_viz_seed", 20260825))
    pit_k_folds = int(cfg.tabicl.get("pit_k_folds", DEFAULT_K_FOLDS))
    # Fitted-GP reference row, same era5_gp_* settings as the era5_fit baseline.
    gp_row_enabled = bool(ecfg.get("era5_viz_gp", True))
    gp_row_kernel = str(ecfg.get("era5_viz_gp_kernel", "matern32"))
    gp_row_n_steps = int(ecfg.get("era5_gp_n_steps_mle", 300))
    gp_row_lr = float(ecfg.get("era5_gp_lr_mle", GP_LR_MLE))
    gp_row_n_restarts = int(ecfg.get("era5_gp_n_restarts_mle", 1))

    lat_bounds, lon_bounds = ERA5_REGIONS[region_name]
    nc_path = fetch_era5(region_name, lat_bounds, lon_bounds, grid_size, n_days_fetch)
    data = load_era5_data(nc_path)
    lat, lon = data["latitude"], data["longitude"]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    D = coords.shape[0]

    rng = np.random.default_rng(seed)
    n_time = data["t2m"].shape[0]
    n_pick = min(n_days_viz, n_time)
    days = sorted(set(np.linspace(0, n_time - 1, n_pick).round().astype(int).tolist()))

    # int() truncates, so the context stays strictly below context_frac of D.
    n_context = max(1, min(int(context_frac * D), D - 1))
    context_idx = rng.choice(D, size=n_context, replace=False)
    context_coords = coords[context_idx]
    x_train_norm, x_test_norm = normalize_features(context_coords, coords)
    x_full = np.concatenate([x_train_norm, x_test_norm], axis=0)
    x_batch = torch.as_tensor(x_full, dtype=torch.float32, device=device).unsqueeze(0)

    true_fields, z_train_per_day = [], []
    dists_per_day, y_mean_per_day, y_std_per_day = [], [], []
    gp_post_per_day: list = []
    gp_post_z_per_day: list = []
    marginal_var_per_day: list = []
    marginal_mean_per_day: list = []
    for d in days:
        frame = data["t2m"][d]
        true_fields.append(frame)
        context_values = frame.ravel()[context_idx]
        z_train_d = compute_context_z_train(x_train_norm, context_values, tabicl_marginal, device, k_folds=pit_k_folds)
        z_train_per_day.append(z_train_d)
        gp_post_per_day.append(
            _era5_viz_gp_posterior(
                x_train_norm, context_values, x_test_norm, gp_row_kernel,
                gp_row_n_steps, gp_row_lr, gp_row_n_restarts, device,
            ) if gp_row_enabled else None
        )
        gp_post_z_per_day.append(
            _era5_viz_gp_posterior_on_z(
                x_train_norm, z_train_d, x_test_norm, gp_row_kernel,
                gp_row_n_steps, gp_row_lr, gp_row_n_restarts, device,
            ) if gp_row_enabled else None
        )
        context_values_t = torch.as_tensor(context_values, dtype=torch.float32, device=device)
        context_values_scaled_t, _, y_mean_t, y_std_t = normalize_targets(context_values_t)
        y_mean_per_day.append(y_mean_t.double())
        y_std_per_day.append(y_std_t.double())
        if tabicl_marginal is None:
            dists_per_day.append(None)
            marginal_var_per_day.append(None)
            marginal_mean_per_day.append(None)
            continue
        with torch.no_grad():
            logits = tabicl_forward(
                tabicl_marginal, x_batch, context_values_scaled_t.unsqueeze(0)
            )  # (1, D, Q)
            dist_d = tabicl_marginal.quantile_dist(logits.reshape(D, -1))
            dists_per_day.append(dist_d)
            # Variance in Kelvin^2: Var[y_scaled | x] * y_std^2.
            marginal_var_per_day.append((dist_d.variance() * y_std_t.double() ** 2).cpu().numpy())
            # Mean in Kelvin.
            marginal_mean_per_day.append((y_mean_t.double() + y_std_t.double() * dist_d.mean().double()).cpu().numpy())

    return {
        "region": region_name, "lat": lat, "lon": lon, "grid_shape": data["t2m"][days[0]].shape,
        "days": days, "true_fields": true_fields, "coords": coords,
        "context_coords": context_coords, "D": D, "n_context": n_context,
        "x_train_norm": x_train_norm, "x_test_norm": x_test_norm,
        "z_train_per_day": z_train_per_day,
        "dists_per_day": dists_per_day, "y_mean_per_day": y_mean_per_day, "y_std_per_day": y_std_per_day,
        "marginal_var_per_day": marginal_var_per_day, "marginal_mean_per_day": marginal_mean_per_day,
        "gp_post_per_day": gp_post_per_day, "gp_post_z_per_day": gp_post_z_per_day,
        "gp_row_kernel": gp_row_kernel,
        "seed": seed,
    }


def _era5_viz_gp_posterior(
    x_train_norm: np.ndarray, context_values: np.ndarray, x_test_norm: np.ndarray,
    kernel_name: str, n_steps: int, lr: float, n_restarts: int, device: str,
) -> "dict | None":
    """GP posterior (mean, Cholesky factor) at the viz grid, fitted by MLE+MAP on the sparse context.

    Same fit as sweep_core._fit_gp_baseline_nll (posterior mode, y standardized
    for fitting and rescaled back). Returns None if fitting or factorization fails.
    """
    from eval.baselines.classical import fit_and_eval_gpytorch

    try:
        mu_y = float(context_values.mean())
        sigma_y = max(float(context_values.std(ddof=1)), 1e-6) if len(context_values) > 1 else 1.0
        X_tr = torch.as_tensor(x_train_norm, dtype=torch.float32, device=device)
        X_te = torch.as_tensor(x_test_norm, dtype=torch.float32, device=device)
        y_tr = torch.as_tensor((context_values - mu_y) / sigma_y, dtype=torch.float32, device=device)
        fit = fit_and_eval_gpytorch(
            X_tr, y_tr, X_te, kernel_name, n_steps=n_steps, lr=lr,
            oracle_mode="posterior", n_restarts=n_restarts,
        )
        mean = (fit["mean"].double() * sigma_y + mu_y).cpu().numpy()
        Sigma = (fit["Sigma"].double() * (sigma_y ** 2)).cpu().numpy()
        return {"mean": mean, "L": safe_cholesky(Sigma), "kernel": kernel_name}
    except Exception as exc:  # noqa: BLE001
        print(f"  [era5_viz_gp:{kernel_name}] fit failed, dropping GP row: {exc}")
        return None


def _era5_viz_gp_posterior_on_z(
    x_train_norm: np.ndarray, z_train: np.ndarray, x_test_norm: np.ndarray,
    kernel_name: str, n_steps: int, lr: float, n_restarts: int, device: str,
) -> "dict | None":
    """GP correlation matrix at the viz grid, fitted on the PIT latent z_train instead of Kelvin.

    Returns None if fitting or factorization fails.
    """
    from eval.baselines.classical import fit_and_eval_gpytorch

    try:
        X_tr = torch.as_tensor(x_train_norm, dtype=torch.float32, device=device)
        X_te = torch.as_tensor(x_test_norm, dtype=torch.float32, device=device)
        z_tr = torch.as_tensor(z_train, dtype=torch.float32, device=device)
        fit = fit_and_eval_gpytorch(
            X_tr, z_tr, X_te, kernel_name, n_steps=n_steps, lr=lr,
            oracle_mode="posterior", n_restarts=n_restarts,
        )
        R = fit["R"].double().cpu().numpy()
        return {"L": safe_cholesky(R), "kernel": kernel_name}
    except Exception as exc:  # noqa: BLE001
        print(f"  [era5_viz_gp_z:{kernel_name}] fit failed, dropping GP-on-z row: {exc}")
        return None


def _era5_viz_gp_field(gp: dict, z_shared: np.ndarray) -> np.ndarray:
    """One draw (D,) from the fitted GP posterior using the shared white noise z_shared."""
    return gp["mean"] + gp["L"] @ z_shared


def _era5_viz_gp_correlation(gp: dict) -> np.ndarray:
    """Correlation matrix of the fitted GP posterior covariance."""
    Sigma = gp["L"] @ gp["L"].T
    std = np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    return Sigma / np.outer(std, std)


def _era5_viz_field(Sigma: np.ndarray, dist, y_mean: torch.Tensor, y_std: torch.Tensor, z_shared: np.ndarray, device: str) -> np.ndarray:
    """One draw (D,) from the copula model: y = F^{-1}(Phi(chol(Sigma) z_shared)).

    Uses the day's frozen marginal dist, or a Gaussian(mean, std) marginal when
    dist is None.
    """
    from scipy.stats import norm

    L = safe_cholesky(Sigma)
    z_copula = L @ z_shared
    if dist is None:
        return y_mean.cpu().numpy() + y_std.cpu().numpy() * z_copula
    u_copula = np.clip(norm.cdf(z_copula), 1e-6, 1.0 - 1e-6)
    u_t = torch.as_tensor(u_copula, dtype=torch.float32, device=device).unsqueeze(-1)
    with torch.no_grad():
        y_pred_scaled = dist.icdf(u_t).squeeze(-1).double()
    return (y_mean + y_std * y_pred_scaled).cpu().numpy()


def _era5_viz_fig(
    model: nn.Module, cfg: DictConfig, vb: dict, jitter: float, device: str,
) -> "tuple[plt.Figure | None, plt.Figure | None]":
    """Build val/era5_predictions and val/era5_residuals from the frozen viz probe.

    Rows: ground truth, fitted-GP sample, copula-model sample, independent
    sample, drawn with shared noise per day. The residual figure subtracts each
    row's own predictive mean. Returns (None, None) for a probe without days;
    the residual figure is None when a marginal mean is missing.
    """
    if not vb["days"]:
        return None, None
    x_train_v = torch.as_tensor(vb["x_train_norm"], dtype=torch.float32, device=device).unsqueeze(0)
    x_test_v = torch.as_tensor(vb["x_test_norm"], dtype=torch.float32, device=device).unsqueeze(0)
    rng_v = np.random.default_rng(vb["seed"])
    R_indep = np.eye(vb["D"])
    predicted_fields, gp_tabicl_fields, independent_fields, gp_fields = [], [], [], []
    predicted_resid, gp_tabicl_resid, independent_resid, gp_resid, true_resid = [], [], [], [], []
    gp_post = vb.get("gp_post_per_day") or [None] * len(vb["days"])
    marginal_mean = vb.get("marginal_mean_per_day") or [None] * len(vb["days"])
    gp_post_z = vb.get("gp_post_z_per_day") or [None] * len(vb["days"])
    for i in range(len(vb["days"])):
        z_train_v = torch.as_tensor(vb["z_train_per_day"][i], dtype=torch.float32, device=device).unsqueeze(0)
        out_v = model({"x_train": x_train_v, "z_train": z_train_v, "x_test": x_test_v})
        Sigma_v = build_sigma(out_v, cfg, jitter=jitter)[0].float().cpu().numpy()
        z_shared = rng_v.standard_normal(vb["D"])
        dist_i, y_mean_i, y_std_i = vb["dists_per_day"][i], vb["y_mean_per_day"][i], vb["y_std_per_day"][i]
        pred_field = _era5_viz_field(Sigma_v, dist_i, y_mean_i, y_std_i, z_shared, device)
        indep_field = _era5_viz_field(R_indep, dist_i, y_mean_i, y_std_i, z_shared, device)
        predicted_fields.append(pred_field)
        independent_fields.append(indep_field)
        mean_i = marginal_mean[i]
        if mean_i is not None:
            predicted_resid.append(pred_field - mean_i)
            independent_resid.append(indep_field - mean_i)
            true_resid.append(vb["true_fields"][i].ravel() - mean_i)
        if gp_post[i] is not None:
            gp_field = _era5_viz_gp_field(gp_post[i], z_shared)
            gp_fields.append(gp_field)
            if mean_i is not None:
                gp_resid.append(gp_field - gp_post[i]["mean"])
        # The z-space GP fit can fail independently of the raw-y fit.
        if gp_post_z[i] is not None:
            gp_tabicl_field = _era5_viz_field(
                _era5_viz_gp_correlation(gp_post_z[i]), dist_i, y_mean_i, y_std_i, z_shared, device,
            )
            gp_tabicl_fields.append(gp_tabicl_field)
            if mean_i is not None:
                gp_tabicl_resid.append(gp_tabicl_field - mean_i)
    # Drop the GP row if any day's fit failed (rows are matched to columns by position).
    oracle_fields = gp_fields if len(gp_fields) == len(vb["days"]) else None
    gp_tabicl_row = gp_tabicl_fields if len(gp_tabicl_fields) == len(vb["days"]) else None
    data_like = {"latitude": vb["lat"], "longitude": vb["lon"], "t2m": dict(zip(vb["days"], vb["true_fields"]))}
    fig_raw = plot_residual_grid(
        data_like, vb["days"], predicted_fields, output_path=None,
        context_coords=vb["context_coords"], independent_fields=independent_fields,
        oracle_fields=oracle_fields, predicted_fields_2=gp_tabicl_row,
        oracle_row_label=f"Fitted GP posterior\n({vb.get('gp_row_kernel', 'gp')})\nsample\nLatitude",
        pred2_row_label="Fitted GP correlation\n+ TabICLv2 marginal\nsample\nLatitude",
        target="raw",
    )
    fig_resid = None
    if len(true_resid) == len(vb["days"]):
        oracle_resid = gp_resid if len(gp_resid) == len(vb["days"]) else None
        gp_tabicl_resid_row = gp_tabicl_resid if len(gp_tabicl_resid) == len(vb["days"]) else None
        fig_resid = plot_mean_removed_grid(
            vb["lat"], vb["lon"], vb["grid_shape"], vb["days"], true_resid, output_path=None,
            predicted_fields=predicted_resid, predicted_fields_2=gp_tabicl_resid_row,
            independent_fields=independent_resid, oracle_fields=oracle_resid,
            context_coords=vb["context_coords"],
            oracle_row_label=f"Fitted GP posterior\n({vb.get('gp_row_kernel', 'gp')})\nsample minus\nGP mean\nLatitude",
        )
    return fig_raw, fig_resid


def _era5_z_samples_fig(
    model: nn.Module, cfg: DictConfig, vb: dict, jitter: float, device: str, n_samples: int = 3,
) -> "plt.Figure | None":
    """Build val/era5_predictions_z: n_samples draws of the latent z on the first viz day.

    Rows: independent (R = I), the copula model's Sigma, and the GP correlation
    fitted on z_train; each column shares one noise draw. Returns None when the
    probe has no days or the GP-on-z fit is missing.
    """
    if not vb["days"]:
        return None
    gp0 = (vb.get("gp_post_z_per_day") or [None])[0]
    if gp0 is None:
        return None
    day0 = vb["days"][0]
    x_train_v = torch.as_tensor(vb["x_train_norm"], dtype=torch.float32, device=device).unsqueeze(0)
    x_test_v = torch.as_tensor(vb["x_test_norm"], dtype=torch.float32, device=device).unsqueeze(0)
    z_train_v = torch.as_tensor(vb["z_train_per_day"][0], dtype=torch.float32, device=device).unsqueeze(0)
    out_v = model({"x_train": x_train_v, "z_train": z_train_v, "x_test": x_test_v})
    Sigma_v = build_sigma(out_v, cfg, jitter=jitter)[0].float().cpu().numpy()
    L_indep = np.eye(vb["D"])
    L_model = safe_cholesky(Sigma_v)
    L_gp = safe_cholesky(_era5_viz_gp_correlation(gp0))

    rng_s = np.random.default_rng(vb["seed"] + 1)
    independent_fields, predicted_fields, gp_fields = [], [], []
    for _ in range(n_samples):
        z_shared = rng_s.standard_normal(vb["D"])
        independent_fields.append(L_indep @ z_shared)
        predicted_fields.append(L_model @ z_shared)
        gp_fields.append(L_gp @ z_shared)

    return plot_z_predictor_samples(
        vb["lat"], vb["lon"], vb["grid_shape"], day0,
        independent_fields, predicted_fields, gp_fields,
        output_path=None, context_coords=vb["context_coords"],
    )


def _era5_marginal_variance_fig(vb: dict) -> "plt.Figure | None":
    """Build val/era5_marginal_variance: per-location predictive variance of the TabICL marginal and the fitted GP.

    Returns None when the probe has no days or a marginal variance is missing;
    the GP row is dropped if any day's GP fit is missing.
    """
    var_fields = vb.get("marginal_var_per_day") or []
    if not vb["days"] or len(var_fields) != len(vb["days"]) or any(v is None for v in var_fields):
        return None
    gp_post = vb.get("gp_post_per_day") or [None] * len(vb["days"])
    gp_var_fields = [np.sum(gp["L"] ** 2, axis=1) for gp in gp_post if gp is not None]
    if len(gp_var_fields) != len(vb["days"]):
        gp_var_fields = None
    return plot_marginal_variance_grid(
        vb["lat"], vb["lon"], vb["grid_shape"], vb["days"], var_fields,
        gp_var_fields=gp_var_fields,
        output_path=None, context_coords=vb["context_coords"],
        gp_row_label=f"Fitted GP\nposterior\n({vb.get('gp_row_kernel', 'gp')})",
    )
