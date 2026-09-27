"""Spatial-correlation diagnostics CLI.

Subcommands, chosen with command=<name> (settings are command.<key>=<value>):
    diagnose: correlation-vs-distance, heatmaps and field panels for one
        real-ERA5 or synthetic config, for one or more checkpoints.
    sweep: scalar and curve metrics for every (checkpoint, config) in a
        profile, written to eval/results/*.json (every family by default).
    baseline: theoretical-law fits to the ground truth for the same profiles.
    report: figures in eval/reports/figures/ from eval/results/*.json.
    all (default): sweep, baseline and report, real and synthetic.

Usage:
    python -m eval.runners.spatial_correlation_eval
    python -m eval.runners.spatial_correlation_eval command=diagnose command.ckpt=kernel-sweep-all-tabicl-retrain-15k
    python -m eval.runners.spatial_correlation_eval command=sweep command.mode=synthetic
    python -m eval.runners.spatial_correlation_eval command=diagnose --cfg job   # every diagnose key
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, fields
from typing import Any, Callable

import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

from eval.configs import (
    constants,
    regions,
)
from eval.configs.checkpoints import CHECKPOINT_FAMILIES, all_family_names, resolve_checkpoint
from eval.data.era5_io import haversine_distance_km, load_era5_data, safe_cholesky
from eval.data.fetch_era5 import fetch as fetch_era5
from eval.runners.hydra_cli import check_choice, check_choices, hydra_entry
from eval.spatial.diagnostics import (
    bin_correlation_by_distance,
    empirical_spatial_correlation,
    extract_model_context_correlation,
    extract_model_dummy_context_correlation,
    extract_model_true_z_train_correlation,
    fit_theoretical_law,
    pair_counts_by_distance,
    pool_yspace_samples_and_correlate,
    predict_copula_residual_field,
    sample_copula_residual_fields,
    sample_simple_kernel_covariance,
)
from eval.spatial.sweep_core import get_model, run_real_config, run_synthetic_config
from eval.viz.correlation_plots import (
    plot_correlation_heatmaps,
    plot_correlation_vs_distance,
    plot_residual_grid,
    plot_synthetic_residual_grid,
)

_RESULTS_DIR = os.path.join(_REPO_ROOT, "eval", "results")
_FIGURES_DIR = os.path.join(_REPO_ROOT, "eval", "reports", "figures")
_DIAGNOSE_DIR = os.path.join(_REPO_ROOT, "eval", "reports", "diagnose")


def _safe_ckpt_tag(token: str) -> str:
    """Filesystem-safe tag for a checkpoint name or path (no '/')."""
    if os.sep in token or (os.altsep and os.altsep in token):
        run_dir = os.path.basename(os.path.dirname(os.path.abspath(token)))
        step_name = os.path.splitext(os.path.basename(token))[0]
        return f"{run_dir}_{step_name}"
    return token


def _diagnose_real(
    ckpt_token: str,
    region: str,
    grid_size: int,
    n_days: int,
    n_context: int,
    device: "str | None",
    seed: int,
    out_dir: str,
) -> None:
    ckpt = resolve_checkpoint(ckpt_token)
    model, cfg, resolved_device, marginal = get_model(ckpt, device)
    lat_bounds, lon_bounds = regions.REGIONS[region]
    nc_path = fetch_era5(region, lat_bounds, lon_bounds, grid_size, n_days)
    data = load_era5_data(nc_path)
    lat, lon = data["latitude"], data["longitude"]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    D = int(coords.shape[0])

    R_emp = empirical_spatial_correlation(data, target="raw")
    R_dummy = extract_model_dummy_context_correlation(model, resolved_device, coords)
    R_indep = np.eye(D)

    rng = np.random.default_rng(seed)
    n_time = data["t2m"].shape[0]
    days = sorted(set(np.linspace(0, n_time - 1, min(6, n_time)).round().astype(int).tolist()))

    n_context_eff = max(1, min(n_context, D - 1))
    context_idx = rng.choice(D, size=n_context_eff, replace=False)
    context_coords = coords[context_idx]

    predicted_fields, independent_fields = [], []
    model_yspace_samples, dummy_yspace_samples = [], []
    for d in days:
        context_values = data["t2m"][d].ravel()[context_idx]
        R_context = extract_model_context_correlation(
            model,
            resolved_device,
            marginal,
            context_coords,
            context_values,
            coords,
            k_folds=constants.PIT_K_FOLDS,
        )
        z_shared = rng.standard_normal(D)
        predicted_fields.append(
            predict_copula_residual_field(
                marginal, context_coords, context_values, coords, R_context, resolved_device, z_shared
            )
        )
        independent_fields.append(
            predict_copula_residual_field(
                marginal, context_coords, context_values, coords, R_indep, resolved_device, z_shared
            )
        )

        # Y-space correlation from pooled samples, with the context-conditioned and
        # the unconditional correlation under the same context and marginal.
        z_batch = rng.standard_normal((constants.N_YSPACE_MC_SAMPLES, D))
        model_yspace_samples.append(
            sample_copula_residual_fields(
                marginal, context_coords, context_values, coords, R_context, resolved_device, z_batch
            )
        )
        dummy_yspace_samples.append(
            sample_copula_residual_fields(
                marginal, context_coords, context_values, coords, R_dummy, resolved_device, z_batch
            )
        )
    R_model_yspace = pool_yspace_samples_and_correlate(model_yspace_samples)
    R_dummy_yspace = pool_yspace_samples_and_correlate(dummy_yspace_samples)

    tag = f"{_safe_ckpt_tag(ckpt_token)}_real_{region}_g{grid_size}"
    dist = haversine_distance_km(coords)
    iu = np.triu_indices_from(R_emp, k=1)
    series = {
        "ground_truth": (dist[iu], R_emp[iu]),
        "model_context": (dist[iu], R_model_yspace[iu]),
        "dummy_context": (dist[iu], R_dummy_yspace[iu]),
    }
    plot_correlation_vs_distance(
        series, os.path.join(out_dir, f"diagnose_distance_{tag}.png"), scatter_series="ground_truth"
    )
    plot_correlation_heatmaps(
        {"ground_truth": R_emp, "model_context": R_model_yspace, "dummy_context": R_dummy_yspace},
        os.path.join(out_dir, f"diagnose_heatmaps_{tag}.png"),
    )
    plot_residual_grid(
        data,
        days,
        predicted_fields,
        os.path.join(out_dir, f"diagnose_residual_grid_{tag}.png"),
        context_coords=context_coords,
        independent_fields=independent_fields,
        target="raw",
    )
    print(f"[diagnose real] {ckpt_token}: done ({tag})")


def _diagnose_synthetic(
    ckpt_token: str,
    kernel: "str | None",
    grid_size: int,
    n_context: int,
    n_draws: int,
    device: "str | None",
    seed: int,
    out_dir: str,
) -> None:
    from copula_inter.data_gen import sigma_to_correlation

    ckpt = resolve_checkpoint(ckpt_token)
    model, cfg, resolved_device, marginal = get_model(ckpt, device)
    rng = np.random.default_rng(seed)

    axis = np.linspace(-1.0, 1.0, grid_size)
    x_grid, y_grid = np.meshgrid(axis, axis)
    coords = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    D = int(coords.shape[0])

    true_cov, kernel_name = sample_simple_kernel_covariance(cfg, coords, kernel, seed)
    R_true_t, _ = sigma_to_correlation(torch.as_tensor(true_cov, dtype=torch.float64))
    R_true = R_true_t.numpy()

    n_context_eff = max(1, min(n_context, D - 1))
    context_idx = rng.choice(D, size=n_context_eff, replace=False)
    context_coords = coords[context_idx]
    K_ff_context = true_cov[np.ix_(context_idx, context_idx)]

    L = safe_cholesky(true_cov)
    R_indep = np.eye(D)
    R_pred_draws, true_fields = [], []
    predicted_fields_true_z, predicted_fields_tabicl_z = [], []
    independent_fields, oracle_marginal_fields = [], []
    for _ in range(n_draws):
        z_true = L @ rng.standard_normal(D)
        context_values = z_true[context_idx]
        R_pred_tabicl_z = extract_model_context_correlation(
            model,
            resolved_device,
            marginal,
            context_coords,
            context_values,
            coords,
            k_folds=constants.PIT_K_FOLDS,
        )
        R_pred_true_z = extract_model_true_z_train_correlation(
            model,
            resolved_device,
            context_coords,
            K_ff_context,
            context_values,
            coords,
        )
        R_pred_draws.append(R_pred_tabicl_z)
        true_fields.append(z_true)
        z_shared = rng.standard_normal(D)
        predicted_fields_true_z.append(
            predict_copula_residual_field(
                marginal, context_coords, context_values, coords, R_pred_true_z, resolved_device, z_shared
            )
        )
        predicted_fields_tabicl_z.append(
            predict_copula_residual_field(
                marginal, context_coords, context_values, coords, R_pred_tabicl_z, resolved_device, z_shared
            )
        )
        oracle_marginal_fields.append(
            predict_copula_residual_field(
                marginal, context_coords, context_values, coords, R_true, resolved_device, z_shared
            )
        )
        independent_fields.append(
            predict_copula_residual_field(
                marginal, context_coords, context_values, coords, R_indep, resolved_device, z_shared
            )
        )
    R_pred_mean = np.mean(R_pred_draws, axis=0)

    tag = f"{_safe_ckpt_tag(ckpt_token)}_synthetic_{kernel_name}_g{grid_size}"
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices_from(R_true, k=1)
    series = {"ground_truth": (dist[iu], R_true[iu]), "model_context": (dist[iu], R_pred_mean[iu])}
    plot_correlation_vs_distance(
        series, os.path.join(out_dir, f"diagnose_distance_{tag}.png"), scatter_series="ground_truth"
    )
    plot_correlation_heatmaps(
        {"ground_truth": R_true, "model_context": R_pred_mean},
        os.path.join(out_dir, f"diagnose_heatmaps_{tag}.png"),
    )
    grid_shape = (grid_size, grid_size)
    true_grids = [f.reshape(grid_shape) for f in true_fields]
    plot_synthetic_residual_grid(
        axis,
        axis,
        grid_shape,
        true_grids,
        predicted_fields_true_z,
        predicted_fields_tabicl_z,
        independent_fields,
        os.path.join(out_dir, f"diagnose_residual_grid_{tag}.png"),
        context_coords=context_coords,
        oracle_fields=oracle_marginal_fields,
    )
    print(f"[diagnose synthetic] {ckpt_token}: kernel={kernel_name}, done ({tag})")


def _diagnose(
    mode: str,
    ckpt_tokens: list,
    region: str,
    grid_size: int,
    kernel: "str | None",
    n_days: int,
    n_context: int,
    n_draws: int,
    device: "str | None",
    seed: int,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for token in ckpt_tokens:
        if mode == "real":
            _diagnose_real(token, region, grid_size, n_days, n_context, device, seed, out_dir)
        else:
            _diagnose_synthetic(token, kernel, grid_size, n_context, n_draws, device, seed, out_dir)


def cmd_diagnose(spec: DiagnoseSpec) -> None:
    tokens = _checkpoint_tokens(spec.ckpt)
    _diagnose(
        spec.mode,
        tokens,
        spec.region,
        spec.grid_size,
        spec.kernel,
        spec.n_days,
        spec.n_context,
        spec.n_draws,
        spec.device,
        spec.seed,
        spec.out_dir,
    )


def _sweep(
    mode: str,
    profile: str,
    checkpoints_arg: "str | None",
    n_context: int,
    n_days: int,
    n_draws: int,
    device: "str | None",
    seed: int,
    out_path: "str | None" = None,
    compute_gp_baseline: bool = True,
    gp_kernels: "list | None" = None,
    gp_n_steps_mle: int = constants.GP_N_STEPS_MLE,
    gp_lr_mle: float = constants.GP_LR_MLE,
    gp_n_restarts_mle: int = constants.GP_N_RESTARTS_MLE,
) -> str:
    tokens = (
        all_family_names()
        if checkpoints_arg is None or checkpoints_arg == "all"
        else [t.strip() for t in checkpoints_arg.split(",") if t.strip()]
    )

    profile_configs = regions.SWEEP_PROFILES[profile] if mode == "real" else constants.SYNTHETIC_SWEEP_PROFILES[profile]

    out_path = out_path or os.path.join(_RESULTS_DIR, f"sweep_{mode}_{profile}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results = []
    for token in tokens:
        ckpt = resolve_checkpoint(token)
        for config_name, axis_name, grid_size in profile_configs:
            if mode == "real":
                r = run_real_config(
                    ckpt,
                    config_name,
                    axis_name,
                    grid_size,
                    n_days=n_days,
                    device=device,
                    seed=seed,
                    n_context=n_context,
                    compute_gp_baseline=compute_gp_baseline,
                    gp_baseline_kernels=gp_kernels,
                    gp_n_steps_mle=gp_n_steps_mle,
                    gp_lr_mle=gp_lr_mle,
                    gp_n_restarts_mle=gp_n_restarts_mle,
                )
            else:
                r = run_synthetic_config(
                    ckpt,
                    config_name,
                    axis_name,
                    grid_size,
                    n_context=n_context,
                    n_draws=n_draws,
                    seed=seed,
                    device=device,
                )
            r["family"] = token
            results.append(r)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")
    return out_path


def cmd_sweep(spec: SweepSpec) -> None:
    _sweep(
        spec.mode,
        spec.profile,
        ",".join(_checkpoint_tokens(spec.checkpoints)),
        spec.n_context,
        spec.n_days,
        spec.n_draws,
        spec.device,
        spec.seed,
        spec.out,
        compute_gp_baseline=spec.gp_baseline,
        gp_kernels=spec.gp_kernels,
        gp_n_steps_mle=spec.gp_n_steps_mle,
        gp_lr_mle=spec.gp_lr_mle,
        gp_n_restarts_mle=spec.gp_n_restarts_mle,
    )


def _baseline(
    mode: str,
    profile: str,
    laws: list,
    n_days: int,
    ckpt_token: "str | None",
    device: "str | None",
    seed: int,
    out_path: "str | None" = None,
) -> str:
    out_path = out_path or os.path.join(_RESULTS_DIR, f"baseline_{mode}.json")
    out = {}

    if mode == "real":
        for config_name, region_name, grid_size in regions.SWEEP_PROFILES[profile]:
            lat_bounds, lon_bounds = regions.REGIONS[region_name]
            nc_path = fetch_era5(region_name, lat_bounds, lon_bounds, grid_size, n_days)
            data = load_era5_data(nc_path)
            lat, lon = data["latitude"], data["longitude"]
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

            R_emp = empirical_spatial_correlation(data, target="raw")
            dist = haversine_distance_km(coords)
            dist_iu = dist[np.triu_indices_from(dist, k=1)]
            max_dist = np.percentile(dist_iu, constants.MAX_DIST_PERCENTILE)
            bin_edges = np.linspace(0.0, max_dist, constants.N_BINS + 1)
            dist_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            pair_counts = pair_counts_by_distance(dist, bin_edges).astype(int)
            rho_emp = bin_correlation_by_distance(R_emp, dist, bin_edges)

            fits = {}
            for law in laws:
                fit = fit_theoretical_law(dist_centers, rho_emp, pair_counts, law)
                fits[law] = fit["r_squared"] if fit else None
                print(f"[{config_name}] {law}: r2={fits[law]}")
            out[config_name] = fits
    else:
        from copula_inter.data_gen import sigma_to_correlation

        ckpt = resolve_checkpoint(ckpt_token or all_family_names()[-1])
        _, cfg, _, _ = get_model(ckpt, device)
        for config_name, kernel_name, grid_size in constants.SYNTHETIC_SWEEP_PROFILES[profile]:
            axis = np.linspace(-1000.0, 1000.0, grid_size)
            x_grid, y_grid = np.meshgrid(axis, axis)
            coords = np.column_stack([x_grid.ravel(), y_grid.ravel()])
            D = int(coords.shape[0])

            true_cov, _ = sample_simple_kernel_covariance(cfg, coords, kernel_name, seed)
            R_true_t, _ = sigma_to_correlation(torch.as_tensor(true_cov, dtype=torch.float64))
            R_true = R_true_t.numpy()

            dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
            dist_iu = dist[np.triu_indices(D, k=1)]
            bin_edges = np.linspace(0.0, dist_iu.max(), constants.N_BINS + 1)
            dist_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            pair_counts = pair_counts_by_distance(dist, bin_edges).astype(int)
            rho_true = bin_correlation_by_distance(R_true, dist, bin_edges)

            fits = {}
            for law in laws:
                fit = fit_theoretical_law(dist_centers, rho_true, pair_counts, law)
                fits[law] = fit["r_squared"] if fit else None
                print(f"[{config_name} | true={kernel_name}] {law}: r2={fits[law]}")
            out[config_name] = fits

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")
    return out_path


def cmd_baseline(spec: LawBaselineSpec) -> None:
    _baseline(spec.mode, spec.profile, spec.laws, spec.n_days, spec.ckpt, spec.device, spec.seed, spec.out)


def _family_style(family_token: str) -> tuple:
    entry = CHECKPOINT_FAMILIES.get(family_token)
    if entry is not None:
        return entry["label"], entry["color"]
    return family_token, None


def _report_mode(results: list, mode: str, out_dir: str, baseline_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = list(dict.fromkeys(r["family"] for r in results))
    configs = list(dict.fromkeys(r["config"] for r in results))
    baselines = json.load(open(baseline_path)) if os.path.exists(baseline_path) else {}

    # --- bar chart: model_r2 per config, grouped by checkpoint family ---
    baseline_laws = list(constants.CURVE_FIT_LAWS) if baselines else []
    n_series = len(families) + len(baseline_laws)
    width = 0.8 / max(n_series, 1)
    x = np.arange(len(configs))

    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(configs)), 6))
    for i, fam in enumerate(families):
        label, color = _family_style(fam)
        lut = {r["config"]: r["model_r2"] for r in results if r["family"] == fam}
        vals = [lut.get(c, np.nan) for c in configs]
        ax.bar(x + (i - n_series / 2 + 0.5) * width, vals, width=width, label=label, color=color)
    for j, law in enumerate(baseline_laws):
        i = len(families) + j
        vals = [baselines.get(c, {}).get(law, np.nan) for c in configs]
        ax.bar(
            x + (i - n_series / 2 + 0.5) * width,
            vals,
            width=width,
            label=f"baseline: {law}",
            hatch="//",
            edgecolor="black",
            linewidth=0.5,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("model_r2 (vs. ground truth)")
    ax.set_title(f"Spatial-correlation model_r2 by config ({mode}) — curve-shape diagnostic, not a scoring rule")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    bar_path = os.path.join(out_dir, f"report_bar_{mode}.png")
    fig.savefig(bar_path, dpi=140)
    plt.close(fig)
    print(f"Saved {bar_path}")

    # Bar chart of total Y-space joint NLL per config (real mode only).
    if all("nll_total" in r for r in results):
        # GP-MLE baseline kernels present in the results (real mode).
        gp_kernels = sorted({k for r in results for k in r.get("gp_baseline_nll", {})})
        n_series_nll = len(families) + len(gp_kernels)
        width_nll = 0.8 / max(n_series_nll, 1)
        gp_lut_by_config = {r["config"]: r["gp_baseline_nll"] for r in results if r.get("gp_baseline_nll")}

        fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(configs)), 6))
        for i, fam in enumerate(families):
            label, color = _family_style(fam)
            lut = {r["config"]: r["nll_total"] for r in results if r["family"] == fam}
            vals = [lut.get(c, np.nan) for c in configs]
            ax.bar(x + (i - n_series_nll / 2 + 0.5) * width_nll, vals, width=width_nll, label=label, color=color)
        for j, kname in enumerate(gp_kernels):
            i = len(families) + j
            vals = [gp_lut_by_config.get(c, {}).get(kname, {}).get("total", np.nan) for c in configs]
            ax.bar(
                x + (i - n_series_nll / 2 + 0.5) * width_nll,
                vals,
                width=width_nll,
                label=f"GP-MLE: {kname}",
                hatch="//",
                edgecolor="black",
                linewidth=0.5,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("total NLL (marginal+copula, nats/pt) — lower is better")
        ax.set_title(f"Spatial total joint NLL by config ({mode})")
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        nll_bar_path = os.path.join(out_dir, f"report_bar_nll_{mode}.png")
        fig.savefig(nll_bar_path, dpi=140)
        plt.close(fig)
        print(f"Saved {nll_bar_path}")

    # Curve overlays per config: rho_emp and each family's rho_model_yspace, both raw-y correlations.
    gt_key = "rho_emp" if mode == "real" else "rho_true"
    pred_key = "rho_model_yspace" if mode == "real" else "rho_pred"
    for config_name in configs:
        recs = [r for r in results if r["config"] == config_name and "dist_centers" in r]
        if not recs:
            continue
        fig, ax = plt.subplots(figsize=(8, 5.5))
        gt_plotted = False
        for r in recs:
            d = np.array(r["dist_centers"])
            if not gt_plotted:
                ax.plot(
                    d, np.array(r[gt_key], dtype=float), color="black", linewidth=3.0, label="Ground truth", zorder=10
                )
                gt_plotted = True
            label, color = _family_style(r["family"])
            ax.plot(
                d,
                np.array(r[pred_key], dtype=float),
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=3.5,
                label=label,
            )
        ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
        ax.set_xlabel("Distance")
        ax.set_ylabel(r"Correlation $\rho$")
        ax.set_title(f"{config_name} ({mode})")
        ax.legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        curve_path = os.path.join(out_dir, f"report_curve_{mode}_{config_name}.png")
        fig.savefig(curve_path, dpi=140)
        plt.close(fig)
        print(f"Saved {curve_path}")


def _report(
    out_dir: "str | None",
    real_results: "str | None",
    synthetic_results: "str | None",
    baseline_real: "str | None",
    baseline_synthetic: "str | None",
) -> None:
    out_dir = out_dir or _FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)

    real_path = real_results or os.path.join(_RESULTS_DIR, "sweep_real_low_context_7config.json")
    synth_path = synthetic_results or os.path.join(_RESULTS_DIR, "sweep_synthetic_low_context_7config.json")
    baseline_real_path = baseline_real or os.path.join(_RESULTS_DIR, "baseline_real.json")
    baseline_synth_path = baseline_synthetic or os.path.join(_RESULTS_DIR, "baseline_synthetic.json")

    if os.path.exists(real_path):
        with open(real_path) as f:
            _report_mode(json.load(f), "real", out_dir, baseline_real_path)
    else:
        print(
            f"No real sweep results at {real_path}; skipping real-mode report figures "
            f"(run command=sweep command.mode=real first, or command=all)."
        )

    if os.path.exists(synth_path):
        with open(synth_path) as f:
            _report_mode(json.load(f), "synthetic", out_dir, baseline_synth_path)
    else:
        print(
            f"No synthetic sweep results at {synth_path}; skipping synthetic-mode report figures "
            f"(run command=sweep command.mode=synthetic first, or command=all)."
        )


def cmd_report(spec: ReportSpec) -> None:
    _report(spec.out_dir, spec.real_results, spec.synthetic_results, spec.baseline_real, spec.baseline_synthetic)


def cmd_all(spec: AllSpec) -> None:
    checkpoints = "all" if spec.checkpoints is None else ",".join(_checkpoint_tokens(spec.checkpoints))
    baseline_synthetic_ckpt = checkpoints.split(",")[-1] if checkpoints != "all" else all_family_names()[-1]
    print("=== [all] 1/5: sweep mode=real ===")
    _sweep(
        "real",
        "low_context_7config",
        checkpoints,
        constants.N_CONTEXT,
        constants.N_DAYS,
        constants.N_SYNTHETIC_DRAWS,
        spec.device,
        constants.SEED,
        compute_gp_baseline=spec.gp_baseline,
    )
    print("=== [all] 2/5: sweep mode=synthetic ===")
    _sweep(
        "synthetic",
        "low_context_7config",
        checkpoints,
        constants.N_CONTEXT,
        constants.N_DAYS,
        constants.N_SYNTHETIC_DRAWS,
        spec.device,
        constants.SEED,
    )
    print("=== [all] 3/5: baseline mode=real ===")
    _baseline(
        "real", "low_context_7config", constants.CURVE_FIT_LAWS, constants.N_DAYS, None, spec.device, constants.SEED
    )
    print("=== [all] 4/5: baseline mode=synthetic ===")
    _baseline(
        "synthetic",
        "low_context_7config",
        constants.CURVE_FIT_LAWS,
        constants.N_DAYS,
        baseline_synthetic_ckpt,
        spec.device,
        constants.SEED,
    )
    print("=== [all] 5/5: report ===")
    _report(None, None, None, None, None)
    print(f"=== [all] done. Figures in {_FIGURES_DIR} ===")


@dataclass
class DiagnoseSpec:
    """Correlation-vs-distance, heatmaps and field panels for one config."""

    # Checkpoint family name, family:step or .pt path, or a list of them ([a,b]) sharing one model cache.
    ckpt: Any = MISSING
    mode: str = "real"
    # [mode=real] Named region (eval/configs/regions.py).
    region: str = "western_europe"
    grid_size: int = 24
    # [mode=synthetic] Ground-truth kernel; random when null.
    kernel: str | None = None
    # [mode=real] Daily snapshots.
    n_days: int = constants.N_DAYS
    n_context: int = constants.N_CONTEXT
    # [mode=synthetic] GP draws averaged per config.
    n_draws: int = constants.N_SYNTHETIC_DRAWS
    device: str | None = None
    seed: int = constants.SEED
    out_dir: str = _DIAGNOSE_DIR


@dataclass
class SweepSpec:
    """Scalar and curve metrics for every (checkpoint, config) in a profile."""

    mode: str = "real"
    # regions.SWEEP_PROFILES (mode=real) or constants.SYNTHETIC_SWEEP_PROFILES (mode=synthetic).
    profile: str = "low_context_7config"
    # "all" (every eval/configs/checkpoints.py family), or one or a list ([a,b]) of names / family:step / .pt paths.
    checkpoints: Any = "all"
    n_context: int = constants.N_CONTEXT
    n_days: int = constants.N_DAYS
    n_draws: int = constants.N_SYNTHETIC_DRAWS
    device: str | None = None
    seed: int = constants.SEED
    # Default: eval/results/sweep_<mode>_<profile>.json.
    out: str | None = None
    # [mode=real] Fit the classical GP-MLE nll_total baseline with these kernels.
    gp_baseline: bool = True
    gp_kernels: list[str] = field(default_factory=lambda: list(constants.GP_BASELINE_KERNELS))
    gp_n_steps_mle: int = constants.GP_N_STEPS_MLE
    gp_lr_mle: float = constants.GP_LR_MLE
    gp_n_restarts_mle: int = constants.GP_N_RESTARTS_MLE


@dataclass
class LawBaselineSpec:
    """Theoretical-law curve fits to the ground truth, no learned model."""

    mode: str = "real"
    profile: str = "low_context_7config"
    laws: list[str] = field(default_factory=lambda: list(constants.CURVE_FIT_LAWS))
    n_days: int = constants.N_DAYS
    # [mode=synthetic] Checkpoint whose saved config supplies the kernel prior; default: the last family.
    ckpt: str | None = None
    device: str | None = None
    seed: int = constants.SEED
    # Default: eval/results/baseline_<mode>.json.
    out: str | None = None


@dataclass
class ReportSpec:
    """Figures from eval/results/*.json; each null path falls back to the default file name."""

    # Default: eval/reports/figures/.
    out_dir: str | None = None
    real_results: str | None = None
    synthetic_results: str | None = None
    baseline_real: str | None = None
    baseline_synthetic: str | None = None


@dataclass
class AllSpec:
    """sweep, baseline and report, real and synthetic, at the default profile."""

    device: str | None = None
    # "all" when null, else what sweep's checkpoints takes.
    checkpoints: Any = None
    # Fit the real-ERA5 classical GP-MLE nll_total baseline.
    gp_baseline: bool = True


@dataclass
class SpatialSpec:
    """One subcommand, chosen with command=<name>; its settings are command.<key>."""

    command: Any = MISSING


_COMMANDS: dict[str, tuple[type, Callable[[Any], None]]] = {
    "diagnose": (DiagnoseSpec, cmd_diagnose),
    "sweep": (SweepSpec, cmd_sweep),
    "baseline": (LawBaselineSpec, cmd_baseline),
    "report": (ReportSpec, cmd_report),
    "all": (AllSpec, cmd_all),
}
for _name, (_schema, _) in _COMMANDS.items():
    ConfigStore.instance().store(group="command", name=_name, node=_schema)


def _checkpoint_tokens(value: Any) -> list[str]:
    """A checkpoint setting (one token, a comma-separated string, or a list) as a list of tokens."""
    items = value.split(",") if isinstance(value, str) else [str(v) for v in value]
    return [t.strip() for t in items if t.strip()]


def validate_spatial_spec(spec: SpatialSpec) -> None:
    """Raise ValueError for a setting outside its allowed values."""
    cmd = spec.command
    device = getattr(cmd, "device", None)
    check_choice("command.device", device, ("cpu", "cuda"))
    if isinstance(cmd, (DiagnoseSpec, SweepSpec, LawBaselineSpec)):
        check_choice("command.mode", cmd.mode, ("real", "synthetic"))
    if isinstance(cmd, DiagnoseSpec):
        check_choice("command.region", cmd.region, regions.REGIONS)
        check_choice("command.kernel", cmd.kernel, constants.SYNTHETIC_SWEEP_KERNELS)
    if isinstance(cmd, SweepSpec):
        check_choices("command.gp_kernels", cmd.gp_kernels, constants.GP_BASELINE_KERNELS)
    if isinstance(cmd, LawBaselineSpec):
        check_choices("command.laws", cmd.laws, constants.CURVE_FIT_LAWS)


def run(spec: SpatialSpec) -> None:
    for schema, cmd in _COMMANDS.values():
        if isinstance(spec.command, schema):
            cmd(spec.command)
            return
    raise TypeError(f"unknown command config {type(spec.command).__name__}")


# Every retired argparse flag maps to the same name under command.
_FLAG_ALIASES = {f.name: f"command.{f.name}" for schema, _ in _COMMANDS.values() for f in fields(schema)} | {
    "no_gp_baseline": "command.gp_baseline=false"
}

_hydra_main = hydra_entry("spatial_correlation_eval", SpatialSpec, run, _FLAG_ALIASES, validate=validate_spatial_spec)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in _COMMANDS:
        raise SystemExit(
            f"spatial_correlation_eval: subcommands are now an override: command={sys.argv[1]} "
            "(its settings are command.<key>=<value>)"
        )
    _hydra_main()


if __name__ == "__main__":
    main()
