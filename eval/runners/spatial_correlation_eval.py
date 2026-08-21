"""spatial_correlation_eval.py — single CLI entrypoint for the whole
spatial-correlation diagnostic/sweep/baseline/report toolchain, replacing
the 22 ad-hoc scripts that used to live in plots/.

Subcommands:
  diagnose  — single/multi-checkpoint deep-dive plots for ONE (region,
              grid_size) real-ERA5 config or ONE synthetic kernel config:
              distance-vs-correlation, correlation heatmaps, and a
              small-multiples residual-field panel. Loops over --ckpt
              in-process (shared model cache), auto-fetches/caches ERA5.
  sweep     — batch scalar+curve metrics over every (checkpoint, config) in
              a named profile (eval/configs/regions.py /
              eval/configs/constants.py), persisted to eval/results/*.json.
              --checkpoints all (default) auto-discovers every registered
              checkpoint family.
  baseline  — direct (non-learned) theoretical-law curve fits against
              ground truth, for the same profiles `sweep` uses. No model/GPU.
  report    — reads eval/results/*.json + eval/configs/checkpoints.py labels
              /colors, writes bar-chart + curve-overlay figures to
              eval/reports/figures/.
  all       — chains sweep (real+synthetic) -> baseline (real+synthetic) ->
              report in one process, using every default above: the
              minimal-intervention, single-command entrypoint.

Usage:
    python eval/runners/spatial_correlation_eval.py all
    python eval/runners/spatial_correlation_eval.py diagnose --ckpt kernel-sweep-all-tabicl-retrain-15k
    python eval/runners/spatial_correlation_eval.py sweep --mode synthetic --checkpoints all
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
from eval.configs import regions  # noqa: E402
from eval.configs.checkpoints import CHECKPOINT_FAMILIES, all_family_names, resolve_checkpoint  # noqa: E402
from eval.data.era5_io import haversine_distance_km, load_era5_data, safe_cholesky  # noqa: E402
from eval.data.fetch_era5 import fetch as fetch_era5  # noqa: E402
from eval.spatial.diagnostics import (  # noqa: E402
    bin_correlation_by_distance,
    empirical_spatial_correlation,
    extract_model_context_correlation,
    extract_model_dummy_context_correlation,
    extract_model_true_z_train_correlation,
    fit_theoretical_law,
    pair_counts_by_distance,
    predict_copula_residual_field,
    sample_simple_kernel_covariance,
)
from eval.spatial.sweep_core import get_model, run_real_config, run_synthetic_config  # noqa: E402
from eval.viz.correlation_plots import (  # noqa: E402
    plot_correlation_heatmaps,
    plot_correlation_vs_distance,
    plot_residual_grid,
    plot_synthetic_residual_grid,
)

_RESULTS_DIR = os.path.join(_REPO_ROOT, "eval", "results")
_FIGURES_DIR = os.path.join(_REPO_ROOT, "eval", "reports", "figures")
_DIAGNOSE_DIR = os.path.join(_REPO_ROOT, "eval", "reports", "diagnose")


def _safe_ckpt_tag(token: str) -> str:
    """Filesystem-safe identifier for `token` (a checkpoint family name or a
    raw .pt path) used in output filenames — a raw path's '/' would
    otherwise be interpreted as directory separators and create bogus
    nested directories instead of a flat file."""
    if os.sep in token or (os.altsep and os.altsep in token):
        run_dir = os.path.basename(os.path.dirname(os.path.abspath(token)))
        step_name = os.path.splitext(os.path.basename(token))[0]
        return f"{run_dir}_{step_name}"
    return token


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------
def _diagnose_real(ckpt_token: str, region: str, grid_size: int, n_days: int, n_context: int,
                    device: "str | None", seed: int, out_dir: str) -> None:
    ckpt = resolve_checkpoint(ckpt_token)
    model, cfg, resolved_device, marginal = get_model(ckpt, device)
    lat_bounds, lon_bounds = regions.REGIONS[region]
    nc_path = fetch_era5(region, lat_bounds, lon_bounds, grid_size, n_days)
    data = load_era5_data(nc_path)
    lat, lon = data["latitude"], data["longitude"]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    D = coords.shape[0]

    R_emp = empirical_spatial_correlation(data, target="raw")
    R_dummy = extract_model_dummy_context_correlation(model, resolved_device, coords)
    R_indep = np.eye(D)

    rng = np.random.default_rng(seed)
    n_time = data["t2m"].shape[0]
    days = sorted(set(np.linspace(0, n_time - 1, min(6, n_time)).round().astype(int).tolist()))

    n_context_eff = max(1, min(n_context, D - 1))
    context_idx = rng.choice(D, size=n_context_eff, replace=False)
    context_coords = coords[context_idx]

    R_context_per_day, predicted_fields, independent_fields = [], [], []
    for d in days:
        context_values = data["t2m"][d].ravel()[context_idx]
        R_context = extract_model_context_correlation(
            model, resolved_device, marginal, context_coords, context_values, coords, k_folds=constants.PIT_K_FOLDS,
        )
        R_context_per_day.append(R_context)
        z_shared = rng.standard_normal(D)
        predicted_fields.append(
            predict_copula_residual_field(marginal, context_coords, context_values, coords, R_context, resolved_device, z_shared)
        )
        independent_fields.append(
            predict_copula_residual_field(marginal, context_coords, context_values, coords, R_indep, resolved_device, z_shared)
        )
    R_context_mean = np.mean(R_context_per_day, axis=0)

    tag = f"{_safe_ckpt_tag(ckpt_token)}_real_{region}_g{grid_size}"
    dist = haversine_distance_km(coords)
    iu = np.triu_indices_from(R_emp, k=1)
    series = {
        "ground_truth": (dist[iu], R_emp[iu]),
        "model_context": (dist[iu], R_context_mean[iu]),
        "dummy_context": (dist[iu], R_dummy[iu]),
    }
    plot_correlation_vs_distance(series, os.path.join(out_dir, f"diagnose_distance_{tag}.png"), scatter_series="ground_truth")
    plot_correlation_heatmaps(
        {"ground_truth": R_emp, "model_context": R_context_mean, "dummy_context": R_dummy},
        os.path.join(out_dir, f"diagnose_heatmaps_{tag}.png"),
    )
    plot_residual_grid(
        data, days, predicted_fields, os.path.join(out_dir, f"diagnose_residual_grid_{tag}.png"),
        context_coords=context_coords, independent_fields=independent_fields, target="raw",
    )
    print(f"[diagnose real] {ckpt_token}: done ({tag})")


def _diagnose_synthetic(ckpt_token: str, kernel: "str | None", grid_size: int, n_context: int, n_draws: int,
                         device: "str | None", seed: int, out_dir: str) -> None:
    from data_gen import sigma_to_correlation

    ckpt = resolve_checkpoint(ckpt_token)
    model, cfg, resolved_device, marginal = get_model(ckpt, device)
    rng = np.random.default_rng(seed)

    axis = np.linspace(-1.0, 1.0, grid_size)
    x_grid, y_grid = np.meshgrid(axis, axis)
    coords = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    D = coords.shape[0]

    true_cov, kernel_name = sample_simple_kernel_covariance(cfg, coords, kernel, seed)
    R_true, _ = sigma_to_correlation(torch.as_tensor(true_cov, dtype=torch.float64))
    R_true = R_true.numpy()

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
            model, resolved_device, marginal, context_coords, context_values, coords, k_folds=constants.PIT_K_FOLDS,
        )
        R_pred_true_z = extract_model_true_z_train_correlation(
            model, resolved_device, context_coords, K_ff_context, context_values, coords,
        )
        R_pred_draws.append(R_pred_tabicl_z)
        true_fields.append(z_true)
        z_shared = rng.standard_normal(D)
        predicted_fields_true_z.append(
            predict_copula_residual_field(marginal, context_coords, context_values, coords, R_pred_true_z, resolved_device, z_shared)
        )
        predicted_fields_tabicl_z.append(
            predict_copula_residual_field(marginal, context_coords, context_values, coords, R_pred_tabicl_z, resolved_device, z_shared)
        )
        oracle_marginal_fields.append(
            predict_copula_residual_field(marginal, context_coords, context_values, coords, R_true, resolved_device, z_shared)
        )
        independent_fields.append(
            predict_copula_residual_field(marginal, context_coords, context_values, coords, R_indep, resolved_device, z_shared)
        )
    R_pred_mean = np.mean(R_pred_draws, axis=0)

    tag = f"{_safe_ckpt_tag(ckpt_token)}_synthetic_{kernel_name}_g{grid_size}"
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices_from(R_true, k=1)
    series = {"ground_truth": (dist[iu], R_true[iu]), "model_context": (dist[iu], R_pred_mean[iu])}
    plot_correlation_vs_distance(series, os.path.join(out_dir, f"diagnose_distance_{tag}.png"), scatter_series="ground_truth")
    plot_correlation_heatmaps(
        {"ground_truth": R_true, "model_context": R_pred_mean}, os.path.join(out_dir, f"diagnose_heatmaps_{tag}.png"),
    )
    grid_shape = (grid_size, grid_size)
    true_grids = [f.reshape(grid_shape) for f in true_fields]
    plot_synthetic_residual_grid(
        axis, axis, grid_shape, true_grids, predicted_fields_true_z, predicted_fields_tabicl_z, independent_fields,
        os.path.join(out_dir, f"diagnose_residual_grid_{tag}.png"),
        context_coords=context_coords, oracle_fields=oracle_marginal_fields,
    )
    print(f"[diagnose synthetic] {ckpt_token}: kernel={kernel_name}, done ({tag})")


def _diagnose(mode: str, ckpt_tokens: list, region: str, grid_size: int, kernel: "str | None",
              n_days: int, n_context: int, n_draws: int, device: "str | None", seed: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for token in ckpt_tokens:
        if mode == "real":
            _diagnose_real(token, region, grid_size, n_days, n_context, device, seed, out_dir)
        else:
            _diagnose_synthetic(token, kernel, grid_size, n_context, n_draws, device, seed, out_dir)


def cmd_diagnose(args) -> None:
    tokens = [t.strip() for t in args.ckpt.split(",") if t.strip()]
    _diagnose(
        args.mode, tokens, args.region, args.grid_size, args.kernel,
        args.n_days, args.n_context, args.n_draws, args.device, args.seed, args.out_dir,
    )


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def _sweep(mode: str, profile: str, checkpoints_arg: "str | None", n_context: int, n_days: int,
           n_draws: int, device: "str | None", seed: int, out_path: "str | None" = None,
           compute_gp_baseline: bool = True, gp_kernels: "list | None" = None,
           gp_n_steps_mle: int = constants.GP_N_STEPS_MLE, gp_lr_mle: float = constants.GP_LR_MLE,
           gp_n_restarts_mle: int = constants.GP_N_RESTARTS_MLE) -> str:
    tokens = all_family_names() if (checkpoints_arg in (None, "all")) else \
        [t.strip() for t in checkpoints_arg.split(",") if t.strip()]

    profile_configs = regions.SWEEP_PROFILES[profile] if mode == "real" else constants.SYNTHETIC_SWEEP_PROFILES[profile]

    out_path = out_path or os.path.join(_RESULTS_DIR, f"sweep_{mode}_{profile}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results = []
    for token in tokens:
        ckpt = resolve_checkpoint(token)
        for config_name, axis_name, grid_size in profile_configs:
            if mode == "real":
                r = run_real_config(
                    ckpt, config_name, axis_name, grid_size, n_days=n_days, device=device, seed=seed, n_context=n_context,
                    compute_gp_baseline=compute_gp_baseline, gp_baseline_kernels=gp_kernels,
                    gp_n_steps_mle=gp_n_steps_mle, gp_lr_mle=gp_lr_mle, gp_n_restarts_mle=gp_n_restarts_mle,
                )
            else:
                r = run_synthetic_config(
                    ckpt, config_name, axis_name, grid_size, n_context=n_context, n_draws=n_draws, seed=seed, device=device,
                )
            r["family"] = token
            results.append(r)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")
    return out_path


def cmd_sweep(args) -> None:
    _sweep(args.mode, args.profile, args.checkpoints, args.n_context, args.n_days, args.n_draws, args.device, args.seed, args.out,
           compute_gp_baseline=not args.no_gp_baseline, gp_kernels=args.gp_kernels,
           gp_n_steps_mle=args.gp_n_steps_mle, gp_lr_mle=args.gp_lr_mle, gp_n_restarts_mle=args.gp_n_restarts_mle)


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------
def _baseline(mode: str, profile: str, laws: list, n_days: int, ckpt_token: "str | None",
              device: "str | None", seed: int, out_path: "str | None" = None) -> str:
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
        from data_gen import sigma_to_correlation

        ckpt = resolve_checkpoint(ckpt_token or all_family_names()[-1])
        _, cfg, _, _ = get_model(ckpt, device)
        for config_name, kernel_name, grid_size in constants.SYNTHETIC_SWEEP_PROFILES[profile]:
            axis = np.linspace(-1000.0, 1000.0, grid_size)
            x_grid, y_grid = np.meshgrid(axis, axis)
            coords = np.column_stack([x_grid.ravel(), y_grid.ravel()])
            D = coords.shape[0]

            true_cov, _ = sample_simple_kernel_covariance(cfg, coords, kernel_name, seed)
            R_true, _ = sigma_to_correlation(torch.as_tensor(true_cov, dtype=torch.float64))
            R_true = R_true.numpy()

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


def cmd_baseline(args) -> None:
    _baseline(args.mode, args.profile, args.laws, args.n_days, args.ckpt, args.device, args.seed, args.out)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
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
        ax.bar(x + (i - n_series / 2 + 0.5) * width, vals, width=width, label=f"baseline: {law}",
               hatch="//", edgecolor="black", linewidth=0.5)
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

    # --- bar chart: total (marginal+copula) joint NLL per config, grouped by
    # family — only present for mode="real" results (sweep_core.py::
    # run_real_config); model_r2 above is a binned correlation-curve-shape
    # diagnostic and not a proper scoring rule, so it can't answer "how many
    # nats worse is the real predictive density" the way this can. ---
    if all("nll_total" in r for r in results):
        # GP-MLE baseline kernels present in the results (real-ERA5 mode
        # only, sweep_core.py::run_real_config's "gp_baseline_nll" field —
        # same value for every family/config row thanks to
        # _fit_gp_baseline_nll's cross-checkpoint cache, so any one matching
        # row's fit works as the lookup source).
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
            ax.bar(x + (i - n_series_nll / 2 + 0.5) * width_nll, vals, width=width_nll,
                   label=f"GP-MLE: {kname}", hatch="//", edgecolor="black", linewidth=0.5)
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

    # --- curve overlays: ground truth vs. every family's predicted curve, one figure per config ---
    gt_key = "rho_emp" if mode == "real" else "rho_true"
    pred_key = "rho_context_mean" if mode == "real" else "rho_pred"
    for config_name in configs:
        recs = [r for r in results if r["config"] == config_name and "dist_centers" in r]
        if not recs:
            continue
        fig, ax = plt.subplots(figsize=(8, 5.5))
        gt_plotted = False
        for r in recs:
            d = np.array(r["dist_centers"])
            if not gt_plotted:
                ax.plot(d, np.array(r[gt_key], dtype=float), color="black", linewidth=3.0, label="Ground truth", zorder=10)
                gt_plotted = True
            label, color = _family_style(r["family"])
            ax.plot(d, np.array(r[pred_key], dtype=float), color=color, linewidth=2.0, marker="o", markersize=3.5, label=label)
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


def _report(out_dir: "str | None", real_results: "str | None", synthetic_results: "str | None",
            baseline_real: "str | None", baseline_synthetic: "str | None") -> None:
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
        print(f"No real sweep results at {real_path}; skipping real-mode report figures "
              f"(run `sweep --mode real` first, or `all`).")

    if os.path.exists(synth_path):
        with open(synth_path) as f:
            _report_mode(json.load(f), "synthetic", out_dir, baseline_synth_path)
    else:
        print(f"No synthetic sweep results at {synth_path}; skipping synthetic-mode report figures "
              f"(run `sweep --mode synthetic` first, or `all`).")


def cmd_report(args) -> None:
    _report(args.out_dir, args.real_results, args.synthetic_results, args.baseline_real, args.baseline_synthetic)


# ---------------------------------------------------------------------------
# all — the minimal-intervention entrypoint
# ---------------------------------------------------------------------------
def cmd_all(args) -> None:
    checkpoints = args.checkpoints or "all"
    baseline_synthetic_ckpt = (
        [t.strip() for t in checkpoints.split(",") if t.strip()][-1]
        if checkpoints != "all" else all_family_names()[-1]
    )
    print("=== [all] 1/5: sweep --mode real ===")
    _sweep("real", "low_context_7config", checkpoints, constants.N_CONTEXT, constants.N_DAYS,
           constants.N_SYNTHETIC_DRAWS, args.device, constants.SEED,
           compute_gp_baseline=not args.no_gp_baseline)
    print("=== [all] 2/5: sweep --mode synthetic ===")
    _sweep("synthetic", "low_context_7config", checkpoints, constants.N_CONTEXT, constants.N_DAYS,
           constants.N_SYNTHETIC_DRAWS, args.device, constants.SEED)
    print("=== [all] 3/5: baseline --mode real ===")
    _baseline("real", "low_context_7config", constants.CURVE_FIT_LAWS, constants.N_DAYS, None,
              args.device, constants.SEED)
    print("=== [all] 4/5: baseline --mode synthetic ===")
    _baseline("synthetic", "low_context_7config", constants.CURVE_FIT_LAWS, constants.N_DAYS,
              baseline_synthetic_ckpt, args.device, constants.SEED)
    print("=== [all] 5/5: report ===")
    _report(None, None, None, None, None)
    print(f"=== [all] done. Figures in {_FIGURES_DIR} ===")


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_diag = sub.add_parser("diagnose", help="Single/multi-checkpoint deep-dive plots for one config.")
    p_diag.add_argument("--ckpt", type=str, required=True,
                         help="Comma-separated list of checkpoint family names (see eval/configs/checkpoints.py), "
                              "'family:step', or raw .pt paths — looped over in-process, sharing the model cache.")
    p_diag.add_argument("--mode", choices=["real", "synthetic"], default="real")
    p_diag.add_argument("--region", choices=list(regions.REGIONS), default="western_europe",
                         help="[--mode real] Named region (eval/configs/regions.py).")
    p_diag.add_argument("--grid-size", type=int, default=24)
    p_diag.add_argument("--kernel", choices=constants.SYNTHETIC_SWEEP_KERNELS, default=None,
                         help="[--mode synthetic] Ground-truth kernel; random if omitted.")
    p_diag.add_argument("--n-days", type=int, default=constants.N_DAYS, help="[--mode real]")
    p_diag.add_argument("--n-context", type=int, default=constants.N_CONTEXT)
    p_diag.add_argument("--n-draws", type=int, default=constants.N_SYNTHETIC_DRAWS, help="[--mode synthetic]")
    p_diag.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p_diag.add_argument("--seed", type=int, default=constants.SEED)
    p_diag.add_argument("--out-dir", type=str, default=_DIAGNOSE_DIR)
    p_diag.set_defaults(func=cmd_diagnose)

    p_sweep = sub.add_parser("sweep", help="Batch scalar+curve metrics over a named config profile.")
    p_sweep.add_argument("--mode", choices=["real", "synthetic"], default="real")
    p_sweep.add_argument("--profile", type=str, default="low_context_7config",
                          help="Named config list — see eval/configs/regions.py:SWEEP_PROFILES "
                               "(--mode real) / eval/configs/constants.py:SYNTHETIC_SWEEP_PROFILES (--mode synthetic).")
    p_sweep.add_argument("--checkpoints", type=str, default="all",
                          help="'all' (default, auto-discovers every eval/configs/checkpoints.py family), or a "
                               "comma-separated list of family names / 'family:step' / raw .pt paths.")
    p_sweep.add_argument("--n-context", type=int, default=constants.N_CONTEXT)
    p_sweep.add_argument("--n-days", type=int, default=constants.N_DAYS, help="[--mode real]")
    p_sweep.add_argument("--n-draws", type=int, default=constants.N_SYNTHETIC_DRAWS, help="[--mode synthetic]")
    p_sweep.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p_sweep.add_argument("--seed", type=int, default=constants.SEED)
    p_sweep.add_argument("--out", type=str, default=None, help="Default: eval/results/sweep_<mode>_<profile>.json")
    p_sweep.add_argument("--gp-kernels", type=str, nargs="+", default=constants.GP_BASELINE_KERNELS,
                          choices=constants.GP_BASELINE_KERNELS,
                          help="[--mode real] Classical-GP-MLE kernels to fit as a real-ERA5 nll_total "
                               "baseline (eval/spatial/sweep_core.py::_fit_gp_baseline_nll).")
    p_sweep.add_argument("--gp-n-steps-mle", type=int, default=constants.GP_N_STEPS_MLE)
    p_sweep.add_argument("--gp-lr-mle", type=float, default=constants.GP_LR_MLE)
    p_sweep.add_argument("--gp-n-restarts-mle", type=int, default=constants.GP_N_RESTARTS_MLE)
    p_sweep.add_argument("--no-gp-baseline", action="store_true",
                          help="[--mode real] Skip the classical-GP-MLE baseline nll_total fit.")
    p_sweep.set_defaults(func=cmd_sweep)

    p_base = sub.add_parser("baseline", help="Direct (non-learned) theoretical-law curve fits against ground truth.")
    p_base.add_argument("--mode", choices=["real", "synthetic"], default="real")
    p_base.add_argument("--profile", type=str, default="low_context_7config")
    p_base.add_argument("--laws", type=str, nargs="+", default=constants.CURVE_FIT_LAWS, choices=list(constants.CURVE_FIT_LAWS))
    p_base.add_argument("--n-days", type=int, default=constants.N_DAYS, help="[--mode real]")
    p_base.add_argument("--ckpt", type=str, default=None,
                         help="[--mode synthetic] Checkpoint to source data_gen.py kernel-prior cfg from "
                              "(no model forward pass) — default: the last registered family.")
    p_base.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p_base.add_argument("--seed", type=int, default=constants.SEED)
    p_base.add_argument("--out", type=str, default=None, help="Default: eval/results/baseline_<mode>.json")
    p_base.set_defaults(func=cmd_baseline)

    p_report = sub.add_parser("report", help="Build bar-chart + curve-overlay figures from eval/results/*.json.")
    p_report.add_argument("--out-dir", type=str, default=None, help="Default: eval/reports/figures/")
    p_report.add_argument("--real-results", type=str, default=None)
    p_report.add_argument("--synthetic-results", type=str, default=None)
    p_report.add_argument("--baseline-real", type=str, default=None)
    p_report.add_argument("--baseline-synthetic", type=str, default=None)
    p_report.set_defaults(func=cmd_report)

    p_all = sub.add_parser("all", help="sweep -> baseline -> report, real+synthetic, zero required flags.")
    p_all.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p_all.add_argument("--checkpoints", type=str, default=None,
                        help="'all' (default, every registered family) or a comma-separated list of "
                             "family names / 'family:step' / raw .pt paths, same as `sweep --checkpoints`.")
    p_all.add_argument("--no-gp-baseline", action="store_true",
                        help="Skip the real-ERA5 classical-GP-MLE baseline nll_total fit (same as "
                             "`sweep --no-gp-baseline`).")
    p_all.set_defaults(func=cmd_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
