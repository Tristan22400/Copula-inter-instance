"""Fixed real-ERA5 validation batches and the figures train.py logs from them."""

from __future__ import annotations

import os

from copula_inter.probe_batches import _name_seed, _tabicl_pit_batch

# P/N (hence attention sequence length T=P+N) are sampled per-shard from a wide
# range (see conf/data/gp_tasks.yaml P_min/P_max, N_min/N_max), so batches vary
# a lot in size while batch_size stays fixed — some shards get much closer to
# the VRAM ceiling than others. When that happens, PyTorch's caching allocator
# can fail a small allocation despite reserved-but-unallocated memory being
# nominally sufficient, because it's fragmented into pieces too small to
# satisfy the request (see the OOM message's "reserved but unallocated"
# figure). expandable_segments avoids this by growing/shrinking allocations
# in-place instead of requiring a fresh contiguous chunk. Must be set before
# the CUDA caching allocator initializes (i.e. before any CUDA call), so this
# goes at the top of the file, before `import torch`. setdefault so an
# explicit environment override still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

# eval/ (regions.py, spatial-correlation probe helpers -- see
# _build_era5_val_batches below) lives at the repo root, not under src/.
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
    """Fixed per-region real-ERA5 probes for the ``era5_fit/<region>``
    validation metrics (see validate()) — the real-data analogue of
    _build_synthetic_kernel_batches above.

    Unlike a kernel_fit/<family> synthetic probe, real ERA5 has no known GP
    oracle (no Sigma_star/R_star), so there is no NLL-gap metric to compute
    here. Instead, eval.spatial.sweep_core.build_era5_probe freezes a ground-
    truth correlation-vs-distance curve (empirical Pearson correlation, the
    same convention eval/runners/spatial_correlation_eval.py's real-mode
    sweep uses) plus a fixed real in-context sample (context coords/values,
    PIT'd once against `tabicl_marginal`) for a handful of ERA5 days per
    region. validate() re-runs only the CURRENT model's forward pass on this
    frozen input every call and scores the resulting correlogram against the
    frozen curve — the ERA5 fetch + PIT cost is paid once, here, not on the
    training loop's hot path.

    `tabicl_marginal` may be None (no PIT checkpoint configured): falls back
    to naive per-context standardization, same as
    eval.spatial.diagnostics.extract_model_context_correlation. In that case
    there is no real predictive density to score a Y-space NLL against, so
    the returned probe carries no "nll_test_z"/"nll_test_log_pdf" and
    validate()'s era5_fit/<region>/y_nll_total block is skipped for every
    region.

    When `tabicl_marginal` IS given, this also runs TabICL's own PIT
    (_tabicl_pit_batch) once on the probe's held-out (never-in-context)
    points (build_era5_probe's nll_test_idx/context_values_per_day/
    nll_test_values_per_day) — the same real-marginal z_test/log_pdf_test
    val/y_nll_total is scored against for the general val set, just frozen
    here alongside z_train since tabicl_marginal doesn't change during
    training either.

    Also fits a classical-GP-MLE baseline (region_batch["gp_baseline_nll"],
    scored by validate() alongside the model's own era5_fit/<region>/
    y_nll_total for a live comparison) via eval.spatial.sweep_core::
    _fit_gp_baseline_nll, independent of tabicl_marginal — same
    fit-once-here-not-per-validate()-call rationale, at deliberately lighter
    settings than that module's own rigor defaults (see the era5_gp_* cfg
    reads below for why).
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

    # Classical-GP-MLE baseline (era5_fit/<region>/gp_baseline_<kernel>_nll_*
    # in validate()) -- a training-time-affordable version of
    # spatial_correlation_eval.py real-mode sweep's own GP_BASELINE_KERNELS
    # fit (eval/spatial/sweep_core.py::_fit_gp_baseline_nll), reused here
    # directly rather than duplicated. That sweep's rigor defaults (all of
    # GP_BASELINE_KERNELS, GP_N_STEPS_MLE=1000, GP_N_RESTARTS_MLE=5) are NOT
    # reused as-is: measured ~17s/kernel/restart/1000-steps on CPU, ~8s on
    # GPU, so 5 kernels x 5 restarts x era5_n_days_probe(3) days x 5 regions
    # would add 30-100+ minutes to every train.py startup. Only 2 of
    # GP_BASELINE_KERNELS by default (matern32 -- this codebase's other
    # standard default kernel -- + rational_quadratic), 1 of the probe's
    # frozen days, 1 restart, and 300 steps keeps this to roughly 20-30s
    # total (still a one-time cost paid here, not on validate()'s hot path
    # -- same precompute-once rationale as the PIT/fetch cost above). Bump
    # era5_gp_baseline_kernels/era5_gp_n_restarts_mle/era5_gp_n_steps_mle
    # back up via cfg for a rarer, higher-fidelity run if the extra startup
    # time is worth it.
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
    """Fixed sparse-context real-ERA5 probe for the qualitative
    ``val/era5_predictions`` figure in validate()'s do_plot block --
    replaces the old val/corr_density_analytic_z + val/corr_grid
    correlation-matrix-vs-oracle plots (see this repo's
    feedback_no_raw_correlation_vs_oracle_comparison note: comparing the
    model's Sigma directly against the GP's exact R_star isn't a valid
    diagnostic once TabICL's PIT is in the loop, since Sigma lives in
    TabICL's own approximate z-space, not the GP's exact one) with
    something directly interpretable: the model's predicted temperature
    field against the real ground truth, on a handful of frozen days, from
    a context sparse enough (< baselines.era5_viz_context_frac, default
    5%, of the grid) to be a genuine spatial-extrapolation test rather than
    near-complete coverage.

    Unlike _build_era5_val_batches (era5_fit/<region>'s NLL probe, which
    uses a denser ~5% context tuned for a stable NLL estimate, not a
    strictly-below-5% one), this picks ONE region and keeps the grid/days/
    context sample fixed across every do_plot call -- only the model's
    forward pass changes step to step, so the figure is directly
    comparable across training.

    Splits precompute (here, once) from live (validate()'s do_plot block,
    every call) the same way _build_era5_val_batches does: the context
    sample, its PIT z_train, and each day's TabICL marginal quantile
    function are all independent of the (still-training) copula model, so
    they're computed once. The marginal quantile function in particular
    (TabICL's QuantileDistribution) is a pure function of its own stored
    tensors once built -- see tabicl's quantile_dist.py:icdf -- so caching
    the `dist` object per day here means validate() never needs to keep
    the (VRAM-heavy) tabicl_marginal net resident, or re-run its forward
    pass, for the rest of training; it only reruns the copula model's own
    forward pass plus a cheap Cholesky sample + icdf lookup.

    Also caches each day's `dist.variance()` (rescaled to real Kelvin^2 by
    the same y_mean/y_std) for val/era5_marginal_variance, and `dist.mean()`
    (rescaled the same way) for val/era5_residuals' mean-removal -- both are
    properties of the frozen marginal alone, so they're likewise computed
    once here rather than in validate()'s do_plot block.
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
    # Fitted-GP reference row (see _era5_viz_gp_posterior). Reuses the
    # era5_gp_*_mle knobs the era5_fit NLL probe's baseline already reads,
    # so there's one place to tune fit fidelity for both.
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

    # int() truncates (not rounds), so this can never land ON the 5%
    # boundary the way a round() could -- strictly < context_frac of D.
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
            # Var[y|x] in real Kelvin^2 = Var[y_scaled|x] * y_std^2 -- dist_d
            # lives in the same context-normalized scale normalize_targets
            # put context_values_scaled_t in (see _era5_viz_field's inverse
            # rescale). QuantileDistribution.variance() is the analytic
            # tail-corrected E[Z^2]-E[Z]^2 formula (quantile_dist.py), so this
            # is exact given the fitted spline/tails, not a sampling estimate.
            marginal_var_per_day.append((dist_d.variance() * y_std_t.double() ** 2).cpu().numpy())
            # E[y|x] in real Kelvin, same rescale as _era5_viz_field's return
            # line (linear, so y_std scales rather than y_std^2) -- the
            # per-location mean val/era5_residuals subtracts off every row.
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
    """Fitted-GP posterior (mean, Cholesky factor) at the viz grid, fitted by
    MLE+MAP on the SAME sparse context the copula model sees -- the reference
    predictor behind val/era5_predictions' "Fitted GP posterior sample" row.

    Same fit as the era5_fit/<region> NLL probe's GP baseline
    (eval/spatial/sweep_core.py::_fit_gp_baseline_nll): identical
    fit_and_eval_gpytorch call, oracle_mode="posterior" (a real
    context-conditioned spatial predictor, not the unconditioned prior), and
    the same z-score-fit-rescale dance -- fit_and_eval_gpytorch's MAP priors
    are tuned to data_gen.py's ~unit-variance synthetic y-scale, so raw
    Kelvin has to be standardized before fitting and the posterior rescaled
    back afterwards, or the prior drags the outputscale/noise to the wrong
    magnitude.

    Precomputed once per viz day (this is a pure function of the frozen
    context sample, not of the still-training copula model, exactly like
    _build_era5_viz_batch's z_train/dists), and the O(D^3) Cholesky is done
    here rather than per do_plot call so drawing the sample later is one
    matvec. Returns None -- caller drops the GP row and the figure renders
    as it did before -- if the fit or the factorization fails, since a
    reference panel is never worth killing a validation pass over.
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
    """Fitted-GP CORRELATION ONLY, MLE-fit directly on the PIT latent
    z_train instead of raw Kelvin -- a SEPARATE fit from
    _era5_viz_gp_posterior, backing val/era5_predictions' "Fitted GP
    correlation + TabICLv2 marginal" row and val/era5_predictions_z's GP
    row, in place of that other function's correlation.

    Why a second fit rather than reusing _era5_viz_gp_posterior's: that fit's
    kernel hyperparameters (lengthscale/outputscale/noise) are a Gaussian-
    likelihood MLE against raw Kelvin, only mean/std-normalized -- real T2m
    still has whatever skew/heteroscedasticity mean/std normalization
    doesn't remove, so a Gaussian likelihood is somewhat misspecified against
    it, which can bias the fitted correlation (e.g. heavy tails inflating
    the noise estimate, over-shrinking off-diagonal correlation). z_train has
    already been Gaussianized by TabICL's own conditional PIT
    (compute_context_z_train) -- fitting the SAME kernel family's
    hyperparameters against it instead removes that marginal-shape
    contamination from the correlation-only estimate, which is the fairest
    classical-GP reference for isolating whether the neural copula's
    correlation beats a classical GP's, holding the (TabICL) marginal fixed
    on both sides.

    No y_mean/y_std rescale-back needed (unlike _era5_viz_gp_posterior):
    z_train is already ~zero-mean/unit-variance by construction (PIT
    output), matching the MAP priors' assumed scale as-is, and only the
    fit's correlation matrix R is kept -- its posterior MEAN is never used
    by either downstream row (both draw a zero-mean copula sample and let
    TabICL's marginal supply location/scale), so it isn't computed here.

    Returns None (caller drops the row) on fit/factorization failure, same
    convention as _era5_viz_gp_posterior.
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
    """One joint draw (D,) from the fitted GP posterior of
    _era5_viz_gp_posterior, using the SAME latent white-noise vector
    `z_shared` that _era5_viz_field injects into the copula model's Sigma.

    Sharing z_shared is the point: the GP row and the copula row then differ
    ONLY in their correlation structure (and in the GP's own Gaussian
    marginal), not in which realization of the noise they happened to draw,
    so a visible difference in smoothness is attributable to the model
    rather than to sampling luck.
    """
    return gp["mean"] + gp["L"] @ z_shared


def _era5_viz_gp_correlation(gp: dict) -> np.ndarray:
    """Convert the fitted GP posterior covariance into its correlation matrix."""
    Sigma = gp["L"] @ gp["L"].T
    std = np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    return Sigma / np.outer(std, std)


def _era5_viz_field(Sigma: np.ndarray, dist, y_mean: torch.Tensor, y_std: torch.Tensor, z_shared: np.ndarray, device: str) -> np.ndarray:
    """One joint draw (D,) from the copula model's implied field for one
    ERA5 viz day: inject the CURRENT model correlation matrix `Sigma` into
    the shared latent Gaussian vector `z_shared` via Cholesky, then map
    through the frozen per-day marginal quantile function `dist` (see
    _build_era5_viz_batch) -- i.e. y = F_hat^{-1}(Phi(z)), the same
    construction as eval.spatial.diagnostics.predict_copula_residual_field,
    just split so only this (cheap) step reruns every do_plot call instead
    of also repeating the (expensive) TabICL forward pass that built `dist`.
    Falls back to a naive Gaussian(mean, std) marginal if `dist` is None
    (no PIT checkpoint configured -- see _build_era5_viz_batch).
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
    """Builds the ``val/era5_predictions`` figure (and its mean-removed
    ``val/era5_residuals`` companion) from a frozen _build_era5_viz_batch
    probe: reruns the CURRENT model's forward pass (the only per-step-
    changing input) to get Sigma for each of the probe's few frozen days,
    samples one field from it via _era5_viz_field, and renders ground-truth
    vs. fitted-GP-posterior vs. predicted (vs. independent, copula switched
    off) small multiples via eval.viz.correlation_plots.plot_residual_grid.
    Returns (None, None) if the probe's grid was degenerate (e.g. 0 valid
    days).

    The fitted-GP row (precomputed by _build_era5_viz_batch, drawn here with
    the SAME z_shared as the copula row) is the reference that makes the
    figure readable: both it and the model row are posterior SAMPLES at <5%
    context, whereas the ground-truth row is a fully-observed realization.
    Comparing a sample against the truth rewards whichever model is most
    overconfident -- emitting the prior correlation rather than the
    posterior gives beautiful smooth fields and a much WORSE held-out NLL.
    Against the GP's own sample, that trade is visible instead of hidden.

    The residual companion figure subtracts each row's OWN predictive mean
    (the frozen TabICL marginal's mean field -- see _build_era5_viz_batch's
    marginal_mean_per_day -- for the ground-truth/predicted/independent
    rows, the fitted GP's own posterior mean for the GP row) at every
    location before plotting, via eval.viz.correlation_plots.
    plot_mean_removed_grid. Raw ERA5 temperature is dominated by a smooth
    lat/lon gradient that swamps the finer cross-location structure Sigma is
    actually scored on; removing the mean isolates that structure so the
    model's residual texture can be compared by eye against the ground
    truth's and the fitted GP's own residual samples. Reuses this
    function's per-day forward pass and z_shared draws rather than
    resampling, so the two figures are built from the identical fields (the
    residual panel is a strict function of the ones the raw panel already
    plots) at no extra model-forward cost. Silently drops (None) if any
    day's marginal mean is missing (no PIT checkpoint configured).
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
        # gp_post_z (correlation fit on z_train) is a SEPARATE fit from
        # gp_post (correlation+mean fit on raw y) -- see
        # _era5_viz_gp_posterior_on_z -- so it fails/succeeds independently
        # and gets its own all-or-nothing gate below.
        if gp_post_z[i] is not None:
            gp_tabicl_field = _era5_viz_field(
                _era5_viz_gp_correlation(gp_post_z[i]), dist_i, y_mean_i, y_std_i, z_shared, device,
            )
            gp_tabicl_fields.append(gp_tabicl_field)
            if mean_i is not None:
                gp_tabicl_resid.append(gp_tabicl_field - mean_i)
    # All-or-nothing: a partially populated oracle row would silently pair
    # day j's GP draw with day k's column (_plot_field_grid zips rows against
    # true_fields positionally), so one failed per-day fit drops the row.
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
    """Builds the ``val/era5_predictions_z`` figure: `n_samples` posterior
    draws of the COPULA LATENT z (no marginal, no ground truth -- a real
    field has no observed z) on ONE fixed day, vb["days"][0] -- the SAME day
    as val/era5_predictions' first column, so the two figures are directly
    comparable. Rows are the three predictors' correlation structures:
    independent (R=I), the copula model's current Sigma (one forward pass,
    reused for all n_samples columns since Sigma doesn't depend on the
    sample), and the fitted-GP baseline's correlation -- the LATTER from
    _era5_viz_gp_posterior_on_z (MLE-fit directly on z_train), not
    _era5_viz_gp_posterior's raw-y fit, since this whole figure lives in
    z-space already and z_train is the honest target for a z-space
    correlation baseline. Every column shares one white-noise draw z_shared
    across all three rows (own RNG, seeded off vb["seed"] + 1 so it never
    perturbs, or is perturbed by, _era5_viz_fig's own per-day z_shared
    sequence), so column-to-column differences are sampling variation and
    row-to-row differences are the correlation structure alone.

    Returns None if the probe has no days, or if the fixed day's GP-on-z fit
    failed/was disabled -- 2 of 3 requested predictors isn't this figure.
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
    """Builds the ``val/era5_marginal_variance`` figure: the frozen per-day
    TabICL marginal's predictive Var[y|x] (real Kelvin^2, cached per day by
    _build_era5_viz_batch) at every grid location, one column per probe day,
    context locations overlaid, plus a second row for the fitted-GP
    baseline's own posterior Var[y|x] -- diag(Sigma_gp) read straight off
    gp_post_per_day[i]["L"] (the SAME fitted covariance _era5_viz_gp_field
    draws samples from, already rescaled to real Kelvin^2 by
    _era5_viz_gp_posterior), no refit needed. Both rows are pure functions
    of the frozen marginal/GP fit + context sample -- no model forward pass,
    no copula, unaffected by training -- so this answers a narrower question
    than val/era5_predictions: does either predictor's OWN uncertainty grow
    with distance from context the way a calibrated spatial predictor's
    should, and does the frozen TabICL marginal track the classical GP's
    behavior or diverge from it?

    Returns None if the probe has no days, or if any day's marginal variance
    is missing (no PIT checkpoint configured -- see _build_era5_viz_batch's
    tabicl_marginal is None branch). The GP row is dropped (all-or-nothing,
    same convention as _era5_viz_fig's oracle row) rather than the whole
    figure if any day's GP fit failed or era5_viz_gp is disabled.
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
