"""
plot_spatial_correlation_diagnostics.py — Spatial correlogram sanity check
for TabICLv2 (CopulaTabICL): does the model's learned inter-instance
dependence structure reproduce the physical spatial correlation decay of the
real ERA5 field?

Four empirical/model curves plus theoretical-law overlays, all binned (or,
for the theory curves, evaluated) by great-circle distance and plotted
together:

  1. Ground truth: empirical correlation of the quantity selected by
     --target across the spatial grid. --target raw (the default) uses the
     raw absolute temperature field Z_true_t itself; --target residual uses
     the 24h persistence residual E_t = Z_true_t - Z_true_{t-24} instead
     (the original convention this script used before --target existed —
     kept for comparison, since differencing removes common-mode structure
     like the seasonal cycle that raw temperature keeps in). See
     get_ground_truth_observations.
  2. Independent TabICLv2: a model with no inter-instance copula assumes
     conditional independence across the grid, so its implied correlation
     matrix IS the identity by construction — no forward pass needed.
  3. Copula model with dummy context: an (as-near-as-architecturally-possible)
     unconditional forward pass of the trained CopulaTabICL checkpoint. This
     is NOT a Bayesian posterior extraction, just a forward pass with a
     content-free dummy context row, matching how the model is trained
     (cfg.data.oracle_mode="prior", see src/data_gen.py) to output R_star
     ignoring in-context conditioning.
  4. Copula model with N context points: a real joint forward pass over all
     D grid points at once, conditioned on a historical in-context sample of
     the SAME --target quantity as (1) (raw temperature by default, or the
     24h persistence residual under --target residual) — so it's conditioned
     on the same physical quantity whose spatial decay it's being compared
     against. This is likewise NOT a Bayesian posterior — it's the same
     forward pass as (3), just with real context instead of a dummy one.
     Context labels z_train are NOT a naive (y - mean) / std standardization:
     they are the K-fold leave-one-out Probability Integral Transform of
     each context point's true value under TabICLv2's own learned marginal
     (see src/pit.py::run_pit) — i.e. u_i = F_hat(y_i | other context
     points), z_i = Phi^-1(u_i) — matching how z_train is actually defined
     during training (data_gen.py's GP-oracle LOO PIT) instead of assuming a
     Gaussian marginal by fiat.
  5. Theoretical decay laws (--theory-models, default: ALL FOUR at once):
     every isotropic correlation law in the reference literature --
     exponential (Hansen & Lebedeff 1987, nu=1/2), Gaussian (nu->infinity),
     Whittle (Matern nu=1, the closed-form omega->0 limit of North, Wang &
     Genton 2011's energy-balance/damped-diffusion model), and the general
     Matern (free smoothness nu, fit directly rather than assumed) -- each
     independently fit by weighted nonlinear least squares to curve (1)
     ONLY (never to the model curves, and never to the synthetic-GP
     fallback's own generating kernel), so the fit is an independent
     physical sanity check: if the ground truth doesn't track a known decay
     law reasonably well (see the printed weighted R^2 per law), something
     is off with the empirical curve itself before comparing it to the
     model at all. Plotting all four together also shows which shape family
     the data actually prefers (e.g. Matern's fitted nu should land close
     to whichever of exponential/Gaussian/Whittle fits best).

Reuses:
  - plots/generate_plots.py: load_era5_data (+ synthetic-GP fallback) and
    haversine_distance_km, so ERA5 I/O and great-circle distance math are
    never reimplemented here.
  - src/model.py: build_copula_transformer + low_rank_correlation, the same
    (W, s) -> Sigma projection used by generate_plots.py's
    build_copula_correlation_fn — this script loads the checkpoint once and
    reuses it for both the dummy-context (3) and real-context (4)
    extractions instead of loading it twice.
  - src/pit.py: load_tabicl + run_pit, the same frozen-TabICL-quantile-head
    K-fold LOO PIT used to build z_train for real (non-GP-oracle) data —
    reused as-is rather than reimplemented with a naive standardization.

Usage:
    python plots/plot_spatial_correlation_diagnostics.py --ckpt ./checkpoints/systematic-composition-8/step_0180000.pt
    # fit the 24h persistence residual instead of the default raw temperature field:
    python plots/plot_spatial_correlation_diagnostics.py --ckpt ./checkpoints/systematic-composition-8/step_0180000.pt --target residual
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
from scipy.optimize import curve_fit
from scipy.special import gamma as gamma_fn, kv as bessel_k

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PLOTS_DIR)
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _PLOTS_DIR not in sys.path:
    sys.path.insert(0, _PLOTS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# src/pit.py imports its src/-local siblings (e.g. data_gen.py) with bare
# names ("from data_gen import ..."), so src/ itself must be on sys.path too
# -- not just the repo root -- for `from src.pit import run_pit` to work.
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from generate_plots import (  # noqa: E402
    haversine_distance_km,
    load_era5_data,
    plot_correlation_matrix_comparison,
    plot_spatial_correlation_diagnostics as plot_generic_diagnostics,
)

def _ckpt_mode_tag(ckpt_path: str, mode: str, target: "str | None" = None) -> str:
    """Short, filesystem-safe tag identifying (checkpoint run dir + step,
    mode, target), used to give every output file this script produces a
    distinct name -- so comparing several checkpoints, --mode real vs.
    synthetic, and/or --target raw vs. residual back-to-back doesn't
    silently overwrite a previous run's plots (all three, and every
    checkpoint, otherwise write the same fixed filenames). `target` is
    only meaningful for --mode real (synthetic mode's ground truth is a
    sampled GP draw, not real temperature, so raw/residual doesn't apply
    there) -- pass None to omit it from the tag."""
    run_dir = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
    step_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    tag = f"{run_dir}_{step_name}_{mode}"
    if target is not None:
        tag = f"{tag}_{target}"
    return tag


# ---------------------------------------------------------------------------
# Ground truth: empirical spatial correlation from 24h persistence residuals
# ---------------------------------------------------------------------------
def compute_persistence_residuals(field_all: np.ndarray) -> np.ndarray:
    """24h persistence residuals E_t = Z_true_t - Z_true_{t-24}.

    `field_all` (n_snapshots, H, W) is one grid snapshot per time index, at a
    fixed cadence of 24h apart (era5_temperature.nc, real or synthetic, is
    one daily snapshot at a fixed hour — see generate_plots.load_era5_data),
    so a lag of 1 index IS a 24h lag; consecutive-day differencing is exactly
    E_t = Z_true_t - Z_true_{t-24} rather than an approximation of it.

    Returns (n_snapshots - 1, H * W).
    """
    n = field_all.shape[0]
    if n < 2:
        raise ValueError(f"Need >= 2 time snapshots to form 24h persistence residuals, got {n}.")
    flat = field_all.reshape(n, -1)
    return flat[1:] - flat[:-1]


def compute_raw_temperature_observations(field_all: np.ndarray) -> np.ndarray:
    """Raw per-day temperature Z_t across the spatial grid, with NO
    differencing -- the --target raw counterpart to
    compute_persistence_residuals's --target residual. Unlike the 24h
    persistence residual (which removes any quantity shared across days,
    e.g. the seasonal cycle or a large-scale weather regime), this keeps
    that common-mode structure in, so its spatial correlation reflects
    "do these two locations tend to have similar absolute temperature",
    not just "do their day-to-day fluctuations move together".

    Returns (n_snapshots, H * W) -- every snapshot is used (no day is lost
    to lag-1 differencing, unlike compute_persistence_residuals).
    """
    n = field_all.shape[0]
    return field_all.reshape(n, -1)


def get_ground_truth_observations(field_all: np.ndarray, target: str) -> np.ndarray:
    """Dispatch on --target: the per-snapshot observation matrix (n, H*W)
    that both empirical_spatial_correlation and main()'s per-day context
    values are derived from -- the single place that decides what "the
    physical quantity being diagnosed" is."""
    if target == "raw":
        return compute_raw_temperature_observations(field_all)
    if target == "residual":
        return compute_persistence_residuals(field_all)
    raise ValueError(f"Unknown target '{target}', choose from 'raw' or 'residual'.")


def empirical_spatial_correlation(data: dict, target: str = "raw") -> np.ndarray:
    """Pearson correlation matrix R_emp (D x D) of the --target quantity
    (raw temperature by default, or the 24h persistence residual -- see
    get_ground_truth_observations)."""
    observations = get_ground_truth_observations(data["t2m"], target)
    return np.corrcoef(observations.T)


def morans_i(field: np.ndarray) -> float:
    """Global Moran's I (Moran, 1950) with rook (4-neighbor) adjacency on a
    regular (H, W) grid: the standard spatial-autocorrelation statistic for
    how smooth/locally coherent a SINGLE snapshot is --

        I = N * sum_edges (x_i - xbar)(x_j - xbar) / (E * sum_i (x_i - xbar)^2)

    where the sum is over unordered rook-adjacent cell pairs (E of them).
    +1 means neighboring cells are highly similar (smooth field), 0 means
    spatially random (no local structure, i.e. noise), negative means
    neighboring cells tend to differ (checkerboard-like roughness).

    This is a DIFFERENT notion of "spatial correlation" than R_emp/R_context
    elsewhere in this script: those measure whether two FIXED locations'
    time series correlate across days; this measures whether one day's
    spatial PATTERN is itself locally smooth.
    """
    x = field - field.mean()
    cross_h = x[:, :-1] * x[:, 1:]
    cross_v = x[:-1, :] * x[1:, :]
    n_edges = cross_h.size + cross_v.size
    numerator = cross_h.sum() + cross_v.sum()
    denominator = (x ** 2).sum()
    return float(field.size * numerator / (n_edges * denominator))


def predict_copula_residual_field(
    tabicl_marginal, context_coords: np.ndarray, context_values: np.ndarray,
    coords_test: np.ndarray, R_context: np.ndarray, device: str, z_shared: np.ndarray,
) -> np.ndarray:
    """One joint draw (D,) from the copula model's implied residual field:
    inject its predicted spatial correlation `R_context` (see
    extract_model_context_correlation, SAME forward pass / SAME context) into
    the given shared latent Gaussian vector `z_shared` via Cholesky, then map
    each coordinate through the frozen TabICL marginal quantile function
    (tabicl_marginal.icdf, conditioned on the same real context) -- i.e.
    y = F_hat^{-1}(Phi(z)), the exact inverse of the PIT definition
    u = F_hat(y), z = Phi^-1(u) this script already uses to build z_train
    elsewhere -- rather than approximating the marginal as Gaussian(mean, std).

    `z_shared` is passed in (not drawn here) so callers can reuse the SAME
    underlying noise across R_context vs. R_indep (see generate_plots.
    plot_spatial_map_comparison's identical convention): the only difference
    between the two resulting fields is then whether cross-location
    correlation was injected, not an unrelated random redraw.

    Falls back to a naive Gaussian(mean, std) marginal if `tabicl_marginal`
    is None (scratch-trained backbone, see load_marginal_tabicl).
    """
    from generate_plots import _safe_cholesky
    from scipy.stats import norm

    L = _safe_cholesky(R_context)
    z_copula = L @ z_shared
    u_copula = np.clip(norm.cdf(z_copula), 1e-6, 1.0 - 1e-6)

    if tabicl_marginal is None:
        y_std = max(context_values.std(), 1e-8)
        return context_values.mean() + y_std * z_copula

    import torch

    x_mean = context_coords.mean(axis=0, keepdims=True)
    x_std = context_coords.std(axis=0, keepdims=True).clip(min=1e-8)
    x_train_norm = (context_coords - x_mean) / x_std
    x_test_norm = (coords_test - x_mean) / x_std

    x_full = np.concatenate([x_train_norm, x_test_norm], axis=0)
    x_batch = torch.as_tensor(x_full, dtype=torch.float32, device=device).unsqueeze(0)  # (1, P+N, p_x)
    y_train_batch = torch.as_tensor(context_values, dtype=torch.float32, device=device).unsqueeze(0)  # (1, P)
    with torch.no_grad():
        logits = tabicl_marginal(x_batch, y_train_batch)  # (1, N, Q) -- N test rows only, per src/pit.py::run_pit
        n_test = coords_test.shape[0]
        dist = tabicl_marginal.quantile_dist(logits.reshape(n_test, -1))
        # icdf's 1-D alpha shape means "same n probability levels evaluated for every
        # batch element" (broadcasts to (*batch_shape, n)) -- NOT "one alpha per batch
        # element". We need the latter (one u per grid point), so pass alpha shaped
        # (*batch_shape, 1) explicitly and drop the resulting trailing singleton dim.
        u_t = torch.as_tensor(u_copula, dtype=torch.float32, device=device).unsqueeze(-1)
        y_pred = dist.icdf(u_t).squeeze(-1)
    return y_pred.cpu().numpy()


def _plot_field_grid(
    lat: np.ndarray, lon: np.ndarray, grid_shape: tuple, true_fields: list, col_titles: list,
    output_path: str, row0_label: str, suptitle: str,
    predicted_fields: "list[np.ndarray] | None" = None,
    independent_fields: "list[np.ndarray] | None" = None,
    oracle_fields: "list[np.ndarray] | None" = None,
    context_coords: "np.ndarray | None" = None,
    pred_row_label: str = "Copula model\n(predicted)\n +marginal\nLatitude",
    indep_row_label: str = "Independent\n(no copula)\nLatitude",
    oracle_row_label: str = "Oracle correlation\n+ marginal\nLatitude",
    xlabel: str = "Longitude", cbar_label: str = "Residual (deg C)",
) -> None:
    """Shared small-multiples renderer behind both plot_residual_grid (real
    ERA5 days) and plot_synthetic_residual_grid (synthetic GP draws): top row
    `true_fields` (already grid_shape-shaped), then an optional
    oracle-correlation row, then optional predicted/independent rows below
    (flat (D,) arrays, reshaped to grid_shape here) -- see plot_residual_grid's
    docstring for the row semantics this preserves. `oracle_fields` is only
    ever passed by plot_synthetic_residual_grid (the real-ERA5 branch has no
    oracle covariance), inserted right below ground truth so the row order
    reads: ground truth -> best-case copula (oracle correlation) -> actual
    model (R_pred) -> no-correlation baseline. All rows share one color scale
    and the Moran's I annotation (see morans_i) so both callers render
    pixel-identically."""
    has_pred = predicted_fields is not None
    has_indep = independent_fields is not None
    has_oracle = oracle_fields is not None
    pred_grids = [f.reshape(grid_shape) for f in predicted_fields] if has_pred else []
    indep_grids = [f.reshape(grid_shape) for f in independent_fields] if has_indep else []
    oracle_grids = [f.reshape(grid_shape) for f in oracle_fields] if has_oracle else []

    vmax = float(np.max(np.abs(true_fields + pred_grids + indep_grids + oracle_grids)))

    def _annotate_morans_i(ax, field):
        # See morans_i's docstring: smoothness of THIS single snapshot, not the
        # cross-day spatial correlation R_emp/R_context already plotted elsewhere.
        ax.text(
            0.97, 0.95, f"$I$={morans_i(field):.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    n_cols = len(true_fields)
    n_rows = 1 + int(has_oracle) + int(has_pred) + int(has_indep)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.8 * n_rows), sharex=True, sharey=True, squeeze=False)
    mesh = None
    for j, (title, field) in enumerate(zip(col_titles, true_fields)):
        mesh = axes[0][j].pcolormesh(lon, lat, field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        axes[0][j].set_title(title, fontsize=9)
        _annotate_morans_i(axes[0][j], field)
    axes[0][0].set_ylabel(row0_label)

    def _plot_row(row_idx, grids, ylabel):
        for j, field in enumerate(grids):
            mesh_local = axes[row_idx][j].pcolormesh(lon, lat, field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
            _annotate_morans_i(axes[row_idx][j], field)
            if context_coords is not None:
                axes[row_idx][j].scatter(
                    context_coords[:, 0], context_coords[:, 1],
                    c="black", s=8, marker="o", linewidths=0.4, edgecolors="white",
                    label="Context points" if j == 0 else None,
                )
        axes[row_idx][0].set_ylabel(ylabel)
        if context_coords is not None:
            axes[row_idx][0].legend(loc="upper left", fontsize=6, framealpha=0.7)
        return mesh_local

    row = 1
    if has_oracle:
        mesh = _plot_row(row, oracle_grids, oracle_row_label)
        row += 1
    if has_pred:
        mesh = _plot_row(row, pred_grids, pred_row_label)
        row += 1
    if has_indep:
        mesh = _plot_row(row, indep_grids, indep_row_label)

    for j in range(n_cols):
        axes[-1][j].set_xlabel(xlabel)
    fig.suptitle(suptitle)
    plt.tight_layout(rect=(0.0, 0.0, 0.93, 0.96))
    fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.85, label=cbar_label)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_residual_grid(
    data: dict, days: list, predicted_fields: "list[np.ndarray] | None", output_path: str,
    context_coords: "np.ndarray | None" = None,
    independent_fields: "list[np.ndarray] | None" = None,
    target: str = "raw",
) -> None:
    """Small-multiples panel of the --target field (raw temperature Z_t by
    default, or the 24h persistence residual E_t = Z_t - Z_{t-24} under
    --target residual) on the lat/lon grid, one column per day in `days`:
    top row is the ground-truth field (the same field that feeds both R_emp
    and the real-context conditioning); if `predicted_fields` is given, the
    next row is the copula model's predicted field for that SAME day (see
    predict_copula_residual_field); if `independent_fields` is also given, a
    third row shows the SAME marginal-per-point prediction with the copula's
    cross-location correlation switched off (R replaced by the identity --
    the "Independent TabICLv2" case already used in the correlogram/
    correlation-matrix plots), isolating what the learned correlation
    structure itself adds on top of the per-point marginal. All rows share
    one color scale so they are directly comparable pixel-for-pixel. If
    `context_coords` is given, the (fixed, same-every-day) real-context
    locations that condition both predicted rows are overlaid as black
    markers on those rows only -- the ground-truth row shows the process
    being measured, not what the model was told about it.
    """
    lat, lon = data["latitude"], data["longitude"]
    grid_shape = data["t2m"][0].shape
    if target == "residual":
        true_fields = [data["t2m"][d] - data["t2m"][d - 1] for d in days]
        col_titles = [f"day {d}: $E_t = Z_{{{d}}} - Z_{{{d - 1}}}$" for d in days]
        suptitle = (
            "24h Persistence Residual Fields: Ground Truth vs. Copula Model Prediction" if predicted_fields is not None
            else "24h Persistence Residual Fields (ground-truth input to $R_{emp}$ and real-context conditioning)"
        )
        cbar_label = "Residual (deg C)"
    else:
        true_fields = [data["t2m"][d] for d in days]
        col_titles = [f"day {d}: $Z_{{{d}}}$ (raw)" for d in days]
        suptitle = (
            "Raw Temperature Fields: Ground Truth vs. Copula Model Prediction" if predicted_fields is not None
            else "Raw Temperature Fields (ground-truth input to $R_{emp}$ and real-context conditioning)"
        )
        cbar_label = "Temperature (deg C)"
    _plot_field_grid(
        lat, lon, grid_shape, true_fields, col_titles, output_path,
        row0_label="Ground truth\nLatitude", suptitle=suptitle,
        predicted_fields=predicted_fields, independent_fields=independent_fields, context_coords=context_coords,
        cbar_label=cbar_label,
    )


def plot_synthetic_residual_grid(
    grid_y: np.ndarray, grid_x: np.ndarray, grid_shape: tuple,
    true_fields: list, predicted_fields: list, independent_fields: list,
    output_path: str, context_coords: "np.ndarray | None" = None,
    oracle_fields: "list[np.ndarray] | None" = None,
) -> None:
    """Synthetic-mode analogue of plot_residual_grid: one column per
    independent GP draw from the sampled ground-truth kernel (see
    run_synthetic_mode) instead of one column per real ERA5 day, with the
    SAME four rows (ground truth / oracle-correlation+marginal /
    copula-predicted / independent) and shared color scale / Moran's I
    annotation, via the same _plot_field_grid renderer plot_residual_grid
    uses. `oracle_fields` (see run_synthetic_mode) reuses the TabICLv2
    marginal from the predicted row but with the TRUE kernel correlation
    matrix instead of the model's R_pred, isolating correlation-estimation
    error from marginal-estimation error -- optional so plot_residual_grid's
    real-ERA5 caller (which has no oracle covariance) is unaffected."""
    col_titles = [f"draw {i + 1}" for i in range(len(true_fields))]
    _plot_field_grid(
        grid_y, grid_x, grid_shape, true_fields, col_titles, output_path,
        row0_label="Ground truth\n(synthetic kernel)\ny", suptitle="Synthetic GP Draws: Ground Truth vs. Copula Model Prediction",
        predicted_fields=predicted_fields, independent_fields=independent_fields, context_coords=context_coords,
        oracle_fields=oracle_fields,
        pred_row_label="Copula model\n(predicted)\ny", indep_row_label="Independent\n(no copula)\ny",
        oracle_row_label="Oracle correlation\n+ TabICLv2 marginal\ny",
        xlabel="x", cbar_label="Field value",
    )


# ---------------------------------------------------------------------------
# TabICLv2 / CopulaTabICL: shared checkpoint loading + dummy/real-context extraction
# ---------------------------------------------------------------------------
def load_copula_model(ckpt_path: str, device: "str | None" = None):
    """Load a CopulaTabICL checkpoint via the repo's single canonical loader
    (inference/copula_inference.py::load_copula_model) rather than
    reimplementing torch.load + build_copula_transformer + state_dict here;
    this wrapper only resolves the auto ("cuda" if available else "cpu")
    device default this script's --device flag relies on.
    """
    import torch

    from inference.copula_inference import load_copula_model as _load_copula_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, cfg = _load_copula_model(ckpt_path, device=device)
    return model, cfg, device


def load_marginal_tabicl(cfg, device: str):
    """Load the frozen, pretrained TabICL quantile regressor used ONLY as a
    marginal-CDF oracle for the PIT transform in
    extract_model_context_correlation — NOT the same object as the
    CopulaTabICL backbone in load_copula_model, whose quantile decoder has
    been stripped and replaced by the copula head (see src/model.py). This
    is the same checkpoint (cfg.tabicl.ckpt) CopulaTabICL's backbone was
    initialized from, loaded a second time with its native quantile head
    intact (see src/pit.py::load_tabicl / run_pit).

    Returns None (with a warning) if the checkpoint's backbone was trained
    from scratch (cfg.tabicl.pretrained=False), since there is then no
    quantile-calibrated marginal model available to PIT against.
    """
    if not bool(cfg.tabicl.get("pretrained", True)):
        print("Warning: cfg.tabicl.pretrained=False — no pretrained quantile "
              "head available for PIT; context z_train will fall back to "
              "naive standardization.")
        return None

    from src.pit import load_tabicl

    return load_tabicl(cfg.tabicl.ckpt, device)


def _forward_correlation(model, device, x_train_norm: np.ndarray, z_train: np.ndarray, x_test_norm: np.ndarray) -> np.ndarray:
    """Shared (x_train, z_train, x_test) -> Sigma forward pass, used by both
    the dummy-context and real-context extractions below (see
    src/model.py:CopulaTabICL and low_rank_correlation)."""
    import torch

    from src.model import low_rank_correlation

    x_train_t = torch.as_tensor(x_train_norm, dtype=torch.float32, device=device).unsqueeze(0)
    x_test_t = torch.as_tensor(x_test_norm, dtype=torch.float32, device=device).unsqueeze(0)
    z_train_t = torch.as_tensor(z_train, dtype=torch.float32, device=device).unsqueeze(0)
    batch = {"x_train": x_train_t, "x_test": x_test_t, "z_train": z_train_t}

    with torch.no_grad():
        out = model(batch)
        Sigma = low_rank_correlation(out["W"], out["s"], jitter=1e-4)
    return Sigma[0].cpu().numpy()


def extract_model_dummy_context_correlation(model, device, coords_test: np.ndarray) -> np.ndarray:
    """Extract the model's correlation matrix via an unconditional forward
    pass — i.e. with no informative historical in-context examples, so the
    output cannot depend on any specific test-time context, only on what
    the model learned as its context-free behavior. This is NOT a Bayesian
    prior extraction in any closed-form sense — just a forward pass fed a
    dummy context (see below).

    CopulaTabICL has no separate closed-form "no-context head": (W, s) are
    always produced from a forward pass over (x_train, z_train, x_test). A
    literal zero-row x_train/z_train (P=0) is not supported by the
    underlying TabICL backbone — its target-aware column embedding
    unconditionally computes `y_train.max()` (see
    tabicl_upstream/src/tabicl/_model/embedding.py), which raises on an
    empty tensor regardless of oracle mode. The closest architecturally-valid
    stand-in for "no historical context" is therefore a single dummy context
    row at x_train=0, z_train=0 (P=1) — a constant, content-free input
    carrying no information about any real historical series. Under
    cfg.data.oracle_mode="prior" training (the current default, see
    conf/data/gp_tasks.yaml), R_star is defined to ignore training-context
    conditioning entirely, so a well-trained model's output here should not
    be sensitive to which dummy value is fed in.
    """
    coords_test = np.asarray(coords_test, dtype=np.float64)
    x_mean = coords_test.mean(axis=0, keepdims=True)
    x_std = coords_test.std(axis=0, keepdims=True).clip(min=1e-8)
    x_test_norm = (coords_test - x_mean) / x_std

    x_train_norm = np.zeros((1, coords_test.shape[1]), dtype=np.float64)
    z_train = np.zeros(1, dtype=np.float64)
    return _forward_correlation(model, device, x_train_norm, z_train, x_test_norm)


def extract_model_context_correlation(
    model, device, tabicl_marginal, context_coords: np.ndarray, context_values: np.ndarray,
    coords_test: np.ndarray, k_folds: int = 10,
) -> np.ndarray:
    """Extract the model's correlation matrix via a single joint forward
    pass over all of `coords_test` at once, conditioned on a real historical
    in-context sample (context_coords, context_values). This is NOT a
    Bayesian posterior extraction — it's the same forward pass as
    extract_model_dummy_context_correlation, just with real context points
    instead of a dummy one.

    z_train is the K-fold leave-one-out PIT of each context point's true
    value under `tabicl_marginal`'s own predicted marginal distribution
    (src/pit.py::run_pit, reused as-is): each context point is held out in
    one of k_folds disjoint folds, its marginal CDF F_hat is predicted from
    the OTHER context points via the frozen pretrained TabICL quantile
    head, u_i = F_hat(y_i) is its resulting quantile, and z_i = Phi^-1(u_i)
    Gaussianizes it. This is the real-data analogue of how z_train is
    defined during training (data_gen.py's GP-oracle LOO PIT, R&W Eq.
    5.12) — same PIT definition, model marginal instead of GP closed form
    — replacing the previous naive (y - mean) / std standardization, which
    assumed a Gaussian marginal instead of estimating one.

    If `tabicl_marginal` is None (scratch-trained backbone, no quantile
    head available), falls back to the naive standardization.
    """
    x_mean = context_coords.mean(axis=0, keepdims=True)
    x_std = context_coords.std(axis=0, keepdims=True).clip(min=1e-8)
    x_train_norm = (context_coords - x_mean) / x_std
    x_test_norm = (coords_test - x_mean) / x_std

    if tabicl_marginal is None:
        y_std = max(context_values.std(), 1e-8)
        z_train = (context_values - context_values.mean()) / y_std
    else:
        import torch

        from src.pit import run_pit

        X_train_t = torch.as_tensor(x_train_norm, dtype=torch.float32, device=device)
        Y_train_t = torch.as_tensor(context_values, dtype=torch.float32, device=device).unsqueeze(-1)  # (P, 1)
        pit_out = run_pit(
            tabicl_marginal, X_train_t, Y_train_t, X_train_t[:1], Y_train_t[:1], k_folds=k_folds,
        )
        z_train = pit_out["z_train"].squeeze(-1).cpu().numpy()  # (P,)

    return _forward_correlation(model, device, x_train_norm, z_train, x_test_norm)


# ---------------------------------------------------------------------------
# Synthetic mode: known ground-truth covariance from a single data_gen.py
# kernel, vs. the model's real forward-pass prediction on a matching draw --
# a no-real-data-confounds sanity check, using the same two generic plots
# (generate_plots.plot_spatial_correlation_diagnostics) as the real branch.
# ---------------------------------------------------------------------------
def sample_simple_kernel_covariance(cfg, coordinates: np.ndarray, seed: "int | None" = None) -> "tuple[np.ndarray, str]":
    """Ground-truth covariance for synthetic mode: samples ONE elementary
    (non-composite) kernel from src/data_gen.py's registry, with
    hyperparameters drawn from the SAME LogNormal/Gamma hyperpriors training
    episodes use (data_gen._kernel_prior_spec, via _build_kernel_component)
    -- bypassing data_gen's sum/product composition path entirely (see
    cfg.data.systematic_composition there), per project convention that a
    synthetic sanity check should isolate one kernel family at a time.

    `coordinates` (M, d) is standardized (zero mean, unit variance) before
    evaluating the kernel, since data_gen's lengthscale prior is calibrated
    for that scale (see _kernel_prior_spec's docstring) -- the same
    normalization convention extract_model_context_correlation already uses
    for x_train/x_test.

    cfg.data.sign_modulation_component_prob (default 0.5, see
    conf/data/gp_tasks.yaml) is forced to 0 for this call: sign modulation
    is a separate per-component training-diversity augmentation, not part of
    "the list of available kernels", and it breaks the constant-diagonal
    prior covariance a single plain kernel should have here.

    Returns (Sigma, kernel_name).
    """
    import random as _random

    import torch
    from omegaconf import OmegaConf

    from data_gen import _build_kernel_component, _COMPOSABLE_KERNELS, _SCALAR_ONLY_KERNELS

    if seed is not None:
        _random.seed(seed)
        torch.manual_seed(seed)

    cfg = OmegaConf.merge(cfg, OmegaConf.create({"data": {"sign_modulation_component_prob": 0.0}}))

    coords = np.asarray(coordinates, dtype=np.float64)
    x_std = (coords - coords.mean(axis=0)) / coords.std(axis=0).clip(min=1e-8)
    k = x_std.shape[1]

    # Exclude "dot_product"/"polynomial" (no real lengthscale -- not a
    # distance-decay kernel) and, for k > 1 coordinates, "cosine" (only PSD
    # for scalar input -- see _SCALAR_ONLY_KERNELS in data_gen.py).
    candidates = [
        name for name in _COMPOSABLE_KERNELS
        if name not in ("dot_product", "polynomial") and not (k > 1 and name in _SCALAR_ONLY_KERNELS)
    ]
    kernel_name = _random.choice(candidates)

    kernel, params = _build_kernel_component(cfg, kernel_name, k=k, B=1, device="cpu")
    x_t = torch.as_tensor(x_std, dtype=torch.get_default_dtype()).unsqueeze(0)  # (1, N, k)
    with torch.no_grad():
        Sigma = kernel(x_t, x_t).to_dense()[0].numpy()

    param_str = ", ".join(f"{name}={v.item():.3f}" for name, v in params.items() if v.numel() == 1)
    print(f"Synthetic ground truth: sampled kernel '{kernel_name}' ({param_str})")
    return Sigma, kernel_name


def run_synthetic_mode(args, rng, model, cfg, device, tabicl_marginal, tag: str) -> None:
    """Synthetic-data branch of main(): draws coordinates on a regular 2D
    grid (so plot_synthetic_residual_grid can pcolormesh them, same as the
    real branch's lon/lat grid), samples a known ground-truth covariance
    from ONE data_gen.py kernel (sample_simple_kernel_covariance), draws
    --n-synthetic-draws independent joint GP samples from it as in-context
    conditioning values, and extracts the model's predicted correlation via
    the SAME extract_model_context_correlation forward pass the real branch
    uses (averaged over the draws, mirroring the real branch's averaging
    over context days) -- then plots the SAME three diagnostics as the real
    branch: distance-vs-correlation, heatmap comparison (both via
    generate_plots.plot_spatial_correlation_diagnostics), and the
    ground-truth/oracle-correlation/predicted/independent field small-multiples
    (via plot_synthetic_residual_grid, sharing predict_copula_residual_field
    with the real branch's plot_residual_grid). The oracle-correlation row
    isolates correlation-estimation error from marginal-estimation error: it
    reuses the model's real TabICLv2 marginal (same as the predicted row) but
    injects the TRUE kernel's correlation matrix instead of the model's own
    R_pred, holding the shared latent noise fixed so the only difference from
    the predicted row is Sigma_hat (model) vs. Sigma_true (oracle).
    """
    import torch

    from generate_plots import _safe_cholesky
    from data_gen import sigma_to_correlation

    grid_size = max(2, int(round(np.sqrt(args.n_synthetic_points))))
    axis = np.linspace(-1.0, 1.0, grid_size)
    x_grid, y_grid = np.meshgrid(axis, axis)
    coords = np.column_stack([x_grid.ravel(), y_grid.ravel()])  # (D, 2)
    D = coords.shape[0]
    if D != args.n_synthetic_points:
        print(f"Note: --n-synthetic-points={args.n_synthetic_points} rounded to the nearest "
              f"square grid, D={D} ({grid_size}x{grid_size}), so it can be shown as a field grid.")

    print(f"Sampling a synthetic ground-truth covariance over {D} points from a single data_gen.py kernel...")
    true_cov, kernel_name = sample_simple_kernel_covariance(cfg, coords, seed=args.seed)

    # sample_simple_kernel_covariance returns a raw GP COVARIANCE (outputscale
    # sampled per-episode, so its diagonal is not generally 1), but
    # predict_copula_residual_field's R_context argument must be a unit-diagonal
    # CORRELATION matrix (it Cholesky-factors R_context and feeds norm.cdf of the
    # resulting latent straight in as the PIT quantile -- see low_rank_correlation
    # in src/model.py, whose Sigma output is likewise unit-diagonal). Reuse the
    # same covariance -> correlation normalization training uses (data_gen.py's
    # sigma_to_correlation) rather than re-deriving it here.
    R_true, _ = sigma_to_correlation(torch.as_tensor(true_cov, dtype=torch.float64))
    R_true = R_true.numpy()

    n_context = min(args.n_synthetic_context, D)
    context_frac = n_context / D
    if context_frac >= 0.01:
        print(f"Warning: --n-synthetic-context={n_context} is {context_frac:.1%} of the grid "
              f"(D={D}) -- consider raising --n-synthetic-points to keep the in-context sample "
              f"under 1% of the field.")
    context_idx = rng.choice(D, size=n_context, replace=False)
    context_coords = coords[context_idx]

    L = _safe_cholesky(true_cov)
    R_indep = np.eye(D)
    R_pred_draws, true_fields, predicted_fields, independent_fields, oracle_marginal_fields = [], [], [], [], []
    for i in range(args.n_synthetic_draws):
        z_true = L @ rng.standard_normal(D)
        context_values = z_true[context_idx]
        print(f"Extracting the joint copula correlation matrix with real context "
              f"(synthetic draw {i + 1}/{args.n_synthetic_draws})...")
        R_pred = extract_model_context_correlation(
            model, device, tabicl_marginal, context_coords, context_values, coords, k_folds=args.pit_k_folds,
        )
        R_pred_draws.append(R_pred)
        true_fields.append(z_true)

        # Same shared latent noise for all three draws below (see
        # predict_copula_residual_field's docstring): isolates what each of the
        # learned cross-location correlation (R_pred) and the oracle correlation
        # (R_true) add on top of the same per-point marginal prediction, rather
        # than an unrelated random redraw -- same convention the real branch's
        # main() loop uses.
        z_shared = rng.standard_normal(D)
        predicted_fields.append(
            predict_copula_residual_field(tabicl_marginal, context_coords, context_values, coords, R_pred, device, z_shared)
        )
        oracle_marginal_fields.append(
            predict_copula_residual_field(tabicl_marginal, context_coords, context_values, coords, R_true, device, z_shared)
        )
        independent_fields.append(
            predict_copula_residual_field(tabicl_marginal, context_coords, context_values, coords, R_indep, device, z_shared)
        )
    predicted_cov = np.mean(R_pred_draws, axis=0)

    print(f"Plotting synthetic ground-truth ('{kernel_name}' kernel) vs. predicted spatial correlation diagnostics...")
    plot_generic_diagnostics(predicted_cov, coords, true_cov=true_cov, tag=tag)

    print(f"Plotting the synthetic ground-truth vs. oracle-correlation vs. copula-model-predicted "
          f"fields for {args.n_synthetic_draws} draws on the same grid...")
    grid_shape = (grid_size, grid_size)
    true_grids = [f.reshape(grid_shape) for f in true_fields]
    plot_synthetic_residual_grid(
        axis, axis, grid_shape, true_grids, predicted_fields, independent_fields,
        os.path.join(_PLOTS_DIR, f"residual_grid_{tag}.png"), context_coords=context_coords,
        oracle_fields=oracle_marginal_fields,
    )


# ---------------------------------------------------------------------------
# Distance binning shared by every curve
# ---------------------------------------------------------------------------
def _bin_indices(d: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Bin index per distance, or -1 for out-of-[bin_edges[0], bin_edges[-1]] values.

    Pairs beyond bin_edges[-1] are DROPPED (index -1), not clipped into the last
    bin: bin_edges may be capped below dist.max() (see --max-dist-percentile) to
    exclude the corner-only, high-variance tail of a bounded non-periodic
    lat/lon domain, and clipping would silently pull that excluded tail back
    into the last visible bin under a misleadingly low distance label.
    """
    n_bins = len(bin_edges) - 1
    bin_idx = np.digitize(d, bin_edges) - 1
    bin_idx[(d < bin_edges[0]) | (d > bin_edges[-1])] = -1
    bin_idx[bin_idx == n_bins] = n_bins - 1  # d == bin_edges[-1] exactly
    return bin_idx


def bin_correlation_by_distance(R: np.ndarray, dist: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Mean correlation per distance bin, over the upper-triangle pairwise entries."""
    iu = np.triu_indices_from(R, k=1)
    corr, d = R[iu], dist[iu]
    n_bins = len(bin_edges) - 1
    bin_idx = _bin_indices(d, bin_edges)
    means = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            means[b] = corr[mask].mean()
    return means


def pair_counts_by_distance(dist: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Number of upper-triangle (i, j) pairs falling in each distance bin.

    On a bounded, non-periodic lat/lon rectangle (see AREA in generate_plots.py),
    the pair population thins out sharply near the max distance -- only the
    handful of pairs straddling opposite corners of the box reach it -- so a
    bin's mean correlation is only as trustworthy as its count here. This is a
    diagnostic for exactly that: low counts in the tail flag bins whose mean is
    a small-sample, corner-biased estimate rather than a real isotropic decay.
    """
    iu_dist = dist[np.triu_indices_from(dist, k=1)]
    n_bins = len(bin_edges) - 1
    bin_idx = _bin_indices(iu_dist, bin_edges)
    counts = np.bincount(bin_idx[bin_idx >= 0], minlength=n_bins)
    return counts[:n_bins]


# ---------------------------------------------------------------------------
# Correlation-matrix seriation: reorder the D grid points so strongly-
# correlated pairs sit near the diagonal in the heatmap comparison.
# ---------------------------------------------------------------------------
def seriate_by_correlation(R: np.ndarray) -> np.ndarray:
    """Permutation of R's indices via average-linkage hierarchical clustering
    with optimal leaf ordering (Bar-Joseph et al. 2001) on the correlation
    distance d_ij = 1 - rho_ij -- the standard "seriation" trick for making a
    correlation-matrix heatmap interpretable. The grid's raw index order
    (raveled lon/lat meshgrid, see main()) has no reason to place spatially
    close -- hence strongly correlated -- points at nearby indices, which is
    why the un-seriated heatmap shows diagonal-adjacent structure only for
    the block that happens to share a latitude row.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    dist = 1.0 - np.clip(R, -1.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce exact symmetry; R may only be numerically symmetric
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average", optimal_ordering=True)
    return np.asarray(leaves_list(Z))


# ---------------------------------------------------------------------------
# Theoretical spatial-correlation decay laws (North, Wang & Genton 2011;
# Whittle 1954; Matern 1960) -- an independent physical sanity check, fit
# ONLY to the ground-truth empirical curve and NOT derived from the model
# or (for the synthetic fallback) the GP's own generating kernel, so a good
# fit is informative rather than circular.
# ---------------------------------------------------------------------------
def exponential_law(r: np.ndarray, L: float) -> np.ndarray:
    """rho(r) = exp(-r / L) -- Hansen & Lebedeff (1987); Matern nu=1/2. Cusped
    (non-smooth) at r=0; the classic empirical baseline, fit this first."""
    return np.exp(-np.asarray(r, dtype=np.float64) / L)


def gaussian_law(r: np.ndarray, L: float) -> np.ndarray:
    """rho(r) = exp(-r^2 / (2 L^2)) -- Matern nu -> infinity, i.e. an
    infinitely smooth field. Usually too smooth for real temperature data."""
    r = np.asarray(r, dtype=np.float64)
    return np.exp(-(r ** 2) / (2.0 * L ** 2))


def matern_law(r: np.ndarray, L: float, nu: float) -> np.ndarray:
    """General Matern correlation: rho(r) = 2^(1-nu)/Gamma(nu) (r/L)^nu K_nu(r/L),
    rho(0)=1 by the x K_nu(x) -> ... limit (handled explicitly below since
    K_nu(0) itself diverges for nu>0). nu=1/2 reduces to exponential_law,
    nu=1 is whittle_law, nu->inf approaches gaussian_law. Free-nu Matern is
    usually the best unconstrained empirical fit (nu~1 typical for real
    temperature fields)."""
    r = np.asarray(r, dtype=np.float64)
    out = np.ones_like(r)
    nz = r > 0
    x = r[nz] / L
    out[nz] = (2.0 ** (1.0 - nu) / gamma_fn(nu)) * (x ** nu) * bessel_k(nu, x)
    return out


def whittle_law(r: np.ndarray, L: float) -> np.ndarray:
    """rho(r) = (r/L) K_1(r/L) -- Matern nu=1, AND the omega->0 (long-time-
    averaging) closed-form limit of the North, Wang & Genton (2011)
    energy-balance / damped-diffusion model (their Eq. 6). This is the one
    law here with an actual physical derivation behind its shape, not just
    a good empirical fit -- see the reference doc section 2.3."""
    return matern_law(r, L, nu=1.0)


# name -> (callable(r, *params), ordered param names); param order must match
# each callable's positional signature for curve_fit's p0/bounds below.
THEORY_LAWS = {
    "exponential": (exponential_law, ["L"]),
    "gaussian": (gaussian_law, ["L"]),
    "whittle": (whittle_law, ["L"]),
    "matern": (matern_law, ["L", "nu"]),
}

# name -> (color, linestyle, linewidth, display label) for the overlay plot.
# matern is drawn bolder/opaque since it's the general (free-nu) parent model
# every other law here is a special case of (see THEORY_LAWS docstrings).
THEORY_STYLE = {
    "exponential": ("purple", "-.", 1.6, "Exponential (Hansen & Lebedeff 1987, $\\nu$=1/2)"),
    "gaussian": ("saddlebrown", "--", 1.6, "Gaussian ($\\nu\\to\\infty$)"),
    "whittle": ("darkgreen", ":", 1.8, "Whittle / EBCM $\\omega\\to0$ limit (North et al. 2011, $\\nu$=1)"),
    "matern": ("magenta", "-", 2.4, "Matérn (free $\\nu$)"),
}

# Published reference decorrelation lengths L (km), independent of the fits
# above -- a literature sanity check, not derived from this dataset. Only
# 'exponential' and 'whittle' have a single reported number to compare
# against; 'gaussian' is used here only as an idealized, infinitely-smooth
# upper bound never fit to real temperature data in the literature, and free-
# nu 'matern' has no canonical L since nu itself varies by fit.
#   - exponential: North, Wang & Genton (2011, J. Climate 24), Fig. 1 -- their
#     own exponential refit of extratropical annual-mean surface-temperature
#     correlation ("modified from HL87"). Hansen & Lebedeff (1987) themselves
#     used a more conservative L~1000 km as a practical station-spacing
#     design choice, not a curve-fit estimate.
#   - whittle: North, Wang & Genton (2011), Fig. 2 -- their Whittle/EBCM fit
#     rK1(r) to annual-mean eastern-Siberia (land) data, reported as "about
#     50% larger than in HL87" (2800 / 1800 = 1.56, consistent with the
#     exponential number above).
LITERATURE_L = {
    "exponential": (1800.0, "North, Wang & Genton 2011, Fig. 1 (extratropical, exponential fit)"),
    "whittle": (2800.0, "North, Wang & Genton 2011, Fig. 2 (eastern Siberia, Whittle/EBCM fit)"),
}


def _correlation_length_guess(dist_centers: np.ndarray, rho: np.ndarray) -> float:
    """Initial L guess for curve_fit: distance at which the empirical curve
    crosses 1/e (the correlation-length definition used throughout the
    reference doc), by linear interpolation between the bracketing bin
    centers. Falls back to half the plotted distance range if the curve
    never crosses 1/e (e.g. too noisy or too short a distance range)."""
    valid = np.isfinite(rho)
    d, r = dist_centers[valid], rho[valid]
    below = np.where(r <= 1.0 / np.e)[0]
    if len(below) == 0 or below[0] == 0:
        return float(d[-1] / 2.0) if len(d) else 1000.0
    i = below[0]
    d0, d1, r0, r1 = d[i - 1], d[i], r[i - 1], r[i]
    if r0 == r1:
        return float(d0)
    frac = (1.0 / np.e - r0) / (r1 - r0)
    return float(d0 + frac * (d1 - d0))


def fit_theoretical_law(
    dist_centers: np.ndarray, rho_emp: np.ndarray, pair_counts: np.ndarray, model: str,
) -> "dict | None":
    """Nonlinear least-squares fit of `model` (see THEORY_LAWS) to the binned
    ground-truth empirical curve, weighted by sqrt(pair_counts) per bin: a
    bin's mean-correlation estimate is lower-variance the more pairs back it
    (see pair_counts_by_distance's docstring on the corner-biased tail), so
    high-count bins should pull the fit harder than sparse tail bins.

    Returns None (with a printed warning) instead of raising if curve_fit
    fails to converge or too few bins are populated, so the caller can just
    skip drawing the theory curve rather than crash the whole diagnostic
    plot over an unfittable curve.
    """
    if model not in THEORY_LAWS:
        raise ValueError(f"Unknown --theory-model '{model}', choose from {sorted(THEORY_LAWS)}.")
    law_fn, param_names = THEORY_LAWS[model]

    mask = np.isfinite(rho_emp) & (pair_counts > 0)
    if mask.sum() < len(param_names) + 1:
        print(f"Warning: too few valid distance bins ({mask.sum()}) to fit '{model}'; skipping theory curve.")
        return None
    d, r, n = dist_centers[mask], rho_emp[mask], pair_counts[mask]
    sigma = 1.0 / np.sqrt(n)  # SEM of a Pearson-r bin mean scales ~ 1/sqrt(n pairs)

    L0 = _correlation_length_guess(dist_centers, rho_emp)
    L_hi = max(50.0 * L0, 10.0 * float(dist_centers[-1]))
    if param_names == ["L"]:
        p0, bounds = [L0], ([1.0], [L_hi])
    else:  # ["L", "nu"]
        p0, bounds = [L0, 1.0], ([1.0, 0.05], [L_hi, 8.0])

    try:
        popt, _ = curve_fit(law_fn, d, r, p0=p0, sigma=sigma, bounds=bounds, maxfev=20000)
    except RuntimeError as exc:
        print(f"Warning: curve_fit failed to converge for '{model}' ({exc}); skipping theory curve.")
        return None

    pred = law_fn(d, *popt)
    resid = r - pred
    weighted_ss_res = float(np.sum((resid / sigma) ** 2))
    r_bar = np.average(r, weights=1.0 / sigma ** 2)
    weighted_ss_tot = float(np.sum(((r - r_bar) / sigma) ** 2))
    r_squared = 1.0 - weighted_ss_res / weighted_ss_tot if weighted_ss_tot > 0 else float("nan")

    return {"model": model, "law_fn": law_fn, "params": dict(zip(param_names, popt)), "r_squared": r_squared}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to a trained CopulaTabICL / TabICLv2 checkpoint "
        "(e.g. ./checkpoints/systematic-composition-8/step_0180000.pt).",
    )
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument(
        "--target", type=str, default="raw", choices=["raw", "residual"],
        help="[--mode real] Physical quantity to diagnose: 'raw' (default) fits the raw absolute "
        "temperature field Z_t itself; 'residual' fits the 24h persistence residual E_t = Z_t - "
        "Z_{t-24} instead (this script's original convention, kept for comparison -- differencing "
        "removes common-mode structure like the seasonal cycle that 'raw' keeps in). Governs the "
        "ground-truth curve (empirical_spatial_correlation), the real-context conditioning values, "
        "and the residual-grid small-multiples panel. Ignored in --mode synthetic (its ground truth "
        "is a sampled GP draw, not real temperature).",
    )
    parser.add_argument(
        "--mode", type=str, default="real", choices=["real", "synthetic"],
        help="'real' (default): the full ERA5/TabICLv2 diagnostic pipeline below, unchanged. "
        "'synthetic': skip ERA5 entirely and instead sample a known ground-truth covariance from "
        "ONE non-composite src/data_gen.py kernel (sample_simple_kernel_covariance) over a "
        "synthetic coordinate grid, then check whether the model's real forward pass "
        "(extract_model_context_correlation, same as the real branch) recovers it -- a "
        "no-real-data-confounds sanity check, producing the two generic diagnostics "
        "(distance-vs-correlation, heatmap comparison) via "
        "generate_plots.plot_spatial_correlation_diagnostics, plus a residual-field small-multiples "
        "panel (plot_synthetic_residual_grid, one column per --n-synthetic-draws draw) -- the "
        "synthetic analogue of --residual-grid-output below. --days/--theory-models/--output/"
        "--residual-grid-output are ignored in this mode (see --n-synthetic-points/--n-synthetic-draws "
        "instead); every output filename is auto-tagged with the checkpoint + mode (see "
        "_ckpt_mode_tag) so repeated comparisons across checkpoints/modes don't overwrite each other.",
    )
    parser.add_argument(
        "--n-synthetic-points", type=int, default=2500,
        help="[--mode synthetic] Number of synthetic 2D coordinates to sample the ground-truth "
        "kernel and model prediction over (rounded up to the nearest square grid -- see "
        "run_synthetic_mode). Default 2500 (50x50) so the default --n-synthetic-context=20 stays "
        "under 1%% of the grid -- a small, genuinely sparse in-context sample rather than a large "
        "fraction of the whole field.",
    )
    parser.add_argument(
        "--n-synthetic-context", type=int, default=20,
        help="[--mode synthetic] Number of historical context points sampled per draw (the "
        "synthetic-mode analogue of --n-context below, kept separate since the real ERA5 grid and "
        "the synthetic grid have very different natural sizes).",
    )
    parser.add_argument(
        "--n-synthetic-draws", type=int, default=12,
        help="[--mode synthetic] Number of independent joint GP draws from the sampled ground-truth "
        "covariance to average the model's predicted correlation over, and the number of columns "
        "shown in the residual-grid small-multiples panel (mirrors the real branch's "
        "averaging over --days).",
    )
    parser.add_argument(
        "--days",
        type=int,
        nargs="+",
        default=None,
        help="Day indices t providing the historical in-context sample: for each t, the model "
        "conditions on the --target quantity (raw temperature field[t] by default, or the 24h "
        "persistence residual field[t] - field[t-1] under --target residual -- same quantity as "
        "the ground-truth curve), sampled at --n-context grid points. Each must be >= 1 under "
        "--target residual (>= 0 under --target raw). One context-conditioned curve is computed "
        "per day and shown faint, plus their mean shown bold, so you can see whether the curve's "
        "shape is a systematic model behavior or single-day noise. Default: an evenly spaced "
        "spread of up to 8 days across the whole dataset.",
    )
    parser.add_argument("--n-context", type=int, default=2500, help="Number of historical context points sampled per day.")
    parser.add_argument(
        "--pit-k-folds", type=int, default=10,
        help="Number of disjoint folds for the K-fold leave-one-out PIT that turns real context "
        "values into z_train (src/pit.py::run_pit). Fixed, small K rather than true LOO -- see "
        "project convention.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=15, help="Number of spatial-distance bins.")
    parser.add_argument(
        "--max-dist-percentile", type=float, default=90.0,
        help="Cap the binned distance range at this percentile of all pairwise distances instead "
        "of the raw max. On a bounded, non-periodic lat/lon rectangle the pairs near the raw max "
        "distance are almost exclusively opposite-corner pairs -- a tiny, geometrically special, "
        "non-isotropic subset -- so their bin mean is a high-variance, corner-biased estimate, not "
        "a real feature of the spatial decay. Set to 100 to restore the old raw-max behavior.",
    )
    parser.add_argument(
        "--theory-models", type=str, nargs="+", default=["exponential", "gaussian", "whittle", "matern"],
        choices=["exponential", "gaussian", "whittle", "matern", "none"],
        help="Theoretical spatial-correlation decay law(s) to fit (weighted nonlinear least squares, each "
        "independently) to the ground-truth empirical curve and overlay, as a physical sanity check "
        "independent of the model (North, Wang & Genton 2011 / Whittle 1954 / Matern 1960). Default plots "
        "every law from the reference literature at once so their shapes can be compared directly: "
        "'exponential' (nu=1/2, Hansen & Lebedeff 1987 baseline), 'gaussian' (nu->inf, infinitely-smooth "
        "upper bound), 'whittle' (nu=1, the physically-derived long-averaging-limit EBCM law), 'matern' "
        "(free smoothness nu, the most flexible fit -- nu is estimated directly by the fit, not assumed). "
        "Pass 'none' alone to disable the overlay entirely.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="[--mode real] Where to save the correlogram PNG. Default: auto-tagged with the "
        "checkpoint + mode (see _ckpt_mode_tag), spatial_correlation_diagnostics_<tag>.png.",
    )
    parser.add_argument(
        "--residual-grid-output", type=str, default=None,
        help="[--mode real] Where to save the small-multiples panel of the --target field (raw "
        "temperature by default, or the 24h persistence residual under --target residual; one "
        "column per --days entry), on the same ERA5 grid used for the correlogram above. "
        "Default: auto-tagged, residual_grid_<tag>.png.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Loading TabICLv2 checkpoint '{args.ckpt}'...")
    model, cfg, device = load_copula_model(args.ckpt, device=args.device)

    print("Loading frozen pretrained TabICL quantile head for context PIT...")
    tabicl_marginal = load_marginal_tabicl(cfg, device)

    tag = _ckpt_mode_tag(args.ckpt, args.mode, target=args.target if args.mode == "real" else None)

    if args.mode == "synthetic":
        run_synthetic_mode(args, rng, model, cfg, device, tabicl_marginal, tag)
        return

    if args.output is None:
        args.output = os.path.join(_PLOTS_DIR, f"spatial_correlation_diagnostics_{tag}.png")
    if args.residual_grid_output is None:
        args.residual_grid_output = os.path.join(_PLOTS_DIR, f"residual_grid_{tag}.png")

    data = load_era5_data()
    lat, lon = data["latitude"], data["longitude"]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])  # (D, 2) = (lon, lat)
    D = coords.shape[0]

    n_days = data["t2m"].shape[0]
    # --target residual needs day d-1 to form E_t = Z_t - Z_{t-24}, so d must start at 1;
    # --target raw has no such dependency and can use day 0 too.
    day_min = 1 if args.target == "residual" else 0
    if args.days is None:
        n_pick = min(8, n_days - day_min)
        args.days = sorted(set(np.linspace(day_min, n_days - 1, n_pick).round().astype(int).tolist()))
    for d in args.days:
        if not (day_min <= d < n_days):
            parser.error(f"each --days value must be in [{day_min}, {n_days - 1}] "
                         f"(--target={args.target} needs day-1 to form a 24h residual), got {d}")

    print(f"Computing empirical spatial correlation from --target={args.target} observations...")
    observations = get_ground_truth_observations(data["t2m"], args.target)  # feeds both R_emp and Plot_generic's baseline
    R_emp = empirical_spatial_correlation(data, target=args.target)

    R_indep = np.eye(D)

    print("Extracting the model's unconditional correlation matrix (dummy context)...")
    R_dummy_context = extract_model_dummy_context_correlation(model, device, coords)

    dist = haversine_distance_km(coords)
    dist_iu = dist[np.triu_indices_from(dist, k=1)]
    max_dist = np.percentile(dist_iu, args.max_dist_percentile)
    bin_edges = np.linspace(0.0, max_dist, args.n_bins + 1)
    dist_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    pair_counts = pair_counts_by_distance(dist, bin_edges)
    print(f"Pairwise distances range [0, {dist_iu.max():.0f}] km; binning out to the "
          f"{args.max_dist_percentile:.0f}th percentile ({max_dist:.0f} km) to avoid the "
          f"corner-only, high-variance tail of the raw max. Pairs per bin:")
    for lo, hi, n in zip(bin_edges[:-1], bin_edges[1:], pair_counts):
        print(f"  [{lo:7.0f}, {hi:7.0f}) km: {n:8d} pairs")

    n_context = min(args.n_context, D)
    context_idx = rng.choice(D, size=n_context, replace=False)  # same context locations every day, only values change
    context_coords = coords[context_idx]

    rho_context_per_day = []
    R_context_per_day = []
    predicted_fields = []
    independent_fields = []
    for d in args.days:
        if args.target == "residual":
            print(f"Extracting the joint copula correlation matrix with real context (context day={d}, "
                  f"conditioned on the 24h persistence residual field[{d}] - field[{d - 1}])...")
            day_values = data["t2m"][d].ravel() - data["t2m"][d - 1].ravel()
        else:
            print(f"Extracting the joint copula correlation matrix with real context (context day={d}, "
                  f"conditioned on the raw temperature field[{d}])...")
            day_values = data["t2m"][d].ravel()
        context_values = day_values[context_idx]
        R_context = extract_model_context_correlation(
            model, device, tabicl_marginal, context_coords, context_values, coords, k_folds=args.pit_k_folds,
        )
        R_context_per_day.append(R_context)
        rho_context_per_day.append(bin_correlation_by_distance(R_context, dist, bin_edges))
        # Same shared latent noise for both draws below (see predict_copula_residual_field's
        # docstring): isolates what the learned cross-location correlation itself adds on
        # top of the per-point marginal prediction, rather than an unrelated random redraw.
        z_shared = rng.standard_normal(D)
        predicted_fields.append(
            predict_copula_residual_field(
                tabicl_marginal, context_coords, context_values, coords, R_context, device, z_shared,
            )
        )
        independent_fields.append(
            predict_copula_residual_field(
                tabicl_marginal, context_coords, context_values, coords, R_indep, device, z_shared,
            )
        )
    rho_context_per_day = np.array(rho_context_per_day)  # (n_days, n_bins)
    rho_context_mean = np.nanmean(rho_context_per_day, axis=0)
    R_context_mean = np.mean(R_context_per_day, axis=0)  # (D, D), elementwise mean matrix over context days

    rho_emp = bin_correlation_by_distance(R_emp, dist, bin_edges)
    rho_indep = bin_correlation_by_distance(R_indep, dist, bin_edges)
    rho_dummy_context = bin_correlation_by_distance(R_dummy_context, dist, bin_edges)

    theory_models = [] if args.theory_models == ["none"] else args.theory_models
    theory_fits = []
    for theory_model in theory_models:
        print(f"Fitting theoretical decay law '{theory_model}' to the ground-truth empirical curve...")
        fit = fit_theoretical_law(dist_centers, rho_emp, pair_counts, theory_model)
        if fit is not None:
            param_str = ", ".join(f"{k}={v:.0f} km" if k == "L" else f"{k}={v:.2f}"
                                   for k, v in fit["params"].items())
            print(f"  Fitted {theory_model}: {param_str}, weighted R^2={fit['r_squared']:.3f}")
            lit = LITERATURE_L.get(theory_model)
            if lit is not None:
                L_lit, citation = lit
                print(f"    Literature reference: L={L_lit:.0f} km ({citation})")
            theory_fits.append(fit)

    fig, (ax, ax_count) = plt.subplots(
        2, 1, figsize=(10.5, 7.3), sharex=True, height_ratios=[3.2, 1],
        gridspec_kw={"hspace": 0.08},
    )
    if args.target == "residual":
        ground_truth_label = ("Ground Truth: empirical corr. of real 24h residuals\n"
                               "$E_t = Z_t - Z_{t-24}$, averaged over all days")
    else:
        ground_truth_label = "Ground Truth: empirical corr. of the raw temperature field $Z_t$"
    ax.plot(dist_centers, rho_emp, "--", color="black", marker="o", label=ground_truth_label)
    r_dense = np.linspace(0.0, bin_edges[-1], 300)
    for fit in theory_fits:
        color, linestyle, linewidth, display_name = THEORY_STYLE[fit["model"]]
        rho_theory = fit["law_fn"](r_dense, *fit["params"].values())
        L_fit = fit["params"]["L"]
        nu_str = f", $\\nu$={fit['params']['nu']:.2f}" if "nu" in fit["params"] else ""
        lit = LITERATURE_L.get(fit["model"])
        lit_str = f", lit. $L$={lit[0]:.0f} km" if lit is not None else ""
        ax.plot(r_dense, rho_theory, linestyle, color=color, linewidth=linewidth,
                label=f"{display_name} fit to ground truth:\n"
                      f"$L$={L_fit:.0f} km{nu_str}{lit_str}, weighted $R^2$={fit['r_squared']:.3f}")
        ax.axvline(L_fit, color=color, linewidth=0.8, linestyle=linestyle, alpha=0.4)
        if lit is not None:
            # Densely-dotted vertical line (same color, no separate legend entry --
            # the literature L is already stated in the fit's own label above) marking
            # the published reference L, for a direct visual fit-vs-literature check.
            ax.axvline(lit[0], color=color, linewidth=1.4, linestyle=(0, (1, 1)), alpha=0.75)
    ax.plot(dist_centers, rho_indep, "-", color="red", marker="^",
            label="Independent TabICLv2: no copula, so $\\rho \\equiv 0$ by construction")
    ax.plot(dist_centers, rho_dummy_context, "-", color="tab:orange", marker="D",
            label="Copula model with dummy context")
    for i, (d, rho_d) in enumerate(zip(args.days, rho_context_per_day)):
        ax.plot(dist_centers, rho_d, "-", color="blue", alpha=0.18, linewidth=1,
                label=f"Copula model with {n_context} context points: individual days\n({len(args.days)} days shown faint)" if i == 0 else None)
    ax.plot(dist_centers, rho_context_mean, "-", color="blue", marker="s", linewidth=2.2,
            label=f"Copula model with {n_context} context points: mean over {len(args.days)} days")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1.0, 1.0)
    ax.set_title("Spatial Correlation Decay: Ground Truth vs. Copula Model (Independent / Dummy Context / With Context)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))

    bin_width = bin_edges[1] - bin_edges[0]
    ax_count.bar(dist_centers, pair_counts, width=bin_width * 0.9, color="steelblue", alpha=0.8)
    ax_count.set_yscale("log")
    ax_count.set_xlabel("Spatial distance (km)")
    ax_count.set_ylabel("Pairs per bin\n(log scale)", fontsize=8)
    ax_count.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")

    print("Seriating the correlation matrix (hierarchical clustering on the ground-truth "
          "correlation distance) so strongly-correlated grid points sit near the diagonal...")
    order = seriate_by_correlation(R_emp)
    print("Plotting predicted (mean over context days) vs. empirical correlation matrix "
          f"on the same {D}-point ERA5 grid, reordered by seriation...")
    plot_correlation_matrix_comparison(
        R_context_mean[np.ix_(order, order)], R_emp[np.ix_(order, order)], pdf=None,
        filename=f"correlation_matrix_comparison_{tag}.png",
    )

    print(f"Plotting the ground-truth vs. copula-model-predicted --target={args.target} fields for "
          f"days {args.days} on the same grid...")
    plot_residual_grid(
        data, args.days, predicted_fields, args.residual_grid_output,
        context_coords=context_coords, independent_fields=independent_fields, target=args.target,
    )

    print("Plotting the generic distance-vs-correlation / heatmap diagnostics "
          f"(baseline = empirical correlation of the real --target={args.target} observations)...")
    plot_generic_diagnostics(R_context_mean, coords, raw_observations=observations, tag=tag)


if __name__ == "__main__":
    main()
