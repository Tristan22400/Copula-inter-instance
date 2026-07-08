"""
Generate the three NeurIPS figures comparing an independent tabular foundation
model (TabICLv2 -- the real, pretrained TabICL regressor for the joint-
probability calibration plot; a Gaussian mock for the spatial-sample panel)
against a copula-based tabular foundation model (Copula-TFM, mocked unless
--ckpt is given) on a spatial temperature field.

Everything this script needs lives in this directory:
  - plots/generate_neurips_plots.py   (this file)
  - plots/era5_temperature.nc         (downloaded or synthetic input data)
  - plots/all_figures.pdf             (single combined output, one page per figure)

Usage:
    python plots/generate_neurips_plots.py
    python plots/generate_neurips_plots.py --ckpt ./checkpoints/copula_transformer_muon/step_0025000.pt
"""

import argparse
import os
import shutil
import sys
from typing import Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.io import netcdf_file
from scipy.special import gammaln, logsumexp
from scipy.stats import chi2, multivariate_normal, norm
from sklearn.gaussian_process.kernels import Matern

# ---------------------------------------------------------------------------
# NeurIPS formatting
# ---------------------------------------------------------------------------
# text.usetex requires a local LaTeX installation. We honor the requested
# style whenever LaTeX is available and degrade gracefully otherwise, so the
# script still runs end-to-end on machines without a TeX distribution.
_USE_TEX = shutil.which("latex") is not None
plt.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": _USE_TEX,
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)
if not _USE_TEX:
    print("Note: no LaTeX install found; falling back to text.usetex=False.")

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PLOTS_DIR, "era5_temperature.nc")
ALL_FIGURES_PATH = os.path.join(PLOTS_DIR, "all_figures.pdf")

RNG = np.random.default_rng(42)

# Bounding box used both for the real ERA5 request and the synthetic fallback:
# North, West, South, East (covers Western Europe, prone to summer heatwaves).
AREA = {"north": 60.0, "west": -10.0, "south": 35.0, "east": 30.0}


# ---------------------------------------------------------------------------
# Phase 1: Data acquisition (ERA5)
# ---------------------------------------------------------------------------
def download_era5(target_path=DATA_PATH, year="2023", month="07", n_days=30):
    """
    Download ERA5 2m temperature, daily at 12:00, over `n_days` summer days,
    for the Europe bounding box in AREA. Requires a configured `~/.cdsapirc`.
    """
    import cdsapi

    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": "2m_temperature",
            "year": year,
            "month": month,
            "day": [f"{d:02d}" for d in range(1, n_days + 1)],
            "time": "12:00",
            "area": [AREA["north"], AREA["west"], AREA["south"], AREA["east"]],
            "format": "netcdf",
        },
        target_path,
    )
    return target_path


def generate_synthetic_era5_field(target_path=DATA_PATH, grid_size=25, n_days=30, seed=42):
    """
    CRITICAL FALLBACK: synthesize a 2D spatial temperature field with a GP
    (Matern 5/2 kernel) so the script runs immediately without a CDS API key.
    Writes a NetCDF3-classic file via scipy.io (no extra geo dependencies).
    """
    rng = np.random.default_rng(seed)
    lat = np.linspace(AREA["south"], AREA["north"], grid_size)
    lon = np.linspace(AREA["west"], AREA["east"], grid_size)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    kernel = Matern(length_scale=8.0, nu=2.5)
    cov = kernel(coords) + 1e-6 * np.eye(coords.shape[0])
    L = np.linalg.cholesky(cov)

    # Warmer south, cooler north -- a simple physical trend under the GP anomaly.
    trend = 25.0 - 0.35 * (lat_grid - AREA["south"])
    fields = np.empty((n_days, grid_size, grid_size))
    for d in range(n_days):
        z = rng.standard_normal(coords.shape[0])
        anomaly = (L @ z).reshape(grid_size, grid_size)
        fields[d] = trend + anomaly

    f = netcdf_file(target_path, "w")
    f.history = "Synthetic ERA5-like field generated with a Matern-5/2 GP fallback."
    f.source = "synthetic_gp_fallback"
    f.createDimension("time", n_days)
    f.createDimension("latitude", grid_size)
    f.createDimension("longitude", grid_size)

    var_t2m = f.createVariable("t2m", "f8", ("time", "latitude", "longitude"))
    var_t2m[:] = fields
    var_lat = f.createVariable("latitude", "f8", ("latitude",))
    var_lat[:] = lat
    var_lon = f.createVariable("longitude", "f8", ("longitude",))
    var_lon[:] = lon
    var_time = f.createVariable("time", "i4", ("time",))
    var_time[:] = np.arange(n_days)
    f.close()
    return target_path


def load_era5_data():
    """Try the real ERA5 download; fall back to the synthetic GP field on any failure."""
    if not os.path.exists(DATA_PATH):
        try:
            download_era5()
            print(f"Downloaded real ERA5 data via cdsapi to {DATA_PATH}.")
        except Exception as exc:
            print(f"cdsapi download unavailable ({exc}); using synthetic GP fallback.")
            generate_synthetic_era5_field()
    else:
        print(f"Using cached data at {DATA_PATH}")

    f = netcdf_file(DATA_PATH, "r", mmap=False)
    t2m = f.variables["t2m"][:].astype(np.float64).copy()
    lat = f.variables["latitude"][:].astype(np.float64).copy()
    lon = f.variables["longitude"][:].astype(np.float64).copy()
    # `source` is only set by generate_synthetic_era5_field -- its absence means
    # this file came from a real cdsapi download, so we can't assume the
    # Matern-5/2 kernel is the true generating process for the ground-truth
    # correlation comparison plot.
    is_synthetic = getattr(f, "source", b"") == b"synthetic_gp_fallback"
    f.close()
    return {"t2m": t2m, "latitude": lat, "longitude": lon, "is_synthetic": is_synthetic}


# ---------------------------------------------------------------------------
# Phase 2: Model mocking / interface
# ---------------------------------------------------------------------------
def TabICLv2_Marginal(context_coords, context_values, coords_test):
    """
    Mock of an independent tabular foundation model: fits a marginal mean/std
    per test location from a labeled context set (X_train, Y_train) only --
    the same context/test split the real Copula-TFM gets -- with no
    cross-location covariance information. Test-point labels are never seen.
    """
    lat_train = context_coords[:, 1]
    design = np.column_stack([np.ones_like(lat_train), lat_train])
    coef, _, _, _ = np.linalg.lstsq(design, context_values, rcond=None)
    a, b = coef

    lat_test = coords_test[:, 1]
    mean = a + b * lat_test

    resid_std = max(float((context_values - (a + b * lat_train)).std()), 1e-6)
    std = np.full(coords_test.shape[0], resid_std)
    return mean, std


def TabICLv2_Regressor():
    """
    Construct the real, pretrained TabICL regressor -- default checkpoint
    'tabicl-regressor-v2-20260212.ckpt', the actual "v2" this script's
    TabICLv2 name refers to. Loading it once and reusing the instance across
    repeated .fit() calls (e.g. once per day) avoids reloading the backbone
    weights every time.
    """
    from tabicl import TabICLRegressor

    return TabICLRegressor()


def TabICLv2_Quantile(context_coords, context_values, coords_test, level=0.9, regressor=None):
    """
    TabICLv2's own quantile output -- queried directly from the real
    TabICL regressor (see TabICLv2_Regressor), not a Gaussian stand-in. It
    is fit in-context on (context_coords, context_values) and asked
    directly for its `level`-th percentile at coords_test via its own
    non-parametric quantile-spline output
    (tabicl._model.quantile_dist.QuantileDistribution) -- no mean/std, no
    norm.ppf inversion.

    Pass an existing `regressor` (from TabICLv2_Regressor) to reuse its
    loaded backbone weights across repeated calls; otherwise a fresh one is
    constructed, which reloads the checkpoint.
    """
    reg = regressor if regressor is not None else TabICLv2_Regressor()
    reg.fit(context_coords, context_values)
    return reg.predict(coords_test, output_type="quantiles", alphas=[level]).ravel()


def Copula_TFM(coords_test, length_scale=8.0, nu=2.5):
    """
    Mock of the Copula-TFM: predicts the full MxM correlation matrix for the
    test locations, as if learned in-context purely from spatial coordinates.
    """
    kernel = Matern(length_scale=length_scale, nu=nu)
    K = kernel(coords_test)
    d = np.sqrt(np.diag(K))
    C = K / np.outer(d, d)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return C


def ground_truth_correlation_matrix(data, coords_test):
    """
    Reference correlation matrix to score a predicted `C` against.

    Synthetic fallback field: `generate_synthetic_era5_field` draws each day
    from an *exact* Matern(length_scale=8.0, nu=2.5) GP, so the analytic
    kernel value at these coordinates IS the true correlation -- not an
    estimate. This is far less noisy than trying to estimate a full MxM
    (e.g. 625x625) correlation matrix from the handful of daily snapshots
    (n_days ~ 30 << M), which would be badly rank-deficient.

    Real ERA5 data has no known analytic covariance, so we fall back to the
    empirical Pearson correlation across daily snapshots, with a printed
    caveat that it is a small-sample estimate.
    """
    if data["is_synthetic"]:
        return Copula_TFM(coords_test, length_scale=8.0, nu=2.5)

    print(
        "Note: real ERA5 data has no analytic ground-truth correlation; using the "
        f"empirical across-day Pearson correlation from n_days={data['t2m'].shape[0]} "
        "snapshots, which is a noisy estimate when n_days << M."
    )
    M = coords_test.shape[0]
    samples = data["t2m"].reshape(data["t2m"].shape[0], M)
    return np.corrcoef(samples.T)


def _safe_cholesky(C, jitter=1e-6, max_tries=6):
    """Cholesky factor of C, adding diagonal jitter if it's not quite PSD.

    A real checkpoint's predicted correlation matrix is PSD by construction
    (low-rank-plus-diagonal, see src/model.py:low_rank_correlation) but float32
    round-trip can leave it *just* outside PSD, which np.linalg.cholesky
    rejects outright.
    """
    C_reg = C
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(C_reg)
        except np.linalg.LinAlgError:
            C_reg = C + jitter * np.eye(C.shape[0])
            jitter *= 10
    w, v = np.linalg.eigh(C_reg)
    return np.linalg.cholesky(v @ np.diag(np.clip(w, 1e-8, None)) @ v.T)


def build_copula_correlation_fn(ckpt_path, device, context_coords, context_values):
    """
    Build a `coords_test -> C_test` function backed by either the real,
    checkpointed Copula-TFM (`--ckpt`) or the Matern-kernel mock above.

    The real model (`src.model.CopulaTabICL`) is an in-context learner: it
    needs (X_train, Y_train) context alongside X_test to produce C_test, so
    `context_coords`/`context_values` (the same 50 context points used in the
    spatial map) are bound once here and reused for every plot.
    """
    if ckpt_path is None:
        return lambda coords_test: Copula_TFM(coords_test)

    import torch
    from omegaconf import OmegaConf

    repo_root = os.path.dirname(PLOTS_DIR)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from src.model import build_copula_transformer, low_rank_correlation

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    model = build_copula_transformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    print(f"Loaded real Copula-TFM checkpoint '{ckpt_path}' (step {ckpt.get('step')}) on {device}.")

    # The real model expects per-episode standardized features and PIT-style
    # standardized context labels z_train (Rasmussen & Williams-style GP
    # residuals during training). We don't have the ground-truth GP posterior
    # for real/synthetic ERA5 fields, so we approximate z_train with an
    # empirical standardization of the context temperatures -- reasonable
    # since the model only needs z_train ~ N(0, 1) marginally.
    x_mean = context_coords.mean(axis=0, keepdims=True)
    x_std = context_coords.std(axis=0, keepdims=True).clip(min=1e-8)
    x_train_norm = (context_coords - x_mean) / x_std

    y_std = max(context_values.std(), 1e-8)
    z_train = (context_values - context_values.mean()) / y_std

    x_train_t = torch.as_tensor(x_train_norm, dtype=torch.float32, device=device).unsqueeze(0)
    z_train_t = torch.as_tensor(z_train, dtype=torch.float32, device=device).unsqueeze(0)

    def corr_fn(coords_test):
        x_test_norm = (coords_test - x_mean) / x_std
        x_test_t = torch.as_tensor(x_test_norm, dtype=torch.float32, device=device).unsqueeze(0)
        batch = {"x_train": x_train_t, "x_test": x_test_t, "z_train": z_train_t}
        with torch.no_grad():
            out = model(batch)
            Sigma = low_rank_correlation(out["W"], out["s"], jitter=1e-4)
        return Sigma[0].cpu().numpy()

    return corr_fn


def _save_fig(fig, filename, pdf, **savefig_kwargs):
    """
    Save `fig` as one page of the combined `pdf` (a PdfPages) if given,
    otherwise fall back to writing it as its own standalone file under
    PLOTS_DIR -- keeps each plot function usable/testable on its own.
    """
    if pdf is not None:
        pdf.savefig(fig, **savefig_kwargs)
        plt.close(fig)
        print(f"Saved {filename} page to {ALL_FIGURES_PATH}")
    else:
        out_path = os.path.join(PLOTS_DIR, filename)
        fig.savefig(out_path, **savefig_kwargs)
        plt.close(fig)
        print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 1: Static vs. Smooth spatial map
# ---------------------------------------------------------------------------
def plot_spatial_map_comparison(data, day, context_idx, C, pdf=None):
    field = data["t2m"][day]
    lat, lon = data["latitude"], data["longitude"]
    grid_size = field.shape[0]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    field_flat = field.ravel()
    M = coords.shape[0]
    context_coords = coords[context_idx]
    context_values = field_flat[context_idx]

    mean, std = TabICLv2_Marginal(context_coords, context_values, coords)

    # Both samples share the SAME underlying white noise draw z_shared -- the
    # only difference between the two panels is whether spatial correlation
    # is injected before the PIT. Drawing z_indep/z_copula independently
    # would compare two unrelated random realizations instead of showing what
    # each model does with the same latent draw.
    z_shared = RNG.standard_normal(M)

    # Independent sample (TabICLv2 assumption): treat z_shared as already iid N(0, I).
    u_indep = norm.cdf(z_shared)
    indep_sample = norm.ppf(u_indep, loc=mean, scale=std)

    # Copula sample: inject spatial correlation into the SAME z_shared via chol(C).
    L = _safe_cholesky(C)
    z_copula = L @ z_shared
    u_copula = norm.cdf(z_copula)
    copula_sample = norm.ppf(u_copula, loc=mean, scale=std)

    fields = [
        field,
        indep_sample.reshape(grid_size, grid_size),
        copula_sample.reshape(grid_size, grid_size),
    ]
    titles = ["True Field", "TabICLv2 (Independent Sample)", "Copula-TFM (Joint Sample)"]
    vmin, vmax = field.min(), field.max()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    mesh = None
    for ax, f, title in zip(axes, fields, titles):
        mesh = ax.pcolormesh(lon, lat, f, cmap="coolwarm", vmin=vmin, vmax=vmax, shading="auto")
        ax.scatter(context_coords[:, 0], context_coords[:, 1], c="black", s=8, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.tight_layout()
    fig.colorbar(mesh, ax=axes.tolist(), shrink=0.85, label="Temperature (deg C)")
    _save_fig(fig, "spatial_map_comparison.pdf", pdf)


# ---------------------------------------------------------------------------
# Plot 2: Correlation vs. distance (implicit variogram)
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(coords):
    """
    Great-circle distance (km) between every pair of (lon, lat) points in
    `coords` (degrees). `coords` here -- and everywhere else in this script --
    is plain Euclidean (lon, lat) in degrees, which is what the Matern kernel
    mock (`Copula_TFM`) and the real model both actually condition on; this
    haversine version is only for turning that into a physically meaningful
    x-axis (km) on the correlation-vs-distance plot.
    """
    lon_rad, lat_rad = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def plot_correlation_vs_distance(data, C, pdf=None):
    lat, lon = data["latitude"], data["longitude"]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    iu = np.triu_indices_from(C, k=1)
    corr = C[iu]
    dist = haversine_distance_km(coords)[iu]

    n_pairs = len(dist)
    max_scatter_points = 4000
    if n_pairs > max_scatter_points:
        sel = RNG.choice(n_pairs, size=max_scatter_points, replace=False)
        dist_plot, corr_plot = dist[sel], corr[sel]
    else:
        dist_plot, corr_plot = dist, corr

    order = np.argsort(dist)
    dist_sorted, corr_sorted = dist[order], corr[order]
    window = max(n_pairs // 50, 1)
    box = np.ones(window) / window
    moving_avg = np.convolve(corr_sorted, box, mode="valid")
    dist_avg = np.convolve(dist_sorted, box, mode="valid")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(dist_plot, corr_plot, s=4, alpha=0.15, color="steelblue", label="Predicted pairs")
    ax.plot(dist_avg, moving_avg, color="crimson", linewidth=2, label="Moving average")
    ax.set_xlabel("Spatial distance (km)")
    ax.set_ylabel(r"Predicted correlation $\rho_{ij}$")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, "correlation_vs_distance.pdf", pdf)


# ---------------------------------------------------------------------------
# Plot 3: Predicted vs. ground-truth correlation matrix
# ---------------------------------------------------------------------------
def plot_correlation_matrix_comparison(C, C_true, pdf=None):
    """
    Directly visualize the model's predicted MxM correlation matrix next to
    the ground-truth reference (see `ground_truth_correlation_matrix`), plus
    a scatter of every pairwise entry with the Pearson r / RMSE between them
    -- the actual quantitative link between "predicted correlation" and
    "ground truth" the model is trying to recover.
    """
    iu = np.triu_indices_from(C, k=1)
    pred_flat, true_flat = C[iu], C_true[iu]
    rmse = float(np.sqrt(np.mean((pred_flat - true_flat) ** 2)))
    mae = float(np.mean(np.abs(pred_flat - true_flat)))
    pearson_r = float(np.corrcoef(pred_flat, true_flat)[0, 1])
    print(
        f"Correlation-matrix agreement vs. ground truth: Pearson r={pearson_r:.3f}, "
        f"RMSE={rmse:.3f}, MAE={mae:.3f} (over {len(pred_flat)} pairs)."
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    im0 = axes[0].imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_title("Predicted Copula-TFM $\\hat{C}$")
    im1 = axes[1].imshow(C_true, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("Ground-truth $C$")
    for ax in axes[:2]:
        ax.set_xlabel("Grid index $j$")
        ax.set_ylabel("Grid index $i$")

    n_scatter = 5000
    if len(pred_flat) > n_scatter:
        sel = RNG.choice(len(pred_flat), size=n_scatter, replace=False)
        sp, st = pred_flat[sel], true_flat[sel]
    else:
        sp, st = pred_flat, true_flat
    axes[2].scatter(st, sp, s=4, alpha=0.15, color="steelblue")
    axes[2].plot([-1, 1], [-1, 1], "k--", linewidth=1, label="Ideal ($y=x$)")
    axes[2].set_xlim(-1, 1)
    axes[2].set_ylim(-1, 1)
    axes[2].set_xlabel(r"Ground-truth $\rho_{ij}$")
    axes[2].set_ylabel(r"Predicted $\hat{\rho}_{ij}$")
    axes[2].set_title(f"$r={pearson_r:.2f}$, RMSE$={rmse:.3f}$")
    axes[2].legend()

    plt.tight_layout()
    fig.colorbar(im1, ax=axes[:2].tolist(), shrink=0.85, label="Correlation")
    _save_fig(fig, "correlation_matrix_comparison.pdf", pdf)


# ---------------------------------------------------------------------------
# Plot 4: Calibration of joint probabilities
# ---------------------------------------------------------------------------
def plot_joint_probability_calibration(data, C, context_idx, n_pairs=400, n_bins=10, pdf=None):
    field_all = data["t2m"]
    n_days, grid_size, _ = field_all.shape
    lat, lon = data["latitude"], data["longitude"]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    M = coords.shape[0]

    samples = field_all.reshape(n_days, M)
    context_coords = coords[context_idx]

    # TabICLv2 (the real, pretrained TabICLRegressor -- see TabICLv2_Quantile)
    # is re-run as a fresh in-context episode for every day: its context
    # LABELS (the true temperatures at the same 50 context locations) are
    # taken from that specific day, so its declared q90[d, i] threshold is
    # refit daily -- exactly as in a real deployment where the model is
    # re-queried once per day rather than fit once (on a single day) and
    # reused across days it never saw. The quantile is the model's own
    # direct output (its non-parametric quantile spline), not derived from a
    # mean/std pair via norm.cdf/norm.ppf: by definition TabICLv2 declares
    # P(X_i > q90[d, i]) = 0.10 on every day. (The norm.ppf a few lines below
    # is unrelated to TabICLv2 -- it's the standard-normal quantile transform
    # that the Gaussian-copula combination formula itself requires to fold
    # in Copula-TFM's correlation C[i,j].) No marginal-miscalibration
    # confound leaks into the comparison against Copula-TFM below -- only
    # the (mis)modeled dependence structure does.
    tabiclv2 = TabICLv2_Regressor()
    q90 = np.stack(
        [
            TabICLv2_Quantile(
                context_coords, field_all[d].ravel()[context_idx], coords, level=0.9, regressor=tabiclv2
            )
            for d in range(n_days)
        ]
    )
    exceed_mask = samples > q90
    p_exceed = 0.10

    idx_i = RNG.integers(0, M, size=n_pairs)
    idx_j = RNG.integers(0, M, size=n_pairs)
    keep = idx_i != idx_j
    idx_i, idx_j = idx_i[keep], idx_j[keep]

    z = norm.ppf(1 - p_exceed)
    empirical, pred_indep, pred_copula = [], [], []
    for i, j in zip(idx_i, idx_j):
        empirical.append(np.mean(exceed_mask[:, i] & exceed_mask[:, j]))
        pred_indep.append(p_exceed * p_exceed)

        rho = C[i, j]
        cov = [[1.0, rho], [rho, 1.0]]
        # P(Z_i > z, Z_j > z) = P(-Z_i < -z, -Z_j < -z) by symmetry of N(0, cov).
        pred_copula.append(multivariate_normal.cdf([-z, -z], mean=[0, 0], cov=cov))

    empirical, pred_indep, pred_copula = map(np.array, (empirical, pred_indep, pred_copula))

    def bin_calibration(pred, emp):
        bins = np.linspace(0, max(pred.max(), 1e-6), n_bins + 1)
        bin_idx = np.clip(np.digitize(pred, bins) - 1, 0, n_bins - 1)
        pred_means, emp_means = [], []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.any():
                pred_means.append(pred[mask].mean())
                emp_means.append(emp[mask].mean())
        return np.array(pred_means), np.array(emp_means)

    pi_indep, ei_indep = bin_calibration(pred_indep, empirical)
    pi_copula, ei_copula = bin_calibration(pred_copula, empirical)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lim = max(pred_indep.max(), pred_copula.max(), empirical.max(), 1e-6) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Ideal calibration ($y=x$)")
    ax.plot(pi_indep, ei_indep, "o-", color="tab:orange", label="TabICLv2 (independent)")
    ax.plot(pi_copula, ei_copula, "s-", color="tab:green", label="Copula-TFM")
    ax.set_xlabel("Predicted joint exceedance probability")
    ax.set_ylabel("Empirical joint exceedance frequency")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, "joint_probability_calibration.pdf", pdf)


# ---------------------------------------------------------------------------
# Plot 5: Quantile calibration (reliability diagram)
# ---------------------------------------------------------------------------
def compute_quantile_ece(
    y_true: np.ndarray,
    y_pred_quantiles: np.ndarray,
    quantiles: Sequence[float],
) -> Tuple[float, np.ndarray]:
    """
    Compute the quantile-regression Expected Calibration Error (ECE).

    For each nominal quantile level, the empirical coverage is the fraction
    of instances where the true value falls at or below the predicted
    quantile value. The ECE is the mean absolute gap between the nominal
    levels and their empirical coverages (0 for perfect calibration).

    Args:
        y_true: 1D array/Series of shape (n_samples,) with observed targets.
        y_pred_quantiles: 2D array/DataFrame of shape (n_samples, n_quantiles);
            column k holds the predicted value for quantiles[k].
        quantiles: Nominal quantile levels in (0, 1), e.g. [0.1, ..., 0.9].

    Returns:
        Tuple (ece, empirical_coverage), where empirical_coverage has shape
        (n_quantiles,) and aligns positionally with `quantiles`.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred_quantiles = np.asarray(y_pred_quantiles)
    quantiles = np.asarray(quantiles, dtype=float)

    if y_pred_quantiles.ndim != 2:
        raise ValueError("y_pred_quantiles must be 2D (n_samples, n_quantiles).")
    if y_pred_quantiles.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"y_true has {y_true.shape[0]} samples but y_pred_quantiles has "
            f"{y_pred_quantiles.shape[0]}."
        )
    if y_pred_quantiles.shape[1] != quantiles.shape[0]:
        raise ValueError(
            f"y_pred_quantiles has {y_pred_quantiles.shape[1]} quantile columns "
            f"but {quantiles.shape[0]} nominal quantiles were given."
        )

    empirical_coverage = np.mean(y_true[:, None] <= y_pred_quantiles, axis=0)
    ece = float(np.mean(np.abs(quantiles - empirical_coverage)))
    return ece, empirical_coverage


def generate_era5_reliability_diagram(
    y_true: np.ndarray,
    y_pred_quantiles: np.ndarray,
    quantiles: Sequence[float],
    pdf: "PdfPages | None" = None,
) -> float:
    """
    Build and save a reliability diagram for TabICLv2's ERA5 quantile
    predictions, with nominal quantile on the x-axis and empirical coverage
    on the y-axis.

    Args:
        y_true: 1D array/Series of observed ERA5 targets.
        y_pred_quantiles: 2D array/DataFrame (n_samples, n_quantiles) of
            predicted quantile values.
        quantiles: Nominal quantile levels matching the columns above.
        pdf: Combined PdfPages to append this figure to as a page; if None,
            the figure is saved as its own standalone file instead.

    Returns:
        The scalar ECE score (also annotated on the figure).
    """
    ece, empirical_coverage = compute_quantile_ece(y_true, y_pred_quantiles, quantiles)
    quantiles = np.asarray(quantiles, dtype=float)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(quantiles, empirical_coverage, "o-", color="tab:blue", label="TabICLv2")
    ax.fill_between(
        quantiles,
        quantiles,
        empirical_coverage,
        color="tab:blue",
        alpha=0.2,
        label="Calibration gap",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal quantile")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Reliability Diagram: TabICLv2 on ERA5 Dataset")
    ax.text(
        0.05,
        0.95,
        f"ECE = {ece:.4f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85),
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save_fig(fig, "era5_reliability_diagram.pdf", pdf, bbox_inches="tight")
    print(f"ECE={ece:.4f}")
    return ece


def plot_era5_quantile_reliability(
    data,
    context_idx,
    quantiles=np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]),
    pdf=None,
):
    """
    Real-data driver for the reliability diagram: context locations
    (`context_idx`) are fixed once, but -- exactly as in
    `plot_joint_probability_calibration` -- the context LABELS are re-drawn
    from that day's field and TabICLv2 is re-queried per day, so we iterate
    over every timestamp in the dataset to build up (y_true, y_pred_quantiles)
    across all days x grid points before scoring calibration.

    The tail levels 0.01/0.05/0.95/0.99 are included alongside the 0.1-step
    grid so the diagram also reports calibration in the extreme tails, not
    just the bulk of the distribution.

    Evaluation is restricted to grid points OUTSIDE `context_idx`: querying
    the model at its own context locations would hand it the true label as a
    training example and then score it against that same label, inflating
    the apparent coverage. All quantile levels for a given day are requested
    from a single fit()+predict() call -- TabICL's quantile spline comes from
    one backbone forward pass regardless of how many `alphas` are requested,
    so refitting/re-forwarding once per quantile level would be redundant.
    """
    field_all = data["t2m"]
    n_days = field_all.shape[0]
    lat, lon = data["latitude"], data["longitude"]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    context_coords = coords[context_idx]

    target_idx = np.setdiff1d(np.arange(coords.shape[0]), context_idx)
    target_coords = coords[target_idx]

    quantiles = np.asarray(quantiles, dtype=float)
    tabiclv2 = TabICLv2_Regressor()
    y_true_chunks, y_pred_chunks = [], []
    for d in range(n_days):
        context_values = field_all[d].ravel()[context_idx]
        tabiclv2.fit(context_coords, context_values)
        preds = tabiclv2.predict(
            target_coords, output_type="quantiles", alphas=list(quantiles)
        )  # (n_target, n_quantiles)
        y_true_chunks.append(field_all[d].ravel()[target_idx])
        y_pred_chunks.append(preds)

    y_true = np.concatenate(y_true_chunks)
    y_pred_quantiles = np.concatenate(y_pred_chunks, axis=0)

    generate_era5_reliability_diagram(y_true, y_pred_quantiles, quantiles, pdf)


# ---------------------------------------------------------------------------
# Plot 6: Multivariate (joint) spatial calibration
#
# TabICLv2 emits independent marginal quantiles per row, so its declared
# joint predictive distribution over a D-dimensional target patch is the
# Independence Copula: H_i(y_i) = prod_d F_{i,d}(y_{i,d}). The four
# diagnostics below all probe whether that independence assumption (and the
# marginals feeding it) are jointly well-calibrated against the true,
# spatially-correlated field -- not just marginally calibrated per grid cell
# (which `compute_quantile_ece` above already checks).
# ---------------------------------------------------------------------------


def calc_kendall_pit(cdf_values: np.ndarray) -> np.ndarray:
    """
    Copula PIT (Kendall's transform) for the independence copula.

    Under H_i(y_i) = prod_d F_{i,d}(y_{i,d}), the copula-level "witness"
    w_i = prod_d F_{i,d}(y_{i,d}) is Uniform(0, 1)-distributed at every
    dimension only if the joint model is exactly correct; W = prod of D iid
    U(0,1) is NOT itself uniform (it concentrates near 0), so w_i must be
    passed through W's own CDF -- the Kendall distribution function -- to
    recover a Uniform(0, 1) PIT value:

        z_i = w_i * sum_{k=0}^{D-1} (-ln w_i)^k / k!

    (equivalently, -ln(w_i) ~ Gamma(D, 1) under H0, and z_i is its survival
    function evaluated at -ln(w_i)). z_i ~ Uniform(0, 1) iff the independence
    copula is correctly calibrated.

    Args:
        cdf_values: (n_samples, D) array of F_{i,d}(y_{i,d}) marginal CDF
            values, one row per instance and one column per spatial
            dimension.

    Returns:
        (n_samples,) array of Kendall PIT values z_i in [0, 1].
    """
    cdf_values = np.clip(np.asarray(cdf_values, dtype=np.float64), 1e-300, 1.0)
    n, D = cdf_values.shape

    log_w = np.log(cdf_values).sum(axis=1)  # log(w_i) = sum_d log F_{i,d}(y_{i,d})
    x = np.clip(-log_w, 0.0, None)  # x_i = -ln(w_i) >= 0

    # log of the k-th series term x^k / k!, summed via logsumexp for numerical
    # stability (naive summation can overflow/underflow for large D or x).
    k = np.arange(D, dtype=np.float64)
    with np.errstate(divide="ignore"):
        log_terms = k[None, :] * np.log(x)[:, None] - gammaln(k + 1.0)[None, :]
    # x_i == 0 (w_i == 1) means k*log(0) is -inf for k>0 and the 0**0
    # convention gives 1 for k=0 -- set that row explicitly rather than rely
    # on IEEE 0*-inf/log(0) arithmetic.
    zero_mask = x == 0.0
    log_terms[zero_mask, :] = -np.inf
    log_terms[zero_mask, 0] = 0.0

    log_series = logsumexp(log_terms, axis=1)  # log sum_k x^k / k!
    z = np.exp(log_w + log_series)  # z_i = w_i * series
    return np.clip(z, 0.0, 1.0)


def plot_kendall_pit(z_values: np.ndarray, ax, n_bins: int = 20):
    """Histogram of Kendall PIT values against the theoretical Uniform(0, 1) density."""
    z_values = np.asarray(z_values, dtype=np.float64)
    ax.hist(
        z_values,
        bins=n_bins,
        range=(0.0, 1.0),
        density=True,
        color="tab:blue",
        alpha=0.75,
        edgecolor="white",
        label="Empirical",
    )
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="Uniform(0, 1)")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Kendall PIT value $z$")
    ax.set_ylabel("Density")
    ax.set_title("Kendall PIT Histogram (Independence Copula)")
    ax.legend()
    return ax


def calc_mahalanobis_distances(
    y_true: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    """
    Mahalanobis distance under a diagonal covariance (Gaussian marginals,
    independence copula): d_i^2 = sum_d (y_{i,d} - mu_{i,d})^2 / sigma^2_{i,d}.
    Under perfect calibration, d_i^2 ~ chi^2_D.

    Args:
        y_true, means, variances: (n_samples, D) arrays.

    Returns:
        (n_samples,) array of squared Mahalanobis distances.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    if not (y_true.shape == means.shape == variances.shape):
        raise ValueError(
            f"y_true, means, variances must share shape (n_samples, D); got "
            f"{y_true.shape}, {means.shape}, {variances.shape}."
        )
    return np.sum((y_true - means) ** 2 / variances, axis=1)


def plot_mahalanobis_pp(distances: np.ndarray, dim_d: int, ax):
    """
    PP-plot (probability-probability plot) of squared Mahalanobis distances
    against the theoretical chi^2_D CDF: for the i-th order statistic
    d^2_(i), plot F_{chi^2_D}(d^2_(i)) against the empirical percentile
    (i - 0.5) / n. Points on the y = x diagonal indicate correct calibration.
    """
    distances = np.sort(np.asarray(distances, dtype=np.float64))
    n = distances.shape[0]
    empirical_p = (np.arange(1, n + 1) - 0.5) / n
    theoretical_p = chi2.cdf(distances, df=dim_d)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal calibration")
    ax.plot(theoretical_p, empirical_p, color="tab:blue", linewidth=1.5, label=f"$\\chi^2_{{{dim_d}}}$ PP-plot")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Theoretical $\chi^2_D$ CDF")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"Mahalanobis $\\chi^2$ PP-Plot ($D={dim_d}$)")
    ax.legend()
    return ax


def calc_exceedance_probs(y_true: np.ndarray, cdf_func, thresholds: np.ndarray):
    """
    Spatial exceedance events and their independence-copula predicted
    probabilities, for a set of thresholds tau_m:

        e_{i,m} = 1[max_d y_{i,d} > tau_m]                (true event)
        pi_{i,m} = 1 - prod_d F_{i,d}(tau_m)              (predicted prob)

    Args:
        y_true: (n_samples, D) array of observed targets.
        cdf_func: callable, cdf_func(tau) -> (n_samples, D) array of marginal
            CDF values F_{i,d}(tau) at threshold tau, for every instance i
            and dimension d.
        thresholds: (n_thresholds,) array of thresholds tau_m.

    Returns:
        Tuple (predicted_probs, true_events), both (n_samples, n_thresholds).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    max_y = y_true.max(axis=1)  # (n_samples,)
    true_events = (max_y[:, None] > thresholds[None, :]).astype(np.float64)

    predicted_probs = np.empty((y_true.shape[0], thresholds.shape[0]), dtype=np.float64)
    for m, tau in enumerate(thresholds):
        F = np.clip(np.asarray(cdf_func(tau), dtype=np.float64), 0.0, 1.0)
        predicted_probs[:, m] = 1.0 - np.prod(F, axis=1)
    return predicted_probs, true_events


def plot_spatial_reliability(predicted_probs: np.ndarray, true_events: np.ndarray, num_bins: int, ax):
    """Binned reliability diagram of predicted vs. empirical spatial exceedance probability."""
    pred_flat = np.asarray(predicted_probs, dtype=np.float64).ravel()
    true_flat = np.asarray(true_events, dtype=np.float64).ravel()

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_idx = np.clip(np.digitize(pred_flat, bins) - 1, 0, num_bins - 1)
    bin_pred, bin_true = [], []
    for b in range(num_bins):
        mask = bin_idx == b
        if mask.any():
            bin_pred.append(pred_flat[mask].mean())
            bin_true.append(true_flat[mask].mean())

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal calibration")
    ax.plot(bin_pred, bin_true, "o-", color="tab:green", label="Spatial exceedance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Predicted exceedance probability $\pi_{i,m}$")
    ax.set_ylabel("Empirical exceedance frequency")
    ax.set_title("Spatial Exceedance Reliability Diagram")
    ax.legend()
    return ax


def calc_spatial_coverage(y_true: np.ndarray, q_lower: np.ndarray, q_upper: np.ndarray) -> float:
    """
    Empirical spatial (joint) central coverage: the fraction of instances
    where EVERY spatial dimension falls within its own central interval,
    P_hat_c = (1/N) sum_i 1[forall d, y_{i,d} in [q_lower_{i,d}, q_upper_{i,d}]].

    Args:
        y_true, q_lower, q_upper: (n_samples, D) arrays.

    Returns:
        Scalar empirical coverage in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    q_lower = np.asarray(q_lower, dtype=np.float64)
    q_upper = np.asarray(q_upper, dtype=np.float64)
    inside = (y_true >= q_lower) & (y_true <= q_upper)  # (n_samples, D)
    return float(inside.all(axis=1).mean())


def plot_spatial_coverage_curve(y_true: np.ndarray, quantile_func, nominal_coverages: np.ndarray, ax):
    """
    Spatial central coverage curve: for each nominal coverage c = 1 - alpha,
    query `quantile_func(alpha)` for the per-dimension central interval
    [Q(alpha/2), Q(1 - alpha/2)] and plot the empirical joint coverage
    (`calc_spatial_coverage`) against c. Points below the y = x diagonal mean
    the joint intervals are too narrow (overconfident independence copula).

    Args:
        y_true: (n_samples, D) array of observed targets.
        quantile_func: callable, quantile_func(alpha) -> (q_lower, q_upper),
            each (n_samples, D), the per-dimension alpha/2 and 1 - alpha/2
            predictive quantiles.
        nominal_coverages: (n_levels,) array of nominal coverages c in (0, 1).
        ax: matplotlib axis to draw on.
    """
    nominal_coverages = np.asarray(nominal_coverages, dtype=np.float64)
    empirical = np.empty_like(nominal_coverages)
    for i, c in enumerate(nominal_coverages):
        alpha = 1.0 - c
        q_lower, q_upper = quantile_func(alpha)
        empirical[i] = calc_spatial_coverage(y_true, q_lower, q_upper)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal calibration")
    ax.plot(nominal_coverages, empirical, "o-", color="tab:purple", label="Spatial coverage")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Nominal joint coverage $c = 1 - \alpha$")
    ax.set_ylabel(r"Empirical spatial coverage $\hat{P}_c$")
    ax.set_title("Spatial Central Coverage Curve")
    ax.legend()
    return ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to a trained Copula-TFM checkpoint (e.g. "
        "./checkpoints/copula_transformer_muon/step_0025000.pt). "
        "If omitted, a Matern-kernel mock stands in for the model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device for real checkpoint inference (default: auto-detect). Ignored without --ckpt.",
    )
    parser.add_argument(
        "--day",
        type=int,
        default=0,
        help="Day index (0-based, into the n_days axis of era5_temperature.nc) to treat as the "
        "'observed' field: it's the True Field panel, and the source of context values "
        "(X_train, Y_train) both for the real Copula-TFM checkpoint and for the "
        "TabICLv2_Marginal fit used in the calibration plot. Default: 0.",
    )
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    data = load_era5_data()

    n_days = data["t2m"].shape[0]
    if not (0 <= args.day < n_days):
        parser.error(f"--day must be in [0, {n_days - 1}], got {args.day}")
    day = args.day

    field0 = data["t2m"][day]
    lat, lon = data["latitude"], data["longitude"]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    M = coords.shape[0]
    n_context = min(50, M)
    context_idx = RNG.choice(M, size=n_context, replace=False)
    context_coords = coords[context_idx]
    context_values = field0.ravel()[context_idx]

    corr_fn = build_copula_correlation_fn(args.ckpt, args.device, context_coords, context_values)
    C = corr_fn(coords)
    C_true = ground_truth_correlation_matrix(data, coords)

    with PdfPages(ALL_FIGURES_PATH) as pdf:
        plot_spatial_map_comparison(data, day, context_idx, C, pdf=pdf)
        plot_correlation_vs_distance(data, C, pdf=pdf)
        plot_correlation_matrix_comparison(C, C_true, pdf=pdf)
        plot_joint_probability_calibration(data, C, context_idx, pdf=pdf)
        plot_era5_quantile_reliability(data, context_idx, pdf=pdf)
    print(f"All figures saved to {ALL_FIGURES_PATH}")


if __name__ == "__main__":
    main()
