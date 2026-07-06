"""
Generate the three NeurIPS figures comparing an independent tabular foundation
model (TabICLv2, mocked) against a copula-based tabular foundation model
(Copula-TFM, mocked) on a spatial temperature field.

Everything this script needs lives in this directory:
  - plots/generate_neurips_plots.py   (this file)
  - plots/era5_temperature.nc         (downloaded or synthetic input data)
  - plots/*.pdf                       (the four output figures)

Usage:
    python plots/generate_neurips_plots.py
    python plots/generate_neurips_plots.py --ckpt ./checkpoints/copula_transformer_muon/step_0025000.pt
"""

import argparse
import os
import shutil
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import netcdf_file
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal, norm
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
def TabICLv2_Marginal(coords_test, field_flat):
    """
    Mock of an independent tabular foundation model: returns a marginal
    mean/std per test location, with no cross-location covariance information.
    """
    lat = coords_test[:, 1]
    mean = 25.0 - 0.35 * (lat - lat.min())
    std = np.full(coords_test.shape[0], field_flat.std() * 0.6)
    return mean, std


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


# ---------------------------------------------------------------------------
# Plot 1: Static vs. Smooth spatial map
# ---------------------------------------------------------------------------
def plot_spatial_map_comparison(data, day, context_idx, C):
    field = data["t2m"][day]
    lat, lon = data["latitude"], data["longitude"]
    grid_size = field.shape[0]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    field_flat = field.ravel()
    M = coords.shape[0]
    context_coords = coords[context_idx]

    mean, std = TabICLv2_Marginal(coords, field_flat)

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
    out_path = os.path.join(PLOTS_DIR, "spatial_map_comparison.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Correlation vs. distance (implicit variogram)
# ---------------------------------------------------------------------------
def plot_correlation_vs_distance(data, C):
    lat, lon = data["latitude"], data["longitude"]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    iu = np.triu_indices_from(C, k=1)
    corr = C[iu]
    dist = squareform(pdist(coords))[iu]

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
    ax.set_xlabel("Spatial distance")
    ax.set_ylabel(r"Predicted correlation $\rho_{ij}$")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "correlation_vs_distance.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Predicted vs. ground-truth correlation matrix
# ---------------------------------------------------------------------------
def plot_correlation_matrix_comparison(C, C_true):
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
    out_path = os.path.join(PLOTS_DIR, "correlation_matrix_comparison.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: Calibration of joint probabilities
# ---------------------------------------------------------------------------
def plot_joint_probability_calibration(data, day, C, n_pairs=400, n_bins=10):
    field_all = data["t2m"]
    n_days, grid_size, _ = field_all.shape
    lat, lon = data["latitude"], data["longitude"]
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    M = coords.shape[0]

    samples = field_all.reshape(n_days, M)
    tau = np.percentile(samples, 90)

    # p_i must be the *predicted* exceedance probability (per the spec), i.e.
    # under TabICLv2's own marginal model -- not the oracle mean/std of the
    # true field. Both the independent (TabICLv2) and copula-combined
    # (Copula-TFM) joint probabilities below reuse this same p_i, p_j so the
    # plot isolates the effect of the dependence structure, not marginal
    # quality: any miscalibration in the independent curve here is inherited
    # from TabICLv2's (mocked) marginal, exactly as in a real deployment.
    mean_pred, std_pred = TabICLv2_Marginal(coords, field_all[day].ravel())
    p_exceed = 1.0 - norm.cdf(tau, loc=mean_pred, scale=std_pred)

    exceed_mask = samples > tau

    idx_i = RNG.integers(0, M, size=n_pairs)
    idx_j = RNG.integers(0, M, size=n_pairs)
    keep = idx_i != idx_j
    idx_i, idx_j = idx_i[keep], idx_j[keep]

    empirical, pred_indep, pred_copula = [], [], []
    for i, j in zip(idx_i, idx_j):
        empirical.append(np.mean(exceed_mask[:, i] & exceed_mask[:, j]))
        p_i, p_j = p_exceed[i], p_exceed[j]
        pred_indep.append(p_i * p_j)

        rho = C[i, j]
        z_i, z_j = norm.ppf(1 - p_i), norm.ppf(1 - p_j)
        cov = [[1.0, rho], [rho, 1.0]]
        # P(Z_i > z_i, Z_j > z_j) = P(-Z_i < -z_i, -Z_j < -z_j) by symmetry of N(0, cov).
        pred_copula.append(multivariate_normal.cdf([-z_i, -z_j], mean=[0, 0], cov=cov))

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
    out_path = os.path.join(PLOTS_DIR, "joint_probability_calibration.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


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
        "'observed' field: it's the True Field panel, the source of context values (X_train, "
        "Y_train) for the real Copula-TFM checkpoint, and the field TabICLv2_Marginal fits its "
        "marginal to for the calibration plot. Default: 0.",
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

    plot_spatial_map_comparison(data, day, context_idx, C)
    plot_correlation_vs_distance(data, C)
    plot_correlation_matrix_comparison(C, C_true)
    plot_joint_probability_calibration(data, day, C)
    print(f"All four NeurIPS figures saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
