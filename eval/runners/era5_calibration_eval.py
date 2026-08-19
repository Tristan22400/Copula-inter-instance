"""era5_calibration_eval.py — real-ERA5 driver for the independence-copula
multivariate spatial calibration diagnostics in eval/spatial/calibration.py
(calc_kendall_pit, calc_mahalanobis_distances, calc_exceedance_probs,
calc_spatial_coverage and their plot_* counterparts). Promoted from
plots/run_era5_eval.py.

All four calibration metrics are, by their own docstrings, testing whether
TabICL's MARGINALS ALONE (ignoring spatial correlation — the independence
copula) are calibrated: they only need real per-cell marginal quantile
predictions, not the correlation/copula model. So the promotion from
plots/run_era5_eval.py's placeholder `MockTabICLv2` to a real model is a
contained swap of the quantile source — via eval/tabicl_utils.py's
make_tabicl_regressor + tabicl_quantiles (the same helpers
eval/runners/run_benchmarks.py already uses) — not new inference logic.

For each timestamp, an in-context-learning episode is built with
`sample_icl_task_from_era5`: a dense target patch inside a lat/lon box, and
a global context of `n_ctx` points from the rest of the grid at that same
timestamp.

Usage:
    python eval/runners/era5_calibration_eval.py \\
        --nc-path /path/to/era5_temperature.nc \\
        --target-lat-bounds 45 50 --target-lon-bounds 0 5

If --nc-path is omitted, a small real ERA5 sample (Western Europe, 10 daily
snapshots from Jan 2023) is auto-downloaded from the public no-auth
ARCO-ERA5 Zarr archive on GCS and cached under eval/data/cache/, so the
script produces real results out of the box with no arguments.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import norm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.spatial import calibration as cal  # noqa: E402
from eval.tabicl_utils import make_tabicl_regressor, tabicl_quantiles  # noqa: E402

_G = 9.80665  # m/s^2, for geopotential (m^2/s^2) -> elevation (m)
_TEMP_VAR_CANDIDATES = ("t2m", "2m_temperature", "temperature", "temp")
_ELEV_VAR_CANDIDATES = (
    "z", "geopotential", "geopotential_at_surface", "surface_geopotential",
    "orography", "elevation", "altitude",
)

# Public, no-auth ARCO-ERA5 archive used to auto-fetch a default real-data
# sample when --nc-path is omitted (see reference memory: no CDS API key
# needed).
_ARCO_ERA5_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
_CACHE_DIR = os.path.join(_REPO_ROOT, "eval", "data", "cache")
_DEFAULT_CACHE_NC = os.path.join(_CACHE_DIR, "era5_calibration_default.nc")
_DEFAULT_TARGET_LAT_BOUNDS = (45.0, 50.0)  # Northern France / Belgium
_DEFAULT_TARGET_LON_BOUNDS = (0.0, 5.0)
_DEFAULT_FETCH_LAT_BOUNDS = (25.0, 72.0)  # Europe-ish context box
_DEFAULT_FETCH_LON_BOUNDS = (0.0, 40.0)  # kept in [0, 360) to match ARCO-ERA5's longitude convention
_DEFAULT_FETCH_TIME_RANGE = ("2023-01-01", "2023-01-10")  # 10 daily (00:00 UTC) snapshots
_DEFAULT_FIGURES_DIR = os.path.join(_REPO_ROOT, "eval", "reports", "figures")

# Quantile levels TabICL is queried at for every ICL episode. Dense enough
# to (a) fit a Gaussian mean/std per target cell via least squares, (b)
# invert the quantile function at an arbitrary threshold by linear
# interpolation, and (c) read off arbitrary alpha/2, 1-alpha/2 central-
# interval bounds for the coverage curve.
ALPHA_GRID = np.round(np.linspace(0.01, 0.99, 99), 2)


def _fetch_default_era5_subset(cache_path: str = _DEFAULT_CACHE_NC) -> str:
    """Download a small real ERA5 sample (2m temperature + surface
    geopotential, Europe box, 10 daily snapshots from Jan 2023) from the
    public no-auth ARCO-ERA5 Zarr archive and cache it as a local NetCDF
    file, so this only runs once. Returns `cache_path`.

    A separate fetch routine from eval/data/fetch_era5.py: this needs a
    real datetime64 'time' coordinate and elevation, which that
    diagnostics-focused fetcher's minimal (t2m/lat/lon/integer-day-index)
    NetCDF3-classic schema deliberately doesn't carry.
    """
    if os.path.exists(cache_path):
        return cache_path
    os.makedirs(_CACHE_DIR, exist_ok=True)
    print(f"No --nc-path given; downloading a small real ERA5 sample to {cache_path} ...")
    ds = xr.open_zarr(_ARCO_ERA5_URL, chunks={"time": 24}, storage_options={"token": "anon"}, consolidated=True)
    lat_lo, lat_hi = _DEFAULT_FETCH_LAT_BOUNDS
    lon_lo, lon_hi = _DEFAULT_FETCH_LON_BOUNDS
    start, end = _DEFAULT_FETCH_TIME_RANGE
    sub = ds[["2m_temperature", "geopotential_at_surface"]].sel(
        latitude=slice(lat_hi, lat_lo),  # ARCO-ERA5 latitude is descending (90 -> -90)
        longitude=slice(lon_lo, lon_hi),
        time=slice(start, end),
    )
    sub = sub.isel(time=slice(0, None, 24))  # hourly -> daily (00:00 UTC)
    sub.load().to_netcdf(cache_path)
    print(f"Saved {cache_path}")
    return cache_path


# ---------------------------------------------------------------------------
# ERA5 -> ICL episode
# ---------------------------------------------------------------------------
def _find_var(ds: xr.Dataset, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def _elevation_field(ds: xr.Dataset, time_idx: int, shape: Tuple[int, int]) -> np.ndarray:
    elev_var = _find_var(ds, _ELEV_VAR_CANDIDATES)
    if elev_var is None:
        print(f"Note: no elevation-like variable found (tried {_ELEV_VAR_CANDIDATES}); using elevation=0.")
        return np.zeros(shape, dtype=np.float64)

    da = ds[elev_var]
    if "time" in da.dims:
        da = da.isel(time=time_idx)
    field = da.values.astype(np.float64)
    if elev_var in ("z", "geopotential", "geopotential_at_surface", "surface_geopotential"):
        field = field / _G
    return field


def _time_features(ds: xr.Dataset, time_idx: int) -> Tuple[float, float, float, float]:
    t = ds["time"].isel(time=time_idx)
    if not np.issubdtype(t.dtype, np.datetime64):
        raise ValueError(
            "sample_icl_task_from_era5 requires a real datetime64 'time' coordinate "
            f"(as provided by an actual ERA5 download); got dtype {t.dtype}."
        )
    day_of_year = float(t.dt.dayofyear.values)
    hour = float(t.dt.hour.values) + float(t.dt.minute.values) / 60.0
    day_angle = 2 * np.pi * day_of_year / 365.25
    hour_angle = 2 * np.pi * hour / 24.0
    return np.cos(day_angle), np.sin(day_angle), np.cos(hour_angle), np.sin(hour_angle)


def sample_icl_task_from_era5(
    ds: xr.Dataset,
    time_idx: int,
    target_lat_bounds: Tuple[float, float],
    target_lon_bounds: Tuple[float, float],
    n_ctx: int = 1000,
    rng: Optional[np.random.Generator] = None,
):
    """
    Build one in-context-learning episode from `ds` at a single timestamp:
    a dense target patch inside the given lat/lon box (the D-dimensional
    joint target), and a global context of up to `n_ctx` points sampled from
    the rest of the grid at the SAME timestamp — excluding the target box
    itself, so context and target never overlap.

    Args:
        ds: xarray.Dataset with dims (time, latitude, longitude), a real
            datetime64 'time' coordinate, a temperature variable, and
            optionally a static elevation/geopotential variable.
        time_idx: Index into the time dimension.
        target_lat_bounds, target_lon_bounds: (min, max) tuples defining the
            dense target patch.
        n_ctx: Number of context points to sample from outside the target
            patch (clipped to the available pool size if smaller).
        rng: numpy Generator used for context sampling; a fresh
            `default_rng(0)` is used if omitted.

    Returns:
        (X_ctx, Y_ctx, X_target, Y_target). X_* arrays have feature columns
        [Lat, Lon, Elev, CosDay, SinDay, CosHour, SinHour]; Y_* are the
        temperature targets.
    """
    rng = rng if rng is not None else np.random.default_rng(0)

    temp_var = _find_var(ds, _TEMP_VAR_CANDIDATES)
    if temp_var is None:
        raise ValueError(f"No recognized temperature variable found; tried {_TEMP_VAR_CANDIDATES}")
    lat_name = _find_var(ds, ("latitude", "lat"))
    lon_name = _find_var(ds, ("longitude", "lon"))
    if lat_name is None or lon_name is None:
        raise ValueError("Dataset must have a 'latitude'/'lat' and 'longitude'/'lon' coordinate.")

    temp_field = ds[temp_var].isel(time=time_idx).values.astype(np.float64)  # (n_lat, n_lon)
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    elev_field = _elevation_field(ds, time_idx, temp_field.shape)
    cos_day, sin_day, cos_hour, sin_hour = _time_features(ds, time_idx)

    def _features(mask):
        n = int(mask.sum())
        return np.column_stack(
            [
                lat_grid[mask], lon_grid[mask], elev_field[mask],
                np.full(n, cos_day), np.full(n, sin_day), np.full(n, cos_hour), np.full(n, sin_hour),
            ]
        )

    lat_lo, lat_hi = target_lat_bounds
    lon_lo, lon_hi = target_lon_bounds
    target_mask = (
        (lat_grid >= lat_lo) & (lat_grid <= lat_hi) & (lon_grid >= lon_lo) & (lon_grid <= lon_hi)
    )
    if not target_mask.any():
        raise ValueError("target_lat_bounds/target_lon_bounds select zero grid points.")

    X_target = _features(target_mask)
    Y_target = temp_field[target_mask]

    ctx_pool_idx = np.flatnonzero(~target_mask.ravel())
    n_ctx_eff = min(n_ctx, ctx_pool_idx.size)
    ctx_flat_idx = rng.choice(ctx_pool_idx, size=n_ctx_eff, replace=False)
    ctx_mask = np.zeros(target_mask.size, dtype=bool)
    ctx_mask[ctx_flat_idx] = True
    ctx_mask = ctx_mask.reshape(target_mask.shape)

    X_ctx = _features(ctx_mask)
    Y_ctx = temp_field[ctx_mask]

    return X_ctx, Y_ctx, X_target, Y_target


# ---------------------------------------------------------------------------
# Quantile-grid adapters (shared by all 4 calibration metrics)
# ---------------------------------------------------------------------------
def _invert_quantile_cdf(y_values: np.ndarray, quantile_values: np.ndarray, alpha_grid: np.ndarray) -> np.ndarray:
    """Per-row inverse of a piecewise-linear quantile function: row i's
    (quantile_values[i], alpha_grid) pairs are the model's declared
    inverse-CDF control points; interpolate to obtain F_i(y_values[i])."""
    y_values = np.asarray(y_values, dtype=np.float64)
    out = np.empty(quantile_values.shape[0], dtype=np.float64)
    for i in range(quantile_values.shape[0]):
        out[i] = np.interp(y_values[i], quantile_values[i], alpha_grid)
    return out


def _quantile_at_alpha(alpha: float, quantile_values: np.ndarray, alpha_grid: np.ndarray) -> np.ndarray:
    """Forward evaluation Q_i(alpha) for every row i, via linear interpolation
    along the (shared) alpha_grid."""
    idx = np.clip(np.searchsorted(alpha_grid, alpha), 1, len(alpha_grid) - 1)
    a0, a1 = alpha_grid[idx - 1], alpha_grid[idx]
    q0, q1 = quantile_values[:, idx - 1], quantile_values[:, idx]
    w = 0.0 if a1 == a0 else (alpha - a0) / (a1 - a0)
    return q0 + w * (q1 - q0)


def _gaussian_mean_variance(quantile_values: np.ndarray, alpha_grid: np.ndarray):
    """Per-row Gaussian (mu, sigma^2) fit by least squares against the
    standard-normal z-scores of alpha_grid: q_i(alpha) ~= mu_i + sigma_i * z(alpha)."""
    z = norm.ppf(alpha_grid)
    design = np.column_stack([np.ones_like(z), z])  # (K, 2)
    coefs = np.linalg.pinv(design) @ quantile_values.T  # (2, n_rows)
    means = coefs[0]
    sigmas = np.clip(coefs[1], 1e-3, None)
    return means, sigmas**2


# ---------------------------------------------------------------------------
# Evaluation loop + figure
# ---------------------------------------------------------------------------
def run_era5_eval(
    nc_path: str,
    target_lat_bounds: Tuple[float, float],
    target_lon_bounds: Tuple[float, float],
    tabicl_ckpt: Optional[str] = None,
    device: Optional[str] = None,
    n_ctx: int = 1000,
    n_timestamps: Optional[int] = None,
    seed: int = 0,
):
    """
    Loop over the first `n_timestamps` (default: all) timestamps in
    `nc_path`, running one ICL episode + real TabICL marginal inference per
    timestamp.

    Returns:
        (y_true, all_quantiles): y_true is (N, D); all_quantiles is
        (N, D, len(ALPHA_GRID)).
    """
    ds = xr.open_dataset(nc_path)
    rng = np.random.default_rng(seed)
    n_times = ds.sizes["time"] if n_timestamps is None else min(n_timestamps, ds.sizes["time"])

    print(f"Loading TabICL marginal model (TabICLRegressor): {tabicl_ckpt or '(default checkpoint)'}")
    regressor = make_tabicl_regressor(checkpoint=tabicl_ckpt, device=device)

    y_true_chunks, quantile_chunks = [], []
    for t in range(n_times):
        X_ctx, Y_ctx, X_target, Y_target = sample_icl_task_from_era5(
            ds, t, target_lat_bounds, target_lon_bounds, n_ctx=n_ctx, rng=rng
        )
        preds = tabicl_quantiles(regressor, X_ctx, Y_ctx, X_target, ALPHA_GRID)  # (D, K)
        y_true_chunks.append(Y_target)
        quantile_chunks.append(preds)
        print(f"  timestamp {t + 1}/{n_times} done", flush=True)
    ds.close()

    y_true = np.stack(y_true_chunks, axis=0)  # (N, D)
    all_quantiles = np.stack(quantile_chunks, axis=0)  # (N, D, K)
    return y_true, all_quantiles


def build_calibration_figure(
    y_true: np.ndarray,
    all_quantiles: np.ndarray,
    alpha_grid: np.ndarray,
    output_path: str,
    exceedance_thresholds: Optional[np.ndarray] = None,
    nominal_coverages: Optional[np.ndarray] = None,
):
    """Build and save the 2x2 multivariate spatial calibration figure."""
    N, D, K = all_quantiles.shape
    flat_q = all_quantiles.reshape(N * D, K)

    means_flat, variances_flat = _gaussian_mean_variance(flat_q, alpha_grid)
    means, variances = means_flat.reshape(N, D), variances_flat.reshape(N, D)

    cdf_at_y = _invert_quantile_cdf(y_true.ravel(), flat_q, alpha_grid).reshape(N, D)
    z_kendall = cal.calc_kendall_pit(cdf_at_y)

    distances = cal.calc_mahalanobis_distances(y_true, means, variances)

    def cdf_func(tau):
        return _invert_quantile_cdf(np.full(N * D, tau), flat_q, alpha_grid).reshape(N, D)

    if exceedance_thresholds is None:
        exceedance_thresholds = np.quantile(y_true, [0.5, 0.75, 0.9, 0.95, 0.99])
    predicted_probs, true_events = cal.calc_exceedance_probs(y_true, cdf_func, exceedance_thresholds)

    def quantile_func(alpha):
        lo = _quantile_at_alpha(alpha / 2, flat_q, alpha_grid).reshape(N, D)
        hi = _quantile_at_alpha(1 - alpha / 2, flat_q, alpha_grid).reshape(N, D)
        return lo, hi

    if nominal_coverages is None:
        nominal_coverages = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    cal.plot_kendall_pit(z_kendall, axes[0, 0])
    cal.plot_mahalanobis_pp(distances, D, axes[0, 1])
    cal.plot_spatial_reliability(predicted_probs, true_events, num_bins=10, ax=axes[1, 0])
    cal.plot_spatial_coverage_curve(y_true, quantile_func, nominal_coverages, axes[1, 1])
    fig.suptitle("TabICL Multivariate Spatial Calibration on ERA5 (independence copula)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--nc-path", type=str, default=None,
        help="Path to a real ERA5 temperature NetCDF file. If omitted, a small real "
        f"sample is auto-downloaded and cached to {_DEFAULT_CACHE_NC}.",
    )
    parser.add_argument(
        "--target-lat-bounds", type=float, nargs=2, default=list(_DEFAULT_TARGET_LAT_BOUNDS),
        metavar=("LAT_MIN", "LAT_MAX"),
    )
    parser.add_argument(
        "--target-lon-bounds", type=float, nargs=2, default=list(_DEFAULT_TARGET_LON_BOUNDS),
        metavar=("LON_MIN", "LON_MAX"),
    )
    parser.add_argument(
        "--tabicl-ckpt", type=str, default=None,
        help="TabICLRegressor checkpoint_version (default: the library's own default checkpoint).",
    )
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--n-ctx", type=int, default=1000, help="Global context points sampled per timestamp.")
    parser.add_argument("--n-timestamps", type=int, default=None, help="Number of timestamps to evaluate (default: all).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=str, default=os.path.join(_DEFAULT_FIGURES_DIR, "era5_multivariate_calibration.pdf"),
    )
    args = parser.parse_args()

    nc_path = args.nc_path if args.nc_path is not None else _fetch_default_era5_subset()

    y_true, all_quantiles = run_era5_eval(
        nc_path, tuple(args.target_lat_bounds), tuple(args.target_lon_bounds),
        tabicl_ckpt=args.tabicl_ckpt, device=args.device,
        n_ctx=args.n_ctx, n_timestamps=args.n_timestamps, seed=args.seed,
    )
    print(f"Evaluated {y_true.shape[0]} timestamps x {y_true.shape[1]} target grid cells.")
    build_calibration_figure(y_true, all_quantiles, ALPHA_GRID, args.output)


if __name__ == "__main__":
    main()
