"""era5_io.py — ERA5 NetCDF reading + geometry helpers shared by every
spatial-correlation diagnostic: great-circle distance, NetCDF loading, and a
jitter-robust Cholesky factor for near-PSD correlation matrices. Promoted
from plots/generate_plots.py (haversine_distance_km, load_era5_data,
_safe_cholesky)."""

from __future__ import annotations

import numpy as np
from scipy.io.netcdf import netcdf_file

from eval.configs.constants import EARTH_RADIUS_KM

__all__ = ["haversine_distance_km", "load_era5_data", "safe_cholesky"]


def haversine_distance_km(coords: np.ndarray) -> np.ndarray:
    """Great-circle distance (km) between every pair of (lon, lat) points in
    `coords` (degrees)."""
    lon_rad, lat_rad = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_era5_data(nc_path: str) -> dict:
    """Read an ERA5-schema NetCDF3-classic file (t2m/latitude/longitude/time
    and static variables — see eval.data.fetch_era5.fetch) into plain numpy arrays."""
    f = netcdf_file(nc_path, "r", mmap=False)
    t2m = f.variables["t2m"][:].astype(np.float64).copy()
    lat = f.variables["latitude"][:].astype(np.float64).copy()
    lon = f.variables["longitude"][:].astype(np.float64).copy()
    out = {"t2m": t2m, "latitude": lat, "longitude": lon}
    for v in (
        "geopotential_at_surface",
        "land_sea_mask",
        "standard_deviation_of_orography",
        "slope_of_sub_gridscale_orography",
    ):
        if v in f.variables:
            out[v] = f.variables[v][:].astype(np.float64).copy()
    f.close()
    return out


def safe_cholesky(C: np.ndarray, jitter: float = 1e-6, max_tries: int = 6) -> np.ndarray:
    """Cholesky factor of C, adding diagonal jitter if it's not quite PSD.

    A real checkpoint's predicted correlation matrix is PSD by construction
    (low-rank-plus-diagonal, see src/model.py:low_rank_correlation) but
    float32 round-trip can leave it *just* outside PSD, which
    np.linalg.cholesky rejects outright.
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
