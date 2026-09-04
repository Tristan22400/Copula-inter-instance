"""fetch_era5_static.py — fetch the 4 time-invariant static surface fields from
ARCO-ERA5 (native 0.25deg global grid, 721x1440) and cache them locally as
eval/data/cache/era5_static.nc.

Variables:
  - geopotential_at_surface (m^2/s^2, float32)
  - land_sea_mask ([0, 1], float32)
  - standard_deviation_of_orography (m, float32)
  - slope_of_sub_gridscale_orography (gradient, float32)

Empirically verified: bit-identical across decades (1990 vs 2020), 0 NaNs.
Total size on disk is ~6 MB.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_CACHE_DIR = os.environ.get(
    "ERA5_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
)
_STATIC_PATH = os.path.join(_CACHE_DIR, "era5_static.nc")
_ARCO_ERA5_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
STATIC_VARS = (
    "geopotential_at_surface",
    "land_sea_mask",
    "standard_deviation_of_orography",
    "slope_of_sub_gridscale_orography",
)

__all__ = ["fetch_static", "load_static", "STATIC_VARS"]


def fetch_static(target_path: str = _STATIC_PATH, force: bool = False) -> str:
    """Fetch the 4 static variables from ARCO-ERA5 and write them to NetCDF3."""
    if os.path.exists(target_path) and not force:
        return target_path

    import netCDF4
    import xarray as xr

    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    print(f"[fetch_era5_static] fetching static fields from ARCO-ERA5 ({_ARCO_ERA5_URL})...")

    ds = xr.open_zarr(_ARCO_ERA5_URL, chunks=None, storage_options={"token": "anon"}, consolidated=True)
    sub = ds[list(STATIC_VARS)].sel(time="2020-01-01T00:00:00").load()

    lat = sub["latitude"].values.astype(np.float64)
    lon = sub["longitude"].values.astype(np.float64)

    print(f"[fetch_era5_static] writing {target_path} (lat={len(lat)}, lon={len(lon)})...")
    nc = netCDF4.Dataset(target_path, "w", format="NETCDF3_CLASSIC")
    nc.history = "ERA5 static fields from ARCO-ERA5 Zarr, native 0.25deg global grid."
    nc.createDimension("latitude", len(lat))
    nc.createDimension("longitude", len(lon))

    var_lat = nc.createVariable("latitude", "f8", ("latitude",))
    var_lat[:] = lat
    var_lon = nc.createVariable("longitude", "f8", ("longitude",))
    var_lon[:] = lon

    for vname in STATIC_VARS:
        arr = sub[vname].values.astype(np.float32)
        if np.isnan(arr).any():
            raise ValueError(f"Found NaNs in static variable {vname}!")
        var = nc.createVariable(vname, "f4", ("latitude", "longitude"))
        var[:] = arr
        print(f"  {vname}: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

    nc.close()
    print(f"[fetch_era5_static] cached {target_path} ({os.path.getsize(target_path) / 1e6:.1f} MB)")
    return target_path


def load_static(path: str = _STATIC_PATH) -> dict[str, np.ndarray]:
    """Load the cached static fields into a dictionary of numpy arrays."""
    if not os.path.exists(path):
        fetch_static(path)

    from scipy.io import netcdf_file

    f = netcdf_file(path, "r", mmap=False)
    out = {
        "latitude": f.variables["latitude"][:].astype(np.float64).copy(),
        "longitude": f.variables["longitude"][:].astype(np.float64).copy(),
    }
    for vname in STATIC_VARS:
        out[vname] = f.variables[vname][:].astype(np.float32).copy()
    f.close()
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default=_STATIC_PATH, help=f"Target NetCDF path (default {_STATIC_PATH}).")
    p.add_argument("--force", action="store_true", help="Force re-fetch.")
    args = p.parse_args()
    fetch_static(args.target, force=args.force)
