"""upgrade_era5_dataset.py — Strategy A implementation.

Upgrades all existing ERA5 NetCDF files in:
  - eval/data/cache/era5_global_train/ (120 files, 10 years 2013-2022)
  - eval/data/cache/era5_global_val/ (12 files, 2023)
  - eval/data/cache/era5_*_g*_d*.nc (regional validation files)

Injects the 4 static variables:
  - geopotential_at_surface
  - land_sea_mask
  - standard_deviation_of_orography
  - slope_of_sub_gridscale_orography

Reads from eval/data/cache/era5_static.nc (which is cached once globally)
and writes in-place to each NetCDF file. Idempotent: skips files that already
contain the static fields.
"""

from __future__ import annotations

import glob
import os
import sys
import time

import netCDF4
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from eval.data.fetch_era5_static import STATIC_VARS, fetch_static, load_static


def upgrade_global_file(path: str, static_dict: dict[str, np.ndarray]) -> bool:
    """Inject the 4 static fields into a full 721x1440 global monthly NetCDF file."""
    try:
        ds = netCDF4.Dataset(path, "r+")
    except Exception as e:
        print(f"Error opening {path}: {e}")
        return False

    existing_vars = set(ds.variables.keys())
    needs_update = any(v not in existing_vars for v in STATIC_VARS)
    if not needs_update:
        ds.close()
        return False

    lat_dim = len(ds.dimensions["latitude"])
    lon_dim = len(ds.dimensions["longitude"])
    if lat_dim != 721 or lon_dim != 1440:
        ds.close()
        raise ValueError(f"{path} has unexpected global dimensions ({lat_dim}, {lon_dim})")

    for vname in STATIC_VARS:
        if vname not in ds.variables:
            var = ds.createVariable(vname, "f4", ("latitude", "longitude"))
            var[:] = static_dict[vname]

    ds.close()
    return True


def upgrade_regional_file(path: str, static_dict: dict[str, np.ndarray]) -> bool:
    """Inject the 4 static fields into a regional/coarsened NetCDF file."""
    try:
        ds = netCDF4.Dataset(path, "r+")
    except Exception as e:
        print(f"Error opening {path}: {e}")
        return False

    existing_vars = set(ds.variables.keys())
    needs_update = any(v not in existing_vars for v in STATIC_VARS)
    if not needs_update:
        ds.close()
        return False

    reg_lat = ds.variables["latitude"][:]
    reg_lon = ds.variables["longitude"][:]
    glob_lat = static_dict["latitude"]
    glob_lon = static_dict["longitude"]

    # Match each regional lat/lon to nearest global grid index
    # (or linear interpolation if coarsened)
    # Using scipy RegularGridInterpolator
    from scipy.interpolate import RegularGridInterpolator

    # Note ARCO-ERA5 lat is descending: 90 down to -90. RegularGridInterpolator requires strictly ascending coords.
    lat_asc = glob_lat[::-1]

    for vname in STATIC_VARS:
        if vname not in ds.variables:
            arr = static_dict[vname]
            arr_asc = arr[::-1, :]  # flip latitude to ascending
            interp = RegularGridInterpolator(
                (lat_asc, glob_lon), arr_asc, method="linear", bounds_error=False, fill_value=None
            )
            # Meshgrid for regional points
            lon_grid, lat_grid = np.meshgrid(reg_lon % 360.0, reg_lat)
            pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
            sub_vals = interp(pts).reshape(len(reg_lat), len(reg_lon)).astype(np.float32)

            var = ds.createVariable(vname, "f4", ("latitude", "longitude"))
            var[:] = sub_vals

    ds.close()
    return True


def main() -> None:
    t0 = time.time()
    cache_dir = os.path.join(_REPO, "eval", "data", "cache")
    static_path = os.path.join(cache_dir, "era5_static.nc")
    if not os.path.exists(static_path):
        fetch_static(static_path)

    print(f"Loading global static reference {static_path}...")
    static = load_static(static_path)

    # 1. Global train files
    train_paths = sorted(glob.glob(os.path.join(cache_dir, "era5_global_train", "era5_global_t2m_*.nc")))
    print(f"\nUpgrading {len(train_paths)} global training files in era5_global_train/...")
    n_up_tr = 0
    for p in train_paths:
        if upgrade_global_file(p, static):
            n_up_tr += 1
    print(f"Updated {n_up_tr}/{len(train_paths)} global train files (remainder already up to date).")

    # 2. Global val files
    val_paths = sorted(glob.glob(os.path.join(cache_dir, "era5_global_val", "era5_global_t2m_*.nc")))
    print(f"\nUpgrading {len(val_paths)} global validation files in era5_global_val/...")
    n_up_val = 0
    for p in val_paths:
        if upgrade_global_file(p, static):
            n_up_val += 1
    print(f"Updated {n_up_val}/{len(val_paths)} global val files (remainder already up to date).")

    # 3. Regional probe files
    reg_paths = sorted(glob.glob(os.path.join(cache_dir, "era5_*_g*.nc")))
    print(f"\nUpgrading {len(reg_paths)} regional probe files in cache/...")
    n_up_reg = 0
    for p in reg_paths:
        if upgrade_regional_file(p, static):
            n_up_reg += 1
    print(f"Updated {n_up_reg}/{len(reg_paths)} regional probe files.")

    print(f"\nStrategy A upgrade completed in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
