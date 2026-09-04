"""fetch_era5_global.py — fetch the FULL global 2m-temperature grid (every
latitude/longitude ARCO-ERA5 carries, no region crop) from the same public,
no-auth ARCO-ERA5 Zarr archive eval/data/fetch_era5.py uses, one calendar
month at a time, cached locally as NetCDF3-classic files compatible with
eval.data.era5_io.load_era5_data's schema (t2m/latitude/longitude/time).

Why global + monthly instead of fetch_era5.py's per-region cache: this feeds
src/era5_live_dataset.py's worldwide finetuning corpus (random region AND
random resolution sampled fresh every training episode, see
eval/data/era5_global_corpus.py) — cropping/coarsening a fixed local global
archive per-episode is far cheaper than a GCS round-trip per episode, and
"a few hundred small regional pulls" would re-download heavily overlapping
lat/lon ranges from GCS many times over. One month of the full 0.25 deg
global grid (721x1440 points, one 00:00 UTC snapshot/day, float32) is
~125 MB, safely under NetCDF3-classic's ~2GiB single-file limit and small
enough that fetching is resumable at monthly granularity (skip whatever is
already cached, like fetch_era5.py's fetch() does per-region).

Usage:
  python eval/data/fetch_era5_global.py --start 2022-01 --n-months 24
"""

from __future__ import annotations

import argparse
import calendar
import os

import numpy as np
from scipy.io import netcdf_file

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "era5_global")
_ARCO_ERA5_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

__all__ = ["cache_path_for", "fetch_month", "fetch_range"]


def cache_path_for(year: int, month: int, cache_dir: str = _CACHE_DIR) -> str:
    return os.path.join(cache_dir, f"era5_global_t2m_{year:04d}{month:02d}.nc")


def fetch_month(year: int, month: int, cache_dir: str = _CACHE_DIR, force: bool = False) -> str:
    """Return the local NetCDF path for the full global grid, every daily
    (00:00 UTC) snapshot in (year, month), fetching from ARCO-ERA5 and
    writing it to the on-disk cache on a miss. `force=True` bypasses the
    cache and re-fetches."""
    target_path = cache_path_for(year, month, cache_dir)
    if os.path.exists(target_path) and not force:
        return target_path

    import xarray as xr

    os.makedirs(cache_dir, exist_ok=True)
    n_days = calendar.monthrange(year, month)[1]
    start = np.datetime64(f"{year:04d}-{month:02d}-01")
    end = start + np.timedelta64(n_days, "D")
    print(
        f"[fetch_era5_global] cache miss for {year:04d}-{month:02d}; fetching global grid "
        f"({n_days} days) from ARCO-ERA5 ({_ARCO_ERA5_URL})..."
    )
    ds = xr.open_zarr(_ARCO_ERA5_URL, chunks={"time": 24}, storage_options={"token": "anon"}, consolidated=True)
    sub = ds[["2m_temperature"]].sel(time=slice(str(start), str(end)))
    sub = sub.isel(time=slice(0, None, 24))  # hourly -> one snapshot/day, always 00:00 UTC
    sub = sub.isel(time=slice(0, n_days))
    if sub.sizes["time"] < n_days:
        raise ValueError(f"Only {sub.sizes['time']} daily snapshots available for {year:04d}-{month:02d}.")

    print(f"Downloading {sub.sizes['time']} days x {sub.sizes['latitude']}x{sub.sizes['longitude']} global grid...")
    sub = sub.load()

    t2m = sub["2m_temperature"].values.astype(np.float32) - np.float32(273.15)  # K -> degC
    lat = sub["latitude"].values.astype(np.float64)
    lon = sub["longitude"].values.astype(np.float64)
    n_time = t2m.shape[0]

    try:
        from eval.data.fetch_era5_static import STATIC_VARS, load_static
    except ModuleNotFoundError:
        from fetch_era5_static import STATIC_VARS, load_static
    static_dict = load_static()

    f = netcdf_file(target_path, "w")
    f.history = (
        f"Full global ERA5 2m_temperature and static surface variables from ARCO-ERA5 Zarr, {year:04d}-{month:02d}, "
        f"{n_time} daily (00:00 UTC) snapshots, native 0.25deg grid."
    )
    f.source = "arco_era5_real_global"
    f.createDimension("time", n_time)
    f.createDimension("latitude", lat.size)
    f.createDimension("longitude", lon.size)
    var_t2m = f.createVariable("t2m", "f4", ("time", "latitude", "longitude"))
    var_t2m[:] = t2m
    var_lat = f.createVariable("latitude", "f8", ("latitude",))
    var_lat[:] = lat
    var_lon = f.createVariable("longitude", "f8", ("longitude",))
    var_lon[:] = lon
    var_time = f.createVariable("time", "i4", ("time",))
    var_time[:] = np.arange(n_time)

    for vname in STATIC_VARS:
        v = f.createVariable(vname, "f4", ("latitude", "longitude"))
        v[:] = static_dict[vname]

    f.close()
    print(f"Cached {target_path} ({os.path.getsize(target_path) / 1e6:.0f} MB)")
    return target_path


def fetch_range(start_year: int, start_month: int, n_months: int, cache_dir: str = _CACHE_DIR, force: bool = False) -> list[str]:
    """Fetch n_months consecutive calendar months starting at (start_year,
    start_month), skipping months already cached (unless force=True).
    Resumable: re-running with the same args after a partial/interrupted run
    only re-fetches what's missing."""
    paths = []
    year, month = start_year, start_month
    for _ in range(n_months):
        paths.append(fetch_month(year, month, cache_dir, force=force))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return paths


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2022-01", help="First calendar month to fetch, YYYY-MM (default 2022-01).")
    p.add_argument("--n-months", type=int, default=24, help="Number of consecutive months to fetch (default 24, ~3GB).")
    p.add_argument("--cache-dir", default=_CACHE_DIR, help=f"Local cache directory (default {_CACHE_DIR}).")
    p.add_argument("--force", action="store_true", help="Re-fetch even if already cached.")
    args = p.parse_args()

    y, m = (int(x) for x in args.start.split("-"))
    fetch_range(y, m, args.n_months, cache_dir=args.cache_dir, force=args.force)
