"""fetch_era5.py — fetch a real ERA5 2m-temperature grid from the public,
no-auth ARCO-ERA5 Zarr archive on GCS, for an arbitrary region/resolution,
with an on-disk cache keyed by (region, grid_size, n_days) so repeated
sweep/diagnose configs over the same region/resolution don't re-fetch from
GCS every time. Promoted from plots/fetch_real_era5_grid.py; writes the same
NetCDF3-classic schema (t2m/latitude/longitude/time) eval.data.era5_io.load_era5_data
reads.

plots/generate_plots.py:download_era5 requires a CDS API key (~/.cdsapirc),
which isn't configured on this machine (see reference memory on ARCO-ERA5) —
this pulls real 2m_temperature from the public archive instead.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.io.netcdf import netcdf_file

_CACHE_DIR = os.environ.get(
    "ERA5_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
)
_ARCO_ERA5_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

__all__ = ["fetch", "cache_path_for"]


def cache_path_for(region_name: str, grid_size: int, n_days: int) -> str:
    return os.path.join(_CACHE_DIR, f"era5_{region_name}_g{grid_size}_d{n_days}.nc")


def fetch(
    region_name: str,
    lat_bounds: tuple,
    lon_bounds: tuple,
    grid_size: int,
    n_days: int,
    start_date: str = "2023-01-01",
    force: bool = False,
) -> str:
    """Return the local NetCDF path for (region_name, grid_size, n_days),
    fetching from ARCO-ERA5 and writing it to the on-disk cache on a miss —
    the automatic-fetch step every `diagnose`/`sweep`/`baseline` config needs,
    with no separate "fetch" command a user has to remember to run first.
    `force=True` bypasses the cache and re-fetches."""
    target_path = cache_path_for(region_name, grid_size, n_days)
    if os.path.exists(target_path) and not force:
        return target_path

    import xarray as xr

    os.makedirs(_CACHE_DIR, exist_ok=True)
    print(
        f"[fetch_era5] cache miss for (region={region_name}, grid_size={grid_size}, "
        f"n_days={n_days}); fetching from ARCO-ERA5 ({_ARCO_ERA5_URL})..."
    )
    ds = xr.open_zarr(_ARCO_ERA5_URL, chunks={"time": 24}, storage_options={"token": "anon"}, consolidated=True)

    lat_lo, lat_hi = lat_bounds
    lon_lo, lon_hi = lon_bounds
    start = np.datetime64(start_date)
    end = start + np.timedelta64(n_days + 5, "D")  # small margin
    sub = ds[["2m_temperature"]].sel(
        latitude=slice(lat_hi, lat_lo),  # ARCO-ERA5 latitude is descending
        longitude=slice(lon_lo % 360, lon_hi % 360),
        time=slice(str(start), str(end)),
    )
    sub = sub.isel(time=slice(0, None, 24))  # hourly -> one snapshot/day, always 00:00 UTC
    sub = sub.isel(time=slice(0, n_days))
    if sub.sizes["time"] < n_days:
        raise ValueError(f"Only {sub.sizes['time']} daily snapshots available, need {n_days}.")
    if sub.sizes["latitude"] == 0 or sub.sizes["longitude"] == 0:
        raise ValueError(
            f"Empty region for lat_bounds={lat_bounds}, lon_bounds={lon_bounds} "
            f"(got {sub.sizes['latitude']}x{sub.sizes['longitude']})."
        )

    print(f"Downloading {sub.sizes['time']} days x {sub.sizes['latitude']}x{sub.sizes['longitude']} grid...")
    sub = sub.load()

    # Coarsen down to ~grid_size x grid_size so D = H*W stays a manageable
    # correlation-matrix size.
    lat_factor = max(sub.sizes["latitude"] // grid_size, 1)
    lon_factor = max(sub.sizes["longitude"] // grid_size, 1)
    if lat_factor > 1 or lon_factor > 1:
        sub = sub.coarsen(latitude=lat_factor, longitude=lon_factor, boundary="trim").mean()

    t2m = sub["2m_temperature"].values.astype(np.float64) - 273.15  # K -> degC
    lat = sub["latitude"].values.astype(np.float64)
    lon = sub["longitude"].values.astype(np.float64)
    n_time = t2m.shape[0]
    print(f"Final grid: {n_time} days x {lat.size}x{lon.size} ({lat.size * lon.size} points).")

    f = netcdf_file(target_path, "w")
    f.history = (
        f"Real ERA5 2m_temperature from ARCO-ERA5 Zarr, {start_date} + {n_time} daily "
        f"(00:00 UTC) snapshots. region={region_name}, lat_bounds={lat_bounds}, lon_bounds={lon_bounds}."
    )
    f.source = "arco_era5_real"
    f.createDimension("time", n_time)
    f.createDimension("latitude", lat.size)
    f.createDimension("longitude", lon.size)

    var_t2m = f.createVariable("t2m", "f8", ("time", "latitude", "longitude"))
    var_t2m[:] = t2m
    var_lat = f.createVariable("latitude", "f8", ("latitude",))
    var_lat[:] = lat
    var_lon = f.createVariable("longitude", "f8", ("longitude",))
    var_lon[:] = lon
    var_time = f.createVariable("time", "i4", ("time",))
    var_time[:] = np.arange(n_time)  # one integer index per day, all at the same UTC hour
    f.close()
    print(f"Cached {target_path}")
    return target_path
