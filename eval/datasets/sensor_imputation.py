"""sensor_imputation.py — Benchmark 2: cross-sectional in-painting on the UCI
"Beijing Multi-Site Air-Quality Data" dataset.

Known constraint: this dataset has only 12 monitoring stations — not enough
distinct spatial points for n_ctx=100/n_test=30 at a single timestamp. Each
(station, day) pair is instead treated as one "instance": features are
[station_lat, station_lon, sin(day_of_year), cos(day_of_year), TEMP, PRES,
DEWP] (weather covariates already in the CSVs), target is the daily-mean
PM2.5 reading. This gives thousands of (station, day) instances to sample
n_ctx + n_test from, at the cost of the split no longer being a literal
single-timestamp mask.

Station lat/lon are not present in the raw CSVs — they are hardcoded below
from public knowledge of the 12 station locations (approximate; not part of
the dataset itself).
"""

from __future__ import annotations

import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

__all__ = ["load_split"]

_ZIP_URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(_HERE), "data", "beijing_pm25")
_ZIP_PATH = os.path.join(_DATA_DIR, "raw.zip")
_EXTRACT_DIR = os.path.join(_DATA_DIR, "extracted")
_CACHE_CSV = os.path.join(_DATA_DIR, "processed_instances.csv")

# Approximate public coordinates of the 12 Beijing air-quality monitoring
# stations (not present in the raw dataset).
_STATION_COORDS = {
    "Aotizhongxin": (39.982, 116.397),
    "Changping": (40.220, 116.230),
    "Dingling": (40.292, 116.220),
    "Dongsi": (39.929, 116.417),
    "Guanyuan": (39.929, 116.339),
    "Gucheng": (39.914, 116.184),
    "Huairou": (40.328, 116.628),
    "Nongzhanguan": (39.933, 116.461),
    "Shunyi": (40.127, 116.655),
    "Tiantan": (39.886, 116.407),
    "Wanliu": (39.987, 116.287),
    "Wanshouxigong": (39.878, 116.352),
}

_FEATURE_COLS = ["lat", "lon", "sin_doy", "cos_doy", "TEMP", "PRES", "DEWP"]


def _download_and_extract() -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    if not os.path.exists(_ZIP_PATH):
        urllib.request.urlretrieve(_ZIP_URL, _ZIP_PATH)
    if not os.path.isdir(_EXTRACT_DIR) or not os.listdir(_EXTRACT_DIR):
        os.makedirs(_EXTRACT_DIR, exist_ok=True)
        with zipfile.ZipFile(_ZIP_PATH) as zf:
            zf.extractall(_EXTRACT_DIR)
        # The UCI archive nests the actual per-station CSVs inside a second
        # zip (PRSA2017_Data_*.zip) alongside a couple of unrelated files.
        for fname in os.listdir(_EXTRACT_DIR):
            if fname.endswith(".zip"):
                with zipfile.ZipFile(os.path.join(_EXTRACT_DIR, fname)) as inner_zf:
                    inner_zf.extractall(_EXTRACT_DIR)


def _find_station_csvs() -> list[str]:
    matches = []
    for root, _dirs, files in os.walk(_EXTRACT_DIR):
        for fname in files:
            if fname.startswith("PRSA_Data_") and fname.endswith(".csv"):
                matches.append(os.path.join(root, fname))
    if not matches:
        raise RuntimeError(f"No PRSA_Data_*.csv files found under {_EXTRACT_DIR}")
    return matches


def _build_instance_table() -> pd.DataFrame:
    if os.path.exists(_CACHE_CSV):
        return pd.read_csv(_CACHE_CSV)

    _download_and_extract()
    csv_paths = _find_station_csvs()

    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        daily = (
            df.groupby(["station", "year", "month", "day"])[["PM2.5", "TEMP", "PRES", "DEWP"]]
            .mean()
            .reset_index()
        )
        frames.append(daily)
    all_daily = pd.concat(frames, ignore_index=True)
    all_daily = all_daily.dropna(subset=["PM2.5", "TEMP", "PRES", "DEWP"])

    all_daily["lat"] = all_daily["station"].map(lambda s: _STATION_COORDS[s][0])
    all_daily["lon"] = all_daily["station"].map(lambda s: _STATION_COORDS[s][1])
    doy = pd.to_datetime(all_daily[["year", "month", "day"]]).dt.dayofyear
    all_daily["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    all_daily["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    os.makedirs(_DATA_DIR, exist_ok=True)
    all_daily.to_csv(_CACHE_CSV, index=False)
    return all_daily


def load_split(
    n_ctx: int = 100, n_test: int = 30, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_train, y_train, X_test, y_test), raw feature scale."""
    table = _build_instance_table()
    rng = np.random.default_rng(seed)

    n_rows = len(table)
    idx = rng.choice(n_rows, size=min(n_ctx + n_test, n_rows), replace=False)
    test_idx = idx[:n_test]
    ctx_idx = idx[n_test:]

    X = table[_FEATURE_COLS].to_numpy(dtype=np.float64)
    y = table["PM2.5"].to_numpy(dtype=np.float64)

    return X[ctx_idx], y[ctx_idx], X[test_idx], y[test_idx]
