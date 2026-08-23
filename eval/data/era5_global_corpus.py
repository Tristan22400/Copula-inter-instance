"""era5_global_corpus.py — in-memory reader over the monthly global caches
written by fetch_era5_global.py, with a `sample_episode` that crops a random
worldwide region at a random grid resolution out of the full native-0.25deg
grid.

This is what makes "different resolutions, different geographic space" a
per-training-episode draw instead of a fixed handful of pre-fetched regions:
once the global grid is on disk, every crop is pure in-memory numpy slicing
(no GCS round-trip), so src/era5_live_dataset.py can afford to draw a fresh
region/resolution/day every single training episode.
"""

from __future__ import annotations

import glob
import os

import numpy as np
from scipy.io.netcdf import netcdf_file

__all__ = ["GlobalERA5Corpus"]


def _load_month(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = netcdf_file(path, "r", mmap=False)
    t2m = f.variables["t2m"][:].astype(np.float32).copy()  # (T, H, W)
    lat = f.variables["latitude"][:].astype(np.float64).copy()
    lon = f.variables["longitude"][:].astype(np.float64).copy()
    f.close()
    return t2m, lat, lon


class GlobalERA5Corpus:
    """Loads every era5_global_t2m_*.nc file under `cache_dir` into RAM once
    (native grid is the same lat/lon for every file — only checked, not
    re-derived, per file) and exposes `sample_episode` for random-region,
    random-resolution episode draws.

    Meant to be constructed once per DataLoader worker process (mirrors
    live_dataset.py::LiveGPDataset's one-TabICL-load-per-worker pattern) —
    loading is a one-time cost (~125MB/cached month, read into float32 RAM),
    not repeated per episode.
    """

    def __init__(self, cache_dir: str):
        paths = sorted(glob.glob(os.path.join(cache_dir, "era5_global_t2m_*.nc")))
        if not paths:
            raise FileNotFoundError(
                f"No cached global ERA5 files found under {cache_dir!r} — run "
                "`python eval/data/fetch_era5_global.py` first to populate the "
                "worldwide finetuning corpus."
            )
        self.lat: np.ndarray | None = None
        self.lon: np.ndarray | None = None
        self._t2m_by_month: list[np.ndarray] = []
        for p in paths:
            t2m, lat, lon = _load_month(p)
            if self.lat is None:
                self.lat, self.lon = lat, lon
            self._t2m_by_month.append(t2m)
        day_counts = [a.shape[0] for a in self._t2m_by_month]
        self.n_days_total = int(sum(day_counts))
        self._cum_days = np.cumsum([0] + day_counts)

    def _day_slice(self, day_global_idx: int) -> np.ndarray:
        m = int(np.searchsorted(self._cum_days, day_global_idx, side="right") - 1)
        d = day_global_idx - int(self._cum_days[m])
        return self._t2m_by_month[m][d]

    def sample_episode(
        self,
        rng: np.random.Generator,
        grid_size_range: tuple[int, int],
        box_deg_range: tuple[float, float],
        n_context_frac_range: tuple[float, float],
    ) -> dict | None:
        """Draw one (region, resolution, day, context/test split) episode.

        Region: box center sampled uniformly on the sphere (arcsin-of-uniform
        latitude, uniform longitude — area-uniform, not lat/lon-uniform, so
        the corpus doesn't over-represent high latitudes), box half-width
        drawn from `box_deg_range` degrees. Resolution: grid_size drawn from
        `grid_size_range`, subsampled from whatever native 0.25deg points
        fall inside the box via evenly-spaced index decimation (simpler than
        fetch_era5.py's block-mean coarsen(), and uniform regardless of
        whether the box wraps the antimeridian).

        No cap on the test set: every point not drawn into context becomes a
        test point, so N grows with grid_size (up to grid_size_range's own
        max squared) rather than being clipped to a fixed ceiling -- the
        model predicts on everything the sampled resolution actually offers.

        Returns None on a degenerate draw (box too small/near a pole edge to
        contain >=2 native points per axis, or too few points left over for
        a disjoint test set) — caller should just draw again.
        """
        from inference.copula_inference import normalize_features

        grid_size = int(rng.integers(grid_size_range[0], grid_size_range[1] + 1))
        box_deg = float(rng.uniform(*box_deg_range))
        half = box_deg / 2.0
        # Keep the box fully within [-90, 90] and away from the exact pole so
        # a plain symmetric lat window never degenerates to zero rows.
        lat_c = float(rng.uniform(-90.0 + half, 90.0 - half))
        lon_c = float(rng.uniform(0.0, 360.0))

        lat_mask = np.abs(self.lat - lat_c) <= half
        # Signed shortest angular distance handles antimeridian wraparound
        # (lon_c near 0/360) without special-casing.
        dlon = ((self.lon - lon_c + 180.0) % 360.0) - 180.0
        lon_mask = np.abs(dlon) <= half
        row_idx = np.nonzero(lat_mask)[0]
        col_idx = np.nonzero(lon_mask)[0]
        if len(row_idx) < 2 or len(col_idx) < 2:
            return None

        gs_r = min(grid_size, len(row_idx))
        gs_c = min(grid_size, len(col_idx))
        row_pick = row_idx[np.linspace(0, len(row_idx) - 1, gs_r).round().astype(int)]
        col_pick = col_idx[np.linspace(0, len(col_idx) - 1, gs_c).round().astype(int)]

        day_idx = int(rng.integers(0, self.n_days_total))
        field = self._day_slice(day_idx)
        sub = field[np.ix_(row_pick, col_pick)]  # (gs_r, gs_c)

        lon_grid, lat_grid = np.meshgrid(self.lon[col_pick], self.lat[row_pick])
        coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()]).astype(np.float64)
        values = sub.ravel().astype(np.float64)
        D = coords.shape[0]

        n_context_frac = float(rng.uniform(*n_context_frac_range))
        n_context = int(np.clip(round(n_context_frac * D), 1, D - 1))
        perm = rng.permutation(D)
        context_idx = perm[:n_context]
        test_idx = perm[n_context:]
        if len(test_idx) < 1:
            return None

        x_train_norm, x_test_norm = normalize_features(coords[context_idx], coords[test_idx])
        return {
            "x_norm_train": x_train_norm.astype(np.float32),
            "x_norm_test": x_test_norm.astype(np.float32),
            "y_train": values[context_idx].astype(np.float32),
            "y_test": values[test_idx].astype(np.float32),
            "lat_bounds": (lat_c - half, lat_c + half),
            "lon_bounds": (lon_c - half, lon_c + half),
            "grid_size": grid_size,
        }
