"""In-memory reader over the monthly global ERA5 caches (fetch_era5_global.py), cropping random regions at random resolutions."""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from scipy.io.netcdf import netcdf_file

from eval.data.fetch_era5_static import STATIC_VARS, load_static

__all__ = ["GlobalERA5Corpus", "load_shared_corpus_arrays", "STATIC_VARS"]


def _load_month(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    f = netcdf_file(path, "r", mmap=False)
    t2m = f.variables["t2m"][:].astype(np.float32).copy()  # (T, H, W)
    lat = f.variables["latitude"][:].astype(np.float64).copy()
    lon = f.variables["longitude"][:].astype(np.float64).copy()
    static = {}
    for v in STATIC_VARS:
        if v in f.variables:
            static[v] = f.variables[v][:].astype(np.float32).copy()
    f.close()
    return t2m, lat, lon, static


class GlobalERA5Corpus:
    """Global ERA5 2 m temperature corpus with random-region, random-resolution episode sampling.

    GlobalERA5Corpus(cache_dir) reads a private copy; from_shared(shared)
    attaches to arrays already placed in shared memory by
    load_shared_corpus_arrays (used by DataLoader workers).
    """

    lat: np.ndarray
    lon: np.ndarray
    static: dict[str, np.ndarray]
    # None when months are memory-mapped from _paths on demand.
    _t2m_by_month: list[np.ndarray] | None
    _paths: list[str] | None

    def __init__(
        self,
        cache_dir: str | None = None,
        *,
        max_months: int | None = None,
        lazy: bool | None = None,
        _shared: dict | None = None,
    ) -> None:
        """max_months keeps only the most recent monthly files (None = all). lazy=None memory-maps on demand when there are more than 60 months."""
        if _shared is not None:
            # Attach to shared storage (zero-copy numpy views).
            self.lat = _shared["lat"].numpy()
            self.lon = _shared["lon"].numpy()
            self._t2m_by_month = [t.numpy() for t in _shared["t2m_by_month"]]
            self.static = {k: v.numpy() for k, v in _shared["static"].items()}
            self._paths = None
            day_counts = [a.shape[0] for a in self._t2m_by_month]
        else:
            if cache_dir is None:
                raise ValueError("GlobalERA5Corpus needs cache_dir unless attaching to shared arrays")
            paths = sorted(glob.glob(os.path.join(cache_dir, "era5_global_t2m_*.nc")))
            if not paths:
                raise FileNotFoundError(
                    f"No cached global ERA5 files found under {cache_dir!r} — run "
                    "`python eval/data/fetch_era5_global.py` first to populate the "
                    "worldwide finetuning corpus."
                )
            if max_months is not None and max_months > 0:
                paths = paths[-int(max_months) :]
            self._paths = paths

            if lazy is None:
                lazy = len(paths) > 60

            lat_arr: np.ndarray | None = None
            lon_arr: np.ndarray | None = None
            static_arrs: dict[str, np.ndarray] = {}

            if lazy:
                self._t2m_by_month = None
                import calendar
                import re

                f0 = netcdf_file(paths[0], "r", mmap=True)
                lat_arr = f0.variables["latitude"][:].astype(np.float64).copy()
                lon_arr = f0.variables["longitude"][:].astype(np.float64).copy()
                for v in STATIC_VARS:
                    if v in f0.variables:
                        static_arrs[v] = f0.variables[v][:].astype(np.float32).copy()
                f0.close()

                day_counts = []
                for p in paths:
                    m = re.search(r"era5_global_t2m_(\d{4})(\d{2})\.nc", p)
                    if m:
                        day_counts.append(calendar.monthrange(int(m.group(1)), int(m.group(2)))[1])
                    else:
                        f = netcdf_file(p, "r", mmap=True)
                        day_counts.append(f.variables["t2m"].shape[0])
                        f.close()
            else:
                months: list[np.ndarray] = []
                for p in paths:
                    t2m, lat, lon, static = _load_month(p)
                    if lat_arr is None:
                        lat_arr, lon_arr = lat, lon
                        if static:
                            static_arrs = static
                    months.append(t2m)
                self._t2m_by_month = months
                day_counts = [a.shape[0] for a in months]
            if not static_arrs:
                st = load_static()
                static_arrs = {k: st[k].astype(np.float32) for k in STATIC_VARS}
            assert lat_arr is not None and lon_arr is not None
            self.lat, self.lon, self.static = lat_arr, lon_arr, static_arrs

        self.n_days_total = int(sum(day_counts))
        self._cum_days = np.cumsum([0] + day_counts)

    @classmethod
    def from_shared(cls, shared: dict) -> "GlobalERA5Corpus":
        """Wrap arrays loaded by load_shared_corpus_arrays (no I/O)."""
        return cls(_shared=shared)

    def _day_slice(self, day_global_idx: int) -> np.ndarray:
        m = int(np.searchsorted(self._cum_days, day_global_idx, side="right") - 1)
        d = day_global_idx - int(self._cum_days[m])
        if self._t2m_by_month is not None:
            return self._t2m_by_month[m][d]
        assert self._paths is not None
        f = netcdf_file(self._paths[m], "r", mmap=True)
        slice_2d = f.variables["t2m"][d].astype(np.float32).copy()
        f.close()
        return slice_2d

    def sample_episode(
        self,
        rng: np.random.Generator,
        grid_size_range: tuple[int, int],
        box_deg_range: tuple[float, float],
        n_context_frac_range: tuple[float, float],
    ) -> dict | None:
        """Draw one (region, resolution, day, context/test split) episode.

        The box centre is area-uniform on the sphere with half-width from
        box_deg_range; grid_size from grid_size_range, subsampled by evenly spaced
        index decimation. Every non-context point is a test point. Returns None on a
        degenerate draw.
        """
        from inference.copula_inference import normalize_features

        grid_size = int(rng.integers(grid_size_range[0], grid_size_range[1] + 1))
        box_deg = float(rng.uniform(*box_deg_range))
        half = box_deg / 2.0
        # Keep the box within [-90, 90] and away from the poles.
        lat_c = float(rng.uniform(-90.0 + half, 90.0 - half))
        lon_c = float(rng.uniform(0.0, 360.0))

        lat_mask = np.abs(self.lat - lat_c) <= half
        # Signed shortest angular distance handles the antimeridian.
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
        static_cols = [self.static[v][np.ix_(row_pick, col_pick)].ravel() for v in STATIC_VARS]
        coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()] + static_cols).astype(np.float64)
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

    def sample_episode_fixed_shape(
        self,
        rng: np.random.Generator,
        grid_size: int,
        box_deg_range: tuple[float, float],
        n_context: int,
    ) -> dict | None:
        """Like sample_episode with exact grid_size and n_context; returns None unless the box holds a full grid_size x grid_size grid, so all episodes share P and N."""
        box_deg = float(rng.uniform(*box_deg_range))
        half = box_deg / 2.0
        lat_c = float(rng.uniform(-90.0 + half, 90.0 - half))
        lon_c = float(rng.uniform(0.0, 360.0))

        lat_mask = np.abs(self.lat - lat_c) <= half
        dlon = ((self.lon - lon_c + 180.0) % 360.0) - 180.0
        lon_mask = np.abs(dlon) <= half
        row_idx = np.nonzero(lat_mask)[0]
        col_idx = np.nonzero(lon_mask)[0]
        if len(row_idx) < grid_size or len(col_idx) < grid_size:
            return None

        row_pick = row_idx[np.linspace(0, len(row_idx) - 1, grid_size).round().astype(int)]
        col_pick = col_idx[np.linspace(0, len(col_idx) - 1, grid_size).round().astype(int)]

        day_idx = int(rng.integers(0, self.n_days_total))
        field = self._day_slice(day_idx)
        sub = field[np.ix_(row_pick, col_pick)]  # (grid_size, grid_size)

        lon_grid, lat_grid = np.meshgrid(self.lon[col_pick], self.lat[row_pick])
        static_cols = [self.static[v][np.ix_(row_pick, col_pick)].ravel() for v in STATIC_VARS]
        coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()] + static_cols).astype(np.float64)
        values = sub.ravel().astype(np.float64)
        D = coords.shape[0]  # == grid_size**2 by construction
        if not (1 <= n_context <= D - 1):
            return None

        perm = rng.permutation(D)
        context_idx = perm[:n_context]
        test_idx = perm[n_context:]

        from inference.copula_inference import normalize_features

        x_train_norm, x_test_norm = normalize_features(coords[context_idx], coords[test_idx])
        return {
            "x_norm_train": x_train_norm.astype(np.float32),
            "x_norm_test": x_test_norm.astype(np.float32),
            "y_train": values[context_idx].astype(np.float32),
            "y_test": values[test_idx].astype(np.float32),
            # Reporting keys (used by eval/data/era5_episodes.py).
            "lat_bounds": (lat_c - half, lat_c + half),
            "lon_bounds": (lon_c - half, lon_c + half),
            "grid_size": grid_size,
        }


def load_shared_corpus_arrays(cache_dir: str) -> dict:
    """Load the corpus once in the main process and move its arrays to shared memory, for workers to attach to via GlobalERA5Corpus.from_shared."""
    # Shared memory needs the months in RAM: never the memory-mapped (lazy) layout.
    corpus = GlobalERA5Corpus(cache_dir, lazy=False)
    assert corpus._t2m_by_month is not None
    return {
        "lat": torch.from_numpy(corpus.lat).share_memory_(),
        "lon": torch.from_numpy(corpus.lon).share_memory_(),
        "t2m_by_month": [torch.from_numpy(a).share_memory_() for a in corpus._t2m_by_month],
        "static": {k: torch.from_numpy(v).share_memory_() for k, v in corpus.static.items()},
    }
