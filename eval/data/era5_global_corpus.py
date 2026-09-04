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
    """Loads every era5_global_t2m_*.nc file under `cache_dir` into RAM once
    (native grid is the same lat/lon for every file — only checked, not
    re-derived, per file) and exposes `sample_episode` for random-region,
    random-resolution episode draws.

    Two construction paths:
      - `GlobalERA5Corpus(cache_dir)`: reads from disk into a private copy.
        Fine for a single-process use (e.g. build_era5_fixed_val_batches).
      - `GlobalERA5Corpus.from_shared(shared)`: attaches to arrays another
        process already loaded via `load_shared_corpus_arrays` and put in
        shared memory -- what src/era5_live_dataset.py's DataLoader workers
        use, so `live_tabicl_num_workers` copies of this corpus don't each
        cost their own ~15GB of system RAM.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        *,
        max_months: int | None = None,
        _shared: dict | None = None,
    ):
        """`max_months` caps how many monthly files are read (the most recent
        ones, since paths are sorted by YYYYMM). The full 120-month corpus is
        ~15GB resident; a caller that only needs an occasional real-data episode
        -- src/finetune_marginal.py's Phase-A mixture, which runs the marginal
        in the MAIN process and so has no shared-memory worker trick to amortize
        the load -- can bound that without needing a separate, hand-curated
        cache directory. None (the default) reads everything, unchanged."""
        if _shared is not None:
            # See from_shared()/load_shared_corpus_arrays() below -- attach
            # to already-loaded, already-shared torch storage instead of
            # touching disk. .numpy() on a share_memory_()'d CPU tensor is a
            # zero-copy view over that shared storage, so every downstream
            # numpy op in sample_episode/sample_episode_fixed_shape below
            # works unmodified against the SAME physical memory every other
            # attached worker reads.
            self.lat = _shared["lat"].numpy()
            self.lon = _shared["lon"].numpy()
            self._t2m_by_month = [t.numpy() for t in _shared["t2m_by_month"]]
            self.static = {k: v.numpy() for k, v in _shared["static"].items()}
        else:
            paths = sorted(glob.glob(os.path.join(cache_dir, "era5_global_t2m_*.nc")))
            if not paths:
                raise FileNotFoundError(
                    f"No cached global ERA5 files found under {cache_dir!r} — run "
                    "`python eval/data/fetch_era5_global.py` first to populate the "
                    "worldwide finetuning corpus."
                )
            if max_months is not None and max_months > 0:
                paths = paths[-int(max_months):]
            self.lat: np.ndarray | None = None
            self.lon: np.ndarray | None = None
            self.static: dict[str, np.ndarray] = {}
            self._t2m_by_month: list[np.ndarray] = []
            for p in paths:
                t2m, lat, lon, static = _load_month(p)
                if self.lat is None:
                    self.lat, self.lon = lat, lon
                    if static:
                        self.static = static
                self._t2m_by_month.append(t2m)
            if not self.static:
                st = load_static()
                self.static = {k: st[k].astype(np.float32) for k in STATIC_VARS}
        day_counts = [a.shape[0] for a in self._t2m_by_month]
        self.n_days_total = int(sum(day_counts))
        self._cum_days = np.cumsum([0] + day_counts)

    @classmethod
    def from_shared(cls, shared: dict) -> "GlobalERA5Corpus":
        """Attach to a corpus already loaded (once) by load_shared_corpus_arrays
        in the main process, instead of re-reading every era5_global_t2m_*.nc
        file from disk into a private copy. Meant to be called inside each
        DataLoader worker's __iter__ (see LiveERA5Dataset) -- cheap (no I/O,
        just wrapping already-shared torch storage as numpy views)."""
        return cls(_shared=shared)

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
        """Like `sample_episode`, but `grid_size`/`n_context` are exact
        integers instead of ranges, and this returns None (redraw) rather
        than silently clipping to fewer native points whenever the drawn box
        doesn't contain a full `grid_size` x `grid_size` grid along either
        axis. That makes every successful draw's P (`n_context`) and N
        (`grid_size**2 - n_context`) identical, letting a caller collect a
        whole *group* of episodes (region/day/box_deg still vary per draw)
        that share P/N -- required for src/pit.py::run_pit_batched, which
        can only PIT a batch of episodes in one TabICL call when they all
        share P/N. Mirrors src/live_dataset.py::LiveGPDataset's group_size
        mechanism (see its docstring: group_size=1 measured ~1.2s/episode of
        TabICL PIT overhead vs ~0.03s/episode grouped).
        """
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
        }


def load_shared_corpus_arrays(cache_dir: str) -> dict:
    """Load `cache_dir`'s corpus from disk ONCE and move its arrays into
    shared CPU memory (torch.Tensor.share_memory_()), so a DataLoader's
    spawned workers can each attach to the SAME physical memory via
    GlobalERA5Corpus.from_shared instead of every worker loading (and
    holding, for the life of the run) its own private ~15GB copy.

    Must be called in the MAIN process before the DataLoader's workers spawn
    -- share_memory_() tensors created ahead of spawn are what let
    torch.multiprocessing's spawn-safe pickling hand out a shared-memory
    handle to each worker instead of serializing (copying) the full array,
    exactly the pattern src/live_dataset.py::build_live_train_loader already
    uses for kernel_weights/tabicl_mix_weights (see LiveGPDataset's
    docstring) -- applied here to O(10GB) corpus data instead of a
    handful-of-floats tensor. Removes the RAM-per-worker constraint that
    used to force live_tabicl_num_workers down to 1 regardless of GPU
    headroom: with one shared copy total regardless of worker count,
    src/era5_live_dataset.py::build_era5_train_loader now sizes
    live_tabicl_num_workers via live_dataset.py's plain GPU-only-bound
    resolve_live_tabicl_num_workers, same as the synthetic-GP path.
    """
    corpus = GlobalERA5Corpus(cache_dir)
    return {
        "lat": torch.from_numpy(corpus.lat).share_memory_(),
        "lon": torch.from_numpy(corpus.lon).share_memory_(),
        "t2m_by_month": [torch.from_numpy(a).share_memory_() for a in corpus._t2m_by_month],
        "static": {k: torch.from_numpy(v).share_memory_() for k, v in corpus.static.items()},
    }
