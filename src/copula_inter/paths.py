"""Checkout-relative default locations: the repo root, the checkpoints tree, the ERA5 caches and the real-data benchmarks.

Every module that needs a path outside the package reads it from here, so the
defaults stay correct wherever a module sits inside the package.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/copula_inter/paths.py -> the checkout root.
REPO_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

_DEFAULT_ERA5_CACHE = REPO_ROOT / "eval" / "data" / "cache"
# Regional fetches and the static fields; the ERA5_CACHE_DIR env var overrides it.
ERA5_CACHE_DIR = Path(os.environ.get("ERA5_CACHE_DIR", _DEFAULT_ERA5_CACHE))
# Global monthly corpora: training, and the held-out validation year (fetch_global --cache-dir).
ERA5_GLOBAL_CACHE_DIR = _DEFAULT_ERA5_CACHE / "era5_global"
ERA5_GLOBAL_VAL_CACHE_DIR = _DEFAULT_ERA5_CACHE / "era5_global_val"
# Real-data benchmarks (eval/data/real): one <name>/ directory per dataset holding the raw files,
# placed by hand, plus the cleaned .npz cache; the REAL_DATA_DIR env var overrides the root
# (a git worktree points it at the main checkout's eval/data/cache).
REAL_DATA_DIR = Path(os.environ.get("REAL_DATA_DIR", _DEFAULT_ERA5_CACHE))
# SDWPF (KDD Cup 2022) wind farm: the SCADA and turbine-location CSVs.
SDWPF_DATA_DIR = REAL_DATA_DIR / "sdwpf"
