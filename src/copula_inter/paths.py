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
# KAUST 2022 competition, sub-competitions 1a/1b: the four training CSVs.
KAUST_DATA_DIR = REAL_DATA_DIR / "kaust2022"
# EMNIST balanced split: the NIST idx files, gzipped as distributed.
EMNIST_DATA_DIR = REAL_DATA_DIR / "emnist"
# California Housing: sklearn's data_home for fetch_california_housing.
CALIFORNIA_DATA_DIR = REAL_DATA_DIR / "california"
# Meuse (R package sp): meuse.csv.
MEUSE_DATA_DIR = REAL_DATA_DIR / "meuse"
# OpenML-CTR23 (suite 353): sklearn's data_home for fetch_openml.
CTR23_DATA_DIR = REAL_DATA_DIR / "ctr23"
# Neural-process benchmark tasks exported by scripts/npbench/export_eeg.py (EEG of Bruinsma et al. 2023).
NPBENCH_DATA_DIR = REAL_DATA_DIR / "npbench"
