"""spatial_housing.py — Benchmark 1: spatial interpolation on California Housing.

Clustered split: a random anchor row defines a spatial test cluster (its
n_test nearest neighbors by lat/lon), and n_ctx context rows are sampled
uniformly from everything else — the same "dense target patch + global
context excluding the patch" idea already used by
plots/run_era5_eval.py::sample_icl_task_from_era5, adapted from a lat/lon
grid to tabular rows.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_california_housing

__all__ = ["load_split"]


def _get_data():
    ds = fetch_california_housing()
    X = ds.data
    y = ds.target
    lat_idx = list(ds.feature_names).index("Latitude")
    lon_idx = list(ds.feature_names).index("Longitude")
    return X, y, lat_idx, lon_idx


def load_split(
    n_ctx: int = 64, n_test: int = 32, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_train, y_train, X_test, y_test), raw feature scale."""
    X, y, lat_idx, lon_idx = _get_data()
    rng = np.random.default_rng(seed)

    n_rows = X.shape[0]
    anchor = rng.integers(n_rows)
    coords = X[:, [lat_idx, lon_idx]]
    dists = np.linalg.norm(coords - coords[anchor], axis=1)
    test_idx = np.argsort(dists)[:n_test]

    remaining = np.setdiff1d(np.arange(n_rows), test_idx)
    ctx_idx = rng.choice(remaining, size=min(n_ctx, len(remaining)), replace=False)

    return X[ctx_idx], y[ctx_idx], X[test_idx], y[test_idx]
