"""Empirical spatial statistics of an ERA5 field: observation targets, correlation, distance binning, Moran's I."""

from __future__ import annotations

import numpy as np

__all__ = [
    "compute_persistence_residuals",
    "compute_raw_temperature_observations",
    "get_ground_truth_observations",
    "empirical_spatial_correlation",
    "morans_i",
    "bin_correlation_by_distance",
    "pair_counts_by_distance",
]


def compute_persistence_residuals(field_all: np.ndarray) -> np.ndarray:
    """24 h persistence residuals E_t = Z_t - Z_{t-1} for daily snapshots field_all (n, H, W); returns (n - 1, H*W)."""
    n = field_all.shape[0]
    if n < 2:
        raise ValueError(f"Need >= 2 time snapshots to form 24h persistence residuals, got {n}.")
    flat = field_all.reshape(n, -1)
    return flat[1:] - flat[:-1]


def compute_raw_temperature_observations(field_all: np.ndarray) -> np.ndarray:
    """Daily temperatures (n, H*W), no differencing."""
    n = field_all.shape[0]
    return field_all.reshape(n, -1)


def get_ground_truth_observations(field_all: np.ndarray, target: str) -> np.ndarray:
    """Observation matrix (n, H*W) for target "raw" or "persistence"."""
    if target == "raw":
        return compute_raw_temperature_observations(field_all)
    if target == "residual":
        return compute_persistence_residuals(field_all)
    raise ValueError(f"Unknown target '{target}', choose from 'raw' or 'residual'.")


def empirical_spatial_correlation(data: dict, target: str = "raw") -> np.ndarray:
    """Pearson correlation matrix (D, D) of the target observations."""
    observations = get_ground_truth_observations(data["t2m"], target)
    return np.corrcoef(observations.T)


def morans_i(field: np.ndarray) -> float:
    """Global Moran's I with rook adjacency on an (H, W) grid (+1 smooth, 0 random, negative checkerboard)."""
    x = field - field.mean()
    cross_h = x[:, :-1] * x[:, 1:]
    cross_v = x[:-1, :] * x[1:, :]
    n_edges = cross_h.size + cross_v.size
    numerator = cross_h.sum() + cross_v.sum()
    denominator = (x**2).sum()
    return float(field.size * numerator / (n_edges * denominator))


def _bin_indices(d: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Bin index per distance, -1 outside [bin_edges[0], bin_edges[-1]]."""
    n_bins = len(bin_edges) - 1
    bin_idx = np.digitize(d, bin_edges) - 1
    bin_idx[(d < bin_edges[0]) | (d > bin_edges[-1])] = -1
    bin_idx[bin_idx == n_bins] = n_bins - 1  # d == bin_edges[-1] exactly
    return bin_idx


def bin_correlation_by_distance(R: np.ndarray, dist: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Mean upper-triangle correlation per distance bin."""
    iu = np.triu_indices_from(R, k=1)
    corr, d = R[iu], dist[iu]
    n_bins = len(bin_edges) - 1
    bin_idx = _bin_indices(d, bin_edges)
    means = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            means[b] = corr[mask].mean()
    return means


def pair_counts_by_distance(dist: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Number of upper-triangle pairs per distance bin."""
    iu_dist = dist[np.triu_indices_from(dist, k=1)]
    n_bins = len(bin_edges) - 1
    bin_idx = _bin_indices(iu_dist, bin_edges)
    counts = np.bincount(bin_idx[bin_idx >= 0], minlength=n_bins)
    return counts[:n_bins]
