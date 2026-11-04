"""Cluster bootstrap for the mean of per-episode statistics drawn from dependent groups.

Episodes that share a cluster (an SDWPF day, an image, a window cell) share
their dependence, so resampling episodes i.i.d. overstates significance.
Resampling whole clusters with replacement keeps it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["cluster_bootstrap_ci"]


def cluster_bootstrap_ci(
    values: np.ndarray | Sequence[float],
    clusters: np.ndarray | Sequence[object],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """(mean, lower, upper): the mean of values and its percentile interval from resampling clusters.

    Each replicate draws as many clusters as there are, with replacement, and
    takes every value of a drawn cluster; its statistic is the mean over the
    values drawn (clusters of unequal size weigh by size, as in the point mean).
    """
    vals = np.asarray(values, dtype=np.float64)
    labels = np.asarray([str(c) for c in clusters])
    if vals.shape != labels.shape:
        raise ValueError(f"values {vals.shape} and clusters {labels.shape} must align")
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    _, inverse = np.unique(labels, return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    sums = np.bincount(inverse, weights=vals, minlength=n_clusters)
    counts = np.bincount(inverse, minlength=n_clusters).astype(np.float64)
    draws = np.random.default_rng(seed).integers(0, n_clusters, size=(n_boot, n_clusters))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return float(vals.mean()), float(lo), float(hi)
