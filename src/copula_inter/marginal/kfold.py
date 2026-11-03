"""PIT of a target from any quantile grid, and the generic K-fold leave-one-out PIT loop.

Backend-agnostic: every marginal backend (marginal.backends) produces a
(n_query, Q) quantile grid in raw y units, and these two functions turn it into
Gaussianized residuals z = Phi^{-1}(F(y)) and log-densities.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import norm

__all__ = ["compute_pit", "kfold_loo_pit"]


def compute_pit(
    quantile_grid: np.ndarray, probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """PIT z-values and log-densities of y_true from a (quantile_grid, probs) pair.

    u = F(y) by linear interpolation; density f = 1 / Q'(u) from a local finite
    difference of the quantile function.
    """
    n = quantile_grid.shape[0]
    u = np.empty(n)
    log_pdf = np.empty(n)
    for i in range(n):
        u[i] = np.interp(y_true[i], quantile_grid[i], probs)
        j = int(np.clip(np.searchsorted(probs, u[i]), 1, len(probs) - 1))
        dQ = quantile_grid[i, j] - quantile_grid[i, j - 1]
        dP = probs[j] - probs[j - 1]
        slope = dQ / max(dP, eps)
        log_pdf[i] = -np.log(max(slope, eps))
    u_clamped = np.clip(u, eps, 1 - eps)
    z = norm.ppf(u_clamped)
    return z, log_pdf


def kfold_loo_pit(
    quantile_fn: Callable[..., np.ndarray],
    X_train: np.ndarray,
    y_train: np.ndarray,
    probs: np.ndarray,
    *,
    k_folds: int = 10,
    eps: float = 1e-6,
    seed: int = 0,
) -> np.ndarray:
    """K-fold PIT of the training set with quantile_fn(X_context, y_context, X_query, fold_idx) -> quantile_grid.

    Returns:
        (n_train,) Gaussianized residuals.
    """
    n = len(y_train)
    k_folds = min(k_folds, n)
    rng = np.random.default_rng(seed)
    fold_id = rng.permutation(n) % k_folds

    z = np.empty(n)
    for k in range(k_folds):
        held = fold_id == k
        rest = ~held
        if held.sum() == 0 or rest.sum() == 0:
            continue
        qgrid_held = quantile_fn(X_train[rest], y_train[rest], X_train[held], k)
        z_held, _ = compute_pit(qgrid_held, probs, y_train[held], eps)
        z[held] = z_held
    return z
