"""Joint NLL under Sklar's theorem for any (quantile_grid, probs, R), via loss.y_space_nll.

Two conventions share the key names "copula"/"marginal":
    own marginal (compute_joint_nll, loss.y_space_nll, classical
        y_space_nlls, eval_checkpoint total_nlls): each method uses its own
        marginal and z; "total" is comparable across methods.
    shared marginal (classical nlls / corr_nll_single, eval_checkpoint
        _print_table): every method uses the same z_test and only R differs;
        there is no marginal or total.
"""

from __future__ import annotations


import numpy as np
import torch
from scipy.stats import norm


from copula_inter.loss import y_space_nll  # noqa: E402

__all__ = ["compute_joint_nll", "compute_pit", "kfold_loo_pit"]


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
    quantile_fn,
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


def compute_joint_nll(
    quantile_grid: np.ndarray,
    probs: np.ndarray,
    R: np.ndarray,
    y_true: np.ndarray,
    eps: float = 1e-6,
) -> dict:
    """Joint NLL of y_true: marginals from (quantile_grid, probs), dependence from R.

    Args:
        quantile_grid: (N, Q), quantile_grid[i, j] = F_i^{-1}(probs[j]).
        probs: (Q,).
        R: (N, N) correlation matrix.
        y_true: (N,).
        eps: probit clamp.

    Returns:
        {"total", "copula", "marginal"} per-instance floats.
    """
    n = quantile_grid.shape[0]
    z, log_pdf = compute_pit(quantile_grid, probs, y_true, eps)

    Sigma = torch.as_tensor(R, dtype=torch.float32).unsqueeze(0)
    z_t = torch.as_tensor(z, dtype=torch.float32).unsqueeze(0)
    log_pdf_t = torch.as_tensor(log_pdf, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, n, dtype=torch.bool)

    out = y_space_nll(Sigma, z_t, log_pdf_t, mask)
    return {k: float(v) for k, v in out.items()}
