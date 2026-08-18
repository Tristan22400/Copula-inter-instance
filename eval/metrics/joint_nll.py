"""joint_nll.py — Joint NLL under Sklar's theorem, for arbitrary (quantile_grid, probs, R).

Thin wrapper around ``src/loss.py::y_space_nll`` — the copula-NLL + marginal-NLL
decomposition is NOT re-derived here; ``y_space_nll`` already implements it
(dense Cholesky via ``_safe_cholesky``, masked padding) and is reused verbatim.
The only new code is turning a generic ``(quantile_grid, probs)`` marginal
representation (which may come from TabICL, a fitted GP, or anything else)
into the ``(z, log_pdf)`` pair ``y_space_nll`` expects — via the same
finite-difference PIT recipe already used in
``experiments/experiment_b_quantitative.py::_quantile_grid_pit`` (ported
verbatim rather than imported, since ``experiments/`` is a script directory,
not a stable package boundary).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from scipy.stats import norm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from loss import y_space_nll  # noqa: E402

__all__ = ["compute_joint_nll", "compute_pit", "kfold_loo_pit"]


def compute_pit(
    quantile_grid: np.ndarray, probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """PIT (Z-values) and log-density for true targets from an ALREADY-BUILT
    (quantile_grid, probs) pair, via linear interpolation for u = F(y) and a
    local finite-difference slope of the quantile function for the density
    (f(z) = 1/Q'(F(z))). Verbatim port of
    experiments/experiment_b_quantitative.py::_quantile_grid_pit. Public (not
    prefixed with ``_``) because callers outside this module use it directly
    to get Z-space residuals for real datasets that have no known generative
    kernel — e.g. as an empirical ground-truth correlation proxy."""
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
    """K-fold leave-fold-out PIT, generic over how the quantile grid for the
    held-out fold is produced. ``quantile_fn(X_context, y_context, X_query,
    fold_idx) -> quantile_grid`` fits/predicts on one fold split; this
    function only owns the fold assignment and the compute_pit call, shared
    by every regressor backend that needs the same recipe (originally
    duplicated per-backend, see eval/tabicl_utils.py::tabicl_loo_pit and
    eval/spatial/marginal_backends.py::loo_pit, now both thin wrappers here).

    Returns:
        Z_train: (n_train,) — Gaussianized PIT residuals
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
    """Joint NLL of y_true under Sklar's theorem: marginals from
    (quantile_grid, probs), dependency structure from correlation matrix R.

    Args:
        quantile_grid: (N, Q) — quantile_grid[i, j] = F_i^{-1}(probs[j])
        probs        : (Q,)
        R            : (N, N) — correlation matrix
        y_true       : (N,)
        eps          : clamp before probit transform

    Returns:
        {"total": float, "copula": float, "marginal": float} — literally
        y_space_nll's own return dict (per-instance-averaged), unpacked to
        Python floats.
    """
    n = quantile_grid.shape[0]
    z, log_pdf = compute_pit(quantile_grid, probs, y_true, eps)

    Sigma = torch.as_tensor(R, dtype=torch.float32).unsqueeze(0)
    z_t = torch.as_tensor(z, dtype=torch.float32).unsqueeze(0)
    log_pdf_t = torch.as_tensor(log_pdf, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, n, dtype=torch.bool)

    out = y_space_nll(Sigma, z_t, log_pdf_t, mask)
    return {k: float(v) for k, v in out.items()}
