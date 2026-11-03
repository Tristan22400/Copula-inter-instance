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

from copula_inter.copula.loss import y_space_nll
from copula_inter.marginal.kfold import compute_pit

__all__ = ["compute_joint_nll"]


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
