"""Per-quantile marginal calibration error (ECE), shared by Phase-A validation and the eval runners."""

from __future__ import annotations

import numpy as np

__all__ = ["compute_quantile_ece"]


def compute_quantile_ece(
    y_true: np.ndarray,
    y_pred_quantiles: np.ndarray,
    quantiles: "list | np.ndarray",
) -> "tuple[float, np.ndarray]":
    """
    Compute the quantile-regression Expected Calibration Error (ECE) — per-
    cell MARGINAL calibration (unlike calc_kendall_pit et al. below, which
    probe the JOINT/independence-copula assumption on top of the marginals).

    For each nominal quantile level, the empirical coverage is the fraction
    of instances where the true value falls at or below the predicted
    quantile value. The ECE is the mean absolute gap between the nominal
    levels and their empirical coverages (0 for perfect calibration).

    Args:
        y_true: 1D array of shape (n_samples,) with observed targets.
        y_pred_quantiles: 2D array of shape (n_samples, n_quantiles); column
            k holds the predicted value for quantiles[k].
        quantiles: Nominal quantile levels in (0, 1), e.g. [0.1, ..., 0.9].

    Returns:
        Tuple (ece, empirical_coverage), where empirical_coverage has shape
        (n_quantiles,) and aligns positionally with `quantiles`.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred_quantiles = np.asarray(y_pred_quantiles)
    quantiles = np.asarray(quantiles, dtype=float)

    if y_pred_quantiles.ndim != 2:
        raise ValueError("y_pred_quantiles must be 2D (n_samples, n_quantiles).")
    if y_pred_quantiles.shape[0] != y_true.shape[0]:
        raise ValueError(f"y_true has {y_true.shape[0]} samples but y_pred_quantiles has {y_pred_quantiles.shape[0]}.")
    if y_pred_quantiles.shape[1] != quantiles.shape[0]:
        raise ValueError(
            f"y_pred_quantiles has {y_pred_quantiles.shape[1]} quantile columns "
            f"but {quantiles.shape[0]} nominal quantiles were given."
        )

    empirical_coverage = np.mean(y_true[:, None] <= y_pred_quantiles, axis=0)
    ece = float(np.mean(np.abs(quantiles - empirical_coverage)))
    return ece, empirical_coverage
