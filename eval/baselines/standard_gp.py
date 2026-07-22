"""standard_gp.py — fully independent end-to-end Gaussian Process baseline.

Unlike the Independent and Copula methods (which both reuse TabICL's marginal
quantile grid and only differ in the correlation matrix R), this baseline
fits its own joint Gaussian directly in raw y-space, with no dependency on
TabICL at all. This is deliberate: it gives a reference model whose marginal
*and* joint quality are both independently estimated, rather than sharing
TabICL's marginals like the other two methods.
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern

__all__ = ["fit_predict"]


def fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Matern-5/2 + white-noise GP by marginal-likelihood optimization
    and return the joint predictive mean and covariance at X_test.

    Returns:
        mean: (N,)
        cov : (N, N)
    """
    kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-2)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=seed,
    )
    gp.fit(X_train, y_train)
    mean, cov = gp.predict(X_test, return_cov=True)
    return mean, cov
