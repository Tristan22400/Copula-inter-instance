"""tabicl_utils.py — thin wrappers around the public ``tabicl.TabICLRegressor``
sklearn-compatible interface, shared by every benchmark that needs TabICL
marginals (as opposed to inference/copula_inference.py's low-level TabICL
model class, which does no target scaling of its own)."""

from __future__ import annotations

import numpy as np

__all__ = ["make_tabicl_regressor", "tabicl_quantiles", "tabicl_loo_pit"]


def make_tabicl_regressor(checkpoint: str | None = None, device: str | None = None):
    """Construct one ``tabicl.TabICLRegressor``, meant to be reused across
    every ``.fit()`` call in a run (as ``plots/generate_plots.py::
    TabICLv2_Regressor`` documents: "Loading it once and reusing the
    instance across repeated .fit() calls ... avoids reloading the backbone
    weights every time"). K-fold PIT alone needs ~10 fit/predict calls per
    episode, so a fresh instance per call would be wasteful.
    """
    from tabicl import TabICLRegressor

    kwargs = {"device": device} if device is not None else {}
    if checkpoint is not None:
        kwargs["checkpoint_version"] = checkpoint
    return TabICLRegressor(**kwargs)


def tabicl_quantiles(
    regressor, X_context: np.ndarray, y_context: np.ndarray, X_query: np.ndarray, probs: np.ndarray
) -> np.ndarray:
    """Fit the public ``tabicl.TabICLRegressor`` in-context and return its
    quantile grid at X_query, in RAW y-units.

    Deliberately uses the sklearn-compatible ``TabICLRegressor`` (as
    ``plots/generate_plots.py::TabICLv2_Regressor`` does) rather than the
    low-level ``TabICL`` model class ``inference/copula_inference.py`` calls
    directly. The low-level class does no target scaling of its own — see
    ``tabicl_upstream/src/tabicl/_model/tabicl.py`` (no normalize/scale in
    its forward pass) vs. ``tabicl_upstream/src/tabicl/_sklearn/regressor.py``
    (``fit()`` step 1 fits a fresh ``StandardScaler`` on y, ``predict()`` step
    5 inverse-transforms back) — so callers of the low-level class must
    replicate that scaling themselves to get sane quantiles on real-scale
    targets. Going through ``TabICLRegressor`` directly means that scaling
    (plus its outlier clipping and ensembling) is the canonical, tested
    implementation instead of a hand-rolled stand-in.

    Args:
        regressor : a TabICLRegressor instance (see make_tabicl_regressor)
        X_context : (n_ctx, d)
        y_context : (n_ctx,) — RAW scale, not pre-normalized
        X_query   : (n_q, d)
        probs     : (Q,) — probability levels to query

    Returns:
        quantile_grid: (n_q, Q), RAW y-units
    """
    regressor.fit(X_context, y_context)
    return regressor.predict(X_query, output_type="quantiles", alphas=list(probs))


def tabicl_loo_pit(
    regressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    probs: np.ndarray,
    k_folds: int = 10,
    eps: float = 1e-6,
    seed: int = 0,
) -> np.ndarray:
    """K-fold leave-fold-out PIT via TabICLRegressor: the same idea as
    inference/copula_inference.py::loo_pit, but through the public,
    canonically-scaled TabICLRegressor interface instead of the low-level
    TabICL model class. For each fold, fits on the other folds and PIT-
    transforms the held-out fold's true y against the fitted quantile grid.

    Returns:
        Z_train: (n_train,) — Gaussianized PIT residuals
    """
    from eval.metrics.joint_nll import compute_pit

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
        qgrid_held = tabicl_quantiles(regressor, X_train[rest], y_train[rest], X_train[held], probs)
        z_held, _ = compute_pit(qgrid_held, probs, y_train[held], eps)
        z[held] = z_held
    return z
