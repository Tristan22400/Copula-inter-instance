"""Helpers around the public tabicl.TabICLRegressor (which scales y itself)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tabicl import TabICLRegressor

__all__ = ["make_tabicl_regressor", "tabicl_quantiles", "tabicl_loo_pit"]

_REPO_ROOT = Path(__file__).resolve().parents[1]


def make_tabicl_regressor(checkpoint: str | None = None, device: str | None = None) -> TabICLRegressor:
    """Build one TabICLRegressor to reuse across .fit() calls.

    checkpoint is a local .ckpt/.pt file (model_path) or a jingang/TabICL HF
    filename (checkpoint_version).
    """
    from tabicl import TabICLRegressor

    kwargs: dict[str, Any] = {"device": device} if device is not None else {}
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.is_absolute() or "/" in checkpoint or checkpoint_path.suffix == ".pt":
            if not checkpoint_path.is_absolute():
                checkpoint_path = _REPO_ROOT / checkpoint_path
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Marginal checkpoint not found: {checkpoint_path}")
            kwargs.update(model_path=str(checkpoint_path), allow_auto_download=False)
        else:
            kwargs["checkpoint_version"] = checkpoint
    return TabICLRegressor(**kwargs)


def tabicl_quantiles(
    regressor: Any, X_context: np.ndarray, y_context: np.ndarray, X_query: np.ndarray, probs: np.ndarray
) -> np.ndarray:
    """Fit TabICLRegressor on the context and return its quantiles at X_query.

    Args:
        regressor: TabICLRegressor.
        X_context: (n_ctx, d).
        y_context: (n_ctx,) raw targets.
        X_query: (n_q, d).
        probs: (Q,) levels.

    Returns:
        (n_q, Q) quantile grid in raw y units.
    """
    regressor.fit(X_context, y_context)
    return regressor.predict(X_query, output_type="quantiles", alphas=list(probs))


def tabicl_loo_pit(
    regressor: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    probs: np.ndarray,
    k_folds: int = 10,
    eps: float = 1e-6,
    seed: int = 0,
) -> np.ndarray:
    """K-fold PIT of the training set through TabICLRegressor (eval/metrics/joint_nll.kfold_loo_pit).

    Returns:
        (n_train,) Gaussianized residuals.
    """
    from eval.metrics.joint_nll import kfold_loo_pit

    return kfold_loo_pit(
        lambda Xc, yc, Xq, _fold: tabicl_quantiles(regressor, Xc, yc, Xq, probs),
        X_train,
        y_train,
        probs,
        k_folds=k_folds,
        eps=eps,
        seed=seed,
    )
