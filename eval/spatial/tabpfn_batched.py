"""Batched PIT for TabPFN v3 across episodes (data.z_train_source=tabpfn), via the public TabPFNRegressor.predict_batched.

All episodes in a call must share their shapes. Needs TABPFN_TOKEN (never
commit it). tests/test_tabpfn_batched.py checks it against the per-episode
path.
"""

from __future__ import annotations

import numpy as np

__all__ = ["tabpfn_run_pit_batched"]


def _quantile_bank_batched(
    regressor, X_context: list, y_context: list, X_query: list, probs: np.ndarray,
) -> np.ndarray:
    """(B, query_rows, len(probs)) quantiles in raw y units from one predict_batched call."""
    results = regressor.predict_batched(
        list(X_context), list(y_context), list(X_query),
        output_type="quantiles", quantiles=list(probs),
    )
    # (n_quantiles, n_query) -> (n_query, n_quantiles).
    return np.stack([np.asarray(r).T for r in results], axis=0)  # (B, query_rows, len(probs))


def tabpfn_run_pit_batched(
    regressor, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray,
    k_folds: int = 10, probs_n: int = 99, eps: float = 1e-6, seed: int = 0,
) -> dict:
    """run_pit_batched for TabPFN, via the shared K-fold driver."""
    from eval.spatial._batched_pit import run_kfold_pit_batched

    return run_kfold_pit_batched(
        lambda X_ctx, y_ctx, X_qry, probs: _quantile_bank_batched(
            regressor, X_ctx, y_ctx, X_qry, probs
        ),
        X_train, Y_train, X_test, Y_test,
        k_folds=k_folds, probs_n=probs_n, eps=eps, seed=seed,
    )
