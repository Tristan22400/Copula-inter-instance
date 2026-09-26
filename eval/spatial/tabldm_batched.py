"""Batched PIT for Xiaomi-TabLDM across episodes (data.z_train_source=tabldm).

TabLDM.predict_stats takes a leading axis of independent tables, so episodes
are fitted with the regressor's own code, grouped by preprocessed shape
(T, H, train_size), forwarded once per group through _batch_forward, and
de-standardized and pooled with each episode's own y scaler.
enhance_candidates=True and a fitted KV cache are rejected.
tests/test_tabldm_batched.py checks it against the per-episode path.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = ["tabldm_run_pit_batched"]


def _episode_member_batch(
    regressor: Any, x_support: np.ndarray, y_support: np.ndarray, x_query: np.ndarray
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Fit the regressor on one episode and return its (members, T, H) / (members, train_size) inputs and its y scaler."""
    from tabldm._sklearn.sklearn_utils import validate_data

    if not getattr(regressor, "_load_model_cached", False):
        orig_load = regressor._load_model

        def _cached_load() -> None:
            if getattr(regressor, "model_", None) is None:
                orig_load()

        regressor._load_model = _cached_load
        regressor._load_model_cached = True

    regressor.fit(x_support, y_support)

    if getattr(regressor, "enhance_candidates", False):
        raise RuntimeError(
            "tabldm_batched does not support enhance_candidates=True: TabLDM's "
            "enhanced predict path only supports output_type='mean' (see "
            "TabLDMRegressor.predict), so it exposes no quantiles to batch."
        )
    if getattr(regressor, "model_kv_cache_", None) is not None:
        raise RuntimeError(
            "tabldm_batched does not support a fitted KV cache: TabLDM's predict "
            "takes a separate cache-keyed branch (_batch_forward_with_cache) that "
            "this module does not mirror. Leave TabLDM's kv_cache at its default "
            "(off), as marginal_backends.py::make_regressor does."
        )

    xq = validate_data(regressor, x_query, reset=False, dtype=None, skip_check_array=True)
    xq = regressor.X_encoder_.transform(xq)
    data = regressor.ensemble_generator_.transform(xq, mode="both")

    # Concatenate member inputs in predict()'s order and average over them.
    xs = np.concatenate([g[0] for g in data.values()], axis=0)  # (members, T, H)
    ys = np.concatenate([g[1] for g in data.values()], axis=0)  # (members, train_size)
    return xs, ys, regressor.y_scaler_


def _group_episode_batches(per_episode: Sequence[tuple[np.ndarray, np.ndarray, Any]]) -> list[list[int]]:
    """Group episodes whose preprocessed inputs have the same shape."""
    groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
    for b, (xs, ys, _) in enumerate(per_episode):
        groups.setdefault((xs.shape, ys.shape), []).append(b)
    return list(groups.values())


def _quantile_bank_batched(
    regressor: Any,
    X_context: list,
    y_context: list,
    X_query: list,
    probs: np.ndarray,
) -> np.ndarray:
    """(B, n_query, len(probs)) quantiles in raw y units, one fused forward per shape group."""
    B = len(X_context)
    per_episode = [_episode_member_batch(regressor, X_context[b], y_context[b], X_query[b]) for b in range(B)]
    banks: list[np.ndarray | None] = [None] * B
    for indices in _group_episode_batches(per_episode):
        members = per_episode[indices[0]][0].shape[0]
        xs = np.concatenate([per_episode[b][0] for b in indices], axis=0)
        ys = np.concatenate([per_episode[b][1] for b in indices], axis=0)
        out = regressor._batch_forward(xs, ys, output_type="quantiles", alphas=list(probs))
        out = np.asarray(out).reshape(len(indices), members, -1, len(probs))
        for local, b in enumerate(indices):
            arr = per_episode[b][2].inverse_transform(out[local].reshape(-1, 1)).reshape(out[local].shape)
            banks[b] = arr.mean(axis=0)
    filled = [bank for bank in banks if bank is not None]
    assert len(filled) == B
    bank = np.stack(filled).astype(np.float64)
    return bank


def tabldm_run_pit_batched(
    regressor: Any,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    k_folds: int = 10,
    probs_n: int = 99,
    eps: float = 1e-6,
    seed: int = 0,
) -> dict:
    """run_pit_batched for TabLDM, via the shared K-fold driver."""
    from eval.spatial._batched_pit import run_kfold_pit_batched

    return run_kfold_pit_batched(
        lambda X_ctx, y_ctx, X_qry, probs: _quantile_bank_batched(regressor, X_ctx, y_ctx, X_qry, probs),
        X_train,
        Y_train,
        X_test,
        Y_test,
        k_folds=k_folds,
        probs_n=probs_n,
        eps=eps,
        seed=seed,
    )
