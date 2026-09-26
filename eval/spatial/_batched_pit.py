"""K-fold PIT driver shared by the batched marginal backends (exaone, tabpfn, tabldm).

The backend supplies bank_fn:

    bank_fn(X_context: list of B (n_context, p_x) arrays,
            y_context: list of B (n_context,) arrays,
            X_query:   list of B (n_query, p_x) arrays,
            probs:     (Q,) array) -> (B, n_query, Q) array in raw y units

probs is the caller's grid; backends with a fixed native grid interpolate
inside bank_fn.
"""

from __future__ import annotations

import numpy as np

__all__ = ["run_kfold_pit_batched"]


def run_kfold_pit_batched(
    bank_fn, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray,
    k_folds: int = 10, probs_n: int = 99, eps: float = 1e-6, seed: int = 0,
) -> dict:
    """K-fold PIT for z_train and a full-context pass for z_test/log_pdf_test, batched across episodes.

    Args:
        bank_fn: the backend's quantile-bank function.
        X_train: (B, P, p_x); X_test: (B, N, p_x).
        Y_train: (B, P); Y_test: (B, N), already scaled by the caller.
        k_folds: clamped to [1, P].
        probs_n: quantile grid size.
        seed: episode b's folds are default_rng(seed + b).permutation(P) % K
            (as eval/metrics/joint_nll.kfold_loo_pit); fold sizes are equal
            across episodes, so folds can be batched.

    Returns:
        dict with z_train (B, P), z_test (B, N), log_pdf_test (B, N) in the
        scaled y units (callers apply the Jacobian).
    """
    from eval.metrics.joint_nll import compute_pit

    B, P, _p_x = X_train.shape
    N = X_test.shape[1]
    K = min(int(k_folds), P)
    probs = np.linspace(1.0 / (probs_n + 1), probs_n / (probs_n + 1), probs_n)

    fold_ids = [np.random.default_rng(seed + b).permutation(P) % K for b in range(B)]

    z_train = np.empty((B, P), dtype=np.float32)
    for k in range(K):
        held = [fold_ids[b] == k for b in range(B)]
        qry_idx = [np.where(held[b])[0] for b in range(B)]
        ctx_idx = [np.where(~held[b])[0] for b in range(B)]
        if qry_idx[0].size == 0 or ctx_idx[0].size == 0:
            continue  # size is shared across b (see docstring); checking b=0 suffices
        X_ctx = [X_train[b][ctx_idx[b]] for b in range(B)]
        y_ctx = [Y_train[b][ctx_idx[b]] for b in range(B)]
        X_qry = [X_train[b][qry_idx[b]] for b in range(B)]
        bank = bank_fn(X_ctx, y_ctx, X_qry, probs)  # (B, F, Q)
        for b in range(B):
            z_held, _ = compute_pit(bank[b], probs, Y_train[b][qry_idx[b]], eps)
            z_train[b, qry_idx[b]] = z_held

    bank_test = bank_fn(
        [X_train[b] for b in range(B)], [Y_train[b] for b in range(B)],
        [X_test[b] for b in range(B)], probs,
    )  # (B, N, Q)
    z_test = np.empty((B, N), dtype=np.float32)
    log_pdf_test = np.empty((B, N), dtype=np.float32)
    for b in range(B):
        z_b, log_pdf_b = compute_pit(bank_test[b], probs, Y_test[b], eps)
        z_test[b] = z_b
        log_pdf_test[b] = log_pdf_b

    return {"z_train": z_train, "z_test": z_test, "log_pdf_test": log_pdf_test}
