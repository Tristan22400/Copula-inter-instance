"""K-fold PIT driver for the batched marginal backends (tabpfn35_fast, kumo_small).

The backend supplies pit_fn:

    pit_fn(X_context: list of B (n_context, p_x) arrays,
           y_context: list of B (n_context,) arrays,
           X_query:   list of B (n_query, p_x) arrays,
           y_query:   list of B (n_query,) arrays) -> (u, log_pdf), each (B, n_query)

u = F(y_query) and log_pdf = log f(y_query) under the backend's full predictive
distribution, in y_context's units (finetune_adapters.backend_pit). No quantile
grid is involved, so z keeps its tails and the density is proper.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.special import ndtri

__all__ = ["backend_pit_fn", "run_kfold_pit_batched"]

PitFn = Callable[[list, list, list, list], tuple[np.ndarray, np.ndarray]]


def backend_pit_fn(name: str, regressor: Any) -> PitFn:
    """pit_fn for a generic backend: finetune_adapters.backend_pit without gradients, as float64 numpy arrays."""
    import torch

    from copula_inter.marginal.finetune_adapters import backend_pit

    def pit_fn(X_ctx: list, y_ctx: list, X_qry: list, y_qry: list) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            u, log_pdf = backend_pit(name, regressor, X_ctx, y_ctx, X_qry, y_qry)
        return u.double().cpu().numpy(), log_pdf.double().cpu().numpy()

    return pit_fn


def _probit(u: np.ndarray, eps: float) -> np.ndarray:
    """Phi^{-1}(u) with u clipped to [eps, 1 - eps], as TabICL's PIT (pit._probit)."""
    return ndtri(np.clip(u, eps, 1.0 - eps))


def run_kfold_pit_batched(
    pit_fn: PitFn,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    k_folds: int = 10,
    eps: float = 1e-6,
    seed: int = 0,
) -> dict:
    """K-fold PIT for z_train and a full-context pass for z_test/log_pdf_test, batched across episodes.

    Args:
        pit_fn: the backend's PIT function (module docstring).
        X_train: (B, P, p_x); X_test: (B, N, p_x).
        Y_train: (B, P); Y_test: (B, N), already scaled by the caller.
        k_folds: clamped to [1, P].
        eps: u is clipped to [eps, 1 - eps] before the probit (|z| <= 4.75 at 1e-6).
        seed: episode b's folds are default_rng(seed + b).permutation(P) % K
            (as eval/metrics/joint_nll.kfold_loo_pit); fold sizes are equal
            across episodes, so folds can be batched.

    Returns:
        dict with z_train (B, P), z_test (B, N), log_pdf_test (B, N) in the
        scaled y units (callers apply the Jacobian).
    """
    B, P, _p_x = X_train.shape
    K = min(int(k_folds), P)

    fold_ids = [np.random.default_rng(seed + b).permutation(P) % K for b in range(B)]

    z_train = np.empty((B, P), dtype=np.float32)
    for k in range(K):
        held = [fold_ids[b] == k for b in range(B)]
        qry_idx = [np.where(held[b])[0] for b in range(B)]
        ctx_idx = [np.where(~held[b])[0] for b in range(B)]
        if qry_idx[0].size == 0 or ctx_idx[0].size == 0:
            continue  # size is shared across b (see docstring); checking b=0 suffices
        u_held, _ = pit_fn(
            [X_train[b][ctx_idx[b]] for b in range(B)],
            [Y_train[b][ctx_idx[b]] for b in range(B)],
            [X_train[b][qry_idx[b]] for b in range(B)],
            [Y_train[b][qry_idx[b]] for b in range(B)],
        )  # (B, F)
        for b in range(B):
            z_train[b, qry_idx[b]] = _probit(u_held[b], eps)

    u_test, log_pdf_test = pit_fn(
        [X_train[b] for b in range(B)],
        [Y_train[b] for b in range(B)],
        [X_test[b] for b in range(B)],
        [Y_test[b] for b in range(B)],
    )  # (B, N)
    return {
        "z_train": z_train,
        "z_test": _probit(u_test, eps).astype(np.float32),
        "log_pdf_test": log_pdf_test.astype(np.float32),
    }
