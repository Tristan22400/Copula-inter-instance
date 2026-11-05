"""K-fold PIT driver for the batched marginal backends (tabpfn35_fast, kumo_small).

The backend supplies pit_fn:

    pit_fn(X_context: list of M (n_context, p_x) arrays,
           y_context: list of M (n_context,) arrays,
           X_query:   list of M (n_query, p_x) arrays,
           y_query:   list of M (n_query,) arrays) -> (u, log_pdf), each (M, n_query)

u = F(y_query) and log_pdf = log f(y_query) under the backend's full predictive
distribution, in y_context's units (finetune_adapters.backend_pit). No quantile
grid is involved, so z keeps its tails and the density is proper.

Both backends run all members of a call in one network forward. The driver
therefore stacks every (fold, episode) pair that shares the context and query
sizes into one call: at most two fold sizes exist (P = q K + r gives r folds of
q + 1 and K - r of q), so a K-fold PIT is <= 3 calls (2 fold groups + the test
pass) instead of K + 1, each split only to bound memory (max_cells).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from scipy.special import ndtri

__all__ = ["backend_pit_fn", "group_by_varying_columns", "run_kfold_pit_batched"]

PitFn = Callable[[list, list, list, list], tuple[np.ndarray, np.ndarray]]

# Members per call are capped so that members * rows * columns stays below this
# (~2M cells: a few hundred MB of activations for either backend).
DEFAULT_MAX_CELLS = 1 << 21


def backend_pit_fn(name: str, regressor: Any) -> PitFn:
    """pit_fn for a generic backend: finetune_adapters.backend_pit without gradients, as float64 numpy arrays."""
    from copula_inter.marginal.finetune_adapters import backend_pit

    def pit_fn(X_ctx: list, y_ctx: list, X_qry: list, y_qry: list) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            u, log_pdf = backend_pit(name, regressor, X_ctx, y_ctx, X_qry, y_qry)
        return u.double().cpu().numpy(), log_pdf.double().cpu().numpy()

    return pit_fn


def group_by_varying_columns(X_context: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Group a batch of contexts (B, n, C) by which feature columns vary over their rows.

    Both libraries drop the columns that are constant over the context (a NaN
    column counts as constant) before the network, so members with different
    kept columns cannot share a forward. Returns [(member indices, kept column
    indices)], one entry per distinct pattern; with a single context row every
    column is kept (sdm.processing.DropConstantColumns).
    """
    B, n, C = X_context.shape
    if n <= 1:
        keep = torch.ones(B, C, dtype=torch.bool, device=X_context.device)
    else:
        keep = (X_context != X_context[:, :1]).any(dim=1) & ~X_context.isnan().all(dim=1)
    patterns, inverse = torch.unique(keep, dim=0, return_inverse=True)
    return [
        (torch.nonzero(inverse == g).squeeze(1), torch.nonzero(patterns[g]).squeeze(1)) for g in range(len(patterns))
    ]


def _probit(u: np.ndarray, eps: float) -> np.ndarray:
    """Phi^{-1}(u) with u clipped to [eps, 1 - eps], as TabICL's PIT (pit._probit)."""
    return ndtri(np.clip(u, eps, 1.0 - eps))


def _chunked_pit(
    pit_fn: PitFn, X_ctx: list, y_ctx: list, X_qry: list, y_qry: list, max_cells: int
) -> tuple[np.ndarray, np.ndarray]:
    """pit_fn over M same-shape members, in chunks of at most max_cells members * rows * columns."""
    M = len(X_ctx)
    rows = X_ctx[0].shape[0] + X_qry[0].shape[0]
    cols = X_ctx[0].shape[1] + 1
    step = max(1, max_cells // (rows * cols))
    parts = [
        pit_fn(X_ctx[i : i + step], y_ctx[i : i + step], X_qry[i : i + step], y_qry[i : i + step])
        for i in range(0, M, step)
    ]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def run_kfold_pit_batched(
    pit_fn: PitFn,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    k_folds: int = 10,
    eps: float = 1e-6,
    seed: int = 0,
    max_cells: int = DEFAULT_MAX_CELLS,
) -> dict:
    """K-fold PIT for z_train and a full-context pass for z_test/log_pdf_test, batched across episodes and folds.

    Args:
        pit_fn: the backend's PIT function (module docstring).
        X_train: (B, P, p_x); X_test: (B, N, p_x).
        Y_train: (B, P); Y_test: (B, N), already scaled by the caller.
        k_folds: clamped to [1, P].
        eps: u is clipped to [eps, 1 - eps] before the probit (|z| <= 4.75 at 1e-6).
        seed: episode b's folds are default_rng(seed + b).permutation(P) % K
            (as eval/metrics/joint_nll.kfold_loo_pit).
        max_cells: memory bound per pit_fn call (members * rows * columns).

    Each held-out point is predicted from the other folds of its own episode
    only; the members of one pit_fn call are independent, so how the
    (fold, episode) pairs are grouped into calls does not change the result.

    Returns:
        dict with z_train (B, P), z_test (B, N), log_pdf_test (B, N) in the
        scaled y units (callers apply the Jacobian).
    """
    B, P, _p_x = X_train.shape
    K = min(int(k_folds), P)

    fold_ids = [np.random.default_rng(seed + b).permutation(P) % K for b in range(B)]
    # (fold k, episode b) members grouped by held-out size; the context size is P minus it.
    groups: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for k in range(K):
        for b in range(B):
            held = fold_ids[b] == k
            qry_idx, ctx_idx = np.where(held)[0], np.where(~held)[0]
            if qry_idx.size and ctx_idx.size:
                groups.setdefault(qry_idx.size, []).append((b, ctx_idx, qry_idx))

    z_train = np.empty((B, P), dtype=np.float32)
    for members in groups.values():
        u_held, _ = _chunked_pit(
            pit_fn,
            [X_train[b][ctx] for b, ctx, _ in members],
            [Y_train[b][ctx] for b, ctx, _ in members],
            [X_train[b][qry] for b, _, qry in members],
            [Y_train[b][qry] for b, _, qry in members],
            max_cells,
        )  # (members, fold size)
        for (b, _, qry), u in zip(members, u_held, strict=True):
            z_train[b, qry] = _probit(u, eps)

    u_test, log_pdf_test = _chunked_pit(pit_fn, list(X_train), list(Y_train), list(X_test), list(Y_test), max_cells)
    return {
        "z_train": z_train,
        "z_test": _probit(u_test, eps).astype(np.float32),
        "log_pdf_test": log_pdf_test.astype(np.float32),
    }
