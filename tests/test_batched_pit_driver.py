"""run_kfold_pit_batched: the fused (fold, episode) batching gives the per-fold K-fold PIT, in <= 3 calls.

A context-dependent stand-in for the backend (a Gaussian fitted to the context
targets, shifted by the query features) makes a wrong context or a misplaced
held-out row visible; no model weights are needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.special import ndtri
from scipy.stats import norm

from copula_inter.marginal.batched_pit import group_by_varying_columns, run_kfold_pit_batched


def _gaussian_pit(X_ctx: list, y_ctx: list, X_qry: list, y_qry: list) -> tuple[np.ndarray, np.ndarray]:
    loc = np.stack([y.mean() + x[:, 0] for y, x in zip(y_ctx, X_qry, strict=True)])
    scale = np.stack([np.full(len(x), y.std() + 0.1 * len(y)) for y, x in zip(y_ctx, X_qry, strict=True)])
    y = np.stack(y_qry)
    return norm.cdf(y, loc, scale), norm.logpdf(y, loc, scale)


def _per_fold_reference(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    K: int,
    seed: int,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    B, P, _ = X_train.shape
    z_train = np.empty((B, P))
    for b in range(B):
        fold = np.random.default_rng(seed + b).permutation(P) % K
        for k in range(K):
            held = fold == k
            u, _ = _gaussian_pit([X_train[b][~held]], [Y_train[b][~held]], [X_train[b][held]], [Y_train[b][held]])
            z_train[b, held] = ndtri(np.clip(u[0], eps, 1 - eps))
    u, lp = _gaussian_pit(list(X_train), list(Y_train), list(X_test), list(Y_test))
    return z_train, ndtri(np.clip(u, eps, 1 - eps)), lp


@pytest.mark.parametrize("P,K", [(23, 5), (20, 4), (6, 10)])
def test_fused_folds_equal_the_per_fold_pit(P: int, K: int) -> None:
    rng = np.random.default_rng(0)
    B, N = 4, 5
    X_train, X_test = rng.normal(size=(B, P, 2)), rng.normal(size=(B, N, 2))
    Y_train, Y_test = rng.normal(size=(B, P)) * 3 + 1, rng.normal(size=(B, N)) * 3 + 1

    calls: list[tuple[int, int, int]] = []

    def pit_fn(X_ctx: list, y_ctx: list, X_qry: list, y_qry: list) -> tuple[np.ndarray, np.ndarray]:
        assert len({x.shape for x in X_ctx}) == 1 and len({x.shape for x in X_qry}) == 1, "a call mixes sizes"
        calls.append((len(X_ctx), X_ctx[0].shape[0], X_qry[0].shape[0]))
        return _gaussian_pit(X_ctx, y_ctx, X_qry, y_qry)

    out = run_kfold_pit_batched(pit_fn, X_train, Y_train, X_test, Y_test, k_folds=K, seed=7)
    z_train, z_test, log_pdf = _per_fold_reference(X_train, Y_train, X_test, Y_test, min(K, P), seed=7)
    np.testing.assert_allclose(out["z_train"], z_train, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out["z_test"], z_test, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out["log_pdf_test"], log_pdf, rtol=1e-5, atol=1e-5)
    assert len(calls) <= 3
    assert sum(m for m, _, _ in calls[:-1]) == B * min(K, P)  # every (fold, episode) once


def test_max_cells_splits_calls_without_changing_the_result() -> None:
    rng = np.random.default_rng(1)
    X_train, X_test = rng.normal(size=(6, 21, 3)), rng.normal(size=(6, 4, 3))
    Y_train, Y_test = rng.normal(size=(6, 21)), rng.normal(size=(6, 4))
    sizes: list[int] = []

    def pit_fn(*args: list) -> tuple[np.ndarray, np.ndarray]:
        sizes.append(len(args[0]))
        return _gaussian_pit(*args)

    whole = run_kfold_pit_batched(_gaussian_pit, X_train, Y_train, X_test, Y_test, k_folds=5)
    split = run_kfold_pit_batched(pit_fn, X_train, Y_train, X_test, Y_test, k_folds=5, max_cells=300)
    assert max(sizes) == 3 and len(sizes) > 3  # 300 // (25 rows * 4 columns) members per call
    for key in whole:
        np.testing.assert_array_equal(split[key], whole[key])


def test_group_by_varying_columns() -> None:
    x = torch.randn(4, 6, 3)
    x[1, :, 0] = 2.0  # constant column
    x[3, :, 2] = float("nan")  # all-NaN column counts as constant
    groups = {tuple(m.tolist()): c.tolist() for m, c in group_by_varying_columns(x)}
    assert groups == {(0, 2): [0, 1, 2], (1,): [1, 2], (3,): [0, 1]}
    single_row = group_by_varying_columns(torch.zeros(2, 1, 3))
    assert [(m.tolist(), c.tolist()) for m, c in single_row] == [([0, 1], [0, 1, 2])]
