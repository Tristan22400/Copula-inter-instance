"""NVIDIA Kumo Tabular (small) as a marginal backend (data.z_train_source=kumo_small).

The model is sdm.models.KumoTabular, used through its public forward with
KUMO_RECIPE: Kumo's default recipe with its ensemble augmentations (random
feature sign flips, a Latin column shuffle and a round-robin choice of feature
transform) pinned to the canonical member, and without the estimator average.
The marginal is then a deterministic function of the episode, episodes batch as
recipe members (each preprocessed on its own context, all run in one model
call), and Phase A trains the same function the PIT reads: KumoTabular.forward
keeps gradients when the module is in train mode.

Kumo's regression head emits 999 quantiles q001..q999, i.e. levels i/1000; the
PIT (kumo_pit) turns them into a full distribution with tails.
Needs the `kumo` uv extra (structured-data-models); weights download from the
nvidia/Kumo-Tabular HF repo (OpenMDW licence).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    import sdm.processing
    from sdm.models import KumoTabular

__all__ = [
    "KUMO_NATIVE_PROBS",
    "kumo_recipe",
    "make_kumo_regressor",
    "kumo_quantiles",
    "kumo_pit",
    "kumo_run_pit_batched",
]

KUMO_SIZE = "small"
KUMO_NATIVE_PROBS = np.arange(1, 1000) / 1000.0


def kumo_recipe() -> sdm.processing.Recipe:
    """sdm.models.kumo.tabular.recipe.default_recipe for numerical features, augmentations pinned to the canonical member."""
    import sdm.processing as sp

    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                numerical=[
                    sp.Cast(torch.float64),
                    sp.DropConstantColumns(),
                    sp.Standardize(eps=1e-6),
                    sp.Clip(-100.0, 100.0),
                    # default_recipe: Choice(Identity, PowerTransform, ..., method="round_robin"),
                    # whose single-member pick is Identity; then FlipSign (random), dropped here.
                    sp.ClipSigma(threshold=4.0),
                ],
            ),
            # default_recipe: ShuffleColumns(method="latin") (random), dropped here.
            sp.SelectColumns(500, method="first"),
            sp.Cast(torch.float32),
        ],
        # default_recipe: Standardize then FlipSign (random), dropped here.
        target=[sp.StypeDispatch(numerical=[sp.Standardize()])],
        # No AverageEstimators: members are episodes, not estimators.
        output=sp.TaskDispatch(regression=[sp.SortQuantiles()]),
    )


def make_kumo_regressor(device: str | None = None) -> KumoTabular:
    """The pretrained Kumo Tabular regressor (an nn.Module, in eval mode)."""
    from sdm.models import KumoTabular

    return KumoTabular(task="regression", size=KUMO_SIZE, pretrained=True, device=device or "cpu")


def kumo_quantiles(
    model: KumoTabular,
    X_context: Sequence[np.ndarray],
    y_context: Sequence[np.ndarray],
    X_query: Sequence[np.ndarray],
    probs: np.ndarray | None = None,
) -> torch.Tensor:
    """(B, n_query, Q) quantiles in y_context's units; differentiable when grad is enabled.

    Episodes must share their context and query sizes. probs=None returns the
    native levels KUMO_NATIVE_PROBS; other levels are read off the native grid
    by linear interpolation in probability.
    """
    device = next(model.parameters()).device
    xc = torch.as_tensor(np.stack(X_context), dtype=torch.float32, device=device)
    yc = torch.as_tensor(np.stack(y_context), dtype=torch.float32, device=device)
    xq = torch.as_tensor(np.stack(X_query), dtype=torch.float32, device=device)
    # KumoTabular.forward keeps gradients only in train mode (no dropout, so the
    # output is the same in both modes).
    model.train(torch.is_grad_enabled())
    out = model(xc, yc.unsqueeze(-1), xq, recipe=kumo_recipe(), num_estimators=None, estimator_batch_size=None)
    q = out.numerical  # (B, n_query, 999)
    if probs is None:
        return q
    return _interp_levels(q, KUMO_NATIVE_PROBS, np.asarray(probs, dtype=np.float64))


def _interp_levels(q: torch.Tensor, levels: np.ndarray, probs: np.ndarray) -> torch.Tensor:
    """Linear interpolation of a quantile grid (..., len(levels)) at probs, clamped to the outer levels."""
    p = np.clip(probs, levels[0], levels[-1])
    hi = np.clip(np.searchsorted(levels, p), 1, len(levels) - 1)
    lo = hi - 1
    w = torch.as_tensor((p - levels[lo]) / (levels[hi] - levels[lo]), dtype=q.dtype, device=q.device)
    lo_t = torch.as_tensor(lo, device=q.device)
    hi_t = torch.as_tensor(hi, device=q.device)
    return q.index_select(-1, lo_t) * (1 - w) + q.index_select(-1, hi_t) * w


def kumo_pit(
    model: KumoTabular,
    X_context: Sequence[np.ndarray],
    y_context: Sequence[np.ndarray],
    X_query: Sequence[np.ndarray],
    y_query: Sequence[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """(u, log_pdf), each (B, n_query): Kumo's 999 native quantiles as a full distribution, evaluated at y_query.

    The distribution is TabICL's QuantileToDistribution on the same levels
    i/1000 (monotone spline inside [q001, q999], exponential tails outside), the
    construction TabICL's own PIT uses, so the density is proper with tails.
    """
    q = kumo_quantiles(model, X_context, y_context, X_query, None)  # (B, n_query, 999)
    B, n_query, Q = q.shape
    dist = _quantile_distribution(q.device)(q.reshape(B * n_query, Q).float())
    y = torch.as_tensor(np.stack(y_query), dtype=torch.float32, device=q.device).reshape(B * n_query)
    return dist.cdf(y).reshape(B, n_query), dist.log_prob(y).reshape(B, n_query)


def _quantile_distribution(device: torch.device) -> torch.nn.Module:
    from tabicl._model.quantile_dist import QuantileToDistribution

    return QuantileToDistribution(num_quantiles=len(KUMO_NATIVE_PROBS)).to(device)


def kumo_run_pit_batched(
    regressor: KumoTabular,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    k_folds: int = 10,
    eps: float = 1e-6,
    seed: int = 0,
) -> dict:
    """run_pit_batched for Kumo, via the shared K-fold driver (one batched forward per fold) and its full distribution."""
    from copula_inter.marginal.batched_pit import backend_pit_fn, run_kfold_pit_batched

    return run_kfold_pit_batched(
        backend_pit_fn("kumo_small", regressor), X_train, Y_train, X_test, Y_test, k_folds=k_folds, eps=eps, seed=seed
    )
