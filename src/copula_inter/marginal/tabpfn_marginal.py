"""PriorLabs TabPFN-3.5 (Fast checkpoint) as a marginal backend (data.z_train_source=tabpfn35_fast).

The model runs through TabPFNRegressor's differentiable-input path
(fit_with_differentiable_input, then forward on a tensor): one estimator, the
identity feature preprocessor and no target transform. That is the one path on
which TabPFN keeps gradients to its weights, so the PIT and Phase A share it,
and a fine-tuned TabPFN is always scored through the function it was trained
through. It differs from TabPFNRegressor.fit/predict, whose numpy feature
preprocessing is not differentiable.

TabPFN predicts a bar distribution (5000 bins with half-normal tails) over the
z-normalized target. The PIT (tabpfn_pit) reads that distribution directly:
its CDF and log-density at y, tails included. Phase A's quantiles come from
bar_icdf, a vectorized, differentiable copy of FullSupportBarDistribution.icdf
(tests/test_marginal_backends.py checks it against the original).

Needs the `tabpfn` uv extra and TABPFN_TOKEN, with the TabPFN-3.5 licence
accepted at https://ux.priorlabs.ai (never commit the key).
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from tabpfn import TabPFNRegressor

__all__ = [
    "TabPFNMarginal",
    "make_tabpfn_regressor",
    "bar_icdf",
    "tabpfn_quantiles",
    "tabpfn_pit",
    "tabpfn_run_pit_batched",
]

TABPFN_NATIVE_QUANTILES = 999
# Keeps the half-normal tail inverse finite (and its gradient defined) at the bin edges.
_TAIL_EPS = 1e-6


class TabPFNMarginal:
    """A TabPFNRegressor on the differentiable-input path, refit per context.

    TabPFNRegressor caches its ensemble configuration from the first fit, which
    fixes the feature count; fit() rebuilds it when the count changes. The
    regressor owns a private copy of the weights (tabpfn's built-model cache
    would otherwise share them across regressors in one process).
    """

    def __init__(self, regressor: TabPFNRegressor) -> None:
        self.regressor = regressor

    @property
    def module(self) -> torch.nn.Module:
        """The trainable network (TabPFNRegressor.model_)."""
        return self.regressor.model_

    @property
    def device(self) -> torch.device:
        return next(self.module.parameters()).device

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        reg = self.regressor
        if getattr(reg, "n_features_in_", X.shape[1]) != X.shape[1]:
            from tabpfn.utils import infer_random_state

            _, rng = infer_random_state(reg.random_state)
            reg.ensemble_configs_, _ = reg._initialize_for_differentiable_input(X=X, rng=rng)
        reg.fit_with_differentiable_input(X, y)

    def logits(self, X_query: torch.Tensor) -> torch.Tensor:
        """(n_query, n_bars) logits over regressor.znorm_space_bardist_."""
        _, outputs, _ = self.regressor.forward(X_query, use_inference_mode=True)
        (out,) = outputs  # one estimator, no target transform: native borders
        return out.reshape(X_query.shape[0], -1)

    def quantiles(
        self, X_context: torch.Tensor, y_context: torch.Tensor, X_query: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        """(n_query, Q) quantiles at probs in y_context's units."""
        self.fit(X_context, y_context)
        reg = self.regressor
        q_z = bar_icdf(reg.znorm_space_bardist_, self.logits(X_query), probs)
        return q_z * reg.y_train_std_ + reg.y_train_mean_

    def pit(
        self, X_context: torch.Tensor, y_context: torch.Tensor, X_query: torch.Tensor, y_query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(u, log_pdf), each (n_query,): CDF and log-density of y_query under the full bar distribution, in y_context's units.

        FullSupportBarDistribution.cdf and its negative log-density (forward)
        include the half-normal tails, so the density is proper on all of R.
        The z-normalization is undone by u(y) = F_z(y_z) and
        log f(y) = log f_z(y_z) - log s, with y_z = (y - m) / s.
        """
        self.fit(X_context, y_context)
        reg = self.regressor
        bardist = reg.znorm_space_bardist_
        logits = self.logits(X_query).float().to(bardist.borders.device)
        mean = torch.as_tensor(reg.y_train_mean_, dtype=torch.float32, device=logits.device)
        std = torch.as_tensor(reg.y_train_std_, dtype=torch.float32, device=logits.device)
        y_z = (y_query.to(logits.device, torch.float32) - mean) / std
        u = bardist.cdf(logits, y_z.unsqueeze(-1)).squeeze(-1)
        log_pdf = -bardist(logits, y_z) - std.log()
        return u, log_pdf


def make_tabpfn_regressor(device: str | None = None) -> TabPFNMarginal:
    """Build the TabPFN-3.5-Fast marginal with its weights loaded (a two-feature warm-up fit loads them)."""
    _require_tabpfn_token()
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    reg = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5_FAST,
        device=device or "cpu",
        n_estimators=1,
        differentiable_input=True,
        random_state=0,
    )
    warm = torch.linspace(-1.0, 1.0, 16).reshape(8, 2)
    reg.fit_with_differentiable_input(warm, warm[:, 0])
    reg.models_ = [copy.deepcopy(m) for m in reg.models_]
    reg.executor_ = None  # still references the shared weights; rebuilt by the next fit
    return TabPFNMarginal(reg)


def _require_tabpfn_token() -> None:
    if not os.environ.get("TABPFN_TOKEN"):
        raise RuntimeError(
            "TabPFN-3.5 requires a one-time license acceptance: open "
            "https://ux.priorlabs.ai, log in, accept the TabPFN-3.5 license, copy your "
            "API key from the account page, then `export TABPFN_TOKEN=...` "
            "(job scripts read ~/.config/tabpfn_token) before running this backend."
        )


def bar_icdf(bardist: Any, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """FullSupportBarDistribution.icdf at every level of probs at once: (..., n_bars), (Q,) -> (..., Q).

    Differentiable in logits; probs must lie strictly inside (0, 1).
    """
    p = logits.float().softmax(-1)
    cum = p.cumsum(-1)
    levels = probs.to(cum).expand(*cum.shape[:-1], -1).contiguous()
    idx = torch.searchsorted(cum.contiguous(), levels).clamp(0, bardist.num_bars - 1)
    before = torch.cat((torch.zeros_like(cum[..., :1]), cum[..., :-1]), -1)
    cond = ((levels - before.gather(-1, idx)) / p.gather(-1, idx)).clamp(0.0, 1.0)
    borders = bardist.borders.to(cum)
    widths = bardist.bucket_widths.to(cum)
    values = borders[idx] + widths[idx] * cond
    # Half-normal tails in the outer bins: HalfNormal(s).icdf(u) = s * ndtri((1 + u) / 2).
    # torch.where back-propagates through both branches, and ndtri(1) = inf, so
    # an interior entry with cond in {0, 1} would send 0 * inf = NaN into the
    # weights. The tail branch therefore only ever sees cond strictly inside
    # (0, 1): a neutral 0.5 where it is not selected, a clamp where it is.
    left_tail, right_tail = idx == 0, idx == bardist.num_bars - 1
    tail_cond = cond.clamp(_TAIL_EPS, 1.0 - _TAIL_EPS)
    left_cond = torch.where(left_tail, tail_cond, torch.full_like(cond, 0.5))
    right_cond = torch.where(right_tail, tail_cond, torch.full_like(cond, 0.5))
    left_scale = bardist.halfnormal_with_p_weight_before(bardist.bucket_widths[0]).scale.to(cum)
    right_scale = bardist.halfnormal_with_p_weight_before(bardist.bucket_widths[-1]).scale.to(cum)
    left = borders[1] - left_scale * torch.special.ndtri((2.0 - left_cond) / 2.0)
    right = borders[-2] + right_scale * torch.special.ndtri((1.0 + right_cond) / 2.0)
    values = torch.where(left_tail, left, values)
    return torch.where(right_tail, right, values)


def tabpfn_quantiles(
    marginal: TabPFNMarginal,
    X_context: Sequence[np.ndarray],
    y_context: Sequence[np.ndarray],
    X_query: Sequence[np.ndarray],
    probs: np.ndarray | None = None,
) -> torch.Tensor:
    """(B, n_query, Q) quantiles in y_context's units, one fit per episode; differentiable when grad is enabled.

    probs=None returns the 999 levels linspace(1/1000, 999/1000, 999).
    """
    if probs is None:
        probs = np.linspace(
            1.0 / (TABPFN_NATIVE_QUANTILES + 1), 1.0 - 1.0 / (TABPFN_NATIVE_QUANTILES + 1), TABPFN_NATIVE_QUANTILES
        )
    device = marginal.device
    probs_t = torch.as_tensor(probs, dtype=torch.float32, device=device)

    def tensor(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    return torch.stack(
        [
            marginal.quantiles(tensor(xc), tensor(yc), tensor(xq), probs_t)
            for xc, yc, xq in zip(X_context, y_context, X_query, strict=True)
        ]
    )


def tabpfn_pit(
    marginal: TabPFNMarginal,
    X_context: Sequence[np.ndarray],
    y_context: Sequence[np.ndarray],
    X_query: Sequence[np.ndarray],
    y_query: Sequence[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """(u, log_pdf), each (B, n_query), from the full bar distribution; one fit per episode."""
    device = marginal.device

    def tensor(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    pairs = [
        marginal.pit(tensor(xc), tensor(yc), tensor(xq), tensor(yq))
        for xc, yc, xq, yq in zip(X_context, y_context, X_query, y_query, strict=True)
    ]
    return torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])


def tabpfn_run_pit_batched(
    regressor: TabPFNMarginal,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    k_folds: int = 10,
    eps: float = 1e-6,
    seed: int = 0,
) -> dict:
    """run_pit_batched for TabPFN, via the shared K-fold driver and the full bar distribution."""
    from copula_inter.marginal.batched_pit import backend_pit_fn, run_kfold_pit_batched

    return run_kfold_pit_batched(
        backend_pit_fn("tabpfn35_fast", regressor),
        X_train,
        Y_train,
        X_test,
        Y_test,
        k_folds=k_folds,
        eps=eps,
        seed=seed,
    )
