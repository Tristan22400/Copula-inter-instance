"""PriorLabs TabPFN-3.5 (Fast checkpoint) as a marginal backend (data.z_train_source=tabpfn35_fast).

The model is TabPFNRegressor's differentiable-input path
(fit_with_differentiable_input, then forward on a tensor): one estimator, the
differentiable feature preprocessor and no target transform. That is the one
path on which TabPFN keeps gradients to its weights, so the PIT and Phase A
share it, and a fine-tuned TabPFN is always scored through the function it was
trained through. It differs from TabPFNRegressor.fit/predict, whose numpy
feature preprocessing is not differentiable. TabPFNMarginal.batched_logits
evaluates that path for a batch of episodes in one network forward instead of
one fit and forward per episode.

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
import math
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
    "bar_pit",
    "tabpfn_quantiles",
    "tabpfn_pit",
    "tabpfn_run_pit_batched",
]

TABPFN_NATIVE_QUANTILES = 999
# Keeps the half-normal tail inverse finite (and its gradient defined) at the bin edges.
_TAIL_EPS = 1e-6


class TabPFNMarginal:
    """A TabPFNRegressor on the differentiable-input path.

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

    def batched_logits(
        self, X_context: torch.Tensor, y_context: torch.Tensor, X_query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Logits over znorm_space_bardist_ for B episodes in one forward per column pattern.

        Args:
            X_context: (B, n, C); y_context: (B, n); X_query: (B, m, C), float32 on the model's device.

        Returns:
            logits (B, m, n_bars) in float32, and the target mean and std (B,)
            that map the bar distribution's z space back to y's units.

        The same function as fit(X_context[b], y_context[b]) then logits(X_query[b])
        per episode, without the per-episode fit: the regressor's CPU
        preprocessing (constant-column removal, column z-norm with the unbiased
        std) is applied per member along the row axis, its GPU pipeline
        (fingerprint feature, column shuffle, soft outlier clip), which accepts a
        batch axis, runs once, and so does the network. Members that keep
        different numbers of columns run in separate forwards. tests/test_marginal_backends.py
        checks it against the per-episode path. Differentiable in the weights.
        """
        from copula_inter.marginal.batched_pit import group_by_kept_columns, take_columns

        B, m = X_query.shape[:2]
        y = y_context.float()
        y_mean = y.mean(dim=1)
        y_std = y.std(dim=1, correction=0).clamp(min=1e-20)
        if bool((y_std <= 1e-12).any()):
            raise ValueError("TabPFN refuses a constant target (std ~ 0) on the differentiable-input path.")
        y_z = (y - y_mean[:, None]) / y_std[:, None]
        groups = group_by_kept_columns(X_context.detach())
        if groups[0][1].shape[1] == 0:
            raise ValueError("All features are constant over a context; TabPFN cannot predict from it.")
        parts = [
            (
                members,
                self._forward_kept(
                    take_columns(X_context[members], cols), y_z[members], take_columns(X_query[members], cols)
                ),
            )
            for members, cols in groups
        ]
        if len(parts) == 1:
            logits = parts[0][1]
        else:
            logits = parts[0][1].new_empty((B, m, parts[0][1].shape[-1]))
            for members, part in parts:
                logits = logits.index_copy(0, members, part)
        return logits, y_mean, y_std

    def _forward_kept(self, X_context: torch.Tensor, y_z: torch.Tensor, X_query: torch.Tensor) -> torch.Tensor:
        """(B, m, n_bars) logits for members whose context columns all vary; y_z is the z-normalized target."""
        from tabpfn.base import resolved_softmax_temperature
        from tabpfn.inference import _model_expectes_task_type_arg
        from tabpfn.preprocessing.datamodel import FeatureModality
        from tabpfn.utils import get_autocast_context

        reg = self.regressor
        n = X_context.shape[1]
        # The GPU pipeline and its feature schema for this column count: one
        # library fit on the first member builds them (its preprocessing has no
        # learned state and is refit on every call below).
        self.fit(X_context[0], y_z[0])
        member = reg.executor_.ensemble_members[0]
        # DifferentiableZNormStep (RemoveConstantFeaturesStep is a no-op here).
        mean = X_context.mean(dim=1, keepdim=True)
        std = X_context.std(dim=1, keepdim=True)
        x_full = torch.cat([(X_context - mean) / std, (X_query - mean) / std], dim=1).transpose(0, 1)
        out = member.gpu_preprocessor(x_full, feature_schema=member.feature_schema, num_train_rows=n)
        model: Any = self.module
        kwargs: dict[str, object] = {"task_type": "regression"} if _model_expectes_task_type_arg(model) else {}
        with get_autocast_context(x_full.device, enabled=reg.use_autocast_):  # off: float32 (make_tabpfn_regressor)
            logits = model(
                out.x,
                y_z.transpose(0, 1),
                only_return_standard_out=True,
                categorical_inds=[out.feature_schema.indices_for(FeatureModality.CATEGORICAL)],
                performance_options=model.get_default_performance_options(),
                **kwargs,
            )  # (m, B, n_bars)
        logits = logits.float()
        temperature = resolved_softmax_temperature(reg)
        if temperature != 1:
            logits = logits / temperature
        return logits.transpose(0, 1)

    def quantiles(
        self, X_context: torch.Tensor, y_context: torch.Tensor, X_query: torch.Tensor, probs: torch.Tensor
    ) -> torch.Tensor:
        """(B, m, Q) quantiles at probs in y_context's units, for (B, n, C), (B, n), (B, m, C) episodes."""
        logits, mean, std = self.batched_logits(X_context, y_context, X_query)
        bardist = self.regressor.znorm_space_bardist_
        q_z = bar_icdf(bardist, logits, probs)
        return q_z * std[:, None, None] + mean[:, None, None]

    def pit(
        self, X_context: torch.Tensor, y_context: torch.Tensor, X_query: torch.Tensor, y_query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(u, log_pdf), each (B, m): CDF and log-density of y_query (B, m) under the full bar distribution, in y_context's units.

        FullSupportBarDistribution.cdf and its negative log-density (forward)
        include the half-normal tails, so the density is proper on all of R.
        The z-normalization is undone by u(y) = F_z(y_z) and
        log f(y) = log f_z(y_z) - log s, with y_z = (y - m) / s.
        """
        logits, mean, std = self.batched_logits(X_context, y_context, X_query)
        y_z = (y_query.to(logits.device, torch.float32) - mean[:, None]) / std[:, None]
        u, log_pdf_z = bar_pit(self.regressor.znorm_space_bardist_, logits, y_z)
        return u, log_pdf_z - std.log()[:, None]


def make_tabpfn_regressor(device: str | None = None) -> TabPFNMarginal:
    """Build the TabPFN-3.5-Fast marginal in float32 with its weights loaded (a two-feature warm-up fit loads them)."""
    _require_tabpfn_token()
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    reg = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5_FAST,
        device=device or "cpu",
        n_estimators=1,
        differentiable_input=True,
        random_state=0,
        # float32, as the frozen TabICL marginal: autocast's fp16/bf16 rounding
        # (|logit error| ~1e-2) would make an episode's PIT depend on which
        # episodes share its batched forward.
        inference_precision=torch.float32,
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


def bar_pit(bardist: Any, logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(F(y), log f(y)) of FullSupportBarDistribution, on logits' device: (..., n_bars), (...) -> (...), (...).

    The same values as bardist.cdf(logits, y[..., None]) and -bardist(logits, y)
    (tests/test_marginal_backends.py checks both, tails included), from one
    log-softmax and one cumsum. The library keeps its borders on the CPU, so
    calling it moved every (n_query, 5000) logit row to the host and evaluated
    the distribution there; it also recomputes the softmax for each method.
    """
    logits = logits.float()
    borders = bardist.borders.to(logits)
    widths = borders[1:] - borders[:-1]
    n_bars = widths.numel()
    y = y.to(logits)
    # map_to_bucket_idx, clamped as both library methods do.
    idx = torch.searchsorted(borders, y.contiguous()) - 1
    idx = torch.where(y == borders[0], 0, idx)
    idx = torch.where(y == borders[-1], n_bars - 1, idx).clamp(0, n_bars - 1)
    log_p = torch.log_softmax(logits, dim=-1)
    p = log_p.exp()
    p_idx = p.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    before = (torch.cumsum(p, dim=-1) - p).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    share = ((y - borders[idx]) / widths[idx]).clamp(0.0, 1.0)
    u = before + p_idx * share
    u = torch.where(y <= borders[0], 0.0, torch.where(y >= borders[-1], 1.0, u))
    # Half-normal tails beyond borders[1] and borders[-2]: HalfNormal(s).cdf(x) = erf(x / (s sqrt 2)).
    s_left = _halfnormal_scale(bardist, bardist.bucket_widths[0]).to(logits)
    s_right = _halfnormal_scale(bardist, bardist.bucket_widths[-1]).to(logits)
    x_left = (borders[1] - y).clamp_min(0.0)
    x_right = (y - borders[-2]).clamp_min(0.0)
    left = p[..., 0] * (1.0 - torch.erf(x_left / (s_left * math.sqrt(2.0))))
    right = 1.0 - p[..., -1] * (1.0 - torch.erf(x_right / (s_right * math.sqrt(2.0))))
    u = torch.where(y < borders[1], left, torch.where(y >= borders[-2], right, u)).clip(0.0, 1.0)

    log_pdf = log_p.gather(-1, idx.unsqueeze(-1)).squeeze(-1) - widths[idx].log()
    # In an outer bin the density is its mass times the half-normal density:
    # log HalfNormal(s).pdf(x) = 0.5 log(2 / pi) - log s - x^2 / (2 s^2).
    half_log_2_over_pi = 0.5 * math.log(2.0 / math.pi)
    x_l = (borders[1] - y).clamp(min=1e-8)
    x_r = (y - borders[-2]).clamp(min=1e-8)
    tail_left = half_log_2_over_pi - s_left.log() - x_l.square() / (2 * s_left**2) + widths[0].log()
    tail_right = half_log_2_over_pi - s_right.log() - x_r.square() / (2 * s_right**2) + widths[-1].log()
    log_pdf = log_pdf + torch.where(idx == 0, tail_left, 0.0) + torch.where(idx == n_bars - 1, tail_right, 0.0)
    return u, log_pdf


def _halfnormal_scale(bardist: Any, width: torch.Tensor) -> torch.Tensor:
    """Scale of the half-normal tail the library attaches to an outer bin of this width."""
    return torch.as_tensor(bardist.halfnormal_with_p_weight_before(width).scale)


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
    """(B, n_query, Q) quantiles in y_context's units, episodes batched; differentiable when grad is enabled.

    Episodes must share their context and query sizes. probs=None returns the
    999 levels linspace(1/1000, 999/1000, 999).
    """
    if probs is None:
        probs = np.linspace(
            1.0 / (TABPFN_NATIVE_QUANTILES + 1), 1.0 - 1.0 / (TABPFN_NATIVE_QUANTILES + 1), TABPFN_NATIVE_QUANTILES
        )
    device = marginal.device
    probs_t = torch.as_tensor(probs, dtype=torch.float32, device=device)
    xc, yc, xq = _stack(device, X_context, y_context, X_query)
    return marginal.quantiles(xc, yc, xq, probs_t)


def tabpfn_pit(
    marginal: TabPFNMarginal,
    X_context: Sequence[np.ndarray],
    y_context: Sequence[np.ndarray],
    X_query: Sequence[np.ndarray],
    y_query: Sequence[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """(u, log_pdf), each (B, n_query), from the full bar distribution; episodes batched (same sizes)."""
    return marginal.pit(*_stack(marginal.device, X_context, y_context, X_query, y_query))


def _stack(device: torch.device, *arrays: Sequence[np.ndarray]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.as_tensor(np.stack(a), dtype=torch.float32, device=device) for a in arrays)


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
    from copula_inter.marginal.batched_pit import TABPFN_MAX_CELLS, backend_pit_fn, run_kfold_pit_batched

    return run_kfold_pit_batched(
        backend_pit_fn("tabpfn35_fast", regressor),
        X_train,
        Y_train,
        X_test,
        Y_test,
        k_folds=k_folds,
        eps=eps,
        seed=seed,
        max_cells=TABPFN_MAX_CELLS,
    )
