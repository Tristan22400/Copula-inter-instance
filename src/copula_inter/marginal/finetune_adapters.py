"""Per-architecture adapters for Phase-A marginal fine-tuning.

For each backbone: which parameters each tier trains, a gradient-carrying
quantile forward, the checkpoint format and how it is loaded back.

    tabicl         tiers 0-3 (pit.run_pit_batched_grad).
    tabpfn35_fast  tier 0 (tabpfn_marginal.tabpfn_quantiles, episodes batched).
    kumo_small     tier 0 (kumo_marginal.kumo_quantiles, episodes batched).

The generic backbones' Phase-A forward is backend_quantiles, the same function
their PIT reads (marginal/backends.py, the *_run_pit_batched drivers), so a
Phase-A checkpoint is scored exactly as it was trained. All-layer LoRA
(lora.apply_lora_all_layers) works for every backbone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Sequence, cast

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from tabicl._model.quantile_dist import QuantileToDistribution

from copula_inter.artifacts import atomic_torch_save
from copula_inter.backend_registry import MARGINAL_BACKENDS, get_backend

__all__ = [
    "TIER0_PATTERNS",
    "MarginalBackbone",
    "load_backbone",
    "resolve_tier",
]

# Tier-0 parameter patterns per backbone: the label path, output norms and the
# quantile decoder (tests/test_marginal_backbones.py checks each matches).
TIER0_PATTERNS: dict[str, tuple[str, ...]] = {
    "tabicl": (
        r"^icl_predictor\.y_encoder\.",
        r"^col_embedder\.y_encoder\.",
        r"^icl_predictor\.ln\.",
        r"^icl_predictor\.tf_icl\.blocks\.\d+\.norm[12]\.",
        r"^icl_predictor\.decoder\.",
    ),
    # TabPFNRegressor.model_ (TabPFNV3p5): target encoders of both stages,
    # the ICL-block norms, the output norm and the regression head.
    "tabpfn35_fast": (
        r"^col_y_encoder\.regression\.",
        r"^col_y_layernorm\.",
        r"^icl_y_encoder\.regression\.",
        r"^icl_y_layernorm\.",
        r"^icl_blocks\.\d+\.(layernorm|layernorm_mlp)\.",
        r"^output_norm\.",
        r"^heads\.mlp_regression\.",
        r"^heads\.output_projection\.",
    ),
    # KumoTabular (task="regression"): target encoders of both stages, the ICL
    # norms and the quantile head.
    "kumo_small": (
        r"^models\.regression\.row_embedding\.y_lin\.",
        r"^models\.regression\.icl_block\.y_lin\.",
        r"^models\.regression\.icl_block\.layers\.\d+\.(query_norm|key_value_norm)\.",
        r"^models\.regression\.icl_block\.norm\.",
        r"^models\.regression\.icl_block\.head\.",
    ),
}


def resolve_tier(backbone_name: str, tier: int) -> int:
    """Return tier if backbone_name supports it, else raise ValueError."""
    if backbone_name not in MARGINAL_BACKENDS:
        raise ValueError(f"Unknown marginal backbone {backbone_name!r}; expected one of {list(MARGINAL_BACKENDS)}.")
    # Highest tier (stage LoRA); all-layer LoRA is not limited by this.
    top = get_backend(backbone_name).max_tier
    if tier > top:
        raise ValueError(
            f"marginal.tier={tier} is not available for backbone {backbone_name!r} "
            f"(max {top}). Tiers >= 1 install LoRA adapters on attention modules "
            f"(src/copula_inter/marginal/lora.py), and {backbone_name}'s attention is not a swappable "
            "nn.Module this repo has an adapter for -- see "
            "src/copula_inter/marginal/finetune_adapters.py's module docstring for the specifics. "
            "Use marginal.tier=0, or add a Parameter-level LoRA to lora.py."
        )
    return int(tier)


@dataclass
class MarginalBackbone:
    """A fine-tunable marginal: module (the trained nn.Module) and handle (the library regressor, None for tabicl)."""

    name: str
    module: nn.Module
    handle: Any = None
    config: dict = field(default_factory=dict)

    @property
    def tier0_patterns(self) -> tuple[str, ...]:
        return TIER0_PATTERNS[self.name]

    def parameters(self, *args: Any, **kwargs: Any) -> Iterator[nn.Parameter]:
        return self.module.parameters(*args, **kwargs)

    def named_parameters(self, *args: Any, **kwargs: Any) -> Iterator[tuple[str, nn.Parameter]]:
        return self.module.named_parameters(*args, **kwargs)

    @property
    def native_quantile_count(self) -> int:
        """Number of quantile levels quantile_forward(probs=None) returns (999 for every backbone here)."""
        own = getattr(self.module, "quantile_dist", None)
        levels = getattr(own, "alpha_levels", None) if own is not None else None
        if levels is not None:
            return int(levels.numel() if torch.is_tensor(levels) else len(levels))
        count = get_backend(self.name).native_quantiles
        if count is None:
            raise RuntimeError(f"cannot determine native quantile count for {self.name!r}")
        return count

    @property
    def native_probs(self) -> np.ndarray:
        """Alpha levels of the native grid: linspace(1/(n+1), n/(n+1), n)."""
        n = self.native_quantile_count
        return np.linspace(1.0 / (n + 1), n / (n + 1), n)

    # -- gradient-carrying quantile forward -----------------------------------
    def quantile_forward(
        self,
        X_context: Sequence[np.ndarray],
        y_context: Sequence[np.ndarray],
        X_query: Sequence[np.ndarray],
        probs: "np.ndarray | None" = None,
    ) -> torch.Tensor:
        """(B, n_query, Q) quantiles in raw y units, with gradients.

        probs=None returns the native decoder grid; explicit probs are interpolated.
        """
        if self.name == "tabicl":
            raise NotImplementedError(
                "tabicl's Phase-A forward goes through pit.py::run_pit_batched_grad, which "
                "finetune_marginal.py calls directly -- it carries K-fold rotation and PIT "
                "semantics this generic per-call interface does not."
            )
        return backend_quantiles(self.name, self.handle, X_context, y_context, X_query, probs)

    def quantile_dist_module(self, probs: "np.ndarray | None" = None) -> QuantileToDistribution:
        """Quantile-grid-to-distribution module for the grid quantile_forward(probs) returns.

        The model's own quantile_dist for the native grid, else TabICL's class.
        """
        from tabicl._model.quantile_dist import QuantileToDistribution

        if probs is None:
            own = getattr(self.module, "quantile_dist", None)
            if own is not None:
                return cast("QuantileToDistribution", own)
            probs = self.native_probs
        return QuantileToDistribution(alpha_levels=list(probs)).to(next(self.module.parameters()).device)

    # -- checkpointing ---------------------------------------------------------
    def save(self, path: str, *, step: int, cfg: DictConfig | None = None, extra: Optional[dict] = None) -> None:
        """Write a Phase-A checkpoint: TabICL's {"config", "state_dict"} for tabicl (loadable by pit.load_tabicl), the same plus a "backbone" tag otherwise."""
        from copula_inter.marginal.lora import merged_base_state_dict_any

        payload = {
            "config": dict(self.config),
            "state_dict": merged_base_state_dict_any(self.module),
            "step": int(step),
            "backbone": self.name,
        }
        if cfg is not None:
            from omegaconf import OmegaConf

            payload["cfg"] = OmegaConf.to_container(cfg, resolve=True)
        if extra:
            payload.update(extra)
        atomic_torch_save(payload, path)


def backend_quantiles(
    name: str,
    regressor: Any,
    X_context: Sequence[np.ndarray],
    y_context: Sequence[np.ndarray],
    X_query: Sequence[np.ndarray],
    probs: "np.ndarray | None" = None,
) -> torch.Tensor:
    """(B, n_query, Q) quantiles of a generic backend in y_context's units; differentiable when grad is enabled.

    The one quantile function of tabpfn35_fast and kumo_small: their PIT
    (inference) and their Phase-A forward (training) both call it.
    probs=None returns the backend's native_quantiles levels.

    An episode whose context features are all constant carries no feature
    information, and both libraries refuse it (they drop constant columns and
    are left with none). It gets the feature-free predictive instead: the
    Gaussian with its context targets' mean and standard deviation, constant in
    the weights.
    """
    if name == "tabpfn35_fast":
        from copula_inter.marginal.tabpfn_marginal import tabpfn_quantiles

        model_quantiles: Callable[..., torch.Tensor] = tabpfn_quantiles
    elif name == "kumo_small":
        from copula_inter.marginal.kumo_marginal import kumo_quantiles

        model_quantiles = kumo_quantiles
    else:
        raise ValueError(f"No generic quantile function for backbone {name!r}.")
    levels = probs if probs is not None else _native_levels(name)
    informative = [b for b, xc in enumerate(X_context) if _has_varying_feature(xc)]
    if len(informative) == len(X_context):
        return model_quantiles(regressor, X_context, y_context, X_query, probs)
    device = next(_trainable_module(name, regressor).parameters()).device
    out = torch.stack(
        [_feature_free_quantiles(y_context[b], X_query[b].shape[0], levels, device) for b in range(len(X_context))]
    )
    if informative:
        q = model_quantiles(
            regressor,
            [X_context[b] for b in informative],
            [y_context[b] for b in informative],
            [X_query[b] for b in informative],
            probs,
        )
        index = torch.as_tensor(informative, device=device)
        out = out.to(q.dtype).index_copy(0, index, q)
    return out


def backend_pit(
    name: str,
    regressor: Any,
    X_context: Sequence[np.ndarray],
    y_context: Sequence[np.ndarray],
    X_query: Sequence[np.ndarray],
    y_query: Sequence[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """(u, log_pdf), each (B, n_query): the predictive CDF and log-density of y_query under the backend's full distribution.

    In y_context's units, evaluated directly at y_query -- no quantile grid:
    tabpfn35_fast reads its bar distribution (5000 bins, half-normal tails),
    kumo_small its 999 quantiles through TabICL's QuantileToDistribution (the
    spline-with-exponential-tails density TabICL's own PIT uses), so all three
    marginals score a proper density with tails.

    Episodes whose context features are all constant get the feature-free
    Gaussian predictive, as in backend_quantiles.
    """
    if name == "tabpfn35_fast":
        from copula_inter.marginal.tabpfn_marginal import tabpfn_pit

        model_pit: Callable[..., tuple[torch.Tensor, torch.Tensor]] = tabpfn_pit
    elif name == "kumo_small":
        from copula_inter.marginal.kumo_marginal import kumo_pit

        model_pit = kumo_pit
    else:
        raise ValueError(f"No generic PIT function for backbone {name!r}.")
    informative = [b for b, xc in enumerate(X_context) if _has_varying_feature(xc)]
    if len(informative) == len(X_context):
        return model_pit(regressor, X_context, y_context, X_query, y_query)
    device = next(_trainable_module(name, regressor).parameters()).device
    pairs = [_feature_free_pit(y_context[b], y_query[b], device) for b in range(len(X_context))]
    u = torch.stack([p[0] for p in pairs])
    log_pdf = torch.stack([p[1] for p in pairs])
    if informative:
        u_inf, lp_inf = model_pit(
            regressor,
            [X_context[b] for b in informative],
            [y_context[b] for b in informative],
            [X_query[b] for b in informative],
            [y_query[b] for b in informative],
        )
        index = torch.as_tensor(informative, device=device)
        u = u.to(u_inf.dtype).index_copy(0, index, u_inf.to(device))
        log_pdf = log_pdf.to(lp_inf.dtype).index_copy(0, index, lp_inf.to(device))
    return u, log_pdf


def _feature_free_pit(
    y_context: np.ndarray, y_query: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """(u, log_pdf) of y_query under N(mean, std) of the context targets."""
    y = np.asarray(y_context, dtype=np.float64)
    std = max(float(y.std(ddof=1)) if y.size > 1 else 0.0, 1e-8)
    normal = torch.distributions.Normal(float(y.mean()), std)
    yq = torch.as_tensor(np.asarray(y_query, dtype=np.float64))
    return normal.cdf(yq).float().to(device), normal.log_prob(yq).float().to(device)


def _has_varying_feature(X_context: np.ndarray) -> bool:
    """Whether a feature column varies over the context rows (TabPFN's and Kumo's constant-column test)."""
    X = np.asarray(X_context)
    varies = (X != X[:1]).any(axis=0) & ~np.isnan(X).all(axis=0)
    return bool(varies.any())


def _native_levels(name: str) -> np.ndarray:
    n = get_backend(name).native_quantiles
    assert n is not None
    return np.linspace(1.0 / (n + 1), n / (n + 1), n)


def _feature_free_quantiles(
    y_context: np.ndarray, n_query: int, levels: np.ndarray, device: torch.device
) -> torch.Tensor:
    """(n_query, Q) quantiles of N(mean, std) of the context targets, the same for every query row."""
    from scipy.stats import norm

    y = np.asarray(y_context, dtype=np.float64)
    std = max(float(y.std(ddof=1)) if y.size > 1 else 0.0, 1e-8)
    q = y.mean() + std * norm.ppf(np.asarray(levels, dtype=np.float64))
    return torch.as_tensor(q, dtype=torch.float32, device=device).expand(n_query, -1)


def load_backbone(name: str, *, ckpt: Optional[str] = None, device: str = "cuda") -> MarginalBackbone:
    """Build a fine-tunable backbone via copula_inter.marginal.backends.make_regressor, optionally loading a Phase-A checkpoint."""
    if name not in MARGINAL_BACKENDS:
        raise ValueError(f"Unknown marginal backbone {name!r}; expected one of {list(MARGINAL_BACKENDS)}.")

    if name == "tabicl":
        from copula_inter.marginal.pit import PRETRAINED_TABICL_CKPT, load_tabicl

        tabicl, config = load_tabicl(ckpt or PRETRAINED_TABICL_CKPT, device, return_config=True)
        return MarginalBackbone(name=name, module=tabicl, handle=None, config=config)

    from copula_inter.marginal.backends import make_regressor

    regressor = make_regressor(name, device=device, ckpt=ckpt)
    module = _trainable_module(name, regressor)
    return MarginalBackbone(name=name, module=module, handle=regressor, config={})


def _trainable_module(name: str, regressor: Any) -> nn.Module:
    """The nn.Module inside a regressor whose parameters Phase A trains."""
    if name == "tabpfn35_fast":
        return cast("nn.Module", regressor.module)  # TabPFNMarginal -> TabPFNRegressor.model_
    if name == "kumo_small":
        return cast("nn.Module", regressor)  # KumoTabular is itself the module
    raise ValueError(f"No trainable module known for backbone {name!r}.")


def assert_patterns_match(module: nn.Module, patterns: Sequence[str]) -> dict[str, int]:
    """{pattern: number of matching parameter tensors}; raise if any pattern matches none."""
    names = [n for n, _ in module.named_parameters()]
    counts, missing = {}, []
    for pat in patterns:
        n_hit = sum(1 for n in names if re.search(pat, n))
        counts[pat] = n_hit
        if n_hit == 0:
            missing.append(pat)
    if missing:
        raise ValueError(
            f"tier-0 patterns matched no parameters: {missing}. The architecture's "
            "parameter names have changed -- update TIER0_PATTERNS in "
            "src/copula_inter/marginal/finetune_adapters.py."
        )
    return counts


def kfold_quantiles_grad(
    backbone: "MarginalBackbone",
    x_train: torch.Tensor,
    y_train_scaled: torch.Tensor,
    x_test: torch.Tensor,
    y_test_scaled: torch.Tensor,
    *,
    k_folds: int,
    probs: "np.ndarray | None" = None,
    fold_subset: Optional[Sequence[int]] = None,
) -> dict:
    """run_pit_batched_grad for non-TabICL backbones: returns q_test, q_train and train_query_idx.

    Uses pit.py's fold geometry (contiguous ceil(P/K) blocks in ascending order),
    which episode_fold_targets assumes.
    """
    import math

    B, P, _ = x_train.shape
    K = max(2, min(int(k_folds), P))
    fold_size = math.ceil(P / K)

    xtr = x_train.detach().cpu().numpy()
    ytr = y_train_scaled.detach().cpu().numpy()
    xte = x_test.detach().cpu().numpy()

    q_test = backbone.quantile_forward(
        [xtr[b] for b in range(B)],
        [ytr[b] for b in range(B)],
        [xte[b] for b in range(B)],
        probs,
    )

    wanted = range(K) if fold_subset is None else sorted({int(k) for k in fold_subset})
    q_train_parts, idx_parts = [], []
    for k in wanted:
        start, end = k * fold_size, min(k * fold_size + fold_size, P)
        if start >= end:
            continue  # empty tail fold when P is not a multiple of fold_size
        qry = np.arange(start, end)
        ctx = np.concatenate([np.arange(0, start), np.arange(end, P)])
        if ctx.size == 0:
            continue
        q_fold = backbone.quantile_forward(
            [xtr[b][ctx] for b in range(B)],
            [ytr[b][ctx] for b in range(B)],
            [xtr[b][qry] for b in range(B)],
            probs,
        )
        q_train_parts.append(q_fold)
        idx_parts.append(torch.as_tensor(qry, dtype=torch.long, device=q_test.device))

    q_train = torch.cat(q_train_parts, dim=1) if q_train_parts else q_test.new_zeros((B, 0, q_test.shape[-1]))
    train_query_idx = torch.cat(idx_parts) if idx_parts else torch.zeros(0, dtype=torch.long, device=q_test.device)
    return {"q_test": q_test, "q_train": q_train, "train_query_idx": train_query_idx}
