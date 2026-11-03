"""Per-architecture adapters for Phase-A marginal fine-tuning.

For each backbone: which parameters each tier trains, a gradient-carrying
quantile forward, the checkpoint format and how it is loaded back.

    tabicl  tiers 0-3 (pit.run_pit_batched_grad).
    tabpfn  tier 0 only; patterns untested, and load_backbone refuses it
            until they are checked against a loaded model.

All-layer LoRA (lora.apply_lora_all_layers) works for every backbone.
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
    # From TabPFN v3's published layout (untested here).
    "tabpfn": (
        r"^y_encoder\.",
        r"^decoder_dict\.",
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
        """Number of quantile levels the decoder emits (999 for every backbone here)."""
        own = getattr(self.module, "quantile_dist", None)
        levels = getattr(own, "alpha_levels", None) if own is not None else None
        if levels is not None:
            return int(levels.numel() if torch.is_tensor(levels) else len(levels))
        if self.handle is not None and hasattr(self.handle, "manifest"):
            return int(self.handle.manifest.output_width)
        raise RuntimeError(f"cannot determine native quantile count for {self.name!r}")

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
        return _QUANTILE_FORWARDS[self.name](self, X_context, y_context, X_query, probs)

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


# Gradient-carrying quantile forwards, one per architecture.
def _tabicl_quantile_forward(
    bb: MarginalBackbone,
    X_context: Sequence[np.ndarray],
    y_context: Sequence[np.ndarray],
    X_query: Sequence[np.ndarray],
    probs: np.ndarray | None,
) -> torch.Tensor:
    raise NotImplementedError(
        "tabicl's Phase-A forward goes through pit.py::run_pit_batched_grad, which "
        "finetune_marginal.py calls directly -- it carries K-fold rotation and PIT "
        "semantics this generic per-call interface does not. This entry exists so "
        "the registry is total; it is never reached for backbone='tabicl'."
    )


_QUANTILE_FORWARDS: dict[str, Callable] = {
    "tabicl": _tabicl_quantile_forward,
}


def load_backbone(name: str, *, ckpt: Optional[str] = None, device: str = "cuda") -> MarginalBackbone:
    """Build a fine-tunable backbone via copula_inter.marginal.backends.make_regressor, optionally loading a Phase-A checkpoint."""
    if name not in MARGINAL_BACKENDS:
        raise ValueError(f"Unknown marginal backbone {name!r}; expected one of {list(MARGINAL_BACKENDS)}.")

    if name == "tabicl":
        from copula_inter.marginal.pit import PRETRAINED_TABICL_CKPT, load_tabicl

        tabicl, config = load_tabicl(ckpt or PRETRAINED_TABICL_CKPT, device, return_config=True)
        return MarginalBackbone(name=name, module=tabicl, handle=None, config=config)

    if name == "tabpfn":
        raise NotImplementedError(
            "Phase-A fine-tuning for tabpfn is wired (TIER0_PATTERNS above, max_tier in backend_registry) "
            "but has never been executed: PriorLabs gates the weights behind a licence "
            "and TABPFN_TOKEN, which this environment does not have, so its parameter "
            "names are taken from published layout rather than from a loaded model. "
            "Set TABPFN_TOKEN, run tests/test_marginal_backbones.py to confirm the "
            "patterns match real parameters, then remove this guard."
        )

    from copula_inter.marginal.backends import make_regressor

    regressor = make_regressor(name, device=device, ckpt=ckpt)
    module = _trainable_module(name, regressor)
    return MarginalBackbone(name=name, module=module, handle=regressor, config={})


def _trainable_module(name: str, regressor: Any) -> nn.Module:
    """The nn.Module inside a regressor whose parameters Phase A trains."""
    if name == "tabpfn":
        return regressor.model_  # unverified, see load_backbone's guard
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
