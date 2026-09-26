"""Phase-A trainable-parameter tiers: tier 0 (label path, ICL norms, decoder) plus LoRA on more attention stages per tier."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch.nn as nn

from copula_inter.lora import (
    apply_lora,
    apply_lora_all_layers,
)
from copula_inter.marginal_backbones import TIER0_PATTERNS as _BACKBONE_TIER0  # noqa: E402
from copula_inter.marginal_backbones import (  # noqa: E402
    assert_patterns_match,
    resolve_tier,
)

if TYPE_CHECKING:
    pass


# Tier 0: the label path, the ICL-stage norms and the decoder (per architecture
# in marginal_backbones.py; re-exported here).
TIER0_PATTERNS: tuple[str, ...] = _BACKBONE_TIER0["tabicl"]


# Tier ladder: tier 0 plus LoRA on successively more attention stages. No full fine-tuning.
TIER_SPECS: dict[int, dict] = {
    0: {
        "lora_stages": [],
        "desc": "label path + ICL norms + decoder (~1.6M, 5.5%)",
    },
    1: {
        "lora_stages": ["icl"],
        "desc": "tier 0 + LoRA on icl_predictor attention",
    },
    2: {
        "lora_stages": ["icl", "row"],
        "desc": "tier 1 + LoRA on row_interactor attention",
    },
    3: {
        "lora_stages": ["icl", "row", "col"],
        "desc": "tier 2 + LoRA on col_embedder attention",
    },
}


def trainable_param_report(module: nn.Module) -> dict:
    """{n_trainable_params, n_total_params, trainable_frac, n_trainable_tensors}."""
    n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in module.parameters())
    return {
        "n_trainable_params": int(n_train),
        "n_total_params": int(n_total),
        "trainable_frac": float(n_train / max(n_total, 1)),
        "n_trainable_tensors": int(sum(1 for p in module.parameters() if p.requires_grad)),
    }


def apply_tier(
    backbone: nn.Module,
    tier: int,
    *,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_target: str = "qkvo",
    extra_patterns: Sequence[str] = (),
    backbone_name: str = "tabicl",
    all_layers: bool = False,
) -> dict:
    """Make the backbone's parameters trainable for the given tier (tier-0 allowlist, plus LoRA for tier >= 1 or all-layer LoRA).

    Returns trainable_param_report(backbone) plus the tier and LoRA settings.
    """
    if tier not in TIER_SPECS:
        raise ValueError(f"Unknown tier {tier}; expected one of {sorted(TIER_SPECS)}.")
    # Raises when the architecture cannot reach the requested tier.
    if not all_layers:
        # all_layers LoRA is not limited by the stage ladder.
        resolve_tier(backbone_name, tier)
    spec = TIER_SPECS[tier]
    stages = list(spec["lora_stages"])
    patterns = tuple(_BACKBONE_TIER0[backbone_name]) + tuple(extra_patterns)
    # Fail if a tier-0 pattern matches nothing.
    assert_patterns_match(backbone, _BACKBONE_TIER0[backbone_name])

    if all_layers:
        # LoRA on every 2-D weight matrix at one shared rank (via parametrization).
        n_replaced = apply_lora_all_layers(
            backbone=backbone,
            rank=int(lora_rank),
            alpha=float(lora_alpha),
            also_trainable=patterns,
        )
    else:
        n_replaced = apply_lora(
            backbone=backbone,
            rank=int(lora_rank) if stages else 0,
            alpha=float(lora_alpha),
            target=lora_target,
            stages=stages,
            also_trainable=patterns,
        )
    report = trainable_param_report(backbone)
    report.update(
        {
            "backbone": backbone_name,
            "tier": tier,
            "tier_desc": "all layers (LoRA on every 2-D weight)" if all_layers else spec["desc"],
            "lora_all_layers": bool(all_layers),
            "lora_rank": int(lora_rank),
            "lora_stages": stages,
            "lora_modules_replaced": n_replaced,
        }
    )
    return report
