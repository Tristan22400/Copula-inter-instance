"""Copula backbones: build the trunk, strip its quantile decoder, and expose any auxiliary loss.

cfg.model.backbone selects "tabicl" or "tabldm" (both have a strippable
icl_predictor.decoder). This is independent of the marginal used for the PIT.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from copula_inter.backend_registry import COPULA_BACKBONES
from tabicl._model.tabicl import TabICL

__all__ = [
    "load_raw_backbone",
    "strip_decoder",
    "moe_aux_loss",
]


def _load_pretrained_tabicl(ckpt_name: str, recompute: bool = False) -> TabICL:
    # ckpt is a jingang/TabICL HF filename or a local .ckpt/.pt path.
    if os.path.isfile(ckpt_name):
        ckpt_path = ckpt_name
    else:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(repo_id="jingang/TabICL", filename=ckpt_name)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Set recompute in the config before construction (nested modules copy it at init).
    ckpt_config = dict(ckpt["config"])
    if recompute:
        ckpt_config["recompute"] = True
    base = TabICL(**ckpt_config)
    base.load_state_dict(ckpt["state_dict"])
    return base


def _build_tabicl_scratch(cfg: DictConfig) -> TabICL:
    """Instantiate a randomly-initialised TabICL from cfg.tabicl.arch."""
    a = cfg.tabicl.get("arch", {})
    return TabICL(
        max_classes=int(a.get("max_classes", 0)),
        num_quantiles=int(a.get("num_quantiles", 999)),
        embed_dim=int(a.get("embed_dim", 128)),
        col_num_blocks=int(a.get("col_num_blocks", 3)),
        col_nhead=int(a.get("col_nhead", 8)),
        col_num_inds=int(a.get("col_num_inds", 128)),
        col_affine=bool(a.get("col_affine", False)),
        col_feature_group=a.get("col_feature_group", "same"),
        col_feature_group_size=int(a.get("col_feature_group_size", 3)),
        col_target_aware=bool(a.get("col_target_aware", True)),
        col_ssmax=a.get("col_ssmax", "qassmax-mlp-elementwise"),
        row_num_blocks=int(a.get("row_num_blocks", 3)),
        row_nhead=int(a.get("row_nhead", 8)),
        row_num_cls=int(a.get("row_num_cls", 4)),
        row_rope_base=float(a.get("row_rope_base", 100000)),
        row_rope_interleaved=bool(a.get("row_rope_interleaved", False)),
        icl_num_blocks=int(a.get("icl_num_blocks", 12)),
        icl_nhead=int(a.get("icl_nhead", 8)),
        icl_ssmax=a.get("icl_ssmax", "qassmax-mlp-elementwise"),
        ff_factor=int(a.get("ff_factor", 2)),
        dropout=float(a.get("dropout", 0.0)),
        activation=a.get("activation", "gelu"),
        norm_first=bool(a.get("norm_first", True)),
        bias_free_ln=bool(a.get("bias_free_ln", False)),
        recompute=bool(a.get("recompute", False)),
    )


# Xiaomi-TabLDM, loaded through eval/spatial/marginal_backends.
def _load_tabldm(cfg: DictConfig) -> nn.Module:
    """Load the pretrained Xiaomi-TabLDM trunk (no from-scratch option).

    cfg.tabicl.recompute=true enables gradient checkpointing on every stage;
    false keeps the checkpoint's own settings.
    """
    if not bool(cfg.tabicl.get("pretrained", True)):
        raise ValueError(
            "model.backbone='tabldm' has no from-scratch architecture — "
            "cfg.tabicl.pretrained=false is not supported for this backbone. "
            "Use model.backbone='tabicl' for a from-scratch run, or leave "
            "tabicl.pretrained at its default (true) here."
        )

    from eval.spatial.marginal_backends import make_regressor

    regressor = make_regressor("tabldm", device="cpu")
    regressor._load_model()
    module = regressor.model_

    if bool(cfg.tabicl.get("recompute", False)):
        n_flipped = 0
        for sub in module.modules():
            if hasattr(sub, "recompute"):
                sub.recompute = True
                n_flipped += 1
        print(
            f"[copula_backbones] tabldm: forced recompute=True on {n_flipped} "
            "submodules (gradient checkpointing escalated for OOM headroom)."
        )

    return module


def load_raw_backbone(name: str, cfg: DictConfig) -> nn.Module:
    """Build the named backbone before decoder stripping, from cfg.tabicl.*."""
    if name == "tabicl":
        pretrained = bool(cfg.tabicl.get("pretrained", True))
        recompute = bool(cfg.tabicl.get("recompute", False))
        if pretrained:
            return _load_pretrained_tabicl(cfg.tabicl.ckpt, recompute=recompute)
        return _build_tabicl_scratch(cfg)
    if name == "tabldm":
        return _load_tabldm(cfg)
    raise ValueError(f"Unknown copula backbone {name!r}; expected one of {list(COPULA_BACKBONES)}.")


def strip_decoder(module: Any) -> int:
    """Replace module.icl_predictor.decoder with nn.Identity and return feature_dim (the decoder's input size)."""
    decoder = module.icl_predictor.decoder
    first_linear = decoder[0]  # nn.Sequential(Linear, GELU, Linear)
    in_features = first_linear.in_features
    module.icl_predictor.decoder = nn.Identity()
    return in_features


def moe_aux_loss(name: str, module: Any) -> Optional[Tensor]:
    """The backbone's auxiliary loss: TabLDM's MoE z-loss + load-balance term, None for TabICL."""
    if name == "tabldm":
        return module.icl_predictor.moe_aux_loss()
    return None
