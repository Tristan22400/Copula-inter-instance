"""copula_backbones.py — per-architecture COPULA backbone adapter.

This is the copula-side analogue of ``src/marginal_backbones.py``: that
module lets Phase A fine-tune a standalone marginal on a choice of tabular
foundation model, and this one lets ``src/model.py``'s CopulaTabICL wrap a
choice of backbone as its frozen-or-finetuned feature extractor. The two are
INDEPENDENT choices — ``cfg.model.backbone`` (this module) selects the
copula's own trunk, while ``cfg.data.z_train_source``/Phase-A's
``marginal.backbone`` (marginal_backbones.py) select the marginal used for
PIT. A run can mix them freely, e.g. a TabLDM copula backbone scored against
a frozen TabICL marginal (the default combination — see below).

WHY ONLY THESE TWO, and why the interface is this narrow: everything
architecture-specific CopulaTabICL needs is:

  1. how to build the pretrained (or from-scratch) trunk,
  2. how to strip its quantile decoder so it emits raw features instead
     (the "feature-extractor" pattern the whole model is built on), and
  3. an optional auxiliary loss term the trunk itself wants added (MoE
     load-balance/z-loss — only TabLDM has one).

TabICL and Xiaomi-TabLDM are the only two backbones registered here because
they are the only two with a strippable ``icl_predictor.decoder``: TabLDM
forks TabICL's architecture wholesale (same col_embedder/row_interactor/
icl_predictor stages, same decoder shape ``Sequential(Linear, GELU,
Linear)``, same ``forward(X, y_train) -> (B, N_test, feature_dim)``
contract — verified empirically, not assumed: stripping the decoder and
running a loss backward through the result reaches 642/643 parameter
tensors). EXAONE/TabPFN (marginal_backbones.py's other two) have no such
swappable stage — EXAONE's attention holds raw Parameters instead of a
child module, and neither exposes a col/row/icl three-stage split — so
neither is a copula-backbone candidate; they stay marginal-only.

RECOMPUTE (gradient checkpointing) is handled differently for the two
backbones — see ``_load_tabldm``'s docstring for why.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_TABICL_SRC = os.path.join(_REPO_ROOT, "tabicl_upstream", "src")
if _TABICL_SRC not in sys.path:
    sys.path.insert(0, _TABICL_SRC)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tabicl._model.tabicl import TabICL  # type: ignore[import]

__all__ = [
    "BACKBONE_NAMES",
    "load_raw_backbone",
    "strip_decoder",
    "moe_aux_loss",
]

BACKBONE_NAMES: tuple[str, ...] = ("tabicl", "tabldm")


# ---------------------------------------------------------------------------
# TabICL — moved verbatim from model.py (byte-identical behaviour).
# ---------------------------------------------------------------------------
def _load_pretrained_tabicl(ckpt_name: str, recompute: bool = False) -> TabICL:
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(repo_id="jingang/TabICL", filename=ckpt_name)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # The checkpoint's saved config carries whatever `recompute` value the
    # original TabICL training run used (checkpointing is a training-time-only
    # memory/compute tradeoff, so it's almost always False in a saved config).
    # Override it here rather than after construction: `recompute` is threaded
    # through TabICL.__init__ into col_embedder/row_interactor/icl_predictor
    # and further down into their own nested encoders, each capturing its own
    # `self.recompute` at construction time — flipping an attribute post-hoc
    # on only the top-level submodules would miss those nested copies. It adds
    # no parameters (pure torch.utils.checkpoint control flow), so this has no
    # effect on `load_state_dict` compatibility below.
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


# ---------------------------------------------------------------------------
# Xiaomi-TabLDM — reuse eval/spatial/marginal_backends.py's loader, not a
# reimplementation (the same "REUSE, NOT REIMPLEMENTATION" reasoning
# eval/spatial/tabldm_batched.py's module docstring gives): the real class is
# TabLDMSparseMoE with a swapped-in ColEmbeddingDualStream column embedder
# and a dropped dense FFN, none of which this repo should re-derive.
# ---------------------------------------------------------------------------
def _load_tabldm(cfg: DictConfig) -> nn.Module:
    """Load the pretrained Xiaomi-TabLDM trunk (occams/Xiaomi-TabLDM on HF).

    No from-scratch path exists: unlike TabICL, there is no published
    architecture spec independent of the released checkpoint's own saved
    config, so ``cfg.tabicl.pretrained=false`` combined with
    ``cfg.model.backbone=tabldm`` raises rather than silently building
    something the checkpoint's config happens to describe.

    RECOMPUTE override — deliberately asymmetric, unlike TabICL's:
    empirically (read off the loaded module, not assumed), the released
    checkpoint's own saved config already enables gradient checkpointing on
    two of its three stages (``row_interactor.encoder_prefix.recompute`` and
    ``icl_predictor.tf_icl.recompute`` both True out of the box) and leaves
    the third off (``col_embedder.tf_col.recompute`` False — TabLDMRegressor.
    _load_model hardcodes ``recompute=False`` when it rebuilds
    ColEmbeddingDualStream post-load, regardless of the checkpoint's own
    config value). All three attributes are read fresh at every forward
    call (``if self.recompute: checkpoint(...)`` — plain mutable instance
    state, not baked into a closure at construction), so flipping them
    post-hoc is safe.

    Given that, ``cfg.tabicl.recompute=true`` here means "escalate": force
    every discovered ``.recompute`` flag to True (useful under CUDA OOM,
    trading compute for the extra activation memory this 71M-param backbone
    needs relative to TabICL's 28.5M). Leaving it at the shared config
    group's default (False) is a no-op — it does NOT downgrade the
    checkpoint's own already-True row/icl settings back to False, unlike
    TabICL's symmetric override. An explicit False would be equally
    surprising to silently honour (it would undo an upstream author's own
    tuning for the two heaviest stages), so this module simply never wires
    "false" to do anything: the only way to make things worse than the
    shipped checkpoint would be to overwrite it, and the config default
    used across every model preset (see conf/model/copula_prod.yaml's
    ``tabicl.recompute: false``) does not opt into that.
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
        print(f"[copula_backbones] tabldm: forced recompute=True on {n_flipped} "
              "submodules (gradient checkpointing escalated for OOM headroom).")

    return module


# ---------------------------------------------------------------------------
# Shared, architecture-agnostic operations
# ---------------------------------------------------------------------------
def load_raw_backbone(name: str, cfg: DictConfig) -> nn.Module:
    """Build the requested backbone's trunk, BEFORE decoder-stripping.

    Reads cfg.tabicl.* regardless of ``name`` — that config group is shared
    across backbones (pretrained/ckpt/recompute/arch for tabicl; pretrained/
    recompute only for tabldm, which has no ckpt/arch of its own). It is
    also read independently by the frozen marginal/PIT path (pit.py,
    live_dataset.py, era5_live_dataset.py) — those are UNAFFECTED by
    ``name``, which only selects the copula's own trunk (see this module's
    docstring).
    """
    if name == "tabicl":
        pretrained = bool(cfg.tabicl.get("pretrained", True))
        recompute = bool(cfg.tabicl.get("recompute", False))
        if pretrained:
            return _load_pretrained_tabicl(cfg.tabicl.ckpt, recompute=recompute)
        return _build_tabicl_scratch(cfg)
    if name == "tabldm":
        return _load_tabldm(cfg)
    raise ValueError(f"Unknown copula backbone {name!r}; expected one of {list(BACKBONE_NAMES)}.")


def strip_decoder(module: nn.Module) -> int:
    """Replace ``module.icl_predictor.decoder`` with ``nn.Identity()``.

    Both backbones share this shape (``icl_predictor.decoder ==
    Sequential(Linear(feature_dim, hidden), GELU, Linear(hidden,
    num_quantiles))`` — verified equal in structure for TabICL and TabLDM,
    just different `feature_dim`/`hidden`/`num_quantiles` numbers), so one
    function serves both. Returns the discovered ``feature_dim`` (the first
    Linear's ``in_features``) so the caller can size ``copula_head``.
    """
    decoder = module.icl_predictor.decoder
    first_linear = decoder[0]  # nn.Sequential(Linear, GELU, Linear)
    in_features = first_linear.in_features
    module.icl_predictor.decoder = nn.Identity()
    return in_features


def moe_aux_loss(name: str, module: nn.Module) -> Optional[Tensor]:
    """This backbone's own auxiliary training loss, or None.

    Only TabLDM has one: its Mixture-of-Experts routing carries a built-in
    z-loss + load-balance term (``icl_predictor.moe_aux_loss()``, already
    weighted by the checkpoint's own ``moe_router_z_loss_coef``/
    ``moe_load_balance_loss_coef`` — confirmed grad-carrying, e.g.
    ``tensor(0.0100, grad_fn=<MeanBackward0>)`` after a real train-mode
    forward). Dropping this term when fine-tuning the trunk end-to-end risks
    expert collapse/imbalance with no training signal correcting it, so
    ``model.py``'s forward surfaces it in the output dict for train.py to
    add to the total loss (see training.moe_aux_weight). TabICL has no MoE
    and returns None unconditionally.
    """
    if name == "tabldm":
        return module.icl_predictor.moe_aux_loss()
    return None
