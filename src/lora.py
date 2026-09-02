"""
lora.py — LoRA adapters for the TabICL backbone inside CopulaTabICL.

Design
------
TabICL's MultiheadAttention stores in_proj_weight (shape 3D×D) as a raw
nn.Parameter and calls multi_head_attention_forward() with it directly.
Standard PEFT libraries can't wrap this; we handle it ourselves.

LoRAMultiheadAttention is a drop-in replacement for tabicl's
MultiheadAttention.  It keeps the pretrained weights frozen as buffers
and adds trainable A/B matrices per target projection (q/k/v/o).
At forward time it computes W_eff = W_frozen + (B @ A) * scale and
passes it to the same multi_head_attention_forward() function.

apply_lora() walks the feature_extractor tree, replaces every
MultiheadAttention found inside the requested stage(s), and freezes
everything else so only LoRA parameters + the copula head are trained.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Lazy import of upstream symbols (avoids circular deps / path issues)
# ---------------------------------------------------------------------------

def _get_mha_class():
    from tabicl._model.layers import MultiheadAttention  # type: ignore[import]
    return MultiheadAttention


def _get_mha_forward():
    from tabicl._model.attention import multi_head_attention_forward  # type: ignore[import]
    return multi_head_attention_forward


def _get_kv_types():
    from tabicl._model.kv_cache import KVCacheEntry  # type: ignore[import]
    from tabicl._model.rope import RotaryEmbedding   # type: ignore[import]
    return KVCacheEntry, RotaryEmbedding


# ---------------------------------------------------------------------------
# LoRAMultiheadAttention
# ---------------------------------------------------------------------------

class LoRAMultiheadAttention(nn.Module):
    """Drop-in replacement for tabicl's MultiheadAttention with LoRA adapters.

    Frozen pretrained weights are stored as buffers (no gradient).
    LoRA A/B matrices for the selected projections are stored as Parameters.

    W_eff = W_frozen + scale * (B @ A),   scale = alpha / rank

    B is zero-initialised so the adapter is a no-op at the start of training.
    A is kaiming-uniform-initialised (standard LoRA practice).

    Args:
        mha   : the original frozen MultiheadAttention to wrap
        rank  : LoRA rank r
        alpha : LoRA scaling alpha (scale = alpha/rank)
        target: string of projection letters to adapt, subset of "qkvo"
    """

    def __init__(
        self,
        mha: nn.Module,
        rank: int,
        alpha: float,
        target: str = "qkvo",
    ) -> None:
        super().__init__()

        D = mha.embed_dim
        self.embed_dim = D
        self.num_heads = mha.num_heads
        self.dropout = mha.dropout
        self.rank = rank
        self.scaling = alpha / rank
        self.target = target

        # --- Frozen pretrained weights as buffers ---
        self.register_buffer("in_proj_weight", mha.in_proj_weight.data.clone())
        if mha.in_proj_bias is not None:
            self.register_buffer("in_proj_bias", mha.in_proj_bias.data.clone())
        else:
            self.in_proj_bias = None  # type: ignore[assignment]

        # out_proj: store weight/bias as buffers, expose via thin wrapper
        self.register_buffer("out_proj_weight", mha.out_proj.weight.data.clone())
        if mha.out_proj.bias is not None:
            self.register_buffer("out_proj_bias", mha.out_proj.bias.data.clone())
        else:
            self.out_proj_bias = None  # type: ignore[assignment]

        # ssmax_layer: keep reference, freeze
        self.ssmax_layer = mha.ssmax_layer  # nn.Module or None
        if self.ssmax_layer is not None:
            for p in self.ssmax_layer.parameters():
                p.requires_grad_(False)

        # --- Trainable LoRA matrices ---
        # A: (rank, D),  B: (D, rank)
        # B zero-init → delta = B@A = 0 at start → exact pretrained behaviour
        for proj in ("q", "k", "v", "o"):
            if proj in target:
                A = nn.Parameter(torch.empty(rank, D))
                B = nn.Parameter(torch.zeros(D, rank))
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                setattr(self, f"lora_A_{proj}", A)
                setattr(self, f"lora_B_{proj}", B)

    # ------------------------------------------------------------------
    # Effective weights (frozen base + LoRA delta)
    # ------------------------------------------------------------------

    def _effective_in_proj_weight(self) -> Tensor:
        W = self.in_proj_weight          # (3D, D) buffer
        D = self.embed_dim
        s = self.scaling
        delta = W.new_zeros(3 * D, D)   # always zero for absent projections
        if "q" in self.target:
            delta[:D] = s * (self.lora_B_q @ self.lora_A_q)
        if "k" in self.target:
            delta[D : 2 * D] = s * (self.lora_B_k @ self.lora_A_k)
        if "v" in self.target:
            delta[2 * D : 3 * D] = s * (self.lora_B_v @ self.lora_A_v)
        return W + delta

    def _effective_out_proj_weight(self) -> Tensor:
        W = self.out_proj_weight         # (D, D) buffer
        if "o" in self.target:
            return W + self.scaling * (self.lora_B_o @ self.lora_A_o)
        return W

    # ------------------------------------------------------------------
    # Forward — identical interface to tabicl's MultiheadAttention
    # ------------------------------------------------------------------

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        cached_kv=None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        rope=None,
        need_kv: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        # Replicate the mask canonicalization from the upstream forward
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="src_mask",
            target_type=query.dtype,
        )
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        mha_forward = _get_mha_forward()
        return mha_forward(
            query,
            self.num_heads,
            self._effective_in_proj_weight(),
            self.in_proj_bias,
            self.dropout,
            self._effective_out_proj_weight(),
            self.out_proj_bias,
            key=key,
            value=value,
            cached_kv=cached_kv,
            training=self.training,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            rope=rope,
            ssmax_layer=self.ssmax_layer,
            need_kv=need_kv,
        )


# ---------------------------------------------------------------------------
# apply_lora — walk the backbone and replace MultiheadAttention modules
# ---------------------------------------------------------------------------

_STAGE_KEYWORDS = {
    "col": "col_embedder",
    "row": "row_interactor",
    "icl": "icl_predictor",
}


def _replace_mha_in_module(
    parent: nn.Module,
    prefix: str,
    rank: int,
    alpha: float,
    target: str,
    stages: List[str],
    MultiheadAttention,
) -> int:
    """Recursively replace MultiheadAttention children; return replacement count."""
    replaced = 0
    for child_name, child in list(parent.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, MultiheadAttention):
            # Only replace if the full name contains one of the requested stage keywords
            stage_match = any(
                _STAGE_KEYWORDS[s] in full_name for s in stages if s in _STAGE_KEYWORDS
            )
            if stage_match:
                lora_mha = LoRAMultiheadAttention(child, rank=rank, alpha=alpha, target=target)
                setattr(parent, child_name, lora_mha)
                replaced += 1
        else:
            replaced += _replace_mha_in_module(
                child, full_name, rank, alpha, target, stages, MultiheadAttention
            )
    return replaced


def is_lora_param_name(name: str) -> bool:
    """True iff *name* (a ``named_parameters()`` key) is a LoRA A/B matrix."""
    return (
        name.startswith("lora_A_")
        or name.startswith("lora_B_")
        or ".lora_A_" in name
        or ".lora_B_" in name
    )


def set_trainable(
    backbone: nn.Module,
    also_trainable: Sequence[str] = (),
) -> int:
    """Freeze every backbone parameter except LoRA adapters and the allowlist.

    The single place that decides what Phase-A / LoRA runs optimize, so
    ``apply_lora`` and the tier routing in ``src/marginal_finetune.py`` cannot
    drift apart on the predicate.

    Args:
        backbone       : module to freeze in place.
        also_trainable : regex patterns (``re.search`` against each
                         ``named_parameters()`` key); a match keeps the
                         parameter trainable alongside the LoRA adapters.
                         Regex rather than plain substrings because the
                         useful selections are conjunctive -- "the norms
                         inside the ICL stack, but not the identically-named
                         norms in col_embedder/row_interactor" is
                         ``r"^icl_predictor\.tf_icl\.blocks\.\d+\.norm[12]\."``
                         and has no substring spelling. A pattern with no
                         metacharacters still behaves as a substring match, so
                         plain names work unchanged. Empty (the default)
                         reproduces the historical LoRA-only behaviour exactly.

    Returns:
        Number of parameter *tensors* left trainable.
    """
    regexes = [re.compile(pat) for pat in also_trainable]
    n_trainable = 0
    for name, param in backbone.named_parameters():
        keep = is_lora_param_name(name) or any(r.search(name) for r in regexes)
        param.requires_grad_(keep)
        n_trainable += int(keep)
    return n_trainable


def apply_lora(
    backbone: nn.Module,
    rank: int,
    alpha: float,
    target: str = "qkvo",
    stages: Sequence[str] = ("icl", "row", "col"),
    also_trainable: Sequence[str] = (),
) -> int:
    """Replace MultiheadAttention modules inside *backbone* with LoRA-augmented versions.

    After replacement:
    - All parameters in *backbone* that are NOT LoRA A/B matrices are frozen,
      except those matching ``also_trainable``.
    - Only ``lora_A_*``/``lora_B_*`` parameters plus the ``also_trainable``
      allowlist inside the backbone are trainable.

    ``rank <= 0`` or an empty ``stages`` degrades cleanly to "no adapters,
    allowlist only" (0 replacements, no error) — this is what makes a
    Tier-0-style run (norms/label-path/decoder, no attention adaptation)
    expressible through the same call as a LoRA tier, instead of needing a
    separate freeze path that could disagree with this one.

    Args:
        backbone       : the TabICL backbone (nn.Module)
        rank           : LoRA rank r; ``<= 0`` disables adapters entirely
        alpha          : LoRA scaling (scale = alpha / rank)
        target         : subset of "qkvo" — which projections to adapt
        stages         : stage names; valid values: "col", "row", "icl"
        also_trainable : regex patterns kept trainable on top of the adapters
                         (see ``set_trainable``)

    Returns:
        Number of MultiheadAttention modules replaced (0 when adapters are off).
    """
    stages = list(stages)
    adapters_requested = int(rank) > 0 and len(stages) > 0

    n_replaced = 0
    if adapters_requested:
        MultiheadAttention = _get_mha_class()
        n_replaced = _replace_mha_in_module(
            backbone, "", int(rank), alpha, target, stages, MultiheadAttention
        )
        if n_replaced == 0:
            raise RuntimeError(
                f"apply_lora found 0 MultiheadAttention modules in stages={stages}. "
                "Check that stage names are correct ('col', 'row', 'icl')."
            )

    set_trainable(backbone, also_trainable)
    return n_replaced


# ---------------------------------------------------------------------------
# Checkpoint helpers — save / load only the lightweight LoRA weights
# ---------------------------------------------------------------------------

def lora_state_dict(model: nn.Module) -> dict:
    """Return the minimal state dict needed to restore a LoRA-tuned model.

    Includes only parameters that require gradients (LoRA A/B + copula head).
    This is typically <1 % of the full model size.
    """
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if any(tag in k for tag in ("lora_A_", "lora_B_", "copula_head"))
    }


def load_lora_state_dict(model: nn.Module, state: dict, strict: bool = True) -> None:
    """Load a LoRA-only state dict into *model* (non-strict by default)."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict:
        non_lora_missing = [k for k in missing if "lora_" not in k and "copula_head" not in k]
        if non_lora_missing:
            raise RuntimeError(f"Unexpected missing keys: {non_lora_missing}")


# ---------------------------------------------------------------------------
# Merge LoRA weights into the frozen base for zero-overhead inference
# ---------------------------------------------------------------------------

def merge_lora_weights(model: nn.Module) -> None:
    """Bake LoRA adapters into the frozen buffers and zero out A/B matrices.

    After calling this, the model behaves identically but LoRAMultiheadAttention
    forward paths are slightly cheaper (no extra matmul).  The operation is
    in-place.  Not reversible without reloading the original checkpoint.
    """
    for module in model.modules():
        if not isinstance(module, LoRAMultiheadAttention):
            continue
        s = module.scaling
        D = module.embed_dim
        with torch.no_grad():
            if "q" in module.target:
                module.in_proj_weight[:D].add_(s * (module.lora_B_q @ module.lora_A_q))
                module.lora_B_q.zero_()
                module.lora_A_q.zero_()
            if "k" in module.target:
                module.in_proj_weight[D : 2 * D].add_(s * (module.lora_B_k @ module.lora_A_k))
                module.lora_B_k.zero_()
                module.lora_A_k.zero_()
            if "v" in module.target:
                module.in_proj_weight[2 * D : 3 * D].add_(s * (module.lora_B_v @ module.lora_A_v))
                module.lora_B_v.zero_()
                module.lora_A_v.zero_()
            if "o" in module.target:
                module.out_proj_weight.add_(s * (module.lora_B_o @ module.lora_A_o))
                module.lora_B_o.zero_()
                module.lora_A_o.zero_()


def merged_base_state_dict(backbone: nn.Module) -> dict:
    """State dict of *backbone* with LoRA deltas baked in and the ORIGINAL
    (adapter-free) TabICL key names restored.

    ``apply_lora`` swaps each ``MultiheadAttention`` for a
    ``LoRAMultiheadAttention``, which renames the pretrained tensors
    (``attn.out_proj.weight`` becomes the buffer ``attn.out_proj_weight``) and
    adds ``lora_A_*``/``lora_B_*``. A checkpoint written from that state dict
    is therefore NOT loadable by a plain ``tabicl._model.tabicl.TabICL``, which
    breaks the whole point of Phase A: its output must be a drop-in
    replacement for ``tabicl.pit_ckpt``, consumable by every existing call
    site through one config line.

    This walks the tree, computes ``W + (alpha/r)*B@A`` for every adapted
    projection, and re-emits it under the name the base model expects, so
    ``TabICL(**config).load_state_dict(merged_base_state_dict(bb))`` succeeds
    strictly. Non-adapted parameters pass through untouched. Unlike
    ``merge_lora_weights`` this is non-destructive -- the live module keeps its
    adapters and can go on training after an intermediate checkpoint write.

    Returns CPU tensors (detached clones), ready for ``torch.save``.
    """
    lora_paths = [
        name for name, m in backbone.named_modules()
        if isinstance(m, LoRAMultiheadAttention)
    ]

    sd: dict = {}
    for key, val in backbone.state_dict().items():
        if any(key.startswith(p + ".") for p in lora_paths):
            continue  # re-emitted below under its base-model name
        sd[key] = val.detach().cpu().clone()

    for path in lora_paths:
        mod = backbone.get_submodule(path)
        sd[f"{path}.in_proj_weight"] = mod._effective_in_proj_weight().detach().cpu().clone()
        if mod.in_proj_bias is not None:
            sd[f"{path}.in_proj_bias"] = mod.in_proj_bias.detach().cpu().clone()
        sd[f"{path}.out_proj.weight"] = mod._effective_out_proj_weight().detach().cpu().clone()
        if mod.out_proj_bias is not None:
            sd[f"{path}.out_proj.bias"] = mod.out_proj_bias.detach().cpu().clone()
        if mod.ssmax_layer is not None:
            for k, v in mod.ssmax_layer.state_dict().items():
                sd[f"{path}.ssmax_layer.{k}"] = v.detach().cpu().clone()

    return sd
