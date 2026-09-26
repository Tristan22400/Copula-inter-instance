"""LoRA adapters for TabICL-style backbones.

LoRAMultiheadAttention replaces a MultiheadAttention (whose in_proj_weight is
a raw Parameter) with frozen base weights plus trainable A/B matrices:
W_eff = W + (B @ A) * alpha / rank. apply_lora swaps them in per stage;
apply_lora_all_layers instead adds a LoRA parametrization to every 2-D weight.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _get_mha_class():
    """Tuple of MultiheadAttention classes LoRAMultiheadAttention can replace (TabICL's, plus TabLDM's if installed)."""
    from tabicl._model.layers import MultiheadAttention  # type: ignore[import]

    classes = [MultiheadAttention]
    try:
        from tabldm._model.layers import MultiheadAttention as TabLDMMHA  # type: ignore[import]
    except Exception:
        pass
    else:
        if TabLDMMHA is not MultiheadAttention:
            classes.append(TabLDMMHA)
    return tuple(classes)


def _get_mha_forward():
    from tabicl._model.attention import multi_head_attention_forward  # type: ignore[import]

    return multi_head_attention_forward


def _get_kv_types():
    from tabicl._model.kv_cache import KVCacheEntry  # type: ignore[import]
    from tabicl._model.rope import RotaryEmbedding  # type: ignore[import]

    return KVCacheEntry, RotaryEmbedding


class LoRAMultiheadAttention(nn.Module):
    """MultiheadAttention with LoRA adapters on the selected projections.

    W_eff = W_frozen + (B @ A) * alpha / rank; frozen weights are buffers, A is
    Kaiming-uniform and B zero-initialized.

    Args:
        mha: the MultiheadAttention to wrap.
        rank: LoRA rank r.
        alpha: scaling (scale = alpha / rank).
        target: projections to adapt, a subset of "qkvo".
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

        # Trainable A: (rank, D), B: (D, rank); B = 0 so the delta starts at zero.
        for proj in ("q", "k", "v", "o"):
            if proj in target:
                A = nn.Parameter(torch.empty(rank, D, device=mha.in_proj_weight.device))
                B = nn.Parameter(torch.zeros(D, rank, device=mha.in_proj_weight.device))
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                setattr(self, f"lora_A_{proj}", A)
                setattr(self, f"lora_B_{proj}", B)

    def _effective_in_proj_weight(self) -> Tensor:
        W = self.in_proj_weight  # (3D, D) buffer
        D = self.embed_dim
        s = self.scaling
        delta = W.new_zeros(3 * D, D)  # always zero for absent projections
        if "q" in self.target:
            delta[:D] = s * (self.lora_B_q @ self.lora_A_q)
        if "k" in self.target:
            delta[D : 2 * D] = s * (self.lora_B_k @ self.lora_A_k)
        if "v" in self.target:
            delta[2 * D : 3 * D] = s * (self.lora_B_v @ self.lora_A_v)
        return W + delta

    def _effective_out_proj_weight(self) -> Tensor:
        W = self.out_proj_weight  # (D, D) buffer
        if "o" in self.target:
            return W + self.scaling * (self.lora_B_o @ self.lora_A_o)
        return W

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
            stage_match = any(_STAGE_KEYWORDS[s] in full_name for s in stages if s in _STAGE_KEYWORDS)
            if stage_match:
                lora_mha = LoRAMultiheadAttention(child, rank=rank, alpha=alpha, target=target)
                setattr(parent, child_name, lora_mha)
                replaced += 1
        else:
            replaced += _replace_mha_in_module(child, full_name, rank, alpha, target, stages, MultiheadAttention)
    return replaced


def is_lora_param_name(name: str) -> bool:
    """True iff *name* (a ``named_parameters()`` key) is a LoRA A/B matrix."""
    return (
        name.startswith("lora_A_")
        or name.startswith("lora_B_")
        or ".lora_A_" in name
        or ".lora_B_" in name
        # Parametrization form (apply_lora_all_layers): the adapter lives at
        # <module>.parametrizations.<weight>.0.{A,B}.
        or (".parametrizations." in name and (name.endswith(".A") or name.endswith(".B")))
    )


def set_trainable(
    backbone: nn.Module,
    also_trainable: Sequence[str] = (),
) -> int:
    """Freeze every backbone parameter except LoRA adapters and parameters matching also_trainable.

    Args:
        backbone: module, frozen in place.
        also_trainable: regex patterns (re.search on named_parameters() keys)
            kept trainable. A parametrization's frozen .original never is.

    Returns:
        Number of trainable parameter tensors.
    """
    regexes = [re.compile(pat) for pat in also_trainable]
    n_trainable = 0
    for name, param in backbone.named_parameters():
        # Never unfreeze a parametrization's frozen .original.
        allow = not is_parametrized_original(name) and any(r.search(name) for r in regexes)
        keep = is_lora_param_name(name) or allow
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
    """Replace the MultiheadAttention modules in the given stages with LoRA versions and freeze the rest.

    rank <= 0 or empty stages installs no adapters (only also_trainable stays
    trainable).

    Args:
        backbone: TabICL-style backbone.
        rank: LoRA rank; <= 0 disables adapters.
        alpha: scaling (scale = alpha / rank).
        target: subset of "qkvo".
        stages: subset of "col", "row", "icl".
        also_trainable: regex patterns kept trainable (see set_trainable).

    Returns:
        Number of modules replaced.
    """
    stages = list(stages)
    adapters_requested = int(rank) > 0 and len(stages) > 0

    n_replaced = 0
    if adapters_requested:
        MultiheadAttention = _get_mha_class()
        n_replaced = _replace_mha_in_module(backbone, "", int(rank), alpha, target, stages, MultiheadAttention)
        if n_replaced == 0:
            raise RuntimeError(
                f"apply_lora found 0 MultiheadAttention modules in stages={stages}. "
                "Check that stage names are correct ('col', 'row', 'icl')."
            )

    set_trainable(backbone, also_trainable)
    return n_replaced


def lora_state_dict(model: nn.Module) -> dict:
    """State dict of the parameters that require gradients (LoRA A/B and the copula head)."""
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


def merge_lora_weights(model: nn.Module) -> None:
    """Fold the LoRA deltas into the frozen weights in place and zero A/B."""
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
    """State dict with LoRA deltas merged and the original TabICL parameter names restored.

    Loads strictly into a stock TabICL; the live module keeps its adapters.
    Returns detached CPU tensors.
    """
    lora_paths = [name for name, m in backbone.named_modules() if isinstance(m, LoRAMultiheadAttention)]

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


# All-layer LoRA via torch.nn.utils.parametrize: adapts parameters rather than
# modules, so it covers architectures without swappable attention (EXAONE).
class LoRAParametrization(nn.Module):
    """Parametrization W -> W + (B @ A) * alpha / rank.

    A/B and the delta are float32, cast back to W's dtype. B is zero-initialized,
    so the weight is unchanged at step 0.
    """

    def __init__(self, weight: Tensor, rank: int, alpha: float):
        super().__init__()
        out_features, in_features = weight.shape[-2], weight.shape[-1]
        self.A = nn.Parameter(torch.zeros(rank, in_features, dtype=torch.float32, device=weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32, device=weight.device))
        self.scaling = float(alpha) / float(rank)

    def forward(self, weight: Tensor) -> Tensor:
        delta = (self.B @ self.A) * self.scaling
        return weight + delta.to(device=weight.device, dtype=weight.dtype).view_as(weight)


def is_parametrized_original(name: str) -> bool:
    """True for the frozen base weight hidden behind a parametrization ("...parametrizations.<name>.original")."""
    return ".parametrizations." in name and name.endswith(".original")


def apply_lora_all_layers(
    backbone: nn.Module,
    rank: int,
    alpha: float,
    also_trainable: Sequence[str] = (),
    skip_patterns: Sequence[str] = (),
) -> int:
    """Add a LoRA parametrization to every 2-D weight parameter of backbone; return how many.

    Attention modules already replaced by apply_lora are skipped (their weights
    are buffers).
    """
    import torch.nn.utils.parametrize as P

    skip = [re.compile(p) for p in skip_patterns]

    # Collect targets before registering, so the adapters' own A/B are not adapted.
    targets = [
        (mod_name, module)
        for mod_name, module in backbone.named_modules()
        if not isinstance(module, LoRAParametrization) and not P.is_parametrized(module)
    ]

    n_adapted = 0
    for mod_name, module in targets:
        for p_name, param in list(module.named_parameters(recurse=False)):
            full = f"{mod_name}.{p_name}" if mod_name else p_name
            if param.dim() != 2 or is_lora_param_name(full):
                continue
            if any(r.search(full) for r in skip):
                continue
            P.register_parametrization(module, p_name, LoRAParametrization(param, rank, alpha))
            n_adapted += 1

    set_trainable(backbone, also_trainable)
    return n_adapted


def merged_base_state_dict_parametrized(backbone: nn.Module) -> dict:
    """merged_base_state_dict for parametrized adapters: effective weights under the original names."""
    import torch.nn.utils.parametrize as P

    effective = {}
    for mod_name, module in backbone.named_modules():
        if not P.is_parametrized(module):
            continue
        for p_name in list(module.parametrizations.keys()):  # type: ignore[union-attr]
            full = f"{mod_name}.{p_name}" if mod_name else p_name
            effective[full] = getattr(module, p_name).detach().cpu().clone()

    out = {}
    for name, tensor in backbone.state_dict().items():
        if ".parametrizations." in name:
            continue  # emitted under its original name from `effective`
        out[name] = tensor.detach().cpu().clone()
    out.update(effective)
    return out


def merged_base_state_dict_any(backbone: nn.Module) -> dict:
    """merged_base_state_dict for whichever adapter style is installed."""
    import torch.nn.utils.parametrize as P

    has_parametrized = any(P.is_parametrized(m) for m in backbone.modules())
    if has_parametrized:
        return merged_base_state_dict_parametrized(backbone)
    return merged_base_state_dict(backbone)
