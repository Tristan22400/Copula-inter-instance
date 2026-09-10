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
    """Every MultiheadAttention class LoRAMultiheadAttention is a valid
    drop-in for, as a tuple suitable for ``isinstance``.

    Xiaomi-TabLDM forks TabICL's attention stack: as of tabldm 0.1.0,
    ``inspect.getsource`` of tabldm._model.layers.MultiheadAttention,
    _model.attention.multi_head_attention_forward, _model.rope.
    RotaryEmbedding and _model.kv_cache.KVCacheEntry are all BYTE-IDENTICAL
    to TabICL's (asserted by tests/test_lora_tabldm_compat.py, so a future
    upstream divergence fails loudly here instead of silently installing
    adapters whose forward no longer matches). They are still distinct class
    OBJECTS, so a lone `isinstance(child, tabicl_MHA)` silently matched
    nothing on a TabLDM backbone and apply_lora raised "found 0
    MultiheadAttention modules" -- tier >= 1 was unreachable for TabLDM for
    that reason alone, not for any architectural one.

    tabldm is optional: absent, this degrades to the TabICL-only tuple.
    """
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
                A = nn.Parameter(torch.empty(rank, D, device=mha.in_proj_weight.device))
                B = nn.Parameter(torch.zeros(D, rank, device=mha.in_proj_weight.device))
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
        # Parametrization form (apply_lora_all_layers): the adapter lives at
        # <module>.parametrizations.<weight>.0.{A,B}.
        or (".parametrizations." in name and (name.endswith(".A") or name.endswith(".B")))
    )


def set_trainable(
    backbone: nn.Module,
    also_trainable: Sequence[str] = (),
) -> int:
    r"""Freeze every backbone parameter except LoRA adapters and the allowlist.

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
        # A parametrization's `.original` is the frozen base weight by
        # construction; never let an allowlist pattern unfreeze it (see
        # is_parametrized_original).
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


# ---------------------------------------------------------------------------
# Universal (all-layers) LoRA
#
# LoRAMultiheadAttention above adapts attention by SWAPPING the module. That
# only reaches architectures whose attention is a swappable nn.Module, which
# made coverage wildly uneven across the marginal backbones: ~91% of TabLDM's
# parameters sit in Linear/MultiheadAttention children, but ~98% of EXAONE's
# are raw nn.Parameters inside custom TensorAttention/FeedForward modules
# (query_weight/key_weight/value_weight/output_weight, 106 modules) with no
# submodule to replace and no way to intercept their forward.
#
# torch.nn.utils.parametrize adapts the PARAMETER instead of the module: after
# registration, every read of `module.weight` returns W + (B@A)*scale, so the
# owning module's forward is untouched and needs to know nothing. That makes
# "every weight matrix, same rank, every architecture" achievable uniformly --
# including for modules this repo does not own and must not edit.
# ---------------------------------------------------------------------------
class LoRAParametrization(nn.Module):
    """``W -> W + (B @ A) * (alpha / rank)`` as a torch parametrization.

    A/B are held in float32 even when the base weight is float16 (EXAONE's
    released weights are), and the delta is computed in float32 before being
    cast back. Optimizer state on fp16 parameters is where small updates
    silently round to zero; the cast back is required anyway, since
    register_parametrization refuses to change a tensor's dtype.

    B is zero-initialised, so the adapted weight is EXACTLY the pretrained one
    at step 0 -- adding adapters never perturbs a pretrained model before any
    training happens (asserted in tests/test_lora_all_layers.py).
    """

    def __init__(self, weight: Tensor, rank: int, alpha: float):
        super().__init__()
        out_features, in_features = weight.shape[-2], weight.shape[-1]
        self.A = nn.Parameter(
            torch.zeros(rank, in_features, dtype=torch.float32, device=weight.device)
        )
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=torch.float32, device=weight.device)
        )
        self.scaling = float(alpha) / float(rank)

    def forward(self, weight: Tensor) -> Tensor:
        delta = (self.B @ self.A) * self.scaling
        return weight + delta.to(device=weight.device, dtype=weight.dtype).view_as(weight)


def is_parametrized_original(name: str) -> bool:
    """True for the frozen base weight a parametrization hides.

    ``register_parametrization`` renames ``foo.weight`` to
    ``foo.parametrizations.weight.original``. That name still matches
    prefix-style tier-0 patterns (``^icl_predictor\\.decoder\\.``), so without
    this guard set_trainable would unfreeze the full pretrained matrix
    alongside its adapter -- i.e. quietly full-fine-tune the very layers LoRA
    was installed on.
    """
    return ".parametrizations." in name and name.endswith(".original")


def apply_lora_all_layers(
    backbone: nn.Module,
    rank: int,
    alpha: float,
    also_trainable: Sequence[str] = (),
    skip_patterns: Sequence[str] = (),
) -> int:
    """Install a LoRA parametrization on EVERY 2-D weight matrix in *backbone*.

    Uniform by construction: one ``rank`` for every layer and every
    architecture, rather than a per-model subset determined by which modules
    happen to be swappable. Returns the number of adapted matrices.

    Only ``dim() == 2`` parameters are adapted -- a low-rank factorisation of a
    1-D tensor is meaningless, so norms and biases are untouched here and stay
    covered by the tier-0 allowlist (``also_trainable``), which is where they
    were already handled.

    Attention modules already swapped by ``apply_lora`` are skipped: their
    pretrained weights live in buffers, not parameters, so they are invisible
    to this walk and cannot be double-adapted.
    """
    import torch.nn.utils.parametrize as P

    skip = [re.compile(p) for p in skip_patterns]

    # Materialise the target list BEFORE registering anything. Registration
    # inserts a `parametrizations` ModuleDict holding a LoRAParametrization,
    # whose own A/B are 2-D parameters -- walking a live tree would adapt the
    # adapters, and then their adapters, until the recursion limit.
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
    """``merged_base_state_dict``'s counterpart for parametrized adapters.

    Returns the state dict under the ORIGINAL parameter names with each
    adapted weight replaced by its effective ``W + (B@A)*scale``, so the file
    loads into a stock model. Non-destructive: the live module keeps its
    adapters and training continues after an intermediate checkpoint write.
    """
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
    """``merged_base_state_dict`` for whichever adapter style is installed.

    A Phase-A checkpoint has to load into a stock model regardless of how it
    was adapted, and the two styles rename tensors differently: module
    replacement turns ``attn.out_proj.weight`` into a buffer
    ``attn.out_proj_weight``, parametrization turns ``foo.weight`` into
    ``foo.parametrizations.weight.original``. Callers should not have to know
    which was used -- picking the wrong merger writes a file that fails
    ``load_state_dict`` at the next run, long after the training spend.
    """
    import torch.nn.utils.parametrize as P

    has_parametrized = any(P.is_parametrized(m) for m in backbone.modules())
    if has_parametrized:
        return merged_base_state_dict_parametrized(backbone)
    return merged_base_state_dict(backbone)
