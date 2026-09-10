"""marginal_backbones.py — per-architecture adapter layer for Phase-A
marginal fine-tuning (src/finetune_marginal.py).

Phase A fine-tunes a STANDALONE tabular foundation model so its marginal
predictive density is better calibrated, then hands the result to a copula
run. It was written for TabICL, and everything architecture-specific about
it lives in four places:

  1. which parameters a tier makes trainable,
  2. how to get a GRADIENT-CARRYING quantile forward out of the model,
  3. what a checkpoint looks like, and
  4. how a later run loads that checkpoint back.

This module is those four things per architecture, so finetune_marginal.py
holds the training loop and the loss (both architecture-agnostic: the loss
is defined on a quantile grid and the targets, see marginal_objective) and
nothing else.

WHAT EACH ARCHITECTURE SUPPORTS, and why it differs -- this is a real
structural difference, not an implementation gap left for later:

  "tabicl"  tiers 0-3. The original. Routed through pit.py::
            run_pit_batched_grad, unchanged.
  "tabldm"  tiers 0-3. Xiaomi-TabLDM forks TabICL's architecture: its
            top-level modules are the same col_embedder / row_interactor /
            icl_predictor, so lora.py's _STAGE_KEYWORDS already address it,
            and its whole attention stack is BYTE-IDENTICAL to TabICL's
            (tests/test_lora_tabldm_compat.py), so LoRAMultiheadAttention is
            a valid drop-in. Only the tier-0 parameter NAMES differ (its ICL
            blocks are `layers.N.{attn,mlp}_norm`, TabICL's are
            `blocks.N.norm[12]`), which is what TIER0_PATTERNS below encodes.
  "exaone"  tier 0 only. EXAONE-Tabular is NOT a TabICL derivative: it has
            no col_embedder/row_interactor/icl_predictor stages for
            _STAGE_KEYWORDS to match, no affine norm parameters at all, and
            -- the blocking one -- its attention holds raw Parameters
            (`transformer.layers.N.item_attention.output_weight`) rather
            than swappable nn.Module attention children, so there is nothing
            for apply_lora to replace. Tier 0 is well-defined and useful
            (label path + the 999-quantile head, ~459K/21.1M = 2.2%,
            comparable to TabICL's tier-0 5.5%); tier >= 1 would need a
            Parameter-level LoRA, which lora.py does not implement. Asking
            for it raises rather than silently training tier 0.
  "tabpfn"  tier 0 only, and NOT execution-verified here -- PriorLabs gates
            the weights behind a licence + TABPFN_TOKEN which this
            environment does not have, so its patterns below are written
            from TabPFN's published module layout and have never been run.
            Same caveat eval/spatial/tabpfn_batched.py already carries. Set
            TABPFN_TOKEN and run tests/test_marginal_backbones.py before
            relying on it.

Gradient flow was verified per architecture, not assumed: a loss on the
quantile output reaches 639/647 parameter tensors for TabLDM and 363/365
for EXAONE (both models' forwards are ordinary autograd graphs; the
`torch.no_grad()`/`torch.inference_mode()` that normally wraps them lives in
the inference wrappers this module deliberately bypasses, exactly as
pit.py::run_pit_batched_grad bypasses run_pit_batched's decorator).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "BACKBONE_NAMES",
    "TIER0_PATTERNS",
    "MAX_TIER",
    "MarginalBackbone",
    "load_backbone",
    "resolve_tier",
]

BACKBONE_NAMES: tuple[str, ...] = ("tabicl", "exaone", "tabpfn", "tabldm")

# Tier-0 = "the label path, the norms the trunk's output is rescaled by, and
# the decoder that turns trunk features into a quantile grid" -- the parts
# that can recalibrate what the frozen trunk already computes. Every pattern
# below was read off the real fitted module's named_parameters(), not guessed
# from the paper (tests/test_marginal_backbones.py asserts each one still
# matches at least one parameter, so an upstream rename fails loudly instead
# of silently training nothing).
TIER0_PATTERNS: dict[str, tuple[str, ...]] = {
    # Unchanged from finetune_marginal.TIER0_PATTERNS -- kept here so all
    # four architectures are described in one place; that module now imports
    # this one.
    "tabicl": (
        r"^icl_predictor\.y_encoder\.",
        r"^col_embedder\.y_encoder\.",
        r"^icl_predictor\.ln\.",
        r"^icl_predictor\.tf_icl\.blocks\.\d+\.norm[12]\.",
        r"^icl_predictor\.decoder\.",
    ),
    # Same shape as TabICL's, retargeted at TabLDM's names: its ICL trunk is
    # `tf_icl.layers.N.{attn,mlp}_norm` (TabICL: `tf_icl.blocks.N.norm[12]`)
    # and it additionally carries the attention-residual norms that give
    # AttnResLightRMSNorm its name.
    "tabldm": (
        r"^icl_predictor\.y_encoder\.",
        r"^col_embedder\.y_encoder\.",
        r"^icl_predictor\.ln\.",
        r"^icl_predictor\.tf_icl\.layers\.\d+\.(attn|mlp)_norm\.",
        r"^icl_predictor\.tf_icl\.attn_res_norms\.\d+\.",
        r"^icl_predictor\.decoder\.",
    ),
    # EXAONE has no affine norms; its tier-0 analogue is the label path plus
    # the head that emits the 999-level quantile bank. feature_summary_tokens
    # /item_summary_tokens are deliberately EXCLUDED: they are learned trunk
    # inputs, not output recalibration, so they belong to a tier that can
    # change how context is aggregated -- which is exactly what tier 0 is
    # defined not to do.
    "exaone": (
        r"^label_encoder\.",
        r"^transformer\.classification_heads\.",
    ),
    # From TabPFN v3's published layout; see the module docstring's caveat --
    # unverified in this environment.
    "tabpfn": (
        r"^y_encoder\.",
        r"^decoder_dict\.",
    ),
}

# Tiers >= 1 install LoRA on attention modules (lora.py). Only architectures
# whose attention is a swappable nn.Module that LoRAMultiheadAttention can
# stand in for can climb the ladder -- see the module docstring.
MAX_TIER: dict[str, int] = {"tabicl": 3, "tabldm": 3, "exaone": 0, "tabpfn": 0}


def resolve_tier(backbone_name: str, tier: int) -> int:
    """Validate ``tier`` for ``backbone_name``, or raise with the reason.

    Raising beats silently clamping: a run launched at tier 1 that quietly
    trained tier 0 would look like "the ladder didn't help" in wandb, which
    is precisely the wrong conclusion to draw.
    """
    if backbone_name not in MAX_TIER:
        raise ValueError(
            f"Unknown marginal backbone {backbone_name!r}; expected one of {list(BACKBONE_NAMES)}."
        )
    top = MAX_TIER[backbone_name]
    if tier > top:
        raise ValueError(
            f"marginal.tier={tier} is not available for backbone {backbone_name!r} "
            f"(max {top}). Tiers >= 1 install LoRA adapters on attention modules "
            f"(src/lora.py), and {backbone_name}'s attention is not a swappable "
            "nn.Module this repo has an adapter for -- see "
            "src/marginal_backbones.py's module docstring for the specifics. "
            "Use marginal.tier=0, or add a Parameter-level LoRA to lora.py."
        )
    return int(tier)


@dataclass
class MarginalBackbone:
    """One fine-tunable marginal, with everything Phase A needs from it.

    ``module`` is the nn.Module whose parameters get trained (and whose
    state_dict is saved). ``handle`` is the library-level wrapper it came
    from -- the sklearn regressor for exaone/tabpfn/tabldm, None for tabicl
    -- kept because those wrappers own the preprocessing that has to run
    per episode before the trunk sees anything.
    """

    name: str
    module: nn.Module
    handle: Any = None
    config: dict = field(default_factory=dict)

    @property
    def tier0_patterns(self) -> tuple[str, ...]:
        return TIER0_PATTERNS[self.name]

    @property
    def max_tier(self) -> int:
        return MAX_TIER[self.name]

    def trainable_report(self) -> dict:
        n_train = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.module.parameters())
        return {
            "backbone": self.name,
            "n_trainable_params": int(n_train),
            "n_total_params": int(n_total),
            "trainable_frac": float(n_train / max(n_total, 1)),
        }

    # -- gradient-carrying quantile forward -----------------------------------
    def quantile_forward(
        self, X_context: Sequence[np.ndarray], y_context: Sequence[np.ndarray],
        X_query: Sequence[np.ndarray], probs: np.ndarray,
    ) -> torch.Tensor:
        """(B, n_query, Q) quantiles in RAW y-units, WITH gradients.

        Same contract as eval/spatial/_batched_pit.py's ``bank_fn``, except
        it returns a grad-carrying torch.Tensor instead of a detached numpy
        array -- the Phase-A loss is defined on these outputs, so the graph
        back to the trunk must survive.
        """
        return _QUANTILE_FORWARDS[self.name](self, X_context, y_context, X_query, probs)

    def quantile_dist_module(self, probs: np.ndarray) -> nn.Module:
        """The parameterless quantile-grid -> distribution head Phase A's loss
        scores through (finetune_marginal.marginal_objective's third argument),
        built on the SAME ``probs`` grid quantile_forward was asked for.

        Deliberately not the model's own ``quantile_dist`` attribute, even
        though TabICL and TabLDM both have one and the class is byte-identical
        between them: that instance is constructed with the model's NATIVE
        alpha levels (999), so handing it a grid of any other size raises a
        shape error deep inside its spline setup ("size of tensor a (18) must
        match tensor b (998)"). Native levels are only guaranteed on the
        TabICL path, whose quantiles come from pit.py::run_pit_batched_grad;
        every backbone routed through quantile_forward picks its own grid.

        Constructing one here is reuse, not reimplementation: the mapping is a
        pure function of (quantile grid, alpha levels) holding no parameters,
        and it is the same class both libraries ship.
        """
        from tabicl._model.quantile_dist import QuantileToDistribution

        return QuantileToDistribution(alpha_levels=list(probs)).to(
            next(self.module.parameters()).device
        )

    # -- checkpointing ---------------------------------------------------------
    def save(self, path: str, *, step: int, cfg=None, extra: Optional[dict] = None) -> None:
        """Write a Phase-A checkpoint in this architecture's own schema.

        For tabicl that is TabICL's ``{"config","state_dict"}`` (consumed by
        pit.load_tabicl, so `tabicl.pit_ckpt=<path>` keeps working). The
        others have no equivalent published loader, so they get the same
        shape plus a ``backbone`` tag, and are loaded back through
        eval/spatial/marginal_backends.py::make_regressor(..., ckpt=path).
        """
        from lora import merged_base_state_dict

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "config": dict(self.config),
            "state_dict": merged_base_state_dict(self.module),
            "step": int(step),
            "backbone": self.name,
        }
        if cfg is not None:
            from omegaconf import OmegaConf

            payload["cfg"] = OmegaConf.to_container(cfg, resolve=True)
        if extra:
            payload.update(extra)
        torch.save(payload, path)


# ---------------------------------------------------------------------------
# Per-architecture gradient-carrying quantile forwards.
#
# Each mirrors its eval/spatial/*_batched.py sibling's preprocessing exactly
# -- same regressor calls, same order -- and differs only in NOT wrapping the
# trunk call in no_grad/inference_mode. Keeping them next to each other here
# (rather than adding a grad flag to those modules) preserves the property
# that every existing caller of the inference modules cannot accidentally
# start building an autograd graph, the same reasoning pit.py gives for
# having a separate run_pit_batched_grad.
# ---------------------------------------------------------------------------
def _tabldm_quantile_forward(bb, X_context, y_context, X_query, probs) -> torch.Tensor:
    from eval.spatial.tabldm_batched import _episode_member_batch

    B = len(X_context)
    per_episode = [
        _episode_member_batch(bb.handle, X_context[b], y_context[b], X_query[b]) for b in range(B)
    ]
    members = per_episode[0][0].shape[0]
    device = next(bb.module.parameters()).device
    xs = torch.from_numpy(np.concatenate([e[0] for e in per_episode], axis=0)).float().to(device)
    ys = torch.from_numpy(np.concatenate([e[1] for e in per_episode], axis=0)).float().to(device)

    # regressor._batch_forward's body, minus its `with torch.no_grad()` and
    # minus the VRAM chunking (a Phase-A batch is already sized to fit).
    out = bb.module.predict_stats(
        xs, ys, output_type="quantiles", alphas=list(probs),
        inference_config=bb.handle.inference_config_,
    )  # (B*members, n_query, Q)
    out = out.reshape(B, members, -1, len(probs))

    # predict()'s own de-standardization, differentiably: StandardScaler's
    # inverse_transform is an affine map, so it is applied as one rather than
    # by calling the sklearn object (which would detach through numpy).
    banks = []
    for b in range(B):
        scaler = per_episode[b][2]
        scale = float(scaler.scale_[0]) if scaler.scale_ is not None else 1.0
        mean = float(scaler.mean_[0]) if scaler.mean_ is not None else 0.0
        banks.append((out[b] * scale + mean).mean(dim=0))
    return torch.stack(banks, dim=0)  # (B, n_query, Q)


def _exaone_quantile_forward(bb, X_context, y_context, X_query, probs) -> torch.Tensor:
    from eval.spatial.exaone_batched import _episode_member_batch

    B = len(X_context)
    per_episode = [
        _episode_member_batch(bb.handle, X_context[b], y_context[b], X_query[b]) for b in range(B)
    ]
    n_passes = len(per_episode[0][0])
    device = next(bb.module.parameters()).device

    pass_outputs = []
    for p in range(n_passes):
        support = torch.cat([per_episode[b][0][p][0] for b in range(B)], dim=0).to(device)
        label = torch.cat([per_episode[b][0][p][1] for b in range(B)], dim=0).to(device)
        query = torch.cat([per_episode[b][0][p][2] for b in range(B)], dim=0).to(device)
        members = per_episode[0][0][p][0].shape[0]
        raw = bb.handle._forward_chunked(support, label, query)  # no inference_mode here
        pass_outputs.append(raw.float().reshape(B, members, query.shape[1], -1))

    pooled = torch.cat(pass_outputs, dim=1)
    pooled = torch.sort(pooled, dim=-1).values.mean(dim=1)  # (B, n_query, native_Q)

    center = torch.tensor([per_episode[b][1] for b in range(B)], device=pooled.device).view(B, 1, 1)
    scale = torch.tensor([per_episode[b][2] for b in range(B)], device=pooled.device).view(B, 1, 1)
    bank = pooled * scale + center

    # EXAONE emits a fixed native grid; interpolate onto the caller's probs
    # the differentiable way (torch, not np.interp -- which would detach).
    native_n = bank.shape[-1]
    native = torch.linspace(
        1.0 / (native_n + 1), native_n / (native_n + 1), native_n,
        device=bank.device, dtype=bank.dtype,
    )
    return _interp_last_dim(bank, native, torch.as_tensor(probs, device=bank.device, dtype=bank.dtype))


def _interp_last_dim(values: torch.Tensor, xp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Differentiable 1-D linear interpolation along the last axis.

    np.interp's autograd-safe equivalent: gradients must reach ``values``,
    which a numpy round-trip would silently sever (producing a Phase-A run
    whose loss never moves the trunk).
    """
    idx = torch.searchsorted(xp, x.contiguous()).clamp(1, xp.numel() - 1)
    lo, hi = idx - 1, idx
    x_lo, x_hi = xp[lo], xp[hi]
    w = ((x - x_lo) / (x_hi - x_lo)).clamp(0.0, 1.0)
    v_lo = values.index_select(-1, lo)
    v_hi = values.index_select(-1, hi)
    return v_lo + (v_hi - v_lo) * w


def _tabicl_quantile_forward(bb, X_context, y_context, X_query, probs) -> torch.Tensor:
    raise NotImplementedError(
        "tabicl's Phase-A forward goes through pit.py::run_pit_batched_grad, which "
        "finetune_marginal.py calls directly -- it carries K-fold rotation and PIT "
        "semantics this generic per-call interface does not. This entry exists so "
        "the registry is total; it is never reached for backbone='tabicl'."
    )


_QUANTILE_FORWARDS: dict[str, Callable] = {
    "tabicl": _tabicl_quantile_forward,
    "tabldm": _tabldm_quantile_forward,
    "exaone": _exaone_quantile_forward,
    "tabpfn": _exaone_quantile_forward,  # placeholder; see load_backbone's guard
}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def load_backbone(name: str, *, ckpt: Optional[str] = None, device: str = "cuda") -> MarginalBackbone:
    """Build a fine-tunable backbone, optionally resuming from a Phase-A
    checkpoint this module wrote.

    Every non-tabicl backbone is constructed through
    eval/spatial/marginal_backends.py::make_regressor, so Phase A trains
    exactly the object the eval/generation paths use at inference -- same
    ensemble settings, same device policy -- rather than a second,
    separately-configured copy that could drift.
    """
    if name not in BACKBONE_NAMES:
        raise ValueError(f"Unknown marginal backbone {name!r}; expected one of {list(BACKBONE_NAMES)}.")

    if name == "tabicl":
        from pit import load_tabicl

        module, config = load_tabicl(ckpt, device, return_config=True) if ckpt else load_tabicl(ckpt, device)
        return MarginalBackbone(name=name, module=module, handle=None, config=config or {})

    if name == "tabpfn":
        raise NotImplementedError(
            "Phase-A fine-tuning for tabpfn is wired (TIER0_PATTERNS/MAX_TIER above) "
            "but has never been executed: PriorLabs gates the weights behind a licence "
            "and TABPFN_TOKEN, which this environment does not have, so its parameter "
            "names are taken from published layout rather than from a loaded model. "
            "Set TABPFN_TOKEN, run tests/test_marginal_backbones.py to confirm the "
            "patterns match real parameters, then remove this guard."
        )

    from eval.spatial.marginal_backends import make_regressor

    regressor = make_regressor(name, device=device)
    module = _trainable_module(name, regressor)
    if ckpt:
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        if payload.get("backbone") not in (None, name):
            raise ValueError(
                f"checkpoint {ckpt} was written for backbone {payload.get('backbone')!r}, "
                f"not {name!r}."
            )
        module.load_state_dict(payload["state_dict"], strict=True)
    return MarginalBackbone(name=name, module=module, handle=regressor, config={})


def _trainable_module(name: str, regressor) -> nn.Module:
    """The nn.Module inside a fitted/constructed regressor whose parameters
    Phase A trains. Each library hangs it off a different attribute."""
    if name == "tabldm":
        if getattr(regressor, "model_", None) is None:
            regressor._load_model()
        return regressor.model_
    if name == "exaone":
        return regressor.model
    if name == "tabpfn":
        return regressor.model_  # unverified, see load_backbone's guard
    raise ValueError(f"No trainable module known for backbone {name!r}.")


def assert_patterns_match(module: nn.Module, patterns: Sequence[str]) -> dict[str, int]:
    """{pattern: n_matching_parameter_tensors}, raising if any matches none.

    Phase A's failure mode without this is silent: a renamed parameter makes
    a tier-0 pattern match nothing, the run trains a smaller set than its
    logs claim, and the only symptom is a worse curve.
    """
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
            "src/marginal_backbones.py."
        )
    return counts


# ---------------------------------------------------------------------------
# K-fold rotation for the generic backends
# ---------------------------------------------------------------------------
def kfold_quantiles_grad(
    backbone: "MarginalBackbone", x_train: torch.Tensor, y_train_scaled: torch.Tensor,
    x_test: torch.Tensor, y_test_scaled: torch.Tensor, *, k_folds: int,
    probs: np.ndarray, fold_subset: Optional[Sequence[int]] = None,
) -> dict:
    """Phase-A's ``run_pit_batched_grad`` for a non-TabICL backbone.

    Returns ``{"q_test", "q_train", "train_query_idx"}`` with exactly the
    shapes and the fold GEOMETRY pit.py::_run_pit_batched_impl produces --
    ``fold_size = ceil(P/K)`` CONTIGUOUS blocks, folds taken in ascending
    order, ``train_query_idx`` listing the scored rows in returned order.

    Matching that convention is not cosmetic. finetune_marginal.py pairs
    these quantiles with ``episode_fold_targets(ep, train_idx, K, ...)``,
    whose analytic target for a row is conditioned on that row's own fold
    complement. A different partition (e.g. eval/spatial/_batched_pit.py's
    random-permutation split, which the INFERENCE path uses) would silently
    score every training row against a target computed from the wrong
    context -- a loss that still decreases, toward the wrong marginal.
    """
    import math

    B, P, _ = x_train.shape
    K = max(2, min(int(k_folds), P))
    fold_size = math.ceil(P / K)

    xtr = x_train.detach().cpu().numpy()
    ytr = y_train_scaled.detach().cpu().numpy()
    xte = x_test.detach().cpu().numpy()

    q_test = backbone.quantile_forward(
        [xtr[b] for b in range(B)], [ytr[b] for b in range(B)], [xte[b] for b in range(B)], probs,
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
            [xtr[b][ctx] for b in range(B)], [ytr[b][ctx] for b in range(B)],
            [xtr[b][qry] for b in range(B)], probs,
        )
        q_train_parts.append(q_fold)
        idx_parts.append(torch.as_tensor(qry, dtype=torch.long, device=q_test.device))

    q_train = (
        torch.cat(q_train_parts, dim=1) if q_train_parts
        else q_test.new_zeros((B, 0, len(probs)))
    )
    train_query_idx = (
        torch.cat(idx_parts) if idx_parts
        else torch.zeros(0, dtype=torch.long, device=q_test.device)
    )
    return {"q_test": q_test, "q_train": q_train, "train_query_idx": train_query_idx}
