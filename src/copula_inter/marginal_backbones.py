"""Per-architecture adapters for Phase-A marginal fine-tuning.

For each backbone: which parameters each tier trains, a gradient-carrying
quantile forward, the checkpoint format and how it is loaded back.

    tabicl  tiers 0-3 (pit.run_pit_batched_grad).
    tabldm  tiers 0-3 (TabICL-identical attention; different tier-0 names).
    exaone  tier 0 only (attention holds raw Parameters; no swappable module).
    tabpfn  tier 0 only; patterns untested (weights are licence-gated).

All-layer LoRA (lora.apply_lora_all_layers) works for every backbone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, cast

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from tabicl._model.quantile_dist import QuantileToDistribution

from copula_inter.artifacts import atomic_torch_save
from copula_inter.backend_registry import BACKENDS

__all__ = [
    "BACKBONE_NAMES",
    "TIER0_PATTERNS",
    "MAX_TIER",
    "MarginalBackbone",
    "load_backbone",
    "resolve_tier",
]

BACKBONE_NAMES: tuple[str, ...] = tuple(BACKENDS)

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
    # TabLDM's ICL norms are tf_icl.layers.N.{attn,mlp}_norm, plus its attention-residual norms.
    "tabldm": (
        r"^icl_predictor\.y_encoder\.",
        r"^col_embedder\.y_encoder\.",
        r"^icl_predictor\.ln\.",
        r"^icl_predictor\.tf_icl\.layers\.\d+\.(attn|mlp)_norm\.",
        r"^icl_predictor\.tf_icl\.attn_res_norms\.\d+\.",
        r"^icl_predictor\.decoder\.",
    ),
    # EXAONE has no affine norms: the label path and the quantile head.
    "exaone": (
        r"^label_encoder\.",
        r"^transformer\.classification_heads\.",
    ),
    # From TabPFN v3's published layout (untested here).
    "tabpfn": (
        r"^y_encoder\.",
        r"^decoder_dict\.",
    ),
}

# Highest tier (stage LoRA) per backbone. All-layer LoRA is not limited by this.
MAX_TIER: dict[str, int] = {name: spec.max_tier for name, spec in BACKENDS.items()}


def resolve_tier(backbone_name: str, tier: int) -> int:
    """Return tier if backbone_name supports it, else raise ValueError."""
    if backbone_name not in MAX_TIER:
        raise ValueError(f"Unknown marginal backbone {backbone_name!r}; expected one of {list(BACKBONE_NAMES)}.")
    top = MAX_TIER[backbone_name]
    if tier > top:
        raise ValueError(
            f"marginal.tier={tier} is not available for backbone {backbone_name!r} "
            f"(max {top}). Tiers >= 1 install LoRA adapters on attention modules "
            f"(src/copula_inter/lora.py), and {backbone_name}'s attention is not a swappable "
            "nn.Module this repo has an adapter for -- see "
            "src/copula_inter/marginal_backbones.py's module docstring for the specifics. "
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
    # Execution settings only; never part of the model's checkpoint schema.
    exaone_chunk_size: int = 1
    exaone_activation_checkpointing: bool = True

    @property
    def tier0_patterns(self) -> tuple[str, ...]:
        return TIER0_PATTERNS[self.name]

    @property
    def max_tier(self) -> int:
        return MAX_TIER[self.name]

    def parameters(self, *args, **kwargs):
        return self.module.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.module.named_parameters(*args, **kwargs)

    def trainable_report(self) -> dict:
        n_train = sum(p.numel() for p in self.module.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.module.parameters())
        return {
            "backbone": self.name,
            "n_trainable_params": int(n_train),
            "n_total_params": int(n_total),
            "trainable_frac": float(n_train / max(n_total, 1)),
        }

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

        The model's own quantile_dist for the native grid (TabICL's class for EXAONE).
        """
        from tabicl._model.quantile_dist import QuantileToDistribution

        if probs is None:
            own = getattr(self.module, "quantile_dist", None)
            if own is not None:
                # TabLDM's class is a byte-identical fork of TabICL's.
                return cast("QuantileToDistribution", own)
            probs = self.native_probs
        return QuantileToDistribution(alpha_levels=list(probs)).to(next(self.module.parameters()).device)

    # -- checkpointing ---------------------------------------------------------
    def save(self, path: str, *, step: int, cfg=None, extra: Optional[dict] = None) -> None:
        """Write a Phase-A checkpoint: TabICL's {"config", "state_dict"} for tabicl (loadable by pit.load_tabicl), the same plus a "backbone" tag otherwise."""
        from copula_inter.lora import merged_base_state_dict_any

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


# Gradient-carrying quantile forwards, one per architecture (same preprocessing
# as the eval/spatial/*_batched.py inference versions, without no_grad).
def _patch_tabldm_inference_manager() -> None:
    try:
        from tabldm._model.inference import InferenceManager, flash_attn3_toggle
    except ImportError:
        return
    if getattr(InferenceManager, "_grad_patched", False):
        return

    _orig_run_forward = InferenceManager._run_forward

    def _grad_aware_run_forward(self, forward_fn, inputs):
        if torch.is_grad_enabled():
            with flash_attn3_toggle(self.use_fa3):
                return forward_fn(**inputs)
        return _orig_run_forward(self, forward_fn, inputs)

    InferenceManager._run_forward = _grad_aware_run_forward
    InferenceManager._grad_patched = True


def _tabldm_quantile_forward(bb, X_context, y_context, X_query, probs) -> torch.Tensor:
    _patch_tabldm_inference_manager()
    from eval.spatial.tabldm_batched import _episode_member_batch, _group_episode_batches

    B = len(X_context)
    per_episode = [_episode_member_batch(bb.handle, X_context[b], y_context[b], X_query[b]) for b in range(B)]
    device = next(bb.module.parameters()).device
    banks: list[torch.Tensor | None] = [None] * B
    for indices in _group_episode_batches(per_episode):
        members = per_episode[indices[0]][0].shape[0]
        xs = torch.from_numpy(np.concatenate([per_episode[b][0] for b in indices], axis=0)).float().to(device)
        ys = torch.from_numpy(np.concatenate([per_episode[b][1] for b in indices], axis=0)).float().to(device)

        # Same forward as the regressor, with autograd enabled.
        kwargs = (
            {"output_type": "raw_quantiles"}
            if probs is None
            else {
                "output_type": "quantiles",
                "alphas": list(probs),
            }
        )
        out = bb.module.predict_stats(
            xs,
            ys,
            inference_config=bb.handle.inference_config_,
            **kwargs,
        )
        out = out.reshape(len(indices), members, -1, out.shape[-1])

        # Apply each episode's inverse scaling without detaching the graph.
        for local, b in enumerate(indices):
            scaler = per_episode[b][2]
            scale = float(scaler.scale_[0]) if scaler.scale_ is not None else 1.0
            mean = float(scaler.mean_[0]) if scaler.mean_ is not None else 0.0
            banks[b] = (out[local] * scale + mean).mean(dim=0)
    filled = [bank for bank in banks if bank is not None]
    assert len(filled) == B
    return torch.stack(filled, dim=0)  # (B, n_query, Q)


def _exaone_grad_forward(
    bb, support: torch.Tensor, label: torch.Tensor, query: torch.Tensor, chunk_size: Optional[int] = None
) -> torch.Tensor:
    """EXAONE forward for training: calls the model directly (no inference KV cache), with activation checkpointing, chunked along the batch axis."""
    from torch.nn.utils import parametrize
    from torch.utils.checkpoint import checkpoint

    if chunk_size is None:
        chunk_size = bb.exaone_chunk_size
    if chunk_size < 1:
        raise ValueError("EXAONE chunk_size must be positive")
    ffn_chunk = 524_288 if support.device.type == "cpu" else 9984
    query_chunk_size = query.shape[1]

    def _model_call(sub_s, sub_l, sub_q):
        # Materialize each LoRA weight once per call, inside the checkpointed function.
        with parametrize.cached():
            return bb.handle.model(
                sub_s,
                sub_l,
                sub_q,
                feedforward_token_chunk=ffn_chunk,
                query_chunk_size=query_chunk_size,
                trusted_internal_inputs=True,
            )

    use_ckpt = (
        bb.exaone_activation_checkpointing
        and torch.is_grad_enabled()
        and any(p.requires_grad for p in bb.module.parameters())
    )
    total_members = support.shape[0]
    if total_members <= chunk_size:
        if use_ckpt:
            return checkpoint(_model_call, support, label, query, use_reentrant=False)
        return _model_call(support, label, query)

    chunks = []
    for start in range(0, total_members, chunk_size):
        stop = min(start + chunk_size, total_members)
        sub_s = support[start:stop]
        sub_l = label[start:stop]
        sub_q = query[start:stop]
        if use_ckpt:
            chunks.append(checkpoint(_model_call, sub_s, sub_l, sub_q, use_reentrant=False))
        else:
            chunks.append(_model_call(sub_s, sub_l, sub_q))
    return torch.cat(chunks, dim=0)


def _exaone_quantile_forward(bb, X_context, y_context, X_query, probs) -> torch.Tensor:
    from eval.spatial.exaone_batched import _episode_member_batch

    B = len(X_context)
    per_episode = [_episode_member_batch(bb.handle, X_context[b], y_context[b], X_query[b]) for b in range(B)]
    n_passes = len(per_episode[0][0])
    device = next(bb.module.parameters()).device

    pass_outputs = []
    for p in range(n_passes):
        support = torch.cat([per_episode[b][0][p][0] for b in range(B)], dim=0).to(device)
        label = torch.cat([per_episode[b][0][p][1] for b in range(B)], dim=0).to(device)
        query = torch.cat([per_episode[b][0][p][2] for b in range(B)], dim=0).to(device)
        members = per_episode[0][0][p][0].shape[0]
        raw = _exaone_grad_forward(bb, support, label, query)
        pass_outputs.append(raw.float().reshape(B, members, query.shape[1], -1))

    pooled = torch.cat(pass_outputs, dim=1)
    pooled = torch.sort(pooled, dim=-1).values.mean(dim=1)  # (B, n_query, native_Q)

    center = torch.tensor([per_episode[b][1] for b in range(B)], device=pooled.device).view(B, 1, 1)
    scale = torch.tensor([per_episode[b][2] for b in range(B)], device=pooled.device).view(B, 1, 1)
    bank = pooled * scale + center

    if probs is None:
        return bank  # already the native 999-level grid

    # Interpolate EXAONE's native grid onto probs differentiably.
    native_n = bank.shape[-1]
    native = torch.linspace(
        1.0 / (native_n + 1),
        native_n / (native_n + 1),
        native_n,
        device=bank.device,
        dtype=bank.dtype,
    )
    return _interp_last_dim(bank, native, torch.as_tensor(probs, device=bank.device, dtype=bank.dtype))


def _interp_last_dim(values: torch.Tensor, xp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Differentiable linear interpolation along the last axis (torch equivalent of np.interp)."""
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


def load_backbone(name: str, *, ckpt: Optional[str] = None, device: str = "cuda") -> MarginalBackbone:
    """Build a fine-tunable backbone via eval/spatial/marginal_backends.make_regressor, optionally loading a Phase-A checkpoint."""
    if name not in BACKBONE_NAMES:
        raise ValueError(f"Unknown marginal backbone {name!r}; expected one of {list(BACKBONE_NAMES)}.")

    if name == "tabicl":
        from copula_inter.pit import PRETRAINED_TABICL_CKPT, load_tabicl

        tabicl, config = load_tabicl(ckpt or PRETRAINED_TABICL_CKPT, device, return_config=True)
        return MarginalBackbone(name=name, module=tabicl, handle=None, config=config)

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
            raise ValueError(f"checkpoint {ckpt} was written for backbone {payload.get('backbone')!r}, not {name!r}.")
        module.load_state_dict(payload["state_dict"], strict=True)
    return MarginalBackbone(name=name, module=module, handle=regressor, config={})


def _trainable_module(name: str, regressor) -> nn.Module:
    """The nn.Module inside a regressor whose parameters Phase A trains."""
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
            "src/copula_inter/marginal_backbones.py."
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
    if backbone.name == "tabldm":
        fold_specs = []
        for k in wanted:
            start, end = k * fold_size, min(k * fold_size + fold_size, P)
            if start >= end:
                continue  # empty tail fold when P is not a multiple of fold_size
            qry = np.arange(start, end)
            ctx = np.concatenate([np.arange(0, start), np.arange(end, P)])
            if ctx.size == 0:
                continue
            fold_specs.append((k, ctx, qry, len(qry)))

        # Group folds by query size so equal-sized folds can be forwarded together
        by_size: dict[int, list] = {}
        for spec in fold_specs:
            by_size.setdefault(spec[3], []).append(spec)

        for _qry_len, group in by_size.items():
            ctx_list = [xtr[b][spec[1]] for spec in group for b in range(B)]
            y_ctx_list = [ytr[b][spec[1]] for spec in group for b in range(B)]
            qry_list = [xtr[b][spec[2]] for spec in group for b in range(B)]
            q_fused = backbone.quantile_forward(ctx_list, y_ctx_list, qry_list, probs)
            for i, spec in enumerate(group):
                q_train_parts.append(q_fused[i * B : (i + 1) * B])
                idx_parts.append(torch.as_tensor(spec[2], dtype=torch.long, device=q_test.device))
    else:
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
