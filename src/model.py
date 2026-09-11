"""
model.py — CopulaTabICL: a tabular ICL backbone as a frozen/finetuned
feature extractor + copula head.

Despite the class name (kept for state-dict/call-site stability — see
copula_backbones.py's docstring for why only two backbones qualify), the
backbone is a CHOICE: ``cfg.model.backbone`` selects "tabicl" (default) or
"tabldm" (Xiaomi-TabLDM), dispatched through src/copula_backbones.py. All
architecture-specific construction (pretrained/scratch loading, decoder
discovery+stripping, an optional MoE auxiliary loss) lives there; this
module only holds the feature-extractor pattern and the copula head itself,
both backbone-agnostic.

Pattern (ResNet/feature-extractor style):
  1. Load the pretrained backbone (TabICL or TabLDM).
  2. STRIP its final quantile decoder by replacing it with ``nn.Identity()``
     — the backbone now emits raw test-instance features of dimension
     ``feature_dim`` (== embed_dim * row_num_cls for both backbones)
     instead of quantile logits.
  3. Add our own ``copula_head : R^{feature_dim} → R^{r+1}`` as a SEPARATE
     module.  Output splits into ``(w_i ∈ R^r, s_i ∈ R)``.

Correlation projection (unconstrained), default "covnorm" parametrization:

    D = diag(softplus(s_i))
    S = W W^T + D
    R = Λ^{-1/2} S Λ^{-1/2},                 Λ = diag(diag(S))
    Σ = Γ^{-1/2} (R + jitter·I) Γ^{-1/2},    Γ = diag(diag(R + jitter·I))

(the second congruence transform renormalizes the jittered matrix back to a
unit diagonal -- see model._renormalize_to_unit_diagonal)

Three alternative parametrizations ("cossim", "tanhnorm", "sparse_covnorm",
selected via cfg.model.correlation_parametrization) live in
correlation_factory.py and are dispatched through low_rank_correlation()/
build_sigma() below.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_TABICL_SRC = os.path.join(_REPO_ROOT, "tabicl_upstream", "src")
if _TABICL_SRC not in sys.path:
    sys.path.insert(0, _TABICL_SRC)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import copula_backbones
from correlation_factory import (
    cossim_correlation,
    sparse_covnorm_correlation,
    tanhnorm_correlation,
)

# Parametrizations whose copula_head output has no trailing scalar column
# (W only, no s) — see CopulaTabICL.__init__.
_NO_SCALAR_COLUMN = {"tanhnorm"}


# ---------------------------------------------------------------------------
# Correlation projection
# ---------------------------------------------------------------------------


def _renormalize_to_unit_diagonal(M: Tensor) -> Tensor:
    """Rescale a symmetric PSD matrix so its diagonal is exactly 1.

    Every parametrization below already builds a unit-diagonal R before
    ``jitter`` is added, but ``R + jitter*I`` shifts the diagonal to
    ``1 + jitter`` while leaving the off-diagonal entries untouched -- so the
    jittered matrix is no longer a valid correlation matrix (R_ii != 1). This
    reapplies the same Λ^{-1/2} (·) Λ^{-1/2} congruence transform used to
    build R in the first place, using the post-jitter diagonal Λ. Off- and
    on-diagonal entries shrink by the same factor (exactly ``1/(1+jitter)``
    when the pre-jitter diagonal was exactly 1), which keeps M PSD -- a
    congruence transform by a positive diagonal matrix preserves
    positive-semidefiniteness.
    """
    diag = M.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)
    inv_sqrt = diag.rsqrt()
    return M * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)


def low_rank_correlation(
    W: Tensor,
    s: Optional[Tensor] = None,
    test_mask: Optional[Tensor] = None,
    jitter: float = 1e-4,
    parametrization: str = "covnorm",
    lam: Optional[Tensor] = None,
) -> Tensor:
    """Build per-batch correlation matrices Σ from raw copula-head outputs.

    Args:
        W      : (B, N, r)
        s      : (B, N) raw scalars — meaning depends on ``parametrization``:
                 softplus(s) diagonal variance for "covnorm"/"sparse_covnorm",
                 a sigmoid gate for "cossim", unused for "tanhnorm".
        test_mask : unused inside; caller slices N_b out of Σ before Cholesky
        jitter : added to the diagonal of Σ for numerical stability, applied
                 uniformly after building Σ regardless of parametrization,
                 then renormalized back to a unit diagonal (see Returns)
        parametrization : one of "covnorm" (default — original behaviour,
                 byte-identical to the pre-existing implementation),
                 "cossim", "tanhnorm", "sparse_covnorm". See
                 correlation_factory.py for the exact math of each.
        lam    : (B,) or (1,) raw threshold, required only for
                 "sparse_covnorm" (see correlation_factory.sparse_covnorm_correlation)

    Returns:
        Sigma : (B, N, N) symmetric PD, unit diagonal EXACTLY (jitter is
                folded in and then renormalized back out via
                ``_renormalize_to_unit_diagonal`` -- see that function's
                docstring for why the naive ``R + jitter*I`` is not itself a
                valid correlation matrix).
    """
    B, N, _ = W.shape
    eye = torch.eye(N, device=W.device, dtype=W.dtype).expand(B, N, N)

    if parametrization == "covnorm":
        D = F.softplus(s)                                   # (B, N) > 0
        S = torch.matmul(W, W.transpose(-1, -2))            # (B, N, N)
        S = S + torch.diag_embed(D)
        diag = S.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)
        inv_sqrt = diag.rsqrt()
        Sigma = S * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)
        return _renormalize_to_unit_diagonal(Sigma + jitter * eye)
    elif parametrization == "cossim":
        factor = cossim_correlation(W, s)
    elif parametrization == "tanhnorm":
        factor = tanhnorm_correlation(W)
    elif parametrization == "sparse_covnorm":
        if lam is None:
            raise ValueError("parametrization='sparse_covnorm' requires `lam`")
        factor = sparse_covnorm_correlation(W, s, lam)
    else:
        raise ValueError(f"unknown correlation parametrization: {parametrization!r}")

    return _renormalize_to_unit_diagonal(factor.dense() + jitter * eye)


def build_sigma(
    out: dict,
    cfg: DictConfig,
    jitter: float = 1e-4,
    test_mask: Optional[Tensor] = None,
) -> Tensor:
    """Dense Σ from a CopulaTabICL forward-pass dict, dispatched by
    ``cfg.model.correlation_parametrization``.

    Single choke point so call sites don't need to know per-parametrization
    argument differences (tanhnorm's ``out`` has no "s" key; sparse_covnorm's
    has an extra "lam" key) — they just call ``build_sigma(out, cfg, ...)``
    instead of ``low_rank_correlation(out["W"], out["s"], ...)`` directly.
    """
    parametrization = cfg.model.get("correlation_parametrization", "covnorm")
    return low_rank_correlation(
        out["W"],
        out.get("s"),
        test_mask=test_mask,
        jitter=jitter,
        parametrization=parametrization,
        lam=out.get("lam"),
    )


# ---------------------------------------------------------------------------
# CopulaTabICL — feature-extractor + copula head
# ---------------------------------------------------------------------------


class CopulaTabICL(nn.Module):
    """A tabular ICL backbone (TabICL or TabLDM) stripped of its quantile
    decoder, with a copula head bolted on. Class name kept for state-dict/
    call-site stability — see model.py's module docstring; the actual
    backbone is a choice, resolved by copula_backbones.py.

    The backbone instance is held as ``self.feature_extractor`` and used as
    a black box: calling it returns (B, N_test, feature_dim) — raw features
    for each test instance — because we have replaced its ICL decoder with
    ``nn.Identity()``.

    ``self.copula_head`` then projects to (W, s) — or just W for
    "tanhnorm", which needs no extra scalar column (see
    correlation_factory.py). "sparse_covnorm" additionally carries a single
    learned soft-threshold shared across the batch (``self.sparse_lambda_raw``)
    — the spec's λ ∈ R^{B×1} is a global learned scalar, not data-conditional,
    so it lives on the module rather than as an extra head output.
    """

    def __init__(
        self,
        base: nn.Module,
        rank: int,
        correlation_parametrization: str = "covnorm",
        backbone_name: str = "tabicl",
    ):
        super().__init__()
        # 1. Discover the feature dimension, then strip the final quantile
        #    decoder — feature-extractor pattern, shared by both backbones
        #    (see copula_backbones.strip_decoder's docstring).
        in_features = copula_backbones.strip_decoder(base)

        # 2. Save the (now feature-only) backbone.
        self.feature_extractor = base
        self.backbone_name = backbone_name
        self.rank = rank
        self.feature_dim = in_features
        self.correlation_parametrization = correlation_parametrization

        # 3. Our own copula head — completely separate module. Output width
        #    varies per parametrization: tanhnorm needs only the r-dim raw
        #    factor, the others also need one trailing scalar column.
        head_out_dim = rank if correlation_parametrization in _NO_SCALAR_COLUMN else rank + 1
        self.copula_head = nn.Linear(in_features, head_out_dim)
        nn.init.normal_(self.copula_head.weight, std=0.02)
        nn.init.zeros_(self.copula_head.bias)

        if correlation_parametrization == "sparse_covnorm":
            # softplus(-6) ~= 0.0025, far below copula_head's ~0.02-std initial
            # W scale, so the soft-threshold starts near-inactive (W_tilde ~= W,
            # matching CovNorm's warm start) instead of zeroing every entry out
            # from step 0. Initializing at 0 (softplus(0) ~= 0.69) is a dead
            # unit: relu's zero-gradient region blocks all gradient to both W
            # and lambda simultaneously, so the threshold could never learn to
            # shrink back down.
            self.sparse_lambda_raw = nn.Parameter(torch.full((1,), -6.0))

    def forward(self, batch: dict) -> dict:
        """Forward over a padded batch from ``dataset.collate_fn``.

        Returns dict(W=(B, N_max, r)), plus "s"=(B, N_max) unless
        correlation_parametrization=="tanhnorm", plus "lam"=(1,) iff
        correlation_parametrization=="sparse_covnorm".
        """
        x_train = batch["x_train"]            # (B, P_max, d_x)
        x_test = batch["x_test"]              # (B, N_max, d_x)
        z_train = batch["z_train"]            # (B, P_max) — Z-space context labels

        X = torch.cat([x_train, x_test], dim=1)            # (B, T, d_x)
        # Backbone in training/eval mode returns (B, N_test, out_dim). With
        # decoder replaced by Identity, out_dim == feature_dim.
        features = self.feature_extractor(X, z_train)      # (B, N_max, feature_dim)

        head_out = self.copula_head(features)              # (B, N_max, head_out_dim)
        W = head_out[..., : self.rank]                      # (B, N_max, r)

        out = {"W": W}
        if self.correlation_parametrization not in _NO_SCALAR_COLUMN:
            out["s"] = head_out[..., self.rank]              # (B, N_max)
        if self.correlation_parametrization == "sparse_covnorm":
            out["lam"] = self.sparse_lambda_raw

        # Backbone's own auxiliary loss (MoE z-loss + load-balance, TabLDM
        # only — see copula_backbones.moe_aux_loss). Surfaced here, the
        # single choke point over this backbone's forward, rather than
        # requiring train.py to know which backbone is loaded.
        aux = copula_backbones.moe_aux_loss(self.backbone_name, self.feature_extractor)
        if aux is not None:
            out["moe_aux_loss"] = aux
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_copula_transformer(cfg: DictConfig) -> CopulaTabICL:
    """Construct CopulaTabICL with the selected backbone.

    Reads:
        cfg.model.backbone             (default "tabicl"; one of
                                         copula_backbones.BACKBONE_NAMES —
                                         "tabicl" | "tabldm". See
                                         src/copula_backbones.py for the
                                         per-architecture construction this
                                         dispatches to.)
        cfg.model.rank
        cfg.model.correlation_parametrization
                                        (default "covnorm"; one of "covnorm",
                                         "cossim", "tanhnorm", "sparse_covnorm"
                                         — see correlation_factory.py)
        cfg.tabicl.pretrained          (default True; tabldm has no
                                         from-scratch path and raises if
                                         this is False)
        cfg.tabicl.ckpt                (tabicl only, only when pretrained=True)
        cfg.tabicl.recompute           (default False; gradient checkpointing
                                         through the backbone — trades
                                         ~20-30% extra compute for a large cut
                                         in peak activation memory, useful when
                                         large N_max/P_max push attention
                                         length T=P+N close to the VRAM
                                         ceiling. For tabldm this is an
                                         escalate-only override — see
                                         copula_backbones._load_tabldm.)
        cfg.tabicl.arch.*              (tabicl only, only when pretrained=False)
        cfg.model.unfreeze_backbone    (default True)
        cfg.lora.enabled               (default False)
        cfg.lora.rank                  (default 8)
        cfg.lora.alpha                 (default 16.0)
        cfg.lora.target                (default "qkvo")
        cfg.lora.stages                (default ["icl", "row", "col"])
    """
    backbone_name = str(cfg.model.get("backbone", "tabicl"))
    if backbone_name not in copula_backbones.BACKBONE_NAMES:
        raise ValueError(
            f"Unknown cfg.model.backbone={backbone_name!r}; expected one of "
            f"{list(copula_backbones.BACKBONE_NAMES)}."
        )
    base = copula_backbones.load_raw_backbone(backbone_name, cfg)

    model = CopulaTabICL(
        base=base,
        rank=int(cfg.model.rank),
        correlation_parametrization=str(cfg.model.get("correlation_parametrization", "covnorm")),
        backbone_name=backbone_name,
    )

    lora_cfg = cfg.get("lora", {})
    if bool(lora_cfg.get("enabled", False)):
        from lora import apply_lora  # type: ignore[import]
        n = apply_lora(
            backbone=model.feature_extractor,
            rank=int(lora_cfg.get("rank", 8)),
            alpha=float(lora_cfg.get("alpha", 16.0)),
            target=str(lora_cfg.get("target", "qkvo")),
            stages=list(lora_cfg.get("stages", ["icl", "row", "col"])),
        )
        print(f"LoRA applied: {n} MultiheadAttention modules replaced "
              f"(rank={lora_cfg.get('rank', 8)}, alpha={lora_cfg.get('alpha', 16.0)}, "
              f"target={lora_cfg.get('target', 'qkvo')}, stages={list(lora_cfg.get('stages', ['icl', 'row', 'col']))})")
        # copula_head is always trainable; backbone LoRA params set by apply_lora
        return model

    unfreeze = bool(cfg.model.get("unfreeze_backbone", True))
    if not unfreeze:
        for p in model.feature_extractor.parameters():
            p.requires_grad_(False)
    return model
