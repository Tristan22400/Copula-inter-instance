"""
model.py — CopulaTabICL: TabICL as a frozen feature extractor + copula head.

Pattern (ResNet/feature-extractor style):
  1. Load the pretrained TabICL regressor.
  2. STRIP its final quantile decoder by replacing it with ``nn.Identity()``
     — TabICL now emits raw test-instance features of dimension
     ``embed_dim * row_num_cls`` instead of quantile logits.
  3. Add our own ``copula_head : R^{icl_dim} → R^{r+1}`` as a SEPARATE
     module.  Output splits into ``(w_i ∈ R^r, s_i ∈ R)``.

Correlation projection (unconstrained), default "covnorm" parametrization:

    D = diag(softplus(s_i))
    S = W W^T + D
    Σ = Λ^{-1/2} S Λ^{-1/2}  +  jitter·I,    Λ = diag(diag(S))

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

from tabicl._model.tabicl import TabICL  # type: ignore[import]

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
                 uniformly after building Σ regardless of parametrization
        parametrization : one of "covnorm" (default — original behaviour,
                 byte-identical to the pre-existing implementation),
                 "cossim", "tanhnorm", "sparse_covnorm". See
                 correlation_factory.py for the exact math of each.
        lam    : (B,) or (1,) raw threshold, required only for
                 "sparse_covnorm" (see correlation_factory.sparse_covnorm_correlation)

    Returns:
        Sigma : (B, N, N) symmetric PD, unit diagonal up to ``jitter``.
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
        return Sigma + jitter * eye
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

    return factor.dense() + jitter * eye


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
    """TabICL stripped of its quantile decoder, with a copula head bolted on.

    The TabICL instance is held as ``self.feature_extractor`` and used as a
    black box: calling it returns (B, N_test, icl_dim) — raw features for
    each test instance — because we have replaced its ICL decoder with
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
        base: TabICL,
        rank: int,
        correlation_parametrization: str = "covnorm",
    ):
        super().__init__()
        # 1. Discover the feature dimension before stripping the decoder.
        decoder = base.icl_predictor.decoder
        first_linear = decoder[0]  # nn.Sequential(Linear, GELU, Linear)
        in_features = first_linear.in_features  # == embed_dim * row_num_cls

        # 2. Strip the final quantile decoder — feature-extractor pattern.
        base.icl_predictor.decoder = nn.Identity()

        # 3. Save the (now feature-only) backbone.
        self.feature_extractor = base
        self.rank = rank
        self.feature_dim = in_features
        self.correlation_parametrization = correlation_parametrization

        # 4. Our own copula head — completely separate module. Output width
        #    varies per parametrization: tanhnorm needs only the r-dim raw
        #    factor, the others also need one trailing scalar column.
        head_out_dim = rank if correlation_parametrization in _NO_SCALAR_COLUMN else rank + 1
        self.copula_head = nn.Sequential(nn.Linear(in_features, in_features * 2), nn.GELU(), nn.Linear(in_features * 2, head_out_dim))
        # Only the OUTPUT layer needs a small init — that's what keeps the raw
        # W/s logits (and hence Sigma) near-identity at step 0 for numerical
        # safety. The hidden layer is a feature transform, not an output; it
        # keeps PyTorch's default Kaiming-uniform init (fan_in-scaled, ~0.088
        # here) so it doesn't attenuate. Previously both layers were forced to
        # std=0.02 (to "match" the output layer) — that shrinks the hidden
        # layer's output ~5x and the gradient reaching it ~5x (~57x at the
        # output layer, compounding through GELU), which was starving the
        # copula head of a usable gradient signal from step 0 and made the
        # correlation term take forever to move off its near-independence init.
        nn.init.zeros_(self.copula_head[0].bias)
        nn.init.normal_(self.copula_head[-1].weight, std=0.02)
        nn.init.zeros_(self.copula_head[-1].bias)

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
        # TabICL in training/eval mode returns (B, N_test, out_dim).
        # With decoder replaced by Identity, out_dim == feature_dim.
        features = self.feature_extractor(X, z_train)      # (B, N_max, icl_dim)

        head_out = self.copula_head(features)              # (B, N_max, head_out_dim)
        W = head_out[..., : self.rank]                      # (B, N_max, r)

        out = {"W": W}
        if self.correlation_parametrization not in _NO_SCALAR_COLUMN:
            out["s"] = head_out[..., self.rank]              # (B, N_max)
        if self.correlation_parametrization == "sparse_covnorm":
            out["lam"] = self.sparse_lambda_raw
        return out


# ---------------------------------------------------------------------------
# Factory
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


def build_copula_transformer(cfg: DictConfig) -> CopulaTabICL:
    """Construct CopulaTabICL with either a pretrained or scratch TabICL backbone.

    Reads:
        cfg.model.rank
        cfg.model.correlation_parametrization
                                        (default "covnorm"; one of "covnorm",
                                         "cossim", "tanhnorm", "sparse_covnorm"
                                         — see correlation_factory.py)
        cfg.tabicl.pretrained          (default True)
        cfg.tabicl.ckpt                (only when pretrained=True)
        cfg.tabicl.recompute           (default False; gradient checkpointing
                                         through the TabICL backbone — trades
                                         ~20-30% extra compute for a large cut
                                         in peak activation memory, useful when
                                         large N_max/P_max push attention
                                         length T=P+N close to the VRAM ceiling)
        cfg.tabicl.arch.*              (only when pretrained=False)
        cfg.model.unfreeze_backbone    (default True)
        cfg.lora.enabled               (default False)
        cfg.lora.rank                  (default 8)
        cfg.lora.alpha                 (default 16.0)
        cfg.lora.target                (default "qkvo")
        cfg.lora.stages                (default ["icl", "row", "col"])
    """
    pretrained = bool(cfg.tabicl.get("pretrained", True))
    recompute = bool(cfg.tabicl.get("recompute", False))
    if pretrained:
        base = _load_pretrained_tabicl(cfg.tabicl.ckpt, recompute=recompute)
    else:
        base = _build_tabicl_scratch(cfg)

    model = CopulaTabICL(
        base=base,
        rank=int(cfg.model.rank),
        correlation_parametrization=str(cfg.model.get("correlation_parametrization", "covnorm")),
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
