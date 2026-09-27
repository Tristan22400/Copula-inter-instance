"""CopulaTabICL: a tabular ICL backbone used as a feature extractor, plus a copula head.

The backbone (cfg.model.backbone: "tabicl" or "tabldm") is built by
copula_backbones.py with its quantile decoder replaced by nn.Identity, so it
emits per-test-row features of size feature_dim. copula_head maps each row to
(w_i in R^r, s_i), and the default "covnorm" parametrization builds

    S = W W^T + diag(softplus(s))
    R = L^{-1/2} S L^{-1/2},               L = diag(diag(S))
    Sigma = G^{-1/2} (R + jitter I) G^{-1/2},  G = diag(diag(R + jitter I))

The other parametrizations ("cossim", "tanhnorm", "sparse_covnorm") live in
correlation_factory.py.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from copula_inter import copula_backbones
from copula_inter.backend_registry import COPULA_BACKBONES
from copula_inter.correlation_factory import (
    LowRankCorrelationFactor,
    cossim_correlation,
    sparse_covnorm_correlation,
    tanhnorm_correlation,
)

# Parametrizations whose copula_head emits W only (no s column).
_NO_SCALAR_COLUMN = {"tanhnorm"}


def _renormalize_to_unit_diagonal(M: Tensor) -> Tensor:
    """Rescale a symmetric PSD matrix to unit diagonal: M -> L^{-1/2} M L^{-1/2}, L = diag(diag(M))."""
    diag = M.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)
    inv_sqrt = diag.rsqrt()
    return M * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)


def _require_s(s: Optional[Tensor], parametrization: str) -> Tensor:
    if s is None:
        raise ValueError(f"parametrization={parametrization!r} requires `s`")
    return s


def low_rank_correlation(
    W: Tensor,
    s: Optional[Tensor] = None,
    test_mask: Optional[Tensor] = None,
    jitter: float = 1e-4,
    parametrization: str = "covnorm",
    lam: Optional[Tensor] = None,
) -> Tensor:
    """Build per-batch correlation matrices Sigma from raw copula-head outputs.

    Args:
        W: (B, N, r) low-rank factor.
        s: (B, N) per-row scalar: softplus(s) diagonal variance for
            "covnorm"/"sparse_covnorm", a sigmoid gate for "cossim", unused for
            "tanhnorm".
        test_mask: unused.
        jitter: added to the diagonal, then renormalized back to unit diagonal.
        parametrization: "covnorm", "cossim", "tanhnorm" or "sparse_covnorm".
        lam: (B,) or (1,) raw soft threshold, required for "sparse_covnorm".

    Returns:
        (B, N, N) symmetric positive-definite Sigma with unit diagonal.
    """
    B, N, _ = W.shape
    eye = torch.eye(N, device=W.device, dtype=W.dtype).expand(B, N, N)

    if parametrization == "covnorm":
        D = F.softplus(_require_s(s, parametrization))  # (B, N) > 0
        S = torch.matmul(W, W.transpose(-1, -2))  # (B, N, N)
        S = S + torch.diag_embed(D)
        diag = S.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)
        inv_sqrt = diag.rsqrt()
        Sigma = S * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)
        return _renormalize_to_unit_diagonal(Sigma + jitter * eye)
    elif parametrization == "cossim":
        factor = cossim_correlation(W, _require_s(s, parametrization))
    elif parametrization == "tanhnorm":
        factor = tanhnorm_correlation(W)
    elif parametrization == "sparse_covnorm":
        if lam is None:
            raise ValueError("parametrization='sparse_covnorm' requires `lam`")
        factor = sparse_covnorm_correlation(W, _require_s(s, parametrization), lam)
    else:
        raise ValueError(f"unknown correlation parametrization: {parametrization!r}")

    return _renormalize_to_unit_diagonal(factor.dense() + jitter * eye)


def low_rank_correlation_factor(
    W: Tensor,
    s: Optional[Tensor] = None,
    jitter: float = 1e-4,
    parametrization: str = "covnorm",
    lam: Optional[Tensor] = None,
) -> LowRankCorrelationFactor:
    """The same Sigma as low_rank_correlation, as a low-rank-plus-diagonal factor.

    With R = U U^T + diag(D) and G = ||U_i||^2 + D_i + jitter,
    Sigma = U' U'^T + diag(D') where U' = G^{-1/2} U and D' = (D + jitter) / G.
    The result's .dense() equals low_rank_correlation(...) up to rounding.
    """
    if parametrization == "covnorm":
        # Same diagonal as low_rank_correlation's covnorm branch (softplus(s), clamp 1e-12).
        D_raw = F.softplus(_require_s(s, parametrization))
        c = ((W * W).sum(-1) + D_raw).clamp_min(1e-12)
        U = W * c.rsqrt().unsqueeze(-1)
        D = D_raw / c
    else:
        if parametrization == "cossim":
            factor = cossim_correlation(W, _require_s(s, parametrization))
        elif parametrization == "tanhnorm":
            factor = tanhnorm_correlation(W)
        elif parametrization == "sparse_covnorm":
            if lam is None:
                raise ValueError("parametrization='sparse_covnorm' requires `lam`")
            factor = sparse_covnorm_correlation(W, _require_s(s, parametrization), lam)
        else:
            raise ValueError(f"unknown correlation parametrization: {parametrization!r}")
        U, D = factor.U, factor.D

    D = D + jitter
    gamma = ((U * U).sum(-1) + D).clamp_min(1e-12)
    return LowRankCorrelationFactor(U=U * gamma.rsqrt().unsqueeze(-1), D=D / gamma)


def build_sigma(
    out: dict,
    cfg: DictConfig,
    jitter: float = 1e-4,
    test_mask: Optional[Tensor] = None,
) -> Tensor:
    """Dense Sigma from a CopulaTabICL forward-pass dict, using cfg.model.correlation_parametrization."""
    parametrization = cfg.model.get("correlation_parametrization", "covnorm")
    return low_rank_correlation(
        out["W"],
        out.get("s"),
        test_mask=test_mask,
        jitter=jitter,
        parametrization=parametrization,
        lam=out.get("lam"),
    )


class CopulaTabICL(nn.Module):
    """Tabular ICL backbone without its quantile decoder, plus a copula head.

    feature_extractor maps a batch to (B, N_test, feature_dim) test-row features.
    copula_head projects them to (W, s), or to W alone for "tanhnorm".
    "sparse_covnorm" also learns one global soft threshold, sparse_lambda_raw.
    """

    def __init__(
        self,
        base: nn.Module,
        rank: int,
        correlation_parametrization: str = "covnorm",
        backbone_name: str = "tabicl",
    ) -> None:
        super().__init__()
        # Discover the feature dimension, then replace the quantile decoder with Identity.
        in_features = copula_backbones.strip_decoder(base)

        # 2. Save the (now feature-only) backbone.
        self.feature_extractor = base
        self.backbone_name = backbone_name
        self.rank = rank
        self.feature_dim = in_features
        self.correlation_parametrization = correlation_parametrization

        # Copula head: r columns for W, plus one for s unless tanhnorm.
        head_out_dim = rank if correlation_parametrization in _NO_SCALAR_COLUMN else rank + 1
        self.copula_head = nn.Linear(in_features, head_out_dim)
        nn.init.normal_(self.copula_head.weight, std=0.02)
        nn.init.zeros_(self.copula_head.bias)

        if correlation_parametrization == "sparse_covnorm":
            # softplus(-6) ~= 0.0025 starts the threshold near-inactive; at 0 the relu has no gradient.
            self.sparse_lambda_raw = nn.Parameter(torch.full((1,), -6.0))

    def forward(self, batch: dict) -> dict:
        """Forward over a padded batch from dataset.collate_fn.

        Returns a dict with W: (B, N_max, r), plus s: (B, N_max) unless the
        parametrization is "tanhnorm", plus lam: (1,) for "sparse_covnorm", plus
        moe_aux_loss when the backbone emits one.
        """
        x_train = batch["x_train"]  # (B, P_max, d_x)
        x_test = batch["x_test"]  # (B, N_max, d_x)
        z_train = batch["z_train"]  # (B, P_max) — Z-space context labels

        X = torch.cat([x_train, x_test], dim=1)  # (B, T, d_x)
        features = self.feature_extractor(X, z_train)  # (B, N_max, feature_dim)
        # Cast backbone features to the head's dtype (TabICL may return float16).
        features = features.to(dtype=self.copula_head.weight.dtype)
        head_out = self.copula_head(features)  # (B, N_max, head_out_dim)
        W = head_out[..., : self.rank]  # (B, N_max, r)

        out = {"W": W}
        if self.correlation_parametrization not in _NO_SCALAR_COLUMN:
            out["s"] = head_out[..., self.rank]  # (B, N_max)
        if self.correlation_parametrization == "sparse_covnorm":
            out["lam"] = self.sparse_lambda_raw

        # Backbone auxiliary loss (TabLDM MoE only).
        aux = copula_backbones.moe_aux_loss(self.backbone_name, self.feature_extractor)
        if aux is not None:
            out["moe_aux_loss"] = aux
        return out


def build_copula_transformer(cfg: DictConfig) -> CopulaTabICL:
    """Construct CopulaTabICL from cfg.

    Reads cfg.model.{backbone, rank, correlation_parametrization,
    unfreeze_backbone}, cfg.tabicl.{pretrained, ckpt, recompute, arch} and
    cfg.lora.{enabled, rank, alpha, target, stages}.
    """
    backbone_name = str(cfg.model.get("backbone", "tabicl"))
    if backbone_name not in COPULA_BACKBONES:
        raise ValueError(f"Unknown cfg.model.backbone={backbone_name!r}; expected one of {list(COPULA_BACKBONES)}.")
    base = copula_backbones.load_raw_backbone(backbone_name, cfg)

    model = CopulaTabICL(
        base=base,
        rank=int(cfg.model.rank),
        correlation_parametrization=str(cfg.model.get("correlation_parametrization", "covnorm")),
        backbone_name=backbone_name,
    )

    lora_cfg = cfg.get("lora", {})
    if bool(lora_cfg.get("enabled", False)):
        from copula_inter.lora import apply_lora

        n = apply_lora(
            backbone=model.feature_extractor,
            rank=int(lora_cfg.get("rank", 8)),
            alpha=float(lora_cfg.get("alpha", 16.0)),
            target=str(lora_cfg.get("target", "qkvo")),
            stages=list(lora_cfg.get("stages", ["icl", "row", "col"])),
        )
        print(
            f"LoRA applied: {n} MultiheadAttention modules replaced "
            f"(rank={lora_cfg.get('rank', 8)}, alpha={lora_cfg.get('alpha', 16.0)}, "
            f"target={lora_cfg.get('target', 'qkvo')}, stages={list(lora_cfg.get('stages', ['icl', 'row', 'col']))})"
        )
        # copula_head is always trainable; backbone LoRA params set by apply_lora
        return model

    unfreeze = bool(cfg.model.get("unfreeze_backbone", True))
    if not unfreeze:
        for p in model.feature_extractor.parameters():
            p.requires_grad_(False)
    return model
