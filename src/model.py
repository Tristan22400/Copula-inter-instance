"""
model.py — CopulaTabICL: TabICL as a frozen feature extractor + copula head.

Pattern (ResNet/feature-extractor style):
  1. Load the pretrained TabICL regressor.
  2. STRIP its final quantile decoder by replacing it with ``nn.Identity()``
     — TabICL now emits raw test-instance features of dimension
     ``embed_dim * row_num_cls`` instead of quantile logits.
  3. Add our own ``copula_head : R^{icl_dim} → R^{r+1}`` as a SEPARATE
     module.  Output splits into ``(w_i ∈ R^r, s_i ∈ R)``.

Correlation projection (unconstrained):

    D = diag(softplus(s_i))
    S = W W^T + D
    Σ = Λ^{-1/2} S Λ^{-1/2}  +  jitter·I,    Λ = diag(diag(S))
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


# ---------------------------------------------------------------------------
# Correlation projection
# ---------------------------------------------------------------------------


def low_rank_correlation(
    W: Tensor,
    s: Tensor,
    test_mask: Optional[Tensor] = None,
    jitter: float = 1e-4,
) -> Tensor:
    """Build per-batch correlation matrices Σ from (W, s).

    Args:
        W      : (B, N, r)
        s      : (B, N)        — raw scalars; softplus(s) sits on the diagonal
        test_mask : unused inside; caller slices N_b out of Σ before Cholesky
        jitter : added to the diagonal of Σ for numerical stability

    Returns:
        Sigma : (B, N, N) symmetric PSD, unit diagonal up to ``jitter``.
    """
    B, N, _ = W.shape
    D = F.softplus(s)                                   # (B, N) > 0
    S = torch.matmul(W, W.transpose(-1, -2))            # (B, N, N)
    S = S + torch.diag_embed(D)
    diag = S.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)
    inv_sqrt = diag.rsqrt()
    Sigma = S * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)
    eye = torch.eye(N, device=W.device, dtype=W.dtype).expand(B, N, N)
    return Sigma + jitter * eye


# ---------------------------------------------------------------------------
# CopulaTabICL — feature-extractor + copula head
# ---------------------------------------------------------------------------


class CopulaTabICL(nn.Module):
    """TabICL stripped of its quantile decoder, with a copula head bolted on.

    The TabICL instance is held as ``self.feature_extractor`` and used as a
    black box: calling it returns (B, N_test, icl_dim) — raw features for
    each test instance — because we have replaced its ICL decoder with
    ``nn.Identity()``.

    ``self.copula_head`` then projects to (W, s).
    """

    def __init__(self, base: TabICL, rank: int):
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

        # 4. Our own copula head — completely separate module.
        self.copula_head = nn.Linear(in_features, rank + 1)
        nn.init.normal_(self.copula_head.weight, std=0.02)
        nn.init.zeros_(self.copula_head.bias)

    def forward(self, batch: dict) -> dict:
        """Forward over a padded batch from ``dataset.collate_fn``.

        Returns dict(W=(B, N_max, r), s=(B, N_max)).
        """
        x_train = batch["x_train"]            # (B, P_max, d_x)
        x_test = batch["x_test"]              # (B, N_max, d_x)
        z_train = batch["z_train"]            # (B, P_max) — Z-space context labels

        X = torch.cat([x_train, x_test], dim=1)            # (B, T, d_x)
        # TabICL in training/eval mode returns (B, N_test, out_dim).
        # With decoder replaced by Identity, out_dim == feature_dim.
        features = self.feature_extractor(X, z_train)      # (B, N_max, icl_dim)

        head_out = self.copula_head(features)              # (B, N_max, r+1)
        W = head_out[..., : self.rank]                     # (B, N_max, r)
        s = head_out[..., self.rank]                       # (B, N_max)
        return {"W": W, "s": s}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _load_pretrained_tabicl(ckpt_name: str) -> TabICL:
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(repo_id="jingang/TabICL", filename=ckpt_name)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base = TabICL(**ckpt["config"])
    base.load_state_dict(ckpt["state_dict"])
    return base


def build_copula_transformer(cfg: DictConfig) -> CopulaTabICL:
    """Construct CopulaTabICL with the pretrained TabICL as feature extractor.

    Reads:
        cfg.model.rank
        cfg.tabicl.ckpt
        cfg.model.unfreeze_backbone (optional, default False)
    """
    base = _load_pretrained_tabicl(cfg.tabicl.ckpt)
    model = CopulaTabICL(base=base, rank=int(cfg.model.rank))

    unfreeze = bool(cfg.model.get("unfreeze_backbone", False))
    if not unfreeze:
        for p in model.feature_extractor.parameters():
            p.requires_grad_(False)
    return model
