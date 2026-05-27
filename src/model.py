"""
model.py — Copula Transformer for inter-instance joint distributions.

Architecture (four stages, then copula head):

  Stage 1: Column-wise ISA (Set Transformer, reused from TabICLv2)
  Stage 2: Row-wise aggregation with CLS tokens + RoPE (reused from TabICLv2)
  Stage 3: ICL transformer — all tokens attend only to train tokens
  Stage 4: CopulaHead — test embeddings → W ∈ R^{N×rank} (raw factor, NOT unit-norm)

Correlation matrix parameterization:
  R_ε = eps * I + W @ W^T          (PSD by construction)
  C_ij = R_ε_ij / sqrt(R_ε_ii * R_ε_jj)   (proper correlation matrix, C_ii = 1)

Initializing W ≈ 0 gives C ≈ I (independence), so the model starts at the
trivial copula and learns correlations from there. The loss function
(copula_nll in loss.py) handles the eps-I + WW^T parameterization directly
via Woodbury / determinant lemma.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# TabICLv2 submodule imports
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_TABICL_SRC = os.path.join(_ROOT, "tabicl_upstream", "src")
if _TABICL_SRC not in sys.path:
    sys.path.insert(0, _TABICL_SRC)

from tabicl._model.embedding import ColEmbedding  # type: ignore[import]
from tabicl._model.encoders import Encoder  # type: ignore[import]
from tabicl._model.interaction import RowInteraction  # type: ignore[import]

# ---------------------------------------------------------------------------
# Attention mask helper
# ---------------------------------------------------------------------------


def build_icl_mask(P: int, N: int, device) -> Tensor:
    """Build additive attention mask for the ICL stage.

    All tokens (train and test) attend ONLY to train tokens.
    Train→test and test→test attention are both blocked.

    Returns:
        Float tensor of shape (T, T) where T = P + N.
        0.0 → allowed, -inf → blocked.
    """
    T = P + N
    mask = torch.zeros(T, T, device=device)
    mask[:, P:] = float("-inf")  # block all attention to test positions
    return mask


# ---------------------------------------------------------------------------
# Copula head
# ---------------------------------------------------------------------------


class CopulaHead(nn.Module):
    """Map test token embeddings → raw factor W ∈ R^{N×rank}.

    The correlation matrix is NOT computed here — loss.py builds:
        R_ε = eps * I + W @ W^T
        C_ij = R_ε_ij / sqrt(R_ε_ii * R_ε_jj)   (proper correlation matrix)

    Initializing the last layer near zero means W ≈ 0 at the start of training,
    which gives C ≈ I (independence copula). The model learns to deviate from
    independence as training progresses — no positive-correlation bias.
    """

    def __init__(self, d_ICL: int, rank: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_ICL, 256),
            nn.GELU(),
            nn.Linear(256, rank),
        )
        # Tiny init: W ≈ 0 → C ≈ I (independence) at the start of training.
        nn.init.normal_(self.mlp[-1].weight, std=0.001)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s_test: Tensor) -> Tensor:
        """
        Args:
            s_test : (B, N_max, d_ICL) — test token ICL embeddings

        Returns:
            W : (B, N_max, rank) — raw (non-normalized) factor matrix.
                The caller builds C from eps*I + W @ W^T.
        """
        return self.mlp(s_test)  # (B, N_max, rank)


# ---------------------------------------------------------------------------
# Full Copula Transformer
# ---------------------------------------------------------------------------


class CopulaTransformer(nn.Module):
    """Copula Transformer for inter-instance joint distributions.

    Takes a batch of GP tasks with normalised features (X_norm) and
    latent z-scores (Z) from the PIT stage and predicts W_tilde, the
    low-rank factor of the inter-instance correlation matrix R.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        L_col: int = 3,
        L_row: int = 3,
        L_ICL: int = 12,
        n_inducing: int = 128,
        n_cls: int = 4,
        rank: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        d_ff = d_model * 2
        d_ICL = d_model * n_cls  # 128 * 4 = 512

        # Stage 1: column-wise ISA (Set Transformer) — from TabICLv2
        self.col_embedder = ColEmbedding(
            embed_dim=d_model,
            num_blocks=L_col,
            nhead=n_heads,
            dim_feedforward=d_ff,
            num_inds=n_inducing,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            affine=False,  # direct set-transformer output
            feature_group="same",  # circular feature grouping (d=1: identity)
            feature_group_size=3,
            target_aware=True,  # inject z-scores into train rows
            max_classes=0,  # regression mode → Linear(1, d_model) TAE
            reserve_cls_tokens=n_cls,  # reserve n_cls slots for CLS tokens
            ssmax="qassmax-mlp-elementwise",
        )

        # Stage 2: row-wise interaction with CLS tokens + RoPE — from TabICLv2
        self.row_interactor = RowInteraction(
            embed_dim=d_model,
            num_blocks=L_row,
            nhead=n_heads,
            dim_feedforward=d_ff,
            num_cls=n_cls,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
        )

        # Stage 3: ICL transformer — shared Encoder with train-only attention
        self.proj_row = nn.Linear(d_ICL, d_ICL)  # Linear_h: row repr → ICL dim
        self.proj_z = nn.Linear(1, d_ICL)  # Linear_z: z-score → ICL dim
        self.e_train = nn.Parameter(torch.zeros(d_ICL))  # learned train-type embedding
        self.e_test = nn.Parameter(torch.zeros(d_ICL))  # learned test-type embedding

        self.icl_encoder = Encoder(
            num_blocks=L_ICL,
            d_model=d_ICL,
            nhead=n_heads,
            dim_feedforward=d_ICL * 2,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            ssmax="qassmax-mlp-elementwise",
        )

        # Stage 4: copula head
        self.copula_head = CopulaHead(d_ICL=d_ICL, rank=rank)

        self._d_ICL = d_ICL

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        """Forward pass.

        Args:
            batch: dict with keys
                x_train    : (B, P_max, d_x)
                z_train    : (B, P_max)        — latent z-scores, 0 for padding
                x_test     : (B, N_max, d_x)
                train_mask : BoolTensor (B, P_max)
                test_mask  : BoolTensor (B, N_max)

        Returns:
            W : (B, N_max, rank) — raw factor (NOT unit-norm).
                loss.py builds R_ε = eps*I + W @ W^T and normalizes to C.
        """
        x_train = batch["x_train"]  # (B, P_max, d_x)
        z_train = batch["z_train"]  # (B, P_max)
        x_test = batch["x_test"]  # (B, N_max, d_x)
        train_mask = batch["train_mask"]  # (B, P_max)
        B = x_train.shape[0]
        P_max = x_train.shape[1]
        N_max = x_test.shape[1]

        # Concatenate train + test features so both get column embeddings in one pass.
        # embed_with_test=False (explicit): ISAB inducing points attend only to the
        # first P_max rows (train positions), so test features do not contribute to
        # the column statistics.  Test tokens receive embeddings shaped by those
        # train-derived statistics via Stage-2 of ISAB — which is the desired
        # transductive behaviour: test features are interpreted in the context of
        # the training column distribution, without influencing that distribution.
        X = torch.cat([x_train, x_test], dim=1)  # (B, T, d_x)

        # ---- Stage 1: column-wise ISA ----
        col_emb = self.col_embedder(
            X, y_train=z_train, embed_with_test=False
        )  # (B, T, G+C, d_model)

        # ---- Stage 2: row-wise interaction ----
        row_repr = self.row_interactor(col_emb)  # (B, T, d_ICL)

        # ---- Stage 3: ICL with z-score injection and train-only attention ----
        s = self.proj_row(row_repr)  # (B, T, d_ICL)

        # Inject z-scores into train token embeddings (masked for padding)
        z_embed = self.proj_z(z_train.unsqueeze(-1))  # (B, P_max, d_ICL)
        z_embed = z_embed * train_mask.unsqueeze(-1).float()  # zero out padding
        s[:, :P_max] = s[:, :P_max] + z_embed

        # Add row-type embeddings
        s[:, :P_max] = s[:, :P_max] + self.e_train
        s[:, P_max:] = s[:, P_max:] + self.e_test

        # Attention mask: all tokens attend only to train positions
        attn_mask = build_icl_mask(P_max, N_max, device=x_train.device)  # (T, T)

        # Run ICL blocks manually to pass attn_mask (Encoder.forward doesn't expose it)
        for block in self.icl_encoder.blocks:
            s = block(q=s, attn_mask=attn_mask)

        # ---- Stage 4: copula head on test tokens only ----
        s_test = s[:, P_max:]  # (B, N_max, d_ICL)
        W = self.copula_head(s_test)  # (B, N_max, rank)

        return W


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_copula_transformer(cfg) -> CopulaTransformer:
    """Build a CopulaTransformer from a Hydra config."""
    m = cfg.model
    return CopulaTransformer(
        d_model=m.d_model,
        n_heads=m.n_heads,
        L_col=m.L_col,
        L_row=m.L_row,
        L_ICL=m.L_ICL,
        n_inducing=m.n_inducing,
        n_cls=m.n_cls,
        rank=m.rank,
        dropout=m.dropout,
    )
