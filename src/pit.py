"""
pit.py — Probability Integral Transform via the frozen TabICL marginal CDF.

For each target dimension j, TabICL's quantile head provides a conditional
predictive distribution over y given (x, context).  Evaluating that
distribution's CDF at the true target maps observations to Uniform(0, 1):

    u_{i,j} = F̂_j(y_{i,j} | x_i, context)

and a probit transform sends them to standard normal Z-space:

    z_{i,j} = Φ⁻¹(u_{i,j}).

Leakage prevention for the training instances is done by **K-fold
partitioning**: the train set is split into K disjoint folds, and for each
fold the held-out points are queried against TabICL using the remaining
K−1 folds as context.  K is small and fixed (default 10) — true LOO
(K = P) is much more accurate but ~K_loo / K_default times slower at
dataset-generation time.  The test instances use the entire training set
as context (single forward pass).

This file makes **no modifications** to ``tabicl_upstream`` — leakage is
handled purely by which points are passed in which forward call.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_TABICL_SRC = os.path.join(_REPO_ROOT, "tabicl_upstream", "src")
if _TABICL_SRC not in sys.path:
    sys.path.insert(0, _TABICL_SRC)


DEFAULT_K_FOLDS = 10


# ---------------------------------------------------------------------------
# TabICL loader
# ---------------------------------------------------------------------------


def load_tabicl(ckpt_name: str, device: str) -> nn.Module:
    """Download (if needed) and load a frozen TabICL regressor."""
    from huggingface_hub import hf_hub_download
    from tabicl._model.tabicl import TabICL  # type: ignore[import]

    ckpt_path = hf_hub_download(repo_id="jingang/TabICL", filename=ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    base = TabICL(**checkpoint["config"])
    base.load_state_dict(checkpoint["state_dict"])
    for p in base.parameters():
        p.requires_grad_(False)
    base.eval()
    base.to(device)
    return base


def _probit(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Clamp u to (eps, 1-eps) then apply Φ⁻¹ via erfinv."""
    u = u.clamp(eps, 1.0 - eps)
    return torch.erfinv(2.0 * u - 1.0) * math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Single-task PIT
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_pit(
    tabicl: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    k_folds: int = DEFAULT_K_FOLDS,
    eps: float = 1e-6,
) -> dict:
    """Run the Probability Integral Transform on one task.

    Args:
        tabicl  : frozen TabICL regressor (max_classes=0)
        X_train : (P, p_x)
        Y_train : (P, d)
        X_test  : (N, p_x)
        Y_test  : (N, d)
        k_folds : number of disjoint folds for the training-set PIT.
                  Bounded above by P; clamp into [2, P].
                  Set to P explicitly for true leave-one-out (slow).
        eps     : clamp before probit.

    Returns dict with:
        z_train      : (P, d)
        z_test       : (N, d)
        log_pdf_test : (N, d)   marginal log-densities at Y_test
    """
    device = X_train.device
    P, p_x = X_train.shape
    N = X_test.shape[0]
    d = Y_train.shape[1]

    K = max(2, min(int(k_folds), P))

    # ------------------------------------------------------------------ #
    # A) Test instances: one forward over the full train context, fused #
    #    across the d target dimensions on the batch axis.                #
    # ------------------------------------------------------------------ #
    X_concat = torch.cat([X_train, X_test], dim=0)                       # (P+N, p_x)
    X_test_batch = X_concat.unsqueeze(0).expand(d, -1, -1).contiguous()  # (d, P+N, p_x)
    y_train_batch = Y_train.permute(1, 0).contiguous()                   # (d, P)

    logits = tabicl(X_test_batch, y_train_batch)                         # (d, N, Q)
    Q = logits.shape[-1]
    dist = tabicl.quantile_dist(logits.reshape(d * N, Q))

    y_test_flat = Y_test.permute(1, 0).reshape(d * N)
    u_test = dist.cdf(y_test_flat).reshape(d, N).permute(1, 0)           # (N, d)
    log_pdf_test = dist.log_prob(y_test_flat).reshape(d, N).permute(1, 0)  # (N, d)

    # ------------------------------------------------------------------ #
    # B) Training instances: K disjoint folds (fixed K, ≪ P)              #
    # ------------------------------------------------------------------ #
    fold_size = math.ceil(P / K)
    u_train = torch.empty(P, d, device=device, dtype=Y_train.dtype)
    indices = torch.arange(P, device=device)

    for k in range(K):
        start = k * fold_size
        end = min(start + fold_size, P)
        if start >= end:
            break

        qry_idx = indices[start:end]
        ctx_mask = torch.ones(P, dtype=torch.bool, device=device)
        ctx_mask[qry_idx] = False
        ctx_idx = indices[ctx_mask]
        F = qry_idx.numel()

        X_fold = torch.cat([X_train[ctx_idx], X_train[qry_idx]], dim=0)    # (P-F+F, p_x)
        X_fold_batch = X_fold.unsqueeze(0).expand(d, -1, -1).contiguous()
        y_ctx_batch = Y_train[ctx_idx].permute(1, 0).contiguous()          # (d, P-F)

        logits_fold = tabicl(X_fold_batch, y_ctx_batch)                    # (d, F, Q)
        dist_fold = tabicl.quantile_dist(logits_fold.reshape(d * F, Q))

        y_qry_flat = Y_train[qry_idx].permute(1, 0).reshape(d * F)
        u_train[qry_idx, :] = (
            dist_fold.cdf(y_qry_flat).reshape(d, F).permute(1, 0)
        )

    z_train = _probit(u_train, eps)
    z_test = _probit(u_test, eps)

    return {
        "z_train": z_train,
        "z_test": z_test,
        "log_pdf_test": log_pdf_test,
    }
