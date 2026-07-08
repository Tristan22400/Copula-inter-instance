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

from data_gen import build_kernel_fn, _safe_cholesky  # noqa: E402

DEFAULT_K_FOLDS = 10


def _optional_param(t: torch.Tensor):
    """Unpack a possibly-ARD-vector task hyperparameter (see data_gen's 0.0
    "not applicable" sentinel convention): None if every entry is the
    sentinel, else a python float (scalar) or the raw tensor (ARD vector)."""
    if torch.all(t == 0.0):
        return None
    return t.item() if t.numel() == 1 else t


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


# ---------------------------------------------------------------------------
# Analytical GP PIT (no model inference required)
# ---------------------------------------------------------------------------


@torch.no_grad()
def gp_analytical_pit(task: dict, eps: float = 1e-6) -> dict:
    """Exact PIT from GP LOO (train) and posterior (test) marginals.

    Since all data is generated from a GP with known hyperparameters, the
    marginal CDFs are available in closed form — no learned regressor needed.

    Test instances:
        y_test[i] | D_train ~ N(mu_star[i], sigma_star[i]²)  (exact)
        z_test[i] = (y_test[i] - mu_star[i]) / sigma_star[i]

    Training instances — exact GP LOO (Rasmussen & Williams, GPML Eq. 5.12):
        sigma²_i^LOO  = 1 / [K_ff⁻¹]_ii
        z_train[i]    = alpha_i / sqrt([K_ff⁻¹]_ii)
        where alpha = K_ff⁻¹ y_train

    Cost: one O(P³) Cholesky per episode vs. O(K × P × forward_pass) for
    the TabICL K-fold approach.

    Args:
        task: raw task dict returned by generate_gp_task (must contain
              kernel, l, alpha2, nugget, period, rq_alpha, l_b, alpha2_b,
              period_b, rq_alpha_b, kernel_feature_indices, x_norm_train,
              y_train, y_test, mu_star, sigma_star).
        eps:  unused (kept for API symmetry with run_pit).

    Returns dict with z_train (P,), z_test (N,), log_pdf_test (N,).
    """
    kernel_name = task["kernel"]
    # scalar, unless the episode was generated ARD (cfg.data.ard=True for
    # rbf/matern32/periodic/rational_quadratic, or always for "hebo"), in
    # which case l is a per-dimension lengthscale vector (k,) — see
    # data_gen._build_scaled_kernel.
    l_tensor = task["l"]
    l      = l_tensor.item() if l_tensor.numel() == 1 else l_tensor
    alpha2 = task["alpha2"].item()
    nugget = task["nugget"].item()
    # 0.0 sentinel means the param is not applicable for this kernel. period
    # is likewise a per-dimension vector under periodic+ARD (gpytorch's
    # PeriodicKernel ties period_length's ard_num_dims to lengthscale's).
    period   = _optional_param(task["period"])
    rq_alpha = task["rq_alpha"].item() if task["rq_alpha"].item() != 0.0 else None
    # Composite ("A+B"/"A*B") kernels' second component — same 0.0 sentinel
    # convention. Omitting these previously made build_kernel_fn silently
    # reconstruct composites with l_b/alpha2_b=None, crashing with a
    # TypeError as soon as component B's kernel function tried to use them.
    # l_b/period_b can be ARD vectors too, same as l/period above, whenever
    # component B is one of the ARD-eligible base kernels under cfg.data.ard.
    l_b        = _optional_param(task["l_b"])
    alpha2_b   = task["alpha2_b"].item() if task["alpha2_b"].item() != 0.0 else None
    period_b   = _optional_param(task["period_b"])
    rq_alpha_b = task["rq_alpha_b"].item() if task["rq_alpha_b"].item() != 0.0 else None

    # active_dims (gpytorch's own kernel kwarg) lets kernel_fn take the
    # full-width x_norm_train straight through and select its k active
    # columns internally — same mechanism data_gen.generate_gp_task uses,
    # so no manual column slicing is needed here either.
    cols = task["kernel_feature_indices"].tolist()
    kernel_fn = build_kernel_fn(
        kernel_name, l, alpha2, period=period, rq_alpha=rq_alpha,
        l_b=l_b, alpha2_b=alpha2_b, period_b=period_b, rq_alpha_b=rq_alpha_b,
        active_dims=cols,
    )
    x_k_train = task["x_norm_train"]   # (P, d_features)
    if kernel_name == "hebo":
        # HEBO+'s Gamma-distributed lengthscale is calibrated for x in
        # [0,1]^k (paper Appendix D) — see data_gen.generate_gp_task's same
        # mapping. Skipping this previously left gp_analytical_pit
        # reconstructing HEBO's kernel over the wrong input domain, silently
        # producing a different z_train than the one generate_gp_task itself
        # computed via _L_ff/_alpha. Safe to map every column: kernel_fn
        # only ever reads the active_dims columns internally.
        x_k_train = torch.special.ndtr(x_k_train)
    y_train    = task["y_train"]                  # (P,)
    y_test     = task["y_test"]                   # (N,)
    mu_star    = task["mu_star"]                  # (N,) posterior mean
    sigma_star = task["sigma_star"]               # (N,) posterior marginal std

    # --- Test: posterior marginals are exact Gaussians ---
    sig_clamped  = sigma_star.clamp(min=1e-8)
    z_test       = (y_test - mu_star) / sig_clamped
    log_pdf_test = (
        -0.5 * math.log(2.0 * math.pi)
        - sig_clamped.log()
        - 0.5 * z_test**2
    )

    # --- Train: exact GP LOO (R&W Eq. 5.12) ---
    # Reuse L_ff and alpha from generate_gp_task when available (B: no double Cholesky).
    # Fall back to kernel reconstruction for tasks loaded from disk.
    P = y_train.shape[0]
    if "_L_ff" in task and "_alpha" in task:
        L     = task["_L_ff"]
        alpha = task["_alpha"]
    else:
        K_ff  = kernel_fn(x_k_train, x_k_train) + nugget * torch.eye(P, device=y_train.device)
        L     = _safe_cholesky(K_ff)
        alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L).squeeze(-1)   # (P,)

    # diag(K_ff^{-1}) = column-wise squared-norm of L^{-1}
    L_inv      = torch.linalg.solve_triangular(
        L, torch.eye(P, device=L.device, dtype=L.dtype), upper=False
    )                                                                      # (P, P)
    K_inv_diag = (L_inv**2).sum(dim=0).clamp(min=1e-12)                   # (P,)
    z_train    = alpha * K_inv_diag.rsqrt()                               # alpha_i/√[K⁻¹]_ii

    return {"z_train": z_train, "z_test": z_test, "log_pdf_test": log_pdf_test}
