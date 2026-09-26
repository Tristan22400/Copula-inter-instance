"""Phase-A marginal objective and diagnostics.

Analytic GP targets, the distillation + pinball + NLL + CRPS loss, the anchor
penalty and the calibration metrics (NLL, CRPS, ECE, KS, rank histogram).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from copula_inter.pit import (
    _kernel_fn_from_task,
    _mean_train_from_task,
    _safe_cholesky,
)

if TYPE_CHECKING:
    from tabicl._model.quantile_dist import QuantileToDistribution


def analytic_marginal_targets(
    task: dict,
    x_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    x_qry: torch.Tensor,
    *,
    kernel_fn: Optional[Callable] = None,
    nugget: Optional[float] = None,
    use_cached_full_context: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact N(mu_i, sigma_i^2) of the observed target at x_qry given (x_ctx, y_ctx).

    sigma^2 = k(x, x) + nugget - ||L^{-1} K_fs||^2 (diagonal only, float64),
    floored at the nugget.

    Args:
        task: episode dict with kernel metadata.
        x_ctx: (P_c, d) context features.
        y_ctx: (P_c,) raw context targets.
        x_qry: (M, d) query features.
        kernel_fn: kernel; rebuilt from task if omitted.
        nugget: noise variance; read from task if omitted.

    Returns:
        (mu, sigma), each (M,) float32 on x_qry's device, raw scale.
    """
    if kernel_fn is None or nugget is None:
        kernel_fn, nugget = _kernel_fn_from_task(task)

    device = x_qry.device
    x_ctx = x_ctx.to(device)
    y_ctx = y_ctx.to(device)
    P_c = x_ctx.shape[0]

    if use_cached_full_context and "_L_ff" in task and "_alpha" in task:
        if P_c != task["x_norm_train"].shape[0]:
            raise ValueError("cached full-context factors require the complete training context")
        L_ff = task["_L_ff"].to(device=device, dtype=torch.float64)
        alpha = task["_alpha"].to(device=device, dtype=torch.float64)
    else:
        K_ff = (kernel_fn(x_ctx, x_ctx) + nugget * torch.eye(P_c, device=device)).double()
        L_ff = _safe_cholesky(K_ff, max_attempts=12)
        mean_ctx = _mean_train_from_task(task, x_ctx).double()
        alpha = torch.cholesky_solve((y_ctx.double() - mean_ctx).unsqueeze(-1), L_ff).squeeze(-1)

    K_sf = kernel_fn(x_qry, x_ctx).double()  # (M, P_c)
    mean_qry = _mean_train_from_task(task, x_qry).double()
    mu = mean_qry + K_sf @ alpha  # (M,)

    V = torch.linalg.solve_triangular(L_ff, K_sf.T, upper=False)  # (P_c, M)
    k_diag = kernel_fn(x_qry, x_qry).diagonal().double() + nugget
    var = (k_diag - (V**2).sum(0)).clamp(min=nugget)
    return mu.float(), var.sqrt().float()


def episode_fold_targets(
    task: dict,
    query_idx: torch.Tensor,
    k_folds: int,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic (mu, sigma) for the training rows in query_idx, each conditioned on the other folds.

    Folds are pit.py's contiguous ceil(P/K) blocks. Returns (mu, sigma) in
    query_idx order, raw scale.
    """
    x_train = task["x_norm_train"].to(device)
    y_train = task["y_train"].to(device)
    P = x_train.shape[0]
    K = max(2, min(int(k_folds), P))
    fold_size = math.ceil(P / K)

    query_idx = query_idx.to(device)
    mu_out = torch.empty(query_idx.numel(), device=device)
    sig_out = torch.empty(query_idx.numel(), device=device)

    fold_of = query_idx // fold_size
    cached = "_L_ff" in task and "_alpha" in task
    if cached:
        L_full = task["_L_ff"].to(device=device, dtype=torch.float64)
        alpha_full = task["_alpha"].to(device=device, dtype=torch.float64)
    else:
        kernel_fn, nugget = _kernel_fn_from_task(task)
    for k in fold_of.unique().tolist():
        sel = (fold_of == k).nonzero(as_tuple=True)[0]  # positions in query_idx
        qry_rows = query_idx[sel]
        start, end = k * fold_size, min((k + 1) * fold_size, P)
        if cached:
            # With precision Lambda and alpha = Lambda (y - mean), conditioning q on the
            # complement gives precision Lambda_qq and mean y_q - Lambda_qq^{-1} alpha_q.
            fold_rows = torch.arange(start, end, device=device)
            eye_q = torch.zeros(P, fold_rows.numel(), dtype=torch.float64, device=device)
            eye_q[fold_rows, torch.arange(fold_rows.numel(), device=device)] = 1.0
            precision_cols = torch.cholesky_solve(eye_q, L_full)
            precision_qq = precision_cols[fold_rows]
            L_qq = _safe_cholesky(precision_qq, max_attempts=12)
            correction = torch.cholesky_solve(alpha_full[fold_rows].unsqueeze(-1), L_qq).squeeze(-1)
            covariance_qq = torch.cholesky_inverse(L_qq)
            positions = qry_rows - start
            mu_k = (y_train[fold_rows].double() - correction)[positions]
            sig_k = covariance_qq.diagonal().clamp(min=1e-12).sqrt()[positions]
        else:
            ctx_mask = torch.ones(P, dtype=torch.bool, device=device)
            ctx_mask[start:end] = False
            ctx_rows = ctx_mask.nonzero(as_tuple=True)[0]
            mu_k, sig_k = analytic_marginal_targets(
                task,
                x_train[ctx_rows],
                y_train[ctx_rows],
                x_train[qry_rows],
                kernel_fn=kernel_fn,
                nugget=nugget,
            )
        mu_out[sel] = mu_k.to(mu_out.dtype)
        sig_out[sel] = sig_k.to(sig_out.dtype)
    return mu_out, sig_out


class MarginalLossWeights:
    """Weights of the Phase-A loss terms: distill (analytic quantiles), pinball, nll, crps and anchor (L2 to pretrained weights)."""

    def __init__(
        self,
        distill: float = 1.0,
        nll: float = 0.0,
        crps: float = 0.0,
        pinball: float = 0.0,
        anchor: float = 0.0,
        huber_delta: float = 1.0,
        tail_power: float = 0.5,
    ) -> None:
        self.distill = float(distill)
        self.nll = float(nll)
        self.crps = float(crps)
        self.pinball = float(pinball)
        self.anchor = float(anchor)
        self.huber_delta = float(huber_delta)
        self.tail_power = float(tail_power)


def quantile_level_weights(alpha_levels: torch.Tensor, tail_power: float = 0.5) -> torch.Tensor:
    """Weights w_k proportional to (alpha_k (1 - alpha_k))^tail_power, normalized to mean 1 (tail_power=0 is uniform)."""
    a = alpha_levels.clamp(1e-9, 1 - 1e-9)
    w = (a * (1.0 - a)) ** float(tail_power)
    return w / w.mean()


def _standard_normal_icdf(alpha: torch.Tensor) -> torch.Tensor:
    return torch.erfinv(2.0 * alpha.clamp(1e-9, 1 - 1e-9) - 1.0) * math.sqrt(2.0)


def marginal_objective(
    q: torch.Tensor,
    y: torch.Tensor,
    quantile_dist: QuantileToDistribution,
    weights: MarginalLossWeights,
    *,
    mu: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    alpha_levels: Optional[torch.Tensor] = None,
) -> dict:
    """Phase-A loss on M query rows, in normalize_targets space.

        L = w_q * Huber(q_ik, mu_i + sigma_i Phi^{-1}(alpha_k))   [distillation]
          + w_p * pinball(alpha_k, y_i - q_ik)
          + w_n * (-log f(y_i))
          + w_c * CRPS(f, y_i)

    nll and crps are always computed and reported even at weight 0.

    Args:
        q: (M, Q) raw decoder quantiles.
        y: (M,) targets.
        quantile_dist: the model's QuantileToDistribution module.
        mu, sigma: (M,) analytic targets; omit to skip distillation.
        target_mask: (M,) bool, rows with an analytic target (None = all).
        alpha_levels: (Q,) levels; read from quantile_dist if omitted.

    Returns:
        dict of scalar tensors: loss and each unweighted term.
    """
    if alpha_levels is None:
        alpha_levels = quantile_dist.alpha_levels.to(q.device, dtype=q.dtype)

    dist = quantile_dist(q)
    nll = -dist.log_prob(y)  # (M,)
    crps = dist.crps(y)  # (M,)
    error = y.unsqueeze(-1) - q
    pinball = torch.maximum(alpha_levels * error, (alpha_levels - 1.0) * error)

    out = {
        "nll": nll.mean(),
        "crps": crps.mean(),
        "pinball": pinball.mean(),
    }
    # Skip zero-weight terms rather than multiplying by 0 (a NaN would still propagate).
    loss = q.sum() * 0.0
    if weights.nll != 0.0:
        loss = loss + weights.nll * out["nll"]
    if weights.crps != 0.0:
        loss = loss + weights.crps * out["crps"]
    if weights.pinball != 0.0:
        loss = loss + weights.pinball * out["pinball"]

    have_target = mu is not None and sigma is not None and weights.distill != 0.0
    if have_target and target_mask is not None:
        have_target = bool(target_mask.any())
    if have_target:
        assert mu is not None and sigma is not None
        if target_mask is not None:
            q_d, mu_d, sig_d = q[target_mask], mu[target_mask], sigma[target_mask]
        else:
            q_d, mu_d, sig_d = q, mu, sigma
        z_target = _standard_normal_icdf(alpha_levels)  # (Q,)
        w = quantile_level_weights(alpha_levels, weights.tail_power)  # (Q,)
        q_target = mu_d.unsqueeze(-1) + sig_d.unsqueeze(-1) * z_target
        per_level = torch.nn.functional.huber_loss(
            q_d,
            q_target,
            reduction="none",
            delta=weights.huber_delta,
        )
        out["distill"] = (per_level * w).mean()
        loss = loss + weights.distill * out["distill"]
    else:
        out["distill"] = torch.zeros((), device=q.device)

    out["loss"] = loss
    return out


class AnchorPenalty:
    """L2 penalty pulling the trainable parameters toward their initial (pretrained) values."""

    def __init__(self, module: nn.Module) -> None:
        self.ref = {name: p.detach().clone() for name, p in module.named_parameters() if p.requires_grad}

    def __call__(self, module: nn.Module) -> torch.Tensor:
        total = None
        for name, p in module.named_parameters():
            if not p.requires_grad or name not in self.ref:
                continue
            term = ((p - self.ref[name]) ** 2).sum()
            total = term if total is None else total + term
        if total is None:
            dev = next(module.parameters()).device
            return torch.zeros((), device=dev)
        return total


def ks_uniform(u: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic of u against Uniform(0, 1)."""
    u = np.sort(np.asarray(u, dtype=float).ravel())
    n = u.size
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def rank_histogram(u: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Normalized histogram of u = F(y): flat is calibrated, U-shaped over-sharp, dome-shaped under-sharp."""
    counts, _ = np.histogram(np.asarray(u, dtype=float).ravel(), bins=n_bins, range=(0.0, 1.0))
    total = counts.sum()
    return counts / total if total else counts.astype(float)


@torch.no_grad()
def marginal_metrics(
    q: torch.Tensor,
    y: torch.Tensor,
    quantile_dist: QuantileToDistribution,
    *,
    log_std: float | torch.Tensor = 0.0,
    y_std: float | torch.Tensor = 1.0,
    eps: float = 1e-6,
    n_rank_bins: int = 20,
) -> dict:
    """Marginal calibration metrics for M query rows.

    q and y are in normalize_targets space; log_std/y_std convert NLL and CRPS to
    raw-y units (nll_raw = nll + log std, crps_raw = crps * std). ECE, KS and the
    rank histogram are scale-free.
    """
    from eval.spatial.calibration import compute_quantile_ece

    alpha = quantile_dist.alpha_levels.to(q.device, dtype=q.dtype)
    dist = quantile_dist(q)
    u = dist.cdf(y)
    nll_scaled = -dist.log_prob(y)
    crps_scaled = dist.crps(y)

    log_std_t = torch.as_tensor(log_std, dtype=nll_scaled.dtype, device=nll_scaled.device)
    y_std_t = torch.as_tensor(y_std, dtype=crps_scaled.dtype, device=crps_scaled.device)

    u_np = u.detach().float().cpu().numpy()
    ece, coverage = compute_quantile_ece(
        y.detach().float().cpu().numpy(),
        q.detach().float().cpu().numpy(),
        alpha.detach().float().cpu().numpy(),
    )
    return {
        "nll": float((nll_scaled + log_std_t).mean()),
        "crps": float((crps_scaled * y_std_t).mean()),
        "ece": float(ece),
        "ks": ks_uniform(u_np),
        "clamp_frac": float(((u <= eps) | (u >= 1.0 - eps)).float().mean()),
        "n": int(y.numel()),
        "_coverage": coverage,
        "_rank_hist": rank_histogram(u_np, n_rank_bins),
    }


@torch.no_grad()
def oracle_marginal_nll(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> float:
    """Mean -log N(y; mu, sigma^2): the analytic floor for the model's marginal NLL."""
    var = sigma.clamp(min=1e-12) ** 2
    nll = 0.5 * (torch.log(2 * math.pi * var) + (y - mu) ** 2 / var)
    return float(nll.mean())
