"""Phase A: fine-tune a standalone TabICL marginal (quantile decoder intact).

Trains the marginal's posterior predictive on GP episodes (optionally mixed
with real ERA5) and writes a TabICL-schema checkpoint usable as
tabicl.pit_ckpt:

    python -m copula_inter.train tabicl.pit_ckpt=<checkpoint>

Usage:
    python -m copula_inter.finetune_marginal
    python -m copula_inter.finetune_marginal marginal.tier=1 training.lr=2e-5
    python -m copula_inter.finetune_marginal wandb.mode=disabled training.steps=20
"""

from __future__ import annotations

import math
import os
import random
import time
import zlib
from typing import Callable, Optional, Sequence

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from copula_inter.artifacts import atomic_torch_save
from copula_inter.config_path import config_dir
from copula_inter.data_gen import generate_gp_batch  # noqa: E402
from copula_inter.lora import (
    apply_lora,
    apply_lora_all_layers,
    merged_base_state_dict_any,
)
from copula_inter.marginal_backbones import TIER0_PATTERNS as _BACKBONE_TIER0  # noqa: E402
from copula_inter.marginal_backbones import (  # noqa: E402
    MarginalBackbone,  # noqa: E402
    assert_patterns_match,
    kfold_quantiles_grad,
    load_backbone,
    resolve_tier,
)
from copula_inter.pit import (
    DEFAULT_K_FOLDS,
    _kernel_fn_from_task,
    _mean_train_from_task,
    _safe_cholesky,
    load_tabicl,  # noqa: E402
    normalize_targets,
    run_pit_batched_grad,
)
from copula_inter.training_core import cosine_lr_lambda  # noqa: E402

# Tier 0: the label path, the ICL-stage norms and the decoder (per architecture
# in marginal_backbones.py; re-exported here).
TIER0_PATTERNS: tuple[str, ...] = _BACKBONE_TIER0["tabicl"]

# Tier ladder: tier 0 plus LoRA on successively more attention stages. No full fine-tuning.
TIER_SPECS: dict[int, dict] = {
    0: {
        "lora_stages": [],
        "desc": "label path + ICL norms + decoder (~1.6M, 5.5%)",
    },
    1: {
        "lora_stages": ["icl"],
        "desc": "tier 0 + LoRA on icl_predictor attention",
    },
    2: {
        "lora_stages": ["icl", "row"],
        "desc": "tier 1 + LoRA on row_interactor attention",
    },
    3: {
        "lora_stages": ["icl", "row", "col"],
        "desc": "tier 2 + LoRA on col_embedder attention",
    },
}


def apply_tier(
    backbone: nn.Module,
    tier: int,
    *,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_target: str = "qkvo",
    extra_patterns: Sequence[str] = (),
    backbone_name: str = "tabicl",
    all_layers: bool = False,
) -> dict:
    """Make the backbone's parameters trainable for the given tier (tier-0 allowlist, plus LoRA for tier >= 1 or all-layer LoRA).

    Returns trainable_param_report(backbone) plus the tier and LoRA settings.
    """
    if tier not in TIER_SPECS:
        raise ValueError(f"Unknown tier {tier}; expected one of {sorted(TIER_SPECS)}.")
    # Raises when the architecture cannot reach the requested tier.
    if not all_layers:
        # all_layers LoRA is not limited by the stage ladder.
        resolve_tier(backbone_name, tier)
    spec = TIER_SPECS[tier]
    stages = list(spec["lora_stages"])
    patterns = tuple(_BACKBONE_TIER0[backbone_name]) + tuple(extra_patterns)
    # Fail if a tier-0 pattern matches nothing.
    assert_patterns_match(backbone, _BACKBONE_TIER0[backbone_name])

    if all_layers:
        # LoRA on every 2-D weight matrix at one shared rank (via parametrization).
        n_replaced = apply_lora_all_layers(
            backbone=backbone,
            rank=int(lora_rank),
            alpha=float(lora_alpha),
            also_trainable=patterns,
        )
    else:
        n_replaced = apply_lora(
            backbone=backbone,
            rank=int(lora_rank) if stages else 0,
            alpha=float(lora_alpha),
            target=lora_target,
            stages=stages,
            also_trainable=patterns,
        )
    report = trainable_param_report(backbone)
    report.update(
        {
            "backbone": backbone_name,
            "tier": tier,
            "tier_desc": "all layers (LoRA on every 2-D weight)" if all_layers else spec["desc"],
            "lora_all_layers": bool(all_layers),
            "lora_rank": int(lora_rank),
            "lora_stages": stages,
            "lora_modules_replaced": n_replaced,
        }
    )
    return report


def trainable_param_report(module: nn.Module) -> dict:
    """{n_trainable_params, n_total_params, trainable_frac, n_trainable_tensors}."""
    n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in module.parameters())
    return {
        "n_trainable_params": int(n_train),
        "n_total_params": int(n_total),
        "trainable_frac": float(n_train / max(n_total, 1)),
        "n_trainable_tensors": int(sum(1 for p in module.parameters() if p.requires_grad)),
    }


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
    quantile_dist: nn.Module,
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
    quantile_dist: nn.Module,
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


def stack_episodes(episodes: Sequence[dict], device: str | torch.device) -> dict:
    """Stack same-shape generate_gp_batch episodes into (B, ...) tensors plus each episode's normalize_targets scale."""
    x_train = torch.stack([e["x_norm_train"] for e in episodes]).to(device)  # (B,P,d)
    y_train = torch.stack([e["y_train"] for e in episodes]).to(device)  # (B,P)
    x_test = torch.stack([e["x_norm_test"] for e in episodes]).to(device)  # (B,N,d)
    y_test = torch.stack([e["y_test"] for e in episodes]).to(device)  # (B,N)

    y_tr_s, y_te_s, means, stds = [], [], [], []
    for b in range(len(episodes)):
        a, c, m, s = normalize_targets(y_train[b], y_test[b])
        y_tr_s.append(a)
        y_te_s.append(c)
        means.append(m)
        stds.append(s)
    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train_raw": y_train,
        "y_test_raw": y_test,
        "y_train_scaled": torch.stack(y_tr_s),
        "y_test_scaled": torch.stack(y_te_s),
        "y_mean": torch.stack(means),
        "y_std": torch.stack(stds),
    }


def phase_a_batch_loss(
    tabicl: "nn.Module | MarginalBackbone",
    episodes: Sequence[dict],
    weights: MarginalLossWeights,
    *,
    k_folds: int = DEFAULT_K_FOLDS,
    folds_per_step: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: str | torch.device = "cuda",
    eps: float = 1e-6,
    timings: Optional[dict[str, float]] = None,
    marginal_probs_n: "int | None" = None,
) -> dict:
    """Forward and loss of one Phase-A step on a batch of GP episodes.

    Scores the N test rows against the full context and folds_per_step of the K
    training folds against their K-1-fold context (default all K). Episodes
    whose kernel cannot be rebuilt get no distillation target but still
    contribute sample-score terms. tabicl is a TabICL module or a
    MarginalBackbone (marginal_backbones.kfold_quantiles_grad, same fold geometry).
    """

    def _mark(name: str, started: float) -> float:
        if timings is not None:
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            timings[name] = timings.get(name, 0.0) + now - started
            return now
        return time.perf_counter()

    if timings is not None and torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    t_part = time.perf_counter()
    B = len(episodes)
    batch = stack_episodes(episodes, device)
    t_part = _mark("collate", t_part)
    P = batch["x_train"].shape[1]
    K = max(2, min(int(k_folds), P))

    # Sample only non-empty folds (for P < K some ceil(P/K) blocks are empty).
    fold_size = math.ceil(P / K)
    n_folds_eff = math.ceil(P / fold_size)
    if folds_per_step is None or folds_per_step >= n_folds_eff:
        fold_subset = None
    else:
        n_f = max(1, int(folds_per_step))
        perm = torch.randperm(n_folds_eff, generator=generator)[:n_f]
        fold_subset = sorted(perm.tolist())

    is_backbone = isinstance(tabicl, MarginalBackbone) and tabicl.name != "tabicl"
    if is_backbone:
        # None: score the model's native decoder grid (999 levels).
        probs = (
            None
            if marginal_probs_n is None
            else np.linspace(
                1.0 / (marginal_probs_n + 1),
                marginal_probs_n / (marginal_probs_n + 1),
                marginal_probs_n,
            )
        )
        out = kfold_quantiles_grad(
            tabicl,
            batch["x_train"],
            batch["y_train_scaled"],
            batch["x_test"],
            batch["y_test_scaled"],
            k_folds=K,
            probs=probs,
            fold_subset=fold_subset,
        )
        quantile_dist = tabicl.quantile_dist_module(probs)
        q_test, q_train = out["q_test"], out["q_train"]
    else:
        module = tabicl.module if isinstance(tabicl, MarginalBackbone) else tabicl
        out = run_pit_batched_grad(
            module,
            batch["x_train"],
            batch["y_train_scaled"].unsqueeze(-1),
            batch["x_test"],
            batch["y_test_scaled"].unsqueeze(-1),
            k_folds=K,
            eps=eps,
            return_quantiles=True,
            fold_subset=fold_subset,
            compute_pit=False,
            fuse_folds=True,
            Y_train_raw=batch["y_train_raw"].unsqueeze(-1),
        )
        quantile_dist = module.quantile_dist
        q_test = out["q_test"].squeeze(2)  # (B, N, Q)
        q_train = out["q_train"].squeeze(2)  # (B, P', Q)
    t_part = _mark("tabicl_forward", t_part)
    if fold_subset is None:
        train_idx = torch.arange(P, device=q_train.device)
    else:
        train_idx = out["train_query_idx"]

    # --- analytic targets, per episode, in normalize_targets space ---------
    M = q_test.shape[1] + q_train.shape[1]
    mu_all = torch.zeros(B, M, device=q_test.device)
    sig_all = torch.ones(B, M, device=q_test.device)
    mask_all = torch.zeros(B, M, dtype=torch.bool, device=q_test.device)
    n_ok = 0
    # Compute analytic targets without autograd.
    with torch.no_grad():
        for b, ep in enumerate(episodes):
            try:
                kernel_fn, nugget = _kernel_fn_from_task(ep)
                mu_te, sig_te = analytic_marginal_targets(
                    ep,
                    batch["x_train"][b],
                    batch["y_train_raw"][b],
                    batch["x_test"][b],
                    kernel_fn=kernel_fn,
                    nugget=nugget,
                    use_cached_full_context=True,
                )
                mu_tr, sig_tr = episode_fold_targets(ep, train_idx, K, device=device)
            except (NotImplementedError, KeyError):
                continue  # configured sample scores may apply; target does not
            n_ok += 1
            m, sd = batch["y_mean"][b], batch["y_std"][b]
            mu_all[b] = torch.cat([(mu_te - m) / sd, (mu_tr - m) / sd])
            sig_all[b] = torch.cat([sig_te / sd, sig_tr / sd])
            mask_all[b] = True
    t_part = _mark("analytic_targets", t_part)

    q_all = torch.cat([q_test, q_train], dim=1)  # (B, N+P', Q)
    y_all = torch.cat([batch["y_test_scaled"], batch["y_train_scaled"][:, train_idx]], dim=1)  # (B, N+P')

    Q = q_all.shape[-1]
    q_flat = q_all.reshape(-1, Q)
    y_flat = y_all.reshape(-1)
    mu_flat = mu_all.reshape(-1)
    sig_flat = sig_all.reshape(-1)
    mask_flat = mask_all.reshape(-1)

    res = marginal_objective(
        q_flat,
        y_flat,
        quantile_dist,
        weights,
        mu=mu_flat,
        sigma=sig_flat,
        target_mask=mask_flat,
    )
    # Report the pre-sort quantile crossing rate (a decoder collapse shows up here).
    res["raw_crossing_frac"] = float((q_flat[:, 1:] < q_flat[:, :-1]).float().mean().detach())
    _mark("objective", t_part)
    res["n_episodes_with_target"] = n_ok
    res["oracle_nll"] = (
        oracle_marginal_nll(y_flat[mask_flat], mu_flat[mask_flat], sig_flat[mask_flat]) if n_ok else float("nan")
    )
    # Gap of the model's marginal NLL to the analytic floor on these rows.
    res["nll_gap_to_oracle"] = float(res["nll"].detach()) - res["oracle_nll"]
    return res


def build_era5_marginal_val_batches(vcfg, device: str | torch.device) -> dict:
    """Fixed per-region real-ERA5 probes for Phase-A validation, holding raw (x, y).

    Same geometry and seeds as the copula run's ERA5 probes
    (sweep_core.build_era5_probe with tabicl_marginal=None), without correlation
    or GP-baseline fields. Data comes from the held-out 2023 period.
    """
    from eval.configs.regions import REGIONS as ERA5_REGIONS
    from eval.spatial.sweep_core import build_era5_probe

    def _g(key, default):
        return vcfg.get(key, default) if hasattr(vcfg, "get") else getattr(vcfg, key, default)

    region_names = list(_g("era5_regions", []) or list(ERA5_REGIONS.keys()))
    grid_size = int(_g("era5_grid_size", 24))
    n_days_fetch = int(_g("era5_n_days_fetch", 60))
    n_days_probe = int(_g("era5_n_days_probe", 3))
    n_context = int(_g("era5_n_context", 30))
    base_seed = int(_g("era5_seed", 20260818))

    batches: dict[str, dict] = {}
    for region in region_names:
        if region not in ERA5_REGIONS:
            continue  # not a registered eval/configs/regions.py entry
        # zlib.crc32 seed, same as probe_batches._name_seed, so both phases use the same points.
        seed = base_seed + (zlib.crc32(region.encode()) % 10_000)
        probe = build_era5_probe(
            region,
            grid_size,
            n_days_fetch,
            n_days_probe,
            n_context,
            n_bins=12,
            tabicl_marginal=None,
            device=str(device),
            seed=seed,
        )
        n_days = probe["context_values_per_day"].shape[0]
        x_tr = torch.as_tensor(probe["x_train_norm"], dtype=torch.float32, device=device)
        x_te = torch.as_tensor(probe["x_nll_test_norm"], dtype=torch.float32, device=device)
        batches[region] = {
            "x_train": x_tr.unsqueeze(0).expand(n_days, -1, -1).contiguous(),
            "x_test": x_te.unsqueeze(0).expand(n_days, -1, -1).contiguous(),
            "y_train": torch.as_tensor(probe["context_values_per_day"], dtype=torch.float32, device=device),
            "y_test": torch.as_tensor(probe["nll_test_values_per_day"], dtype=torch.float32, device=device),
        }
    return batches


@torch.no_grad()
def validate_era5_marginal(
    tabicl: "nn.Module | MarginalBackbone",
    batches: dict,
    *,
    eps: float = 1e-6,
    marginal_probs_n: "int | None" = None,
) -> dict:
    """Marginal metrics per ERA5 region and their means: val_marginal/<region>/{nll, crps, ece, ks, clamp_frac}, val_marginal/mean_*.

    Query points are outside the context, so one full-context forward
    (fold_subset=[]) is used.
    """
    per_region: dict[str, dict] = {}
    is_backbone = isinstance(tabicl, MarginalBackbone) and tabicl.name != "tabicl"
    if is_backbone:
        probs = (
            None
            if marginal_probs_n is None
            else np.linspace(1.0 / (marginal_probs_n + 1), marginal_probs_n / (marginal_probs_n + 1), marginal_probs_n)
        )
        quantile_dist = tabicl.quantile_dist_module(probs)

    for region, b in batches.items():
        y_tr_s, y_te_s, mean, std = [], [], [], []
        for d in range(b["y_train"].shape[0]):
            a, c, m, sd = normalize_targets(b["y_train"][d], b["y_test"][d])
            y_tr_s.append(a)
            y_te_s.append(c)
            mean.append(m)
            std.append(sd)
        y_tr_s = torch.stack(y_tr_s)
        y_te_s = torch.stack(y_te_s)
        std_t = torch.stack(std)

        if is_backbone:
            xtr = b["x_train"].detach().cpu().numpy()
            ytr = y_tr_s.detach().cpu().numpy()
            xte = b["x_test"].detach().cpu().numpy()
            days = b["y_train"].shape[0]
            q = tabicl.quantile_forward(
                [xtr[d] for d in range(days)],
                [ytr[d] for d in range(days)],
                [xte[d] for d in range(days)],
                probs,
            )
        else:
            module = tabicl.module if isinstance(tabicl, MarginalBackbone) else tabicl
            out = run_pit_batched_grad(
                module,
                b["x_train"],
                y_tr_s.unsqueeze(-1),
                b["x_test"],
                y_te_s.unsqueeze(-1),
                k_folds=2,
                eps=eps,
                return_quantiles=True,
                fold_subset=[],
                compute_pit=False,
                Y_train_raw=b["y_train"].unsqueeze(-1),
            )
            q = out["q_test"].squeeze(2)  # (days, N, Q)
            quantile_dist = module.quantile_dist

        # Per-day std for the raw-nats conversion.
        n_q = q.shape[1]
        log_std = std_t.log().unsqueeze(1).expand(-1, n_q).reshape(-1)
        y_std = std_t.unsqueeze(1).expand(-1, n_q).reshape(-1)
        m = marginal_metrics(
            q.reshape(-1, q.shape[-1]),
            y_te_s.reshape(-1),
            quantile_dist,
            log_std=log_std,
            y_std=y_std,
            eps=eps,
        )
        per_region[region] = m

    metrics: dict[str, float] = {}
    for region, m in per_region.items():
        for k in ("nll", "crps", "ece", "ks", "clamp_frac"):
            metrics[f"val_marginal/{region}/{k}"] = m[k]
    if per_region:
        for k in ("nll", "crps", "ece", "ks", "clamp_frac"):
            metrics[f"val_marginal/mean_{k}"] = float(np.mean([m[k] for m in per_region.values()]))
    return metrics


@torch.no_grad()
def validate_synthetic_marginal(
    tabicl: "nn.Module | MarginalBackbone",
    episode_batches: Sequence[Sequence[dict]],
    *,
    k_folds: int = DEFAULT_K_FOLDS,
    eps: float = 1e-6,
    device: str | torch.device = "cuda",
    marginal_probs_n: "int | None" = None,
) -> dict:
    """Marginal metrics on the fixed GP validation set: val_marginal/gp/nll, nll_oracle, nll_gap_to_oracle and the training objective."""
    # Also report the training objective on the validation episodes.
    metric_w = MarginalLossWeights(distill=1.0, nll=0.0, crps=0.0)
    nlls, crpss, distills, oracles, crossings = [], [], [], [], []
    for episodes in episode_batches:
        res = phase_a_batch_loss(
            tabicl,
            episodes,
            metric_w,
            k_folds=k_folds,
            folds_per_step=None,
            device=device,
            eps=eps,
            marginal_probs_n=marginal_probs_n,
        )
        nlls.append(float(res["nll"]))
        crpss.append(float(res["crps"]))
        distills.append(float(res["distill"]))
        oracles.append(res["oracle_nll"])
        crossings.append(res["raw_crossing_frac"])
    out = {
        "val_marginal/gp/nll": float(np.mean(nlls)) if nlls else float("nan"),
        "val_marginal/gp/crps": float(np.mean(crpss)) if crpss else float("nan"),
        "val_marginal/gp/distill": (float(np.mean(distills)) if distills else float("nan")),
        "val_marginal/gp/nll_oracle": float(np.nanmean(oracles)) if oracles else float("nan"),
        "val_marginal/gp/raw_crossing_frac": (float(np.mean(crossings)) if crossings else float("nan")),
    }
    out["val_marginal/gp/nll_gap_to_oracle"] = out["val_marginal/gp/nll"] - out["val_marginal/gp/nll_oracle"]
    return out


def save_marginal_checkpoint(
    path: str,
    backbone: nn.Module,
    tabicl_config: dict,
    *,
    step: int,
    cfg=None,
    extra: Optional[dict] = None,
) -> None:
    """Write a TabICL-schema checkpoint ({"config", "state_dict"}, LoRA merged) that pit.load_tabicl can read.

    step, cfg and extra are stored alongside.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "config": dict(tabicl_config),
        "state_dict": merged_base_state_dict_any(backbone),
        "step": int(step),
    }
    if cfg is not None:
        from omegaconf import OmegaConf

        payload["cfg"] = OmegaConf.to_container(cfg, resolve=True)
    if extra:
        payload.update(extra)
    atomic_torch_save(payload, path)


def _resolve_device(spec: str) -> str:
    if spec != "auto":
        return spec
    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ERA5EpisodeSampler:
    """Batches of real-ERA5 episodes with one shared P and N (sample_episode_fixed_shape); region, day and box vary."""

    def __init__(
        self, corpus, *, grid_size: int, n_context: int, box_deg_range: tuple[float, float], seed: int
    ) -> None:
        self.corpus = corpus
        self.grid_size = int(grid_size)
        self.n_context = int(n_context)
        self.box_deg_range = box_deg_range
        self.rng = np.random.default_rng(seed)

    def batch(self, B: int, max_tries: int = 200) -> list[dict]:
        out: list[dict] = []
        tries = 0
        while len(out) < B and tries < max_tries * B:
            tries += 1
            ep = self.corpus.sample_episode_fixed_shape(self.rng, self.grid_size, self.box_deg_range, self.n_context)
            if ep is None:
                continue
            out.append(
                {
                    "x_norm_train": torch.as_tensor(ep["x_norm_train"]),
                    "y_train": torch.as_tensor(ep["y_train"]),
                    "x_norm_test": torch.as_tensor(ep["x_norm_test"]),
                    "y_test": torch.as_tensor(ep["y_test"]),
                }
            )
        if len(out) < B:
            raise RuntimeError(
                f"ERA5 sampler produced {len(out)}/{B} episodes in {tries} draws at "
                f"grid_size={self.grid_size}, n_context={self.n_context}, "
                f"box_deg_range={self.box_deg_range}. Widen box_deg_max or lower "
                f"grid_size."
            )
        return out


def _gp_cfg(cfg: DictConfig) -> DictConfig:
    """A config with the data group and a seed, as generate_gp_batch expects."""
    return OmegaConf.create({"data": OmegaConf.to_container(cfg.data, resolve=True), "seed": int(cfg.seed)})


def _generate_phase_a_gp_batch(gp_cfg: DictConfig, batch_size: int, device: str, *, max_rounds: int = 20) -> list[dict]:
    """generate_gp_batch with P, N and d pinned after the first call, so every episode has the same shape."""
    episodes = generate_gp_batch(gp_cfg, batch_size, device, return_kernel_metadata=True)
    P = int(episodes[0]["x_norm_train"].shape[0])
    N = int(episodes[0]["x_norm_test"].shape[0])
    d = int(episodes[0]["x_norm_train"].shape[1])
    out = [ep for ep in episodes if ep["x_norm_train"].shape == (P, d) and ep["x_norm_test"].shape == (N, d)]
    if len(out) == batch_size:
        return out

    fixed = OmegaConf.create(OmegaConf.to_container(gp_cfg, resolve=True))
    fixed.data.P_min = fixed.data.P_max = P
    fixed.data.N_min = fixed.data.N_max = N
    base_seed = int(gp_cfg.seed)
    for round_idx in range(1, max_rounds + 1):
        fixed.seed = base_seed + round_idx * 1_000_003
        out.extend(
            generate_gp_batch(
                fixed,
                batch_size - len(out),
                device,
                return_kernel_metadata=True,
                d_override=d,
            )
        )
        if len(out) >= batch_size:
            return out[:batch_size]
    raise RuntimeError(
        f"Phase-A GP generator produced only {len(out)}/{batch_size} episodes "
        f"with fixed shape P={P}, N={N}, d={d} after {max_rounds} retries."
    )


def _build_gp_val_batches(cfg: DictConfig, device: str) -> list[list[dict]]:
    """Fixed synthetic GP validation batches, drawn once with their own seed."""
    gp_cfg = _gp_cfg(cfg)
    batches = []
    for i in range(int(cfg.validation.gp_n_batches)):
        gp_cfg.seed = int(cfg.validation.gp_seed) + i
        batches.append(
            _generate_phase_a_gp_batch(
                gp_cfg,
                int(cfg.validation.gp_batch_size),
                device,
            )
        )
    return batches


@hydra.main(config_path=config_dir(__file__), config_name="finetune_marginal", version_base=None)
def main(cfg: DictConfig) -> None:
    device = _resolve_device(str(cfg.training.device))
    torch.set_float32_matmul_precision(str(cfg.training.matmul_precision))
    _seed_everything(int(cfg.seed))
    print(OmegaConf.to_yaml(cfg))

    # ---- model + tier routing -------------------------------------------
    backbone_name = str(cfg.marginal.get("backbone", "tabicl"))
    _probs_n_cfg = cfg.marginal.get("probs_n", None)
    marginal_probs_n = None if _probs_n_cfg is None else int(_probs_n_cfg)
    if backbone_name == "tabicl":
        # Unchanged path: load_tabicl owns TabICL's own checkpoint schema.
        tabicl, tabicl_config = load_tabicl(str(cfg.marginal.ckpt), device, trainable=True, return_config=True)
        trainable_module = tabicl
    else:
        backbone_obj = load_backbone(backbone_name, ckpt=cfg.marginal.get("resume_ckpt", None), device=device)
        if backbone_name == "exaone":
            backbone_obj.exaone_chunk_size = int(cfg.marginal.exaone.chunk_size)
            if backbone_obj.exaone_chunk_size < 1:
                raise ValueError("marginal.exaone.chunk_size must be positive")
            backbone_obj.exaone_activation_checkpointing = bool(cfg.marginal.exaone.activation_checkpointing)
        tabicl, tabicl_config = backbone_obj, {}
        trainable_module = backbone_obj.module
        for p_ in trainable_module.parameters():
            p_.requires_grad_(True)
    report = apply_tier(
        trainable_module,
        int(cfg.marginal.tier),
        lora_rank=int(cfg.marginal.lora_rank),
        lora_alpha=float(cfg.marginal.lora_alpha),
        lora_target=str(cfg.marginal.lora_target),
        backbone_name=backbone_name,
        all_layers=bool(cfg.marginal.get("lora_all_layers", True)),
    )
    trainable_module.to(device)
    print(
        f"[{report.get('backbone', 'tabicl')} tier {report['tier']}] {report['tier_desc']}: "
        f"{report['n_trainable_params']:,} / {report['n_total_params']:,} trainable "
        f"({100 * report['trainable_frac']:.2f}%), "
        f"{report['lora_modules_replaced']} LoRA module(s) at rank {report['lora_rank']}"
    )

    weights = MarginalLossWeights(
        distill=float(cfg.marginal.loss.distill),
        nll=float(cfg.marginal.loss.nll),
        crps=float(cfg.marginal.loss.crps),
        pinball=float(cfg.marginal.loss.pinball),
        anchor=float(cfg.marginal.loss.anchor),
        huber_delta=float(cfg.marginal.loss.huber_delta),
        tail_power=float(cfg.marginal.loss.tail_power),
    )
    # Separate loss weights for synthetic and ERA5 batches.
    era5_loss_cfg = cfg.marginal.era5.loss
    era5_weights = MarginalLossWeights(
        distill=0.0,
        nll=float(era5_loss_cfg.nll),
        crps=float(era5_loss_cfg.crps),
        pinball=float(era5_loss_cfg.pinball),
        anchor=weights.anchor,
        huber_delta=weights.huber_delta,
        tail_power=weights.tail_power,
    )
    anchor = AnchorPenalty(trainable_module) if weights.anchor > 0 else None

    # AdamW over all trainable parameters in one group.
    params = [p for p in trainable_module.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("Tier routing left no trainable parameters.")
    adam_eps = 1e-4 if any(p.dtype == torch.float16 for p in params) else 1e-8
    opt = torch.optim.AdamW(
        params,
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        eps=adam_eps,
    )
    total_steps = int(cfg.training.steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: cosine_lr_lambda(
            s,
            int(cfg.training.warmup_steps),
            total_steps,
            float(cfg.training.lr_min_frac),
        ),
    )

    # ---- data ------------------------------------------------------------
    gp_cfg = _gp_cfg(cfg)
    eps = float(cfg.marginal.pit_eps)
    k_folds = int(cfg.marginal.k_folds)
    folds_per_step = cfg.marginal.folds_per_step
    folds_per_step = None if folds_per_step is None else int(folds_per_step)
    mix_frac = float(cfg.marginal.era5.mix_frac)
    if not 0.0 <= mix_frac <= 1.0:
        raise ValueError(f"marginal.era5.mix_frac must be in [0, 1], got {mix_frac}")

    def _has_sample_objective(w: MarginalLossWeights) -> bool:
        return any(value != 0.0 for value in (w.distill, w.nll, w.crps, w.pinball))

    # Fail if every loss weight is zero.
    if mix_frac < 1.0 and not _has_sample_objective(weights):
        raise ValueError("Synthetic batches have no non-zero marginal loss weight.")
    if mix_frac > 0.0 and not _has_sample_objective(era5_weights):
        raise ValueError("ERA5 batches have no non-zero marginal loss weight.")

    era5_sampler = None
    if mix_frac > 0:
        from eval.data.era5_global_corpus import GlobalERA5Corpus

        corpus = GlobalERA5Corpus(
            str(cfg.marginal.era5.corpus_dir),
            max_months=int(cfg.marginal.era5.max_months),
        )
        era5_sampler = ERA5EpisodeSampler(
            corpus,
            grid_size=int(cfg.marginal.era5.grid_size),
            n_context=int(cfg.marginal.era5.n_context),
            box_deg_range=(
                float(cfg.marginal.era5.box_deg_min),
                float(cfg.marginal.era5.box_deg_max),
            ),
            seed=int(cfg.seed) + 7717,
        )
        print(f"[era5] mixture on: {corpus.n_days_total} days loaded, mix_frac={mix_frac}")

    print("[val] building fixed validation sets (one-off ERA5 fetch/crop)...")
    era5_val = build_era5_marginal_val_batches(cfg.validation, device)
    gp_val = _build_gp_val_batches(cfg, device)
    print(f"[val] {len(era5_val)} ERA5 region(s), {len(gp_val)} synthetic GP batch(es)")

    # ---- wandb -----------------------------------------------------------
    run = None
    if str(cfg.wandb.mode) != "disabled":
        import wandb

        run = wandb.init(
            project=str(cfg.wandb.project),
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=str(cfg.wandb.mode),
        )
        wandb.watch(trainable_module, log="gradients", log_freq=max(1, int(cfg.training.log_every)))
        wandb.log({f"model/{k}": v for k, v in report.items() if isinstance(v, (int, float))}, step=0)

    def _log(payload: dict, step: int) -> None:
        if run is not None:
            run.log(payload, step=step)

    def _validate(step: int) -> dict[str, float]:
        t0 = time.time()
        t_era5 = time.time()
        metrics = validate_era5_marginal(tabicl, era5_val, eps=eps, marginal_probs_n=marginal_probs_n)
        metrics["val_marginal/era5_seconds"] = time.time() - t_era5
        t_gp = time.time()
        metrics.update(
            validate_synthetic_marginal(
                tabicl,
                gp_val,
                k_folds=k_folds,
                eps=eps,
                device=device,
                marginal_probs_n=marginal_probs_n,
            )
        )
        metrics["val_marginal/gp_seconds"] = time.time() - t_gp
        metrics["val_marginal/seconds"] = time.time() - t0
        _log(metrics, step)
        print(
            f"[val step {step}] "
            f"era5 nll={metrics.get('val_marginal/mean_nll', float('nan')):.4f} "
            f"ece={metrics.get('val_marginal/mean_ece', float('nan')):.4f} "
            f"ks={metrics.get('val_marginal/mean_ks', float('nan')):.4f} | "
            f"gp nll={metrics.get('val_marginal/gp/nll', float('nan')):.4f} "
            f"distill={metrics.get('val_marginal/gp/distill', float('nan')):.4f} "
            f"oracle={metrics.get('val_marginal/gp/nll_oracle', float('nan')):.4f} "
            f"gap={metrics.get('val_marginal/gp/nll_gap_to_oracle', float('nan')):.4f} | "
            f"{metrics['val_marginal/seconds']:.2f}s "
            f"(era5 {metrics['val_marginal/era5_seconds']:.2f}s, "
            f"gp {metrics['val_marginal/gp_seconds']:.2f}s)"
        )
        return metrics

    def _save(step: int, tag: str = "") -> str | None:
        if cfg.training.ckpt_dir is None:
            return None
        name = f"step_{step:07d}{tag}.pt"
        path = os.path.join(str(cfg.training.ckpt_dir), name)
        tier_extra = {"tier_report": {k: v for k, v in report.items() if isinstance(v, (int, float, str))}}
        if isinstance(tabicl, MarginalBackbone):
            # Non-TabICL backbones write their own checkpoint format.
            tabicl.save(path, step=step, cfg=cfg, extra=tier_extra)
        else:
            save_marginal_checkpoint(
                path,
                tabicl,
                tabicl_config,
                step=step,
                cfg=cfg,
                extra=tier_extra,
            )
        print(f"[ckpt] {path}")
        return path

    # ---- train -----------------------------------------------------------
    initial_metrics = _validate(0)
    selection_metric = str(cfg.training.get("selection_metric", "val_marginal/mean_nll"))
    if selection_metric not in initial_metrics:
        raise KeyError(
            f"training.selection_metric={selection_metric!r} was not emitted by "
            f"validation. Available metrics: {sorted(initial_metrics)}"
        )
    best_value = float(initial_metrics[selection_metric])
    if not math.isfinite(best_value):
        raise RuntimeError(f"Initial selection metric {selection_metric} is non-finite: {best_value}")
    best_step = 0
    selection_min_delta = float(cfg.training.get("selection_min_delta", 0.0))

    def _snapshot_trainable() -> dict[str, torch.Tensor]:
        # Keep only the trainable tensors for best-checkpoint selection.
        return {name: p.detach().cpu().clone() for name, p in trainable_module.named_parameters() if p.requires_grad}

    def _restore_trainable(state: dict[str, torch.Tensor]) -> None:
        named = dict(trainable_module.named_parameters())
        with torch.no_grad():
            for name, value in state.items():
                named[name].copy_(value.to(device=named[name].device))

    best_state = _snapshot_trainable()

    def _consider_validation(step: int, metrics: dict[str, float]) -> None:
        nonlocal best_step, best_value, best_state
        value = float(metrics[selection_metric])
        if math.isfinite(value) and value < best_value - selection_min_delta:
            best_step = step
            best_value = value
            best_state = _snapshot_trainable()
            print(f"[selection] new best {selection_metric}={best_value:.6f} at step {best_step}")

    rng = np.random.default_rng(int(cfg.seed) + 991)
    gen = torch.Generator().manual_seed(int(cfg.seed) + 13)
    B = int(cfg.training.batch_size)
    t_last = time.time()
    profile_steps = int(cfg.training.get("profile_steps", 0))
    profile_totals: dict[str, float] = {}

    for step in range(1, total_steps + 1):
        profiling = step <= profile_steps
        if profiling and device.startswith("cuda"):
            torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        data_started = step_started
        use_era5 = era5_sampler is not None and rng.random() < mix_frac
        if use_era5:
            episodes = era5_sampler.batch(B)
            episodes = [{k: v.to(device) for k, v in ep.items()} for ep in episodes]
            w = era5_weights
        else:
            gp_cfg.seed = int(cfg.seed) * 1_000_003 + step
            episodes = _generate_phase_a_gp_batch(gp_cfg, B, device)
            w = weights
        if profiling and device.startswith("cuda"):
            torch.cuda.synchronize(device)
        data_seconds = time.perf_counter() - data_started

        part_timings: dict[str, float] | None = {} if profiling else None
        res = phase_a_batch_loss(
            tabicl,
            episodes,
            w,
            k_folds=k_folds,
            folds_per_step=folds_per_step,
            generator=gen,
            device=device,
            eps=eps,
            timings=part_timings,
            marginal_probs_n=marginal_probs_n,
        )
        loss = res["loss"]
        anchor_val = 0.0
        if anchor is not None:
            a = anchor(trainable_module)
            loss = loss + weights.anchor * a
            anchor_val = a.detach().item()

        backward_started = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, float(cfg.training.clip_grad_norm))
        if profiling and device.startswith("cuda"):
            torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_started
        optimizer_started = time.perf_counter()
        opt.step()
        sched.step()
        if profiling and device.startswith("cuda"):
            torch.cuda.synchronize(device)
        optimizer_seconds = time.perf_counter() - optimizer_started

        if profiling:
            measured = {
                "data": data_seconds,
                **(part_timings or {}),
                "backward_and_clip": backward_seconds,
                "optimizer": optimizer_seconds,
                "total": time.perf_counter() - step_started,
            }
            for key, value in measured.items():
                profile_totals[key] = profile_totals.get(key, 0.0) + value
            print("[profile step %d] %s" % (step, " ".join(f"{key}={value:.4f}s" for key, value in measured.items())))
            if step == profile_steps:
                means = {key: value / profile_steps for key, value in profile_totals.items()}
                print("[profile mean] " + " ".join(f"{key}={value:.4f}s" for key, value in means.items()))
                _log({f"profile/{key}_seconds": value for key, value in means.items()}, step)

        if step % int(cfg.training.log_every) == 0:
            dt = (time.time() - t_last) / int(cfg.training.log_every)
            t_last = time.time()
            payload = {
                "train/loss": loss.detach().item(),
                "train/nll": res["nll"].detach().item(),
                "train/crps": res["crps"].detach().item(),
                "train/pinball": res["pinball"].detach().item(),
                "train/distill": res["distill"].detach().item(),
                "train/raw_crossing_frac": res["raw_crossing_frac"],
                "train/anchor": anchor_val,
                "train/grad_norm": gnorm.detach().item(),
                "train/lr": sched.get_last_lr()[0],
                "train/sec_per_step": dt,
                "train/is_era5_batch": float(use_era5),
                "train/P": int(episodes[0]["x_norm_train"].shape[0]),
            }
            if not use_era5:
                payload["train/nll_oracle"] = res["oracle_nll"]
                payload["train/nll_gap_to_oracle"] = res["nll_gap_to_oracle"]
            _log(payload, step)
            print(
                f"step {step:>7} loss={loss.detach().item():.4f} "
                f"nll={res['nll'].detach().item():.4f} "
                f"distill={res['distill'].detach().item():.4f} "
                f"pinball={res['pinball'].detach().item():.4f} "
                f"cross={res['raw_crossing_frac']:.3%} "
                f"gap={res.get('nll_gap_to_oracle', float('nan')):.4f} "
                f"lr={sched.get_last_lr()[0]:.2e} {dt:.2f}s/step" + ("  [era5]" if use_era5 else "")
            )

        hooks_started = time.time()
        if step % int(cfg.training.val_every) == 0:
            _consider_validation(step, _validate(step))
        if step % int(cfg.training.save_every) == 0:
            _save(step)
        # Do not charge validation/checkpoint I/O to the next sec_per_step window.
        t_last += time.time() - hooks_started

    if total_steps % int(cfg.training.val_every) != 0:
        _consider_validation(total_steps, _validate(total_steps))

    if bool(cfg.training.get("restore_best", True)):
        _restore_trainable(best_state)
        print(f"[selection] restored step {best_step} with {selection_metric}={best_value:.6f} before final export")
        export_step = best_step
    else:
        export_step = total_steps
    final = _save(export_step, tag="_final")
    if final:
        print(
            "\nPhase A done. Use it as the copula run's marginal with:\n"
            f"    python -m copula_inter.train tabicl.pit_ckpt={os.path.abspath(final)}\n"
            "and measure it first with:\n"
            f"    python eval/runners/marginal_calibration_eval.py --ckpt {os.path.abspath(final)}"
        )
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
