"""
evaluate_baselines.py — Baseline comparison for inter-instance copula.

Compares the pretrained CopulaTabICL checkpoint against classical baselines
and a per-episode trained small transformer on held-out PIT episodes.

Methods
-------
  independence       : R = I_N (copula NLL = 0.0 always, reference point)
  gp_prior_rbf       : RBF prior correlation at test points (median bandwidth,
                       no conditioning on z_train)
  gp_mle_rbf         : GP posterior with MLE-fitted RBF {l, α², σ²_n}
  gp_mle_matern32    : GP posterior with MLE-fitted Matérn-3/2 kernel
  gp_mle_periodic    : GP posterior with MLE-fitted Periodic kernel (+ period)
  gp_mle_rq          : GP posterior with MLE-fitted Rational Quadratic (+ rq_α)
  per_ep_transformer : Small set-transformer trained from scratch on this episode
  icl                : Pretrained CopulaTabICL checkpoint (in-context learning)
  oracle             : Ground-truth R_star from episode file (lower bound)

Usage
-----
    python src/evaluate_baselines.py \\
        --config conf/config.yaml \\
        --ckpt   ./checkpoints/copula_transformer/step_XXXXXX_final.pt \\
        [--n_episodes 50]         # episodes to evaluate
        [--episode_idx 0]         # starting episode index
        [--n_steps_mle 300]       # Adam steps for GP MLE fitting
        [--lr_mle 0.05]           # learning rate for GP MLE
        [--n_steps_per_ep 500]    # training steps for PerEpisodeTransformer
        [--patience_per_ep 100]   # early stopping patience (steps without improvement)
        [--plot_episode 0]        # local episode index to plot corr_grid for
        [--out_dir ./plots]       # directory to save corr_grid figure
        [--device auto]
        [--seed 42]
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor
from torch.optim import Adam

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from data_gen import (
    _sq_dist,
    _dist,
    matern32_kernel,
    periodic_kernel,
    rational_quadratic_kernel,
    rbf_kernel,
    sigma_to_correlation,
)
from loss import _safe_cholesky, oracle_copula_nll
from model import build_copula_transformer, low_rank_correlation


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# GP baselines
# ---------------------------------------------------------------------------


def _gp_lml(
    z_train: Tensor,  # (P,)
    K: Tensor,        # (P, P) kernel matrix (already includes noise on diagonal)
) -> Tensor:
    """Gaussian log marginal likelihood: -0.5*(log|K| + z^T K^{-1} z + P log 2π)."""
    P = z_train.shape[0]
    L = _safe_cholesky(K)
    log_det = 2.0 * L.diagonal().log().sum()
    alpha = torch.cholesky_solve(z_train.unsqueeze(-1), L).squeeze(-1)
    quad = z_train @ alpha
    return -0.5 * (log_det + quad + P * math.log(2.0 * math.pi))


def gp_mle_fit(
    X_train: Tensor,
    z_train: Tensor,
    kernel_name: str,
    n_steps: int = 300,
    lr: float = 0.05,
) -> dict:
    """Fit GP kernel hyperparameters by maximising the log marginal likelihood.

    Returns a dict of fitted scalar tensors: l, alpha2, noise (and period or
    rq_alpha for kernels that need them).
    """
    dtype, device = X_train.dtype, X_train.device
    P = X_train.shape[0]
    I = torch.eye(P, dtype=dtype, device=device)

    log_l      = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
    log_alpha2 = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
    log_noise  = nn.Parameter(torch.full((1,), -2.0, dtype=dtype, device=device))

    extra_params: list[nn.Parameter] = []
    if kernel_name == "periodic":
        log_period = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
        extra_params.append(log_period)
    elif kernel_name == "rational_quadratic":
        log_rq_alpha = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
        extra_params.append(log_rq_alpha)

    opt = Adam([log_l, log_alpha2, log_noise] + extra_params, lr=lr)

    for _ in range(n_steps):
        opt.zero_grad()

        l       = log_l.clamp(-4.0, 4.0).exp()
        alpha2  = log_alpha2.clamp(-3.0, 3.0).exp()
        noise   = log_noise.clamp(-8.0, 2.0).exp()

        if kernel_name == "rbf":
            K_prior = alpha2 * torch.exp(-_sq_dist(X_train, X_train) / (2.0 * l**2))
        elif kernel_name == "matern32":
            r = _dist(X_train, X_train)
            s = math.sqrt(3.0) * r / l
            K_prior = alpha2 * (1.0 + s) * torch.exp(-s)
        elif kernel_name == "periodic":
            period  = log_period.clamp(-2.0, 2.0).exp()   # type: ignore[possibly-undefined]
            r = _dist(X_train, X_train)
            K_prior = alpha2 * torch.exp(-2.0 * torch.sin(math.pi * r / period) ** 2 / l**2)
        elif kernel_name == "rational_quadratic":
            rq_a   = log_rq_alpha.clamp(-2.0, 2.0).exp()  # type: ignore[possibly-undefined]
            sq     = _sq_dist(X_train, X_train)
            K_prior = alpha2 * (1.0 + sq / (2.0 * rq_a * l**2)) ** (-rq_a)
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")

        K = K_prior + noise * I
        lml = _gp_lml(z_train, K)
        (-lml).backward()
        opt.step()

    with torch.no_grad():
        result = {
            "l":      log_l.clamp(-4.0, 4.0).exp().item(),
            "alpha2": log_alpha2.clamp(-3.0, 3.0).exp().item(),
            "noise":  log_noise.clamp(-8.0, 2.0).exp().item(),
        }
        if kernel_name == "periodic":
            result["period"]   = log_period.clamp(-2.0, 2.0).exp().item()   # type: ignore[possibly-undefined]
        elif kernel_name == "rational_quadratic":
            result["rq_alpha"] = log_rq_alpha.clamp(-2.0, 2.0).exp().item() # type: ignore[possibly-undefined]
    return result


def gp_posterior_corr(
    X_train: Tensor,
    z_train: Tensor,
    X_test: Tensor,
    kernel_name: str,
    params: dict,
    jitter: float = 1e-6,
) -> Tensor:
    """GP posterior correlation matrix at test points given training context.

    Args:
        X_train : (P, d_x)
        z_train : (P,)
        X_test  : (N, d_x)
        kernel_name: one of rbf | matern32 | periodic | rational_quadratic
        params  : dict with l, alpha2, noise (+ period or rq_alpha)
        jitter  : added to posterior covariance diagonal for numerical stability

    Returns:
        R : (N, N) correlation matrix
    """
    P, N = X_train.shape[0], X_test.shape[0]
    dtype, device = X_train.dtype, X_train.device
    l, alpha2, noise = params["l"], params["alpha2"], params["noise"]
    period   = params.get("period", 1.0)
    rq_alpha = params.get("rq_alpha", 1.0)

    kw = dict(l=l, alpha2=alpha2, period=period, rq_alpha=rq_alpha)

    _KERNELS = {
        "rbf": rbf_kernel,
        "matern32": matern32_kernel,
        "periodic": periodic_kernel,
        "rational_quadratic": rational_quadratic_kernel,
    }
    kfn = _KERNELS[kernel_name]

    with torch.no_grad():
        K_ff = kfn(X_train, X_train, **kw) + noise * torch.eye(P, dtype=dtype, device=device)
        K_sf = kfn(X_test, X_train, **kw)    # (N, P)
        K_ss = kfn(X_test, X_test, **kw)     # (N, N)

        L_ff = _safe_cholesky(K_ff)
        V = torch.linalg.solve_triangular(L_ff, K_sf.T, upper=False)  # (P, N)
        Sigma_post = K_ss - V.T @ V + jitter * torch.eye(N, dtype=dtype, device=device)
        Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)

    R, _ = sigma_to_correlation(Sigma_post)
    return R


def gp_prior_corr_rbf(X_test: Tensor) -> Tensor:
    """RBF prior correlation at test points with median bandwidth (no training data)."""
    N = X_test.shape[0]
    sq = _sq_dist(X_test, X_test)
    h2 = torch.pdist(X_test).pow(2).median().clamp(min=1e-6)
    R = torch.exp(-sq / (2.0 * h2))
    R = R / R.diagonal().clamp(min=1e-8).sqrt().unsqueeze(-1)
    R = R / R.diagonal().clamp(min=1e-8).sqrt().unsqueeze(-2)
    return R


# ---------------------------------------------------------------------------
# Per-episode small transformer
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _SelfAttn(nn.Module):
    def __init__(self, m: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(m)
        self.norm2 = nn.LayerNorm(m)
        self.attn = nn.MultiheadAttention(m, n_heads, dropout=dropout, batch_first=True)
        d_ff = max(round(8 / 3 * m / 32) * 32, 32)
        self.ff = nn.Sequential(
            nn.Linear(m, d_ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_ff, m)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class _CrossAttn(nn.Module):
    def __init__(self, m: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q  = nn.LayerNorm(m)
        self.norm_kv = nn.LayerNorm(m)
        self.attn = nn.MultiheadAttention(m, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        h, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
                         need_weights=False)
        return q + self.drop(h)


class PerEpisodeTransformer(nn.Module):
    """Small set-to-correlation transformer, trained from scratch on a single episode.

    Forward:
        X_ctx : (n_sup, d_x),  z_ctx : (n_sup,)  — training context
        X_qry : (n_qry, d_x)                     — query features
        → W   : (n_qry, r),   s : (n_qry,)

    The (W, s) pair feeds into ``low_rank_correlation`` (from model.py) to produce
    the (n_qry × n_qry) inter-instance correlation matrix, matching the CopulaTabICL
    output convention.
    """

    def __init__(
        self,
        d_x: int,
        m: int = 32,
        r: int = 4,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.r = r

        self.x_enc   = _MLP(d_x, m, m, dropout)
        self.row_enc = _MLP(m + 1, m, m, dropout)
        self.self_attn = nn.ModuleList([_SelfAttn(m, n_heads, dropout) for _ in range(n_layers)])
        self.W_q       = nn.Linear(m, m)
        self.cross_attn = _CrossAttn(m, n_heads, dropout)
        self.head      = nn.Linear(m, r + 1)

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, X_ctx: Tensor, z_ctx: Tensor, X_qry: Tensor) -> tuple[Tensor, Tensor]:
        n_sup, n_qry = X_ctx.shape[0], X_qry.shape[0]

        ex = self.x_enc(X_ctx)                                          # (n_sup, m)
        row = self.row_enc(torch.cat([ex, z_ctx.unsqueeze(-1)], dim=-1))  # (n_sup, m)
        row = row.unsqueeze(0)                                          # (1, n_sup, m)
        for block in self.self_attn:
            row = block(row)

        eq = self.x_enc(X_qry).unsqueeze(0)                            # (1, n_qry, m)
        q_emb = self.W_q(eq)
        h = self.cross_attn(q_emb, row).squeeze(0)                     # (n_qry, m)

        out = self.head(h)                                              # (n_qry, r+1)
        W = out[:, : self.r]                                            # (n_qry, r)
        s = out[:, self.r]                                              # (n_qry,)
        return W, s


def _corr_nll_single(R: Tensor, z: Tensor) -> float:
    """Copula NLL for a single (N, N) correlation matrix and (N,) z-vector."""
    N = z.shape[0]
    mask = torch.ones(1, N, dtype=torch.bool, device=z.device)
    return oracle_copula_nll(R.unsqueeze(0), z.unsqueeze(0), mask).item()


def train_per_episode(
    X_train: Tensor,
    z_train: Tensor,
    r: int,
    n_steps: int = 500,
    lr: float = 1e-3,
    patience: int = 100,
    val_every: int = 10,
    device: torch.device = torch.device("cpu"),
) -> PerEpisodeTransformer:
    """Train a PerEpisodeTransformer on one episode's training instances.

    Uses a fixed 20% val split for early stopping; the remaining 80% pool is
    randomly split 80/20 into support/query at each training step.
    """
    d_x = X_train.shape[1]
    P   = X_train.shape[0]

    n_val  = max(2, int(round(0.2 * P)))
    perm   = torch.randperm(P, device=device)
    val_idx, pool_idx = perm[:n_val], perm[n_val:]

    X_val,  z_val  = X_train[val_idx],  z_train[val_idx]
    X_pool, z_pool = X_train[pool_idx], z_train[pool_idx]
    n_pool = X_pool.shape[0]

    model = PerEpisodeTransformer(d_x, r=r).to(device)
    opt   = Adam(model.parameters(), lr=lr)

    best_val  = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    model.train()
    for step in range(n_steps):
        n_sup = max(1, int(round(0.8 * n_pool)))
        perm_p = torch.randperm(n_pool, device=device)
        X_s, z_s = X_pool[perm_p[:n_sup]], z_pool[perm_p[:n_sup]]
        X_q, z_q = X_pool[perm_p[n_sup:]], z_pool[perm_p[n_sup:]]

        if X_q.shape[0] < 2:
            continue

        W, s = model(X_s, z_s, X_q)
        Sigma = low_rank_correlation(W.unsqueeze(0), s.unsqueeze(0)).squeeze(0)
        mask  = torch.ones(1, X_q.shape[0], dtype=torch.bool, device=device)
        loss  = oracle_copula_nll(Sigma.unsqueeze(0), z_q.unsqueeze(0), mask)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % val_every == 0:
            model.eval()
            with torch.no_grad():
                W_v, s_v = model(X_pool, z_pool, X_val)
                Sv = low_rank_correlation(W_v.unsqueeze(0), s_v.unsqueeze(0)).squeeze(0)
                val_nll = _corr_nll_single(Sv, z_val)
            model.train()

            if val_nll < best_val - 1e-4:
                best_val   = val_nll
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += val_every

            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------


def plot_corr_grid(
    estimators: dict[str, Tensor],
    oracle_R: Tensor,
    title: str = "",
    max_show: int = 40,
) -> "plt.Figure":  # type: ignore[name-defined]
    """Side-by-side heatmaps of oracle R_star vs each estimator's predicted R.

    Args:
        estimators : {label: (N, N) tensor}
        oracle_R   : (N, N) tensor — ground-truth correlation
        title      : overall figure title
        max_show   : max N to display (subsampled if larger)
    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = ["oracle"] + list(estimators.keys())
    mats   = [oracle_R.cpu().float()] + [v.cpu().float() for v in estimators.values()]

    N = oracle_R.shape[0]
    if N > max_show:
        idx = torch.linspace(0, N - 1, max_show).long()
        mats = [m[idx][:, idx] for m in mats]

    n_cols = len(labels)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    for ax, lbl, R in zip(axes, labels, mats):
        R_np = R.numpy()
        sns.heatmap(
            R_np,
            ax=ax,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar=lbl == labels[-1],
        )
        color = "red" if lbl == "oracle" else "black"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2 if lbl == "oracle" else 1)
        ax.set_title(lbl, fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-episode evaluation
# ---------------------------------------------------------------------------


def _eval_episode(
    ep_path: str,
    icl_model: nn.Module,
    icl_rank: int,
    n_steps_mle: int,
    lr_mle: float,
    n_steps_per_ep: int,
    patience_per_ep: int,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Tensor], Tensor]:
    """Evaluate all methods on one episode.

    Returns:
        nlls      : {method_name: copula_nll_float}
        R_dict    : {method_name: (N, N) correlation tensor} — for plotting
        R_oracle  : (N, N) oracle correlation tensor
    """
    ep         = torch.load(ep_path, map_location=device, weights_only=True)
    X_train    = ep["x_norm_train"].to(device)   # (P, d_x)
    z_train    = ep["z_train"].to(device)         # (P,)
    X_test     = ep["x_norm_test"].to(device)     # (N, d_x)
    z_test     = ep["z_test"].to(device)          # (N,)
    R_oracle   = ep["R_star"].to(device)          # (N, N)

    P, d_x = X_train.shape
    N      = X_test.shape[0]
    nlls: dict[str, float]    = {}
    R_dict: dict[str, Tensor] = {}

    # --- independence ---
    R_I = torch.eye(N, dtype=X_train.dtype, device=device)
    nlls["independence"] = _corr_nll_single(R_I, z_test)
    R_dict["independence"] = R_I

    # --- GP prior RBF ---
    R_prior = gp_prior_corr_rbf(X_test)
    nlls["gp_prior_rbf"] = _corr_nll_single(R_prior, z_test)
    R_dict["gp_prior_rbf"] = R_prior

    # --- GP MLE baselines ---
    _GP_KERNELS = ["rbf", "matern32", "periodic", "rational_quadratic"]
    _LABEL_MAP  = {
        "rbf":                "gp_mle_rbf",
        "matern32":           "gp_mle_matern32",
        "periodic":           "gp_mle_periodic",
        "rational_quadratic": "gp_mle_rq",
    }
    for kname in _GP_KERNELS:
        label = _LABEL_MAP[kname]
        try:
            params = gp_mle_fit(X_train, z_train, kname,
                                n_steps=n_steps_mle, lr=lr_mle)
            R_gp = gp_posterior_corr(X_train, z_train, X_test, kname, params)
            nlls[label]  = _corr_nll_single(R_gp, z_test)
            R_dict[label] = R_gp
        except Exception as exc:
            print(f"  [{label}] failed: {exc}")
            nlls[label]  = float("nan")
            R_dict[label] = R_I.clone()

    # --- per-episode transformer ---
    try:
        per_ep_model = train_per_episode(
            X_train, z_train, r=icl_rank,
            n_steps=n_steps_per_ep, patience=patience_per_ep,
            device=device,
        )
        with torch.no_grad():
            W_te, s_te = per_ep_model(X_train, z_train, X_test)
            Sigma_te   = low_rank_correlation(W_te.unsqueeze(0), s_te.unsqueeze(0)).squeeze(0)
        nlls["per_ep_transformer"]  = _corr_nll_single(Sigma_te, z_test)
        R_dict["per_ep_transformer"] = Sigma_te
    except Exception as exc:
        print(f"  [per_ep_transformer] failed: {exc}")
        nlls["per_ep_transformer"]  = float("nan")
        R_dict["per_ep_transformer"] = R_I.clone()

    # --- ICL model ---
    try:
        train_mask = torch.ones(1, P, dtype=torch.bool, device=device)
        batch = {
            "x_train":   X_train.unsqueeze(0),
            "x_test":    X_test.unsqueeze(0),
            "z_train":   z_train.unsqueeze(0),
            "train_mask": train_mask,
        }
        with torch.no_grad():
            out   = icl_model(batch)
            Sigma_icl = low_rank_correlation(out["W"], out["s"])  # (1, N, N)
        R_icl = Sigma_icl[0, :N, :N]
        nlls["icl"]  = _corr_nll_single(R_icl, z_test)
        R_dict["icl"] = R_icl
    except Exception as exc:
        print(f"  [icl] failed: {exc}")
        nlls["icl"]  = float("nan")
        R_dict["icl"] = R_I.clone()

    # --- oracle ---
    nlls["oracle"]  = _corr_nll_single(R_oracle, z_test)
    R_dict["oracle"] = R_oracle

    return nlls, R_dict, R_oracle


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


_METHOD_ORDER = [
    ("independence",       "Independence"),
    ("gp_prior_rbf",       "GP-Prior-RBF"),
    ("gp_mle_rbf",         "GP-MLE-RBF"),
    ("gp_mle_matern32",    "GP-MLE-Matern32"),
    ("gp_mle_periodic",    "GP-MLE-Periodic"),
    ("gp_mle_rq",          "GP-MLE-RQ"),
    ("per_ep_transformer", "PerEp-Transformer"),
    ("icl",                "ICL (pretrained)"),
    ("oracle",             "Oracle"),
]


def _print_table(all_nlls: list[dict[str, float]]) -> None:
    means = {k: float(np.nanmean([m.get(k, float("nan")) for m in all_nlls]))
             for k, _ in _METHOD_ORDER}
    stds  = {k: float(np.nanstd( [m.get(k, float("nan")) for m in all_nlls]))
             for k, _ in _METHOD_ORDER}

    col = 22
    total = col + 2 * 12
    print(f"\n{'─' * total}")
    print(f"Inter-instance copula NLL (z-space) — lower is better  [N={len(all_nlls)} episodes]")
    print(f"{'─' * total}")
    print(f"{'Method':<{col}}{'Mean NLL':>12}{'Std NLL':>12}")
    print(f"{'─' * col}{'─' * 12}{'─' * 12}")
    for key, label in _METHOD_ORDER:
        m, s = means.get(key, float("nan")), stds.get(key, float("nan"))
        marker = ""
        if key == "icl":
            marker = "  ← our model"
        elif key == "oracle":
            marker = "  ← lower bound"
        print(f"{label:<{col}}{m:>12.4f}{s:>12.4f}{marker}")
    print(f"{'─' * total}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ICL checkpoint vs baselines on inter-instance copula episodes"
    )
    parser.add_argument("--config",       default="conf/config.yaml")
    parser.add_argument("--ckpt",         required=True)
    parser.add_argument("--n_episodes",   type=int,   default=50)
    parser.add_argument("--episode_idx",  type=int,   default=0)
    parser.add_argument("--n_steps_mle",  type=int,   default=300,
                        help="Adam steps for GP kernel MLE fitting")
    parser.add_argument("--lr_mle",       type=float, default=0.05,
                        help="Learning rate for GP MLE Adam")
    parser.add_argument("--n_steps_per_ep", type=int, default=500,
                        help="Training steps for PerEpisodeTransformer")
    parser.add_argument("--patience_per_ep", type=int, default=100,
                        help="Early stopping patience for PerEpisodeTransformer")
    parser.add_argument("--plot_episode", type=int,   default=0,
                        help="Local episode index to generate the corr_grid plot for")
    parser.add_argument("--out_dir",      default="./plots",
                        help="Directory for saved corr_grid figure")
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    _set_seed(args.seed)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    print(f"Device: {device}")

    cfg = OmegaConf.load(args.config)

    # ---- Load ICL model ----
    print(f"\nLoading ICL checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    icl_cfg = ckpt.get("cfg", cfg)
    if isinstance(icl_cfg, dict):
        icl_cfg = OmegaConf.create(icl_cfg)
    icl_model = build_copula_transformer(icl_cfg).to(device)
    icl_model.load_state_dict(ckpt["model_state"])
    icl_model.eval()
    icl_rank = int(icl_cfg.model.rank)
    n_params = sum(p.numel() for p in icl_model.parameters())
    print(f"ICL model parameters: {n_params:,}  rank={icl_rank}")

    dataset_dir = cfg.training.dataset_dir
    n_ep = args.n_episodes

    all_nlls: list[dict[str, float]] = []
    plot_R_dict: dict[str, Tensor] | None = None
    plot_R_oracle: Tensor | None = None

    print(f"\nEvaluating {n_ep} episodes from {dataset_dir} (start={args.episode_idx})")
    print(f"  GP MLE: {args.n_steps_mle} steps | PerEp: {args.n_steps_per_ep} steps "
          f"(patience={args.patience_per_ep})")

    for local_i in range(n_ep):
        ep_i   = args.episode_idx + local_i
        ep_path = os.path.join(dataset_dir, f"task_{ep_i:06d}.pt")
        if not os.path.exists(ep_path):
            print(f"  [ep {ep_i}] file not found, skipping")
            continue

        nlls, R_dict, R_oracle = _eval_episode(
            ep_path=ep_path,
            icl_model=icl_model,
            icl_rank=icl_rank,
            n_steps_mle=args.n_steps_mle,
            lr_mle=args.lr_mle,
            n_steps_per_ep=args.n_steps_per_ep,
            patience_per_ep=args.patience_per_ep,
            device=device,
        )
        all_nlls.append(nlls)

        if local_i == args.plot_episode:
            plot_R_dict   = R_dict
            plot_R_oracle = R_oracle

        icl_nll = nlls.get("icl", float("nan"))
        ora_nll = nlls.get("oracle", float("nan"))
        print(f"  ep {ep_i:04d}: icl={icl_nll:.4f}  oracle={ora_nll:.4f}")

    if not all_nlls:
        print("No episodes evaluated successfully.")
        return

    _print_table(all_nlls)

    # ---- Correlation heatmap ----
    if plot_R_dict is not None and plot_R_oracle is not None:
        import matplotlib
        matplotlib.use("Agg")

        os.makedirs(args.out_dir, exist_ok=True)
        # Exclude oracle from estimators dict (it's passed separately)
        estimators = {k: v for k, v in plot_R_dict.items() if k != "oracle"}
        fig = plot_corr_grid(
            estimators=estimators,
            oracle_R=plot_R_oracle,
            title=f"Correlation estimators — episode {args.episode_idx + args.plot_episode}",
        )
        out_path = os.path.join(args.out_dir, f"corr_grid_ep{args.plot_episode}.png")
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        print(f"Saved corr_grid to: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
