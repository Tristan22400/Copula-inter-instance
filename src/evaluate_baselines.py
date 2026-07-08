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
  gp_mle_ard_rbf     : Same, with one lengthscale per input dimension (ARD)
  gp_mle_matern32    : GP posterior with MLE-fitted Matérn-3/2 kernel
  gp_mle_ard_matern32: Matérn-3/2 with ARD lengthscales
  gp_mle_periodic    : GP posterior with MLE-fitted Periodic kernel (+ period)
  gp_mle_ard_periodic: Periodic with one lengthscale + period per input dimension (ARD)
  gp_mle_rq          : GP posterior with MLE-fitted Rational Quadratic (+ rq_α)
  gp_mle_ard_rq      : Rational Quadratic with ARD lengthscales
  gp_mle_dot_product : GP posterior with MLE-fitted linear/dot-product kernel
                       (variance + noise term fitted)
  dkl_rbf/matern32/rq/dot_product :
                       Deep Kernel Learning — MLP(d_x→32→16) feature extractor
                       feeding a GP layer (chosen kernel), trained jointly by
                       maximising the marginal log-likelihood
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
        [--n_steps_mle 300]       # Adam steps for GP MLE fitting (also used for ARD variants)
        [--lr_mle 0.05]           # learning rate for GP MLE
        [--n_steps_dkl 300]       # Adam steps for Deep Kernel Learning (MLP+GP) fitting
        [--lr_dkl 0.01]           # learning rate for DKL Adam
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

import gpytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor
from torch.optim import Adam

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from data_gen import _sq_dist, sigma_to_correlation
from dataset import CopulaDataset
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
# GP baselines (via GPyTorch)
# ---------------------------------------------------------------------------
# GPyTorch kernels register their hyperparameters as ordinary constrained
# nn.Parameters, so a standard ExactGP + ExactMarginalLogLikelihood + Adam
# loop backprops through lengthscale/outputscale/period/alpha fine on its
# own — no hand-rolled kernel formulas or NaN-safe distance helpers needed.
# (The previous hand-rolled version existed because gpytorch's *setters*,
# e.g. `kernel.lengthscale = value`, do a non-differentiable in-place copy;
# that only breaks autograd if you call the setter mid-training-loop, which
# this code never does — hyperparameters are optimised as registered
# parameters throughout.)

_ARD_ELIGIBLE = {
    "rbf": True,
    "matern32": True,
    # GPyTorch's PeriodicKernel(ard_num_dims=d_x) fits both a per-dimension
    # lengthscale and a per-dimension period_length, so ARD is meaningful here
    # too (unlike the old hand-rolled formula, which collapsed to a scalar
    # Euclidean distance before applying period/lengthscale).
    "periodic": True,
    "rational_quadratic": True,
    "dot_product": False,
}


class _ExactGPModel(gpytorch.models.ExactGP):
    """ExactGP wrapper over one of the five baseline kernels, optionally
    preceded by a learned feature extractor (Deep Kernel Learning)."""

    def __init__(
        self,
        X_train: Tensor,
        z_train: Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel_name: str,
        ard_num_dims: int | None = None,
        feature_extractor: nn.Module | None = None,
    ) -> None:
        super().__init__(X_train, z_train, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ZeroMean()

        # PeriodicKernel reads `kwargs.get("ard_num_dims", 1)` directly rather
        # than through the base Kernel class's None-handling, so passing
        # ard_num_dims=None explicitly (vs. omitting it) breaks it. Omit the
        # kwarg entirely rather than pass None, for every kernel uniformly.
        ard_kw = {} if ard_num_dims is None else {"ard_num_dims": ard_num_dims}

        if kernel_name == "rbf":
            base = gpytorch.kernels.RBFKernel(**ard_kw)
        elif kernel_name == "matern32":
            base = gpytorch.kernels.MaternKernel(nu=1.5, **ard_kw)
        elif kernel_name == "periodic":
            base = gpytorch.kernels.PeriodicKernel(**ard_kw)
        elif kernel_name == "rational_quadratic":
            base = gpytorch.kernels.RQKernel(**ard_kw)
        elif kernel_name == "dot_product":
            base = gpytorch.kernels.LinearKernel()
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")

        # LinearKernel already carries its own learnable `variance` scale;
        # wrapping it in ScaleKernel too would just add a second, unidentifiable
        # scale factor on top.
        self.covar_module = (
            base if kernel_name == "dot_product" else gpytorch.kernels.ScaleKernel(base)
        )

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def fit_and_eval_gpytorch(
    X_train: Tensor,
    z_train: Tensor,
    X_test: Tensor,
    kernel_name: str,
    n_steps: int,
    lr: float,
    ard: bool = False,
    feature_extractor: nn.Module | None = None,
    jitter: float = 1e-6,
) -> Tensor:
    """Fit a GP (optionally over a learned feature extractor, i.e. DKL) by
    maximising the exact marginal log-likelihood, and return the posterior
    correlation matrix at X_test.

    kernel_name: one of rbf | matern32 | periodic | rational_quadratic | dot_product.
    ard: one lengthscale per input dimension instead of a single shared scalar
         (only meaningful for rbf/matern32/rational_quadratic; see _ARD_ELIGIBLE).
         Always ignored (no ARD) for dot_product, which has no lengthscale.
    feature_extractor: if given, the GP is fit on feature_extractor(X) instead
         of X directly (Deep Kernel Learning); "periodic" is not PD in a fixed-
         dimensional latent space and is rejected in that case.

    Folding observation noise into both the training and test covariance
    mirrors data_gen.gp_posterior's latent=False convention used to build the
    oracle R_star: `likelihood(model(X_test))` returns Cov(f*|D) + noise*I,
    which for dot_product (kernel rank <= d_x, frequently < N) is what keeps
    the returned covariance full rank instead of numerically singular.
    """
    if kernel_name == "periodic" and feature_extractor is not None:
        raise ValueError("kernel_name='periodic' is not PD in a >1D DKL latent space")

    d_x = X_train.shape[1]
    ard_num_dims = d_x if (ard and kernel_name != "dot_product") else None

    # Interval mirrors the previous hand-rolled code's log_noise clamp range
    # (exp(-8)..exp(2)) — keeps optimisation stable / prevents noise collapsing
    # to (near-)zero, which is what made the old dot_product fit ill-conditioned.
    noise_constraint = gpytorch.constraints.Interval(math.exp(-8.0), math.exp(2.0))
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
    likelihood.noise = 0.1

    model = _ExactGPModel(
        X_train, z_train, likelihood, kernel_name,
        ard_num_dims=ard_num_dims, feature_extractor=feature_extractor,
    ).to(X_train.device)
    # likelihood is registered as a submodule of model (gpytorch.models.ExactGP
    # sets self.likelihood = likelihood), so model.to(...) above already moved
    # it in place — no separate likelihood.to(...) needed.

    model.train()
    likelihood.train()
    # gpytorch.models.ExactGP.__init__ already stores `self.likelihood =
    # likelihood`, registering it as a submodule — model.parameters() already
    # includes the noise parameter; adding likelihood.parameters() on top
    # would just duplicate it in the optimizer's param groups.
    opt = Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(n_steps):
        opt.zero_grad()
        loss = -mll(model(X_train), z_train)
        loss.backward()
        opt.step()

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        Sigma_post = likelihood(model(X_test)).covariance_matrix
        N = X_test.shape[0]
        Sigma_post = 0.5 * (Sigma_post + Sigma_post.T) + jitter * torch.eye(
            N, dtype=Sigma_post.dtype, device=Sigma_post.device
        )

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


# ---------------------------------------------------------------------------
# Deep Kernel Learning (DKL)
# ---------------------------------------------------------------------------

# _MLP is already exactly Linear(d_x,32) -> SiLU -> Dropout -> Linear(32,16)
# when instantiated with dropout=0.0 (a no-op) — reused here under a
# descriptive alias rather than duplicating the class. DKL itself is just
# fit_and_eval_gpytorch(..., feature_extractor=DKLFeatureExtractor(...)):
# the GP hyperparameters on the MLP's latent space use a single scalar
# lengthscale (ard=False) — the MLP's own linear output layer can already
# learn an arbitrary per-latent-dimension scaling, so a separate ARD
# lengthscale would be redundant and would only add optimisation instability.
# "periodic" is excluded by the caller (_eval_episode): it is only a valid PD
# kernel in 1D, but the latent space has a fixed dimensionality, so it can't
# be routed around the way ARD-periodic is for the raw-input GP baselines.
DKLFeatureExtractor = _MLP


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

        # NOTE: zero-initializing head.weight is a dead end here — Sigma is
        # built from W @ W.T (a bilinear form in the first r head outputs),
        # so dSigma/dW ∝ W vanishes exactly at W=0, and the unit-diagonal
        # normalization makes Sigma == I regardless of s when W=0. Both paths
        # have zero gradient, so the model can never leave Sigma=I. Use a
        # small random init to break the saddle point.
        nn.init.normal_(self.head.weight, std=1e-2)
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

    # r is normally icl_rank (the pretrained model's rank, e.g. 32) — sized for
    # a model pretrained across millions of episodes. Trained from scratch on a
    # single episode's P instances (as few as ~13, ~8 after the support/query
    # split below), that many free low-rank factors overfits badly: verified
    # empirically that more training steps at r=32 makes some episodes *worse*
    # (correlation matrix collapses to near-singular, off-the-charts NLL),
    # while capping r relative to P keeps it stable. r=4 (the class default)
    # is never exceeded for very small P.
    r = max(2, min(r, P // 4))

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
    ep: dict,
    icl_model: nn.Module,
    icl_rank: int,
    n_steps_mle: int,
    lr_mle: float,
    n_steps_dkl: int,
    lr_dkl: float,
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

    # --- GP MLE baselines (plain + ARD for lengthscale kernels) ---
    _GP_KERNELS = ["rbf", "matern32", "periodic", "rational_quadratic", "dot_product"]
    _LABEL_MAP  = {
        ("rbf", False):                "gp_mle_rbf",
        ("rbf", True):                 "gp_mle_ard_rbf",
        ("matern32", False):           "gp_mle_matern32",
        ("matern32", True):            "gp_mle_ard_matern32",
        ("periodic", False):           "gp_mle_periodic",
        ("periodic", True):            "gp_mle_ard_periodic",
        ("rational_quadratic", False): "gp_mle_rq",
        ("rational_quadratic", True):  "gp_mle_ard_rq",
        ("dot_product", False):        "gp_mle_dot_product",
    }
    # GPyTorch's PeriodicKernel sums per-dimension sin² terms inside a single
    # exp() (a product of per-dimension periodic kernels), which is valid PSD
    # for any d_x — unlike the old hand-rolled formula (Euclidean distance
    # collapsed to a scalar first), which was only PSD for d_x == 1. No gate
    # needed here anymore. ARD is meaningful too (see _ARD_ELIGIBLE): it fits
    # a per-dimension lengthscale *and* period_length instead of one shared pair.
    for kname in _GP_KERNELS:
        for ard in ([False, True] if _ARD_ELIGIBLE[kname] else [False]):
            label = _LABEL_MAP[(kname, ard)]
            try:
                R_gp = fit_and_eval_gpytorch(X_train, z_train, X_test, kname,
                                              n_steps=n_steps_mle, lr=lr_mle, ard=ard)
                nlls[label]  = _corr_nll_single(R_gp, z_test)
                R_dict[label] = R_gp
            except Exception as exc:
                print(f"  [{label}] failed: {exc}")
                nlls[label]  = float("nan")
                R_dict[label] = R_I.clone()

    # --- Deep Kernel Learning (MLP + GP, jointly trained), across multiple kernels ---
    # "periodic" excluded: not PD in the fixed 16-dim latent space at any dimensionality.
    _DKL_KERNELS   = ["rbf", "matern32", "rational_quadratic", "dot_product"]
    _DKL_LABEL_MAP = {
        "rbf":                "dkl_rbf",
        "matern32":           "dkl_matern32",
        "rational_quadratic": "dkl_rq",
        "dot_product":        "dkl_dot_product",
    }
    for kname in _DKL_KERNELS:
        label = _DKL_LABEL_MAP[kname]
        try:
            mlp = DKLFeatureExtractor(d_x, hidden=32, out_dim=16, dropout=0.0).to(device)
            R_dkl = fit_and_eval_gpytorch(X_train, z_train, X_test, kname,
                                           n_steps=n_steps_dkl, lr=lr_dkl,
                                           ard=False, feature_extractor=mlp)
            nlls[label]  = _corr_nll_single(R_dkl, z_test)
            R_dict[label] = R_dkl
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
    ("independence",        "Independence"),
    ("gp_prior_rbf",        "GP-Prior-RBF"),
    ("gp_mle_rbf",          "GP-MLE-RBF"),
    ("gp_mle_ard_rbf",      "GP-MLE-ARD-RBF"),
    ("gp_mle_matern32",     "GP-MLE-Matern32"),
    ("gp_mle_ard_matern32", "GP-MLE-ARD-Matern32"),
    ("gp_mle_periodic",     "GP-MLE-Periodic"),
    ("gp_mle_ard_periodic", "GP-MLE-ARD-Periodic"),
    ("gp_mle_rq",           "GP-MLE-RQ"),
    ("gp_mle_ard_rq",       "GP-MLE-ARD-RQ"),
    ("gp_mle_dot_product",  "GP-MLE-DotProduct"),
    ("dkl_rbf",             "Deep Kernel Learning (RBF)"),
    ("dkl_matern32",        "Deep Kernel Learning (Matern32)"),
    ("dkl_rq",              "Deep Kernel Learning (RQ)"),
    ("dkl_dot_product",     "Deep Kernel Learning (DotProduct)"),
    ("per_ep_transformer",  "PerEp-Transformer"),
    ("icl",                 "ICL (pretrained)"),
    ("oracle",              "Oracle"),
]


def _print_table(all_nlls: list[dict[str, float]]) -> None:
    means = {k: float(np.nanmean([m.get(k, float("nan")) for m in all_nlls]))
             for k, _ in _METHOD_ORDER}
    stds  = {k: float(np.nanstd( [m.get(k, float("nan")) for m in all_nlls]))
             for k, _ in _METHOD_ORDER}

    col = max(22, max(len(label) for _, label in _METHOD_ORDER) + 2)
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
    parser.add_argument("--n_steps_mle",  type=int,   default=1000,
                        help="Adam steps for GP kernel MLE fitting (also used for ARD variants)")
    parser.add_argument("--lr_mle",       type=float, default=0.05,
                        help="Learning rate for GP MLE Adam")
    parser.add_argument("--n_steps_dkl",  type=int,   default=5000,
                        help="Adam steps for Deep Kernel Learning (MLP+GP) fitting")
    parser.add_argument("--lr_dkl",       type=float, default=0.01,
                        help="Learning rate for DKL Adam")
    parser.add_argument("--n_steps_per_ep", type=int, default=5000,
                        help="Training steps for PerEpisodeTransformer")
    parser.add_argument("--patience_per_ep", type=int, default=500,
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
    icl_model.load_state_dict(ckpt.get("model_state", ckpt.get("state_dict")))
    icl_model.eval()
    icl_rank = int(icl_cfg.model.rank)
    n_params = sum(p.numel() for p in icl_model.parameters())
    print(f"ICL model parameters: {n_params:,}  rank={icl_rank}")

    dataset_dir = cfg.training.dataset_dir
    n_ep = args.n_episodes

    dataset = CopulaDataset(episode_dir=dataset_dir)
    n_available = len(dataset)

    all_nlls: list[dict[str, float]] = []
    plot_R_dict: dict[str, Tensor] | None = None
    plot_R_oracle: Tensor | None = None

    print(f"\nEvaluating {n_ep} episodes from {dataset_dir} (start={args.episode_idx})")
    print(f"  Dataset size: {n_available} episodes")
    print(f"  GP MLE: {args.n_steps_mle} steps | DKL: {args.n_steps_dkl} steps | "
          f"PerEp: {args.n_steps_per_ep} steps (patience={args.patience_per_ep})")

    for local_i in range(n_ep):
        ep_i = args.episode_idx + local_i
        if ep_i >= n_available:
            print(f"  [ep {ep_i}] index out of range ({n_available} available), skipping")
            continue

        ep = dataset[ep_i]
        nlls, R_dict, R_oracle = _eval_episode(
            ep=ep,
            icl_model=icl_model,
            icl_rank=icl_rank,
            n_steps_mle=args.n_steps_mle,
            lr_mle=args.lr_mle,
            n_steps_dkl=args.n_steps_dkl,
            lr_dkl=args.lr_dkl,
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
