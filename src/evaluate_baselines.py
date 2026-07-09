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
from gpytorch.priors import GammaPrior, LogNormalPrior, Prior
from omegaconf import OmegaConf
from torch import Tensor
from torch.optim import Adam

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from data_gen import _sq_dist, sigma_to_correlation
from dataset import CopulaDataset
from loss import oracle_copula_nll
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
# GP baselines (GPyTorch)
# ---------------------------------------------------------------------------
# ExactGP + ExactMarginalLogLikelihood + Adam(model.parameters()) backprops
# through kernel hyperparameters (registered as ordinary constrained
# nn.Parameters) just fine — no hand-rolled kernel math or NaN-safe distance
# helpers needed, unlike an earlier version of this file which duplicated
# data_gen.py's kernel formulas here to work around gpytorch's `.lengthscale =`
# / `.outputscale =` setters (those do an in-place `.initialize()` copy that
# breaks autograd — a real footgun, but only if you use the setters inside a
# training loop, which we don't).

_ARD_ELIGIBLE = {
    "rbf": True,
    "matern32": True,
    # gpytorch.kernels.PeriodicKernel sums per-dimension sin² terms inside a
    # single exp() (a product of per-dimension periodic kernels), which is
    # PSD for any ard_num_dims — unlike the hand-rolled scalar-Euclidean-
    # distance formula this file used to have, which was only PD in 1D.
    "periodic": True,
    "rational_quadratic": True,
    "dot_product": False,
}

# Default hyperprior constants, mirroring data_gen.py's _kernel_prior_spec /
# _nugget_prior fallback defaults (the actual per-dataset values live in the
# checkpoint's own saved cfg.data and are threaded in via prior_cfg — these
# are only the fallback if a key happens to be missing there).
_DEFAULT_PRIOR_CFG: dict[str, float] = {
    "l_lognormal_loc": 0.0,
    "l_lognormal_scale": 0.7,
    "alpha2_gamma_concentration": 4.0,
    "alpha2_gamma_rate": 3.0,
    "period_lognormal_loc": math.log(1.2),
    "period_lognormal_scale": 0.4,
    "rq_alpha_gamma_concentration": 2.0,
    "rq_alpha_gamma_rate": 1.0,
    "nugget_lognormal_loc": -4.63,
    "nugget_lognormal_scale": 0.5,
}


def _kernel_priors(prior_cfg: dict, kernel_name: str, ard: bool = False) -> dict[str, Prior]:
    """Build the same LogNormal/Gamma hyperpriors data_gen.py's generative
    process samples ground-truth hyperparameters from (see
    data_gen._kernel_prior_spec / _nugget_prior), for MAP — instead of plain
    MLE — fitting of the GP baselines here.

    Plain MLE over an ARD lengthscale vector has no reason to prefer "large
    lengthscale" (irrelevant dimension) over "small lengthscale" (active
    dimension, poorly identified from limited context) when both explain the
    P training points similarly well — the marginal-likelihood surface is
    flat/multimodal in that direction. Registering these priors on the
    kernel/likelihood makes ExactMarginalLogLikelihood add their log-density
    automatically (gpytorch sums every registered prior's log_prob via
    Module.named_priors() — no separate loss-term plumbing needed), pulling
    the fit toward the same hyperparameter regime the episodes were actually
    generated from.

    ard=True omits the lengthscale prior specifically: LogNormal(l_loc, l_scale)
    describes the lengthscale of a dimension already known to be active — its
    median-1 pull actively discourages the "grow arbitrarily large" behaviour
    an ARD lengthscale needs to correctly flag a dimension as irrelevant.
    Measured on one rbf-prior-2 episode (10 dims, ~60-90% inactive):
    registering it homogenised all 10 fitted lengthscales into a narrow
    0.4-1.0 band (vs. a plain-MLE run that let 2 of them grow to ~4, correctly
    flagging those as inactive) and *reduced* recovered off-diagonal
    correlation (max |corr| 0.60 -> 0.40) despite restarts making the fit
    reliable/reproducible across inits. Outputscale/noise priors showed no
    such conflict (they aren't asked to do variable selection) and stay on
    unconditionally.
    """
    cfg = {**_DEFAULT_PRIOR_CFG, **prior_cfg}
    if kernel_name == "dot_product":
        return {"variance_prior": GammaPrior(cfg["alpha2_gamma_concentration"], cfg["alpha2_gamma_rate"])}
    priors: dict[str, Prior] = {
        "outputscale_prior": GammaPrior(cfg["alpha2_gamma_concentration"], cfg["alpha2_gamma_rate"]),
    }
    if not ard:
        priors["lengthscale_prior"] = LogNormalPrior(cfg["l_lognormal_loc"], cfg["l_lognormal_scale"])
    if kernel_name == "periodic":
        priors["period_length_prior"] = LogNormalPrior(cfg["period_lognormal_loc"], cfg["period_lognormal_scale"])
    elif kernel_name == "rational_quadratic":
        priors["alpha_prior"] = GammaPrior(cfg["rq_alpha_gamma_concentration"], cfg["rq_alpha_gamma_rate"])
    return priors


def _noise_prior(prior_cfg: dict) -> LogNormalPrior:
    cfg = {**_DEFAULT_PRIOR_CFG, **prior_cfg}
    return LogNormalPrior(cfg["nugget_lognormal_loc"], cfg["nugget_lognormal_scale"])


def _lengthscale_init_prior(prior_cfg: dict) -> LogNormalPrior:
    """Same LogNormal distribution _kernel_priors would've used for the
    lengthscale prior — used here only to sample a fresh *initial* value per
    restart when ard=True omits it as a registered MAP prior (see
    _kernel_priors' ard docs), so ARD restarts still diversify their starting
    point instead of all beginning from gpytorch's identical default init."""
    cfg = {**_DEFAULT_PRIOR_CFG, **prior_cfg}
    return LogNormalPrior(cfg["l_lognormal_loc"], cfg["l_lognormal_scale"])


def _randomize_init(
    model: "_ExactGPModel",
    kernel_priors: dict[str, Prior],
    kernel_name: str,
    lengthscale_init_prior: Prior | None = None,
) -> None:
    """Sample a fresh initial value for every registered prior's parameter —
    called once per restart (before .train()/optimizer construction, so the
    in-place `.initialize()` copy these setters do is safe; see the module
    note above about not using them mid-training) to diversify restarts
    instead of re-running Adam from the same fixed gpytorch default init
    every time.
    """
    base = model.covar_module if kernel_name == "dot_product" else model.covar_module.base_kernel
    # gpytorch's Kernel base class defines a `lengthscale` property on every
    # kernel (returning None for kernels with has_lengthscale=False, e.g.
    # LinearKernel) rather than raising AttributeError, so hasattr() alone
    # can't distinguish "no lengthscale" from "has one" — check the value too.
    has_lengthscale = getattr(base, "lengthscale", None) is not None
    device = base.lengthscale.device if has_lengthscale else next(model.parameters()).device
    ls_prior = kernel_priors.get("lengthscale_prior", lengthscale_init_prior)
    if ls_prior is not None and has_lengthscale:
        base.lengthscale = ls_prior.sample(base.lengthscale.shape).to(device)
    if "period_length_prior" in kernel_priors:
        base.period_length = kernel_priors["period_length_prior"].sample(base.period_length.shape).to(device)
    if "alpha_prior" in kernel_priors:
        base.alpha = kernel_priors["alpha_prior"].sample(base.alpha.shape).to(device)
    if "variance_prior" in kernel_priors:
        model.covar_module.variance = kernel_priors["variance_prior"].sample(model.covar_module.variance.shape).to(device)
    if "outputscale_prior" in kernel_priors:
        model.covar_module.outputscale = kernel_priors["outputscale_prior"].sample(model.covar_module.outputscale.shape).to(device)


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
        kernel_priors: dict[str, Prior] | None = None,
    ) -> None:
        super().__init__(X_train, z_train, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ZeroMean()
        kp = kernel_priors or {}

        # PeriodicKernel reads `kwargs.get("ard_num_dims", 1)` directly rather
        # than through the base Kernel class's None-handling, so passing
        # ard_num_dims=None explicitly (vs. omitting it) breaks it. Omit the
        # kwarg entirely rather than pass None, for every kernel uniformly.
        ard_kw = {} if ard_num_dims is None else {"ard_num_dims": ard_num_dims}

        if kernel_name == "rbf":
            base = gpytorch.kernels.RBFKernel(lengthscale_prior=kp.get("lengthscale_prior"), **ard_kw)
        elif kernel_name == "matern32":
            base = gpytorch.kernels.MaternKernel(nu=1.5, lengthscale_prior=kp.get("lengthscale_prior"), **ard_kw)
        elif kernel_name == "periodic":
            base = gpytorch.kernels.PeriodicKernel(
                lengthscale_prior=kp.get("lengthscale_prior"),
                period_length_prior=kp.get("period_length_prior"),
                **ard_kw,
            )
        elif kernel_name == "rational_quadratic":
            base = gpytorch.kernels.RQKernel(lengthscale_prior=kp.get("lengthscale_prior"), alpha_prior=kp.get("alpha_prior"), **ard_kw)
        elif kernel_name == "dot_product":
            base = gpytorch.kernels.LinearKernel(variance_prior=kp.get("variance_prior"))
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")

        # LinearKernel already has its own learnable `variance` scale;
        # wrapping it in ScaleKernel on top would just be a redundant,
        # unidentifiable second scale factor.
        self.covar_module = (
            base if kernel_name == "dot_product"
            else gpytorch.kernels.ScaleKernel(base, outputscale_prior=kp.get("outputscale_prior"))
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
    oracle_mode: str = "prior",
    prior_cfg: dict | None = None,
    n_restarts: int = 1,
) -> Tensor:
    """Fit a GP (optionally over a learned feature extractor, i.e. DKL) by
    maximising the exact marginal log-likelihood, and return the correlation
    matrix at X_test to compare against the episode's oracle R_star.

    oracle_mode must match how the episode's own R_star was built (see
    data_gen.py's oracle_mode branch):
      - "posterior": R_star conditions on (X_train, z_train), so we score the
        fitted kernel's true GP posterior at X_test — likelihood(model(X_test))
        folds observation noise into the returned covariance matrix, mirroring
        data_gen.gp_posterior's latent=False convention (needed so
        dot_product's rank-deficient K_ss — rank <= d_x, often < N — doesn't
        come out singular).
      - "prior": R_star ignores training conditioning entirely (raw kernel
        structure among test points only). Scoring the conditioned posterior
        here would answer a different question than what R_star asks — with P
        up to several hundred context points and a small nugget, the posterior
        shrinks off-diagonal correlation toward ~0 regardless of how well the
        kernel hyperparameters were identified, a systematic bias specific to
        GP-MLE/DKL (per_ep_transformer and the ICL model are trained
        end-to-end against this same R_star target, so they don't have this
        mismatch). Instead, evaluate the *fitted* kernel's own prior
        covariance at X_test — model.forward(X_test) bypasses
        ExactGP.__call__'s posterior conditioning and returns
        mean_module/covar_module evaluated directly at X_test, then
        likelihood(...) adds the fitted noise, exactly mirroring data_gen.py's
        Sigma_star = K_ss (kernel + nugget, no conditioning).

    prior_cfg / n_restarts: registers the same LogNormal/Gamma hyperpriors
    data_gen.py's generative process uses (see _kernel_priors), turning plain
    MLE into MAP, and repeats the fit from n_restarts independent random
    inits (each sampled from those same priors — see _randomize_init),
    keeping whichever restart reaches the best final training loss. Plain MLE
    over an ARD lengthscale vector has no reason to prefer "large lengthscale
    = irrelevant dimension" over "small lengthscale = active dimension,
    poorly identified from limited context" when both explain the P training
    points similarly well (the marginal-likelihood surface is flat/multimodal
    along that direction) — regularising toward the true generative
    hyperparameter regime, and giving the optimiser several independent shots
    at escaping a bad local optimum, both target that failure mode directly.
    """
    if kernel_name == "periodic" and feature_extractor is not None:
        raise ValueError("kernel_name='periodic' is not PD in a >1D DKL latent space")
    if ard and feature_extractor is not None:
        # ard_num_dims below is derived from d_x (X_train's raw column count),
        # but the base kernel actually sees feature_extractor(x) — a
        # differently-shaped latent tensor. Silently using d_x here would
        # either crash with a lengthscale/tensor shape mismatch, or (if d_x
        # happens to equal the extractor's out_dim) silently "work" while
        # tying each lengthscale to the wrong axis. Neither call site combines
        # ard=True with a feature_extractor today; fail loudly instead of
        # leaving this as a landmine for a future caller.
        raise ValueError(
            "ard=True is not supported together with a feature_extractor: "
            "ARD lengthscale count must match the extractor's output "
            "dimension, not X_train's raw column count (d_x)."
        )

    d_x = X_train.shape[1]
    ard_num_dims = d_x if (ard and kernel_name != "dot_product") else None
    kernel_priors = _kernel_priors(prior_cfg or {}, kernel_name, ard=ard)
    noise_prior = _noise_prior(prior_cfg or {})
    lengthscale_init_prior = _lengthscale_init_prior(prior_cfg or {})

    best_loss: float | None = None
    best_model: _ExactGPModel | None = None
    best_likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None

    for _ in range(max(1, n_restarts)):
        # Mirrors the previous hand-rolled code's log_noise clamp range
        # (exp(-8)..exp(2)) — keeps optimisation stable / prevents noise
        # collapsing to (near-)zero. Built fresh every restart: gpytorch's
        # Interval holds its bounds as plain (non-parameter) tensors, so
        # reusing one Interval instance across multiple GaussianLikelihoods
        # in this loop means the *next* likelihood constructed from it
        # inherits whatever device the *previous* iteration's `.to(device)`
        # left those bounds on, in-place — a shared-mutable-state footgun
        # that surfaces as a CPU/CUDA device-mismatch on restart 2+.
        noise_constraint = gpytorch.constraints.Interval(math.exp(-8.0), math.exp(2.0))
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint, noise_prior=noise_prior,
        )
        # Sampled (not fixed at 0.1) so restarts are actually independent —
        # clamped strictly inside noise_constraint's open interval.
        likelihood.noise = noise_prior.sample(likelihood.noise.shape).to(X_train.device).clamp(
            min=math.exp(-8.0) * 1.01, max=math.exp(2.0) * 0.99
        )

        model = _ExactGPModel(
            X_train, z_train, likelihood, kernel_name,
            ard_num_dims=ard_num_dims, feature_extractor=feature_extractor,
            kernel_priors=kernel_priors,
        ).to(X_train.device)
        _randomize_init(model, kernel_priors, kernel_name, lengthscale_init_prior=lengthscale_init_prior)

        model.train()
        likelihood.train()
        opt = Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        loss = None
        for _step in range(n_steps):
            opt.zero_grad()
            loss = -mll(model(X_train), z_train)
            loss.backward()
            opt.step()

        final_loss = loss.item()
        if best_loss is None or final_loss < best_loss:
            best_loss, best_model, best_likelihood = final_loss, model, likelihood

    model, likelihood = best_model, best_likelihood
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model.forward(X_test)) if oracle_mode == "prior" else likelihood(model(X_test))
        Sigma_post = pred.covariance_matrix
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
# descriptive alias rather than duplicating the class.
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
    oracle_mode: str = "prior",
    prior_cfg: dict | None = None,
    n_restarts_mle: int = 1,
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
    for kname in _GP_KERNELS:
        for ard in ([False, True] if _ARD_ELIGIBLE[kname] else [False]):
            label = _LABEL_MAP[(kname, ard)]
            try:
                R_gp = fit_and_eval_gpytorch(X_train, z_train, X_test, kname,
                                             n_steps=n_steps_mle, lr=lr_mle, ard=ard,
                                             oracle_mode=oracle_mode, prior_cfg=prior_cfg,
                                             n_restarts=n_restarts_mle)
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
                                          ard=False, feature_extractor=mlp,
                                          oracle_mode=oracle_mode, prior_cfg=prior_cfg)
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
    parser.add_argument("--dataset_dir",  default=None,
                        help="Episode directory to evaluate on (overrides "
                             "training.dataset_dir from --config)")
    parser.add_argument("--n_episodes",   type=int,   default=50)
    parser.add_argument("--episode_idx",  type=int,   default=0)
    parser.add_argument("--n_steps_mle",  type=int,   default=1000,
                        help="Adam steps for GP kernel MLE fitting (also used for ARD variants)")
    parser.add_argument("--lr_mle",       type=float, default=0.05,
                        help="Learning rate for GP MLE Adam")
    parser.add_argument("--n_restarts_mle", type=int, default=5,
                        help="Independent random restarts per GP-MLE kernel fit (each "
                             "initialised by sampling from the same LogNormal/Gamma "
                             "hyperpriors data_gen.py's generative process uses — see "
                             "fit_and_eval_gpytorch's prior_cfg/n_restarts docs); keeps "
                             "whichever restart reaches the best final training loss. "
                             "GP-ARD's marginal-likelihood surface is multimodal enough "
                             "that a single Adam run from a fixed init is unreliable.")
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
    parser.add_argument("--oracle_mode",  default=None, choices=["prior", "posterior"],
                        help="How R_star was built for this dataset (see data_gen.py's "
                             "oracle_mode). Determines whether GP-MLE/DKL score the fitted "
                             "kernel's posterior (conditioned on X_train) or its raw prior "
                             "covariance at X_test. Default: read from the checkpoint's own "
                             "saved training config (cfg.data.oracle_mode), falling back to "
                             "'prior' if that's absent (this repo's current datasets all use "
                             "oracle_mode=prior).")
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

    # GP-MLE/DKL must score against the same convention used to build this
    # dataset's R_star ("prior" ignores training conditioning entirely,
    # "posterior" conditions on X_train) — see fit_and_eval_gpytorch's
    # docstring. Read from the checkpoint's own saved training cfg by
    # default, since that's the actual generation config for this run's
    # dataset; falls back to "prior" (this repo's current datasets all use
    # oracle_mode=prior, unlike data_gen.py's own historical "posterior"
    # default for dataset *generation*).
    oracle_mode = args.oracle_mode or OmegaConf.select(icl_cfg, "data.oracle_mode", default="prior")
    print(f"Oracle mode: {oracle_mode}")

    # GP-MLE/DKL hyperpriors (see _kernel_priors/_noise_prior): read the exact
    # LogNormal/Gamma constants this checkpoint's dataset was generated with,
    # falling back to _DEFAULT_PRIOR_CFG (data_gen.py's own defaults) for any
    # missing key (e.g. an older checkpoint saved before a given key existed).
    data_cfg = OmegaConf.select(icl_cfg, "data", default=None)
    prior_cfg = OmegaConf.to_container(data_cfg) if data_cfg is not None else {}
    print(f"GP-MLE restarts: {args.n_restarts_mle}")

    dataset_dir = args.dataset_dir or cfg.training.dataset_dir
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
            oracle_mode=oracle_mode,
            prior_cfg=prior_cfg,
            n_restarts_mle=args.n_restarts_mle,
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
