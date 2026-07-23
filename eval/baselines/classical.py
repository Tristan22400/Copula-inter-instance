"""classical.py — every non-ICL baseline scored against a copula episode,
fit together in one place so they always run under identical conventions
(same oracle_mode, same hyperpriors, same y-space fitting target).

Methods
-------
  independence       : R = I_N (copula NLL = 0.0 always, reference point)
  gp_prior_rbf       : RBF prior correlation at test points (median bandwidth,
                       no conditioning on z_train)
  gp_mle_rbf         : GP posterior with MLE-fitted RBF {l, alpha^2, sigma^2_n},
                       fit in raw y-space
  gp_mle_ard_rbf     : Same, with one lengthscale per input dimension (ARD)
  gp_mle_matern32    : GP posterior with MLE-fitted Matern-3/2 kernel, fit in
                       raw y-space
  gp_mle_ard_matern32: Matern-3/2 with ARD lengthscales
  gp_mle_periodic    : GP posterior with MLE-fitted Periodic kernel (+ period),
                       fit in raw y-space
  gp_mle_ard_periodic: Periodic with one lengthscale + period per input dimension (ARD)
  gp_mle_rq          : GP posterior with MLE-fitted Rational Quadratic (+ rq_alpha),
                       fit in raw y-space
  gp_mle_ard_rq      : Rational Quadratic with ARD lengthscales
  gp_mle_dot_product : GP posterior with MLE-fitted linear/dot-product kernel
                       (variance + noise term fitted), fit in raw y-space
  dkl_rbf/matern32/rq/dot_product :
                       Deep Kernel Learning — MLP(d_x->32->16) feature extractor
                       feeding a GP layer (chosen kernel), fit in raw y-space,
                       trained jointly by maximising the marginal log-likelihood
  per_ep_transformer : Small set-transformer trained from scratch on this episode,
                       fit against a z-scored (no oracle kernel) transform of
                       y_train, analogous to gp_mle/dkl's y-space fit

Entry point is ``eval_baselines_episode`` — everything else in this module is
an implementation detail of one of the methods above. ``eval_checkpoint.py``
calls it once per episode and merges the result with the ICL model's own
score (which lives outside this file: it's the one thing that changes
between runs, not a baseline).

Baseline caching
----------------
GP-MLE (with restarts)/DKL/per_ep_transformer fitting dominates evaluation
runtime and, unlike the ICL model under test, only depends on the
episode-generating config and the fitting hyperparameters passed to
``eval_baselines_episode`` — not on which checkpoint is being evaluated.
``baseline_fingerprint`` + ``episode_cache_key`` + ``load_baseline_cache`` /
``save_baseline_cache`` let a runner cache results per episode across
repeated runs against new checkpoints; any change to a fitting
hyperparameter or the episode config invalidates the cache automatically.
"""

from __future__ import annotations

import copy
import math
import os
import sys

import gpytorch
import torch
import torch.nn as nn
from gpytorch.priors import GammaPrior, LogNormalPrior, Prior
from omegaconf import OmegaConf
from torch import Tensor
from torch.optim import Adam

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from loss import oracle_copula_nll  # noqa: E402
from model import low_rank_correlation  # noqa: E402

__all__ = [
    "corr_nll_single",
    "gp_prior_corr_rbf",
    "eval_baselines_episode",
    "baseline_fingerprint",
    "episode_cache_key",
    "load_baseline_cache",
    "save_baseline_cache",
]


def corr_nll_single(R: Tensor, z: Tensor) -> float:
    """Copula NLL for a single (N, N) correlation matrix and (N,) z-vector."""
    N = z.shape[0]
    mask = torch.ones(1, N, dtype=torch.bool, device=z.device)
    return oracle_copula_nll(R.unsqueeze(0), z.unsqueeze(0), mask).item()


# ---------------------------------------------------------------------------
# GP baselines (GPyTorch)
# ---------------------------------------------------------------------------
# ExactGP + ExactMarginalLogLikelihood + Adam(model.parameters()) backprops
# through kernel hyperparameters (registered as ordinary constrained
# nn.Parameters) just fine — no hand-rolled kernel math or NaN-safe distance
# helpers needed.

_ARD_ELIGIBLE = {
    "rbf": True,
    "matern32": True,
    # gpytorch.kernels.PeriodicKernel sums per-dimension sin^2 terms inside a
    # single exp() (a product of per-dimension periodic kernels), which is
    # PSD for any ard_num_dims.
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
    process samples ground-truth hyperparameters from, for MAP — instead of
    plain MLE — fitting of the GP baselines here.

    Plain MLE over an ARD lengthscale vector has no reason to prefer "large
    lengthscale" (irrelevant dimension) over "small lengthscale" (active
    dimension, poorly identified from limited context) when both explain the
    P training points similarly well — the marginal-likelihood surface is
    flat/multimodal in that direction. Registering these priors on the
    kernel/likelihood makes ExactMarginalLogLikelihood add their log-density
    automatically, pulling the fit toward the same hyperparameter regime the
    episodes were actually generated from.

    ard=True omits the lengthscale prior specifically: LogNormal(l_loc, l_scale)
    describes the lengthscale of a dimension already known to be active — its
    median-1 pull actively discourages the "grow arbitrarily large" behaviour
    an ARD lengthscale needs to correctly flag a dimension as irrelevant.
    Outputscale/noise priors showed no such conflict (they aren't asked to do
    variable selection) and stay on unconditionally.
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
    restart when ard=True omits it as a registered MAP prior, so ARD restarts
    still diversify their starting point instead of all beginning from
    gpytorch's identical default init."""
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
    in-place `.initialize()` copy these setters do is safe) to diversify
    restarts instead of re-running Adam from the same fixed gpytorch default
    init every time.
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
        y_train: Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel_name: str,
        ard_num_dims: int | None = None,
        feature_extractor: nn.Module | None = None,
        kernel_priors: dict[str, Prior] | None = None,
    ) -> None:
        super().__init__(X_train, y_train, likelihood)
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
    y_train: Tensor,
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
    """Fit a GP (optionally over a learned feature extractor, i.e. DKL) on the
    raw y-space target by maximising the exact marginal log-likelihood, and
    return the correlation matrix at X_test to compare against the episode's
    oracle R_star.

    Fits against y_train (not the PIT-transformed z_train) because z_train is
    itself derived from the true generating kernel's own Cholesky factor —
    feeding that into an independently fit "baseline" GP would leak oracle
    kernel information a real baseline never has access to, and would fit
    against a variable already whitened to unit variance, at odds with the
    alpha2/nugget hyperpriors below (which mirror data_gen.py's own *y-space*
    generative kernel-hyperparameter priors). The resulting covariance is
    converted to a correlation matrix (sigma_to_correlation), which is
    coordinate-free, so scoring against z_test downstream is unaffected by
    y_train's absolute scale/mean.

    oracle_mode must match how the episode's own R_star was built (see
    data_gen.py's oracle_mode branch):
      - "posterior": R_star conditions on (X_train, y_train), so we score the
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
    keeping whichever restart reaches the best final training loss.

    When feature_extractor is given (DKL), training instead holds out a 20%
    validation split of X_train/y_train and keeps whichever training step's
    weights reached the best held-out predictive NLL, instead of just running
    to n_steps and keeping the final weights. The plain (no feature_extractor)
    GP-MLE fit above has only 2-3 free hyperparameters, already regularised by
    kernel_priors/noise_prior — but DKL's MLP is unregularised and free to
    rescale its own output to defeat those priors. Empirically this drives the
    fitted noise toward the noise_constraint floor, near-interpolating y_train
    while collapsing every X_test feature into a near-constant direction
    (off-diagonal correlation -> ~1) — training loss keeps improving long
    after held-out NLL has turned catastrophically worse than independence,
    so a fixed step count with no validation signal silently picks the worst
    point on that curve. Skipped for P < 8 (too few points for a meaningful
    split); falls back to training on the full set with no early stopping,
    same as the no-feature-extractor path.
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

    P = X_train.shape[0]
    use_val = feature_extractor is not None and P >= 8
    if use_val:
        n_val = max(2, int(round(0.2 * P)))
        perm = torch.randperm(P, device=X_train.device)
        val_idx, fit_idx = perm[:n_val], perm[n_val:]
        X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        val_every = max(1, n_steps // 100)
    else:
        X_fit, y_fit = X_train, y_train

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
            X_fit, y_fit, likelihood, kernel_name,
            ard_num_dims=ard_num_dims, feature_extractor=feature_extractor,
            kernel_priors=kernel_priors,
        ).to(X_train.device)
        _randomize_init(model, kernel_priors, kernel_name, lengthscale_init_prior=lengthscale_init_prior)

        model.train()
        likelihood.train()
        opt = Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        best_step_val: float = float("inf")
        best_step_state: tuple[dict, dict] | None = None
        loss = None
        for step in range(n_steps):
            opt.zero_grad()
            loss = -mll(model(X_fit), y_fit)
            loss.backward()
            opt.step()

            if use_val and (step % val_every == 0 or step == n_steps - 1):
                model.eval()
                likelihood.eval()
                with torch.no_grad():
                    val_nll = -likelihood(model(X_val)).log_prob(y_val).item() / X_val.shape[0]
                model.train()
                likelihood.train()
                if val_nll < best_step_val:
                    best_step_val = val_nll
                    best_step_state = (
                        copy.deepcopy(model.state_dict()),
                        copy.deepcopy(likelihood.state_dict()),
                    )

        if use_val:
            model.load_state_dict(best_step_state[0])
            likelihood.load_state_dict(best_step_state[1])
            final_loss = best_step_val
        else:
            final_loss = loss.item()

        if best_loss is None or final_loss < best_loss:
            best_loss, best_model, best_likelihood = final_loss, model, likelihood

    model, likelihood = best_model, best_likelihood
    if use_val:
        # The val split only mattered for picking which training step's
        # weights to keep; restore full-context conditioning for the final
        # posterior/prior evaluation below (oracle_mode="posterior" needs the
        # model literally conditioned on all of X_train — oracle_mode="prior"
        # ignores stored train data entirely via model.forward(), so this is
        # a no-op for that path). strict=False since the point count changes
        # from len(fit_idx) to P.
        model.set_train_data(inputs=X_train, targets=y_train, strict=False)
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model.forward(X_test)) if oracle_mode == "prior" else likelihood(model(X_test))
        Sigma_post = pred.covariance_matrix
        N = X_test.shape[0]
        Sigma_post = 0.5 * (Sigma_post + Sigma_post.T) + jitter * torch.eye(
            N, dtype=Sigma_post.dtype, device=Sigma_post.device
        )

    from data_gen import sigma_to_correlation  # noqa: E402  (lazy: keeps module import light for callers that only need corr_nll_single/gp_prior_corr_rbf)

    R, _ = sigma_to_correlation(Sigma_post)
    return R


def gp_prior_corr_rbf(X_test: Tensor) -> Tensor:
    """RBF prior correlation at test points with median bandwidth (no training data)."""
    from data_gen import _sq_dist  # noqa: E402

    N = X_test.shape[0]
    sq = _sq_dist(X_test, X_test)
    h2 = torch.pdist(X_test).pow(2).median().clamp(min=1e-6)
    R = torch.exp(-sq / (2.0 * h2))
    R = R / R.diagonal().clamp(min=1e-8).sqrt().unsqueeze(-1)
    R = R / R.diagonal().clamp(min=1e-8).sqrt().unsqueeze(-2)
    return R


# ---------------------------------------------------------------------------
# Per-episode small transformer + Deep Kernel Learning building blocks
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
        self.norm_q = nn.LayerNorm(m)
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
        -> W  : (n_qry, r),   s : (n_qry,)

    The (W, s) pair feeds into ``low_rank_correlation`` (from model.py) to produce
    the (n_qry x n_qry) inter-instance correlation matrix, matching the CopulaTabICL
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

        self.x_enc = _MLP(d_x, m, m, dropout)
        self.row_enc = _MLP(m + 1, m, m, dropout)
        self.self_attn = nn.ModuleList([_SelfAttn(m, n_heads, dropout) for _ in range(n_layers)])
        self.W_q = nn.Linear(m, m)
        self.cross_attn = _CrossAttn(m, n_heads, dropout)
        self.head = nn.Linear(m, r + 1)

        # NOTE: zero-initializing head.weight is a dead end here — Sigma is
        # built from W @ W.T (a bilinear form in the first r head outputs),
        # so dSigma/dW ∝ W vanishes exactly at W=0, and the unit-diagonal
        # normalization makes Sigma == I regardless of s when W=0. Both paths
        # have zero gradient, so the model can never leave Sigma=I. Use a
        # small random init to break the saddle point.
        nn.init.normal_(self.head.weight, std=1e-2)
        nn.init.zeros_(self.head.bias)

    def forward(self, X_ctx: Tensor, z_ctx: Tensor, X_qry: Tensor) -> tuple[Tensor, Tensor]:
        ex = self.x_enc(X_ctx)                                            # (n_sup, m)
        row = self.row_enc(torch.cat([ex, z_ctx.unsqueeze(-1)], dim=-1))  # (n_sup, m)
        row = row.unsqueeze(0)                                           # (1, n_sup, m)
        for block in self.self_attn:
            row = block(row)

        eq = self.x_enc(X_qry).unsqueeze(0)                              # (1, n_qry, m)
        q_emb = self.W_q(eq)
        h = self.cross_attn(q_emb, row).squeeze(0)                       # (n_qry, m)

        out = self.head(h)                                                # (n_qry, r+1)
        W = out[:, : self.r]                                              # (n_qry, r)
        s = out[:, self.r]                                                # (n_qry,)
        return W, s


def _standardize_y(y: Tensor) -> Tensor:
    """Z-score y using only its own sample mean/std — no oracle kernel.

    Unlike arbitrary real-world targets, data_gen.py's y_all is a direct
    MultivariateNormal(0, K) + Gaussian-noise sample; tabiclv2_warp_features
    only warps X, never y. So y's marginal is exactly Gaussian by
    construction, and plain affine standardization already gives ~N(0,1)
    margins — the exact PIT for a known-Gaussian family, unlike a rank-based
    transform. This never touches the true generating kernel K_ff, so it
    carries none of the oracle-kernel leakage fit_and_eval_gpytorch's
    docstring warns about. Gives per_ep_transformer standard-normal-margin
    inputs (required by oracle_copula_nll's Gaussian-copula density) without
    the oracle shortcut, matching how gp_mle/dkl are fit against raw y_train
    instead of z_train.
    """
    return (y - y.mean()) / y.std(unbiased=True).clamp(min=1e-6)


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
    P = X_train.shape[0]

    # r is normally icl_rank (the pretrained model's rank, e.g. 32) — sized for
    # a model pretrained across millions of episodes. Trained from scratch on a
    # single episode's P instances (as few as ~13, ~8 after the support/query
    # split below), that many free low-rank factors overfits badly: verified
    # empirically that more training steps at r=32 makes some episodes *worse*
    # (correlation matrix collapses to near-singular, off-the-charts NLL),
    # while capping r relative to P keeps it stable. r=4 (the class default)
    # is never exceeded for very small P.
    r = max(2, min(r, P // 4))

    n_val = max(2, int(round(0.2 * P)))
    perm = torch.randperm(P, device=device)
    val_idx, pool_idx = perm[:n_val], perm[n_val:]

    X_val, z_val = X_train[val_idx], z_train[val_idx]
    X_pool, z_pool = X_train[pool_idx], z_train[pool_idx]
    n_pool = X_pool.shape[0]

    model = PerEpisodeTransformer(d_x, r=r).to(device)
    opt = Adam(model.parameters(), lr=lr)

    best_val = float("inf")
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
        mask = torch.ones(1, X_q.shape[0], dtype=torch.bool, device=device)
        loss = oracle_copula_nll(Sigma.unsqueeze(0), z_q.unsqueeze(0), mask)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % val_every == 0:
            model.eval()
            with torch.no_grad():
                W_v, s_v = model(X_pool, z_pool, X_val)
                Sv = low_rank_correlation(W_v.unsqueeze(0), s_v.unsqueeze(0)).squeeze(0)
                val_nll = corr_nll_single(Sv, z_val)
            model.train()

            if val_nll < best_val - 1e-4:
                best_val = val_nll
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
# Per-episode evaluation — every baseline at once
# ---------------------------------------------------------------------------


def eval_baselines_episode(
    ep: dict,
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
) -> tuple[dict[str, float], dict[str, Tensor]]:
    """Evaluate every classical/fitted baseline (everything except the ICL
    model and the oracle) on one episode, under identical conventions.

    Split out from ICL/oracle evaluation so these (expensive: many Adam
    restarts per GP-MLE kernel, DKL training, per-episode-transformer
    training) results can be cached across repeated eval_checkpoint.py runs
    that only change the checkpoint under test — see baseline_fingerprint /
    load_baseline_cache / save_baseline_cache below.

    Returns:
        nlls      : {method_name: copula_nll_float}
        R_dict    : {method_name: (N, N) correlation tensor} — for plotting
    """
    X_train = ep["x_norm_train"].to(device)      # (P, d_x)
    y_train = ep["y_train"].to(device)            # (P,)  raw target, used to fit the GP-MLE/DKL baselines
    z_train_self = _standardize_y(y_train)        # (P,) z-scored y_train, used to train per_ep_transformer
    X_test = ep["x_norm_test"].to(device)         # (N, d_x)
    z_test = ep["z_test"].to(device)              # (N,)

    N = X_test.shape[0]
    nlls: dict[str, float] = {}
    R_dict: dict[str, Tensor] = {}

    # --- independence ---
    R_I = torch.eye(N, dtype=X_train.dtype, device=device)
    nlls["independence"] = corr_nll_single(R_I, z_test)
    R_dict["independence"] = R_I

    # --- GP prior RBF ---
    R_prior = gp_prior_corr_rbf(X_test)
    nlls["gp_prior_rbf"] = corr_nll_single(R_prior, z_test)
    R_dict["gp_prior_rbf"] = R_prior

    # --- GP MLE baselines (plain + ARD for lengthscale kernels) ---
    _GP_KERNELS = ["rbf", "matern32", "periodic", "rational_quadratic", "dot_product"]
    _LABEL_MAP = {
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
                R_gp = fit_and_eval_gpytorch(X_train, y_train, X_test, kname,
                                             n_steps=n_steps_mle, lr=lr_mle, ard=ard,
                                             oracle_mode=oracle_mode, prior_cfg=prior_cfg,
                                             n_restarts=n_restarts_mle)
                nlls[label] = corr_nll_single(R_gp, z_test)
                R_dict[label] = R_gp
            except Exception as exc:
                print(f"  [{label}] failed: {exc}")
                nlls[label] = float("nan")
                R_dict[label] = R_I.clone()

    # --- Deep Kernel Learning (MLP + GP, jointly trained), across multiple kernels ---
    # "periodic" excluded: not PD in the fixed 16-dim latent space at any dimensionality.
    _DKL_KERNELS = ["rbf", "matern32", "rational_quadratic", "dot_product"]
    _DKL_LABEL_MAP = {
        "rbf":                "dkl_rbf",
        "matern32":           "dkl_matern32",
        "rational_quadratic": "dkl_rq",
        "dot_product":        "dkl_dot_product",
    }
    for kname in _DKL_KERNELS:
        label = _DKL_LABEL_MAP[kname]
        try:
            mlp = DKLFeatureExtractor(X_train.shape[1], hidden=32, out_dim=16, dropout=0.0).to(device)
            R_dkl = fit_and_eval_gpytorch(X_train, y_train, X_test, kname,
                                          n_steps=n_steps_dkl, lr=lr_dkl,
                                          ard=False, feature_extractor=mlp,
                                          oracle_mode=oracle_mode, prior_cfg=prior_cfg)
            nlls[label] = corr_nll_single(R_dkl, z_test)
            R_dict[label] = R_dkl
        except Exception as exc:
            print(f"  [{label}] failed: {exc}")
            nlls[label] = float("nan")
            R_dict[label] = R_I.clone()

    # --- per-episode transformer ---
    # Trained/queried against z_train_self (z-scored from y_train), not the
    # oracle z_train — see _standardize_y's docstring. Scored against z_test
    # (oracle) below, same as every other baseline.
    try:
        per_ep_model = train_per_episode(
            X_train, z_train_self, r=icl_rank,
            n_steps=n_steps_per_ep, patience=patience_per_ep,
            device=device,
        )
        with torch.no_grad():
            W_te, s_te = per_ep_model(X_train, z_train_self, X_test)
            Sigma_te = low_rank_correlation(W_te.unsqueeze(0), s_te.unsqueeze(0)).squeeze(0)
        nlls["per_ep_transformer"] = corr_nll_single(Sigma_te, z_test)
        R_dict["per_ep_transformer"] = Sigma_te
    except Exception as exc:
        print(f"  [per_ep_transformer] failed: {exc}")
        nlls["per_ep_transformer"] = float("nan")
        R_dict["per_ep_transformer"] = R_I.clone()

    return nlls, R_dict


# ---------------------------------------------------------------------------
# Baseline cache
# ---------------------------------------------------------------------------
#
# GP-MLE (with restarts)/DKL/per_ep_transformer fitting dominates a
# checkpoint-evaluation run's runtime and, unlike the ICL model under test,
# is unaffected by which checkpoint we're evaluating — only by the
# episode-generating config and the fitting hyperparameters below. Caching
# those results to disk lets repeated "just check the new checkpoint" runs
# skip straight to the ICL forward pass + oracle NLL for every episode,
# instead of re-fitting ~15 baselines per episode from scratch.


def baseline_fingerprint(
    icl_cfg,
    live_generate: bool,
    dataset_dir: str | None,
    seed: int,
    icl_rank: int,
    oracle_mode: str,
    n_steps_mle: int,
    lr_mle: float,
    n_restarts_mle: int,
    n_steps_dkl: int,
    lr_dkl: float,
    n_steps_per_ep: int,
    patience_per_ep: int,
) -> dict:
    """Everything that determines the *baseline* fit results for an episode,
    other than which episode it is (see episode_cache_key for that half).

    Includes cfg.data (the generating distribution episodes are drawn from,
    and the source of prior_cfg's hyperpriors) and icl_rank (sizes
    per_ep_transformer's low-rank factor — see train_per_episode's docstring)
    since both change what a "correct" baseline fit looks like, even though
    neither is a baseline-fitting hyperparameter in the argparse sense.
    Deliberately excludes the rest of icl_cfg (e.g. model architecture,
    optimizer settings) — those affect the ICL model, not the baselines being
    cached here, and including them would invalidate the cache every time an
    unrelated training run tweaks something baselines never see.
    """
    data_cfg = OmegaConf.select(icl_cfg, "data", default=None)
    return {
        "data_cfg": OmegaConf.to_container(data_cfg) if data_cfg is not None else {},
        "icl_rank": icl_rank,
        "live_generate": live_generate,
        "dataset_dir": os.path.abspath(dataset_dir) if (dataset_dir and not live_generate) else None,
        "seed": seed,
        "oracle_mode": oracle_mode,
        "n_steps_mle": n_steps_mle,
        "lr_mle": lr_mle,
        "n_restarts_mle": n_restarts_mle,
        "n_steps_dkl": n_steps_dkl,
        "lr_dkl": lr_dkl,
        "n_steps_per_ep": n_steps_per_ep,
        "patience_per_ep": patience_per_ep,
    }


def episode_cache_key(live_generate: bool, dataset_dir: str | None, seed: int, local_i: int, ep_i: int) -> str:
    """Identifies which episode a cached baseline result belongs to.

    Live episodes are fully determined by (seed, local_i); dataset episodes
    by (dataset_dir, ep_i)."""
    if live_generate:
        return f"live:seed{seed}:idx{local_i}"
    return f"dataset:{os.path.abspath(dataset_dir)}:idx{ep_i}"


def load_baseline_cache(path: str, fingerprint: dict) -> dict[str, dict]:
    """Load {episode_key: {"nlls": ..., "R_dict": ...}} from path if its
    stored fingerprint matches; otherwise (missing file, or a fingerprint
    mismatch meaning the cache was built under different generation/fitting
    settings) start from an empty cache rather than serving stale results."""
    if not os.path.exists(path):
        return {}
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"  [baseline_cache] failed to load {path}: {exc} — starting fresh")
        return {}
    if blob.get("fingerprint") != fingerprint:
        print(f"  [baseline_cache] {path} was built with different generation/fitting "
              "settings — ignoring it and refitting all baselines")
        return {}
    entries = blob.get("entries", {})
    print(f"  [baseline_cache] loaded {len(entries)} cached episode(s) from {path}")
    return entries


def save_baseline_cache(path: str, fingerprint: dict, entries: dict[str, dict]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({"fingerprint": fingerprint, "entries": entries}, path)
    print(f"  [baseline_cache] saved {len(entries)} episode(s) to {path}")
