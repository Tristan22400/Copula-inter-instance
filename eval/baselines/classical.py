"""Non-ICL baselines scored on a copula episode, all under the same conventions.

Methods:
    independence: R = I (copula NLL 0).
    gp_prior_rbf: RBF prior correlation with a median bandwidth.
    gp_mle_{rbf, matern32, periodic, rq, dot_product, polynomial} and ARD
        variants (gp_mle_ard_*): GP posterior with MAP-fitted hyperparameters,
        fitted on raw y; polynomial searches every degree in
        [poly_power_min, poly_power_max].
    dkl_{rbf, matern32, rq, dot_product}: deep kernel learning, an MLP
        (d_x -> 32 -> 16) feeding a GP, trained on the marginal likelihood.
    per_ep_transformer: small set transformer trained on the episode's
        z-scored y_train.

eval_baselines_episode is the entry point. baseline_fingerprint,
episode_cache_key, save_baseline_entry and load_baseline_cache cache results
per episode (one file each), keyed so that any change to the generating
config or fitting settings invalidates them.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import warnings
from typing import Callable

import gpytorch
import torch
import torch.nn as nn
from gpytorch.priors import GammaPrior, LogNormalPrior, Prior
from linear_operator.utils.warnings import NumericalWarning
from omegaconf import OmegaConf
from torch import Tensor
from torch.optim import Adam

# Silence psd_safe_cholesky's jitter warnings during fitting.
warnings.filterwarnings("ignore", category=NumericalWarning)


from copula_inter.artifacts import atomic_torch_save  # noqa: E402
from copula_inter.dataset_manifest import dataset_identity  # noqa: E402
from copula_inter.loss import gp_oracle_y_nll, oracle_copula_nll  # noqa: E402
from copula_inter.model import low_rank_correlation  # noqa: E402
from eval.results import NAN_PARTS as _NAN_PARTS  # noqa: E402

__all__ = [
    "corr_nll_single",
    "gp_prior_corr_rbf",
    "eval_baselines_episode",
    "baseline_fingerprint",
    "episode_cache_key",
    "load_baseline_cache",
    "save_baseline_cache",
    "save_baseline_entry",
    "GP_VAL_SELECT_MODES",
    "EXPECTED_BASELINE_KEYS",
    "assert_shared_z_test",
    "fit_zero_mean_gp_on_marginal",
    "zero_mean_gp_prior_cfg",
]


def assert_shared_z_test(z_test: Tensor, ep: dict) -> None:
    """Raise if the z_test being scored is not ep["z_test"] (the shared-marginal table needs one z_test for every method)."""
    ep_z_test = ep["z_test"].to(z_test.device, z_test.dtype)
    assert torch.equal(z_test, ep_z_test), (
        "z_test used for a shared-marginal copula-NLL score has diverged "
        "from ep['z_test'] — this breaks _print_table's cross-method "
        "ranking (see assert_shared_z_test's docstring)."
    )

# Keys eval_baselines_episode returns; cached entries missing any are refitted.
EXPECTED_BASELINE_KEYS = frozenset(
    {"independence", "gp_prior_rbf", "per_ep_transformer"}
    | {
        "gp_mle_rbf", "gp_mle_ard_rbf",
        "gp_mle_matern32", "gp_mle_ard_matern32",
        "gp_mle_periodic", "gp_mle_ard_periodic",
        "gp_mle_rq", "gp_mle_ard_rq",
        "gp_mle_dot_product", "gp_mle_polynomial",
    }
    | {"dkl_rbf", "dkl_matern32", "dkl_rq", "dkl_dot_product"}
)

# Bump when a fitting algorithm changes in a way the fingerprint cannot detect.
_BASELINE_ALGO_VERSION = 4


GP_VAL_SELECT_MODES = ("ard", "always", "never")


def _resolve_val_select(mode: str, ard: bool) -> bool:
    """Whether this kernel selects its fit on a held-out split.

    "ard" (default): only ARD kernels; "always"; "never" (lowest training loss).
    """
    if mode == "always":
        return True
    if mode == "never":
        return False
    if mode == "ard":
        return ard
    raise ValueError(
        f"gp_val_select={mode!r} is not one of {GP_VAL_SELECT_MODES}"
    )


def corr_nll_single(R: Tensor, z: Tensor) -> float:
    """Copula NLL of an (N,) z under an (N, N) correlation matrix, with the shared z_test."""
    N = z.shape[0]
    mask = torch.ones(1, N, dtype=torch.bool, device=z.device)
    return oracle_copula_nll(R.unsqueeze(0), z.unsqueeze(0), mask).item()


# GP baselines (gpytorch).

# Matern nu per kernel name.
_MATERN_NU = {"matern12": 0.5, "matern32": 1.5, "matern52": 2.5}

_ARD_ELIGIBLE = {
    "rbf": True,
    "matern12": True,
    "matern32": True,
    "matern52": True,
    # PeriodicKernel is PSD for any ard_num_dims.
    "periodic": True,
    "rational_quadratic": True,
    "dot_product": False,
    # PolynomialKernel has no lengthscale.
    "polynomial": False,
}

# Default polynomial degree when poly_power is not given (not trainable in gpytorch).
_POLY_BASELINE_POWER = 2


def _poly_degree_candidates(prior_cfg: dict) -> list[int]:
    """Integer degrees poly_power_min..poly_power_max that polynomial episodes can have."""
    power_min = int(prior_cfg.get("poly_power_min", 2))
    power_max = int(prior_cfg.get("poly_power_max", 4))
    return list(range(power_min, power_max + 1))


# Fallback hyperprior constants (data_gen defaults), used when prior_cfg lacks a key.
_DEFAULT_PRIOR_CFG: dict[str, float] = {
    "l_lognormal_loc": 0.0,
    "l_lognormal_scale": 0.7,
    "alpha2_gamma_concentration": 4.0,
    "alpha2_gamma_rate": 3.0,
    "period_lognormal_loc": math.log(1.2),
    "period_lognormal_scale": 0.4,
    "rq_alpha_gamma_concentration": 2.0,
    "rq_alpha_gamma_rate": 1.0,
    "poly_offset_gamma_concentration": 2.0,
    "poly_offset_gamma_rate": 1.0,
    "nugget_lognormal_loc": -4.63,
    "nugget_lognormal_scale": 0.5,
}


def _kernel_priors(prior_cfg: dict, kernel_name: str, ard: bool = False) -> dict[str, Prior]:
    """LogNormal/Gamma hyperpriors matching data_gen's generative priors, for MAP fitting.

    With ard=True the lengthscale prior is omitted.
    """
    cfg = {**_DEFAULT_PRIOR_CFG, **prior_cfg}
    if kernel_name == "dot_product":
        return {"variance_prior": GammaPrior(cfg["alpha2_gamma_concentration"], cfg["alpha2_gamma_rate"])}
    if kernel_name == "polynomial":
        # PolynomialKernel is wrapped in ScaleKernel, so it keeps the outputscale prior.
        return {
            "outputscale_prior": GammaPrior(cfg["alpha2_gamma_concentration"], cfg["alpha2_gamma_rate"]),
            "offset_prior": GammaPrior(cfg["poly_offset_gamma_concentration"], cfg["poly_offset_gamma_rate"]),
        }
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
    """The lengthscale prior, used only to sample initial ARD lengthscales per restart."""
    cfg = {**_DEFAULT_PRIOR_CFG, **prior_cfg}
    return LogNormalPrior(cfg["l_lognormal_loc"], cfg["l_lognormal_scale"])


def _randomize_init(
    model: "_ExactGPModel",
    kernel_priors: dict[str, Prior],
    kernel_name: str,
    lengthscale_init_prior: Prior | None = None,
) -> None:
    """Sample a fresh initial value for every registered prior's parameter (once per restart)."""
    base = model.covar_module if kernel_name == "dot_product" else model.covar_module.base_kernel
    # Kernels without a lengthscale return None for .lengthscale.
    has_lengthscale = getattr(base, "lengthscale", None) is not None
    device = base.lengthscale.device if has_lengthscale else next(model.parameters()).device
    ls_prior = kernel_priors.get("lengthscale_prior", lengthscale_init_prior)
    if ls_prior is not None and has_lengthscale:
        base.lengthscale = ls_prior.sample(base.lengthscale.shape).to(device)
    if "period_length_prior" in kernel_priors:
        base.period_length = kernel_priors["period_length_prior"].sample(base.period_length.shape).to(device)
    if "alpha_prior" in kernel_priors:
        base.alpha = kernel_priors["alpha_prior"].sample(base.alpha.shape).to(device)
    if "offset_prior" in kernel_priors:
        base.offset = kernel_priors["offset_prior"].sample(base.offset.shape).to(device)
    if "variance_prior" in kernel_priors:
        model.covar_module.variance = kernel_priors["variance_prior"].sample(model.covar_module.variance.shape).to(device)
    if "outputscale_prior" in kernel_priors:
        model.covar_module.outputscale = kernel_priors["outputscale_prior"].sample(model.covar_module.outputscale.shape).to(device)


class _ExactGPModel(gpytorch.models.ExactGP):
    """ExactGP over one baseline kernel with a zero mean, optionally after a learned feature extractor (DKL)."""

    def __init__(
        self,
        X_train: Tensor,
        y_train: Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel_name: str,
        ard_num_dims: int | None = None,
        feature_extractor: nn.Module | None = None,
        kernel_priors: dict[str, Prior] | None = None,
        poly_power: int = _POLY_BASELINE_POWER,
    ) -> None:
        super().__init__(X_train, y_train, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ZeroMean()
        kp = kernel_priors or {}

        # Omit ard_num_dims instead of passing None (PeriodicKernel does not accept None).
        ard_kw = {} if ard_num_dims is None else {"ard_num_dims": ard_num_dims}

        if kernel_name == "rbf":
            base = gpytorch.kernels.RBFKernel(lengthscale_prior=kp.get("lengthscale_prior"), **ard_kw)
        elif kernel_name in _MATERN_NU:
            base = gpytorch.kernels.MaternKernel(
                nu=_MATERN_NU[kernel_name], lengthscale_prior=kp.get("lengthscale_prior"), **ard_kw,
            )
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
        elif kernel_name == "polynomial":
            base = gpytorch.kernels.PolynomialKernel(power=poly_power, offset_prior=kp.get("offset_prior"))
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")

        # LinearKernel has its own variance, so no ScaleKernel; PolynomialKernel gets one.
        self.covar_module = (
            base if kernel_name == "dot_product"
            else gpytorch.kernels.ScaleKernel(base, outputscale_prior=kp.get("outputscale_prior"))
        )

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def _val_check_steps(n_steps: int) -> set[int]:
    """Steps at which to evaluate held-out NLL: every 10 steps up to 500, plus every n_steps // 100."""
    coarse_every = max(1, n_steps // 100)
    dense = range(0, min(n_steps, 500), 10)
    coarse = range(0, n_steps, coarse_every)
    return set(dense) | set(coarse) | {n_steps - 1}


def fit_and_eval_gpytorch(
    X_train: Tensor,
    y_train: Tensor,
    X_test: Tensor,
    kernel_name: str,
    n_steps: int,
    lr: float,
    ard: bool = False,
    feature_extractor_factory: Callable[[], nn.Module] | None = None,
    jitter: float = 1e-6,
    oracle_mode: str = "prior",
    prior_cfg: dict | None = None,
    n_restarts: int = 1,
    val_select: bool = False,
) -> dict[str, Tensor]:
    """Fit a GP (or DKL) on raw y_train and return {"R", "mean", "Sigma"} at X_test.

    Maximizes the exact marginal likelihood plus the hyperpriors (MAP) from
    n_restarts random initializations and keeps the best. oracle_mode="posterior"
    returns the fitted posterior at X_test (with noise); "prior" returns the
    fitted kernel's prior covariance at X_test (with noise). "polynomial" repeats
    the fit for every candidate degree. With a feature extractor, or when
    val_select applies, 20% of the training points are held out and the step
    with the best held-out NLL is kept (not below P = 8). mean and Sigma are in
    raw y units.
    """
    if kernel_name == "periodic" and feature_extractor_factory is not None:
        raise ValueError("kernel_name='periodic' is not PD in a >1D DKL latent space")
    if ard and feature_extractor_factory is not None:
        # ARD with a feature extractor is not supported.
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
    use_val = (feature_extractor_factory is not None or val_select) and P >= 8
    if use_val:
        n_val = max(2, int(round(0.2 * P)))
        perm = torch.randperm(P, device=X_train.device)
        val_idx, fit_idx = perm[:n_val], perm[n_val:]
        X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        val_check_steps = _val_check_steps(n_steps)
    else:
        X_fit, y_fit = X_train, y_train

    # Only polynomial has degrees to search.
    poly_powers = _poly_degree_candidates(prior_cfg or {}) if kernel_name == "polynomial" else [_POLY_BASELINE_POWER]

    best_loss: float | None = None
    best_model: _ExactGPModel | None = None
    best_likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None

    for poly_power in poly_powers:
        for _ in range(max(1, n_restarts)):
          try:
            # Noise constraint exp(-8)..exp(2), built fresh each restart (its bounds are mutable tensors).
            noise_constraint = gpytorch.constraints.Interval(math.exp(-8.0), math.exp(2.0))
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=noise_constraint, noise_prior=noise_prior,
            )
            # Random initial noise inside the constraint.
            likelihood.noise = noise_prior.sample(likelihood.noise.shape).to(X_train.device).clamp(
                min=math.exp(-8.0) * 1.01, max=math.exp(2.0) * 0.99
            )

            # Fresh feature extractor each restart.
            feature_extractor = (
                feature_extractor_factory() if feature_extractor_factory is not None else None
            )
            model = _ExactGPModel(
                X_fit, y_fit, likelihood, kernel_name,
                ard_num_dims=ard_num_dims, feature_extractor=feature_extractor,
                kernel_priors=kernel_priors, poly_power=poly_power,
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

                if use_val and step in val_check_steps:
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
          except Exception as exc:
            # Skip a (degree, restart) whose kernel matrix is not PSD; raise only if all fail.
            print(f"  [gp_mle_polynomial power={poly_power}] restart failed: {exc}")
            continue

    if best_model is None:
        raise RuntimeError(
            f"fit_and_eval_gpytorch(kernel_name={kernel_name!r}): every candidate degree/restart "
            f"combination in {poly_powers} failed (see printed exceptions above)."
        )
    model, likelihood = best_model, best_likelihood
    if use_val:
        # Condition on the full training set for the final evaluation.
        model.set_train_data(inputs=X_train, targets=y_train, strict=False)
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model.forward(X_test)) if oracle_mode == "prior" else likelihood(model(X_test))
        mean_post = pred.mean
        Sigma_post = pred.covariance_matrix
        N = X_test.shape[0]
        Sigma_post = 0.5 * (Sigma_post + Sigma_post.T) + jitter * torch.eye(
            N, dtype=Sigma_post.dtype, device=Sigma_post.device
        )

    from copula_inter.data_gen import (
        sigma_to_correlation,  # noqa: E402  (lazy: keeps module import light for callers that only need corr_nll_single/gp_prior_corr_rbf)
    )

    R, _ = sigma_to_correlation(Sigma_post)
    return {"R": R, "mean": mean_post, "Sigma": Sigma_post}


# Zero-mean GP fitted on a real marginal's z_train (not cached with the other
# baselines, since it depends on the marginal).


def zero_mean_gp_prior_cfg(prior_cfg: dict | None = None) -> dict:
    """Hyperprior overrides for fitting on a unit-variance z_train: outputscale prior Gamma(2, 2)."""
    cfg = dict(prior_cfg or {})
    cfg["alpha2_gamma_concentration"] = 2.0
    cfg["alpha2_gamma_rate"] = 2.0
    return cfg


def fit_zero_mean_gp_on_marginal(
    X_train: Tensor,
    z_train: Tensor,
    X_test: Tensor,
    kernel_name: str,
    n_steps: int,
    lr: float,
    n_restarts: int,
    oracle_mode: str = "prior",
    prior_cfg: dict | None = None,
    jitter: float = 1e-6,
) -> dict[str, Tensor]:
    """Zero-mean GP-MLE on (X_train, z_train), z_train being a real marginal's PIT; returns {"R", "mean", "Sigma"} in z space."""
    return fit_and_eval_gpytorch(
        X_train, z_train, X_test, kernel_name,
        n_steps=n_steps, lr=lr, ard=False, jitter=jitter,
        oracle_mode=oracle_mode, prior_cfg=zero_mean_gp_prior_cfg(prior_cfg),
        n_restarts=n_restarts, val_select=False,
    )


def gp_prior_corr_rbf(X_test: Tensor) -> Tensor:
    """RBF prior correlation at the test points with a median bandwidth."""
    from copula_inter.data_gen import _sq_dist  # noqa: E402

    sq = _sq_dist(X_test, X_test)
    h2 = torch.pdist(X_test).pow(2).median().clamp(min=1e-6)
    R = torch.exp(-sq / (2.0 * h2))
    R = R / R.diagonal().clamp(min=1e-8).sqrt().unsqueeze(-1)
    R = R / R.diagonal().clamp(min=1e-8).sqrt().unsqueeze(-2)
    return R


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


# Feature extractor for DKL: Linear(d_x, 32) -> SiLU -> Linear(32, 16).
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
    """Small set transformer trained from scratch on one episode.

    forward(X_ctx (n_sup, d_x), z_ctx (n_sup,), X_qry (n_qry, d_x)) returns
    W (n_qry, r) and s (n_qry,), fed to model.low_rank_correlation.
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

        # Small random head init (W = 0 is a zero-gradient saddle at Sigma = I).
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
    """Z-score y with its own sample mean and std."""
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
    """Train a PerEpisodeTransformer on one episode.

    A fixed 20% split is used for early stopping; the rest is split 80/20 into
    support/query at every step. The rank is capped relative to P.
    """
    d_x = X_train.shape[1]
    P = X_train.shape[0]

    # Cap the rank relative to P (large ranks overfit a single episode).
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
    n_restarts_dkl: int = 1,
    fit_seed: int | None = None,
    gp_val_select: str = "ard",
) -> tuple[dict[str, float], dict[str, Tensor], dict[str, dict[str, float]]]:
    """Evaluate every baseline (all but the ICL model and the oracle) on one episode.

    fit_seed, when given, reseeds torch first so the fit depends only on the
    episode and settings (derive it from the episode's global index).

    Returns:
        nlls: {method: copula NLL} against the shared z_test.
        R_dict: {method: (N, N) correlation}.
        y_space_nlls: {method: {total, marginal, copula}} in raw y units under
            each fitted method's own marginal (not the unfitted references).
    """
    if fit_seed is not None:
        torch.manual_seed(fit_seed)

    X_train = ep["x_norm_train"].to(device)      # (P, d_x)
    y_train = ep["y_train"].to(device)            # (P,)  raw target, used to fit the GP-MLE/DKL baselines
    z_train_self = _standardize_y(y_train)        # (P,) z-scored y_train, used to train per_ep_transformer
    X_test = ep["x_norm_test"].to(device)         # (N, d_x)
    z_test = ep["z_test"].to(device)              # (N,)
    assert_shared_z_test(z_test, ep)
    y_test = ep["y_test"].to(device)              # (N,)  raw target, for total Y-space NLL

    N = X_test.shape[0]
    nlls: dict[str, float] = {}
    R_dict: dict[str, Tensor] = {}
    y_space_nlls: dict[str, dict[str, float]] = {}
    test_mask = torch.ones(1, N, dtype=torch.bool, device=device)

    def _nll_parts(mean: Tensor, Sigma: Tensor) -> dict[str, float]:
        parts = gp_oracle_y_nll(
            Sigma.unsqueeze(0), mean.unsqueeze(0), y_test.unsqueeze(0), test_mask,
        )
        return {k: v.item() for k, v in parts.items()}

    # --- independence ---
    R_I = torch.eye(N, dtype=X_train.dtype, device=device)
    nlls["independence"] = corr_nll_single(R_I, z_test)
    R_dict["independence"] = R_I

    # --- GP prior RBF ---
    R_prior = gp_prior_corr_rbf(X_test)
    nlls["gp_prior_rbf"] = corr_nll_single(R_prior, z_test)
    R_dict["gp_prior_rbf"] = R_prior

    # --- GP MLE baselines (plain + ARD for lengthscale kernels) ---
    _GP_KERNELS = ["rbf", "matern32", "periodic", "rational_quadratic", "dot_product", "polynomial"]
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
        ("polynomial", False):         "gp_mle_polynomial",
    }
    for kname in _GP_KERNELS:
        for ard in ([False, True] if _ARD_ELIGIBLE[kname] else [False]):
            label = _LABEL_MAP[(kname, ard)]
            try:
                # Held-out selection per _resolve_val_select.
                fit = fit_and_eval_gpytorch(X_train, y_train, X_test, kname,
                                             n_steps=n_steps_mle, lr=lr_mle, ard=ard,
                                             oracle_mode=oracle_mode, prior_cfg=prior_cfg,
                                             n_restarts=n_restarts_mle,
                                             val_select=_resolve_val_select(
                                                 gp_val_select, ard))
                nlls[label] = corr_nll_single(fit["R"], z_test)
                R_dict[label] = fit["R"]
                y_space_nlls[label] = _nll_parts(fit["mean"], fit["Sigma"])
            except Exception as exc:
                print(f"  [{label}] failed: {exc}")
                nlls[label] = float("nan")
                R_dict[label] = R_I.clone()
                y_space_nlls[label] = _NAN_PARTS.copy()

    # DKL kernels (periodic excluded).
    _DKL_KERNELS = ["rbf", "matern32", "rational_quadratic", "dot_product"]
    _DKL_LABEL_MAP = {
        "rbf":                "dkl_rbf",
        "matern32":           "dkl_matern32",
        "rational_quadratic": "dkl_rq",
        "dot_product":        "dkl_dot_product",
    }
    def _make_dkl_mlp(d_x: int = X_train.shape[1], dev: torch.device = device) -> nn.Module:
        return DKLFeatureExtractor(d_x, hidden=32, out_dim=16, dropout=0.0).to(dev)

    for kname in _DKL_KERNELS:
        label = _DKL_LABEL_MAP[kname]
        try:
            fit = fit_and_eval_gpytorch(X_train, y_train, X_test, kname,
                                        n_steps=n_steps_dkl, lr=lr_dkl,
                                        ard=False, feature_extractor_factory=_make_dkl_mlp,
                                        oracle_mode=oracle_mode, prior_cfg=prior_cfg,
                                        n_restarts=n_restarts_dkl)
            nlls[label] = corr_nll_single(fit["R"], z_test)
            R_dict[label] = fit["R"]
            y_space_nlls[label] = _nll_parts(fit["mean"], fit["Sigma"])
        except Exception as exc:
            print(f"  [{label}] failed: {exc}")
            nlls[label] = float("nan")
            R_dict[label] = R_I.clone()
            # Failed fits get _NAN_PARTS (a dict) so the cache entry stays valid.
            y_space_nlls[label] = _NAN_PARTS.copy()

    # per_ep_transformer: fitted on z-scored y_train, scored against the shared z_test;
    # its Y-space NLL uses the empirical Gaussian marginal.
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
        mean_tr = y_train.mean()
        std_tr = y_train.std(unbiased=True).clamp(min=1e-6)
        y_space_nlls["per_ep_transformer"] = _nll_parts(
            mean_tr.expand(N), (std_tr ** 2) * Sigma_te,
        )
    except Exception as exc:
        print(f"  [per_ep_transformer] failed: {exc}")
        nlls["per_ep_transformer"] = float("nan")
        R_dict["per_ep_transformer"] = R_I.clone()
        y_space_nlls["per_ep_transformer"] = _NAN_PARTS.copy()

    return nlls, R_dict, y_space_nlls


# Baseline cache.


def baseline_fingerprint(
    gen_cfg,
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
    gp_val_select: str = "ard",
    n_restarts_dkl: int = 1,
) -> dict:
    """Digest of everything, besides the episode itself, that determines the baseline fits.

    Includes gen_cfg.data (the generating config, not the checkpoint's own),
    icl_rank, oracle_mode and the fitting settings.
    """
    data_cfg = OmegaConf.select(gen_cfg, "data", default=None)
    return {
        "algo_version": _BASELINE_ALGO_VERSION,
        "data_cfg": OmegaConf.to_container(data_cfg) if data_cfg is not None else {},
        "icl_rank": icl_rank,
        "live_generate": live_generate,
        "dataset_dir": os.path.abspath(dataset_dir) if (dataset_dir and not live_generate) else None,
        "dataset_identity": (
            dataset_identity(dataset_dir)
            if dataset_dir and not live_generate and os.path.isdir(dataset_dir) else None
        ),
        "seed": seed,
        "oracle_mode": oracle_mode,
        "n_steps_mle": n_steps_mle,
        "lr_mle": lr_mle,
        "n_restarts_mle": n_restarts_mle,
        "n_steps_dkl": n_steps_dkl,
        "lr_dkl": lr_dkl,
        "n_restarts_dkl": n_restarts_dkl,
        "n_steps_per_ep": n_steps_per_ep,
        "patience_per_ep": patience_per_ep,
        "gp_val_select": gp_val_select,
    }


def episode_cache_key(
    live_generate: bool, dataset_dir: str | None, seed: int, ep_i: int,
    *, source: str | None = None,
) -> str:
    """Cache key of one episode.

        "{source}:seed{seed}:idx{ep_i}"   when source is given (e.g. "era5")
        "live:seed{seed}:idx{ep_i}"       for live generation
        "dataset:{dataset_dir}:idx{ep_i}" for an on-disk dataset

    ep_i is the episode's global index.
    """
    if source is not None:
        return f"{source}:seed{seed}:idx{ep_i}"
    if live_generate:
        return f"live:seed{seed}:idx{ep_i}"
    return f"dataset:{os.path.abspath(dataset_dir)}:idx{ep_i}"


def _shard_dir(path: str) -> str:
    """Directory holding this cache's per-episode shards (<path>.d)."""
    return f"{path}.d"


def _shard_name(cache_key: str) -> str:
    """Filename of one episode's shard: a hash of its cache key (the key is also stored inside)."""
    return hashlib.sha1(cache_key.encode()).hexdigest() + ".pt"


def save_baseline_entry(path: str, fingerprint: dict, cache_key: str, entry: dict) -> None:
    """Write one episode's fitted baselines to its own shard under <path>.d/."""
    d = _shard_dir(path)
    os.makedirs(d, exist_ok=True)
    dest = os.path.join(d, _shard_name(cache_key))
    atomic_torch_save({"fingerprint": fingerprint, "cache_key": cache_key, "entry": entry}, dest)


def load_baseline_cache(path: str, fingerprint: dict) -> dict[str, dict]:
    """Load {episode_key: entry} from the per-episode shards and the legacy single file, keeping fingerprint matches.

    Shards win over the legacy file; unreadable shards are skipped.
    """
    entries: dict[str, dict] = {}
    n_stale = 0

    if os.path.exists(path):
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            if blob.get("fingerprint") == fingerprint:
                entries.update(blob.get("entries", {}))
                print(f"  [baseline_cache] loaded {len(entries)} episode(s) from {path}")
            else:
                n_stale += len(blob.get("entries", {}))
        except Exception as exc:
            print(f"  [baseline_cache] failed to load {path}: {exc} — ignoring it")

    d = _shard_dir(path)
    if os.path.isdir(d):
        n_shard = 0
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".pt"):
                continue
            try:
                blob = torch.load(os.path.join(d, fn), map_location="cpu", weights_only=False)
            except Exception as exc:
                print(f"  [baseline_cache] shard {fn} unreadable ({exc}) — will refit it")
                continue
            if blob.get("fingerprint") != fingerprint:
                n_stale += 1
                continue
            entries[blob["cache_key"]] = blob["entry"]
            n_shard += 1
        if n_shard:
            print(f"  [baseline_cache] loaded {n_shard} episode(s) from {d}")

    if n_stale:
        print(f"  [baseline_cache] ignored {n_stale} entr(ies) built under different "
              "generation/fitting settings — those episodes will be refitted")
    return entries


def save_baseline_cache(path: str, fingerprint: dict, entries: dict[str, dict]) -> None:
    """Write the whole cache to one file atomically (legacy layout, still readable)."""
    atomic_torch_save({"fingerprint": fingerprint, "entries": entries}, path)
    print(f"  [baseline_cache] saved {len(entries)} episode(s) to {path}", flush=True)
