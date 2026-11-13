"""Classical GP-MLE baseline: MAP-fitted gpytorch GP posterior (or prior) correlation at the test points.

fit_and_eval_gpytorch is the single fit routine. eval.baselines.classical
runs it per episode for the baseline tables, and the training loop's ERA5
probes (training.era5_probes) use fit_gp_baseline_nll as a fixed reference NLL.
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Callable

import gpytorch
import numpy as np
import torch
import torch.nn as nn
from gpytorch.priors import GammaPrior, LogNormalPrior, Prior
from linear_operator.utils.warnings import NumericalWarning
from torch import Tensor
from torch.optim import Adam

from copula_inter.copula.loss import gp_oracle_y_nll

__all__ = [
    "ARD_ELIGIBLE",
    "GP_LR_MLE",
    "GP_MEANS",
    "fit_and_eval_gpytorch",
    "fit_gp_baseline_nll",
    "gp_map_objective",
    "laplace_log_evidence",
    "sample_prior_gp",
]

# Default Adam learning rate of the MAP fit (eval_checkpoint and the ERA5 training probes).
GP_LR_MLE = 0.05

# Silence psd_safe_cholesky's jitter warnings during fitting.
warnings.filterwarnings("ignore", category=NumericalWarning)


# Mean functions of the fitted GP: zero, a learned constant, or a learned affine function of x.
GP_MEANS = ("zero", "constant", "linear")


# Matern nu per kernel name.
_MATERN_NU = {"matern12": 0.5, "matern32": 1.5, "matern52": 2.5}


ARD_ELIGIBLE = {
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


# Fallback hyperprior constants (gp_episodes defaults), used when prior_cfg lacks a key.
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
    """LogNormal/Gamma hyperpriors matching gp_episodes's generative priors, for MAP fitting.

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
        model.covar_module.variance = (
            kernel_priors["variance_prior"].sample(model.covar_module.variance.shape).to(device)
        )
    if "outputscale_prior" in kernel_priors:
        model.covar_module.outputscale = (
            kernel_priors["outputscale_prior"].sample(model.covar_module.outputscale.shape).to(device)
        )


def _sampled_likelihood(noise_prior: Prior, device: torch.device) -> gpytorch.likelihoods.GaussianLikelihood:
    """Gaussian likelihood with noise in exp(-8)..exp(2) under noise_prior, its noise drawn from that prior.

    Built fresh per fit (the constraint's bounds are mutable tensors).
    """
    noise_constraint = gpytorch.constraints.Interval(math.exp(-8.0), math.exp(2.0))
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint, noise_prior=noise_prior)
    # Random initial noise inside the constraint.
    likelihood.noise = (
        noise_prior.sample(likelihood.noise.shape).to(device).clamp(min=math.exp(-8.0) * 1.01, max=math.exp(2.0) * 0.99)
    )
    return likelihood


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
        mean: str = "zero",
    ) -> None:
        super().__init__(X_train, y_train, likelihood)
        self.feature_extractor = feature_extractor
        # The mean acts on the raw inputs (before any feature extractor), initialized at y_train's mean.
        y_bar = float(y_train.mean()) if y_train.numel() else 0.0
        if mean == "zero":
            self.mean_module: gpytorch.means.Mean = gpytorch.means.ZeroMean()
        elif mean == "constant":
            self.mean_module = gpytorch.means.ConstantMean()
            self.mean_module.initialize(constant=y_bar)
        elif mean == "linear":
            self.mean_module = gpytorch.means.LinearMean(input_size=X_train.shape[-1])
            self.mean_module.initialize(weights=torch.zeros_like(self.mean_module.weights), bias=y_bar)
        else:
            raise ValueError(f"Unknown GP mean {mean!r}; expected one of {GP_MEANS}")
        kp = kernel_priors or {}

        # Omit ard_num_dims instead of passing None (PeriodicKernel does not accept None).
        ard_kw = {} if ard_num_dims is None else {"ard_num_dims": ard_num_dims}

        if kernel_name == "rbf":
            base = gpytorch.kernels.RBFKernel(lengthscale_prior=kp.get("lengthscale_prior"), **ard_kw)
        elif kernel_name in _MATERN_NU:
            base = gpytorch.kernels.MaternKernel(
                nu=_MATERN_NU[kernel_name],
                lengthscale_prior=kp.get("lengthscale_prior"),
                **ard_kw,
            )
        elif kernel_name == "periodic":
            base = gpytorch.kernels.PeriodicKernel(
                lengthscale_prior=kp.get("lengthscale_prior"),
                period_length_prior=kp.get("period_length_prior"),
                **ard_kw,
            )
        elif kernel_name == "rational_quadratic":
            base = gpytorch.kernels.RQKernel(
                lengthscale_prior=kp.get("lengthscale_prior"), alpha_prior=kp.get("alpha_prior"), **ard_kw
            )
        elif kernel_name == "dot_product":
            base = gpytorch.kernels.LinearKernel(variance_prior=kp.get("variance_prior"))
        elif kernel_name == "polynomial":
            base = gpytorch.kernels.PolynomialKernel(power=poly_power, offset_prior=kp.get("offset_prior"))
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")

        # LinearKernel has its own variance, so no ScaleKernel; PolynomialKernel gets one.
        self.covar_module = (
            base
            if kernel_name == "dot_product"
            else gpytorch.kernels.ScaleKernel(base, outputscale_prior=kp.get("outputscale_prior"))
        )

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, self.covar_module(x))


def _submodule_attr(root: nn.Module, dotted: str) -> Tensor:
    """root.<dotted> by attribute access (sees the tensors torch.func.functional_call swaps in)."""
    obj: object = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    assert isinstance(obj, Tensor), dotted
    return obj


def _raw_param_priors(model: _ExactGPModel) -> dict[str, Prior]:
    """{raw parameter name: its registered hyperprior}, e.g. covar_module.raw_outputscale -> outputscale_prior.

    gpytorch registers "<stem>_prior" on the module holding "raw_<stem>"; raise if a
    prior does not follow that convention (its density could not be placed on a raw parameter).
    """
    param_names = {n for n, _ in model.named_parameters()}
    out: dict[str, Prior] = {}
    for name, _module, prior, _closure, _setting in model.named_priors():
        prefix, _, prior_name = name.rpartition(".")
        stem = prior_name.removesuffix("_prior")
        raw = f"{prefix}.raw_{stem}" if prefix else f"raw_{stem}"
        if raw not in param_names:
            raise ValueError(f"prior {name!r} has no raw parameter {raw!r}")
        out[raw] = prior
    return out


class _RawLogJoint(nn.Module):
    """log p(y | g(r)) + sum_j [log pi_j(g_j(r_j)) + log |g_j'(r_j)|] over the raw hyperparameters r.

    g_j is parameter j's gpytorch constraint transform (identity if unconstrained), so the
    prior term is the hyperprior's density pushed to raw coordinates. Parameters without a
    prior (mean function) get a flat density there.
    """

    def __init__(self, model: _ExactGPModel, X: Tensor, y: Tensor, priors: dict[str, Prior]) -> None:
        super().__init__()
        self.model = model
        self.X, self.y = X, y
        self.priors = priors
        self.constraints = {n.removesuffix("_constraint"): c for n, c in model.named_constraints()}

    def forward(self) -> Tensor:
        out = self.model.likelihood(self.model(self.X)).log_prob(self.y)
        for name, prior in self.priors.items():
            raw = _submodule_attr(self.model, name)
            constraint = self.constraints.get(name)
            if constraint is None:
                out = out + prior.log_prob(raw).sum()
                continue
            value = constraint.transform(raw)
            # The transforms are elementwise, so d(sum g)/dr is the Jacobian's diagonal.
            (dvalue,) = torch.autograd.grad(value.sum(), raw, create_graph=True)
            out = out + prior.log_prob(value).sum() + torch.log(dvalue).sum()
        return out


def laplace_log_evidence(
    model: _ExactGPModel,
    X: Tensor,
    y: Tensor,
    extra_priors: dict[str, Prior] | None = None,
    max_iter: int = 200,
) -> float:
    """Laplace approximation of the evidence log p(y | M) = log int p(y | theta) p(theta) dtheta.

    Works in the raw (unconstrained) hyperparameters r, where the prior carries the
    constraint's Jacobian, so the result does not depend on the parameterization:
        log p(y | M) ~= L(r*) + (k / 2) log(2 pi) - (1 / 2) log det H,
        L(r) = log p(y | g(r)) + log pi(r),  r* = argmax L,  H = -grad^2 L(r*),
    with k = dim r. r* is found by L-BFGS from the fitted model (a copy, in float64), whose
    MAP objective omits the Jacobian (and, for held-out selected fits, is not at the mode).
    extra_priors adds priors for raw parameters fitted without one (e.g. ARD lengthscales);
    parameters left without a prior (mean function) are flat, a constant shared by every
    kernel family with the same mean, so it cancels when comparing families. Returns nan if
    H is not positive definite or the optimization fails.
    """
    m = copy.deepcopy(model).double()
    X64, y64 = X.double(), y.double()
    m.set_train_data(inputs=X64, targets=y64, strict=False)
    m.train()
    priors = _raw_param_priors(m)
    for name, prior in (extra_priors or {}).items():
        priors.setdefault(name, copy.deepcopy(prior).double())
    joint = _RawLogJoint(m, X64, y64, priors)

    names = [n for n, p in m.named_parameters() if p.requires_grad]
    params = [m.get_parameter(n) for n in names]
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        opt.zero_grad()
        loss = -joint()
        loss.backward()
        return loss

    try:
        opt.step(closure)
        shapes = [p.shape for p in params]
        sizes = [p.numel() for p in params]
        flat = torch.cat([p.detach().reshape(-1) for p in params])

        def neg_joint(v: Tensor) -> Tensor:
            swapped = {f"model.{n}": c.reshape(s) for n, c, s in zip(names, torch.split(v, sizes), shapes)}
            return -torch.func.functional_call(joint, swapped, ())

        H = torch.autograd.functional.hessian(neg_joint, flat)
        assert isinstance(H, Tensor)
        # The Jacobian term differentiates through r, so r must require grad here too.
        log_joint = -neg_joint(flat.clone().requires_grad_(True)).item()
    except Exception as exc:  # a non-PSD kernel matrix along the line search
        print(f"  [laplace_log_evidence] failed: {exc}")
        return float("nan")
    L, info = torch.linalg.cholesky_ex(0.5 * (H + H.T))
    if int(info) != 0 or not math.isfinite(log_joint):
        return float("nan")
    k = flat.numel()
    logdet = 2.0 * torch.log(torch.diagonal(L)).sum().item()
    return log_joint + 0.5 * k * math.log(2.0 * math.pi) - 0.5 * logdet


def gp_map_objective(model: _ExactGPModel, X: Tensor, y: Tensor) -> float:
    """The MAP fit's objective J(theta) = log p(y | X, theta) + sum_j log pi_j(theta_j) at the model's hyperparameters.

    The priors are the ones registered on the model, evaluated on the constrained values
    (no Jacobian), exactly what fit_and_eval_gpytorch maximizes (times |y|).
    """
    model.set_train_data(inputs=X, targets=y, strict=False)
    model.train()
    with torch.no_grad():
        out = model.likelihood(model(X)).log_prob(y)
        for _name, module, prior, closure, _setting in model.named_priors():
            out = out + prior.log_prob(closure(module)).sum()
    return float(out)


def sample_prior_gp(
    X: Tensor, kernel_name: str, prior_cfg: dict | None = None, mean_constant: float = 0.0
) -> _ExactGPModel:
    """A constant-mean GP over X whose kernel hyperparameters and noise are drawn from the MAP fit's own hyperpriors.

    The well-specified generator: data sampled from it (model.train(); likelihood(model(X)).sample())
    follow exactly the model that fit_and_eval_gpytorch(kernel_name, ard=False, mean="constant") fits.
    """
    cfg = prior_cfg or {}
    kernel_priors = _kernel_priors(cfg, kernel_name)
    likelihood = _sampled_likelihood(_noise_prior(cfg), X.device)
    y0 = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
    model = _ExactGPModel(X, y0, likelihood, kernel_name, kernel_priors=kernel_priors, mean="constant").to(X.device)
    _randomize_init(model, kernel_priors, kernel_name, lengthscale_init_prior=_lengthscale_init_prior(cfg))
    model.mean_module.initialize(constant=mean_constant)
    return model


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
    mean: str = "zero",
    laplace: bool = False,
) -> dict[str, Tensor]:
    """Fit a GP (or DKL) on raw y_train; return {"R", "mean", "Sigma", "log_ml", "n_hparams", "log_evidence", "map_objective"}.

    Maximizes the exact marginal likelihood plus the hyperpriors (MAP) from
    n_restarts random initializations and keeps the best. oracle_mode="posterior"
    returns the fitted posterior at X_test (with noise); "prior" returns the
    fitted kernel's prior covariance at X_test (with noise). "polynomial" repeats
    the fit for every candidate degree. With a feature extractor, or when
    val_select applies, 20% of the training points are held out and the step
    with the best held-out NLL is kept (not below P = 8). mean and Sigma are in
    raw y units. mean is the GP mean function (GP_MEANS). log_ml is the log
    marginal likelihood log p(y_train | X_train, theta_hat) of the kept fit
    (no hyperprior term), n_hparams its trainable parameter count; together
    they give a context-only model-selection score (e.g. BIC). With laplace=True
    (no feature extractor), log_evidence is laplace_log_evidence of the kept
    fit, with the lengthscale prior on ARD lengthscales; otherwise nan.
    map_objective is gp_map_objective of the kept fit on the full context.
    """
    if mean not in GP_MEANS:
        raise ValueError(f"Unknown GP mean {mean!r}; expected one of {GP_MEANS}")
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
                likelihood = _sampled_likelihood(noise_prior, X_train.device)

                # Fresh feature extractor each restart.
                feature_extractor = feature_extractor_factory() if feature_extractor_factory is not None else None
                model = _ExactGPModel(
                    X_fit,
                    y_fit,
                    likelihood,
                    kernel_name,
                    ard_num_dims=ard_num_dims,
                    feature_extractor=feature_extractor,
                    kernel_priors=kernel_priors,
                    poly_power=poly_power,
                    mean=mean,
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
                    assert best_step_state is not None
                    model.load_state_dict(best_step_state[0])
                    likelihood.load_state_dict(best_step_state[1])
                    final_loss = best_step_val
                else:
                    assert loss is not None
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
    assert best_likelihood is not None
    model, likelihood = best_model, best_likelihood
    if use_val:
        # Condition on the full training set for the final evaluation.
        model.set_train_data(inputs=X_train, targets=y_train, strict=False)
    # Log marginal likelihood of the full context under the kept hyperparameters (train mode = prior marginal).
    model.train()
    likelihood.train()
    with torch.no_grad():
        log_ml = likelihood(model(X_train)).log_prob(y_train)
    n_hparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    map_objective = gp_map_objective(model, X_train, y_train)
    model.train()
    likelihood.train()
    log_evidence = float("nan")
    if laplace and feature_extractor_factory is None:
        extra: dict[str, Prior] = {}
        if ard_num_dims is not None:
            # ARD fits carry no lengthscale prior; score them under the shared one.
            extra["covar_module.base_kernel.raw_lengthscale"] = lengthscale_init_prior
        log_evidence = laplace_log_evidence(model, X_train, y_train, extra)
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

    from copula_inter.data.gp_episodes import (
        sigma_to_correlation,  # (lazy: keeps module import light for callers that only need corr_nll_single/gp_prior_corr_rbf)
    )

    R, _ = sigma_to_correlation(Sigma_post)
    return {
        "R": R,
        "mean": mean_post,
        "Sigma": Sigma_post,
        "log_ml": log_ml.detach(),
        "n_hparams": torch.tensor(float(n_hparams)),
        "log_evidence": torch.tensor(log_evidence),
        "map_objective": torch.tensor(map_objective),
    }


# GP baseline NLL cache (fits do not depend on the checkpoint).
_GP_BASELINE_CACHE: dict = {}


def fit_gp_baseline_nll(
    cache_key: tuple,
    x_train_norm: np.ndarray,
    context_values: np.ndarray,
    x_test_norm: np.ndarray,
    y_test: np.ndarray,
    kernel_names: list,
    n_steps: int,
    lr: float,
    n_restarts: int,
    device: str,
) -> dict:
    """Classical GP-MLE baseline Y-space NLL (total, marginal, copula) per kernel on the held-out points.

    Fitted on standardized context values (posterior mode) and rescaled to
    Kelvin before scoring with gp_oracle_y_nll; cached per cache_key.
    """
    key = cache_key + (tuple(kernel_names), n_steps, lr, n_restarts)
    if key in _GP_BASELINE_CACHE:
        return _GP_BASELINE_CACHE[key]

    X_train = torch.as_tensor(x_train_norm, dtype=torch.float32, device=device)
    X_test = torch.as_tensor(x_test_norm, dtype=torch.float32, device=device)
    y_test_t = torch.as_tensor(y_test, dtype=torch.float32, device=device)

    mu_y = float(context_values.mean())
    sigma_y = max(float(context_values.std(ddof=1)), 1e-6) if len(context_values) > 1 else 1.0
    y_train_std = torch.as_tensor((context_values - mu_y) / sigma_y, dtype=torch.float32, device=device)

    n = X_test.shape[0]
    mask = torch.ones(1, n, dtype=torch.bool, device=device)

    out = {}
    for kname in kernel_names:
        try:
            fit = fit_and_eval_gpytorch(
                X_train,
                y_train_std,
                X_test,
                kname,
                n_steps=n_steps,
                lr=lr,
                oracle_mode="posterior",
                n_restarts=n_restarts,
            )
            mean_real = fit["mean"] * sigma_y + mu_y
            Sigma_real = fit["Sigma"] * (sigma_y**2)
            parts = gp_oracle_y_nll(
                Sigma_real.unsqueeze(0),
                mean_real.unsqueeze(0),
                y_test_t.unsqueeze(0),
                mask,
            )
            out[kname] = {k: float(v) for k, v in parts.items()}
        except Exception as exc:  # noqa: BLE001
            print(f"  [gp_baseline:{kname}] failed: {exc}")
            out[kname] = {"total": float("nan"), "marginal": float("nan"), "copula": float("nan")}

    _GP_BASELINE_CACHE[key] = out
    return out
