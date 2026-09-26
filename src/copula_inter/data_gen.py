"""GP episode generation for the inter-instance copula.

Each episode samples a GP kernel and its hyperparameters, draws P + N points,
normalizes features over all P + N, samples targets jointly from the GP, and
returns the prior correlation R_star at the test points plus the PIT
inputs (z_train, z_test, log_pdf_test).

Kernels (gpytorch, hyperparameters from LogNormal/Gamma priors, see
_kernel_prior_spec and _nugget_prior):
    rbf, matern12/32/52, cosine, periodic, rational_quadratic,
    dot_product (LinearKernel, variance = alpha2, no lengthscale),
    polynomial (alpha2 * (x1.x2 + c)^d; c stored in "l", d in "power",
        one d per generate_gp_batch call).

cfg.data options:
    kernel / kernels: one kernel name, or a list sampled uniformly (default "rbf").
    ard: per-dimension lengthscales for rbf/matern/periodic/rational_quadratic;
        isotropic_ratio collapses an episode's ARD lengthscale to one value.
    Composite "A+B" / "A*B": every pair of _COMPOSABLE_KERNELS, see
        COMPOSITE_KERNELS.
    systematic_composition: sample a random-length +/* chain per call
        (_sample_kernel_chain_structure); chain names are not in
        KERNEL_REGISTRY and are rebuilt from kernel_component_params.
    sign_modulation_component_prob / _outer_prob: wrap components or the
        composed kernel in SignModulatedKernel,
        K'(x1, x2) = K(x1, x2) s(x1) s(x2), s(x) = tanh(a (w.x + b)).
    d_features, or d_features_lognormal_loc/scale: total feature count per
        call (a batch mixing shards with different d cannot be collated).
"""

from __future__ import annotations

import functools
import itertools
import math
import random
import re
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import gpytorch
import numpy as np
import torch
from gpytorch.priors import GammaPrior, LogNormalPrior, Prior
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.utils.errors import NanError, NotPSDError
from torch import Tensor

from copula_inter.episode_contracts import assemble_episodes
from copula_inter.feature_transforms import (
    apply_kernel_hidden_warp,
    apply_mlp_feature_mixing,
)
from copula_inter.loss import _safe_cholesky
from copula_inter.type_aliases import Device, HasDataConfig

if TYPE_CHECKING:
    from copula_inter.pit import TabICLLike

# Force exact Cholesky solves for every covariance up to this size (gpytorch uses CG above max_cholesky_size).
_MAX_CHOLESKY = 8192

# z_train_source values with a batched PIT module, as lazy importers (heavy optional dependencies).
from copula_inter.backend_registry import batched_backend_factories

_BATCHED_MARGINAL_BACKENDS: Dict[str, Callable[[], Callable]] = batched_backend_factories()


def _seed_everything(seed: int) -> None:
    """Seed python, numpy and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe even with a single GPU / no GPU


def _sq_dist(X1: Tensor, X2: Tensor) -> Tensor:
    """Squared Euclidean distance matrix (n1, n2)."""
    diff = X1.unsqueeze(1) - X2.unsqueeze(0)  # (n1, n2, d)
    return (diff**2).sum(-1)


def _dist(X1: Tensor, X2: Tensor) -> Tensor:
    """Euclidean distance matrix (n1, n2)."""
    return _sq_dist(X1, X2).clamp(min=0.0).sqrt()


# Kernel construction: _sample_episode_kernel samples B episodes' kernels;
# build_kernel_fn rebuilds a kernel from concrete saved hyperparameters.

_BASE_GPYTORCH_KERNEL_CLS: Dict[str, Callable[..., gpytorch.kernels.Kernel]] = {
    "rbf": gpytorch.kernels.RBFKernel,
    "matern12": functools.partial(gpytorch.kernels.MaternKernel, nu=0.5),
    "matern32": functools.partial(gpytorch.kernels.MaternKernel, nu=1.5),
    "matern52": functools.partial(gpytorch.kernels.MaternKernel, nu=2.5),
    "cosine": gpytorch.kernels.CosineKernel,
    "periodic": gpytorch.kernels.PeriodicKernel,
    "rational_quadratic": gpytorch.kernels.RQKernel,
}

# Schema name -> gpytorch attribute for the extra (non-lengthscale) parameters.
_EXTRA_PARAM_TO_ATTR: Dict[str, str] = {"period": "period_length", "rq_alpha": "alpha"}


@dataclass
class KernelPriorSpec:
    """Hyperprior distributions for one base kernel family.

    lengthscale_prior(k) is the prior over lengthscale_attr (the lengthscale, or
    cosine's period_length) for k active dims; ard=True samples one value per dim.
    """

    lengthscale_prior: Callable[[int], Prior]
    outputscale_prior: Prior
    lengthscale_attr: str = "lengthscale"
    extra_priors: Dict[str, Prior] = field(default_factory=dict)
    ard: bool = False
    # Per-episode probability of collapsing an ARD lengthscale/period to one value.
    isotropic_ratio: float = 0.0


# Kernels whose lengthscale can be ARD (cosine's period_length is always scalar).
_ARD_ELIGIBLE_KERNELS = frozenset({"rbf", "matern12", "matern32", "matern52", "periodic", "rational_quadratic"})


def _kernel_prior_spec(cfg: HasDataConfig, kernel_name: str) -> KernelPriorSpec:
    """LogNormal/Gamma hyperprior spec for one base kernel family, overridable via cfg.data."""
    isotropic_ratio = float(getattr(cfg.data, "isotropic_ratio", 0.0))

    l_loc = float(getattr(cfg.data, "l_lognormal_loc", 0.0))
    l_scale = float(getattr(cfg.data, "l_lognormal_scale", 0.7))
    a_conc = float(getattr(cfg.data, "alpha2_gamma_concentration", 4.0))
    a_rate = float(getattr(cfg.data, "alpha2_gamma_rate", 3.0))
    ard = bool(getattr(cfg.data, "ard", False)) and kernel_name in _ARD_ELIGIBLE_KERNELS

    # Shift the lengthscale prior by 0.25*log(k) (k capped) so correlations don't vanish as k grows.
    k_exponent = float(getattr(cfg.data, "l_lognormal_k_exponent", 0.25))
    k_cap = float(getattr(cfg.data, "l_lognormal_k_cap", 15))

    def lengthscale_prior(k: int) -> LogNormalPrior:
        shift = k_exponent * math.log(max(min(k, k_cap), 1))
        return LogNormalPrior(l_loc + shift, l_scale)

    # cosine's shape parameter is period_length.
    lengthscale_attr = "period_length" if kernel_name == "cosine" else "lengthscale"

    extra_priors: Dict[str, Prior] = {}
    if kernel_name == "periodic":
        p_loc = float(getattr(cfg.data, "period_lognormal_loc", math.log(1.2)))
        p_scale = float(getattr(cfg.data, "period_lognormal_scale", 0.4))
        extra_priors["period"] = LogNormalPrior(p_loc, p_scale)
    elif kernel_name == "rational_quadratic":
        rq_conc = float(getattr(cfg.data, "rq_alpha_gamma_concentration", 2.0))
        rq_rate = float(getattr(cfg.data, "rq_alpha_gamma_rate", 1.0))
        extra_priors["rq_alpha"] = GammaPrior(rq_conc, rq_rate)

    return KernelPriorSpec(
        lengthscale_prior=lengthscale_prior,
        outputscale_prior=GammaPrior(a_conc, a_rate),
        lengthscale_attr=lengthscale_attr,
        extra_priors=extra_priors,
        ard=ard,
        isotropic_ratio=isotropic_ratio,
    )


def _nugget_prior(cfg: HasDataConfig, kernel_name: str) -> LogNormalPrior:
    """Noise (nugget) prior shared by every kernel; default LogNormal(-4.63, 0.5)."""
    loc = float(getattr(cfg.data, "nugget_lognormal_loc", -4.63))
    scale = float(getattr(cfg.data, "nugget_lognormal_scale", 0.5))
    return LogNormalPrior(loc, scale)


def _build_likelihood(
    cfg: HasDataConfig, kernel_name: str, B: int, device: Device
) -> gpytorch.likelihoods.GaussianLikelihood:
    """Sample B episodes' noise from _nugget_prior as a GaussianLikelihood (.noise is the nugget)."""
    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([B])).to(device)
    likelihood.noise = _nugget_prior(cfg, kernel_name).sample(torch.Size([B])).to(device)
    return likelihood


def _collapse_isotropic(sample: Tensor, iso_mask: Optional[Tensor]) -> Tensor:
    """For episodes flagged in iso_mask, set every entry of the last (ARD) axis to the first entry.

    No-op when iso_mask is None or the last axis has size 1.
    """
    if iso_mask is None or sample.shape[-1] == 1:
        return sample
    collapsed = sample[..., :1].expand_as(sample)
    mask = iso_mask.view(-1, *([1] * (sample.dim() - 1)))
    return torch.where(mask, collapsed, sample)


def _build_scaled_kernel(
    name: str, spec: KernelPriorSpec, k: int, B: int, device: Device, active_dims: Optional[List[int]] = None
) -> tuple[gpytorch.kernels.Kernel, Dict[str, Tensor]]:
    """Sample B episodes' hyperparameters for one base kernel.

    Returns (ScaleKernel(base) with batch_shape=[B], params dict keyed l, alpha2
    and any extra prior names). active_dims selects columns from the full-width input.
    """
    batch_shape = torch.Size([B])
    kernel_kwargs: Dict = {"batch_shape": batch_shape}
    if active_dims is not None:
        kernel_kwargs["active_dims"] = active_dims
    if spec.ard:
        kernel_kwargs["ard_num_dims"] = k
    # Move the kernel to `device` before assigning sampled values.
    base = _BASE_GPYTORCH_KERNEL_CLS[name](**kernel_kwargs).to(device)

    # One isotropic-collapse draw per episode, shared by lengthscale and period.
    iso_mask = torch.rand(B, device=device) < spec.isotropic_ratio if spec.ard and spec.isotropic_ratio > 0.0 else None

    l_attr = getattr(base, spec.lengthscale_attr)
    l_sample = spec.lengthscale_prior(k).sample(l_attr.shape).to(device)
    l_sample = _collapse_isotropic(l_sample, iso_mask)
    setattr(base, spec.lengthscale_attr, l_sample)

    scaled = gpytorch.kernels.ScaleKernel(base, batch_shape=batch_shape).to(device)
    a_sample = spec.outputscale_prior.sample(scaled.outputscale.shape).to(device)
    scaled.outputscale = a_sample

    l_flat = l_sample.reshape(B, -1)
    params: Dict[str, Tensor] = {
        "l": l_flat.squeeze(-1) if l_flat.shape[-1] == 1 else l_flat,
        "alpha2": a_sample.reshape(B),
    }
    for schema_name, prior in spec.extra_priors.items():
        attr_name = _EXTRA_PARAM_TO_ATTR[schema_name]
        attr = getattr(base, attr_name)
        sample = prior.sample(attr.shape).to(device)
        # period is ARD-shaped under spec.ard and collapses with the same mask; rq_alpha is never ARD.
        sample = _collapse_isotropic(sample, iso_mask)
        setattr(base, attr_name, sample)
        sample_flat = sample.reshape(B, -1)
        params[schema_name] = sample_flat.squeeze(-1) if sample_flat.shape[-1] == 1 else sample_flat

    return scaled, params


class SignModulatedKernel(gpytorch.kernels.Kernel):
    """Sign modulation: K'(x1, x2) = K(x1, x2) * s(x1) * s(x2), s(x) = tanh(a (w.x[active_cols] + b)).

    One (w, b, a) per episode (batch_shape=[B]). K' stays PSD (Schur product with
    the rank-1 s s^T). active_cols is the wrapped kernel's own column subset
    (None = all columns).
    """

    def __init__(
        self,
        base_kernel: gpytorch.kernels.Kernel,
        w: Tensor,
        b: Tensor,
        a: Tensor,
        active_dims: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> None:
        # batch_shape comes from w's leading dim (B,).
        super().__init__(batch_shape=torch.Size([w.shape[0]]), **kwargs)
        self.base_kernel = base_kernel
        self.register_buffer("w", w)  # (B, k)
        self.register_buffer("b", b)  # (B,)
        self.register_buffer("a", a)  # (B,) sharpness, > 0
        self.active_cols = list(active_dims) if active_dims is not None else None

    def _signs(self, x: Tensor) -> Tensor:
        """s(x) = tanh(a * (w . x[..., active_cols] + b)), shape (..., n) for x of shape (..., n, d)."""
        cols = self.active_cols
        x_active = x[..., cols] if cols is not None else x
        w = self.w.unsqueeze(-2)  # (B, 1, k)
        b = self.b.unsqueeze(-1)  # (B, 1)
        a = self.a.unsqueeze(-1)  # (B, 1)
        z = (x_active * w).sum(-1) + b
        return torch.tanh(a * z)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params: Any) -> Tensor:
        K = self.base_kernel(x1, x2, diag=diag, **params)
        K = K.to_dense() if hasattr(K, "to_dense") else K
        s1 = self._signs(x1)
        s2 = self._signs(x2)
        if diag:
            return K * s1 * s2
        return K * s1.unsqueeze(-1) * s2.unsqueeze(-2)


class _DenseComposedKernel(gpytorch.kernels.Kernel):
    """Combine two kernels with dense tensor + or * instead of gpytorch's lazy AdditiveKernel/ProductKernel.

    Avoids LinearOperator's eager low-rank factorization, which raises on the
    near-singular sums that dot_product/polynomial components produce.
    """

    def __init__(
        self, kernel_a: gpytorch.kernels.Kernel, op: str, kernel_b: gpytorch.kernels.Kernel, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert op in ("+", "*"), f"op must be '+' or '*', got {op!r}"
        self.kernel_a = kernel_a
        self.op = op
        self.kernel_b = kernel_b

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params: Any) -> Tensor:
        a = self.kernel_a(x1, x2, diag=diag, **params)
        b = self.kernel_b(x1, x2, diag=diag, **params)
        a = a.to_dense() if hasattr(a, "to_dense") else a
        b = b.to_dense() if hasattr(b, "to_dense") else b
        return a + b if self.op == "+" else a * b


def _sample_sign_modulation(cfg: HasDataConfig, k: int, B: int, device: Device) -> tuple[Tensor, Tensor, Tensor]:
    """Sample one hyperplane per episode: w ~ N(0, I_k)/sqrt(k), b ~ N(0, 1), a ~ LogNormal(sharpness loc, scale)."""
    w = torch.randn(B, k, device=device) / math.sqrt(max(k, 1))
    b = torch.randn(B, device=device)
    a_loc = float(getattr(cfg.data, "sign_modulation_sharpness_lognormal_loc", math.log(3.0)))
    a_scale = float(getattr(cfg.data, "sign_modulation_sharpness_lognormal_scale", 0.5))
    a = LogNormalPrior(a_loc, a_scale).sample(torch.Size([B])).to(device)
    return w, b, a


def _maybe_wrap_sign_modulated(
    cfg: HasDataConfig,
    kernel: gpytorch.kernels.Kernel,
    prob: float,
    k: int,
    B: int,
    device: Device,
    active_dims: Optional[List[int]],
    param_suffix: str = "",
) -> tuple[gpytorch.kernels.Kernel, Dict[str, Tensor]]:
    """With probability prob (once per call), wrap kernel in SignModulatedKernel with per-episode (w, b, a).

    Returns (kernel, params) with sign_applied{suffix} (0.0/1.0), sign_w{suffix}
    (B, k), sign_b{suffix} (B,) and sign_a{suffix} (B,); zero-filled when not applied.
    """
    if prob > 0.0 and random.random() < prob:
        w, b, a = _sample_sign_modulation(cfg, k, B, device)
        wrapped = SignModulatedKernel(kernel, w, b, a, active_dims=active_dims)
        params = {
            f"sign_applied{param_suffix}": torch.ones(B, device=device),
            f"sign_w{param_suffix}": w,
            f"sign_b{param_suffix}": b,
            f"sign_a{param_suffix}": a,
        }
        return wrapped, params
    params = {
        f"sign_applied{param_suffix}": torch.zeros(B, device=device),
        f"sign_w{param_suffix}": torch.zeros(B, max(k, 1), device=device),
        f"sign_b{param_suffix}": torch.zeros(B, device=device),
        f"sign_a{param_suffix}": torch.zeros(B, device=device),
    }
    return kernel, params


def _build_kernel_component(
    cfg: HasDataConfig,
    name: str,
    k: int,
    B: int,
    device: Device,
    active_dims: Optional[List[int]] = None,
    d_total: Optional[int] = None,
) -> tuple[gpytorch.kernels.Kernel, Dict[str, Tensor]]:
    """Build one elementary kernel for B episodes and its sampled hyperparameter dict.

    Applies per-component sign modulation (sign_modulation_component_prob).
    "dot_product" is a bare LinearKernel whose variance is drawn from the alpha2
    prior; it always uses every column (active_dims is ignored), and d_total sizes
    its sign-modulation hyperplane (defaults to k).
    """
    sign_prob = float(getattr(cfg.data, "sign_modulation_component_prob", 0.0))

    if name == "dot_product":
        kernel = gpytorch.kernels.LinearKernel(batch_shape=torch.Size([B])).to(device)
        a_conc = float(getattr(cfg.data, "alpha2_gamma_concentration", 4.0))
        a_rate = float(getattr(cfg.data, "alpha2_gamma_rate", 3.0))
        a_sample = GammaPrior(a_conc, a_rate).sample(kernel.variance.shape).to(device)
        kernel.variance = a_sample
        params: Dict[str, Tensor] = {
            "l": torch.zeros(B, device=device),
            "alpha2": a_sample.reshape(B),
        }
        # dot_product's hyperplane spans every column (d_total).
        kernel, sign_params = _maybe_wrap_sign_modulated(
            cfg, kernel, sign_prob, d_total if d_total is not None else k, B, device, active_dims=None
        )
        params.update(sign_params)
        return kernel, params
    if name == "polynomial":
        # One polynomial degree per call (PolynomialKernel accepts a single power).
        power_min = int(getattr(cfg.data, "poly_power_min", 2))
        power_max = int(getattr(cfg.data, "poly_power_max", 4))
        power = random.randint(power_min, power_max)
        kernel_kwargs: Dict = {"power": power, "batch_shape": torch.Size([B])}
        if active_dims is not None:
            kernel_kwargs["active_dims"] = active_dims
        base = gpytorch.kernels.PolynomialKernel(**kernel_kwargs).to(device)
        o_conc = float(getattr(cfg.data, "poly_offset_gamma_concentration", 2.0))
        o_rate = float(getattr(cfg.data, "poly_offset_gamma_rate", 1.0))
        o_sample = GammaPrior(o_conc, o_rate).sample(base.offset.shape).to(device)
        base.offset = o_sample

        scaled = gpytorch.kernels.ScaleKernel(base, batch_shape=torch.Size([B])).to(device)
        a_conc = float(getattr(cfg.data, "alpha2_gamma_concentration", 4.0))
        a_rate = float(getattr(cfg.data, "alpha2_gamma_rate", 3.0))
        a_sample = GammaPrior(a_conc, a_rate).sample(scaled.outputscale.shape).to(device)
        scaled.outputscale = a_sample

        params = {
            # The offset is stored in the "l" slot.
            "l": o_sample.reshape(B),
            "alpha2": a_sample.reshape(B),
            "power": torch.full((B,), float(power), device=device),
        }
        scaled, sign_params = _maybe_wrap_sign_modulated(cfg, scaled, sign_prob, k, B, device, active_dims=active_dims)
        params.update(sign_params)
        return scaled, params
    spec = _kernel_prior_spec(cfg, name)
    scaled, params = _build_scaled_kernel(name, spec, k, B, device, active_dims=active_dims)
    scaled, sign_params = _maybe_wrap_sign_modulated(cfg, scaled, sign_prob, k, B, device, active_dims=active_dims)
    params.update(sign_params)
    return scaled, params


def _sample_episode_kernel(
    cfg: HasDataConfig,
    kernel_name: str,
    k: int,
    B: int,
    device: Device,
    active_dims: Optional[List[int]] = None,
    d_total: Optional[int] = None,
) -> tuple[gpytorch.kernels.Kernel, Dict[str, Tensor]]:
    """Sample B episodes' kernel for kernel_name (base or "A+B"/"A*B").

    Returns (gpytorch Kernel with batch_shape=[B], params) with keys l, alpha2,
    period, rq_alpha, power, their _b versions, and sign_* / sign_*_b /
    sign_*_outer; not-applicable entries are 0.0. active_dims: columns the kernel
    uses (None = all). Applies whole-kernel sign modulation
    (sign_modulation_outer_prob) last.
    """
    d_total = d_total if d_total is not None else k
    composite = _parse_composite(kernel_name)
    if composite is None:
        kernel, params = _build_kernel_component(
            cfg, kernel_name, k, B, device, active_dims=active_dims, d_total=d_total
        )
    else:
        name_a, op, name_b = composite
        kernel_a, params_a = _build_kernel_component(
            cfg, name_a, k, B, device, active_dims=active_dims, d_total=d_total
        )
        kernel_b, params_b = _build_kernel_component(
            cfg, name_b, k, B, device, active_dims=active_dims, d_total=d_total
        )
        kernel = _DenseComposedKernel(kernel_a, op, kernel_b)
        params = dict(params_a)
        for key, val in params_b.items():
            params[f"{key}_b"] = val

    for key in ("period", "rq_alpha", "power", "l_b", "alpha2_b", "period_b", "rq_alpha_b", "power_b"):
        params.setdefault(key, torch.zeros(B, device=device))

    outer_prob = float(getattr(cfg.data, "sign_modulation_outer_prob", 0.0))
    kernel, outer_params = _maybe_wrap_sign_modulated(
        cfg, kernel, outer_prob, k, B, device, active_dims=active_dims, param_suffix="_outer"
    )
    params.update(outer_params)

    return kernel, params


def _wrap_concrete_sign_modulated(
    kernel: gpytorch.kernels.Kernel,
    sign_w: Optional[Tensor],
    sign_b: Optional[Tensor],
    sign_a: Optional[Tensor],
    active_dims: Optional[List[int]],
) -> gpytorch.kernels.Kernel:
    """Wrap a non-batched kernel in SignModulatedKernel with given sign_w/sign_b/sign_a.

    No-op when sign_w or sign_b is None. sign_a=None uses a very large sharpness
    (a hard sign).
    """
    if sign_w is None or sign_b is None:
        return kernel
    w_t = sign_w if torch.is_tensor(sign_w) else torch.as_tensor(sign_w, dtype=torch.get_default_dtype())
    b_t = sign_b if torch.is_tensor(sign_b) else torch.as_tensor(sign_b, dtype=torch.get_default_dtype())
    if sign_a is None:
        a_t = torch.full_like(b_t, 1e6)
    else:
        a_t = sign_a if torch.is_tensor(sign_a) else torch.as_tensor(sign_a, dtype=torch.get_default_dtype())
    w_t = w_t.reshape(1, -1)
    b_t = b_t.reshape(1)
    a_t = a_t.reshape(1)
    return SignModulatedKernel(kernel, w_t, b_t, a_t, active_dims=active_dims)


def _build_concrete_kernel(
    name: str,
    l: float | Tensor,
    alpha2: float | Tensor,
    *,
    period: Optional[float | Tensor] = None,
    rq_alpha: Optional[float] = None,
    power: Optional[float | int] = None,
    active_dims: Optional[List[int]] = None,
    sign_w: Optional[Tensor] = None,
    sign_b: Optional[Tensor] = None,
    sign_a: Optional[Tensor] = None,
) -> gpytorch.kernels.Kernel:
    """Build a non-batched gpytorch Kernel with given hyperparameter values (used by build_kernel_fn).

    "dot_product" is a bare LinearKernel on every column (l and active_dims
    ignored). "polynomial" reads its offset from l and its degree from power
    (default 2). sign_w/sign_b/sign_a, when given, wrap the result via
    _wrap_concrete_sign_modulated.
    """
    if name == "dot_product":
        kernel = gpytorch.kernels.LinearKernel()
        kernel.variance = torch.as_tensor(alpha2, dtype=torch.get_default_dtype()).reshape(kernel.variance.shape)
        return _wrap_concrete_sign_modulated(kernel, sign_w, sign_b, sign_a, active_dims=None)

    if name == "polynomial":
        power_int = int(round(float(power))) if power is not None else 2
        kernel_kwargs: dict[str, Any] = {"power": power_int}
        if active_dims is not None:
            kernel_kwargs["active_dims"] = active_dims
        base = gpytorch.kernels.PolynomialKernel(**kernel_kwargs)
        offset_t = l if torch.is_tensor(l) else torch.as_tensor(l, dtype=torch.get_default_dtype())
        base.offset = offset_t.reshape(base.offset.shape)
        scale = gpytorch.kernels.ScaleKernel(base)
        scale.outputscale = torch.as_tensor(alpha2, dtype=torch.get_default_dtype()).reshape(scale.outputscale.shape)
        return _wrap_concrete_sign_modulated(scale, sign_w, sign_b, sign_a, active_dims=active_dims)

    l_t = l if torch.is_tensor(l) else torch.as_tensor(l, dtype=torch.get_default_dtype())
    # A multi-element l means ARD; gpytorch needs ard_num_dims at construction.
    kernel_kwargs = {"ard_num_dims": l_t.numel()} if l_t.numel() > 1 else {}
    if active_dims is not None:
        kernel_kwargs["active_dims"] = active_dims
    base = _BASE_GPYTORCH_KERNEL_CLS[name](**kernel_kwargs)
    attr = "period_length" if name == "cosine" else "lengthscale"
    setattr(base, attr, l_t.reshape(getattr(base, attr).shape))
    if name == "periodic" and period is not None:
        period_t = period if torch.is_tensor(period) else torch.as_tensor(float(period))
        base.period_length = period_t.reshape(base.period_length.shape)
    if name == "rational_quadratic" and rq_alpha is not None:
        base.alpha = torch.as_tensor(float(rq_alpha)).reshape(base.alpha.shape)

    scale = gpytorch.kernels.ScaleKernel(base)
    scale.outputscale = torch.as_tensor(alpha2, dtype=torch.get_default_dtype()).reshape(scale.outputscale.shape)
    return _wrap_concrete_sign_modulated(scale, sign_w, sign_b, sign_a, active_dims=active_dims)


def build_kernel_fn(
    kernel_name: str,
    l: float | Tensor,
    alpha2: float | Tensor,
    *,
    period: Optional[float | Tensor] = None,
    rq_alpha: Optional[float] = None,
    power: Optional[float | int] = None,
    l_b: Optional[float | Tensor] = None,
    alpha2_b: Optional[float | Tensor] = None,
    period_b: Optional[float | Tensor] = None,
    rq_alpha_b: Optional[float] = None,
    power_b: Optional[float | int] = None,
    active_dims: Optional[List[int]] = None,
    sign_w: Optional[Tensor] = None,
    sign_b: Optional[Tensor] = None,
    sign_a: Optional[Tensor] = None,
    sign_w_b: Optional[Tensor] = None,
    sign_b_b: Optional[Tensor] = None,
    sign_a_b: Optional[Tensor] = None,
    sign_w_outer: Optional[Tensor] = None,
    sign_b_outer: Optional[Tensor] = None,
    sign_a_outer: Optional[Tensor] = None,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Return a kernel(X1, X2) -> K callable with the given hyperparameters.

    l/period (and their _b versions) may be ARD vectors. The _b arguments are
    component B of an "A+B"/"A*B" composite; power is the polynomial degree.
    active_dims: columns both components use (None = all). sign_w/sign_b/sign_a
    (and _b) are per-component sign-modulation hyperplanes, sign_*_outer the
    whole-kernel one applied last; None means not applied.
    """
    composite = _parse_composite(kernel_name)
    if composite is None:
        kernel = _build_concrete_kernel(
            kernel_name,
            l,
            alpha2,
            period=period,
            rq_alpha=rq_alpha,
            power=power,
            active_dims=active_dims,
            sign_w=sign_w,
            sign_b=sign_b,
            sign_a=sign_a,
        )
    else:
        name_a, op, name_b = composite
        if l_b is None or alpha2_b is None:
            raise ValueError(f"composite kernel {kernel_name!r} needs l_b and alpha2_b")
        kernel_a = _build_concrete_kernel(
            name_a,
            l,
            alpha2,
            period=period,
            rq_alpha=rq_alpha,
            power=power,
            active_dims=active_dims,
            sign_w=sign_w,
            sign_b=sign_b,
            sign_a=sign_a,
        )
        kernel_b = _build_concrete_kernel(
            name_b,
            l_b,
            alpha2_b,
            period=period_b,
            rq_alpha=rq_alpha_b,
            power=power_b,
            active_dims=active_dims,
            sign_w=sign_w_b,
            sign_b=sign_b_b,
            sign_a=sign_a_b,
        )
        kernel = _DenseComposedKernel(kernel_a, op, kernel_b)

    kernel = _wrap_concrete_sign_modulated(kernel, sign_w_outer, sign_b_outer, sign_a_outer, active_dims=active_dims)

    # Move the kernel to X1's device at call time.
    return lambda X1, X2: kernel.to(X1.device)(X1, X2).to_dense()


# Kernel registry: named free functions that evaluate build_kernel_fn.
# Hyperparameters are assigned in place, so these do not backpropagate into them.


def rbf_kernel(X1: Tensor, X2: Tensor, *, l: float | Tensor, alpha2: float | Tensor, **_: Any) -> Tensor:
    """Squared exponential (RBF), via gpytorch.kernels.RBFKernel."""
    return build_kernel_fn("rbf", l, alpha2)(X1, X2)


def matern12_kernel(X1: Tensor, X2: Tensor, *, l: float | Tensor, alpha2: float | Tensor, **_: Any) -> Tensor:
    """Matern nu=1/2, via gpytorch.kernels.MaternKernel(nu=0.5)."""
    return build_kernel_fn("matern12", l, alpha2)(X1, X2)


def matern32_kernel(X1: Tensor, X2: Tensor, *, l: float | Tensor, alpha2: float | Tensor, **_: Any) -> Tensor:
    """Matern nu=3/2, via gpytorch.kernels.MaternKernel(nu=1.5)."""
    return build_kernel_fn("matern32", l, alpha2)(X1, X2)


def matern52_kernel(X1: Tensor, X2: Tensor, *, l: float | Tensor, alpha2: float | Tensor, **_: Any) -> Tensor:
    """Matern nu=5/2, via gpytorch.kernels.MaternKernel(nu=2.5)."""
    return build_kernel_fn("matern52", l, alpha2)(X1, X2)


def cosine_kernel(X1: Tensor, X2: Tensor, *, l: float | Tensor, alpha2: float | Tensor, **_: Any) -> Tensor:
    """Cosine (spectral), via gpytorch.kernels.CosineKernel."""
    return build_kernel_fn("cosine", l, alpha2)(X1, X2)


def periodic_kernel(
    X1: Tensor, X2: Tensor, *, l: float | Tensor, alpha2: float | Tensor, period: float = 1.0, **_: Any
) -> Tensor:
    """Periodic, via gpytorch.kernels.PeriodicKernel."""
    return build_kernel_fn("periodic", l, alpha2, period=period)(X1, X2)


def rational_quadratic_kernel(
    X1: Tensor, X2: Tensor, *, l: float | Tensor, alpha2: float | Tensor, rq_alpha: float = 1.0, **_: Any
) -> Tensor:
    """Rational quadratic, via gpytorch.kernels.RQKernel."""
    return build_kernel_fn("rational_quadratic", l, alpha2, rq_alpha=rq_alpha)(X1, X2)


def dot_product_kernel(X1: Tensor, X2: Tensor, *, alpha2: float = 1.0, **_: Any) -> Tensor:
    """Linear kernel alpha2 * X1 @ X2^T, via gpytorch.kernels.LinearKernel (l is ignored)."""
    return build_kernel_fn("dot_product", 0.0, alpha2)(X1, X2)


def polynomial_kernel(
    X1: Tensor, X2: Tensor, *, l: float | Tensor, alpha2: float | Tensor, power: float = 2.0, **_: Any
) -> Tensor:
    """Polynomial kernel alpha2 * (x1.x2 + c)^d; l holds the offset c and power the degree d."""
    return build_kernel_fn("polynomial", l, alpha2, power=power)(X1, X2)


KERNEL_REGISTRY: Dict[str, Callable[..., Tensor]] = {
    "rbf": rbf_kernel,
    "matern12": matern12_kernel,
    "matern32": matern32_kernel,
    "matern52": matern52_kernel,
    "cosine": cosine_kernel,
    "periodic": periodic_kernel,
    "rational_quadratic": rational_quadratic_kernel,
    "dot_product": dot_product_kernel,
    "polynomial": polynomial_kernel,
}


# Composite kernels: sum / product of every pair of base kernels.
_COMPOSABLE_KERNELS: List[str] = [
    "rbf",
    "matern12",
    "matern32",
    "matern52",
    "cosine",
    "periodic",
    "rational_quadratic",
    "dot_product",
    "polynomial",
]

# Kernels that are PSD only for scalar inputs; composites containing one use k=1.
_SCALAR_ONLY_KERNELS = {"cosine"}


def _parse_composite(name: str) -> Optional[tuple]:
    """Split "A+B" / "A*B" into (name_a, op, name_b), or None if not composite."""
    for op in ("+", "*"):
        if op in name:
            a, _, b = name.partition(op)
            if a in _COMPOSABLE_KERNELS and b in _COMPOSABLE_KERNELS:
                return a, op, b
    return None


def _kernel_needs_scalar_input(kernel_name: str) -> bool:
    """True if the kernel, or any component of a composite or chain, requires k=1 input dims."""
    return any(part in _SCALAR_ONLY_KERNELS for part in re.split(r"[+*]", kernel_name))


def _composite_kernel(
    X1: Tensor,
    X2: Tensor,
    *,
    kernel_name: str,
    l: float,
    alpha2: float,
    l_b: Optional[float] = None,
    alpha2_b: Optional[float] = None,
    period: Optional[float] = None,
    period_b: Optional[float] = None,
    rq_alpha: Optional[float] = None,
    rq_alpha_b: Optional[float] = None,
    power: Optional[float] = None,
    power_b: Optional[float] = None,
    **_: Any,
) -> Tensor:
    """Evaluate a registered "A+B" / "A*B" kernel through build_kernel_fn."""
    fn = build_kernel_fn(
        kernel_name,
        l,
        alpha2,
        period=period,
        rq_alpha=rq_alpha,
        power=power,
        l_b=l_b,
        alpha2_b=alpha2_b,
        period_b=period_b,
        rq_alpha_b=rq_alpha_b,
        power_b=power_b,
    )
    return fn(X1, X2)


COMPOSITE_KERNELS: List[str] = []
for _name_a, _name_b in itertools.combinations(_COMPOSABLE_KERNELS, 2):
    for _op in ("+", "*"):
        _combo_name = f"{_name_a}{_op}{_name_b}"
        KERNEL_REGISTRY[_combo_name] = functools.partial(_composite_kernel, kernel_name=_combo_name)
        COMPOSITE_KERNELS.append(_combo_name)
del _name_a, _name_b, _op, _combo_name

ALL_KERNELS: List[str] = list(KERNEL_REGISTRY.keys())


def _sample_d_features(cfg: HasDataConfig) -> int:
    """Total feature count d for this call.

    d ~ round(LogNormal(d_features_lognormal_loc, _scale)), at least 2, when both
    keys are set; otherwise cfg.data.d_features.
    """
    loc = getattr(cfg.data, "d_features_lognormal_loc", None)
    scale = getattr(cfg.data, "d_features_lognormal_scale", None)
    if loc is None or scale is None:
        return int(cfg.data.d_features)
    return max(2, round(random.lognormvariate(float(loc), float(scale))))


def _sample_active_dims(d_total: int, cfg: HasDataConfig) -> List[int]:
    """Sorted column indices the kernel uses.

    A fraction ~ Uniform[inactive_frac_min, inactive_frac_max] of the d_total
    columns is left inactive; all columns are active when the keys are absent.
    """
    frac_min = float(getattr(cfg.data, "inactive_frac_min", 0.0))
    frac_max = float(getattr(cfg.data, "inactive_frac_max", 0.0))
    frac = random.uniform(frac_min, frac_max)
    k = d_total - round(frac * d_total)
    k = max(1, min(k, d_total))
    return sorted(random.sample(range(d_total), k))


def _weights_for_pool(pool: List[str], kernel_weights: Optional[Tensor]) -> Optional[List[float]]:
    """Map _COMPOSABLE_KERNELS-ordered weights onto pool, renormalized; None when kernel_weights is None."""
    if kernel_weights is None:
        return None
    idx = [_COMPOSABLE_KERNELS.index(name) for name in pool]
    sub = [float(kernel_weights[i]) for i in idx]
    total = sum(sub)
    if total <= 0:
        return None
    return [w / total for w in sub]


def _tabicl_mix_prob_for_kernel(kernel_name: str, tabicl_mix_weights: Optional[Tensor]) -> float:
    """Probability of using the TabICL PIT z_train instead of the analytic one for this call's kernel.

    Uses the largest weight among the kernel's components in tabicl_mix_weights
    (_COMPOSABLE_KERNELS-ordered). Returns 1.0 when tabicl_mix_weights is None.
    """
    if tabicl_mix_weights is None:
        return 1.0
    members = [n for n in re.split(r"[+*]", kernel_name) if n in _COMPOSABLE_KERNELS]
    if not members:
        return 0.0
    idx = [_COMPOSABLE_KERNELS.index(n) for n in members]
    return float(max(tabicl_mix_weights[i] for i in idx))


def _resolve_kernel_name(cfg: HasDataConfig, kernel_weights: Optional[Tensor] = None) -> str:
    """Pick one task's kernel from cfg.data.kernel or cfg.data.kernels (optionally weighted by kernel_weights)."""
    data = cfg.data
    if hasattr(data, "kernel") and data.kernel:
        name = str(data.kernel)
        if name not in KERNEL_REGISTRY:
            raise ValueError(f"Unknown kernel '{name}'. Choose from {ALL_KERNELS}.")
        return name
    if hasattr(data, "kernels") and data.kernels:
        pool = list(data.kernels)
        for k in pool:
            if k not in KERNEL_REGISTRY:
                raise ValueError(f"Unknown kernel '{k}'. Choose from {ALL_KERNELS}.")
        weights = None
        if all(k in _COMPOSABLE_KERNELS for k in pool):
            weights = _weights_for_pool(pool, kernel_weights)
        return random.choices(pool, weights=weights, k=1)[0] if weights else random.choice(pool)
    return "rbf"


def _sample_kernel_chain_structure(
    cfg: HasDataConfig, kernel_weights: Optional[Tensor] = None
) -> tuple[List[str], List[str], str]:
    """Sample a kernel chain for systematic composition.

    m ~ round(LogNormal(composite_num_kernels_lognormal_loc, _scale)), clipped to
    [composite_num_kernels_min, composite_num_kernels_max]; m kernels drawn with
    replacement from _COMPOSABLE_KERNELS minus composite_exclude_kernels
    (weighted by kernel_weights if given), joined left to right by m-1 random
    +/* ops.

    Returns:
        (names, ops, chain_name), chain_name like "rbf+cosine*periodic".
    """
    exclude = set(getattr(cfg.data, "composite_exclude_kernels", None) or [])
    pool = [k for k in _COMPOSABLE_KERNELS if k not in exclude]
    if not pool:
        raise ValueError(
            f"composite_exclude_kernels={sorted(exclude)} excludes every kernel "
            f"in _COMPOSABLE_KERNELS={_COMPOSABLE_KERNELS}"
        )
    lo = int(getattr(cfg.data, "composite_num_kernels_min", 1))
    hi = int(getattr(cfg.data, "composite_num_kernels_max", 4))
    m_loc = float(getattr(cfg.data, "composite_num_kernels_lognormal_loc", 0.55))
    m_scale = float(getattr(cfg.data, "composite_num_kernels_lognormal_scale", 1.05))
    m = min(max(round(random.lognormvariate(m_loc, m_scale)), lo), hi)
    names = random.choices(pool, weights=_weights_for_pool(pool, kernel_weights), k=m)
    ops = [random.choice(("+", "*")) for _ in range(m - 1)]
    chain_name = names[0] + "".join(f"{op}{name}" for op, name in zip(ops, names[1:]))
    return names, ops, chain_name


def _build_kernel_chain(
    cfg: HasDataConfig,
    names: List[str],
    ops: List[str],
    k: int,
    B: int,
    device: Device,
    active_dims: Optional[List[int]] = None,
    d_total: Optional[int] = None,
) -> tuple[gpytorch.kernels.Kernel, List[Dict[str, Tensor]], Dict[str, Tensor]]:
    """Build a systematic-composition chain for B episodes.

    Each component comes from _build_kernel_component (with its own sign
    modulation); components are combined left to right by ops, then the whole
    chain may get outer sign modulation.

    Returns:
        (kernel, component_params list in names order, outer sign params dict).
    """
    built = [
        _build_kernel_component(cfg, name, k, B, device, active_dims=active_dims, d_total=d_total) for name in names
    ]
    kernel = built[0][0]
    for op, (comp_kernel, _) in zip(ops, built[1:]):
        kernel = _DenseComposedKernel(kernel, op, comp_kernel)
    component_params = [params for _, params in built]

    outer_prob = float(getattr(cfg.data, "sign_modulation_outer_prob", 0.0))
    kernel, outer_params = _maybe_wrap_sign_modulated(
        cfg, kernel, outer_prob, k, B, device, active_dims=active_dims, param_suffix="_outer"
    )

    return kernel, component_params, outer_params


class _MeanFunctionBank(gpytorch.means.Mean):
    """Per-episode GP mean function: linear (or constant), exponential, or sparse anomaly.

    family_onehot selects each episode's family (an all-zero row gives a zero
    mean). Linear uses per-feature weights; exponential and anomaly project x onto
    a unit-norm direction. The exponent is clamped to [-10, 10].
    """

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor,
        exp_direction: Tensor,
        exp_rate: Tensor,
        exp_scale: Tensor,
        anomaly_direction: Tensor,
        anomaly_threshold: Tensor,
        anomaly_magnitude: Tensor,
        family_onehot: Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight)  # (B, d)
        self.register_buffer("bias", bias)  # (B,)
        self.register_buffer("exp_direction", exp_direction)  # (B, d), unit norm
        self.register_buffer("exp_rate", exp_rate)  # (B,)
        self.register_buffer("exp_scale", exp_scale)  # (B,)
        self.register_buffer("anomaly_direction", anomaly_direction)  # (B, d), unit norm
        self.register_buffer("anomaly_threshold", anomaly_threshold)  # (B,)
        self.register_buffer("anomaly_magnitude", anomaly_magnitude)  # (B,)
        self.register_buffer("family_onehot", family_onehot)  # (B, 3): [linear, exponential, anomaly]

    def forward(self, x: Tensor) -> Tensor:  # x: (B, n, d)
        linear_val = (x * self.weight.unsqueeze(1)).sum(-1) + self.bias.unsqueeze(-1)

        exp_proj = (x * self.exp_direction.unsqueeze(1)).sum(-1)
        exponent = torch.clamp(self.exp_rate.unsqueeze(-1) * exp_proj, min=-10.0, max=10.0)
        exp_val = self.exp_scale.unsqueeze(-1) * torch.exp(exponent)

        anomaly_proj = (x * self.anomaly_direction.unsqueeze(1)).sum(-1)
        anomaly_hit = (anomaly_proj > self.anomaly_threshold.unsqueeze(-1)).to(x.dtype)
        anomaly_val = anomaly_hit * self.anomaly_magnitude.unsqueeze(-1)

        stacked = torch.stack([linear_val, exp_val, anomaly_val], dim=-1)  # (B, n, 3)
        return (stacked * self.family_onehot.unsqueeze(1)).sum(-1)


def _sample_mean_module(
    cfg: HasDataConfig, d: int, B: int, device: Device
) -> tuple[gpytorch.means.Mean, Dict[str, Tensor]]:
    """Sample B episodes' GP mean function (cfg.data.mean_fn_*).

    With probability mean_fn_prob an episode gets a non-zero mean whose family
    (linear, exponential, anomaly) ~ Categorical(mean_fn_family_probs); a linear
    mean has a trend with probability mean_fn_linear_prob, else only a bias.
    mean_fn_enabled=False returns gpytorch ZeroMean.

    Returns:
        (mean_module, params) with mean_weight (B, d), mean_bias (B,),
        mean_nonzero (B,), mean_family (B,) in {0 linear, 1 exponential,
        2 anomaly}, mean_linear (B,), mean_exp_direction (B, d), mean_exp_rate,
        mean_exp_scale, mean_anomaly_direction (B, d), mean_anomaly_threshold and
        mean_anomaly_magnitude (B,). Unused families' params are 0.
    """
    batch_shape = torch.Size([B])

    if not bool(getattr(cfg.data, "mean_fn_enabled", False)):
        mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape).to(device)
        params = {
            "mean_weight": torch.zeros(B, d, device=device),
            "mean_bias": torch.zeros(B, device=device),
            "mean_nonzero": torch.zeros(B, dtype=torch.bool, device=device),
            "mean_family": torch.zeros(B, dtype=torch.long, device=device),
            "mean_linear": torch.zeros(B, dtype=torch.bool, device=device),
            "mean_exp_direction": torch.zeros(B, d, device=device),
            "mean_exp_rate": torch.zeros(B, device=device),
            "mean_exp_scale": torch.zeros(B, device=device),
            "mean_anomaly_direction": torch.zeros(B, d, device=device),
            "mean_anomaly_threshold": torch.zeros(B, device=device),
            "mean_anomaly_magnitude": torch.zeros(B, device=device),
        }
        return mean_module, params

    prob_nonzero = float(getattr(cfg.data, "mean_fn_prob", 0.5))
    prob_linear = float(getattr(cfg.data, "mean_fn_linear_prob", 0.5))
    weight_std = float(getattr(cfg.data, "mean_fn_weight_std", 0.5))
    bias_std = float(getattr(cfg.data, "mean_fn_bias_std", 1.0))
    family_probs = list(getattr(cfg.data, "mean_fn_family_probs", [0.5, 0.25, 0.25]))
    exp_rate_std = float(getattr(cfg.data, "mean_fn_exp_rate_std", 0.5))
    exp_scale_std = float(getattr(cfg.data, "mean_fn_exp_scale_std", 1.0))
    anomaly_frac = float(getattr(cfg.data, "mean_fn_anomaly_frac", 0.1))
    anomaly_magnitude_std = float(getattr(cfg.data, "mean_fn_anomaly_magnitude_std", 2.0))

    nonzero_mask = torch.rand(B, device=device) < prob_nonzero

    family_weights = torch.tensor(family_probs, device=device, dtype=torch.float32)
    family_idx = torch.multinomial(family_weights.expand(B, -1), 1, replacement=True).squeeze(-1)  # (B,) in {0,1,2}
    family_onehot = torch.zeros(B, 3, device=device)
    family_onehot.scatter_(1, family_idx.unsqueeze(-1), 1.0)
    family_onehot = family_onehot * nonzero_mask.unsqueeze(-1)  # zero out episodes with no mean at all

    is_linear = nonzero_mask & (family_idx == 0)
    linear_mask = is_linear & (torch.rand(B, device=device) < prob_linear)
    weight = torch.randn(B, d, device=device) * weight_std * linear_mask.unsqueeze(-1)
    bias = torch.randn(B, device=device) * bias_std * is_linear

    exp_direction = torch.randn(B, d, device=device)
    exp_direction = exp_direction / exp_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    exp_rate = torch.randn(B, device=device) * exp_rate_std
    exp_scale = torch.randn(B, device=device) * exp_scale_std

    anomaly_direction = torch.randn(B, d, device=device)
    anomaly_direction = anomaly_direction / anomaly_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # A unit-direction projection of z-normalized features is ~N(0, 1), so this threshold fires on ~anomaly_frac of points.
    anomaly_threshold_value = math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * (1.0 - anomaly_frac) - 1.0))
    anomaly_threshold = anomaly_threshold_value.to(device).expand(B).clone()
    anomaly_magnitude = torch.randn(B, device=device) * anomaly_magnitude_std

    mean_module = _MeanFunctionBank(
        weight=weight,
        bias=bias,
        exp_direction=exp_direction,
        exp_rate=exp_rate,
        exp_scale=exp_scale,
        anomaly_direction=anomaly_direction,
        anomaly_threshold=anomaly_threshold,
        anomaly_magnitude=anomaly_magnitude,
        family_onehot=family_onehot,
    ).to(device)

    params = {
        "mean_weight": weight,
        "mean_bias": bias,
        "mean_nonzero": nonzero_mask,
        "mean_family": family_idx,
        "mean_linear": linear_mask,
        "mean_exp_direction": exp_direction,
        "mean_exp_rate": exp_rate,
        "mean_exp_scale": exp_scale,
        "mean_anomaly_direction": anomaly_direction,
        "mean_anomaly_threshold": anomaly_threshold,
        "mean_anomaly_magnitude": anomaly_magnitude,
    }
    return mean_module, params


def gp_posterior(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    kernel_fn: Callable[[Tensor, Tensor], Tensor],
    noise: float,
    *,
    latent: bool = True,
    return_factors: bool = False,
) -> tuple:
    """Analytical GP posterior at the test points.

    Args:
        latent: posterior over f* (K_ss without noise) instead of noisy y*.
        return_factors: also return (L_ff, alpha).

    Returns:
        mu_star (N,), Sigma_star (N, N), and, if return_factors, L_ff (P, P) and
        alpha = K_ff^{-1} y_train (P,).
    """
    P, N = x_train.shape[0], x_test.shape[0]

    K_ff = kernel_fn(x_train, x_train) + noise * torch.eye(P, device=x_train.device)
    K_sf = kernel_fn(x_test, x_train)  # (N, P)
    K_ss = kernel_fn(x_test, x_test)
    if not latent:
        K_ss = K_ss + noise * torch.eye(N, device=x_test.device)

    L_ff = _safe_cholesky(K_ff, max_attempts=12)
    alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L_ff).squeeze(-1)  # (P,)

    mu_star = K_sf @ alpha  # (N,)

    V = torch.linalg.solve_triangular(L_ff, K_sf.T, upper=False)  # (P, N)
    Sigma_star = K_ss - V.T @ V  # (N, N)
    Sigma_star = 0.5 * (Sigma_star + Sigma_star.T)
    if return_factors:
        return mu_star, Sigma_star, L_ff, alpha
    return mu_star, Sigma_star


def sigma_to_correlation(Sigma: Tensor) -> tuple[Tensor, Tensor]:
    """Convert a covariance matrix to (correlation matrix, marginal std)."""
    sigma = Sigma.diagonal().clamp(min=1e-10).sqrt()  # (N,)
    D_inv = torch.diag(1.0 / sigma)
    R = D_inv @ Sigma @ D_inv
    # Renormalize with the original sigma to remove float32 drift symmetrically.
    d = R.diagonal().clamp(min=1e-10).sqrt()
    R = R / (d.unsqueeze(0) * d.unsqueeze(1))
    return R, sigma


@torch.no_grad()
def generate_gp_task(cfg: HasDataConfig) -> Dict[str, Tensor]:
    """Generate one GP episode: generate_gp_batch(cfg, 1, "cpu", return_kernel_metadata=True)[0].

    Keys: x_norm_train, y_train, x_norm_test, y_test, z_train, z_test,
    log_pdf_test, R_star, Sigma_star, mu_star, sigma_star, n_train, n_test, the
    kernel hyperparameters, kernel_feature_indices, and the unsaved _L_ff/_alpha
    Cholesky factors.
    """
    return generate_gp_batch(cfg, 1, "cpu", return_kernel_metadata=True)[0]


def _gathered_psd_safe_cholesky(K: Tensor, label: str, max_tries: int = 6) -> tuple[Tensor, Tensor]:
    """Batched Cholesky (B, N, N) -> (L, failed).

    One cholesky_ex on the whole batch, then gpytorch's psd_safe_cholesky
    (escalating jitter) only on the episodes that failed. Episodes that are NaN or
    still not PSD at maximum jitter get an identity L and failed=True, and the
    discard rate is logged.
    """
    L, info = torch.linalg.cholesky_ex(K)
    failed = info.ne(0)
    if not failed.any():
        return L, failed
    idx = failed.nonzero(as_tuple=True)[0]
    eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
    try:
        L[idx] = psd_safe_cholesky(K[idx], max_tries=max_tries)
        return L, torch.zeros_like(failed)
    except (NotPSDError, NanError):
        jitter0 = gpytorch.settings.cholesky_jitter.value(K.dtype)
        max_jitter = jitter0 * (10 ** (max_tries - 1))
        K_boosted = K[idx] + max_jitter * eye
        nan_mask = torch.isnan(K_boosted).any(dim=-1).any(dim=-1)
        if nan_mask.any():
            K_boosted = K_boosted.clone()
            K_boosted[nan_mask] = eye
        L_sub, info_sub = torch.linalg.cholesky_ex(K_boosted)
        L[idx] = L_sub
        still_failed = torch.zeros_like(failed)
        still_failed[idx] = info_sub.ne(0) | nan_mask
        warnings.warn(
            f"{label}: {int(still_failed.sum())}/{K.shape[0]} episodes fell back "
            f"to an identity Cholesky factor (unrecoverable even at jitter="
            f"{max_jitter:.1e}, or had NaN entries) and will be discarded.",
            RuntimeWarning,
        )
        L[still_failed] = eye.unsqueeze(0).expand_as(L[still_failed])
        return L, still_failed


def _batched_cholesky(K: Tensor) -> tuple[Tensor, Tensor]:
    """Cholesky of K_ff for the LOO PIT; see _gathered_psd_safe_cholesky."""
    return _gathered_psd_safe_cholesky(K, label="_batched_cholesky (K_ff)")


def _psd_safe_batch(K: Tensor, max_tries: int = 6) -> tuple[Tensor, Tensor]:
    """Cholesky of K_all for sampling; see _gathered_psd_safe_cholesky.

    Episodes with NaN kernel entries are replaced by identity and marked failed.
    """
    nan_failed = torch.isnan(K).any(dim=-1).any(dim=-1)
    if nan_failed.any():
        eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
        K = K.clone()
        K[nan_failed] = eye
        warnings.warn(
            f"_psd_safe_batch: {int(nan_failed.sum())}/{K.shape[0]} episodes had "
            f"NaN entries in K_all (kernel evaluation produced NaN) and will be "
            f"discarded.",
            RuntimeWarning,
        )
    L, failed = _gathered_psd_safe_cholesky(K, label="_psd_safe_batch (K_all)", max_tries=max_tries)
    return L, failed | nan_failed


def tabiclv2_warp_features(x: Tensor, seed: Optional[int] = None) -> Tensor:
    """Warp each feature column with one of 11 random marginal transforms (TabICLv2-style).

    Transforms include heavy tails, power laws, ordinal steps, bimodal mixtures,
    periodicity, Cauchy outliers, zero inflation, bounded ranges and left skew.

    Args:
        x: (B, T, d) or (T, d) standard-normal features.
        seed: if given, reseed all RNGs first.

    Returns:
        Tensor shaped like x, each (episode, column) warped independently.
    """
    if seed is not None:
        _seed_everything(seed)

    added_batch_dim = x.dim() == 2
    if added_batch_dim:
        x = x.unsqueeze(0)

    B, T, d = x.shape
    warped_x = x.clone()
    choices = torch.randint(0, 11, (B, d), device=x.device)

    for b in range(B):
        for col in range(d):
            c = choices[b, col].item()
            col_data = warped_x[b, :, col]

            if c == 0:  # Identity — Standard Normal baseline
                continue
            elif c == 1:  # Signed-square — mild heavy tails
                warped_x[b, :, col] = torch.sign(col_data) * (col_data**2)
            elif c == 2:  # Cube — Student-T-like heavy tails
                warped_x[b, :, col] = col_data**3
            elif c == 3:  # Log-normal / exponential — right-skewed power law
                # Clamp before exp() to avoid float overflow.
                warped_x[b, :, col] = torch.exp(col_data.clamp(min=-5.0, max=4.0))
            elif c == 4:  # Quantization — ordinal / discrete steps
                warped_x[b, :, col] = torch.round(col_data * 2.0) / 2.0
            elif c == 5:  # Bimodal mixture — mixed populations
                mask = torch.rand_like(col_data) > 0.5
                shift = torch.randn(1, device=x.device).item() * 4.0
                col_data[mask] += shift
            elif c == 6:  # Cyclic — seasonal / periodic features
                freq = torch.rand(1, device=x.device).item() * 3.0 + 0.5
                warped_x[b, :, col] = torch.sin(col_data * freq)
            elif c == 7:  # Cauchy — extreme heavy tails, undefined variance
                u = torch.erf(col_data / math.sqrt(2.0))
                # Scale by 0.95 to keep tan() away from its asymptotes.
                warped_x[b, :, col] = torch.tan(u * (math.pi / 2.0 * 0.95))
            elif c == 8:  # Zero-inflation — point mass at 0 mixed with a continuous tail
                spike_frac = float(torch.empty(1).uniform_(0.2, 0.6))
                mask = torch.rand_like(col_data) < spike_frac
                warped_x[b, :, col] = torch.where(mask, torch.zeros_like(col_data), col_data)
            elif c == 9:  # Bounded / sigmoid squash — proportions, percentages, probabilities
                scale = float(torch.empty(1).uniform_(0.5, 3.0))
                warped_x[b, :, col] = torch.sigmoid(col_data * scale)
            elif c == 10:  # Left-skew — mirror of the log-normal/exponential (c == 3) above
                warped_x[b, :, col] = -torch.exp((-col_data).clamp(min=-5.0, max=4.0))

    if added_batch_dim:
        warped_x = warped_x.squeeze(0)
    return warped_x


# Structural feature-warp categories, ported from TempoPFN's offline augmentor
# but applied to the inputs x, so R_star stays exact. Two-level sampling: 2-6
# categories without replacement, then one op per category in this order.
# "seasonality" keeps only amplitude_modulation; "analytic" is {smooth, first
# derivative, second derivative, integral}.
_STRUCTURAL_CATEGORIES: List[str] = [
    "invariances",
    "structure",
    "seasonality",
    "artifacts",
    "analytic",
    "discrete",
]

_CATEGORY_OPS: Dict[str, List[str]] = {
    "invariances": ["yflip", "time_flip"],
    "structure": ["regime_change", "shock_recovery"],
    "seasonality": ["amplitude_modulation"],
    "artifacts": ["resample_artifact"],
    "analytic": ["differential"],
    "discrete": ["quantize", "censor"],
}

# TempoPFN's own default category weights (offline_per_sample_iid_augmentations.py).
_DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = {
    "invariances": 0.6,
    "structure": 0.6,
    "seasonality": 0.5,
    "artifacts": 0.3,
    "analytic": 0.4,
    "discrete": 0.6,
}

# Sub-op weights for categories with more than one op.
_CATEGORY_SUB_OP_WEIGHTS: Dict[str, Dict[str, float]] = {
    "discrete": {"quantize": 0.6, "censor": 0.4},
}


def _structural_warp_column(col_data: Tensor, op: str, use_index_axis: bool = False) -> Tensor:
    """Apply one structural transform to one feature column (T,).

    The pseudo-time axis is the value rank within the column, or the row index
    when use_index_axis is True.
    """
    T = col_data.shape[0]
    device = col_data.device
    std = col_data.std()
    if not torch.isfinite(std) or std <= 0:
        std = torch.ones((), device=device)

    # Ops that don't need either pseudo-time axis at all.
    if op == "yflip":
        return -col_data

    if op == "censor":
        # Elementwise clip between two random quantiles — no ordering needed.
        q_low, q_high = float(torch.rand(1)), float(torch.rand(1))
        q_low, q_high = min(q_low, q_high), max(q_low, q_high)
        sorted_vals = torch.sort(col_data).values
        lo = sorted_vals[int(q_low * (T - 1))]
        hi = sorted_vals[int(q_high * (T - 1))]
        # No-op when the two quantile indices coincide (clamping would flatten the column).
        if not torch.isfinite(hi - lo) or (hi - lo).item() <= 0:
            return col_data
        return col_data.clamp(min=lo.item(), max=hi.item())

    if op == "quantize":
        # Snap to the nearest of n_levels levels: {min, max} plus random interior points.
        lo, hi = col_data.min(), col_data.max()
        if not torch.isfinite(hi - lo) or (hi - lo).item() <= 0:
            return col_data
        n_levels = int(torch.randint(3, 11, (1,)).item())
        n_interior = max(0, n_levels - 2)
        interior = lo + (hi - lo) * torch.rand(n_interior, device=device)
        levels = torch.sort(torch.cat([lo.view(1), hi.view(1), interior])).values
        idx = torch.argmin((col_data.unsqueeze(1) - levels.unsqueeze(0)).abs(), dim=1)
        return levels[idx]

    # Remaining ops use a pseudo-time axis: value rank, or row index if use_index_axis.
    if use_index_axis:
        sort_idx = torch.arange(T, device=device)
    else:
        sort_idx = torch.argsort(col_data)
    sorted_vals = col_data[sort_idx]
    rank = torch.arange(T, device=device, dtype=torch.float32)

    if op == "time_flip":
        # Reverse along the active pseudo-time axis.
        transformed = sorted_vals.flip(dims=[0])

    elif op == "regime_change":
        min_seg = max(4, T // 16)
        valid_hi = T - min_seg
        if valid_hi <= min_seg:
            transformed = sorted_vals
        else:
            num_cp = int(torch.randint(1, 4, (1,)).item())
            valid = torch.arange(min_seg, valid_hi, device=device)
            num_cp = min(num_cp, valid.numel())
            cp = torch.sort(valid[torch.randperm(valid.numel(), device=device)[:num_cp]]).values
            boundaries = torch.cat(
                [
                    torch.zeros(1, device=device, dtype=cp.dtype),
                    cp,
                    torch.full((1,), T, device=device, dtype=cp.dtype),
                ]
            )
            transformed = sorted_vals.clone()
            for i in range(boundaries.numel() - 1):
                s, e = int(boundaries[i]), int(boundaries[i + 1])
                if e <= s:
                    continue
                seg = sorted_vals[s:e]
                scale = float(torch.empty(1).uniform_(0.8, 1.25))
                shift = float(torch.randn(1)) * 0.15 * std.item()
                seg_mean = seg.mean()
                transformed[s:e] = (seg - seg_mean) * scale + seg_mean + shift

    elif op == "shock_recovery":
        t_lo = max(1, T // 16)
        t_hi = max(t_lo + 1, T - T // 16)
        t0 = int(torch.randint(t_lo, t_hi, (1,)).item())
        mag = float(torch.empty(1).uniform_(0.5, 2.0)) * std.item()
        if torch.rand(1).item() < 0.5:
            mag = -mag
        half_life = max(1.0, float(torch.empty(1).uniform_(0.05, 0.3)) * T)
        decay = torch.exp(-(rank - t0).clamp(min=0) / half_life)
        transformed = sorted_vals + mag * decay

    elif op == "amplitude_modulation":
        # Rescale one contiguous window's amplitude around its local mean.
        min_w = max(4, T // 16)
        max_w = max(min_w + 1, T // 2)
        win = int(torch.randint(min_w, max_w + 1, (1,)).item())
        start = int(torch.randint(0, max(1, T - win) + 1, (1,)).item())
        end = min(T, start + win)
        transformed = sorted_vals.clone()
        seg = sorted_vals[start:end]
        if seg.numel() > 0:
            seg_mean = seg.mean()
            amp = float(torch.empty(1).uniform_(0.5, 1.8))
            transformed[start:end] = (seg - seg_mean) * amp + seg_mean

    elif op == "differential":
        # Smooth, 1st derivative, 2nd derivative or cumulative integral of a box-smoothed column, rescaled to the original range.
        k = max(3, T // 32)
        k = k + 1 if k % 2 == 0 else k
        box = torch.ones(k, device=device) / k
        padded = torch.nn.functional.pad(sorted_vals.view(1, 1, -1), (k // 2, k // 2), mode="reflect")
        smoothed = torch.nn.functional.conv1d(padded, box.view(1, 1, -1)).view(-1)

        sub_op = int(torch.randint(0, 4, (1,)).item())
        if sub_op == 0:
            raw = smoothed
        elif sub_op == 1:  # first derivative
            sk = torch.tensor([-1.0, 0.0, 1.0], device=device)
            p = torch.nn.functional.pad(smoothed.view(1, 1, -1), (1, 1), mode="reflect")
            raw = torch.nn.functional.conv1d(p, sk.view(1, 1, -1)).view(-1)
        elif sub_op == 2:  # second derivative
            sk = torch.tensor([1.0, -2.0, 1.0], device=device)
            p = torch.nn.functional.pad(smoothed.view(1, 1, -1), (1, 1), mode="reflect")
            raw = torch.nn.functional.conv1d(p, sk.view(1, 1, -1)).view(-1)
        else:  # cumulative integral, running from the left or right
            if torch.rand(1).item() < 0.5:
                raw = torch.cumsum(smoothed, dim=0)
            else:
                raw = torch.flip(torch.cumsum(torch.flip(smoothed, dims=[0]), dim=0), dims=[0])

        r_min, r_max = raw.min(), raw.max()
        s_min, s_max = sorted_vals.min(), sorted_vals.max()
        # No-op when the transformed column is flat.
        if not torch.isfinite(r_max - r_min) or (r_max - r_min).item() <= 1e-8:
            transformed = sorted_vals
        else:
            denom = r_max - r_min
            transformed = (raw - r_min) / denom * (s_max - s_min) + s_min

    elif op == "resample_artifact":
        # Downsample with a random phase, then upsample (linear, step-hold or linear+smooth).
        max_factor = max(2, min(8, T // 32))
        factor = int(torch.randint(2, max_factor + 1, (1,)).item())
        offset = int(torch.randint(0, factor, (1,)).item())
        ds_idx = torch.arange(offset, T, factor, device=device)
        if ds_idx.numel() < 3:
            transformed = sorted_vals
        else:
            ds_vals_np = sorted_vals[ds_idx].detach().cpu().numpy()
            ds_idx_np = ds_idx.detach().cpu().numpy().astype(np.float64)
            rank_np = rank.detach().cpu().numpy()
            mode_idx = int(torch.multinomial(torch.tensor([0.5, 0.2, 0.3]), 1).item())
            if mode_idx == 0:  # linear
                us_np = np.interp(rank_np, ds_idx_np, ds_vals_np)
            elif mode_idx == 1:  # step-hold: forward-fill from the last downsampled point
                us_np = ds_vals_np[np.searchsorted(ds_idx_np, rank_np, side="right") - 1]
            else:  # linear + light smoothing
                us_np = np.interp(rank_np, ds_idx_np, ds_vals_np)
                sm_k = max(3, T // 128)
                sm_kernel = np.ones(sm_k) / sm_k
                us_np = np.convolve(us_np, sm_kernel, mode="same")
            transformed = torch.from_numpy(us_np).to(device=device, dtype=sorted_vals.dtype)

    else:
        raise ValueError(f"Unknown structural warp op '{op}'")

    warped = torch.empty_like(col_data)
    warped[sort_idx] = transformed
    return warped


def _sample_structural_ops(category_weights: Dict[str, float], num_ops_min: int, num_ops_max: int) -> List[str]:
    """Draw 2..6 categories without replacement (weighted), then one op per category in canonical order."""
    eligible = [c for c in _STRUCTURAL_CATEGORIES if category_weights.get(c, 0.0) > 0.0]
    if not eligible:
        return []
    k = min(int(torch.randint(num_ops_min, num_ops_max + 1, (1,)).item()), len(eligible))
    weights = torch.tensor([category_weights[c] for c in eligible], dtype=torch.float32)
    weights = weights / weights.sum()
    idx = torch.multinomial(weights, k, replacement=False)
    chosen_categories = {eligible[i] for i in idx.tolist()}

    ops: List[str] = []
    for category in _STRUCTURAL_CATEGORIES:  # fixed canonical order, not draw order
        if category not in chosen_categories:
            continue
        candidates = _CATEGORY_OPS[category]
        if len(candidates) == 1:
            ops.append(candidates[0])
            continue
        sub_weights_map = _CATEGORY_SUB_OP_WEIGHTS.get(category)
        if sub_weights_map is None:
            sub_weights = torch.ones(len(candidates))
        else:
            sub_weights = torch.tensor([sub_weights_map[c] for c in candidates], dtype=torch.float32)
        sub_weights = sub_weights / sub_weights.sum()
        pick = int(torch.multinomial(sub_weights, 1).item())
        ops.append(candidates[pick])
    return ops


def _sample_structural_category_mask(
    M: int,
    category_weights: Dict[str, float],
    num_ops_min: int,
    num_ops_max: int,
    device: Device,
) -> Tuple[Tensor, List[str]]:
    """Batched category selection for M draws (Gumbel top-k).

    Each row picks k in [num_ops_min, num_ops_max] categories without replacement
    from the non-zero-weight ones.

    Returns:
        (chosen_mask (M, len(eligible)) bool, eligible category list).
    """
    eligible = [c for c in _STRUCTURAL_CATEGORIES if category_weights.get(c, 0.0) > 0.0]
    n_elig = len(eligible)
    if n_elig == 0:
        return torch.zeros(M, 0, dtype=torch.bool, device=device), eligible

    weights = torch.tensor([category_weights[c] for c in eligible], dtype=torch.float32, device=device)
    log_w = torch.log(weights / weights.sum())
    u = torch.rand(M, n_elig, device=device).clamp_min(1e-12)
    gumbel = -torch.log((-torch.log(u)).clamp_min(1e-12))
    scores = log_w.unsqueeze(0) + gumbel
    # rank[m, i] = position of category i in row m's Gumbel-perturbed order.
    rank = torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1)

    k = torch.randint(num_ops_min, num_ops_max + 1, (M,), device=device)
    k = torch.clamp(k, max=n_elig)
    chosen_mask = rank < k.unsqueeze(1)
    return chosen_mask, eligible


def _structural_warp_batch(col_data: Tensor, op: str, use_index_axis: Tensor) -> Tensor:
    """Batched _structural_warp_column: apply op to each row of col_data (M, T).

    use_index_axis is (M,) bool. "resample_artifact" loops over rows with
    _structural_warp_column.
    """
    M, T = col_data.shape
    device = col_data.device

    if op == "resample_artifact":
        out = torch.empty_like(col_data)
        for i in range(M):
            out[i] = _structural_warp_column(col_data[i], op, use_index_axis=bool(use_index_axis[i]))
        return out

    std = col_data.std(dim=1)
    std = torch.where(torch.isfinite(std) & (std > 0), std, torch.ones_like(std))

    if op == "yflip":
        return -col_data

    if op == "censor":
        q = torch.rand(M, 2, device=device)
        q_low = q.min(dim=1).values
        q_high = q.max(dim=1).values
        sorted_vals, _ = torch.sort(col_data, dim=1)
        lo_idx = (q_low * (T - 1)).long().clamp(0, T - 1)
        hi_idx = (q_high * (T - 1)).long().clamp(0, T - 1)
        lo = torch.gather(sorted_vals, 1, lo_idx.unsqueeze(1)).squeeze(1)
        hi = torch.gather(sorted_vals, 1, hi_idx.unsqueeze(1)).squeeze(1)
        # Coinciding quantile indices: use +-inf bounds so the clamp is a no-op.
        degenerate = ~torch.isfinite(hi - lo) | ((hi - lo) <= 0)
        lo_eff = torch.where(degenerate, torch.full_like(lo, float("-inf")), lo)
        hi_eff = torch.where(degenerate, torch.full_like(hi, float("inf")), hi)
        return torch.clamp(col_data, min=lo_eff.unsqueeze(1), max=hi_eff.unsqueeze(1))

    if op == "quantize":
        lo = col_data.min(dim=1).values
        hi = col_data.max(dim=1).values
        degenerate = ~torch.isfinite(hi - lo) | ((hi - lo) <= 0)
        max_interior = 8  # n_levels in [3,10] -> n_interior in [1,8]
        n_levels = torch.randint(3, 11, (M,), device=device)
        n_interior = (n_levels - 2).clamp(min=0)
        interior_raw = torch.rand(M, max_interior, device=device)
        interior = lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * interior_raw
        slot_idx = torch.arange(max_interior, device=device).unsqueeze(0)
        interior_mask = slot_idx < n_interior.unsqueeze(1)
        # Padding levels are +inf so they never win the nearest-level argmin.
        interior = torch.where(interior_mask, interior, torch.full_like(interior, float("inf")))
        levels = torch.cat([lo.unsqueeze(1), hi.unsqueeze(1), interior], dim=1)
        levels, _ = torch.sort(levels, dim=1)
        diff = (col_data.unsqueeze(2) - levels.unsqueeze(1)).abs()
        idx = diff.argmin(dim=2)
        quantized = torch.gather(levels, 1, idx)
        return torch.where(degenerate.unsqueeze(1), col_data, quantized)

    # Pseudo-time axis per row: value rank, or row index if use_index_axis.
    argsort_idx = torch.argsort(col_data, dim=1)
    index_idx = torch.arange(T, device=device).unsqueeze(0).expand(M, T)
    sort_idx = torch.where(use_index_axis.unsqueeze(1), index_idx, argsort_idx)
    sorted_vals = torch.gather(col_data, 1, sort_idx)
    rank = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).expand(M, T)

    if op == "time_flip":
        transformed = sorted_vals.flip(dims=[1])

    elif op == "regime_change":
        min_seg = max(4, T // 16)
        valid_hi = T - min_seg
        if valid_hi <= min_seg:
            transformed = sorted_vals
        else:
            # Each row draws up to 3 changepoints from the shared candidate range.
            valid = torch.arange(min_seg, valid_hi, device=device)
            n_valid = valid.numel()
            max_cp = 3
            num_cp = torch.randint(1, 4, (M,), device=device).clamp(max=n_valid)
            keys = torch.rand(M, n_valid, device=device)
            take = min(max_cp, n_valid)
            order = torch.argsort(keys, dim=1)[:, :take]  # (M, take)
            if take < max_cp:
                # Fewer candidates than slots: pad with 0 (masked out below).
                pad = torch.zeros(M, max_cp - take, dtype=order.dtype, device=device)
                order = torch.cat([order, pad], dim=1)
            chosen_pos = valid[order]  # (M, max_cp)
            slot_idx = torch.arange(max_cp, device=device).unsqueeze(0)
            valid_mask = slot_idx < num_cp.unsqueeze(1)
            # Unused slots are T, giving empty trailing segments.
            chosen_pos = torch.where(valid_mask, chosen_pos, torch.full_like(chosen_pos, T))
            cp_sorted, _ = torch.sort(chosen_pos, dim=1)
            boundaries = torch.cat(
                [
                    torch.zeros(M, 1, dtype=cp_sorted.dtype, device=device),
                    cp_sorted,
                    torch.full((M, 1), T, dtype=cp_sorted.dtype, device=device),
                ],
                dim=1,
            )
            pos = torch.arange(T, device=device).unsqueeze(0)
            transformed = sorted_vals.clone()
            for i in range(boundaries.shape[1] - 1):
                s = boundaries[:, i].unsqueeze(1)
                e = boundaries[:, i + 1].unsqueeze(1)
                in_seg = (pos >= s) & (pos < e)
                seg_count = in_seg.sum(dim=1).clamp(min=1)
                seg_mean = (sorted_vals * in_seg).sum(dim=1) / seg_count
                scale = torch.empty(M, device=device).uniform_(0.8, 1.25)
                shift = torch.randn(M, device=device) * 0.15 * std
                new_vals = (
                    (sorted_vals - seg_mean.unsqueeze(1)) * scale.unsqueeze(1)
                    + seg_mean.unsqueeze(1)
                    + shift.unsqueeze(1)
                )
                transformed = torch.where(in_seg, new_vals, transformed)

    elif op == "shock_recovery":
        lo_t = max(1, T // 16)
        hi_t = max(lo_t + 1, T - T // 16)
        t0 = torch.randint(lo_t, hi_t, (M,), device=device).float()
        mag_abs = torch.empty(M, device=device).uniform_(0.5, 2.0) * std
        sign = torch.where(torch.rand(M, device=device) < 0.5, -1.0, 1.0)
        mag = mag_abs * sign
        half_life = torch.empty(M, device=device).uniform_(0.05, 0.3) * T
        half_life = half_life.clamp(min=1.0)
        decay = torch.exp(-(rank - t0.unsqueeze(1)).clamp(min=0) / half_life.unsqueeze(1))
        transformed = sorted_vals + mag.unsqueeze(1) * decay

    elif op == "amplitude_modulation":
        min_w = max(4, T // 16)
        max_w = max(min_w + 1, T // 2)
        win = torch.randint(min_w, max_w + 1, (M,), device=device)
        span = torch.clamp(T - win, min=1)  # mirrors max(1, T - win)
        start = (torch.rand(M, device=device) * (span + 1).float()).floor().long().clamp(max=span)
        end = (start + win).clamp(max=T)
        pos = torch.arange(T, device=device).unsqueeze(0)
        in_seg = (pos >= start.unsqueeze(1)) & (pos < end.unsqueeze(1))
        seg_count = in_seg.sum(dim=1).clamp(min=1)
        seg_mean = (sorted_vals * in_seg).sum(dim=1) / seg_count
        amp = torch.empty(M, device=device).uniform_(0.5, 1.8)
        new_vals = (sorted_vals - seg_mean.unsqueeze(1)) * amp.unsqueeze(1) + seg_mean.unsqueeze(1)
        transformed = torch.where(in_seg, new_vals, sorted_vals)

    elif op == "differential":
        # Box average and 3-tap derivatives via slicing/cumsum (same result as conv1d, faster here).
        k = max(3, T // 32)
        k = k + 1 if k % 2 == 0 else k
        pad_k = k // 2
        padded = torch.nn.functional.pad(sorted_vals.unsqueeze(1), (pad_k, pad_k), mode="reflect").squeeze(1)
        csum = torch.nn.functional.pad(padded.cumsum(dim=1), (1, 0))  # csum[:,0] = 0
        smoothed = (csum[:, k:] - csum[:, :-k]) / k  # k-wide windowed mean, (M, T)

        # Compute all 4 sub-ops for the batch, then gather each row's choice.
        p1 = torch.nn.functional.pad(smoothed.unsqueeze(1), (1, 1), mode="reflect").squeeze(1)  # (M, T+2)
        raw_d1 = p1[:, 2:] - p1[:, :-2]
        raw_d2 = p1[:, :-2] - 2 * p1[:, 1:-1] + p1[:, 2:]
        int_fwd = torch.cumsum(smoothed, dim=1)
        int_bwd = torch.flip(torch.cumsum(torch.flip(smoothed, dims=[1]), dim=1), dims=[1])
        int_dir = torch.rand(M, 1, device=device) < 0.5
        raw_int = torch.where(int_dir, int_fwd, int_bwd)

        candidates = torch.stack([smoothed, raw_d1, raw_d2, raw_int], dim=1)  # (M, 4, T)
        sub_op = torch.randint(0, 4, (M,), device=device)
        raw = torch.gather(candidates, 1, sub_op.view(M, 1, 1).expand(-1, 1, T)).squeeze(1)

        r_min = raw.min(dim=1).values
        r_max = raw.max(dim=1).values
        s_min = sorted_vals.min(dim=1).values
        s_max = sorted_vals.max(dim=1).values
        degenerate = ~torch.isfinite(r_max - r_min) | ((r_max - r_min) <= 1e-8)
        denom = torch.where(degenerate, torch.ones_like(r_max), r_max - r_min)
        rescaled = (raw - r_min.unsqueeze(1)) / denom.unsqueeze(1) * (s_max - s_min).unsqueeze(1) + s_min.unsqueeze(1)
        transformed = torch.where(degenerate.unsqueeze(1), sorted_vals, rescaled)

    else:
        raise ValueError(f"Unknown structural warp op '{op}'")

    warped = torch.empty_like(col_data)
    warped.scatter_(1, sort_idx, transformed)
    return warped


def apply_structural_feature_warp(x: Tensor, cfg: HasDataConfig, device: Device) -> Tensor:
    """Apply TempoPFN-style structural transforms to gated feature columns, per episode.

    Each column is gated with structural_warp_prob; gated columns get one op from
    each of structural_warp_num_ops_min..max categories (weighted by
    structural_warp_category_weights), applied in canonical order. Order-dependent
    ops use value rank, or the row index with probability
    structural_warp_index_axis_ratio (chosen once per column).

    Args:
        x: (B, T, d) inputs.
        cfg: config; reads cfg.data.structural_warp_* (disabled by default).
        device: unused.

    Returns:
        (B, T, d) tensor.
    """
    if not bool(getattr(cfg.data, "structural_warp_enabled", False)):
        return x

    prob = float(getattr(cfg.data, "structural_warp_prob", 0.3))
    if prob <= 0.0:
        return x

    category_weights = dict(getattr(cfg.data, "structural_warp_category_weights", _DEFAULT_CATEGORY_WEIGHTS))

    num_ops_max = int(getattr(cfg.data, "structural_warp_num_ops_max", 6))
    num_ops_max = max(1, min(num_ops_max, len(_STRUCTURAL_CATEGORIES)))
    num_ops_min = int(getattr(cfg.data, "structural_warp_num_ops_min", 2))
    num_ops_min = max(1, min(num_ops_min, num_ops_max))

    index_axis_enabled = bool(getattr(cfg.data, "structural_warp_index_axis_enabled", False))
    index_axis_ratio = float(getattr(cfg.data, "structural_warp_index_axis_ratio", 0.0))

    B, T, d = x.shape
    dev = x.device

    # Per-(episode, column) gate in one draw.
    gate = torch.rand(B, d, device=dev) < prob
    gated_idx = gate.reshape(-1).nonzero(as_tuple=True)[0]
    if gated_idx.numel() == 0:
        return x.clone()  # matches the original's unconditional x.clone() up front

    if index_axis_enabled:
        axis_gate = torch.rand(B, d, device=dev) < index_axis_ratio
    else:
        axis_gate = torch.zeros(B, d, dtype=torch.bool, device=dev)

    # (B, T, d) -> (B*d, T); row b*d + col is column (b, col).
    flat = x.permute(0, 2, 1).reshape(B * d, T).clone()
    gated_cols = flat[gated_idx]
    use_index_axis = axis_gate.reshape(-1)[gated_idx]

    M = gated_idx.numel()
    chosen_mask, eligible = _sample_structural_category_mask(M, category_weights, num_ops_min, num_ops_max, dev)

    # Loop over categories, each applied to the gated columns that chose it.
    for category in _STRUCTURAL_CATEGORIES:
        if category not in eligible:
            continue
        cat_col = eligible.index(category)
        cat_mask = chosen_mask[:, cat_col]
        if not bool(cat_mask.any()):
            continue
        sel = cat_mask.nonzero(as_tuple=True)[0]
        candidates = _CATEGORY_OPS[category]
        if len(candidates) == 1:
            op = candidates[0]
            gated_cols[sel] = _structural_warp_batch(gated_cols[sel], op, use_index_axis[sel])
        else:
            sub_weights_map = _CATEGORY_SUB_OP_WEIGHTS.get(category)
            if sub_weights_map is None:
                sub_w = torch.ones(len(candidates))
            else:
                sub_w = torch.tensor([sub_weights_map[c] for c in candidates], dtype=torch.float32)
            sub_w = sub_w / sub_w.sum()
            picks = torch.multinomial(sub_w, sel.numel(), replacement=True)
            for k_op, op in enumerate(candidates):
                op_sel = sel[picks == k_op]
                if op_sel.numel() == 0:
                    continue
                gated_cols[op_sel] = _structural_warp_batch(gated_cols[op_sel], op, use_index_axis[op_sel])

    flat[gated_idx] = gated_cols
    warped_x = flat.reshape(B, d, T).permute(0, 2, 1).contiguous()
    return warped_x


# z_train corruption: optional robustness augmentation toward N(0, 1) noise.
DEFAULT_Z_CORRUPTION_RHO_BETA_A = 2.0
DEFAULT_Z_CORRUPTION_RHO_BETA_B = 3.0


def corrupt_z_train(z_train: Tensor, data_cfg: Any) -> Tensor:
    """Mix z_train with i.i.d. N(0, 1) noise per episode (cfg.data.z_train_corruption_*).

        z = sqrt(rho) * z_train + sqrt(1 - rho) * eps,   rho ~ Beta(a, b)

    applied to a fraction of episodes; a no-op unless z_train_corruption_enabled.

    Args:
        z_train: (B, P).
        data_cfg: cfg.data.

    Returns:
        (B, P) tensor.
    """
    # getattr so plain dataclass configs work too.
    if not bool(getattr(data_cfg, "z_train_corruption_enabled", False)):
        return z_train

    prob = float(getattr(data_cfg, "z_train_corruption_prob", 0.5))
    if prob <= 0.0:
        return z_train

    beta_a = float(getattr(data_cfg, "z_train_corruption_rho_beta_a", DEFAULT_Z_CORRUPTION_RHO_BETA_A))
    beta_b = float(getattr(data_cfg, "z_train_corruption_rho_beta_b", DEFAULT_Z_CORRUPTION_RHO_BETA_B))

    B, P = z_train.shape
    device = z_train.device

    apply_ep = torch.rand(B, device=device) < prob  # (B,) which episodes get corrupted at all
    if not bool(apply_ep.any()):
        return z_train

    rho = torch.distributions.Beta(beta_a, beta_b).sample((B,)).to(device=device, dtype=z_train.dtype)
    noise = torch.randn(B, P, device=device, dtype=z_train.dtype)

    sqrt_rho = rho.clamp(0.0, 1.0).sqrt().unsqueeze(-1)
    sqrt_1m_rho = (1.0 - rho).clamp(0.0, 1.0).sqrt().unsqueeze(-1)
    z_blend = sqrt_rho * z_train + sqrt_1m_rho * noise

    return torch.where(apply_ep.unsqueeze(-1), z_blend, z_train)


def _max_batch_for_context(B: int, T: int, device: Device) -> int:
    """Largest episode batch that fits in free CUDA memory for context length T = P + N (<= B)."""
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return B
    free_bytes, _ = torch.cuda.mem_get_info(device)
    # About 6 (B, T, T) float32 buffers are live at peak; budget half of free memory.
    bytes_per_episode = 6 * T * T * 4
    budget = 0.5 * free_bytes
    return max(1, min(B, int(budget // bytes_per_episode)))


def _evaluate_kernel_dense(kernel_obj: gpytorch.kernels.Kernel, x_norm: Tensor) -> Tensor:
    """Evaluate kernel_obj(x_norm) as a dense (B, T, T) tensor."""
    return kernel_obj(x_norm).to_dense()


@torch.no_grad()
def _generate_gp_batch_raw(
    cfg: HasDataConfig,
    B: int,
    device: Device = "cpu",
    *,
    return_kernel_metadata: bool = False,
    d_override: Optional[int] = None,
    tabicl_model: Optional[TabICLLike] = None,
    tabicl_k_folds: int = 10,
    tabicl_split_calib_frac: float = 0.0,
    kernel_weights: Optional[Tensor] = None,
    tabicl_mix_weights: Optional[Tensor] = None,
    marginal_backend: Optional[str] = None,
    marginal_regressor: Any = None,
    marginal_probs_n: int = 99,
    raw_y_override: bool = False,
) -> List[Dict[str, Tensor]]:
    """Generate up to B GP episodes in one vectorized call.

    All episodes share one kernel family, (P, N) and active_dims, with independent
    hyperparameters, noise and features. Episodes whose Cholesky fails are
    dropped, so fewer than B may be returned; use generate_gp_batch for exactly B.
    If cfg.seed is set all RNGs are reseeded from it.

    Args:
        cfg: config.
        B: number of episodes.
        device: "cpu" or "cuda".
        return_kernel_metadata: also return kernel name, hyperparameters,
            kernel_feature_indices, mlp_mixed and the _L_ff/_alpha factors.
        tabicl_model: replace z_train with this TabICL's K-fold PIT.
        tabicl_k_folds: K for that PIT.
        tabicl_split_calib_frac: > 0 uses a calibration-split PIT with an extra
            round(frac * P) context points instead of K-fold; features are then
            normalized over train + test + calibration points.
        kernel_weights: _COMPOSABLE_KERNELS-ordered sampling weights.
        tabicl_mix_weights: per-family probability that the z_train override
            applies to this call (None = always).
        marginal_backend: a non-TabICL marginal (eval/spatial/marginal_backends)
            whose PIT replaces z_train, z_test and log_pdf_test.
        marginal_regressor: that backend's regressor.
        marginal_probs_n: quantile grid size for marginal_backend.
        raw_y_override: z_train = per-episode z-scored y_train (no PIT).

    Returns:
        list of episode dicts.
    """
    seed = getattr(cfg, "seed", None)
    if seed is not None:
        _seed_everything(seed)

    # d_override pins d across generate_gp_batch's top-up calls.
    d = d_override if d_override is not None else _sample_d_features(cfg)

    # Shared settings for this batch. systematic_composition samples a chain instead of cfg.data.kernel(s).
    systematic = bool(getattr(cfg.data, "systematic_composition", False))
    if systematic:
        chain_names, chain_ops, kernel_name = _sample_kernel_chain_structure(cfg, kernel_weights=kernel_weights)
    else:
        kernel_name = _resolve_kernel_name(cfg, kernel_weights=kernel_weights)
    P = random.randint(cfg.data.P_min, cfg.data.P_max)
    N = random.randint(cfg.data.N_min, cfg.data.N_max)
    # Calibration-only points for tabicl_split_calib_frac > 0, placed after train and
    # test so the train/test sample does not depend on them.
    P_C = max(1, round(tabicl_split_calib_frac * P)) if tabicl_split_calib_frac > 0 else 0
    T = P + N + P_C
    # Cap B so the (B, T, T) buffers fit in free memory.
    B = _max_batch_for_context(B, T, device)

    # active_dims (and k) are shared by all episodes in the call. periodic is capped
    # to k=1 (the period is not identifiable in higher dimensions).
    kernel_cols: Optional[List[int]]
    if _kernel_needs_scalar_input(kernel_name) or "periodic" in kernel_name:
        kernel_cols = [random.randint(0, d - 1)]
    elif kernel_name == "dot_product":
        # dot_product uses every column.
        kernel_cols = None
    else:
        kernel_cols = _sample_active_dims(d, cfg)
    k = d if kernel_cols is None else len(kernel_cols)

    # --- Per-episode hyperparameters + noise (B independent draws in one call) ---
    if systematic:
        kernel_obj, component_params, outer_sign_params = _build_kernel_chain(
            cfg, chain_names, chain_ops, k, B, device, active_dims=kernel_cols, d_total=d
        )
        # Chains fill the flat schema with 0.0 (per-component values are in
        # component_params); outer sign-modulation params stay in the flat schema.
        params = {
            key: torch.zeros(B, device=device)
            for key in (
                "l",
                "alpha2",
                "period",
                "rq_alpha",
                "power",
                "l_b",
                "alpha2_b",
                "period_b",
                "rq_alpha_b",
                "power_b",
            )
        }
        params.update(outer_sign_params)
    else:
        kernel_obj, params = _sample_episode_kernel(cfg, kernel_name, k, B, device, active_dims=kernel_cols, d_total=d)
    likelihood = _build_likelihood(cfg, kernel_name, B, device)
    nugget = likelihood.noise.reshape(B)  # "nugget" name kept for the saved-metadata schema

    # Mean function, one per episode.
    mean_module, mean_params = _sample_mean_module(cfg, d, B, device)

    # --- Features (B, T, d) ~ N(0, 1), warped, normalised per episode ---
    x_raw = torch.randn(B, T, d, device=device)
    x_raw = tabiclv2_warp_features(x_raw)
    x_raw = apply_structural_feature_warp(x_raw, cfg, device)
    if return_kernel_metadata:
        x_raw, mlp_mixed = apply_mlp_feature_mixing(x_raw, cfg, device, return_gate=True)
    else:
        x_raw = apply_mlp_feature_mixing(x_raw, cfg, device)
    x_norm = (x_raw - x_raw.mean(1, keepdim=True)) / x_raw.std(1, keepdim=True).clamp(min=1e-8)

    # The kernel and mean are evaluated on x_kernel, a hidden transform of x_norm
    # (identity unless kernel_hidden_enabled); the model sees x_norm.
    if return_kernel_metadata:
        x_kernel, kernel_hidden_applied = apply_kernel_hidden_warp(x_norm, cfg, device, return_gate=True)
    else:
        x_kernel = apply_kernel_hidden_warp(x_norm, cfg, device)

    # Joint prior covariance (B, T, T): dense kernel + nugget on the diagonal. Only
    # kernel evaluation can raise; factorization happens per episode in
    # _psd_safe_batch.
    with gpytorch.settings.max_cholesky_size(_MAX_CHOLESKY):
        try:
            K_full_dense = _evaluate_kernel_dense(kernel_obj, x_kernel)  # (B, T, T), no nugget yet
        except (NotPSDError, torch.linalg.LinAlgError):
            warnings.warn(
                f"_generate_gp_batch_raw: kernel evaluation for this "
                f"{B}-episode batch (kernel={kernel_name!r}) raised NotPSDError "
                f"or LinAlgError; discarding the whole batch and resampling.",
                RuntimeWarning,
            )
            return []
    nugget_eye = torch.eye(T, device=device, dtype=K_full_dense.dtype).expand(B, T, T)
    K_all_raw = K_full_dense + likelihood.noise.reshape(B, 1, 1) * nugget_eye

    # K_all = L_all L_all^T from a PSD-repaired Cholesky, so the sample y_all and the
    # reported covariances come from the same PSD matrix.
    L_all, failed_all = _psd_safe_batch(K_all_raw)
    K_all = L_all @ L_all.mT  # (B, T, T), PSD by construction
    y_all = (L_all @ torch.randn(B, T, 1, device=device)).squeeze(-1)  # zero-mean GP sample
    # Add the mean function (evaluated on x_kernel).
    y_all = y_all + mean_module(x_kernel)

    x_norm_train = x_norm[:, :P]  # (B, P, d) -- model-visible, saved/returned below
    x_norm_test = x_norm[:, P : P + N]  # (B, N, d)
    x_norm_calib = x_norm[:, P + N :]  # (B, P_C, d) -- tabicl_split PIT context only
    x_kernel_train = x_kernel[:, :P]  # (B, P, d) -- oracle-only, never saved/returned
    x_kernel_test = x_kernel[:, P : P + N]  # (B, N, d)
    y_train = y_all[:, :P]  # (B, P)
    y_test = y_all[:, P : P + N]  # (B, N)
    y_calib = y_all[:, P + N :]  # (B, P_C)

    # --- Sub-matrices of K_all (nugget already on diagonal) ---
    K_ff = K_all[:, :P, :P]  # (B, P, P) -- P_C never enters K_ff/LOO/oracle
    K_ss = K_all[:, P : P + N, P : P + N]  # (B, N, N)

    # LOO PIT needs L_ff and alpha = K_ff^{-1} (y_train - mean_train).
    L_ff, failed_ff = _batched_cholesky(K_ff)
    mean_train = mean_module(x_kernel_train)  # (B, P)
    alpha = torch.cholesky_solve((y_train - mean_train).unsqueeze(-1), L_ff).squeeze(-1)  # (B, P)

    # Episodes whose Cholesky failed are dropped at the end.
    discard = failed_all | failed_ff

    oracle_mode = getattr(cfg.data, "oracle_mode", "prior")
    if oracle_mode == "prior":
        # Prior oracle: R_star is the prior correlation of the test block of K_all.
        # The copula target is the posterior correlation R_post (see z_test below).
        mu_star = mean_module(x_kernel_test)
        Sigma_star = K_ss
    else:
        raise ValueError(f"Unknown data.oracle_mode '{oracle_mode}'; only 'prior' is supported.")
    Sigma_star = 0.5 * (Sigma_star + Sigma_star.permute(0, 2, 1))

    # sigma_to_correlation (batched)
    var_diag = Sigma_star.diagonal(dim1=1, dim2=2).clamp(min=1e-10)  # (B, N)
    sigma_star = var_diag.sqrt()
    inv_s = var_diag.rsqrt()
    R_star = Sigma_star * inv_s.unsqueeze(1) * inv_s.unsqueeze(2)  # (B, N, N)
    d_diag = R_star.diagonal(dim1=1, dim2=2).clamp(min=1e-10).sqrt()
    R_star = R_star / (d_diag.unsqueeze(1) * d_diag.unsqueeze(2))

    # Prior correlation among the test points (same as R_star; kept for the schema).
    prior_var = K_ss.diagonal(dim1=1, dim2=2).clamp(min=1e-10)  # (B, N)
    prior_inv = prior_var.rsqrt()
    R_prior = K_ss * prior_inv.unsqueeze(1) * prior_inv.unsqueeze(2)  # (B, N, N)
    pd_diag = R_prior.diagonal(dim1=1, dim2=2).clamp(min=1e-10).sqrt()
    R_prior = R_prior / (pd_diag.unsqueeze(1) * pd_diag.unsqueeze(2))

    # LOO PIT for z_train; diag(K_ff^{-1}) is the column-wise squared norm of L_ff^{-1}.
    eye_P = torch.eye(P, device=device)
    L_inv = torch.linalg.solve_triangular(L_ff, eye_P.unsqueeze(0).expand(B, -1, -1), upper=False)  # (B, P, P)
    K_inv_diag = (L_inv**2).sum(dim=1).clamp(min=1e-12)  # (B, P)
    z_train = alpha * K_inv_diag.rsqrt()  # (B, P)

    # Posterior PIT for z_test: standardize by the GP posterior marginals
    # N(mu_post_i, Sigma_post_ii), matching what a TabICL marginal conditioned on
    # the context gives. mu_star/sigma_star stay prior quantities. Only diag of the
    # Schur complement is used (each entry >= nugget).
    K_sf = K_all[:, P : P + N, :P]  # (B, N, P)
    V_sf = torch.linalg.solve_triangular(L_ff, K_sf.mT, upper=False)  # (B, P, N)
    mu_post = mu_star + torch.bmm(K_sf, alpha.unsqueeze(-1)).squeeze(-1)  # (B, N)
    var_post = K_ss.diagonal(dim1=1, dim2=2) - (V_sf**2).sum(dim=1)  # (B, N)
    var_post = var_post.clamp(min=likelihood.noise.reshape(B, 1).clamp(min=1e-10))
    sig_c = var_post.sqrt()
    z_test = (y_test - mu_post) / sig_c  # (B, N)
    log_pdf_test = -0.5 * math.log(2.0 * math.pi) - sig_c.log() - 0.5 * z_test**2  # (B, N)

    # Discard episodes whose z_train is non-finite or has degenerate spread.
    non_finite = ~torch.isfinite(z_train).all(dim=1)
    z_std = z_train.std(dim=1)
    degen = non_finite | (z_std < 0.1) | (z_std > 3.0)
    if degen.any():
        warnings.warn(
            f"generate_gp_batch: {int(degen.sum())}/{B} episodes have degenerate LOO z "
            f"({int(non_finite.sum())} non-finite) and will be discarded.",
            RuntimeWarning,
        )
    discard = discard | degen

    # Discard episodes whose every active kernel dimension collapsed to a constant
    # (R_star would then be constant).
    active_cols = kernel_cols if kernel_cols is not None else list(range(d))
    active_stds = x_norm[:, :, active_cols].std(dim=1)  # (B, len(active_cols))
    degenerate_active_col = (active_stds.max(dim=1).values) < 1e-4
    if degenerate_active_col.any():
        warnings.warn(
            f"generate_gp_batch: {int(degenerate_active_col.sum())}/{B} episodes have a "
            f"degenerate (near-constant) active kernel column and will be discarded.",
            RuntimeWarning,
        )
    discard = discard | degenerate_active_col

    # Reconstruct full posterior covariance (for Y-space oracle)
    Sigma_full = R_star * sigma_star.unsqueeze(1) * sigma_star.unsqueeze(2)  # (B, N, N)

    # z_train override from a marginal model (z_train_source tabicl / tabicl_split /
    # backends), after the degenerate-episode checks. Targets are z-scored per
    # episode first. With tabicl_mix_weights the override applies with the
    # kernel's mix probability.
    if tabicl_mix_weights is not None:
        apply_tabicl = tabicl_model is not None and (
            random.random() < _tabicl_mix_prob_for_kernel(kernel_name, tabicl_mix_weights)
        )
    else:
        apply_tabicl = tabicl_model is not None

    if raw_y_override:
        # "y_train": z_train is the z-scored target; z_test/log_pdf_test stay analytic.
        y_mean = y_train.mean(dim=1, keepdim=True)
        y_std = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
        z_train = ((y_train - y_mean) / y_std).detach()
    elif marginal_backend in _BATCHED_MARGINAL_BACKENDS:
        # Batched PIT for a non-TabICL backend: (k_folds + 1) fused forwards for the call.
        y_mean = y_train.mean(dim=1, keepdim=True)
        y_std = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
        y_train_s = ((y_train - y_mean) / y_std).detach().cpu().numpy()
        y_test_s = ((y_test - y_mean) / y_std).detach().cpu().numpy()
        x_train_np = x_norm_train.detach().cpu().numpy()
        x_test_np = x_norm_test.detach().cpu().numpy()
        base_seed = int(getattr(cfg, "seed", None) or 0)
        _run_batched = _BATCHED_MARGINAL_BACKENDS[marginal_backend]()
        out = _run_batched(
            marginal_regressor,
            x_train_np,
            y_train_s,
            x_test_np,
            y_test_s,
            k_folds=tabicl_k_folds,
            probs_n=marginal_probs_n,
            seed=base_seed,
        )
        z_train = torch.from_numpy(out["z_train"]).to(device=device)
        z_test = torch.from_numpy(out["z_test"]).to(device=device)
        # Jacobian back to raw-y nats.
        log_pdf_test = torch.from_numpy(out["log_pdf_test"]).to(device=device) - y_std.log()  # (B,N) - (B,1) broadcast
    elif marginal_backend not in (None, "tabicl"):
        # Per-episode K-fold PIT for backends without a batched module (slow).
        assert marginal_backend is not None
        from eval.metrics.joint_nll import compute_pit
        from eval.spatial.marginal_backends import loo_pit as _backend_loo_pit
        from eval.spatial.marginal_backends import quantiles as _backend_quantiles

        y_mean = y_train.mean(dim=1, keepdim=True)
        y_std = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
        probs = np.linspace(1.0 / (marginal_probs_n + 1), marginal_probs_n / (marginal_probs_n + 1), marginal_probs_n)
        base_seed = int(getattr(cfg, "seed", None) or 0)
        z_train_np = np.empty((B, P), dtype=np.float32)
        z_test_np = np.empty((B, N), dtype=np.float32)
        log_pdf_np = np.empty((B, N), dtype=np.float32)
        for b in range(B):
            xc = x_norm_train[b].detach().cpu().numpy()
            xq = x_norm_test[b].detach().cpu().numpy()
            y_std_b = float(y_std[b])
            yc = ((y_train[b] - y_mean[b]) / y_std[b]).detach().cpu().numpy()
            yq = ((y_test[b] - y_mean[b]) / y_std[b]).detach().cpu().numpy()
            seed_b = (base_seed + b) % (2**31)
            z_train_np[b] = _backend_loo_pit(
                marginal_backend,
                marginal_regressor,
                xc,
                yc,
                probs,
                k_folds=tabicl_k_folds,
                seed=seed_b,
            )
            q_test = _backend_quantiles(marginal_backend, marginal_regressor, xc, yc, xq, probs, seed=seed_b)
            z_test_b, log_pdf_b = compute_pit(q_test, probs, yq)
            z_test_np[b] = z_test_b
            # Jacobian back to raw-y nats.
            log_pdf_np[b] = log_pdf_b - math.log(y_std_b)
        z_train = torch.from_numpy(z_train_np).to(device=device)
        z_test = torch.from_numpy(z_test_np).to(device=device)
        log_pdf_test = torch.from_numpy(log_pdf_np).to(device=device)
    elif apply_tabicl and tabicl_split_calib_frac > 0:
        # "tabicl_split": one forward with the calibration points as context.
        from copula_inter.pit import run_pit_calib_split_batched  # local: pit.py imports from this module

        assert tabicl_model is not None

        y_mean = y_train.mean(dim=1, keepdim=True)
        y_std = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
        y_train_scaled = ((y_train - y_mean) / y_std).unsqueeze(-1)  # (B, P, 1)
        y_calib_scaled = ((y_calib - y_mean) / y_std).unsqueeze(-1)  # (B, P_C, 1)
        split_pit = run_pit_calib_split_batched(
            tabicl_model,
            x_norm_train,
            y_train_scaled,
            x_norm_calib,
            y_calib_scaled,
            Y_query_raw=y_train.unsqueeze(-1),
            Y_calib_raw=y_calib.unsqueeze(-1),
        )
        z_train = split_pit["z_train"].squeeze(-1)  # (B, P)
    elif apply_tabicl:
        # "tabicl": K-fold PIT on the training points and a full-context PIT on the
        # real test points; z_train, z_test and log_pdf_test all come from TabICL.
        from copula_inter.pit import run_pit_batched  # local: pit.py imports from this module

        assert tabicl_model is not None

        y_mean = y_train.mean(dim=1, keepdim=True)
        y_std = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
        y_train_scaled = ((y_train - y_mean) / y_std).unsqueeze(-1)  # (B, P, 1)
        y_test_scaled = ((y_test - y_mean) / y_std).unsqueeze(-1)  # (B, N, 1)
        tabicl_pit = run_pit_batched(
            tabicl_model,
            x_norm_train,
            y_train_scaled,
            x_norm_test,
            y_test_scaled,
            k_folds=tabicl_k_folds,
            Y_train_raw=y_train.unsqueeze(-1),
        )
        z_train = tabicl_pit["z_train"].squeeze(-1)  # (B, P)
        z_test = tabicl_pit["z_test"].squeeze(-1)  # (B, N)
        # Jacobian back to raw-y nats: log p_raw = log p_scaled - log(std).
        log_pdf_test = tabicl_pit["log_pdf_test"].squeeze(-1) - y_std.log()  # (B, N) - (B, 1) broadcast

    # Optional z_train corruption, skipped when this call used the adaptive TabICL mix.
    if not (tabicl_mix_weights is not None and apply_tabicl):
        z_train = corrupt_z_train(z_train, cfg.data)

    # --- Pack into list of dicts (single D→H transfer) ---
    tensors = {
        "x_norm_train": x_norm_train.cpu(),
        "x_norm_test": x_norm_test.cpu(),
        "y_train": y_train.cpu(),
        "y_test": y_test.cpu(),
        "z_train": z_train.cpu(),
        "z_test": z_test.cpu(),
        "log_pdf_test": log_pdf_test.cpu(),
        "R_star": R_star.cpu(),
        "R_prior": R_prior.cpu(),
        "Sigma_star": Sigma_full.cpu(),
        "mu_star": mu_star.cpu(),
        "sigma_star": sigma_star.cpu(),
    }

    # Discard any episode with a non-finite saved field.
    non_finite = torch.zeros(B, dtype=torch.bool)
    for _t in tensors.values():
        non_finite = non_finite | ~_t.reshape(_t.shape[0], -1).isfinite().all(dim=1)
    if non_finite.any():
        warnings.warn(
            f"generate_gp_batch: {int(non_finite.sum())}/{B} episodes contain "
            f"NaN/Inf in a saved field and will be discarded.",
            RuntimeWarning,
        )
    discard = discard | non_finite.to(discard.device)

    n_tr = torch.tensor(P)
    n_te = torch.tensor(N)
    extra: Dict[str, object] = {"n_train": n_tr, "n_test": n_te}

    if return_kernel_metadata:
        # Per-episode hyperparameters and factors plus the call-shared kernel name and
        # active_dims. Chains carry per-component sign fields in kernel_component_params.
        flat_keys = [
            "l",
            "alpha2",
            "period",
            "rq_alpha",
            "power",
            "l_b",
            "alpha2_b",
            "period_b",
            "rq_alpha_b",
            "power_b",
            "sign_applied_outer",
            "sign_w_outer",
            "sign_b_outer",
            "sign_a_outer",
        ]
        if not systematic:
            flat_keys += ["sign_applied", "sign_w", "sign_b", "sign_a"]
            if _parse_composite(kernel_name) is not None:
                flat_keys += ["sign_applied_b", "sign_w_b", "sign_b_b", "sign_a_b"]
        for key in flat_keys:
            tensors[key] = params[key].cpu()
        tensors["nugget"] = nugget.cpu()
        tensors["mlp_mixed"] = mlp_mixed.cpu()
        tensors["kernel_hidden_applied"] = kernel_hidden_applied.cpu()
        tensors["x_kernel_train"] = x_kernel_train.cpu()
        tensors["x_kernel_test"] = x_kernel_test.cpu()
        for key in (
            "mean_weight",
            "mean_bias",
            "mean_nonzero",
            "mean_family",
            "mean_linear",
            "mean_exp_direction",
            "mean_exp_rate",
            "mean_exp_scale",
            "mean_anomaly_direction",
            "mean_anomaly_threshold",
            "mean_anomaly_magnitude",
        ):
            tensors[key] = mean_params[key].cpu()
        tensors["_L_ff"] = L_ff
        tensors["_alpha"] = alpha
        extra["kernel"] = kernel_name
        extra["kernel_feature_indices"] = torch.tensor(
            kernel_cols if kernel_cols is not None else list(range(d)), dtype=torch.long
        )

    # Drop discarded episodes from the per-episode tensors (the extra fields are call-shared).
    return assemble_episodes(
        tensors,
        extra,
        discard,
        (chain_names, chain_ops, component_params) if return_kernel_metadata and systematic else None,
    )


def generate_gp_batch(
    cfg: HasDataConfig,
    B: int,
    device: Device = "cpu",
    *,
    return_kernel_metadata: bool = False,
    tabicl_model: Optional[TabICLLike] = None,
    tabicl_k_folds: int = 10,
    tabicl_split_calib_frac: float = 0.0,
    d_override: Optional[int] = None,
    kernel_weights: Optional[Tensor] = None,
    tabicl_mix_weights: Optional[Tensor] = None,
    marginal_backend: Optional[str] = None,
    marginal_regressor: Any = None,
    marginal_probs_n: int = 99,
    raw_y_override: bool = False,
) -> List[Dict[str, Tensor]]:
    """Generate exactly B GP episodes, topping up discarded ones with further calls.

    Top-up calls resample kernel, P, N and active_dims but keep d (pinned to
    d_override if given, else to the first call's d). Raises after max_rounds.
    """
    base_seed = getattr(cfg, "seed", None)
    episodes = _generate_gp_batch_raw(
        cfg,
        B,
        device,
        return_kernel_metadata=return_kernel_metadata,
        d_override=d_override,
        tabicl_model=tabicl_model,
        tabicl_k_folds=tabicl_k_folds,
        tabicl_split_calib_frac=tabicl_split_calib_frac,
        kernel_weights=kernel_weights,
        tabicl_mix_weights=tabicl_mix_weights,
        marginal_backend=marginal_backend,
        marginal_regressor=marginal_regressor,
        marginal_probs_n=marginal_probs_n,
        raw_y_override=raw_y_override,
    )
    # Pin every top-up round to the first round's d (or d_override).
    d_fixed: Optional[int]
    if episodes:
        d_fixed = int(episodes[0]["x_norm_train"].shape[-1])
    else:
        d_fixed = d_override
    max_rounds = 20
    for round_idx in range(1, max_rounds + 1):
        if len(episodes) >= B:
            break
        shortfall = B - len(episodes)
        if base_seed is not None:
            # Offset the seed per round so retries draw new kernels.
            setattr(cfg, "seed", base_seed + round_idx * 104_729)
        new_episodes = _generate_gp_batch_raw(
            cfg,
            shortfall,
            device,
            return_kernel_metadata=return_kernel_metadata,
            d_override=d_fixed,
            tabicl_model=tabicl_model,
            tabicl_k_folds=tabicl_k_folds,
            tabicl_split_calib_frac=tabicl_split_calib_frac,
            kernel_weights=kernel_weights,
            tabicl_mix_weights=tabicl_mix_weights,
            marginal_backend=marginal_backend,
            marginal_regressor=marginal_regressor,
            marginal_probs_n=marginal_probs_n,
            raw_y_override=raw_y_override,
        )
        if d_fixed is None and new_episodes:
            d_fixed = int(new_episodes[0]["x_norm_train"].shape[-1])
        episodes += new_episodes
    if base_seed is not None:
        setattr(cfg, "seed", base_seed)
    if len(episodes) < B:
        raise RuntimeError(
            f"generate_gp_batch: could not assemble {B} valid episodes after "
            f"{max_rounds} top-up rounds ({len(episodes)} obtained) — the kernel/config "
            f"combination in this call appears to be persistently non-PSD."
        )
    return episodes[:B]
