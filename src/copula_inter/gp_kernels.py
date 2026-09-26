"""GP kernels for episode generation.

Hyperparameter priors, batched and concrete gpytorch kernels, sign modulation,
and the kernel registry (base kernels plus "A+B" / "A*B" composites).

Kernels (gpytorch, hyperparameters from LogNormal/Gamma priors, see
_kernel_prior_spec and _nugget_prior):
    rbf, matern12/32/52, cosine, periodic, rational_quadratic,
    dot_product (LinearKernel, variance = alpha2, no lengthscale),
    polynomial (alpha2 * (x1.x2 + c)^d; c stored in "l", d in "power",
        one d per generate_gp_batch call).
"""

from __future__ import annotations

import functools
import itertools
import math
import random
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import gpytorch
import torch
from gpytorch.priors import GammaPrior, LogNormalPrior, Prior
from torch import Tensor

from copula_inter.type_aliases import Device, HasDataConfig

if TYPE_CHECKING:
    pass


def _sq_dist(X1: Tensor, X2: Tensor) -> Tensor:
    """Squared Euclidean distance matrix (n1, n2)."""
    diff = X1.unsqueeze(1) - X2.unsqueeze(0)  # (n1, n2, d)
    return (diff**2).sum(-1)


def _dist(X1: Tensor, X2: Tensor) -> Tensor:
    """Euclidean distance matrix (n1, n2)."""
    return _sq_dist(X1, X2).clamp(min=0.0).sqrt()


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


def _register_composite_kernels() -> List[str]:
    """Add every "A+B" / "A*B" pair of _COMPOSABLE_KERNELS to KERNEL_REGISTRY; return their names."""
    names = []
    for name_a, name_b in itertools.combinations(_COMPOSABLE_KERNELS, 2):
        for op in ("+", "*"):
            combo = f"{name_a}{op}{name_b}"
            KERNEL_REGISTRY[combo] = functools.partial(_composite_kernel, kernel_name=combo)
            names.append(combo)
    return names


COMPOSITE_KERNELS: List[str] = _register_composite_kernels()


ALL_KERNELS: List[str] = list(KERNEL_REGISTRY.keys())
