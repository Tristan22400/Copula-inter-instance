"""Per-call sampling of the GP prior.

Feature count, active dimensions, kernel name or composition chain, and the
mean function.

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

import math
import random
import re
from typing import Dict, List, Optional

import gpytorch
import torch
from torch import Tensor

from copula_inter.gp_kernels import (
    _COMPOSABLE_KERNELS,
    ALL_KERNELS,
    KERNEL_REGISTRY,
    _build_kernel_component,
    _DenseComposedKernel,
    _maybe_wrap_sign_modulated,
)
from copula_inter.type_aliases import Device, HasDataConfig


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
    return max(float(tabicl_mix_weights[i]) for i in idx)


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
