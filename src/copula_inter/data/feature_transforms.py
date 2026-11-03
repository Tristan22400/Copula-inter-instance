"""Random feature maps used before GP kernel evaluation."""

from __future__ import annotations

import math
import random
from typing import List, Literal, Optional, overload

import torch
from torch import Tensor

from copula_inter.rng import seed_everything
from copula_inter.type_aliases import Device, HasDataConfig

# Activations for MLP feature mixing.
_MLP_MIX_ACTIVATIONS: List[str] = ["linear", "relu", "sigmoid", "sin", "mod", "leaky_relu"]


def _apply_mlp_activation(x: Tensor, name: str) -> Tensor:
    """Elementwise nonlinearity for one MLP-mixing layer. `x` is any shape."""
    if name == "linear":
        return x
    if name == "relu":
        return torch.relu(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "sin":
        return torch.sin(x)
    if name == "mod":
        # Remainder with a fixed 2*pi period.
        return torch.remainder(x, 2 * math.pi)
    if name == "leaky_relu":
        return torch.nn.functional.leaky_relu(x, negative_slope=0.1)
    raise ValueError(f"Unknown MLP-mixing activation '{name}'")


@overload
def apply_mlp_feature_mixing(
    x: Tensor, cfg: HasDataConfig, device: Device, *, return_gate: Literal[False] = ...
) -> Tensor: ...
@overload
def apply_mlp_feature_mixing(
    x: Tensor, cfg: HasDataConfig, device: Device, *, return_gate: Literal[True]
) -> tuple[Tensor, Tensor]: ...
def apply_mlp_feature_mixing(
    x: Tensor, cfg: HasDataConfig, device: Device, *, return_gate: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    """Mix the input feature columns of gated episodes through a small random MLP.

    The number of layers and their activations are shared by the call; weights
    and biases are drawn per episode. Episodes are gated with mlp_mixing_prob.

    Args:
        x: (B, T, d) features, before per-episode normalization.
        cfg: config; reads cfg.data.mlp_mixing_* (disabled by default).
        device: device for the sampled weights.
        return_gate: also return the (B,) bool gate.

    Returns:
        (B, T, d) tensor, or (x, gate) if return_gate.
    """
    if not bool(getattr(cfg.data, "mlp_mixing_enabled", False)):
        if return_gate:
            return x, torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        return x

    mixing_prob = float(getattr(cfg.data, "mlp_mixing_prob", 0.3))
    if mixing_prob <= 0.0:
        if return_gate:
            return x, torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        return x

    L_min = int(getattr(cfg.data, "mlp_num_layers_min", 1))
    L_max = int(getattr(cfg.data, "mlp_num_layers_max", 2))
    w_std = float(getattr(cfg.data, "mlp_mix_weight_std", 1.0))

    B, T, d = x.shape
    L = random.randint(L_min, L_max)
    # Activation sequence shared by the whole call; weights per episode.
    activations = [random.choice(_MLP_MIX_ACTIVATIONS) for _ in range(L)]

    x_mixed = x
    for act_name in activations:
        # 1/sqrt(d) fan-in scaling.
        W_l = torch.randn(B, d, d, device=device) * (w_std / math.sqrt(d))
        b_l = torch.randn(B, 1, d, device=device) * w_std
        x_mixed = torch.einsum("btd,bde->bte", x_mixed, W_l) + b_l
        x_mixed = _apply_mlp_activation(x_mixed, act_name)

    gate_1d = torch.rand(B, device=device) < mixing_prob  # (B,)
    gate = gate_1d[:, None, None]  # (B,1,1)
    # (B, 1, 1) gate shape broadcasts against (B, T, d).
    x_out = torch.where(gate, x_mixed, x)
    if return_gate:
        return x_out, gate_1d
    return x_out


@overload
def apply_kernel_hidden_warp(
    x_norm: Tensor, cfg: HasDataConfig, device: Device, *, return_gate: Literal[False] = ...
) -> Tensor: ...
@overload
def apply_kernel_hidden_warp(
    x_norm: Tensor, cfg: HasDataConfig, device: Device, *, return_gate: Literal[True]
) -> tuple[Tensor, Tensor]: ...
def apply_kernel_hidden_warp(
    x_norm: Tensor, cfg: HasDataConfig, device: Device, *, return_gate: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    """Warp normalized inputs through a hidden bottleneck MLP seen only by the kernel and mean.

    Down-projects to width r < d, applies random nonlinear layers, projects back
    to d and z-normalizes per episode. The model's x_norm is unchanged. Episodes
    are gated with kernel_hidden_prob.

    Args:
        x_norm: (B, T, d) normalized model inputs.
        cfg: config; reads cfg.data.kernel_hidden_* (disabled by default).
        device: device for the sampled weights.
        return_gate: also return the (B,) bool gate.

    Returns:
        (B, T, d) x_kernel (x_norm for ungated episodes), or (x_kernel, gate).
    """
    if not bool(getattr(cfg.data, "kernel_hidden_enabled", False)):
        if return_gate:
            return x_norm, torch.zeros(x_norm.shape[0], dtype=torch.bool, device=x_norm.device)
        return x_norm

    hidden_prob = float(getattr(cfg.data, "kernel_hidden_prob", 1.0))
    if hidden_prob <= 0.0:
        if return_gate:
            return x_norm, torch.zeros(x_norm.shape[0], dtype=torch.bool, device=x_norm.device)
        return x_norm

    H_min = int(getattr(cfg.data, "kernel_hidden_layers_min", 1))
    H_max = int(getattr(cfg.data, "kernel_hidden_layers_max", 2))
    w_std = float(getattr(cfg.data, "kernel_hidden_weight_std", 1.0))
    bottleneck_frac = float(getattr(cfg.data, "kernel_hidden_bottleneck_frac", 0.5))

    B, T, d = x_norm.shape
    # Clamp r to [1, d-1].
    r = max(1, min(d - 1, round(bottleneck_frac * d)))

    W_down = torch.randn(B, d, r, device=device) * (w_std / math.sqrt(d))
    b_down = torch.randn(B, 1, r, device=device) * w_std
    h = torch.einsum("btd,bdr->btr", x_norm, W_down) + b_down

    H = random.randint(H_min, H_max)
    # Activation sequence per call, weights per episode.
    activations = [random.choice(_MLP_MIX_ACTIVATIONS) for _ in range(H)]
    for act_name in activations:
        W_l = torch.randn(B, r, r, device=device) * (w_std / math.sqrt(r))
        b_l = torch.randn(B, 1, r, device=device) * w_std
        h = torch.einsum("btr,brs->bts", h, W_l) + b_l
        h = _apply_mlp_activation(h, act_name)

    W_up = torch.randn(B, r, d, device=device) * (w_std / math.sqrt(r))
    b_up = torch.randn(B, 1, d, device=device) * w_std
    x_hidden = torch.einsum("btr,brd->btd", h, W_up) + b_up
    x_hidden = (x_hidden - x_hidden.mean(1, keepdim=True)) / x_hidden.std(1, keepdim=True).clamp(min=1e-8)

    gate_1d = torch.rand(B, device=device) < hidden_prob  # (B,)
    gate = gate_1d[:, None, None]  # (B,1,1), see apply_mlp_feature_mixing
    x_kernel = torch.where(gate, x_hidden, x_norm)
    if return_gate:
        return x_kernel, gate_1d
    return x_kernel


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
        seed_everything(seed)

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
