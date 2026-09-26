"""Random feature maps used before GP kernel evaluation."""

from __future__ import annotations

import math
import random
from typing import List

import torch
from torch import Tensor

# Activation bank for MLP feature mixing (adapted from CauKer's SCM activation
# set, applied here to GP *input coordinates* rather than sampled *outputs* —
# see apply_mlp_feature_mixing's docstring for why this preserves exact
# analytic Gaussianity while CauKer's approach would not).
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
        # Remainder by a fixed period (not data-dependent) keeps this a pure
        # deterministic function of x alone -> still a valid PSD-preserving
        # feature map; 2*pi period avoids introducing a new magic-number
        # scale unrelated to the 'sin' branch above.
        return torch.remainder(x, 2 * math.pi)
    if name == "leaky_relu":
        return torch.nn.functional.leaky_relu(x, negative_slope=0.1)
    raise ValueError(f"Unknown MLP-mixing activation '{name}'")


def apply_mlp_feature_mixing(
    x: Tensor, cfg, device, *, return_gate: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    """Randomly mix the GP's input feature columns through a small stack of
    dense affine + nonlinearity layers, applied to input coordinates x (never
    to sampled outputs y) so k(f(x_i), f(x_j)) remains a valid PSD kernel for
    the fixed deterministic map f = this mixing stack composed with
    tabiclv2_warp_features -- preserving EXACT analytic Gaussianity (closed-
    form GP posterior/Cholesky oracle), unlike CauKer's SCM approach of mixing
    sampled *outputs* through a random DAG (which would force Monte Carlo).

    Structure mirrors the rest of this file's "shared structure across the
    batch, independent per-episode parameters" convention (kernel_name,
    active_dims, tabiclv2_warp_features's per-(episode,column) transform
    choice): the number of layers L and each layer's activation name are
    sampled ONCE per batch call (shared across all B episodes); each layer's
    weight matrix and bias are sampled independently PER EPISODE and applied
    via a batched einsum (no Python loop over B).

    Note on active_dims/ARD semantics: this is a DENSE mix (every output
    column is a combination of every input column), so it partially subverts
    the "inactive_frac_min/max leaves some columns as pure noise" contract
    downstream in _sample_active_dims -- post-mixing, no column is purely
    irrelevant anymore. This is an accepted trade-off for increased task
    diversity, not a bug.

    Args:
        x: (B, T, d) tensor, already warped by tabiclv2_warp_features, NOT
            yet z-normalised (this runs before the existing per-episode
            mean/std normalisation step).
        cfg: Hydra config; reads cfg.data.mlp_mixing_* keys (see
            conf/data/gp_tasks.yaml), all optional/backward-compatible via
            getattr defaults (mlp_mixing_enabled defaults False -> exact
            no-op, byte-for-byte, for every existing config/dataset).
        device: torch device string, threaded through for the new W_l/b_l
            parameter tensors (same convention as the rest of this file).
        return_gate: if True, also return the (B,) bool tensor recording
            which episodes were actually mixed (used by generate_gp_batch's
            return_kernel_metadata=True path to report mlp-mixing usage per
            episode). Default False preserves the original single-tensor
            return type/behavior for every existing call site.

    Returns:
        (B, T, d) tensor, same shape/dtype as x. Episodes not selected by the
        per-episode Bernoulli gate (mlp_mixing_prob) are returned unchanged.
        If return_gate=True, returns (x, gate) instead, where gate is a (B,)
        bool tensor (all False when mixing is disabled/no-op).
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
    # Activation sequence shared across the whole batch call (same granularity
    # as kernel_name/P/N/active_dims above) -- NOT per-episode, so every mixed
    # episode in this batch call shares one topology, differing only in the
    # sampled W_l/b_l weight values.
    activations = [random.choice(_MLP_MIX_ACTIVATIONS) for _ in range(L)]

    x_mixed = x
    for act_name in activations:
        # 1/sqrt(d) fan-in scaling keeps the pre-activation roughly variance-
        # preserving (same purpose as Xavier/He init) -- an empirically-tuned
        # default rather than an analytically-guaranteed bound; validated by
        # test_mlp_mixing_goldilocks_and_psd in tests/test_data.py.
        W_l = torch.randn(B, d, d, device=device) * (w_std / math.sqrt(d))
        b_l = torch.randn(B, 1, d, device=device) * w_std
        x_mixed = torch.einsum("btd,bde->bte", x_mixed, W_l) + b_l
        x_mixed = _apply_mlp_activation(x_mixed, act_name)

    gate_1d = torch.rand(B, device=device) < mixing_prob  # (B,)
    gate = gate_1d[:, None, None]  # (B,1,1)
    # (B,1,1) is required for correct broadcast against (B,T,d); a bare (B,)
    # shape misaligns on the trailing (T, d) dims instead of the batch dim.
    x_out = torch.where(gate, x_mixed, x)
    if return_gate:
        return x_out, gate_1d
    return x_out


def apply_kernel_hidden_warp(
    x_norm: Tensor, cfg, device, *, return_gate: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
    """Warp the already-normalised GP input coordinates through a SEPARATE,
    hidden-only stack (down-projection -> nonlinear mixing at reduced width
    -> up-projection -> per-episode z-norm) that only the kernel/mean oracle
    ever sees. The model's own x_norm_train/x_norm_test (saved/returned by
    _generate_gp_batch_raw) are computed from `x_norm` directly and never
    touch this function's output.

    Why this exists: without it, kernel_obj/mean_module are evaluated on the
    exact same tensor the model is given (x_norm), so for any kernel that's a
    function of pairwise distance (rbf/matern/rq/...) the model can solve
    "predict R_star" by recognising the kernel family/hyperparameters
    straight from its own input and applying the closed-form formula, never
    needing the in-context y samples at all. Hiding the oracle's coordinate
    system behind another stack of the SAME kind as apply_mlp_feature_mixing
    would not be enough on its own: fan-in-scaled random d->d layers are, in
    expectation, close to a uniform rescaling of every pairwise distance
    (`E[||delta @ W||^2] = w_std^2 * ||delta||^2`, independent of delta's
    direction -- the same variance-preservation argument as He/Xavier init),
    so a model could still absorb the whole stack as one unknown scalar
    lengthscale correction. The down-projection to width r < d below is what
    actually breaks this: for delta = x_i - x_j, `delta @ W_down` depends
    only on delta's component inside the random r-dimensional row space of
    W_down; the (d-r)-dimensional orthogonal component is annihilated
    outright, so no single per-episode rescaling can map kernel-space
    distances back to model-space ones.

    R_star stays an exact oracle regardless of any of this: this function's
    output is still a deterministic function of the full realised randomness
    of the episode (x_norm plus this function's own fresh
    W_down/W_l/W_up/b_* draws) -- the same PSD/exactness argument already
    relied on for tabiclv2_warp_features/apply_mlp_feature_mixing composes
    through this additional deterministic map without change.

    Args:
        x_norm: (B, T, d) tensor, the fully-formed model-visible input (post
            tabiclv2_warp_features / apply_structural_feature_warp /
            apply_mlp_feature_mixing / z-normalisation) -- untouched by this
            function; its output is consumed ONLY by kernel_obj/mean_module
            in _generate_gp_batch_raw.
        cfg: Hydra config; reads cfg.data.kernel_hidden_* keys (see
            conf/data/gp_tasks.yaml), all optional/backward-compatible via
            getattr defaults (kernel_hidden_enabled defaults False -> exact
            no-op, byte-for-byte, for every existing config/dataset).
        device: torch device string, threaded through for the new weight
            tensors (same convention as apply_mlp_feature_mixing).
        return_gate: if True, also return the (B,) bool tensor recording
            which episodes actually got the hidden warp (used by
            generate_gp_batch's return_kernel_metadata=True path, same
            convention as apply_mlp_feature_mixing's mlp_mixed). Default
            False preserves a single-tensor return for every existing call
            site.

    Returns:
        (B, T, d) tensor, same shape/dtype as x_norm (down-then-up-projected
        back to width d, so kernel_obj's active_dims/d_total=d wiring needs
        no changes downstream). Episodes not selected by the per-episode
        Bernoulli gate (kernel_hidden_prob) fall back to x_norm unchanged
        (oracle == model input for those episodes, same partial-coverage
        convention as apply_mlp_feature_mixing's mlp_mixing_prob). If
        return_gate=True, returns (x_kernel, gate) instead, where gate is a
        (B,) bool tensor (all False when disabled/no-op).
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
    # Clamp to [1, d-1] so r is always a genuine rank reduction, even for the
    # small d (~8-15 typical, see d_features_lognormal_* in
    # conf/data/gp_tasks.yaml) this repo generates episodes with.
    r = max(1, min(d - 1, round(bottleneck_frac * d)))

    W_down = torch.randn(B, d, r, device=device) * (w_std / math.sqrt(d))
    b_down = torch.randn(B, 1, r, device=device) * w_std
    h = torch.einsum("btd,bdr->btr", x_norm, W_down) + b_down

    H = random.randint(H_min, H_max)
    # Same batch-shared-topology / per-episode-weights convention as
    # apply_mlp_feature_mixing: activation sequence sampled once per batch
    # call, weight matrices independent per episode.
    activations = [random.choice(_MLP_MIX_ACTIVATIONS) for _ in range(H)]
    for act_name in activations:
        W_l = torch.randn(B, r, r, device=device) * (w_std / math.sqrt(r))
        b_l = torch.randn(B, 1, r, device=device) * w_std
        h = torch.einsum("btr,brs->bts", h, W_l) + b_l
        h = _apply_mlp_activation(h, act_name)

    W_up = torch.randn(B, r, d, device=device) * (w_std / math.sqrt(r))
    b_up = torch.randn(B, 1, d, device=device) * w_std
    x_hidden = torch.einsum("btr,brd->btd", h, W_up) + b_up
    x_hidden = (
        (x_hidden - x_hidden.mean(1, keepdim=True))
        / x_hidden.std(1, keepdim=True).clamp(min=1e-8)
    )

    gate_1d = torch.rand(B, device=device) < hidden_prob  # (B,)
    gate = gate_1d[:, None, None]  # (B,1,1), see apply_mlp_feature_mixing
    x_kernel = torch.where(gate, x_hidden, x_norm)
    if return_gate:
        return x_kernel, gate_1d
    return x_kernel
