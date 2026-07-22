"""synthetic_bbo.py — Benchmark 3: d-dimensional synthetic GP with a known
ground-truth PRIOR correlation matrix at the test points.

The kernel itself is sampled with the exact same machinery
generate_gp_batch/generate_gp_task use during training
(_sample_kernel_chain_structure + _build_kernel_chain + _build_likelihood, all
driven by a Hydra cfg) rather than a hand-rolled reimplementation of the
chain-length/lengthscale/noise priors. This matters: a checkpoint's cfg fixes
not just which kernels are allowed (cfg.data.composite_exclude_kernels) but
also how many get chained together (composite_num_kernels_min/max) and how
long a typical lengthscale is (cfg.data.l_lognormal_loc/scale, k-scaled) —
getting either of those wrong silently produces a ground truth that decays
with distance much faster or slower than what the checkpoint actually
trained on, which shows up as a systematic (not random) miscalibration when
scoring corr_frob, even though the model's predictions look qualitatively
reasonable in a heatmap/corr-vs-distance plot. Pass the loaded checkpoint's
own cfg in (see run_benchmarks.py) rather than the module-level default.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import qmc

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data_gen import (  # noqa: E402
    _build_kernel_chain,
    _build_likelihood,
    _kernel_needs_scalar_input,
    _safe_cholesky,
    _sample_kernel_chain_structure,
    _seed_everything,
    gp_posterior,
    sigma_to_correlation,
)

__all__ = ["load_split"]

_PERTURB_STD = 0.05
_N_OPTIMA_SEEDS = 3

# Mirrors data_gen.py's own getattr-defaults for every key
# _sample_kernel_chain_structure/_build_kernel_chain/_build_likelihood read —
# used only when a caller doesn't have a checkpoint cfg on hand (standalone
# import/testing). composite_exclude_kernels defaults to periodic+cosine
# (the two data_gen._COMPOSABLE_KERNELS members whose values can go negative)
# since this benchmark's whole point is a positive-correlation ground truth.
_DEFAULT_CFG = OmegaConf.create(
    {
        "data": {
            "composite_exclude_kernels": ["periodic", "cosine"],
            "composite_num_kernels_min": 1,
            "composite_num_kernels_max": 4,
            "l_lognormal_loc": 0.0,
            "l_lognormal_scale": 0.7,
            "l_lognormal_k_exponent": 0.25,
            "l_lognormal_k_cap": 15,
            "alpha2_gamma_concentration": 4.0,
            "alpha2_gamma_rate": 3.0,
            "nugget_lognormal_loc": -4.63,
            "nugget_lognormal_scale": 0.5,
            "ard": False,
            "isotropic_ratio": 0.0,
        }
    }
)


def _sample_episode_kernel_fn(cfg, d: int, rng_np: np.random.Generator):
    """One episode's (kernel_fn, noise_variance), sampled via cfg's own
    chain/lengthscale/nugget priors (data_gen.py) — same as generate_gp_batch's
    dim-selection: scalar-only/periodic components get a single active column
    (k=1), everything else uses all d columns (k=d), matching this
    benchmark's "every feature is relevant" design (no inactive noise dims).
    """
    chain_names, chain_ops, kernel_name = _sample_kernel_chain_structure(cfg)
    if _kernel_needs_scalar_input(kernel_name) or "periodic" in kernel_name:
        active_dims = [int(rng_np.integers(0, d))]
    else:
        active_dims = None
    k = d if active_dims is None else len(active_dims)

    kernel_obj, _, _ = _build_kernel_chain(cfg, chain_names, chain_ops, k, B=1, device="cpu", active_dims=active_dims)
    likelihood = _build_likelihood(cfg, kernel_name, B=1, device="cpu")
    noise_var = float(likelihood.noise.item())

    def kernel_fn(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        return kernel_obj(X1, X2).to_dense().reshape(X1.shape[0], X2.shape[0])

    return kernel_fn, noise_var


@torch.no_grad()
def load_split(
    d: int,
    n_ctx: int,
    n_test: int,
    seed: int | None = None,
    cfg=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_train, y_train, X_test, y_test, R_ground_truth).

    Context points are Sobol-sampled in [0,1]^d. Test/query points are half a
    fresh Sobol batch (exploration) and half small Gaussian perturbations
    around the top-3 lowest-y context points (BBO minimization convention) —
    a lightweight stand-in for q-EI candidate proposals.

    `cfg` should be the actual checkpoint's training cfg (see
    inference.copula_inference.load_copula_model's return value) so the
    sampled kernel chain/lengthscale/noise match that checkpoint's training
    distribution; falls back to _DEFAULT_CFG otherwise. All kernel-sampling
    RNG (chain structure, hyperparameters) is reseeded from `seed` via
    data_gen._seed_everything for reproducibility.

    Kernel evaluation happens on features z-scored using context-only
    mean/std (test points share ~the same marginal distribution by
    construction, so this is a close proxy for jointly z-scoring ctx+test as
    data_gen.py's x_norm does, without the circularity of needing X_test's
    stats before X_test exists — X_test's "exploit" half depends on y_ctx).
    This scaling matters because the lengthscale prior is calibrated for
    unit-variance features; skipping it would make a fixed lengthscale
    reflect a shorter effective length in this benchmark's raw [0,1]^d domain
    than in training's z-scored domain, decaying with distance too fast
    relative to what the checkpoint learned. Only the correlation math sees
    the z-scored features — the returned X_train/X_test stay raw, since
    run_benchmarks.py's normalize_features re-standardizes them anyway before
    handing them to the actual models under test.

    y_test is drawn from the GP's noiseless posterior given context
    (matching this repo's "test targets are true function values, no
    observation noise" convention).

    R_ground_truth is the PRIOR correlation among test points — kernel
    structure only, no conditioning on context (data_gen.py's
    oracle_mode="prior": Sigma_star = K_ss, mu_star = 0) — rather than the
    conditioned posterior correlation, matching checkpoints trained with
    cfg.data.oracle_mode == "prior" (whose R_star never depended on context).
    Noise variance is folded into the diagonal, matching the nugget
    data_gen.py's K_ss carries.
    """
    cfg = _DEFAULT_CFG if cfg is None else cfg
    rng_np = np.random.default_rng(seed)
    torch_seed = int(rng_np.integers(0, 2**31 - 1))
    rng_torch = torch.Generator().manual_seed(torch_seed)
    _seed_everything(int(rng_np.integers(0, 2**31 - 1)))

    kernel_fn, noise_var = _sample_episode_kernel_fn(cfg, d, rng_np)

    sobol = qmc.Sobol(d=d, seed=rng_np)
    X_ctx = sobol.random(n_ctx)
    ctx_mean, ctx_std = X_ctx.mean(axis=0), X_ctx.std(axis=0)
    ctx_std_safe = np.clip(ctx_std, 1e-8, None)
    X_ctx_norm_t = torch.as_tensor((X_ctx - ctx_mean) / ctx_std_safe, dtype=torch.float32)

    K_ctx = kernel_fn(X_ctx_norm_t, X_ctx_norm_t) + noise_var * torch.eye(n_ctx)
    L_ctx = _safe_cholesky(K_ctx)
    eps_ctx = torch.randn(n_ctx, generator=rng_torch)
    f_ctx = (L_ctx @ eps_ctx.unsqueeze(-1)).squeeze(-1)
    y_ctx = f_ctx + np.sqrt(noise_var) * torch.randn(n_ctx, generator=rng_torch)

    n_explore = n_test // 2
    n_exploit = n_test - n_explore
    X_explore = sobol.random(n_explore)
    best_idx = np.argsort(y_ctx.numpy())[:_N_OPTIMA_SEEDS]
    centers = X_ctx[best_idx]
    picks = rng_np.integers(0, len(centers), size=n_exploit)
    perturb = rng_np.normal(0.0, _PERTURB_STD, size=(n_exploit, d))
    X_exploit = np.clip(centers[picks] + perturb, 0.0, 1.0)
    X_test = np.concatenate([X_explore, X_exploit], axis=0)
    X_test_norm_t = torch.as_tensor((X_test - ctx_mean) / ctx_std_safe, dtype=torch.float32)

    mu_star, Sigma_latent = gp_posterior(
        X_ctx_norm_t, y_ctx, X_test_norm_t, kernel_fn, noise=noise_var, latent=True
    )
    L_test = _safe_cholesky(Sigma_latent)
    eps_test = torch.randn(n_test, generator=rng_torch)
    y_test_t = mu_star + (L_test @ eps_test.unsqueeze(-1)).squeeze(-1)

    K_ss_prior = kernel_fn(X_test_norm_t, X_test_norm_t) + noise_var * torch.eye(n_test)
    R_true, _ = sigma_to_correlation(K_ss_prior)

    return X_ctx, y_ctx.numpy(), X_test, y_test_t.numpy(), R_true.numpy()
