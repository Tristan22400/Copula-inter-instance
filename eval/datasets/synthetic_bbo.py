"""Synthetic d-dimensional GP benchmark with a known prior correlation at the test points.

Kernels are sampled with data_gen's own chain, lengthscale and nugget priors
from the given cfg; pass the checkpoint's own cfg so the ground truth matches
its training distribution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import qmc

from copula_inter.data_gen import (  # noqa: E402
    _build_kernel_chain,
    _build_likelihood,
    _kernel_needs_scalar_input,
    _safe_cholesky,
    _sample_kernel_chain_structure,
    _seed_everything,
    gp_posterior,
    sigma_to_correlation,
)

if TYPE_CHECKING:
    from copula_inter.type_aliases import HasDataConfig

__all__ = ["load_split"]

_PERTURB_STD = 0.05
_N_OPTIMA_SEEDS = 3

# Defaults for data_gen's keys when no checkpoint cfg is given (periodic and cosine excluded).
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


def _sample_episode_kernel_fn(
    cfg: HasDataConfig, d: int, rng_np: np.random.Generator
) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], float]:
    """One episode's (kernel_fn, noise_variance) from cfg's priors; scalar-only and periodic components use one column, others all d."""
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
    cfg: HasDataConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_test, y_test, R_ground_truth).

    Context points are Sobol samples in [0, 1]^d; test points are half a new
    Sobol batch and half Gaussian perturbations around the three lowest-y
    context points. The kernel is evaluated on features z-scored with the
    context's statistics (returned X stay raw). y_test comes from the noiseless
    posterior; R_ground_truth is the prior test correlation with the noise on the
    diagonal. All sampling is seeded from seed.
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

    mu_star, Sigma_latent = gp_posterior(X_ctx_norm_t, y_ctx, X_test_norm_t, kernel_fn, noise=noise_var, latent=True)
    L_test = _safe_cholesky(Sigma_latent)
    eps_test = torch.randn(n_test, generator=rng_torch)
    y_test_t = mu_star + (L_test @ eps_test.unsqueeze(-1)).squeeze(-1)

    K_ss_prior = kernel_fn(X_test_norm_t, X_test_norm_t) + noise_var * torch.eye(n_test)
    R_true, _ = sigma_to_correlation(K_ss_prior)

    return X_ctx, y_ctx.numpy(), X_test, y_test_t.numpy(), R_true.numpy()
