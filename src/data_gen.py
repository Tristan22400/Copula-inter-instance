"""
data_gen.py — Stage A: GP task generation for inter-instance copula.

Each task samples a random GP with an RBF kernel, draws P+N instances,
normalises features over the full P+N set, samples targets jointly from
the GP, computes the analytical posterior correlation matrix R*, and saves
all required tensors.

The copula in this project captures correlations between the N TEST instances
(scalar targets), not between target dimensions.  The "dimensions" of the
joint distribution are the N query instances themselves.
"""

from __future__ import annotations

import math
import random
from typing import Dict

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------


def rbf_kernel(X1: Tensor, X2: Tensor, l: float, alpha2: float) -> Tensor:
    diff = X1.unsqueeze(1) - X2.unsqueeze(0)  # (n1, n2, d)
    sq_dist = (diff**2).sum(-1)
    return alpha2 * torch.exp(-sq_dist / (2 * l**2))


def matern32_kernel(X1: Tensor, X2: Tensor, l: float, alpha2: float) -> Tensor:
    diff = X1.unsqueeze(1) - X2.unsqueeze(0)  # (n1, n2, d)
    r = (diff**2).sum(-1).clamp(min=0).sqrt()
    s = math.sqrt(3) * r / l
    return alpha2 * (1.0 + s) * torch.exp(-s)


def matern52_kernel(X1: Tensor, X2: Tensor, l: float, alpha2: float) -> Tensor:
    diff = X1.unsqueeze(1) - X2.unsqueeze(0)  # (n1, n2, d)
    r = (diff**2).sum(-1).clamp(min=0).sqrt()
    s = math.sqrt(5) * r / l
    return alpha2 * (1.0 + s + s**2 / 3.0) * torch.exp(-s)


def periodic_kernel(
    X1: Tensor, X2: Tensor, l: float, alpha2: float, period: float
) -> Tensor:
    diff = X1.unsqueeze(1) - X2.unsqueeze(0)  # (n1, n2, d)
    r = (diff**2).sum(-1).clamp(min=0).sqrt()
    return alpha2 * torch.exp(-2.0 * torch.sin(math.pi * r / period) ** 2 / l**2)


_KERNEL_REGISTRY = {
    "rbf": rbf_kernel,
    "matern32": matern32_kernel,
    "matern52": matern52_kernel,
    "periodic": periodic_kernel,
}


def gp_posterior(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    l: float,
    alpha2: float,
    noise: float,
    kernel_fn,
    kernel_kwargs: dict,
) -> tuple[Tensor, Tensor]:
    """Analytical GP posterior for any stationary kernel.

    Returns:
        mu_star   : (N,)   — posterior mean at test points
        Sigma_star: (N, N) — posterior covariance at test points
    """
    P, N = x_train.shape[0], x_test.shape[0]

    K_ff = kernel_fn(x_train, x_train, l, alpha2, **kernel_kwargs)  # (P, P)
    K_ff = K_ff + noise * torch.eye(P, device=K_ff.device)

    K_sf = kernel_fn(x_test, x_train, l, alpha2, **kernel_kwargs)  # (N, P)
    K_ss = kernel_fn(x_test, x_test, l, alpha2, **kernel_kwargs)  # (N, N)
    K_ss = K_ss + noise * torch.eye(N, device=K_ss.device)

    # K_ff^{-1} y_train via Cholesky
    L_ff = torch.linalg.cholesky(K_ff)
    alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L_ff).squeeze(-1)  # (P,)

    mu_star = K_sf @ alpha  # (N,)

    # K_sf @ K_ff^{-1} @ K_fs via Cholesky: V = L_ff^{-1} K_fs
    K_fs = K_sf.T  # (P, N)
    V = torch.linalg.solve_triangular(L_ff, K_fs, upper=False)  # (P, N)
    Sigma_star = K_ss - V.T @ V  # (N, N)

    # Symmetrize for numerical stability
    Sigma_star = 0.5 * (Sigma_star + Sigma_star.T)
    return mu_star, Sigma_star


def sigma_to_correlation(Sigma: Tensor) -> tuple[Tensor, Tensor]:
    """Convert covariance matrix to correlation matrix and marginal std."""
    sigma = Sigma.diagonal().clamp(min=1e-10).sqrt()  # (N,)
    D_inv = torch.diag(1.0 / sigma)
    R = D_inv @ Sigma @ D_inv
    # Ensure unit diagonal (guard against numerical drift)
    R = R / R.diagonal().clamp(min=1e-10).sqrt().unsqueeze(0)
    R = R / R.diagonal().clamp(min=1e-10).sqrt().unsqueeze(1)
    return R, sigma


# ---------------------------------------------------------------------------
# Task generator
# ---------------------------------------------------------------------------


def generate_gp_task(cfg) -> Dict[str, Tensor]:
    """Sample one GP task with a randomly chosen kernel family.

    Kernels sampled uniformly from: RBF, Matérn-3/2, Matérn-5/2, Periodic.
    This creates diverse correlation structures (smooth, rough, oscillatory)
    that the copula model must learn to capture from features alone.

    Keys returned:
        x_norm_train : (P, d_x)  normalised train features
        y_train      : (P,)      train targets
        x_norm_test  : (N, d_x)  normalised test features
        y_test       : (N,)      test targets
        R_star       : (N, N)    ground-truth test correlation matrix
        mu_star      : (N,)      GP posterior mean at test points
        sigma_star   : (N,)      GP posterior marginal std at test points
        n_train      : int       P (as 0-dim tensor)
        n_test       : int       N (as 0-dim tensor)
        kernel       : str       kernel family name
    """
    d = cfg.data.d_features

    # 1. Sample kernel family and hyperparameters
    kernel_name = random.choice(["rbf", "matern32", "matern52", "periodic"])
    kernel_fn = _KERNEL_REGISTRY[kernel_name]

    l = random.uniform(cfg.data.l_min, cfg.data.l_max)
    alpha2 = random.uniform(cfg.data.alpha2_min, cfg.data.alpha2_max)
    noise = random.uniform(cfg.data.noise_min, cfg.data.noise_max)

    kernel_kwargs = {}
    if kernel_name == "periodic":
        period = random.uniform(cfg.data.period_min, cfg.data.period_max)
        kernel_kwargs["period"] = period

    # 2. Sample dataset sizes
    P = random.randint(cfg.data.P_min, cfg.data.P_max)
    N = random.randint(cfg.data.N_min, cfg.data.N_max)

    # 3. Sample features x ~ U([-5, 5]^d), all P+N instances
    x_raw = torch.rand(P + N, d) * 10.0 - 5.0

    # 4. Normalise features using statistics over ALL P+N instances
    mu_x = x_raw.mean(0)
    std_x = x_raw.std(0).clamp(min=1e-8)
    x_norm = (x_raw - mu_x) / std_x

    # 5. Build kernel and sample y ~ GP(0, K + noise*I) jointly
    x_all = x_norm  # (P+N, d)
    K_all = kernel_fn(x_all, x_all, l, alpha2, **kernel_kwargs)
    K_all = K_all + noise * torch.eye(P + N)

    try:
        L_all = torch.linalg.cholesky(K_all)
    except torch.linalg.LinAlgError:
        K_all = K_all + 1e-4 * torch.eye(P + N)
        L_all = torch.linalg.cholesky(K_all)
    y_all = L_all @ torch.randn(P + N)

    # 6. Split into train / test
    x_norm_train = x_norm[:P]
    y_train = y_all[:P]
    x_norm_test = x_norm[P:]
    y_test = y_all[P:]

    # 7. Compute GP posterior at test points (for oracle evaluation)
    mu_star, Sigma_star = gp_posterior(
        x_norm_train, y_train, x_norm_test, l, alpha2, noise, kernel_fn, kernel_kwargs
    )
    R_star, sigma_star = sigma_to_correlation(Sigma_star)

    return {
        "x_norm_train": x_norm_train,  # (P, d_x)
        "y_train": y_train,  # (P,)
        "x_norm_test": x_norm_test,  # (N, d_x)
        "y_test": y_test,  # (N,)
        "R_star": R_star,  # (N, N)
        "mu_star": mu_star,  # (N,)
        "sigma_star": sigma_star,  # (N,)
        "n_train": torch.tensor(P),
        "n_test": torch.tensor(N),
        "kernel": kernel_name,
    }
