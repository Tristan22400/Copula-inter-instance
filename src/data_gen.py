"""
data_gen.py — Stage A: GP task generation for inter-instance copula.

Each task samples a random GP with a configurable PSD kernel, draws P+N
instances, normalises features over the full P+N set, samples targets jointly
from the GP, computes the analytical posterior correlation matrix R*, and saves
all required tensors.

Supported kernels
-----------------
  rbf                 — Squared Exponential / RBF
  matern32            — Matérn ν=3/2
  cosine              — Cosine (spectral): k(r) = alpha2 * cos(2π r / l)
  periodic            — Periodic: k(r) = alpha2 * exp(-2 sin²(π r / period) / l²)
  rational_quadratic  — Rational Quadratic: k(r) = alpha2 * (1 + r²/(2α l²))^{-α}
  dot_product         — Linear + bias: k(x1,x2) = alpha2 + x1ᵀx2

Kernel selection (cfg.data.kernel / cfg.data.kernels)
------------------------------------------------------
  cfg.data.kernel   : str          → use this single kernel for every task
  cfg.data.kernels  : list[str]    → sample uniformly at task generation time
  If both are absent the default is "rbf".
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor

from loss import _safe_cholesky

# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------


def _sq_dist(X1: Tensor, X2: Tensor) -> Tensor:
    """Squared Euclidean distance matrix (n1, n2)."""
    diff = X1.unsqueeze(1) - X2.unsqueeze(0)  # (n1, n2, d)
    return (diff**2).sum(-1)


def _dist(X1: Tensor, X2: Tensor) -> Tensor:
    """Euclidean distance matrix (n1, n2)."""
    return _sq_dist(X1, X2).clamp(min=0.0).sqrt()


# ---------------------------------------------------------------------------
# PSD kernel functions  k(X1, X2) -> (n1, n2) matrix
# ---------------------------------------------------------------------------


def rbf_kernel(X1: Tensor, X2: Tensor, *, l: float, alpha2: float, **_) -> Tensor:
    """Squared Exponential (RBF): alpha2 * exp(-r² / (2 l²))."""
    return alpha2 * torch.exp(-_sq_dist(X1, X2) / (2.0 * l**2))


def matern32_kernel(X1: Tensor, X2: Tensor, *, l: float, alpha2: float, **_) -> Tensor:
    """Matérn ν=3/2: alpha2 * (1 + √3 r/l) * exp(-√3 r/l)."""
    r = _dist(X1, X2)
    s = math.sqrt(3.0) * r / l
    return alpha2 * (1.0 + s) * torch.exp(-s)


def cosine_kernel(X1: Tensor, X2: Tensor, *, l: float, alpha2: float, **_) -> Tensor:
    """Cosine (spectral): alpha2 * cos(2π r / l).

    Stationary PSD kernel whose spectral density is a pair of Dirac deltas at
    ±1/l.  Small negative eigenvalues from floating-point rounding are handled
    by the jitter already applied in generate_gp_task.
    """
    r = _dist(X1, X2)
    return alpha2 * torch.cos(2.0 * math.pi * r / l)


def periodic_kernel(
    X1: Tensor,
    X2: Tensor,
    *,
    l: float,
    alpha2: float,
    period: float = 1.0,
    **_,
) -> Tensor:
    """Periodic: alpha2 * exp(-2 sin²(π r / period) / l²)."""
    r = _dist(X1, X2)
    return alpha2 * torch.exp(-2.0 * torch.sin(math.pi * r / period) ** 2 / l**2)


def rational_quadratic_kernel(
    X1: Tensor,
    X2: Tensor,
    *,
    l: float,
    alpha2: float,
    rq_alpha: float = 1.0,
    **_,
) -> Tensor:
    """Rational Quadratic: alpha2 * (1 + r² / (2 α l²))^{-α}."""
    sq = _sq_dist(X1, X2)
    return alpha2 * (1.0 + sq / (2.0 * rq_alpha * l**2)) ** (-rq_alpha)


def dot_product_kernel(
    X1: Tensor, X2: Tensor, *, **_
) -> Tensor:
    """Linear + bias: alpha2 + X1 @ X2ᵀ.

    PSD because K = XᵀX is a sum of two PSD matrices.
    Length-scale l is unused; geometry is determined by the feature space.
    """
    return X1 @ X2.T


# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------

KERNEL_REGISTRY: Dict[str, Callable[..., Tensor]] = {
    "rbf": rbf_kernel,
    "matern32": matern32_kernel,
    "cosine": cosine_kernel,
    "periodic": periodic_kernel,
    "rational_quadratic": rational_quadratic_kernel,
    "dot_product": dot_product_kernel,
}

ALL_KERNELS: List[str] = list(KERNEL_REGISTRY.keys())


def build_kernel_fn(
    kernel_name: str,
    l: float,
    alpha2: float,
    *,
    period: Optional[float] = None,
    rq_alpha: Optional[float] = None,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Return a kernel(X1, X2) -> K callable with hyperparameters baked in."""
    fn = KERNEL_REGISTRY[kernel_name]
    kwargs: Dict = dict(l=l, alpha2=alpha2)
    if period is not None:
        kwargs["period"] = period
    if rq_alpha is not None:
        kwargs["rq_alpha"] = rq_alpha
    return lambda X1, X2: fn(X1, X2, **kwargs)


def _sample_kernel_cols(d_total: int, cfg) -> List[int]:
    """Return a sorted list of column indices that the kernel will use.

    k ~ Uniform[d_kernel_min, d_kernel_max]; falls back to all columns when
    the config keys are absent (backward compat with old episode files).
    """
    d_min = int(getattr(cfg.data, "d_kernel_min", d_total))
    d_max = int(getattr(cfg.data, "d_kernel_max", d_total))
    d_min = min(d_min, d_total)
    d_max = min(d_max, d_total)
    k = random.randint(d_min, d_max)
    return sorted(random.sample(range(d_total), k))


def _resolve_kernel_name(cfg) -> str:
    """Pick which kernel to use for one task based on config."""
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
        return random.choice(pool)
    return "rbf"


# ---------------------------------------------------------------------------
# GP posterior (kernel-agnostic)
# ---------------------------------------------------------------------------


def gp_posterior(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    kernel_fn: Callable[[Tensor, Tensor], Tensor],
    noise: float,
    *,
    latent: bool = False,
) -> tuple[Tensor, Tensor]:
    """Analytical GP posterior for an arbitrary stationary kernel.

    Args:
        latent: if True, return posterior over f* (latent GP), not noisy y*.
                K_ss excludes the noise term so that R* reflects kernel structure
                rather than being diluted by σ² in the diagonal.

    Returns:
        mu_star   : (N,)   — posterior mean at test points
        Sigma_star: (N, N) — posterior covariance at test points
    """
    P, N = x_train.shape[0], x_test.shape[0]

    K_ff = kernel_fn(x_train, x_train) + noise * torch.eye(P, device=x_train.device)
    K_sf = kernel_fn(x_test, x_train)   # (N, P)
    K_ss = kernel_fn(x_test, x_test)
    if not latent:
        K_ss = K_ss + noise * torch.eye(N, device=x_test.device)

    L_ff = _safe_cholesky(K_ff, max_attempts=12)
    alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L_ff).squeeze(-1)  # (P,)

    mu_star = K_sf @ alpha  # (N,)

    V = torch.linalg.solve_triangular(L_ff, K_sf.T, upper=False)  # (P, N)
    Sigma_star = K_ss - V.T @ V  # (N, N)
    Sigma_star = 0.5 * (Sigma_star + Sigma_star.T)
    return mu_star, Sigma_star


def sigma_to_correlation(Sigma: Tensor) -> tuple[Tensor, Tensor]:
    """Convert covariance matrix to correlation matrix and marginal std."""
    sigma = Sigma.diagonal().clamp(min=1e-10).sqrt()  # (N,)
    D_inv = torch.diag(1.0 / sigma)
    R = D_inv @ Sigma @ D_inv
    # One-shot re-normalization using the original sigma (symmetric in i,j).
    # D_inv @ Sigma @ D_inv already gives diagonal=1 for PSD Sigma; this just
    # corrects any float32 rounding drift without introducing asymmetry.
    d = R.diagonal().clamp(min=1e-10).sqrt()
    R = R / (d.unsqueeze(0) * d.unsqueeze(1))
    return R, sigma


# ---------------------------------------------------------------------------
# Task generator
# ---------------------------------------------------------------------------


def generate_gp_task(cfg) -> Dict[str, Tensor]:
    """Sample one GP task and return a dict of tensors.

    The kernel operates on a random subset of k columns (k ~ Uniform[d_kernel_min,
    d_kernel_max]); the full d_features columns are returned so the model must
    identify which features drive the correlations.

    cosine and periodic kernels are capped to k=1 (they are PSD only for scalar inputs).

    A nugget ~ U[nugget_min, nugget_max] is added to the diagonal for guaranteed PSD
    and controls posterior tightness (replaces the former separate noise parameter).

    Keys returned:
        x_norm_train          : (P, d_features)  normalised train features (full)
        y_train               : (P,)             train targets
        x_norm_test           : (N, d_features)  normalised test features (full)
        y_test                : (N,)             test targets
        R_star                : (N, N)           ground-truth test correlation matrix
        mu_star               : (N,)             GP posterior mean at test points
        sigma_star            : (N,)             GP posterior marginal std at test points
        n_train               : int              P (as 0-dim tensor)
        n_test                : int              N (as 0-dim tensor)
        l                     : float            kernel length scale (scalar tensor)
        alpha2                : float            kernel variance / bias (scalar tensor)
        nugget                : float            diagonal regulariser ~ U[nugget_min, nugget_max]
        kernel                : str              name of the kernel used
        period                : float            period param (periodic kernel only, else 0.0)
        rq_alpha              : float            alpha param (rational_quadratic only, else 0.0)
        kernel_feature_indices: (k,)             column indices used by the kernel (metadata)
    """
    d = cfg.data.d_features

    # 1. Choose kernel and active columns
    kernel_name = _resolve_kernel_name(cfg)

    # cosine and periodic are PSD only for scalar (1D) inputs; cap to k=1
    if kernel_name in ("cosine", "periodic"):
        kernel_cols = [random.randint(0, d - 1)]
    else:
        kernel_cols = _sample_kernel_cols(d, cfg)

    # 2. Sample shared GP hyperparameters
    l = random.uniform(cfg.data.l_min, cfg.data.l_max)
    if kernel_name == "dot_product":
        a2_min = float(0.0)
        a2_max = float(0.0)
    else:
        a2_min = float(cfg.data.alpha2_min)
        a2_max = float(cfg.data.alpha2_max)
    alpha2 = random.uniform(a2_min, a2_max)

    # Nugget: single diagonal regulariser ~ U[nugget_min, nugget_max].
    # Replaces the separate noise parameter (the two were always summed anyway).
    nugget_min = float(getattr(cfg.data, "nugget_min", 0.1))
    nugget_max = float(getattr(cfg.data, "nugget_max", 1.0))
    nugget = random.uniform(nugget_min, nugget_max)

    # 3. Extra hyperparameters for specific kernels
    period: Optional[float] = None
    rq_alpha: Optional[float] = None

    if kernel_name == "periodic":
        p_min = float(getattr(cfg.data, "period_min", 0.5))
        p_max = float(getattr(cfg.data, "period_max", 3.0))
        period = random.uniform(p_min, p_max)

    if kernel_name == "rational_quadratic":
        rq_min = float(getattr(cfg.data, "rq_alpha_min", 0.1))
        rq_max = float(getattr(cfg.data, "rq_alpha_max", 5.0))
        rq_alpha = random.uniform(rq_min, rq_max)

    # 4. Sample dataset sizes
    P = random.randint(cfg.data.P_min, cfg.data.P_max)
    N = random.randint(cfg.data.N_min, cfg.data.N_max)

    # 5. Sample features x ~ U([-5, 5]^d), normalise over all P+N instances
    x_raw = torch.rand(P + N, d) * 10.0 - 5.0
    mu_x = x_raw.mean(0)
    std_x = x_raw.std(0).clamp(min=1e-8)
    x_norm = (x_raw - mu_x) / std_x                    # full (P+N, d_features)
    x_k = x_norm[:, kernel_cols]                        # kernel sub-matrix (P+N, k)

    # 6. Build kernel function and sample y ~ GP(0, K + nugget·I) jointly
    kernel_fn = build_kernel_fn(kernel_name, l, alpha2, period=period, rq_alpha=rq_alpha)
    K_all = kernel_fn(x_k, x_k) + nugget * torch.eye(P + N)

    y_all = _safe_cholesky(K_all, max_attempts=12) @ torch.randn(P + N)

    # 7. Split into train / test (full features returned to model)
    x_norm_train = x_norm[:P]
    y_train = y_all[:P]
    x_norm_test = x_norm[P:]
    y_test = y_all[P:]

    # 8. Compute R_star from the GP prior at test points.
    # K_ss = kernel(X_test, X_test) + nugget·I.  The nugget floor guarantees
    # min_eig(K_ss) ≥ nugget > 0 (no near-singular oracle).  Off-diagonal
    # correlations span ±alpha2/(alpha2+nugget) with an arcsine distribution,
    # giving a rich and diverse training signal.  mu_star is set to the GP
    # posterior mean so the Y-space marginals remain accurate.
    x_k_train = x_k[:P]
    x_k_test = x_k[P:]
    Sigma_star = kernel_fn(x_k_test, x_k_test) + nugget * torch.eye(N)
    mu_star, _ = gp_posterior(x_k_train, y_train, x_k_test, kernel_fn, nugget)
    R_star, sigma_star = sigma_to_correlation(Sigma_star)

    return {
        "x_norm_train": x_norm_train,                           # (P, d_features)
        "y_train": y_train,                                      # (P,)
        "x_norm_test": x_norm_test,                             # (N, d_features)
        "y_test": y_test,                                        # (N,)
        "R_star": R_star,                                        # (N, N)
        "mu_star": mu_star,                                      # (N,)
        "sigma_star": sigma_star,                                # (N,)
        "n_train": torch.tensor(P),
        "n_test": torch.tensor(N),
        "l": torch.tensor(l),
        "alpha2": torch.tensor(alpha2),
        "nugget": torch.tensor(nugget),
        "kernel": kernel_name,
        "period": torch.tensor(period if period is not None else 0.0),
        "rq_alpha": torch.tensor(rq_alpha if rq_alpha is not None else 0.0),
        "kernel_feature_indices": torch.tensor(kernel_cols, dtype=torch.long),
    }
