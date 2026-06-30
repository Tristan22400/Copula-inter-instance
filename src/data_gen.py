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
    X1: Tensor, X2: Tensor, **_
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
    return_factors: bool = False,
) -> tuple:
    """Analytical GP posterior for an arbitrary stationary kernel.

    Args:
        latent: if True, return posterior over f* (latent GP), not noisy y*.
                K_ss excludes the noise term so that R* reflects kernel structure
                rather than being diluted by σ² in the diagonal.
        return_factors: if True, also return (L_ff, alpha) so the caller can
                reuse them for the LOO PIT without a second Cholesky.

    Returns:
        mu_star   : (N,)   — posterior mean at test points
        Sigma_star: (N, N) — posterior covariance at test points
        L_ff      : (P, P) — Cholesky of K_ff  (only if return_factors=True)
        alpha     : (P,)   — K_ff^{-1} y_train (only if return_factors=True)
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
    if return_factors:
        return mu_star, Sigma_star, L_ff, alpha
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

    # 8. Compute R_star from the GP posterior at test points.
    # Sigma_star_posterior = K_ss + nugget·I − K_st K_ff⁻¹ K_ts  accounts for
    # what the training context already explains.  Off-diagonal entries shrink
    # relative to the prior, giving the correct oracle for the copula the model
    # must learn: residual dependence after conditioning on training data.
    x_k_train = x_k[:P]
    x_k_test = x_k[P:]
    mu_star, Sigma_star, L_ff, alpha = gp_posterior(
        x_k_train, y_train, x_k_test, kernel_fn, nugget, return_factors=True
    )
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
        # Ephemeral Cholesky factors — consumed by gp_analytical_pit, not saved to disk.
        "_L_ff": L_ff,    # (P, P) Cholesky of K_ff
        "_alpha": alpha,  # (P,)   K_ff^{-1} y_train
    }


# ---------------------------------------------------------------------------
# Batched generation (C: vectorised over B episodes simultaneously)
# ---------------------------------------------------------------------------


def _batched_cholesky(K: Tensor) -> Tensor:
    """Batched Cholesky (B, N, N) → (B, N, N) with automatic jitter for failures."""
    L, info = torch.linalg.cholesky_ex(K)
    failed = info.ne(0)
    if not failed.any():
        return L
    eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
    for jitter in (1e-5, 1e-4, 1e-3, 1e-2, 0.1):
        if not failed.any():
            break
        K = K.clone()
        K[failed] = K[failed] + jitter * eye
        L_new, info_new = torch.linalg.cholesky_ex(K)
        L[failed] = L_new[failed]
        failed = info_new.ne(0)
    if failed.any():
        # Last resort: replace with identity so the episode is invalid but non-crashing.
        L[failed] = eye.unsqueeze(0).expand_as(L[failed])
    return L


def _batched_kernel_matrix(
    kernel_name: str,
    x_k: Tensor,                        # (B, T, k)
    l: Tensor,                          # (B,)
    alpha2: Tensor,                     # (B,)
    period: Optional[Tensor] = None,    # (B,) — periodic kernel only
    rq_alpha: Optional[Tensor] = None,  # (B,) — rational_quadratic only
) -> Tensor:
    """Build B kernel matrices for T points in one vectorised call.

    Returns (B, T, T) WITHOUT nugget on the diagonal.
    The caller is responsible for adding nugget * I.
    """
    B, T, _ = x_k.shape
    l3     = l.view(B, 1, 1)
    alpha3 = alpha2.view(B, 1, 1)

    if kernel_name == "dot_product":
        return x_k @ x_k.permute(0, 2, 1)  # (B, T, T)

    diff  = x_k.unsqueeze(2) - x_k.unsqueeze(1)  # (B, T, T, k)
    sq    = (diff ** 2).sum(-1)                    # (B, T, T)
    r     = sq.clamp(min=0.0).sqrt()

    if kernel_name == "rbf":
        return alpha3 * torch.exp(-sq / (2.0 * l3 ** 2))

    if kernel_name == "matern32":
        s = math.sqrt(3.0) * r / l3
        return alpha3 * (1.0 + s) * torch.exp(-s)

    if kernel_name == "cosine":
        return alpha3 * torch.cos(2.0 * math.pi * r / l3)

    if kernel_name == "periodic":
        p3 = period.view(B, 1, 1)
        return alpha3 * torch.exp(-2.0 * torch.sin(math.pi * r / p3) ** 2 / l3 ** 2)

    if kernel_name == "rational_quadratic":
        r3 = rq_alpha.view(B, 1, 1)
        return alpha3 * (1.0 + sq / (2.0 * r3 * l3 ** 2)) ** (-r3)

    raise ValueError(f"Unknown kernel '{kernel_name}' in _batched_kernel_matrix.")


def generate_gp_batch(cfg, B: int, device: str = "cpu") -> List[Dict[str, Tensor]]:
    """Generate B GP episodes in a single vectorised call.

    All B episodes share one kernel type and one (P, N) size (both sampled
    once per call) but have independent hyperparameters and feature draws.
    This removes the Python-loop overhead of B separate generate_gp_task calls
    and enables GPU or CPU-SIMD acceleration for the linear-algebra steps.

    The returned dicts have the same schema as the episodes saved by
    generate_pit_dataset.py (no kernel metadata).

    Args:
        cfg    : Hydra config (same as generate_gp_task).
        B      : number of episodes to generate in this batch.
        device : torch device string ("cpu" or "cuda").

    Returns:
        list of B episode dicts ready for torch.save.
    """
    import warnings

    d          = cfg.data.d_features
    nugget_min = float(getattr(cfg.data, "nugget_min", 0.1))
    nugget_max = float(getattr(cfg.data, "nugget_max", 1.0))

    # --- Shared settings for this batch ---
    kernel_name = _resolve_kernel_name(cfg)
    P = random.randint(cfg.data.P_min, cfg.data.P_max)
    N = random.randint(cfg.data.N_min, cfg.data.N_max)
    T = P + N

    if kernel_name in ("cosine", "periodic"):
        k = 1
    elif kernel_name == "dot_product":
        k = d
    else:
        k = random.randint(
            int(getattr(cfg.data, "d_kernel_min", 1)),
            min(int(getattr(cfg.data, "d_kernel_max", 4)), d),
        )

    # --- Per-episode hyperparameters ---
    l      = torch.rand(B, device=device) * (cfg.data.l_max      - cfg.data.l_min)      + cfg.data.l_min
    nugget = torch.rand(B, device=device) * (nugget_max           - nugget_min)           + nugget_min
    alpha2 = (
        torch.zeros(B, device=device)
        if kernel_name == "dot_product"
        else torch.rand(B, device=device) * (cfg.data.alpha2_max - cfg.data.alpha2_min) + cfg.data.alpha2_min
    )

    period = rq_alpha = None
    if kernel_name == "periodic":
        p_min  = float(getattr(cfg.data, "period_min",   0.5))
        p_max  = float(getattr(cfg.data, "period_max",   3.0))
        period = torch.rand(B, device=device) * (p_max - p_min) + p_min
    if kernel_name == "rational_quadratic":
        rq_min  = float(getattr(cfg.data, "rq_alpha_min", 0.1))
        rq_max  = float(getattr(cfg.data, "rq_alpha_max", 5.0))
        rq_alpha = torch.rand(B, device=device) * (rq_max - rq_min) + rq_min

    # --- Features (B, T, d) ~ U[-5, 5], normalised per episode ---
    x_raw  = torch.rand(B, T, d, device=device) * 10.0 - 5.0
    x_norm = (x_raw - x_raw.mean(1, keepdim=True)) / x_raw.std(1, keepdim=True).clamp(min=1e-8)

    # --- Select kernel columns ---
    if kernel_name == "dot_product":
        x_k = x_norm                                                                # (B, T, d)
    elif kernel_name in ("cosine", "periodic"):
        col = torch.randint(0, d, (B,), device=device)                             # (B,)
        x_k = x_norm.gather(2, col.view(B, 1, 1).expand(B, T, 1))                 # (B, T, 1)
    else:
        # Vectorised random column selection: argsort of uniform noise picks k cols
        col_idx = torch.rand(B, d, device=device).argsort(dim=1)[:, :k]           # (B, k)
        x_k = x_norm.gather(2, col_idx.unsqueeze(1).expand(B, T, k))              # (B, T, k)

    # --- Build K_all (B, T, T) and sample y jointly ---
    K_all = _batched_kernel_matrix(kernel_name, x_k, l, alpha2, period, rq_alpha)
    K_all = K_all + nugget.view(B, 1, 1) * torch.eye(T, device=device)
    K_all = 0.5 * (K_all + K_all.permute(0, 2, 1))               # symmetrize float32 drift

    L_all = _batched_cholesky(K_all)                              # (B, T, T)
    y_all = (L_all @ torch.randn(B, T, 1, device=device)).squeeze(-1)  # (B, T)

    x_norm_train = x_norm[:, :P]   # (B, P, d)
    x_norm_test  = x_norm[:, P:]   # (B, N, d)
    y_train      = y_all[:,  :P]   # (B, P)
    y_test       = y_all[:,  P:]   # (B, N)

    # --- Sub-matrices of K_all (nugget already on diagonal) ---
    K_ff = K_all[:, :P, :P]   # (B, P, P)
    K_sf = K_all[:, P:, :P]   # (B, N, P)
    K_ss = K_all[:, P:, P:]   # (B, N, N)

    # --- GP posterior (batched) ---
    L_ff  = _batched_cholesky(K_ff)
    alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L_ff).squeeze(-1)          # (B, P)
    mu_star  = (K_sf @ alpha.unsqueeze(-1)).squeeze(-1)                             # (B, N)

    V          = torch.linalg.solve_triangular(L_ff, K_sf.permute(0, 2, 1), upper=False)  # (B, P, N)
    Sigma_star = K_ss - V.permute(0, 2, 1) @ V                                     # (B, N, N)
    Sigma_star = 0.5 * (Sigma_star + Sigma_star.permute(0, 2, 1))

    # sigma_to_correlation (batched)
    var_diag   = Sigma_star.diagonal(dim1=1, dim2=2).clamp(min=1e-10)              # (B, N)
    sigma_star = var_diag.sqrt()
    inv_s      = var_diag.rsqrt()
    R_star     = Sigma_star * inv_s.unsqueeze(1) * inv_s.unsqueeze(2)             # (B, N, N)
    d_diag     = R_star.diagonal(dim1=1, dim2=2).clamp(min=1e-10).sqrt()
    R_star     = R_star / (d_diag.unsqueeze(1) * d_diag.unsqueeze(2))

    # --- LOO PIT for z_train (R&W Eq. 5.12, batched) ---
    # diag(K_ff^{-1}) = column-squared-norm of L_ff^{-1}
    eye_P      = torch.eye(P, device=device)
    L_inv      = torch.linalg.solve_triangular(
        L_ff, eye_P.unsqueeze(0).expand(B, -1, -1), upper=False
    )                                                                               # (B, P, P)
    K_inv_diag = (L_inv ** 2).sum(dim=1).clamp(min=1e-12)                         # (B, P)
    z_train    = alpha * K_inv_diag.rsqrt()                                       # (B, P)

    # --- Posterior PIT for z_test ---
    sig_c        = sigma_star.clamp(min=1e-8)
    z_test       = (y_test - mu_star) / sig_c                                      # (B, N)
    log_pdf_test = (
        -0.5 * math.log(2.0 * math.pi) - sig_c.log() - 0.5 * z_test ** 2
    )                                                                               # (B, N)

    # LOO residuals are N(0,1) by construction (R&W Eq. 5.12); no empirical
    # rescaling needed.  Filter degenerate episodes instead.
    z_std = z_train.std(dim=1)
    degen = (z_std < 0.1) | (z_std > 3.0)
    if degen.any():
        warnings.warn(
            f"generate_gp_batch: {int(degen.sum())}/{B} episodes have degenerate LOO z.",
            RuntimeWarning,
        )

    # Reconstruct full posterior covariance (for Y-space oracle)
    Sigma_full = R_star * sigma_star.unsqueeze(1) * sigma_star.unsqueeze(2)       # (B, N, N)

    # --- Pack into list of dicts (single D→H transfer) ---
    tensors = {
        "x_norm_train": x_norm_train.cpu(),
        "x_norm_test":  x_norm_test.cpu(),
        "y_train":      y_train.cpu(),
        "y_test":       y_test.cpu(),
        "z_train":      z_train.cpu(),
        "z_test":       z_test.cpu(),
        "log_pdf_test": log_pdf_test.cpu(),
        "R_star":       R_star.cpu(),
        "Sigma_star":   Sigma_full.cpu(),
        "mu_star":      mu_star.cpu(),
        "sigma_star":   sigma_star.cpu(),
    }
    n_tr = torch.tensor(P)
    n_te = torch.tensor(N)
    return [
        {key: val[b] for key, val in tensors.items()} | {"n_train": n_tr, "n_test": n_te}
        for b in range(B)
    ]
