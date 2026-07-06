"""
data_gen.py — Stage A: GP task generation for inter-instance copula.

Each task samples a random GP with a configurable PSD kernel, draws P+N
instances, normalises features over the full P+N set, samples targets jointly
from the GP, computes the analytical correlation matrix R* at the test points
(from either the GP posterior conditioned on training data or the raw GP
prior, per cfg.data.oracle_mode), and saves all required tensors.

Supported kernels
-----------------
  rbf                 — Squared Exponential / RBF
  matern32            — Matérn ν=3/2
  cosine              — Cosine (spectral): k(r) = alpha2 * cos(2π r / l)
  periodic            — Periodic: k(r) = alpha2 * exp(-2 sin²(π r / period) / l²)
  rational_quadratic  — Rational Quadratic: k(r) = alpha2 * (1 + r²/(2α l²))^{-α}
  dot_product         — Linear (dot product): k(x1,x2) = x1ᵀx2
  lsh_forest          — LSH Forest: k(x1,x2) = alpha2 * (fraction of random-hyperplane
                         trees where x1,x2 land in the same leaf). Ultrametric,
                         block-diagonal structure (sharp regime changes, like a
                         decision tree ensemble) instead of a smooth distance decay.

Composite kernels ("A+B" / "A*B")
---------------------------------
  Sums and products of PSD kernels are PSD, so every pair drawn from
  {rbf, matern32, cosine, periodic, rational_quadratic} is auto-registered
  under both operators, e.g. "rbf+periodic" (locally periodic: smooth decay
  times exact periodicity) or "matern32*cosine" (spectral windowing). See
  COMPOSITE_KERNELS for the full list. dot_product and lsh_forest are not
  composable (irregular hyperparameter signatures).

Kernel selection (cfg.data.kernel / cfg.data.kernels)
------------------------------------------------------
  cfg.data.kernel   : str          → use this single kernel for every task
                                     (any entry in ALL_KERNELS, including composites)
  cfg.data.kernels  : list[str]    → sample uniformly at task generation time
  If both are absent the default is "rbf".
"""

from __future__ import annotations

import functools
import itertools
import math
import random
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from loss import _safe_cholesky


def _seed_everything(seed: int) -> None:
    """Seed python/numpy/torch RNGs for reproducible data generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe even with a single GPU / no GPU


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
    """Linear (dot product): X1 @ X2ᵀ.

    PSD because K = XᵀX is PSD. Has no lengthscale or variance hyperparameter;
    geometry is determined entirely by the feature space.
    """
    return X1 @ X2.T


def _lsh_leaf_ids(X: Tensor, W: Tensor, b: Tensor) -> Tensor:
    """Hash each row of X through every tree, return one leaf id per tree.

    X: (n, k), W: (k, num_trees, depth), b: (num_trees, depth) -> (n, num_trees) int64.
    Each of the `depth` random hyperplanes per tree contributes one bit; the
    `depth` bits are packed into a single leaf id so a same-leaf test across a
    tree is one integer comparison instead of `depth` boolean comparisons.
    """
    depth = W.shape[-1]
    bits = torch.einsum("nk,ksd->nsd", X, W) + b > 0  # (n, num_trees, depth)
    weights = (2 ** torch.arange(depth, device=X.device)).to(torch.int64)
    return (bits.long() * weights).sum(-1)  # (n, num_trees)


def lsh_forest_kernel(
    X1: Tensor, X2: Tensor, *, alpha2: float, lsh_W: Tensor, lsh_b: Tensor, **_
) -> Tensor:
    """LSH Forest: alpha2 * (fraction of trees where x1, x2 land in the same leaf).

    Each tree is a set of `depth` random hyperplanes that partitions the input
    space into up to 2**depth cells (a random-hyperplane / SimHash forest).
    Within one tree, the leaf-membership indicator matrix is a block-diagonal
    0/1 matrix, which is PSD; a convex average (over trees) of PSD matrices is
    PSD, so the kernel is valid for any alpha2 >= 0. This produces sharp,
    ultrametric block structure (like a decision-tree ensemble) instead of the
    smooth distance-based decay of the other stationary kernels.

    lsh_W/lsh_b must be the SAME realization for every kernel_fn(X1, X2) call
    within one episode — gp_posterior calls the kernel three times (K_ff, K_sf,
    K_ss) with different point sets, so the hyperplanes are sampled once per
    episode (see generate_gp_task/generate_gp_batch) and passed in here rather
    than being resampled on every call.
    """
    ids1 = _lsh_leaf_ids(X1, lsh_W, lsh_b)  # (n1, num_trees)
    ids2 = _lsh_leaf_ids(X2, lsh_W, lsh_b)  # (n2, num_trees)
    matches = ids1.unsqueeze(1) == ids2.unsqueeze(0)  # (n1, n2, num_trees)
    return alpha2 * matches.float().mean(-1)


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
    "lsh_forest": lsh_forest_kernel,
}


# ---------------------------------------------------------------------------
# Composite kernels: sum / product of two base kernels
# ---------------------------------------------------------------------------
# Sums and products of PSD kernels are PSD, so "rbf+periodic" (locally
# periodic — smooth decay times exact periodicity) or "matern32*cosine"
# (spectral windowing) are valid kernels without any new math. Restricted to
# the kernels below because they share one calling convention (l, alpha2,
# plus an optional named extra); dot_product and lsh_forest have irregular
# signatures (no lengthscale / a per-episode random forest) and are left out
# of composites for now.
_COMPOSABLE_KERNELS: List[str] = ["rbf", "matern32", "cosine", "periodic", "rational_quadratic"]

# Kernels whose PSD guarantee only holds for scalar (1D) inputs — composites
# that include one of these must also cap the active kernel dimensionality
# to k=1 (see generate_gp_task / generate_gp_batch).
_SCALAR_ONLY_KERNELS = {"cosine", "periodic"}


def _parse_composite(name: str) -> Optional[tuple]:
    """Split "A+B" / "A*B" into (name_a, op, name_b), or None if not composite."""
    for op in ("+", "*"):
        if op in name:
            a, _, b = name.partition(op)
            if a in _COMPOSABLE_KERNELS and b in _COMPOSABLE_KERNELS:
                return a, op, b
    return None


def _kernel_needs_scalar_input(kernel_name: str) -> bool:
    """True if this kernel (or either half of a composite) requires k=1 input dims."""
    composite = _parse_composite(kernel_name)
    if composite is not None:
        name_a, _, name_b = composite
        return name_a in _SCALAR_ONLY_KERNELS or name_b in _SCALAR_ONLY_KERNELS
    return kernel_name in _SCALAR_ONLY_KERNELS


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
    **_,
) -> Tensor:
    """Evaluate a registered "A+B" / "A*B" composite kernel.

    Component A uses (l, alpha2, period, rq_alpha); component B uses the
    "_b"-suffixed counterparts — independent hyperparameters per component,
    sampled once per episode by the caller (same convention as period/rq_alpha
    for the simple kernels).
    """
    name_a, op, name_b = _parse_composite(kernel_name)
    kwargs_a: Dict = dict(l=l, alpha2=alpha2)
    if name_a == "periodic":
        kwargs_a["period"] = period
    if name_a == "rational_quadratic":
        kwargs_a["rq_alpha"] = rq_alpha
    kwargs_b: Dict = dict(l=l_b, alpha2=alpha2_b)
    if name_b == "periodic":
        kwargs_b["period"] = period_b
    if name_b == "rational_quadratic":
        kwargs_b["rq_alpha"] = rq_alpha_b

    K_a = KERNEL_REGISTRY[name_a](X1, X2, **kwargs_a)
    K_b = KERNEL_REGISTRY[name_b](X1, X2, **kwargs_b)
    return K_a + K_b if op == "+" else K_a * K_b


COMPOSITE_KERNELS: List[str] = []
for _name_a, _name_b in itertools.combinations(_COMPOSABLE_KERNELS, 2):
    for _op in ("+", "*"):
        _combo_name = f"{_name_a}{_op}{_name_b}"
        KERNEL_REGISTRY[_combo_name] = functools.partial(_composite_kernel, kernel_name=_combo_name)
        COMPOSITE_KERNELS.append(_combo_name)
del _name_a, _name_b, _op, _combo_name

ALL_KERNELS: List[str] = list(KERNEL_REGISTRY.keys())


def build_kernel_fn(
    kernel_name: str,
    l: float,
    alpha2: float,
    *,
    period: Optional[float] = None,
    rq_alpha: Optional[float] = None,
    lsh_W: Optional[Tensor] = None,
    lsh_b: Optional[Tensor] = None,
    l_b: Optional[float] = None,
    alpha2_b: Optional[float] = None,
    period_b: Optional[float] = None,
    rq_alpha_b: Optional[float] = None,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Return a kernel(X1, X2) -> K callable with hyperparameters baked in.

    lsh_W/lsh_b (lsh_forest only) must be one fixed realization sampled once
    per episode by the caller — see lsh_forest_kernel's docstring for why.
    l_b/alpha2_b/period_b/rq_alpha_b are the second component's hyperparameters
    for composite ("A+B" / "A*B") kernels — see _composite_kernel.
    """
    fn = KERNEL_REGISTRY[kernel_name]
    kwargs: Dict = dict(l=l, alpha2=alpha2)
    if period is not None:
        kwargs["period"] = period
    if rq_alpha is not None:
        kwargs["rq_alpha"] = rq_alpha
    if lsh_W is not None:
        kwargs["lsh_W"] = lsh_W
    if lsh_b is not None:
        kwargs["lsh_b"] = lsh_b
    if l_b is not None:
        kwargs["l_b"] = l_b
    if alpha2_b is not None:
        kwargs["alpha2_b"] = alpha2_b
    if period_b is not None:
        kwargs["period_b"] = period_b
    if rq_alpha_b is not None:
        kwargs["rq_alpha_b"] = rq_alpha_b
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


def _length_scale_range(cfg, k: int, P: int) -> tuple[float, float]:
    """(l_min, l_max), optionally scaled by sqrt(k) and/or P-density — see _sample_length_scale."""
    l_min = float(cfg.data.l_min)
    l_max = float(cfg.data.l_max)
    if getattr(cfg.data, "l_scale_by_sqrt_k", False):
        scale = math.sqrt(max(k, 1))
        l_min *= scale
        l_max *= scale
    if getattr(cfg.data, "l_scale_by_P", False):
        P_ref = float(getattr(cfg.data, "l_P_ref", 32.0))
        density_scale = (P_ref / max(P, 1)) ** (1.0 / max(k, 1))
        l_min *= density_scale
        l_max *= density_scale
    return l_min, l_max


def _sample_length_scale(cfg, k: int, P: int) -> float:
    """Sample the GP length-scale l, honouring three optional cfg flags.

    l_scale_by_sqrt_k: multiplies [l_min, l_max] by sqrt(k) before sampling.
        Squared distance sum_{i=1}^k (x1_i - x2_i)^2 grows ~linearly with k
        (active kernel dims), so without this correction, k=1 and k=4 tasks
        sample from very different effective-correlation regimes even though
        l is drawn from the same range — this is what produces a bimodal
        (collapsed-to-0 vs. spread-out) mixture in R_star.
    l_scale_by_P: multiplies [l_min, l_max] by (l_P_ref / P)^(1/k) before
        sampling. Train/test points are drawn i.i.d. from the same fixed
        domain, so larger P packs training points more densely — any test
        point ends up with a training neighbour within one length-scale,
        and GP conditioning shrinks R_star towards 0 regardless of the
        kernel hyperparameters. Shrinking l with P keeps the *local*
        neighbour count (and hence how much conditioning explains away)
        roughly constant across the whole P range, so R_star stays
        meaningful even at large, realistic P (e.g. 32-512).
    l_log_uniform: samples l ~ LogUniform instead of Uniform. RBF correlation
        decays as exp(-k / l^2), so a linear-uniform l barely visits the
        high-correlation regime; log-uniform spreads mass evenly across
        decades of l and gives a much more uniform R_star.
    """
    l_min, l_max = _length_scale_range(cfg, k, P)
    if getattr(cfg.data, "l_log_uniform", False):
        return math.exp(random.uniform(math.log(l_min), math.log(l_max)))
    return random.uniform(l_min, l_max)


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

    If cfg.seed is set (same field train.py uses via `torch.manual_seed(cfg.seed)`),
    it seeds python/numpy/torch RNGs, making the kernel/hyperparameter choice
    (kernel_name, P, N, l, alpha2, nugget, ...), the feature warp, and y
    sampling all reproducible. Calling this repeatedly with the same cfg.seed
    restarts every RNG at the same point every time.

    Keys returned:
        x_norm_train          : (P, d_features)  normalised train features (full)
        y_train               : (P,)             train targets
        x_norm_test           : (N, d_features)  normalised test features (full)
        y_test                : (N,)             test targets
        R_star                : (N, N)           ground-truth test correlation matrix
                                                  (posterior or prior, per cfg.data.oracle_mode)
        mu_star               : (N,)             mean at test points (posterior mean, or 0 for prior)
        sigma_star            : (N,)             marginal std at test points (posterior or prior)
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
    seed = getattr(cfg, "seed", None)
    if seed is not None:
        _seed_everything(seed)

    d = cfg.data.d_features

    # 1. Choose kernel and active columns
    kernel_name = _resolve_kernel_name(cfg)

    # cosine and periodic (and any composite containing one of them) are PSD
    # only for scalar (1D) inputs; cap to k=1
    if _kernel_needs_scalar_input(kernel_name):
        kernel_cols = [random.randint(0, d - 1)]
    else:
        kernel_cols = _sample_kernel_cols(d, cfg)

    # 2. Sample dataset sizes (needed before l — see l_scale_by_P)
    P = random.randint(cfg.data.P_min, cfg.data.P_max)
    N = random.randint(cfg.data.N_min, cfg.data.N_max)

    # 3. Sample shared GP hyperparameters
    l = _sample_length_scale(cfg, len(kernel_cols), P)
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

    # 4. Extra hyperparameters for specific kernels
    period: Optional[float] = None
    rq_alpha: Optional[float] = None
    l_b: Optional[float] = None
    alpha2_b: Optional[float] = None
    period_b: Optional[float] = None
    rq_alpha_b: Optional[float] = None

    composite = _parse_composite(kernel_name)
    component_a = composite[0] if composite is not None else kernel_name
    component_b = composite[2] if composite is not None else None

    p_min = float(getattr(cfg.data, "period_min", 0.5))
    p_max = float(getattr(cfg.data, "period_max", 3.0))
    rq_min = float(getattr(cfg.data, "rq_alpha_min", 0.1))
    rq_max = float(getattr(cfg.data, "rq_alpha_max", 5.0))

    if component_a == "periodic":
        period = random.uniform(p_min, p_max)
    if component_a == "rational_quadratic":
        rq_alpha = random.uniform(rq_min, rq_max)

    if composite is not None:
        # Component B: independent length-scale/variance draw, same ranges as A.
        l_b = _sample_length_scale(cfg, len(kernel_cols), P)
        alpha2_b = random.uniform(a2_min, a2_max)
        if component_b == "periodic":
            period_b = random.uniform(p_min, p_max)
        if component_b == "rational_quadratic":
            rq_alpha_b = random.uniform(rq_min, rq_max)

    lsh_W: Optional[Tensor] = None
    lsh_b: Optional[Tensor] = None
    if kernel_name == "lsh_forest":
        num_trees = int(getattr(cfg.data, "lsh_num_trees", 20))
        depth = int(getattr(cfg.data, "lsh_depth", 4))
        # One fixed hyperplane forest for the whole episode — K_ff/K_sf/K_ss
        # (computed by three separate kernel_fn calls below) must agree on it.
        lsh_W = torch.randn(len(kernel_cols), num_trees, depth)
        lsh_b = torch.randn(num_trees, depth)

    # 5. Sample features x ~ N(0, 1)^d, warp marginals (TabICLv2-style feature
    # heterogeneity), normalise over all P+N instances
    x_raw = torch.randn(P + N, d)
    x_raw = tabiclv2_warp_features(x_raw)
    mu_x = x_raw.mean(0)
    std_x = x_raw.std(0).clamp(min=1e-8)
    x_norm = (x_raw - mu_x) / std_x                    # full (P+N, d_features)
    x_k = x_norm[:, kernel_cols]                        # kernel sub-matrix (P+N, k)

    # 6. Build kernel function and sample y ~ GP(0, K + nugget·I) jointly
    kernel_fn = build_kernel_fn(
        kernel_name, l, alpha2, period=period, rq_alpha=rq_alpha, lsh_W=lsh_W, lsh_b=lsh_b,
        l_b=l_b, alpha2_b=alpha2_b, period_b=period_b, rq_alpha_b=rq_alpha_b,
    )
    K_all = kernel_fn(x_k, x_k) + nugget * torch.eye(P + N)

    y_all = _safe_cholesky(K_all, max_attempts=12) @ torch.randn(P + N)

    # 7. Split into train / test (full features returned to model)
    x_norm_train = x_norm[:P]
    y_train = y_all[:P]
    x_norm_test = x_norm[P:]
    y_test = y_all[P:]

    # 8. Compute R_star at test points, from either the GP posterior or the
    # GP prior, per cfg.data.oracle_mode (default "posterior" — unchanged
    # behaviour).
    # posterior: Sigma_star = K_ss + nugget·I − K_st K_ff⁻¹ K_ts accounts for
    #   what the training context already explains — off-diagonal entries
    #   shrink relative to the prior, giving the oracle for residual
    #   dependence after conditioning on training data.
    # prior: Sigma_star = K_ss + nugget·I, ignoring the training context —
    #   the oracle is the raw kernel structure among test points.
    x_k_train = x_k[:P]
    x_k_test = x_k[P:]
    oracle_mode = getattr(cfg.data, "oracle_mode", "posterior")
    if oracle_mode == "prior":
        # z_train's LOO PIT still needs L_ff/alpha from K_ff regardless of
        # which oracle is used for the test-side R_star/mu_star/sigma_star.
        K_ff = kernel_fn(x_k_train, x_k_train) + nugget * torch.eye(P, device=x_k_train.device)
        L_ff = _safe_cholesky(K_ff, max_attempts=12)
        alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L_ff).squeeze(-1)
        Sigma_star = kernel_fn(x_k_test, x_k_test) + nugget * torch.eye(N, device=x_k_test.device)
        mu_star = torch.zeros(N, device=x_k_test.device)
    elif oracle_mode == "posterior":
        mu_star, Sigma_star, L_ff, alpha = gp_posterior(
            x_k_train, y_train, x_k_test, kernel_fn, nugget, return_factors=True
        )
    else:
        raise ValueError(f"Unknown data.oracle_mode '{oracle_mode}'; expected 'prior' or 'posterior'.")
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
        "l_b": torch.tensor(l_b if l_b is not None else 0.0),
        "alpha2_b": torch.tensor(alpha2_b if alpha2_b is not None else 0.0),
        "period_b": torch.tensor(period_b if period_b is not None else 0.0),
        "rq_alpha_b": torch.tensor(rq_alpha_b if rq_alpha_b is not None else 0.0),
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


def _batched_lsh_leaf_ids(x_k: Tensor, W: Tensor, b: Tensor) -> Tensor:
    """Batched version of _lsh_leaf_ids.

    x_k: (B, T, k), W: (B, k, num_trees, depth), b: (B, num_trees, depth)
    -> (B, T, num_trees) int64 leaf id per (episode, point, tree).
    """
    depth = W.shape[-1]
    bits = torch.einsum("btk,bksd->btsd", x_k, W) + b.unsqueeze(1) > 0  # (B, T, num_trees, depth)
    weights = (2 ** torch.arange(depth, device=x_k.device)).to(torch.int64)
    return (bits.long() * weights).sum(-1)  # (B, T, num_trees)


def _batched_single_kernel_matrix(
    kernel_name: str,
    x_k: Tensor,                        # (B, T, k)
    l: Tensor,                          # (B,)
    alpha2: Tensor,                     # (B,)
    period: Optional[Tensor] = None,    # (B,) — periodic kernel only
    rq_alpha: Optional[Tensor] = None,  # (B,) — rational_quadratic only
    lsh_W: Optional[Tensor] = None,     # (B, k, num_trees, depth) — lsh_forest only
    lsh_b: Optional[Tensor] = None,     # (B, num_trees, depth) — lsh_forest only
) -> Tensor:
    """Build B matrices for one non-composite kernel (no nugget on the diagonal)."""
    B, T, _ = x_k.shape
    l3     = l.view(B, 1, 1)
    alpha3 = alpha2.view(B, 1, 1)

    if kernel_name == "dot_product":
        return x_k @ x_k.permute(0, 2, 1)  # (B, T, T)

    if kernel_name == "lsh_forest":
        ids = _batched_lsh_leaf_ids(x_k, lsh_W, lsh_b)         # (B, T, num_trees)
        matches = ids.unsqueeze(2) == ids.unsqueeze(1)          # (B, T, T, num_trees)
        return alpha3 * matches.float().mean(dim=-1)

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

    raise ValueError(f"Unknown kernel '{kernel_name}' in _batched_single_kernel_matrix.")


def _batched_kernel_matrix(
    kernel_name: str,
    x_k: Tensor,                        # (B, T, k)
    l: Tensor,                          # (B,)
    alpha2: Tensor,                     # (B,)
    period: Optional[Tensor] = None,    # (B,) — periodic kernel/component only
    rq_alpha: Optional[Tensor] = None,  # (B,) — rational_quadratic kernel/component only
    lsh_W: Optional[Tensor] = None,     # (B, k, num_trees, depth) — lsh_forest only
    lsh_b: Optional[Tensor] = None,     # (B, num_trees, depth) — lsh_forest only
    l_b: Optional[Tensor] = None,       # (B,) — composite kernels only, component B
    alpha2_b: Optional[Tensor] = None,  # (B,) — composite kernels only, component B
    period_b: Optional[Tensor] = None,  # (B,) — composite kernels only, component B
    rq_alpha_b: Optional[Tensor] = None,  # (B,) — composite kernels only, component B
) -> Tensor:
    """Build B kernel matrices for T points in one vectorised call.

    Returns (B, T, T) WITHOUT nugget on the diagonal.
    The caller is responsible for adding nugget * I.
    """
    composite = _parse_composite(kernel_name)
    if composite is None:
        return _batched_single_kernel_matrix(
            kernel_name, x_k, l, alpha2, period=period, rq_alpha=rq_alpha, lsh_W=lsh_W, lsh_b=lsh_b
        )

    name_a, op, name_b = composite
    K_a = _batched_single_kernel_matrix(name_a, x_k, l, alpha2, period=period, rq_alpha=rq_alpha)
    K_b = _batched_single_kernel_matrix(name_b, x_k, l_b, alpha2_b, period=period_b, rq_alpha=rq_alpha_b)
    return K_a + K_b if op == "+" else K_a * K_b


def tabiclv2_warp_features(x: Tensor, seed: Optional[int] = None) -> Tensor:
    """Warp each feature column with one of 8 random marginal transforms.

    Simulates the extreme marginal heterogeneity of real tabular data
    (TabICLv2): heavy tails, power laws, ordinal steps, bimodal mixtures,
    periodicity, and Cauchy outliers, applied on top of a Standard Normal
    baseline. Intended to run before any per-episode mean/std normalisation,
    so downstream kernel/covariance code keeps operating on calibrated,
    unit-scale features while the model still sees the warped shape.

    Args:
        x: (B, T, d) or (T, d) tensor of Standard Normal features.
        seed: if given, seeds python/numpy/torch RNGs so the warp choice and
            all sampled transform parameters are reproducible. Leave None
            when called from generate_gp_task/generate_gp_batch — those
            already seed globally before calling this, so reseeding here
            would just restart the same streams.

    Returns:
        Tensor of the same shape as `x`, with each (episode, column) warped
        independently by a randomly chosen transform.
    """
    if seed is not None:
        _seed_everything(seed)

    added_batch_dim = x.dim() == 2
    if added_batch_dim:
        x = x.unsqueeze(0)

    B, T, d = x.shape
    warped_x = x.clone()
    choices = torch.randint(0, 8, (B, d), device=x.device)

    for b in range(B):
        for col in range(d):
            c = choices[b, col].item()
            col_data = warped_x[b, :, col]

            if c == 0:  # Identity — Standard Normal baseline
                continue
            elif c == 1:  # Signed-square — mild heavy tails
                warped_x[b, :, col] = torch.sign(col_data) * (col_data ** 2)
            elif c == 2:  # Cube — Student-T-like heavy tails
                warped_x[b, :, col] = col_data ** 3
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

    if added_batch_dim:
        warped_x = warped_x.squeeze(0)
    return warped_x


def generate_gp_batch(cfg, B: int, device: str = "cpu") -> List[Dict[str, Tensor]]:
    """Generate B GP episodes in a single vectorised call.

    All B episodes share one kernel type and one (P, N) size (both sampled
    once per call) but have independent hyperparameters and feature draws.
    This removes the Python-loop overhead of B separate generate_gp_task calls
    and enables GPU or CPU-SIMD acceleration for the linear-algebra steps.

    The returned dicts have the same schema as the episodes saved by
    generate_pit_dataset.py (no kernel metadata).

    If cfg.seed is set, it seeds python/numpy/torch RNGs, making the
    kernel/shape choice (kernel_name, P, N, k), hyperparameters (l, nugget,
    alpha2, ...), feature sampling/warp, and y sampling all reproducible.
    Note that calling this repeatedly with the same cfg.seed (e.g. once per
    shard in generate_pit_dataset.py) restarts every RNG at the same point
    every call — vary cfg.seed per call (e.g. `cfg.seed + shard_idx`) if you
    need distinct shards.

    Args:
        cfg    : Hydra config (same as generate_gp_task).
        B      : number of episodes to generate in this batch.
        device : torch device string ("cpu" or "cuda").

    Returns:
        list of B episode dicts ready for torch.save.
    """
    import warnings

    seed = getattr(cfg, "seed", None)
    if seed is not None:
        _seed_everything(seed)

    d          = cfg.data.d_features
    nugget_min = float(getattr(cfg.data, "nugget_min", 0.1))
    nugget_max = float(getattr(cfg.data, "nugget_max", 1.0))

    # --- Shared settings for this batch ---
    kernel_name = _resolve_kernel_name(cfg)
    P = random.randint(cfg.data.P_min, cfg.data.P_max)
    N = random.randint(cfg.data.N_min, cfg.data.N_max)
    T = P + N

    composite = _parse_composite(kernel_name)
    component_a = composite[0] if composite is not None else kernel_name
    component_b = composite[2] if composite is not None else None

    if _kernel_needs_scalar_input(kernel_name):
        k = 1
    elif kernel_name in ("dot_product", "lsh_forest"):
        # Every hyperplane split can draw on all d columns (no lengthscale to
        # dilute with irrelevant dims, unlike rbf/matern32/rational_quadratic).
        k = d
    else:
        k = random.randint(
            int(getattr(cfg.data, "d_kernel_min", 1)),
            min(int(getattr(cfg.data, "d_kernel_max", 4)), d),
        )

    # --- Per-episode hyperparameters ---
    l_min, l_max = _length_scale_range(cfg, k, P)
    if getattr(cfg.data, "l_log_uniform", False):
        l = torch.exp(
            torch.rand(B, device=device) * (math.log(l_max) - math.log(l_min)) + math.log(l_min)
        )
    else:
        l = torch.rand(B, device=device) * (l_max - l_min) + l_min
    nugget = torch.rand(B, device=device) * (nugget_max           - nugget_min)           + nugget_min
    alpha2 = (
        torch.zeros(B, device=device)
        if kernel_name == "dot_product"
        else torch.rand(B, device=device) * (cfg.data.alpha2_max - cfg.data.alpha2_min) + cfg.data.alpha2_min
    )

    p_min  = float(getattr(cfg.data, "period_min",   0.5))
    p_max  = float(getattr(cfg.data, "period_max",   3.0))
    rq_min = float(getattr(cfg.data, "rq_alpha_min", 0.1))
    rq_max = float(getattr(cfg.data, "rq_alpha_max", 5.0))

    period = rq_alpha = None
    if component_a == "periodic":
        period = torch.rand(B, device=device) * (p_max - p_min) + p_min
    if component_a == "rational_quadratic":
        rq_alpha = torch.rand(B, device=device) * (rq_max - rq_min) + rq_min

    # --- Component B (composite kernels only): independent hyperparameters,
    # same ranges as component A.
    l_b = alpha2_b = period_b = rq_alpha_b = None
    if composite is not None:
        if getattr(cfg.data, "l_log_uniform", False):
            l_b = torch.exp(
                torch.rand(B, device=device) * (math.log(l_max) - math.log(l_min)) + math.log(l_min)
            )
        else:
            l_b = torch.rand(B, device=device) * (l_max - l_min) + l_min
        alpha2_b = torch.rand(B, device=device) * (cfg.data.alpha2_max - cfg.data.alpha2_min) + cfg.data.alpha2_min
        if component_b == "periodic":
            period_b = torch.rand(B, device=device) * (p_max - p_min) + p_min
        if component_b == "rational_quadratic":
            rq_alpha_b = torch.rand(B, device=device) * (rq_max - rq_min) + rq_min

    lsh_W = lsh_b = None
    if kernel_name == "lsh_forest":
        num_trees = int(getattr(cfg.data, "lsh_num_trees", 20))
        depth = int(getattr(cfg.data, "lsh_depth", 4))
        # One fixed hyperplane forest per episode, shared by every submatrix
        # of K_all — see lsh_forest_kernel's docstring for why this matters.
        lsh_W = torch.randn(B, k, num_trees, depth, device=device)
        lsh_b = torch.randn(B, num_trees, depth, device=device)

    # --- Features (B, T, d) ~ N(0, 1), warped, normalised per episode ---
    x_raw  = torch.randn(B, T, d, device=device)
    x_raw  = tabiclv2_warp_features(x_raw)
    x_norm = (x_raw - x_raw.mean(1, keepdim=True)) / x_raw.std(1, keepdim=True).clamp(min=1e-8)

    # --- Select kernel columns ---
    if kernel_name == "dot_product":
        x_k = x_norm                                                                # (B, T, d)
    elif _kernel_needs_scalar_input(kernel_name):
        col = torch.randint(0, d, (B,), device=device)                             # (B,)
        x_k = x_norm.gather(2, col.view(B, 1, 1).expand(B, T, 1))                 # (B, T, 1)
    else:
        # Vectorised random column selection: argsort of uniform noise picks k cols
        col_idx = torch.rand(B, d, device=device).argsort(dim=1)[:, :k]           # (B, k)
        x_k = x_norm.gather(2, col_idx.unsqueeze(1).expand(B, T, k))              # (B, T, k)

    # --- Build K_all (B, T, T) and sample y jointly ---
    K_all = _batched_kernel_matrix(
        kernel_name, x_k, l, alpha2, period, rq_alpha, lsh_W, lsh_b,
        l_b=l_b, alpha2_b=alpha2_b, period_b=period_b, rq_alpha_b=rq_alpha_b,
    )
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

    # --- GP posterior/prior (batched) ---
    # z_train's LOO PIT always needs L_ff/alpha from K_ff, regardless of which
    # oracle drives the test-side R_star/mu_star/sigma_star below.
    L_ff  = _batched_cholesky(K_ff)
    alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L_ff).squeeze(-1)          # (B, P)

    oracle_mode = getattr(cfg.data, "oracle_mode", "posterior")
    if oracle_mode == "prior":
        # Prior oracle: ignore training conditioning — R_star reflects the raw
        # kernel structure among test points; mu_star is the GP prior mean (0).
        mu_star    = torch.zeros(B, N, device=device)
        Sigma_star = K_ss
    elif oracle_mode == "posterior":
        mu_star    = (K_sf @ alpha.unsqueeze(-1)).squeeze(-1)                             # (B, N)
        V          = torch.linalg.solve_triangular(L_ff, K_sf.permute(0, 2, 1), upper=False)  # (B, P, N)
        Sigma_star = K_ss - V.permute(0, 2, 1) @ V                                     # (B, N, N)
    else:
        raise ValueError(f"Unknown data.oracle_mode '{oracle_mode}'; expected 'prior' or 'posterior'.")
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
