"""Inference API for the copula model: marginals, PIT, correlation and sampling.

Features must be standardized first with normalize_features (jointly over
train and test). Targets are z-scored internally before reaching TabICL.

Marginal backends, both returning (quantile_grid (n_query, Q), probs):
    get_marginal_quantiles: TabICL (999-level grid, or icdf at given probs).
    get_marginal_quantiles_pfn4bo: PFN4BO bar distribution (x mapped to
        [0, 1] with the Gaussian CDF, y Yeo-Johnson transformed).
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.stats import norm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_PFNS4BO_ROOT = os.path.join(_REPO_ROOT, "pfns4bo_upstream")

from copula_inter.model import CopulaTabICL, build_copula_transformer, low_rank_correlation  # noqa: E402
from copula_inter.pit import load_tabicl, normalize_targets, run_pit  # noqa: E402

if TYPE_CHECKING:
    from tabicl._model.tabicl import TabICL

__all__ = [
    "normalize_features",
    "load_tabicl_marginal",
    "get_marginal_quantiles",
    "loo_pit",
    "load_copula_model",
    "get_test_correlation",
    "load_pfn4bo",
    "get_marginal_quantiles_pfn4bo",
    "sample_trajectories",
]


def normalize_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score features with the joint train+test mean and std per column (the training convention).

    Args:
        X_train: (P, d) raw features.
        X_test: (N, d) raw features.

    Returns:
        (X_train_norm, X_test_norm).
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    X_all = np.concatenate([X_train, X_test], axis=0)
    mean = X_all.mean(axis=0, keepdims=True)
    std_raw = X_all.std(axis=0, ddof=1, keepdims=True)
    is_constant = std_raw < 1e-6
    std = np.where(is_constant, 1.0, std_raw)
    norm_tr = np.where(is_constant, 0.0, (X_train - mean) / std)
    norm_te = np.where(is_constant, 0.0, (X_test - mean) / std)
    return norm_tr, norm_te


def load_tabicl_marginal(ckpt_name: str, device: str) -> torch.nn.Module:
    """Load a frozen TabICL regressor (pit.load_tabicl)."""
    return load_tabicl(ckpt_name, device)


@torch.no_grad()
def get_marginal_quantiles(
    tabicl: TabICL,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    probs: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """TabICL's predictive quantiles at X_query.

    Args:
        tabicl: TabICL regressor.
        X_context: (n_ctx, d).
        y_context: (n_ctx,) raw targets (z-scored internally).
        X_query: (n_q, d).
        probs: optional (Q,) levels; None uses TabICL's 999-level grid,
            otherwise QuantileDistribution.icdf at these levels.

    Returns:
        quantile_grid (n_q, Q) in raw y units and the probs used.
    """
    device = next(tabicl.parameters()).device
    dtype = next(tabicl.parameters()).dtype

    y_ctx_raw_t = torch.as_tensor(np.asarray(y_context), dtype=dtype, device=device)
    y_ctx_t, _, y_mean, y_std = normalize_targets(y_ctx_raw_t)

    X_ctx_t = torch.as_tensor(np.asarray(X_context), dtype=dtype, device=device)
    X_qry_t = torch.as_tensor(np.asarray(X_query), dtype=dtype, device=device)

    X_full = torch.cat([X_ctx_t, X_qry_t], dim=0).unsqueeze(0)  # (1, T, d_x)
    y_ctx_b = y_ctx_t.unsqueeze(0)  # (1, P)

    raw_quantiles = tabicl(X_full, y_ctx_b)  # (1, n_q, 999)
    dist = tabicl.quantile_dist(raw_quantiles)

    if probs is None:
        quantile_grid = dist.quantiles[0]
        probs_out = dist.alpha_levels
    else:
        probs_t = torch.as_tensor(np.asarray(probs), dtype=raw_quantiles.dtype, device=device)
        quantile_grid = dist.icdf(probs_t)[0]  # (n_q, len(probs))
        probs_out = probs_t

    # Upcast float16 quantiles before un-scaling (large y values overflow float16).
    quantile_grid = y_mean.double() + y_std.double() * quantile_grid.double()
    return quantile_grid.cpu().numpy(), probs_out.cpu().numpy()


def loo_pit(
    tabicl: TabICL,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_folds: int = 10,
    eps: float = 1e-6,
) -> np.ndarray:
    """K-fold PIT of the training set through TabICL (k_folds=len(X_train) for true LOO).

    Args:
        tabicl: TabICL regressor.
        X_train: (P, d).
        y_train: (P,) raw targets.
        k_folds: number of folds.
        eps: probit clamp.

    Returns:
        (P,) Gaussianized residuals.
    """
    device = next(tabicl.parameters()).device
    dtype = next(tabicl.parameters()).dtype

    X_t = torch.as_tensor(np.asarray(X_train), dtype=dtype, device=device)
    y_train_raw_t = torch.as_tensor(np.asarray(y_train), dtype=dtype, device=device)
    y_train_scaled, _, _, _ = normalize_targets(y_train_raw_t)
    y_t = y_train_scaled.unsqueeze(-1)  # (P, 1)

    out = run_pit(
        tabicl,
        X_t,
        y_t,
        X_t[:1],
        y_t[:1],
        k_folds=k_folds,
        eps=eps,
        Y_train_raw=y_train_raw_t.unsqueeze(-1),
    )
    return out["z_train"].squeeze(-1).cpu().numpy()


def _resolve_copula_checkpoint(ckpt_path: str) -> str:
    """Resolve a checkpoint file; for a directory, the highest step_<n>[_final].pt (final preferred on ties)."""
    if not os.path.isdir(ckpt_path):
        return ckpt_path

    candidates = []
    for name in os.listdir(ckpt_path):
        match = re.fullmatch(r"step_(\d+)(?:_final)?\.pt", name)
        path = os.path.join(ckpt_path, name)
        if match and os.path.isfile(path):
            candidates.append((int(match.group(1)), name.endswith("_final.pt"), path))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint files named step_<number>.pt or step_<number>_final.pt found in directory '{ckpt_path}'."
        )

    _, _, resolved = max(candidates)
    print(f"Resolved checkpoint directory '{ckpt_path}' to '{resolved}'.")
    return resolved


def load_copula_model(
    ckpt_path: str,
    config_path: Optional[str] = None,
    device: str = "cpu",
) -> tuple[CopulaTabICL, DictConfig]:
    """Load a CopulaTabICL checkpoint.

    Args:
        ckpt_path: checkpoint file, or a directory of step_<n>.pt files.
        config_path: config to use when the checkpoint has no saved cfg.
        device: torch device.

    Returns:
        (model in eval mode, cfg).
    """
    ckpt_path = _resolve_copula_checkpoint(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ckpt.get("cfg")
    if cfg is None:
        if config_path is None:
            raise ValueError(f"Checkpoint '{ckpt_path}' has no saved 'cfg' and no config_path was given.")
        cfg = OmegaConf.load(config_path)
    elif isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    model = build_copula_transformer(cfg).to(device)
    state = ckpt.get("model_state", ckpt.get("state_dict"))
    raw = getattr(model, "_orig_mod", model)
    raw.load_state_dict(state)
    model.eval()
    print(f"Loaded CopulaTabICL checkpoint '{ckpt_path}' (step {ckpt.get('step')}) on {device}.")
    return model, cfg


@torch.no_grad()
def get_test_correlation(
    copula_model: CopulaTabICL,
    X_train: np.ndarray,
    Z_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """The copula model's (N, N) test correlation, symmetrized with a unit diagonal.

    Args:
        copula_model: loaded CopulaTabICL.
        X_train: (P, d).
        Z_train: (P,) Gaussianized residuals.
        X_test: (N, d).
    """
    device = next(copula_model.parameters()).device
    dtype = next(copula_model.parameters()).dtype

    x_train_t = torch.as_tensor(np.asarray(X_train), dtype=dtype, device=device).unsqueeze(0)
    x_test_t = torch.as_tensor(np.asarray(X_test), dtype=dtype, device=device).unsqueeze(0)
    z_train_t = torch.as_tensor(np.asarray(Z_train), dtype=dtype, device=device).unsqueeze(0)

    batch = {"x_train": x_train_t, "x_test": x_test_t, "z_train": z_train_t}
    out = copula_model(batch)
    Sigma = low_rank_correlation(out["W"], out["s"])  # (1, N, N)

    R = Sigma[0].cpu().numpy()
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


def _patch_pfns4bo_torch_compat() -> None:
    """Re-export typing.Optional from torch.nn.modules.transformer, which pfns4bo imports from there."""
    import typing

    import torch.nn.modules.transformer as _t

    if not hasattr(_t, "Optional"):
        setattr(_t, "Optional", typing.Optional)


def load_pfn4bo(model_name: str = "hebo_plus_model", device: str = "cpu") -> torch.nn.Module:
    """Load a pretrained PFN4BO model from pfns4bo_upstream by attribute name (downloads on first use)."""
    _patch_pfns4bo_torch_compat()
    import pfns4bo

    model_path = getattr(pfns4bo, model_name)
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def _vectorized_bar_icdf(criterion: Any, logits: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """BarDistribution.icdf at many probability levels at once.

    Args:
        criterion: BarDistribution (uses .borders and .num_bars).
        logits: (..., num_bars).
        probs: (Q,).

    Returns:
        (..., Q) quantiles.
    """
    p = torch.softmax(logits, dim=-1)  # (..., B)
    cumprobs = torch.cumsum(p, dim=-1)  # (..., B)
    prefix_shape = cumprobs.shape[:-1]

    probs_b = probs.view(*([1] * len(prefix_shape)), -1).expand(*prefix_shape, -1).contiguous()
    idx = torch.searchsorted(cumprobs.contiguous(), probs_b).clamp(0, criterion.num_bars - 1)

    zeros = torch.zeros(*prefix_shape, 1, device=logits.device, dtype=logits.dtype)
    cumprobs_padded = torch.cat([zeros, cumprobs], dim=-1)
    left_cum = cumprobs_padded.gather(-1, idx)
    rest_prob = probs_b - left_cum

    left_border = criterion.borders[idx]
    right_border = criterion.borders[idx + 1]
    bucket_p = p.gather(-1, idx).clamp(min=1e-12)

    return left_border + (right_border - left_border) * rest_prob / bucket_p


def _yeo_johnson_valid_domain(lam: float) -> tuple[float, float]:
    """Interval of transformed values that the inverse Yeo-Johnson transform at lam accepts."""
    t_min = -1.0 / (lam - 2.0) if lam > 2.0 else -np.inf
    t_max = -1.0 / lam if lam < 0.0 else np.inf
    return t_min, t_max


@torch.no_grad()
def get_marginal_quantiles_pfn4bo(
    pfn4bo_model: torch.nn.Module,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    probs: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """PFN4BO's predictive quantiles at X_query.

    x is mapped to [0, 1]^d with the Gaussian CDF; y is Yeo-Johnson transformed
    (fit on y_context) and the quantiles are transformed back, clipped to the
    invertible domain and to a multiple of the context range.

    Args:
        pfn4bo_model: loaded PFN4BO model.
        X_context: (n_ctx, d).
        y_context: (n_ctx,).
        X_query: (n_q, d).
        probs: optional (Q,) levels (default: the 999-level grid).

    Returns:
        quantile_grid (n_q, Q) in original y units and the probs used.
    """
    from sklearn.preprocessing import PowerTransformer

    if probs is None:
        probs = np.linspace(0.0, 1.0, 999 + 2)[1:-1]
    probs = np.asarray(probs)

    device = next(pfn4bo_model.parameters()).device

    y_ctx = np.asarray(y_context, dtype=np.float64).reshape(-1, 1)
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    y_ctx_transformed = pt.fit_transform(y_ctx).reshape(-1)

    X_ctx = np.asarray(X_context, dtype=np.float64)
    X_qry = np.asarray(X_query, dtype=np.float64)
    x_ctx_t = torch.special.ndtr(torch.as_tensor(X_ctx, dtype=torch.float32))
    x_qry_t = torch.special.ndtr(torch.as_tensor(X_qry, dtype=torch.float32))

    n_ctx = x_ctx_t.shape[0]
    x_full = torch.cat([x_ctx_t, x_qry_t], dim=0).to(device).unsqueeze(1)  # (T, 1, d_x)
    y_full = torch.as_tensor(y_ctx_transformed, dtype=torch.float32, device=device).unsqueeze(1)  # (P, 1)

    # The output covers only the query positions: (n_q, 1, num_bars).
    logits = pfn4bo_model((None, x_full, y_full), single_eval_pos=n_ctx)  # (n_q, 1, num_bars)
    logits_query = logits[:, 0, :]  # (n_q, num_bars)

    criterion = pfn4bo_model.criterion
    probs_t = torch.as_tensor(probs, dtype=logits_query.dtype, device=device)
    quantile_grid_transformed = _vectorized_bar_icdf(criterion, logits_query, probs_t)  # (n_q, Q)

    # Clip to the inverse transform's domain (outside it sklearn returns NaN).
    t_min, t_max = _yeo_johnson_valid_domain(float(pt.lambdas_[0]))
    margin = 1e-6
    flat = quantile_grid_transformed.cpu().numpy().reshape(-1, 1).astype(np.float64)
    flat = np.clip(flat, t_min + margin, t_max - margin)

    n_q = quantile_grid_transformed.shape[0]
    quantile_grid = pt.inverse_transform(flat).reshape(n_q, -1)

    # Clip to a multiple of the context range.
    y_ctx_min, y_ctx_max = float(y_ctx.min()), float(y_ctx.max())
    y_range = max(y_ctx_max - y_ctx_min, 1e-6)
    quantile_grid = np.clip(quantile_grid, y_ctx_min - 10 * y_range, y_ctx_max + 10 * y_range)

    return quantile_grid, probs


def sample_trajectories(
    quantile_grid: np.ndarray,
    probs: np.ndarray,
    R: np.ndarray,
    n_samples: int,
    eps_reg: float = 1e-6,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, int]:
    """Sample correlated trajectories: z = chol(R + eps_reg I) eps, y_i = F_i^{-1}(Phi(z_i)).

    Probabilities are clipped to [probs[0], probs[-1]].

    Args:
        quantile_grid: (n_test, Q), quantile_grid[i, j] = F_i^{-1}(probs[j]).
        probs: (Q,) levels.
        R: (n_test, n_test) correlation (np.eye for independence).
        n_samples: number of trajectories.
        eps_reg: diagonal jitter.
        rng: optional np.random.Generator.

    Returns:
        samples (n_samples, n_test) and the number of clipped probabilities.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_test = R.shape[0]
    R_safe = R + eps_reg * np.eye(n_test)
    # Add jitter, then renormalize to a unit diagonal.
    scale = np.sqrt(np.diag(R_safe))
    R_safe = R_safe / np.outer(scale, scale)
    L = np.linalg.cholesky(R_safe)

    eps = rng.standard_normal((n_samples, n_test))
    z = eps @ L.T  # (n_samples, n_test)
    p = norm.cdf(z)

    lo, hi = probs[0], probs[-1]
    n_clipped = int(np.sum((p < lo) | (p > hi)))
    p_clipped = np.clip(p, lo, hi)

    samples = np.empty_like(p_clipped)
    for i in range(n_test):
        samples[:, i] = np.interp(p_clipped[:, i], probs, quantile_grid[i])

    return samples, n_clipped
