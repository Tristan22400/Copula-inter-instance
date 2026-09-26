"""Spatial-correlation diagnostics: model correlation extraction, distance binning and decay-law fits."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gamma as gamma_fn
from scipy.special import kv as bessel_k

from eval.data.era5_io import safe_cholesky

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from copula_inter.model import CopulaTabICL
    from copula_inter.pit import TabICLLike
    from copula_inter.type_aliases import Device, HasDataConfig
    from tabicl._model.tabicl import TabICL

__all__ = [
    "compute_persistence_residuals",
    "compute_raw_temperature_observations",
    "get_ground_truth_observations",
    "empirical_spatial_correlation",
    "morans_i",
    "predict_copula_residual_field",
    "sample_copula_residual_fields",
    "pool_yspace_samples_and_correlate",
    "load_marginal_tabicl",
    "extract_model_dummy_context_correlation",
    "compute_context_z_train",
    "extract_model_context_correlation",
    "extract_model_true_z_train_correlation",
    "sample_simple_kernel_covariance",
    "build_synthetic_grid_task",
    "bin_correlation_by_distance",
    "pair_counts_by_distance",
    "seriate_by_correlation",
    "THEORY_LAWS",
    "THEORY_STYLE",
    "LITERATURE_L",
    "fit_theoretical_law",
]


def compute_persistence_residuals(field_all: np.ndarray) -> np.ndarray:
    """24 h persistence residuals E_t = Z_t - Z_{t-1} for daily snapshots field_all (n, H, W); returns (n - 1, H*W)."""
    n = field_all.shape[0]
    if n < 2:
        raise ValueError(f"Need >= 2 time snapshots to form 24h persistence residuals, got {n}.")
    flat = field_all.reshape(n, -1)
    return flat[1:] - flat[:-1]


def compute_raw_temperature_observations(field_all: np.ndarray) -> np.ndarray:
    """Daily temperatures (n, H*W), no differencing."""
    n = field_all.shape[0]
    return field_all.reshape(n, -1)


def get_ground_truth_observations(field_all: np.ndarray, target: str) -> np.ndarray:
    """Observation matrix (n, H*W) for target "raw" or "persistence"."""
    if target == "raw":
        return compute_raw_temperature_observations(field_all)
    if target == "residual":
        return compute_persistence_residuals(field_all)
    raise ValueError(f"Unknown target '{target}', choose from 'raw' or 'residual'.")


def empirical_spatial_correlation(data: dict, target: str = "raw") -> np.ndarray:
    """Pearson correlation matrix (D, D) of the target observations."""
    observations = get_ground_truth_observations(data["t2m"], target)
    return np.corrcoef(observations.T)


def morans_i(field: np.ndarray) -> float:
    """Global Moran's I with rook adjacency on an (H, W) grid (+1 smooth, 0 random, negative checkerboard)."""
    x = field - field.mean()
    cross_h = x[:, :-1] * x[:, 1:]
    cross_v = x[:-1, :] * x[1:, :]
    n_edges = cross_h.size + cross_v.size
    numerator = cross_h.sum() + cross_v.sum()
    denominator = (x**2).sum()
    return float(field.size * numerator / (n_edges * denominator))


def sample_copula_residual_fields(
    tabicl_marginal: TabICLLike | None,
    context_coords: np.ndarray,
    context_values: np.ndarray,
    coords_test: np.ndarray,
    R_context: np.ndarray,
    device: str,
    z_shared_batch: np.ndarray,
) -> np.ndarray:
    """K joint draws (K, D) from the copula model: y_k = F^{-1}(Phi(L z_shared_batch[k])), L = chol(R_context).

    One marginal forward pass is shared by the K draws. Uses a Gaussian
    (mean, std) marginal when tabicl_marginal is None.
    """
    from scipy.stats import norm

    L = safe_cholesky(R_context)
    z_copula = L @ np.asarray(z_shared_batch).T  # (D, K)
    u_copula = np.clip(norm.cdf(z_copula), 1e-6, 1.0 - 1e-6)  # (D, K)

    y_mean = context_values.mean()
    y_std = max(context_values.std(), 1e-8)

    if tabicl_marginal is None:
        return (y_mean + y_std * z_copula).T  # (K, D)

    import torch

    from copula_inter.pit import normalize_targets
    from inference.copula_inference import normalize_features

    x_train_norm, x_test_norm = normalize_features(context_coords, coords_test)

    x_full = np.concatenate([x_train_norm, x_test_norm], axis=0)
    x_batch = torch.as_tensor(x_full, dtype=torch.float32, device=device).unsqueeze(0)  # (1, P+N, p_x)
    context_values_t = torch.as_tensor(context_values, dtype=torch.float32, device=device)
    context_values_scaled_t, _, y_mean_t, y_std_t = normalize_targets(context_values_t)
    y_train_batch = context_values_scaled_t.unsqueeze(0)  # (1, P)
    with torch.no_grad():
        logits = tabicl_marginal(x_batch, y_train_batch)  # (1, N, Q) -- N test rows only
        n_test = coords_test.shape[0]
        dist = tabicl_marginal.quantile_dist(logits.reshape(n_test, -1))  # batch_shape=(N,)
        u_t = torch.as_tensor(u_copula, dtype=torch.float32, device=device)  # (N, K)
        y_pred_scaled = dist.icdf(u_t).double()  # (N, K)
    return (y_mean_t.double() + y_std_t.double() * y_pred_scaled).T.cpu().numpy()  # (K, N)


def pool_yspace_samples_and_correlate(samples_per_day: list) -> np.ndarray:
    """Correlation of all days' (K, D) samples pooled into one (n_days*K, D) matrix."""
    return np.corrcoef(np.concatenate(samples_per_day, axis=0).T)


def predict_copula_residual_field(
    tabicl_marginal: TabICLLike | None,
    context_coords: np.ndarray,
    context_values: np.ndarray,
    coords_test: np.ndarray,
    R_context: np.ndarray,
    device: str,
    z_shared: np.ndarray,
) -> np.ndarray:
    """One draw (D,) from the copula model (sample_copula_residual_fields with K=1)."""
    return sample_copula_residual_fields(
        tabicl_marginal,
        context_coords,
        context_values,
        coords_test,
        R_context,
        device,
        np.asarray(z_shared)[None, :],
    )[0]


def load_marginal_tabicl(cfg: DictConfig, device: str) -> TabICL | None:
    """Load the frozen TabICL marginal named by pit.resolve_pit_ckpt(cfg), or None (with a warning) if none or loading fails."""
    from copula_inter.pit import load_tabicl, resolve_pit_ckpt

    source = resolve_pit_ckpt(cfg)
    if source is None:
        print(
            "Warning: cfg.tabicl.pretrained=False and no pit_ckpt set — "
            "no usable marginal for PIT; context z_train will fall back "
            "to naive standardization."
        )
        return None

    try:
        return load_tabicl(source, device)
    except Exception as exc:  # noqa: BLE001
        print(
            f"Warning: failed to load marginal '{source}' ({exc}); "
            "context z_train will fall back to naive standardization."
        )
        return None


def _forward_correlation(
    model: CopulaTabICL, device: Device, x_train_norm: np.ndarray, z_train: np.ndarray, x_test_norm: np.ndarray
) -> np.ndarray:
    """Model forward (x_train, z_train, x_test) -> dense Sigma."""
    import torch

    from copula_inter.model import low_rank_correlation

    x_train_t = torch.as_tensor(x_train_norm, dtype=torch.float32, device=device).unsqueeze(0)
    x_test_t = torch.as_tensor(x_test_norm, dtype=torch.float32, device=device).unsqueeze(0)
    z_train_t = torch.as_tensor(z_train, dtype=torch.float32, device=device).unsqueeze(0)
    batch = {"x_train": x_train_t, "x_test": x_test_t, "z_train": z_train_t}

    with torch.no_grad():
        out = model(batch)
        Sigma = low_rank_correlation(out["W"], out["s"], jitter=1e-4)
    return Sigma[0].cpu().numpy()


def extract_model_dummy_context_correlation(model: CopulaTabICL, device: Device, coords_test: np.ndarray) -> np.ndarray:
    """Model correlation with a single dummy context row (x = 0, z = 0)."""
    coords_test = np.asarray(coords_test, dtype=np.float64)
    x_mean = coords_test.mean(axis=0, keepdims=True)
    x_std = coords_test.std(axis=0, keepdims=True).clip(min=1e-8)
    x_test_norm = (coords_test - x_mean) / x_std

    x_train_norm = np.zeros((1, coords_test.shape[1]), dtype=np.float64)
    z_train = np.zeros(1, dtype=np.float64)
    return _forward_correlation(model, device, x_train_norm, z_train, x_test_norm)


def compute_context_z_train(
    x_train_norm: np.ndarray,
    context_values: np.ndarray,
    tabicl_marginal: TabICLLike | None,
    device: Device,
    k_folds: int = 10,
) -> np.ndarray:
    """K-fold PIT z_train of a real context sample under tabicl_marginal (standardized values when it is None)."""
    if tabicl_marginal is None:
        y_mean = context_values.mean()
        y_std = max(context_values.std(), 1e-8)
        return (context_values - y_mean) / y_std

    import torch

    from copula_inter.pit import normalize_targets, run_pit

    X_train_t = torch.as_tensor(x_train_norm, dtype=torch.float32, device=device)
    context_values_t = torch.as_tensor(context_values, dtype=torch.float32, device=device)
    context_values_scaled_t, _, _, _ = normalize_targets(context_values_t)
    Y_train_t = context_values_scaled_t.unsqueeze(-1)  # (P, 1)
    pit_out = run_pit(
        tabicl_marginal,
        X_train_t,
        Y_train_t,
        X_train_t[:1],
        Y_train_t[:1],
        k_folds=k_folds,
        Y_train_raw=context_values_t.unsqueeze(-1),
    )
    return pit_out["z_train"].squeeze(-1).cpu().numpy()  # (P,)


def extract_model_context_correlation(
    model: CopulaTabICL,
    device: Device,
    tabicl_marginal: TabICLLike | None,
    context_coords: np.ndarray,
    context_values: np.ndarray,
    coords_test: np.ndarray,
    k_folds: int = 10,
) -> np.ndarray:
    """Model correlation at coords_test given a real context sample (z_train from compute_context_z_train)."""
    from inference.copula_inference import normalize_features

    x_train_norm, x_test_norm = normalize_features(context_coords, coords_test)
    z_train = compute_context_z_train(x_train_norm, context_values, tabicl_marginal, device, k_folds)
    return _forward_correlation(model, device, x_train_norm, z_train, x_test_norm)


def _exact_gp_loo_z_train(K_ff: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact zero-mean GP LOO z-scores: alpha = K_ff^{-1} y, z_i = alpha_i / sqrt([K_ff^{-1}]_ii)."""
    from scipy.linalg import cho_solve, solve_triangular

    L = safe_cholesky(K_ff)
    alpha = cho_solve((L, True), y)
    L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True)
    K_inv_diag = np.clip(np.sum(L_inv**2, axis=0), 1e-12, None)
    return alpha / np.sqrt(K_inv_diag)


def extract_model_true_z_train_correlation(
    model: CopulaTabICL,
    device: Device,
    context_coords: np.ndarray,
    K_ff_context: np.ndarray,
    context_values: np.ndarray,
    coords_test: np.ndarray,
) -> np.ndarray:
    """Model correlation conditioned on the exact GP LOO z_train (synthetic mode only)."""
    from inference.copula_inference import normalize_features

    x_train_norm, x_test_norm = normalize_features(context_coords, coords_test)
    z_train_true = _exact_gp_loo_z_train(K_ff_context, context_values)
    return _forward_correlation(model, device, x_train_norm, z_train_true, x_test_norm)


def sample_simple_kernel_covariance(
    cfg: HasDataConfig,
    coordinates: np.ndarray,
    kernel_name: "str | None" = None,
    seed: "int | None" = None,
) -> "tuple[np.ndarray, str]":
    """Covariance of one elementary data_gen kernel (random or given) at standardized coordinates, with hyperparameters from the training priors.

    Returns (Sigma, kernel_name).
    """
    import random as _random

    import torch
    from omegaconf import OmegaConf

    from copula_inter.gp_kernels import _COMPOSABLE_KERNELS, _SCALAR_ONLY_KERNELS, _build_kernel_component

    if seed is not None:
        _random.seed(seed)
        torch.manual_seed(seed)

    cfg = OmegaConf.merge(cfg, OmegaConf.create({"data": {"sign_modulation_component_prob": 0.0}}))

    coords = np.asarray(coordinates, dtype=np.float64)
    x_std = (coords - coords.mean(axis=0)) / coords.std(axis=0).clip(min=1e-8)
    k = x_std.shape[1]

    if kernel_name is None:
        # Exclude kernels without a lengthscale, and cosine for k > 1.
        candidates = [
            name
            for name in _COMPOSABLE_KERNELS
            if name not in ("dot_product", "polynomial") and not (k > 1 and name in _SCALAR_ONLY_KERNELS)
        ]
        kernel_name = _random.choice(candidates)

    kernel, params = _build_kernel_component(cfg, kernel_name, k=k, B=1, device="cpu")
    x_t = torch.as_tensor(x_std, dtype=torch.get_default_dtype()).unsqueeze(0)  # (1, N, k)
    with torch.no_grad():
        Sigma = kernel(x_t, x_t).to_dense()[0].numpy()

    param_str = ", ".join(f"{name}={v.item():.3f}" for name, v in params.items() if v.numel() == 1)
    print(f"Synthetic ground truth: kernel='{kernel_name}' ({param_str})")
    return Sigma, kernel_name


def build_synthetic_grid_task(
    cfg: HasDataConfig,
    kernel_name: str,
    grid_size: int,
    n_context: int,
    n_bins: int,
    seed: int,
    *,
    min_context: int = 1,
) -> dict:
    """Synthetic task: grid_size x grid_size grid on [-1000, 1000]^2, one sampled covariance, distance/bin/pair-count arrays and a context sample.

    The returned rng has already drawn the context indices; keep using it.
    """
    import torch

    from copula_inter.data_gen import sigma_to_correlation

    rng = np.random.default_rng(seed)
    axis = np.linspace(-1000.0, 1000.0, grid_size)
    x_grid, y_grid = np.meshgrid(axis, axis)
    coords = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    D = coords.shape[0]

    true_cov, _ = sample_simple_kernel_covariance(cfg, coords, kernel_name, seed)
    R_true_t, _ = sigma_to_correlation(torch.as_tensor(true_cov, dtype=torch.float64))
    R_true = R_true_t.numpy()

    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    bin_edges = np.linspace(0.0, dist[np.triu_indices(D, k=1)].max(), n_bins + 1)
    pair_counts = pair_counts_by_distance(dist, bin_edges).astype(float)

    n_context_eff = max(min_context, min(n_context, D - 1))
    context_idx = rng.choice(D, size=n_context_eff, replace=False)
    context_coords = coords[context_idx]

    return {
        "rng": rng,
        "coords": coords,
        "D": D,
        "true_cov": true_cov,
        "R_true": R_true,
        "dist": dist,
        "bin_edges": bin_edges,
        "pair_counts": pair_counts,
        "context_idx": context_idx,
        "context_coords": context_coords,
        "n_context_eff": n_context_eff,
        "L": safe_cholesky(true_cov),
    }


def _bin_indices(d: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Bin index per distance, -1 outside [bin_edges[0], bin_edges[-1]]."""
    n_bins = len(bin_edges) - 1
    bin_idx = np.digitize(d, bin_edges) - 1
    bin_idx[(d < bin_edges[0]) | (d > bin_edges[-1])] = -1
    bin_idx[bin_idx == n_bins] = n_bins - 1  # d == bin_edges[-1] exactly
    return bin_idx


def bin_correlation_by_distance(R: np.ndarray, dist: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Mean upper-triangle correlation per distance bin."""
    iu = np.triu_indices_from(R, k=1)
    corr, d = R[iu], dist[iu]
    n_bins = len(bin_edges) - 1
    bin_idx = _bin_indices(d, bin_edges)
    means = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            means[b] = corr[mask].mean()
    return means


def pair_counts_by_distance(dist: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Number of upper-triangle pairs per distance bin."""
    iu_dist = dist[np.triu_indices_from(dist, k=1)]
    n_bins = len(bin_edges) - 1
    bin_idx = _bin_indices(iu_dist, bin_edges)
    counts = np.bincount(bin_idx[bin_idx >= 0], minlength=n_bins)
    return counts[:n_bins]


def seriate_by_correlation(R: np.ndarray) -> np.ndarray:
    """Index permutation from average-linkage clustering with optimal leaf ordering on 1 - rho."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    dist = 1.0 - np.clip(R, -1.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce exact symmetry; R may only be numerically symmetric
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average", optimal_ordering=True)
    return np.asarray(leaves_list(Z))


# Theoretical correlation decay laws, fitted to the empirical curve only.
def exponential_law(r: np.ndarray, L: float) -> np.ndarray:
    """rho(r) = exp(-r / L) — Hansen & Lebedeff (1987); Matern nu=1/2."""
    return np.exp(-np.asarray(r, dtype=np.float64) / L)


def gaussian_law(r: np.ndarray, L: float) -> np.ndarray:
    """rho(r) = exp(-r^2 / (2 L^2)) — Matern nu -> infinity."""
    r = np.asarray(r, dtype=np.float64)
    return np.exp(-(r**2) / (2.0 * L**2))


def matern_law(r: np.ndarray, L: float, nu: float) -> np.ndarray:
    """General Matern correlation: rho(r) = 2^(1-nu)/Gamma(nu) (r/L)^nu K_nu(r/L)."""
    r = np.asarray(r, dtype=np.float64)
    out = np.ones_like(r)
    nz = r > 0
    x = r[nz] / L
    out[nz] = (2.0 ** (1.0 - nu) / gamma_fn(nu)) * (x**nu) * bessel_k(nu, x)
    return out


def whittle_law(r: np.ndarray, L: float) -> np.ndarray:
    """rho(r) = (r/L) K_1(r/L) — Matern nu=1, and the omega->0 closed-form
    limit of the North, Wang & Genton (2011) energy-balance model."""
    return matern_law(r, L, nu=1.0)


def rational_quadratic_law(r: np.ndarray, L: float, alpha: float) -> np.ndarray:
    """rho(r) = (1 + r^2 / (2 alpha L^2))^(-alpha) — a scale mixture of
    Gaussian kernels with Gamma-distributed lengthscales."""
    r = np.asarray(r, dtype=np.float64)
    return (1.0 + (r**2) / (2.0 * alpha * L**2)) ** (-alpha)


# name -> (callable(r, *params), ordered param names)
THEORY_LAWS: dict[str, tuple[Callable[..., np.ndarray], list[str]]] = {
    "exponential": (exponential_law, ["L"]),
    "gaussian": (gaussian_law, ["L"]),
    "whittle": (whittle_law, ["L"]),
    "matern": (matern_law, ["L", "nu"]),
    "rational_quadratic": (rational_quadratic_law, ["L", "alpha"]),
}

# name -> (color, linestyle, linewidth, display label)
THEORY_STYLE = {
    "exponential": ("purple", "-.", 1.6, "Exponential (Hansen & Lebedeff 1987, $\\nu$=1/2)"),
    "gaussian": ("saddlebrown", "--", 1.6, "Gaussian ($\\nu\\to\\infty$)"),
    "whittle": ("darkgreen", ":", 1.8, "Whittle / EBCM $\\omega\\to0$ limit (North et al. 2011, $\\nu$=1)"),
    "matern": ("magenta", "-", 2.4, "Matérn (free $\\nu$)"),
    "rational_quadratic": ("teal", "-.", 1.8, "Rational Quadratic (scale mixture of Gaussians)"),
}

# Published reference decorrelation lengths L (km).
LITERATURE_L = {
    "exponential": (1800.0, "North, Wang & Genton 2011, Fig. 1 (extratropical, exponential fit)"),
    "whittle": (2800.0, "North, Wang & Genton 2011, Fig. 2 (eastern Siberia, Whittle/EBCM fit)"),
}


def _correlation_length_guess(dist_centers: np.ndarray, rho: np.ndarray) -> float:
    """Distance where the empirical curve crosses 1/e (linear interpolation), as a starting L."""
    valid = np.isfinite(rho)
    d, r = dist_centers[valid], rho[valid]
    below = np.where(r <= 1.0 / math.e)[0]
    if len(below) == 0 or below[0] == 0:
        return float(d[-1] / 2.0) if len(d) else 1000.0
    i = below[0]
    d0, d1, r0, r1 = d[i - 1], d[i], r[i - 1], r[i]
    if r0 == r1:
        return float(d0)
    frac = (1.0 / math.e - r0) / (r1 - r0)
    return float(d0 + frac * (d1 - d0))


def fit_theoretical_law(
    dist_centers: np.ndarray,
    rho_emp: np.ndarray,
    pair_counts: np.ndarray,
    model: str,
) -> "dict | None":
    """Weighted (sqrt(pair_counts)) least-squares fit of a decay law to the binned empirical curve; None if the fit fails."""
    if model not in THEORY_LAWS:
        raise ValueError(f"Unknown theory model '{model}', choose from {sorted(THEORY_LAWS)}.")
    law_fn, param_names = THEORY_LAWS[model]

    mask = np.isfinite(rho_emp) & (pair_counts > 0)
    if mask.sum() < len(param_names) + 1:
        print(f"Warning: too few valid distance bins ({mask.sum()}) to fit '{model}'; skipping.")
        return None
    d, r, n = dist_centers[mask], rho_emp[mask], pair_counts[mask]
    sigma = 1.0 / np.sqrt(n)  # SEM of a Pearson-r bin mean scales ~ 1/sqrt(n pairs)

    L0 = _correlation_length_guess(dist_centers, rho_emp)
    L_hi = max(50.0 * L0, 10.0 * float(dist_centers[-1]))
    if param_names == ["L"]:
        p0, bounds = [L0], ([1.0], [L_hi])
    else:  # ["L", "nu"]
        p0, bounds = [L0, 1.0], ([1.0, 0.05], [L_hi, 8.0])

    try:
        popt, _ = curve_fit(law_fn, d, r, p0=p0, sigma=sigma, bounds=bounds, maxfev=20000)
    except RuntimeError as exc:
        print(f"Warning: curve_fit failed to converge for '{model}' ({exc}); skipping.")
        return None

    pred = law_fn(d, *popt)
    resid = r - pred
    weighted_ss_res = float(np.sum((resid / sigma) ** 2))
    r_bar = np.average(r, weights=1.0 / sigma**2)
    weighted_ss_tot = float(np.sum(((r - r_bar) / sigma) ** 2))
    r_squared = 1.0 - weighted_ss_res / weighted_ss_tot if weighted_ss_tot > 0 else float("nan")

    return {"model": model, "law_fn": law_fn, "params": dict(zip(param_names, popt)), "r_squared": r_squared}
