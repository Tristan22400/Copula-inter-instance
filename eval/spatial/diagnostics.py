"""diagnostics.py — CopulaTabICL checkpoint loading, dummy/real-context
correlation extraction, distance binning, and theoretical decay-law curve
fitting: the core engine behind every spatial-correlation diagnostic/sweep
tool. Promoted from plots/plot_spatial_correlation_diagnostics.py.

Reuses:
  - eval/data/era5_io.py: safe_cholesky (was generate_plots._safe_cholesky).
  - inference/copula_inference.py: load_copula_model, normalize_features —
    the repo's single canonical checkpoint loader / feature-normalization
    convention, not reimplemented here.
  - src/pit.py: load_tabicl + run_pit — the same frozen-TabICL-quantile-head
    K-fold LOO PIT used to build z_train for real (non-GP-oracle) data.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gamma as gamma_fn, kv as bessel_k

from eval.data.era5_io import safe_cholesky

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = [
    "compute_persistence_residuals",
    "compute_raw_temperature_observations",
    "get_ground_truth_observations",
    "empirical_spatial_correlation",
    "morans_i",
    "predict_copula_residual_field",
    "load_copula_model",
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


# ---------------------------------------------------------------------------
# Ground truth: empirical spatial correlation from raw temperature / 24h
# persistence residuals
# ---------------------------------------------------------------------------
def compute_persistence_residuals(field_all: np.ndarray) -> np.ndarray:
    """24h persistence residuals E_t = Z_true_t - Z_true_{t-24}.

    `field_all` (n_snapshots, H, W) is one grid snapshot per time index, at a
    fixed cadence of 24h apart, so a lag of 1 index IS a 24h lag. Returns
    (n_snapshots - 1, H * W).
    """
    n = field_all.shape[0]
    if n < 2:
        raise ValueError(f"Need >= 2 time snapshots to form 24h persistence residuals, got {n}.")
    flat = field_all.reshape(n, -1)
    return flat[1:] - flat[:-1]


def compute_raw_temperature_observations(field_all: np.ndarray) -> np.ndarray:
    """Raw per-day temperature Z_t across the spatial grid, with NO
    differencing. Returns (n_snapshots, H * W)."""
    n = field_all.shape[0]
    return field_all.reshape(n, -1)


def get_ground_truth_observations(field_all: np.ndarray, target: str) -> np.ndarray:
    """Dispatch on `target`: the per-snapshot observation matrix (n, H*W)
    both empirical_spatial_correlation and real-context conditioning values
    are derived from."""
    if target == "raw":
        return compute_raw_temperature_observations(field_all)
    if target == "residual":
        return compute_persistence_residuals(field_all)
    raise ValueError(f"Unknown target '{target}', choose from 'raw' or 'residual'.")


def empirical_spatial_correlation(data: dict, target: str = "raw") -> np.ndarray:
    """Pearson correlation matrix R_emp (D x D) of `target` (raw temperature
    by default, or the 24h persistence residual)."""
    observations = get_ground_truth_observations(data["t2m"], target)
    return np.corrcoef(observations.T)


def morans_i(field: np.ndarray) -> float:
    """Global Moran's I (Moran, 1950) with rook (4-neighbor) adjacency on a
    regular (H, W) grid: how smooth/locally coherent a SINGLE snapshot is.
    +1 = smooth field, 0 = spatially random, negative = checkerboard-like."""
    x = field - field.mean()
    cross_h = x[:, :-1] * x[:, 1:]
    cross_v = x[:-1, :] * x[1:, :]
    n_edges = cross_h.size + cross_v.size
    numerator = cross_h.sum() + cross_v.sum()
    denominator = (x ** 2).sum()
    return float(field.size * numerator / (n_edges * denominator))


def predict_copula_residual_field(
    tabicl_marginal, context_coords: np.ndarray, context_values: np.ndarray,
    coords_test: np.ndarray, R_context: np.ndarray, device: str, z_shared: np.ndarray,
) -> np.ndarray:
    """One joint draw (D,) from the copula model's implied residual field:
    inject `R_context` into the shared latent Gaussian vector `z_shared` via
    Cholesky, then map each coordinate through the frozen TabICL marginal
    quantile function (conditioned on the same real context) — i.e.
    y = F_hat^{-1}(Phi(z)). Falls back to a naive Gaussian(mean, std)
    marginal if `tabicl_marginal` is None (scratch-trained backbone).
    """
    from scipy.stats import norm

    L = safe_cholesky(R_context)
    z_copula = L @ z_shared
    u_copula = np.clip(norm.cdf(z_copula), 1e-6, 1.0 - 1e-6)

    y_mean = context_values.mean()
    y_std = max(context_values.std(), 1e-8)

    if tabicl_marginal is None:
        return y_mean + y_std * z_copula

    import torch

    from inference.copula_inference import normalize_features
    from pit import normalize_targets

    x_train_norm, x_test_norm = normalize_features(context_coords, coords_test)

    x_full = np.concatenate([x_train_norm, x_test_norm], axis=0)
    x_batch = torch.as_tensor(x_full, dtype=torch.float32, device=device).unsqueeze(0)  # (1, P+N, p_x)
    context_values_t = torch.as_tensor(context_values, dtype=torch.float32, device=device)
    context_values_scaled_t, _, y_mean_t, y_std_t = normalize_targets(context_values_t)
    y_train_batch = context_values_scaled_t.unsqueeze(0)  # (1, P)
    with torch.no_grad():
        logits = tabicl_marginal(x_batch, y_train_batch)  # (1, N, Q) -- N test rows only
        n_test = coords_test.shape[0]
        dist = tabicl_marginal.quantile_dist(logits.reshape(n_test, -1))
        u_t = torch.as_tensor(u_copula, dtype=torch.float32, device=device).unsqueeze(-1)
        y_pred_scaled = dist.icdf(u_t).squeeze(-1).double()
    return (y_mean_t.double() + y_std_t.double() * y_pred_scaled).cpu().numpy()


# ---------------------------------------------------------------------------
# TabICLv2 / CopulaTabICL: shared checkpoint loading + dummy/real-context extraction
# ---------------------------------------------------------------------------
def load_copula_model(ckpt_path: str, device: "str | None" = None):
    """Load a CopulaTabICL checkpoint via the repo's single canonical loader
    (inference/copula_inference.py::load_copula_model); resolves the auto
    ("cuda" if available else "cpu") device default."""
    import torch

    from inference.copula_inference import load_copula_model as _load_copula_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, cfg = _load_copula_model(ckpt_path, device=device)
    return model, cfg, device


def load_marginal_tabicl(cfg, device: str):
    """Load the frozen, pretrained TabICL quantile regressor used ONLY as a
    marginal-CDF oracle for the PIT transform in
    extract_model_context_correlation — NOT the same object as the
    CopulaTabICL backbone in load_copula_model. Returns None (with a
    warning) if the checkpoint's backbone was trained from scratch."""
    if not bool(cfg.tabicl.get("pretrained", True)):
        print("Warning: cfg.tabicl.pretrained=False — no pretrained quantile "
              "head available for PIT; context z_train will fall back to "
              "naive standardization.")
        return None

    from src.pit import load_tabicl

    return load_tabicl(cfg.tabicl.ckpt, device)


def _forward_correlation(model, device, x_train_norm: np.ndarray, z_train: np.ndarray, x_test_norm: np.ndarray) -> np.ndarray:
    """Shared (x_train, z_train, x_test) -> Sigma forward pass, used by both
    the dummy-context and real-context extractions below."""
    import torch

    from src.model import low_rank_correlation

    x_train_t = torch.as_tensor(x_train_norm, dtype=torch.float32, device=device).unsqueeze(0)
    x_test_t = torch.as_tensor(x_test_norm, dtype=torch.float32, device=device).unsqueeze(0)
    z_train_t = torch.as_tensor(z_train, dtype=torch.float32, device=device).unsqueeze(0)
    batch = {"x_train": x_train_t, "x_test": x_test_t, "z_train": z_train_t}

    with torch.no_grad():
        out = model(batch)
        Sigma = low_rank_correlation(out["W"], out["s"], jitter=1e-4)
    return Sigma[0].cpu().numpy()


def extract_model_dummy_context_correlation(model, device, coords_test: np.ndarray) -> np.ndarray:
    """Extract the model's correlation matrix via an unconditional forward
    pass — a single dummy context row at x_train=0, z_train=0 (P=1), the
    closest architecturally-valid stand-in for "no historical context" (see
    CopulaTabICL's target-aware column embedding, which requires P>=1)."""
    coords_test = np.asarray(coords_test, dtype=np.float64)
    x_mean = coords_test.mean(axis=0, keepdims=True)
    x_std = coords_test.std(axis=0, keepdims=True).clip(min=1e-8)
    x_test_norm = (coords_test - x_mean) / x_std

    x_train_norm = np.zeros((1, coords_test.shape[1]), dtype=np.float64)
    z_train = np.zeros(1, dtype=np.float64)
    return _forward_correlation(model, device, x_train_norm, z_train, x_test_norm)


def compute_context_z_train(
    x_train_norm: np.ndarray, context_values: np.ndarray, tabicl_marginal, device: str, k_folds: int = 10,
) -> np.ndarray:
    """K-fold leave-one-out PIT z_train for a real in-context sample
    (`x_train_norm` already through `normalize_features`), under
    `tabicl_marginal`'s own predicted marginal distribution (src/pit.py::run_pit):
    u_i = F_hat(y_i), z_i = Phi^-1(u_i) — the real-data analogue of how
    z_train is defined during training (data_gen.py's GP-oracle LOO PIT).
    Falls back to naive standardization if `tabicl_marginal` is None.

    Shared by extract_model_context_correlation (checkpoint-sweep diagnostics,
    below) and src/train.py's era5_fit validation probe — both need this same
    context-PIT step, split out from the model forward pass that follows it
    so the training loop can freeze z_train once while re-running the forward
    pass on the currently-training model every validate() call.
    """
    if tabicl_marginal is None:
        y_mean = context_values.mean()
        y_std = max(context_values.std(), 1e-8)
        return (context_values - y_mean) / y_std

    import torch

    from pit import normalize_targets
    from src.pit import run_pit

    X_train_t = torch.as_tensor(x_train_norm, dtype=torch.float32, device=device)
    context_values_t = torch.as_tensor(context_values, dtype=torch.float32, device=device)
    context_values_scaled_t, _, _, _ = normalize_targets(context_values_t)
    Y_train_t = context_values_scaled_t.unsqueeze(-1)  # (P, 1)
    pit_out = run_pit(
        tabicl_marginal, X_train_t, Y_train_t, X_train_t[:1], Y_train_t[:1], k_folds=k_folds,
    )
    return pit_out["z_train"].squeeze(-1).cpu().numpy()  # (P,)


def extract_model_context_correlation(
    model, device, tabicl_marginal, context_coords: np.ndarray, context_values: np.ndarray,
    coords_test: np.ndarray, k_folds: int = 10,
) -> np.ndarray:
    """Extract the model's correlation matrix via a single joint forward
    pass over all of `coords_test` at once, conditioned on a real historical
    in-context sample (context_coords, context_values). See
    compute_context_z_train for the z_train derivation.
    """
    from inference.copula_inference import normalize_features

    x_train_norm, x_test_norm = normalize_features(context_coords, coords_test)
    z_train = compute_context_z_train(x_train_norm, context_values, tabicl_marginal, device, k_folds)
    return _forward_correlation(model, device, x_train_norm, z_train, x_test_norm)


def _exact_gp_loo_z_train(K_ff: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact zero-mean GP leave-one-out PIT z-scores (Rasmussen & Williams,
    GPML Eq. 5.12), computed directly from the TRUE synthetic kernel
    covariance restricted to the context points:

        alpha = K_ff^-1 y
        z_train[i] = alpha_i / sqrt([K_ff^-1]_ii)
    """
    from scipy.linalg import cho_solve, solve_triangular

    L = safe_cholesky(K_ff)
    alpha = cho_solve((L, True), y)
    L_inv = solve_triangular(L, np.eye(L.shape[0]), lower=True)
    K_inv_diag = np.clip(np.sum(L_inv ** 2, axis=0), 1e-12, None)
    return alpha / np.sqrt(K_inv_diag)


def extract_model_true_z_train_correlation(
    model, device, context_coords: np.ndarray, K_ff_context: np.ndarray,
    context_values: np.ndarray, coords_test: np.ndarray,
) -> np.ndarray:
    """Extract the model's correlation matrix conditioned on the EXACT GP
    leave-one-out z_train (_exact_gp_loo_z_train) instead of
    extract_model_context_correlation's TabICLv2 K-fold PIT *estimate* of
    the same quantity — only available in synthetic mode, where the true
    generating kernel (and hence the true z_train) is known."""
    from inference.copula_inference import normalize_features

    x_train_norm, x_test_norm = normalize_features(context_coords, coords_test)
    z_train_true = _exact_gp_loo_z_train(K_ff_context, context_values)
    return _forward_correlation(model, device, x_train_norm, z_train_true, x_test_norm)


# ---------------------------------------------------------------------------
# Synthetic mode: known ground-truth covariance from a single data_gen.py kernel
# ---------------------------------------------------------------------------
def sample_simple_kernel_covariance(
    cfg, coordinates: np.ndarray, kernel_name: "str | None" = None, seed: "int | None" = None,
) -> "tuple[np.ndarray, str]":
    """Ground-truth covariance for synthetic mode: samples ONE elementary
    (non-composite) kernel from src/data_gen.py's registry (random choice if
    `kernel_name` is None, else the given one — e.g. for constants.SYNTHETIC_SWEEP_KERNELS
    sweeps), with hyperparameters drawn from the SAME LogNormal/Gamma
    hyperpriors training episodes use.

    `coordinates` (M, d) is standardized (zero mean, unit variance) before
    evaluating the kernel, since data_gen's lengthscale prior is calibrated
    for that scale.

    Returns (Sigma, kernel_name).
    """
    import random as _random

    import torch
    from omegaconf import OmegaConf

    from data_gen import _build_kernel_component, _COMPOSABLE_KERNELS, _SCALAR_ONLY_KERNELS

    if seed is not None:
        _random.seed(seed)
        torch.manual_seed(seed)

    cfg = OmegaConf.merge(cfg, OmegaConf.create({"data": {"sign_modulation_component_prob": 0.0}}))

    coords = np.asarray(coordinates, dtype=np.float64)
    x_std = (coords - coords.mean(axis=0)) / coords.std(axis=0).clip(min=1e-8)
    k = x_std.shape[1]

    if kernel_name is None:
        # Exclude "dot_product"/"polynomial" (no real lengthscale) and, for
        # k > 1 coordinates, "cosine" (only PSD for scalar input).
        candidates = [
            name for name in _COMPOSABLE_KERNELS
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
    cfg, kernel_name: str, grid_size: int, n_context: int, n_bins: int, seed: int, *, min_context: int = 1,
) -> dict:
    """Shared synthetic-task setup: the grid_size x grid_size coordinate
    grid, ONE known ground-truth covariance sampled from it
    (sample_simple_kernel_covariance), the derived distance/bin/pair-count
    arrays, and one context-point sample — everything a synthetic sweep task
    needs before it's free to diverge on what it does with a context sample
    (average many draws' predicted correlation, as
    sweep_core.py::run_synthetic_config does; or compare a single draw's
    exact-GP z_train against several marginal backends' K-fold PIT estimate
    of it, as debug/stages/s7_backbone.py::run_task does) — previously
    duplicated between those two call sites.

    The returned ``rng`` has already drawn `context_idx`; callers that need
    further random draws from the SAME stream (e.g. sampling z_true =
    L @ rng.standard_normal(D)) should keep using it rather than creating a
    new Generator, to keep one `seed` fully determining the whole task.

    Coordinate SCALE is arbitrary (sample_simple_kernel_covariance z-scores
    before evaluating the kernel), so [-1000, 1000] is used purely so
    `dist`'s magnitude clears fit_theoretical_law's hardcoded L >= 1.0 lower
    bound (calibrated for real ERA5 km-distances).
    """
    import torch

    from data_gen import sigma_to_correlation

    rng = np.random.default_rng(seed)
    axis = np.linspace(-1000.0, 1000.0, grid_size)
    x_grid, y_grid = np.meshgrid(axis, axis)
    coords = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    D = coords.shape[0]

    true_cov, _ = sample_simple_kernel_covariance(cfg, coords, kernel_name, seed)
    R_true, _ = sigma_to_correlation(torch.as_tensor(true_cov, dtype=torch.float64))
    R_true = R_true.numpy()

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


# ---------------------------------------------------------------------------
# Distance binning shared by every curve
# ---------------------------------------------------------------------------
def _bin_indices(d: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Bin index per distance, or -1 for out-of-[bin_edges[0], bin_edges[-1]]
    values. Pairs beyond bin_edges[-1] are DROPPED, not clipped into the
    last bin (see pair_counts_by_distance's docstring on the corner-biased
    tail)."""
    n_bins = len(bin_edges) - 1
    bin_idx = np.digitize(d, bin_edges) - 1
    bin_idx[(d < bin_edges[0]) | (d > bin_edges[-1])] = -1
    bin_idx[bin_idx == n_bins] = n_bins - 1  # d == bin_edges[-1] exactly
    return bin_idx


def bin_correlation_by_distance(R: np.ndarray, dist: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Mean correlation per distance bin, over the upper-triangle pairwise entries."""
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
    """Number of upper-triangle (i, j) pairs falling in each distance bin.

    On a bounded, non-periodic lat/lon rectangle the pair population thins
    out sharply near the max distance — a bin's mean correlation is only as
    trustworthy as its count here."""
    iu_dist = dist[np.triu_indices_from(dist, k=1)]
    n_bins = len(bin_edges) - 1
    bin_idx = _bin_indices(iu_dist, bin_edges)
    counts = np.bincount(bin_idx[bin_idx >= 0], minlength=n_bins)
    return counts[:n_bins]


def seriate_by_correlation(R: np.ndarray) -> np.ndarray:
    """Permutation of R's indices via average-linkage hierarchical clustering
    with optimal leaf ordering (Bar-Joseph et al. 2001) on the correlation
    distance d_ij = 1 - rho_ij — makes a correlation-matrix heatmap
    interpretable by placing strongly-correlated pairs near the diagonal."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    dist = 1.0 - np.clip(R, -1.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce exact symmetry; R may only be numerically symmetric
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average", optimal_ordering=True)
    return np.asarray(leaves_list(Z))


# ---------------------------------------------------------------------------
# Theoretical spatial-correlation decay laws (North, Wang & Genton 2011;
# Whittle 1954; Matern 1960) — an independent physical sanity check, fit
# ONLY to the ground-truth empirical curve, never the model curve.
# ---------------------------------------------------------------------------
def exponential_law(r: np.ndarray, L: float) -> np.ndarray:
    """rho(r) = exp(-r / L) — Hansen & Lebedeff (1987); Matern nu=1/2."""
    return np.exp(-np.asarray(r, dtype=np.float64) / L)


def gaussian_law(r: np.ndarray, L: float) -> np.ndarray:
    """rho(r) = exp(-r^2 / (2 L^2)) — Matern nu -> infinity."""
    r = np.asarray(r, dtype=np.float64)
    return np.exp(-(r ** 2) / (2.0 * L ** 2))


def matern_law(r: np.ndarray, L: float, nu: float) -> np.ndarray:
    """General Matern correlation: rho(r) = 2^(1-nu)/Gamma(nu) (r/L)^nu K_nu(r/L)."""
    r = np.asarray(r, dtype=np.float64)
    out = np.ones_like(r)
    nz = r > 0
    x = r[nz] / L
    out[nz] = (2.0 ** (1.0 - nu) / gamma_fn(nu)) * (x ** nu) * bessel_k(nu, x)
    return out


def whittle_law(r: np.ndarray, L: float) -> np.ndarray:
    """rho(r) = (r/L) K_1(r/L) — Matern nu=1, and the omega->0 closed-form
    limit of the North, Wang & Genton (2011) energy-balance model."""
    return matern_law(r, L, nu=1.0)


def rational_quadratic_law(r: np.ndarray, L: float, alpha: float) -> np.ndarray:
    """rho(r) = (1 + r^2 / (2 alpha L^2))^(-alpha) — a scale mixture of
    Gaussian kernels with Gamma-distributed lengthscales."""
    r = np.asarray(r, dtype=np.float64)
    return (1.0 + (r ** 2) / (2.0 * alpha * L ** 2)) ** (-alpha)


# name -> (callable(r, *params), ordered param names)
THEORY_LAWS = {
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

# Published reference decorrelation lengths L (km) — a literature sanity
# check, independent of the fits above.
LITERATURE_L = {
    "exponential": (1800.0, "North, Wang & Genton 2011, Fig. 1 (extratropical, exponential fit)"),
    "whittle": (2800.0, "North, Wang & Genton 2011, Fig. 2 (eastern Siberia, Whittle/EBCM fit)"),
}


def _correlation_length_guess(dist_centers: np.ndarray, rho: np.ndarray) -> float:
    """Initial L guess for curve_fit: distance at which the empirical curve
    crosses 1/e, by linear interpolation between bracketing bin centers."""
    valid = np.isfinite(rho)
    d, r = dist_centers[valid], rho[valid]
    below = np.where(r <= 1.0 / np.e)[0]
    if len(below) == 0 or below[0] == 0:
        return float(d[-1] / 2.0) if len(d) else 1000.0
    i = below[0]
    d0, d1, r0, r1 = d[i - 1], d[i], r[i - 1], r[i]
    if r0 == r1:
        return float(d0)
    frac = (1.0 / np.e - r0) / (r1 - r0)
    return float(d0 + frac * (d1 - d0))


def fit_theoretical_law(
    dist_centers: np.ndarray, rho_emp: np.ndarray, pair_counts: np.ndarray, model: str,
) -> "dict | None":
    """Nonlinear least-squares fit of `model` (see THEORY_LAWS) to the binned
    ground-truth empirical curve, weighted by sqrt(pair_counts) per bin.

    Returns None (with a printed warning) instead of raising if curve_fit
    fails to converge or too few bins are populated.
    """
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
    r_bar = np.average(r, weights=1.0 / sigma ** 2)
    weighted_ss_tot = float(np.sum(((r - r_bar) / sigma) ** 2))
    r_squared = 1.0 - weighted_ss_res / weighted_ss_tot if weighted_ss_tot > 0 else float("nan")

    return {"model": model, "law_fn": law_fn, "params": dict(zip(param_names, popt)), "r_squared": r_squared}
