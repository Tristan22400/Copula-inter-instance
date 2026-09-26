"""GP episode generation for the inter-instance copula.

Each episode samples a GP kernel and its hyperparameters, draws P + N points,
normalizes features over all P + N, samples targets jointly from the GP, and
returns the prior correlation R_star at the test points plus the PIT
inputs (z_train, z_test, log_pdf_test).

Kernels and their priors live in gp_kernels, the per-call prior sampling
(feature count, kernel choice, composition chains, mean function) in
kernel_sampling, and the feature warps in feature_transforms and
structural_warps.
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import gpytorch
import numpy as np
import torch
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.utils.errors import NanError, NotPSDError
from torch import Tensor

from copula_inter.backend_registry import batched_backend_factories
from copula_inter.episode_contracts import assemble_episodes
from copula_inter.feature_transforms import (
    apply_kernel_hidden_warp,
    apply_mlp_feature_mixing,
    tabiclv2_warp_features,
)
from copula_inter.gp_kernels import (
    _build_likelihood,
    _kernel_needs_scalar_input,
    _parse_composite,
    _sample_episode_kernel,
)
from copula_inter.kernel_sampling import (
    _build_kernel_chain,
    _resolve_kernel_name,
    _sample_active_dims,
    _sample_d_features,
    _sample_kernel_chain_structure,
    _sample_mean_module,
    _tabicl_mix_prob_for_kernel,
)
from copula_inter.loss import _safe_cholesky
from copula_inter.rng import seed_everything
from copula_inter.structural_warps import apply_structural_feature_warp
from copula_inter.type_aliases import Device, HasDataConfig

if TYPE_CHECKING:
    from copula_inter.pit import TabICLLike

# Force exact Cholesky solves for every covariance up to this size (gpytorch uses CG above max_cholesky_size).
_MAX_CHOLESKY = 8192

# z_train_source values with a batched PIT module, as lazy importers (heavy optional dependencies).
_BATCHED_MARGINAL_BACKENDS: Dict[str, Callable[[], Callable]] = batched_backend_factories()


# Kernel construction: _sample_episode_kernel samples B episodes' kernels;
# build_kernel_fn rebuilds a kernel from concrete saved hyperparameters.


# Kernel registry: named free functions that evaluate build_kernel_fn.
# Hyperparameters are assigned in place, so these do not backpropagate into them.


def gp_posterior(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    kernel_fn: Callable[[Tensor, Tensor], Tensor],
    noise: float,
    *,
    latent: bool = True,
    return_factors: bool = False,
) -> tuple:
    """Analytical GP posterior at the test points.

    Args:
        latent: posterior over f* (K_ss without noise) instead of noisy y*.
        return_factors: also return (L_ff, alpha).

    Returns:
        mu_star (N,), Sigma_star (N, N), and, if return_factors, L_ff (P, P) and
        alpha = K_ff^{-1} y_train (P,).
    """
    P, N = x_train.shape[0], x_test.shape[0]

    K_ff = kernel_fn(x_train, x_train) + noise * torch.eye(P, device=x_train.device)
    K_sf = kernel_fn(x_test, x_train)  # (N, P)
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
    """Convert a covariance matrix to (correlation matrix, marginal std)."""
    sigma = Sigma.diagonal().clamp(min=1e-10).sqrt()  # (N,)
    D_inv = torch.diag(1.0 / sigma)
    R = D_inv @ Sigma @ D_inv
    # Renormalize with the original sigma to remove float32 drift symmetrically.
    d = R.diagonal().clamp(min=1e-10).sqrt()
    R = R / (d.unsqueeze(0) * d.unsqueeze(1))
    return R, sigma


@torch.no_grad()
def generate_gp_task(cfg: HasDataConfig) -> Dict[str, Tensor]:
    """Generate one GP episode: generate_gp_batch(cfg, 1, "cpu", return_kernel_metadata=True)[0].

    Keys: x_norm_train, y_train, x_norm_test, y_test, z_train, z_test,
    log_pdf_test, R_star, Sigma_star, mu_star, sigma_star, n_train, n_test, the
    kernel hyperparameters, kernel_feature_indices, and the unsaved _L_ff/_alpha
    Cholesky factors.
    """
    return generate_gp_batch(cfg, 1, "cpu", return_kernel_metadata=True)[0]


def _gathered_psd_safe_cholesky(K: Tensor, label: str, max_tries: int = 6) -> tuple[Tensor, Tensor]:
    """Batched Cholesky (B, N, N) -> (L, failed).

    One cholesky_ex on the whole batch, then gpytorch's psd_safe_cholesky
    (escalating jitter) only on the episodes that failed. Episodes that are NaN or
    still not PSD at maximum jitter get an identity L and failed=True, and the
    discard rate is logged.
    """
    L, info = torch.linalg.cholesky_ex(K)
    failed = info.ne(0)
    if not failed.any():
        return L, failed
    idx = failed.nonzero(as_tuple=True)[0]
    eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
    try:
        L[idx] = psd_safe_cholesky(K[idx], max_tries=max_tries)
        return L, torch.zeros_like(failed)
    except (NotPSDError, NanError):
        jitter0 = gpytorch.settings.cholesky_jitter.value(K.dtype)
        max_jitter = jitter0 * (10 ** (max_tries - 1))
        K_boosted = K[idx] + max_jitter * eye
        nan_mask = torch.isnan(K_boosted).any(dim=-1).any(dim=-1)
        if nan_mask.any():
            K_boosted = K_boosted.clone()
            K_boosted[nan_mask] = eye
        L_sub, info_sub = torch.linalg.cholesky_ex(K_boosted)
        L[idx] = L_sub
        still_failed = torch.zeros_like(failed)
        still_failed[idx] = info_sub.ne(0) | nan_mask
        warnings.warn(
            f"{label}: {int(still_failed.sum())}/{K.shape[0]} episodes fell back "
            f"to an identity Cholesky factor (unrecoverable even at jitter="
            f"{max_jitter:.1e}, or had NaN entries) and will be discarded.",
            RuntimeWarning,
        )
        L[still_failed] = eye.unsqueeze(0).expand_as(L[still_failed])
        return L, still_failed


def _batched_cholesky(K: Tensor) -> tuple[Tensor, Tensor]:
    """Cholesky of K_ff for the LOO PIT; see _gathered_psd_safe_cholesky."""
    return _gathered_psd_safe_cholesky(K, label="_batched_cholesky (K_ff)")


def _psd_safe_batch(K: Tensor, max_tries: int = 6) -> tuple[Tensor, Tensor]:
    """Cholesky of K_all for sampling; see _gathered_psd_safe_cholesky.

    Episodes with NaN kernel entries are replaced by identity and marked failed.
    """
    nan_failed = torch.isnan(K).any(dim=-1).any(dim=-1)
    if nan_failed.any():
        eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
        K = K.clone()
        K[nan_failed] = eye
        warnings.warn(
            f"_psd_safe_batch: {int(nan_failed.sum())}/{K.shape[0]} episodes had "
            f"NaN entries in K_all (kernel evaluation produced NaN) and will be "
            f"discarded.",
            RuntimeWarning,
        )
    L, failed = _gathered_psd_safe_cholesky(K, label="_psd_safe_batch (K_all)", max_tries=max_tries)
    return L, failed | nan_failed


# z_train corruption: optional robustness augmentation toward N(0, 1) noise.
DEFAULT_Z_CORRUPTION_RHO_BETA_A = 2.0
DEFAULT_Z_CORRUPTION_RHO_BETA_B = 3.0


def corrupt_z_train(z_train: Tensor, data_cfg: Any) -> Tensor:
    """Mix z_train with i.i.d. N(0, 1) noise per episode (cfg.data.z_train_corruption_*).

        z = sqrt(rho) * z_train + sqrt(1 - rho) * eps,   rho ~ Beta(a, b)

    applied to a fraction of episodes; a no-op unless z_train_corruption_enabled.

    Args:
        z_train: (B, P).
        data_cfg: cfg.data.

    Returns:
        (B, P) tensor.
    """
    # getattr so plain dataclass configs work too.
    if not bool(getattr(data_cfg, "z_train_corruption_enabled", False)):
        return z_train

    prob = float(getattr(data_cfg, "z_train_corruption_prob", 0.5))
    if prob <= 0.0:
        return z_train

    beta_a = float(getattr(data_cfg, "z_train_corruption_rho_beta_a", DEFAULT_Z_CORRUPTION_RHO_BETA_A))
    beta_b = float(getattr(data_cfg, "z_train_corruption_rho_beta_b", DEFAULT_Z_CORRUPTION_RHO_BETA_B))

    B, P = z_train.shape
    device = z_train.device

    apply_ep = torch.rand(B, device=device) < prob  # (B,) which episodes get corrupted at all
    if not bool(apply_ep.any()):
        return z_train

    rho = torch.distributions.Beta(beta_a, beta_b).sample((B,)).to(device=device, dtype=z_train.dtype)
    noise = torch.randn(B, P, device=device, dtype=z_train.dtype)

    sqrt_rho = rho.clamp(0.0, 1.0).sqrt().unsqueeze(-1)
    sqrt_1m_rho = (1.0 - rho).clamp(0.0, 1.0).sqrt().unsqueeze(-1)
    z_blend = sqrt_rho * z_train + sqrt_1m_rho * noise

    return torch.where(apply_ep.unsqueeze(-1), z_blend, z_train)


def _max_batch_for_context(B: int, T: int, device: Device) -> int:
    """Largest episode batch that fits in free CUDA memory for context length T = P + N (<= B)."""
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return B
    free_bytes, _ = torch.cuda.mem_get_info(device)
    # About 6 (B, T, T) float32 buffers are live at peak; budget half of free memory.
    bytes_per_episode = 6 * T * T * 4
    budget = 0.5 * free_bytes
    return max(1, min(B, int(budget // bytes_per_episode)))


def _evaluate_kernel_dense(kernel_obj: gpytorch.kernels.Kernel, x_norm: Tensor) -> Tensor:
    """Evaluate kernel_obj(x_norm) as a dense (B, T, T) tensor."""
    return kernel_obj(x_norm).to_dense()


@dataclass
class _CallShape:
    """Settings every episode of one generation call shares."""

    d: int
    kernel_name: str
    systematic: bool
    chain_names: List[str]
    chain_ops: List[str]
    P: int
    N: int
    P_C: int  # calibration-only points (tabicl_split), after train and test
    B: int
    kernel_cols: Optional[List[int]]  # None = every column

    @property
    def T(self) -> int:
        return self.P + self.N + self.P_C

    @property
    def k(self) -> int:
        return self.d if self.kernel_cols is None else len(self.kernel_cols)


def _sample_call_shape(
    cfg: HasDataConfig,
    B: int,
    device: Device,
    d_override: Optional[int],
    kernel_weights: Optional[Tensor],
    tabicl_split_calib_frac: float,
) -> _CallShape:
    """Draw the call-shared kernel, feature count, (P, N) and active dims; cap B to fit memory."""
    # d_override pins d across generate_gp_batch's top-up calls.
    d = d_override if d_override is not None else _sample_d_features(cfg)

    # systematic_composition samples a chain instead of cfg.data.kernel(s).
    systematic = bool(getattr(cfg.data, "systematic_composition", False))
    chain_names: List[str] = []
    chain_ops: List[str] = []
    if systematic:
        chain_names, chain_ops, kernel_name = _sample_kernel_chain_structure(cfg, kernel_weights=kernel_weights)
    else:
        kernel_name = _resolve_kernel_name(cfg, kernel_weights=kernel_weights)
    P = random.randint(cfg.data.P_min, cfg.data.P_max)
    N = random.randint(cfg.data.N_min, cfg.data.N_max)
    # Calibration-only points for tabicl_split_calib_frac > 0, placed after train and
    # test so the train/test sample does not depend on them.
    P_C = max(1, round(tabicl_split_calib_frac * P)) if tabicl_split_calib_frac > 0 else 0
    # Cap B so the (B, T, T) buffers fit in free memory.
    B = _max_batch_for_context(B, P + N + P_C, device)

    # periodic is capped to k=1 (the period is not identifiable in higher dimensions).
    kernel_cols: Optional[List[int]]
    if _kernel_needs_scalar_input(kernel_name) or "periodic" in kernel_name:
        kernel_cols = [random.randint(0, d - 1)]
    elif kernel_name == "dot_product":
        # dot_product uses every column.
        kernel_cols = None
    else:
        kernel_cols = _sample_active_dims(d, cfg)
    return _CallShape(d, kernel_name, systematic, chain_names, chain_ops, P, N, P_C, B, kernel_cols)


@dataclass
class _EpisodePriors:
    """Per-episode kernel, hyperparameters, noise and mean function (B independent draws)."""

    kernel_obj: gpytorch.kernels.Kernel
    params: Dict[str, Tensor]
    component_params: List[Dict[str, Tensor]]  # systematic chains only
    likelihood: gpytorch.likelihoods.GaussianLikelihood
    mean_module: gpytorch.means.Mean
    mean_params: Dict[str, Tensor]


def _sample_episode_priors(cfg: HasDataConfig, shape: _CallShape, device: Device) -> _EpisodePriors:
    """Draw every episode's kernel hyperparameters, noise and mean function."""
    B = shape.B
    component_params: List[Dict[str, Tensor]] = []
    if shape.systematic:
        kernel_obj, component_params, outer_sign_params = _build_kernel_chain(
            cfg,
            shape.chain_names,
            shape.chain_ops,
            shape.k,
            B,
            device,
            active_dims=shape.kernel_cols,
            d_total=shape.d,
        )
        # Chains fill the flat schema with 0.0 (per-component values are in
        # component_params); outer sign-modulation params stay in the flat schema.
        params = {
            key: torch.zeros(B, device=device)
            for key in (
                "l",
                "alpha2",
                "period",
                "rq_alpha",
                "power",
                "l_b",
                "alpha2_b",
                "period_b",
                "rq_alpha_b",
                "power_b",
            )
        }
        params.update(outer_sign_params)
    else:
        kernel_obj, params = _sample_episode_kernel(
            cfg, shape.kernel_name, shape.k, B, device, active_dims=shape.kernel_cols, d_total=shape.d
        )
    likelihood = _build_likelihood(cfg, shape.kernel_name, B, device)
    mean_module, mean_params = _sample_mean_module(cfg, shape.d, B, device)
    return _EpisodePriors(kernel_obj, params, component_params, likelihood, mean_module, mean_params)


@dataclass
class _Features:
    x_norm: Tensor  # (B, T, d) model-visible, z-scored per episode
    x_kernel: Tensor  # (B, T, d) what the kernel and mean see (hidden warp of x_norm)
    mlp_mixed: Optional[Tensor]  # (B,) gates, only with kernel metadata
    kernel_hidden_applied: Optional[Tensor]


def _sample_features(cfg: HasDataConfig, shape: _CallShape, device: Device, with_gates: bool) -> _Features:
    """Features (B, T, d) ~ N(0, 1), warped and normalised per episode, plus the kernel's hidden view."""
    x_raw = torch.randn(shape.B, shape.T, shape.d, device=device)
    x_raw = tabiclv2_warp_features(x_raw)
    x_raw = apply_structural_feature_warp(x_raw, cfg, device)
    mlp_mixed = kernel_hidden_applied = None
    if with_gates:
        x_raw, mlp_mixed = apply_mlp_feature_mixing(x_raw, cfg, device, return_gate=True)
    else:
        x_raw = apply_mlp_feature_mixing(x_raw, cfg, device)
    x_norm = (x_raw - x_raw.mean(1, keepdim=True)) / x_raw.std(1, keepdim=True).clamp(min=1e-8)

    # The kernel and mean are evaluated on x_kernel, a hidden transform of x_norm
    # (identity unless kernel_hidden_enabled); the model sees x_norm.
    if with_gates:
        x_kernel, kernel_hidden_applied = apply_kernel_hidden_warp(x_norm, cfg, device, return_gate=True)
    else:
        x_kernel = apply_kernel_hidden_warp(x_norm, cfg, device)
    return _Features(x_norm, x_kernel, mlp_mixed, kernel_hidden_applied)


@dataclass
class _GPDraw:
    """One joint GP sample per episode and its exact (analytic) oracle and PIT."""

    x_norm_train: Tensor  # (B, P, d)
    x_norm_test: Tensor  # (B, N, d)
    x_norm_calib: Tensor  # (B, P_C, d), tabicl_split PIT context only
    x_kernel_train: Tensor  # oracle-only, never returned
    x_kernel_test: Tensor
    y_train: Tensor  # (B, P)
    y_test: Tensor  # (B, N)
    y_calib: Tensor  # (B, P_C)
    L_ff: Tensor  # (B, P, P) Cholesky of the training covariance
    alpha: Tensor  # (B, P) K_ff^{-1} (y_train - mean_train)
    mu_star: Tensor  # (B, N) prior mean at the test points
    sigma_star: Tensor  # (B, N) prior std
    R_star: Tensor  # (B, N, N) prior correlation
    R_prior: Tensor  # (B, N, N) same as R_star, kept for the schema
    Sigma_full: Tensor  # (B, N, N) R_star rescaled by sigma_star
    z_train: Tensor  # (B, P) exact LOO PIT
    z_test: Tensor  # (B, N) exact posterior PIT
    log_pdf_test: Tensor  # (B, N)
    discard: Tensor  # (B,) bool, Cholesky failures


def _draw_gp(
    cfg: HasDataConfig, shape: _CallShape, priors: _EpisodePriors, feats: _Features, device: Device
) -> Optional[_GPDraw]:
    """Sample y jointly from each episode's GP; None when kernel evaluation fails for the whole batch."""
    B, P, N, T = shape.B, shape.P, shape.N, shape.T
    x_norm, x_kernel = feats.x_norm, feats.x_kernel
    likelihood, mean_module = priors.likelihood, priors.mean_module

    # Joint prior covariance (B, T, T): dense kernel + nugget on the diagonal. Only
    # kernel evaluation can raise; factorization happens per episode in
    # _psd_safe_batch.
    with gpytorch.settings.max_cholesky_size(_MAX_CHOLESKY):
        try:
            K_full_dense = _evaluate_kernel_dense(priors.kernel_obj, x_kernel)  # (B, T, T), no nugget yet
        except (NotPSDError, torch.linalg.LinAlgError):
            warnings.warn(
                f"_generate_gp_batch_raw: kernel evaluation for this "
                f"{B}-episode batch (kernel={shape.kernel_name!r}) raised NotPSDError "
                f"or LinAlgError; discarding the whole batch and resampling.",
                RuntimeWarning,
            )
            return None
    nugget_eye = torch.eye(T, device=device, dtype=K_full_dense.dtype).expand(B, T, T)
    K_all_raw = K_full_dense + likelihood.noise.reshape(B, 1, 1) * nugget_eye

    # K_all = L_all L_all^T from a PSD-repaired Cholesky, so the sample y_all and the
    # reported covariances come from the same PSD matrix.
    L_all, failed_all = _psd_safe_batch(K_all_raw)
    K_all = L_all @ L_all.mT  # (B, T, T), PSD by construction
    y_all = (L_all @ torch.randn(B, T, 1, device=device)).squeeze(-1)  # zero-mean GP sample
    # Add the mean function (evaluated on x_kernel).
    y_all = y_all + mean_module(x_kernel)

    x_kernel_train = x_kernel[:, :P]
    x_kernel_test = x_kernel[:, P : P + N]
    y_train = y_all[:, :P]
    y_test = y_all[:, P : P + N]

    # --- Sub-matrices of K_all (nugget already on diagonal) ---
    K_ff = K_all[:, :P, :P]  # (B, P, P) -- P_C never enters K_ff/LOO/oracle
    K_ss = K_all[:, P : P + N, P : P + N]  # (B, N, N)

    # LOO PIT needs L_ff and alpha = K_ff^{-1} (y_train - mean_train).
    L_ff, failed_ff = _batched_cholesky(K_ff)
    mean_train = mean_module(x_kernel_train)  # (B, P)
    alpha = torch.cholesky_solve((y_train - mean_train).unsqueeze(-1), L_ff).squeeze(-1)  # (B, P)

    oracle_mode = getattr(cfg.data, "oracle_mode", "prior")
    if oracle_mode == "prior":
        # Prior oracle: R_star is the prior correlation of the test block of K_all.
        # The copula target is the posterior correlation R_post (see z_test below).
        mu_star = mean_module(x_kernel_test)
        Sigma_star = K_ss
    else:
        raise ValueError(f"Unknown data.oracle_mode '{oracle_mode}'; only 'prior' is supported.")
    Sigma_star = 0.5 * (Sigma_star + Sigma_star.permute(0, 2, 1))

    # sigma_to_correlation (batched)
    var_diag = Sigma_star.diagonal(dim1=1, dim2=2).clamp(min=1e-10)  # (B, N)
    sigma_star = var_diag.sqrt()
    inv_s = var_diag.rsqrt()
    R_star = Sigma_star * inv_s.unsqueeze(1) * inv_s.unsqueeze(2)  # (B, N, N)
    d_diag = R_star.diagonal(dim1=1, dim2=2).clamp(min=1e-10).sqrt()
    R_star = R_star / (d_diag.unsqueeze(1) * d_diag.unsqueeze(2))

    # Prior correlation among the test points (same as R_star; kept for the schema).
    prior_var = K_ss.diagonal(dim1=1, dim2=2).clamp(min=1e-10)  # (B, N)
    prior_inv = prior_var.rsqrt()
    R_prior = K_ss * prior_inv.unsqueeze(1) * prior_inv.unsqueeze(2)  # (B, N, N)
    pd_diag = R_prior.diagonal(dim1=1, dim2=2).clamp(min=1e-10).sqrt()
    R_prior = R_prior / (pd_diag.unsqueeze(1) * pd_diag.unsqueeze(2))

    # LOO PIT for z_train; diag(K_ff^{-1}) is the column-wise squared norm of L_ff^{-1}.
    eye_P = torch.eye(P, device=device)
    L_inv = torch.linalg.solve_triangular(L_ff, eye_P.unsqueeze(0).expand(B, -1, -1), upper=False)  # (B, P, P)
    K_inv_diag = (L_inv**2).sum(dim=1).clamp(min=1e-12)  # (B, P)
    z_train = alpha * K_inv_diag.rsqrt()  # (B, P)

    # Posterior PIT for z_test: standardize by the GP posterior marginals
    # N(mu_post_i, Sigma_post_ii), matching what a TabICL marginal conditioned on
    # the context gives. mu_star/sigma_star stay prior quantities. Only diag of the
    # Schur complement is used (each entry >= nugget).
    K_sf = K_all[:, P : P + N, :P]  # (B, N, P)
    V_sf = torch.linalg.solve_triangular(L_ff, K_sf.mT, upper=False)  # (B, P, N)
    mu_post = mu_star + torch.bmm(K_sf, alpha.unsqueeze(-1)).squeeze(-1)  # (B, N)
    var_post = K_ss.diagonal(dim1=1, dim2=2) - (V_sf**2).sum(dim=1)  # (B, N)
    var_post = var_post.clamp(min=likelihood.noise.reshape(B, 1).clamp(min=1e-10))
    sig_c = var_post.sqrt()
    z_test = (y_test - mu_post) / sig_c  # (B, N)
    log_pdf_test = -0.5 * math.log(2.0 * math.pi) - sig_c.log() - 0.5 * z_test**2  # (B, N)

    # Full prior covariance at the test points (for the Y-space oracle).
    Sigma_full = R_star * sigma_star.unsqueeze(1) * sigma_star.unsqueeze(2)  # (B, N, N)

    return _GPDraw(
        x_norm_train=x_norm[:, :P],
        x_norm_test=x_norm[:, P : P + N],
        x_norm_calib=x_norm[:, P + N :],
        x_kernel_train=x_kernel_train,
        x_kernel_test=x_kernel_test,
        y_train=y_train,
        y_test=y_test,
        y_calib=y_all[:, P + N :],
        L_ff=L_ff,
        alpha=alpha,
        mu_star=mu_star,
        sigma_star=sigma_star,
        R_star=R_star,
        R_prior=R_prior,
        Sigma_full=Sigma_full,
        z_train=z_train,
        z_test=z_test,
        log_pdf_test=log_pdf_test,
        discard=failed_all | failed_ff,
    )


def _degenerate_episodes(draw: _GPDraw, shape: _CallShape, x_norm: Tensor) -> Tensor:
    """(B,) bool: episodes with a degenerate LOO z or whose every active kernel column is near-constant."""
    B = shape.B
    non_finite = ~torch.isfinite(draw.z_train).all(dim=1)
    z_std = draw.z_train.std(dim=1)
    degen = non_finite | (z_std < 0.1) | (z_std > 3.0)
    if degen.any():
        warnings.warn(
            f"generate_gp_batch: {int(degen.sum())}/{B} episodes have degenerate LOO z "
            f"({int(non_finite.sum())} non-finite) and will be discarded.",
            RuntimeWarning,
        )

    # Every active kernel dimension collapsed to a constant would make R_star constant.
    active_cols = shape.kernel_cols if shape.kernel_cols is not None else list(range(shape.d))
    active_stds = x_norm[:, :, active_cols].std(dim=1)  # (B, len(active_cols))
    degenerate_active_col = (active_stds.max(dim=1).values) < 1e-4
    if degenerate_active_col.any():
        warnings.warn(
            f"generate_gp_batch: {int(degenerate_active_col.sum())}/{B} episodes have a "
            f"degenerate (near-constant) active kernel column and will be discarded.",
            RuntimeWarning,
        )
    return degen | degenerate_active_col


def _backend_pit_per_episode(
    cfg: HasDataConfig,
    draw: _GPDraw,
    shape: _CallShape,
    device: Device,
    marginal_backend: str,
    marginal_regressor: Any,
    k_folds: int,
    probs_n: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """K-fold PIT, episode by episode, through a backend without a batched module (slow)."""
    from eval.metrics.joint_nll import compute_pit
    from eval.spatial.marginal_backends import loo_pit as _backend_loo_pit
    from eval.spatial.marginal_backends import quantiles as _backend_quantiles

    B, P, N = shape.B, shape.P, shape.N
    y_train, y_test = draw.y_train, draw.y_test
    y_mean = y_train.mean(dim=1, keepdim=True)
    y_std = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
    probs = np.linspace(1.0 / (probs_n + 1), probs_n / (probs_n + 1), probs_n)
    base_seed = int(getattr(cfg, "seed", None) or 0)
    z_train_np = np.empty((B, P), dtype=np.float32)
    z_test_np = np.empty((B, N), dtype=np.float32)
    log_pdf_np = np.empty((B, N), dtype=np.float32)
    for b in range(B):
        xc = draw.x_norm_train[b].detach().cpu().numpy()
        xq = draw.x_norm_test[b].detach().cpu().numpy()
        y_std_b = float(y_std[b])
        yc = ((y_train[b] - y_mean[b]) / y_std[b]).detach().cpu().numpy()
        yq = ((y_test[b] - y_mean[b]) / y_std[b]).detach().cpu().numpy()
        seed_b = (base_seed + b) % (2**31)
        z_train_np[b] = _backend_loo_pit(
            marginal_backend,
            marginal_regressor,
            xc,
            yc,
            probs,
            k_folds=k_folds,
            seed=seed_b,
        )
        q_test = _backend_quantiles(marginal_backend, marginal_regressor, xc, yc, xq, probs, seed=seed_b)
        z_test_b, log_pdf_b = compute_pit(q_test, probs, yq)
        z_test_np[b] = z_test_b
        # Jacobian back to raw-y nats.
        log_pdf_np[b] = log_pdf_b - math.log(y_std_b)
    return (
        torch.from_numpy(z_train_np).to(device=device),
        torch.from_numpy(z_test_np).to(device=device),
        torch.from_numpy(log_pdf_np).to(device=device),
    )


def _marginal_pit(
    cfg: HasDataConfig,
    draw: _GPDraw,
    shape: _CallShape,
    device: Device,
    *,
    apply_tabicl: bool,
    tabicl_model: Optional[TabICLLike],
    tabicl_k_folds: int,
    tabicl_split_calib_frac: float,
    marginal_backend: Optional[str],
    marginal_regressor: Any,
    marginal_probs_n: int,
    raw_y_override: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """(z_train, z_test, log_pdf_test): the exact PIT, or its replacement by a marginal model.

    Targets are z-scored per episode before a marginal model sees them, and its
    log-densities are converted back to raw-y nats.
    """
    z_train, z_test, log_pdf_test = draw.z_train, draw.z_test, draw.log_pdf_test
    y_train, y_test, y_calib = draw.y_train, draw.y_test, draw.y_calib
    if raw_y_override:
        # "y_train": z_train is the z-scored target; z_test/log_pdf_test stay analytic.
        y_mean = y_train.mean(dim=1, keepdim=True)
        y_std = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
        z_train = ((y_train - y_mean) / y_std).detach()
    elif marginal_backend in _BATCHED_MARGINAL_BACKENDS:
        # Batched PIT for a non-TabICL backend: (k_folds + 1) fused forwards for the call.
        y_mean = y_train.mean(dim=1, keepdim=True)
        y_std = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
        y_train_s = ((y_train - y_mean) / y_std).detach().cpu().numpy()
        y_test_s = ((y_test - y_mean) / y_std).detach().cpu().numpy()
        x_train_np = draw.x_norm_train.detach().cpu().numpy()
        x_test_np = draw.x_norm_test.detach().cpu().numpy()
        base_seed = int(getattr(cfg, "seed", None) or 0)
        _run_batched = _BATCHED_MARGINAL_BACKENDS[marginal_backend]()
        out = _run_batched(
            marginal_regressor,
            x_train_np,
            y_train_s,
            x_test_np,
            y_test_s,
            k_folds=tabicl_k_folds,
            probs_n=marginal_probs_n,
            seed=base_seed,
        )
        z_train = torch.from_numpy(out["z_train"]).to(device=device)
        z_test = torch.from_numpy(out["z_test"]).to(device=device)
        # Jacobian back to raw-y nats.
        log_pdf_test = torch.from_numpy(out["log_pdf_test"]).to(device=device) - y_std.log()  # (B,N) - (B,1) broadcast
    elif marginal_backend is not None and marginal_backend != "tabicl":
        z_train, z_test, log_pdf_test = _backend_pit_per_episode(
            cfg, draw, shape, device, marginal_backend, marginal_regressor, tabicl_k_folds, marginal_probs_n
        )
    elif apply_tabicl and tabicl_split_calib_frac > 0:
        # "tabicl_split": one forward with the calibration points as context.
        from copula_inter.pit import run_pit_calib_split_batched  # local: pit.py imports from this module

        assert tabicl_model is not None

        y_mean = y_train.mean(dim=1, keepdim=True)
        y_std = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
        y_train_scaled = ((y_train - y_mean) / y_std).unsqueeze(-1)  # (B, P, 1)
        y_calib_scaled = ((y_calib - y_mean) / y_std).unsqueeze(-1)  # (B, P_C, 1)
        split_pit = run_pit_calib_split_batched(
            tabicl_model,
            draw.x_norm_train,
            y_train_scaled,
            draw.x_norm_calib,
            y_calib_scaled,
            Y_query_raw=y_train.unsqueeze(-1),
            Y_calib_raw=y_calib.unsqueeze(-1),
        )
        z_train = split_pit["z_train"].squeeze(-1)  # (B, P)
    elif apply_tabicl:
        # "tabicl": K-fold PIT on the training points and a full-context PIT on the
        # real test points; z_train, z_test and log_pdf_test all come from TabICL.
        from copula_inter.pit import run_pit_batched  # local: pit.py imports from this module

        assert tabicl_model is not None

        y_mean = y_train.mean(dim=1, keepdim=True)
        y_std = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
        y_train_scaled = ((y_train - y_mean) / y_std).unsqueeze(-1)  # (B, P, 1)
        y_test_scaled = ((y_test - y_mean) / y_std).unsqueeze(-1)  # (B, N, 1)
        tabicl_pit = run_pit_batched(
            tabicl_model,
            draw.x_norm_train,
            y_train_scaled,
            draw.x_norm_test,
            y_test_scaled,
            k_folds=tabicl_k_folds,
            Y_train_raw=y_train.unsqueeze(-1),
        )
        z_train = tabicl_pit["z_train"].squeeze(-1)  # (B, P)
        z_test = tabicl_pit["z_test"].squeeze(-1)  # (B, N)
        # Jacobian back to raw-y nats: log p_raw = log p_scaled - log(std).
        log_pdf_test = tabicl_pit["log_pdf_test"].squeeze(-1) - y_std.log()  # (B, N) - (B, 1) broadcast
    return z_train, z_test, log_pdf_test


# Flat per-episode hyperparameter keys saved with kernel metadata.
_FLAT_KERNEL_KEYS = [
    "l",
    "alpha2",
    "period",
    "rq_alpha",
    "power",
    "l_b",
    "alpha2_b",
    "period_b",
    "rq_alpha_b",
    "power_b",
    "sign_applied_outer",
    "sign_w_outer",
    "sign_b_outer",
    "sign_a_outer",
]
_MEAN_KEYS = (
    "mean_weight",
    "mean_bias",
    "mean_nonzero",
    "mean_family",
    "mean_linear",
    "mean_exp_direction",
    "mean_exp_rate",
    "mean_exp_scale",
    "mean_anomaly_direction",
    "mean_anomaly_threshold",
    "mean_anomaly_magnitude",
)


def _kernel_metadata(
    shape: _CallShape, priors: _EpisodePriors, feats: _Features, draw: _GPDraw
) -> tuple[Dict[str, Tensor], Dict[str, object]]:
    """Per-episode hyperparameters and factors, plus the call-shared kernel name and active dims."""
    flat_keys = list(_FLAT_KERNEL_KEYS)
    if not shape.systematic:
        # Chains carry per-component sign fields in kernel_component_params instead.
        flat_keys += ["sign_applied", "sign_w", "sign_b", "sign_a"]
        if _parse_composite(shape.kernel_name) is not None:
            flat_keys += ["sign_applied_b", "sign_w_b", "sign_b_b", "sign_a_b"]
    assert feats.mlp_mixed is not None and feats.kernel_hidden_applied is not None
    tensors = {key: priors.params[key].cpu() for key in flat_keys}
    tensors["nugget"] = priors.likelihood.noise.reshape(shape.B).cpu()  # name kept for the saved schema
    tensors["mlp_mixed"] = feats.mlp_mixed.cpu()
    tensors["kernel_hidden_applied"] = feats.kernel_hidden_applied.cpu()
    tensors["x_kernel_train"] = draw.x_kernel_train.cpu()
    tensors["x_kernel_test"] = draw.x_kernel_test.cpu()
    for key in _MEAN_KEYS:
        tensors[key] = priors.mean_params[key].cpu()
    tensors["_L_ff"] = draw.L_ff
    tensors["_alpha"] = draw.alpha
    extra: Dict[str, object] = {
        "kernel": shape.kernel_name,
        "kernel_feature_indices": torch.tensor(
            shape.kernel_cols if shape.kernel_cols is not None else list(range(shape.d)), dtype=torch.long
        ),
    }
    return tensors, extra


@torch.no_grad()
def _generate_gp_batch_raw(
    cfg: HasDataConfig,
    B: int,
    device: Device = "cpu",
    *,
    return_kernel_metadata: bool = False,
    d_override: Optional[int] = None,
    tabicl_model: Optional[TabICLLike] = None,
    tabicl_k_folds: int = 10,
    tabicl_split_calib_frac: float = 0.0,
    kernel_weights: Optional[Tensor] = None,
    tabicl_mix_weights: Optional[Tensor] = None,
    marginal_backend: Optional[str] = None,
    marginal_regressor: Any = None,
    marginal_probs_n: int = 99,
    raw_y_override: bool = False,
) -> List[Dict[str, Tensor]]:
    """Generate up to B GP episodes in one vectorized call.

    All episodes share one kernel family, (P, N) and active_dims, with independent
    hyperparameters, noise and features. Episodes whose Cholesky fails are
    dropped, so fewer than B may be returned; use generate_gp_batch for exactly B.
    If cfg.seed is set all RNGs are reseeded from it.

    Args:
        cfg: config.
        B: number of episodes.
        device: "cpu" or "cuda".
        return_kernel_metadata: also return kernel name, hyperparameters,
            kernel_feature_indices, mlp_mixed and the _L_ff/_alpha factors.
        tabicl_model: replace z_train with this TabICL's K-fold PIT.
        tabicl_k_folds: K for that PIT.
        tabicl_split_calib_frac: > 0 uses a calibration-split PIT with an extra
            round(frac * P) context points instead of K-fold; features are then
            normalized over train + test + calibration points.
        kernel_weights: _COMPOSABLE_KERNELS-ordered sampling weights.
        tabicl_mix_weights: per-family probability that the z_train override
            applies to this call (None = always).
        marginal_backend: a non-TabICL marginal (eval/spatial/marginal_backends)
            whose PIT replaces z_train, z_test and log_pdf_test.
        marginal_regressor: that backend's regressor.
        marginal_probs_n: quantile grid size for marginal_backend.
        raw_y_override: z_train = per-episode z-scored y_train (no PIT).

    Returns:
        list of episode dicts.
    """
    seed = getattr(cfg, "seed", None)
    if seed is not None:
        seed_everything(seed)

    shape = _sample_call_shape(cfg, B, device, d_override, kernel_weights, tabicl_split_calib_frac)
    priors = _sample_episode_priors(cfg, shape, device)
    feats = _sample_features(cfg, shape, device, with_gates=return_kernel_metadata)
    draw = _draw_gp(cfg, shape, priors, feats, device)
    if draw is None:
        return []
    discard = draw.discard | _degenerate_episodes(draw, shape, feats.x_norm)

    # With tabicl_mix_weights the TabICL override applies with the kernel's mix probability.
    if tabicl_mix_weights is not None:
        apply_tabicl = tabicl_model is not None and (
            random.random() < _tabicl_mix_prob_for_kernel(shape.kernel_name, tabicl_mix_weights)
        )
    else:
        apply_tabicl = tabicl_model is not None
    z_train, z_test, log_pdf_test = _marginal_pit(
        cfg,
        draw,
        shape,
        device,
        apply_tabicl=apply_tabicl,
        tabicl_model=tabicl_model,
        tabicl_k_folds=tabicl_k_folds,
        tabicl_split_calib_frac=tabicl_split_calib_frac,
        marginal_backend=marginal_backend,
        marginal_regressor=marginal_regressor,
        marginal_probs_n=marginal_probs_n,
        raw_y_override=raw_y_override,
    )
    # Optional z_train corruption, skipped when this call used the adaptive TabICL mix.
    if not (tabicl_mix_weights is not None and apply_tabicl):
        z_train = corrupt_z_train(z_train, cfg.data)

    # --- Pack into list of dicts (single D→H transfer) ---
    tensors = {
        "x_norm_train": draw.x_norm_train.cpu(),
        "x_norm_test": draw.x_norm_test.cpu(),
        "y_train": draw.y_train.cpu(),
        "y_test": draw.y_test.cpu(),
        "z_train": z_train.cpu(),
        "z_test": z_test.cpu(),
        "log_pdf_test": log_pdf_test.cpu(),
        "R_star": draw.R_star.cpu(),
        "R_prior": draw.R_prior.cpu(),
        "Sigma_star": draw.Sigma_full.cpu(),
        "mu_star": draw.mu_star.cpu(),
        "sigma_star": draw.sigma_star.cpu(),
    }

    # Discard any episode with a non-finite saved field.
    non_finite = torch.zeros(shape.B, dtype=torch.bool)
    for _t in tensors.values():
        non_finite = non_finite | ~_t.reshape(_t.shape[0], -1).isfinite().all(dim=1)
    if non_finite.any():
        warnings.warn(
            f"generate_gp_batch: {int(non_finite.sum())}/{shape.B} episodes contain "
            f"NaN/Inf in a saved field and will be discarded.",
            RuntimeWarning,
        )
    discard = discard | non_finite.to(discard.device)

    extra: Dict[str, object] = {"n_train": torch.tensor(shape.P), "n_test": torch.tensor(shape.N)}
    if return_kernel_metadata:
        meta_tensors, meta_extra = _kernel_metadata(shape, priors, feats, draw)
        tensors.update(meta_tensors)
        extra.update(meta_extra)

    # Drop discarded episodes from the per-episode tensors (the extra fields are call-shared).
    chains = (
        (shape.chain_names, shape.chain_ops, priors.component_params)
        if return_kernel_metadata and shape.systematic
        else None
    )
    return assemble_episodes(tensors, extra, discard, chains)


def generate_gp_batch(
    cfg: HasDataConfig,
    B: int,
    device: Device = "cpu",
    *,
    return_kernel_metadata: bool = False,
    tabicl_model: Optional[TabICLLike] = None,
    tabicl_k_folds: int = 10,
    tabicl_split_calib_frac: float = 0.0,
    d_override: Optional[int] = None,
    kernel_weights: Optional[Tensor] = None,
    tabicl_mix_weights: Optional[Tensor] = None,
    marginal_backend: Optional[str] = None,
    marginal_regressor: Any = None,
    marginal_probs_n: int = 99,
    raw_y_override: bool = False,
) -> List[Dict[str, Tensor]]:
    """Generate exactly B GP episodes, topping up discarded ones with further calls.

    Top-up calls resample kernel, P, N and active_dims but keep d (pinned to
    d_override if given, else to the first call's d). Raises after max_rounds.
    """
    base_seed = getattr(cfg, "seed", None)
    episodes = _generate_gp_batch_raw(
        cfg,
        B,
        device,
        return_kernel_metadata=return_kernel_metadata,
        d_override=d_override,
        tabicl_model=tabicl_model,
        tabicl_k_folds=tabicl_k_folds,
        tabicl_split_calib_frac=tabicl_split_calib_frac,
        kernel_weights=kernel_weights,
        tabicl_mix_weights=tabicl_mix_weights,
        marginal_backend=marginal_backend,
        marginal_regressor=marginal_regressor,
        marginal_probs_n=marginal_probs_n,
        raw_y_override=raw_y_override,
    )
    # Pin every top-up round to the first round's d (or d_override).
    d_fixed: Optional[int]
    if episodes:
        d_fixed = int(episodes[0]["x_norm_train"].shape[-1])
    else:
        d_fixed = d_override
    max_rounds = 20
    for round_idx in range(1, max_rounds + 1):
        if len(episodes) >= B:
            break
        shortfall = B - len(episodes)
        if base_seed is not None:
            # Offset the seed per round so retries draw new kernels.
            setattr(cfg, "seed", base_seed + round_idx * 104_729)
        new_episodes = _generate_gp_batch_raw(
            cfg,
            shortfall,
            device,
            return_kernel_metadata=return_kernel_metadata,
            d_override=d_fixed,
            tabicl_model=tabicl_model,
            tabicl_k_folds=tabicl_k_folds,
            tabicl_split_calib_frac=tabicl_split_calib_frac,
            kernel_weights=kernel_weights,
            tabicl_mix_weights=tabicl_mix_weights,
            marginal_backend=marginal_backend,
            marginal_regressor=marginal_regressor,
            marginal_probs_n=marginal_probs_n,
            raw_y_override=raw_y_override,
        )
        if d_fixed is None and new_episodes:
            d_fixed = int(new_episodes[0]["x_norm_train"].shape[-1])
        episodes += new_episodes
    if base_seed is not None:
        setattr(cfg, "seed", base_seed)
    if len(episodes) < B:
        raise RuntimeError(
            f"generate_gp_batch: could not assemble {B} valid episodes after "
            f"{max_rounds} top-up rounds ({len(episodes)} obtained) — the kernel/config "
            f"combination in this call appears to be persistently non-PSD."
        )
    return episodes[:B]
