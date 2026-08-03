"""
pit.py — Probability Integral Transform via the frozen TabICL marginal CDF.

For each target dimension j, TabICL's quantile head provides a conditional
predictive distribution over y given (x, context).  Evaluating that
distribution's CDF at the true target maps observations to Uniform(0, 1):

    u_{i,j} = F̂_j(y_{i,j} | x_i, context)

and a probit transform sends them to standard normal Z-space:

    z_{i,j} = Φ⁻¹(u_{i,j}).

Leakage prevention for the training instances is done by **K-fold
partitioning**: the train set is split into K disjoint folds, and for each
fold the held-out points are queried against TabICL using the remaining
K−1 folds as context.  K is small and fixed (default 10) — true LOO
(K = P) is much more accurate but ~K_loo / K_default times slower at
dataset-generation time.  The test instances use the entire training set
as context (single forward pass).

This file makes **no modifications** to ``tabicl_upstream`` — leakage is
handled purely by which points are passed in which forward call.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_TABICL_SRC = os.path.join(_REPO_ROOT, "tabicl_upstream", "src")
if _TABICL_SRC not in sys.path:
    sys.path.insert(0, _TABICL_SRC)

from data_gen import build_kernel_fn, _safe_cholesky  # noqa: E402

DEFAULT_K_FOLDS = 10


def _optional_param(t: torch.Tensor):
    """Unpack a possibly-ARD-vector task hyperparameter (see data_gen's 0.0
    "not applicable" sentinel convention): None if every entry is the
    sentinel, else a python float (scalar) or the raw tensor (ARD vector)."""
    if torch.all(t == 0.0):
        return None
    return t.item() if t.numel() == 1 else t


def _mean_train_from_task(task: dict, x: torch.Tensor) -> torch.Tensor:
    """Reconstruct mean_module(x) from a task dict's saved mean-bank params
    (see data_gen._MeanFunctionBank / _sample_mean_module) -- same formula,
    unbatched. Needed so gp_analytical_pit's LOO alpha can be residualized
    against the same mean the episode was actually generated with (see
    data_gen._generate_gp_batch_raw's alpha computation for why: R&W Eq.
    5.12 is derived for a zero-mean joint Gaussian). Older tasks saved before
    "mean_*" existed default to an all-zero (no-op) mean, same convention as
    the sign-modulation fields above.
    """
    zero = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    if not bool(task.get("mean_nonzero", torch.tensor(False)).item()):
        return zero

    family = int(task["mean_family"].item())
    if family == 0:  # linear (incl. constant-only, when mean_weight is all-zero)
        return (x * task["mean_weight"]).sum(-1) + task["mean_bias"]
    elif family == 1:  # exponential
        proj = (x * task["mean_exp_direction"]).sum(-1)
        exponent = torch.clamp(task["mean_exp_rate"] * proj, min=-10.0, max=10.0)
        return task["mean_exp_scale"] * torch.exp(exponent)
    elif family == 2:  # sparse anomaly
        proj = (x * task["mean_anomaly_direction"]).sum(-1)
        hit = (proj > task["mean_anomaly_threshold"]).to(x.dtype)
        return hit * task["mean_anomaly_magnitude"]
    raise ValueError(f"Unknown mean_family {family}; expected 0, 1, or 2.")


# ---------------------------------------------------------------------------
# TabICL loader
# ---------------------------------------------------------------------------


def load_tabicl(ckpt_name: str, device: str) -> nn.Module:
    """Download (if needed) and load a frozen TabICL regressor."""
    from huggingface_hub import hf_hub_download
    from tabicl._model.tabicl import TabICL  # type: ignore[import]

    ckpt_path = hf_hub_download(repo_id="jingang/TabICL", filename=ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    base = TabICL(**checkpoint["config"])
    base.load_state_dict(checkpoint["state_dict"])
    for p in base.parameters():
        p.requires_grad_(False)
    base.eval()
    base.to(device)
    return base


def _probit(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Clamp u to (eps, 1-eps) then apply Φ⁻¹ via erfinv."""
    u = u.clamp(eps, 1.0 - eps)
    return torch.erfinv(2.0 * u - 1.0) * math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Single-task PIT
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_pit(
    tabicl: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    k_folds: int = DEFAULT_K_FOLDS,
    eps: float = 1e-6,
) -> dict:
    """Run the Probability Integral Transform on one task.

    Args:
        tabicl  : frozen TabICL regressor (max_classes=0)
        X_train : (P, p_x)
        Y_train : (P, d)
        X_test  : (N, p_x)
        Y_test  : (N, d)
        k_folds : number of disjoint folds for the training-set PIT.
                  Bounded above by P; clamp into [2, P].
                  Set to P explicitly for true leave-one-out (slow).
        eps     : clamp before probit.

    Returns dict with:
        z_train      : (P, d)
        z_test       : (N, d)
        log_pdf_test : (N, d)   marginal log-densities at Y_test
    """
    device = X_train.device
    P, p_x = X_train.shape
    N = X_test.shape[0]
    d = Y_train.shape[1]

    K = max(2, min(int(k_folds), P))

    # ------------------------------------------------------------------ #
    # A) Test instances: one forward over the full train context, fused #
    #    across the d target dimensions on the batch axis.                #
    # ------------------------------------------------------------------ #
    X_concat = torch.cat([X_train, X_test], dim=0)                       # (P+N, p_x)
    X_test_batch = X_concat.unsqueeze(0).expand(d, -1, -1).contiguous()  # (d, P+N, p_x)
    y_train_batch = Y_train.permute(1, 0).contiguous()                   # (d, P)

    logits = tabicl(X_test_batch, y_train_batch)                         # (d, N, Q)
    Q = logits.shape[-1]
    dist = tabicl.quantile_dist(logits.reshape(d * N, Q))

    y_test_flat = Y_test.permute(1, 0).reshape(d * N)
    u_test = dist.cdf(y_test_flat).reshape(d, N).permute(1, 0)           # (N, d)
    log_pdf_test = dist.log_prob(y_test_flat).reshape(d, N).permute(1, 0)  # (N, d)

    # ------------------------------------------------------------------ #
    # B) Training instances: K disjoint folds (fixed K, ≪ P)              #
    # ------------------------------------------------------------------ #
    fold_size = math.ceil(P / K)
    u_train = torch.empty(P, d, device=device, dtype=Y_train.dtype)
    indices = torch.arange(P, device=device)

    for k in range(K):
        start = k * fold_size
        end = min(start + fold_size, P)
        if start >= end:
            break

        qry_idx = indices[start:end]
        ctx_mask = torch.ones(P, dtype=torch.bool, device=device)
        ctx_mask[qry_idx] = False
        ctx_idx = indices[ctx_mask]
        F = qry_idx.numel()

        X_fold = torch.cat([X_train[ctx_idx], X_train[qry_idx]], dim=0)    # (P-F+F, p_x)
        X_fold_batch = X_fold.unsqueeze(0).expand(d, -1, -1).contiguous()
        y_ctx_batch = Y_train[ctx_idx].permute(1, 0).contiguous()          # (d, P-F)

        logits_fold = tabicl(X_fold_batch, y_ctx_batch)                    # (d, F, Q)
        dist_fold = tabicl.quantile_dist(logits_fold.reshape(d * F, Q))

        y_qry_flat = Y_train[qry_idx].permute(1, 0).reshape(d * F)
        u_train[qry_idx, :] = (
            dist_fold.cdf(y_qry_flat).reshape(d, F).permute(1, 0)
        )

    z_train = _probit(u_train, eps)
    z_test = _probit(u_test, eps)

    return {
        "z_train": z_train,
        "z_test": z_test,
        "log_pdf_test": log_pdf_test,
    }


# ---------------------------------------------------------------------------
# Analytical GP PIT (no model inference required)
# ---------------------------------------------------------------------------


@torch.no_grad()
def gp_analytical_pit(task: dict, eps: float = 1e-6) -> dict:
    """Exact PIT from GP LOO (train) and posterior (test) marginals.

    Since all data is generated from a GP with known hyperparameters, the
    marginal CDFs are available in closed form — no learned regressor needed.

    Test instances:
        y_test[i] | D_train ~ N(mu_star[i], sigma_star[i]²)  (exact)
        z_test[i] = (y_test[i] - mu_star[i]) / sigma_star[i]

    Training instances — exact GP LOO (Rasmussen & Williams, GPML Eq. 5.12),
    derived for a zero-mean joint Gaussian, so alpha uses the mean-residual
    (y_train - mean_train), not y_train directly, whenever the episode's
    mean bank (see data_gen._MeanFunctionBank) is non-zero:
        sigma²_i^LOO  = 1 / [K_ff⁻¹]_ii
        z_train[i]    = alpha_i / sqrt([K_ff⁻¹]_ii)
        where alpha = K_ff⁻¹ (y_train - mean_train), mean_train = mean_module(x_train)

    Cost: one O(P³) Cholesky per episode vs. O(K × P × forward_pass) for
    the TabICL K-fold approach.

    Args:
        task: raw task dict returned by generate_gp_task (must contain
              kernel, l, alpha2, nugget, period, rq_alpha, power, l_b,
              alpha2_b, period_b, rq_alpha_b, power_b, kernel_feature_indices,
              x_norm_train, y_train, y_test, mu_star, sigma_star, and the
              mean_* fields from data_gen._sample_mean_module — see
              _mean_train_from_task).
        eps:  unused (kept for API symmetry with run_pit).

    Returns dict with z_train (P,), z_test (N,), log_pdf_test (N,).
    """
    kernel_name = task["kernel"]
    # scalar, unless the episode was generated ARD (cfg.data.ard=True for
    # rbf/matern32/periodic/rational_quadratic), in which case l is a
    # per-dimension lengthscale vector (k,) — see data_gen._build_scaled_kernel.
    l_tensor = task["l"]
    l      = l_tensor.item() if l_tensor.numel() == 1 else l_tensor
    alpha2 = task["alpha2"].item()
    nugget = task["nugget"].item()
    # 0.0 sentinel means the param is not applicable for this kernel. period
    # is likewise a per-dimension vector under periodic+ARD (gpytorch's
    # PeriodicKernel ties period_length's ard_num_dims to lengthscale's).
    period   = _optional_param(task["period"])
    rq_alpha = task["rq_alpha"].item() if task["rq_alpha"].item() != 0.0 else None
    # "polynomial"'s integer degree — same 0.0 sentinel convention (its own
    # default is never 0, see data_gen's poly_power_min/max).
    power    = task["power"].item() if task["power"].item() != 0.0 else None
    # Composite ("A+B"/"A*B") kernels' second component — same 0.0 sentinel
    # convention. Omitting these previously made build_kernel_fn silently
    # reconstruct composites with l_b/alpha2_b=None, crashing with a
    # TypeError as soon as component B's kernel function tried to use them.
    # l_b/period_b can be ARD vectors too, same as l/period above, whenever
    # component B is one of the ARD-eligible base kernels under cfg.data.ard.
    l_b        = _optional_param(task["l_b"])
    alpha2_b   = task["alpha2_b"].item() if task["alpha2_b"].item() != 0.0 else None
    period_b   = _optional_param(task["period_b"])
    rq_alpha_b = task["rq_alpha_b"].item() if task["rq_alpha_b"].item() != 0.0 else None
    power_b    = task["power_b"].item() if task["power_b"].item() != 0.0 else None

    # Sign-modulation hyperplanes (see data_gen.SignModulatedKernel /
    # cfg.data.sign_modulation_component_prob / sign_modulation_outer_prob):
    # gated on the explicit sign_applied*/1.0 sentinel rather than
    # _optional_param's "all entries are 0.0" check, since sign_w is a
    # random N(0, I_k) draw that isn't guaranteed nonzero even when applied
    # (unlike l/period/etc., whose priors never actually produce exactly 0).
    # Absent entirely for episodes saved before this feature existed (older
    # datasets on disk) -- task.get(..., 0.0) defaults to "not applied".
    # sign_a (the tanh sharpness -- see SignModulatedKernel) may itself be
    # absent even when sign_applied*==1.0, for datasets saved by the earlier
    # hard-sign() version of this feature (which had no sharpness knob);
    # data_gen._wrap_concrete_sign_modulated substitutes a very large `a` in
    # that case, numerically recovering the hard sign() those episodes were
    # actually generated with, so their saved z_train/z_test still round-trip.
    def _sign_pair(applied_key: str, w_key: str, b_key: str, a_key: str):
        applied = task.get(applied_key)
        if applied is None or applied.item() == 0.0:
            return None, None, None
        return task[w_key], task[b_key], task.get(a_key)

    sign_w, sign_b, sign_a = _sign_pair("sign_applied", "sign_w", "sign_b", "sign_a")
    sign_w_b, sign_b_b, sign_a_b = _sign_pair("sign_applied_b", "sign_w_b", "sign_b_b", "sign_a_b")
    sign_w_outer, sign_b_outer, sign_a_outer = _sign_pair(
        "sign_applied_outer", "sign_w_outer", "sign_b_outer", "sign_a_outer"
    )

    # active_dims (gpytorch's own kernel kwarg) lets kernel_fn take the
    # full-width x_norm_train straight through and select its k active
    # columns internally — same mechanism data_gen.generate_gp_task uses,
    # so no manual column slicing is needed here either.
    cols = task["kernel_feature_indices"].tolist()
    kernel_fn = build_kernel_fn(
        kernel_name, l, alpha2, period=period, rq_alpha=rq_alpha, power=power,
        l_b=l_b, alpha2_b=alpha2_b, period_b=period_b, rq_alpha_b=rq_alpha_b, power_b=power_b,
        active_dims=cols,
        sign_w=sign_w, sign_b=sign_b, sign_a=sign_a,
        sign_w_b=sign_w_b, sign_b_b=sign_b_b, sign_a_b=sign_a_b,
        sign_w_outer=sign_w_outer, sign_b_outer=sign_b_outer, sign_a_outer=sign_a_outer,
    )
    x_k_train = task["x_norm_train"]   # (P, d_features)
    y_train    = task["y_train"]                  # (P,)
    y_test     = task["y_test"]                   # (N,)
    mu_star    = task["mu_star"]                  # (N,) posterior mean
    sigma_star = task["sigma_star"]               # (N,) posterior marginal std

    # --- Test: posterior marginals are exact Gaussians ---
    sig_clamped  = sigma_star.clamp(min=1e-8)
    z_test       = (y_test - mu_star) / sig_clamped
    log_pdf_test = (
        -0.5 * math.log(2.0 * math.pi)
        - sig_clamped.log()
        - 0.5 * z_test**2
    )

    # --- Train: exact GP LOO (R&W Eq. 5.12) ---
    # Reuse L_ff and alpha from generate_gp_task when available (B: no double Cholesky).
    # Fall back to kernel reconstruction for tasks loaded from disk.
    P = y_train.shape[0]
    if "_L_ff" in task and "_alpha" in task:
        L     = task["_L_ff"]
        alpha = task["_alpha"]
    else:
        K_ff       = kernel_fn(x_k_train, x_k_train) + nugget * torch.eye(P, device=y_train.device)
        L          = _safe_cholesky(K_ff)
        mean_train = _mean_train_from_task(task, x_k_train)
        alpha      = torch.cholesky_solve((y_train - mean_train).unsqueeze(-1), L).squeeze(-1)  # (P,)

    # diag(K_ff^{-1}) = column-wise squared-norm of L^{-1}
    L_inv      = torch.linalg.solve_triangular(
        L, torch.eye(P, device=L.device, dtype=L.dtype), upper=False
    )                                                                      # (P, P)
    K_inv_diag = (L_inv**2).sum(dim=0).clamp(min=1e-12)                   # (P,)
    z_train    = alpha * K_inv_diag.rsqrt()                               # alpha_i/√[K⁻¹]_ii

    return {"z_train": z_train, "z_test": z_test, "log_pdf_test": log_pdf_test}


# ---------------------------------------------------------------------------
# z_train corruption (training-time robustness augmentation)
# ---------------------------------------------------------------------------
#
# Motivation (see plots/plot_spatial_correlation_diagnostics.py's real-mode
# diagnostic): CopulaTabICL is trained exclusively on the EXACT closed-form
# GP-LOO whitened residual (gp_analytical_pit above / data_gen.py's batched
# equivalent), but at deployment on any dataset without a known generating
# kernel (e.g. real ERA5), z_train can only be estimated via run_pit's K-fold
# TabICL-marginal quantile PIT -- measured to recover only
# corr(z_exact, z_tabicl) ~= 0.6-0.65 with the true whitened residual, flat
# across k_folds (not a fold-size artifact). The trained model turned out to
# have essentially zero tolerance for this: real-context predictions
# collapsed toward ~0 correlation (near-neighbour truth 0.97 -> predicted
# 0.05-0.13). This is a train/deploy distribution-shift problem, not a
# lengthscale-prior or evaluation-methodology problem (both were ruled out).
#
# Fix: blend z_train toward i.i.d. N(0, 1) noise during training so the model
# never gets to rely on it being perfectly whitened -- standard input-noise-
# augmentation logic, with the blend strength (see DEFAULT_Z_CORRUPTION_RHO_
# BETA_A/B below) calibrated to the measured ~0.6-0.65 real-world signal
# correlation above. This is safe here specifically because
# cfg.data.oracle_mode="prior" decouples the training TARGET
# (R_star = kernel(x_test, x_test)) from z_train's realized values entirely,
# so corrupting z_train never requires inventing a different loss target --
# the task stays "predict the same R_star", just from a noisier context
# signal.
DEFAULT_Z_CORRUPTION_RHO_BETA_A = 2.0
DEFAULT_Z_CORRUPTION_RHO_BETA_B = 3.0


def corrupt_z_train(
    z_train: torch.Tensor,
    train_mask: torch.Tensor,
    data_cfg,
) -> torch.Tensor:
    """Randomly corrupt a batch of exact GP-LOO z_train toward i.i.d. N(0, 1)
    noise, per data_cfg.z_train_corruption_* knobs (see conf/data/gp_tasks.yaml).
    No-op (returns z_train unchanged) unless
    data_cfg.z_train_corruption_enabled is True.

    Per corrupted episode:
        z_corrupted = sqrt(rho) * z_train + sqrt(1 - rho) * N(0, 1)
    where rho ~ Beta(a, b) is the per-episode "signal fraction" (rho=1 leaves
    z_train untouched; rho=0 fully replaces it with pure noise). Since
    z_train and the i.i.d. noise term are independent and both unit-variance,
    this construction gives corr(z_train, z_corrupted) = sqrt(rho) in
    expectation, which is what the default Beta(2, 3) shape (E[sqrt(rho)]
    ~= 0.61, p10~=0.14, p90~=0.68) is calibrated against -- centered near the
    measured TabICL-marginal-PIT signal correlation (~0.6-0.65) with spread
    toward both a near-clean and a more severely corrupted regime.

    Args:
        z_train    : (B, P_max) exact GP-LOO whitened residual.
        train_mask : (B, P_max) bool, True at valid (non-padding) context
                     positions -- collate_fn pads z_train with zeros past
                     each episode's true P, so the corrupted output is
                     re-masked to keep padding at exactly zero (matching the
                     uncorrupted convention downstream code -- e.g. loss
                     masking -- already relies on).
        data_cfg   : cfg.data (Hydra DictConfig) -- see conf/data/gp_tasks.yaml
                     for the z_train_corruption_* keys this reads. Lives under
                     cfg.data (not cfg.training) since it's a data-generation-
                     time modulation, same as sign_modulation_component_prob/
                     mlp_mixing_enabled/etc -- read fresh every call so both
                     live_generation and Hydra CLI overrides (e.g. an oarsub
                     command's data.z_train_corruption_enabled=true) apply
                     without any separate wiring.

    Returns:
        (B, P_max) corrupted z_train, same dtype/device as the input.
    """
    if not bool(data_cfg.get("z_train_corruption_enabled", False)):
        return z_train

    prob = float(data_cfg.get("z_train_corruption_prob", 0.5))
    beta_a = float(data_cfg.get("z_train_corruption_rho_beta_a", DEFAULT_Z_CORRUPTION_RHO_BETA_A))
    beta_b = float(data_cfg.get("z_train_corruption_rho_beta_b", DEFAULT_Z_CORRUPTION_RHO_BETA_B))
    if prob <= 0.0:
        return z_train

    B, P = z_train.shape
    device = z_train.device
    mask_f = train_mask.to(dtype=z_train.dtype)

    apply_ep = torch.rand(B, device=device) < prob  # (B,) which episodes get corrupted at all
    if not bool(apply_ep.any()):
        return z_train

    rho = torch.distributions.Beta(beta_a, beta_b).sample((B,)).to(device=device, dtype=z_train.dtype)
    noise = torch.randn(B, P, device=device, dtype=z_train.dtype)

    sqrt_rho = rho.clamp(0.0, 1.0).sqrt().unsqueeze(-1)
    sqrt_1m_rho = (1.0 - rho).clamp(0.0, 1.0).sqrt().unsqueeze(-1)
    z_blend = sqrt_rho * z_train + sqrt_1m_rho * noise

    z_out = torch.where(apply_ep.unsqueeze(-1), z_blend, z_train)
    return z_out * mask_f  # re-zero padding regardless of which branch fed it
