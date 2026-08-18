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

``run_pit_calib_split_batched`` offers a cheaper alternative to the K-fold
rotation: leakage is instead avoided by drawing a separate calibration point
set that is disjoint from the training set by construction (never a K-fold
rotation of the same pool), so every training point can be scored against it
in a single forward pass. See its docstring for the cost/quality trade-off.

This file makes **no modifications** to ``tabicl_upstream`` — leakage is
handled purely by which points are passed in which forward call.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

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


def resolve_pit_ckpt(cfg) -> str | None:
    """Which checkpoint (if any) to load as a frozen TabICL marginal for
    PIT-ing episodes the way real (non-GP) deployment data would be seen.

    Shared by every caller that needs this resolution (train.py's z_train
    sim-to-real diagnostic, generate_pit_dataset.py's data.z_train_source=
    tabicl/tabicl_split, live_dataset.py's live-generation equivalent) — was
    previously train.py-local (train.py::_resolve_pit_ckpt), moved here so
    live_dataset.py can reuse it without importing from train.py (which
    itself imports live_dataset.py, so the reverse import would cycle).

    This is a separate model instance from the backbone a given run actually
    trains, so its checkpoint is its own knob (tabicl.pit_ckpt) rather than
    reusing tabicl.pretrained/tabicl.ckpt (which describe that run's own
    backbone) — a from-scratch backbone (tabicl.pretrained=false, e.g.
    copula_nano) can still opt in by setting pit_ckpt explicitly, since
    PIT-ing episodes with a released checkpoint's quantile head doesn't
    depend on the run's own architecture. Defaults to tabicl.ckpt when the
    backbone itself is pretrained (copula_prod's original behaviour), else
    None (diagnostic/override off) unless pit_ckpt is set explicitly.
    """
    pit_ckpt = cfg.tabicl.get("pit_ckpt", None)
    if pit_ckpt is None and bool(cfg.tabicl.get("pretrained", True)):
        pit_ckpt = cfg.tabicl.get("ckpt", None)
    return pit_ckpt


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


def normalize_targets(
    y_train: torch.Tensor, y_test: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Z-score targets to the scale the frozen TabICL quantile head expects.

    ``run_pit`` (and the raw TabICL module it wraps) does no target scaling
    of its own — unlike ``tabicl.TabICLRegressor.fit()``, which always fits
    a fresh StandardScaler on y before calling this same underlying model
    (see ``tabicl_upstream/.../_sklearn/regressor.py``). Every call site
    that hands ``y`` to ``run_pit`` (or the raw module directly) must
    replicate that scaling first, or absolute-scale targets (e.g. real-world
    units, or a synthetic draw with a random GammaPrior-drawn outputscale)
    saturate the frozen quantile head's CDF into its extreme tail for every
    point alike, collapsing the returned PIT residuals'/quantiles' spread
    instead of reflecting the true per-point rank.

    ``y_test``, if given, is scaled with ``y_train``'s own mean/std (never
    its own) — mirrors ``TabICLRegressor.fit()``, which fits its scaler on
    training data only, and matches every real deployment where test
    targets are unknown at normalization time.

    Args:
        y_train : (P,) training targets, raw scale.
        y_test  : optional (N,) test targets, raw scale.

    Returns:
        (y_train_scaled, y_test_scaled_or_None, mean, std). ``mean``/``std``
        are needed to un-scale any raw-y-unit output (e.g. a quantile
        value) or Jacobian-correct any log-density output computed from the
        scaled call: ``log p_raw(y) = log p_scaled(y_scaled) - log(std)``.
    """
    mean = y_train.mean()
    std = y_train.std().clamp(min=1e-8)
    y_train_scaled = (y_train - mean) / std
    y_test_scaled = (y_test - mean) / std if y_test is not None else None
    return y_train_scaled, y_test_scaled, mean, std


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
# Batch-of-episodes PIT (dataset-generation use)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_pit_batched(
    tabicl: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    k_folds: int = DEFAULT_K_FOLDS,
    eps: float = 1e-6,
) -> dict:
    """``run_pit``, vectorised over a leading batch-of-episodes axis B.

    Only valid when every episode in the batch shares the same P and N —
    true for one ``data_gen._generate_gp_batch_raw`` call (all B episodes in
    a shard-generation batch share P/N by construction), which is the
    intended caller. The B and target-dim (d) axes are folded together into
    TabICL's own batch axis (mirrors how ``run_pit`` already folds d alone),
    so this costs one (B*d)-batched forward pass per fold instead of B
    separate single-episode ``run_pit`` calls — B*(K+1)x fewer Python-level
    TabICL invocations, though the K-fold loop's iteration count (and hence
    wall-clock scaling in K) is unchanged.

    Args:
        tabicl  : frozen TabICL regressor (max_classes=0)
        X_train : (B, P, p_x)
        Y_train : (B, P, d)
        X_test  : (B, N, p_x)
        Y_test  : (B, N, d)
        k_folds : as in ``run_pit`` — clamped into [2, P], shared by every
                  episode in the batch since P is shared.
        eps     : clamp before probit.

    Returns dict with:
        z_train      : (B, P, d)
        z_test       : (B, N, d)
        log_pdf_test : (B, N, d)   marginal log-densities at Y_test
    """
    device = X_train.device
    B, P, p_x = X_train.shape
    N = X_test.shape[1]
    d = Y_train.shape[2]

    K = max(2, min(int(k_folds), P))

    # ------------------------------------------------------------------ #
    # A) Test instances: one forward, batch axis = B*d.                   #
    # ------------------------------------------------------------------ #
    X_concat = torch.cat([X_train, X_test], dim=1)                              # (B, P+N, p_x)
    X_test_batch = (
        X_concat.unsqueeze(1).expand(B, d, P + N, p_x).reshape(B * d, P + N, p_x).contiguous()
    )
    y_train_batch = Y_train.permute(0, 2, 1).reshape(B * d, P).contiguous()     # (B*d, P)

    logits = tabicl(X_test_batch, y_train_batch)                                # (B*d, N, Q)
    Q = logits.shape[-1]
    dist = tabicl.quantile_dist(logits.reshape(B * d * N, Q))

    y_test_flat = Y_test.permute(0, 2, 1).reshape(B * d * N)
    u_test = dist.cdf(y_test_flat).reshape(B, d, N).permute(0, 2, 1)            # (B, N, d)
    log_pdf_test = dist.log_prob(y_test_flat).reshape(B, d, N).permute(0, 2, 1)  # (B, N, d)

    # ------------------------------------------------------------------ #
    # B) Training instances: K disjoint folds (fixed K, ≪ P), batch axis  #
    #    = B*d, fold membership shared across the batch since P is.       #
    # ------------------------------------------------------------------ #
    fold_size = math.ceil(P / K)
    u_train = torch.empty(B, P, d, device=device, dtype=Y_train.dtype)
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

        X_fold = torch.cat([X_train[:, ctx_idx], X_train[:, qry_idx]], dim=1)   # (B, P-F+F, p_x)
        X_fold_batch = (
            X_fold.unsqueeze(1).expand(B, d, X_fold.shape[1], p_x)
            .reshape(B * d, X_fold.shape[1], p_x).contiguous()
        )
        y_ctx_batch = (
            Y_train[:, ctx_idx].permute(0, 2, 1).reshape(B * d, P - F).contiguous()
        )

        logits_fold = tabicl(X_fold_batch, y_ctx_batch)                        # (B*d, F, Q)
        dist_fold = tabicl.quantile_dist(logits_fold.reshape(B * d * F, Q))

        y_qry_flat = Y_train[:, qry_idx].permute(0, 2, 1).reshape(B * d * F)
        u_train[:, qry_idx, :] = (
            dist_fold.cdf(y_qry_flat).reshape(B, d, F).permute(0, 2, 1)
        )

    z_train = _probit(u_train, eps)
    z_test = _probit(u_test, eps)

    return {
        "z_train": z_train,
        "z_test": z_test,
        "log_pdf_test": log_pdf_test,
    }


# ---------------------------------------------------------------------------
# Calibration-split PIT (single forward pass, no K-fold rotation)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_pit_calib_split_batched(
    tabicl: nn.Module,
    X_query: torch.Tensor,
    Y_query: torch.Tensor,
    X_calib: torch.Tensor,
    Y_calib: torch.Tensor,
    eps: float = 1e-6,
) -> dict:
    """One-pass alternative to ``run_pit_batched``'s K-fold query-side PIT.

    ``X_calib``/``Y_calib`` is a *separate* point set (disjoint from
    ``X_query`` by construction, e.g. extra points drawn from the same
    episode's generating process but never part of the official training
    set) used as TabICL's context. Every query point is then scored against
    that single context in one forward pass — no leakage, since a query
    point was never a member of its own context, so no fold rotation is
    needed. This is exactly ``run_pit_batched``'s part A (the test-side PIT,
    which already scores ``X_test`` against the full ``X_train`` context in
    one pass), generalised to an arbitrary context/query pair.

    Cost: 1 forward pass total, vs. ``k_folds`` for ``run_pit_batched``'s
    K-fold query-side PIT. Trade-off: every query point shares the same,
    fixed-size context, rather than K-fold's near-full-pool context per
    point — quality depends on how large/informative ``X_calib`` is (see
    ``conf/data/gp_tasks.yaml``'s ``z_train_split_calib_frac``).

    Args:
        tabicl  : frozen TabICL regressor (max_classes=0)
        X_query : (B, P_Q, p_x)
        Y_query : (B, P_Q, d) — only used to evaluate the CDF, never
                  passed to TabICL as context.
        X_calib : (B, P_C, p_x)
        Y_calib : (B, P_C, d)
        eps     : clamp before probit.

    Returns dict with:
        z_train : (B, P_Q, d)
    """
    device = X_query.device
    B, P_Q, p_x = X_query.shape
    P_C = X_calib.shape[1]
    d = Y_query.shape[2]

    X_concat = torch.cat([X_calib, X_query], dim=1)                          # (B, P_C+P_Q, p_x)
    X_batch = (
        X_concat.unsqueeze(1).expand(B, d, P_C + P_Q, p_x).reshape(B * d, P_C + P_Q, p_x).contiguous()
    )
    y_calib_batch = Y_calib.permute(0, 2, 1).reshape(B * d, P_C).contiguous()  # (B*d, P_C)

    logits = tabicl(X_batch, y_calib_batch)                                  # (B*d, P_Q, Q)
    Q = logits.shape[-1]
    dist = tabicl.quantile_dist(logits.reshape(B * d * P_Q, Q))

    y_query_flat = Y_query.permute(0, 2, 1).reshape(B * d * P_Q)
    u_query = dist.cdf(y_query_flat).reshape(B, d, P_Q).permute(0, 2, 1)     # (B, P_Q, d)
    z_train = _probit(u_query, eps)

    return {"z_train": z_train}


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
