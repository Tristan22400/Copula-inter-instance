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

from data_gen import build_kernel_fn, _safe_cholesky, sigma_to_correlation  # noqa: E402

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
    # TabICL's InferenceManager auto-batches large forward calls and, under
    # low free GPU memory, may offload its output to CPU regardless of the
    # input device (see inference.py's _resolve_offload_mode) -- re-sync
    # onto `device` so quantile_dist's internal tensors never end up on a
    # different device than y_test_flat/y_qry_flat below.
    logits = logits.to(device)
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
        logits_fold = logits_fold.to(device)  # see run_pit's offload-mode comment above
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
    # See run_pit's offload-mode comment: TabICL's InferenceManager can return
    # its output on CPU under GPU memory pressure regardless of input device.
    logits = logits.to(device)
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
        logits_fold = logits_fold.to(device)  # see run_pit's offload-mode comment above
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
    # See run_pit's offload-mode comment: TabICL's InferenceManager can return
    # its output on CPU under GPU memory pressure regardless of input device.
    logits = logits.to(device)
    Q = logits.shape[-1]
    dist = tabicl.quantile_dist(logits.reshape(B * d * P_Q, Q))

    y_query_flat = Y_query.permute(0, 2, 1).reshape(B * d * P_Q)
    u_query = dist.cdf(y_query_flat).reshape(B, d, P_Q).permute(0, 2, 1)     # (B, P_Q, d)
    z_train = _probit(u_query, eps)

    return {"z_train": z_train}


# ---------------------------------------------------------------------------
# Analytical GP PIT (no model inference required)
# ---------------------------------------------------------------------------


def _sign_triple(d: dict, applied_key: str, w_key: str, b_key: str, a_key: str):
    """(sign_w, sign_b, sign_a) from a dict's sign_applied* 0.0/1.0 sentinel,
    or (None, None, None) if not applied -- shared by _kernel_fn_from_task's
    flat schema and _kernel_fn_from_chain_task's per-component schema (see
    data_gen.SignModulatedKernel / cfg.data.sign_modulation_component_prob /
    sign_modulation_outer_prob). Gated on the explicit sign_applied* sentinel
    rather than an "all entries are 0.0" check, since sign_w is a random
    N(0, I_k) draw that isn't guaranteed nonzero even when applied (unlike
    l/period/etc., whose priors never actually produce exactly 0). Absent
    entirely for episodes saved before this feature existed -- d.get(...)
    defaults to "not applied". sign_a (the tanh sharpness) may itself be
    absent even when sign_applied*==1.0, for datasets saved by the earlier
    hard-sign() version of this feature (no sharpness knob yet);
    data_gen._wrap_concrete_sign_modulated substitutes a very large `a` in
    that case, numerically recovering the hard sign() those episodes were
    actually generated with.
    """
    applied = d.get(applied_key)
    if applied is None or applied.item() == 0.0:
        return None, None, None
    return d[w_key], d[b_key], d.get(a_key)


def _kernel_fn_from_chain_task(task: dict):
    """Reconstruct (kernel_fn, nugget) for a systematic-composition chain
    episode (cfg.data.systematic_composition=True, this repo's default —
    see data_gen.py's "Systematic composition" docstring section and
    generate_gp_batch's return_kernel_metadata handling).

    Builds each chain link as its own single-component build_kernel_fn call
    from task["kernel_component_params"][i] (one dict per component, already
    per-episode-indexed — see _build_kernel_chain/_build_kernel_component),
    then combines the resulting dense kernel matrices left-to-right per
    task["kernel_ops"] ("+"/"*") — the exact same left-to-right dense
    combination data_gen._DenseComposedKernel used to build the kernel this
    episode was actually generated from, just replayed here as plain tensor
    ops instead of a gpytorch Kernel object.

    Component param dicts only carry the keys relevant to that component's
    kernel type (see data_gen._build_scaled_kernel: "period"/"rq_alpha" are
    present only when that kernel family uses them, no 0.0-sentinel filler
    the way the flat/non-systematic schema does) — handled below via `in`
    checks before falling back to build_kernel_fn's own None defaults.

    Raises NotImplementedError if the whole-chain "outer" sign-modulation
    wrap (cfg.data.sign_modulation_outer_prob, applied once to the fully
    composed chain — see data_gen.SignModulatedKernel /
    _wrap_concrete_sign_modulated) was actually used for this episode: that
    wrap needs applying AFTER combination, which isn't implemented here.
    Defaults to 0.0 (off) in every existing config, so this only fires for
    episodes deliberately generated with that feature turned on.
    """
    names = task["kernel_components"]
    ops = task["kernel_ops"]
    comp_params = task["kernel_component_params"]
    nugget = task["nugget"].item()
    cols = task["kernel_feature_indices"].tolist()

    sign_w_outer, sign_b_outer, sign_a_outer = _sign_triple(
        task, "sign_applied_outer", "sign_w_outer", "sign_b_outer", "sign_a_outer"
    )
    if sign_w_outer is not None:
        raise NotImplementedError(
            "_kernel_fn_from_chain_task: whole-chain outer sign modulation "
            "(cfg.data.sign_modulation_outer_prob) is not supported for "
            "systematic-composition chain reconstruction."
        )

    component_fns = []
    for name, params in zip(names, comp_params):
        l_t = params["l"]
        l = l_t.item() if l_t.numel() == 1 else l_t
        alpha2 = params["alpha2"].item()
        period = _optional_param(params["period"]) if "period" in params else None
        rq_alpha = (
            params["rq_alpha"].item() if "rq_alpha" in params and params["rq_alpha"].item() != 0.0 else None
        )
        power = (
            params["power"].item() if "power" in params and params["power"].item() != 0.0 else None
        )
        sign_w, sign_b, sign_a = _sign_triple(params, "sign_applied", "sign_w", "sign_b", "sign_a")
        component_fns.append(build_kernel_fn(
            name, l, alpha2, period=period, rq_alpha=rq_alpha, power=power,
            active_dims=cols, sign_w=sign_w, sign_b=sign_b, sign_a=sign_a,
        ))

    def kernel_fn(X1, X2):
        K = component_fns[0](X1, X2)
        for op, fn in zip(ops, component_fns[1:]):
            Ki = fn(X1, X2)
            K = K + Ki if op == "+" else K * Ki
        return K

    return kernel_fn, nugget


def _kernel_fn_from_task(task: dict):
    """Reconstruct (kernel_fn, nugget) from a task dict's saved kernel
    metadata (return_kernel_metadata=True schema — see data_gen.generate_gp_task
    / generate_gp_batch's return_kernel_metadata handling). Shared by
    gp_analytical_pit (train-side LOO) and gp_analytical_posterior (test-side
    Schur-complement conditioning) so both reconstruct the exact same kernel.

    Dispatches to _kernel_fn_from_chain_task for systematic-composition
    chain episodes (identified by "kernel_components" in task) — those don't
    fit the flat l/alpha2/l_b/alpha2_b schema handled below at all.
    """
    if "kernel_components" in task:
        return _kernel_fn_from_chain_task(task)

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

    sign_w, sign_b, sign_a = _sign_triple(task, "sign_applied", "sign_w", "sign_b", "sign_a")
    sign_w_b, sign_b_b, sign_a_b = _sign_triple(task, "sign_applied_b", "sign_w_b", "sign_b_b", "sign_a_b")
    sign_w_outer, sign_b_outer, sign_a_outer = _sign_triple(
        task, "sign_applied_outer", "sign_w_outer", "sign_b_outer", "sign_a_outer"
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
    return kernel_fn, nugget


@torch.no_grad()
def gp_analytical_pit(task: dict, eps: float = 1e-6) -> dict:
    """Exact PIT from GP LOO (train) and posterior (test) marginals.

    Since all data is generated from a GP with known hyperparameters, the
    marginal CDFs are available in closed form — no learned regressor needed.

    Test instances — the exact GP POSTERIOR marginals, conditioned on the
    realized (x_train, y_train):
        mu_post[i]  = mean(x_test[i]) + [K_sf K_ff^-1 (y_train - mean_train)]_i
        var_post[i] = (K_ss)_ii - [K_sf K_ff^-1 K_fs]_ii
        y_test[i] | D_train ~ N(mu_post[i], var_post[i])   (exact)
        z_test[i]  = (y_test[i] - mu_post[i]) / sqrt(var_post[i])

    NOT (mu_star, sigma_star): under data.oracle_mode="prior" -- the only
    supported mode (data_gen.py's oracle_mode branch) -- those are the PRIOR
    mean/std, which is a different distribution. This function is the exact-GP
    stand-in for what a frozen TabICL supplies, and run_pit (above) calls
    TabICL with the context labels in-context, so its z_test/log_pdf_test are
    POSTERIOR PREDICTIVE quantities; standardizing by the prior here would make
    z_train_source="analytic" and "tabicl" two different problems rather than
    two estimates of the same z. It also changes the copula head's training
    target from the conditional correlation R_post to the context-blind prior
    correlation R_star -- see data_gen._generate_gp_batch_raw's "Posterior PIT
    for z_test" comment for that derivation.

    Only the DIAGONAL of the Schur complement is used, so this does not
    reintroduce the PSD failure that retired oracle_mode="posterior":
    Sigma_post = Cov(f|train) + nugget*I with the first term PSD, hence every
    diagonal entry is >= nugget > 0 with no eigenvalue repair needed.

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
              x_norm_train, x_norm_test, y_train, y_test, mu_star, and the
              mean_* fields from data_gen._sample_mean_module — see
              _mean_train_from_task). mu_star is read as the PRIOR mean at
              the test points (mean_module(x_norm_test)), the same way
              gp_analytical_posterior reads it.
        eps:  unused (kept for API symmetry with run_pit).

    Returns dict with z_train (P,), z_test (N,), log_pdf_test (N,).
    """
    kernel_fn, nugget = _kernel_fn_from_task(task)
    x_k_train = task["x_norm_train"]   # (P, d_features)
    x_k_test   = task["x_norm_test"]              # (N, d_features)
    y_train    = task["y_train"]                  # (P,)
    y_test     = task["y_test"]                   # (N,)
    mu_star    = task["mu_star"]                  # (N,) PRIOR mean at the test points

    # --- L_ff / alpha: needed by BOTH the test-side posterior marginals and
    # the train-side LOO, so resolved once up front. Reuse the factors
    # generate_gp_task cached when available (no double Cholesky); fall back to
    # kernel reconstruction for tasks loaded from disk.
    P = y_train.shape[0]
    if "_L_ff" in task and "_alpha" in task:
        L     = task["_L_ff"]
        alpha = task["_alpha"]
    else:
        K_ff       = kernel_fn(x_k_train, x_k_train) + nugget * torch.eye(P, device=y_train.device)
        L          = _safe_cholesky(K_ff)
        mean_train = _mean_train_from_task(task, x_k_train)
        alpha      = torch.cholesky_solve((y_train - mean_train).unsqueeze(-1), L).squeeze(-1)  # (P,)

    # --- Test: exact GP posterior marginals (see the docstring) ---
    # Same block conventions as gp_analytical_posterior: K_sf noise-free,
    # K_ss with the nugget on its diagonal. Only diag(K_ss - K_sf K_ff^-1 K_fs)
    # is formed, and it is bounded below by the nugget by construction.
    x_ref     = x_k_test.to(L.device)
    K_sf      = kernel_fn(x_ref, x_k_train.to(L.device))                       # (N, P)
    K_ss_diag = (
        kernel_fn(x_ref, x_ref).diagonal() + nugget
    )                                                                          # (N,)
    V_sf      = torch.linalg.solve_triangular(L, K_sf.T.to(L.dtype), upper=False)  # (P, N)
    mu_post   = mu_star.to(L.device) + K_sf @ alpha                            # (N,)
    var_post  = (K_ss_diag - (V_sf ** 2).sum(dim=0)).clamp(min=max(nugget, 1e-12))
    sig_clamped  = var_post.sqrt()
    z_test       = (y_test.to(L.device) - mu_post) / sig_clamped
    log_pdf_test = (
        -0.5 * math.log(2.0 * math.pi)
        - sig_clamped.log()
        - 0.5 * z_test**2
    )

    # --- Train: exact GP LOO (R&W Eq. 5.12) ---
    # diag(K_ff^{-1}) = column-wise squared-norm of L^{-1}
    L_inv      = torch.linalg.solve_triangular(
        L, torch.eye(P, device=L.device, dtype=L.dtype), upper=False
    )                                                                      # (P, P)
    K_inv_diag = (L_inv**2).sum(dim=0).clamp(min=1e-12)                   # (P,)
    z_train    = alpha * K_inv_diag.rsqrt()                               # alpha_i/√[K⁻¹]_ii

    return {"z_train": z_train, "z_test": z_test, "log_pdf_test": log_pdf_test}


def mvn_nll(y: torch.Tensor, mean: torch.Tensor, Sigma: torch.Tensor) -> float:
    """Full multivariate-normal negative log-likelihood of ``y`` under
    ``N(mean, Sigma)``, in raw-y units, run in float64 for the Cholesky
    solve (matches every other linear-algebra call site in this file).

    For any jointly-Gaussian predictive (mean, Sigma) — e.g. a fitted GP's
    own posterior/prior — this single formula already equals
    marginal-NLL + copula-NLL (Sklar's decomposition collapses to one term
    when the marginals are Gaussian), so callers with a Gaussian predictive
    don't need to split it via PIT/z-scoring at all.
    """
    y, mean, Sigma = y.double(), mean.double(), Sigma.double()
    L = _safe_cholesky(Sigma)
    resid = (y - mean).unsqueeze(-1)
    sol = torch.cholesky_solve(resid, L)
    quad = (resid * sol).sum()
    log_det = 2.0 * torch.log(torch.diagonal(L)).sum()
    n = y.shape[0]
    return (0.5 * (n * math.log(2.0 * math.pi) + log_det + quad)).item()


def mvn_nll_parts(y: torch.Tensor, mean: torch.Tensor, Sigma: torch.Tensor) -> dict:
    """Same quantity as mvn_nll, additionally split into its Sklar
    marginal/copula components (raw-sum units, i.e. NOT divided by n — same
    unnormalized convention as mvn_nll itself and as gp_analytical_posterior's
    nll_prior/nll_post; contrast with loss.gp_oracle_y_nll, the batched,
    per-point-normalized equivalent used for the GP-MLE/DKL baselines).

    marginal is the sum of each dimension's own univariate Gaussian NLL
    under (mean_i, Sigma_ii); copula is defined as total - marginal, which
    is exact by construction (Sklar's theorem collapses to one term for a
    jointly-Gaussian predictive — see mvn_nll's docstring), not a
    separately-verified quantity.
    """
    total = mvn_nll(y, mean, Sigma)
    y64, mean64 = y.double(), mean.double()
    std = Sigma.double().diagonal().clamp(min=1e-12).sqrt()
    z = (y64 - mean64) / std
    marginal = (0.5 * math.log(2.0 * math.pi) + std.log() + 0.5 * z ** 2).sum().item()
    return {"total": total, "marginal": marginal, "copula": total - marginal}


@torch.no_grad()
def gp_analytical_posterior(task: dict, eig_floor: float = 1e-6) -> dict:
    """Exact GP posterior correlation among test points, conditioned on the
    realized (x_train, y_train) via the Schur complement -- the "mechanism 2"
    term cfg.data.oracle_mode="prior" (data_gen.py's only supported mode)
    deliberately leaves out of R_star: R_star there is the raw/unconditional
    kernel correlation K_ss, never K_ss - K_sf K_ff^-1 K_fs. This function
    computes that missing conditioned quantity directly, for use as a true
    Bayes-optimal reference at EVAL time (see
    eval/runners/eval_checkpoint.py's "GP oracle total NLL (Y-space)"
    prior-vs-posterior report, printed by _print_y_space_oracle) — it does
    not touch training data generation or cfg.data.oracle_mode at all.

    Reuses the exact kernel _kernel_fn_from_task reconstructs (same one
    gp_analytical_pit's LOO z_train uses), and, when the task carries them
    (return_kernel_metadata=True — see generate_gp_batch), the same
    _L_ff/_alpha Cholesky factors gp_analytical_pit reuses, so this never
    repeats the O(P^3) factorization already paid for elsewhere.

    Supports both the flat and systematic-composition-chain kernel schemas
    (via _kernel_fn_from_task's dispatch) — raises NotImplementedError only
    for the rare case of whole-chain outer sign modulation on a chain
    episode (see _kernel_fn_from_chain_task's docstring; off by default in
    every existing config). Callers should catch NotImplementedError/KeyError
    and report "unavailable" rather than crash a whole eval run over one
    episode's kernel family.

    All linear algebra (the Schur complement itself, plus the eigendecomposition
    used for the PSD repair below) runs in float64 -- kernel evaluation stays
    in the kernel's native float32, matching every other call site in this
    file. This is the fix for the numerical gap that got cfg.data.oracle_mode
    ="posterior" removed from data_gen.py: that removed implementation *did*
    already use float64 for the Schur complement (then cast back to float32),
    yet still occasionally left the minimum eigenvalue of the result below
    the PSD floor for composite kernels, because nothing ever checked for
    it. Here, any residual eigenvalue below eig_floor is explicitly detected
    and clamped up (an eigenvalue-floor repair, not a discard) before
    converting to a correlation matrix, since an eval script wants a number
    for every episode, not a silently dropped one.

    eig_floor is RELATIVE to Sigma_post's own diagonal scale, not an
    absolute eigenvalue cutoff -- see the repair below. Non-stationary
    kernels (e.g. "polynomial", k = alpha2*(x1.x2+c)^d) can put K_ss's
    diagonal anywhere from O(1e3) to O(1e11) within a single episode; an
    absolute floor of 1e-6 -- fine for an O(1)-scale RBF/Matern posterior --
    is an absurdly overconfident "we know this to 1e-6 out of 1e11" claim
    once the repair fires on such an episode, and blows the Gaussian NLL's
    residual^2/(2*eigenvalue) term up by many orders of magnitude (this is
    what was producing oracle_diag/gap_nll around -109 / y_nll_oracle_posterior
    around 110 nats/point in training logs -- an artifact of this repair, not
    a real property of the posterior).

    The effective floor is actually max(eig_floor * scale, nugget), not the
    relative term alone. Reason: y_test = f_test + eps with eps ~ iid
    N(0, nugget) independent of training (see generate_gp_batch's K_all =
    K_full + likelihood.noise * I, the same nugget this function reads off
    the task below) -- so Sigma_post = Cov(f_test | train) + nugget*I, and
    since Cov(f_test | train) is itself PSD, EVERY eigenvalue of the true
    Sigma_post is provably >= nugget. This is a hard lower bound, not a
    heuristic, and it catches a failure mode the scale-relative term alone
    misses: a composite/chain kernel whose Sigma_post has small overall
    scale (so eig_floor*scale is tiny) can still develop a numerically
    near-zero or slightly negative eigenvalue from Schur-complement
    cancellation (K_ss - V.T @ V, two O(1)-ish quantities subtracted) even
    though the true eigenvalue can't be below nugget -- flooring only to
    eig_floor*scale there reintroduces the exact same
    residual^2/(2*eigenvalue) blowup this repair exists to prevent, just at
    a smaller absolute scale. Motivated by a z_train_source=tabicl
    live-generation run whose fixed 208-episode val set scored
    val/y_nll_oracle_posterior ~68 nats/point (oracle_diag/gap_nll ~-67,
    almost entirely in the copula term) even with the scale-relative-only
    floor in place -- the exact offending episode wasn't recovered (CUDA's
    RNG stream isn't reproducible process-to-process, see
    live_dataset.py's fixed-val-set seeding), so this fix is the
    mathematical guarantee above applied proactively rather than a
    confirmed root cause for that specific run.

    Returns dict with mu_post (N,), Sigma_post (N,N), R_post (N,N) — all
    float32 — plus min_eig (float, pre-repair, for diagnostics), repaired
    (bool, whether the eigenvalue floor actually fired), and nll_prior/
    nll_post (float, the total Y-space multivariate-normal NLL of y_test
    under the unconditioned prior N(mean_test, K_ss) vs. the conditioned
    posterior N(mu_post, Sigma_post) — see the comment above the
    `mvn_nll_parts` calls below for why THESE two numbers, not
    corr_nll_single(R_post, z_test), are the correct prior-vs-posterior
    comparison), plus nll_prior_marginal/nll_prior_copula and
    nll_post_marginal/nll_post_copula (float, mvn_nll_parts' Sklar split of
    the two totals above, same raw-sum units).
    """
    kernel_fn, nugget = _kernel_fn_from_task(task)
    # x_norm_train/test are always .cpu()'d before being packed into an
    # episode dict (see generate_gp_batch's tensors dict), but _L_ff/_alpha
    # are deliberately left device-resident for reuse (see that same
    # function's comment) — mismatched whenever the episode was
    # live-generated on GPU, so move x/y onto whatever device _L_ff/_alpha
    # already live on (falling back to x_train's own device when absent,
    # i.e. the recompute-from-scratch branch below, which never leaves CPU).
    ref_device = task["_L_ff"].device if "_L_ff" in task else task["x_norm_train"].device
    x_train = task["x_norm_train"].to(ref_device)   # (P, d), float32
    x_test  = task["x_norm_test"].to(ref_device)    # (N, d), float32
    y_train = task["y_train"].to(ref_device)         # (P,)
    P, N = x_train.shape[0], x_test.shape[0]

    if "_L_ff" in task and "_alpha" in task:
        L_ff  = task["_L_ff"].double()
        alpha = task["_alpha"].double()
    else:
        K_ff       = kernel_fn(x_train, x_train) + nugget * torch.eye(P, device=x_train.device)
        L_ff       = _safe_cholesky(K_ff).double()
        mean_train = _mean_train_from_task(task, x_train)
        alpha      = torch.cholesky_solve(
            (y_train - mean_train).double().unsqueeze(-1), L_ff
        ).squeeze(-1)

    # K_sf carries no noise term (measurement noise is independent across
    # distinct points, train vs. test included); K_ss does, on its diagonal
    # only, matching K_ff's own convention above and data_gen.py's K_all
    # (nugget added once to the full (T,T) diagonal before slicing out the
    # K_ff/K_ss blocks) — so this is the posterior over noisy y_test, the
    # same quantity oracle_mode="prior"'s R_star (also a K_ss slice of that
    # same K_all) already represents unconditionally.
    K_sf = kernel_fn(x_test, x_train).double()                                              # (N, P)
    K_ss = (kernel_fn(x_test, x_test) + nugget * torch.eye(N, device=x_test.device)).double()  # (N, N)

    V = torch.linalg.solve_triangular(L_ff, K_sf.T, upper=False)   # (P, N)
    Sigma_post = K_ss - V.T @ V
    Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)

    # mu_star under oracle_mode="prior" is exactly mean_module(x_test) (see
    # data_gen.py's oracle_mode branch: "mu_star = mean_module(x_norm_test)")
    # -- directly reusable as the prior mean term here, no separate
    # mean_module reconstruction needed.
    mean_test = task["mu_star"].to(ref_device).double()
    mu_post = mean_test + K_sf @ alpha

    # eig_floor scales with Sigma_post's own diagonal magnitude (see the
    # docstring) rather than acting as a fixed absolute cutoff -- otherwise
    # a non-stationary kernel whose K_ss diagonal legitimately spans many
    # orders of magnitude (e.g. "polynomial") gets floored to a fixed 1e-6
    # regardless of scale, which manufactures an enormous, meaningless
    # nll_post once any residual falls along that floored direction. Also
    # floored at `nugget` itself (see the docstring's PSD-decomposition
    # argument: Sigma_post = Cov(f_test|train) + nugget*I with the first
    # term PSD, so nugget is a hard, non-heuristic lower bound on every
    # eigenvalue) -- catches small-overall-scale composite kernels where
    # eig_floor*scale alone would floor below that hard bound.
    scale = Sigma_post.diagonal().abs().max().clamp(min=1e-12).item()
    eig_floor_eff = max(eig_floor * scale, nugget)
    eigvals = torch.linalg.eigvalsh(Sigma_post)
    min_eig = eigvals.min().item()
    repaired = min_eig < eig_floor_eff
    if repaired:
        eigvals_c, eigvecs = torch.linalg.eigh(Sigma_post)
        Sigma_post = eigvecs @ torch.diag(eigvals_c.clamp(min=eig_floor_eff)) @ eigvecs.T
        Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)

    R_post, _ = sigma_to_correlation(Sigma_post.float())

    # --- Total (marginal + copula) Y-space NLL, prior vs. posterior -------
    # Deliberately NOT scored via corr_nll_single(R_post, z_test): z_test is
    # standardized under the PRIOR's own (mu_star, sigma_star) (see
    # data_gen.py's oracle_mode="prior" branch), so it has unit marginal
    # variance only under the prior, not under the posterior -- Var(z_test |
    # x_train, y_train) = diag(Sigma_post)/sigma_star_prior^2 is generally
    # < 1 once conditioned, since conditioning shrinks variance (R&W §2.2).
    # Reusing that same unit-variance-assuming z against R_post's copula
    # formula (which assumes z ~ N(0, R) with unit marginal variance) scores
    # a distribution nothing was actually drawn from, and is NOT guaranteed
    # to be a lower bound relative to the prior's own z-space score -- this
    # was verified empirically (it comes out *worse*, i.e. a higher NLL,
    # than the prior on a real smoke test episode, the opposite of a true
    # lower bound). The full multivariate-normal log density below has no
    # such assumption -- N(mu, Sigma) at y_test is valid for ANY (mu, Sigma)
    # pair, prior or posterior alike -- so this is the sound version of
    # "posterior is a true lower bound": E[nll_post] <= E[nll_prior] holds
    # here because Bayesian conditioning on (x_train, y_train) is exactly
    # the NLL-minimizing update given that information (Bayes-optimality of
    # the posterior predictive under log-loss), with no unit-variance
    # assumption anywhere to violate.
    y_test = task["y_test"].to(ref_device).double()

    K_ss_sym = 0.5 * (K_ss + K_ss.T)
    prior_parts = mvn_nll_parts(y_test, mean_test, K_ss_sym)
    post_parts  = mvn_nll_parts(y_test, mu_post, Sigma_post)

    return {
        "mu_post":    mu_post.float(),
        "Sigma_post": Sigma_post.float(),
        "R_post":     R_post,
        "min_eig":    min_eig,
        "repaired":   repaired,
        "nll_prior":  prior_parts["total"],
        "nll_post":   post_parts["total"],
        # Marginal/copula split of the two totals above (same raw-sum units,
        # see mvn_nll_parts) — lets callers show the same per-episode
        # breakdown for the oracle that eval_baselines_episode/_eval_icl_episode
        # now expose for every fitted baseline and the ICL model.
        "nll_prior_marginal": prior_parts["marginal"],
        "nll_prior_copula":   prior_parts["copula"],
        "nll_post_marginal":  post_parts["marginal"],
        "nll_post_copula":    post_parts["copula"],
    }


def gaussian_corr_kl(R_model: torch.Tensor, R_post: torch.Tensor) -> float:
    """KL( N(0, R_post) || N(0, R_model) ) per point -- a correlation-only
    divergence with a true zero floor.

        corr_kl/n = 0.5 * [ tr(R_model^-1 R_post) - n + log|R_model| - log|R_post| ] / n

    >= 0, and == 0 iff R_model == R_post.

    Why this exists alongside oracle_diag/gap_nll. gap_nll is a Monte-Carlo
    estimate of KL(true posterior || model predictive) from ONE realized
    y_test, so it carries sampling noise and its per-episode value can be
    negative. This is a functional of the two matrices alone -- no y_test, no
    noise -- so it isolates the copula head's correlation error exactly, and a
    zero here means the predicted correlation IS the posterior correlation.

    Both arguments must be correlation matrices (unit diagonal): R_model comes
    from model.low_rank_correlation (unit diagonal by construction, plus
    jitter) and R_post from gp_analytical_posterior. Returns +inf rather than
    raising when R_model is not positive definite even after that jitter --
    one bad episode must never take down a validation pass, same policy as
    episode_posterior_ceiling's None return.
    """
    A = R_model.double()
    B = R_post.double()
    n = A.shape[-1]
    try:
        L = torch.linalg.cholesky(A)
    except Exception:
        return float("inf")
    if not torch.isfinite(L).all():
        return float("inf")
    log_det_model = 2.0 * torch.log(torch.diagonal(L)).sum()
    trace = torch.diagonal(torch.cholesky_solve(B, L)).sum()
    sign, log_det_post = torch.linalg.slogdet(B)
    if sign.item() <= 0:
        return float("inf")
    val = 0.5 * (trace - n + log_det_model - log_det_post) / n
    return float(val.item())


@torch.no_grad()
def episode_posterior_ceiling(task: dict, eig_floor: float = 1e-6) -> "dict | None":
    """gp_analytical_posterior reduced to the handful of numbers a validation
    loop actually consumes, per-point-normalized, computed ONCE per episode.

    Why this exists: the Bayes-optimal ceiling is a property of the EPISODE
    alone -- it never touches the model -- yet train.py::validate() used to
    call gp_analytical_posterior on every episode of a FIXED probe/val set on
    every single validate() call. Those episodes are generated once
    (build_fixed_live_val_batches' live_val_seed, _build_synthetic_kernel_
    batches' synth_seed), so every call after the first recomputed a constant:
    an O(N^3) float64 eigendecomposition per episode per family per
    validation. scripts/train_fast.py already made exactly this argument for
    its own debug loop ("a property of the fixed episodes alone, independent
    of the model being trained, so it's computed once here rather than every
    validation call"); this brings the real training loop to parity.

    Callers precompute a list of these alongside the episodes and hand them to
    validate(), which then just averages floats. That in turn makes raising
    baselines.synth_n_episodes / training.val_episodes cheap: it costs a
    one-time startup pass instead of scaling the per-validation cost.

    Returns None -- rather than raising -- for the two cases the callers all
    already treat as "ceiling unavailable for this episode": a kernel schema
    gp_analytical_posterior doesn't support (NotImplementedError, e.g.
    whole-chain outer sign modulation) or an episode missing the metadata it
    needs (KeyError, e.g. loaded from an on-disk shard written without
    return_kernel_metadata=True). One skipped episode must never take down a
    whole validation pass.

    All three NLL fields are divided by this episode's own n_test, matching
    loss.y_space_nll's per-point convention, so they can be averaged directly
    against the model's own totals. off_R_post is the strict upper triangle of
    the true posterior correlation matrix R_post, the operand for the
    predicted-vs-oracle correlation diagnostics; None when n_test < 2 (no
    off-diagonal pairs exist).
    """
    try:
        post = gp_analytical_posterior(task, eig_floor=eig_floor)
    except (KeyError, NotImplementedError):
        return None
    n = int(task["x_norm_test"].shape[0])
    if n < 1:
        return None
    off_R_post = None
    if n >= 2:
        ri, ci = torch.triu_indices(n, n, offset=1)
        off_R_post = post["R_post"][ri, ci].cpu().numpy()
    return {
        "n_test": n,
        "nll_post": post["nll_post"] / n,
        "nll_post_marginal": post["nll_post_marginal"] / n,
        "nll_post_copula": post["nll_post_copula"] / n,
        "off_R_post": off_R_post,
        # Full (n, n) matrix, not just off_R_post's strict upper triangle:
        # gaussian_corr_kl needs a Cholesky of it. Kept as float32 on CPU --
        # at the pinned probe/val sizes this is a few MB across the whole
        # cached probe set, against the O(n^3) float64 eigendecomposition it
        # saves on every validate() call.
        "R_post": post["R_post"],
    }
