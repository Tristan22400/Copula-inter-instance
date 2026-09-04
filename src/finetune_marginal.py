"""finetune_marginal.py — Phase A entry point.

Fine-tunes a STANDALONE TabICL (quantile decoder intact) so its marginal
posterior predictive is correct for the GP prior the copula is trained on, then
writes it in TabICL's own checkpoint schema so the copula run picks it up with:

    python src/train.py tabicl.pit_ckpt=<checkpoints/marginal_finetune/...pt>

Usage
-----
    python src/finetune_marginal.py
    python src/finetune_marginal.py marginal.tier=1 training.lr=2e-5
    python src/finetune_marginal.py wandb.mode=disabled training.steps=20   # smoke

A real Hydra application, not an argparse -> override translator that shells out
to train.py the way ``src/finetune_era5.py`` does: the Phase-A objective needs
its own model construction, its own loss and its own validation, so there is no
train.py invocation to translate INTO. Every knob is therefore a normal Hydra
override and the composed config is snapshotted into each checkpoint.

Why this is a separate loop rather than a ``training.objective: marginal`` branch
inside ``src/train.py``: that file's ~3000-line ``main`` is built end to end
around the copula path — live GP/ERA5 DataLoaders whose workers each hold their
own frozen TabICL, the copula head, z_train collation, Sigma diagnostics,
correlogram probes, Muon param groups. Phase A shares none of it: no DataLoader
at all (the trainable marginal must live in the MAIN process, since gradients do
not cross process boundaries and nothing can push updated weights into spawned
workers), no copula head, no z. Threading a second objective through that main
would mean touching model construction, data, loss, validation and
checkpointing, putting the working copula path at risk for no reuse. What IS
shared is shared by import — ``pit`` (the PIT forward), ``data_gen`` (the prior),
``lora`` (the freeze predicate), ``train.cosine_lr_lambda`` (the schedule) and
``eval.spatial.sweep_core`` (the ERA5 probe geometry) — so there is no duplicated
logic, only a duplicated ``for step in range(...)``.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _REPO_ROOT, os.path.join(_REPO_ROOT, "tabicl_upstream", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gen import generate_gp_batch  # noqa: E402
from pit import load_tabicl  # noqa: E402
from train import cosine_lr_lambda  # noqa: E402



import zlib
from typing import Callable, Optional, Sequence
import torch.nn as nn
from lora import apply_lora, merged_base_state_dict
from pit import (
    DEFAULT_K_FOLDS,
    _kernel_fn_from_task,
    _mean_train_from_task,
    _safe_cholesky,
    normalize_targets,
    run_pit_batched_grad,
)

# ===========================================================================
# Deliverable 1 — parameter adaptation and routing
# ===========================================================================

# Tier 0: the label path, the norms of the stage that does in-context
# learning, and the module that literally emits the marginal.
#
# Routing rationale (checkpoint tabicl-regressor-v2-20260212: col_embedder
# 0.875M @ width 128, row_interactor 0.397M @ width 128, icl_predictor 27.27M @
# width 512, of which decoder 1.55M):
#   * col_embedder / row_interactor only ever see x. They are structurally
#     uninvolved in mapping *labels* to a predictive law -- the one exception is
#     col_embedder.y_encoder, which is how context labels enter at all under
#     col_target_aware=True, hence its inclusion here.
#   * icl_predictor is where in-context learning happens: it ingests context
#     labels through its own y_encoder, attends across rows, and decodes the
#     predictive. Marginal correctness lives there.
#
# These are exactly the modules whose INPUT DISTRIBUTION changed (the label
# path), the ones that renormalize it (the norms), and the one that emits the
# thing being corrected (the decoder). ~1.56M params, 5.5% of 28.5M.
#
# Regex, not substrings: "the norms inside the ICL stack" has no substring
# spelling that excludes the identically-named norms in col_embedder and
# row_interactor.
TIER0_PATTERNS: tuple[str, ...] = (
    r"^icl_predictor\.y_encoder\.",
    r"^col_embedder\.y_encoder\.",
    r"^icl_predictor\.ln\.",
    r"^icl_predictor\.tf_icl\.blocks\.\d+\.norm[12]\.",
    r"^icl_predictor\.decoder\.",
)

# Tier ladder. Deliberately a ladder and not a guess: Tier 0 can only rescale
# and remap what the trunk already computes; it cannot change *how much context
# row j influences query row i*. Posterior contraction with context density is
# an attention-pattern property, so whether the pretrained attention already
# implements a good enough GP-like aggregation is an empirical question,
# answered by climbing this ladder and watching the NLL gap to the analytic
# oracle plateau (or not).
#
# Full fine-tuning is deliberately absent: src/muon.py self-declares Muon "may
# not work well for finetuning pretrained models", and full FT risks
# catastrophic forgetting of the general tabular marginal that is the entire
# reason a TabICL marginal transfers to real ERA5/UCI data at all.
TIER_SPECS: dict[int, dict] = {
    0: {
        "lora_stages": [],
        "desc": "label path + ICL norms + decoder (~1.6M, 5.5%)",
    },
    1: {
        "lora_stages": ["icl"],
        "desc": "tier 0 + LoRA on icl_predictor attention",
    },
    2: {
        "lora_stages": ["icl", "row"],
        "desc": "tier 1 + LoRA on row_interactor attention",
    },
    3: {
        "lora_stages": ["icl", "row", "col"],
        "desc": "tier 2 + LoRA on col_embedder attention",
    },
}


def apply_tier(
    backbone: nn.Module,
    tier: int,
    *,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_target: str = "qkvo",
    extra_patterns: Sequence[str] = (),
) -> dict:
    """Route Phase-A trainability over *backbone* according to ``tier``.

    Every tier keeps the Tier-0 allowlist trainable; tiers >= 1 additionally
    install LoRA adapters on the listed stages' attention. Both go through
    ``lora.apply_lora``/``lora.set_trainable``, so the freeze predicate is
    defined in exactly one place and a tier can never disagree with what a LoRA
    run does.

    Returns a report dict (also the thing to log as ``n_trainable_params`` so
    the ladder is visible in the wandb run table).
    """
    if tier not in TIER_SPECS:
        raise ValueError(f"Unknown tier {tier}; expected one of {sorted(TIER_SPECS)}.")
    spec = TIER_SPECS[tier]
    stages = list(spec["lora_stages"])
    patterns = tuple(TIER0_PATTERNS) + tuple(extra_patterns)

    n_replaced = apply_lora(
        backbone=backbone,
        rank=int(lora_rank) if stages else 0,
        alpha=float(lora_alpha),
        target=lora_target,
        stages=stages,
        also_trainable=patterns,
    )
    report = trainable_param_report(backbone)
    report.update(
        {
            "tier": tier,
            "tier_desc": spec["desc"],
            "lora_stages": stages,
            "lora_modules_replaced": n_replaced,
        }
    )
    return report


def trainable_param_report(module: nn.Module) -> dict:
    """``{n_trainable_params, n_total_params, trainable_frac, n_trainable_tensors}``."""
    n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in module.parameters())
    return {
        "n_trainable_params": int(n_train),
        "n_total_params": int(n_total),
        "trainable_frac": float(n_train / max(n_total, 1)),
        "n_trainable_tensors": int(sum(1 for p in module.parameters() if p.requires_grad)),
    }


# ===========================================================================
# Analytic targets — the exact marginal posterior predictive
# ===========================================================================


def analytic_marginal_targets(
    task: dict,
    x_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    x_qry: torch.Tensor,
    *,
    kernel_fn: Optional[Callable] = None,
    nugget: Optional[float] = None,
    use_cached_full_context: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact ``N(mu_i, sigma_i^2)`` for the OBSERVABLE target at ``x_qry``,
    conditioned on ``(x_ctx, y_ctx)``.

    This is the Phase-A regression target. It differs from
    ``pit.gp_analytical_posterior`` in three ways, each deliberate:

    * **arbitrary context subset.** ``gp_analytical_posterior`` always conditions
      on the episode's full ``x_norm_train``. Phase A must condition on whatever
      the model actually saw in that forward — the K-1 folds for a K-fold query
      row, all P rows for a test row — because the whole thing being taught is
      how the predictive contracts with context.
    * **diagonal only.** The marginal needs ``diag(Sigma_post)``, not the (N,N)
      matrix. Computed as ``k(x,x) + nugget - ||L^-1 K_fs||^2`` columnwise, so
      cost is O(P^3 + P^2 N) with no N^2 term.
    * **observable y, not latent f.** ``nugget`` is added to the query diagonal.
      ``data_gen.gp_posterior`` defaults to ``latent=True`` (posterior over f*,
      noise excluded); using that here would make every target systematically
      over-sharp and teach the head to be overconfident. The nugget is also the
      hard lower bound on the variance — ``Sigma_post = Cov(f|D) + nugget*I``
      with the first term PSD — so it doubles as the clamp floor, exactly as
      ``gp_analytical_posterior`` argues.

    Linear algebra runs in float64 (kernel evaluation stays in the kernel's
    native float32), matching every other analytic path in ``pit.py``.

    Args:
        task      : episode dict with ``return_kernel_metadata=True`` fields.
        x_ctx     : (P_c, d) context features, normalized space.
        y_ctx     : (P_c,)   context targets, RAW scale.
        x_qry     : (M, d)   query features.
        kernel_fn : reconstructed kernel; recomputed from ``task`` if omitted.
        nugget    : observation-noise variance; read from ``task`` if omitted.

    Returns:
        ``(mu, sigma)``, both (M,) float32, on ``x_qry``'s device, RAW scale.
    """
    if kernel_fn is None or nugget is None:
        kernel_fn, nugget = _kernel_fn_from_task(task)

    device = x_qry.device
    x_ctx = x_ctx.to(device)
    y_ctx = y_ctx.to(device)
    P_c = x_ctx.shape[0]

    if use_cached_full_context and "_L_ff" in task and "_alpha" in task:
        if P_c != task["x_norm_train"].shape[0]:
            raise ValueError("cached full-context factors require the complete training context")
        L_ff = task["_L_ff"].to(device=device, dtype=torch.float64)
        alpha = task["_alpha"].to(device=device, dtype=torch.float64)
    else:
        K_ff = (kernel_fn(x_ctx, x_ctx) + nugget * torch.eye(P_c, device=device)).double()
        L_ff = _safe_cholesky(K_ff, max_attempts=12)
        mean_ctx = _mean_train_from_task(task, x_ctx).double()
        alpha = torch.cholesky_solve(
            (y_ctx.double() - mean_ctx).unsqueeze(-1), L_ff
        ).squeeze(-1)

    K_sf = kernel_fn(x_qry, x_ctx).double()                       # (M, P_c)
    mean_qry = _mean_train_from_task(task, x_qry).double()
    mu = mean_qry + K_sf @ alpha                                  # (M,)

    V = torch.linalg.solve_triangular(L_ff, K_sf.T, upper=False)  # (P_c, M)
    k_diag = kernel_fn(x_qry, x_qry).diagonal().double() + nugget
    var = (k_diag - (V ** 2).sum(0)).clamp(min=nugget)
    return mu.float(), var.sqrt().float()


def episode_fold_targets(
    task: dict,
    query_idx: torch.Tensor,
    k_folds: int,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic ``(mu, sigma)`` for the K-fold training-query rows in
    ``query_idx``, each conditioned on ITS OWN fold's context.

    ``query_idx`` is the ``train_query_idx`` a fold-subsetted
    ``pit.run_pit_batched_grad`` returns: the concatenation, in fold order, of
    the contiguous index blocks it actually scored. The fold membership is
    re-derived here from the SAME ``fold_size = ceil(P/K)`` rule rather than
    being passed in, so the target's conditioning set is provably the model's
    conditioning set — if that rule ever changes in ``pit.py``, this must be
    updated with it, and the ``tests/test_marginal_finetune.py`` fold-agreement
    test is what catches the drift.

    Returns ``(mu, sigma)``, both ``(len(query_idx),)``, RAW scale, in
    ``query_idx``'s own order.
    """
    x_train = task["x_norm_train"].to(device)
    y_train = task["y_train"].to(device)
    P = x_train.shape[0]
    K = max(2, min(int(k_folds), P))
    fold_size = math.ceil(P / K)

    query_idx = query_idx.to(device)
    mu_out = torch.empty(query_idx.numel(), device=device)
    sig_out = torch.empty(query_idx.numel(), device=device)

    fold_of = (query_idx // fold_size)
    cached = "_L_ff" in task and "_alpha" in task
    if cached:
        L_full = task["_L_ff"].to(device=device, dtype=torch.float64)
        alpha_full = task["_alpha"].to(device=device, dtype=torch.float64)
    else:
        kernel_fn, nugget = _kernel_fn_from_task(task)
    for k in fold_of.unique().tolist():
        sel = (fold_of == k).nonzero(as_tuple=True)[0]           # positions in query_idx
        qry_rows = query_idx[sel]
        start, end = k * fold_size, min((k + 1) * fold_size, P)
        if cached:
            # For a joint Gaussian with full precision Lambda and
            # alpha=Lambda@(y-mean), conditioning q on the complement gives
            # precision Lambda_qq and mean y_q-Lambda_qq^-1 alpha_q.
            fold_rows = torch.arange(start, end, device=device)
            eye_q = torch.zeros(P, fold_rows.numel(), dtype=torch.float64, device=device)
            eye_q[fold_rows, torch.arange(fold_rows.numel(), device=device)] = 1.0
            precision_cols = torch.cholesky_solve(eye_q, L_full)
            precision_qq = precision_cols[fold_rows]
            L_qq = _safe_cholesky(precision_qq, max_attempts=12)
            correction = torch.cholesky_solve(
                alpha_full[fold_rows].unsqueeze(-1), L_qq
            ).squeeze(-1)
            covariance_qq = torch.cholesky_inverse(L_qq)
            positions = qry_rows - start
            mu_k = (y_train[fold_rows].double() - correction)[positions]
            sig_k = covariance_qq.diagonal().clamp(min=1e-12).sqrt()[positions]
        else:
            ctx_mask = torch.ones(P, dtype=torch.bool, device=device)
            ctx_mask[start:end] = False
            ctx_rows = ctx_mask.nonzero(as_tuple=True)[0]
            mu_k, sig_k = analytic_marginal_targets(
                task, x_train[ctx_rows], y_train[ctx_rows], x_train[qry_rows],
                kernel_fn=kernel_fn, nugget=nugget,
            )
        mu_out[sel] = mu_k.to(mu_out.dtype)
        sig_out[sel] = sig_k.to(sig_out.dtype)
    return mu_out, sig_out


# ===========================================================================
# Deliverable 3b — the Phase A objective
# ===========================================================================


class MarginalLossWeights:
    """Weights for the marginal fine-tuning loss terms.

    ``distill`` is the dense analytic-quantile diagnostic; ``pinball`` is the
    production, strictly consistent sample quantile score; ``nll`` and
    distribution-based ``crps`` are available as opt-in ablations;
    ``anchor`` is the anti-forgetting pull toward the pretrained weights.

    Set ``distill=0`` and ``pinball>0`` on a batch with no analytic target
    (the real-ERA5 mixture fraction). Pinball is higher variance than exact
    distillation but remains statistically consistent for every quantile.
    """

    def __init__(
        self,
        distill: float = 1.0,
        nll: float = 0.0,
        crps: float = 0.0,
        pinball: float = 0.0,
        anchor: float = 0.0,
        huber_delta: float = 1.0,
        tail_power: float = 0.5,
    ) -> None:
        self.distill = float(distill)
        self.nll = float(nll)
        self.crps = float(crps)
        self.pinball = float(pinball)
        self.anchor = float(anchor)
        self.huber_delta = float(huber_delta)
        self.tail_power = float(tail_power)


def quantile_level_weights(
    alpha_levels: torch.Tensor, tail_power: float = 0.5
) -> torch.Tensor:
    """``w_k ∝ (alpha_k (1-alpha_k))^tail_power``, normalized to mean 1.

    De-emphasizes the extreme levels in the distillation term. Two reasons, not
    one: ``Phi^-1(0.001) = -3.09`` so the outermost levels carry the largest
    residuals and would dominate a uniformly-weighted loss; and the outer 20
    levels in each tail are exactly what
    ``QuantileDistribution``'s ``TAIL_QUANTILES_FOR_ESTIMATION = 20`` uses to
    fit its parametric tail, i.e. they are consumed by an extrapolation model
    rather than read off directly, so forcing them to the analytic Gaussian
    value is both the hardest ask and the least load-bearing one.

    ``tail_power=0`` gives uniform weights (the ablation).
    """
    a = alpha_levels.clamp(1e-9, 1 - 1e-9)
    w = (a * (1.0 - a)) ** float(tail_power)
    return w / w.mean()


def _standard_normal_icdf(alpha: torch.Tensor) -> torch.Tensor:
    return torch.erfinv(2.0 * alpha.clamp(1e-9, 1 - 1e-9) - 1.0) * math.sqrt(2.0)


def marginal_objective(
    q: torch.Tensor,
    y: torch.Tensor,
    quantile_dist: nn.Module,
    weights: MarginalLossWeights,
    *,
    mu: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    alpha_levels: Optional[torch.Tensor] = None,
) -> dict:
    """Phase-A loss on ``M`` query rows, all in ``normalize_targets`` space.

        L = w_q * Huber( q_ik, mu_i + sigma_i Phi^-1(alpha_k) )       [distillation]
          + w_p * pinball(alpha_k, y_i - q_ik)                       [quantile score]
          + w_n * ( -log f(y_i) )                                    [NLL,  proper]
          + w_c * CRPS( f, y_i )                                     [CRPS, proper]

    The analytic term is direct quantile regression in ``normalize_targets``
    space.  An earlier version standardized each prediction error by the
    *posterior* ``sigma_i``.  Although that has the same pointwise optimum, its
    gradient is proportional to ``1 / sigma_i`` and made nearly deterministic
    GP rows dominate every clipped optimizer step.  On a fixed validation set,
    that objective made both GP and ERA5 NLL worse while direct ERA5 pinball
    improved both.  The inputs and targets here have already been normalized by
    the context target standard deviation, so a second, per-query scale division
    is neither needed nor desirable.

    The terms have the same population optimum, but not comparable decoder
    gradients. ``QuantileDistribution.log_prob`` first locates the two knots
    surrounding the single observed ``y`` with ``searchsorted`` and then
    differentiates only through that local interval. With 999 decoder outputs,
    this makes the NLL gradient both extremely sparse and orders of magnitude
    larger per active knot than the dense CRPS/distillation gradients. In
    practice it teaches the raw decoder to permute/collapse knots; the
    distribution's sorting step hides that from the training NLL until held-out
    NLL and calibration deteriorate. Distribution-based CRPS also sees sorted
    knots, making it invariant to raw-output permutations. Therefore ``nll``
    and ``crps`` are opt-in diagnostic weights (zero in the shipped
    configuration), while both are always computed and reported. Direct
    pinball loss preserves quantile identities on real data, and analytic
    quantile distillation does so on synthetic data.

    Args:
        q             : (M, Q) raw decoder quantile values, scaled space.
        y             : (M,)   observed targets, scaled space.
        quantile_dist : the model's ``QuantileToDistribution`` module.
        mu, sigma     : (M,) analytic targets, scaled space. Omit to skip the
                        distillation term (real-data mixture batches).
        target_mask   : (M,) bool; rows with a usable analytic target. Rows
                        outside it still contribute configured sample scores;
                        a kernel family this repo cannot reconstruct
                        (``_kernel_fn_from_task`` raising) is a missing dense
                        target, not a missing observation, so dropping the
                        whole episode would throw away potentially valid
                        sample-score signal. ``None`` means "all rows".
        alpha_levels  : (Q,) nominal levels; taken from ``quantile_dist`` if
                        omitted.

    Returns a dict of scalar tensors: ``loss`` plus each term unweighted, so
    the logger can show what is actually moving.
    """
    if alpha_levels is None:
        alpha_levels = quantile_dist.alpha_levels.to(q.device, dtype=q.dtype)

    dist = quantile_dist(q)
    nll = -dist.log_prob(y)                     # (M,)
    crps = dist.crps(y)                         # (M,)
    error = y.unsqueeze(-1) - q
    pinball = torch.maximum(alpha_levels * error, (alpha_levels - 1.0) * error)

    out = {
        "nll": nll.mean(),
        "crps": crps.mean(),
        "pinball": pinball.mean(),
    }
    # Do not spell this as ``0 * nll``: apart from retaining the pathological
    # sparse NLL graph, IEEE arithmetic would let a diagnostic NaN contaminate
    # an otherwise finite objective even when its configured weight is zero.
    loss = q.sum() * 0.0
    if weights.nll != 0.0:
        loss = loss + weights.nll * out["nll"]
    if weights.crps != 0.0:
        loss = loss + weights.crps * out["crps"]
    if weights.pinball != 0.0:
        loss = loss + weights.pinball * out["pinball"]

    have_target = mu is not None and sigma is not None and weights.distill != 0.0
    if have_target and target_mask is not None:
        have_target = bool(target_mask.any())
    if have_target:
        if target_mask is not None:
            q_d, mu_d, sig_d = q[target_mask], mu[target_mask], sigma[target_mask]
        else:
            q_d, mu_d, sig_d = q, mu, sigma
        z_target = _standard_normal_icdf(alpha_levels)                      # (Q,)
        w = quantile_level_weights(alpha_levels, weights.tail_power)        # (Q,)
        q_target = mu_d.unsqueeze(-1) + sig_d.unsqueeze(-1) * z_target
        per_level = torch.nn.functional.huber_loss(
            q_d, q_target, reduction="none",
            delta=weights.huber_delta,
        )
        out["distill"] = (per_level * w).mean()
        loss = loss + weights.distill * out["distill"]
    else:
        out["distill"] = torch.zeros((), device=q.device)

    out["loss"] = loss
    return out


class AnchorPenalty:
    """L2 pull of the trainable parameters toward their pretrained values.

    The anti-forgetting term. A TabICL marginal is worth fine-tuning precisely
    because it transfers to real tabular/ERA5 data with non-Gaussian marginals;
    the GP prior has Gaussian marginals, so an unconstrained run can improve
    synthetic nats by collapsing the head toward "always Gaussian" and destroy
    exactly the property that made it useful. This, the low LR, the tier
    topology, the real-ERA5 mixture and the ``run_benchmarks.py`` regression
    gate are five independent guards on that one failure mode.

    Snapshots only the trainable tensors (Tier 0 is ~1.6M floats, ~6MB).
    """

    def __init__(self, module: nn.Module) -> None:
        self.ref = {
            name: p.detach().clone()
            for name, p in module.named_parameters()
            if p.requires_grad
        }

    def __call__(self, module: nn.Module) -> torch.Tensor:
        total = None
        for name, p in module.named_parameters():
            if not p.requires_grad or name not in self.ref:
                continue
            term = ((p - self.ref[name]) ** 2).sum()
            total = term if total is None else total + term
        if total is None:
            dev = next(module.parameters()).device
            return torch.zeros((), device=dev)
        return total


# ===========================================================================
# Metrics — marginal only. No Sigma, no copula term, nothing correlation-shaped.
# ===========================================================================


def ks_uniform(u: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic of ``u`` against Uniform(0,1).

    ``u = F(y)`` is uniform iff the predictive law is correct, so this is the
    single most direct scalar test of marginal calibration. Computed by hand
    (rather than ``scipy.stats.kstest``) to keep the two-sided sup over both
    ECDF branches explicit and avoid a scipy import in the training loop.
    """
    u = np.sort(np.asarray(u, dtype=float).ravel())
    n = u.size
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def rank_histogram(u: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Normalized rank histogram of ``u = F(y)``.

    Flat = calibrated. U-shaped = over-sharp (observations fall in the tails too
    often). Dome-shaped = under-sharp. This is the diagnostic that says *how* a
    non-flat PIT is wrong, which a single KS number cannot; no rank-histogram
    code existed in the repo before this.
    """
    counts, _ = np.histogram(np.asarray(u, dtype=float).ravel(), bins=n_bins, range=(0.0, 1.0))
    total = counts.sum()
    return counts / total if total else counts.astype(float)


@torch.no_grad()
def marginal_metrics(
    q: torch.Tensor,
    y: torch.Tensor,
    quantile_dist: nn.Module,
    *,
    log_std: float | torch.Tensor = 0.0,
    y_std: float | torch.Tensor = 1.0,
    eps: float = 1e-6,
    n_rank_bins: int = 20,
) -> dict:
    """Marginal-only calibration metrics for ``M`` query rows.

    ``q``/``y`` are in ``normalize_targets`` space. ``log_std``/``y_std`` are
    that normalization's own scale, used to report RAW-y units:
    ``log p_raw(y) = log p_scaled(y_s) - log(std)`` (so ``nll_raw = nll_scaled +
    log std``) and ``crps_raw = crps_scaled * std`` — the same Jacobian
    convention ``train.py::_tabicl_pit_batch`` and ``data_gen`` already use, so
    these numbers are directly comparable to ``val/y_nll_marginal``.

    ECE, KS and the rank histogram are scale-free and identical in either space.
    """
    from eval.spatial.calibration import compute_quantile_ece

    alpha = quantile_dist.alpha_levels.to(q.device, dtype=q.dtype)
    dist = quantile_dist(q)
    u = dist.cdf(y)
    nll_scaled = -dist.log_prob(y)
    crps_scaled = dist.crps(y)

    log_std_t = torch.as_tensor(log_std, dtype=nll_scaled.dtype, device=nll_scaled.device)
    y_std_t = torch.as_tensor(y_std, dtype=crps_scaled.dtype, device=crps_scaled.device)

    u_np = u.detach().float().cpu().numpy()
    ece, coverage = compute_quantile_ece(
        y.detach().float().cpu().numpy(),
        q.detach().float().cpu().numpy(),
        alpha.detach().float().cpu().numpy(),
    )
    return {
        "nll": float((nll_scaled + log_std_t).mean()),
        "crps": float((crps_scaled * y_std_t).mean()),
        "ece": float(ece),
        "ks": ks_uniform(u_np),
        "clamp_frac": float(((u <= eps) | (u >= 1.0 - eps)).float().mean()),
        "n": int(y.numel()),
        "_coverage": coverage,
        "_rank_hist": rank_histogram(u_np, n_rank_bins),
    }


@torch.no_grad()
def oracle_marginal_nll(
    y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> float:
    """``-log N(y; mu, sigma^2)`` averaged — the analytic floor the model's own
    marginal NLL is measured against.

    The gap ``nll_model - nll_oracle`` in nats/point is the headline number for
    this whole workstream and the acceptance criterion for a Phase-A run: it is
    the part of the joint NLL that ``y_space_nll``'s marginal term currently
    cannot improve because that term has no trainable parameters at all.
    """
    var = sigma.clamp(min=1e-12) ** 2
    nll = 0.5 * (torch.log(2 * math.pi * var) + (y - mu) ** 2 / var)
    return float(nll.mean())


# ===========================================================================
# One training step's worth of work on a batch of synthetic GP episodes
# ===========================================================================


def stack_episodes(episodes: Sequence[dict], device: str | torch.device) -> dict:
    """Collate ``data_gen.generate_gp_batch``'s per-episode dicts into the
    (B, ...) tensors ``pit.run_pit_batched*`` wants, plus the per-episode
    ``normalize_targets`` scale.

    No padding and no masks: every episode from one ``generate_gp_batch`` call
    shares P and N by construction (that is the precondition
    ``run_pit_batched`` documents), which is exactly why Phase A generates
    batch-at-a-time instead of sampling episodes independently.
    """
    x_train = torch.stack([e["x_norm_train"] for e in episodes]).to(device)   # (B,P,d)
    y_train = torch.stack([e["y_train"] for e in episodes]).to(device)        # (B,P)
    x_test = torch.stack([e["x_norm_test"] for e in episodes]).to(device)     # (B,N,d)
    y_test = torch.stack([e["y_test"] for e in episodes]).to(device)          # (B,N)

    y_tr_s, y_te_s, means, stds = [], [], [], []
    for b in range(len(episodes)):
        a, c, m, s = normalize_targets(y_train[b], y_test[b])
        y_tr_s.append(a)
        y_te_s.append(c)
        means.append(m)
        stds.append(s)
    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train_raw": y_train,
        "y_test_raw": y_test,
        "y_train_scaled": torch.stack(y_tr_s),
        "y_test_scaled": torch.stack(y_te_s),
        "y_mean": torch.stack(means),
        "y_std": torch.stack(stds),
    }


def phase_a_batch_loss(
    tabicl: nn.Module,
    episodes: Sequence[dict],
    weights: MarginalLossWeights,
    *,
    k_folds: int = DEFAULT_K_FOLDS,
    folds_per_step: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: str | torch.device = "cuda",
    eps: float = 1e-6,
    timings: Optional[dict[str, float]] = None,
) -> dict:
    """One Phase-A training step's forward + loss on a batch of GP episodes.

    Query rows are scored under exactly the conditioning deployment uses:
    ``N`` test rows against the full ``P``-row context, plus ``folds_per_step``
    of the ``K`` contiguous training folds against their own ``K-1``-fold
    context. Both are conditioned on the context that forward actually saw, so
    there is no train/serve skew in context size or fold geometry — and because
    ``P`` is resampled per batch by ``generate_gp_batch``, the model sees a
    range of context densities rather than memorizing one.

    ``folds_per_step`` (default: all ``K``) subsamples the fold rotation. Phase
    A never needs a complete ``z_train`` — it scores each query row's own
    predictive density, not its PIT residual — so scoring a random subset is an
    unbiased estimate of the same objective at 1/K the forward cost, with the
    fold geometry untouched.

    Episodes whose kernel family cannot be reconstructed
    (``NotImplementedError``/``KeyError`` from ``_kernel_fn_from_task``, e.g.
    whole-chain outer sign modulation) have no distillation target rather than
    crashing the run. They still contribute any configured sample-score terms;
    with the shipped pinball loss they still provide valid sample supervision.
    The precedent ``gp_analytical_posterior``'s callers already set.
    """
    def _mark(name: str, started: float) -> float:
        if timings is not None:
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            timings[name] = timings.get(name, 0.0) + now - started
            return now
        return time.perf_counter()

    if timings is not None and torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    t_part = time.perf_counter()
    B = len(episodes)
    batch = stack_episodes(episodes, device)
    t_part = _mark("collate", t_part)
    P = batch["x_train"].shape[1]
    K = max(2, min(int(k_folds), P))

    # Only the folds that are actually non-empty are eligible. pit.py splits P
    # into ceil(P/K)-sized contiguous blocks, so for P=16, K=10 the block size
    # is 2 and folds 8 and 9 cover nothing -- sampling from range(K) would then
    # request an empty fold and score no rows at all. This is not hypothetical:
    # Phase A deliberately draws P from a wide range (conf/data/
    # gp_tasks_marginal.yaml), so small-P batches with P < K are routine.
    fold_size = math.ceil(P / K)
    n_folds_eff = math.ceil(P / fold_size)
    if folds_per_step is None or folds_per_step >= n_folds_eff:
        fold_subset = None
    else:
        n_f = max(1, int(folds_per_step))
        perm = torch.randperm(n_folds_eff, generator=generator)[:n_f]
        fold_subset = sorted(perm.tolist())

    out = run_pit_batched_grad(
        tabicl,
        batch["x_train"],
        batch["y_train_scaled"].unsqueeze(-1),
        batch["x_test"],
        batch["y_test_scaled"].unsqueeze(-1),
        k_folds=K,
        eps=eps,
        return_quantiles=True,
        fold_subset=fold_subset,
        compute_pit=False,
        fuse_folds=True,
    )
    t_part = _mark("tabicl_forward", t_part)

    q_test = out["q_test"].squeeze(2)                                # (B, N, Q)
    q_train = out["q_train"].squeeze(2)                              # (B, P', Q)
    if fold_subset is None:
        train_idx = torch.arange(P, device=q_train.device)
    else:
        train_idx = out["train_query_idx"]

    # --- analytic targets, per episode, in normalize_targets space ---------
    M = q_test.shape[1] + q_train.shape[1]
    mu_all = torch.zeros(B, M, device=q_test.device)
    sig_all = torch.ones(B, M, device=q_test.device)
    mask_all = torch.zeros(B, M, dtype=torch.bool, device=q_test.device)
    n_ok = 0
    # These are fixed supervision targets, never trainable quantities. gpytorch
    # kernel objects own requires-grad parameters by default, so without this
    # guard autograd retains and traverses their entire Cholesky/solve graph
    # even though those ephemeral parameters are absent from the optimizer.
    with torch.no_grad():
        for b, ep in enumerate(episodes):
            try:
                kernel_fn, nugget = _kernel_fn_from_task(ep)
                mu_te, sig_te = analytic_marginal_targets(
                    ep,
                    batch["x_train"][b], batch["y_train_raw"][b], batch["x_test"][b],
                    kernel_fn=kernel_fn, nugget=nugget, use_cached_full_context=True,
                )
                mu_tr, sig_tr = episode_fold_targets(ep, train_idx, K, device=device)
            except (NotImplementedError, KeyError):
                continue  # configured sample scores may apply; target does not
            n_ok += 1
            m, sd = batch["y_mean"][b], batch["y_std"][b]
            mu_all[b] = torch.cat([(mu_te - m) / sd, (mu_tr - m) / sd])
            sig_all[b] = torch.cat([sig_te / sd, sig_tr / sd])
            mask_all[b] = True
    t_part = _mark("analytic_targets", t_part)

    q_all = torch.cat([q_test, q_train], dim=1)                       # (B, N+P', Q)
    y_all = torch.cat(
        [batch["y_test_scaled"], batch["y_train_scaled"][:, train_idx]], dim=1
    )                                                                 # (B, N+P')

    Q = q_all.shape[-1]
    q_flat = q_all.reshape(-1, Q)
    y_flat = y_all.reshape(-1)
    mu_flat = mu_all.reshape(-1)
    sig_flat = sig_all.reshape(-1)
    mask_flat = mask_all.reshape(-1)

    res = marginal_objective(
        q_flat, y_flat, tabicl.quantile_dist, weights,
        mu=mu_flat, sigma=sig_flat, target_mask=mask_flat,
    )
    # QuantileDistribution sorts raw decoder outputs before scoring them. That
    # is useful at inference, but can conceal a decoder collapse during
    # training. Keep the pre-sort crossing rate visible in every train/val
    # record; the failed NLL-optimized run climbed from ~0.6% to ~49%.
    res["raw_crossing_frac"] = float(
        (q_flat[:, 1:] < q_flat[:, :-1]).float().mean().detach()
    )
    _mark("objective", t_part)
    res["n_episodes_with_target"] = n_ok
    res["oracle_nll"] = (
        oracle_marginal_nll(y_flat[mask_flat], mu_flat[mask_flat], sig_flat[mask_flat])
        if n_ok
        else float("nan")
    )
    # The headline number: how many nats/point the model's marginal is above the
    # analytic floor on exactly these query rows, under exactly this
    # conditioning. This is what a Phase-A run has to drive toward zero.
    res["nll_gap_to_oracle"] = float(res["nll"].detach()) - res["oracle_nll"]
    return res


# ===========================================================================
# Deliverable 3d — validation on real ERA5 regions, MARGINAL METRICS ONLY
# ===========================================================================


def build_era5_marginal_val_batches(vcfg, device: str | torch.device) -> dict:
    """Fixed per-region real-ERA5 probes for Phase-A validation, carrying RAW
    ``(x, y)`` only.

    Same geometry as the copula run's own ERA5 validation
    (``train.py::_build_era5_val_batches`` -> ``sweep_core.build_era5_probe``):
    the five ``baselines.era5_regions``, ``era5_grid_size``, ``era5_n_context``,
    the same fixed per-region seed. Reusing that geometry is the point — it
    makes a Phase-A run's real-data numbers directly comparable to the copula
    run's, on the same points.

    Two deliberate differences:

    * **``tabicl_marginal=None``.** ``build_era5_probe`` would otherwise PIT the
      context ONCE here and cache ``z_train``. That caching is valid only
      because the marginal is frozen in the copula run; in Phase A the marginal
      changes every step, so a cached PIT would silently freeze the validation
      metric at its step-0 value. Passing None skips it, and this function keeps
      the raw values instead — the fetch/crop cost stays a one-off, only the
      forward repeats.
    * **no correlogram, no GP baseline, no Sigma.** Phase A is marginal-only;
      ``rho_emp``/``pair_counts``/``gp_baseline_nll`` are copula diagnostics and
      are dropped here rather than computed and ignored.

    The probes come from ``eval/data/fetch_era5.py``, whose ``start_date``
    defaults to ``2023-01-01`` — the same held-out year as
    ``eval/data/cache/era5_global_val/``, and disjoint from the 2013-2022
    ``era5_global_train/`` months the training mixture draws from. Validation is
    therefore on genuinely unseen time by construction, not by convention.
    """
    from eval.configs.regions import REGIONS as ERA5_REGIONS
    from eval.spatial.sweep_core import build_era5_probe

    def _g(key, default):
        return vcfg.get(key, default) if hasattr(vcfg, "get") else getattr(vcfg, key, default)

    region_names = list(_g("era5_regions", []) or list(ERA5_REGIONS.keys()))
    grid_size = int(_g("era5_grid_size", 24))
    n_days_fetch = int(_g("era5_n_days_fetch", 60))
    n_days_probe = int(_g("era5_n_days_probe", 3))
    n_context = int(_g("era5_n_context", 30))
    base_seed = int(_g("era5_seed", 20260818))

    batches: dict[str, dict] = {}
    for region in region_names:
        if region not in ERA5_REGIONS:
            continue  # not a registered eval/configs/regions.py entry
        # zlib.crc32, not hash(): Python's str hash is salted per process
        # (PYTHONHASHSEED), so hash() here would draw a DIFFERENT context
        # sample every run and make the validation curve incomparable across
        # runs -- and incomparable with the copula run's own era5_fit probes.
        # This is byte-identical to train.py::_name_seed, deliberately, so both
        # phases validate on exactly the same points.
        seed = base_seed + (zlib.crc32(region.encode()) % 10_000)
        probe = build_era5_probe(
            region, grid_size, n_days_fetch, n_days_probe, n_context,
            n_bins=12, tabicl_marginal=None, device=str(device), seed=seed,
        )
        n_days = probe["context_values_per_day"].shape[0]
        x_tr = torch.as_tensor(probe["x_train_norm"], dtype=torch.float32, device=device)
        x_te = torch.as_tensor(probe["x_nll_test_norm"], dtype=torch.float32, device=device)
        batches[region] = {
            "x_train": x_tr.unsqueeze(0).expand(n_days, -1, -1).contiguous(),
            "x_test": x_te.unsqueeze(0).expand(n_days, -1, -1).contiguous(),
            "y_train": torch.as_tensor(
                probe["context_values_per_day"], dtype=torch.float32, device=device),
            "y_test": torch.as_tensor(
                probe["nll_test_values_per_day"], dtype=torch.float32, device=device),
        }
    return batches


@torch.no_grad()
def validate_era5_marginal(
    tabicl: nn.Module, batches: dict, *, eps: float = 1e-6
) -> dict:
    """Marginal-only metrics per region, plus across-region means.

    Emits ``val_marginal/<region>/{nll,crps,ece,ks,clamp_frac}`` and
    ``val_marginal/mean_*``. Nothing correlation-shaped: no Sigma, no copula
    term, no comparison of anything to an oracle ``R_star`` — per this repo's
    standing rule that TabICL's PIT distorts z-space, so only full predictive
    densities are valid comparisons.

    The query points are held out by construction (``build_era5_probe`` draws
    them from the never-in-context remainder), so this uses
    ``fold_subset=[]``: one forward with the full context, no leakage-avoiding
    fold rotation needed. It still goes through ``run_pit_batched_grad`` so the
    forward/CDF/log_prob path is byte-identical to training's.
    """
    per_region: dict[str, dict] = {}
    for region, b in batches.items():
        y_tr_s, y_te_s, mean, std = [], [], [], []
        for d in range(b["y_train"].shape[0]):
            a, c, m, sd = normalize_targets(b["y_train"][d], b["y_test"][d])
            y_tr_s.append(a); y_te_s.append(c); mean.append(m); std.append(sd)
        y_tr_s = torch.stack(y_tr_s)
        y_te_s = torch.stack(y_te_s)
        std_t = torch.stack(std)

        out = run_pit_batched_grad(
            tabicl, b["x_train"], y_tr_s.unsqueeze(-1),
            b["x_test"], y_te_s.unsqueeze(-1),
            k_folds=2, eps=eps, return_quantiles=True, fold_subset=[],
            compute_pit=False,
        )
        q = out["q_test"].squeeze(2)                                  # (days, N, Q)
        # Per-day std, broadcast over that day's query rows, so the raw-nats
        # Jacobian is each day's own -- days are normalized independently.
        n_q = q.shape[1]
        log_std = std_t.log().unsqueeze(1).expand(-1, n_q).reshape(-1)
        y_std = std_t.unsqueeze(1).expand(-1, n_q).reshape(-1)
        m = marginal_metrics(
            q.reshape(-1, q.shape[-1]), y_te_s.reshape(-1), tabicl.quantile_dist,
            log_std=log_std, y_std=y_std, eps=eps,
        )
        per_region[region] = m

    metrics: dict[str, float] = {}
    for region, m in per_region.items():
        for k in ("nll", "crps", "ece", "ks", "clamp_frac"):
            metrics[f"val_marginal/{region}/{k}"] = m[k]
    if per_region:
        for k in ("nll", "crps", "ece", "ks", "clamp_frac"):
            metrics[f"val_marginal/mean_{k}"] = float(
                np.mean([m[k] for m in per_region.values()])
            )
    return metrics


@torch.no_grad()
def validate_synthetic_marginal(
    tabicl: nn.Module,
    episode_batches: Sequence[Sequence[dict]],
    *,
    k_folds: int = DEFAULT_K_FOLDS,
    eps: float = 1e-6,
    device: str | torch.device = "cuda",
) -> dict:
    """Synthetic-GP counterpart of the ERA5 pass: the analytic headroom.

    ``val_marginal/gp/nll`` vs ``val_marginal/gp/nll_oracle`` and their
    difference ``val_marginal/gp/nll_gap_to_oracle`` — the number this whole
    workstream exists to drive toward zero, and the one that says whether a run
    that looks good on ERA5 is actually learning the posterior or just learning
    to be blurry.

    Watch it against the ERA5 block: a run that improves synthetic-GP nats while
    ``val_marginal/mean_nll``/``mean_ece`` degrade is overfitting to the GP
    prior's Gaussianity and should be stopped.
    """
    # Report the actual synthetic training objective on the fixed validation
    # episodes too. Previously validation only reported density scores, so a
    # run could not distinguish poor loss generalization from a broken update.
    metric_w = MarginalLossWeights(distill=1.0, nll=0.0, crps=0.0)
    nlls, crpss, distills, oracles, crossings = [], [], [], [], []
    for episodes in episode_batches:
        res = phase_a_batch_loss(
            tabicl, episodes, metric_w, k_folds=k_folds,
            folds_per_step=None, device=device, eps=eps,
        )
        nlls.append(float(res["nll"]))
        crpss.append(float(res["crps"]))
        distills.append(float(res["distill"]))
        oracles.append(res["oracle_nll"])
        crossings.append(res["raw_crossing_frac"])
    out = {
        "val_marginal/gp/nll": float(np.mean(nlls)) if nlls else float("nan"),
        "val_marginal/gp/crps": float(np.mean(crpss)) if crpss else float("nan"),
        "val_marginal/gp/distill": (
            float(np.mean(distills)) if distills else float("nan")
        ),
        "val_marginal/gp/nll_oracle": float(np.nanmean(oracles)) if oracles else float("nan"),
        "val_marginal/gp/raw_crossing_frac": (
            float(np.mean(crossings)) if crossings else float("nan")
        ),
    }
    out["val_marginal/gp/nll_gap_to_oracle"] = out["val_marginal/gp/nll"] - out["val_marginal/gp/nll_oracle"]
    return out


# ===========================================================================
# Checkpointing — TabICL-native, so the output IS a pit_ckpt
# ===========================================================================


def save_marginal_checkpoint(
    path: str,
    backbone: nn.Module,
    tabicl_config: dict,
    *,
    step: int,
    cfg=None,
    extra: Optional[dict] = None,
) -> None:
    """Write a Phase-A checkpoint in TabICL's OWN ``{"config", "state_dict"}``
    schema, with LoRA deltas merged into the base weights.

    Deliberately not ``train.py::save_checkpoint``'s schema. That one is for
    copula checkpoints, whose consumer is ``eval/configs/checkpoints.py`` and
    the eval runners. A Phase-A artifact's consumer is ``pit.load_tabicl``,
    which constructs ``TabICL(**checkpoint["config"])`` and calls
    ``load_state_dict`` strictly — so the file has to carry a ``config`` key and
    adapter-free parameter names, or the "drop-in ``tabicl.pit_ckpt``" promise
    is false. ``step``/``cfg``/``extra`` ride along beside them for provenance;
    ``load_tabicl`` ignores extra keys.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "config": dict(tabicl_config),
        "state_dict": merged_base_state_dict(backbone),
        "step": int(step),
    }
    if cfg is not None:
        from omegaconf import OmegaConf

        payload["cfg"] = OmegaConf.to_container(cfg, resolve=True)
    if extra:
        payload.update(extra)
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------


def _resolve_device(spec: str) -> str:
    if spec != "auto":
        return spec
    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ERA5EpisodeSampler:
    """Draws batches of real-ERA5 episodes that share P and N.

    ``run_pit_batched`` can only fold a batch into TabICL's own batch axis when
    every episode shares P/N, so this uses the corpus's
    ``sample_episode_fixed_shape`` (exact grid_size / n_context, redraw on a
    degenerate box) rather than ``sample_episode``'s ranges — the same reason
    ``src/era5_live_dataset.py`` groups its draws.

    Region, day and box width still vary per episode, so a batch is a genuine
    spread of real spatial fields at one shape, not one field repeated.
    """

    def __init__(self, corpus, *, grid_size: int, n_context: int,
                 box_deg_range: tuple[float, float], seed: int) -> None:
        self.corpus = corpus
        self.grid_size = int(grid_size)
        self.n_context = int(n_context)
        self.box_deg_range = box_deg_range
        self.rng = np.random.default_rng(seed)

    def batch(self, B: int, max_tries: int = 200) -> list[dict]:
        out: list[dict] = []
        tries = 0
        while len(out) < B and tries < max_tries * B:
            tries += 1
            ep = self.corpus.sample_episode_fixed_shape(
                self.rng, self.grid_size, self.box_deg_range, self.n_context
            )
            if ep is None:
                continue
            out.append(
                {
                    "x_norm_train": torch.as_tensor(ep["x_norm_train"]),
                    "y_train": torch.as_tensor(ep["y_train"]),
                    "x_norm_test": torch.as_tensor(ep["x_norm_test"]),
                    "y_test": torch.as_tensor(ep["y_test"]),
                }
            )
        if len(out) < B:
            raise RuntimeError(
                f"ERA5 sampler produced {len(out)}/{B} episodes in {tries} draws at "
                f"grid_size={self.grid_size}, n_context={self.n_context}, "
                f"box_deg_range={self.box_deg_range}. Widen box_deg_max or lower "
                f"grid_size."
            )
        return out


def _gp_cfg(cfg: DictConfig) -> DictConfig:
    """The shape ``data_gen.generate_gp_batch`` expects: a top-level config with
    a ``data`` group and a ``seed``, not the ``data`` group on its own.

    ``generate_gp_batch`` reads ``cfg.data.*`` for the prior and
    ``getattr(cfg, "seed")`` for reproducibility, so handing it ``cfg.data``
    directly raises ``Missing key data``. Built once and re-seeded per call
    rather than reconstructed, since the prior block is large.
    """
    return OmegaConf.create(
        {"data": OmegaConf.to_container(cfg.data, resolve=True), "seed": int(cfg.seed)}
    )


def _generate_phase_a_gp_batch(
    gp_cfg: DictConfig, batch_size: int, device: str, *, max_rounds: int = 20
) -> list[dict]:
    """Generate a shape-homogeneous batch even when GP episodes are discarded.

    ``generate_gp_batch`` guarantees the requested count, but its numerical-
    failure top-ups intentionally resample P/N because its usual consumers pad
    rows in a DataLoader. Phase A stacks directly for one batched TabICL call,
    so after the first returned shape is chosen, retries must pin P, N and d.
    """
    episodes = generate_gp_batch(
        gp_cfg, batch_size, device, return_kernel_metadata=True
    )
    P = int(episodes[0]["x_norm_train"].shape[0])
    N = int(episodes[0]["x_norm_test"].shape[0])
    d = int(episodes[0]["x_norm_train"].shape[1])
    out = [
        ep for ep in episodes
        if ep["x_norm_train"].shape == (P, d) and ep["x_norm_test"].shape == (N, d)
    ]
    if len(out) == batch_size:
        return out

    fixed = OmegaConf.create(OmegaConf.to_container(gp_cfg, resolve=True))
    fixed.data.P_min = fixed.data.P_max = P
    fixed.data.N_min = fixed.data.N_max = N
    base_seed = int(gp_cfg.seed)
    for round_idx in range(1, max_rounds + 1):
        fixed.seed = base_seed + round_idx * 1_000_003
        out.extend(generate_gp_batch(
            fixed, batch_size - len(out), device,
            return_kernel_metadata=True, d_override=d,
        ))
        if len(out) >= batch_size:
            return out[:batch_size]
    raise RuntimeError(
        f"Phase-A GP generator produced only {len(out)}/{batch_size} episodes "
        f"with fixed shape P={P}, N={N}, d={d} after {max_rounds} retries."
    )


def _build_gp_val_batches(cfg: DictConfig, device: str) -> list[list[dict]]:
    """A FIXED synthetic-GP validation set, drawn once with its own seed.

    Fixed rather than freshly sampled per call so that a change in
    ``val_marginal/gp/nll_gap_to_oracle`` between two validations is the model
    moving, not the episodes changing — the gap is a few nats on a quantity with
    real per-episode spread, so resampling would bury the signal in draw noise.
    """
    gp_cfg = _gp_cfg(cfg)
    batches = []
    for i in range(int(cfg.validation.gp_n_batches)):
        gp_cfg.seed = int(cfg.validation.gp_seed) + i
        batches.append(
            _generate_phase_a_gp_batch(
                gp_cfg, int(cfg.validation.gp_batch_size), device,
            )
        )
    return batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="../conf", config_name="finetune_marginal", version_base=None)
def main(cfg: DictConfig) -> None:
    device = _resolve_device(str(cfg.training.device))
    torch.set_float32_matmul_precision(str(cfg.training.matmul_precision))
    _seed_everything(int(cfg.seed))
    print(OmegaConf.to_yaml(cfg))

    # ---- model + tier routing -------------------------------------------
    tabicl, tabicl_config = load_tabicl(
        str(cfg.marginal.ckpt), device, trainable=True, return_config=True
    )
    report = apply_tier(
        tabicl,
        int(cfg.marginal.tier),
        lora_rank=int(cfg.marginal.lora_rank),
        lora_alpha=float(cfg.marginal.lora_alpha),
        lora_target=str(cfg.marginal.lora_target),
    )
    tabicl.to(device)
    print(
        f"[tier {report['tier']}] {report['tier_desc']}: "
        f"{report['n_trainable_params']:,} / {report['n_total_params']:,} trainable "
        f"({100 * report['trainable_frac']:.2f}%), "
        f"{report['lora_modules_replaced']} LoRA module(s)"
    )

    weights = MarginalLossWeights(
        distill=float(cfg.marginal.loss.distill),
        nll=float(cfg.marginal.loss.nll),
        crps=float(cfg.marginal.loss.crps),
        pinball=float(cfg.marginal.loss.pinball),
        anchor=float(cfg.marginal.loss.anchor),
        huber_delta=float(cfg.marginal.loss.huber_delta),
        tail_power=float(cfg.marginal.loss.tail_power),
    )
    # Both sources use raw quantile-index-aware pinball in the shipped config.
    # Synthetic weights remain separate because exact-target distillation is a
    # useful opt-in diagnostic there, while ERA5 has no analytic target.
    era5_loss_cfg = cfg.marginal.era5.loss
    era5_weights = MarginalLossWeights(
        distill=0.0,
        nll=float(era5_loss_cfg.nll),
        crps=float(era5_loss_cfg.crps),
        pinball=float(era5_loss_cfg.pinball),
        anchor=weights.anchor,
        huber_delta=weights.huber_delta, tail_power=weights.tail_power,
    )
    anchor = AnchorPenalty(tabicl) if weights.anchor > 0 else None

    # ---- optimizer -------------------------------------------------------
    # AdamW, one group, no ndim split. Two reasons this is not train.py's
    # Muon setup: src/muon.py's own header warns Muon "may not work well for
    # finetuning pretrained models", which is precisely this; and train.py's
    # positional optimizer-state restore (load_checkpoint matches Adam/Muon
    # moments to params by position in the flattened list) is a hazard the
    # moment param groups change, which a tier ladder does by construction.
    # A single group over `requires_grad` params has no positional ambiguity.
    params = [p for p in tabicl.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("Tier routing left no trainable parameters.")
    opt = torch.optim.AdamW(
        params, lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    total_steps = int(cfg.training.steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: cosine_lr_lambda(
            s, int(cfg.training.warmup_steps), total_steps,
            float(cfg.training.lr_min_frac),
        ),
    )

    # ---- data ------------------------------------------------------------
    gp_cfg = _gp_cfg(cfg)
    eps = float(cfg.marginal.pit_eps)
    k_folds = int(cfg.marginal.k_folds)
    folds_per_step = cfg.marginal.folds_per_step
    folds_per_step = None if folds_per_step is None else int(folds_per_step)
    mix_frac = float(cfg.marginal.era5.mix_frac)
    if not 0.0 <= mix_frac <= 1.0:
        raise ValueError(f"marginal.era5.mix_frac must be in [0, 1], got {mix_frac}")

    def _has_sample_objective(w: MarginalLossWeights) -> bool:
        return any(value != 0.0 for value in (w.distill, w.nll, w.crps, w.pinball))

    # Fail loudly instead of launching an expensive run whose loss is exactly
    # zero. This also catches misspelled/partial Hydra loss overrides early.
    if mix_frac < 1.0 and not _has_sample_objective(weights):
        raise ValueError("Synthetic batches have no non-zero marginal loss weight.")
    if mix_frac > 0.0 and not _has_sample_objective(era5_weights):
        raise ValueError("ERA5 batches have no non-zero marginal loss weight.")

    era5_sampler = None
    if mix_frac > 0:
        from eval.data.era5_global_corpus import GlobalERA5Corpus

        corpus = GlobalERA5Corpus(
            str(cfg.marginal.era5.corpus_dir),
            max_months=int(cfg.marginal.era5.max_months),
        )
        era5_sampler = ERA5EpisodeSampler(
            corpus,
            grid_size=int(cfg.marginal.era5.grid_size),
            n_context=int(cfg.marginal.era5.n_context),
            box_deg_range=(
                float(cfg.marginal.era5.box_deg_min),
                float(cfg.marginal.era5.box_deg_max),
            ),
            seed=int(cfg.seed) + 7717,
        )
        print(f"[era5] mixture on: {corpus.n_days_total} days loaded, mix_frac={mix_frac}")

    print("[val] building fixed validation sets (one-off ERA5 fetch/crop)...")
    era5_val = build_era5_marginal_val_batches(cfg.validation, device)
    gp_val = _build_gp_val_batches(cfg, device)
    print(f"[val] {len(era5_val)} ERA5 region(s), {len(gp_val)} synthetic GP batch(es)")

    # ---- wandb -----------------------------------------------------------
    run = None
    if str(cfg.wandb.mode) != "disabled":
        import wandb

        run = wandb.init(
            project=str(cfg.wandb.project),
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=str(cfg.wandb.mode),
        )
        wandb.watch(tabicl, log="gradients", log_freq=max(1, int(cfg.training.log_every)))
        wandb.log({f"model/{k}": v for k, v in report.items()
                   if isinstance(v, (int, float))}, step=0)

    def _log(payload: dict, step: int) -> None:
        if run is not None:
            run.log(payload, step=step)

    def _validate(step: int) -> dict[str, float]:
        t0 = time.time()
        t_era5 = time.time()
        metrics = validate_era5_marginal(tabicl, era5_val, eps=eps)
        metrics["val_marginal/era5_seconds"] = time.time() - t_era5
        t_gp = time.time()
        metrics.update(
            validate_synthetic_marginal(
                tabicl, gp_val, k_folds=k_folds, eps=eps, device=device
            )
        )
        metrics["val_marginal/gp_seconds"] = time.time() - t_gp
        metrics["val_marginal/seconds"] = time.time() - t0
        _log(metrics, step)
        print(
            f"[val step {step}] "
            f"era5 nll={metrics.get('val_marginal/mean_nll', float('nan')):.4f} "
            f"ece={metrics.get('val_marginal/mean_ece', float('nan')):.4f} "
            f"ks={metrics.get('val_marginal/mean_ks', float('nan')):.4f} | "
            f"gp nll={metrics.get('val_marginal/gp/nll', float('nan')):.4f} "
            f"distill={metrics.get('val_marginal/gp/distill', float('nan')):.4f} "
            f"oracle={metrics.get('val_marginal/gp/nll_oracle', float('nan')):.4f} "
            f"gap={metrics.get('val_marginal/gp/nll_gap_to_oracle', float('nan')):.4f} | "
            f"{metrics['val_marginal/seconds']:.2f}s "
            f"(era5 {metrics['val_marginal/era5_seconds']:.2f}s, "
            f"gp {metrics['val_marginal/gp_seconds']:.2f}s)"
        )
        return metrics

    def _save(step: int, tag: str = "") -> str | None:
        if cfg.training.ckpt_dir is None:
            return None
        name = f"step_{step:07d}{tag}.pt"
        path = os.path.join(str(cfg.training.ckpt_dir), name)
        save_marginal_checkpoint(
            path, tabicl, tabicl_config, step=step, cfg=cfg,
            extra={"tier_report": {k: v for k, v in report.items()
                                   if isinstance(v, (int, float, str))}},
        )
        print(f"[ckpt] {path}")
        return path

    # ---- train -----------------------------------------------------------
    initial_metrics = _validate(0)
    selection_metric = str(cfg.training.get(
        "selection_metric", "val_marginal/mean_nll"
    ))
    if selection_metric not in initial_metrics:
        raise KeyError(
            f"training.selection_metric={selection_metric!r} was not emitted by "
            f"validation. Available metrics: {sorted(initial_metrics)}"
        )
    best_value = float(initial_metrics[selection_metric])
    if not math.isfinite(best_value):
        raise RuntimeError(
            f"Initial selection metric {selection_metric} is non-finite: {best_value}"
        )
    best_step = 0
    selection_min_delta = float(cfg.training.get("selection_min_delta", 0.0))

    def _snapshot_trainable() -> dict[str, torch.Tensor]:
        # Frozen base tensors never change. Keeping only trainable tensors makes
        # validation selection cheap even for the full pretrained backbone.
        return {
            name: p.detach().cpu().clone()
            for name, p in tabicl.named_parameters()
            if p.requires_grad
        }

    def _restore_trainable(state: dict[str, torch.Tensor]) -> None:
        named = dict(tabicl.named_parameters())
        with torch.no_grad():
            for name, value in state.items():
                named[name].copy_(value.to(device=named[name].device))

    best_state = _snapshot_trainable()

    def _consider_validation(step: int, metrics: dict[str, float]) -> None:
        nonlocal best_step, best_value, best_state
        value = float(metrics[selection_metric])
        if math.isfinite(value) and value < best_value - selection_min_delta:
            best_step = step
            best_value = value
            best_state = _snapshot_trainable()
            print(
                f"[selection] new best {selection_metric}={best_value:.6f} "
                f"at step {best_step}"
            )

    rng = np.random.default_rng(int(cfg.seed) + 991)
    gen = torch.Generator().manual_seed(int(cfg.seed) + 13)
    B = int(cfg.training.batch_size)
    t_last = time.time()
    profile_steps = int(cfg.training.get("profile_steps", 0))
    profile_totals: dict[str, float] = {}

    for step in range(1, total_steps + 1):
        profiling = step <= profile_steps
        if profiling and device.startswith("cuda"):
            torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        data_started = step_started
        use_era5 = era5_sampler is not None and rng.random() < mix_frac
        if use_era5:
            episodes = era5_sampler.batch(B)
            episodes = [
                {k: v.to(device) for k, v in ep.items()} for ep in episodes
            ]
            w = era5_weights
        else:
            gp_cfg.seed = int(cfg.seed) * 1_000_003 + step
            episodes = _generate_phase_a_gp_batch(gp_cfg, B, device)
            w = weights
        if profiling and device.startswith("cuda"):
            torch.cuda.synchronize(device)
        data_seconds = time.perf_counter() - data_started

        part_timings: dict[str, float] | None = {} if profiling else None
        res = phase_a_batch_loss(
            tabicl, episodes, w,
            k_folds=k_folds, folds_per_step=folds_per_step,
            generator=gen, device=device, eps=eps,
            timings=part_timings,
        )
        loss = res["loss"]
        anchor_val = 0.0
        if anchor is not None:
            a = anchor(tabicl)
            loss = loss + weights.anchor * a
            anchor_val = a.detach().item()

        backward_started = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            params, float(cfg.training.clip_grad_norm)
        )
        if profiling and device.startswith("cuda"):
            torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_started
        optimizer_started = time.perf_counter()
        opt.step()
        sched.step()
        if profiling and device.startswith("cuda"):
            torch.cuda.synchronize(device)
        optimizer_seconds = time.perf_counter() - optimizer_started

        if profiling:
            measured = {
                "data": data_seconds,
                **(part_timings or {}),
                "backward_and_clip": backward_seconds,
                "optimizer": optimizer_seconds,
                "total": time.perf_counter() - step_started,
            }
            for key, value in measured.items():
                profile_totals[key] = profile_totals.get(key, 0.0) + value
            print("[profile step %d] %s" % (
                step, " ".join(f"{key}={value:.4f}s" for key, value in measured.items())
            ))
            if step == profile_steps:
                means = {key: value / profile_steps for key, value in profile_totals.items()}
                print("[profile mean] " + " ".join(
                    f"{key}={value:.4f}s" for key, value in means.items()
                ))
                _log({f"profile/{key}_seconds": value for key, value in means.items()}, step)

        if step % int(cfg.training.log_every) == 0:
            dt = (time.time() - t_last) / int(cfg.training.log_every)
            t_last = time.time()
            payload = {
                "train/loss": loss.detach().item(),
                "train/nll": res["nll"].detach().item(),
                "train/crps": res["crps"].detach().item(),
                "train/pinball": res["pinball"].detach().item(),
                "train/distill": res["distill"].detach().item(),
                "train/raw_crossing_frac": res["raw_crossing_frac"],
                "train/anchor": anchor_val,
                "train/grad_norm": gnorm.detach().item(),
                "train/lr": sched.get_last_lr()[0],
                "train/sec_per_step": dt,
                "train/is_era5_batch": float(use_era5),
                "train/P": int(episodes[0]["x_norm_train"].shape[0]),
            }
            if not use_era5:
                payload["train/nll_oracle"] = res["oracle_nll"]
                payload["train/nll_gap_to_oracle"] = res["nll_gap_to_oracle"]
            _log(payload, step)
            print(
                f"step {step:>7} loss={loss.detach().item():.4f} "
                f"nll={res['nll'].detach().item():.4f} "
                f"distill={res['distill'].detach().item():.4f} "
                f"pinball={res['pinball'].detach().item():.4f} "
                f"cross={res['raw_crossing_frac']:.3%} "
                f"gap={res.get('nll_gap_to_oracle', float('nan')):.4f} "
                f"lr={sched.get_last_lr()[0]:.2e} {dt:.2f}s/step"
                + ("  [era5]" if use_era5 else "")
            )

        hooks_started = time.time()
        if step % int(cfg.training.val_every) == 0:
            _consider_validation(step, _validate(step))
        if step % int(cfg.training.save_every) == 0:
            _save(step)
        # Do not charge validation/checkpoint I/O to the next sec_per_step window.
        t_last += time.time() - hooks_started

    if total_steps % int(cfg.training.val_every) != 0:
        _consider_validation(total_steps, _validate(total_steps))

    if bool(cfg.training.get("restore_best", True)):
        _restore_trainable(best_state)
        print(
            f"[selection] restored step {best_step} with "
            f"{selection_metric}={best_value:.6f} before final export"
        )
        export_step = best_step
    else:
        export_step = total_steps
    final = _save(export_step, tag="_final")
    if final:
        print(
            "\nPhase A done. Use it as the copula run's marginal with:\n"
            f"    python src/train.py tabicl.pit_ckpt={os.path.abspath(final)}\n"
            "and measure it first with:\n"
            f"    python eval/runners/marginal_calibration_eval.py --ckpt {os.path.abspath(final)}"
        )
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
