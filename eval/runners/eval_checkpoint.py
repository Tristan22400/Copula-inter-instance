"""eval_checkpoint.py — Evaluate a CopulaTabICL checkpoint against every
classical baseline (see eval/baselines/classical.py) plus the oracle
lower bound, on held-out PIT episodes.

This is the CLAUDE.md-documented daily workflow's evaluation entrypoint,
replacing src/evaluate_baselines.py.

Usage
-----
    python eval/runners/eval_checkpoint.py \\
        --config conf/config.yaml \\
        --ckpt   ./checkpoints/copula_transformer/step_XXXXXX_final.pt \\
        [--n_episodes 50]         # episodes to evaluate
        [--episode_idx 0]         # starting episode index
        [--n_steps_mle 300]       # Adam steps for GP MLE fitting (also used for ARD variants)
        [--lr_mle 0.05]           # learning rate for GP MLE
        [--n_steps_dkl 300]       # Adam steps for Deep Kernel Learning (MLP+GP) fitting
        [--lr_dkl 0.01]           # learning rate for DKL Adam
        [--n_steps_per_ep 500]    # training steps for PerEpisodeTransformer
        [--patience_per_ep 100]   # early stopping patience (steps without improvement)
        [--z_train_source oracle] # or 'tabicl': feed the ICL model TabICL's own
                                   # K-fold PIT estimate of z_train instead of the
                                   # exact GP-LOO one, to measure the sim-to-real gap
        [--tabicl_ckpt ...]        # TabICL checkpoint for --z_train_source=tabicl
        [--tabicl_pit_k_folds 10]  # K-fold count for --z_train_source=tabicl
        [--plot_episode 0]        # local episode index to plot corr_grid for
        [--out_dir ./eval/results]  # directory to save corr_grid figure
        [--device auto]
        [--seed 42]
        [--baseline_cache ./baseline_cache.pt]  # cache fitted baseline results across runs
        [--no_baseline_cache]      # disable the cache entirely
        [--refresh_baselines]      # ignore cached entries, refit and overwrite them

Baseline caching is handled by eval/baselines/classical.py (see its module
docstring): GP-MLE/DKL/per_ep_transformer fitting dominates runtime and is
unaffected by which checkpoint is under test, so repeated runs against a new
checkpoint reuse the cached fits and only redo the cheap ICL forward pass +
oracle NLL.

With --live_generate (the default), the episodes themselves come from
--config's own cfg.data — resolved through Hydra's defaults list, NOT the
checkpoint's own saved training cfg (see _load_full_config) — so the same
--config + --seed always produces the same episodes and the same baseline
cache fingerprint no matter which --ckpt you point at, even across
checkpoints trained under different cfg.data. The tradeoff: every checkpoint
is scored against one shared distribution (whatever --config currently
says) rather than its own training distribution. Edit --config or pass a
different file to change that distribution deliberately.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from collections import Counter

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch import Tensor

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gen import _parse_composite, generate_gp_batch  # noqa: E402
from dataset import CopulaDataset  # noqa: E402
from inference.copula_inference import load_copula_model  # noqa: E402
from model import low_rank_correlation  # noqa: E402
from pit import DEFAULT_K_FOLDS, gp_analytical_posterior, load_tabicl, normalize_targets, run_pit  # noqa: E402

from eval.baselines.classical import (  # noqa: E402
    EXPECTED_BASELINE_KEYS,
    baseline_fingerprint,
    corr_nll_single,
    episode_cache_key,
    eval_baselines_episode,
    load_baseline_cache,
    save_baseline_cache,
)
from eval.viz.correlation_plots import plot_corr_grid  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_full_config(config_path: str) -> OmegaConf:
    """Resolve --config through Hydra's defaults list (model/data groups +
    _self_), the same composition train.py's @hydra.main gets, instead of a
    bare OmegaConf.load (which would leave cfg.data missing entirely — see
    conf/config.yaml's `defaults:` block).

    This is deliberately independent of any checkpoint: it's the fixed
    episode-generating distribution used for live generation and for the
    baseline cache's fingerprint (see main()), so switching --ckpt between
    checkpoints trained under different cfg.data no longer invalidates cached
    baseline fits — only editing --config itself, or passing a different
    one, does.
    """
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        return hydra.compose(config_name=config_name)


# ---------------------------------------------------------------------------
# ICL model + oracle evaluation (the cheap, per-checkpoint part)
# ---------------------------------------------------------------------------


def _eval_icl_episode(
    ep: dict,
    icl_model: nn.Module,
    device: torch.device,
    z_train_override: Tensor | None = None,
) -> tuple[dict[str, float], dict[str, Tensor], Tensor, dict[str, float]]:
    """Evaluate just the ICL model + oracle lower bound on one episode — the
    cheap, per-checkpoint part of the comparison (no fitting/training loop),
    always recomputed even when the baseline results are served from cache.

    z_train_override, when given (see --z_train_source=tabicl in main()),
    replaces the episode's own exact GP-LOO z_train as the ICL model's
    conditioning input — everything else (z_test, R_oracle, the baselines)
    still scores/fits against the episode's true values, since only the
    model's *input* is meant to change, not what "correct" means.

    Returns (nlls, R_dict, R_oracle, y_space_nlls) — y_space_nlls is a
    separate {"prior": ..., "posterior": ...} dict (see gp_analytical_
    posterior's docstring for why these two aren't folded into `nlls`
    alongside icl/oracle/baselines: they're a full multivariate-normal
    Y-space NLL, not the z-space copula-only NLL every other entry in
    `nlls` is, so they're not in the same units/comparable via the same
    table — nan values for both keys when unavailable).
    """
    X_train = ep["x_norm_train"].to(device)   # (P, d_x)
    z_train = (
        z_train_override.to(device) if z_train_override is not None
        else ep["z_train"].to(device)
    )                                            # (P,)  ICL's conditioning input — oracle LOO-PIT residual by default
    X_test  = ep["x_norm_test"].to(device)     # (N, d_x)
    z_test  = ep["z_test"].to(device)          # (N,)
    R_oracle = ep["R_star"].to(device)         # (N, N)

    P, N = X_train.shape[0], X_test.shape[0]
    nlls: dict[str, float] = {}
    R_dict: dict[str, Tensor] = {}
    R_I = torch.eye(N, dtype=X_train.dtype, device=device)

    try:
        train_mask = torch.ones(1, P, dtype=torch.bool, device=device)
        batch = {
            "x_train":   X_train.unsqueeze(0),
            "x_test":    X_test.unsqueeze(0),
            "z_train":   z_train.unsqueeze(0),
            "train_mask": train_mask,
        }
        with torch.no_grad():
            out = icl_model(batch)
            Sigma_icl = low_rank_correlation(
                out["W"],
                out.get("s"),
                parametrization=getattr(icl_model, "correlation_parametrization", "covnorm"),
                lam=out.get("lam"),
            )  # (1, N, N)
        R_icl = Sigma_icl[0, :N, :N]
        nlls["icl"] = corr_nll_single(R_icl, z_test)
        R_dict["icl"] = R_icl
    except Exception as exc:
        print(f"  [icl] failed: {exc}")
        nlls["icl"] = float("nan")
        R_dict["icl"] = R_I.clone()

    nlls["oracle"] = corr_nll_single(R_oracle, z_test)
    R_dict["oracle"] = R_oracle

    # "oracle" above is the PRIOR reference (cfg.data.oracle_mode="prior" —
    # the only mode data_gen.py's training pipeline supports): R_star = raw
    # kernel correlation among test points, never conditioned on the
    # realized (x_train, y_train). It is NOT a true lower bound in the sense
    # of "the best achievable full predictive NLL" — the actual Bayes-optimal
    # reference additionally Schur-complement-conditions on context, computed
    # below via gp_analytical_posterior as a SEPARATE total Y-space NLL (not
    # folded into `nlls`/`R_dict`'s z-space-copula-only comparison — see that
    # function's docstring for why the two aren't in the same units). R_post
    # is still stashed into R_dict purely for the correlation-grid plot (a
    # descriptive visual of what conditioning does to the correlation
    # structure), never scored against z_test.
    y_space_nlls = {"prior": float("nan"), "posterior": float("nan")}
    try:
        post = gp_analytical_posterior(ep)
        R_dict["oracle_posterior"] = post["R_post"].to(device)
        y_space_nlls = {"prior": post["nll_prior"], "posterior": post["nll_post"]}
        if post["repaired"]:
            print(f"    [oracle_posterior] PSD repair fired (min_eig={post['min_eig']:.2e})")
    except (KeyError, NotImplementedError) as exc:
        R_dict["oracle_posterior"] = R_I.clone()
        print(f"  [oracle_posterior] unavailable: {exc}")

    return nlls, R_dict, R_oracle, y_space_nlls


def _tabicl_z_train(
    ep: dict,
    tabicl_marginal: nn.Module,
    k_folds: int,
    device: torch.device,
) -> Tensor | None:
    """K-fold PIT z_train from the frozen TabICL marginal (pit.py::run_pit),
    in place of the episode's exact GP-LOO z_train — the same "does the
    model's correlation prediction hold up against TabICL's own estimated
    marginals instead of the oracle ones" check src/train.py's
    _build_tabicl_val_z runs during training, used here at eval time via
    --z_train_source=tabicl.

    Returns None (caller falls back to the oracle z_train) when the episode
    has fewer than 2 training points, since run_pit's fold split needs at
    least that many.

    y_train is z-scored via pit.normalize_targets before reaching the raw
    TabICL module: run_pit does no target scaling of its own (unlike
    tabicl.TabICLRegressor.fit(), which fits a fresh StandardScaler before
    ever calling this same underlying model). Every other run_pit call site
    in the repo (inference/copula_inference.py::loo_pit,
    train.py::_build_tabicl_val_z) goes through the same helper, so this
    conditioning input is computed identically everywhere. Episode
    y_train's scale is not fixed — outputscale is drawn from a GammaPrior
    (data_gen.py's generative process) — so an unscaled call risks
    saturating the pretrained quantile head's CDF into its extreme tail for
    every point alike on high-outputscale episodes, collapsing z_train's
    spread instead of reflecting the true per-point rank.
    """
    X_train = ep["x_norm_train"].to(device)   # (P, d_x)
    y_train = ep["y_train"].to(device)         # (P,)
    P = X_train.shape[0]
    if P < 2:
        return None
    y_train_scaled, _, _, _ = normalize_targets(y_train)
    Y_train = y_train_scaled.unsqueeze(-1)      # (P, 1)
    pit_out = run_pit(
        tabicl_marginal, X_train, Y_train, X_train[:1], Y_train[:1], k_folds=k_folds,
    )
    return pit_out["z_train"].squeeze(-1)      # (P,)


def _make_folds(n: int, k: int, seed: int) -> list[Tensor]:
    """Deterministic, per-episode partition of the n test-point indices into
    k disjoint folds of near-equal size (sizes differ by at most 1) —
    independent of the global RNG (a fresh CPU-seeded Generator), so it
    doesn't perturb the GP-MLE/DKL restarts' own randomness elsewhere in the
    run.
    """
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    base, extra = divmod(n, k)
    folds: list[Tensor] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        folds.append(perm[start:start + size])
        start += size
    return folds


def _select_best_baseline_cv(
    baseline_R: dict[str, Tensor], z_test: Tensor, n_folds: int, min_fold_size: int, seed: int,
) -> tuple[float, str | None, list[dict]]:
    """Pick the per-episode best *fitted* baseline honestly via nested
    (leave-one-fold-out) cross-validation over the n test points, instead of
    argmin-ing directly over the same z_test the winner is then scored on
    (a selection-bias/winner's-curse leak — see Cawley & Talbot 2010, "On
    Over-fitting in Model Selection and Subsequent Selection Bias in
    Performance Evaluation") and instead of a single fixed val/test split
    (this function's predecessor, _select_best_baseline_holdout), which
    permanently sacrifices a fraction of the points to selection alone —
    wasteful and noisy at this repo's small N (N_min=8).

    For each of K folds: rank candidates by NLL on the other K-1 folds
    (val), then score the winner's NLL on the held-out fold (test) — every
    point plays val in K-1 folds and test in exactly 1, so no point is ever
    used to both select and score the same baseline.

    R_star being a valid (N, N) correlation matrix means any principal
    submatrix R[idx][:, idx] is too, so this needs no refit — the same NxN
    correlation each baseline already produced at fit time is just scored
    against different index subsets.

    K = min(n_folds, n // min_fold_size), so no fold — nor its (K-1)-fold
    val complement, which is always >= one fold's own size — ever scores a
    candidate on fewer than min_fold_size points. Returns (nan, None, [])
    when there are no fitted candidates or this leaves K < 2 (no CV
    possible, n_test too small): with min_fold_size=20 and this repo's
    N_min=8/N_max=128 (uniform), that's true for about a quarter of
    episodes (N < 40) — those simply contribute no best_baseline value
    rather than a noisy one (see _print_table's valid-count note).

    The pooled NLL returned is the size-weighted average of the K held-out
    fold NLLs (each already normalized by its own fold size in
    corr_nll_single) — i.e. the total unnormalized NLL summed across the K
    independent fold-blocks, divided by n. fold_details records each fold's
    selection/scores for diagnostics (console printing).
    """
    keys = [k for k in baseline_R if k not in _NON_FITTED_EXCLUDED]
    n = z_test.shape[0]
    if not keys or n // min_fold_size < 2:
        return float("nan"), None, []

    k_folds = min(n_folds, n // min_fold_size)
    folds = [f.to(z_test.device) for f in _make_folds(n, k_folds, seed)]

    def _sub_nll(key: str, idx: Tensor) -> float:
        R = baseline_R[key]
        R_sub = R.index_select(0, idx).index_select(1, idx)
        return corr_nll_single(R_sub, z_test.index_select(0, idx))

    fold_details: list[dict] = []
    weighted_sum = 0.0
    for i, test_idx in enumerate(folds):
        val_idx = torch.cat([f for j, f in enumerate(folds) if j != i])
        val_nll = {key: _sub_nll(key, val_idx) for key in keys}
        selected = min(val_nll, key=val_nll.get)
        test_nll = _sub_nll(selected, test_idx)
        weighted_sum += test_idx.numel() * test_nll
        fold_details.append({
            "fold": i, "size": test_idx.numel(), "selected": selected,
            "val_nll": val_nll[selected], "test_nll": test_nll,
        })

    pooled_nll = weighted_sum / n
    mode_key = Counter(fd["selected"] for fd in fold_details).most_common(1)[0][0]
    return pooled_nll, mode_key, fold_details


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

_METHOD_ORDER = [
    ("independence",        "Independence"),
    ("gp_prior_rbf",        "GP-Prior-RBF"),
    ("gp_mle_rbf",          "GP-MLE-RBF"),
    ("gp_mle_ard_rbf",      "GP-MLE-ARD-RBF"),
    ("gp_mle_matern32",     "GP-MLE-Matern32"),
    ("gp_mle_ard_matern32", "GP-MLE-ARD-Matern32"),
    ("gp_mle_periodic",     "GP-MLE-Periodic"),
    ("gp_mle_ard_periodic", "GP-MLE-ARD-Periodic"),
    ("gp_mle_rq",           "GP-MLE-RQ"),
    ("gp_mle_ard_rq",       "GP-MLE-ARD-RQ"),
    ("gp_mle_dot_product",  "GP-MLE-DotProduct"),
    ("gp_mle_polynomial",   "GP-MLE-Polynomial"),
    ("dkl_rbf",             "Deep Kernel Learning (RBF)"),
    ("dkl_matern32",        "Deep Kernel Learning (Matern32)"),
    ("dkl_rq",              "Deep Kernel Learning (RQ)"),
    ("dkl_dot_product",     "Deep Kernel Learning (DotProduct)"),
    ("per_ep_transformer",  "PerEp-Transformer"),
    ("best_baseline",       "Best-of-Baselines (per-episode)"),
    ("icl",                 "ICL (pretrained)"),
    ("oracle",              "Oracle (prior)"),
]

# Excluded from the "5 best baselines" ranking: independence/gp_prior_rbf
# are trivial, no-fit reference points rather than baselines, icl/oracle
# aren't baselines at all (icl is our model, oracle is a reference, not a
# fitted candidate), and best_baseline is itself derived from this same
# ranking (added after it's computed each episode — see main()'s loop).
_NON_FITTED_EXCLUDED = {"independence", "gp_prior_rbf", "icl", "oracle", "best_baseline"}

# oracle_posterior only ever appears in R_dict (the correlation-grid plot),
# never in `nlls`/_METHOD_ORDER's z-space table — see
# _eval_icl_episode's docstring for why its NLL isn't comparable in that
# table's units. Kept here only so the plot panel gets a readable title.
_METHOD_LABELS = dict(_METHOD_ORDER) | {"oracle_posterior": "Oracle (posterior)"}


def _kernel_composition_label(ep: dict) -> str:
    """Human-readable kernel-composition string for one episode (e.g.
    "rbf(ARD)+periodic, mlp-mixing"), built from the return_kernel_metadata
    fields generate_gp_batch attaches — absent entirely for episodes loaded
    from a pre-built dataset that didn't request that metadata (the common
    case for existing PIT datasets on disk today)."""
    if "kernel" not in ep:
        return "unavailable (pass --dataset_dir with pre-generated metadata, or use --live_generate)"

    if "kernel_components" in ep:
        parts = [ep["kernel_components"][0]]
        for op, comp_name in zip(ep["kernel_ops"], ep["kernel_components"][1:]):
            parts.append(op)
            parts.append(comp_name)
        label = " ".join(parts)
        ard_tags = [
            comp_name + "(ARD)"
            for comp_name, comp_params in zip(ep["kernel_components"], ep["kernel_component_params"])
            if torch.is_tensor(comp_params.get("l")) and comp_params["l"].numel() > 1
        ]
    else:
        label = ep["kernel"]
        composite = _parse_composite(ep["kernel"])
        ard_tags = []
        if composite is None:
            if torch.is_tensor(ep.get("l")) and ep["l"].numel() > 1:
                ard_tags.append(f"{ep['kernel']}(ARD)")
        else:
            name_a, _op, name_b = composite
            if torch.is_tensor(ep.get("l")) and ep["l"].numel() > 1:
                ard_tags.append(f"{name_a}(ARD)")
            if torch.is_tensor(ep.get("l_b")) and ep["l_b"].numel() > 1:
                ard_tags.append(f"{name_b}(ARD)")

    if ard_tags:
        label = f"{label}  [{', '.join(ard_tags)}]"
    if bool(ep.get("mlp_mixed", False)):
        label = f"{label}, mlp-mixing"
    return label


def _live_generate_alternating(gen_cfg, n_ep: int, device, seed: int) -> list[dict]:
    """Live-generate n_ep episodes, forcing every even local index (0, 2, 4,
    ...) to a single elementary kernel (no composition) so each consecutive
    pair of evaluated episodes includes one non-composite draw — otherwise
    non-composite episodes are rare under this repo's default composite
    kernel counts.

    generate_gp_batch samples its kernel structure once per call and shares
    it across the whole batch, so getting per-episode composition variety at
    all requires B=1 calls rather than a single batched B=n_ep call. Each
    call gets its own seed (seed + local_i): generate_gp_batch reseeds every
    RNG from cfg.seed at the start of each call, so reusing one seed across
    calls would otherwise resample the identical episode n_ep times.
    """
    episodes: list[dict] = []
    for local_i in range(n_ep):
        ep_cfg = copy.deepcopy(gen_cfg)
        ep_cfg.seed = seed + local_i
        if local_i % 2 == 0:
            # Force non-composite for both kernel-selection modes
            # _resolve_kernel_name / _sample_kernel_chain_structure support.
            if bool(getattr(ep_cfg.data, "systematic_composition", False)):
                ep_cfg.data.composite_num_kernels_min = 1
                ep_cfg.data.composite_num_kernels_max = 1
            else:
                fixed = getattr(ep_cfg.data, "kernel", None)
                if fixed:
                    composite = _parse_composite(str(fixed))
                    if composite is not None:
                        ep_cfg.data.kernel = composite[0]
                elif getattr(ep_cfg.data, "kernels", None):
                    pool = [k for k in ep_cfg.data.kernels if _parse_composite(k) is None]
                    if not pool:
                        raise ValueError(
                            f"cfg.data.kernels={list(ep_cfg.data.kernels)} contains only "
                            "composite kernels; cannot force a non-composite episode."
                        )
                    ep_cfg.data.kernels = pool
                # else: _resolve_kernel_name's own "rbf" default, already non-composite.
        episodes.extend(generate_gp_batch(ep_cfg, 1, device, return_kernel_metadata=True))
    return episodes


def _print_table(all_nlls: list[dict[str, float]], z_train_source: str = "oracle") -> None:
    means = {k: float(np.nanmean([m.get(k, float("nan")) for m in all_nlls]))
             for k, _ in _METHOD_ORDER}
    stds  = {k: float(np.nanstd( [m.get(k, float("nan")) for m in all_nlls]))
             for k, _ in _METHOD_ORDER}

    col = max(22, max(len(label) for _, label in _METHOD_ORDER) + 2)
    total = col + 2 * 12
    print(f"\n{'─' * total}")
    print(f"Inter-instance copula NLL (z-space) — lower is better  [N={len(all_nlls)} episodes]")
    print(f"ICL z_train source: {z_train_source}"
          + ("  (exact GP-LOO PIT)" if z_train_source == "oracle" else "  (TabICL K-fold PIT estimate)"))
    print(f"{'─' * total}")
    print(f"{'Method':<{col}}{'Mean NLL':>12}{'Std NLL':>12}")
    print(f"{'─' * col}{'─' * 12}{'─' * 12}")
    for key, label in _METHOD_ORDER:
        m, s = means.get(key, float("nan")), stds.get(key, float("nan"))
        marker = ""
        if key == "best_baseline":
            n_valid = sum(1 for ep_m in all_nlls if not np.isnan(ep_m.get(key, float("nan"))))
            marker = (f"  ← per-episode best baseline (nested CV; "
                      f"valid for {n_valid}/{len(all_nlls)} episodes)")
        elif key == "icl":
            marker = "  ← our model"
        elif key == "oracle":
            marker = "  ← unconditional kernel corr. among test pts (NOT Bayes-optimal; see GP oracle Y-space NLL below)"
        print(f"{label:<{col}}{m:>12.4f}{s:>12.4f}{marker}")
    print(f"{'─' * total}\n")


def _print_y_space_oracle(y_space_nlls: list[dict[str, float]]) -> None:
    """Total (marginal + copula) Y-space multivariate-normal GP oracle NLL,
    prior vs. posterior — see gp_analytical_posterior's docstring for why
    this is a SEPARATE table from _print_table's z-space copula-only NLL
    (different units, not just a missing row there): this one directly
    answers "how much does conditioning on context help, in the units of a
    real predictive log-likelihood" and posterior <= prior is a real,
    provable guarantee here (Bayes-optimality of the posterior predictive
    under log-loss), unlike a same-z_test z-space copula comparison.

    Episodes where gp_analytical_posterior was unavailable (systematic-chain
    kernel with whole-chain outer sign modulation, or a --dataset_dir
    episode missing kernel metadata) contribute NaN to both columns and are
    excluded via nanmean/nanstd, same convention as _print_table.
    """
    prior_vals = [d["prior"] for d in y_space_nlls]
    post_vals  = [d["posterior"] for d in y_space_nlls]
    n_valid = sum(1 for d in y_space_nlls if not np.isnan(d["posterior"]))
    print(f"GP oracle total NLL (Y-space, marginal+copula) — lower is better, "
          f"posterior <= prior is a Bayes-optimality guarantee here "
          f"[valid for {n_valid}/{len(y_space_nlls)} episodes]")
    print(f"  prior (unconditioned):      mean={np.nanmean(prior_vals):.4f}  std={np.nanstd(prior_vals):.4f}")
    print(f"  posterior (Schur-conditioned): mean={np.nanmean(post_vals):.4f}  std={np.nanstd(post_vals):.4f}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ICL checkpoint vs baselines on inter-instance copula episodes"
    )
    parser.add_argument("--config",       default="conf/config.yaml",
                        help="Hydra config defining the eval-episode-generating "
                             "distribution (cfg.data) for --live_generate, "
                             "resolved through its own defaults list — "
                             "independent of --ckpt's saved training cfg. "
                             "Keeping this fixed is what lets the baseline "
                             "cache survive switching checkpoints.")
    parser.add_argument("--ckpt",         required=True)
    parser.add_argument("--dataset_dir",  default=None,
                        help="Episode directory to evaluate on (overrides "
                             "training.dataset_dir from --config). Passing "
                             "this disables --live_generate by default.")
    parser.add_argument("--live_generate", action=argparse.BooleanOptionalAction, default=None,
                        help="Generate evaluation episodes on the fly via "
                             "data_gen.generate_gp_batch(..., return_kernel_metadata=True) "
                             "instead of loading a pre-built PIT dataset directory. Default: "
                             "True unless --dataset_dir is given. --episode_idx is ignored "
                             "in this mode (episodes are freshly sampled, not indexed).")
    parser.add_argument("--n_episodes",   type=int,   default=30)
    parser.add_argument("--episode_idx",  type=int,   default=0)
    parser.add_argument("--n_steps_mle",  type=int,   default=1000,
                        help="Adam steps for GP kernel MLE fitting (also used for ARD variants)")
    parser.add_argument("--lr_mle",       type=float, default=0.05,
                        help="Learning rate for GP MLE Adam")
    parser.add_argument("--n_restarts_mle", type=int, default=5,
                        help="Independent random restarts per GP-MLE kernel fit (each "
                             "initialised by sampling from the same LogNormal/Gamma "
                             "hyperpriors data_gen.py's generative process uses); keeps "
                             "whichever restart reaches the best final training loss.")
    parser.add_argument("--n_steps_dkl",  type=int,   default=5000,
                        help="Adam steps for Deep Kernel Learning (MLP+GP) fitting")
    parser.add_argument("--lr_dkl",       type=float, default=0.01,
                        help="Learning rate for DKL Adam")
    parser.add_argument("--n_steps_per_ep", type=int, default=5000,
                        help="Training steps for PerEpisodeTransformer")
    parser.add_argument("--patience_per_ep", type=int, default=500,
                        help="Early stopping patience for PerEpisodeTransformer")
    parser.add_argument("--z_train_source", default="oracle", choices=["oracle", "tabicl"],
                        help="What the ICL model conditions on for each episode's z_train. "
                             "'oracle' (default): the episode's exact GP-LOO PIT residual "
                             "(R&W Eq. 5.12) computed from the true generating kernel — "
                             "unavailable on real data, but exact here since the kernel is "
                             "known by construction. 'tabicl': a K-fold cross-fitted PIT "
                             "estimate from the frozen TabICL marginal (pit.py::run_pit), "
                             "the same proxy real-world evaluation is stuck with — use this "
                             "to measure the sim-to-real gap between the two (icl_nll under "
                             "each source is unaffected in every other respect: z_test, "
                             "R_oracle, and every baseline still score/fit against the "
                             "episode's true values). Adds the cost of one extra frozen "
                             "TabICL forward pass per fold per episode.")
    parser.add_argument("--tabicl_ckpt",  default=None,
                        help="TabICL checkpoint filename for --z_train_source=tabicl. "
                             "Default: read from --config's cfg.tabicl.ckpt.")
    parser.add_argument("--tabicl_pit_k_folds", type=int, default=None,
                        help="K-fold count for --z_train_source=tabicl's run_pit call. "
                             f"Default: cfg.tabicl.pit_k_folds, falling back to "
                             f"pit.DEFAULT_K_FOLDS ({DEFAULT_K_FOLDS}).")
    parser.add_argument("--plot_episode", type=int,   default=0,
                        help="Local episode index to generate the corr_grid plot for")
    parser.add_argument("--out_dir",      default=os.path.join(_REPO_ROOT, "eval", "results"),
                        help="Directory for saved corr_grid figure")
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--n_folds",      type=int,   default=5,
                        help="Number of folds for nested (leave-one-fold-out) "
                             "cross-validation of the per-episode best_baseline pick: "
                             "each fold's held-out NLL is scored using a baseline "
                             "selected (argmin NLL) on the other K-1 folds only, and "
                             "the K fold NLLs are pooled — instead of argmin-ing "
                             "directly over z_test, which lets the winner peek at the "
                             "very data it's scored on (see _select_best_baseline_cv). "
                             "Actual K is capped at n_test // 3 per episode, so small-N "
                             "episodes automatically use fewer folds. The fold split is "
                             "deterministic per episode (seeded from --seed + episode "
                             "index) and never touches icl/oracle or any individual "
                             "baseline's own diagnostic row, which keep scoring against "
                             "the full test set as before.")
    parser.add_argument("--min_fold_size", type=int, default=20,
                        help="Minimum points required on both sides of every "
                             "leave-one-fold-out split (a fold's own size, and its "
                             "(K-1)-fold val complement) before a candidate baseline's "
                             "NLL there is trusted enough to base a selection on. Below "
                             "this, ranking ~12 candidate kernels by NLL on a handful of "
                             "points is closer to a coin flip than a real comparison, and "
                             "an occasional spurious pick of a blow-up-prone candidate "
                             "(e.g. dot_product/polynomial kernels, whose NLL is "
                             "unbounded above when misspecified) can drag the averaged "
                             "best_baseline mean above even a single steady fixed kernel "
                             "used for every episode. Effective K per episode is "
                             "min(--n_folds, n_test // min_fold_size); episodes where "
                             "that's < 2 report best_baseline=nan (excluded from its "
                             "mean/std, not silently zero) instead of forcing a low-"
                             "confidence selection.")
    parser.add_argument("--min_test_points", type=int, default=None,
                        help="Floor on eval episodes' test-point count N, so every episode "
                             "is valid for the nested-CV best_baseline selection (see "
                             "--min_fold_size: _select_best_baseline_cv needs n_test // "
                             "min_fold_size >= 2 folds, i.e. n_test >= 2 * min_fold_size, "
                             "or it reports best_baseline=nan for that episode — silently "
                             "excluding it from the summary table's best_baseline mean while "
                             "every other row's mean still includes it). Default: "
                             "2 * --min_fold_size (40 under the argparse defaults). For "
                             "--live_generate (the default), this raises cfg.data.N_min for "
                             "this run only — conf/data/gp_tasks.yaml's own N_min=8, and "
                             "hence training's distribution, is untouched. For --dataset_dir, "
                             "episode sizes are fixed by the pre-built dataset and can't be "
                             "regenerated, so episodes below this floor are skipped instead.")
    parser.add_argument("--oracle_mode",  default=None, choices=["prior", "posterior"],
                        help="How R_star was built for this dataset. Determines whether "
                             "GP-MLE/DKL score the fitted kernel's posterior (conditioned "
                             "on X_train) or its raw prior covariance at X_test. Default: "
                             "read from the checkpoint's own saved training config "
                             "(cfg.data.oracle_mode), falling back to 'prior' if absent.")
    parser.add_argument("--baseline_cache", default="./baseline_cache.pt",
                        help="Path to a cache file storing every classical baseline's fitted "
                             "NLL/correlation results, keyed per-episode. These are the "
                             "expensive, checkpoint-independent part of the comparison; the "
                             "ICL model + oracle are always recomputed fresh since they're "
                             "what actually changes between runs. A cache entry is only "
                             "reused when the episode-generating config and every baseline-"
                             "fitting hyperparameter below match exactly what produced it "
                             "(see eval.baselines.classical.baseline_fingerprint) — otherwise "
                             "it's recomputed and the cache updated in place.")
    parser.add_argument("--no_baseline_cache", action="store_true",
                        help="Disable baseline caching entirely: always recompute, never "
                             "read or write --baseline_cache.")
    parser.add_argument("--refresh_baselines", action="store_true",
                        help="Recompute every baseline even if a matching cache entry "
                             "exists, overwriting it (still writes --baseline_cache unless "
                             "--no_baseline_cache is also given).")
    args = parser.parse_args()

    _set_seed(args.seed)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    print(f"Device: {device}")

    # cfg is the eval-episode-generating config, resolved from --config's own
    # Hydra defaults (model/data groups + _self_) — deliberately NOT the
    # checkpoint's own saved training cfg. Keeping it fixed across --ckpt
    # values is what lets the baseline cache (see fingerprint below) survive
    # switching between checkpoints trained under different cfg.data: the
    # tradeoff is that all checkpoints are now scored against one shared
    # distribution rather than each against its own training distribution.
    # Point --config at a different file (or edit this one) to change it.
    cfg = _load_full_config(args.config)

    # ---- Load ICL model ----
    print(f"\nLoading ICL checkpoint: {args.ckpt}")
    icl_model, icl_cfg = load_copula_model(args.ckpt, config_path=args.config, device=str(device))
    icl_rank = int(icl_cfg.model.rank)
    n_params = sum(p.numel() for p in icl_model.parameters())
    print(f"ICL model parameters: {n_params:,}  rank={icl_rank}")

    # ---- Optionally load a second, frozen TabICL marginal purely to
    # K-fold-PIT each episode's z_train (see --z_train_source's help text) ----
    tabicl_marginal: nn.Module | None = None
    tabicl_pit_k_folds = DEFAULT_K_FOLDS
    if args.z_train_source == "tabicl":
        tabicl_ckpt = args.tabicl_ckpt or OmegaConf.select(cfg, "tabicl.ckpt", default=None)
        if not tabicl_ckpt:
            raise ValueError(
                "--z_train_source=tabicl requires a TabICL checkpoint: pass --tabicl_ckpt "
                "or set cfg.tabicl.ckpt in --config."
            )
        tabicl_pit_k_folds = args.tabicl_pit_k_folds or int(
            OmegaConf.select(cfg, "tabicl.pit_k_folds", default=DEFAULT_K_FOLDS)
        )
        print(f"\nLoading frozen TabICL marginal for --z_train_source=tabicl: {tabicl_ckpt} "
              f"(k_folds={tabicl_pit_k_folds})")
        tabicl_marginal = load_tabicl(tabicl_ckpt, str(device))
    print(f"z_train source (ICL conditioning input): {args.z_train_source}")

    # GP-MLE/DKL must score against the same convention used to build this
    # run's R_star ("prior" ignores training conditioning entirely,
    # "posterior" conditions on X_train) — see classical.fit_and_eval_gpytorch's
    # docstring. Read from cfg (the fixed eval-generating config above), the
    # actual generation config for these episodes; falls back to "prior"
    # (this repo's current datasets all use oracle_mode=prior, unlike
    # data_gen.py's own historical "posterior" default for dataset
    # *generation*).
    oracle_mode = args.oracle_mode or OmegaConf.select(cfg, "data.oracle_mode", default="prior")
    print(f"Oracle mode: {oracle_mode}")

    # GP-MLE/DKL hyperpriors: read the exact LogNormal/Gamma constants these
    # episodes are actually generated with (cfg, not the checkpoint's own
    # training cfg — see cfg's definition above), falling back to
    # classical._DEFAULT_PRIOR_CFG for any missing key.
    data_cfg = OmegaConf.select(cfg, "data", default=None)
    prior_cfg = OmegaConf.to_container(data_cfg) if data_cfg is not None else {}
    print(f"GP-MLE restarts: {args.n_restarts_mle}")

    # Live-generate by default, unless the user points at a fixed dataset
    # with --dataset_dir (see --live_generate's help text).
    live_generate = args.live_generate if args.live_generate is not None else (args.dataset_dir is None)

    n_ep = args.n_episodes
    all_nlls: list[dict[str, float]] = []
    all_y_space_nlls: list[dict[str, float]] = []
    plot_R_dict: dict[str, Tensor] | None = None
    plot_R_oracle: Tensor | None = None
    plot_best_key: str | None = None
    plot_best_R: Tensor | None = None

    # Eval-only floor on episodes' test-point count N (see --min_test_points'
    # help text) — 2 * --min_fold_size by default, the minimum n_test
    # _select_best_baseline_cv needs for >=2 CV folds.
    min_test_points = args.min_test_points if args.min_test_points is not None else 2 * args.min_fold_size

    if live_generate:
        # cfg (the fixed eval-generating config, not icl_cfg) drives live
        # generation — same source already used for prior_cfg above — so
        # every checkpoint evaluated against this --config gets identical
        # episodes for a given seed, regardless of what that checkpoint was
        # itself trained on.
        if cfg.data.N_min < min_test_points:
            print(f"Raising eval episode N_min {cfg.data.N_min} -> {min_test_points} "
                  f"(--min_test_points) for this run only — training's own "
                  "conf/data/gp_tasks.yaml N_min is untouched")
            cfg.data.N_min = min_test_points
            if cfg.data.N_max < cfg.data.N_min:
                cfg.data.N_max = cfg.data.N_min
        print(f"\nLive-generating {n_ep} episodes via generate_gp_batch "
              f"(return_kernel_metadata=True), seed={args.seed}, "
              "alternating every-other episode to a non-composite kernel")
        live_episodes = _live_generate_alternating(cfg, n_ep, device, args.seed)
    else:
        dataset_dir = args.dataset_dir or cfg.training.dataset_dir
        dataset = CopulaDataset(episode_dir=dataset_dir)
        n_available = len(dataset)
        print(f"\nEvaluating {n_ep} episodes from {dataset_dir} (start={args.episode_idx})")
        print(f"  Dataset size: {n_available} episodes")

    print(f"  GP MLE: {args.n_steps_mle} steps | DKL: {args.n_steps_dkl} steps | "
          f"PerEp: {args.n_steps_per_ep} steps (patience={args.patience_per_ep})")

    # ---- Baseline cache: skip re-fitting GP-MLE/DKL/per_ep_transformer for
    # episodes already scored under an identical generation/fitting config ----
    use_cache = not args.no_baseline_cache
    fingerprint = baseline_fingerprint(
        cfg, live_generate, args.dataset_dir, args.seed, icl_rank, oracle_mode,
        args.n_steps_mle, args.lr_mle, args.n_restarts_mle,
        args.n_steps_dkl, args.lr_dkl, args.n_steps_per_ep, args.patience_per_ep,
    )
    cache_entries = load_baseline_cache(args.baseline_cache, fingerprint) if use_cache else {}
    cache_dirty = False

    for local_i in range(n_ep):
        if live_generate:
            ep_i = local_i
            ep = live_episodes[local_i]
        else:
            ep_i = args.episode_idx + local_i
            if ep_i >= n_available:
                print(f"  [ep {ep_i}] index out of range ({n_available} available), skipping")
                continue
            ep = dataset[ep_i]
            n_test = ep["z_test"].shape[0]
            if n_test < min_test_points:
                print(f"  [ep {ep_i}] only {n_test} test points (< --min_test_points="
                      f"{min_test_points}), skipping — best_baseline needs enough for "
                      ">=2 nested-CV folds")
                continue

        cache_key = episode_cache_key(live_generate, args.dataset_dir, args.seed, local_i, ep_i)
        cached = cache_entries.get(cache_key) if (use_cache and not args.refresh_baselines) else None
        if cached is not None and not EXPECTED_BASELINE_KEYS.issubset(cached["nlls"].keys()):
            # Same episode/fingerprint, but this entry predates a baseline
            # that was added to eval_baselines_episode since it was cached
            # (e.g. gp_mle_polynomial) — refit everything for this episode
            # rather than silently serving a result with that key missing.
            missing = EXPECTED_BASELINE_KEYS - cached["nlls"].keys()
            print(f"  [ep {ep_i}] cached baselines missing {sorted(missing)} — refitting")
            cached = None
        if cached is not None:
            baseline_nlls = cached["nlls"]
            baseline_R    = {k: v.to(device) for k, v in cached["R_dict"].items()}
        else:
            baseline_nlls, baseline_R = eval_baselines_episode(
                ep=ep,
                icl_rank=icl_rank,
                n_steps_mle=args.n_steps_mle,
                lr_mle=args.lr_mle,
                n_steps_dkl=args.n_steps_dkl,
                lr_dkl=args.lr_dkl,
                n_steps_per_ep=args.n_steps_per_ep,
                patience_per_ep=args.patience_per_ep,
                device=device,
                oracle_mode=oracle_mode,
                prior_cfg=prior_cfg,
                n_restarts_mle=args.n_restarts_mle,
            )
            if use_cache:
                cache_entries[cache_key] = {
                    "nlls": baseline_nlls,
                    "R_dict": {k: v.cpu() for k, v in baseline_R.items()},
                }
                cache_dirty = True

        z_train_override = None
        if tabicl_marginal is not None:
            z_train_override = _tabicl_z_train(
                ep=ep, tabicl_marginal=tabicl_marginal, k_folds=tabicl_pit_k_folds, device=device,
            )
            if z_train_override is None:
                print(f"  [ep {ep_i}] fewer than 2 training points — "
                      "falling back to oracle z_train for this episode")

        icl_nlls, icl_R, R_oracle, y_space_nlls = _eval_icl_episode(
            ep=ep, icl_model=icl_model, device=device, z_train_override=z_train_override,
        )
        all_y_space_nlls.append(y_space_nlls)

        nlls   = {**baseline_nlls, **icl_nlls}
        R_dict = {**baseline_R, **icl_R}

        icl_nll = nlls.get("icl", float("nan"))
        ora_nll = nlls.get("oracle", float("nan"))
        ranked_baselines = sorted(
            ((k, v) for k, v in nlls.items() if k not in _NON_FITTED_EXCLUDED),
            key=lambda kv: kv[1],
        )
        top5 = ranked_baselines[:5]
        # Per-episode best fitted baseline's NLL, selected via nested
        # (leave-one-fold-out) CV over this episode's test points (see
        # _select_best_baseline_cv) — averaging this across episodes (see
        # _print_table's "Best-of-Baselines" row) is a tighter,
        # per-episode-optimal reference than any single baseline's own
        # average, so its gap to ICL's mean is the real "how much is ICL
        # leaving on the table vs. always picking the best baseline" number,
        # without the winner having been picked by peeking at the same
        # z_test it's scored on.
        holdout_seed = (args.seed * 1_000_003 + ep_i) % (2 ** 31 - 1)
        best_nll, mode_key, fold_details = _select_best_baseline_cv(
            baseline_R, ep["z_test"].to(device), args.n_folds, args.min_fold_size, holdout_seed,
        )
        nlls["best_baseline"] = best_nll
        all_nlls.append(nlls)

        if local_i == args.plot_episode:
            plot_R_dict   = R_dict
            plot_R_oracle = R_oracle
            if mode_key is not None:
                plot_best_key = mode_key
                plot_best_R   = R_dict[mode_key]

        print(f"  ep {ep_i:04d}: kernel={_kernel_composition_label(ep)}")
        print(f"    icl={icl_nll:.4f}  oracle(prior, z-space copula)={ora_nll:.4f}  "
              f"GP-oracle-y-space(prior={y_space_nlls['prior']:.4f}, "
              f"posterior={y_space_nlls['posterior']:.4f})")
        if fold_details:
            fold_summary = ", ".join(
                f"fold{fd['fold']}={_METHOD_LABELS.get(fd['selected'], fd['selected'])}"
                for fd in fold_details
            )
            print(f"    best_baseline (nested {len(fold_details)}-fold CV, "
                  f"pooled test NLL)={best_nll:.4f}")
            print(f"      per-fold picks: {fold_summary}")
        else:
            print("    best_baseline: unavailable (too few test points for nested CV)")
        print("    top-5 baselines (lowest NLL on full test set, diagnostic only — "
              "not the selection used for best_baseline above):")
        for key, val in top5:
            print(f"      {_METHOD_LABELS.get(key, key):<28}{val:.4f}")

    if use_cache and cache_dirty:
        save_baseline_cache(args.baseline_cache, fingerprint, cache_entries)

    if not all_nlls:
        print("No episodes evaluated successfully.")
        return

    _print_table(all_nlls, z_train_source=args.z_train_source)
    _print_y_space_oracle(all_y_space_nlls)

    # ---- Correlation heatmap ----
    if plot_R_dict is not None and plot_R_oracle is not None:
        import matplotlib
        matplotlib.use("Agg")

        os.makedirs(args.out_dir, exist_ok=True)
        # Exclude oracle from estimators dict (it's passed separately). Move
        # icl to the end and insert the best-performing fitted baseline for
        # this episode (lowest NLL, excluding icl/oracle/independence/
        # gp_prior_rbf — same ranking as the "top-5 baselines" console print
        # above) right before it, so oracle / best-baseline / icl sit next to
        # each other for a quick visual comparison instead of having to scan
        # all the individual baseline panels.
        estimators = {k: v for k, v in plot_R_dict.items() if k != "oracle"}
        icl_panel = estimators.pop("icl", None)
        if plot_best_key is not None:
            best_label = f"best_baseline ({_METHOD_LABELS.get(plot_best_key, plot_best_key)})"
            estimators[best_label] = plot_best_R
        if icl_panel is not None:
            estimators["icl"] = icl_panel
        fig = plot_corr_grid(
            estimators=estimators,
            oracle_R=plot_R_oracle,
            title=f"Correlation estimators — episode {args.episode_idx + args.plot_episode}",
        )
        out_path = os.path.join(args.out_dir, f"corr_grid_ep{args.plot_episode}.png")
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        print(f"Saved corr_grid to: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
