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
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
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

from eval.baselines.classical import (  # noqa: E402
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


# ---------------------------------------------------------------------------
# ICL model + oracle evaluation (the cheap, per-checkpoint part)
# ---------------------------------------------------------------------------


def _eval_icl_episode(
    ep: dict,
    icl_model: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Tensor], Tensor]:
    """Evaluate just the ICL model + oracle lower bound on one episode — the
    cheap, per-checkpoint part of the comparison (no fitting/training loop),
    always recomputed even when the baseline results are served from cache.
    """
    X_train = ep["x_norm_train"].to(device)   # (P, d_x)
    z_train = ep["z_train"].to(device)         # (P,)  oracle LOO-PIT residual, used to train icl only
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
            Sigma_icl = low_rank_correlation(out["W"], out["s"])  # (1, N, N)
        R_icl = Sigma_icl[0, :N, :N]
        nlls["icl"] = corr_nll_single(R_icl, z_test)
        R_dict["icl"] = R_icl
    except Exception as exc:
        print(f"  [icl] failed: {exc}")
        nlls["icl"] = float("nan")
        R_dict["icl"] = R_I.clone()

    nlls["oracle"] = corr_nll_single(R_oracle, z_test)
    R_dict["oracle"] = R_oracle

    return nlls, R_dict, R_oracle


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
    ("dkl_rbf",             "Deep Kernel Learning (RBF)"),
    ("dkl_matern32",        "Deep Kernel Learning (Matern32)"),
    ("dkl_rq",              "Deep Kernel Learning (RQ)"),
    ("dkl_dot_product",     "Deep Kernel Learning (DotProduct)"),
    ("per_ep_transformer",  "PerEp-Transformer"),
    ("icl",                 "ICL (pretrained)"),
    ("oracle",              "Oracle"),
]

# Excluded from the "5 best baselines" ranking: independence/gp_prior_rbf
# are trivial, no-fit reference points rather than baselines, and icl/oracle
# aren't baselines at all (icl is our model, oracle is the lower bound).
_NON_FITTED_EXCLUDED = {"independence", "gp_prior_rbf", "icl", "oracle"}

_METHOD_LABELS = dict(_METHOD_ORDER)


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


def _live_generate_alternating(icl_cfg, n_ep: int, device, seed: int) -> list[dict]:
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
        ep_cfg = copy.deepcopy(icl_cfg)
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


def _print_table(all_nlls: list[dict[str, float]]) -> None:
    means = {k: float(np.nanmean([m.get(k, float("nan")) for m in all_nlls]))
             for k, _ in _METHOD_ORDER}
    stds  = {k: float(np.nanstd( [m.get(k, float("nan")) for m in all_nlls]))
             for k, _ in _METHOD_ORDER}

    col = max(22, max(len(label) for _, label in _METHOD_ORDER) + 2)
    total = col + 2 * 12
    print(f"\n{'─' * total}")
    print(f"Inter-instance copula NLL (z-space) — lower is better  [N={len(all_nlls)} episodes]")
    print(f"{'─' * total}")
    print(f"{'Method':<{col}}{'Mean NLL':>12}{'Std NLL':>12}")
    print(f"{'─' * col}{'─' * 12}{'─' * 12}")
    for key, label in _METHOD_ORDER:
        m, s = means.get(key, float("nan")), stds.get(key, float("nan"))
        marker = ""
        if key == "icl":
            marker = "  ← our model"
        elif key == "oracle":
            marker = "  ← lower bound"
        print(f"{label:<{col}}{m:>12.4f}{s:>12.4f}{marker}")
    print(f"{'─' * total}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ICL checkpoint vs baselines on inter-instance copula episodes"
    )
    parser.add_argument("--config",       default="conf/config.yaml")
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
    parser.add_argument("--plot_episode", type=int,   default=0,
                        help="Local episode index to generate the corr_grid plot for")
    parser.add_argument("--out_dir",      default=os.path.join(_REPO_ROOT, "eval", "results"),
                        help="Directory for saved corr_grid figure")
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--seed",         type=int,   default=42)
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

    cfg = OmegaConf.load(args.config)

    # ---- Load ICL model ----
    print(f"\nLoading ICL checkpoint: {args.ckpt}")
    icl_model, icl_cfg = load_copula_model(args.ckpt, config_path=args.config, device=str(device))
    icl_rank = int(icl_cfg.model.rank)
    n_params = sum(p.numel() for p in icl_model.parameters())
    print(f"ICL model parameters: {n_params:,}  rank={icl_rank}")

    # GP-MLE/DKL must score against the same convention used to build this
    # dataset's R_star ("prior" ignores training conditioning entirely,
    # "posterior" conditions on X_train) — see classical.fit_and_eval_gpytorch's
    # docstring. Read from the checkpoint's own saved training cfg by
    # default, since that's the actual generation config for this run's
    # dataset; falls back to "prior" (this repo's current datasets all use
    # oracle_mode=prior, unlike data_gen.py's own historical "posterior"
    # default for dataset *generation*).
    oracle_mode = args.oracle_mode or OmegaConf.select(icl_cfg, "data.oracle_mode", default="prior")
    print(f"Oracle mode: {oracle_mode}")

    # GP-MLE/DKL hyperpriors: read the exact LogNormal/Gamma constants this
    # checkpoint's dataset was generated with, falling back to
    # classical._DEFAULT_PRIOR_CFG for any missing key (e.g. an older
    # checkpoint saved before a given key existed).
    data_cfg = OmegaConf.select(icl_cfg, "data", default=None)
    prior_cfg = OmegaConf.to_container(data_cfg) if data_cfg is not None else {}
    print(f"GP-MLE restarts: {args.n_restarts_mle}")

    # Live-generate by default, unless the user points at a fixed dataset
    # with --dataset_dir (see --live_generate's help text).
    live_generate = args.live_generate if args.live_generate is not None else (args.dataset_dir is None)

    n_ep = args.n_episodes
    all_nlls: list[dict[str, float]] = []
    plot_R_dict: dict[str, Tensor] | None = None
    plot_R_oracle: Tensor | None = None

    if live_generate:
        # icl_cfg is the checkpoint's own saved training config (same source
        # already used for prior_cfg above), so live-generated episodes match
        # the kernel family/hyperprior distribution the ICL model was
        # actually trained on.
        print(f"\nLive-generating {n_ep} episodes via generate_gp_batch "
              f"(return_kernel_metadata=True), seed={args.seed}, "
              "alternating every-other episode to a non-composite kernel")
        live_episodes = _live_generate_alternating(icl_cfg, n_ep, device, args.seed)
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
        icl_cfg, live_generate, args.dataset_dir, args.seed, icl_rank, oracle_mode,
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

        cache_key = episode_cache_key(live_generate, args.dataset_dir, args.seed, local_i, ep_i)
        cached = cache_entries.get(cache_key) if (use_cache and not args.refresh_baselines) else None
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

        icl_nlls, icl_R, R_oracle = _eval_icl_episode(ep=ep, icl_model=icl_model, device=device)

        nlls   = {**baseline_nlls, **icl_nlls}
        R_dict = {**baseline_R, **icl_R}
        all_nlls.append(nlls)

        if local_i == args.plot_episode:
            plot_R_dict   = R_dict
            plot_R_oracle = R_oracle

        icl_nll = nlls.get("icl", float("nan"))
        ora_nll = nlls.get("oracle", float("nan"))
        top5 = sorted(
            ((k, v) for k, v in nlls.items() if k not in _NON_FITTED_EXCLUDED),
            key=lambda kv: kv[1],
        )[:5]
        print(f"  ep {ep_i:04d}: kernel={_kernel_composition_label(ep)}")
        print(f"    icl={icl_nll:.4f}  oracle={ora_nll:.4f}")
        print("    top-5 baselines (lowest NLL, fitted only):")
        for key, val in top5:
            print(f"      {_METHOD_LABELS.get(key, key):<28}{val:.4f}")

    if use_cache and cache_dirty:
        save_baseline_cache(args.baseline_cache, fingerprint, cache_entries)

    if not all_nlls:
        print("No episodes evaluated successfully.")
        return

    _print_table(all_nlls)

    # ---- Correlation heatmap ----
    if plot_R_dict is not None and plot_R_oracle is not None:
        import matplotlib
        matplotlib.use("Agg")

        os.makedirs(args.out_dir, exist_ok=True)
        # Exclude oracle from estimators dict (it's passed separately)
        estimators = {k: v for k, v in plot_R_dict.items() if k != "oracle"}
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
