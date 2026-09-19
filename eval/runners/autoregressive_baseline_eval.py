"""eval/runners/autoregressive_baseline_eval.py — compare the trained Copula
Model against an autoregressive marginal-chain baseline
(src/autoregressive_baseline.py) that uses ONLY the same frozen/finetuned
TabICL marginal, no copula head at all: for a query set of size N, it chains
N single-point forward passes, growing the context by one row after each
query point is processed (Bruinsma et al., ICLR 2023's "Autoregressive
Conditional Neural Process" construction).

Both methods condition on the same synthetic GP episodes eval_checkpoint.py
already generates (--live_generate) and are scored in the same raw-nats,
per-point convention eval_checkpoint.py's _print_total_nll_table uses
(src/loss.py::y_space_nll's "total" is already averaged over each episode's
own N), so the two printed numbers are directly comparable.

NLL uses TEACHER FORCING (the true past target is appended to the context at
each autoregressive step) — the exact chain-rule decomposition of the true
joint density. Sampling (for the comparison plot) uses ANCESTRAL SAMPLING
(the drawn value is appended instead) — a different quantity by design, a
draw from the model's implied joint rather than a density evaluation. Query
order is the episode's existing x_norm_test order, not randomized/averaged
over permutations.

Usage:
    python eval/runners/autoregressive_baseline_eval.py \
        --ckpt checkpoints/copula_nano/copula-finetune-marginal-float32/step_0630000.pt \
        --tabicl_marginal_ckpt era5-33y \
        --n_episodes 20 --n_plot_episodes 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autoregressive_baseline import (  # noqa: E402
    _marginal_forward_dist,
    run_autoregressive_chain_nll,
    run_autoregressive_chain_sample,
)
from inference.copula_inference import load_copula_model  # noqa: E402
from pit import (  # noqa: E402
    DEFAULT_K_FOLDS,
    configure_tabicl_inference_amp,
    load_tabicl,
    normalize_targets,
)

from eval.configs.checkpoints import resolve_marginal_checkpoint  # noqa: E402
from eval.data.era5_io import safe_cholesky  # noqa: E402
from eval.runners.eval_checkpoint import (  # noqa: E402
    _eval_icl_episode,
    _live_generate_alternating,
    _load_full_config,
    _marginal_pit,
    _set_seed,
)
from eval.viz.sample_comparison_plots import plot_sample_comparison  # noqa: E402

_DEFAULT_CKPT = os.path.join(
    _REPO_ROOT, "checkpoints", "copula_nano", "copula-finetune-marginal-float32",
    "step_0630000.pt",
)


def _one_sample_pair(
    tabicl_marginal, icl_model, R_icl, X_train, y_train, X_test, y_train_scaled, mean, std, device,
):
    """One joint sample from each method, unscaled back to raw y-units.

    Copula Model: one batched marginal forward pass over all of X_test
    (fixed, shared context — the whole point of an explicit copula is not
    needing to grow the context), combined with the copula's own Sigma via
    Cholesky + a single shared white-noise vector (same pattern as
    src/train.py::_era5_viz_field, generalized off its ERA5-specific bits).

    Autoregressive marginal-chain: run_autoregressive_chain_sample's
    sequential ancestral-sampling loop.
    """
    N = X_test.shape[0]
    Sigma_np = R_icl.detach().to(torch.float64).cpu().numpy()
    L = torch.from_numpy(safe_cholesky(Sigma_np)).to(device=device, dtype=X_test.dtype)
    z_shared = torch.randn(N, device=device, dtype=X_test.dtype)
    z_copula = L @ z_shared
    u_copula = torch.clamp(0.5 * (1.0 + torch.erf(z_copula / (2.0 ** 0.5))), 1e-6, 1.0 - 1e-6)

    marginal_dist = _marginal_forward_dist(tabicl_marginal, X_train, y_train_scaled.unsqueeze(-1), X_test)
    # icdf treats a plain (n,) alpha as n shared quantile levels broadcast
    # across the whole batch (-> a (batch, n) grid), not one alpha per
    # distribution -- an explicit trailing size-1 axis (src/train.py's
    # _era5_viz_field:855-857) selects one quantile per distribution instead.
    y_copula_scaled = marginal_dist.icdf(u_copula.unsqueeze(-1)).squeeze(-1)
    y_copula_sample = (mean + std * y_copula_scaled).detach().cpu().numpy()

    ar_sample_scaled = run_autoregressive_chain_sample(
        tabicl_marginal, X_train, y_train_scaled.unsqueeze(-1), X_test,
    )
    ar_sample = (mean + std * ar_sample_scaled).detach().cpu().numpy()
    return y_copula_sample, ar_sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the Copula Model against an autoregressive marginal-chain "
                     "baseline that uses only the same frozen marginal, no copula head."
    )
    parser.add_argument("--config", default="conf/config.yaml",
                         help="Hydra config defining the eval-episode-generating distribution "
                              "(cfg.data), same convention as eval_checkpoint.py's --config.")
    parser.add_argument("--ckpt", default=_DEFAULT_CKPT, help="Copula Model checkpoint.")
    parser.add_argument("--tabicl_marginal_ckpt", default="era5-33y",
                         help="Marginal checkpoint shared by BOTH methods -- a "
                              "MARGINAL_FAMILIES name (eval/configs/checkpoints.py) or a raw "
                              "path. Default 'era5-33y': TabICL v2 fine-tuned on ERA5.")
    parser.add_argument("--n_episodes", type=int, default=20,
                         help="Fewer than eval_checkpoint.py's default (30) -- the "
                              "autoregressive chain does N forward passes per episode "
                              "instead of 1, so wall-clock is much higher.")
    parser.add_argument("--n_plot_episodes", type=int, default=3,
                         help="How many of the evaluated episodes additionally get a "
                              "sample-comparison PNG (see --out_dir).")
    parser.add_argument("--tabicl_pit_k_folds", type=int, default=DEFAULT_K_FOLDS)
    parser.add_argument("--tabicl_amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out_dir", default=os.path.join(_REPO_ROOT, "eval", "results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    _set_seed(args.seed)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    print(f"Device: {device}")

    cfg = _load_full_config(args.config)

    print(f"\nLoading Copula Model checkpoint: {args.ckpt}")
    icl_model, icl_cfg = load_copula_model(args.ckpt, config_path=args.config, device=str(device))
    print(f"Copula Model parameters: {sum(p.numel() for p in icl_model.parameters()):,}  "
          f"rank={int(icl_cfg.model.rank)}")

    marginal_ckpt_path = resolve_marginal_checkpoint(args.tabicl_marginal_ckpt)
    print(f"\nLoading shared TabICL marginal: {marginal_ckpt_path}")
    tabicl_marginal = load_tabicl(marginal_ckpt_path, str(device))
    configure_tabicl_inference_amp(args.tabicl_amp)
    print(f"Frozen TabICL marginal inference AMP={'on' if args.tabicl_amp else 'off (float32)'}")

    print(f"\nLive-generating {args.n_episodes} episodes via generate_gp_batch, seed={args.seed}")
    episodes = _live_generate_alternating(cfg, args.n_episodes, device, args.seed)

    copula_totals, ar_totals = [], []
    t0 = time.time()
    for i, ep in enumerate(episodes):
        marginal_pit = _marginal_pit(ep, tabicl_marginal, args.tabicl_pit_k_folds, device)
        if marginal_pit is None:
            print(f"  episode {i}: fewer than 2 train points, skipping (both methods need a real context)")
            continue

        _, R_dict, _, _, icl_y_parts = _eval_icl_episode(ep, icl_model, device, marginal_pit)
        copula_total = icl_y_parts["total"]

        X_train = ep["x_norm_train"].to(device)
        y_train = ep["y_train"].to(device)
        X_test = ep["x_norm_test"].to(device)
        y_test = ep["y_test"].to(device)
        N = X_test.shape[0]
        y_train_scaled, y_test_scaled, mean, std = normalize_targets(y_train, y_test)

        log_probs = run_autoregressive_chain_nll(
            tabicl_marginal,
            X_train, y_train_scaled.unsqueeze(-1),
            X_test, y_test_scaled.unsqueeze(-1),
        )
        ar_total = float((-log_probs.sum() + N * std.log()).item()) / N

        copula_totals.append(copula_total)
        ar_totals.append(ar_total)
        print(f"  episode {i}: Copula Model total={copula_total:.4f} nats/pt   "
              f"AR-chain total={ar_total:.4f} nats/pt   (N={N})")

        if i < args.n_plot_episodes:
            y_copula_sample, ar_sample = _one_sample_pair(
                tabicl_marginal, icl_model, R_dict["icl"],
                X_train, y_train, X_test, y_train_scaled, mean, std, device,
            )
            out_path = os.path.join(args.out_dir, f"sample_comparison_ep{i}.png")
            plot_sample_comparison(
                x_train=X_train.detach().cpu().numpy(),
                y_train=y_train.detach().cpu().numpy(),
                x_test=X_test.detach().cpu().numpy(),
                y_test_true=y_test.detach().cpu().numpy(),
                y_copula_sample=y_copula_sample,
                y_ar_chain_sample=ar_sample,
                output_path=out_path,
                title=f"Episode {i}",
            )
            print(f"    wrote {out_path}")

    elapsed = time.time() - t0
    print(f"\n{'─' * 70}")
    print(f"Total NLL (Y-space, marginal+copula, shared marginal) — lower is better  "
          f"[N={len(copula_totals)} episodes, {elapsed:.1f}s]")
    print(f"{'─' * 70}")
    print(f"{'Method':<45}{'Mean nats/pt':>14}{'Std':>10}")
    print(f"{'Copula Model':<45}{np.nanmean(copula_totals):>14.4f}{np.nanstd(copula_totals):>10.4f}")
    print(f"{'Autoregressive marginal-chain':<45}{np.nanmean(ar_totals):>14.4f}{np.nanstd(ar_totals):>10.4f}")
    print(f"{'─' * 70}\n")


if __name__ == "__main__":
    main()
