"""Compare the copula model with the autoregressive marginal chain (eval/baselines/autoregressive.py) on live GP episodes.

Both are scored in per-point raw nats; the chain uses teacher forcing in the
episode's own test order. Sample plots use ancestral sampling.

Usage:
    python -m eval.runners.autoregressive_baseline_eval ckpt=<copula checkpoint> \
        tabicl_marginal_ckpt=era5-33y n_episodes=20 n_plot_episodes=3
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from copula_inter.model import CopulaTabICL
    from copula_inter.pit import TabICLLike
    from copula_inter.type_aliases import Device

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

from copula_inter.pit import (  # noqa: E402
    DEFAULT_K_FOLDS,
    configure_tabicl_inference_amp,
    load_tabicl,
    normalize_targets,
    tabicl_forward,
)
from eval.baselines.autoregressive import autoregressive_log_pdf  # noqa: E402
from eval.configs.checkpoints import resolve_marginal_checkpoint  # noqa: E402
from eval.data.era5_io import safe_cholesky  # noqa: E402
from eval.runners.episode_scoring import _eval_icl_episode, _marginal_pit  # noqa: E402
from eval.runners.eval_checkpoint import _set_seed  # noqa: E402
from eval.runners.eval_inputs import _live_generate_alternating, _load_full_config  # noqa: E402
from eval.runners.hydra_cli import hydra_entry  # noqa: E402
from eval.viz.sample_comparison_plots import plot_sample_comparison  # noqa: E402
from inference.copula_inference import load_copula_model  # noqa: E402

_DEFAULT_CKPT = os.path.join(
    _REPO_ROOT,
    "checkpoints",
    "copula_nano",
    "copula-finetune-marginal-float32",
    "step_0630000.pt",
)


def _one_sample_pair(
    tabicl_marginal: TabICLLike,
    icl_model: CopulaTabICL,
    R_icl: torch.Tensor,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_train_scaled: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: Device,
) -> tuple[np.ndarray, np.ndarray]:
    """One joint sample per method in raw y units: the copula model (one marginal pass plus chol(Sigma) noise) and the chain (ancestral sampling)."""
    N = X_test.shape[0]
    Sigma_np = R_icl.detach().to(torch.float64).cpu().numpy()
    L = torch.from_numpy(safe_cholesky(Sigma_np)).to(device=device, dtype=X_test.dtype)
    z_shared = torch.randn(N, device=device, dtype=X_test.dtype)
    z_copula = L @ z_shared
    u_copula = torch.clamp(0.5 * (1.0 + torch.erf(z_copula / (2.0**0.5))), 1e-6, 1.0 - 1e-6)

    logits = tabicl_forward(
        tabicl_marginal,
        torch.cat([X_train, X_test]).unsqueeze(0),
        y_train_scaled.unsqueeze(0),
    ).to(device)
    marginal_dist = tabicl_marginal.quantile_dist(logits[0])
    # Trailing size-1 axis: one quantile level per distribution.
    y_copula_scaled = marginal_dist.icdf(u_copula.unsqueeze(-1)).squeeze(-1)
    y_copula_sample = (mean + std * y_copula_scaled).detach().cpu().numpy()

    ar = autoregressive_log_pdf(
        tabicl_marginal,
        X_train[None],
        y_train[None],
        X_test[None],
        torch.zeros_like(X_test[None, :, 0]),
        order="natural",
        conditioning="sample",
    )
    ar_sample = ar["appended"][0].detach().cpu().numpy()
    return y_copula_sample, ar_sample


@dataclass
class AutoregressiveBaselineSpec:
    """Compare the copula model against the autoregressive marginal chain on the same frozen marginal."""

    # Hydra config defining the episode distribution (same convention as eval_checkpoint's config).
    config: str = "conf/config.yaml"
    # Copula model checkpoint.
    ckpt: str = _DEFAULT_CKPT
    # Marginal shared by both methods: a MARGINAL_FAMILIES name or a path (era5-33y: TabICL v2 fine-tuned on ERA5).
    tabicl_marginal_ckpt: str = "era5-33y"
    # The chain does N forward passes per episode, so fewer than eval_checkpoint's 30.
    n_episodes: int = 20
    # Evaluated episodes that also get a sample-comparison PNG in out_dir.
    n_plot_episodes: int = 3
    tabicl_pit_k_folds: int = DEFAULT_K_FOLDS
    tabicl_amp: bool = True
    out_dir: str = os.path.join(_REPO_ROOT, "eval", "results")
    seed: int = 42
    device: str = "auto"


def run(args: AutoregressiveBaselineSpec) -> None:
    _set_seed(args.seed)
    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    print(f"Device: {device}")

    cfg = _load_full_config(args.config)

    print(f"\nLoading Copula Model checkpoint: {args.ckpt}")
    icl_model, icl_cfg = load_copula_model(args.ckpt, config_path=args.config, device=str(device))
    print(
        f"Copula Model parameters: {sum(p.numel() for p in icl_model.parameters()):,}  rank={int(icl_cfg.model.rank)}"
    )

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

        log_pdf = autoregressive_log_pdf(
            tabicl_marginal,
            X_train[None],
            y_train[None],
            X_test[None],
            y_test[None],
            order="natural",
        )["log_pdf"]
        ar_total = float(-log_pdf.sum().item()) / N

        copula_totals.append(copula_total)
        ar_totals.append(ar_total)
        print(
            f"  episode {i}: Copula Model total={copula_total:.4f} nats/pt   "
            f"AR-chain total={ar_total:.4f} nats/pt   (N={N})"
        )

        if i < args.n_plot_episodes:
            y_copula_sample, ar_sample = _one_sample_pair(
                tabicl_marginal,
                icl_model,
                R_dict["icl"],
                X_train,
                y_train,
                X_test,
                y_train_scaled,
                mean,
                std,
                device,
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
    print(
        f"Total NLL (Y-space, marginal+copula, shared marginal) — lower is better  "
        f"[N={len(copula_totals)} episodes, {elapsed:.1f}s]"
    )
    print(f"{'─' * 70}")
    print(f"{'Method':<45}{'Mean nats/pt':>14}{'Std':>10}")
    print(f"{'Copula Model':<45}{np.nanmean(copula_totals):>14.4f}{np.nanstd(copula_totals):>10.4f}")
    print(f"{'Autoregressive marginal-chain':<45}{np.nanmean(ar_totals):>14.4f}{np.nanstd(ar_totals):>10.4f}")
    print(f"{'─' * 70}\n")


main = hydra_entry("autoregressive_baseline_eval", AutoregressiveBaselineSpec, run)

if __name__ == "__main__":
    main()
