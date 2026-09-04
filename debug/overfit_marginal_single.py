#!/usr/bin/env python3
"""Full-path single-episode overfit check for marginal fine-tuning.

This deliberately uses analytic quantile distillation: unlike sample pinball,
it has a well-defined distributional target even when the same GP realization
is repeated. A healthy run must drive the loss close to zero and the model NLL
close to the analytic marginal NLL. Production training uses pinball on fresh
episodes; this script answers the narrower question "can gradients traverse
the real fold/normalization/TabICL/loss path and memorize one task?"

Usage:
    python debug/overfit_marginal_single.py
    python debug/overfit_marginal_single.py --steps 100 --tier 0 --lr 1e-4
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from hydra import compose, initialize_config_dir

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "tabicl_upstream", "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from finetune_marginal import _generate_phase_a_gp_batch, _gp_cfg  # noqa: E402
from marginal_finetune import (  # noqa: E402
    MarginalLossWeights,
    apply_tier,
    phase_a_batch_loss,
)
from pit import load_tabicl  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tier", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-final-ratio", type=float, default=0.02)
    return parser.parse_args()


def _metrics(model, episode, weights, device: str) -> dict:
    with torch.no_grad():
        result = phase_a_batch_loss(
            model, [episode], weights, k_folds=2, folds_per_step=None,
            device=device,
        )
    return {
        "distill": result["distill"].item(),
        "nll": result["nll"].item(),
        "oracle_nll": result["oracle_nll"],
        "nll_gap": result["nll_gap_to_oracle"],
        "crossing": result["raw_crossing_frac"],
    }


def main() -> None:
    args = _args()
    torch.manual_seed(args.seed)

    overrides = [
        "data.P_min=12", "data.P_max=12", "data.N_min=8", "data.N_max=8",
        "data.d_features=2", "data.systematic_composition=false", "data.kernel=rbf",
        "data.structural_warp_enabled=false", "data.mlp_mixing_enabled=false",
        "data.mean_fn_enabled=false", "marginal.era5.mix_frac=0",
    ]
    with initialize_config_dir(config_dir=os.path.join(_ROOT, "conf"), version_base=None):
        cfg = compose(config_name="finetune_marginal", overrides=overrides)

    gp_cfg = _gp_cfg(cfg)
    gp_cfg.seed = args.seed
    episode = _generate_phase_a_gp_batch(gp_cfg, 1, args.device)[0]
    # The episode is fixed supervision. This is defensive for hand-built
    # episodes too, whose temporary gpytorch kernel can otherwise retain a graph.
    episode = {
        key: value.detach() if torch.is_tensor(value) else value
        for key, value in episode.items()
    }

    model, _ = load_tabicl(
        str(cfg.marginal.ckpt), args.device, trainable=True, return_config=True
    )
    report = apply_tier(
        model, args.tier, lora_rank=int(cfg.marginal.lora_rank),
        lora_alpha=float(cfg.marginal.lora_alpha),
        lora_target=str(cfg.marginal.lora_target),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    weights = MarginalLossWeights(
        distill=1.0, nll=0.0, crps=0.0, pinball=0.0,
        tail_power=float(cfg.marginal.loss.tail_power),
    )

    initial = _metrics(model, episode, weights, args.device)
    print(f"trainable={report['n_trainable_params']:,} initial={initial}")
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        result = phase_a_batch_loss(
            model, [episode], weights, k_folds=2, folds_per_step=None,
            device=args.device,
        )
        result["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            current = _metrics(model, episode, weights, args.device)
            print(f"step={step:4d} grad={grad_norm.item():.5f} metrics={current}")

    final = _metrics(model, episode, weights, args.device)
    ratio = final["distill"] / max(initial["distill"], 1e-12)
    # On one finite realization a misspecified model can have lower empirical
    # NLL than the data-generating distribution by chance. The valid check is
    # movement toward the oracle NLL, not necessarily downward movement.
    oracle_gap_improved = abs(final["nll_gap"]) < abs(initial["nll_gap"])
    if ratio > args.max_final_ratio or not oracle_gap_improved:
        raise SystemExit(
            "FAIL: full-path single-episode overfit did not converge: "
            f"distill ratio={ratio:.6f}, |oracle gap| "
            f"{abs(initial['nll_gap']):.6f} -> {abs(final['nll_gap']):.6f}"
        )
    print(
        "PASS: full-path overfit converged; "
        f"distill ratio={ratio:.6f}, |oracle gap| "
        f"{abs(initial['nll_gap']):.6f} -> {abs(final['nll_gap']):.6f}."
    )


if __name__ == "__main__":
    main()
