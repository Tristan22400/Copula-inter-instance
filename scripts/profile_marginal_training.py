#!/usr/bin/env python
"""Microbenchmark the Phase-A marginal training step on a real CUDA device."""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext

import torch
from hydra import compose, initialize_config_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "tabicl_upstream", "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from finetune_marginal import _generate_phase_a_gp_batch, _gp_cfg, _seed_everything
from marginal_finetune import MarginalLossWeights, apply_tier, phase_a_batch_loss
from pit import load_tabicl


def _sync() -> None:
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--p", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--folds-per-step", type=int, default=2)
    parser.add_argument("--amp", choices=("off", "bf16", "fp16"), default="off")
    parser.add_argument("--matmul-precision", choices=("highest", "high"), default="highest")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device = "cuda"
    torch.set_float32_matmul_precision(args.matmul_precision)
    _seed_everything(42)
    with initialize_config_dir(config_dir=os.path.join(ROOT, "conf"), version_base=None):
        cfg = compose(
            config_name="finetune_marginal",
            overrides=[
                f"training.batch_size={args.batch_size}",
                f"data.P_min={args.p}", f"data.P_max={args.p}",
                f"data.N_min={args.n}", f"data.N_max={args.n}",
                "wandb.mode=disabled", "marginal.era5.mix_frac=0",
            ],
        )

    t0 = time.perf_counter()
    tabicl, _ = load_tabicl(str(cfg.marginal.ckpt), device, trainable=True, return_config=True)
    report = apply_tier(tabicl, int(cfg.marginal.tier))
    params = [p for p in tabicl.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg.training.lr))
    _sync()
    print(f"gpu={torch.cuda.get_device_name()} model_load_s={time.perf_counter() - t0:.3f} "
          f"trainable={report['n_trainable_params']}", flush=True)

    weights = MarginalLossWeights(
        distill=float(cfg.marginal.loss.distill), nll=float(cfg.marginal.loss.nll),
        crps=float(cfg.marginal.loss.crps), tail_power=float(cfg.marginal.loss.tail_power),
    )
    gp_cfg = _gp_cfg(cfg)
    gen = torch.Generator().manual_seed(55)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    totals = {name: 0.0 for name in (
        "generation", "collate", "tabicl_forward", "analytic_targets",
        "objective", "backward", "optimizer", "step",
    )}
    measured = 0
    torch.cuda.reset_peak_memory_stats()

    for step in range(args.warmup + args.steps):
        measure = step >= args.warmup
        _sync()
        step_start = time.perf_counter()
        gp_cfg.seed = 42000000 + step
        t = time.perf_counter()
        episodes = _generate_phase_a_gp_batch(gp_cfg, args.batch_size, device)
        _sync()
        generation = time.perf_counter() - t

        parts: dict[str, float] | None = {} if measure else None
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None else nullcontext()
        )
        with amp_ctx:
            res = phase_a_batch_loss(
                tabicl, episodes, weights, k_folds=int(cfg.marginal.k_folds),
                folds_per_step=args.folds_per_step, generator=gen, device=device,
                eps=float(cfg.marginal.pit_eps), timings=parts,
            )
        if not torch.isfinite(res["loss"]):
            raise RuntimeError(f"non-finite loss under amp={args.amp}")
        t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        res["loss"].backward()
        _sync()
        backward = time.perf_counter() - t
        t = time.perf_counter()
        opt.step()
        _sync()
        optimizer = time.perf_counter() - t
        elapsed = time.perf_counter() - step_start

        if measure:
            measured += 1
            totals["generation"] += generation
            for key, value in (parts or {}).items():
                totals[key] += value
            totals["backward"] += backward
            totals["optimizer"] += optimizer
            totals["step"] += elapsed
            detail = " ".join(f"{key}={value:.3f}" for key, value in (parts or {}).items())
            print(f"step={measured} generation={generation:.3f} {detail} "
                  f"backward={backward:.3f} optimizer={optimizer:.3f} total={elapsed:.3f}",
                  flush=True)

    print("mean_seconds " + " ".join(
        f"{key}={value / measured:.4f}" for key, value in totals.items()
    ), flush=True)
    print(f"peak_cuda_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}", flush=True)


if __name__ == "__main__":
    main()
