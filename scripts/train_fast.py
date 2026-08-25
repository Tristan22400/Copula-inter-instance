#!/usr/bin/env python3
"""
train_fast.py — instant-startup debug trainer for the Copula Transformer.

train.py's startup (baselines.enabled's 8 synthetic-kernel probes + a TabICL
K-fold PIT pass over every one of them, a real ERA5 fetch + classical-GP-MLE
baseline fit, a second frozen-TabICL "sim-to-real diagnostic" load, a
500-episode fixed validation set built through TabICL's own K-fold PIT, live
DataLoader worker spawn, wandb.init's network round-trip) takes ~7 minutes
before the first training step runs. That's dead time when what you actually
need is "is this model/config training at all, and does the loss move" —
e.g. debugging a run that looks stuck.

This script builds the exact same Hydra config train.py would (same
`model=`/`data=` groups and CLI overrides), the exact same model, optimizer
(Muon), LR schedule, AMP setup, and per-step forward/loss/backward/clip/step
logic — imported directly from src/train.py, not reimplemented, so a step
here behaves identically to a step in the real run. It only diverges from
train.py in what it skips: no baselines/era5 probes, no wandb, no persistent
DataLoader workers, and a small in-process-generated validation set instead
of the fixed 500-episode one. First step happens in seconds, most of that
being CUDA context init + (if data.z_train_source=tabicl, the default) one
frozen-TabICL load.

Not a replacement for train.py — no wandb logging, no baselines/era5
validation metrics. Checkpointing (training.ckpt_dir/training.resume_ckpt)
uses train.py's own save_checkpoint/load_checkpoint, so a checkpoint saved
here is a normal checkpoint train.sh can resume, and training.resume_ckpt
here can load the actual stuck run's checkpoint to debug from where it left
off. For a full production run, use train.sh.

Usage (same Hydra override syntax as train.py):
    python scripts/train_fast.py
    python scripts/train_fast.py training.resume_ckpt=./checkpoints/copula_transformer/step_0029999.pt
    python scripts/train_fast.py model=copula_nano training.steps=200
    python scripts/train_fast.py data.z_train_source=analytic   # skip TabICL PIT entirely -- fastest possible start
    python scripts/train_fast.py training.batch_size=8 data.N_max=64  # shrink episodes for an even faster loop
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

# Force line-buffered stdout even when piped/redirected (e.g. `| tee log.txt`)
# -- this script's entire point is watching output live to tell a genuinely
# stuck run apart from one that's just quiet because Python block-buffers
# non-tty stdout. Without this, output can sit in the buffer indefinitely.
sys.stdout.reconfigure(line_buffering=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler

from data_gen import generate_gp_batch
from dataset import collate_fn
from model import build_copula_transformer
from muon import Muon
from pit import load_tabicl, resolve_pit_ckpt
from train import (
    _forward_and_loss,
    _run_train_step,
    _sigma_stats,
    cosine_lr_lambda,
    load_checkpoint,
    save_checkpoint,
)

# Debug-loop cadence -- deliberately NOT tied to training.log_every/val_every
# (those default to 200/1000, tuned for multi-day production runs, not a
# "watch it start" debug session). Override by editing these constants
# directly if you want a different cadence.
DEBUG_LOG_EVERY = 1
DEBUG_VAL_EVERY = 20
# How many of live_dataset.py::build_fixed_live_val_batches' fixed val
# batches to reproduce here (see _build_debug_val_batch below) -- 2 keeps
# startup near-instant; raise for a lower-variance (but slower to build)
# val estimate.
DEBUG_VAL_N_BATCHES = 2


def _build_debug_val_batch(cfg: DictConfig, t: DictConfig, device: str, gen_device: str,
                            tabicl_model, tabicl_k_folds: int, tabicl_split_calib_frac: float):
    """The first `DEBUG_VAL_N_BATCHES` batches of train.py's own fixed
    live-generation validation set (see live_dataset.py::
    build_fixed_live_val_batches) -- NOT an independent sample.

    generate_gp_batch fully reseeds python/numpy/torch RNGs from cfg.seed on
    every call (see data_gen.py's module docstring), so batch i there is a
    deterministic function of (cfg, val_seed + i*104_729, batch_size,
    tabicl checkpoint weights) alone. Reusing that exact seed formula here
    with the same training.live_val_seed/training.batch_size/data config/
    tabicl checkpoint as the run being debugged reproduces those episodes
    byte-for-byte -- so this val loss is a genuine (if smaller/higher-
    variance) subsample of whatever train.sh logs as val/y_nll_total, not a
    different validation distribution. That equivalence breaks the moment
    training.batch_size, training.live_val_seed, the data.* config, or the
    resolved TabICL checkpoint differ from the run you're comparing against.
    """
    # Each batch is kept SEPARATELY collated, exactly like
    # build_fixed_live_val_batches's own `batches: List[dict]` -- d_features
    # is sampled once per generate_gp_batch call (data_gen.py::
    # _sample_d_features) and can differ across the two calls below, so
    # concatenating their episodes into one collate_fn call would crash the
    # same way a cross-shard variable-d training batch would.
    val_seed = int(t.get("live_val_seed", 20260723))
    batch_size = int(t.batch_size)
    batches = []
    n_episodes = 0
    for i in range(DEBUG_VAL_N_BATCHES):
        val_cfg = OmegaConf.merge(cfg, OmegaConf.create({"seed": val_seed + i * 104_729}))
        episodes = generate_gp_batch(
            val_cfg, batch_size, device=gen_device,
            tabicl_model=tabicl_model, tabicl_k_folds=tabicl_k_folds,
            tabicl_split_calib_frac=tabicl_split_calib_frac,
        )
        n_episodes += len(episodes)
        batch = {k: v.to(device, non_blocking=True) for k, v in collate_fn(episodes).items()}
        batches.append(batch)
    return n_episodes, val_seed, batch_size, batches


def _build_episode_batch(cfg: DictConfig, n: int, seed: int, device: str,
                          tabicl_model, tabicl_k_folds: int, tabicl_split_calib_frac: float,
                          gen_device: str, return_kernel_metadata: bool = False):
    call_cfg = OmegaConf.merge(cfg, OmegaConf.create({"seed": seed}))
    episodes = generate_gp_batch(
        call_cfg, n, device=gen_device,
        tabicl_model=tabicl_model, tabicl_k_folds=tabicl_k_folds,
        tabicl_split_calib_frac=tabicl_split_calib_frac,
        return_kernel_metadata=return_kernel_metadata,
    )
    batch = {k: v.to(device, non_blocking=True) for k, v in collate_fn(episodes).items()}
    return episodes, batch


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    t_script0 = time.perf_counter()
    torch.manual_seed(cfg.seed)
    t = cfg.training
    device = (
        "cuda" if t.device == "auto" and torch.cuda.is_available()
        else ("cpu" if t.device == "auto" else t.device)
    )

    # conf/config.yaml's training.steps default (1_000_000) is sized for a
    # real production run, not a "watch it start" debug session -- if the
    # caller didn't lower it explicitly, cap it here instead of silently
    # looping for days. Explicit training.steps=N overrides always win since
    # they replace this value before this check ever runs.
    if int(t.steps) >= 100_000:
        print(f"[train_fast] training.steps={int(t.steps)} looks like the production default -- capping to 60 for this debug run (pass training.steps=N to override).")
        t.steps = 60

    z_train_source = str(cfg.data.get("z_train_source", "analytic"))
    tabicl_model = None
    gen_device = "cpu"
    tabicl_k_folds = int(cfg.data.get("z_train_tabicl_k_folds", 10))
    tabicl_split_calib_frac = (
        float(cfg.data.get("z_train_split_calib_frac", 1.0))
        if z_train_source == "tabicl_split" else 0.0
    )
    if z_train_source in ("tabicl", "tabicl_split"):
        ckpt = resolve_pit_ckpt(cfg)
        if ckpt is None:
            raise ValueError(
                f"data.z_train_source={z_train_source} requires a resolvable TabICL "
                "checkpoint (tabicl.ckpt with tabicl.pretrained=true, or tabicl.pit_ckpt) "
                "-- or pass data.z_train_source=analytic to skip PIT entirely."
            )
        print(f"[train_fast] Loading frozen TabICL marginal for PIT: {ckpt}")
        t_pit0 = time.perf_counter()
        tabicl_model = load_tabicl(ckpt, device)
        gen_device = device
        print(f"[train_fast] TabICL marginal loaded in {time.perf_counter() - t_pit0:.1f}s")

    print(
        f"[train_fast] model={cfg.model.get('rank')}rank/"
        f"{cfg.model.get('correlation_parametrization', 'covnorm')} "
        f"data=P[{cfg.data.P_min}..{cfg.data.P_max}] N[{cfg.data.N_min}..{cfg.data.N_max}] "
        f"z_train_source={z_train_source} device={device}"
    )

    t_val0 = time.perf_counter()
    n_val_debug, val_seed, val_batch_size, val_batches = _build_debug_val_batch(
        cfg, t, device, gen_device, tabicl_model, tabicl_k_folds, tabicl_split_calib_frac,
    )
    print(
        f"[train_fast] Built val set: first {n_val_debug} episodes "
        f"({DEBUG_VAL_N_BATCHES} batches of {val_batch_size}) of train.sh's own fixed "
        f"val set (live_val_seed={val_seed}) in {time.perf_counter() - t_val0:.1f}s -- "
        "byte-identical to (a prefix of) train.sh's val/y_nll_total ONLY if "
        "training.batch_size/training.live_val_seed/data.*/the resolved TabICL "
        "checkpoint match the run you're comparing against."
    )

    t_model0 = time.perf_counter()
    model = build_copula_transformer(cfg).to(device)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_fast] Model built in {time.perf_counter() - t_model0:.1f}s ({n_train_params:,} trainable params)")

    trainable = [p for p in model.parameters() if p.requires_grad]
    muon_params = [p for p in trainable if p.ndim >= 2]
    adamw_params = [p for p in trainable if p.ndim < 2]
    optimizer = Muon(
        [
            {
                "params": muon_params, "use_muon": True, "lr": t.muon_lr,
                "weight_decay": t.muon_weight_decay, "momentum": t.muon_momentum,
                "matched_adamw_rms": t.muon_matched_adamw_rms, "ns_steps": t.muon_ns_steps,
                "nesterov": t.muon_nesterov, "adamw_betas": tuple(t.muon_adamw_betas),
                "adamw_eps": t.muon_adamw_eps,
            },
            {
                "params": adamw_params, "use_muon": False, "lr": t.muon_lr,
                "weight_decay": 0.0, "adamw_betas": tuple(t.muon_adamw_betas),
                "adamw_eps": t.muon_adamw_eps,
            },
        ]
    )
    lr_min_frac = t.muon_lr_min / t.muon_lr

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = GradScaler(device=device) if (use_amp and amp_dtype == torch.float16) else None

    start_step = 0
    resume_ckpt = t.get("resume_ckpt", None)
    if resume_ckpt:
        ckpt_step = load_checkpoint(resume_ckpt, model, device, optimizer=optimizer, scaler=scaler)
        if bool(t.get("resume_reset_schedule", False)):
            print(f"[train_fast] Resumed weights+optimizer from {resume_ckpt} (step {ckpt_step}) -- resetting to step 0")
        else:
            start_step = ckpt_step
            print(f"[train_fast] Resumed weights+optimizer from {resume_ckpt} -- continuing from step {start_step}")
    if start_step > 0:
        for group in optimizer.param_groups:
            group["initial_lr"] = t.muon_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_lr_lambda(s, t.warmup_steps, t.steps, lr_min_frac),
        last_epoch=start_step - 1 if start_step > 0 else -1,
    )

    jitter = float(cfg.model.get("sigma_jitter", 1e-4))
    parametrization = str(cfg.model.get("correlation_parametrization", "covnorm"))
    nll_weight = float(t.get("nll_weight", 1.0))
    aux_mae_weight = float(t.get("aux_mae_weight", 0.0))
    triu_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    print(f"[train_fast] Ready to train after {time.perf_counter() - t_script0:.1f}s (steps={int(t.steps)}, batch_size={int(t.batch_size)})")
    print("[train_fast] Training loop started (Ctrl-C to stop)\n")

    model.train()
    for step in range(start_step + 1, int(t.steps) + 1):
        step_t0 = time.perf_counter()
        _, batch = _build_episode_batch(
            cfg, int(t.batch_size), seed=int(cfg.seed) + step * 104_729, device=device,
            tabicl_model=tabicl_model, tabicl_k_folds=tabicl_k_folds,
            tabicl_split_calib_frac=tabicl_split_calib_frac, gen_device=gen_device,
        )
        data_ms = (time.perf_counter() - step_t0) * 1000.0

        optimizer.zero_grad(set_to_none=True)
        out, Sigma, parts, loss, aux_mae, grad_norm = _run_train_step(
            model=model, optimizer=optimizer, scheduler=scheduler, trainable=trainable,
            batch=batch, device=device, use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler,
            clip_grad_norm=float(t.clip_grad_norm), nll_weight=nll_weight, aux_mae_weight=aux_mae_weight,
            jitter=jitter, triu_cache=triu_cache, phase_start=lambda: None, phase_end=lambda name, s: None,
            parametrization=parametrization,
        )
        step_ms = (time.perf_counter() - step_t0) * 1000.0

        if step % DEBUG_LOG_EVERY == 0:
            stats = _sigma_stats(Sigma.detach(), batch["test_mask"])
            gn = float(grad_norm) if grad_norm is not None else float("nan")
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[{step:6d}/{int(t.steps)}] loss={loss.item():.4f} "
                f"(cop={parts['copula'].item():.4f} mar={parts['marginal'].item():.4f} aux={aux_mae.item():.4f}) "
                f"offdiag_mean={stats['offdiag_mean']:+.4f} offdiag_std={stats['offdiag_std']:.4f} "
                f"grad_norm={gn:.3f} lr={lr_now:.2e} | data={data_ms:.0f}ms step={step_ms:.0f}ms"
            )

        if step % DEBUG_VAL_EVERY == 0 or step == int(t.steps):
            model.eval()
            totals, copulas, marginals = [], [], []
            with torch.no_grad():
                for vb in val_batches:
                    _, _, val_parts, _, _ = _forward_and_loss(
                        model=model, batch=vb, device=device, use_amp=use_amp, amp_dtype=amp_dtype,
                        nll_weight=nll_weight, aux_mae_weight=0.0, jitter=jitter, triu_cache=triu_cache,
                        parametrization=parametrization,
                    )
                    totals.append(val_parts["total"].item())
                    copulas.append(val_parts["copula"].item())
                    marginals.append(val_parts["marginal"].item())
            model.train()
            print(
                f"          -- val({n_val_debug}) total={sum(totals) / len(totals):.4f} "
                f"copula={sum(copulas) / len(copulas):.4f} marginal={sum(marginals) / len(marginals):.4f}"
            )

    if t.get("ckpt_dir", None):
        save_checkpoint(model, optimizer, scheduler, cfg, int(t.steps))
        print(f"[train_fast] Saved checkpoint to {t.ckpt_dir}")
    print("\n[train_fast] Done.")


if __name__ == "__main__":
    main()
