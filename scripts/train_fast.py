#!/usr/bin/env python3
"""Fast-start debug trainer for the copula model.

Builds the same Hydra config, model, optimizer, schedule, AMP setup and
training step as copula_inter.train, but skips the startup probes, wandb and
persistent workers, and validates on the first DEBUG_VAL_N_BATCHES batches of
the live validation set, whose z comes from a TabICL PIT regardless of
data.z_train_source. Checkpoints use the same format.

Usage:
    python scripts/train_fast.py
    python scripts/train_fast.py training.resume_ckpt=<checkpoint.pt>
    python scripts/train_fast.py model=copula_nano training.steps=200
    python scripts/train_fast.py data.z_train_source=analytic
    python scripts/train_fast.py training.batch_size=8 data.N_max=64
    python scripts/train_fast.py data.z_train_tabicl_mix_enabled=true \
        data.z_train_tabicl_mix_floor_frac=0.5 data.z_train_tabicl_mix_max_frac=0.5
"""

from __future__ import annotations

import io
import os
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from copula_inter.pit import TabICLLike

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

# Line-buffered stdout even when piped.
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(line_buffering=True)


import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler

from copula_inter.backend_registry import TABICL_Z_TRAIN_SOURCES
from copula_inter.backend_registry import z_train_source as z_train_source_of
from copula_inter.checkpointing import load_checkpoint, save_checkpoint
from copula_inter.config_path import merge_configs
from copula_inter.data_gen import generate_gp_batch
from copula_inter.dataset import collate_fn
from copula_inter.gp_kernels import _COMPOSABLE_KERNELS
from copula_inter.model import build_copula_transformer
from copula_inter.muon import Muon
from copula_inter.pit import gp_analytical_posterior, load_tabicl, resolve_pit_ckpt
from copula_inter.probe_batches import _sigma_stats
from copula_inter.rng import resolve_device
from copula_inter.training_core import _forward_and_loss, _run_train_step, cosine_lr_lambda

# Debug logging and validation cadence.
DEBUG_LOG_EVERY = 1
DEBUG_VAL_EVERY = 20
# Number of validation batches reproduced.
DEBUG_VAL_N_BATCHES = 2


def _build_debug_val_batch(
    cfg: DictConfig,
    t: DictConfig,
    device: str,
    gen_device: str,
    tabicl_model: TabICLLike | None,
    tabicl_k_folds: int,
    tabicl_split_calib_frac: float,
) -> tuple[int, int, int, list[dict], float]:
    """The first DEBUG_VAL_N_BATCHES batches of the live validation set (same seeds), PIT'd with tabicl_model.

    Returns (n_episodes, val_seed, batch_size, batches, oracle_copula_nll), the
    last being the mean per-point analytic posterior copula NLL
    (gp_analytical_posterior), the fixed operand of the copula gap.
    """
    # Keep batches separately collated (d_features may differ between calls).
    val_seed = int(t.get("live_val_seed", 20260723))
    batch_size = int(t.batch_size)
    batches = []
    n_episodes = 0
    oracle_copula_per_point: list[float] = []
    for i in range(DEBUG_VAL_N_BATCHES):
        val_cfg = merge_configs(cfg, OmegaConf.create({"seed": val_seed + i * 104_729}))
        episodes = generate_gp_batch(
            val_cfg,
            batch_size,
            device=gen_device,
            tabicl_model=tabicl_model,
            tabicl_k_folds=tabicl_k_folds,
            tabicl_split_calib_frac=tabicl_split_calib_frac,
            return_kernel_metadata=True,
        )
        n_episodes += len(episodes)
        for ep in episodes:
            try:
                post = gp_analytical_posterior(ep)
            except (NotImplementedError, KeyError):
                # Skip the oracle for kernels that cannot be rebuilt.
                continue
            n_test_ep = int(ep["x_norm_test"].shape[0])
            oracle_copula_per_point.append(float(post["nll_post_copula"]) / n_test_ep)
        batch = {k: v.to(device, non_blocking=True) for k, v in collate_fn(episodes).items()}
        batches.append(batch)
    oracle_copula_nll = (
        sum(oracle_copula_per_point) / len(oracle_copula_per_point) if oracle_copula_per_point else float("nan")
    )
    return n_episodes, val_seed, batch_size, batches, oracle_copula_nll


def _build_episode_batch(
    cfg: DictConfig,
    n: int,
    seed: int,
    device: str,
    tabicl_model: TabICLLike | None,
    tabicl_k_folds: int,
    tabicl_split_calib_frac: float,
    gen_device: str,
    return_kernel_metadata: bool = False,
    tabicl_mix_weights: torch.Tensor | None = None,
) -> tuple[list[dict[str, torch.Tensor]], dict]:
    call_cfg = merge_configs(cfg, OmegaConf.create({"seed": seed}))
    episodes = generate_gp_batch(
        call_cfg,
        n,
        device=gen_device,
        tabicl_model=tabicl_model,
        tabicl_k_folds=tabicl_k_folds,
        tabicl_split_calib_frac=tabicl_split_calib_frac,
        tabicl_mix_weights=tabicl_mix_weights,
        return_kernel_metadata=return_kernel_metadata,
    )
    batch = {k: v.to(device, non_blocking=True) for k, v in collate_fn(episodes).items()}
    return episodes, batch


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    t_script0 = time.perf_counter()
    torch.manual_seed(cfg.seed)
    t = cfg.training
    device = resolve_device(t.device)

    # Cap training.steps unless it was overridden.
    if int(t.steps) >= 100_000:
        print(
            f"[train_fast] training.steps={int(t.steps)} looks like the production default -- capping to 60 for this debug run (pass training.steps=N to override)."
        )
        t.steps = 60

    z_train_source = z_train_source_of(cfg)
    tabicl_model = None
    gen_device = "cpu"
    tabicl_k_folds = int(cfg.data.get("z_train_tabicl_k_folds", 10))
    tabicl_split_calib_frac = (
        float(cfg.data.get("z_train_split_calib_frac", 1.0)) if z_train_source == "tabicl_split" else 0.0
    )

    # Fixed-fraction TabICL z_train mixing only (floor_frac == max_frac).
    mix_enabled = bool(cfg.data.get("z_train_tabicl_mix_enabled", False))
    tabicl_mix_weights = None
    if mix_enabled:
        floor_frac = float(cfg.data.get("z_train_tabicl_mix_floor_frac", 0.05))
        max_frac = float(cfg.data.get("z_train_tabicl_mix_max_frac", 0.35))
        if floor_frac != max_frac:
            raise ValueError(
                f"data.z_train_tabicl_mix_enabled=true with floor_frac={floor_frac} != "
                f"max_frac={max_frac} needs the adaptive per-kernel-family gap "
                "measurement train_fast.py deliberately skips -- set both to the same "
                "fixed mixing fraction (e.g. 0.5 for a 50/50 alternation), or run "
                "src/copula_inter/train.py directly for the adaptive version."
            )
        tabicl_mix_weights = torch.full((len(_COMPOSABLE_KERNELS),), floor_frac, dtype=torch.float32)

    if z_train_source in TABICL_Z_TRAIN_SOURCES or mix_enabled:
        ckpt = resolve_pit_ckpt(cfg)
        if ckpt is None:
            raise ValueError(
                f"data.z_train_source={z_train_source} (or data.z_train_tabicl_mix_enabled=true) "
                "requires a resolvable TabICL checkpoint (tabicl.ckpt with tabicl.pretrained=true, "
                "or tabicl.pit_ckpt) -- or pass data.z_train_source=analytic and "
                "data.z_train_tabicl_mix_enabled=false to skip PIT entirely."
            )
        print(f"[train_fast] Loading frozen TabICL marginal for PIT: {ckpt}")
        t_pit0 = time.perf_counter()
        tabicl_model = load_tabicl(ckpt, device)
        gen_device = device
        print(f"[train_fast] TabICL marginal loaded in {time.perf_counter() - t_pit0:.1f}s")

    mix_desc = f" mix_frac={float(tabicl_mix_weights[0]):.2f}" if tabicl_mix_weights is not None else ""
    print(
        f"[train_fast] model={cfg.model.get('rank')}rank/"
        f"{cfg.model.get('correlation_parametrization', 'covnorm')} "
        f"data=P[{cfg.data.P_min}..{cfg.data.P_max}] N[{cfg.data.N_min}..{cfg.data.N_max}] "
        f"z_train_source={z_train_source}{mix_desc} device={device}"
    )

    # Validation z always comes from a TabICL PIT (reuses the training marginal if loaded).
    if tabicl_model is not None:
        val_tabicl_model, val_gen_device = tabicl_model, gen_device
    else:
        ckpt = resolve_pit_ckpt(cfg)
        if ckpt is None:
            raise ValueError(
                "train_fast.py's debug val set always scores z_test through real TabICL "
                "PIT, which requires a resolvable TabICL checkpoint (tabicl.ckpt with "
                "tabicl.pretrained=true, or tabicl.pit_ckpt)."
            )
        print(f"[train_fast] Loading frozen TabICL marginal for val z_test: {ckpt}")
        t_pit0 = time.perf_counter()
        val_tabicl_model = load_tabicl(ckpt, device)
        val_gen_device = device
        print(f"[train_fast] TabICL marginal loaded in {time.perf_counter() - t_pit0:.1f}s")

    t_val0 = time.perf_counter()
    n_val_debug, val_seed, val_batch_size, val_batches, oracle_copula_nll = _build_debug_val_batch(
        cfg,
        t,
        device,
        val_gen_device,
        val_tabicl_model,
        tabicl_k_folds,
        tabicl_split_calib_frac,
    )
    print(
        f"[train_fast] Built val set: first {n_val_debug} episodes "
        f"({DEBUG_VAL_N_BATCHES} batches of {val_batch_size}) of train.sh's own fixed "
        f"val set (live_val_seed={val_seed}) in {time.perf_counter() - t_val0:.1f}s -- "
        "byte-identical to (a prefix of) train.sh's val/y_nll_total ONLY if "
        "training.batch_size/training.live_val_seed/data.*/the resolved TabICL "
        "checkpoint match the run you're comparing against."
    )
    print(
        f"[train_fast] Oracle (exact GP posterior) copula NLL on this val set: {oracle_copula_nll:.4f} -- the copula_gap below is the model's copula NLL minus this."
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
                "params": muon_params,
                "use_muon": True,
                "lr": t.muon_lr,
                "weight_decay": t.muon_weight_decay,
                "momentum": t.muon_momentum,
                "matched_adamw_rms": t.muon_matched_adamw_rms,
                "ns_steps": t.muon_ns_steps,
                "nesterov": t.muon_nesterov,
                "adamw_betas": tuple(t.muon_adamw_betas),
                "adamw_eps": t.muon_adamw_eps,
            },
            {
                "params": adamw_params,
                "use_muon": False,
                "lr": t.muon_lr,
                "weight_decay": 0.0,
                "adamw_betas": tuple(t.muon_adamw_betas),
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
            print(
                f"[train_fast] Resumed weights+optimizer from {resume_ckpt} (step {ckpt_step}) -- resetting to step 0"
            )
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
    triu_cache: dict[int, torch.Tensor] = {}

    print(
        f"[train_fast] Ready to train after {time.perf_counter() - t_script0:.1f}s (steps={int(t.steps)}, batch_size={int(t.batch_size)})"
    )
    print("[train_fast] Training loop started (Ctrl-C to stop)\n")

    model.train()
    for step in range(start_step + 1, int(t.steps) + 1):
        step_t0 = time.perf_counter()
        _, batch = _build_episode_batch(
            cfg,
            int(t.batch_size),
            seed=int(cfg.seed) + step * 104_729,
            device=device,
            tabicl_model=tabicl_model,
            tabicl_k_folds=tabicl_k_folds,
            tabicl_split_calib_frac=tabicl_split_calib_frac,
            gen_device=gen_device,
            tabicl_mix_weights=tabicl_mix_weights,
        )
        data_ms = (time.perf_counter() - step_t0) * 1000.0

        optimizer.zero_grad(set_to_none=True)
        out, Sigma, parts, loss, aux_mae, grad_norm = _run_train_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            trainable=trainable,
            batch=batch,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            clip_grad_norm=float(t.clip_grad_norm),
            nll_weight=nll_weight,
            aux_mae_weight=aux_mae_weight,
            jitter=jitter,
            triu_cache=triu_cache,
            phase_start=lambda: None,
            phase_end=lambda name, s: None,
            parametrization=parametrization,
        )
        step_ms = (time.perf_counter() - step_t0) * 1000.0

        if step % DEBUG_LOG_EVERY == 0:
            stats = _sigma_stats(Sigma.dense().detach(), batch["test_mask"])
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
                        model=model,
                        batch=vb,
                        device=device,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        nll_weight=nll_weight,
                        aux_mae_weight=0.0,
                        jitter=jitter,
                        triu_cache=triu_cache,
                        parametrization=parametrization,
                    )
                    totals.append(val_parts["total"].item())
                    copulas.append(val_parts["copula"].item())
                    marginals.append(val_parts["marginal"].item())
            model.train()
            model_copula_nll = sum(copulas) / len(copulas)
            copula_gap = model_copula_nll - oracle_copula_nll
            print(
                f"          -- val({n_val_debug}) total={sum(totals) / len(totals):.4f} "
                f"copula={model_copula_nll:.4f} marginal={sum(marginals) / len(marginals):.4f} "
                f"| copula_gap={copula_gap:+.4f} (vs. oracle {oracle_copula_nll:.4f} -- lower is better, 0 = Bayes-optimal)"
            )

    if t.get("ckpt_dir", None):
        save_checkpoint(model, optimizer, scheduler, cfg, int(t.steps))
        print(f"[train_fast] Saved checkpoint to {t.ckpt_dir}")
    print("\n[train_fast] Done.")


if __name__ == "__main__":
    main()
