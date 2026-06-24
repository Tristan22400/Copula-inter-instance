"""
train.py — Train the Copula Transformer in Y-space NLL via Sklar's theorem.

Loss:  L = Copula_NLL(z_test; Σ̂) + Marginal_NLL(y_test; TabICL log-pdf)
Σ̂ is built by ``low_rank_correlation(W, s)`` from the model output.

Usage:
    python src/train.py
    python src/train.py training.steps=500 training.dataset_dir=./data/debug_latent
    WANDB_MODE=disabled python src/train.py training.steps=200
"""

from __future__ import annotations

import math
import os
import sys
from glob import glob
from itertools import cycle

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import wandb

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataset import CopulaDataset, collate_fn
from loss import gp_oracle_y_nll, oracle_copula_nll, y_space_nll
from model import build_copula_transformer, low_rank_correlation


def cosine_lr_lambda(step: int, warmup: int, total: int, lr_min_frac: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min_frac + (1.0 - lr_min_frac) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


@torch.no_grad()
def validate(model: nn.Module, val_files: list, cfg: DictConfig, device: str) -> dict:
    model.eval()
    val_loader = DataLoader(
        CopulaDataset(file_list=val_files),
        batch_size=cfg.training.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
    )
    jitter = float(cfg.model.get("sigma_jitter", 1e-4))

    tot, cop, mar, ora, ora_cop, ora_mar, ora_cop_z = [], [], [], [], [], [], []
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        Sigma = low_rank_correlation(
            out["W"].float(), out["s"].float(), batch["test_mask"], jitter=jitter
        )
        parts = y_space_nll(
            Sigma,
            batch["z_test"].float(),
            batch["log_pdf_test"].float(),
            batch["test_mask"],
        )
        oracle_parts = gp_oracle_y_nll(
            batch["Sigma_star"].float(),
            batch["mu_star"].float(),
            batch["y_test"].float(),
            batch["test_mask"],
        )
        ora_cop_z_val = oracle_copula_nll(
            batch["R_star"].float(),
            batch["z_test"].float(),
            batch["test_mask"],
        )
        tot.append(parts["total"].item())
        cop.append(parts["copula"].item())
        mar.append(parts["marginal"].item())
        ora.append(oracle_parts["total"].item())
        ora_cop.append(oracle_parts["copula"].item())
        ora_mar.append(oracle_parts["marginal"].item())
        ora_cop_z.append(ora_cop_z_val.item())

    metrics = {
        "y_nll_total": sum(tot) / len(tot),
        "y_nll_copula": sum(cop) / len(cop),
        "y_nll_marginal": sum(mar) / len(mar),
        "y_nll_oracle": sum(ora) / len(ora),
        "y_nll_oracle_copula": sum(ora_cop) / len(ora_cop),
        "y_nll_oracle_marginal": sum(ora_mar) / len(ora_mar),
        "y_nll_oracle_copula_z": sum(ora_cop_z) / len(ora_cop_z),
    }
    metrics["oracle_gap"] = metrics["y_nll_total"] - metrics["y_nll_oracle"]
    metrics["copula_gap"] = metrics["y_nll_copula"] - metrics["y_nll_oracle_copula_z"]
    model.train()
    return metrics


def save_checkpoint(model, optimizer, scheduler, cfg, step: int) -> None:
    os.makedirs(cfg.training.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.training.ckpt_dir, f"step_{step:07d}.pt")
    raw = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "step": step,
            "state_dict": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cfg": OmegaConf.to_container(cfg),
        },
        path,
    )


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = (
        "cuda" if cfg.training.device == "auto" and torch.cuda.is_available()
        else ("cpu" if cfg.training.device == "auto" else cfg.training.device)
    )

    t = cfg.training
    run_name = (
        f"copula_r={cfg.model.rank}"
        f"_lr={t.lr}_steps={t.steps}"
        f"_unfreeze={bool(cfg.model.get('unfreeze_backbone', False))}"
    )
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity if cfg.wandb.entity else None,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    all_files = sorted(glob(os.path.join(t.dataset_dir, "*.pt")))
    if not all_files:
        raise RuntimeError(
            f"No episode files in {t.dataset_dir}. Run generate_pit_dataset.py first."
        )
    n_val = max(1, int(len(all_files) * t.val_fraction))
    val_files = all_files[:n_val]
    train_files = all_files[n_val:]
    print(f"Train: {len(train_files)} | Val: {len(val_files)} episodes")

    train_loader = DataLoader(
        CopulaDataset(file_list=train_files),
        batch_size=t.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    torch._dynamo.config.capture_scalar_outputs = True
    model = build_copula_transformer(cfg).to(device)
    wandb.watch(model, log="gradients", log_freq=500)

    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_train_params:,}")
    wandb.config.update({"n_trainable_params": n_train_params})

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=t.lr, weight_decay=0.0)
    lr_min_frac = t.lr_min / t.lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_lr_lambda(s, t.warmup_steps, t.steps, lr_min_frac),
    )

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = GradScaler(device=device) if (use_amp and amp_dtype == torch.float16) else None

    jitter = float(cfg.model.get("sigma_jitter", 1e-4))

    model.train()
    data_iter = cycle(train_loader)

    for step in range(t.steps + 1):
        batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            out = model(batch)
        # Loss in float32 — Cholesky / log-det want full precision.
        Sigma = low_rank_correlation(
            out["W"].float(), out["s"].float(), batch["test_mask"], jitter=jitter
        )
        parts = y_space_nll(
            Sigma,
            batch["z_test"].float(),
            batch["log_pdf_test"].float(),
            batch["test_mask"],
        )
        loss = parts["total"]

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, t.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, t.clip_grad_norm)
            optimizer.step()

        scheduler.step()

        if step % t.log_every == 0:
            lr_now = scheduler.get_last_lr()[0]
            wandb.log(
                {
                    "train/y_nll_total": loss.item(),
                    "train/y_nll_copula": parts["copula"].item(),
                    "train/y_nll_marginal": parts["marginal"].item(),
                    "train/lr": lr_now,
                },
                step=step,
            )
            print(
                f"[{step:6d}] total={loss.item():.4f} "
                f"cop={parts['copula'].item():.4f} "
                f"mar={parts['marginal'].item():.4f}  lr={lr_now:.2e}"
            )

        if step % t.val_every == 0 and step > 0:
            metrics = validate(model, val_files, cfg, device)
            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=step)
            print(
                f"[{step:6d}] val total={metrics['y_nll_total']:.4f} "
                f"oracle={metrics['y_nll_oracle']:.4f} "
                f"gap={metrics['oracle_gap']:.4f}  "
                f"cop={metrics['y_nll_copula']:.4f} "
                f"ora_cop_z={metrics['y_nll_oracle_copula_z']:.4f} "
                f"cop_gap={metrics['copula_gap']:.4f}  "
                f"ora_cop_y={metrics['y_nll_oracle_copula']:.4f} "
                f"ora_mar={metrics['y_nll_oracle_marginal']:.4f}"
            )

        if step % t.save_every == 0 and step > 0:
            save_checkpoint(model, optimizer, scheduler, cfg, step)

    save_checkpoint(model, optimizer, scheduler, cfg, t.steps)
    wandb.finish()


if __name__ == "__main__":
    main()
