"""
train.py — Training loop for the inter-instance Copula Transformer.

Trains on pre-computed PIT episodes (Stage A+B output) to predict the
low-rank inter-instance correlation matrix W_tilde.

Usage:
    python src/train.py
    python src/train.py training.steps=200 training.dataset_dir=./data/debug_latent
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
from loss import copula_nll, oracle_copula_nll
from model import build_copula_transformer

# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------


def cosine_lr_lambda(step: int, warmup: int, total: int, lr_min_frac: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min_frac + (1.0 - lr_min_frac) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


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

    nll_pred_list, nll_oracle_list, frob_list, pearson_list = [], [], [], []

    eps = 0.1  # must match copula_nll default

    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        W = model(batch).float()

        nll_pred = copula_nll(W, batch["z_test"].float(), batch["test_mask"])
        nll_oracle = oracle_copula_nll(
            batch["R_star"].float(), batch["z_test"].float(), batch["test_mask"]
        )
        nll_pred_list.append(nll_pred.item())
        nll_oracle_list.append(nll_oracle.item())

        # Materialise C_hat = normalized correlation matrix from R_ε = eps*I + W W^T
        WWT = torch.bmm(W, W.transpose(-2, -1))  # (B, N_max, N_max)
        R_eps = eps * torch.eye(W.shape[1], device=device).unsqueeze(0) + WWT
        R_diag = R_eps.diagonal(dim1=-2, dim2=-1)  # (B, N_max)
        D_inv_sqrt = 1.0 / R_diag.clamp(min=1e-8).sqrt()
        C_hat = D_inv_sqrt.unsqueeze(-1) * R_eps * D_inv_sqrt.unsqueeze(-2)

        Bval = W.shape[0]
        for b in range(Bval):
            N = batch["n_test"][b].item()
            R_h = C_hat[b, :N, :N].detach()
            R_s = batch["R_star"][b, :N, :N]

            # Frobenius error per entry
            frob_list.append((R_h - R_s).norm(p="fro").item() / N)

            # Off-diagonal Pearson correlation
            mask = ~torch.eye(N, dtype=torch.bool, device=device)
            rh_off = R_h[mask].cpu()
            rs_off = R_s[mask].cpu()
            if rh_off.std() > 1e-8 and rs_off.std() > 1e-8:
                pearson = torch.corrcoef(torch.stack([rh_off, rs_off]))[0, 1].item()
                pearson_list.append(pearson)

    metrics = {
        "copula_nll": sum(nll_pred_list) / len(nll_pred_list),
        "oracle_nll": sum(nll_oracle_list) / len(nll_oracle_list),
        "frob_error": sum(frob_list) / len(frob_list),
    }
    if pearson_list:
        metrics["offdiag_pearson"] = sum(pearson_list) / len(pearson_list)
    # NLL gap: how far above the oracle lower bound the model sits (lower is better)
    metrics["nll_gap"] = metrics["copula_nll"] - metrics["oracle_nll"]

    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(model, optimizer, scheduler, cfg, step: int) -> None:
    os.makedirs(cfg.training.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.training.ckpt_dir, f"step_{step:07d}.pt")
    # Unwrap torch.compile wrapper if present
    raw_model = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "step": step,
            "state_dict": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cfg": OmegaConf.to_container(cfg),
        },
        path,
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)

    device = (
        "cuda"
        if cfg.training.device == "auto" and torch.cuda.is_available()
        else ("cpu" if cfg.training.device == "auto" else cfg.training.device)
    )

    # ---- W&B run name encodes key hyperparameters ----
    m = cfg.model
    t = cfg.training
    run_name = (
        f"lr={t.lr}"
        f"_steps={t.steps}"
        f"_d={m.d_model}"
        f"_Lcol={m.L_col}_Lrow={m.L_row}_LICL={m.L_ICL}"
        f"_H={m.n_heads}"
        f"_r={m.rank}"
    )
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity if cfg.wandb.entity else None,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # ---- Data split ----
    all_files = sorted(glob(os.path.join(t.dataset_dir, "*.pt")))
    if len(all_files) == 0:
        raise RuntimeError(
            f"No episode files found in {t.dataset_dir}. "
            "Run generate_pit_dataset.py first."
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

    # ---- Model ----
    torch._dynamo.config.capture_scalar_outputs = (
        True  # needed for TabICLv2 .item() in embedding
    )
    model = build_copula_transformer(cfg).to(device)
    model = torch.compile(model, mode="default")
    wandb.watch(model, log="gradients", log_freq=500)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params})

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t.lr, weight_decay=t.weight_decay
    )
    lr_min_frac = t.lr_min / t.lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_lr_lambda(s, t.warmup_steps, t.steps, lr_min_frac),
    )

    # ---- AMP ----
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler(device=device) if use_amp else None

    # ---- Training loop ----
    model.train()
    data_iter = cycle(train_loader)

    for step in range(t.steps + 1):
        batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            W = model(batch)
        # Loss in float32: Woodbury has large intermediate values in bfloat16
        loss = copula_nll(W.float(), batch["z_test"].float(), batch["test_mask"])

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), t.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), t.clip_grad_norm)
            optimizer.step()

        scheduler.step()

        # ---- Logging ----
        if step % t.log_every == 0:
            lr_now = scheduler.get_last_lr()[0]
            wandb.log(
                {"train/copula_nll": loss.item(), "train/lr": lr_now},
                step=step,
            )
            print(f"[{step:6d}] loss={loss.item():.4f}  lr={lr_now:.2e}")

        # ---- Validation ----
        if step % t.val_every == 0 and step > 0:
            metrics = validate(model, val_files, cfg, device)
            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=step)
            print(
                f"[{step:6d}] val copula_nll={metrics['copula_nll']:.4f} "
                f"oracle_nll={metrics['oracle_nll']:.4f} "
                f"nll_gap={metrics['nll_gap']:.4f} "
                f"frob={metrics['frob_error']:.4f}"
            )

        # ---- Checkpoint ----
        if step % t.save_every == 0 and step > 0:
            save_checkpoint(model, optimizer, scheduler, cfg, step)

    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, cfg, t.steps)
    wandb.finish()


if __name__ == "__main__":
    main()
