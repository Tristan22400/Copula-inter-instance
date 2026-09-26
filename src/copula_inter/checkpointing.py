"""Save and load copula training checkpoints."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.amp import GradScaler

from copula_inter.artifacts import atomic_torch_save


def save_checkpoint(model, optimizer, scheduler, cfg, step: int, scaler=None) -> None:
    if cfg.training.ckpt_dir is None:
        return
    os.makedirs(cfg.training.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.training.ckpt_dir, f"step_{step:07d}.pt")
    raw = getattr(model, "_orig_mod", model)
    atomic_torch_save(
        {
            "step": step,
            "state_dict": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "cfg": OmegaConf.to_container(cfg),
        },
        path,
    )


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
) -> int:
    """Restore model weights and optimizer/scaler state; return the saved step (0 if absent).

    If the state dict does not match the model exactly, load non-strictly and
    skip optimizer state (its entries are matched by parameter position).
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"resume_ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    raw = getattr(model, "_orig_mod", model)
    ckpt_state = ckpt["state_dict"]
    model_keys = set(raw.state_dict().keys())
    ckpt_keys = set(ckpt_state.keys())
    if model_keys != ckpt_keys:
        missing, unexpected = sorted(model_keys - ckpt_keys), sorted(ckpt_keys - model_keys)
        print(
            f"[load_checkpoint] {ckpt_path} was saved by a different model architecture: "
            f"missing key(s) (kept at random init) {missing}; unexpected key(s) (dropped) {unexpected}. "
            f"Loading weights non-strict and skipping optimizer state restore (param positions "
            f"downstream of the changed layer would otherwise be misaligned)."
        )
        raw.load_state_dict(ckpt_state, strict=False)
    else:
        raw.load_state_dict(ckpt_state)
        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0))
