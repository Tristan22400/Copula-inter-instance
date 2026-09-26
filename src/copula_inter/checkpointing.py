"""Save and load copula training checkpoints."""

from __future__ import annotations

import os

# P/N (hence attention sequence length T=P+N) are sampled per-shard from a wide
# range (see conf/data/gp_tasks.yaml P_min/P_max, N_min/N_max), so batches vary
# a lot in size while batch_size stays fixed — some shards get much closer to
# the VRAM ceiling than others. When that happens, PyTorch's caching allocator
# can fail a small allocation despite reserved-but-unallocated memory being
# nominally sufficient, because it's fragmented into pieces too small to
# satisfy the request (see the OOM message's "reserved but unallocated"
# figure). expandable_segments avoids this by growing/shrinking allocations
# in-place instead of requiring a fresh contiguous chunk. Must be set before
# the CUDA caching allocator initializes (i.e. before any CUDA call), so this
# goes at the top of the file, before `import torch`. setdefault so an
# explicit environment override still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


import matplotlib

matplotlib.use("Agg")
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.amp import GradScaler

# eval/ (regions.py, spatial-correlation probe helpers -- see
# _build_era5_val_batches below) lives at the repo root, not under src/.
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
    """Restore model weights and optimizer/scaler state from a checkpoint.

    Optimizer moments (Adam/Muon) and the AMP grad scaler state are restored
    so the run doesn't have to relearn gradient statistics from scratch.
    Returns the step the checkpoint was saved at (0 for legacy checkpoints
    without a "step" key), which the caller uses to decide where the LR
    schedule resumes — see the `resume_reset_schedule` handling in train().

    If the checkpoint predates an architecture change (e.g. a head gaining
    extra layers), its state_dict keys won't match the live model 1:1. We
    load non-strict in that case (matching tensors restored, new ones keep
    their random init) and skip restoring optimizer state entirely — Muon/
    Adam match state to params by position in the flattened param list, and
    an inserted/removed tensor shifts every param after it, which would
    silently apply the wrong param's moments downstream of the changed layer.
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
