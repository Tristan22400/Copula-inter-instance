"""Run setup: global RNG seeding and device selection."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU and every CUDA device) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> str:
    """spec, with "auto" meaning CUDA when available, else CPU."""
    if spec != "auto":
        return spec
    return "cuda" if torch.cuda.is_available() else "cpu"
