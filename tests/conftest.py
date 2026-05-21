"""Shared pytest fixtures for the inter-instance copula test suite."""

from __future__ import annotations

import os
import sys

import pytest
import torch
from omegaconf import OmegaConf

# Make src/ importable
_TESTS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_TESTS), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


@pytest.fixture(scope="session")
def small_cfg():
    """Minimal Hydra-like config for fast CPU tests (no GPU required)."""
    return OmegaConf.create(
        {
            "model": {
                "d_model": 32,
                "n_heads": 4,
                "L_col": 1,
                "L_row": 1,
                "L_ICL": 2,
                "n_inducing": 16,
                "n_cls": 4,
                "rank": 2,
                "dropout": 0.0,
            },
            "data": {
                "d_features": 1,
                "P_min": 5,
                "P_max": 10,
                "N_min": 3,
                "N_max": 6,
                "n_tasks": 4,
                "l_min": 0.5,
                "l_max": 1.5,
                "alpha2_min": 0.5,
                "alpha2_max": 1.5,
                "noise_min": 0.05,
                "noise_max": 0.2,
                "raw_dir": "/tmp/test_gp_raw",
                "latent_dir": "/tmp/test_pit",
                "resume": False,
            },
        }
    )


def make_batch(B: int = 2, P: int = 10, N: int = 5, d_x: int = 1) -> dict:
    """Return a fully-valid padded batch (uniform P and N for simplicity)."""
    return {
        "x_train": torch.randn(B, P, d_x),
        "z_train": torch.randn(B, P),
        "x_test": torch.randn(B, N, d_x),
        "z_test": torch.randn(B, N),
        "train_mask": torch.ones(B, P, dtype=torch.bool),
        "test_mask": torch.ones(B, N, dtype=torch.bool),
        "R_star": torch.eye(N).unsqueeze(0).expand(B, -1, -1).clone(),
        "n_train": torch.full((B,), P, dtype=torch.long),
        "n_test": torch.full((B,), N, dtype=torch.long),
    }
