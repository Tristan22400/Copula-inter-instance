"""Tests for data_gen.corrupt_z_train: off by default, a no-op at prob 0, and corr(z_train, z) ~ sqrt(rho)."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from copula_inter.data_gen import corrupt_z_train


def make_z_train(B: int = 200, P: int = 40, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, P, generator=g)


def test_disabled_is_noop() -> None:
    z_train = make_z_train()
    cfg = OmegaConf.create({})  # z_train_corruption_enabled absent -> defaults False
    out = corrupt_z_train(z_train, cfg)
    assert torch.equal(out, z_train)


def test_zero_prob_is_noop() -> None:
    z_train = make_z_train()
    cfg = OmegaConf.create(
        {
            "z_train_corruption_enabled": True,
            "z_train_corruption_prob": 0.0,
        }
    )
    out = corrupt_z_train(z_train, cfg)
    assert torch.equal(out, z_train)


def test_achieves_target_correlation() -> None:
    """corr(z_train, z_corrupted) ~ sqrt(rho)."""
    torch.manual_seed(0)
    B, P = 4000, 1  # many independent episodes, single "point" per episode
    # One point per episode: measure the across-episode correlation.
    z_train = make_z_train(B=B, P=P, seed=1)

    cfg = OmegaConf.create(
        {
            "z_train_corruption_enabled": True,
            "z_train_corruption_prob": 1.0,
            # rho fixed near a known value.
            "z_train_corruption_rho_beta_a": 5000.0,
            "z_train_corruption_rho_beta_b": 5000.0 * (1.0 - 0.5) / 0.5,  # mean rho ~= 0.5
        }
    )
    out = corrupt_z_train(z_train, cfg)

    achieved = torch.corrcoef(torch.stack([z_train.squeeze(-1), out.squeeze(-1)]))[0, 1].item()
    expected = 0.5**0.5  # sqrt(rho), rho ~= 0.5
    assert abs(achieved - expected) < 0.03, f"achieved corr={achieved:.3f}, expected~={expected:.3f}"
