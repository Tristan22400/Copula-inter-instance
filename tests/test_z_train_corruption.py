"""
test_z_train_corruption.py — Sanity checks for pit.py::corrupt_z_train, the
z_train robustness augmentation (see its docstring / conf/data/gp_tasks.yaml's
z_train_corruption_* knobs for the motivation: CopulaTabICL is trained
exclusively on the exact closed-form GP-LOO whitened residual, but real
(non-GP) deployment data can only ever produce an approximate PIT).

Tests verify:
  1. Disabled by default (data_cfg without z_train_corruption_enabled) is
     an exact no-op.
  2. z_train_corruption_prob=0 is an exact no-op.
  3. corr(z_train, z_corrupted) ~= sqrt(rho) in expectation, matching the
     closed-form derivation in the docstring (z_train and the i.i.d. noise
     term are independent, both unit-variance).
  4. Padding (train_mask=False positions) stays exactly zero after
     corruption.
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from pit import corrupt_z_train


def make_batch(B: int = 200, P: int = 40, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    z_train = torch.randn(B, P, generator=g)
    train_mask = torch.ones(B, P, dtype=torch.bool)
    return z_train, train_mask


def test_disabled_is_noop():
    z_train, train_mask = make_batch()
    cfg = OmegaConf.create({})  # z_train_corruption_enabled absent -> defaults False
    out = corrupt_z_train(z_train, train_mask, cfg)
    assert torch.equal(out, z_train)


def test_zero_prob_is_noop():
    z_train, train_mask = make_batch()
    cfg = OmegaConf.create({
        "z_train_corruption_enabled": True,
        "z_train_corruption_prob": 0.0,
    })
    out = corrupt_z_train(z_train, train_mask, cfg)
    assert torch.equal(out, z_train)


def test_noise_blend_achieves_target_correlation():
    """corr(z_train, z_corrupted) should track sqrt(rho) in expectation
    (independent unit-variance blend), matching the closed-form
    Cov(z, blend) = sqrt(rho), Var(blend) = 1 derivation in the docstring."""
    torch.manual_seed(0)
    B, P = 4000, 1  # many independent episodes, single "point" per episode
    # so the resulting (B,) vectors of z_train / z_corrupted let us measure
    # the ACROSS-EPISODE correlation directly, isolating the per-element
    # blend behaviour from any within-episode structure.
    z_train, train_mask = make_batch(B=B, P=P, seed=1)

    cfg = OmegaConf.create({
        "z_train_corruption_enabled": True,
        "z_train_corruption_prob": 1.0,
        # Fix rho tightly around a known value so the achieved correlation
        # has a precise closed-form target to compare against.
        "z_train_corruption_rho_beta_a": 5000.0,
        "z_train_corruption_rho_beta_b": 5000.0 * (1.0 - 0.5) / 0.5,  # mean rho ~= 0.5
    })
    out = corrupt_z_train(z_train, train_mask, cfg)

    achieved = torch.corrcoef(torch.stack([z_train.squeeze(-1), out.squeeze(-1)]))[0, 1].item()
    expected = 0.5 ** 0.5  # sqrt(rho), rho ~= 0.5
    assert abs(achieved - expected) < 0.03, f"achieved corr={achieved:.3f}, expected~={expected:.3f}"


def test_padding_stays_zero():
    torch.manual_seed(2)
    B, P = 64, 20
    z_train, train_mask = make_batch(B=B, P=P, seed=2)
    # Make half of each episode padding, with nonzero garbage in the padded
    # z_train slots (collate_fn itself always zero-pads, but the corruption
    # function shouldn't assume that -- it should mask explicitly).
    train_mask[:, P // 2:] = False
    z_train[:, P // 2:] = 999.0

    cfg = OmegaConf.create({
        "z_train_corruption_enabled": True,
        "z_train_corruption_prob": 1.0,
    })
    out = corrupt_z_train(z_train, train_mask, cfg)
    assert torch.all(out[:, P // 2:] == 0.0)
