"""
test_z_train_corruption.py — Sanity checks for pit.py::corrupt_z_train, the
z_train robustness augmentation (see its docstring / conf/config.yaml's
training.z_train_corruption_* knobs for the motivation: CopulaTabICL is
trained exclusively on the exact closed-form GP-LOO whitened residual, but
real (non-GP) deployment data can only ever produce an approximate PIT).

Tests verify:
  1. Disabled by default (training_cfg without z_train_corruption_enabled) is
     an exact no-op.
  2. z_train_corruption_prob=0 (or a curriculum ramp that hasn't started yet)
     is an exact no-op.
  3. corr(z_train, z_corrupted) ~= sqrt(rho) in expectation, matching the
     closed-form derivation in the docstring (z_train and the i.i.d. noise
     term are independent, both unit-variance).
  4. Padding (train_mask=False positions) stays exactly zero after
     corruption.
  5. linear_warmup curriculum ramps effective corruption probability
     smoothly from 0 at step 0 to the target prob at warmup_steps.
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
    out = corrupt_z_train(z_train, train_mask, cfg, step=100000)
    assert torch.equal(out, z_train)


def test_zero_prob_is_noop():
    z_train, train_mask = make_batch()
    cfg = OmegaConf.create({
        "z_train_corruption_enabled": True,
        "z_train_corruption_prob": 0.0,
    })
    out = corrupt_z_train(z_train, train_mask, cfg, step=100000)
    assert torch.equal(out, z_train)


def test_warmup_not_started_is_noop():
    z_train, train_mask = make_batch()
    cfg = OmegaConf.create({
        "z_train_corruption_enabled": True,
        "z_train_corruption_prob": 1.0,
        "z_train_corruption_curriculum": "linear_warmup",
        "z_train_corruption_warmup_steps": 10000,
    })
    out = corrupt_z_train(z_train, train_mask, cfg, step=0)
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
        "z_train_corruption_curriculum": "none",
        # Fix rho tightly around a known value so the achieved correlation
        # has a precise closed-form target to compare against.
        "z_train_corruption_rho_beta_a": 5000.0,
        "z_train_corruption_rho_beta_b": 5000.0 * (1.0 - 0.5) / 0.5,  # mean rho ~= 0.5
    })
    out = corrupt_z_train(z_train, train_mask, cfg, step=999999)

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
        "z_train_corruption_curriculum": "none",
    })
    out = corrupt_z_train(z_train, train_mask, cfg, step=999999)
    assert torch.all(out[:, P // 2:] == 0.0)


def test_linear_warmup_ramps_prob():
    """Corruption FREQUENCY across many single-episode calls should scale
    with the curriculum ramp fraction (step / warmup_steps)."""
    torch.manual_seed(4)
    warmup_steps = 1000
    cfg = OmegaConf.create({
        "z_train_corruption_enabled": True,
        "z_train_corruption_prob": 1.0,
        "z_train_corruption_curriculum": "linear_warmup",
        "z_train_corruption_warmup_steps": warmup_steps,
    })

    def frac_corrupted(step, n=3000):
        z_train, train_mask = make_batch(B=n, P=1, seed=step)
        out = corrupt_z_train(z_train, train_mask, cfg, step=step)
        return (out != z_train).any(dim=1).float().mean().item()

    f_half = frac_corrupted(warmup_steps // 2)
    f_full = frac_corrupted(warmup_steps)
    assert 0.3 < f_half < 0.7, f"expected ~0.5 corrupted at half warmup, got {f_half:.3f}"
    assert f_full > 0.9, f"expected ~all corrupted at full warmup, got {f_full:.3f}"
