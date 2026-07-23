"""
test_resume_checkpoint.py — Verify resume_ckpt restarts training at step 0.

train.load_checkpoint() used to restore optimizer/scheduler/scaler state and
return `ckpt["step"] + 1`, so resuming continued the previous run's LR
schedule and step count. It now restores only the model weights and returns
nothing: resuming always restarts at step 0 with a fresh warmup/cosine
schedule and the full step budget.

Tests verify:
  1. load_checkpoint restores model weights from a checkpoint saved by
     save_checkpoint
  2. load_checkpoint does NOT mutate optimizer state (no momentum/step
     buffers leak from the checkpoint's optimizer)
  3. load_checkpoint does NOT mutate scheduler state (last_epoch stays at
     whatever the fresh scheduler was already at, e.g. 0)
  4. load_checkpoint returns None (call sites must not derive a resume step
     from it; train.py's main() hardcodes start_step = 0 instead)
  5. load_checkpoint still raises FileNotFoundError for a missing path
  6. load_checkpoint works through torch.compile's `_orig_mod` wrapper
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from train import load_checkpoint, save_checkpoint


def make_model_optimizer_scheduler(seed: int):
    torch.manual_seed(seed)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    return model, optimizer, scheduler


def train_a_step(model, optimizer, scheduler):
    """Run one optimizer step so optimizer/scheduler state is non-trivial."""
    out = model(torch.randn(2, 4))
    out.sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


@pytest.fixture
def saved_ckpt(tmp_path):
    model, optimizer, scheduler = make_model_optimizer_scheduler(seed=0)
    train_a_step(model, optimizer, scheduler)  # gives optimizer momentum buffers, scheduler last_epoch=1

    cfg = OmegaConf.create({"training": {"ckpt_dir": str(tmp_path)}})
    save_checkpoint(model, optimizer, scheduler, cfg, step=42)
    return str(tmp_path / "step_0000042.pt"), model


def test_load_checkpoint_restores_weights(saved_ckpt):
    ckpt_path, saved_model = saved_ckpt
    fresh_model, _, _ = make_model_optimizer_scheduler(seed=1)  # different init
    assert not torch.equal(fresh_model.weight, saved_model.weight)

    load_checkpoint(ckpt_path, fresh_model, device="cpu")

    assert torch.equal(fresh_model.weight, saved_model.weight)
    assert torch.equal(fresh_model.bias, saved_model.bias)


def test_load_checkpoint_does_not_touch_optimizer_or_scheduler(saved_ckpt):
    ckpt_path, _ = saved_ckpt
    fresh_model, fresh_optimizer, fresh_scheduler = make_model_optimizer_scheduler(seed=1)

    optimizer_state_before = fresh_optimizer.state_dict()
    scheduler_state_before = fresh_scheduler.state_dict()
    assert optimizer_state_before["state"] == {}  # never stepped
    assert scheduler_state_before["last_epoch"] == 0

    load_checkpoint(ckpt_path, fresh_model, device="cpu")

    assert fresh_optimizer.state_dict() == optimizer_state_before
    assert fresh_scheduler.state_dict() == scheduler_state_before


def test_load_checkpoint_returns_none(saved_ckpt):
    ckpt_path, _ = saved_ckpt
    fresh_model, _, _ = make_model_optimizer_scheduler(seed=1)
    assert load_checkpoint(ckpt_path, fresh_model, device="cpu") is None


def test_load_checkpoint_signature_has_no_optimizer_scheduler_params():
    """Guards against re-introducing optimizer/scheduler restoration."""
    params = list(inspect.signature(load_checkpoint).parameters)
    assert params == ["ckpt_path", "model", "device"]


def test_load_checkpoint_missing_file_raises(tmp_path):
    fresh_model, _, _ = make_model_optimizer_scheduler(seed=1)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(str(tmp_path / "does_not_exist.pt"), fresh_model, device="cpu")


def test_load_checkpoint_through_compile_wrapper(saved_ckpt):
    """torch.compile wraps the model in an OptimizedModule exposing `_orig_mod`;
    load_checkpoint must load into that inner module, matching save_checkpoint's
    symmetric unwrap."""
    ckpt_path, saved_model = saved_ckpt
    fresh_model, _, _ = make_model_optimizer_scheduler(seed=1)

    class FakeCompiledWrapper(nn.Module):
        def __init__(self, orig_mod):
            super().__init__()
            self._orig_mod = orig_mod

    wrapper = FakeCompiledWrapper(fresh_model)
    load_checkpoint(ckpt_path, wrapper, device="cpu")

    assert torch.equal(fresh_model.weight, saved_model.weight)
