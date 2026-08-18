"""
test_resume_checkpoint.py — Verify resume_ckpt continues the LR schedule.

Earlier history: load_checkpoint() used to restore optimizer/scheduler/scaler
state and return `ckpt["step"] + 1`; it was later changed to restore only
model weights, then optimizer/scaler restoration was reinstated (commit
09ea72e) as opt-in params — but scheduler continuation was left out, so every
resume restarted at step 0 with a fresh warmup/cosine schedule stacked on top
of already-warmed-up optimizer moments. That combination spiked the
effective step size right after resume, which is the fix here: load_checkpoint
now returns the checkpoint's step, and train.py (see cosine_lr_lambda and the
scheduler construction in train()) uses it to continue the same cosine curve
by default, with `training.resume_reset_schedule=true` as the opt-out back to
the old from-scratch behavior.

Tests verify:
  1. load_checkpoint restores model weights from a checkpoint saved by
     save_checkpoint
  2. load_checkpoint does NOT mutate optimizer state when optimizer isn't
     passed in (no momentum/step buffers leak from the checkpoint)
  3. load_checkpoint does NOT mutate scheduler state (there is no scheduler
     param to opt into this, unlike optimizer/scaler)
  4. load_checkpoint returns the checkpoint's saved step (0 for a legacy
     checkpoint dict without a "step" key)
  5. load_checkpoint still raises FileNotFoundError for a missing path
  6. load_checkpoint works through torch.compile's `_orig_mod` wrapper
  7. cosine_lr_lambda clamps progress at 1.0 instead of swinging back upward
     past the end of the schedule
  8. constructing a LambdaLR with last_epoch=start_step-1 (train()'s
     resume-continuation pattern) reproduces the LR an uninterrupted run
     would have had at that step, matching a scheduler stepped there directly
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from train import cosine_lr_lambda, load_checkpoint, save_checkpoint


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


def test_load_checkpoint_does_not_touch_optimizer_or_scheduler_when_not_passed(saved_ckpt):
    ckpt_path, _ = saved_ckpt
    fresh_model, fresh_optimizer, fresh_scheduler = make_model_optimizer_scheduler(seed=1)

    optimizer_state_before = fresh_optimizer.state_dict()
    scheduler_state_before = fresh_scheduler.state_dict()
    assert optimizer_state_before["state"] == {}  # never stepped
    assert scheduler_state_before["last_epoch"] == 0

    load_checkpoint(ckpt_path, fresh_model, device="cpu")  # optimizer/scaler omitted

    assert fresh_optimizer.state_dict() == optimizer_state_before
    assert fresh_scheduler.state_dict() == scheduler_state_before


def test_load_checkpoint_returns_step(saved_ckpt):
    ckpt_path, _ = saved_ckpt
    fresh_model, _, _ = make_model_optimizer_scheduler(seed=1)
    assert load_checkpoint(ckpt_path, fresh_model, device="cpu") == 42


def test_load_checkpoint_returns_zero_for_legacy_checkpoint_without_step(tmp_path):
    """Legacy checkpoints saved before "step" was added to the dict should
    resume like a from-scratch run rather than crashing."""
    fresh_model, _, _ = make_model_optimizer_scheduler(seed=1)
    torch.save({"state_dict": fresh_model.state_dict()}, tmp_path / "legacy.pt")

    fresh_model2, _, _ = make_model_optimizer_scheduler(seed=1)
    assert load_checkpoint(str(tmp_path / "legacy.pt"), fresh_model2, device="cpu") == 0


def test_load_checkpoint_signature_has_no_scheduler_param():
    """Guards against re-introducing direct scheduler-state restoration
    (optimizer/scaler restoration was intentionally reinstated in commit
    09ea72e as opt-in params, default None) — schedule continuation instead
    goes through the returned step, not a restored scheduler object."""
    params = inspect.signature(load_checkpoint).parameters
    assert list(params) == ["ckpt_path", "model", "device", "optimizer", "scaler"]
    assert params["optimizer"].default is None
    assert params["scaler"].default is None


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


def test_cosine_lr_lambda_clamps_past_total():
    warmup, total, lr_min_frac = 10, 100, 0.01
    at_end = cosine_lr_lambda(total, warmup, total, lr_min_frac)
    past_end = cosine_lr_lambda(total + 50, warmup, total, lr_min_frac)
    way_past_end = cosine_lr_lambda(total * 3, warmup, total, lr_min_frac)

    assert past_end == pytest.approx(lr_min_frac)
    assert way_past_end == pytest.approx(lr_min_frac)
    assert at_end == pytest.approx(past_end)


def test_resume_continues_cosine_schedule_instead_of_restarting():
    """Mirrors train()'s resume-continuation pattern: construct a fresh
    LambdaLR at last_epoch=-1 and step it up to `resume_step`, vs. directly
    constructing one with last_epoch=resume_step-1 (what train() does after
    resume, once param groups have 'initial_lr' rebased to the base LR) —
    the two must land on the same LR, proving resume doesn't reset warmup."""
    warmup, total, base_lr, lr_min = 10, 100, 0.1, 0.001
    lr_min_frac = lr_min / base_lr
    resume_step = 55

    def make_optimizer():
        model = nn.Linear(4, 4)
        return torch.optim.SGD(model.parameters(), lr=base_lr)

    lr_lambda = lambda s: cosine_lr_lambda(s, warmup, total, lr_min_frac)

    uninterrupted_opt = make_optimizer()
    uninterrupted_sched = torch.optim.lr_scheduler.LambdaLR(uninterrupted_opt, lr_lambda=lr_lambda)
    for _ in range(resume_step):
        uninterrupted_sched.step()

    resumed_opt = make_optimizer()
    for group in resumed_opt.param_groups:
        group["initial_lr"] = base_lr
    resumed_sched = torch.optim.lr_scheduler.LambdaLR(
        resumed_opt, lr_lambda=lr_lambda, last_epoch=resume_step - 1
    )

    assert resumed_sched.get_last_lr() == pytest.approx(uninterrupted_sched.get_last_lr())
    # Sanity: this is mid-decay, not the from-scratch warmup-start LR a
    # step-0 restart would have produced.
    fresh_restart_lr = base_lr * lr_lambda(0)
    assert resumed_sched.get_last_lr()[0] != pytest.approx(fresh_restart_lr)
