"""Regression tests for training-step CUDA OOM cleanup."""

from __future__ import annotations

import gc
import traceback
import weakref

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import train


class _TinyModel(nn.Module):
    def __init__(self, n_test: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, n_test, 2))

    def forward(self, batch):
        return {
            "W": self.weight.expand(batch["x_train"].shape[0], -1, -1),
            "s": self.weight[..., :1],
        }


def test_oom_unwinds_train_step_graph(monkeypatch):
    """A failed optimizer step must not keep the step graph alive."""
    n_test = 3
    model = _TinyModel(n_test)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    batch = {
        "x_train": torch.zeros(2, 4, 1),
        "test_mask": torch.ones(2, n_test, dtype=torch.bool),
        "z_test": torch.zeros(2, n_test),
        "log_pdf_test": torch.zeros(2, n_test),
    }
    graph_ref = {}

    def fake_correlation(W, _s, _mask, jitter, **_kwargs):
        graph_ref["tensor"] = weakref.ref(W)
        return W[..., :1] @ W[..., :1].transpose(-1, -2)

    def fake_nll(Sigma, *_args):
        total = Sigma.square().mean()
        return {"total": total, "copula": total, "marginal": total}

    monkeypatch.setattr(train, "low_rank_correlation", fake_correlation)
    monkeypatch.setattr(train, "y_space_nll", fake_nll)

    def fail_step(*_args, **_kwargs):
        raise torch.cuda.OutOfMemoryError("synthetic OOM")

    monkeypatch.setattr(optimizer, "step", fail_step)

    try:
        train._run_train_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            trainable=list(model.parameters()),
            batch=batch,
            device="cpu",
            use_amp=False,
            amp_dtype=torch.float32,
            scaler=None,
            clip_grad_norm=1.0,
            nll_weight=1.0,
            aux_mae_weight=0.0,
            jitter=1e-4,
            triu_cache={},
            phase_start=lambda: None,
            phase_end=lambda *_args: None,
        )
    except torch.cuda.OutOfMemoryError as exc:
        # Match the production handler: clear traceback-held locals before
        # checking allocator-visible graph lifetime.
        traceback.clear_frames(exc.__traceback__)
        del exc
    else:
        pytest.fail("synthetic OOM was not raised")

    # This mirrors the caller's recovery path before empty_cache().
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    assert graph_ref["tensor"]() is None


def _fake_device_properties(total_gb: float):
    class _Props:
        total_memory = total_gb * 1e9

    return lambda _device: _Props()


def test_reserve_headroom_noop_when_tabicl_not_live(monkeypatch):
    """No live-generation TabICL workers -> nothing should cap this process."""
    calls = []
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *a: calls.append(a))
    cfg = OmegaConf.create({"data": {"z_train_source": "analytic", "z_train_tabicl_mix_enabled": False}})
    t = OmegaConf.create({"live_tabicl_num_workers": 2})
    train._reserve_gpu_headroom_for_live_tabicl(cfg, t, "cuda")
    assert calls == []


def test_reserve_headroom_noop_on_cpu(monkeypatch):
    """TabICL mix enabled but device=cpu (no GPU workers to protect) -> no-op."""
    calls = []
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *a: calls.append(a))
    cfg = OmegaConf.create({"data": {"z_train_tabicl_mix_enabled": True}})
    t = OmegaConf.create({"live_tabicl_num_workers": 2})
    train._reserve_gpu_headroom_for_live_tabicl(cfg, t, "cpu")
    assert calls == []


def test_reserve_headroom_noop_when_zero_workers(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *a: calls.append(a))
    monkeypatch.setattr(torch.cuda, "get_device_properties", _fake_device_properties(24.0))
    cfg = OmegaConf.create({"data": {"z_train_source": "tabicl"}})
    t = OmegaConf.create({"live_tabicl_num_workers": 0})
    train._reserve_gpu_headroom_for_live_tabicl(cfg, t, "cuda")
    assert calls == []


def test_reserve_headroom_caps_fraction_for_tabicl_workers(monkeypatch):
    """2 workers * 2.5GB + 1.0GB flat = 6GB headroom on a 24GB card -> 75%.

    The 2.5GB per worker is _LIVE_TABICL_WORKER_FIXED_OVERHEAD_GB (0.5) +
    _LIVE_TABICL_WORKER_PER_EPISODE_GB (0.02) * group_size, where
    group_size = training.batch_size * training.live_tabicl_group_multiplier
    (see train.py::_reserve_gpu_headroom_for_live_tabicl) -- so batch_size=50
    at the group_multiplier default of 2 gives group_size=100 -> 0.5 + 2.0.
    batch_size is REQUIRED in this fixture: the headroom formula started
    scaling with it when live_tabicl worker auto-sizing landed, and omitting
    it raises ConfigAttributeError rather than falling back to a default.
    """
    captured = {}
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction",
        lambda frac, dev: captured.update(fraction=frac, device=dev),
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", _fake_device_properties(24.0))
    cfg = OmegaConf.create({"data": {"z_train_tabicl_mix_enabled": True}})
    t = OmegaConf.create({"live_tabicl_num_workers": 2, "batch_size": 50})
    train._reserve_gpu_headroom_for_live_tabicl(cfg, t, "cuda")
    assert captured["device"] == 0
    assert captured["fraction"] == pytest.approx(0.75)


def test_reserve_headroom_clamps_fraction_floor(monkeypatch):
    """A huge worker count shouldn't starve this process itself below 50%."""
    captured = {}
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction",
        lambda frac, dev: captured.update(fraction=frac, device=dev),
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", _fake_device_properties(24.0))
    cfg = OmegaConf.create({"data": {"z_train_source": "tabicl_split"}})
    t = OmegaConf.create({"live_tabicl_num_workers": 50, "batch_size": 50})
    train._reserve_gpu_headroom_for_live_tabicl(cfg, t, "cuda")
    assert captured["fraction"] == pytest.approx(0.5)
