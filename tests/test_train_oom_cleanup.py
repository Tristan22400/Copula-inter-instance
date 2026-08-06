"""Regression tests for training-step CUDA OOM cleanup."""

from __future__ import annotations

import gc
import traceback
import weakref

import pytest
import torch
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
