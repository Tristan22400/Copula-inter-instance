"""test_era5_marginal_backend.py — the real-ERA5 finetune path's marginal
backend selection (era5_live_dataset.py::_resolve_marginal) and its PIT
dispatch (_pit_group / _pit_episode).

Before this, the ERA5 path was structurally TabICL-only: _pit_group called
pit.py::run_pit_batched directly, so data.z_train_source had no effect there
at all -- an ERA5 finetune launched with z_train_source=exaone silently
trained against TabICL's marginal instead. These tests pin down both halves
of the fix: that the knob is READ (and that TabICL-ish values still mean
TabICL), and that a selected backend actually reaches the PIT.

The corpus-dependent parts of the module are not exercised here -- these
call the PIT helpers directly with synthetic tensors, so no ERA5 download
is needed.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
from omegaconf import OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("analytic", None),      # no generating GP on real data -> stay on TabICL
        ("tabicl", None),
        ("tabicl_split", None),
        ("exaone", "exaone"),
        ("tabpfn", "tabpfn"),
        ("tabldm", "tabldm"),
    ],
)
def test_resolve_marginal_maps_z_train_source(source, expected):
    from era5_live_dataset import _resolve_marginal

    cfg = OmegaConf.create({"data": {"z_train_source": source}})
    backend, probs_n = _resolve_marginal(cfg)
    assert backend == expected
    assert probs_n == 99


def test_resolve_marginal_reads_probs_n():
    from era5_live_dataset import _resolve_marginal

    cfg = OmegaConf.create({"data": {"z_train_source": "tabldm", "z_train_marginal_probs_n": 33}})
    assert _resolve_marginal(cfg) == ("tabldm", 33)


def test_resolve_marginal_rejects_typo():
    """The z_train_source typo class that silently no-opped the synthetic
    path (see tests/test_z_train_source_validation.py) must not be able to
    silently no-op this one either."""
    from era5_live_dataset import _resolve_marginal

    cfg = OmegaConf.create({"data": {"z_train_source": "tabicl-split"}})
    with pytest.raises(ValueError, match="Unknown data.z_train_source"):
        _resolve_marginal(cfg)


@pytest.mark.parametrize("backend", ["tabldm"])
def test_pit_group_and_episode_agree_under_backend(backend):
    """A selected backend reaches the PIT, and the grouped path agrees with
    the single-episode one it shares a batched module with."""
    pytest.importorskip(backend, reason=f"{backend} not installed")
    from era5_live_dataset import _pit_episode, _pit_group
    from eval.spatial.marginal_backends import make_regressor

    regressor = make_regressor(backend, device="cpu")
    torch.manual_seed(0)
    B, P, N, p_x = 2, 8, 3, 3
    x_train, x_test = torch.randn(B, P, p_x), torch.randn(B, N, p_x)
    y_train, y_test = torch.randn(B, P) * 2 + 1, torch.randn(B, N) * 2 + 1

    kw = dict(
        marginal_backend=backend, marginal_regressor=regressor,
        marginal_probs_n=9, seed=7,
    )
    grouped = _pit_group(x_train, y_train, x_test, y_test, None, 2, **kw)
    assert grouped["z_train"].shape == (B, P)
    assert grouped["z_test"].shape == (B, N)
    assert grouped["log_pdf_test"].shape == (B, N)
    assert all(torch.isfinite(v).all() for v in grouped.values())

    single = _pit_episode(x_train[0], y_train[0], x_test[0], y_test[0], None, 2, **kw)
    torch.testing.assert_close(grouped["z_train"][0], single["z_train"], atol=1e-3, rtol=0)
    torch.testing.assert_close(grouped["log_pdf_test"][0], single["log_pdf_test"], atol=1e-2, rtol=0)
