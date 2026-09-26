"""Tests for the ERA5 path's marginal selection (_resolve_marginal) and PIT dispatch (_pit_group, _pit_episode), on synthetic tensors."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf


@pytest.mark.parametrize(
    "source,expected",
    [
        ("analytic", None),  # no generating GP on real data -> stay on TabICL
        ("tabicl", None),
        ("tabicl_split", None),
        ("exaone", "exaone"),
        ("tabpfn", "tabpfn"),
        ("tabldm", "tabldm"),
    ],
)
def test_resolve_marginal_maps_z_train_source(source, expected):
    from copula_inter.era5_live_dataset import _resolve_marginal

    cfg = OmegaConf.create({"data": {"z_train_source": source}})
    backend, probs_n = _resolve_marginal(cfg)
    assert backend == expected
    assert probs_n == 99


def test_resolve_marginal_reads_probs_n():
    from copula_inter.era5_live_dataset import _resolve_marginal

    cfg = OmegaConf.create({"data": {"z_train_source": "tabldm", "z_train_marginal_probs_n": 33}})
    assert _resolve_marginal(cfg) == ("tabldm", 33)


def test_resolve_marginal_rejects_typo():
    """_resolve_marginal rejects unknown z_train_source values."""
    from copula_inter.era5_live_dataset import _resolve_marginal

    cfg = OmegaConf.create({"data": {"z_train_source": "tabicl-split"}})
    with pytest.raises(ValueError, match="Unknown data.z_train_source"):
        _resolve_marginal(cfg)


@pytest.mark.parametrize("backend", ["tabldm"])
def test_pit_group_and_episode_agree_under_backend(backend):
    """A backend reaches the PIT, and the grouped and single-episode paths agree."""
    pytest.importorskip(backend, reason=f"{backend} not installed")
    from copula_inter.era5_live_dataset import _pit_episode, _pit_group
    from eval.spatial.marginal_backends import make_regressor

    regressor = make_regressor(backend, device="cpu")
    torch.manual_seed(0)
    B, P, N, p_x = 2, 8, 3, 3
    x_train, x_test = torch.randn(B, P, p_x), torch.randn(B, N, p_x)
    y_train, y_test = torch.randn(B, P) * 2 + 1, torch.randn(B, N) * 2 + 1

    kw = dict(
        marginal_backend=backend,
        marginal_regressor=regressor,
        marginal_probs_n=9,
        seed=7,
    )
    grouped = _pit_group(x_train, y_train, x_test, y_test, None, 2, **kw)
    assert grouped["z_train"].shape == (B, P)
    assert grouped["z_test"].shape == (B, N)
    assert grouped["log_pdf_test"].shape == (B, N)
    assert all(torch.isfinite(v).all() for v in grouped.values())

    single = _pit_episode(x_train[0], y_train[0], x_test[0], y_test[0], None, 2, **kw)
    torch.testing.assert_close(grouped["z_train"][0], single["z_train"], atol=1e-3, rtol=0)
    torch.testing.assert_close(grouped["log_pdf_test"][0], single["log_pdf_test"], atol=1e-2, rtol=0)
