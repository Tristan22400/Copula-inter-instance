"""Tests for the size_fit/* validation metrics: the copula scored on a fixed (P, N) grid against the GP posterior."""

from __future__ import annotations

import math

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from copula_inter.copula.model import build_copula_transformer
from copula_inter.training.probe_batches import _build_size_grid_batches, size_grid_key
from copula_inter.training.validation import _size_fit_figure, _size_fit_metrics, validate

P_GRID = [4, 12]
N_GRID = [3, 8]


def _cfg(small_cfg: DictConfig, small_model_cfg: DictConfig) -> DictConfig:
    """small_cfg's data block with a scratch backbone, an rbf prior and a 2 x 2 size grid."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    assert isinstance(cfg, DictConfig)
    cfg.model = OmegaConf.create(OmegaConf.to_container(small_model_cfg.model, resolve=True))
    cfg.tabicl = OmegaConf.create(OmegaConf.to_container(small_model_cfg.tabicl, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.baselines = OmegaConf.create(
        {"size_grid_P": P_GRID, "size_grid_N": N_GRID, "size_grid_n_episodes": 3, "size_grid_batch_size": 2}
    )
    return cfg


@pytest.fixture
def cfg(small_cfg: DictConfig, small_model_cfg: DictConfig) -> DictConfig:
    return _cfg(small_cfg, small_model_cfg)


def test_grid_cells_have_exact_sizes_and_a_finite_ceiling(cfg: DictConfig) -> None:
    """Every cell holds exactly P context and N test points, whatever cfg.data's own ranges are."""
    cells = _build_size_grid_batches(cfg, "cpu")
    assert set(cells) == {(P, N) for P in P_GRID for N in N_GRID}
    for (P, N), cell in cells.items():
        batch = cell["batch"]
        assert batch["x_train"].shape[:2] == (3, P)
        assert batch["x_test"].shape[:2] == (3, N)
        assert bool(batch["train_mask"].all()) and bool(batch["test_mask"].all())
        assert np.isfinite(cell["oracle_total"]).all()
        assert np.isfinite(cell["oracle_copula"]).all()


def test_grid_is_deterministic(cfg: DictConfig) -> None:
    """Fixed per-cell seeds: two builds give the same episodes."""
    a = _build_size_grid_batches(cfg, "cpu")
    b = _build_size_grid_batches(cfg, "cpu")
    for pn in a:
        assert bool((a[pn]["batch"]["y_test"] == b[pn]["batch"]["y_test"]).all())


def test_size_fit_metrics_are_consistent(cfg: DictConfig) -> None:
    """Per cell only gap_nll and copula_capture; per-P values average the cells; spread = max - min."""
    model = build_copula_transformer(cfg)
    cells = _build_size_grid_batches(cfg, "cpu")
    m = _size_fit_metrics(model, cfg, 1e-4, cells)
    for P, N in cells:
        k = f"size_fit/{size_grid_key(P, N)}"
        assert {name.removeprefix(k + "/") for name in m if name.startswith(k + "/")} == {"gap_nll", "copula_capture"}
        assert math.isfinite(m[f"{k}/gap_nll"])
    per_P = [m[f"size_fit/P{P}/gap_nll"] for P in P_GRID]
    for P, value in zip(P_GRID, per_P):
        assert value == pytest.approx(np.mean([m[f"size_fit/P{P}_N{N}/gap_nll"] for N in N_GRID]))
    assert m["size_fit/P_spread_gap_nll"] == pytest.approx(max(per_P) - min(per_P))
    assert _size_fit_figure(m, cells) is not None


def test_validate_reports_size_fit_only_when_given_the_grid(cfg: DictConfig) -> None:
    """validate() adds size_fit/* and its figure from size_grid_batches, and nothing without them."""
    model = build_copula_transformer(cfg)
    cells = _build_size_grid_batches(cfg, "cpu")
    with_grid, figs = validate(model, [], cfg, "cpu", do_plot=True, size_grid_batches=cells)
    assert "size_fit/mean_gap_nll" in with_grid
    assert "val/size_fit_grid" in figs
    without, _ = validate(model, [], cfg, "cpu", do_plot=True)
    assert not any(k.startswith("size_fit/") for k in without)
