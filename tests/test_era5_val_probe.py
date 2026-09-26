"""Tests for the real-ERA5 validation probes (_build_era5_val_batches, the era5_fit/<region> scoring) and sweep_core's weighted_* helpers.

Fetches a tiny ERA5 grid (4 x 4, 2 days) from the public ARCO-ERA5 archive on
first run (network) and caches it under eval/data/cache/.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig


_TESTS = os.path.dirname(os.path.abspath(__file__))

from copula_inter.config_path import merge_configs
from copula_inter.era5_probes import _build_era5_val_batches  # noqa: E402
from copula_inter.loss import y_space_nll  # noqa: E402
from copula_inter.model import build_copula_transformer, build_sigma  # noqa: E402
from eval.spatial.diagnostics import bin_correlation_by_distance  # noqa: E402
from eval.spatial.sweep_core import (  # noqa: E402
    build_era5_probe,
    weighted_corr,
    weighted_r2,
    weighted_rmse_bias,
)

_TINY_REGION = "western_europe"
_TINY_GRID = 4
_TINY_DAYS_FETCH = 2
_TINY_DAYS_PROBE = 1
_TINY_CONTEXT = 5
_TINY_BINS = 4


class FakeTabICL(nn.Module):
    """Deterministic stand-in for run_pit's interface: forward(X, y) -> logits, quantile_dist(logits) -> distribution with cdf/log_prob."""

    def __init__(self, q: int = 2) -> None:
        super().__init__()
        self.q = q

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d, T, _ = X.shape
        P = y.shape[1]
        n = T - P
        g = torch.Generator().manual_seed(int(X.sum().item() * 1000) % 2**31)
        return torch.randn(d, n, self.q, generator=g)

    def quantile_dist(self, logits_flat: torch.Tensor) -> torch.distributions.Normal:
        loc = logits_flat[:, 0]
        scale = torch.nn.functional.softplus(logits_flat[:, 1]) + 1e-3
        return torch.distributions.Normal(loc, scale)


@pytest.fixture(scope="module")
def tabicl_fake() -> FakeTabICL:
    return FakeTabICL()


def test_weighted_corr_identical_curves_is_one() -> None:
    rho = np.array([0.9, 0.5, 0.2, 0.05])
    w = np.array([10.0, 8.0, 5.0, 2.0])
    assert weighted_corr(rho, rho, w) == pytest.approx(1.0)


def test_weighted_rmse_bias_zero_for_identical_curves() -> None:
    rho = np.array([0.9, 0.5, 0.2, 0.05])
    w = np.array([10.0, 8.0, 5.0, 2.0])
    rmse, bias = weighted_rmse_bias(rho, rho, w)
    assert rmse == pytest.approx(0.0, abs=1e-8)
    assert bias == pytest.approx(0.0, abs=1e-8)


def test_weighted_r2_perfect_fit_is_one() -> None:
    rho = np.array([0.9, 0.5, 0.2, 0.05])
    w = np.array([10.0, 8.0, 5.0, 2.0])
    assert weighted_r2(rho, rho, w) == pytest.approx(1.0)


def test_weighted_corr_nan_with_too_few_valid_points() -> None:
    a = np.array([0.9, np.nan, np.nan, np.nan])
    b = np.array([0.9, 0.5, 0.2, 0.05])
    w = np.array([10.0, 8.0, 5.0, 2.0])
    assert np.isnan(weighted_corr(a, b, w))


def test_build_era5_probe_shapes_and_finite(tabicl_fake: FakeTabICL) -> None:
    probe = build_era5_probe(
        _TINY_REGION,
        _TINY_GRID,
        _TINY_DAYS_FETCH,
        _TINY_DAYS_PROBE,
        _TINY_CONTEXT,
        _TINY_BINS,
        tabicl_fake,
        "cpu",
        seed=123,
    )
    D = _TINY_GRID * _TINY_GRID
    n_context = min(_TINY_CONTEXT, D - 1)

    assert probe["D"] == D
    assert probe["n_context"] == n_context
    assert probe["x_train_norm"].shape == (n_context, 6)
    assert probe["x_test_norm"].shape == (D, 6)
    assert probe["z_train_per_day"].shape == (_TINY_DAYS_PROBE, n_context)
    assert probe["dist"].shape == (D, D)
    assert probe["rho_emp"].shape == (_TINY_BINS,)
    assert probe["pair_counts"].shape == (_TINY_BINS,)
    assert np.isfinite(probe["x_train_norm"]).all()
    assert np.isfinite(probe["z_train_per_day"]).all()
    # 120 pairs over 4 bins: the nearest bin is populated.
    assert probe["pair_counts"][0] > 0
    assert np.isfinite(probe["rho_emp"][0])


def test_build_era5_probe_deterministic(tabicl_fake: FakeTabICL) -> None:
    """build_era5_probe gives the same probe for the same seed."""
    p1 = build_era5_probe(
        _TINY_REGION,
        _TINY_GRID,
        _TINY_DAYS_FETCH,
        _TINY_DAYS_PROBE,
        _TINY_CONTEXT,
        _TINY_BINS,
        tabicl_fake,
        "cpu",
        seed=99,
    )
    p2 = build_era5_probe(
        _TINY_REGION,
        _TINY_GRID,
        _TINY_DAYS_FETCH,
        _TINY_DAYS_PROBE,
        _TINY_CONTEXT,
        _TINY_BINS,
        tabicl_fake,
        "cpu",
        seed=99,
    )
    np.testing.assert_array_equal(p1["z_train_per_day"], p2["z_train_per_day"])
    np.testing.assert_array_equal(p1["rho_emp"], p2["rho_emp"])
    np.testing.assert_array_equal(p1["x_train_norm"], p2["x_train_norm"])


def test_build_era5_probe_none_marginal_uses_naive_standardization() -> None:
    probe = build_era5_probe(
        _TINY_REGION,
        _TINY_GRID,
        _TINY_DAYS_FETCH,
        _TINY_DAYS_PROBE,
        _TINY_CONTEXT,
        _TINY_BINS,
        None,
        "cpu",
        seed=7,
    )
    z = probe["z_train_per_day"][0]
    assert np.isfinite(z).all()
    assert z.mean() == pytest.approx(0.0, abs=1e-6)
    assert z.std() == pytest.approx(1.0, abs=1e-6)


def _tiny_era5_cfg(seed: int = 555) -> DictConfig:
    return OmegaConf.create(
        {
            "baselines": {
                "era5_regions": [_TINY_REGION],
                "era5_grid_size": _TINY_GRID,
                "era5_n_days_fetch": _TINY_DAYS_FETCH,
                "era5_n_days_probe": _TINY_DAYS_PROBE,
                "era5_n_context": _TINY_CONTEXT,
                "era5_n_bins": _TINY_BINS,
                "era5_seed": seed,
                # GP baseline off here for speed (tested separately below).
                "era5_gp_baseline": False,
            },
            # tabicl.pit_k_folds, as in copula_prod.yaml.
            "tabicl": {"pit_k_folds": 5},
        }
    )


def test_build_era5_val_batches_shapes(tabicl_fake: FakeTabICL) -> None:
    batches = _build_era5_val_batches(_tiny_era5_cfg(), tabicl_fake, "cpu")
    assert set(batches.keys()) == {_TINY_REGION}

    probe = batches[_TINY_REGION]
    D = _TINY_GRID * _TINY_GRID
    n_context = min(_TINY_CONTEXT, D - 1)
    n_nll = min(30, D - n_context)  # eval.configs.constants.N_NLL_TEST, capped
    b = probe["batch"]
    assert b["x_train"].shape == (_TINY_DAYS_PROBE, n_context, 6)
    assert b["x_test"].shape == (_TINY_DAYS_PROBE, D, 6)
    assert b["z_train"].shape == (_TINY_DAYS_PROBE, n_context)
    assert b["test_mask"].shape == (_TINY_DAYS_PROBE, D)
    assert b["test_mask"].dtype == torch.bool
    assert bool(b["test_mask"].all())
    assert probe["dist"].shape == (D, D)
    assert probe["rho_emp"].shape == (_TINY_BINS,)
    assert probe["pair_counts"].shape == (_TINY_BINS,)

    # Y-space NLL inputs, present with a marginal.
    assert probe["nll_test_idx"].shape == (n_nll,)
    assert probe["nll_test_idx"].max() < D  # indices into the D-point grid
    assert probe["nll_test_z"].shape == (_TINY_DAYS_PROBE, n_nll)
    assert probe["nll_test_log_pdf"].shape == (_TINY_DAYS_PROBE, n_nll)
    assert torch.isfinite(probe["nll_test_z"]).all()
    assert torch.isfinite(probe["nll_test_log_pdf"]).all()


def test_build_era5_val_batches_none_marginal_skips_nll() -> None:
    """Without a marginal the probe has no nll_test_* keys."""
    batches = _build_era5_val_batches(_tiny_era5_cfg(), None, "cpu")
    probe = batches[_TINY_REGION]
    assert "nll_test_z" not in probe
    assert "nll_test_log_pdf" not in probe
    assert "nll_test_idx" not in probe


def test_build_era5_val_batches_gp_baseline(tabicl_fake: FakeTabICL) -> None:
    """era5_gp_baseline=True adds a GP-MLE baseline NLL per kernel to each probe (tiny settings)."""
    cfg = _tiny_era5_cfg()
    cfg.baselines.era5_gp_baseline = True
    cfg.baselines.era5_gp_baseline_kernels = ["rbf"]
    cfg.baselines.era5_gp_n_steps_mle = 20
    cfg.baselines.era5_gp_n_restarts_mle = 1
    cfg.baselines.era5_gp_baseline_n_days = 1

    batches = _build_era5_val_batches(cfg, tabicl_fake, "cpu")
    probe = batches[_TINY_REGION]
    assert "gp_baseline_nll" in probe
    assert set(probe["gp_baseline_nll"].keys()) == {"rbf"}
    parts = probe["gp_baseline_nll"]["rbf"]
    assert set(parts.keys()) == {"total", "marginal", "copula"}
    for v in parts.values():
        assert math.isfinite(v)
    assert parts["total"] == pytest.approx(parts["marginal"] + parts["copula"], abs=1e-3)


def test_build_era5_val_batches_gp_baseline_disabled_by_default_cfg(tabicl_fake: FakeTabICL) -> None:
    """era5_gp_baseline defaults to True when the key is absent."""
    cfg = OmegaConf.create(
        {
            "baselines": {
                "era5_regions": [_TINY_REGION],
                "era5_grid_size": _TINY_GRID,
                "era5_n_days_fetch": _TINY_DAYS_FETCH,
                "era5_n_days_probe": _TINY_DAYS_PROBE,
                "era5_n_context": _TINY_CONTEXT,
                "era5_n_bins": _TINY_BINS,
                "era5_gp_baseline_kernels": ["rbf"],
                "era5_gp_n_steps_mle": 20,
                "era5_gp_n_restarts_mle": 1,
            },
            "tabicl": {"pit_k_folds": 5},
        }
    )
    batches = _build_era5_val_batches(cfg, tabicl_fake, "cpu")
    assert "gp_baseline_nll" in batches[_TINY_REGION]


def test_build_era5_val_batches_skips_unregistered_region(tabicl_fake: FakeTabICL) -> None:
    cfg = OmegaConf.create(
        {
            "baselines": {"era5_regions": ["not_a_real_region"]},
            "tabicl": {"pit_k_folds": 5},
        }
    )
    assert _build_era5_val_batches(cfg, tabicl_fake, "cpu") == {}


def test_era5_fit_scoring_with_tiny_model(small_model_cfg: DictConfig, tabicl_fake: FakeTabICL) -> None:
    torch.manual_seed(0)
    model = build_copula_transformer(small_model_cfg)
    # No model.eval(), as in validate().

    cfg = merge_configs(
        small_model_cfg,
        OmegaConf.create({"model": {"sigma_jitter": 1e-4}}),
    )
    era5_val_batches = _build_era5_val_batches(_tiny_era5_cfg(seed=321), tabicl_fake, "cpu")
    probe = era5_val_batches[_TINY_REGION]

    with torch.no_grad():
        out = model(probe["batch"])
    Sigma = build_sigma(out, cfg, jitter=1e-4, test_mask=probe["batch"]["test_mask"])

    D = _TINY_GRID * _TINY_GRID
    assert Sigma.shape == (_TINY_DAYS_PROBE, D, D)

    R_mean = Sigma.float().mean(dim=0).detach().cpu().numpy()
    # low_rank_correlation's own contract: unit-diagonal correlation matrix.
    np.testing.assert_allclose(np.diagonal(R_mean), 1.0, atol=1e-3)

    rho_context = bin_correlation_by_distance(R_mean, probe["dist"], probe["bin_edges"])
    shape_corr = weighted_corr(rho_context, probe["rho_emp"], probe["pair_counts"])
    rmse, bias = weighted_rmse_bias(rho_context, probe["rho_emp"], probe["pair_counts"])
    model_r2 = weighted_r2(rho_context, probe["rho_emp"], probe["pair_counts"])

    assert math.isfinite(rmse)
    assert math.isfinite(model_r2)
    assert math.isfinite(bias)
    # Bounded shape_corr/model_r2 (formulas are tested above).
    if not math.isnan(shape_corr):
        assert -1.0 - 1e-6 <= shape_corr <= 1.0 + 1e-6

    # Y-space NLL from the same forward, on the held-out points.
    idx = torch.as_tensor(probe["nll_test_idx"], dtype=torch.long)
    Sigma_nll = Sigma.index_select(1, idx).index_select(2, idx)
    z_nll, log_pdf_nll = probe["nll_test_z"], probe["nll_test_log_pdf"]
    mask_nll = torch.ones_like(z_nll, dtype=torch.bool)
    parts = y_space_nll(Sigma_nll, z_nll, log_pdf_nll, mask_nll)

    assert math.isfinite(parts["total"].item())
    assert math.isfinite(parts["marginal"].item())
    assert math.isfinite(parts["copula"].item())
    assert parts["total"].item() == pytest.approx(parts["marginal"].item() + parts["copula"].item(), abs=1e-3)
