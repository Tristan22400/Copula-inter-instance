"""
test_era5_val_probe.py — Sanity checks for the real-ERA5 validation-loop
probes added to train.py (_build_era5_val_batches, era5_fit/<region>/*
validate() metrics) and their shared scoring helpers promoted to public in
eval/spatial/sweep_core.py (weighted_corr/weighted_rmse_bias/weighted_r2,
build_era5_probe).

FakeTabICL mirrors test_tabicl_z_diagnostic.py's stand-in: it only needs to
satisfy run_pit's interface (forward(X, y) -> logits,
.quantile_dist(logits) -> a distribution with .cdf/.log_prob), and being a
fixed-seed pure function of its input shapes, is fully deterministic --
which build_era5_probe's "compute z_train once, cache" design relies on
(mirrors _build_tabicl_val_z's identical rationale for the synthetic-episode
case).

These tests fetch a TINY real ERA5 grid (grid_size=4, n_days_fetch=2) from
the public, no-auth ARCO-ERA5 archive on GCS on first run (network
required) and cache it under eval/data/cache/ (gitignored) for every
subsequent run -- the same auto-fetch/cache behavior every diagnose/sweep
CLI command in eval/runners/spatial_correlation_eval.py already relies on.

Tests verify:
  1. weighted_corr/weighted_rmse_bias/weighted_r2 (promoted from private
     helpers) still score identical curves as a perfect fit, and still
     return NaN (not raise) when too few bins are populated.
  2. build_era5_probe returns correctly-shaped, finite frozen probe data,
     is deterministic across repeated calls with the same seed (the
     "compute once, cache" contract _build_era5_val_batches relies on), and
     falls back to naive per-context standardization when tabicl_marginal
     is None.
  3. train.py::_build_era5_val_batches assembles those probes into
     model-forward-ready batches (x_train/x_test/z_train/test_mask), and
     silently skips region names not in eval/configs/regions.py.
  4. The era5_fit/<region>/* scoring block validate() runs (model forward
     -> build_sigma -> bin_correlation_by_distance -> weighted_*) produces
     a valid unit-diagonal correlation matrix and finite rmse/bias when run
     against a real (tiny, scratch-initialized) CopulaTabICL model.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

_TESTS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eval.spatial.diagnostics import bin_correlation_by_distance  # noqa: E402
from eval.spatial.sweep_core import (  # noqa: E402
    build_era5_probe,
    weighted_corr,
    weighted_r2,
    weighted_rmse_bias,
)
from loss import y_space_nll  # noqa: E402
from model import build_copula_transformer, build_sigma  # noqa: E402
from train import _build_era5_val_batches  # noqa: E402

_TINY_REGION = "western_europe"
_TINY_GRID = 4
_TINY_DAYS_FETCH = 2
_TINY_DAYS_PROBE = 1
_TINY_CONTEXT = 5
_TINY_BINS = 4


class FakeTabICL(nn.Module):
    """Minimal stand-in satisfying pit.py::run_pit's interface: forward(X, y)
    -> logits (d, N, Q) for the N query rows after the P context rows in X;
    quantile_dist(logits) -> a distribution with .cdf/.log_prob. Seeded
    per-call so it's a deterministic pure function of (X, y)'s shapes/values,
    matching a real frozen, eval-mode model's determinism (see
    test_tabicl_z_diagnostic.py's identical fake)."""

    def __init__(self, q: int = 2):
        super().__init__()
        self.q = q

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d, T, _ = X.shape
        P = y.shape[1]
        n = T - P
        g = torch.Generator().manual_seed(int(X.sum().item() * 1000) % 2**31)
        return torch.randn(d, n, self.q, generator=g)

    def quantile_dist(self, logits_flat: torch.Tensor):
        loc = logits_flat[:, 0]
        scale = torch.nn.functional.softplus(logits_flat[:, 1]) + 1e-3
        return torch.distributions.Normal(loc, scale)


@pytest.fixture(scope="module")
def tabicl_fake():
    return FakeTabICL()


# ---------------------------------------------------------------------------
# weighted_corr / weighted_rmse_bias / weighted_r2
# ---------------------------------------------------------------------------
def test_weighted_corr_identical_curves_is_one():
    rho = np.array([0.9, 0.5, 0.2, 0.05])
    w = np.array([10.0, 8.0, 5.0, 2.0])
    assert weighted_corr(rho, rho, w) == pytest.approx(1.0)


def test_weighted_rmse_bias_zero_for_identical_curves():
    rho = np.array([0.9, 0.5, 0.2, 0.05])
    w = np.array([10.0, 8.0, 5.0, 2.0])
    rmse, bias = weighted_rmse_bias(rho, rho, w)
    assert rmse == pytest.approx(0.0, abs=1e-8)
    assert bias == pytest.approx(0.0, abs=1e-8)


def test_weighted_r2_perfect_fit_is_one():
    rho = np.array([0.9, 0.5, 0.2, 0.05])
    w = np.array([10.0, 8.0, 5.0, 2.0])
    assert weighted_r2(rho, rho, w) == pytest.approx(1.0)


def test_weighted_corr_nan_with_too_few_valid_points():
    a = np.array([0.9, np.nan, np.nan, np.nan])
    b = np.array([0.9, 0.5, 0.2, 0.05])
    w = np.array([10.0, 8.0, 5.0, 2.0])
    assert np.isnan(weighted_corr(a, b, w))


# ---------------------------------------------------------------------------
# build_era5_probe (real, tiny fetch)
# ---------------------------------------------------------------------------
def test_build_era5_probe_shapes_and_finite(tabicl_fake):
    probe = build_era5_probe(
        _TINY_REGION, _TINY_GRID, _TINY_DAYS_FETCH, _TINY_DAYS_PROBE,
        _TINY_CONTEXT, _TINY_BINS, tabicl_fake, "cpu", seed=123,
    )
    D = _TINY_GRID * _TINY_GRID
    n_context = min(_TINY_CONTEXT, D - 1)

    assert probe["D"] == D
    assert probe["n_context"] == n_context
    assert probe["x_train_norm"].shape == (n_context, 2)
    assert probe["x_test_norm"].shape == (D, 2)
    assert probe["z_train_per_day"].shape == (_TINY_DAYS_PROBE, n_context)
    assert probe["dist"].shape == (D, D)
    assert probe["rho_emp"].shape == (_TINY_BINS,)
    assert probe["pair_counts"].shape == (_TINY_BINS,)
    assert np.isfinite(probe["x_train_norm"]).all()
    assert np.isfinite(probe["z_train_per_day"]).all()
    # The tiny 4x4 grid still yields 120 upper-triangle pairs over 4 bins,
    # so the nearest-neighbor bin must be populated.
    assert probe["pair_counts"][0] > 0
    assert np.isfinite(probe["rho_emp"][0])


def test_build_era5_probe_deterministic(tabicl_fake):
    """The whole point of precomputing this once (see build_era5_probe's
    docstring): a frozen (context, day) sample must give the same probe
    every time, or caching it in _build_era5_val_batches would silently go
    stale."""
    p1 = build_era5_probe(
        _TINY_REGION, _TINY_GRID, _TINY_DAYS_FETCH, _TINY_DAYS_PROBE,
        _TINY_CONTEXT, _TINY_BINS, tabicl_fake, "cpu", seed=99,
    )
    p2 = build_era5_probe(
        _TINY_REGION, _TINY_GRID, _TINY_DAYS_FETCH, _TINY_DAYS_PROBE,
        _TINY_CONTEXT, _TINY_BINS, tabicl_fake, "cpu", seed=99,
    )
    np.testing.assert_array_equal(p1["z_train_per_day"], p2["z_train_per_day"])
    np.testing.assert_array_equal(p1["rho_emp"], p2["rho_emp"])
    np.testing.assert_array_equal(p1["x_train_norm"], p2["x_train_norm"])


def test_build_era5_probe_none_marginal_uses_naive_standardization():
    probe = build_era5_probe(
        _TINY_REGION, _TINY_GRID, _TINY_DAYS_FETCH, _TINY_DAYS_PROBE,
        _TINY_CONTEXT, _TINY_BINS, None, "cpu", seed=7,
    )
    z = probe["z_train_per_day"][0]
    assert np.isfinite(z).all()
    assert z.mean() == pytest.approx(0.0, abs=1e-6)
    assert z.std() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# train.py::_build_era5_val_batches
# ---------------------------------------------------------------------------
def _tiny_era5_cfg(seed: int = 555) -> "OmegaConf":
    return OmegaConf.create({
        "baselines": {
            "era5_regions": [_TINY_REGION],
            "era5_grid_size": _TINY_GRID,
            "era5_n_days_fetch": _TINY_DAYS_FETCH,
            "era5_n_days_probe": _TINY_DAYS_PROBE,
            "era5_n_context": _TINY_CONTEXT,
            "era5_n_bins": _TINY_BINS,
            "era5_seed": seed,
        },
        # _build_era5_val_batches reads tabicl.pit_k_folds (mirrors
        # conf/model/copula_prod.yaml's real `tabicl:` section) to run
        # TabICL's own PIT on the probe's held-out points -- see
        # test_build_era5_val_batches_shapes' nll_test_z/log_pdf assertions.
        "tabicl": {"pit_k_folds": 5},
    })


def test_build_era5_val_batches_shapes(tabicl_fake):
    batches = _build_era5_val_batches(_tiny_era5_cfg(), tabicl_fake, "cpu")
    assert set(batches.keys()) == {_TINY_REGION}

    probe = batches[_TINY_REGION]
    D = _TINY_GRID * _TINY_GRID
    n_context = min(_TINY_CONTEXT, D - 1)
    n_nll = min(30, D - n_context)  # eval.configs.constants.N_NLL_TEST, capped
    b = probe["batch"]
    assert b["x_train"].shape == (_TINY_DAYS_PROBE, n_context, 2)
    assert b["x_test"].shape == (_TINY_DAYS_PROBE, D, 2)
    assert b["z_train"].shape == (_TINY_DAYS_PROBE, n_context)
    assert b["test_mask"].shape == (_TINY_DAYS_PROBE, D)
    assert b["test_mask"].dtype == torch.bool
    assert bool(b["test_mask"].all())
    assert probe["dist"].shape == (D, D)
    assert probe["rho_emp"].shape == (_TINY_BINS,)
    assert probe["pair_counts"].shape == (_TINY_BINS,)

    # Real, non-oracle Y-space NLL ingredients (validate()'s
    # era5_fit/<region>/y_nll_total/marginal/copula) -- only present when
    # tabicl_marginal is not None (see _build_era5_val_batches' docstring).
    assert probe["nll_test_idx"].shape == (n_nll,)
    assert probe["nll_test_idx"].max() < D  # indices into the D-point grid
    assert probe["nll_test_z"].shape == (_TINY_DAYS_PROBE, n_nll)
    assert probe["nll_test_log_pdf"].shape == (_TINY_DAYS_PROBE, n_nll)
    assert torch.isfinite(probe["nll_test_z"]).all()
    assert torch.isfinite(probe["nll_test_log_pdf"]).all()


def test_build_era5_val_batches_none_marginal_skips_nll():
    """No PIT checkpoint configured -> no real predictive density to score a
    Y-space NLL against, so the probe carries no nll_test_* keys (validate()
    guards on this to skip era5_fit/<region>/y_nll_total for every region)."""
    batches = _build_era5_val_batches(_tiny_era5_cfg(), None, "cpu")
    probe = batches[_TINY_REGION]
    assert "nll_test_z" not in probe
    assert "nll_test_log_pdf" not in probe
    assert "nll_test_idx" not in probe


def test_build_era5_val_batches_skips_unregistered_region(tabicl_fake):
    cfg = OmegaConf.create({
        "baselines": {"era5_regions": ["not_a_real_region"]},
        "tabicl": {"pit_k_folds": 5},
    })
    assert _build_era5_val_batches(cfg, tabicl_fake, "cpu") == {}


# ---------------------------------------------------------------------------
# era5_fit/<region>/* scoring block (validate()'s new logic), run against a
# real (scratch, tiny) CopulaTabICL model
# ---------------------------------------------------------------------------
def test_era5_fit_scoring_with_tiny_model(small_model_cfg, tabicl_fake):
    torch.manual_seed(0)
    model = build_copula_transformer(small_model_cfg)
    # validate() deliberately never calls model.eval() (see its docstring) --
    # mirror that here rather than the more common eval()-mode test pattern.

    cfg = OmegaConf.merge(
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
    assert math.isfinite(bias)
    # shape_corr/model_r2 need >=3 populated bins; with 120 pairs over 4 bins
    # on this tiny grid that's expected, but assert boundedness rather than
    # an exact value -- the scoring formula's correctness is already covered
    # by the weighted_* unit tests above.
    if not math.isnan(shape_corr):
        assert -1.0 - 1e-6 <= shape_corr <= 1.0 + 1e-6

    # Real, non-oracle Y-space NLL block (validate()'s
    # era5_fit/<region>/y_nll_total/marginal/copula): reuses the SAME Sigma
    # forward pass above, sliced to nll_test_idx, scored against the frozen
    # TabICL PIT probe["nll_test_z"]/["nll_test_log_pdf"].
    idx = torch.as_tensor(probe["nll_test_idx"], dtype=torch.long)
    Sigma_nll = Sigma.index_select(1, idx).index_select(2, idx)
    z_nll, log_pdf_nll = probe["nll_test_z"], probe["nll_test_log_pdf"]
    mask_nll = torch.ones_like(z_nll, dtype=torch.bool)
    parts = y_space_nll(Sigma_nll, z_nll, log_pdf_nll, mask_nll)

    assert math.isfinite(parts["total"].item())
    assert math.isfinite(parts["marginal"].item())
    assert math.isfinite(parts["copula"].item())
    assert parts["total"].item() == pytest.approx(
        parts["marginal"].item() + parts["copula"].item(), abs=1e-3
    )
