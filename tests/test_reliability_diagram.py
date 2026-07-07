"""
test_reliability_diagram.py — Sanity checks for the quantile reliability
diagram code in plots/generate_plots.py.

compute_quantile_ece is exercised on synthetic data with a controlled
miscalibration bias (ECE should track the injected bias, and be ~0 for a
perfectly calibrated forecaster).

plot_era5_quantile_reliability is exercised with a fake TabICLv2 regressor to
regression-test two properties that are easy to silently break:
  - context_idx locations must be excluded from the evaluated (y_true,
    y_pred_quantiles) set (querying the model at its own context points would
    let it condition on the true label it's scored against, inflating the
    apparent coverage).
  - all quantile levels for a given day come from a single fit()+predict()
    call, not one fit() per quantile level (TabICL's quantile spline comes
    from one backbone forward pass regardless of how many alphas are
    requested).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from scipy.stats import norm

_TESTS = os.path.dirname(os.path.abspath(__file__))
_PLOTS = os.path.join(os.path.dirname(_TESTS), "plots")
if _PLOTS not in sys.path:
    sys.path.insert(0, _PLOTS)

import generate_plots as gp  # noqa: E402


def test_compute_quantile_ece_perfect_calibration():
    quantiles = np.arange(0.1, 1.0, 0.1)
    rng = np.random.default_rng(0)
    mu, sigma = 15.0, 5.0
    n = 5000

    y_true = rng.normal(mu, sigma, size=n)
    quantile_values = norm.ppf(quantiles, loc=mu, scale=sigma)
    y_pred_quantiles = np.broadcast_to(quantile_values, (n, len(quantiles))).copy()

    ece, empirical_coverage = gp.compute_quantile_ece(y_true, y_pred_quantiles, quantiles)
    assert ece < 0.02
    assert np.all(np.diff(empirical_coverage) > 0)


def test_compute_quantile_ece_detects_miscalibration():
    quantiles = np.arange(0.1, 1.0, 0.1)
    rng = np.random.default_rng(0)
    mu, sigma = 15.0, 5.0
    n = 5000

    y_true = rng.normal(mu, sigma, size=n)
    skewed_quantiles = np.clip(quantiles + 0.15 * (quantiles - 0.5), 0.01, 0.99)
    quantile_values = norm.ppf(skewed_quantiles, loc=mu, scale=sigma)
    y_pred_quantiles = np.broadcast_to(quantile_values, (n, len(quantiles))).copy()

    ece, _ = gp.compute_quantile_ece(y_true, y_pred_quantiles, quantiles)
    assert ece > 0.02


def test_compute_quantile_ece_shape_validation():
    quantiles = np.arange(0.1, 1.0, 0.1)
    y_true = np.zeros(10)
    bad_pred = np.zeros((9, len(quantiles)))  # wrong n_samples vs. y_true
    with pytest.raises(ValueError):
        gp.compute_quantile_ece(y_true, bad_pred, quantiles)


class _FakeTabICLRegressor:
    """Deterministic stand-in for tabicl.TabICLRegressor: records call counts
    instead of running the real pretrained backbone, so the leakage/batching
    logic in plot_era5_quantile_reliability can be regression-tested cheaply.
    """

    def __init__(self):
        self.fit_calls = 0
        self.predict_alphas = []
        self._X = None
        self._y = None

    def fit(self, X, y):
        self.fit_calls += 1
        self._X, self._y = np.asarray(X), np.asarray(y)
        return self

    def predict(self, X_test, output_type="quantiles", alphas=None):
        assert output_type == "quantiles"
        self.predict_alphas.append(list(alphas))
        X_test = np.asarray(X_test)
        dists = np.linalg.norm(X_test[:, None, :] - self._X[None, :, :], axis=-1)
        nearest = self._y[np.argmin(dists, axis=1)]
        return np.broadcast_to(nearest[:, None], (X_test.shape[0], len(alphas))).copy()


def test_reliability_diagram_excludes_context_and_batches_alphas(monkeypatch):
    created = {}

    class _Tracked(_FakeTabICLRegressor):
        def __init__(self):
            super().__init__()
            created["reg"] = self

    monkeypatch.setattr(gp, "TabICLv2_Regressor", _Tracked)

    captured = {}

    def fake_generate(y_true, y_pred_quantiles, quantiles, out_path):
        captured["y_true"] = y_true
        captured["y_pred_quantiles"] = y_pred_quantiles
        return 0.0

    monkeypatch.setattr(gp, "generate_era5_reliability_diagram", fake_generate)

    grid_size, n_days = 5, 3
    rng = np.random.default_rng(0)
    data = {
        "t2m": rng.normal(size=(n_days, grid_size, grid_size)),
        "latitude": np.linspace(0, 1, grid_size),
        "longitude": np.linspace(0, 1, grid_size),
    }
    M = grid_size * grid_size
    context_idx = np.array([0, 3, 7, 12, 20])
    quantiles = np.array([0.1, 0.5, 0.9])

    gp.plot_era5_quantile_reliability(data, context_idx, quantiles=quantiles)

    n_target = M - len(context_idx)
    assert captured["y_true"].shape[0] == n_days * n_target
    assert captured["y_pred_quantiles"].shape == (n_days * n_target, len(quantiles))

    reg = created["reg"]
    assert reg.fit_calls == n_days
    assert len(reg.predict_alphas) == n_days
    assert all(len(a) == len(quantiles) for a in reg.predict_alphas)
