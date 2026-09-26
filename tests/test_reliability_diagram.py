"""Tests for compute_quantile_ece and plot_era5_quantile_reliability (with a fake regressor): context points are excluded, and one fit per day serves all quantile levels."""

from __future__ import annotations

import os

import numpy as np
import pytest
from pytest import MonkeyPatch
from scipy.stats import norm

_TESTS = os.path.dirname(os.path.abspath(__file__))

from eval.spatial import calibration as gp  # noqa: E402


def test_compute_quantile_ece_perfect_calibration() -> None:
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


def test_compute_quantile_ece_detects_miscalibration() -> None:
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


def test_compute_quantile_ece_shape_validation() -> None:
    quantiles = np.arange(0.1, 1.0, 0.1)
    y_true = np.zeros(10)
    bad_pred = np.zeros((9, len(quantiles)))  # wrong n_samples vs. y_true
    with pytest.raises(ValueError):
        gp.compute_quantile_ece(y_true, bad_pred, quantiles)


class _FakeTabICLRegressor:
    """Deterministic stand-in for TabICLRegressor that counts calls."""

    def __init__(self) -> None:
        self.fit_calls = 0
        self.predict_alphas = []
        self._X = None
        self._y = None

    def fit(self, X, y):
        self.fit_calls += 1
        self._X, self._y = np.asarray(X), np.asarray(y)
        return self

    def predict(self, X_test, output_type: str = "quantiles", alphas=None):
        assert output_type == "quantiles"
        self.predict_alphas.append(list(alphas))
        X_test = np.asarray(X_test)
        dists = np.linalg.norm(X_test[:, None, :] - self._X[None, :, :], axis=-1)
        nearest = self._y[np.argmin(dists, axis=1)]
        return np.broadcast_to(nearest[:, None], (X_test.shape[0], len(alphas))).copy()


def test_reliability_diagram_excludes_context_and_batches_alphas(monkeypatch: MonkeyPatch) -> None:
    created = {}

    class _Tracked(_FakeTabICLRegressor):
        def __init__(self) -> None:
            super().__init__()
            created["reg"] = self

    monkeypatch.setattr(gp, "make_tabicl_regressor", lambda checkpoint=None, device=None: _Tracked())

    captured = {}

    def fake_generate(y_true, y_pred_quantiles, quantiles, out_path) -> float:
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
