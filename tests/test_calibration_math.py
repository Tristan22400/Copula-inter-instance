"""
test_calibration_math.py — Null-hypothesis correctness checks for the
multivariate spatial calibration metrics in plots/generate_plots.py
(calc_kendall_pit, calc_mahalanobis_distances, calc_exceedance_probs,
calc_spatial_coverage) before running them on real ERA5 data.

Each test constructs synthetic data under perfect calibration (H0: the
declared independence-copula model matches the true generating process) and
checks the metric's known closed-form null distribution -- Uniform(0, 1) for
the Kendall PIT, chi^2_D for the Mahalanobis distance, and c^D for spatial
coverage at nominal level c (independent Uniform(0, 1) marginals).

Run directly:
    pytest tests/test_calibration_math.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from scipy.stats import kstest, norm

_TESTS = os.path.dirname(os.path.abspath(__file__))
_PLOTS = os.path.join(os.path.dirname(_TESTS), "plots")
if _PLOTS not in sys.path:
    sys.path.insert(0, _PLOTS)

import generate_plots as gp  # noqa: E402

_SEED = 0


def test_kendall_pit_null():
    """Y ~ N(0, I_D) scored against standard-normal marginal CDFs: the
    resulting Kendall PIT values must be indistinguishable from Uniform(0, 1)."""
    rng = np.random.default_rng(_SEED)
    n, D = 4000, 5

    y = rng.standard_normal((n, D))
    cdf_values = norm.cdf(y)

    z = gp.calc_kendall_pit(cdf_values)
    assert z.shape == (n,)
    assert np.all((z >= 0.0) & (z <= 1.0))

    stat, p_value = kstest(z, "uniform")
    assert p_value > 0.05, f"Kendall PIT failed KS-test against Uniform(0,1): stat={stat:.4f}, p={p_value:.4f}"


def test_kendall_pit_null_various_dims():
    """The Kendall transform must hold under H0 for D=1 (identity case) and larger D."""
    rng = np.random.default_rng(_SEED + 1)
    n = 4000
    for D in (1, 2, 10):
        y = rng.standard_normal((n, D))
        cdf_values = norm.cdf(y)
        z = gp.calc_kendall_pit(cdf_values)
        _, p_value = kstest(z, "uniform")
        assert p_value > 0.01, f"D={D}: Kendall PIT failed KS-test, p={p_value:.4f}"


def test_kendall_pit_detects_miscalibration():
    """An overconfident independence copula (true correlation ignored, CDFs
    computed from too-narrow marginals) must be rejected by the KS-test."""
    rng = np.random.default_rng(_SEED)
    n, D = 4000, 5

    y = rng.standard_normal((n, D))
    # Declare the marginals as N(0, 0.5^2) -- too narrow relative to the true
    # N(0, 1) generating process -- so the model is miscalibrated.
    cdf_values = norm.cdf(y, loc=0.0, scale=0.5)

    z = gp.calc_kendall_pit(cdf_values)
    _, p_value = kstest(z, "uniform")
    assert p_value < 0.05


def test_mahalanobis_null():
    """Y ~ N(0, I_D) with mu=0, sigma^2=1: d^2 must pass a KS-test against chi2(df=D)."""
    rng = np.random.default_rng(_SEED)
    n, D = 4000, 5

    y = rng.standard_normal((n, D))
    means = np.zeros((n, D))
    variances = np.ones((n, D))

    d2 = gp.calc_mahalanobis_distances(y, means, variances)
    assert d2.shape == (n,)
    assert np.all(d2 >= 0.0)

    stat, p_value = kstest(d2, "chi2", args=(D,))
    assert p_value > 0.05, f"Mahalanobis distances failed KS-test against chi2(df={D}): stat={stat:.4f}, p={p_value:.4f}"


def test_mahalanobis_shape_mismatch_raises():
    y = np.zeros((10, 3))
    means = np.zeros((10, 3))
    variances = np.ones((9, 3))  # wrong n_samples
    with pytest.raises(ValueError):
        gp.calc_mahalanobis_distances(y, means, variances)


def test_mahalanobis_detects_miscalibration():
    """Declaring variance=1 when the true generating variance is 4 must
    inflate d^2 well beyond a chi2_D null, and get rejected by the KS-test."""
    rng = np.random.default_rng(_SEED)
    n, D = 4000, 5

    y = rng.normal(loc=0.0, scale=2.0, size=(n, D))  # true variance = 4
    means = np.zeros((n, D))
    variances = np.ones((n, D))  # declared variance = 1 (overconfident)

    d2 = gp.calc_mahalanobis_distances(y, means, variances)
    _, p_value = kstest(d2, "chi2", args=(D,))
    assert p_value < 0.05


def test_spatial_coverage_null():
    """Independent Uniform(0, 1) marginals, nominal bounds [0.05, 0.95]:
    empirical joint coverage should be approximately 0.90^D."""
    rng = np.random.default_rng(_SEED)
    n, D = 20000, 4

    y = rng.uniform(0.0, 1.0, size=(n, D))
    q_lower = np.full((n, D), 0.05)
    q_upper = np.full((n, D), 0.95)

    coverage = gp.calc_spatial_coverage(y, q_lower, q_upper)
    expected = 0.90**D
    # Binomial standard error at n=20000 trials, p=expected.
    se = np.sqrt(expected * (1 - expected) / n)
    assert abs(coverage - expected) < 6 * se, (
        f"Empirical coverage {coverage:.4f} too far from 0.90^{D}={expected:.4f} "
        f"(6*SE={6 * se:.4f})"
    )


def test_spatial_coverage_curve_matches_calc(monkeypatch):
    """plot_spatial_coverage_curve must query quantile_func at alpha/2 and
    1 - alpha/2 for each nominal coverage and reproduce calc_spatial_coverage."""
    rng = np.random.default_rng(_SEED)
    n, D = 5000, 3
    y = rng.uniform(0.0, 1.0, size=(n, D))

    def quantile_func(alpha):
        lo = np.full((n, D), alpha / 2)
        hi = np.full((n, D), 1 - alpha / 2)
        return lo, hi

    nominal_coverages = np.array([0.5, 0.8, 0.9])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    gp.plot_spatial_coverage_curve(y, quantile_func, nominal_coverages, ax)
    plt.close(fig)

    for c in nominal_coverages:
        alpha = 1.0 - c
        q_lower, q_upper = quantile_func(alpha)
        expected = gp.calc_spatial_coverage(y, q_lower, q_upper)
        assert abs(expected - (1 - alpha) ** D) < 0.05


def test_exceedance_probs_null():
    """Independence copula with correctly-specified Gaussian marginals: the
    predicted exceedance probability must equal the true exceedance frequency
    within Monte Carlo error, i.e. the reliability curve lies on y = x."""
    rng = np.random.default_rng(_SEED)
    n, D = 20000, 3

    y = rng.standard_normal((n, D))

    def cdf_func(tau):
        return np.broadcast_to(norm.cdf(tau), (n, D))

    thresholds = np.array([-1.0, 0.0, 1.0, 2.0])
    predicted_probs, true_events = gp.calc_exceedance_probs(y, cdf_func, thresholds)

    assert predicted_probs.shape == (n, len(thresholds))
    assert true_events.shape == (n, len(thresholds))

    empirical = true_events.mean(axis=0)
    predicted = predicted_probs[0]  # identical across rows here (constant cdf_func)
    se = np.sqrt(predicted * (1 - predicted) / n)
    assert np.all(np.abs(empirical - predicted) < 6 * np.clip(se, 1e-6, None))
