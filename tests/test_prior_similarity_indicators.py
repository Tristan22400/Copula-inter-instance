"""Ground-truth checks for eval/prior_similarity/indicators.py.

Every indicator in the suite is used to argue that one prior is closer to ERA5
than another, so each one needs a case where its correct value is known
analytically. These tests build fields whose answer is known and assert the
indicator recovers it.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.prior_similarity.bundles import FieldBundle, _euclidean, _matern_corr, _simulate_from_correlation
from eval.prior_similarity.indicators import (
    _eigen_stats,
    compute_all,
    tier0_geometry,
    tier1_nonstationarity,
    tier1_second_order,
    tier2_posterior,
    tier3_marginal,
)


def _lattice_bundle(gs=16, L=4.0, nu=1.5, R=2000, seed=0, sd=None):
    ax = np.arange(gs, dtype=np.float64)
    gx, gy = np.meshgrid(ax, ax)
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    dist = _euclidean(coords)
    C = _matern_corr(dist, L, nu)
    if sd is not None:
        # Deliberately NOT renormalized back to a correlation matrix: the
        # non-stationary marginal variance is the thing under test.
        C = C * np.outer(sd, sd)
    rng = np.random.default_rng(seed)
    fields = _simulate_from_correlation(C, R, rng)
    return FieldBundle(coords=coords, fields=fields, source="test", grid_shape=(gs, gs),
                       dist=dist, R_prior_analytic=C)


def _cloud_bundle(D=256, d=9, L=1.5, R=2000, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.normal(size=(D, d))
    dist = _euclidean(coords)
    C = _matern_corr(dist, L, 1.5)
    fields = _simulate_from_correlation(C, R, rng)
    return FieldBundle(coords=coords, fields=fields, source="test_cloud", grid_shape=None, dist=dist)


def test_geometry_separates_lattice_from_cloud():
    """nn_cv and spread_over_nn are the two Tier-0 claims: a lattice is
    perfectly regular and spans ~sqrt(D) spacings; a Gaussian cloud in ~9
    dimensions is irregular and spans far fewer."""
    lat = tier0_geometry(_lattice_bundle(gs=16))
    cloud = tier0_geometry(_cloud_bundle(D=256, d=9))

    assert lat["nn_cv"] < 0.02, "a perfect lattice must have ~zero NN-distance spread"
    assert cloud["nn_cv"] > 0.15, "a Gaussian point cloud must have visibly irregular spacing"
    # 16x16 lattice: median pair distance / spacing is ~sqrt(D)/2 territory.
    assert lat["spread_over_nn"] > 2.0 * cloud["spread_over_nn"]


def test_range_over_nn_recovers_the_generating_lengthscale():
    """The headline Tier-1 scalar has to be in the right units: a Matern with
    L = 4 lattice spacings must give a 0.5-crossing of a few spacings, and
    doubling L must roughly double it."""
    r4 = tier1_second_order(_lattice_bundle(gs=20, L=4.0), med_nn=1.0)["range_over_nn"]
    r8 = tier1_second_order(_lattice_bundle(gs=20, L=8.0, seed=1), med_nn=1.0)["range_over_nn"]
    assert 2.0 < r4 < 8.0, f"L=4 spacings should give a 0.5-crossing of a few spacings, got {r4}"
    assert r8 > 1.5 * r4, f"doubling the lengthscale must lengthen the range: {r4} -> {r8}"


def test_eigen_stats_on_known_spectra():
    D = 100
    assert _eigen_stats(np.eye(D), "p")["p_evr32"] == pytest.approx(32 / D, abs=1e-9)
    assert _eigen_stats(np.eye(D), "p")["p_eff_rank_frac"] == pytest.approx(1.0, abs=1e-9)
    rank1 = np.ones((D, D))
    assert _eigen_stats(rank1, "p")["p_evr8"] == pytest.approx(1.0, abs=1e-9)
    assert _eigen_stats(rank1, "p")["p_eff_rank"] == pytest.approx(1.0, abs=1e-6)


def test_posterior_conditioning_reduces_variance_and_is_psd():
    """Tier 2's whole argument rests on the Schur complement being trustworthy
    on ill-conditioned correlations, which is where the raw implementation
    blew up to 1e7 before shrinkage + PSD repair."""
    out = tier2_posterior(_lattice_bundle(gs=16, L=6.0), med_nn=1.0)
    assert 0.0 < out["post05_var_ratio"] < 1.0
    assert out["post20_var_ratio"] < out["post05_var_ratio"], "more context must remove more variance"
    for k in ("post05_od_mu", "post20_od_mu", "post05_evr32", "post20_evr32"):
        assert 0.0 <= out[k] <= 1.0, f"{k} out of range: {out[k]}"
    assert out["post05_evr32"] <= out["post05_evr64"] <= out["post05_evr128"]


def test_nonstationarity_is_near_zero_for_a_stationary_field():
    """The indicator must report ~0 on a genuinely stationary field, otherwise
    a nonzero reading on ERA5 means nothing. This also fixes the estimation-
    noise floor at the realization count the runner uses."""
    stat = tier1_nonstationarity(_lattice_bundle(gs=16, L=4.0), med_nn=1.0)
    gs = 16
    ramp = np.exp(np.linspace(-1.0, 1.0, gs))[:, None] * np.ones((1, gs))
    nonstat = tier1_nonstationarity(_lattice_bundle(gs=gs, L=4.0, sd=ramp.ravel()), med_nn=1.0)
    assert stat["nonstat_var_cv"] < 0.15
    assert nonstat["nonstat_var_cv"] > 2.0 * stat["nonstat_var_cv"]


def test_increment_kurtosis_is_zero_for_a_gaussian_field():
    """increment_exkurt is the 'this is not a GP' detector; it must read ~0 on
    an actual GP or every reading on ERA5 is uninterpretable."""
    out = tier3_marginal(_lattice_bundle(gs=16, L=4.0))
    assert abs(out["increment_exkurt"]) < 0.6
    assert abs(out["spatial_exkurt"]) < 0.8


def test_marginal_indicators_see_a_monotone_warp():
    """The proposed marginal warp (plan section 2.5) must move the marginal
    indicators while leaving the correlation structure alone -- that invariance
    is the reason the warp is free."""
    b = _lattice_bundle(gs=16, L=4.0)
    plain = tier3_marginal(b)
    warped = FieldBundle(coords=b.coords, fields=np.sinh(1.0 + 0.8 * b.fields),
                         source="warped", grid_shape=b.grid_shape, dist=b.dist)
    w = tier3_marginal(warped)
    assert w["spatial_skew_abs"] > plain["spatial_skew_abs"] + 0.3
    assert w["spatial_exkurt"] > plain["spatial_exkurt"] + 1.0
    # Rank correlation is untouched by a monotone map, so the copula target is.
    from scipy.stats import spearmanr

    rho_a = spearmanr(b.fields[:, 0], b.fields[:, 5]).statistic
    rho_b = spearmanr(warped.fields[:, 0], warped.fields[:, 5]).statistic
    assert rho_a == pytest.approx(rho_b, abs=1e-9)


def test_compute_all_is_finite_and_source_agnostic():
    """The runner tabulates whatever compute_all returns; a NaN that is not a
    documented 'undefined here' case would silently poison a median."""
    for b in (_lattice_bundle(gs=14), _cloud_bundle(D=196, d=9)):
        row = compute_all(b)
        assert row["source"] == b.source
        expect_nan = {"spec_slope", "spec_nyquist_excess", "morans_i"} if b.grid_shape is None else set()
        bad = [k for k, v in row.items()
               if isinstance(v, float) and not np.isfinite(v)
               and k not in expect_nan
               and k not in {"range_over_nn", "range_e_over_nn", "det_range_over_nn",
                             "det_range_e_over_nn", "nonstat_range_cv"}]
        assert not bad, f"unexpected non-finite indicators for {b.source}: {bad}"
