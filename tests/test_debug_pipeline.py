"""Fast CPU checks for the debug/ pipeline."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

_TESTS = os.path.dirname(os.path.abspath(__file__))


def test_rank_ceiling_recovers_exact_low_rank_target():
    """Fitting rank r to a rank-r covnorm matrix drives the ceiling loss to ~0."""
    from copula_inter.model import low_rank_correlation
    from debug.stages.s1_rank_ceiling import fit_rank_ceiling

    torch.manual_seed(0)
    N, r = 24, 2
    W_true = torch.randn(1, N, r) * 0.8
    s_true = torch.randn(1, N) * 0.3
    R_true = low_rank_correlation(W_true, s_true, jitter=1e-4, parametrization="covnorm")

    per_ep, _ = fit_rank_ceiling(R_true, r, steps=800, lr=0.05, jitter=1e-4, device="cpu")
    assert per_ep.item() < 0.02, f"expected near-zero ceiling loss for an exactly-representable target, got {per_ep.item()}"


def test_rank_ceiling_monotone_in_rank():
    """The rank-8 ceiling loss is <= the rank-2 one on the same target."""
    from copula_inter.model import low_rank_correlation
    from debug.stages.s1_rank_ceiling import fit_rank_ceiling

    torch.manual_seed(1)
    N = 32
    W_true = torch.randn(1, N, 6) * 0.6
    s_true = torch.randn(1, N) * 0.3
    R_true = low_rank_correlation(W_true, s_true, jitter=1e-4, parametrization="covnorm")

    loss_r2, _ = fit_rank_ceiling(R_true, 2, steps=400, lr=0.05, device="cpu")
    loss_r16, _ = fit_rank_ceiling(R_true, 16, steps=400, lr=0.05, device="cpu")
    assert loss_r16.item() <= loss_r2.item() + 1e-3


def test_clamping_census_all_saturated():
    from debug.stages.s2_uspace import U_SPLINE_KNOT, _clamp_stats

    n_points = 50
    u_fully_saturated = [np.full(n_points, U_SPLINE_KNOT / 2.0)]  # every point below the spline threshold
    stats = _clamp_stats(u_fully_saturated)
    assert stats["pooled_frac_spline_saturated"] == pytest.approx(1.0)
    assert stats["n_episodes_gt_1pct_saturated"] == 1
    assert stats["n_episodes_total"] == 1


def test_clamping_census_none_saturated():
    from debug.stages.s2_uspace import _clamp_stats

    n_points = 50
    u_uniform = [np.linspace(0.1, 0.9, n_points)]
    stats = _clamp_stats(u_uniform)
    assert stats["pooled_frac_spline_saturated"] == pytest.approx(0.0)
    assert stats["n_episodes_gt_1pct_saturated"] == 0


def test_u_from_z_roundtrips_probit():
    """u_from_z inverts pit._probit (clamped values come back at the clamp)."""
    from copula_inter.pit import _probit
    from debug.stages.s2_uspace import U_HARD_CLAMP, u_from_z

    u = torch.tensor([0.5, 0.1, 0.9, 1e-9, 1.0 - 1e-9])
    z = _probit(u)
    u_back = u_from_z(z)
    expected = np.array([0.5, 0.1, 0.9, U_HARD_CLAMP, 1.0 - U_HARD_CLAMP])
    np.testing.assert_allclose(u_back, expected, atol=1e-4)


def test_build_config_applies_dotted_overrides():
    from debug.config import build_config

    dcfg = build_config(
        overrides=["data.P_min=17", "data.P_max=17", "model.rank=64"],
        n_episodes=3, device="cpu", seed=1,
    )
    assert int(dcfg.cfg.data.P_min) == 17
    assert int(dcfg.cfg.data.P_max) == 17
    assert int(dcfg.cfg.model.rank) == 64


def test_build_config_rejects_malformed_override():
    from debug.config import build_config

    with pytest.raises(ValueError):
        build_config(overrides=["not_a_key_value_pair"], device="cpu")


def test_s0_posterior_signal_uses_per_point_normalization():
    """S0 divides the posterior copula NLL by n_test."""
    from debug.stages.s0_signal import run_one_P
    from debug.config import build_config

    dcfg = build_config(overrides=["data.P_min=8", "data.P_max=8"], n_episodes=2, device="cpu", seed=42)
    result = run_one_P(dcfg, P=8, n_episodes=2)
    if result["n_episodes_scored"] == 0:
        pytest.skip("no episodes scored for this seed (rare unsupported kernel schema)")
    # Per-point values are O(1), not a sum over N.
    assert abs(result["copula_nll_per_point"]["mean"]) < 50.0
