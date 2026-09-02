"""tests/test_debug_pipeline.py — sanity checks for the debug/ pipeline
(see debug/README.md). Fast, CPU-only, no TabICL/network dependency.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_TESTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TESTS)
for _p in (os.path.join(_ROOT, "debug"), os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# S1: rank-ceiling fitter
# ---------------------------------------------------------------------------

def test_rank_ceiling_recovers_exact_low_rank_target():
    """A correlation matrix built with model.py's OWN covnorm parametrization
    at rank r lies exactly inside the rank-r covnorm family -- fitting rank r
    to it should drive the ceiling loss (expected copula NLL vs. itself)
    down to ~0, not just "small"."""
    from model import low_rank_correlation
    from stages.s1_rank_ceiling import fit_rank_ceiling

    torch.manual_seed(0)
    N, r = 24, 2
    W_true = torch.randn(1, N, r) * 0.8
    s_true = torch.randn(1, N) * 0.3
    R_true = low_rank_correlation(W_true, s_true, jitter=1e-4, parametrization="covnorm")

    per_ep, _ = fit_rank_ceiling(R_true, r, steps=800, lr=0.05, jitter=1e-4, device="cpu")
    assert per_ep.item() < 0.02, f"expected near-zero ceiling loss for an exactly-representable target, got {per_ep.item()}"


def test_rank_ceiling_monotone_in_rank():
    """A higher rank can only do at least as well: r=8's ceiling loss on a
    generic R must be <= r=2's (both fit to the SAME target, more capacity
    can't hurt the population-level optimum)."""
    from model import low_rank_correlation
    from stages.s1_rank_ceiling import fit_rank_ceiling

    torch.manual_seed(1)
    N = 32
    W_true = torch.randn(1, N, 6) * 0.6
    s_true = torch.randn(1, N) * 0.3
    R_true = low_rank_correlation(W_true, s_true, jitter=1e-4, parametrization="covnorm")

    loss_r2, _ = fit_rank_ceiling(R_true, 2, steps=400, lr=0.05, device="cpu")
    loss_r16, _ = fit_rank_ceiling(R_true, 16, steps=400, lr=0.05, device="cpu")
    assert loss_r16.item() <= loss_r2.item() + 1e-3


# ---------------------------------------------------------------------------
# S2: clamping census
# ---------------------------------------------------------------------------

def test_clamping_census_all_saturated():
    from stages.s2_uspace import U_SPLINE_KNOT, _clamp_stats

    n_points = 50
    u_fully_saturated = [np.full(n_points, U_SPLINE_KNOT / 2.0)]  # every point below the spline threshold
    stats = _clamp_stats(u_fully_saturated)
    assert stats["pooled_frac_spline_saturated"] == pytest.approx(1.0)
    assert stats["n_episodes_gt_1pct_saturated"] == 1
    assert stats["n_episodes_total"] == 1


def test_clamping_census_none_saturated():
    from stages.s2_uspace import _clamp_stats

    n_points = 50
    u_uniform = [np.linspace(0.1, 0.9, n_points)]
    stats = _clamp_stats(u_uniform)
    assert stats["pooled_frac_spline_saturated"] == pytest.approx(0.0)
    assert stats["n_episodes_gt_1pct_saturated"] == 0


def test_u_from_z_roundtrips_probit():
    """u_from_z is the exact inverse of pit.py::_probit -- round-tripping a
    non-saturated u through _probit -> u_from_z should recover it, and a
    saturated u should come back exactly at the clamp bound."""
    from pit import _probit
    from stages.s2_uspace import U_HARD_CLAMP, u_from_z

    u = torch.tensor([0.5, 0.1, 0.9, 1e-9, 1.0 - 1e-9])
    z = _probit(u)
    u_back = u_from_z(z)
    expected = np.array([0.5, 0.1, 0.9, U_HARD_CLAMP, 1.0 - U_HARD_CLAMP])
    np.testing.assert_allclose(u_back, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# config.py: override parser
# ---------------------------------------------------------------------------

def test_build_config_applies_dotted_overrides():
    from config import build_config

    dcfg = build_config(
        overrides=["data.P_min=17", "data.P_max=17", "model.rank=64"],
        n_episodes=3, device="cpu", seed=1,
    )
    assert int(dcfg.cfg.data.P_min) == 17
    assert int(dcfg.cfg.data.P_max) == 17
    assert int(dcfg.cfg.model.rank) == 64


def test_build_config_rejects_malformed_override():
    from config import build_config

    with pytest.raises(ValueError):
        build_config(overrides=["not_a_key_value_pair"], device="cpu")


# ---------------------------------------------------------------------------
# S0: per-point normalization matches loss.py::oracle_copula_nll's own convention
# ---------------------------------------------------------------------------

def test_s0_posterior_signal_uses_per_point_normalization():
    """gp_analytical_posterior's nll_post_copula is a raw sum over N test
    points (not yet per-point); s0_signal.py must divide by n_test itself
    to match every other per-point metric in this pipeline -- this test
    guards that division against silently disappearing in a refactor."""
    from stages.s0_signal import run_one_P
    from config import build_config

    dcfg = build_config(overrides=["data.P_min=8", "data.P_max=8"], n_episodes=2, device="cpu", seed=42)
    result = run_one_P(dcfg, P=8, n_episodes=2)
    if result["n_episodes_scored"] == 0:
        pytest.skip("no episodes scored for this seed (rare unsupported kernel schema)")
    # A per-point copula NLL for a handful of test points should be a small
    # (O(1)-ish) number, not a raw sum over N~256 points (which would be
    # orders of magnitude larger) -- catches a missing "/ n_test" directly.
    assert abs(result["copula_nll_per_point"]["mean"]) < 50.0
