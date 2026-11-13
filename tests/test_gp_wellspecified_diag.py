"""D1/D2 well-specified GP-MLE diagnostics: episode scoring and the summary."""

from __future__ import annotations

import math

import pytest

from eval.runners.gp_wellspecified_diag import GpDiagSpec, diagnose_episode, summarize


def test_diagnose_episode_scores_fit_and_oracle() -> None:
    """A well-specified RBF episode: finite totals, and a converged fit reaches J(theta_true) (D2 passes)."""
    spec = GpDiagSpec(n_query=16, n_steps=300, n_restarts=2, ard=[False, True])
    rows = diagnose_episode(spec, "rbf", 30, 0)
    assert [r["ard"] for r in rows] == [False, True]
    for r in rows:
        assert math.isfinite(r["oracle_total"]) and math.isfinite(r["fit_total"])
        assert r["gap"] == pytest.approx(r["fit_total"] - r["oracle_total"])
        # The fit cannot beat the known-theta posterior by much on one episode (catches an unconditioned oracle).
        assert r["gap"] > -0.5
    assert rows[0]["j_fit"] >= rows[0]["j_true"] - spec.j_tol


def test_summarize_gap_and_d2_failure_rate() -> None:
    rows = [
        {"kernel": "rbf", "ard": False, "P": 10, "gap": g, "j_fit": jf, "j_true": 0.0}
        for g, jf in [(0.1, 0.0), (0.3, -1.0), (0.2, 0.5), (math.nan, math.nan)]
    ]
    (s,) = summarize(rows, j_tol=0.05)
    assert s["n"] == 3 and s["n_failed_fits"] == 1
    assert s["gap_mean"] == pytest.approx(0.2) and s["gap_median"] == pytest.approx(0.2)
    assert s["d2_failure_rate"] == pytest.approx(1 / 3)
    assert s["d2_worst_dj"] == pytest.approx(-1.0)
