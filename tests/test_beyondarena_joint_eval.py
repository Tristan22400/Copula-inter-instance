"""Pure helpers of the BeyondArena joint track: blocks, cross-fitting folds, ICC, rank truncation, sampling, metrics."""

from __future__ import annotations

import numpy as np
import pytest

from eval.runners.beyondarena_joint_eval import (
    anova_icc,
    block_metrics,
    cross_fit_folds,
    exchangeable,
    keep_columns,
    make_blocks,
    sample_block,
    truncate_rank,
)


def test_grouped_blocks_are_single_groups_capped() -> None:
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(10), [1, 3, 5, 200, 2, 2, 2, 2, 2, 2])
    idx = np.arange(len(g))
    blocks = make_blocks("grouped", idx, g, np.full(len(g), np.nan), 32, 128, 1000, rng)
    assert all(len(np.unique(g[b])) == 1 for b in blocks)
    assert max(len(b) for b in blocks) == 128
    assert 0 not in {int(g[b[0]]) for b in blocks}  # the 1-row group is dropped
    exact = make_blocks("grouped", idx, g, np.full(len(g), np.nan), 4, 4, 1000, rng, exact=True)
    assert {len(b) for b in exact} == {4} and len(exact) == 2  # groups of 5 and 200 rows


def test_temporal_blocks_are_consecutive_windows() -> None:
    rng = np.random.default_rng(0)
    t = rng.permutation(100).astype(float)
    blocks = make_blocks("temporal", np.arange(100), np.full(100, -1), t, 10, 128, 1000, rng)
    assert len(blocks) == 10
    for b in blocks:
        assert np.ptp(t[b]) == 9  # 10 consecutive time stamps


def test_max_blocks_caps_the_count() -> None:
    rng = np.random.default_rng(0)
    blocks = make_blocks("iid", np.arange(1000), np.full(1000, -1), np.full(1000, np.nan), 8, 8, 7, rng)
    assert len(blocks) == 7


def test_struct_folds_keep_groups_whole_and_time_contiguous() -> None:
    rng = np.random.default_rng(1)
    g = rng.integers(0, 40, 500)
    f = cross_fit_folds("grouped", "struct", g, np.zeros(500), 5, rng)
    for gid in np.unique(g):
        assert len(np.unique(f[g == gid])) == 1
    t = rng.random(500)
    f = cross_fit_folds("temporal", "struct", np.zeros(500), t, 5, rng)
    order = np.argsort(t)
    assert np.all(np.diff(f[order]) >= 0)
    f = cross_fit_folds("grouped", "random", g, t, 5, rng)
    assert np.bincount(f).min() == 100


def test_anova_icc_recovers_the_intraclass_correlation() -> None:
    rng = np.random.default_rng(0)
    rho, K, n = 0.4, 400, 8
    z = np.sqrt(rho) * np.repeat(rng.standard_normal(K), n) + np.sqrt(1 - rho) * rng.standard_normal(K * n)
    assert anova_icc(z, np.repeat(np.arange(K), n)) == pytest.approx(rho, abs=0.05)
    assert abs(anova_icc(rng.standard_normal(K * n), np.repeat(np.arange(K), n))) < 0.05


def test_truncate_rank_keeps_the_unit_diagonal() -> None:
    rng = np.random.default_rng(0)
    U = rng.standard_normal((20, 6)) * 0.3
    D = 1.0 - (U * U).sum(1)
    Ur, Dr = truncate_rank(U, D, 2)
    assert Ur.shape == (20, 2)
    np.testing.assert_allclose(np.diag(Ur @ Ur.T) + Dr, 1.0, atol=1e-12)


def test_sampling_keeps_marginals_and_hits_the_correlation() -> None:
    rng = np.random.default_rng(0)
    levels = np.linspace(0.001, 0.999, 999)
    from scipy.stats import norm

    q = np.tile(norm.ppf(levels), (3, 1))
    eps = rng.standard_normal((20000, 3))
    _, Y_ind = sample_block(np.eye(3), eps, q, levels)
    _, Y_exc = sample_block(exchangeable(3, 0.6), eps, q, levels)
    # Same marginal law under both (the copula only couples the rows).
    for i in range(3):
        assert np.quantile(Y_ind[:, i], [0.1, 0.5, 0.9]) == pytest.approx(
            np.quantile(Y_exc[:, i], [0.1, 0.5, 0.9]), abs=0.05
        )
    assert np.corrcoef(Y_exc.T)[0, 1] == pytest.approx(0.6, abs=0.03)


def test_block_mean_coverage_needs_the_dependence() -> None:
    """With rho = 0.5 across 32 rows, independent samples under-cover the block mean; the true copula covers it."""
    rng = np.random.default_rng(0)
    levels = np.linspace(0.001, 0.999, 999)
    from scipy.stats import norm

    b, rho = 32, 0.5
    q = np.tile(norm.ppf(levels), (b, 1))
    cov = {"indep": [], "exch": []}
    es = {"indep": [], "exch": []}
    for _ in range(200):
        y = np.sqrt(rho) * rng.standard_normal() + np.sqrt(1 - rho) * rng.standard_normal(b)
        eps = rng.standard_normal((500, b))
        for name, sig in (("indep", np.eye(b)), ("exch", exchangeable(b, rho))):
            m = block_metrics(y, sample_block(sig, eps, q, levels)[1], 0.5)
            cov[name].append(m["mean_cov90"])
            es[name].append(m["es"])
    assert np.mean(cov["exch"]) == pytest.approx(0.9, abs=0.06)
    assert np.mean(cov["indep"]) < 0.5
    assert np.mean(es["exch"]) < np.mean(es["indep"])


def test_keep_columns_caps_width_and_keeps_the_group_column() -> None:
    cols = keep_columns(742, 500, 741, seed=42)
    assert len(cols) == 500 and 741 in cols and np.all(np.diff(cols) > 0)
    np.testing.assert_array_equal(cols, keep_columns(742, 500, 741, seed=42))
    assert len(keep_columns(742, 500, -1, seed=42)) == 500
