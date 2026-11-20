"""eval.beyondarena.joint_metrics against exact dense computations and a synthetic grouped dataset."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from eval.beyondarena.joint_metrics import (
    _blockexch_logdensity,
    anova_icc,
    block_ids_for,
    joint_scores,
    keep_columns,
    lowrank_copula_logdensity,
    sample_normal_scores,
    truncate_rank,
)


def _dense_logc(S: np.ndarray, z: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(S)
    return float(-0.5 * (logdet + z @ np.linalg.solve(S, z) - z @ z))


def test_lowrank_sampling_and_density_match_dense() -> None:
    rng = np.random.default_rng(0)
    U = rng.standard_normal((12, 3))
    U *= np.sqrt(rng.uniform(0.1, 0.8, 12) / (U * U).sum(1))[:, None]  # a valid correlation: |U_i|^2 < 1
    D = 1.0 - (U * U).sum(1)
    S = U @ U.T + np.diag(D)
    gen = torch.Generator().manual_seed(0)
    Z = sample_normal_scores("lowrank", 12, 200_000, gen, torch.device("cpu"), U=torch.tensor(U), D=torch.tensor(D))
    np.testing.assert_allclose(np.cov(Z.numpy().T), S, atol=0.02)
    z = rng.standard_normal(12)
    assert lowrank_copula_logdensity(torch.tensor(U), torch.tensor(D), torch.tensor(z)) == pytest.approx(
        _dense_logc(S, z), abs=1e-8
    )


def test_block_exchangeable_density_matches_dense() -> None:
    rng = np.random.default_rng(1)
    ids = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
    rho = 0.35
    S = np.where(ids[:, None] == ids[None], rho, 0.0)
    np.fill_diagonal(S, 1.0)
    z = rng.standard_normal(len(ids))
    assert _blockexch_logdensity(torch.tensor(z), torch.tensor(ids), rho) == pytest.approx(_dense_logc(S, z), abs=1e-10)


def test_helpers() -> None:
    assert len(keep_columns(742, 500, 741, seed=1)) == 500 and 741 in keep_columns(742, 500, 741, seed=1)
    U = np.random.default_rng(0).standard_normal((20, 6)) * 0.3
    Ur, Dr = truncate_rank(U, 1.0 - (U * U).sum(1), 2)
    np.testing.assert_allclose((Ur * Ur).sum(1) + Dr, 1.0, atol=1e-12)
    t = np.random.default_rng(0).permutation(100).astype(float)
    ids = block_ids_for("temporal", None, t, 10, seed=0)
    assert all(np.ptp(t[ids == k]) == 9 for k in range(10))
    z = np.sqrt(0.4) * np.repeat(np.random.default_rng(0).standard_normal(300), 8) + np.sqrt(
        0.6
    ) * np.random.default_rng(1).standard_normal(2400)
    assert anova_icc(z, np.repeat(np.arange(300), 8)) == pytest.approx(0.4, abs=0.06)


def _gaussian_pred(y_mean: np.ndarray, y: np.ndarray, rho: float) -> dict:
    from scipy.stats import norm

    levels = np.linspace(0.001, 0.999, 999)
    return {
        "q": (y_mean[:, None] + norm.ppf(levels)[None]).astype(np.float32),
        "levels": levels,
        "z": y - y_mean,
        "logf": norm.logpdf(y - y_mean),
        "scale": 1.0,
        "rho_exch": rho,
    }


def test_joint_scores_reward_the_true_dependence_and_keep_marginals() -> None:
    rng = np.random.default_rng(0)
    G, n, rho = 60, 20, 0.5
    groups = np.repeat(np.arange(G), n).astype(str)
    y = np.sqrt(rho) * np.repeat(rng.standard_normal(G), n) + np.sqrt(1 - rho) * rng.standard_normal(G * n)
    pred = _gaussian_pred(np.zeros(G * n), y, rho)
    s = joint_scores(pred, y, "grouped", groups, None, n_samples=400, device="cpu", seed=0)
    ind, exc = s["indep"], s["exch"]
    assert ind["grid_crps"] == exc["grid_crps"]  # the marginal is untouched by the copula
    assert ind["sample_crps"] == pytest.approx(ind["grid_crps"], rel=0.03)
    assert exc["sample_crps"] == pytest.approx(ind["grid_crps"], rel=0.03)
    assert ind["joint_nll"] == pytest.approx(-np.mean(pred["logf"]), abs=1e-9)
    assert exc["joint_nll"] < ind["joint_nll"] - 0.1
    assert exc["energy"] < ind["energy"]
    assert exc["agg_group_cov90"] == pytest.approx(0.9, abs=0.12)
    assert ind["agg_group_cov90"] < 0.5
    assert exc["rho_pred"] == pytest.approx(rho) and ind["icc_test_z"] == pytest.approx(rho, abs=0.15)
    assert math.isfinite(ind["variogram_struct"]) and math.isfinite(ind["variogram_uniform"])
