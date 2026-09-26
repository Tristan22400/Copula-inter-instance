"""Tests for inference/copula_inference.py with hand-built quantile grids and a fake copula model."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
from scipy.stats import norm

_TESTS = os.path.dirname(os.path.abspath(__file__))

from copula_inter.model import low_rank_correlation  # noqa: E402
from inference.copula_inference import (  # noqa: E402
    _resolve_copula_checkpoint,
    get_test_correlation,
    normalize_features,
    sample_trajectories,
)
from tabicl._model.quantile_dist import QuantileDistribution  # noqa: E402


def test_resolve_copula_checkpoint_directory_uses_highest_step(tmp_path):
    """A checkpoint directory selects the last numerical training snapshot."""
    (tmp_path / "step_0000010.pt").touch()
    (tmp_path / "step_0000020.pt").touch()
    final = tmp_path / "step_0000020_final.pt"
    final.touch()
    (tmp_path / "notes.pt").touch()

    assert _resolve_copula_checkpoint(str(tmp_path)) == str(final)


def test_resolve_copula_checkpoint_file_is_unchanged(tmp_path):
    checkpoint = tmp_path / "arbitrary-name.pt"
    checkpoint.touch()

    assert _resolve_copula_checkpoint(str(checkpoint)) == str(checkpoint)


def test_pit_recovers_standard_normal_from_exact_quantile_grid():
    """cdf + probit on an exact Gaussian quantile grid recovers ~N(0, 1)."""
    rng = np.random.default_rng(0)
    probs = np.linspace(0.0, 1.0, 999 + 2)[1:-1]
    mu, sigma = 2.0, 1.5

    quantiles = norm.ppf(probs, loc=mu, scale=sigma)  # exact F^{-1}(probs)
    quantiles_t = torch.as_tensor(quantiles, dtype=torch.float64).unsqueeze(0)  # batch_shape=(1,)
    probs_t = torch.as_tensor(probs, dtype=torch.float64)

    dist = QuantileDistribution(quantiles_t, alpha_levels=probs_t)

    y_true = norm.rvs(loc=mu, scale=sigma, size=5000, random_state=rng)
    y_true_t = torch.as_tensor(y_true, dtype=torch.float64)

    u = dist.cdf(y_true_t).squeeze(0).numpy()
    z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))

    assert abs(z.mean()) < 0.1
    assert abs(z.std() - 1.0) < 0.1


def test_pit_recovers_standard_normal_from_skewnorm_grid():
    """Same with a skew-normal marginal."""
    from scipy.stats import skewnorm

    rng = np.random.default_rng(1)
    probs = np.linspace(0.0, 1.0, 999 + 2)[1:-1]
    a = 4.0  # skewness parameter

    quantiles = skewnorm.ppf(probs, a)
    quantiles_t = torch.as_tensor(quantiles, dtype=torch.float64).unsqueeze(0)
    probs_t = torch.as_tensor(probs, dtype=torch.float64)

    dist = QuantileDistribution(quantiles_t, alpha_levels=probs_t)

    y_true = skewnorm.rvs(a, size=5000, random_state=rng)
    y_true_t = torch.as_tensor(y_true, dtype=torch.float64)

    u = dist.cdf(y_true_t).squeeze(0).numpy()
    z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))

    assert abs(z.mean()) < 0.15
    assert abs(z.std() - 1.0) < 0.15


def _standard_normal_grid(n_test: int, probs: np.ndarray) -> np.ndarray:
    return np.tile(norm.ppf(probs), (n_test, 1))


def test_sample_trajectories_recovers_known_correlation():
    n_test = 4
    probs = np.linspace(0.0, 1.0, 999 + 2)[1:-1]
    quantile_grid = _standard_normal_grid(n_test, probs)

    rho = 0.6
    idx = np.arange(n_test)
    R = rho ** np.abs(idx[:, None] - idx[None, :])

    samples_small, _ = sample_trajectories(
        quantile_grid, probs, R, n_samples=300, rng=np.random.default_rng(10)
    )
    samples_large, _ = sample_trajectories(
        quantile_grid, probs, R, n_samples=30000, rng=np.random.default_rng(11)
    )

    err_small = np.linalg.norm(np.corrcoef(samples_small, rowvar=False) - R)
    err_large = np.linalg.norm(np.corrcoef(samples_large, rowvar=False) - R)

    assert err_large < err_small


def test_sample_trajectories_identity_gives_near_zero_cross_correlation():
    n_test = 5
    probs = np.linspace(0.0, 1.0, 999 + 2)[1:-1]
    quantile_grid = _standard_normal_grid(n_test, probs)
    R = np.eye(n_test)

    samples, _ = sample_trajectories(
        quantile_grid, probs, R, n_samples=20000, rng=np.random.default_rng(20)
    )
    corr = np.corrcoef(samples, rowvar=False)
    off_diag = corr[~np.eye(n_test, dtype=bool)]

    assert np.abs(off_diag).max() < 0.1


def test_sample_trajectories_clip_diagnostic_flags_narrow_grid():
    n_test = 3
    R = np.eye(n_test)

    # Narrow grid: most probabilities are clipped.
    probs_narrow = np.linspace(0.3, 0.7, 50)
    grid_narrow = np.tile(np.linspace(-0.5, 0.5, 50), (n_test, 1))
    _, n_clipped_narrow = sample_trajectories(
        grid_narrow, probs_narrow, R, n_samples=2000, rng=np.random.default_rng(30)
    )

    # Wide grid: essentially the whole standard-normal mass is covered.
    probs_wide = np.linspace(1e-6, 1 - 1e-6, 999)
    grid_wide = np.tile(norm.ppf(probs_wide), (n_test, 1))
    _, n_clipped_wide = sample_trajectories(
        grid_wide, probs_wide, R, n_samples=2000, rng=np.random.default_rng(31)
    )

    assert n_clipped_narrow > 0
    assert n_clipped_wide < n_clipped_narrow


class _FakeCopulaModel(torch.nn.Module):
    """Stands in for CopulaTabICL: forward(batch) -> {"W", "s"}, ignoring the batch."""

    def __init__(self, W: torch.Tensor, s: torch.Tensor):
        super().__init__()
        self.W = W
        self.s = s
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict) -> dict:
        return {"W": self.W, "s": self.s}


def test_get_test_correlation_is_symmetric_and_unit_diagonal():
    torch.manual_seed(0)
    n_test, rank = 6, 2
    W = torch.randn(1, n_test, rank) * 0.3
    s = torch.randn(1, n_test)
    model = _FakeCopulaModel(W, s)

    X_train = np.random.randn(4, 1)
    Z_train = np.random.randn(4)
    X_test = np.random.randn(n_test, 1)

    R = get_test_correlation(model, X_train, Z_train, X_test)

    assert np.allclose(np.diag(R), 1.0)
    assert np.allclose(R, R.T)
    eigvals = np.linalg.eigvalsh(R)
    assert eigvals.min() > -1e-6


def test_get_test_correlation_matches_low_rank_correlation_up_to_postprocessing():
    """get_test_correlation is low_rank_correlation plus symmetrization and a unit diagonal."""
    torch.manual_seed(1)
    n_test, rank = 5, 3
    W = torch.randn(1, n_test, rank) * 0.5
    s = torch.randn(1, n_test)
    model = _FakeCopulaModel(W, s)

    X_train = np.random.randn(3, 1)
    Z_train = np.random.randn(3)
    X_test = np.random.randn(n_test, 1)

    R = get_test_correlation(model, X_train, Z_train, X_test)
    Sigma_raw = low_rank_correlation(W, s)[0].detach().numpy()

    Sigma_expected = 0.5 * (Sigma_raw + Sigma_raw.T)
    np.fill_diagonal(Sigma_expected, 1.0)

    assert np.allclose(R, Sigma_expected, atol=1e-6)


def test_normalize_features_gives_zero_mean_unit_std_jointly_over_train_and_test():
    rng = np.random.default_rng(0)
    X_train = rng.uniform(10.0, 20.0, size=(7, 2))  # arbitrary raw scale/offset
    X_test = rng.uniform(10.0, 20.0, size=(13, 2))

    X_train_norm, X_test_norm = normalize_features(X_train, X_test)

    combined = np.concatenate([X_train_norm, X_test_norm], axis=0)
    assert np.allclose(combined.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(combined.std(axis=0, ddof=1), 1.0, atol=1e-8)
    assert X_train_norm.shape == X_train.shape
    assert X_test_norm.shape == X_test.shape


def test_normalize_features_uses_joint_not_train_only_statistics():
    """normalize_features uses the joint train+test mean and std."""
    X_test = np.linspace(0.0, 1.0, 50).reshape(-1, 1)
    X_train = X_test[:5]  # narrow, non-representative subset

    X_train_norm, X_test_norm = normalize_features(X_train, X_test)

    joint = np.concatenate([X_train, X_test])
    assert not np.isclose(X_train.mean(), joint.mean(), atol=1e-3)  # train-only stats would differ

    expected_train_norm = (X_train - joint.mean()) / joint.std(ddof=1)
    assert np.allclose(X_train_norm, expected_train_norm)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
