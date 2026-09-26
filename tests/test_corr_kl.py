"""Tests for pit.gaussian_corr_kl: zero iff equal, positive otherwise, equal to torch's MVN KL, inf (not an exception) for a singular input."""

from __future__ import annotations

import math

import pytest
import torch

from copula_inter.pit import gaussian_corr_kl


def _random_correlation(n, seed, rank=None):
    """Random PD correlation matrix (low-rank plus diagonal, unit diagonal)."""
    g = torch.Generator().manual_seed(seed)
    r = rank or max(2, n // 2)
    W = torch.randn(n, r, generator=g, dtype=torch.float64)
    S = W @ W.T + torch.diag(torch.rand(n, generator=g, dtype=torch.float64) + 0.5)
    d = S.diagonal().sqrt()
    return S / torch.outer(d, d)


@pytest.mark.parametrize("n", [3, 8, 25])
def test_zero_iff_identical(n) -> None:
    """The floor: KL(R || R) == 0 exactly."""
    R = _random_correlation(n, seed=n)
    assert abs(gaussian_corr_kl(R, R)) < 1e-9


@pytest.mark.parametrize("n", [3, 8, 25])
def test_strictly_positive_when_different(n) -> None:
    """Positive in both argument orders when the matrices differ."""
    A = _random_correlation(n, seed=n)
    B = _random_correlation(n, seed=n + 100)
    assert gaussian_corr_kl(A, B) > 1e-6
    assert gaussian_corr_kl(B, A) > 1e-6


@pytest.mark.parametrize("n", [4, 12])
def test_matches_torch_kl_divergence(n) -> None:
    """Matches torch.distributions: gaussian_corr_kl(R_model, R_post) = KL(N(0, R_post) || N(0, R_model))."""
    R_model = _random_correlation(n, seed=n + 7)
    R_post = _random_correlation(n, seed=n + 21)
    p = torch.distributions.MultivariateNormal(torch.zeros(n, dtype=torch.float64), R_post)
    q = torch.distributions.MultivariateNormal(torch.zeros(n, dtype=torch.float64), R_model)
    expected = float(torch.distributions.kl_divergence(p, q)) / n
    assert math.isclose(gaussian_corr_kl(R_model, R_post), expected, rel_tol=1e-8, abs_tol=1e-10)


def test_scales_with_distance_from_the_target() -> None:
    """Increases monotonically as R_model moves from R_post toward the identity."""
    n = 12
    R_post = _random_correlation(n, seed=3)
    eye = torch.eye(n, dtype=torch.float64)
    vals = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        R = (1 - t) * R_post + t * eye
        vals.append(gaussian_corr_kl(R, R_post))
    assert abs(vals[0]) < 1e-9
    assert all(b > a for a, b in zip(vals, vals[1:])), vals


def test_singular_model_returns_inf_not_an_exception() -> None:
    """A singular R_model returns inf."""
    n = 6
    R_post = _random_correlation(n, seed=11)
    singular = torch.ones(n, n, dtype=torch.float64)  # rank 1, unit diagonal
    assert gaussian_corr_kl(singular, R_post) == float("inf")


def test_accepts_float32_inputs() -> None:
    """float32 inputs give the float64 result."""
    n = 10
    R_post = _random_correlation(n, seed=5)
    R_model = _random_correlation(n, seed=6)
    got32 = gaussian_corr_kl(R_model.float(), R_post.float())
    got64 = gaussian_corr_kl(R_model, R_post)
    assert math.isclose(got32, got64, rel_tol=1e-4, abs_tol=1e-6)
