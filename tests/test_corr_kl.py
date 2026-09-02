"""test_corr_kl.py — pins pit.gaussian_corr_kl, the correlation-only
divergence with a true zero floor.

oracle_diag/gap_nll is a Monte-Carlo estimate of KL(true posterior || model
predictive) from ONE realized y_test: noisy, and negative per-episode often
enough that only its mean is interpretable. gaussian_corr_kl is the noise-free
companion -- a functional of the two correlation matrices alone -- so it is
exactly zero iff the predicted correlation IS the posterior correlation, which
makes it the metric that says whether the copula head has converged rather than
just how the last draw scored.

Cases: the zero floor, strict positivity off it, agreement with torch's own
KL for multivariate normals (an independent implementation), and the
non-raising failure mode on a singular input.
"""

from __future__ import annotations

import math

import pytest
import torch

from pit import gaussian_corr_kl


def _random_correlation(n, seed, rank=None):
    """A random PD correlation matrix via a low-rank + diagonal factor,
    normalized to unit diagonal -- the same construction shape
    model.low_rank_correlation produces."""
    g = torch.Generator().manual_seed(seed)
    r = rank or max(2, n // 2)
    W = torch.randn(n, r, generator=g, dtype=torch.float64)
    S = W @ W.T + torch.diag(torch.rand(n, generator=g, dtype=torch.float64) + 0.5)
    d = S.diagonal().sqrt()
    return S / torch.outer(d, d)


@pytest.mark.parametrize("n", [3, 8, 25])
def test_zero_iff_identical(n):
    """The floor: KL(R || R) == 0 exactly."""
    R = _random_correlation(n, seed=n)
    assert abs(gaussian_corr_kl(R, R)) < 1e-9


@pytest.mark.parametrize("n", [3, 8, 25])
def test_strictly_positive_when_different(n):
    """Off the floor it is strictly positive, in both argument orders (KL is
    not symmetric, but both directions are still > 0)."""
    A = _random_correlation(n, seed=n)
    B = _random_correlation(n, seed=n + 100)
    assert gaussian_corr_kl(A, B) > 1e-6
    assert gaussian_corr_kl(B, A) > 1e-6


@pytest.mark.parametrize("n", [4, 12])
def test_matches_torch_kl_divergence(n):
    """Cross-check against torch.distributions -- an independent
    implementation of the same quantity. Note the argument order:
    gaussian_corr_kl(R_model, R_post) is KL(N(0,R_post) || N(0,R_model)),
    i.e. the TRUE distribution first, matching gap_nll's orientation.
    """
    R_model = _random_correlation(n, seed=n + 7)
    R_post = _random_correlation(n, seed=n + 21)
    p = torch.distributions.MultivariateNormal(torch.zeros(n, dtype=torch.float64), R_post)
    q = torch.distributions.MultivariateNormal(torch.zeros(n, dtype=torch.float64), R_model)
    expected = float(torch.distributions.kl_divergence(p, q)) / n
    assert math.isclose(gaussian_corr_kl(R_model, R_post), expected, rel_tol=1e-8, abs_tol=1e-10)


def test_scales_with_distance_from_the_target():
    """Interpolating R_model from R_post toward the identity increases the
    divergence monotonically -- the property that makes it readable as a
    training curve."""
    n = 12
    R_post = _random_correlation(n, seed=3)
    eye = torch.eye(n, dtype=torch.float64)
    vals = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        R = (1 - t) * R_post + t * eye
        vals.append(gaussian_corr_kl(R, R_post))
    assert abs(vals[0]) < 1e-9
    assert all(b > a for a, b in zip(vals, vals[1:])), vals


def test_singular_model_returns_inf_not_an_exception():
    """One numerically degenerate episode must not take down a validation
    pass -- same policy as episode_posterior_ceiling's None return."""
    n = 6
    R_post = _random_correlation(n, seed=11)
    singular = torch.ones(n, n, dtype=torch.float64)   # rank 1, unit diagonal
    assert gaussian_corr_kl(singular, R_post) == float("inf")


def test_accepts_float32_inputs():
    """validate() holds Sigma in float32; the function must promote internally
    rather than lose the log-det to single precision."""
    n = 10
    R_post = _random_correlation(n, seed=5)
    R_model = _random_correlation(n, seed=6)
    got32 = gaussian_corr_kl(R_model.float(), R_post.float())
    got64 = gaussian_corr_kl(R_model, R_post)
    assert math.isclose(got32, got64, rel_tol=1e-4, abs_tol=1e-6)
