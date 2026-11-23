"""grid_score: the oracle NLL it scores against."""

from __future__ import annotations

import torch

from eval.runners.grid_score import oracle_nll


def test_grid_oracle_nll_is_the_gaussian_nll_per_row() -> None:
    torch.manual_seed(0)
    N = 6
    A = torch.randn(N, N, dtype=torch.float64)
    C = A @ A.T + N * torch.eye(N, dtype=torch.float64)
    sigma = C.diagonal().sqrt()
    R = C / (sigma[:, None] * sigma[None, :])
    mu, y = torch.randn(N, dtype=torch.float64), torch.randn(N, dtype=torch.float64)
    expected = -torch.distributions.MultivariateNormal(mu, C).log_prob(y) / N
    assert torch.allclose(oracle_nll(y[None], mu[None], sigma[None], R[None])[0], expected)
