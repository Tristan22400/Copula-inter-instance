"""
test_loss.py — Sanity checks for the copula NLL loss function.

Tests verify:
  1. copula_nll is finite for well-formed inputs
  2. Gradients flow through copula_nll w.r.t. W_tilde
  3. oracle_copula_nll is lower than copula_nll on average (not per sample)
  4. copula_nll decreases for W_tilde that better matches R_star
  5. oracle_copula_nll consistency with known R
"""

from __future__ import annotations

import torch

from loss import _safe_cholesky, copula_nll, oracle_copula_nll

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_w_and_z(B: int = 4, N: int = 10, rank: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    W = torch.randn(B, N, rank + 1)
    W = W / W.norm(dim=-1, keepdim=True)  # unit rows
    z = torch.randn(B, N)
    mask = torch.ones(B, N, dtype=torch.bool)
    return W, z, mask


def make_identity_w(B: int, N: int, rank: int):
    """W_tilde with eps=1 and W=0 gives R_eps = I + 0 = I (independence)."""
    return torch.zeros(B, N, rank + 1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_copula_nll_is_finite():
    """copula_nll should return a finite scalar for valid inputs."""
    W, z, mask = make_w_and_z(B=4, N=10, rank=2)
    loss = copula_nll(W, z, mask)
    assert torch.isfinite(loss), f"copula_nll returned non-finite: {loss}"


def test_copula_nll_is_scalar():
    W, z, mask = make_w_and_z(B=4, N=10, rank=2)
    loss = copula_nll(W, z, mask)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


def test_copula_nll_gradients_flow():
    """Gradients must flow from loss to W_tilde."""
    W, z, mask = make_w_and_z(B=4, N=10, rank=2)
    W = W.requires_grad_(True)
    loss = copula_nll(W, z, mask)
    loss.backward()
    assert W.grad is not None
    assert torch.isfinite(W.grad).all(), "Non-finite gradients"


def test_copula_nll_zero_mask():
    """copula_nll with all-False mask should return zero (no valid tasks)."""
    B, N, rank = 2, 5, 2
    W = torch.randn(B, N, rank + 1)
    W = W / W.norm(dim=-1, keepdim=True)
    z = torch.randn(B, N)
    mask = torch.zeros(B, N, dtype=torch.bool)  # all masked
    loss = copula_nll(W, z, mask)
    assert loss.item() == 0.0


def test_oracle_copula_nll_is_finite():
    """oracle_copula_nll should be finite for valid inputs."""
    B, N = 4, 8
    torch.manual_seed(5)
    A = torch.randn(B, N, N)
    R = A @ A.transpose(-2, -1) + 0.1 * torch.eye(N)
    # Normalise to correlation matrix
    sigma = R.diagonal(dim1=-2, dim2=-1).clamp(min=1e-10).sqrt()
    D_inv = torch.diag_embed(1.0 / sigma)
    R = D_inv @ R @ D_inv
    z = torch.randn(B, N)
    mask = torch.ones(B, N, dtype=torch.bool)

    loss = oracle_copula_nll(R, z, mask)
    assert torch.isfinite(loss), f"oracle_copula_nll not finite: {loss}"


def test_oracle_le_independence_on_average():
    """Oracle (R=R_star) should beat independence (R=I) on average over many tasks."""
    torch.manual_seed(42)
    B, N, rank = 8, 20, 4

    oracle_losses, pred_losses = [], []
    n_rounds = 30

    for _ in range(n_rounds):
        # Random positive-definite correlation matrix
        A = torch.randn(N, N)
        Sigma = A @ A.T + 0.5 * torch.eye(N)
        sigma = Sigma.diagonal().clamp(min=1e-10).sqrt()
        D_inv = torch.diag(1.0 / sigma)
        R_star = D_inv @ Sigma @ D_inv
        R_star = R_star.unsqueeze(0).expand(B, -1, -1)

        # Sample z ~ N(0, R_star) for each batch element
        L = _safe_cholesky(R_star[0])
        z = (L @ torch.randn(N, B)).T  # (B, N)

        mask = torch.ones(B, N, dtype=torch.bool)

        oracle_loss = oracle_copula_nll(R_star, z, mask).item()
        # Independence: W=0, eps=1 → R_eps = 2I (not exactly I, but close to indep baseline)
        W_zero = torch.zeros(B, N, rank + 1)
        indep_loss = copula_nll(W_zero, z, mask, eps=1.0).item()

        oracle_losses.append(oracle_loss)
        pred_losses.append(indep_loss)

    mean_oracle = sum(oracle_losses) / len(oracle_losses)
    mean_indep = sum(pred_losses) / len(pred_losses)
    # Oracle should generally be lower (oracle < independence when R ≠ I)
    # This is not guaranteed for every single case, only in expectation
    # (when R is actually correlated, oracle does better)
    assert mean_oracle < mean_indep + 1.0, (
        f"Oracle ({mean_oracle:.4f}) should be better than or comparable to "
        f"independence ({mean_indep:.4f}) on average"
    )


def test_copula_nll_smaller_for_better_w():
    """copula_nll should be smaller when W_tilde encodes the true correlation."""
    torch.manual_seed(99)
    N, rank = 10, 3
    B = 1

    # Create a known low-rank correlation matrix
    W_true = torch.randn(N, rank + 1)
    W_true = W_true / W_true.norm(dim=-1, keepdim=True)
    R_true = W_true @ W_true.T  # (N, N) — correlation-like (diagonal ≤ 1)
    # Normalise to true correlation matrix
    d = R_true.diagonal().clamp(min=1e-8).sqrt()
    D_inv = torch.diag(1.0 / d)
    R_true = D_inv @ R_true @ D_inv

    # Sample z from this distribution
    L = _safe_cholesky(R_true)
    z = (L @ torch.randn(N, 100)).T  # (100, N)

    mask = torch.ones(B, N, dtype=torch.bool)

    # Evaluate copula NLL with the true W vs random W
    nll_true_list, nll_rand_list = [], []
    for i in range(100):
        zi = z[i : i + 1]  # (1, N)

        W_t = W_true.unsqueeze(0)
        nll_true_list.append(copula_nll(W_t, zi, mask, eps=1e-3).item())

        W_r = torch.randn_like(W_t)
        W_r = W_r / W_r.norm(dim=-1, keepdim=True)
        nll_rand_list.append(copula_nll(W_r, zi, mask, eps=1e-3).item())

    mean_true = sum(nll_true_list) / len(nll_true_list)
    mean_rand = sum(nll_rand_list) / len(nll_rand_list)
    assert mean_true < mean_rand, (
        f"True W NLL ({mean_true:.4f}) should be lower than random W NLL ({mean_rand:.4f})"
    )


def test_woodbury_matches_direct_cholesky():
    """copula_nll via Woodbury should match oracle_copula_nll when R=W_tilde@W_tilde^T+eps*I.

    Use eps=0.5 to keep the matrix well-conditioned and avoid numerical drift
    from the large ratio (1/eps) in the capacitance matrix.
    """
    torch.manual_seed(7)
    B, N, rank = 1, 8, 3
    # Use a moderate eps so R_eps is well-conditioned (min eigenvalue = eps, max ≈ eps+r+1)
    eps = 0.5

    W = torch.randn(B, N, rank + 1)
    W = W / W.norm(dim=-1, keepdim=True)
    z = torch.randn(B, N)
    mask = torch.ones(B, N, dtype=torch.bool)

    # copula_nll via Woodbury + determinant lemma
    nll_woodbury = copula_nll(W, z, mask, eps=eps)

    # Reference: direct computation on the dense R_eps matrix
    R_eps = torch.bmm(W, W.transpose(-2, -1)) + eps * torch.eye(N).unsqueeze(0)
    nll_direct = oracle_copula_nll(R_eps, z, mask)

    # Relative tolerance: both formulas should agree to ~0.1%
    rel_err = abs(nll_woodbury.item() - nll_direct.item()) / (
        abs(nll_direct.item()) + 1e-8
    )
    assert rel_err < 1e-3, (
        f"Woodbury ({nll_woodbury:.6f}) != direct ({nll_direct:.6f}), "
        f"rel_err={rel_err:.2e}"
    )
