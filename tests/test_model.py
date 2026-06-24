"""
test_model.py — Structural property tests for the Copula Transformer.

Tests verify:
  1. Output shape
  2. Unit row norms on W_tilde (→ unit diagonal on R)
  3. R = W W^T is PSD
  4. Test-instance permutation equivariance
  5. Train-instance permutation invariance
  6. ICL attention mask blocks test-position attention
"""

from __future__ import annotations

import pytest
import torchs
from conftest import make_batch

from model import build_copula_transformer, build_icl_mask

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_and_cfg(small_cfg):
    torch.manual_seed(0)
    model = build_copula_transformer(small_cfg)
    model.eval()
    return model, small_cfg


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def permute_test(batch: dict, perm: list) -> dict:
    """Return a new batch with test instances reordered by perm."""
    b = {k: v.clone() for k, v in batch.items()}
    b["x_test"] = b["x_test"][:, perm]
    b["z_test"] = b["z_test"][:, perm]
    b["test_mask"] = b["test_mask"][:, perm]
    return b


def permute_train(batch: dict, perm: list) -> dict:
    """Return a new batch with train instances reordered by perm."""
    b = {k: v.clone() for k, v in batch.items()}
    b["x_train"] = b["x_train"][:, perm]
    b["z_train"] = b["z_train"][:, perm]
    b["train_mask"] = b["train_mask"][:, perm]
    return b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_output_shape(model_and_cfg):
    model, cfg = model_and_cfg
    B, P, N = 2, 10, 5
    batch = make_batch(B=B, P=P, N=N)
    with torch.no_grad():
        W = model(batch)
    rank = cfg.model.rank
    assert W.shape == (B, N, rank + 1), (
        f"Expected ({B}, {N}, {rank + 1}), got {W.shape}"
    )


def test_unit_row_norms(model_and_cfg):
    """W_tilde rows must have unit norm (so R_ii = 1 by construction)."""
    model, _ = model_and_cfg
    batch = make_batch(B=2, P=10, N=5)
    with torch.no_grad():
        W = model(batch)
    norms = W.norm(dim=-1)  # (B, N)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"Row norms not 1: min={norms.min():.6f}, max={norms.max():.6f}"
    )


def test_psd_correlation_matrix(model_and_cfg):
    """R = W_tilde @ W_tilde^T must be PSD (all eigenvalues >= 0)."""
    model, _ = model_and_cfg
    batch = make_batch(B=2, P=10, N=5)
    with torch.no_grad():
        W = model(batch)
    R = torch.bmm(W, W.transpose(-2, -1))  # (B, N, N)
    for b in range(R.shape[0]):
        eigvals = torch.linalg.eigvalsh(R[b])
        assert (eigvals >= -1e-5).all(), (
            f"Batch {b}: negative eigenvalues: {eigvals[eigvals < 0]}"
        )


def test_unit_diagonal(model_and_cfg):
    """Diagonal of R = W W^T should be 1 since ||w̃_j|| = 1."""
    model, _ = model_and_cfg
    batch = make_batch(B=2, P=10, N=5)
    with torch.no_grad():
        W = model(batch)
    R = torch.bmm(W, W.transpose(-2, -1))
    diag = R.diagonal(dim1=-2, dim2=-1)  # (B, N)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5), (
        f"Diagonal not 1: {diag}"
    )


def test_permutation_equivariance_test_instances(model_and_cfg):
    """Permuting test instances should permute W_tilde rows by same permutation."""
    model, _ = model_and_cfg
    torch.manual_seed(42)
    batch = make_batch(B=2, P=10, N=5)
    perm = [2, 0, 4, 1, 3]

    with torch.no_grad():
        W1 = model(batch)
        W2 = model(permute_test(batch, perm))

    assert torch.allclose(W1[:, perm], W2, atol=1e-4), (
        f"Max diff: {(W1[:, perm] - W2).abs().max():.6f}"
    )


def test_permutation_invariance_train_instances(model_and_cfg):
    """Permuting train instances should not change the output W_tilde."""
    model, _ = model_and_cfg
    torch.manual_seed(42)
    batch = make_batch(B=2, P=10, N=5)
    perm = list(torch.randperm(10).numpy())

    with torch.no_grad():
        W1 = model(batch)
        W2 = model(permute_train(batch, perm))

    assert torch.allclose(W1, W2, atol=1e-4), (
        f"Max diff after train permutation: {(W1 - W2).abs().max():.6f}"
    )


def test_icl_mask_structure():
    """build_icl_mask should block all attention to test positions."""
    P, N = 10, 5
    mask = build_icl_mask(P, N, device="cpu")  # (T, T) float
    T = P + N

    # Everything attending to test positions (cols P..T-1) must be -inf
    assert torch.all(mask[:, P:] == float("-inf")), (
        "Mask should block all tokens from attending to test positions"
    )

    # Attention to train positions (cols 0..P-1) must be 0 (allowed)
    assert torch.all(mask[:, :P] == 0.0), (
        "Mask should allow all tokens to attend to train positions"
    )

    assert mask.shape == (T, T)


def test_forward_with_padding(model_and_cfg):
    """Model should handle batches with different P and N (via padding)."""
    from dataset import collate_fn

    model, _ = model_and_cfg
    torch.manual_seed(7)

    # Manually build padded batch with two different sizes
    samples = [
        {
            "x_norm_train": torch.randn(8, 1),
            "z_train": torch.randn(8),
            "x_norm_test": torch.randn(4, 1),
            "z_test": torch.randn(4),
            "R_star": torch.eye(4),
            "mu_star": torch.zeros(4),
            "sigma_star": torch.ones(4),
            "n_train": torch.tensor(8),
            "n_test": torch.tensor(4),
        },
        {
            "x_norm_train": torch.randn(6, 1),
            "z_train": torch.randn(6),
            "x_norm_test": torch.randn(3, 1),
            "z_test": torch.randn(3),
            "R_star": torch.eye(3),
            "mu_star": torch.zeros(3),
            "sigma_star": torch.ones(3),
            "n_train": torch.tensor(6),
            "n_test": torch.tensor(3),
        },
    ]
    batch = collate_fn(samples)

    with torch.no_grad():
        W = model(batch)

    # W shape: (2, N_max=4, rank+1)
    assert W.shape[0] == 2
    # Row norms should still be 1 for all positions (including padded)
    norms = W.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
