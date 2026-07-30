"""test_model.py — Structural property tests for CopulaTabICL.

Tests verify:
  1. Output shapes of (W, s)
  2. low_rank_correlation(W, s) produces a unit-diagonal, PSD Sigma
  3. Test-instance permutation equivariance
  4. Train-instance permutation invariance
  5. Test instances are independent of one another (no cross-leakage)
  6. Forward pass handles padded batches (variable P/N per sample)
"""

from __future__ import annotations

import pytest
import torch
from conftest import make_batch

from model import build_copula_transformer, low_rank_correlation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_and_cfg(small_model_cfg):
    """A scratch CopulaTabICL, warm-started with one optimizer step.

    TabICL's attention/FF output projections are zero-initialized (a
    ReZero-style stability trick, see tabicl_upstream _model/layers.py
    MultiheadAttentionBlock.init_weights), so a freshly-constructed model is
    an exact identity map: every attention block collapses to its residual
    input, and row/test representations come out identical regardless of
    per-row content. One dummy gradient step moves the projections off of
    that degenerate fixed point so the structural properties below actually
    probe the architecture instead of a constant function.
    """
    torch.manual_seed(0)
    model = build_copula_transformer(small_model_cfg)
    model.train()  # eval() would route through TabICL's inference manager,
    # which auto-selects a CUDA execution device even for this CPU-only
    # scratch model whenever CUDA is available on the host.

    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    warm_batch = make_batch(B=2, P=10, N=5)
    out = model(warm_batch)
    loss = out["W"].pow(2).sum() + out["s"].pow(2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()

    return model, small_model_cfg


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
        out = model(batch)
    rank = cfg.model.rank
    assert out["W"].shape == (B, N, rank), (
        f"Expected W {(B, N, rank)}, got {out['W'].shape}"
    )
    assert out["s"].shape == (B, N), (
        f"Expected s {(B, N)}, got {out['s'].shape}"
    )


def test_correlation_unit_diagonal(model_and_cfg):
    """low_rank_correlation(W, s) must have Sigma_ii == 1 (up to jitter)."""
    model, _ = model_and_cfg
    batch = make_batch(B=2, P=10, N=5)
    with torch.no_grad():
        out = model(batch)
        Sigma = low_rank_correlation(out["W"], out["s"], batch["test_mask"])
    diag = Sigma.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-3), (
        f"Diagonal not 1: {diag}"
    )


def test_correlation_is_psd(model_and_cfg):
    """low_rank_correlation(W, s) must be PSD (all eigenvalues >= 0)."""
    model, _ = model_and_cfg
    batch = make_batch(B=2, P=10, N=5)
    with torch.no_grad():
        out = model(batch)
        Sigma = low_rank_correlation(out["W"], out["s"], batch["test_mask"])
    for b in range(Sigma.shape[0]):
        eigvals = torch.linalg.eigvalsh(Sigma[b])
        assert (eigvals >= -1e-4).all(), (
            f"Batch {b}: negative eigenvalues: {eigvals[eigvals < 0]}"
        )


def test_permutation_equivariance_test_instances(model_and_cfg):
    """Permuting test instances should permute (W, s) rows by the same permutation."""
    model, _ = model_and_cfg
    torch.manual_seed(42)
    batch = make_batch(B=2, P=10, N=5)
    perm = [2, 0, 4, 1, 3]

    with torch.no_grad():
        out1 = model(batch)
        out2 = model(permute_test(batch, perm))

    assert torch.allclose(out1["W"][:, perm], out2["W"], atol=1e-4), (
        f"W max diff: {(out1['W'][:, perm] - out2['W']).abs().max():.6f}"
    )
    assert torch.allclose(out1["s"][:, perm], out2["s"], atol=1e-4), (
        f"s max diff: {(out1['s'][:, perm] - out2['s']).abs().max():.6f}"
    )


def test_permutation_invariance_train_instances(model_and_cfg):
    """Permuting train instances should not change the output (W, s)."""
    model, _ = model_and_cfg
    torch.manual_seed(42)
    batch = make_batch(B=2, P=10, N=5)
    perm = list(torch.randperm(10).numpy())

    with torch.no_grad():
        out1 = model(batch)
        out2 = model(permute_train(batch, perm))

    assert torch.allclose(out1["W"], out2["W"], atol=1e-4), (
        f"W max diff after train permutation: {(out1['W'] - out2['W']).abs().max():.6f}"
    )
    assert torch.allclose(out1["s"], out2["s"], atol=1e-4), (
        f"s max diff after train permutation: {(out1['s'] - out2['s']).abs().max():.6f}"
    )


def test_test_instances_are_independent(model_and_cfg):
    """Perturbing one test instance's input must not change any other test
    instance's output — the ICL stage must not let test rows attend to
    each other."""
    model, _ = model_and_cfg
    torch.manual_seed(7)
    batch = make_batch(B=2, P=10, N=5)

    batch_perturbed = {k: v.clone() for k, v in batch.items()}
    batch_perturbed["x_test"][:, 0] += 100.0

    with torch.no_grad():
        out1 = model(batch)
        out2 = model(batch_perturbed)

    w_diff = (out1["W"] - out2["W"]).abs().sum(dim=-1)  # (B, N)
    s_diff = (out1["s"] - out2["s"]).abs()               # (B, N)

    # The perturbed instance (index 0) is expected to change.
    assert (w_diff[:, 0] > 1e-6).all() or (s_diff[:, 0] > 1e-6).all(), (
        "Perturbing test instance 0 should change its own output"
    )
    # No other test instance may be affected.
    assert torch.allclose(w_diff[:, 1:], torch.zeros_like(w_diff[:, 1:]), atol=1e-6), (
        f"Perturbing test instance 0 leaked into other test instances: {w_diff[:, 1:]}"
    )
    assert torch.allclose(s_diff[:, 1:], torch.zeros_like(s_diff[:, 1:]), atol=1e-6), (
        f"Perturbing test instance 0 leaked into other test instances: {s_diff[:, 1:]}"
    )


def test_forward_with_padding(model_and_cfg):
    """Model should handle batches with different P and N per sample (via padding)."""
    model, _ = model_and_cfg
    torch.manual_seed(3)

    d_x = 1
    P_max, N_max = 8, 4
    x_train = torch.zeros(2, P_max, d_x)
    z_train = torch.zeros(2, P_max)
    x_test = torch.zeros(2, N_max, d_x)
    z_test = torch.zeros(2, N_max)
    train_mask = torch.zeros(2, P_max, dtype=torch.bool)
    test_mask = torch.zeros(2, N_max, dtype=torch.bool)

    # Sample 0: full P=8, N=4. Sample 1: P=6, N=3 (padded to P_max/N_max).
    x_train[0] = torch.randn(P_max, d_x)
    z_train[0] = torch.randn(P_max)
    train_mask[0] = True
    x_train[1, :6] = torch.randn(6, d_x)
    z_train[1, :6] = torch.randn(6)
    train_mask[1, :6] = True

    x_test[0] = torch.randn(N_max, d_x)
    z_test[0] = torch.randn(N_max)
    test_mask[0] = True
    x_test[1, :3] = torch.randn(3, d_x)
    z_test[1, :3] = torch.randn(3)
    test_mask[1, :3] = True

    batch = {
        "x_train": x_train, "z_train": z_train,
        "x_test": x_test, "z_test": z_test,
        "train_mask": train_mask, "test_mask": test_mask,
        "n_train": torch.tensor([8, 6]), "n_test": torch.tensor([4, 3]),
    }

    with torch.no_grad():
        out = model(batch)

    rank = 2
    assert out["W"].shape == (2, N_max, rank)
    assert out["s"].shape == (2, N_max)
    assert torch.isfinite(out["W"]).all()
    assert torch.isfinite(out["s"]).all()
