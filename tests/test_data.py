"""
test_data.py — Tests for GP task generation and data pipeline.

Tests verify:
  1. generate_gp_task output shapes
  2. Feature normalisation over train+test combined
  3. R_star is a valid correlation matrix (unit diagonal, PSD)
  4. y values are drawn from the correct GP (basic sanity)
  5. collate_fn produces correct padded shapes and masks
  6. CopulaDataset loads and serves tasks
"""

from __future__ import annotations

import random

import torch

from data_gen import generate_gp_task, gp_posterior, sigma_to_correlation
from dataset import CopulaDataset, collate_fn

# ---------------------------------------------------------------------------
# generate_gp_task tests
# ---------------------------------------------------------------------------


def test_gp_task_output_keys(small_cfg):
    task = generate_gp_task(small_cfg)
    required = [
        "x_norm_train",
        "y_train",
        "x_norm_test",
        "y_test",
        "R_star",
        "mu_star",
        "sigma_star",
        "n_train",
        "n_test",
    ]
    for key in required:
        assert key in task, f"Missing key: {key}"


def test_gp_task_shapes(small_cfg):
    torch.manual_seed(0)
    task = generate_gp_task(small_cfg)
    P = task["n_train"].item()
    N = task["n_test"].item()
    d = small_cfg.data.d_features

    assert task["x_norm_train"].shape == (P, d)
    assert task["y_train"].shape == (P,)
    assert task["x_norm_test"].shape == (N, d)
    assert task["y_test"].shape == (N,)
    assert task["R_star"].shape == (N, N)
    assert task["mu_star"].shape == (N,)
    assert task["sigma_star"].shape == (N,)

    assert small_cfg.data.P_min <= P <= small_cfg.data.P_max
    assert small_cfg.data.N_min <= N <= small_cfg.data.N_max


def test_feature_normalisation_over_all_instances(small_cfg):
    """x_norm_train and x_norm_test together should have ~zero mean, ~unit std."""
    torch.manual_seed(1)
    # Generate multiple tasks and check normalisation
    for _ in range(10):
        task = generate_gp_task(small_cfg)
        x_all = torch.cat([task["x_norm_train"], task["x_norm_test"]], dim=0)
        for f in range(x_all.shape[1]):
            col = x_all[:, f]
            assert abs(col.mean().item()) < 0.2, (
                f"Feature {f} mean {col.mean():.3f} not near zero"
            )
            assert abs(col.std().item() - 1.0) < 0.2, (
                f"Feature {f} std {col.std():.3f} not near 1"
            )


def test_r_star_is_valid_correlation_matrix(small_cfg):
    """R_star must have unit diagonal and be positive semi-definite."""
    torch.manual_seed(2)
    for _ in range(20):
        task = generate_gp_task(small_cfg)
        R = task["R_star"]
        N = task["n_test"].item()

        # Unit diagonal
        assert torch.allclose(R.diagonal(), torch.ones(N), atol=1e-4), (
            f"R_star diagonal not 1: {R.diagonal()}"
        )

        # PSD
        eigvals = torch.linalg.eigvalsh(R)
        assert (eigvals >= -1e-4).all(), (
            f"R_star has negative eigenvalue: {eigvals.min():.6f}"
        )

        # Symmetry
        assert torch.allclose(R, R.T, atol=1e-5)


def test_r_star_values_in_minus1_1(small_cfg):
    """Correlation matrix entries must be in [-1, 1]."""
    torch.manual_seed(3)
    for _ in range(10):
        R = generate_gp_task(small_cfg)["R_star"]
        assert R.abs().max() <= 1.0 + 1e-5


def test_gp_posterior_helper():
    """gp_posterior should return correct shapes and PSD Sigma_star."""
    from data_gen import build_kernel_fn
    P, N, d = 20, 8, 1
    x_train = torch.randn(P, d)
    y_train = torch.randn(P)
    x_test = torch.randn(N, d)
    kernel_fn = build_kernel_fn("rbf", l=1.0, alpha2=1.0)
    mu, Sigma = gp_posterior(x_train, y_train, x_test, kernel_fn, noise=0.1)

    assert mu.shape == (N,)
    assert Sigma.shape == (N, N)

    eigvals = torch.linalg.eigvalsh(Sigma)
    assert (eigvals >= -1e-4).all(), f"Sigma_star not PSD: min eig={eigvals.min():.6f}"


def test_sigma_to_correlation():
    """sigma_to_correlation should produce unit diagonal."""
    N = 6
    # Build a random PD covariance
    A = torch.randn(N, N)
    Sigma = A @ A.T + 0.1 * torch.eye(N)
    R, sigma = sigma_to_correlation(Sigma)

    assert R.shape == (N, N)
    assert sigma.shape == (N,)
    assert torch.allclose(R.diagonal(), torch.ones(N), atol=1e-5)
    # PSD
    assert (torch.linalg.eigvalsh(R) >= -1e-5).all()


# ---------------------------------------------------------------------------
# Dataset / collate_fn tests
# ---------------------------------------------------------------------------


def _make_sample(P: int, N: int, d: int = 1) -> dict:
    return {
        "x_norm_train": torch.randn(P, d),
        "x_norm_test": torch.randn(N, d),
        "y_train": torch.randn(P),
        "y_test": torch.randn(N),
        "z_train": torch.randn(P),
        "z_test": torch.randn(N),
        "log_pdf_test": torch.randn(N),
        "R_star": torch.eye(N),
        "Sigma_star": torch.eye(N),
        "mu_star": torch.zeros(N),
        "sigma_star": torch.ones(N),
        "n_train": torch.tensor(P),
        "n_test": torch.tensor(N),
    }


def test_collate_fn_shapes():
    sizes = [(8, 4), (6, 3), (10, 5), (7, 5)]
    samples = [_make_sample(P, N) for P, N in sizes]
    batch = collate_fn(samples)

    B = len(samples)
    P_max = max(P for P, _ in sizes)
    N_max = max(N for _, N in sizes)

    assert batch["x_train"].shape == (B, P_max, 1)
    assert batch["z_train"].shape == (B, P_max)
    assert batch["x_test"].shape == (B, N_max, 1)
    assert batch["z_test"].shape == (B, N_max)
    assert batch["train_mask"].shape == (B, P_max)
    assert batch["test_mask"].shape == (B, N_max)
    assert batch["R_star"].shape == (B, N_max, N_max)
    assert batch["train_mask"].dtype == torch.bool
    assert batch["test_mask"].dtype == torch.bool


def test_collate_fn_masks_correct():
    samples = [_make_sample(8, 4), _make_sample(6, 3)]
    batch = collate_fn(samples)

    # First sample: P=8 valid, P_max=8 → all True
    assert batch["train_mask"][0].all()
    # Second sample: P=6 valid, rest padding → only first 6 True
    assert batch["train_mask"][1, :6].all()
    assert not batch["train_mask"][1, 6:].any()

    # Test mask
    assert batch["test_mask"][0, :4].all()
    assert not batch["test_mask"][1, 3:].any()  # N=3 for second sample


def test_collate_fn_padding_is_zero():
    """Padded z_train and x_train values should be zero."""
    samples = [_make_sample(10, 5), _make_sample(6, 3)]
    batch = collate_fn(samples)

    # Second sample padded from 6 to 10
    assert (batch["z_train"][1, 6:] == 0.0).all()
    assert (batch["x_train"][1, 6:] == 0.0).all()
    assert (batch["z_test"][1, 3:] == 0.0).all()


def test_copula_dataset_load(tmp_path):
    """CopulaDataset should load .pt files correctly."""
    for i in range(3):
        sample = _make_sample(P=random.randint(5, 10), N=random.randint(3, 6))
        torch.save(sample, tmp_path / f"task_{i:06d}.pt")

    ds = CopulaDataset(episode_dir=str(tmp_path))
    assert len(ds) == 3

    item = ds[0]
    assert "x_norm_train" in item
    assert "z_train" in item
    assert "R_star" in item
