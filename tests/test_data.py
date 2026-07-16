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

import pytest
import torch
from omegaconf import OmegaConf

from data_gen import ALL_KERNELS, _kernel_needs_scalar_input, generate_gp_task, gp_posterior, sigma_to_correlation
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


# Goldilocks band (mirrors src/diag_kernels.py's Stage-3 thresholds): R_star
# must reflect real dependence — not collapsed toward independence (screening
# effect) and not saturated near +-1 everywhere (trivial task).
_COLLAPSE_THRESHOLD = 0.01
_DEGENERATE_THRESHOLD = 0.95


@pytest.mark.parametrize("kernel_name", ALL_KERNELS)
def test_kernel_goldilocks_and_psd(small_cfg, kernel_name):
    """Every registered kernel must produce a valid, non-trivial R_star.

    One shared test parametrized over every entry in data_gen.ALL_KERNELS,
    rather than a bespoke test per kernel, so newly registered kernels are
    automatically held to the same PSD + Goldilocks bar as the existing
    ones without needing a new test written by hand.
    """
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = kernel_name
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 1 / 3     # (6-4)/6 -> k up to 4
    cfg.data.inactive_frac_max = 5 / 6     # (6-1)/6 -> k down to 1

    torch.manual_seed(abs(hash(kernel_name)) % (2**31))
    off_diag_abs = []
    for _ in range(20):
        task = generate_gp_task(cfg)
        R = task["R_star"]
        N = R.shape[0]

        assert torch.allclose(R.diagonal(), torch.ones(N), atol=1e-4), (
            f"{kernel_name}: diagonal not 1: {R.diagonal()}"
        )
        assert torch.allclose(R, R.T, atol=1e-5), f"{kernel_name}: not symmetric"
        eigvals = torch.linalg.eigvalsh(R)
        assert (eigvals >= -1e-4).all(), (
            f"{kernel_name}: not PSD (min eig={eigvals.min():.6f})"
        )
        assert R.abs().max() <= 1.0 + 1e-5, f"{kernel_name}: value outside [-1, 1]"

        mask = ~torch.eye(N, dtype=torch.bool)
        off_diag_abs.append(R[mask].abs())

    mean_abs_r = torch.cat(off_diag_abs).mean().item()
    assert mean_abs_r > _COLLAPSE_THRESHOLD, (
        f"{kernel_name}: screening effect, mean|r*_offdiag|={mean_abs_r:.4f}"
    )
    assert mean_abs_r < _DEGENERATE_THRESHOLD, (
        f"{kernel_name}: degenerate/trivial, mean|r*_offdiag|={mean_abs_r:.4f}"
    )


def test_kernel_needs_scalar_input_handles_n_way_chains():
    """Regression test: _kernel_needs_scalar_input used to route through
    _parse_composite, which only handles exactly 2 parts via .partition() —
    for a 3-way systematic-composition chain like "rbf+cosine*periodic",
    that mis-parsed as non-composite and silently returned False even though
    cosine (scalar-only) is present. The generic re.split-based
    implementation must catch cosine anywhere in the chain, regardless of
    position or chain length."""
    assert _kernel_needs_scalar_input("rbf+cosine*periodic") is True
    assert _kernel_needs_scalar_input("periodic*matern32+cosine") is True
    assert _kernel_needs_scalar_input("rbf+periodic*matern32") is False
    # Existing base-kernel / 2-way-composite behaviour must be unchanged.
    assert _kernel_needs_scalar_input("cosine") is True
    assert _kernel_needs_scalar_input("rbf") is False
    assert _kernel_needs_scalar_input("rbf+cosine") is True
    assert _kernel_needs_scalar_input("rbf+periodic") is False


def test_systematic_composition_goldilocks_and_psd(small_cfg):
    """cfg.data.systematic_composition=True (CauKer-style chain sampling)
    must produce a valid R_star on every draw, same hard invariants as
    test_kernel_goldilocks_and_psd. Not ALL_KERNELS-parametrized (chain
    names are sampled at runtime, unbounded cardinality) and only keeps the
    _COLLAPSE_THRESHOLD lower-bound Goldilocks check — the upper
    (_DEGENERATE_THRESHOLD) bound is expected to trip legitimately for
    short/product-heavy chains and would be flaky here."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.systematic_composition = True
    cfg.data.composite_num_kernels_min = 1
    cfg.data.composite_num_kernels_max = 3
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 1 / 3
    cfg.data.inactive_frac_max = 5 / 6

    torch.manual_seed(123)
    off_diag_abs = []
    for _ in range(20):
        task = generate_gp_task(cfg)
        R = task["R_star"]
        N = R.shape[0]

        assert torch.allclose(R.diagonal(), torch.ones(N), atol=1e-4), (
            f"{task['kernel']}: diagonal not 1: {R.diagonal()}"
        )
        assert torch.allclose(R, R.T, atol=1e-5), f"{task['kernel']}: not symmetric"
        eigvals = torch.linalg.eigvalsh(R)
        assert (eigvals >= -1e-4).all(), (
            f"{task['kernel']}: not PSD (min eig={eigvals.min():.6f})"
        )
        assert R.abs().max() <= 1.0 + 1e-5, f"{task['kernel']}: value outside [-1, 1]"

        mask = ~torch.eye(N, dtype=torch.bool)
        off_diag_abs.append(R[mask].abs())

    mean_abs_r = torch.cat(off_diag_abs).mean().item()
    assert mean_abs_r > _COLLAPSE_THRESHOLD, (
        f"systematic_composition: screening effect, mean|r*_offdiag|={mean_abs_r:.4f}"
    )


_ARD_ELIGIBLE_KERNELS = ["rbf", "matern32", "rational_quadratic", "periodic"]


@pytest.mark.parametrize("kernel_name", _ARD_ELIGIBLE_KERNELS)
def test_ard_samples_per_dimension_lengthscale(small_cfg, kernel_name):
    """cfg.data.ard=True gives an ARD lengthscale vector (k,) instead of a
    shared isotropic scalar, and the analytical-PIT kernel reconstruction
    (pit.gp_analytical_pit -> data_gen.build_kernel_fn) round-trips it
    correctly (matches the cached _L_ff/_alpha result from generation)."""
    from pit import gp_analytical_pit

    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = kernel_name
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 0.5    # (6-3)/6 -> fixed k=3
    cfg.data.inactive_frac_max = 0.5
    cfg.data.ard = True

    # "periodic" is always capped to k=1 active dims (see data_gen.py), so
    # its ARD vector squeezes to a plain scalar, same as the non-ARD case.
    expected_shape = () if kernel_name == "periodic" else (3,)
    torch.manual_seed(abs(hash("ard_" + kernel_name)) % (2**31))
    task = generate_gp_task(cfg)
    assert task["l"].shape == expected_shape, (
        f"{kernel_name}: expected shape {expected_shape}, got {tuple(task['l'].shape)}"
    )

    cached = gp_analytical_pit(task)
    reconstructed_task = {k: v for k, v in task.items() if k not in ("_L_ff", "_alpha")}
    reconstructed = gp_analytical_pit(reconstructed_task)
    assert torch.allclose(cached["z_train"], reconstructed["z_train"], atol=1e-3)
    assert torch.allclose(cached["z_test"], reconstructed["z_test"], atol=1e-3)


def test_ard_default_false_keeps_isotropic_lengthscale(small_cfg):
    """Without cfg.data.ard, lengthscale stays a shared scalar even for k>1
    (unchanged pre-ARD behaviour)."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 0.5    # (6-3)/6 -> fixed k=3
    cfg.data.inactive_frac_max = 0.5

    torch.manual_seed(0)
    task = generate_gp_task(cfg)
    assert task["l"].shape == (), f"expected isotropic scalar, got shape {tuple(task['l'].shape)}"


def test_ard_not_applied_to_cosine_or_dot_product(small_cfg):
    """cfg.data.ard=True is a silent no-op for kernels where ARD isn't
    structurally possible ("cosine": gpytorch hardcodes period_length to a
    scalar) or not applicable ("dot_product": no lengthscale)."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 0.5    # (6-3)/6 -> fixed k=3
    cfg.data.inactive_frac_max = 0.5
    cfg.data.ard = True

    torch.manual_seed(0)
    cfg.data.kernel = "cosine"
    task = generate_gp_task(cfg)
    assert task["l"].shape == (), "cosine's period_length must stay scalar under ard=True"

    torch.manual_seed(0)
    cfg.data.kernel = "dot_product"
    task = generate_gp_task(cfg)  # must not raise
    assert task["alpha2"].numel() == 1


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
