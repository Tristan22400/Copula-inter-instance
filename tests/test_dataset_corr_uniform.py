"""
test_dataset_corr_uniform.py — Verify structural properties of R_star in a dataset folder.

Since data_gen uses latent=False (posterior over noisy y*, not latent f*), R_star
captures the *observable* correlation — bounded by alpha2/(alpha2+nugget) < 1.
The tests below check properties appropriate for this regime:
  - Well-conditioned (min_eig ≥ 0.01): oracle NLL cannot explode
  - Symmetric: mean ≈ 0, ~50% negative entries
  - Some diversity: positive std, both signs present
  - Unit diagonal: R_star is a proper correlation matrix

Run against a specific folder:
    DATASET_DIR=./data/pit_episodes pytest tests/test_dataset_corr_uniform.py -v

The test is skipped when the folder does not exist or is empty.
Default folder: ./data/pit_cosine-new (matches training config).
"""

from __future__ import annotations

import os
import random

import pytest
import torch
from scipy import stats

_DEFAULT_DIR = "./data/pit_cosine-new"


@pytest.fixture(scope="module")
def dataset_dir():
    return os.environ.get("DATASET_DIR", _DEFAULT_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_EPISODES = 500   # episodes to sample
_SEED = 0


def _load_episodes(folder: str, n: int, seed: int):
    """Load up to *n* episodes; return (off_diag_values, min_eigenvalues)."""
    paths = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".pt")
    )
    rng = random.Random(seed)
    rng.shuffle(paths)
    paths = paths[:n]

    values = []
    min_eigs = []
    for p in paths:
        ep = torch.load(p, map_location="cpu", weights_only=False)
        R = ep["R_star"]          # (N, N)
        N = R.shape[0]
        mask = ~torch.eye(N, dtype=torch.bool)
        values.append(R[mask].flatten())
        min_eigs.append(torch.linalg.eigvalsh(R).min().item())

    return torch.cat(values), min_eigs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def episode_data(dataset_dir):
    if not os.path.isdir(dataset_dir):
        pytest.skip(f"Dataset folder not found: {dataset_dir}")
    pts = [f for f in os.listdir(dataset_dir) if f.endswith(".pt")]
    if len(pts) == 0:
        pytest.skip(f"Dataset folder is empty: {dataset_dir}")
    return _load_episodes(dataset_dir, _N_EPISODES, _SEED)


@pytest.fixture(scope="module")
def off_diag(episode_data):
    return episode_data[0]


@pytest.fixture(scope="module")
def min_eigenvalues(episode_data):
    return episode_data[1]


# ---------------------------------------------------------------------------
# Tests — correlation values
# ---------------------------------------------------------------------------


def test_correlations_have_both_signs(off_diag):
    """Both positive and negative off-diagonal entries must exist."""
    assert off_diag.min().item() < -0.02, (
        f"Min correlation {off_diag.min().item():.3f} — no negative correlations found"
    )
    assert off_diag.max().item() > 0.02, (
        f"Max correlation {off_diag.max().item():.3f} — no positive correlations found"
    )


def test_correlations_mean_near_zero(off_diag):
    """Distribution should be symmetric: mean of off-diagonal R_star ≈ 0."""
    mean = off_diag.mean().item()
    assert abs(mean) < 0.15, f"Mean {mean:.3f} too far from 0 — distribution is skewed"


def test_correlations_negative_fraction(off_diag):
    """Between 25 % and 75 % of off-diagonal entries should be negative."""
    neg_frac = (off_diag < 0).float().mean().item()
    assert neg_frac > 0.25, f"Only {neg_frac:.1%} negative — distribution too positive"
    assert neg_frac < 0.75, f"{neg_frac:.1%} negative — distribution too negative"


def test_correlations_std_nonzero(off_diag):
    """Standard deviation must be positive — not all the same value."""
    std = off_diag.std().item()
    assert std > 0.02, f"Std {std:.3f} essentially zero — all correlations identical"


def test_unit_diagonal(dataset_dir):
    """R_star must have unit diagonal (proper correlation matrix)."""
    pts = sorted(f for f in os.listdir(dataset_dir) if f.endswith(".pt"))[:20]
    for fname in pts:
        ep = torch.load(os.path.join(dataset_dir, fname), map_location="cpu", weights_only=False)
        R = ep["R_star"]
        diag_err = (R.diagonal() - 1.0).abs().max().item()
        assert diag_err < 1e-4, (
            f"{fname}: diagonal of R_star deviates from 1 by {diag_err:.2e}"
        )


# ---------------------------------------------------------------------------
# Tests — numerical conditioning (latent=False guarantee)
# ---------------------------------------------------------------------------


def test_r_star_well_conditioned(min_eigenvalues):
    """All R_star matrices must have minimum eigenvalue ≥ 0.001.

    With latent=False, the nugget noise floor in K_ss prevents R_star from
    being rank-deficient. Near-singular R_star (min_eig ~ 1e-7) causes
    oracle_copula_nll to blow up (z^T R^{-1} z >> 1).
    """
    bad = [v for v in min_eigenvalues if v < 0.001]
    assert len(bad) == 0, (
        f"{len(bad)}/{len(min_eigenvalues)} episodes have min_eig < 0.001; "
        f"smallest: {min(bad):.2e}. R_star is near-singular — check if "
        f"latent=False is in effect in data_gen.generate_gp_task."
    )


def test_r_star_psd(min_eigenvalues):
    """R_star must be positive semi-definite (no negative eigenvalues)."""
    neg = [v for v in min_eigenvalues if v < -1e-5]
    assert len(neg) == 0, (
        f"{len(neg)} episodes have negative min eigenvalue (most negative: {min(neg):.2e})"
    )
