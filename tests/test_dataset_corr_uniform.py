"""
test_dataset_corr_uniform.py — Verify structural properties of R_star in a dataset folder.

data_gen uses the GP *posterior* covariance K_ss + nugget·I − K_st K_ff⁻¹ K_ts as R_star.
This gives:
  - PSD (residual uncertainty after conditioning on training data)
  - Unit diagonal: R_star is a proper correlation matrix
  - Off-diagonal entries smaller than the prior (training data explains part of the correlation)
  - Both positive and negative entries (oscillatory kernels like cosine)

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


def _iter_episodes(folder: str, shuffle_seed: int | None = None):
    """Yield episode dicts from a folder — handles both shard and individual layout.

    Shard files are shuffled (not their contents) before reading, so callers that
    only need the first *n* episodes can stop early without loading every shard —
    important for datasets with hundreds of GB across many shards.
    """
    paths = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".pt") and f != "meta.pt"
    )
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(paths)
    for p in paths:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(obj, list):           # shard: list of episode dicts
            yield from obj
        elif isinstance(obj, dict):         # individual task_*.pt
            yield obj


def _load_episodes(folder: str, n: int, seed: int):
    """Load up to *n* episodes (stopping early); return (off_diag_values, min_eigenvalues)."""
    episodes = []
    for ep in _iter_episodes(folder, shuffle_seed=seed):
        episodes.append(ep)
        if len(episodes) >= n:
            break

    values = []
    min_eigs = []
    for ep in episodes:
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
    """Mean of off-diagonal R_star should not be extreme.

    The cosine prior with l ~ U(1, 10) has a slight positive mean (~0.20) because
    large l values keep 2π*dist/l small (near cos(0)=1). Threshold is 0.30 to
    catch degenerate cases (e.g. all-ones or collapsed kernel) while tolerating
    this natural bias.
    """
    mean = off_diag.mean().item()
    assert abs(mean) < 0.30, f"Mean {mean:.3f} too far from 0 — distribution may be degenerate"


def test_correlations_negative_fraction(off_diag):
    """Between 25 % and 75 % of off-diagonal entries should be negative."""
    neg_frac = (off_diag < 0).float().mean().item()
    assert neg_frac > 0.25, f"Only {neg_frac:.1%} negative — distribution too positive"
    assert neg_frac < 0.75, f"{neg_frac:.1%} negative — distribution too negative"


def test_correlations_std_nonzero(off_diag):
    """Standard deviation must be non-trivial — posterior R_star has residual structure.

    With the GP posterior, off-diagonal correlations are shrunk by the Schur complement
    (training data explains part of the prior correlation). Expected std ≈ 0.1–0.30
    depending on the kernel and the P/N ratio. A value below 0.1 indicates the posterior
    correlations have collapsed near zero (e.g. a rank-limited kernel like dot_product
    with P >> d_features, where training data pins down the signal almost completely).
    """
    std = off_diag.std().item()
    assert std > 0.1, (
        f"Std {std:.4f} too low — posterior R_star correlations appear degenerate."
    )


def test_unit_diagonal(dataset_dir):
    """R_star must have unit diagonal (proper correlation matrix)."""
    if not os.path.isdir(dataset_dir):
        pytest.skip(f"Dataset folder not found: {dataset_dir}")
    episodes = []
    for ep in _iter_episodes(dataset_dir):
        episodes.append(ep)
        if len(episodes) >= 20:
            break
    if not episodes:
        pytest.skip(f"Dataset folder is empty: {dataset_dir}")
    for i, ep in enumerate(episodes):
        R = ep["R_star"]
        diag_err = (R.diagonal() - 1.0).abs().max().item()
        assert diag_err < 1e-4, (
            f"episode[{i}]: diagonal of R_star deviates from 1 by {diag_err:.2e}"
        )


# ---------------------------------------------------------------------------
# Tests — numerical conditioning (latent=False guarantee)
# ---------------------------------------------------------------------------


def test_r_star_well_conditioned(min_eigenvalues):
    """All R_star matrices must have minimum eigenvalue ≥ 0.0001.

    With latent=False, the nugget noise floor in K_ss prevents R_star from
    being rank-deficient. Near-singular R_star (min_eig ~ 1e-7) causes
    oracle_copula_nll to blow up (z^T R^{-1} z >> 1).
    """
    bad = [v for v in min_eigenvalues if v < 0.0001]
    assert len(bad) == 0, (
        f"{len(bad)}/{len(min_eigenvalues)} episodes have min_eig < 0.0001; "
        f"smallest: {min(bad):.2e}. R_star is near-singular — check if "
        f"latent=False is in effect in data_gen.generate_gp_task."
    )


def test_r_star_psd(min_eigenvalues):
    """R_star must be positive semi-definite (no negative eigenvalues)."""
    neg = [v for v in min_eigenvalues if v < -1e-5]
    assert len(neg) == 0, (
        f"{len(neg)} episodes have negative min eigenvalue (most negative: {min(neg):.2e})"
    )
