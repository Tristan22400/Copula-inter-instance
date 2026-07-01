"""test_dataset_corr_nonneg.py — Verify R_star structure for non-negative kernels
(rbf, matern32, rational_quadratic, periodic, dot_product).

Unlike cosine (oscillatory, tests/test_dataset_corr_uniform.py), these kernels have
a non-negative prior: K(x1, x2) >= 0 everywhere. Posterior conditioning on training
data can still push a handful of entries slightly negative (Simpson's-paradox-style
explaining-away), but the bulk of R_star should be non-negative and spread across
[0, 1] rather than collapsed near 0 (no structure) or saturated near 1 (near-singular).

Run against a specific folder:
    DATASET_DIR=./data/rbf-posterior-tuned/pit pytest tests/test_dataset_corr_nonneg.py -v

The test is skipped when the folder does not exist or is empty.
"""

from __future__ import annotations

import os
import random

import pytest
import torch

_DEFAULT_DIR = "./data/rbf-posterior-tuned/pit"


@pytest.fixture(scope="module")
def dataset_dir():
    return os.environ.get("DATASET_DIR", _DEFAULT_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_EPISODES = 500
_SEED = 0


def _iter_episodes(folder: str):
    paths = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".pt") and f != "meta.pt"
    )
    for p in paths:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(obj, list):
            yield from obj
        elif isinstance(obj, dict):
            yield obj


def _load_episodes(folder: str, n: int, seed: int):
    all_eps = list(_iter_episodes(folder))
    rng = random.Random(seed)
    rng.shuffle(all_eps)
    episodes = all_eps[:n]

    values = []
    min_eigs = []
    for ep in episodes:
        R = ep["R_star"]
        N = R.shape[0]
        mask = ~torch.eye(N, dtype=torch.bool)
        values.append(R[mask].flatten())
        min_eigs.append(torch.linalg.eigvalsh(R).min().item())

    return torch.cat(values), min_eigs


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


def test_correlations_mostly_nonnegative(off_diag):
    """A non-negative-kernel prior should leave only a small negative tail after conditioning."""
    neg_frac = (off_diag < -0.01).float().mean().item()
    assert neg_frac < 0.15, (
        f"{neg_frac:.1%} of entries are negative — too much for a non-negative-prior kernel"
    )


def test_correlations_span_meaningful_range(off_diag):
    """Off-diagonal values must reach well above 0 — not collapsed to a near-identity matrix."""
    q95 = off_diag.quantile(0.95).item()
    assert q95 > 0.15, f"95th percentile {q95:.3f} too low — correlations look collapsed near 0"


def test_correlations_not_saturated(off_diag):
    """Off-diagonal values must not pile up near 1 — that means posterior is near-singular."""
    frac_sat = (off_diag > 0.9).float().mean().item()
    assert frac_sat < 0.05, f"{frac_sat:.1%} of entries > 0.9 — matrices look saturated near 1"


def test_correlations_not_all_near_zero(off_diag):
    """Most entries near-zero (matrix ~= identity) means R_star carries no signal."""
    frac_near_zero = (off_diag.abs() < 0.02).float().mean().item()
    assert frac_near_zero < 0.85, (
        f"{frac_near_zero:.1%} of entries are ~0 — R_star looks like a matrix full of 0s"
    )


def test_correlations_std_nonzero(off_diag):
    """Standard deviation must be non-trivial — degenerate kernel collapses correlations to zero."""
    std = off_diag.std().item()
    assert std > 0.03, f"Std {std:.4f} too low — R_star correlations appear degenerate."


def test_unit_diagonal(dataset_dir):
    if not os.path.isdir(dataset_dir):
        pytest.skip(f"Dataset folder not found: {dataset_dir}")
    episodes = list(_iter_episodes(dataset_dir))
    if not episodes:
        pytest.skip(f"Dataset folder is empty: {dataset_dir}")
    for i, ep in enumerate(episodes[:20]):
        R = ep["R_star"]
        diag_err = (R.diagonal() - 1.0).abs().max().item()
        assert diag_err < 1e-4, (
            f"episode[{i}]: diagonal of R_star deviates from 1 by {diag_err:.2e}"
        )


# ---------------------------------------------------------------------------
# Tests — numerical conditioning (latent=False guarantee)
# ---------------------------------------------------------------------------


def test_r_star_well_conditioned(min_eigenvalues):
    """All R_star matrices must have minimum eigenvalue >= 0.001 (see test_dataset_corr_uniform.py)."""
    bad = [v for v in min_eigenvalues if v < 0.001]
    assert len(bad) == 0, (
        f"{len(bad)}/{len(min_eigenvalues)} episodes have min_eig < 0.001; "
        f"smallest: {min(bad):.2e}. R_star is near-singular."
    )


def test_r_star_psd(min_eigenvalues):
    neg = [v for v in min_eigenvalues if v < -1e-5]
    assert len(neg) == 0, (
        f"{len(neg)} episodes have negative min eigenvalue (most negative: {min(neg):.2e})"
    )
