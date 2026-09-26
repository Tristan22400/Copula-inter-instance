"""Structural checks of R_star (the prior test correlation) in a dataset folder.

    DATASET_DIR=./data/pit_episodes pytest tests/test_dataset_corr_uniform.py -v

Skipped when the folder is missing or empty (default ./data/pit_cosine-new).
"""

from __future__ import annotations

import os
import random
from typing import Any, Iterator

import pytest
import torch

_DEFAULT_DIR = "./data/pit_cosine-new"


@pytest.fixture(scope="module")
def dataset_dir() -> str:
    return os.environ.get("DATASET_DIR", _DEFAULT_DIR)


_N_EPISODES = 500  # episodes to sample
_SEED = 0


def _iter_episodes(folder: str, shuffle_seed: int | None = None) -> Iterator[dict]:
    """Yield episode dicts from a folder (shards in shuffled order, or individual files)."""
    paths = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pt") and f != "meta.pt")
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(paths)
    for p in paths:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(obj, list):  # shard: list of episode dicts
            yield from obj
        elif isinstance(obj, dict):  # individual task_*.pt
            yield obj


def _load_episodes(folder: str, n: int, seed: int) -> tuple[torch.Tensor, list[Any]]:
    """Load up to *n* episodes (stopping early); return (off_diag_values, min_eigenvalues)."""
    episodes = []
    for ep in _iter_episodes(folder, shuffle_seed=seed):
        episodes.append(ep)
        if len(episodes) >= n:
            break

    values = []
    min_eigs = []
    for ep in episodes:
        R = ep["R_star"]  # (N, N)
        N = R.shape[0]
        mask = ~torch.eye(N, dtype=torch.bool)
        values.append(R[mask].flatten())
        min_eigs.append(torch.linalg.eigvalsh(R).min().item())

    return torch.cat(values), min_eigs


@pytest.fixture(scope="module")
def episode_data(dataset_dir: str) -> tuple[torch.Tensor, list[Any]]:
    if not os.path.isdir(dataset_dir):
        pytest.skip(f"Dataset folder not found: {dataset_dir}")
    pts = [f for f in os.listdir(dataset_dir) if f.endswith(".pt")]
    if len(pts) == 0:
        pytest.skip(f"Dataset folder is empty: {dataset_dir}")
    return _load_episodes(dataset_dir, _N_EPISODES, _SEED)


@pytest.fixture(scope="module")
def off_diag(episode_data: tuple[torch.Tensor, list[Any]]) -> torch.Tensor:
    return episode_data[0]


@pytest.fixture(scope="module")
def min_eigenvalues(episode_data: tuple[torch.Tensor, list[Any]]) -> list[Any]:
    return episode_data[1]


def test_correlations_have_both_signs(off_diag: torch.Tensor) -> None:
    """Both positive and negative off-diagonal entries must exist."""
    assert off_diag.min().item() < -0.02, (
        f"Min correlation {off_diag.min().item():.3f} — no negative correlations found"
    )
    assert off_diag.max().item() > 0.02, f"Max correlation {off_diag.max().item():.3f} — no positive correlations found"


def test_correlations_mean_near_zero(off_diag: torch.Tensor) -> None:
    """|mean off-diagonal R_star| < 0.30."""
    mean = off_diag.mean().item()
    assert abs(mean) < 0.30, f"Mean {mean:.3f} too far from 0 — distribution may be degenerate"


def test_correlations_negative_fraction(off_diag: torch.Tensor) -> None:
    """Between 25 % and 75 % of off-diagonal entries should be negative."""
    neg_frac = (off_diag < 0).float().mean().item()
    assert neg_frac > 0.25, f"Only {neg_frac:.1%} negative — distribution too positive"
    assert neg_frac < 0.75, f"{neg_frac:.1%} negative — distribution too negative"


def test_correlations_std_nonzero(off_diag: torch.Tensor) -> None:
    """The off-diagonal std is at least 0.1."""
    std = off_diag.std().item()
    assert std > 0.1, f"Std {std:.4f} too low — posterior R_star correlations appear degenerate."


def test_unit_diagonal(dataset_dir: str) -> None:
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
        assert diag_err < 1e-4, f"episode[{i}]: diagonal of R_star deviates from 1 by {diag_err:.2e}"


def test_r_star_well_conditioned(min_eigenvalues: list[Any]) -> None:
    """Every R_star has minimum eigenvalue >= 1e-4."""
    bad = [v for v in min_eigenvalues if v < 0.0001]
    assert len(bad) == 0, (
        f"{len(bad)}/{len(min_eigenvalues)} episodes have min_eig < 0.0001; "
        f"smallest: {min(bad):.2e}. R_star is near-singular — check if "
        f"latent=False is in effect in data_gen.generate_gp_task."
    )


def test_r_star_psd(min_eigenvalues: list[Any]) -> None:
    """R_star must be positive semi-definite (no negative eigenvalues)."""
    neg = [v for v in min_eigenvalues if v < -1e-5]
    assert len(neg) == 0, f"{len(neg)} episodes have negative min eigenvalue (most negative: {min(neg):.2e})"
