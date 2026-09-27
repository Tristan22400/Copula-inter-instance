"""Structural checks of R_star (the prior test correlation) for a sign-changing kernel.

Episodes are generated live from the default data config with a fixed cosine
kernel and the analytic marginal, so no dataset folder or TabICL model is needed.
"""

from __future__ import annotations

import pytest
import torch

import copula_inter
from copula_inter.config_path import compose_config, config_dir
from copula_inter.data_gen import generate_gp_batch

_KERNEL = "cosine"
_N_EPISODES = 256
_N_TEST = 64
_SEED = 0


def _live_r_star(kernel: str, n_episodes: int, n_test: int, seed: int) -> list[torch.Tensor]:
    """R_star of n_episodes live GP episodes drawn with one fixed kernel."""
    cfg = compose_config(
        config_dir(copula_inter.__file__),
        "config",
        [
            f"data.kernel={kernel}",
            "data.systematic_composition=false",
            "data.z_train_source=analytic",
            f"data.N_min={n_test}",
            f"data.N_max={n_test}",
            f"seed={seed}",
        ],
    )
    torch.manual_seed(seed)
    return [ep["R_star"] for ep in generate_gp_batch(cfg, n_episodes, "cpu")]


@pytest.fixture(scope="module")
def r_stars() -> list[torch.Tensor]:
    return _live_r_star(_KERNEL, _N_EPISODES, _N_TEST, _SEED)


@pytest.fixture(scope="module")
def off_diag(r_stars: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([R[~torch.eye(R.shape[0], dtype=torch.bool)] for R in r_stars])


@pytest.fixture(scope="module")
def min_eigenvalues(r_stars: list[torch.Tensor]) -> list[float]:
    return [torch.linalg.eigvalsh(R.double()).min().item() for R in r_stars]


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
    assert std > 0.1, f"Std {std:.4f} too low — R_star correlations appear degenerate."


def test_unit_diagonal(r_stars: list[torch.Tensor]) -> None:
    """R_star must have unit diagonal (proper correlation matrix)."""
    for i, R in enumerate(r_stars):
        diag_err = (R.diagonal() - 1.0).abs().max().item()
        assert diag_err < 1e-4, f"episode[{i}]: diagonal of R_star deviates from 1 by {diag_err:.2e}"


def test_r_star_well_conditioned(min_eigenvalues: list[float]) -> None:
    """Every R_star has minimum eigenvalue >= 1e-4."""
    bad = [v for v in min_eigenvalues if v < 0.0001]
    assert len(bad) == 0, (
        f"{len(bad)}/{len(min_eigenvalues)} episodes have min_eig < 0.0001; "
        f"smallest: {min(bad):.2e}. R_star is near-singular."
    )


def test_r_star_psd(min_eigenvalues: list[float]) -> None:
    """R_star must be positive semi-definite (no negative eigenvalues)."""
    neg = [v for v in min_eigenvalues if v < -1e-5]
    assert len(neg) == 0, f"{len(neg)} episodes have negative min eigenvalue (most negative: {min(neg):.2e})"
