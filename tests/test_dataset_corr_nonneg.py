"""Structural checks of R_star for non-negative kernels (rbf, matern32, rational_quadratic, periodic).

Episodes are generated live from the default data config, one fixed kernel per
batch, with the analytic marginal, so no dataset folder or TabICL model is needed.
R_star is the prior test correlation (data_gen supports oracle_mode=prior only),
so for these kernels it has no negative entries.
"""

from __future__ import annotations

import pytest
import torch

import copula_inter
from copula_inter.config_path import compose_config, config_dir
from copula_inter.data_gen import generate_gp_batch

_KERNELS = ("rbf", "matern32", "rational_quadratic", "periodic")
_EPISODES_PER_KERNEL = 64
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
    return [R for kernel in _KERNELS for R in _live_r_star(kernel, _EPISODES_PER_KERNEL, _N_TEST, _SEED)]


@pytest.fixture(scope="module")
def off_diag(r_stars: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([R[~torch.eye(R.shape[0], dtype=torch.bool)] for R in r_stars])


@pytest.fixture(scope="module")
def min_eigenvalues(r_stars: list[torch.Tensor]) -> list[float]:
    return [torch.linalg.eigvalsh(R.double()).min().item() for R in r_stars]


def test_correlations_non_negative(off_diag: torch.Tensor) -> None:
    """A non-negative kernel's prior correlation has no negative entries (up to rounding)."""
    assert off_diag.min().item() > -1e-5, f"Min correlation {off_diag.min().item():.2e} — negative entry found"


def test_correlations_span_meaningful_range(off_diag: torch.Tensor) -> None:
    """Off-diagonal values must reach well above 0 — not collapsed to a near-identity matrix."""
    q95 = off_diag.quantile(0.95).item()
    assert q95 > 0.15, f"95th percentile {q95:.3f} too low — correlations look collapsed near 0"


def test_correlations_not_all_near_zero(off_diag: torch.Tensor) -> None:
    """Most entries near-zero (matrix ~= identity) means R_star carries no signal."""
    frac_near_zero = (off_diag.abs() < 0.02).float().mean().item()
    assert frac_near_zero < 0.85, f"{frac_near_zero:.1%} of entries are ~0 — R_star looks like a matrix full of 0s"


def test_correlations_std_nonzero(off_diag: torch.Tensor) -> None:
    """Standard deviation must be non-trivial — degenerate kernel collapses correlations to zero."""
    std = off_diag.std().item()
    assert std > 0.03, f"Std {std:.4f} too low — R_star correlations appear degenerate."


def test_unit_diagonal(r_stars: list[torch.Tensor]) -> None:
    for i, R in enumerate(r_stars):
        diag_err = (R.diagonal() - 1.0).abs().max().item()
        assert diag_err < 1e-4, f"episode[{i}]: diagonal of R_star deviates from 1 by {diag_err:.2e}"


def test_r_star_well_conditioned(min_eigenvalues: list[float]) -> None:
    """Every R_star has minimum eigenvalue >= 1e-3."""
    bad = [v for v in min_eigenvalues if v < 0.001]
    assert len(bad) == 0, (
        f"{len(bad)}/{len(min_eigenvalues)} episodes have min_eig < 0.001; "
        f"smallest: {min(bad):.2e}. R_star is near-singular."
    )


def test_r_star_psd(min_eigenvalues: list[float]) -> None:
    neg = [v for v in min_eigenvalues if v < -1e-5]
    assert len(neg) == 0, f"{len(neg)} episodes have negative min eigenvalue (most negative: {min(neg):.2e})"
