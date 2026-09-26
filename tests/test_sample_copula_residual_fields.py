"""Tests for diagnostics.sample_copula_residual_fields and predict_copula_residual_field (Gaussian fallback marginal only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from eval.spatial.diagnostics import (  # noqa: E402
    predict_copula_residual_field,
    sample_copula_residual_fields,
)


def _random_correlation(rng: np.random.Generator, d: int) -> np.ndarray:
    A = rng.standard_normal((d, d))
    R = A @ A.T
    std = np.sqrt(np.diag(R))
    R = R / np.outer(std, std)
    np.fill_diagonal(R, 1.0)
    return R


@pytest.fixture
def toy_task() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    D = 8
    return {
        "rng": rng,
        "D": D,
        "R": _random_correlation(rng, D),
        "context_coords": rng.standard_normal((4, 2)),
        "context_values": rng.standard_normal(4) * 5.0 + 280.0,
        "coords_test": rng.standard_normal((D, 2)),
    }


def test_batched_shape(toy_task: dict[str, Any]) -> None:
    K = 7
    z_batch = toy_task["rng"].standard_normal((K, toy_task["D"]))
    out = sample_copula_residual_fields(
        None,
        toy_task["context_coords"],
        toy_task["context_values"],
        toy_task["coords_test"],
        toy_task["R"],
        "cpu",
        z_batch,
    )
    assert out.shape == (K, toy_task["D"])
    assert np.all(np.isfinite(out))


def test_batched_matches_single_sample_rowwise(toy_task: dict[str, Any]) -> None:
    """Each row of a K-sample batch equals predict_copula_residual_field for that z."""
    K = 5
    z_batch = toy_task["rng"].standard_normal((K, toy_task["D"]))
    batch = sample_copula_residual_fields(
        None,
        toy_task["context_coords"],
        toy_task["context_values"],
        toy_task["coords_test"],
        toy_task["R"],
        "cpu",
        z_batch,
    )
    for k in range(K):
        single = predict_copula_residual_field(
            None,
            toy_task["context_coords"],
            toy_task["context_values"],
            toy_task["coords_test"],
            toy_task["R"],
            "cpu",
            z_batch[k],
        )
        assert np.allclose(batch[k], single)


def test_naive_fallback_is_affine_in_z(toy_task: dict[str, Any]) -> None:
    """With the Gaussian fallback, the pooled samples' correlation recovers R_context."""
    rng = np.random.default_rng(1)
    K = 4000  # enough draws for a stable empirical correlation at D=8
    z_batch = rng.standard_normal((K, toy_task["D"]))
    samples = sample_copula_residual_fields(
        None,
        toy_task["context_coords"],
        toy_task["context_values"],
        toy_task["coords_test"],
        toy_task["R"],
        "cpu",
        z_batch,
    )
    R_empirical = np.corrcoef(samples.T)
    assert np.allclose(R_empirical, toy_task["R"], atol=0.05)


def test_single_sample_wrapper_matches_batch_of_one(toy_task: dict[str, Any]) -> None:
    z = toy_task["rng"].standard_normal(toy_task["D"])
    single = predict_copula_residual_field(
        None,
        toy_task["context_coords"],
        toy_task["context_values"],
        toy_task["coords_test"],
        toy_task["R"],
        "cpu",
        z,
    )
    batch = sample_copula_residual_fields(
        None,
        toy_task["context_coords"],
        toy_task["context_values"],
        toy_task["coords_test"],
        toy_task["R"],
        "cpu",
        z[None, :],
    )
    assert single.shape == (toy_task["D"],)
    assert np.allclose(single, batch[0])
