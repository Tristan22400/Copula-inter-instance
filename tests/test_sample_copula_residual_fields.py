"""test_sample_copula_residual_fields.py — regression coverage for
eval/spatial/diagnostics.py::sample_copula_residual_fields (the batched
y-space sampler behind sweep_core.py::run_real_config's model_r2/shape_corr
fix) and predict_copula_residual_field (now a thin K=1 wrapper around it).

Exercises the tabicl_marginal=None (naive Gaussian fallback) path only —
fast, deterministic, no network/model download. The tabicl_marginal-given
path (the real TabICL QuantileDistribution.icdf, whose (*batch_shape, n)
argument contract is stricter than torch.distributions.Normal's ordinary
broadcasting, and does NOT match the FakeTabICL fixture other test files use
for pit.py::run_pit) was verified manually against the real pretrained
checkpoint instead of committed here as a fast unit test — see the PR/commit
description.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
def toy_task():
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


def test_batched_shape(toy_task):
    K = 7
    z_batch = toy_task["rng"].standard_normal((K, toy_task["D"]))
    out = sample_copula_residual_fields(
        None, toy_task["context_coords"], toy_task["context_values"], toy_task["coords_test"],
        toy_task["R"], "cpu", z_batch,
    )
    assert out.shape == (K, toy_task["D"])
    assert np.all(np.isfinite(out))


def test_batched_matches_single_sample_rowwise(toy_task):
    """Each row of a K-sample batch must equal what
    predict_copula_residual_field returns for that same z row — batching
    must not change per-sample values, only amortize the (real-marginal-
    only) forward pass across them."""
    K = 5
    z_batch = toy_task["rng"].standard_normal((K, toy_task["D"]))
    batch = sample_copula_residual_fields(
        None, toy_task["context_coords"], toy_task["context_values"], toy_task["coords_test"],
        toy_task["R"], "cpu", z_batch,
    )
    for k in range(K):
        single = predict_copula_residual_field(
            None, toy_task["context_coords"], toy_task["context_values"], toy_task["coords_test"],
            toy_task["R"], "cpu", z_batch[k],
        )
        assert np.allclose(batch[k], single)


def test_naive_fallback_is_affine_in_z(toy_task):
    """tabicl_marginal=None's naive fallback is y = y_mean + y_std * z_copula
    -- a single shared (scalar) affine map applied identically to every
    grid point, so the resulting SAMPLES' empirical correlation should
    recover R_context (up to Monte Carlo noise) exactly like the z-space
    correlation would, which is the whole point of the y-space fix relying
    on many pooled draws (see run_real_config's N_YSPACE_MC_SAMPLES) rather
    than a single sample."""
    rng = np.random.default_rng(1)
    K = 4000  # enough draws for a stable empirical correlation at D=8
    z_batch = rng.standard_normal((K, toy_task["D"]))
    samples = sample_copula_residual_fields(
        None, toy_task["context_coords"], toy_task["context_values"], toy_task["coords_test"],
        toy_task["R"], "cpu", z_batch,
    )
    R_empirical = np.corrcoef(samples.T)
    assert np.allclose(R_empirical, toy_task["R"], atol=0.05)


def test_single_sample_wrapper_matches_batch_of_one(toy_task):
    z = toy_task["rng"].standard_normal(toy_task["D"])
    single = predict_copula_residual_field(
        None, toy_task["context_coords"], toy_task["context_values"], toy_task["coords_test"],
        toy_task["R"], "cpu", z,
    )
    batch = sample_copula_residual_fields(
        None, toy_task["context_coords"], toy_task["context_values"], toy_task["coords_test"],
        toy_task["R"], "cpu", z[None, :],
    )
    assert single.shape == (toy_task["D"],)
    assert np.allclose(single, batch[0])
