"""test_correlation_factory.py — verify src/correlation_factory.py.

Checks, per parametrization (CovNorm / CosSim / TanhNorm / SparseCovNorm):
  1. Unit diagonal, symmetry, strict positive-definiteness of R = dense().
  2. Woodbury log_det()/solve()/inverse() match dense torch.linalg ops.
  3. Gradients through the raw unconstrained inputs are finite.
"""

from __future__ import annotations

import pytest
import torch

from correlation_factory import (
    LowRankCorrelationFactor,
    covnorm_correlation,
    cossim_correlation,
    sparse_covnorm_correlation,
    tanhnorm_correlation,
)

torch.manual_seed(0)

D = 12  # ambient dimension
R = 3  # rank


def _random_inputs(batch: int, requires_grad: bool = False):
    def mk(*shape):
        t = torch.randn(*shape, dtype=torch.float64) * 1.5
        if requires_grad:
            t.requires_grad_(True)
        return t

    return {
        "W": mk(batch, D, R),
        "v": mk(batch, D),
        "V": mk(batch, D, R),
        "g": mk(batch, D),
        "lam_raw": mk(batch),
    }


def _build(name: str, inputs: dict) -> LowRankCorrelationFactor:
    if name == "covnorm":
        return covnorm_correlation(inputs["W"], inputs["v"])
    if name == "cossim":
        return cossim_correlation(inputs["V"], inputs["g"])
    if name == "tanhnorm":
        return tanhnorm_correlation(inputs["W"])
    if name == "sparse_covnorm":
        return sparse_covnorm_correlation(inputs["W"], inputs["v"], inputs["lam_raw"])
    raise ValueError(name)


PARAMETRIZATIONS = ["covnorm", "cossim", "tanhnorm", "sparse_covnorm"]
BATCH_SIZES = [1, 4]


@pytest.mark.parametrize("name", PARAMETRIZATIONS)
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_unit_diagonal(name, batch):
    inputs = _random_inputs(batch)
    factor = _build(name, inputs)
    Rd = factor.dense()
    diag = Rd.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)


@pytest.mark.parametrize("name", PARAMETRIZATIONS)
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_symmetry(name, batch):
    inputs = _random_inputs(batch)
    factor = _build(name, inputs)
    Rd = factor.dense()
    assert torch.allclose(Rd, Rd.transpose(-1, -2), atol=1e-10)


@pytest.mark.parametrize("name", PARAMETRIZATIONS)
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_strictly_positive_definite(name, batch):
    inputs = _random_inputs(batch)
    factor = _build(name, inputs)
    Rd = factor.dense()
    # Raises if not PD; also confirms no NaN/Inf reached the diagonal.
    L = torch.linalg.cholesky(Rd)
    assert torch.isfinite(L).all()


@pytest.mark.parametrize("name", PARAMETRIZATIONS)
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_log_det_matches_dense(name, batch):
    inputs = _random_inputs(batch)
    factor = _build(name, inputs)
    Rd = factor.dense()
    expected = torch.linalg.slogdet(Rd).logabsdet
    actual = factor.log_det()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("name", PARAMETRIZATIONS)
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_inverse_matches_dense(name, batch):
    inputs = _random_inputs(batch)
    factor = _build(name, inputs)
    Rd = factor.dense()
    expected = torch.linalg.inv(Rd)
    actual = factor.inverse()
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    # Symmetric by construction of the Woodbury formula.
    assert torch.allclose(actual, actual.transpose(-1, -2), atol=1e-8)


@pytest.mark.parametrize("name", PARAMETRIZATIONS)
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_solve_matches_dense(name, batch):
    inputs = _random_inputs(batch)
    factor = _build(name, inputs)
    Rd = factor.dense()
    z = torch.randn(batch, D, dtype=torch.float64)
    expected = torch.linalg.solve(Rd, z.unsqueeze(-1)).squeeze(-1)
    actual = factor.solve(z)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)

    expected_quad = (z * expected).sum(-1)
    actual_quad = factor.quad_form(z)
    assert torch.allclose(actual_quad, expected_quad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("name", PARAMETRIZATIONS)
def test_gradients_are_finite(name):
    inputs = _random_inputs(batch=4, requires_grad=True)
    factor = _build(name, inputs)
    loss = factor.dense().sum() + factor.log_det().sum()
    loss.backward()
    for key, t in inputs.items():
        if not t.requires_grad or t.grad is None:
            continue
        assert torch.isfinite(t.grad).all(), f"non-finite grad for {name}/{key}"
