"""Checks for benchmark inputs, risk reducers, and equal-budget BO."""

import math

import pytest
import torch

from eval.environment_calibration import TRUE_PARAMETERS, calibration_pool, concentration
from eval.risk_bo import run_bo, var
from eval.risk_bo_benchmarks import benchmark_design, marzat_f6
from eval.runners.risk_bo import RiskBOSpec, validate


@pytest.mark.parametrize(
    ("problem", "scenarios", "features", "measure"),
    [
        ("branin-williams", 12, 4, "cvar"),
        ("f6", 8, 7, "cvar"),
        ("portfolio", 10, 5, "var"),
        ("covid-testing", 27, 5, "cvar"),
    ],
)
def test_benchmark_rows(problem: str, scenarios: int, features: int, measure: str) -> None:
    benchmark = benchmark_design(problem, 8, 42, 43)
    assert benchmark.inputs.shape == (8, scenarios, features)
    assert benchmark.measure == measure
    assert benchmark.weights.sum().item() == pytest.approx(1)
    assert (benchmark.weights > 0).all()
    if problem == "covid-testing":
        assert (benchmark.inputs[..., :2].sum(-1) <= 1).all()
        assert benchmark.weights.min().item() == 1 / 64
        assert benchmark.weights.max().item() == 1 / 8


def test_f6_uses_all_three_environmental_penalties() -> None:
    # At x=0, f6 is 2u + v + 5w - u² - v² - w².
    x = torch.tensor([[0.0, 0.0, 0.0, 0.0, -2.0, 1.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(marzat_f6(x), torch.tensor([-2.0, 0.0]))


def test_var_at_a_discrete_boundary() -> None:
    y = torch.arange(10, dtype=torch.float64)
    w = torch.full_like(y, 0.1)
    torch.testing.assert_close(var(y, w, 0.8), y[7])
    torch.testing.assert_close(var(y.flip(0), w, 0.81), y[8])


def test_environment_calibration_readings_and_fixed_reference() -> None:
    pool, identity = calibration_pool(8, seed=42, device="cpu")
    assert pool.inputs.shape == (8, 12, 6)
    assert pool.outcomes.shape == (8, 12)
    grid = pool.inputs[0, :, -2:]
    torch.testing.assert_close(grid[:, 0], grid.new_tensor([0.0] * 4 + [1.0] * 4 + [2.5] * 4))
    torch.testing.assert_close(grid[:, 1], grid.new_tensor([15.0, 30.0, 45.0, 60.0] * 3))
    params = grid.new_tensor(TRUE_PARAMETERS)
    reference = concentration(params, grid)
    torch.testing.assert_close(reference[0], grid.new_tensor(10 / math.sqrt(4 * math.pi * 0.07 * 15)))
    assert reference.tolist() == identity["reference"]
    assert pool.objective(reference) == 0
    torch.testing.assert_close(pool.objective(reference + 1), grid.new_tensor(12.0))
    assert (pool.objective(pool.outcomes) > 0).all()


def test_environment_calibration_charges_one_call_per_complete_vector() -> None:
    pool, _ = calibration_pool(8, seed=42, device="cpu")
    trace = run_bo(
        pool.inputs,
        torch.tensor([0, 1]),
        None,
        seed=1,
        n_designs=5,
        batch_size=2,
        n_draws=4,
        benchmark=pool,
        acquisition="random",
    )
    assert trace.evaluations == [2, 4, 5]
    actual_error = pool.objective(pool.outcomes[trace.designs])
    assert trace.regret == [float(actual_error[:count].min()) for count in trace.evaluations]


@pytest.mark.parametrize("overrides", [{"risk_measure": "var"}, {"acquisition": "rho-kg"}])
def test_environment_calibration_requires_the_composite_protocol(overrides: dict[str, str]) -> None:
    spec = RiskBOSpec(problem="environment-calibration")
    for key, value in overrides.items():
        setattr(spec, key, value)
    with pytest.raises(ValueError, match="squared error and whole-design evaluations"):
        validate(spec)
