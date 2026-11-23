"""Regression tests for scenario risk, shared paths and equal-budget BO."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from eval.risk_bo import (
    SCENARIOS,
    WEIGHTS,
    RiskKnowledgeGradient,
    RiskPool,
    branin_williams,
    cvar,
    max_value_entropy,
    run_bo,
    scenario_inputs,
    select_designs,
)
from eval.risk_bo_models import GPRiskSurrogate, gaussian_paths
from eval.risk_bo_point import _paths, run_point_bo


def test_scenarios_use_published_coordinates_and_unequal_weights() -> None:
    x = scenario_inputs(torch.tensor([[0.1, 0.9]], dtype=torch.float64))
    assert x.shape == (1, 12, 4)
    torch.testing.assert_close(x[0, :, 1:3], x.new_tensor(SCENARIOS))
    torch.testing.assert_close(x[0, :, 0], x.new_full((12,), 0.1))
    torch.testing.assert_close(x[0, :, 3], x.new_full((12,), 0.9))
    assert sum(WEIGHTS) == pytest.approx(1)
    assert (branin_williams(x) > 0).all()


def test_weighted_cvar_includes_fraction_of_boundary_scenario() -> None:
    y = torch.tensor([[100.0, 20.0, 0.0], [20.0, 20.0, 20.0]], dtype=torch.float64)
    w = y.new_tensor([0.1, 0.3, 0.6])
    torch.testing.assert_close(cvar(y, w), y.new_tensor([14 / 0.3, 20.0]))
    order = [2, 0, 1]
    torch.testing.assert_close(cvar(y[:, order], w[order]), cvar(y, w))


def test_scenario_dependence_changes_single_design_expected_improvement() -> None:
    # Both scenarios have the same marginal in both columns. In candidate 0
    # they are coupled and in candidate 1 all four independent outcomes occur.
    draws = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 10.0]], [[10.0, 10.0], [10.0, 0.0]], [[10.0, 10.0], [10.0, 10.0]]]
    )
    risks = cvar(draws, draws.new_tensor([0.5, 0.5]))
    torch.testing.assert_close(risks.mean(0), draws.new_tensor([5.0, 7.5]))
    assert select_designs(risks, 8.0, 1).tolist() == [0]
    assert select_designs(risks.flip(1), 8.0, 1).tolist() == [1]


def test_batch_ei_accounts_for_inter_design_redundancy() -> None:
    risks = torch.tensor([[0.0, 0.0, 10.0], [10.0, 10.0, 2.0], [0.0, 0.0, 10.0], [10.0, 10.0, 2.0]])
    assert select_designs(risks, 10.0, 1).tolist() == [0]
    assert select_designs(risks, 10.0, 2, block_size=1).tolist() == [0, 2]


def test_mes_tracks_the_minimum_from_joint_draws() -> None:
    risks = torch.tensor([[0.0, 4.0], [0.0, 5.0], [1.0, 4.0], [1.0, 5.0]])
    scores = max_value_entropy(risks, incumbent=3.0, bins=2)
    torch.testing.assert_close(scores, risks.new_tensor([0.69314718, 0]))
    torch.testing.assert_close(max_value_entropy(risks, incumbent=0.0, bins=2), torch.zeros(2))


def test_ucb_uses_the_published_beta_and_minimizes_loss() -> None:
    risk = torch.tensor([[-1.0, -0.7], [1.0, -0.7]])
    assert select_designs(risk, 3.0, 1, acquisition="ucb").tolist() == [1]


@dataclass
class RecordingSurrogate:
    counts: list[int]

    def draw(
        self, x_train: Tensor, y_train: Tensor, x_query: Tensor, base: Tensor, seed: int
    ) -> tuple[Tensor, Callable[[], Tensor]]:
        self.counts.append(len(y_train))
        assert len(y_train) == len(x_train)
        assert x_query.shape[0] == base.shape[1]
        draws = base + 50
        # Identity covariance gives the same paths as independence under shared draws.
        identity = gaussian_paths(base, base.new_zeros((base.shape[1], 2)), base.new_ones(base.shape[1]), seed)
        return draws, lambda: identity + 50


def test_single_scenario_paths_hold_known_values_fixed() -> None:
    x = torch.arange(4, dtype=torch.float64).reshape(-1, 1)
    indices = torch.tensor([3, 0])
    base = x.new_tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    paths = _paths(x, indices, x.new_tensor([10, 15]), torch.tensor([1, 0, 3]), base, RecordingSurrogate([]), 0)
    torch.testing.assert_close(paths, x.new_tensor([[51, 15, 10], [55, 15, 10]]))


@pytest.mark.parametrize("acquisition", ["rho-random", "rho-kg-apx", "rho-kg"])
def test_single_scenario_policy_uses_unique_queries_and_true_recommendation_risk(acquisition: str) -> None:
    x = torch.arange(12, dtype=torch.float64).reshape(3, 2, 2)
    pool = RiskPool(x, x.new_tensor([[1, 0], [1, 2], [0, 3]]), x.new_tensor([0.5, 0.5]), 0.75, "cvar")
    trace = run_point_bo(
        pool,
        torch.tensor([1, 3]),
        RecordingSurrogate([]),
        seed=0,
        budget=5,
        n_draws=4,
        acquisition=acquisition,
        kg=RiskKnowledgeGradient(2, 2, 2),
    )
    assert trace.evaluations == [2, 3, 4, 5]
    assert len(set(zip(trace.designs, trace.scenarios, strict=True))) == 5
    assert trace.regret == [float(pool.risk(pool.losses)[i]) - 1 for i in trace.recommendations]


@pytest.mark.parametrize(("q", "evaluations"), [(1, [24, 36, 48, 60, 72]), (2, [24, 48, 72])])
def test_equal_budget_and_identity_control(q: int, evaluations: list[int]) -> None:
    pool = torch.quasirandom.SobolEngine(2, scramble=True, seed=42).draw(8, dtype=torch.float64)
    model = RecordingSurrogate([])
    trace = run_bo(pool, torch.tensor([0, 2]), model, seed=42, n_designs=6, batch_size=q, n_draws=8)
    assert trace.evaluations == evaluations
    assert model.counts == evaluations[:-1]
    assert len(set(trace.designs)) == len(trace.designs) == 6
    assert not any(trace.dependence_changed_choice)
    assert len(trace.dependence_changed_choice) == len(evaluations) - 1
    assert trace.same_history_regret_delta == [0.0] * (len(evaluations) - 1)
    assert trace.regret == sorted(trace.regret, reverse=True)
    assert trace.regret[-1] >= 0


def test_same_history_delta_scores_the_alternative_without_observing_it() -> None:
    class DisagreeingSurrogate:
        def draw(
            self, x_train: Tensor, y_train: Tensor, x_query: Tensor, base: Tensor, seed: int
        ) -> tuple[Tensor, Callable[[], Tensor]]:
            return base.new_tensor([[2.0, 3.0]]), lambda: base.new_tensor([[3.0, 2.0]])

    x = torch.arange(4, dtype=torch.float64).reshape(4, 1, 1)
    pool = RiskPool(x, x.new_tensor([[5.0], [6.0], [2.0], [3.0]]), x.new_tensor([1.0]), 0.75, "cvar")
    trace = run_bo(
        x, torch.tensor([0, 1]), DisagreeingSurrogate(), seed=0, n_designs=3, batch_size=1, n_draws=1, benchmark=pool
    )
    assert trace.designs == [0, 1, 2]
    assert trace.dependence_changed_choice == [True]
    assert trace.same_history_regret_delta == [-1.0]


@pytest.mark.parametrize("log_output", [False, True])
def test_gp_draws_finite_reproducible_paths(log_output: bool) -> None:
    pool = torch.quasirandom.SobolEngine(2, scramble=True, seed=42).draw(4, dtype=torch.float64)
    x = scenario_inputs(pool)
    base = torch.randn(4, 24, dtype=torch.float64, generator=torch.Generator().manual_seed(12))
    model = GPRiskSurrogate(log_output=log_output, n_steps=2)
    y = branin_williams(x[:2]).flatten()
    draws, control = model.draw(x[:2].flatten(0, 1), y, x[2:].flatten(0, 1), base, seed=0)
    repeated, _ = model.draw(x[:2].flatten(0, 1), y, x[2:].flatten(0, 1), base, seed=0)
    assert control is None and draws.shape == base.shape
    assert torch.isfinite(draws).all()
    if log_output:
        assert (draws > 0).all()
    torch.testing.assert_close(draws, repeated)
