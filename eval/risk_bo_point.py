"""Single-scenario BoRisk policies on a finite design–scenario grid."""

from __future__ import annotations

import time

import torch
from torch import Tensor

from eval.risk_bo import RiskKnowledgeGradient, RiskPool, Surrogate, Trace


def _paths(
    x: Tensor,
    indices: Tensor,
    observations: Tensor,
    queries: Tensor,
    base: Tensor,
    surrogate: Surrogate,
    seed: int,
) -> Tensor:
    """Draw unknown rows jointly and retain actual or fantasized values on known rows."""
    context_positions = torch.full((len(x),), -1, dtype=torch.long, device=x.device)
    context_positions[indices] = torch.arange(len(indices), device=x.device)
    positions = context_positions[queries]
    known = positions >= 0
    paths = base.new_empty((len(base), len(queries)))
    paths[:, known] = observations[positions[known]]
    if bool((~known).any()):
        unknown = queries[~known]
        draws, _ = surrogate.draw(x[indices], observations, x[unknown], base[:, unknown], seed)
        if draws.shape != (len(base), len(unknown)) or not torch.isfinite(draws).all():
            raise ValueError("surrogate returned invalid scenario draws")
        paths[:, ~known] = draws.to(paths)
    return paths


def _select_kg(
    benchmark: RiskPool,
    surrogate: Surrogate,
    x: Tensor,
    indices: Tensor,
    observations: Tensor,
    paths: Tensor,
    base: Tensor,
    available: Tensor,
    design_rows: Tensor,
    seed: int,
    kg: RiskKnowledgeGradient,
    approximate: bool,
) -> Tensor:
    risks = benchmark.risk(paths[:, design_rows])
    promising = risks.mean(0).argsort(stable=True)[: kg.recommendations]
    uncertainty = paths[:, design_rows].std(0, correction=0) * benchmark.weights
    action_rows = design_rows[promising].flatten()
    action_rows = action_rows[available[action_rows]]
    if not len(action_rows):
        action_rows = available.nonzero().flatten()
    score = uncertainty.flatten()[action_rows]
    actions = action_rows[score.argsort(descending=True, stable=True)[: kg.candidates]]
    visited = (indices // design_rows.shape[1]).unique()
    scores = []
    for action in actions.unbind():
        design = action // design_rows.shape[1]
        recommendations = (
            torch.cat((visited, design.reshape(1))).unique()
            if approximate
            else torch.arange(len(design_rows), device=x.device)
        )
        query = design_rows[recommendations].flatten()
        context = torch.cat((indices, action.reshape(1)))
        future = []
        for fantasy, value in enumerate(paths[: kg.fantasies, action].unbind()):
            draws = _paths(
                x,
                context,
                torch.cat((observations, value.reshape(1))),
                query,
                base,
                surrogate,
                seed + 104_729 * (fantasy + 1),
            )
            future.append(benchmark.risk(draws.reshape(len(base), len(recommendations), -1)).mean(0).min())
        scores.append(torch.stack(future).mean())
    return actions[torch.stack(scores).argmin()].reshape(1)


def run_point_bo(
    benchmark: RiskPool,
    initial: Tensor,
    surrogate: Surrogate,
    *,
    seed: int,
    budget: int,
    n_draws: int,
    acquisition: str,
    kg: RiskKnowledgeGradient,
) -> Trace:
    """Run a single-scenario policy and score the posterior recommendation after each call."""
    n_designs, n_scenarios, _ = benchmark.inputs.shape
    x = benchmark.inputs.flatten(0, 1)
    losses = benchmark.losses.flatten()
    truth = benchmark.risk(benchmark.losses)
    optimum = float(truth.min())
    rows = torch.arange(len(x), device=x.device)
    design_rows = rows.reshape(n_designs, n_scenarios)
    indices = initial.clone()
    observations = losses[indices]
    available = torch.ones(len(x), dtype=torch.bool, device=x.device)
    available[indices] = False
    generator = torch.Generator(device=x.device)
    evaluations: list[int] = []
    regret: list[float] = []
    recommendations: list[int] = []
    start = time.perf_counter()
    while True:
        step_seed = seed * 1_000_003 + len(indices)
        generator.manual_seed(step_seed)
        base = torch.randn(n_draws, len(x), device=x.device, dtype=x.dtype, generator=generator)
        paths = _paths(x, indices, observations, rows, base, surrogate, step_seed)
        predicted = benchmark.risk(paths[:, design_rows]).mean(0)
        recommendation = int(predicted.argmin())
        recommendations.append(recommendation)
        evaluations.append(len(indices))
        regret.append(float(truth[recommendation]) - optimum)
        if len(indices) >= budget:
            break
        if acquisition == "rho-random":
            eligible = available.nonzero().flatten()
            choice = eligible[torch.randint(len(eligible), (1,), device=x.device, generator=generator)]
        else:
            choice = _select_kg(
                benchmark,
                surrogate,
                x,
                indices,
                observations,
                paths,
                base,
                available,
                design_rows,
                step_seed,
                kg,
                approximate=acquisition == "rho-kg-apx",
            )
        indices = torch.cat((indices, choice))
        observations = torch.cat((observations, losses[choice]))
        available[choice] = False
    return Trace(
        (indices // n_scenarios).tolist(),
        evaluations,
        regret,
        time.perf_counter() - start,
        [],
        (indices % n_scenarios).tolist(),
        recommendations,
    )
