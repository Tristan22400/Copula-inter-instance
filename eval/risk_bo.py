"""Branin–Williams and expected improvement in scenario CVaR."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

import torch
from torch import Tensor

POINT_ACQUISITIONS = ("rho-random", "rho-kg-apx", "rho-kg")
ACQUISITIONS = ("ei", "pi", "mes", "kg", "ucb", "random", *POINT_ACQUISITIONS)

# Cakmak et al. (NeurIPS 2020), supplementary Table 1. The coordinates
# (x1, x4) are the design and (x2, x3) are the environmental scenario.
SCENARIOS = tuple((u, v) for u in (0.25, 0.5, 0.75) for v in (0.2, 0.4, 0.6, 0.8))
WEIGHTS = (0.0375, 0.0875, 0.0875, 0.0375, 0.075, 0.175, 0.175, 0.075, 0.0375, 0.0875, 0.0875, 0.0375)


def scenario_inputs(designs: Tensor) -> Tensor:
    """Expand (designs, 2) to (designs, scenarios, 4) in the published coordinate order."""
    w = designs.new_tensor(SCENARIOS).expand(len(designs), -1, -1)
    x = designs[:, None, :].expand(-1, len(SCENARIOS), -1)
    return torch.cat((x[..., :1], w, x[..., 1:]), dim=-1)


def _branin(u: Tensor, v: Tensor) -> Tensor:
    return (v - 5.1 / (4 * math.pi**2) * u.square() + 5 / math.pi * u - 6).square() + (
        10 * (1 - 1 / (8 * math.pi)) * u.cos() + 10
    )


def branin_williams(inputs: Tensor) -> Tensor:
    """Evaluate the deterministic Branin–Williams function on the last axis."""
    x1, x2, x3, x4 = inputs.unbind(-1)
    return _branin(15 * x1 - 5, 15 * x2) * _branin(15 * x3 - 5, 15 * x4)


def cvar(losses: Tensor, weights: Tensor, alpha: float = 0.7) -> Tensor:
    """Average the largest losses over the upper (1-alpha) probability mass."""
    values, order = losses.sort(dim=-1, descending=True)
    w = weights[order]
    tail = 1 - alpha
    used = (tail - (w.cumsum(-1) - w)).clamp_min(0).minimum(w)
    return (values * used).sum(-1) / tail


def var(losses: Tensor, weights: Tensor, alpha: float) -> Tensor:
    """Return the smallest loss whose discrete cumulative probability reaches alpha."""
    values, order = losses.sort(dim=-1)
    mass = weights[order].cumsum(-1)
    index = (mass < alpha - 4 * torch.finfo(weights.dtype).eps).sum(-1, keepdim=True)
    return values.gather(-1, index).squeeze(-1)


@dataclass(frozen=True)
class RiskPool:
    inputs: Tensor
    losses: Tensor
    weights: Tensor
    alpha: float
    measure: Literal["var", "cvar"]

    def risk(self, losses: Tensor) -> Tensor:
        reducer = var if self.measure == "var" else cvar
        return reducer(losses, self.weights, self.alpha)


@dataclass(frozen=True)
class BundlePool:
    """Vector outcomes, their scalar objective, and the cost of revealing a vector."""

    inputs: Tensor
    outcomes: Tensor
    objective: Callable[[Tensor], Tensor]
    evaluations_per_design: int
    optimum: float | None = None


def select_designs(
    risk: Tensor, incumbent: float, batch_size: int, block_size: int = 16, acquisition: str = "ei"
) -> Tensor:
    """Select unevaluated designs from the posterior draws of their risks."""
    if acquisition == "ucb":
        return (risk.mean(0) - math.sqrt(0.2) * risk.std(0, correction=0)).argmin().reshape(1)
    if acquisition == "mes":
        return max_value_entropy(risk, incumbent).argmax().reshape(1)
    gain = (incumbent - risk).clamp_min(0)
    if acquisition == "pi":
        gain = (gain > 0).to(gain)
    if batch_size == 1:
        return gain.mean(0).argmax().reshape(1)
    if batch_size != 2 or risk.shape[1] < 2:
        raise ValueError("select_designs requires one design or a pair of distinct designs")
    n = gain.shape[1]
    ids = torch.arange(n, device=gain.device)
    scores = []
    for start in range(0, n, block_size):
        block = gain[:, start : start + block_size, None]
        score = torch.maximum(block, gain[:, None, :]).mean(0)
        scores.append(score.masked_fill(ids[None, :] <= ids[start : start + block_size, None], -torch.inf))
    best = torch.cat(scores, dim=0).flatten().argmax()
    return torch.stack((best // n, best % n))


def max_value_entropy(risk: Tensor, incumbent: float, bins: int = 6) -> Tensor:
    """Histogram estimate of information about the minimum risk from joint posterior draws."""
    levels = torch.arange(1, bins, device=risk.device, dtype=risk.dtype) / bins
    optimum = risk.amin(1).clamp_max(incumbent)
    optimum_bin = (optimum[:, None] > torch.quantile(optimum, levels)[None]).sum(-1)
    edges = torch.quantile(risk, levels, dim=0)
    design_bin = (risk[:, None, :] > edges[None]).sum(1)
    cells = bins * design_bin + optimum_bin[:, None]
    counts = risk.new_zeros((bins * bins, risk.shape[1])).scatter_add_(0, cells, torch.ones_like(risk))
    joint = counts.reshape(bins, bins, -1) / len(risk)
    product = joint.sum(0, keepdim=True) * joint.sum(1, keepdim=True)
    return (torch.special.xlogy(joint, joint) - torch.special.xlogy(joint, product)).sum((0, 1))


class Surrogate(Protocol):
    """Posterior paths over the same ordered design–scenario inputs for all draws."""

    def draw(
        self, x_train: Tensor, y_train: Tensor, x_query: Tensor, base: Tensor, seed: int
    ) -> tuple[Tensor, Callable[[], Tensor] | None]:
        """Return paths and an optional deferred independence check with the fitted marginals."""
        ...


@dataclass(frozen=True)
class RiskKnowledgeGradient:
    candidates: int = 3
    recommendations: int = 8
    fantasies: int = 4

    def select(
        self,
        benchmark: BundlePool,
        surrogate: Surrogate,
        x_train: Tensor,
        y_train: Tensor,
        queries: Tensor,
        draws: Tensor,
        risks: Tensor,
        base: Tensor,
        incumbent: float,
        seed: int,
    ) -> Tensor:
        """Approximate bundle KG with an EI action shortlist and fixed recommendation set."""
        improvement = (incumbent - risks).clamp_min(0).mean(0)
        candidates = improvement.argsort(descending=True, stable=True)[: self.candidates]
        recommendations = risks.mean(0).argsort(stable=True)[: self.recommendations]
        n_fantasies = min(self.fantasies, len(draws))
        scores = []
        for action in candidates.unbind():
            future = recommendations[recommendations != action]
            future_inputs = queries[future].flatten(0, 1)
            context = torch.cat((x_train, queries[action]))
            outcomes = draws[:n_fantasies, action]
            updated = []
            for fantasy, loss in enumerate(outcomes.unbind()):
                best = benchmark.objective(loss).clamp_max(incumbent)
                if len(future):
                    paths, _ = surrogate.draw(
                        context,
                        torch.cat((y_train, loss.to(y_train))),
                        future_inputs,
                        base[:, future].flatten(1),
                        seed + 104_729 * (fantasy + 1),
                    )
                    expected_risk = benchmark.objective(paths.reshape(len(base), len(future), queries.shape[1])).mean(0)
                    best = torch.minimum(best, expected_risk.min())
                updated.append(best)
            # The current best posterior recommendation has the same score
            # for every action, so minimizing the expected future score
            # is equivalent to maximizing its expected improvement.
            scores.append(torch.stack(updated).mean())
        return candidates[torch.stack(scores).argmin()].reshape(1)


@dataclass
class Trace:
    designs: list[int]
    evaluations: list[int]
    regret: list[float]
    wall_seconds: float
    dependence_changed_choice: list[bool]
    scenarios: list[int] = field(default_factory=list)
    recommendations: list[int] = field(default_factory=list)
    same_history_regret_delta: list[float] = field(default_factory=list)


def run_bo(
    pool: Tensor,
    initial: Tensor,
    surrogate: Surrogate | None,
    *,
    seed: int,
    n_designs: int,
    batch_size: int,
    n_draws: int,
    benchmark: BundlePool | RiskPool | None = None,
    acquisition: str = "ei",
    knowledge_gradient: RiskKnowledgeGradient | None = None,
) -> Trace:
    """Run fixed-pool BO, revealing the vector outcomes of selected designs."""
    if benchmark is None:
        x = scenario_inputs(pool)
        benchmark = RiskPool(x, branin_williams(x), x.new_tensor(WEIGHTS), 0.7, "cvar")
    if isinstance(benchmark, RiskPool):
        benchmark = BundlePool(benchmark.inputs, benchmark.losses, benchmark.risk, benchmark.inputs.shape[1])
    x = benchmark.inputs
    n_scenarios = x.shape[1]
    cost = benchmark.evaluations_per_design
    optimum = benchmark.optimum
    if optimum is None:
        optimum = float(benchmark.objective(benchmark.outcomes).min())
    selected = initial.clone()
    observed = benchmark.outcomes[selected]
    incumbent = float(benchmark.objective(observed).min())
    evaluations = [selected.numel() * cost]
    regret = [incumbent - optimum]
    changes = []
    same_history_regret_delta = []
    duration = 0.0
    available = torch.ones(len(pool), dtype=torch.bool, device=pool.device)
    available[selected] = False
    generator = torch.Generator(device=pool.device)
    while selected.numel() < n_designs:
        candidates = available.nonzero().flatten()
        size = min(batch_size, n_designs - selected.numel())
        # Keep noise keyed to pool row and evaluation count even after policies diverge.
        step_seed = seed * 1_000_003 + selected.numel()
        generator.manual_seed(step_seed)
        noise = torch.randn(n_draws, *x.shape[:2], generator=generator, device=x.device, dtype=x.dtype)
        query = x[candidates].flatten(0, 1)
        start = time.perf_counter()
        x_train = x[selected].flatten(0, 1)
        y_train = observed.flatten()
        base = noise[:, candidates]
        independent = None
        if acquisition == "random":
            choice = torch.randperm(len(candidates), device=x.device, generator=generator)[:size]
        else:
            assert surrogate is not None
            draws, independent = surrogate.draw(x_train, y_train, query, base.flatten(1), step_seed)
            if draws.shape != (n_draws, len(query)) or not torch.isfinite(draws).all():
                raise ValueError("surrogate must return finite posterior draws with shape (n_draws, n_query)")
            paths = draws.reshape(n_draws, len(candidates), n_scenarios)
            risk = benchmark.objective(paths)
            if acquisition == "kg":
                kg = knowledge_gradient or RiskKnowledgeGradient()
                choice = kg.select(
                    benchmark, surrogate, x_train, y_train, x[candidates], paths, risk, base, incumbent, step_seed
                )
            else:
                choice = select_designs(risk, incumbent, size, acquisition=acquisition)
        chosen = candidates[choice]
        new_y = benchmark.outcomes[chosen]
        new_score = float(benchmark.objective(new_y).min())
        duration += time.perf_counter() - start
        if independent is not None and acquisition != "kg":
            reference = benchmark.objective(independent().reshape(n_draws, len(candidates), n_scenarios))
            reference_choice = select_designs(reference, incumbent, size, acquisition=acquisition)
            changes.append(bool((choice != reference_choice).any()))
            # Score the counterfactual without adding its outcome to the model context.
            reference_score = float(benchmark.objective(benchmark.outcomes[candidates[reference_choice]]).min())
            same_history_regret_delta.append(min(incumbent, new_score) - min(incumbent, reference_score))
        incumbent = min(incumbent, new_score)
        available[chosen] = False
        selected = torch.cat((selected, chosen))
        observed = torch.cat((observed, new_y))
        evaluations.append(selected.numel() * cost)
        regret.append(incumbent - optimum)
    return Trace(
        selected.tolist(), evaluations, regret, duration, changes, same_history_regret_delta=same_history_regret_delta
    )
