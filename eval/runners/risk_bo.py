"""Compare TabPFN, its matching copulas, and scenario-level GPs on BoRisk benchmarks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch

from copula_inter.artifacts import artifact_identity, atomic_write, canonical_digest
from copula_inter.paths import CHECKPOINTS_DIR
from eval.environment_calibration import PROBLEM as CALIBRATION_PROBLEM
from eval.environment_calibration import IndependentOutputGP, calibration_pool
from eval.risk_bo import (
    ACQUISITIONS,
    POINT_ACQUISITIONS,
    BundlePool,
    RiskKnowledgeGradient,
    RiskPool,
    Surrogate,
    Trace,
    run_bo,
)
from eval.risk_bo_benchmarks import PROBLEMS, benchmark_design, prepare_problem
from eval.risk_bo_models import GPRiskSurrogate, TabPFNRiskSurrogate, checkpoint_paths
from eval.risk_bo_point import run_point_bo
from eval.runners.hydra_cli import check_choices, hydra_entry

DEFAULT_POLICIES = ("tabpfn-pre", "tabpfn-pre-copula", "tabpfn-ft", "tabpfn-ft-copula", "gp", "gp-log")
CALIBRATION_POLICIES = ("tabpfn-pre", "tabpfn-pre-copula", "gp", "gp-matern32", "gp-independent-outputs")
POLICIES = (*DEFAULT_POLICIES, "gp-matern32", "gp-independent-outputs", "random")


@dataclass
class RiskBOSpec:
    problem: str = "branin-williams"
    risk_measure: str = "default"
    acquisition: str = "ei"
    kg_candidates: int = 3
    kg_recommendations: int = 8
    kg_fantasies: int = 4
    point_initial: int = 8
    point_budget: int = 24
    scenario_seed: int = 43
    asset_dir: str = "results/borisk_assets"
    covid_python: str | None = None
    prepare_only: bool = False
    policies: list[str] | None = None
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2])
    pool_size: int = 256
    pool_seed: int = 42
    seed: int = 0
    n_seeds: int = 30
    n_initial: int = 6
    n_designs: int = 20
    n_draws: int = 512
    query_batch: int = 512
    k_folds: int = 10
    gp_steps: int = 80
    scale_gp_inputs: bool = False
    device: str = "cuda"
    checkpoint_dir: str | None = None
    out_dir: str = "results/risk_bo"


@dataclass
class Run:
    policy: str
    batch_size: int
    seed: int
    trace: Trace


def validate(spec: RiskBOSpec) -> None:
    check_choices("problem", [spec.problem], (*PROBLEMS, CALIBRATION_PROBLEM))
    check_choices("risk_measure", [spec.risk_measure], ("default", "var", "cvar"))
    check_choices("acquisition", [spec.acquisition], ACQUISITIONS)
    if spec.problem == CALIBRATION_PROBLEM:
        if spec.risk_measure != "default" or spec.acquisition in POINT_ACQUISITIONS:
            raise ValueError("environment-calibration uses squared error and whole-design evaluations")
    elif spec.policies is not None and "gp-independent-outputs" in spec.policies:
        raise ValueError("gp-independent-outputs requires environment-calibration")
    if spec.pool_size < 1:
        raise ValueError("pool_size must be positive")
    if min(spec.seed, spec.pool_seed, spec.scenario_seed) < 0:
        raise ValueError("seed, pool_seed and scenario_seed must be nonnegative")
    if spec.prepare_only:
        return
    if spec.policies is not None:
        check_choices("policies", spec.policies, POLICIES)
        if not spec.policies or len(set(spec.policies)) != len(spec.policies):
            raise ValueError("policies must be nonempty and unique")
        if ("random" in spec.policies) != (spec.acquisition == "random") or (
            spec.acquisition == "random" and spec.policies != ["random"]
        ):
            raise ValueError("random acquisition uses only the model-free random policy")
    if not spec.batch_sizes or len(set(spec.batch_sizes)) != len(spec.batch_sizes) or set(spec.batch_sizes) - {1, 2}:
        raise ValueError("batch_sizes must contain one or both of 1 and 2")
    if spec.acquisition in ("mes", "ucb", "kg", *POINT_ACQUISITIONS):
        if spec.batch_sizes != [1]:
            raise ValueError(f"{spec.acquisition} requires batch_sizes=[1]")
    if spec.acquisition in ("kg", "rho-kg-apx", "rho-kg"):
        if min(spec.kg_candidates, spec.kg_recommendations, spec.kg_fantasies) < 1:
            raise ValueError("knowledge gradient requires positive shortlist sizes and fantasy count")
    if spec.acquisition in POINT_ACQUISITIONS and not 2 <= spec.point_initial < spec.point_budget:
        raise ValueError("need 2 <= point_initial < point_budget")
    if not 2 <= spec.n_initial < spec.n_designs <= spec.pool_size:
        raise ValueError("need 2 <= n_initial < n_designs <= pool_size")
    if spec.n_seeds < 1 or spec.n_draws < 1 or spec.query_batch < 1 or spec.k_folds < 2 or spec.gp_steps < 1:
        raise ValueError("need positive n_seeds, n_draws, query_batch, gp_steps and k_folds >= 2")


def _identity(spec: RiskBOSpec, root: Path, policies: list[str], oracle_identity: object) -> str:
    paths = set()
    for policy in policies:
        if policy.startswith("tabpfn"):
            marginal, copula = checkpoint_paths(root, fine_tuned=policy.startswith("tabpfn-ft"))
            if marginal is not None:
                paths.add(marginal)
            if policy.endswith("-copula"):
                paths.add(copula)
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"Missing BO checkpoint {path}. Set checkpoint_dir to its checkpoints root.")
    config = asdict(spec) | {"policies": policies}
    if not spec.scale_gp_inputs:
        config.pop("scale_gp_inputs")
    if spec.acquisition == "ei":
        config.pop("acquisition")
    if spec.acquisition not in ("kg", "rho-kg-apx", "rho-kg"):
        for key in ("kg_candidates", "kg_recommendations", "kg_fantasies"):
            config.pop(key)
    if spec.acquisition not in POINT_ACQUISITIONS:
        config.pop("point_initial")
        config.pop("point_budget")
    # Preserve compatibility with completed Branin–Williams runs.
    payload: dict[str, object] = {
        "spec": config,
        "checkpoints": {str(p): artifact_identity(p) for p in sorted(paths)},
    }
    if spec.problem == "branin-williams" and spec.risk_measure in ("default", "cvar"):
        for key in ("problem", "risk_measure", "scenario_seed", "asset_dir", "covid_python", "prepare_only"):
            config.pop(key)
    else:
        payload["oracle"] = oracle_identity
    return canonical_digest(payload)


def _model(policy: str, spec: RiskBOSpec, root: Path) -> Surrogate:
    if policy == "gp-independent-outputs":
        return IndependentOutputGP(n_steps=spec.gp_steps)
    if policy.startswith("gp"):
        return GPRiskSurrogate(
            log_output=policy == "gp-log",
            n_steps=spec.gp_steps,
            kernel_name="matern32" if policy == "gp-matern32" else "matern52",
        )
    return TabPFNRiskSurrogate(
        root,
        device=spec.device,
        fine_tuned=policy.startswith("tabpfn-ft"),
        coupled=policy.endswith("-copula"),
        query_batch=spec.query_batch,
        k_folds=spec.k_folds,
    )


def _summary(runs: list[Run]) -> dict[str, object]:
    groups: dict[tuple[str, int], dict[int, Trace]] = {}
    for rec in runs:
        groups.setdefault((rec.policy, rec.batch_size), {})[rec.seed] = rec.trace

    rows = {}
    for (method, q), group in sorted(groups.items()):
        traces = list(group.values())
        changes = [changed for trace in traces for changed in trace.dependence_changed_choice]
        immediate_deltas = [delta for trace in traces for delta in trace.same_history_regret_delta]
        rows[f"{method}/q{q}"] = {
            "count": len(traces),
            "mean_final_regret": float(np.mean([tr.regret[-1] for tr in traces])),
            "median_wall_seconds": float(np.median([tr.wall_seconds for tr in traces])),
            "same_history_choice_change_fraction": float(np.mean(changes)) if changes else None,
            "same_history_mean_one_step_regret_delta": float(np.mean(immediate_deltas)) if immediate_deltas else None,
        }

    comparisons = []
    for method in ("tabpfn-pre-copula", "tabpfn-ft-copula"):
        for q in (1, 2):
            for reference in (method.removesuffix("-copula"), "gp", "gp-log", "gp-matern32", "gp-independent-outputs"):
                comparisons.append(((method, q), (reference, q)))
    comparisons.extend(((method, 2), (method, 1)) for method in POLICIES)
    deltas = {}
    rng = np.random.default_rng(0)
    for test_key, ref_key in comparisons:
        test, ref = groups.get(test_key, {}), groups.get(ref_key, {})
        seeds = sorted(test.keys() & ref.keys())
        if not seeds:
            continue
        diff = np.array([test[s].regret[-1] - ref[s].regret[-1] for s in seeds])
        interval = None
        if len(diff) > 1:
            bootstrap = rng.choice(diff, size=(5000, len(diff)), replace=True).mean(axis=1)
            interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        name = f"{test_key[0]}/q{test_key[1]} minus {ref_key[0]}/q{ref_key[1]}"
        deltas[name] = {"paired_seeds": len(seeds), "mean_regret_delta": float(diff.mean()), "ci95": interval}
    return {"methods": rows, "paired_final_regret": deltas}


def _plot(path: Path, runs: list[Run], acquisition: str, evaluation_label: str) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    batches = sorted({rec.batch_size for rec in runs})
    fig = Figure(figsize=(6 * len(batches), 4), layout="constrained")
    canvas = FigureCanvasAgg(fig)
    axes = fig.subplots(1, len(batches), squeeze=False)
    point_protocol = acquisition in POINT_ACQUISITIONS
    for ax, q in zip(axes[0], batches, strict=True):
        for policy in POLICIES:
            traces = [rec.trace for rec in runs if rec.batch_size == q and rec.policy == policy]
            if traces:
                mean = np.mean([trace.regret for trace in traces], axis=0)
                ax.step(traces[0].evaluations, mean, where="post", label=policy)
        ax.set(
            title="One scenario per round" if point_protocol else f"{q} design{'s' if q > 1 else ''} per round",
            xlabel=evaluation_label,
            ylabel="Mean recommendation regret" if point_protocol else "Mean best observed regret",
        )
        ax.legend()
    atomic_write(path, canvas.print_png)


def run(spec: RiskBOSpec) -> None:
    validate(spec)
    root = Path(spec.checkpoint_dir).expanduser().resolve() if spec.checkpoint_dir else CHECKPOINTS_DIR
    out = Path(spec.out_dir)
    if spec.problem != "branin-williams" and out == Path("results/risk_bo"):
        out = out / spec.problem
    if spec.acquisition != "ei" and spec.out_dir == "results/risk_bo":
        out = out / spec.acquisition
    device = "cpu" if spec.prepare_only else spec.device
    benchmark: BundlePool | RiskPool
    oracle_identity: object
    metadata: dict[str, object]
    if spec.problem == CALIBRATION_PROBLEM:
        benchmark, oracle_identity = calibration_pool(spec.pool_size, spec.pool_seed, device)
        metadata = {
            "objective": "sum-squared-error",
            "readings_per_design": benchmark.inputs.shape[1],
            "known_optimum": benchmark.optimum,
            "pool_minimum_loss": float(benchmark.objective(benchmark.outcomes).min()),
        }
        defaults = list(CALIBRATION_POLICIES)
        evaluation_label = "Simulator evaluations"
    else:
        assets = Path(spec.asset_dir).expanduser().resolve()
        design = benchmark_design(spec.problem, spec.pool_size, spec.pool_seed, spec.scenario_seed)
        if spec.risk_measure != "default":
            design = replace(design, measure=cast(Literal["var", "cvar"], spec.risk_measure))
        benchmark, oracle_identity = prepare_problem(design, assets, spec.covid_python, assets / "cache", device)
        metadata = {
            "risk_measure": design.measure,
            "risk_alpha": design.alpha,
            "scenarios_per_design": design.inputs.shape[1],
            "simulator_replications_per_scenario": 10 if spec.problem == "covid-testing" else 1,
        }
        defaults = list(DEFAULT_POLICIES if spec.problem == "branin-williams" else DEFAULT_POLICIES[:-1])
        evaluation_label = "Scenario evaluations"
    n_scenarios = benchmark.inputs.shape[1]
    print(f"{spec.problem}/{spec.acquisition}: {spec.pool_size} designs, {metadata}", flush=True)
    if spec.prepare_only:
        return
    if spec.acquisition == "random":
        defaults = ["random"]
    policies = spec.policies if spec.policies is not None else defaults
    if "gp-log" in policies:
        outcomes = benchmark.outcomes if isinstance(benchmark, BundlePool) else benchmark.losses
        if not bool((outcomes > 0).all()):
            raise ValueError("gp-log requires strictly positive outcomes. Use gp for this benchmark.")
    result_path = out / "traces.json"
    digest = _identity(spec, root, policies, oracle_identity)
    runs: list[Run] = []
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved["fingerprint"] != digest:
            raise ValueError(f"{result_path} belongs to another experiment. Choose a new out_dir.")
        runs = [Run(row["policy"], row["batch_size"], row["seed"], Trace(**row["trace"])) for row in saved["runs"]]

    def save() -> None:
        document = {
            "fingerprint": digest,
            "spec": asdict(spec) | {"policies": policies},
            "oracle": oracle_identity,
            **metadata,
            "runs": [asdict(record) for record in runs],
            "summary": _summary(runs),
        }

        def write(tmp: str) -> None:
            Path(tmp).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")

        atomic_write(result_path, write)

    done = {(rec.policy, rec.batch_size, rec.seed) for rec in runs}
    pool = benchmark.inputs
    point_protocol = spec.acquisition in POINT_ACQUISITIONS
    kg = RiskKnowledgeGradient(spec.kg_candidates, spec.kg_recommendations, spec.kg_fantasies)
    if point_protocol and spec.point_budget > len(pool) * n_scenarios:
        raise ValueError("point_budget exceeds the number of design–scenario rows")
    for policy in policies:
        missing = [
            (q, seed)
            for q in spec.batch_sizes
            for seed in range(spec.seed, spec.seed + spec.n_seeds)
            if (policy, q, seed) not in done
        ]
        if not missing:
            continue
        model = None if policy == "random" else _model(policy, spec, root)
        model_pool = benchmark
        if spec.scale_gp_inputs and policy.startswith("gp"):
            lo = pool.amin((0, 1))
            width = pool.amax((0, 1)) - lo
            model_pool = replace(benchmark, inputs=(pool - lo) / width.clamp_min(1e-12))
        for q, seed in missing:
            init_rng = torch.Generator().manual_seed(seed)
            if point_protocol:
                assert model is not None and isinstance(model_pool, RiskPool)
                initial = torch.randperm(len(pool) * n_scenarios, generator=init_rng)[: spec.point_initial]
                trace = run_point_bo(
                    model_pool,
                    initial.to(pool.device),
                    model,
                    seed=seed,
                    budget=spec.point_budget,
                    n_draws=spec.n_draws,
                    acquisition=spec.acquisition,
                    kg=kg,
                )
            else:
                initial = torch.randperm(len(pool), generator=init_rng)[: spec.n_initial].to(pool.device)
                trace = run_bo(
                    pool,
                    initial,
                    model,
                    seed=seed,
                    n_designs=spec.n_designs,
                    batch_size=q,
                    n_draws=spec.n_draws,
                    benchmark=model_pool,
                    acquisition=spec.acquisition,
                    knowledge_gradient=kg,
                )
            runs.append(Run(policy, q, seed, trace))
            save()
            print(
                f"{policy} q={q} seed={seed} final regret={trace.regret[-1]:.6g} time={trace.wall_seconds:.2f}s",
                flush=True,
            )
        del model
    _plot(out / "regret.png", runs, spec.acquisition, evaluation_label)


main = hydra_entry("risk_bo", RiskBOSpec, run, validate=validate)


if __name__ == "__main__":
    main()
