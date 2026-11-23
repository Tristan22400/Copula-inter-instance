"""Run and plot the BoRisk acquisition comparison on the fixed scenario grids."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from copula_inter.artifacts import atomic_write
from eval.risk_bo import POINT_ACQUISITIONS, Trace
from eval.risk_bo_benchmarks import PROBLEMS
from eval.runners.hydra_cli import check_choices, hydra_entry
from eval.runners.risk_bo import RiskBOSpec, Run, run, validate

BUNDLE_ACQUISITIONS = ("ei", "mes", "kg", "ucb", "random")
PAPER_ACQUISITIONS = (*BUNDLE_ACQUISITIONS, *POINT_ACQUISITIONS)
MODELS = ("tabpfn-pre", "tabpfn-pre-copula", "gp", "gp-matern32")
PROBLEM_LABELS = dict(zip(PROBLEMS, ("Branin–Williams", "Marzat f₆", "Portfolio", "COVID-19 testing allocation")))
ACQUISITION_LABELS = dict(zip(PAPER_ACQUISITIONS, ("EI", "MES", "KG", "UCB", "Random", "ρ-random", "ρKG-apx", "ρKG")))
MODEL_LABELS: dict[str, tuple[str, str, Literal["-", "--", "-.", ":"]]] = {
    "tabpfn-pre": ("TabPFN-3.5", "#0072b2", "--"),
    "tabpfn-pre-copula": ("TabPFN-3.5 + copula", "#d55e00", "-"),
    "gp": ("GP Matérn-5/2", "#009e73", "-."),
    "gp-matern32": ("GP Matérn-3/2", "#884997", ":"),
    "random": ("Random", "#565656", "-"),
}


@dataclass
class CompareSpec:
    problems: list[str] = field(default_factory=lambda: list(PROBLEMS))
    acquisitions: list[str] = field(default_factory=lambda: list(PAPER_ACQUISITIONS))
    seed: int = 0
    n_seeds: int = 6
    pool_size: int = 24
    n_initial: int = 4
    n_designs: int = 10
    point_initial: int = 8
    point_budget: int = 20
    n_draws: int = 96
    gp_steps: int = 24
    kg_candidates: int = 2
    kg_recommendations: int = 8
    kg_fantasies: int = 3
    device: str = "cuda"
    asset_dir: str = "results/borisk_assets"
    checkpoint_dir: str | None = None
    out_dir: str = "results/borisk_paper_acquisitions"


def _specs(spec: CompareSpec) -> list[RiskBOSpec]:
    return [
        RiskBOSpec(
            problem=problem,
            risk_measure="var" if problem == "branin-williams" else "default",
            acquisition=acquisition,
            kg_candidates=spec.kg_candidates,
            kg_recommendations=spec.kg_recommendations,
            kg_fantasies=spec.kg_fantasies,
            point_initial=spec.point_initial,
            point_budget=spec.point_budget,
            asset_dir=spec.asset_dir,
            policies=["random"] if acquisition == "random" else list(MODELS),
            batch_sizes=[1],
            pool_size=spec.pool_size,
            seed=spec.seed,
            n_seeds=spec.n_seeds,
            n_initial=spec.n_initial,
            n_designs=spec.n_designs,
            n_draws=spec.n_draws,
            gp_steps=spec.gp_steps,
            scale_gp_inputs=True,
            device=spec.device,
            checkpoint_dir=spec.checkpoint_dir,
            out_dir=str(Path(spec.out_dir) / problem / acquisition),
        )
        for problem in spec.problems
        for acquisition in spec.acquisitions
    ]


def _validate(spec: CompareSpec) -> None:
    check_choices("problems", spec.problems, PROBLEMS)
    check_choices("acquisitions", spec.acquisitions, PAPER_ACQUISITIONS)
    for key, choices in (("problems", spec.problems), ("acquisitions", spec.acquisitions)):
        if not choices or len(set(choices)) != len(choices):
            raise ValueError(f"{key} must be nonempty and unique")
    for row in _specs(spec):
        validate(row)


def _plot(directory: Path, problem: str, acquisitions: dict[str, list[Run]], risk_label: str) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    rows = [
        ([a for a in BUNDLE_ACQUISITIONS if a in acquisitions], "Whole-design evaluations", "Best observed regret"),
        ([a for a in POINT_ACQUISITIONS if a in acquisitions], "One-scenario evaluations", "Recommendation regret"),
    ]
    rows = [(names, title, ylabel) for names, title, ylabel in rows if names]
    width = max(6.6, 3.05 * max(len(names) for names, _, _ in rows))
    legend_columns = 3 if width < 10 else 5
    fig = Figure(
        figsize=(width, 3.8 * len(rows) + 1.4),
        dpi=160,
        layout="constrained",
    )
    canvas = FigureCanvasAgg(fig)
    fig.suptitle(f"{PROBLEM_LABELS[problem]} · {risk_label}", fontsize=17, fontweight="bold")
    grid = fig.add_gridspec(len(rows) + 1, 1, height_ratios=[0.2] + [1] * len(rows))
    legend = fig.add_subplot(grid[0])
    legend.set_axis_off()
    policies = {row.policy for runs in acquisitions.values() for row in runs}
    handles = [
        Line2D([], [], color=color, linestyle=style, label=label, linewidth=2)
        for policy, (label, color, style) in MODEL_LABELS.items()
        if policy in policies
    ]
    legend.legend(handles=handles, loc="center", ncols=legend_columns, frameon=False, columnspacing=1.3)

    for i, (names, title, ylabel) in enumerate(rows, start=1):
        panel = fig.add_subfigure(grid[i])
        panel.suptitle(title, fontsize=12, fontweight="medium")
        axes = panel.subplots(1, len(names), sharex=True, sharey=True, squeeze=False)[0]
        for ax, acquisition in zip(axes, names, strict=True):
            runs = acquisitions[acquisition]
            for policy, (_, color, style) in MODEL_LABELS.items():
                traces = [row.trace for row in runs if row.policy == policy]
                if not traces:
                    continue
                mean = np.mean([trace.regret for trace in traces], axis=0)
                ax.step(traces[0].evaluations, mean, where="post", color=color, linestyle=style, linewidth=1.7)
                ax.scatter(traces[0].evaluations[-1], mean[-1], color=color, s=10, zorder=3)
            ax.set_title(f"{ACQUISITION_LABELS[acquisition]}  ·  n={len({row.seed for row in runs})}")
            ax.yaxis.grid(True, alpha=0.22)
            ax.spines[["right", "top"]].set_visible(False)
        axes[0].set_ylim(bottom=0)
        panel.supxlabel("Scenario evaluations")
        panel.supylabel(f"Mean {ylabel.lower()}")

    atomic_write(directory / "regret.png", canvas.print_png)


def run_comparison(spec: CompareSpec) -> None:
    for problem in spec.problems:
        runs: dict[str, list[Run]] = {}
        for settings in _specs(spec):
            if settings.problem != problem:
                continue
            run(settings)
            saved = json.loads((Path(settings.out_dir) / "traces.json").read_text())
            runs[settings.acquisition] = [
                Run(row["policy"], row["batch_size"], row["seed"], Trace(**row["trace"])) for row in saved["runs"]
            ]
            risk_label = (
                f"{saved['risk_measure'].replace('cvar', 'CVaR').replace('var', 'VaR')}({saved['risk_alpha']:.2f})"
            )
            _plot(Path(spec.out_dir) / problem, problem, runs, risk_label)


main = hydra_entry("risk_bo_compare", CompareSpec, run_comparison, validate=_validate)


if __name__ == "__main__":
    main()
