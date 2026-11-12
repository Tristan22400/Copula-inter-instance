"""Compare training runs from wandb: full NLL = marginal + copula, each against independence and the oracle.

Usage:
    python -m eval.runners.wandb_report filter=_PN_ since=2026-11-09
    python -m eval.runners.wandb_report ids=[splmz6y3,0jkldstb] out=report.json

Runs are labelled by the basename of training.ckpt_dir. All values are nats per test point, read at
the last logged validation step:
    full          val/y_nll_total         learned marginal + copula, on the marginal's own z
    marginal      val/y_nll_marginal      the same run's full NLL under independence (copula = 0)
    copula        val/y_nll_copula        full - marginal: < 0 helps, > 0 is worse than independence
    copula exact  oracle_diag/copula_nll  copula NLL on analytic z (exact marginal): the GP correlation only
    oracle        the GP posterior: oracle full = oracle marginal + oracle copula, oracle copula = -headroom
Full NLLs are comparable only between runs scored on the same validation set (same oracle full), and
the marginal part only between runs whose marginal setup (listed at the end) differs as intended.
The size grid (runs trained with baselines.size_grid_enabled=true) gives the exact-z copula NLL per
(P, N) cell as size_fit/<cell>/gap_nll - headroom; independence is 0 and the oracle is -headroom.
"""

from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from eval.runners.hydra_cli import hydra_entry

VAL_KEYS: dict[str, str] = {
    "full": "val/y_nll_total",
    "marginal": "val/y_nll_marginal",
    "copula": "val/y_nll_copula",
    "copula_exact": "oracle_diag/copula_nll",
    "oracle_full": "val/y_nll_oracle_posterior",
    "headroom": "oracle_diag/copula_headroom",
}
_CELL_HEADROOM = re.compile(r"^val/size_fit/P(\d+)_N(\d+)/copula_headroom$")


def gap_key(P: int, N: int) -> str:
    return f"val/size_fit/P{P}_N{N}/gap_nll"


@dataclass
class RunData:
    """What wandb holds for one run: identity, validation history rows, size-grid headroom and history rows."""

    label: str
    run_id: str
    state: str
    marginal_setup: str
    val_rows: list[dict[str, float]]
    headroom: dict[tuple[int, int], float] = field(default_factory=dict)
    grid_rows: list[dict[str, float]] = field(default_factory=list)


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def decompose(row: dict[str, float]) -> dict[str, float]:
    """One validation row split into marginal + copula for the run and the oracle, and the excess of each part."""
    out = {name: float(row.get(key, math.nan)) for name, key in VAL_KEYS.items()}
    out["step"] = float(row.get("_step", math.nan))
    out["oracle_copula"] = -out["headroom"]
    out["oracle_marginal"] = out["oracle_full"] - out["oracle_copula"]
    out["excess"] = out["full"] - out["oracle_full"]
    out["excess_marginal"] = out["marginal"] - out["oracle_marginal"]
    out["excess_copula"] = out["copula"] - out["oracle_copula"]
    return out


def best_row(rows: list[dict[str, float]], key: str) -> dict[str, float] | None:
    """The row with the lowest finite value of key, or None."""
    scored = [r for r in rows if _finite(r.get(key))]
    return min(scored, key=lambda r: r[key]) if scored else None


def _mean_by_P(values: dict[tuple[int, int], float]) -> dict[int, float]:
    Ps = sorted({P for P, _ in values})
    return {P: sum(v for (p, _), v in values.items() if p == P) / sum(1 for p, _ in values if p == P) for P in Ps}


def grid_copula(row: dict[str, float], headroom: dict[tuple[int, int], float]) -> dict[str, Any]:
    """Exact-z copula NLL per cell, its per-P and grid means, and the same for the oracle (-headroom)."""
    cells = {
        pn: float(row[gap_key(*pn)]) - H for pn, H in headroom.items() if _finite(row.get(gap_key(*pn))) and _finite(H)
    }
    if not cells:
        return {}
    oracle = {pn: -headroom[pn] for pn in cells}
    mean = sum(cells.values()) / len(cells)
    oracle_mean = sum(oracle.values()) / len(oracle)
    return {
        "step": float(row.get("_step", math.nan)),
        "cells": {f"P{P}_N{N}": v for (P, N), v in cells.items()},
        "per_P": _mean_by_P(cells),
        "oracle_per_P": _mean_by_P(oracle),
        "mean": mean,
        "oracle_mean": oracle_mean,
        "captured": mean / oracle_mean if oracle_mean < 0 else math.nan,
    }


def summarize_run(run: RunData) -> dict[str, Any] | None:
    """The JSON-able record of one run, or None when it has no validation row yet."""
    rows = [r for r in run.val_rows if _finite(r.get(VAL_KEYS["full"]))]
    if not rows:
        return None
    best_full = best_row(rows, VAL_KEYS["full"])
    best_exact = best_row(rows, VAL_KEYS["copula_exact"])
    grid_rows = [g for g in run.grid_rows if any(_finite(v) for k, v in g.items() if k != "_step")]
    return {
        "label": run.label,
        "run_id": run.run_id,
        "state": run.state,
        "marginal_setup": run.marginal_setup,
        "last": decompose(rows[-1]),
        "best_full": decompose(best_full) if best_full else None,
        "best_copula_exact": decompose(best_exact) if best_exact else None,
        "grid_first": grid_copula(grid_rows[0], run.headroom) if grid_rows else {},
        "grid_last": grid_copula(grid_rows[-1], run.headroom) if grid_rows else {},
    }


def _fmt(x: object, spec: str = "+.4f") -> str:
    return format(x, spec) if _finite(x) else "—"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def _sort_value(record: dict[str, Any], sort_by: str) -> float:
    if sort_by == "grid":
        value = record["grid_last"].get("mean", math.nan)
    else:
        value = record["last"].get(sort_by, math.nan)
    return value if _finite(value) else math.inf


def render(records: list[dict[str, Any]]) -> str:
    """Markdown: one decomposition table per validation set, the best steps, the size grid, the marginal setups."""
    out: list[str] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(_fmt(r["last"]["oracle_full"]), []).append(r)
    for oracle_full, group in groups.items():
        o = group[0]["last"]
        out.append(
            f"## Validation set with oracle full {oracle_full} = marginal {_fmt(o['oracle_marginal'])} "
            f"+ copula {_fmt(o['oracle_copula'])}\n"
        )
        out.append(
            markdown_table(
                [
                    "run",
                    "state",
                    "step",
                    "marginal (= indep.)",
                    "+ copula",
                    "**full**",
                    "excess: marg. + cop.",
                    "copula exact z",
                ],
                [
                    [
                        r["label"],
                        r["state"],
                        str(int(r["last"]["step"])),
                        _fmt(r["last"]["marginal"]),
                        _fmt(r["last"]["copula"]),
                        f"**{_fmt(r['last']['full'])}**",
                        f"{_fmt(r['last']['excess_marginal'])} {_fmt(r['last']['excess_copula'])}",
                        _fmt(r["last"]["copula_exact"]),
                    ]
                    for r in group
                ],
            )
            + "\n"
        )

    def _at(rec: dict[str, Any] | None, key: str) -> str:
        return f"{_fmt(rec[key])} @ {int(rec['step'])}" if rec else "—"

    out.append("## Best validation step\n")
    out.append(
        markdown_table(
            ["run", "best full", "best copula exact z", "last copula exact z"],
            [
                [
                    r["label"],
                    _at(r["best_full"], "full"),
                    _at(r["best_copula_exact"], "copula_exact"),
                    _fmt(r["last"]["copula_exact"]),
                ]
                for r in records
            ],
        )
        + "\n"
    )
    gridded = [r for r in records if r["grid_last"]]
    if gridded:
        ref = gridded[0]["grid_last"]
        Ps = sorted(ref["per_P"])
        out.append("## Size grid: copula NLL on exact z, mean over N (independence = 0, oracle = -headroom)\n")
        rows = [
            ["**oracle**", ""] + [_fmt(ref["oracle_per_P"][P]) for P in Ps] + [_fmt(ref["oracle_mean"]), "100%", ""]
        ]
        for r in gridded:
            g0, g1 = r["grid_first"], r["grid_last"]
            rows.append(
                [r["label"], str(int(g1["step"]))]
                + [_fmt(g1["per_P"].get(P)) for P in Ps]
                + [_fmt(g1["mean"]), _fmt(g1["captured"], ".1%"), f"{_fmt(g0.get('mean'))} @ {int(g0['step'])}"]
            )
        out.append(
            markdown_table(["run", "step"] + [f"P{P}" for P in Ps] + ["grid mean", "captured", "first logged"], rows)
            + "\n"
        )
    out.append("## Marginal setup\n")
    out.extend(f"- {r['label']} ({r['run_id']}, {r['state']}): {r['marginal_setup']}" for r in records)
    return "\n".join(out)


# ---- wandb access ---------------------------------------------------------------------------------


def _get(config: dict[str, Any], dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _base(path: Any) -> str:
    return os.path.basename(str(path).rstrip("/")) if path else "-"


def marginal_setup(config: dict[str, Any]) -> str:
    """The settings that decide the run's marginal and starting point, in one line."""
    source = _get(config, "data.z_train_source") or "tabicl"
    parts = [f"z_train_source={source}"]
    if source == "tabicl":
        parts.append(f"pit_ckpt={_base(_get(config, 'tabicl.pit_ckpt'))}")
    else:
        parts.append(f"marginal_ckpt={_base(_get(config, 'data.z_train_marginal_ckpt'))}")
    parts.append(f"P=[{_get(config, 'data.P_min')},{_get(config, 'data.P_max')}]")
    parts.append(f"N=[{_get(config, 'data.N_min')},{_get(config, 'data.N_max')}]")
    resume = _get(config, "training.resume_ckpt")
    if resume:
        parts.append(f"resumed from {_base(os.path.dirname(str(resume)))}/{_base(resume)}")
    return ", ".join(parts)


def run_label(run: Any) -> str:
    """The basename of the run's training.ckpt_dir, else its wandb name."""
    ckpt_dir = _get(dict(run.config), "training.ckpt_dir")
    return _base(ckpt_dir) if ckpt_dir else str(run.name)


def fetch_run(run: Any) -> RunData:
    """Read one wandb.apis.public.Run into RunData."""
    config = dict(run.config)
    label = run_label(run)
    val_rows = [
        {k: v for k, v in row.items() if v is not None} for row in run.scan_history(keys=[*VAL_KEYS.values(), "_step"])
    ]
    headroom: dict[tuple[int, int], float] = {}
    for key, value in run.summary.items():
        m = _CELL_HEADROOM.match(key)
        if m and _finite(value):
            headroom[(int(m.group(1)), int(m.group(2)))] = float(value)
    grid_rows: list[dict[str, float]] = []
    if headroom:
        keys = [gap_key(*pn) for pn in sorted(headroom)] + ["_step"]
        grid_rows = [{k: v for k, v in row.items() if v is not None} for row in run.scan_history(keys=keys)]
    return RunData(label, run.id, run.state, marginal_setup(config), val_rows, headroom, grid_rows)


@dataclass
class WandbReportSpec:
    """Compare training runs from wandb on full NLL (marginal + copula) and the size grid."""

    # entity/project to read.
    project: str = "iliass-lasri-team/copula-inter-instance"
    # Run ids to read; when set, filter/since/states/max_runs are ignored.
    ids: list[str] = field(default_factory=list)
    # Regex searched in each run's label (basename of training.ckpt_dir, else the run name).
    filter: str | None = None
    # Only runs created on or after this date (YYYY-MM-DD).
    since: str | None = None
    # Keep runs in these wandb states (running, finished, crashed, failed, killed); empty keeps all.
    states: list[str] = field(default_factory=lambda: ["running", "finished"])
    # Newest runs to scan before filtering.
    max_runs: int = 200
    # Sort key: a decomposition field (full, marginal, copula, copula_exact, excess, ...) or grid.
    sort_by: str = "full"
    # Also write the per-run records as JSON here.
    out: str | None = None


def validate_spec(spec: WandbReportSpec) -> None:
    if spec.filter is not None:
        re.compile(spec.filter)
    if spec.since is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", spec.since):
        raise ValueError(f"since={spec.since!r} is not YYYY-MM-DD")


def run(spec: WandbReportSpec) -> None:
    import wandb

    api = wandb.Api()
    if spec.ids:
        runs = [api.run(f"{spec.project}/{rid}") for rid in spec.ids]
    else:
        filters = {"created_at": {"$gte": f"{spec.since}T00:00:00"}} if spec.since else None
        runs = []
        for i, r in enumerate(api.runs(spec.project, filters=filters, order="-created_at")):
            if i >= spec.max_runs:
                break
            if spec.states and r.state not in spec.states:
                continue
            if spec.filter is not None and not re.search(spec.filter, run_label(r)):
                continue
            runs.append(r)
    with ThreadPoolExecutor(max_workers=8) as pool:  # history reads are network-bound
        data = list(pool.map(fetch_run, runs))
    records = [rec for rec in (summarize_run(d) for d in data) if rec is not None]
    records.sort(key=lambda rec: _sort_value(rec, spec.sort_by))
    print(render(records) if records else "No run with a validation row matched.")
    if spec.out:
        with open(spec.out, "w") as fh:
            json.dump(records, fh, indent=1, default=str)
        print(f"\nwrote {len(records)} records to {spec.out}")


main = hydra_entry("wandb_report", WandbReportSpec, run, validate=validate_spec)

if __name__ == "__main__":
    main()
