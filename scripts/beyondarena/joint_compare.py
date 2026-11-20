"""Joint-track results through TabArena's own aggregation (beyondarena-joint venv).

Usage:
    python joint_compare.py                       # run=main under results/beyondarena/joint_ta

1. Flattens every results.pkl of the run into <run>/joint_scores.csv: one row per (marginal, variant,
   dataset, split) with every joint score, the point-prediction RMSE (metric_error) and the run facts.
2. For each joint score in `metrics` and each scope in `scopes` (all, iid, grouped, temporal): an Elo
   ranking of the joint methods (marginal + variant) with TabArena's own Elo
   (bencheval BenchmarkEvaluator.compute_elo: Bradley-Terry MLE, 400 scale, bootstrap 95% CI over tasks), on the
   BeyondArena per-dataset aggregation (one task per dataset, the score averaged over its splits). Only
   the datasets on which every compared method has every split enter (no imputation), the development
   datasets are left out (`exclude_dev`), and the mixed model (grouped tasks only) enters the grouped scope
   only. Elo is pinned to 1000 at `calibration`. Each dataset's scores are shifted by their minimum (Elo
   uses within-dataset comparisons only; the evaluator needs errors >= 0). Also the mean rank and the
   win rate. Writes <run>/compare/<metric>/<scope>.csv. (TabArena's context.compare cannot be used: it
   requires every method, cached baselines included, on every task of the subset.)
"""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "out_root": str(REPO / "results/beyondarena/joint_ta"),
    "run": "main",
    "metrics": ["energy", "joint_nll", "variogram_struct", "agg_group_crps", "agg_B32_crps", "sample_crps"],
    "scopes": ["all", "iid", "grouped", "temporal"],
    "exclude_dev": True,
    "calibration": "tabicl-ft+indep",
}
DEV = ("miami_housing", "qsar_tid_11", "wine_quality")
SCOPE_REGIME = {"iid": "iid", "grouped": "grouped", "temporal": "temporal"}


def _complete_datasets(df: pd.DataFrame, methods: list[str]) -> set[str]:
    """Datasets on which every method in `methods` has the same, full number of splits."""
    n = df[df.method.isin(methods)].groupby(["dataset", "method"]).size().unstack()
    n = n.reindex(columns=methods)
    ok = n.notna().all(axis=1) & n.eq(n.max(axis=1), axis=0).all(axis=1)
    return set(n.index[ok])


def _load(p: Path) -> dict:
    raw = p.read_bytes()
    return pickle.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULTS), OmegaConf.from_cli())
    run = Path(cfg.out_root) / str(cfg.run)
    tags: dict[str, str] = {}
    for f in run.glob("configs_*.json"):
        tags |= json.loads(f.read_text())["tags"]
    results, rows = [], []
    for p in sorted(run.glob("data/*/*/*/results.pkl")):
        r = _load(p)
        if "joint_metrics" not in r:
            continue
        base = str(r["framework"]).replace("[New] ", "")
        tag = tags.get(base, "lmm" if "LMM" in base.upper() else base)
        tm = r["task_metadata"]
        for variant, scores in r["joint_metrics"].items():
            rows.append(
                {
                    "marginal": tag,
                    "variant": variant,
                    "method": f"{tag}+{variant}",
                    "dataset": tm["name"],
                    "fold": tm["fold"],
                    "repeat": tm["repeat"],
                    "rmse": r["metric_error"],
                    **scores,
                    **{f"info_{k}": v for k, v in r.get("joint_info", {}).items()},
                }
            )
        results.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(run / "joint_scores.csv", index=False)
    print(f"{len(results)} split results, {len(df)} rows -> {run / 'joint_scores.csv'}")
    if df.empty:
        return

    from bencheval.evaluator import BenchmarkEvaluator

    regime = df.groupby("dataset")["info_regime"].first()
    for metric in cfg.metrics:
        if metric not in df:
            continue
        for scope in cfg.scopes:
            sub = df[df[metric].notna()]
            if scope != "all":
                sub = sub[sub.dataset.map(regime) == SCOPE_REGIME[str(scope)]]
            if scope != "grouped":
                sub = sub[sub.marginal != "lmm"]
            if cfg.exclude_dev:
                sub = sub[~sub.dataset.str.startswith(DEV)]
            methods = sorted(sub.method.unique())
            keep = _complete_datasets(sub, methods)
            if len(keep) < 2 or len(methods) < 2:
                continue
            per_ds = sub[sub.dataset.isin(keep)].groupby(["method", "dataset"])[metric].mean().rename("err")
            per_ds = per_ds.reset_index()
            per_ds["err"] -= per_ds.groupby("dataset")["err"].transform("min")
            per_ds = per_ds.rename(columns={"dataset": "task", "err": "metric_error"})
            ev = BenchmarkEvaluator(
                method_col="method", task_col="task", error_col="metric_error", columns_to_agg_extra=None
            )
            calib = str(cfg.calibration) if cfg.calibration in methods else None
            elo = ev.compute_elo(
                results_per_task=per_ds,
                calibration_framework=calib,
                calibration_elo=1000 if calib else None,
                BOOTSTRAP_ROUNDS=200,
            )
            wide = per_ds.pivot(index="task", columns="method", values="metric_error")
            rank = wide.rank(axis=1, method="average").mean()
            m = len(methods)
            win = {
                c: float(
                    (
                        (wide[c].to_numpy()[:, None] < wide.to_numpy()).sum()
                        + 0.5 * ((wide[c].to_numpy()[:, None] == wide.to_numpy()).sum() - len(wide))
                    )
                    / (len(wide) * (m - 1))
                )
                for c in methods
            }
            lb = elo.join(rank.rename("mean_rank")).join(pd.Series(win, name="winrate"))
            lb = lb.sort_values("elo", ascending=False).reset_index()
            out = run / "compare" / metric
            out.mkdir(parents=True, exist_ok=True)
            lb.to_csv(out / f"{scope}.csv", index=False)
            print(f"\n== {metric} | {scope}: {len(keep)} datasets, {m} methods (calibration {calib} = 1000)")
            print(lb.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
