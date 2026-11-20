"""Export BeyondArena regression splits to .npz for the joint track (runs in the tabarena venv).

Usage (scripts/beyondarena/export_splits.sh activates the tabarena venv):
    python export_splits.py datasets='[telemonitoring_parkinsons_biomedical_voi-c608f1a3d898]' probe=true
    python export_splits.py                      # every core2k regression split

Per (dataset, repeat, fold) of the official subset, writes
<out_dir>/<dataset>/r<repeat>_f<fold>.npz with
    X_tr, X_te   float32 features after TabArena's model-agnostic preprocessing, fitted on the
                 training rows only, with categorical columns as ordinal codes (unseen -> NaN)
    y_tr, y_te   float64 targets
    g_tr, g_te   int64 group codes of `group_on` (-1 when the task has none)
    t_tr, t_te   float64 time of `time_on` in seconds or raw units (NaN when the task has none)
    group_feature  index of the group column among the X columns, -1 if TabArena removed it
    n_train_full   training rows before subsampling: training sets above max_train_rows (100k)
                   keep a seeded subsample (the preprocessing is fitted on all of them)
plus <dataset>/meta.json (task metadata, feature names, splits).

Deviation from the official pipeline (docs/beyondarena/deviations.md): semantic text
features are disabled (no precomputed embedding cache), and nothing is bagged.
Rerunning skips files that exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "subset": ["core2k"],
    "datasets": [],
    "out_dir": str(REPO / "data/beyondarena"),
    "probe": False,
    "max_train_rows": 100_000,
}


def _time_values(col) -> np.ndarray:  # pd.Series -> float seconds (datetimes) or raw numbers
    import pandas as pd

    if pd.api.types.is_datetime64_any_dtype(col):
        return col.astype("int64").to_numpy(dtype=np.float64) / 1e9
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().mean() > 0.99:
        return num.to_numpy(dtype=np.float64)
    return pd.to_datetime(col, errors="coerce").astype("int64").to_numpy(dtype=np.float64) / 1e9


def _numeric(frame, ref_categories: dict | None):  # categoricals -> codes against the training categories
    import pandas as pd

    out, cats = {}, {} if ref_categories is None else ref_categories
    for c in frame.columns:
        s = frame[c]
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_string_dtype(s):
            if ref_categories is None:
                cats[c] = list(pd.Series(s.astype("string").dropna().unique()).sort_values())
            codes = pd.Categorical(s.astype("string"), categories=cats[c]).codes.astype(np.float64)
            codes[codes < 0] = np.nan
            out[c] = codes
        elif pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(np.float64).to_numpy()
        else:
            out[c] = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    return np.column_stack([out[c] for c in frame.columns]).astype(np.float32), cats


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULTS), OmegaConf.from_cli())
    from tabarena.benchmark.preprocessing.model_agnostic_default_preprocessing import (
        TabArenaModelAgnosticPreprocessing,
    )
    from tabarena.benchmark.task.user_task import UserTask
    from tabarena.contexts import BeyondArenaContext

    ctx = BeyondArenaContext()
    filters: dict = {"problem_types": ["regression"], "subset": list(cfg.subset)}
    if cfg.datasets:
        filters["dataset_names"] = list(cfg.datasets)
    tmc = ctx._official_task_metadata_collection().subset_tasks(**filters).materialize()
    probed: set[str] = set()
    for ttm in tmc.tasks:
        name = ttm.tabarena_task_name
        ddir = Path(cfg.out_dir) / name
        ddir.mkdir(parents=True, exist_ok=True)
        task = UserTask.from_task_id_str(ttm.task_id_str).load()
        group_on, time_on = ttm.group_on, ttm.time_on
        splits = sorted((int(s.repeat), int(s.fold)) for s in ttm.splits_metadata.values())
        # The collection holds one entry per split of a dataset: merge them into one meta.json.
        old = json.loads((ddir / "meta.json").read_text()) if (ddir / "meta.json").exists() else {}
        splits = sorted({tuple(x) for x in old.get("splits", [])} | set(splits))
        meta = {
            "dataset": name,
            "task_type": ttm.task_type,
            "group_on": group_on,
            "time_on": time_on,
            "group_labels": str(ttm.group_labels),
            "eval_metric": ttm.eval_metric,
            "splits": splits,
        }
        for repeat, fold in splits:
            path = ddir / f"r{repeat}_f{fold}.npz"
            if cfg.probe and name in probed:
                break
            if path.exists() and not cfg.probe:
                continue
            X_tr, y_tr, X_te, y_te = task.get_train_test_split(fold=fold, repeat=repeat)
            g_cols = [group_on] if isinstance(group_on, str) else list(group_on or [])
            if g_cols:
                keys = X_tr[g_cols].astype("string").agg("|".join, axis=1)
                keys_te = X_te[g_cols].astype("string").agg("|".join, axis=1)
                uniq = {k: i for i, k in enumerate(sorted(set(keys) | set(keys_te)))}
                g_tr = keys.map(uniq).to_numpy(np.int64)
                g_te = keys_te.map(uniq).to_numpy(np.int64)
            else:
                g_tr, g_te = np.full(len(X_tr), -1, np.int64), np.full(len(X_te), -1, np.int64)
            t_tr = _time_values(X_tr[time_on]) if time_on else np.full(len(X_tr), np.nan)
            t_te = _time_values(X_te[time_on]) if time_on else np.full(len(X_te), np.nan)

            fg = TabArenaModelAgnosticPreprocessing(enable_sematic_text_features=False, verbosity=0)
            P_tr = fg.fit_transform(X_tr, y_tr)
            P_te = fg.transform(X_te)
            A_tr, cats = _numeric(P_tr, None)
            A_te, _ = _numeric(P_te, cats)
            cols = list(P_tr.columns)
            gf = cols.index(g_cols[0]) if len(g_cols) == 1 and g_cols[0] in cols else -1
            meta["features"] = cols
            meta["raw_columns"] = list(X_tr.columns)
            if cfg.probe:
                n_te_groups = len(set(g_te.tolist()))
                shared = len(set(g_tr.tolist()) & set(g_te.tolist())) if g_cols else 0
                print(
                    f"{name} r{repeat}f{fold}: P={len(X_tr)} N={len(X_te)} raw cols={len(X_tr.columns)} "
                    f"-> {len(cols)} features; group_on={group_on} in raw X: {bool(g_cols) and g_cols[0] in X_tr}, "
                    f"kept as feature: {gf >= 0}; test groups {n_te_groups}, shared with train {shared}; "
                    f"time_on={time_on} t_tr max {np.nanmax(t_tr) if time_on else None} t_te min "
                    f"{np.nanmin(t_te) if time_on else None}; NaN frac {np.isnan(A_tr).mean():.3f}"
                )
                print("  features:", cols[:40])
                probed.add(name)
                break
            # The joint track's marginal context is at most 50k rows, so keep a seeded 100k-row
            # subsample of large training sets (the preprocessing above was fitted on all of them).
            keep = np.arange(len(A_tr))
            if len(keep) > int(cfg.max_train_rows):
                rng = np.random.default_rng([0, repeat, fold])
                keep = np.sort(rng.choice(len(keep), int(cfg.max_train_rows), replace=False))
            np.savez_compressed(
                path,
                X_tr=A_tr[keep],
                X_te=A_te,
                y_tr=y_tr.to_numpy(np.float64)[keep],
                y_te=y_te.to_numpy(np.float64),
                g_tr=g_tr[keep],
                g_te=g_te,
                t_tr=t_tr[keep],
                t_te=t_te,
                group_feature=np.int64(gf),
                n_train_full=np.int64(len(A_tr)),
            )
        if not cfg.probe:
            (ddir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    main()
