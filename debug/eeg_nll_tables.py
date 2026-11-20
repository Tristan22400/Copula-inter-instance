"""NLL tables of the EEG benchmark (lower is better): marginal, copula and total NLL per method.

Every method is scored on the same episodes: the GP subset of each unit (eval/runners/eeg_gp_baselines.py),
joined by episode index with the TFM rows of eval/runners/eeg_joint_eval.py. AR rows use the episodes of
that subset that also carry the AR chain. One table for interpolation tasks, one for forecasting tasks
(Bruinsma reconstruction predicts the future of one electrode, so it is a forecasting task).

Usage (node, main venv, from the checkout root): python debug/eeg_nll_tables.py [out_dir]
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

L = {"tabicl": "TabICLv2", "tabpfn35_fast": "TabPFN-3.5", "kumo_small": "Kumo"}
INTERP = [
    ("Bruinsma Int (µV)", "arcnp_interpolation"),
    ("Hassan Int M=16", "hassan_int_N*_M16"),
    ("Hassan Int M=64", "hassan_int_N*_M64"),
]
FORECAST = [
    ("Bruinsma Rec (µV)", "arcnp_reconstruction"),
    ("Hassan For M=16", "hassan_for_N*_M16"),
    ("Hassan For M=64", "hassan_for_N*_M64"),
]


def _ci(v: np.ndarray) -> str:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "—"
    se = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else float("nan")
    return f"{v.mean():.3f} ± {1.96 * se:.3f}"


def _rows(out_dir: str, pattern: str) -> dict[str, dict[str, np.ndarray]] | None:
    """Per-method (marginal, copula, total) NLL arrays on the GP subset of every unit matching pattern."""
    acc: dict[str, dict[str, list[np.ndarray]]] = {}

    def add(method: str, marg: np.ndarray, cop: np.ndarray) -> None:
        d = acc.setdefault(method, {"marginal": [], "copula": [], "total": []})
        d["marginal"].append(marg)
        d["copula"].append(cop)
        d["total"].append(marg + cop)

    units = sorted(glob.glob(os.path.join(out_dir, pattern, "gp.npz")))
    if not units:
        return None
    for gp_path in units:
        g = np.load(gp_path)
        add("GP joint (MAP, BIC)", g["gp_marginal"], g["gp_copula"])
        add("GP diagonal", g["gp_marginal"], np.zeros_like(g["gp_marginal"]))
        for m, lab in L.items():
            mp = os.path.join(os.path.dirname(gp_path), f"{m}.npz")
            if not os.path.isfile(mp):
                continue
            a = np.load(mp)
            pos = {int(j): i for i, j in enumerate(a["index"])}
            ix = np.array([pos[int(j)] for j in g["index"]])
            marg = -a["ll_indep"][ix]
            gap = np.abs(marg - g[f"{m}_marginal"]).max()
            if gap > 1e-3:
                print(f"WARNING {gp_path} {m}: marginal NLL differs from the main run by {gap:.2e}", file=sys.stderr)
            add(f"{lab} + indep", marg, np.zeros_like(marg))
            add(f"{lab} + copula (ours)", marg, a["copula_nll"][ix])
            add(f"{lab} + GP corr.", marg, g[f"{m}_gpcorr_copula"])
            ar_total = -a["ll_ar"][ix]
            add(f"{lab} + AR (subset)", np.where(np.isfinite(ar_total), marg, np.nan), ar_total - marg)
    return {k: {kk: np.concatenate(vv) for kk, vv in d.items()} for k, d in acc.items()}


def table(out_dir: str, title: str, groups: list[tuple[str, str]]) -> None:
    print(f"\n### {title}\n")
    print("| Task | Method | n | Marginal NLL | Copula NLL | Total NLL |")
    print("|---|---|---|---|---|---|")
    for label, pattern in groups:
        rows = _rows(out_dir, pattern)
        if rows is None:
            print(f"| {label} | (GP subset not scored yet) | | | | |")
            continue
        for method, d in rows.items():
            n = int(np.isfinite(d["total"]).sum())
            print(f"| {label} | {method} | {n} | {_ci(d['marginal'])} | {_ci(d['copula'])} | {_ci(d['total'])} |")


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "results/npbench/eeg"
    print("NLL per target value (lower is better), mean ± 1.96 SE, on each unit's GP subset (Hassan pooled over N).")
    table(out_dir, "Interpolation", INTERP)
    table(out_dir, "Forecasting", FORECAST)


if __name__ == "__main__":
    main()
