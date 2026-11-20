"""BeyondArena study report: tables, hypothesis tests and figures (docs/beyondarena/preregistration.md).

Usage:
    python -m eval.runners.beyondarena_report                 # every result found under results/beyondarena

Inputs:
- joint track: results/beyondarena/joint/<tag>/<model>/<dataset>/<split>.csv.gz (beyondarena_joint_eval);
  tag "main" is the confirmatory run, the other tags are ablations (ablation_tags below).
- official track: results/beyondarena/official/compare/<scope>/ (scripts/beyondarena/official_compare.py).
Statistics: each metric is averaged over the splits and blocks of a dataset; the CI is a percentile
bootstrap of the median over datasets; tests are one-sided Wilcoxon signed-rank over datasets
(Holm over the secondary copulas). Development datasets are left out of every test.
Outputs under out_dir: per-dataset tables (.csv), hypotheses.json, and fig_*.pdf / .png.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from eval.runners.hydra_cli import hydra_entry

__all__ = ["BeyondArenaReportSpec", "bootstrap_median", "dataset_table", "gains"]

PRIMARY = "tabicl-ft"
METHOD_STYLE = {  # fixed categorical order (dataviz palette, validated): colour + marker per method
    "copula": ("#2a78d6", "o"),
    "indep": ("#eb6834", "s"),
    "exch": ("#1baf7a", "^"),
    "lmm": ("#eda100", "D"),
}
REGIME_ORDER = ("grouped", "temporal", "iid")
METRICS = (
    "es",
    "vs",
    "joint_nll",
    "mean_crps",
    "mean_cov90",
    "mean_width90",
    "grid_crps",
    "row_crps_sample",
    "rho_pred",
)


@dataclass
class BeyondArenaReportSpec:
    """Where the results are and which runs are ablations."""

    results_dir: str | None = None  # default copula_inter.paths.BEYONDARENA_RESULTS_DIR
    out_dir: str | None = None  # default <results_dir>/report
    dev_datasets: list[str] = field(
        default_factory=lambda: ["miami_housing-77ada7242bed", "qsar_tid_11-34170685abe8", "wine_quality-6ad3969edbaa"]
    )
    # tag -> label of the ablation runs of the primary model (results/beyondarena/joint/<tag>/tabicl-ft).
    ablation_tags: dict[str, str] = field(
        default_factory=lambda: {
            "main": "default (struct, rank full, P=1000)",
            "xfit-random": "random cross-fitting",
            "drop-group": "group id dropped",
            "rank1": "rank 1",
            "rank4": "rank 4",
            "rank16": "rank 16",
            "rank64": "rank 64",
            "pcop256": "P_cop = 256",
            "pcop4000": "P_cop = 4000",
            "pcop50000": "P_cop = full context",
        }
    )
    margin: float = 1.02
    n_boot: int = 10_000
    seed: int = 0


# ---------------------------------------------------------------- statistics


def bootstrap_median(x: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """(median, 2.5%, 97.5%) of the median of x, resampling its entries (datasets)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, x.size, size=(n_boot, x.size))
    med = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))


def _wilcoxon_greater(x: np.ndarray) -> float:
    from scipy.stats import wilcoxon

    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x) & (x != 0)]
    return float(wilcoxon(x, alternative="greater").pvalue) if x.size >= 2 else float("nan")


def _summ(x: np.ndarray, spec: BeyondArenaReportSpec) -> dict[str, float]:
    med, lo, hi = bootstrap_median(x, spec.n_boot, spec.seed)
    xs = np.asarray(x, dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    return {
        "n": int(xs.size),
        "median": med,
        "ci_lo": lo,
        "ci_hi": hi,
        "wins": int((xs > 0).sum()),
        "p_greater": _wilcoxon_greater(xs),
    }


# ---------------------------------------------------------------- joint track


def load_joint(root: str) -> pd.DataFrame:
    """Every split table under root (<model>/<dataset>/<split>.csv.gz), with the per-split JSON summaries joined."""
    frames = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(root, "*", "*", "*.csv.gz")))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    rows = []
    for p in glob.glob(os.path.join(root, "*", "*.json")):
        with open(p) as fh:
            blob = json.load(fh)
        for key, s in blob["splits"].items():
            rows.append({"model": blob["spec"]["method"], "dataset": blob["meta"]["dataset"], "split": key} | s)
    if rows:
        df = df.merge(pd.DataFrame(rows).drop(columns=["regime"]), on=["model", "dataset", "split"], how="left")
    return df


def dataset_table(df: pd.DataFrame, B: int = 0) -> pd.DataFrame:
    """Mean over splits and blocks of every metric, per model, method, dataset (block size B; 0 = primary)."""
    sub = df[df.B == B]
    extra = [c for c in ("icc_test_z", "icc_ctx_z", "rho_exch", "seconds_copula_per_row") if c in sub]
    return sub.groupby(["model", "method", "regime", "dataset"], as_index=False)[[*METRICS, *extra]].mean()


def gains(table: pd.DataFrame, method: str = "copula") -> pd.DataFrame:
    """Per model and dataset: relative gains of `method` over indep (positive = better) on every joint metric."""
    a = table[table.method == method].set_index(["model", "regime", "dataset"])
    b = table[table.method == "indep"].set_index(["model", "regime", "dataset"])
    a, b = a.align(b, join="inner")
    out = pd.DataFrame(index=a.index)
    out["g_es"] = 1.0 - a.es / b.es
    out["g_vs"] = 1.0 - a.vs / b.vs
    out["g_mean_crps"] = 1.0 - a.mean_crps / b.mean_crps
    out["d_nll"] = b.joint_nll - a.joint_nll  # nats per point, positive = better
    out["d_cov_err"] = (b.mean_cov90 - 0.9).abs() - (a.mean_cov90 - 0.9).abs()
    out["crps_gap"] = a.row_crps_sample - b.row_crps_sample  # must be ~0 (same marginal)
    out["rho_pred"] = a.rho_pred
    if "icc_test_z" in a:
        out["icc_test_z"] = a.icc_test_z
    return out.reset_index()


def joint_hypotheses(g: pd.DataFrame, spec: BeyondArenaReportSpec) -> dict[str, Any]:
    """H2 and H3 decisions from the per-dataset gains (dev datasets excluded)."""
    from eval.runners.ctr23_full_eval import holm

    g = g[~g.dataset.isin(spec.dev_datasets)]
    res: dict[str, Any] = {}
    for regime in ("grouped", "temporal"):
        per_model = {}
        for model, sub in g[g.regime == regime].groupby("model"):
            per_model[model] = {
                k: _summ(sub[k].to_numpy(), spec) for k in ("g_es", "g_vs", "d_nll", "g_mean_crps", "d_cov_err")
            }
        sec = [m for m in per_model if m != PRIMARY]
        if sec:
            adj = holm(np.array([per_model[m]["g_es"]["p_greater"] for m in sec]))
            for m, p in zip(sec, adj):
                per_model[m]["g_es"]["p_holm"] = float(p)
        prim = per_model.get(PRIMARY, {}).get("g_es")
        verdict = "not run"
        if prim:
            ok = prim["median"] > 0 and prim["ci_lo"] > 0 and prim["p_greater"] < 0.05
            verdict = "supported" if ok else ("refuted" if prim["ci_hi"] < 0 else "not supported")
        res[f"H2_{regime}"] = {"verdict": verdict, "models": per_model}
    iid = g[(g.regime == "iid") & (g.model == PRIMARY)]
    s = _summ(iid.g_es.to_numpy(), spec)
    rho = float(np.median(iid.rho_pred.abs())) if len(iid) else float("nan")
    ok = s["ci_lo"] > -0.01 and s["ci_hi"] < 0.01 and rho < 0.05
    res["H3"] = {
        "verdict": "supported" if ok else "not supported",
        "g_es": s,
        "median_abs_rho_pred": rho,
        "d_nll": _summ(iid.d_nll.to_numpy(), spec),
    }
    res["marginal_check_max_abs_crps_gap"] = float(g.crps_gap.abs().max()) if len(g) else float("nan")
    return res


def ablations(root: str, spec: BeyondArenaReportSpec) -> pd.DataFrame:
    """Median ES gain (with CI) of the primary model for every ablation tag, on grouped + temporal datasets."""
    rows = []
    base: pd.DataFrame | None = None
    for tag, label in spec.ablation_tags.items():
        df = load_joint(os.path.join(root, "joint", tag))
        if df.empty:
            continue
        g = gains(dataset_table(df))
        g = g[(g.model == PRIMARY) & g.regime.isin(["grouped", "temporal"]) & ~g.dataset.isin(spec.dev_datasets)]
        if tag == "main":
            base = g
        for regime, sub in g.groupby("regime"):
            r = {"tag": tag, "label": label, "regime": regime} | _summ(sub.g_es.to_numpy(), spec)
            if base is not None and tag != "main":  # paired difference to the default run (H4 for xfit-random)
                m = sub.merge(base[base.regime == regime], on="dataset", suffixes=("", "_main"))
                diff = m.g_es_main - m.g_es
                r |= {f"diff_{k}": v for k, v in _summ(diff.to_numpy(), spec).items()}
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- official track


OFFICIAL = {
    "ours": "[New] TA-TABICLv2-ERA5FT (default)",
    "repro": "[New] TA-TABICLv2 (default)",
    "TabICLv2": "TA-TABICLv2 (default)",
    "TabPFN-3.5-Fast": "TA-TABPFN-3.5-FAST (default)",
    "Kumo-Small": "TA-KUMO-TABULAR-SMALL (default)",
}


def official(root: str, meta_dir: str, spec: BeyondArenaReportSpec) -> dict[str, Any]:
    """H1: log RMSE ratio of ours over TabICLv2 per dataset (imputed baselines dropped), Elo per scope."""
    out: dict[str, Any] = {}
    per = os.path.join(root, "compare", "all", "results_per_split.csv")
    if not os.path.isfile(per):
        return {"H1": {"verdict": "not run"}}
    r = pd.read_csv(per)
    r = r[r.method.isin(OFFICIAL.values())]
    # compare() imputes the splits a new method has not run: drop those rows of our runs (a cached
    # baseline's imputed rows stay flagged and are dropped per comparison below).
    imputed = r.imputed.astype(str).str.lower().isin(["true", "1", "1.0"])
    r = r[~(r.method.str.startswith("[New]") & imputed)]
    regime = {}
    for p in glob.glob(os.path.join(meta_dir, "*", "meta.json")):
        with open(p) as fh:
            m = json.load(fh)
        regime[m["dataset"]] = {"random": "iid"}.get(m["task_type"], m["task_type"])
    r["regime"] = r.dataset.map(regime)
    ds = r.groupby(["method", "dataset", "regime"], as_index=False).agg(
        rmse=("metric_error", "mean"), imputed=("imputed", "max")
    )
    wide = ds.pivot_table(index=["dataset", "regime"], columns="method", values="rmse")
    imp = ds.pivot_table(index=["dataset", "regime"], columns="method", values="imputed")
    res: dict[str, Any] = {}
    for name in ("TabICLv2", "TabPFN-3.5-Fast", "Kumo-Small"):
        col = OFFICIAL[name]
        if col not in wide or OFFICIAL["ours"] not in wide:
            continue
        ok = imp[col].fillna(1).eq(0)  # imputed cached baseline -> dropped
        lr = np.log(wide[OFFICIAL["ours"]] / wide[col])[ok].dropna()
        lr = lr[~lr.index.get_level_values("dataset").isin(spec.dev_datasets)]
        res[name] = {
            "all": _summ(-lr.to_numpy(), spec)
            | {"median_log_ratio": float(np.median(lr)), "n_imputed_dropped": int((~ok).sum())}
        }
        for rg, sub in lr.groupby(level="regime"):
            res[name][rg] = {
                "median_log_ratio": float(np.median(sub)),
                "n": int(sub.size),
                "ci": bootstrap_median(sub.to_numpy(), spec.n_boot, spec.seed),
            }
    out["log_rmse_ratio_vs"] = res
    if "TabICLv2" in res:
        med, lo, hi = bootstrap_median(
            np.log(wide[OFFICIAL["ours"]] / wide[OFFICIAL["TabICLv2"]])[imp[OFFICIAL["TabICLv2"]].fillna(1).eq(0)]
            .dropna()
            .to_numpy(),
            spec.n_boot,
            spec.seed,
        )
        verdict = "supported" if hi < np.log(spec.margin) else ("refuted" if lo > 0 else "inconclusive")
        out["H1"] = {
            "verdict": verdict,
            "median_log_ratio": med,
            "ci": [lo, hi],
            "margin_log": float(np.log(spec.margin)),
        }
    if OFFICIAL["repro"] in wide:  # control: our rerun of TabICLv2 against its cache, lite split
        lr = np.log(wide[OFFICIAL["repro"]] / wide[OFFICIAL["TabICLv2"]]).dropna()
        out["repro_control"] = {
            "n": int(lr.size),
            "median_log_ratio": float(np.median(lr)),
            "max_abs_log_ratio": float(lr.abs().max()),
        }
    elo = {}
    for scope in ("all", "iid", "grouped", "temporal"):
        p = os.path.join(root, "compare", scope, "leaderboard.csv")
        if os.path.isfile(p):
            lb = pd.read_csv(p)
            keep = lb[lb.method.isin(OFFICIAL.values())]
            elo[scope] = keep[[c for c in ("method", "elo", "elo+", "elo-", "rank") if c in keep]].to_dict("records")
    out["elo"] = elo
    wide.reset_index().to_csv(os.path.join(spec.out_dir or ".", "official_rmse_per_dataset.csv"), index=False)
    return out


# ---------------------------------------------------------------- figures


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e5e0",
            "grid.linewidth": 0.6,
            "axes.edgecolor": "#52514e",
            "lines.linewidth": 1.5,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def _save(fig: Any, out: str, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))


def figures(df: pd.DataFrame, g: pd.DataFrame, abl: pd.DataFrame, out: str, spec: BeyondArenaReportSpec) -> None:
    import matplotlib.pyplot as plt

    _style()
    prim = df[(df.model == PRIMARY) & ~df.dataset.isin(spec.dev_datasets)]
    regimes = [r for r in REGIME_ORDER if r in set(prim.regime)]
    if not regimes:  # only development datasets so far
        return

    # 1. Coverage of the block-mean 90% interval against block size.
    fig, axes = plt.subplots(1, len(regimes), figsize=(2.3 * len(regimes), 2.0), sharey=True, squeeze=False)
    for ax, rg in zip(axes[0], regimes):
        sub = prim[(prim.regime == rg) & (prim.B > 0)]
        per = sub.groupby(["method", "B", "dataset"]).mean_cov90.mean().groupby(["method", "B"]).agg(["mean", "sem"])
        for m, (c, mk) in METHOD_STYLE.items():
            if m not in per.index.get_level_values(0):
                continue
            s = per.loc[m]
            ax.errorbar(s.index, s["mean"], yerr=s["sem"], color=c, marker=mk, ms=4, capsize=0, label=m)
        ax.axhline(0.9, color="#52514e", ls="--", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_title(rg)
        ax.set_xlabel("block size $B$")
    axes[0][0].set_ylabel("coverage of 90% block-mean interval")
    axes[0][-1].legend(frameon=False, loc="lower left")
    _save(fig, out, "fig_coverage_vs_block_size")
    plt.close(fig)

    # 2. Energy-score gain per dataset, by split type (primary model; copula and exch over indep).
    gp = g[(g.model == PRIMARY)].copy()
    ge = gains(dataset_table(df), "exch")
    ge = ge[ge.model == PRIMARY][["dataset", "g_es"]].rename(columns={"g_es": "g_es_exch"})
    gp = gp.merge(ge, on="dataset", how="left")
    gp["order"] = gp.regime.map({r: i for i, r in enumerate(REGIME_ORDER)})
    gp = gp.sort_values(["order", "g_es"])
    fig, ax = plt.subplots(figsize=(3.4, 0.14 * len(gp) + 0.8))
    y = np.arange(len(gp))
    ax.scatter(100 * gp.g_es, y, color=METHOD_STYLE["copula"][0], marker="o", s=14, label="copula", zorder=3)
    ax.scatter(100 * gp.g_es_exch, y, color=METHOD_STYLE["exch"][0], marker="^", s=14, label="exch", zorder=2)
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_yticks(
        y, [f"{d.split('-')[0][:28]}" + (" (dev)" if d in spec.dev_datasets else "") for d in gp.dataset], fontsize=5
    )
    for rg in regimes:  # regime bands
        ys = y[gp.regime.to_numpy() == rg]
        if ys.size:
            ax.text(1.01, ys.mean(), rg, transform=ax.get_yaxis_transform(), va="center", fontsize=7, color="#52514e")
    ax.set_xlabel("energy-score gain over indep. sampling (%)")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, out, "fig_es_gain_per_dataset")
    plt.close(fig)

    # 3. Predicted against empirical within-block correlation (per dataset).
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    for rg, mk in zip(REGIME_ORDER, ("o", "s", "^")):
        sub = gp[gp.regime == rg]
        ax.scatter(
            sub.icc_test_z,
            sub.rho_pred,
            marker=mk,
            s=16,
            color=METHOD_STYLE["copula"][0],
            facecolor="none" if rg == "iid" else METHOD_STYLE["copula"][0],
            label=rg,
        )
    lim = [
        min(-0.05, float(np.nanmin(gp[["icc_test_z", "rho_pred"]].to_numpy()))),
        max(0.5, float(np.nanmax(gp[["icc_test_z", "rho_pred"]].to_numpy()))),
    ]
    ax.plot(lim, lim, color="#52514e", lw=0.8, ls="--")
    ax.set_xlabel("empirical within-block corr. (ICC of test $z$)")
    ax.set_ylabel("predicted mean corr. (copula)")
    ax.legend(frameon=False)
    _save(fig, out, "fig_pred_vs_empirical_corr")
    plt.close(fig)

    # 4. Ablations: median ES gain with its CI, per regime.
    if not abl.empty:
        labels = list(dict.fromkeys(abl.label))
        fig, ax = plt.subplots(figsize=(3.4, 0.22 * len(labels) + 0.8))
        for k, (rg, mk) in enumerate(zip(("grouped", "temporal"), ("o", "s"))):
            sub = abl[abl.regime == rg].set_index("label").reindex(labels)
            yy = np.arange(len(labels)) + (k - 0.5) * 0.25
            ax.errorbar(
                100 * sub["median"],
                yy,
                xerr=[100 * (sub["median"] - sub.ci_lo), 100 * (sub.ci_hi - sub["median"])],
                fmt=mk,
                color=METHOD_STYLE["copula"][0],
                mfc="white" if k else METHOD_STYLE["copula"][0],
                ms=4,
                capsize=0,
                label=rg,
            )
        ax.axvline(0, color="#52514e", lw=0.8)
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_xlabel("median ES gain over indep. (%), 95% CI")
        ax.legend(frameon=False)
        _save(fig, out, "fig_ablations")
        plt.close(fig)

    # 5. Runtime against block size: copula forward (per-row cost x B) + sampling/scoring.
    sub = prim[prim.B > 0]
    fig, ax = plt.subplots(figsize=(2.6, 2.2))
    for m, (c, mk) in METHOD_STYLE.items():
        s = sub[sub.method == m].groupby("B").seconds_sample.median()
        if s.empty:
            continue
        if m == "copula" and "seconds_copula_per_row" in sub:
            s = s + sub[sub.method == m].groupby("B").seconds_copula_per_row.median() * s.index.to_numpy()
        ax.plot(
            s.index,
            s.to_numpy(),
            color=c,
            marker=mk,
            ms=4,
            label=m + (" (fwd + sample)" if m == "copula" else " (sample)"),
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("block size $B$")
    ax.set_ylabel("seconds per block (median)")
    ax.legend(frameon=False)
    _save(fig, out, "fig_runtime_vs_block_size")
    plt.close(fig)


# ---------------------------------------------------------------- entry


def run(spec: BeyondArenaReportSpec) -> None:
    from copula_inter.paths import BEYONDARENA_DATA_DIR, BEYONDARENA_RESULTS_DIR

    root = spec.results_dir or str(BEYONDARENA_RESULTS_DIR)
    spec.out_dir = out = spec.out_dir or os.path.join(root, "report")
    os.makedirs(out, exist_ok=True)
    result: dict[str, Any] = {}
    df = load_joint(os.path.join(root, "joint", "main"))
    if not df.empty:
        table = dataset_table(df)
        table.to_csv(os.path.join(out, "joint_per_dataset.csv"), index=False)
        dataset_table(df[df.B > 0].assign(B=0)).to_csv(os.path.join(out, "joint_sweep_pooled.csv"), index=False)
        df[df.B > 0].groupby(["model", "method", "regime", "dataset", "B"], as_index=False)[
            list(METRICS)
        ].mean().to_csv(os.path.join(out, "joint_by_block_size.csv"), index=False)
        g = gains(table)
        g.to_csv(os.path.join(out, "joint_gains_per_dataset.csv"), index=False)
        result |= joint_hypotheses(g, spec)
        abl = ablations(root, spec)
        abl.to_csv(os.path.join(out, "ablations.csv"), index=False)
        h4 = abl[abl.tag == "xfit-random"] if not abl.empty else abl
        if not h4.empty:
            result["H4"] = {
                rg: {k: r[k] for k in r.index if k.startswith("diff_")}
                | {
                    "verdict": "supported"
                    if r["diff_median"] > 0 and r["diff_ci_lo"] > 0 and r["diff_p_greater"] < 0.05
                    else "not supported"
                }
                for rg, r in h4.set_index("regime").iterrows()
            }
        figures(df, g, abl, out, spec)
    result["official"] = official(os.path.join(root, "official"), str(BEYONDARENA_DATA_DIR), spec)
    with open(os.path.join(out, "hypotheses.json"), "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    print(
        json.dumps(
            {k: (v.get("verdict") if isinstance(v, dict) else v) for k, v in result.items()}, indent=1, default=str
        )
    )


main = hydra_entry("beyondarena_report", BeyondArenaReportSpec, run)


if __name__ == "__main__":
    main()
