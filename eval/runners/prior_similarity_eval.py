#!/usr/bin/env python
"""prior_similarity_eval.py — how far is the synthetic training prior from the
real ERA5 episode distribution, on indicators computed identically for both?

    # the Phase-0 baseline table: real ERA5 vs. today's src/data_gen.py prior
    python eval/runners/prior_similarity_eval.py --n-bundles 40

    # add the reference lattice+Matern probe (eval/prior_similarity/bundles.py)
    python eval/runners/prior_similarity_eval.py --sources era5,synthetic_current,lattice_matern_probe

Outputs (under --out-dir, default eval/reports/prior_similarity/):
  indicators.csv     one row per bundle, every indicator
  summary.md         per-source median [IQR] and the real-vs-synthetic gap
  summary.json       the same, machine-readable
  correlogram.png    mean correlation vs. distance in nearest-neighbour units
  c2st.json          two-sample-classifier AUC + the most discriminative
                     indicators (i.e. what to fix first)

The indicator definitions and the reasoning behind each tier live in
eval/prior_similarity/indicators.py; the episode sources in bundles.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.prior_similarity import bundles as B  # noqa: E402
from eval.prior_similarity.indicators import INDICATOR_TIERS, compute_all  # noqa: E402

DEFAULT_OUT = os.path.join(_REPO_ROOT, "eval", "reports", "prior_similarity")
ERA5_CACHE = os.path.join(_REPO_ROOT, "eval", "data", "cache", "era5_global_train")


def _load_cfg():
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=os.path.join(_REPO_ROOT, "conf"), version_base=None):
        return hydra.compose(config_name="config")


def build_bundles(args) -> list:
    out = []
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    if "era5" in sources:
        pool = B.ERA5Pool(args.era5_cache, n_months=args.era5_months, seed=args.seed)
        rng = np.random.default_rng(args.seed)
        got = 0
        while got < args.n_bundles:
            b = pool.sample_bundle(
                rng,
                grid_size_range=(args.grid_min, args.grid_max),
                box_deg_range=(args.box_min, args.box_max),
                n_realizations=args.n_realizations,
            )
            if b is None:
                continue
            out.append(b)
            got += 1
            print(f"[era5] bundle {got}/{args.n_bundles} D={b.D} gs={b.meta['grid_size']} "
                  f"box={b.meta['box_deg']:.1f}deg lat={b.meta['lat_c']:.1f}")
        del pool

    if "synthetic_current" in sources:
        cfg = _load_cfg()
        # Match the ERA5 side's D range so the comparison is not confounded by
        # episode size; everything else about the prior is left exactly as the
        # committed config has it.
        if args.match_size:
            cfg.data.N_min = args.grid_min ** 2
            cfg.data.N_max = args.grid_max ** 2
        print(f"[synthetic_current] N in [{cfg.data.N_min}, {cfg.data.N_max}], "
              f"P in [{cfg.data.P_min}, {cfg.data.P_max}], d~LogNormal")
        out += B.sample_synthetic_bundles(
            cfg, args.n_bundles, n_realizations=args.n_realizations,
            device=args.device, seed=args.seed,
        )

    if "lattice_matern_probe" in sources:
        out += B.sample_lattice_matern_bundles(
            args.n_bundles, n_realizations=args.n_realizations,
            seed=args.seed, grid_size_range=(args.grid_min, args.grid_max),
        )
    return out


def summarize(rows: list, out_dir: str) -> dict:
    import pandas as pd

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("_curve")} for r in rows])
    df.to_csv(os.path.join(out_dir, "indicators.csv"), index=False)

    sources = list(dict.fromkeys(df["source"]))
    ref = "era5" if "era5" in sources else sources[0]
    summary = {}
    lines = ["# Prior-similarity indicators", "",
             f"Reference source: `{ref}`. Each cell is median [q25, q75] over bundles.",
             f"`gap` = (median_source - median_{ref}) / (IQR_{ref}/1.349), i.e. how many "
             "reference standard deviations the synthetic prior sits away on that indicator.", ""]

    for tier, names in INDICATOR_TIERS.items():
        names = [n for n in names if n in df.columns]
        if not names:
            continue
        lines += [f"## {tier}", "",
                  "| indicator | " + " | ".join(sources) + " | " + " | ".join(f"gap[{s}]" for s in sources if s != ref) + " |",
                  "|---|" + "---|" * (len(sources) + len(sources) - 1)]
        for n in names:
            cells, gaps = [], []
            r_med = np.nanmedian(df.loc[df["source"] == ref, n].values.astype(float))
            r_iqr = np.nanpercentile(df.loc[df["source"] == ref, n].values.astype(float), 75) - \
                np.nanpercentile(df.loc[df["source"] == ref, n].values.astype(float), 25)
            r_sd = max(r_iqr / 1.349, 1e-9)
            for s in sources:
                v = df.loc[df["source"] == s, n].values.astype(float)
                med = np.nanmedian(v)
                q1, q3 = np.nanpercentile(v, 25), np.nanpercentile(v, 75)
                cells.append(f"{med:.3g} [{q1:.3g}, {q3:.3g}]")
                summary.setdefault(s, {})[n] = {"median": float(med), "q25": float(q1), "q75": float(q3)}
                if s != ref:
                    gaps.append(f"**{(med - r_med) / r_sd:+.1f}**")
            lines.append(f"| `{n}` | " + " | ".join(cells) + " | " + " | ".join(gaps) + " |")
        lines.append("")

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n".join(lines))
    return summary


def c2st(rows: list, out_dir: str, ref: str = "era5") -> dict:
    """Two-sample classifier test: can a small model tell a real ERA5 bundle
    from a synthetic one given only its indicator vector? AUC 0.5 means the
    prior is indistinguishable on everything measured here; the permutation
    importances rank what to fix first."""
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    from threadpoolctl import threadpool_limits

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("_curve")} for r in rows])
    feats = [c for c in df.columns if c not in ("source", "D", "d_x", "R_realizations", "med_nn")]
    results = {}
    for s in df["source"].unique():
        if s == ref:
            continue
        sub = df[df["source"].isin([ref, s])]
        X = sub[feats].values.astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = (sub["source"] == s).values.astype(int)
        if len(np.unique(y)) < 2 or len(y) < 12:
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3, random_state=0)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        # A few dozen rows by a few dozen columns is far too small to
        # parallelize: left at the machine default this stage spun 48 OpenMP
        # threads at ~3700% CPU for many minutes on an 80-row problem, dwarfing
        # the indicator computation it exists to summarize.
        with threadpool_limits(limits=2):
            p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
            auc = float(roc_auc_score(y, p))
            clf.fit(X, y)
            imp = permutation_importance(clf, X, y, n_repeats=15, random_state=0)
        order = np.argsort(imp.importances_mean)[::-1][:10]
        results[s] = {
            "auc": auc,
            "top_discriminative": [{"indicator": feats[i], "importance": float(imp.importances_mean[i])}
                                   for i in order if imp.importances_mean[i] > 0],
        }
        print(f"\n[C2ST] {ref} vs {s}: AUC = {auc:.3f}  (0.5 = indistinguishable)")
        for e in results[s]["top_discriminative"][:6]:
            print(f"    {e['indicator']:<28} {e['importance']:.3f}")
    with open(os.path.join(out_dir, "c2st.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_correlograms(rows: list, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in ((axes[0], "_curve", "raw field"), (axes[1], "det__curve", "plane-detrended field")):
        by_src = {}
        for r in rows:
            if key not in r:
                continue
            by_src.setdefault(r["source"], []).append(r[key])
        for src, curves in by_src.items():
            grid = np.linspace(0, 12, 60)
            stack = []
            for c, rho, _ in curves:
                ok = np.isfinite(rho)
                if ok.sum() >= 3:
                    stack.append(np.interp(grid, c[ok], rho[ok], left=np.nan, right=np.nan))
            if not stack:
                continue
            S = np.asarray(stack)
            med = np.nanmedian(S, axis=0)
            lo, hi = np.nanpercentile(S, 25, axis=0), np.nanpercentile(S, 75, axis=0)
            ax.plot(grid, med, label=src, lw=2)
            ax.fill_between(grid, lo, hi, alpha=0.18)
        ax.axhline(0.5, color="k", lw=0.6, ls=":")
        ax.set_xlabel("distance / median nearest-neighbour spacing")
        ax.set_ylabel("correlation")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
    fig.suptitle("Prior correlation vs. distance, in units of the design's own sample spacing")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "correlogram.png"), dpi=140)
    print(f"\nwrote {os.path.join(out_dir, 'correlogram.png')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="era5,synthetic_current")
    ap.add_argument("--n-bundles", type=int, default=40, help="bundles per source")
    ap.add_argument("--n-realizations", type=int, default=1200,
                    help="days (ERA5) / GP draws (synthetic) per bundle; must exceed D for a "
                         "usable empirical covariance")
    ap.add_argument("--era5-months", type=int, default=40, help="whole months read into the ERA5 pool")
    ap.add_argument("--era5-cache", default=ERA5_CACHE)
    ap.add_argument("--grid-min", type=int, default=8)
    ap.add_argument("--grid-max", type=int, default=24)
    ap.add_argument("--box-min", type=float, default=5.0)
    ap.add_argument("--box-max", type=float, default=25.0)
    ap.add_argument("--match-size", action="store_true", default=True,
                    help="force the synthetic prior's N range to match the ERA5 grid range so the "
                         "comparison is not confounded by episode size")
    ap.add_argument("--no-match-size", dest="match_size", action="store_false")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    bundles = build_bundles(args)
    print(f"\ncomputing indicators for {len(bundles)} bundles ...")
    rows = []
    for i, b in enumerate(bundles):
        rows.append(compute_all(b, seed=args.seed + i))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(bundles)}")

    summarize(rows, args.out_dir)
    plot_correlograms(rows, args.out_dir)
    if len({r["source"] for r in rows}) > 1:
        c2st(rows, args.out_dir)
    print(f"\nreports in {args.out_dir}")


if __name__ == "__main__":
    main()
