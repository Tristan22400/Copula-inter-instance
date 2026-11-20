"""Interim analysis of results/beyondarena/joint_ta/main/joint_scores.csv (complete datasets only, dev excluded)."""

import glob
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from eval.runners.beyondarena_report import bootstrap_median

DEV = {"miami_housing", "qsar_tid_11", "wine_quality"}
df = pd.read_csv("results/beyondarena/joint_ta/main/joint_scores.csv")
df["regime"] = df["info_regime"]
n_splits = {}
for p in glob.glob("data/beyondarena/*/meta.json"):
    m = json.load(open(p))
    n_splits[m["dataset"].rsplit("-", 1)[0]] = len(m["splits"])
df["ds"] = df.dataset.str.rsplit("-", n=1).str[0].where(df.dataset.str.contains("-"), df.dataset)
df["ds"] = [d if d in n_splits else d.rsplit("-", 1)[0] for d in df.dataset]
cnt = df[df.variant.isin(["indep", "lmm"])].groupby(["marginal", "ds"]).size()
complete = {k for k, v in cnt.items() if v >= n_splits.get(k[1], 10**9)}
df = df[[(m, d) in complete for m, d in zip(df.marginal, df.ds)] & ~df.ds.isin(DEV)]
cols = [
    "energy",
    "joint_nll",
    "variogram_struct",
    "variogram_uniform",
    "sample_crps",
    "grid_crps",
    "agg_group_cov90",
    "agg_B32_cov90",
    "agg_fold_cov90",
    "agg_group_crps",
    "agg_B32_crps",
    "rho_pred",
    "icc_test_z",
    "rmse",
]
d = df.groupby(["marginal", "variant", "regime", "ds"])[[c for c in cols if c in df]].mean().reset_index()


def gain(a, b, key, rel=True):
    m = a.merge(b, on=["marginal", "regime", "ds"], suffixes=("_a", "_b"))
    return m, (1 - m[f"{key}_a"] / m[f"{key}_b"]) if rel else (m[f"{key}_b"] - m[f"{key}_a"])


print("datasets complete per marginal/regime (dev excluded):")
print(d[d.variant == "indep"].groupby(["marginal", "regime"]).size().unstack().to_string())
for marg in sorted(d.marginal.unique()):
    if marg == "lmm":
        continue
    for rg in ("grouped", "temporal", "iid"):
        ind = d[(d.marginal == marg) & (d.variant == "indep") & (d.regime == rg)]
        if ind.empty:
            continue
        rows = []
        for var in ("copula", "exch"):
            v = d[(d.marginal == marg) & (d.variant == var) & (d.regime == rg)]
            m, g = gain(v, ind, "energy")
            _, nll = gain(v, ind, "joint_nll", rel=False)
            med, lo, hi = bootstrap_median(100 * g.to_numpy())
            x = g.to_numpy()
            p = wilcoxon(x[x != 0], alternative="greater").pvalue if (x != 0).sum() >= 2 else float("nan")
            r = {
                "vs indep": var,
                "n": len(x),
                "ES gain % med [CI]": f"{med:+.3f} [{lo:+.3f},{hi:+.3f}]",
                "ES wins": f"{(x > 0).sum()}/{len(x)}",
                "p (1-sided)": f"{p:.3g}",
                "NLL gain med": f"{np.median(nll):+.4f}",
                "NLL wins": f"{(nll > 0).sum()}/{len(x)}",
            }
            # Variogram scores (order 0.5): uniform random pairs, and structured pairs (same group /
            # time-adjacent; none on iid). Gain % = 100 (1 - VS_var / VS_indep), > 0 = better than indep.
            for tag in ("uniform", "struct"):
                key = f"variogram_{tag}"
                if f"{key}_a" not in m or m[f"{key}_a"].isna().all():
                    continue
                _, gv = gain(v, ind, key)
                xv = gv.dropna().to_numpy()
                if len(xv) == 0:
                    continue
                medv, lov, hiv = bootstrap_median(100 * xv)
                pv = wilcoxon(xv[xv != 0], alternative="greater").pvalue if (xv != 0).sum() >= 2 else float("nan")
                r[f"VS-{tag[:4]} gain % [CI]"] = f"{medv:+.2f} [{lov:+.2f},{hiv:+.2f}]"
                r[f"VS-{tag[:4]} wins, p"] = f"{(xv > 0).sum()}/{len(xv)}, {pv:.2g}"
            agg = "agg_group_cov90" if rg == "grouped" else "agg_B32_cov90"
            r[f"cov90 ({agg[4:-6]}) indep->var"] = f"{np.median(m[f'{agg}_b']):.2f} -> {np.median(m[f'{agg}_a']):.2f}"
            r["rho_pred med"] = f"{np.median(m['rho_pred_a']):.3f}"
            r["icc_test med"] = f"{np.median(m['icc_test_z_a']):.3f}"
            r["max|CRPS gap|"] = f"{(m['sample_crps_a'] - m['sample_crps_b']).abs().max():.1e}"
            rows.append(r)
        print(f"\n### {marg} | {rg}")
        print(pd.DataFrame(rows).to_string(index=False))
lmm = d[d.marginal == "lmm"]
if not lmm.empty:
    print(
        "\n### lmm (grouped):",
        lmm[["ds", "energy", "joint_nll", "agg_group_cov90", "rho_pred", "icc_test_z"]].round(3).to_string(index=False),
    )
prim = d[(d.marginal == "tabicl-ft") & (d.regime == "grouped")].pivot_table(
    index="ds", columns="variant", values=["energy", "joint_nll", "agg_group_cov90", "rho_pred"]
)
print("\n### tabicl-ft grouped, per dataset\n", prim.round(3).to_string())
