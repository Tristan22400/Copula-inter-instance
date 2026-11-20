"""Interim summary of the stopped (capped-block) joint track: complete datasets only, dev excluded."""

import glob
import json

import pandas as pd

from eval.runners.beyondarena_report import BeyondArenaReportSpec, bootstrap_median, dataset_table, gains, load_joint

spec = BeyondArenaReportSpec()
complete = set()
for p in glob.glob("results/beyondarena/joint/main/*/*.json"):
    b = json.load(open(p))
    meta = json.load(open(f"data/beyondarena/{b['meta']['dataset']}/meta.json"))
    if len(b["splits"]) == len(meta["splits"]):
        complete.add((b["spec"]["method"], b["meta"]["dataset"]))
df = load_joint("results/beyondarena/joint/main")
df = df[[(m, d) in complete for m, d in zip(df.model, df.dataset)] & ~df.dataset.isin(spec.dev_datasets)]
t = dataset_table(df)
rows = []
for m in ("copula", "exch"):
    g = gains(t, m)
    for (model, rg), s in g.groupby(["model", "regime"]):
        med, lo, hi = bootstrap_median(100 * s.g_es.to_numpy())
        rows.append(
            {
                "model": model,
                "regime": rg,
                "vs_indep": m,
                "n": len(s),
                "ES gain % [CI]": f"{med:+.2f} [{lo:+.2f},{hi:+.2f}]",
                "ES wins": f"{(s.g_es > 0).sum()}/{len(s)}",
                "NLL gain med": round(float(s.d_nll.median()), 3),
                "NLL wins": f"{(s.d_nll > 0).sum()}/{len(s)}",
                "max |CRPS gap|": f"{s.crps_gap.abs().max():.1e}",
            }
        )
print(pd.DataFrame(rows).sort_values(["regime", "vs_indep", "model"]).to_string(index=False))
cov = t.pivot_table(index=["model", "regime"], columns="method", values="mean_cov90", aggfunc="median").round(2)
print("\nmedian over datasets of block-mean 90% coverage (B = primary blocks):\n", cov.to_string())
gp = gains(t)
print("\npredicted vs empirical correlation, tabicl-ft:")
print(gp[gp.model == "tabicl-ft"].groupby("regime")[["rho_pred", "icc_test_z"]].median().round(3).to_string())
x = load_joint("results/beyondarena/joint/xfit-random")
if not x.empty:
    gx = gains(dataset_table(x))
    gm = gp[gp.model == "tabicl-ft"]
    mm = gx.merge(gm, on=["model", "regime", "dataset"], suffixes=("_rand", "_struct"))
    print("\nH4 interim (struct - random ES gain, %):")
    print(
        (100 * (mm.set_index(["regime", "dataset"]).g_es_struct - mm.set_index(["regime", "dataset"]).g_es_rand))
        .round(3)
        .to_string()
    )
