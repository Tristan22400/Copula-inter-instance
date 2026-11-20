"""Compare the mixed-model baseline before (run main) and after the fix (run lmm_v2), per grouped dataset."""

import pandas as pd

R = "results/beyondarena/joint_ta"
cols = ["rmse", "joint_nll", "energy", "agg_group_cov90", "rho_pred", "icc_test_z"]
out = []
for run in ("main", "lmm_v2"):
    df = pd.read_csv(f"{R}/{run}/joint_scores.csv")
    df = df[(df.marginal == "lmm") & (df.variant == "lmm")].copy()
    df["ds"] = df.dataset.str.rsplit("-", n=1).str[0]
    g = df.groupby("ds")
    t = g[cols].median()
    if "info_fit" in df:
        t["reml_share"] = g["info_fit"].apply(lambda s: (s == "reml").mean())
    for k in ("info_n_axes", "info_s_u2", "info_s_e2", "info_var_resid_ols", "info_reml_mean_shift_sd"):
        if k in df:
            t[k[5:]] = g[k].median()
    t["run"] = run
    out.append(t.reset_index())
print(pd.concat(out).sort_values(["ds", "run"]).round(4).to_string(index=False))
ref = pd.read_csv(f"{R}/main/joint_scores.csv")
ref = ref[(ref.marginal == "tabicl-ft") & (ref.variant == "indep")].copy()
ref["ds"] = ref.dataset.str.rsplit("-", n=1).str[0]
print("\ntabicl-ft indep, same datasets (reference):")
print(ref[ref.ds.isin(out[1].ds)].groupby("ds")[cols].median().round(4).to_string())
