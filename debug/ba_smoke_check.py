import glob
import gzip
import pickle

for p in sorted(glob.glob("results/beyondarena/joint_ta/smoke/data/*/*/*/results.pkl")):
    raw = open(p, "rb").read()
    r = pickle.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    jm, info = r.get("joint_metrics"), r.get("joint_info", {})
    print(
        f"\n{r['framework']} | {r['task_metadata']['name'][:30]} | RMSE {r['metric_error']:.4f} | "
        f"P_ctx {info.get('P_ctx')} rank {info.get('rank')} regime {info.get('regime')} "
        f"rho_exch {info.get('rho_exch', float('nan')):.3f} t_joint {info.get('seconds_predict_joint', 0):.1f}s"
    )
    for v, s in (jm or {}).items():
        print(
            f"  {v:7s} energy {s['energy']:.4f} nll {s['joint_nll']:+.4f} grid_crps {s['grid_crps']:.4f} "
            f"sample_crps {s['sample_crps']:.4f} rho_pred {s['rho_pred']:.3f} icc_test {s['icc_test_z']:.3f} "
            f"cov_group {s.get('agg_group_cov90', float('nan')):.2f} cov_fold {s.get('agg_fold_cov90', float('nan')):.0f} "
            f"N {s['n_test']:.0f} M {s['n_samples']:.0f}"
        )
