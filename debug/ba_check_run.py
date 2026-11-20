import collections
import glob
import gzip
import pickle
import sys

method = sys.argv[1]
paths = sorted(glob.glob(f"results/beyondarena/joint_ta/main/data/{method}/*/*/results.pkl"))
c: collections.Counter[str] = collections.Counter()
for p in paths:
    raw = open(p, "rb").read()
    r = pickle.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    jm = r.get("joint_metrics")
    c["with_joint" if jm else "without_joint"] += 1
    c[r["task_metadata"]["name"][:28]] += 1
print(len(paths), "results;", dict(c))
for p in paths[:: max(1, len(paths) // 4)]:
    raw = open(p, "rb").read()
    r = pickle.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    info = r.get("joint_info", {})
    print(
        f"\n{r['task_metadata']['name'][:28]} f{r['task_metadata']['fold']} r{r['task_metadata']['repeat']} RMSE {r['metric_error']:.4g} "
        f"P_ctx {info.get('P_ctx')} rank {info.get('rank')} t_joint {info.get('seconds_predict_joint', 0):.1f}s rho_exch {info.get('rho_exch', 0):.3f}"
    )
    for v, s in (r.get("joint_metrics") or {}).items():
        print(
            f"  {v:7s} energy {s['energy']:.4f} nll {s['joint_nll']:+.4f} grid {s['grid_crps']:.4f} sample {s['sample_crps']:.4f} rho_pred {s['rho_pred']:.3f} fold_cov {s.get('agg_fold_cov90', float('nan')):.0f} B32_cov {s.get('agg_B32_cov90', float('nan')):.2f}"
        )
