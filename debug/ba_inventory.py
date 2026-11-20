import collections
import glob
import json
import os

for tag in sorted(os.listdir("results/beyondarena/joint")):
    for model in sorted(os.listdir(f"results/beyondarena/joint/{tag}")):
        c: collections.Counter[str] = collections.Counter()
        full: collections.Counter[str] = collections.Counter()
        for p in glob.glob(f"results/beyondarena/joint/{tag}/{model}/*.json"):
            b = json.load(open(p))
            meta = json.load(open(f"data/beyondarena/{b['meta']['dataset']}/meta.json"))
            rg = {"random": "iid"}.get(meta["task_type"], meta["task_type"])
            c[rg] += 1
            full[rg] += len(b["splits"]) == len(meta["splits"])
        print(f"{tag:12s} {model:11s} " + "  ".join(f"{k}: {full[k]} complete / {c[k]} started" for k in sorted(c)))
