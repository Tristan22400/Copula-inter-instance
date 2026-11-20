"""BeyondArena official track: leaderboards (Elo) of our runs against the cached baselines (tabarena venv).

Usage:
    python official_compare.py                    # every method dir under results/beyondarena/official

Loads every results.pkl that official_track.py cached, registers them next to the cached BeyondArena
baselines, and calls the context's own compare() (its Elo, ranks and per-split errors) on the
core2k regression tasks, overall and per split regime. Writes, under <out_root>/compare/<scope>/,
TabArena's leaderboard CSVs (with results_per_split.csv and its `imputed` flag) and leaderboard.csv.
"""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "out_root": str(REPO / "results/beyondarena/official"),
    # all = every core2k regression task; iid / grouped / temporal = that split regime only.
    "scopes": ["all", "iid", "grouped", "temporal"],
}


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULTS), OmegaConf.from_cli())
    from tabarena.contexts import BeyondArenaContext

    root = Path(cfg.out_root)
    results = []
    for p in sorted(root.glob("*/data/*/*/*/results.pkl")):
        raw = p.read_bytes()
        results.append(pickle.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw))
    print(f"{len(results)} cached results under {root}")
    for scope in cfg.scopes:
        extra = [] if scope == "all" else [str(scope)]
        ctx = BeyondArenaContext()
        ctx.register(results, new_result_prefix="[New] ")
        out = root / "compare" / scope
        lb = ctx.compare(output_dir=out, subset=["core2k", "regression", *extra], plot=False)
        lb.to_csv(out / "leaderboard.csv", index=False)
        cols = [c for c in ("method", "elo", "elo+", "elo-", "rank", "imputed", "n_datasets") if c in lb.columns]
        print(f"\n== {scope}\n{lb[cols].head(45).to_string(index=False)}")


if __name__ == "__main__":
    main()
