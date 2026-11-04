"""Paired per-episode NLL differences with cluster-bootstrap CIs, from eval_checkpoint output.dump_episodes files.

Usage:
    # Run B minus run A, per method and NLL term (e.g. F1b vs F0 on the same episodes):
    python -m eval.runners.paired_compare a=<dump_f0.json> b=<dump_f1b.json>
    # Within one run: the copula row minus independence, autoregressive and best_baseline:
    python -m eval.runners.paired_compare a=<dump.json>

Episodes are matched on ep_i; their episode meta (rows, split, group) must be
identical, else the runs did not see the same episodes and the comparison
raises. Every delta is per point (the total-NLL table's units); negative means
B (or the copula) is better. The CI resamples the episode meta's "cluster"
(cluster=meta, the dataset's CI unit) or single episodes (cluster=episode).
win_rate is the fraction of episodes with delta < 0. folds=[...] restricts
both runs to episodes whose meta fold is listed (cross-fitting reads the other
folds only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eval.metrics.cluster_bootstrap import cluster_bootstrap_ci
from eval.runners.hydra_cli import check_choice, hydra_entry

__all__ = ["PairedCompareSpec", "paired_deltas", "within_run_deltas"]

TERMS = ("total", "marginal", "copula")
# The within-run contrasts: (label, copula-model key, reference key).
WITHIN_RUN = (
    ("copula - independence", "icl", "independence_marginal"),
    ("copula - autoregressive", "icl", "autoregressive"),
    ("copula - best_baseline", "icl", "best_baseline"),
)


@dataclass
class PairedCompareSpec:
    """Paired NLL deltas between two dumps (B - A), or the copula's contrasts within one dump (A)."""

    a: str = ""
    b: str | None = None
    # "meta": resample the episode meta's cluster (the dataset's CI unit); "episode": single episodes.
    cluster: str = "meta"
    n_boot: int = 2000
    seed: int = 0
    level: float = 0.95
    # Keep only episodes of these folds (meta "fold"); empty keeps all.
    folds: list[int] = field(default_factory=list)
    # Write the result table as JSON here.
    out_json: str | None = None


def _load(path: str) -> dict[int, dict[str, Any]]:
    with open(path) as fh:
        blob = json.load(fh)
    return {int(e["ep_i"]): e for e in blob["episodes"]}


def _keep(ep: dict[str, Any], folds: list[int]) -> bool:
    return not folds or int(ep.get("episode", {}).get("fold", -1)) in folds


def _cluster(ep: dict[str, Any], mode: str) -> str:
    if mode == "episode":
        return f"ep{ep['ep_i']}"
    meta = ep.get("episode")
    if meta is None or "cluster" not in meta:
        raise ValueError(f"episode {ep['ep_i']} has no meta cluster; use cluster=episode")
    return str(meta["cluster"])


def _term(ep: dict[str, Any], key: str, term: str) -> float:
    if term == "total" and key == "best_baseline":
        # best_baseline has no own marginal: its row is the shared-z copula NLL.
        return float("nan")
    return float((ep["total_nlls"].get(key) or {}).get(term, float("nan")))


def _summary(deltas: list[float], clusters: list[str], spec: PairedCompareSpec) -> dict[str, float]:
    d = np.asarray(deltas, dtype=np.float64)
    ok = np.isfinite(d)
    if not ok.any():
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "win_rate": float("nan")}
    cl = [c for c, k in zip(clusters, ok) if k]
    mean, lo, hi = cluster_bootstrap_ci(d[ok], cl, n_boot=spec.n_boot, seed=spec.seed, level=spec.level)
    return {
        "n": int(ok.sum()),
        "n_clusters": len(set(cl)),
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "win_rate": float((d[ok] < 0).mean()),
    }


def paired_deltas(spec: PairedCompareSpec) -> dict[str, dict[str, dict[str, float]]]:
    """{method: {term: summary}} of B - A over the episodes both dumps share; raises on mismatched meta."""
    assert spec.b is not None
    a, b = _load(spec.a), _load(spec.b)
    shared = sorted(set(a) & set(b))
    if set(a) != set(b):
        print(f"  note: {len(set(a) ^ set(b))} episodes appear in one dump only and are dropped")
    for i in shared:
        if a[i].get("episode") != b[i].get("episode"):
            raise ValueError(f"episode {i}: the two dumps hold different episodes (meta differs)")
    shared = [i for i in shared if _keep(a[i], spec.folds)]
    methods = sorted(set().union(*(a[i]["total_nlls"].keys() for i in shared))) if shared else []
    clusters = [_cluster(a[i], spec.cluster) for i in shared]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for m in methods:
        out[m] = {t: _summary([_term(b[i], m, t) - _term(a[i], m, t) for i in shared], clusters, spec) for t in TERMS}
    # best_baseline is scored on the shared z only: compare its copula NLL from the z-space rows.
    out["best_baseline"] = {
        "copula": _summary(
            [
                float(b[i]["nlls"].get("best_baseline", np.nan)) - float(a[i]["nlls"].get("best_baseline", np.nan))
                for i in shared
            ],
            clusters,
            spec,
        )
    }
    return out


def within_run_deltas(spec: PairedCompareSpec) -> dict[str, dict[str, dict[str, float]]]:
    """{contrast: {term: summary}} of the copula row minus each reference within dump A."""
    a = _load(spec.a)
    eps = [a[i] for i in sorted(a) if _keep(a[i], spec.folds)]
    clusters = [_cluster(e, spec.cluster) for e in eps]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for label, ours, ref in WITHIN_RUN:
        if ref == "best_baseline":
            # best_baseline is a z-space selection: compare copula NLLs against the shared z_test.
            vals = [float(e["nlls"].get(ours, np.nan)) - float(e["nlls"].get(ref, np.nan)) for e in eps]
            out[label] = {"copula (shared z)": _summary(vals, clusters, spec)}
            continue
        out[label] = {t: _summary([_term(e, ours, t) - _term(e, ref, t) for e in eps], clusters, spec) for t in TERMS}
    return out


def _print(results: dict[str, dict[str, dict[str, float]]], title: str, level: float) -> None:
    print(f"\n{title}  (per-point nats; negative = better; {level:.0%} cluster-bootstrap CI)")
    print(f"{'contrast / method':<44}{'term':<20}{'n':>6}{'clusters':>9}{'mean Δ':>10}{'CI':>22}{'win':>7}")
    for name, terms in results.items():
        for term, s in terms.items():
            if not s.get("n"):
                continue
            ci = f"[{s['lo']:+.4f}, {s['hi']:+.4f}]"
            print(
                f"{name:<44}{term:<20}{s['n']:>6}{s.get('n_clusters', 0):>9}"
                f"{s['mean']:>+10.4f}{ci:>22}{s['win_rate']:>7.2f}"
            )


def run(spec: PairedCompareSpec) -> None:
    """Print (and optionally save) the paired deltas."""
    check_choice("cluster", spec.cluster, ("meta", "episode"))
    if not spec.a:
        raise ValueError("a=<dump.json> is required")
    if spec.b is None:
        results = within_run_deltas(spec)
        _print(results, f"Within-run contrasts of {spec.a}", spec.level)
    else:
        results = paired_deltas(spec)
        _print(results, f"B - A: {spec.b}  minus  {spec.a}", spec.level)
    if spec.out_json:
        with open(spec.out_json, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"Wrote {spec.out_json}")


main = hydra_entry("paired_compare", PairedCompareSpec, run)


if __name__ == "__main__":
    main()
