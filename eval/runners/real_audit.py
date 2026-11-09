"""Model-free audit of a real-data benchmark (eval/data/real): cleaning counts, atoms/ties, layout, draw statistics.

Usage:
    python -m eval.runners.real_audit real.dataset=kaust real.kaust.variant=1a_2
    python -m eval.runners.real_audit real.dataset=ctr23 ctr23_all=true    # atom rule over the whole suite

Loads and verifies the dataset exactly as eval_checkpoint does, prints the
dataset-level checks of its module (below), then draws n_episodes episodes
with the smoke seed (43, never a reported seed) and reports the skip rate,
draw time, episode shapes, cluster and fold counts, and within-episode target
ties. Nothing here reads a model or a score. SDWPF has its own audit
(sdwpf_audit: the direction checks need the SCADA angles).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eval.runners.eval_args import RealSpec
from eval.runners.hydra_cli import hydra_entry

__all__ = ["RealAuditSpec", "audit"]


@dataclass
class RealAuditSpec:
    """Which real dataset to audit (real.dataset and its real.<name> sub-config) and how many draws to inspect."""

    real: RealSpec = field(default_factory=RealSpec)
    n_episodes: int = 200
    # The smoke/debug seed: none of these draws is a reported episode.
    seed: int = 43
    # CTR23 only: fetch every suite dataset and apply the atom rule (the exclusion list).
    ctr23_all: bool = False
    out_json: str | None = None


def _q(v: np.ndarray, qs: tuple[float, ...] = (0.0, 0.05, 0.5, 0.95, 1.0)) -> dict[str, float]:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return {f"q{int(q * 100)}": float(np.quantile(v, q)) for q in qs} if v.size else {}


def _tie_frac(y: np.ndarray) -> float:
    from eval.data.real.ctr23 import tie_fraction

    return tie_fraction(np.asarray(y))


def _semivariogram(x: np.ndarray, y: np.ndarray, z: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """Empirical semivariogram 0.5 E[(z_i - z_j)^2] over random pairs, binned by distance (unit-square lags)."""
    n = len(z)
    i, j = rng.integers(n, size=(2, 400_000))
    d = np.hypot(x[i] - x[j], y[i] - y[j])
    g = 0.5 * (z[i] - z[j]) ** 2
    edges = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
    out: dict[str, float] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() >= 50:
            out[f"[{lo},{hi})"] = float(g[m].mean())
    out["var_z"] = float(z.var())
    return out


def _audit_kaust(data: Any, cfg: Any) -> dict[str, Any]:
    from eval.data.real.kaust import EXPECTED_ROWS, N_STRIPS

    cells = cfg.window_cells
    ix = np.minimum((data.x * cells).astype(int), cells - 1)
    iy = np.minimum((data.y * cells).astype(int), cells - 1)
    per_cell = np.bincount(ix * cells + iy, minlength=cells * cells)
    strip = np.minimum((data.x * N_STRIPS).astype(int), N_STRIPS - 1)
    xy = np.round(np.stack([data.x, data.y], 1), 6)
    n_dup_xy = int(len(xy) - len(np.unique(xy, axis=0)))
    return {
        "n_points": data.n,
        "expected_rows": EXPECTED_ROWS[data.variant],
        "paper_rows": 90_000 if data.variant.startswith("1a") else 900_000,
        "n_unique_x": int(len(np.unique(np.round(data.x, 6)))),
        "n_unique_y": int(len(np.unique(np.round(data.y, 6)))),
        "duplicate_locations": n_dup_xy,
        "z": _q(data.z) | {"mean": float(data.z.mean()), "std": float(data.z.std())},
        "z_tie_frac": _tie_frac(data.z),
        "points_per_strip": np.bincount(strip, minlength=N_STRIPS).tolist(),
        "points_per_cell": _q(per_cell),
        "cells_below_P_plus_N": int((per_cell < cfg.n_context + cfg.n_query).sum()),
        "semivariogram": _semivariogram(data.x, data.y, data.z, np.random.default_rng(0)),
    }


def _audit_emnist(data: Any, cfg: Any) -> dict[str, Any]:
    px = data.images.reshape(len(data.images), -1)
    return {
        "n_images": int(len(data.images)),
        "classes": int(len(np.unique(data.labels))),
        "frac_pixels_0": float((px == 0).mean()),
        "frac_pixels_255": float((px == 255).mean()),
        "frac_pixels_interior": float(((px > 0) & (px < 255)).mean()),
        "mean_intensity": float(px.mean() / 255.0),
        "per_image_frac_0": _q((px == 0).mean(axis=1)),
    }


def _audit_california(data: Any, cfg: Any) -> dict[str, Any]:
    return {
        "n_raw": data.n_raw,
        "n_censored_dropped": data.n_censored,
        "cap_value": data.cap_value,
        "n_rows": int(len(data.y)),
        "n_kept_at_5.0": int((data.y == 5.0).sum()),
        "y": _q(data.y),
        "y_tie_frac_after_cleaning": _tie_frac(data.y),
    }


def _audit_meuse(data: Any, cfg: Any, seed: int) -> dict[str, Any]:
    from eval.data.real.meuse import N_POINTS, repetition_parts

    counts = np.zeros(N_POINTS, dtype=int)
    for rep in range(cfg.n_repeats):
        for part in repetition_parts(seed, rep, cfg.n_folds):
            counts[part] += 1
    return {
        "n_points": int(len(data.log_zinc)),
        "log_zinc": _q(data.log_zinc),
        "log_zinc_tie_frac": _tie_frac(data.log_zinc),
        "times_queried_min_max": [int(counts.min()), int(counts.max())],
        "query_sizes": sorted({len(p) for p in repetition_parts(seed, 0, cfg.n_folds)}),
    }


def _audit_ctr23(data: Any, cfg: Any) -> dict[str, Any]:
    return {
        "name": data.name,
        "n_raw": data.n_raw,
        "n_missing_target": data.n_missing_target,
        "n_rows": int(len(data.y)),
        "n_features": int(data.X.shape[1]),
        "tie_frac": data.tie_frac,
        "atom_frac": data.atom_frac,
        "y": _q(data.y),
    }


def _ctr23_suite(spec: RealAuditSpec) -> dict[str, Any]:
    """Every CTR23 dataset: rows, features, largest atom, tie fraction, and whether the atom rule excludes it."""
    import dataclasses
    import urllib.request

    from eval.data.real import ctr23

    try:
        with urllib.request.urlopen(f"https://www.openml.org/api/v1/json/study/{ctr23.SUITE_ID}", timeout=30) as r:
            tasks = sorted(int(t) for t in json.load(r)["study"]["tasks"]["task_id"])
        pinned = sorted(t for t, _ in ctr23.CTR23_TASKS.values())
        suite_check = "matches the API" if tasks == pinned else f"DIFFERS from the API: {tasks}"
    except OSError as exc:
        suite_check = f"not checked (offline: {exc})"
    rows: dict[str, Any] = {}
    for data_id in ctr23.CTR23_TASKS:
        cfg = dataclasses.replace(spec.real.ctr23, data_id=data_id, max_atom_frac=1.0)
        try:
            d = ctr23.load(cfg, verbose=False)
        except ctr23.Ctr23DataError as exc:
            rows[str(data_id)] = {"error": str(exc)}
            continue
        excluded = d.atom_frac > spec.real.ctr23.max_atom_frac
        rows[str(data_id)] = {
            "name": d.name,
            "n_rows": int(len(d.y)),
            "n_features": int(d.X.shape[1]),
            "atom_frac": round(d.atom_frac, 4),
            "tie_frac": round(d.tie_frac, 4),
            "excluded": bool(excluded),
        }
        print(
            f"  {data_id:>6} {d.name:<40} n={len(d.y):>7} d={d.X.shape[1]:>4} atom={d.atom_frac:6.3f} ties={d.tie_frac:6.3f}"
            + ("  EXCLUDED" if excluded else "")
        )
    kept = [k for k, v in rows.items() if v.get("excluded") is False]
    return {"suite_check": suite_check, "datasets": rows, "n_kept": len(kept), "kept_data_ids": kept}


def audit(spec: RealAuditSpec) -> dict[str, Any]:
    """The audit result as a JSON-able dict (also printed)."""
    from eval.data.real import dataset_module
    from eval.runners.eval_args import real_dataset_spec_of

    name = spec.real.dataset
    assert name is not None
    if name == "ctr23" and spec.ctr23_all:
        return {"dataset": name, "suite": _ctr23_suite(spec)}
    mod = dataset_module(name)
    cfg = real_dataset_spec_of(spec.real)
    mod.validate(cfg)
    t0 = time.perf_counter()
    data = mod.load(cfg)
    result: dict[str, Any] = {"dataset": name, "config": cfg.__dict__, "load_s": round(time.perf_counter() - t0, 2)}
    level = {
        "kaust": _audit_kaust,
        "emnist": _audit_emnist,
        "california": _audit_california,
        "ctr23": _audit_ctr23,
    }
    if name == "meuse":
        result["dataset_checks"] = _audit_meuse(data, cfg, spec.seed)
    elif name in level:
        result["dataset_checks"] = level[name](data, cfg)

    shapes: list[tuple[int, int, int]] = []
    clusters: set[str] = set()
    folds: list[Any] = []
    ties: list[float] = []
    y_sd: list[float] = []
    t0 = time.perf_counter()
    for ep_i in range(spec.n_episodes):
        d = mod.draw(data, np.random.default_rng([spec.seed, ep_i]), cfg, seed=spec.seed, ep_i=ep_i)
        if d is None:
            continue
        overlap = set(map(tuple, np.asarray(d.x_train).tolist())) & set(map(tuple, np.asarray(d.x_test).tolist()))
        shapes.append((len(d.y_train), len(d.y_test), int(np.asarray(d.x_train).shape[1])))
        clusters.add(str(d.meta["cluster"]))
        folds.append(d.meta.get("fold"))
        y = np.concatenate([d.y_train, d.y_test])
        ties.append(_tie_frac(y))
        y_sd.append(float(np.std(y)))
        if overlap and name in ("kaust", "emnist", "meuse"):
            result.setdefault("warnings", []).append(f"ep {ep_i}: a context and a query row share their features")
    dt = time.perf_counter() - t0
    result["draws"] = {
        "attempted": spec.n_episodes,
        "drawn": len(shapes),
        "ms_per_draw": round(1000 * dt / max(spec.n_episodes, 1), 2),
        "shapes_P_N_d": sorted({str(s) for s in shapes}),
        "n_clusters": len(clusters),
        "folds": {str(k): folds.count(k) for k in sorted(set(folds), key=str)},
        "within_episode_tie_frac": _q(np.array(ties)),
        "episode_y_std": _q(np.array(y_sd)),
    }
    return result


def run(spec: RealAuditSpec) -> None:
    """Print (and optionally save) the audit."""
    result = audit(spec)
    print(json.dumps(result, indent=1, default=str))
    if spec.out_json:
        with open(spec.out_json, "w") as fh:
            json.dump(result, fh, indent=1, default=str)
        print(f"Wrote {spec.out_json}")


def _validate(spec: RealAuditSpec) -> None:
    from eval.data.real import REAL_DATASETS

    if spec.real.dataset not in REAL_DATASETS or spec.real.dataset == "sdwpf":
        raise ValueError(
            f"real.dataset must be one of {[d for d in REAL_DATASETS if d != 'sdwpf']} (SDWPF: sdwpf_audit)"
        )
    if spec.n_episodes < 1:
        raise ValueError("n_episodes must be at least 1")


main = hydra_entry("real_audit", RealAuditSpec, run, validate=_validate)


if __name__ == "__main__":
    main()
