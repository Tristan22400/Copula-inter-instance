"""Model-free audit of the SDWPF wind farm, per day-block fold: what the data supports before any model is scored.

Usage:
    python -m eval.runners.sdwpf_audit [data_dir=<dir>] [output_dir=results/sdwpf_audit]

Prints, and writes to output_dir as audit.json, per_day_atoms.csv and farm_direction_rose.png:
  - record, turbine and day counts, and the rows each cleaning rule drops;
  - the rated power (rated_quantile of cleaned Patv over all days, as the episodes use);
  - the atom fractions (Patv == 0, Patv >= atom_hi_frac * rated), overall, per fold and per day;
  - eligible hourly timestamps per fold and context size, under three readings of partial load
    (below), the last being the episode rule exactly;
  - pairwise turbine distances and the farm extent;
  - the F1b direction audit on all days, per fold: the spread of phi_i = (Ndir + Wdir) mod 360
    across turbines per timestamp, the distribution of the farm direction phi, and the mean
    Spearman correlation of u (positive downwind) with Patv for each direction_offset_deg in
    {0, 180, 90, -90}; wakes make it negative under the right convention;
  - the cross-fitted direction decision for each fold (below);
  - the empirical Patv correlation of turbine pairs by distance.

partial_load readings: strict (every valid turbine of the timestamp is inside), pooled (at
least P + N valid turbines are), draw (expected count when P + N turbines are drawn from all
valid ones and the timestamp is kept only if none is an atom: the episode rule).

Cross-fitted direction decision (fixed before any number was seen). For held-out fold k, on
the other four folds' hourly timestamps with at least P + N partial-load turbines (the pooled
reading, the one closest to the episodes): the chosen offset is the one with the most negative
mean Spearman(u, Patv); it is "clearly negative" when its 95 % day-bootstrap CI lies entirely
below 0. F1b is admissible for fold k only if the chosen offset is clearly negative and the
median per-timestamp circular std of phi_i on those folds is at most spread_max_deg (30 deg).
Fold k's own timestamps never enter its decision. phi in this audit is the circular mean over
all partial-load turbines of the timestamp; inside an episode it is over the context rows only.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.special import gammaln
from scipy.stats import rankdata

from copula_inter.paths import SDWPF_DATA_DIR
from eval.data.real.sdwpf import (
    ALL_DAYS,
    DIRECTION_OFFSETS,
    FOLD_DAYS,
    STEPS_PER_DAY,
    STEPS_PER_HOUR,
    SdwpfTable,
    absolute_wind_direction,
    circular_mean_deg,
    estimate_rated_power,
    load_sdwpf,
    wind_aligned_axes,
)
from eval.metrics.cluster_bootstrap import cluster_bootstrap_ci
from eval.runners.hydra_cli import hydra_entry

_SECTORS = 16
FOLDS = {f"fold{k}": days for k, days in enumerate(FOLD_DAYS)}


@dataclass
class SdwpfAuditSpec:
    """Model-free audit of the SDWPF files; nothing here reads a model or a score."""

    data_dir: str = str(SDWPF_DATA_DIR)
    output_dir: str = "results/sdwpf_audit"
    n_query: int = 64
    # Context sizes whose eligible-timestamp counts are reported: the main P first, then the P sweep.
    n_context: list[int] = field(default_factory=lambda: [32, 16, 64])
    # Partial load is atom_lo < Patv < atom_hi_frac * rated.
    atom_lo: float = 0.0
    atom_hi_frac: float = 0.99
    rated_quantile: float = 0.999
    # F1b is inadmissible for a fold when the median per-timestamp spread of phi_i exceeds this.
    spread_max_deg: float = 30.0
    distance_bins: int = 12
    # Angular resolution of the most-negative wake-axis scan.
    scan_step_deg: float = 10.0
    n_boot: int = 2000
    seed: int = 0


def _log_comb(n: np.ndarray, k: int) -> np.ndarray:
    return np.asarray(gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0))


def _quantiles(values: np.ndarray, qs: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)) -> dict[str, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {f"p{round(q * 100)}": float("nan") for q in qs}
    return {f"p{round(q * 100)}": float(np.quantile(vals, q)) for q in qs}


def _spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman correlation of each row of a (k, m) with b (m,)."""
    ra = rankdata(a, axis=1)
    rb = rankdata(b)
    ra = ra - ra.mean(axis=1, keepdims=True)
    rb = rb - rb.mean()
    denom = np.linalg.norm(ra, axis=1) * np.linalg.norm(rb)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray((ra @ rb) / denom)


def _angle_diff(a: np.ndarray, b: np.ndarray | float) -> np.ndarray:
    """Signed difference a - b wrapped to [-180, 180)."""
    return np.asarray(np.mod(np.asarray(a) - b + 180.0, 360.0) - 180.0)


def _fold_of_steps(steps: np.ndarray) -> np.ndarray:
    """The fold index of each step (its day's 49-day block)."""
    day = steps // STEPS_PER_DAY + 1
    return np.asarray((day - 1) // (FOLD_DAYS[0][1] - FOLD_DAYS[0][0] + 1))


def _atoms(table: SdwpfTable, rated: float, atom_lo: float, atom_hi_frac: float) -> dict[str, Any]:
    hi = atom_hi_frac * rated
    valid = table.valid
    patv = table.patv

    def fractions(steps: np.ndarray) -> dict[str, float]:
        p = patv[steps][valid[steps]].astype(np.float64)
        return {
            "n": int(p.size),
            "eq_0": float(np.mean(p == 0)),
            "le_atom_lo": float(np.mean(p <= atom_lo)),
            "in_0_10kw": float(np.mean((p > 0) & (p <= 10))),
            "ge_atom_hi": float(np.mean(p >= hi)),
            "ge_0.95_rated": float(np.mean(p >= 0.95 * rated)),
            "gt_rated": float(np.mean(p > rated)),
        }

    out: dict[str, Any] = {"atom_hi_kw": hi, "overall": fractions(np.arange(table.n_steps))}
    for name, days in FOLDS.items():
        out[name] = fractions(table.period_steps(days))
    n_days = table.n_steps // STEPS_PER_DAY
    per_step_valid = valid.sum(axis=1)
    with np.errstate(invalid="ignore"):
        per_step_zero = (valid & (patv == 0)).sum(axis=1)
        per_step_hi = (valid & (patv >= hi)).sum(axis=1)
    n_valid_day = per_step_valid.reshape(n_days, STEPS_PER_DAY).sum(axis=1)
    safe = np.maximum(n_valid_day, 1)
    zero_day = per_step_zero.reshape(n_days, STEPS_PER_DAY).sum(axis=1) / safe
    hi_day = per_step_hi.reshape(n_days, STEPS_PER_DAY).sum(axis=1) / safe
    out["per_day"] = {
        "day": list(range(1, n_days + 1)),
        "n_valid": n_valid_day.tolist(),
        "frac_zero": zero_day.tolist(),
        "frac_hi": hi_day.tolist(),
    }
    out["per_day_summary"] = {"frac_zero": _quantiles(zero_day), "frac_hi": _quantiles(hi_day)}
    return out


def _eligibility(
    table: SdwpfTable, rated: float, *, atom_lo: float, atom_hi_frac: float, n_context: list[int], n_query: int
) -> dict[str, Any]:
    """Eligible hourly timestamps per fold and P, under the strict, pooled and draw (episode-rule) readings."""
    valid = table.valid
    with np.errstate(invalid="ignore"):
        inside = valid & (table.patv > atom_lo) & (table.patv < atom_hi_frac * rated)
    n_valid = valid.sum(axis=1)
    n_inside = inside.sum(axis=1)
    out: dict[str, Any] = {}
    for name, days in FOLDS.items():
        steps = table.period_steps(days)
        steps = steps[steps % STEPS_PER_HOUR == 0]
        nv, ni = n_valid[steps], n_inside[steps]
        rows: dict[str, Any] = {"hourly_timestamps": int(steps.size), "n_valid_per_timestamp": _quantiles(nv)}
        for P in n_context:
            k = P + n_query
            enough = nv >= k
            p_clean = np.where(
                enough & (ni >= k), np.exp(_log_comb(ni.astype(float), k) - _log_comb(nv.astype(float), k)), 0.0
            )
            rows[f"P{P}_N{n_query}"] = {
                "all": int(enough.sum()),
                "partial_strict": int((enough & (ni == nv)).sum()),
                "partial_pooled": int((ni >= k).sum()),
                "partial_draw_expected": float(p_clean.sum()),
            }
        out[name] = rows
    return out


def _geometry(table: SdwpfTable) -> dict[str, Any]:
    xy = np.stack([table.x, table.y], axis=1)
    d = pdist(xy)
    full = squareform(d)
    np.fill_diagonal(full, np.inf)
    return {
        "pair_distance_m": {"min": float(d.min()), "median": float(np.median(d)), "max": float(d.max())},
        "nearest_neighbour_m": _quantiles(full.min(axis=1)),
        "extent_m": {
            "x": [float(table.x.min()), float(table.x.max())],
            "y": [float(table.y.min()), float(table.y.max())],
        },
    }


def _offset_stats(
    rho: np.ndarray, days: np.ndarray, sel: np.ndarray, n_boot: int, seed: int
) -> dict[str, dict[str, Any]]:
    """Per offset: mean, 95 % day-bootstrap CI and quantiles of the per-timestamp Spearman rows in sel."""
    out: dict[str, dict[str, Any]] = {}
    for j, off in enumerate(DIRECTION_OFFSETS):
        col = rho[sel, j]
        ok = np.isfinite(col)
        mean, lo, up = cluster_bootstrap_ci(col[ok], days[sel][ok], n_boot=n_boot, seed=seed)
        out[f"{off:+d}"] = {
            "n": int(ok.sum()),
            "mean": mean,
            "ci95": [lo, up],
            **_quantiles(col, (0.1, 0.5, 0.9)),
            "frac_negative": float(np.mean(col[ok] < 0)) if ok.any() else float("nan"),
        }
    return out


def _direction_checks(
    table: SdwpfTable,
    rated: float,
    *,
    atom_lo: float,
    atom_hi_frac: float,
    min_turbines: int,
    spread_max_deg: float,
    scan_step_deg: float,
    n_boot: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Spread, distribution and frame alignment of phi_i per fold, and the cross-fitted F1b decisions."""
    valid = table.valid
    steps = np.arange(0, table.n_steps, STEPS_PER_HOUR)
    steps = steps[valid[steps].sum(axis=1) >= min_turbines]
    fold = _fold_of_steps(steps)
    days = (steps // STEPS_PER_DAY + 1).astype(str)
    phi_i = absolute_wind_direction(table.ndir[steps], table.wdir[steps])  # NaN where invalid
    farm_phi, resultant = circular_mean_deg(phi_i, axis=1)
    with np.errstate(divide="ignore"):
        circ_std = np.rad2deg(np.sqrt(-2.0 * np.log(np.clip(resultant, 1e-12, 1.0))))
    med_abs_dev = np.nanmedian(np.abs(_angle_diff(phi_i, farm_phi[:, None])), axis=1)
    hi = atom_hi_frac * rated
    with np.errstate(invalid="ignore"):
        inside = valid[steps] & (table.patv[steps] > atom_lo) & (table.patv[steps] < hi)
    strict = inside.sum(axis=1) == valid[steps].sum(axis=1)
    pooled = inside.sum(axis=1) >= min_turbines

    spread: dict[str, Any] = {}
    for k, name in enumerate(FOLDS):
        sel = fold == k
        spread[name] = {
            "timestamps": int(sel.sum()),
            "circular_std_deg": _quantiles(circ_std[sel], (0.5, 0.95)),
            "median_abs_dev_deg": _quantiles(med_abs_dev[sel], (0.5, 0.95)),
        }
    # A turbine whose yaw sensor is offset shows a steady bias against the farm mean.
    bias, _ = circular_mean_deg(_angle_diff(phi_i, farm_phi[:, None]), axis=0)
    bias_abs = np.abs(_angle_diff(bias, 0.0))
    spread["per_turbine_bias_deg"] = {
        **_quantiles(bias_abs, (0.5, 0.9)),
        "max": float(np.nanmax(bias_abs)),
        "n_over_30": int(np.sum(bias_abs > 30)),
        "worst_turbines": [int(t) for t in table.turb_ids[np.argsort(-np.nan_to_num(bias_abs))[:5]]],
    }

    edges = np.linspace(0.0, 360.0, _SECTORS + 1)
    distribution: dict[str, Any] = {"sector_centres_deg": edges[:-1].tolist()}
    for k, name in enumerate(FOLDS):
        sel = fold == k
        # Sectors centred on N, NNE, ...: shift by half a sector.
        hist, _ = np.histogram(np.mod(farm_phi[sel] + 180.0 / _SECTORS, 360.0), bins=edges)
        m, res = circular_mean_deg(farm_phi[sel])
        distribution[name] = {
            "sector_fraction": (hist / max(1, hist.sum())).tolist(),
            "circular_mean_deg": float(m),
            "resultant_length": float(res),
        }

    # Spearman(u, Patv) across partial-load turbines per timestamp, for each offset and a scan of axes.
    scan = np.arange(0.0, 360.0, scan_step_deg)
    readings: dict[str, Any] = {}
    rho_pooled = np.full((len(steps), len(DIRECTION_OFFSETS)), np.nan)
    for reading, rows_mask in (("partial_strict", strict), ("partial_pooled", pooled)):
        rows = np.flatnonzero(rows_mask)
        rho = np.full((len(steps), len(DIRECTION_OFFSETS)), np.nan)
        best = np.full(len(steps), np.nan)
        for ri in rows:
            r = int(ri)
            cols = np.flatnonzero(inside[r] if reading == "partial_pooled" else valid[steps[r]])
            phi_r, _ = circular_mean_deg(phi_i[r, cols])
            angles = np.concatenate([float(phi_r) + np.asarray(DIRECTION_OFFSETS, dtype=float), scan])
            u, _ = wind_aligned_axes(table.x[cols][None, :], table.y[cols][None, :], angles[:, None])
            corr = _spearman_rows(u, table.patv[steps[r], cols].astype(np.float64))
            rho[r] = corr[: len(DIRECTION_OFFSETS)]
            scan_corr = corr[len(DIRECTION_OFFSETS) :]
            if np.isfinite(scan_corr).any():
                best[r] = _angle_diff(scan[np.nanargmin(scan_corr)], float(phi_r))
        if reading == "partial_pooled":
            rho_pooled = rho
        per_fold: dict[str, Any] = {}
        for k, name in enumerate(FOLDS):
            sel = rows_mask & (fold == k)
            ok = sel & np.isfinite(best)
            dm, dr = circular_mean_deg(best[ok]) if ok.any() else (np.nan, np.nan)
            per_fold[name] = {
                "timestamps": int(sel.sum()),
                "offsets": _offset_stats(rho, days, sel, n_boot, seed),
                # Where the most-negative wake axis sits relative to phi: a tight resultant = phi is informative.
                "best_axis_minus_phi": {"mean_deg": float(dm), "resultant": float(dr)},
            }
        readings[reading] = per_fold

    decisions: dict[str, Any] = {}
    for k, name in enumerate(FOLDS):
        train = pooled & (fold != k)
        stats = _offset_stats(rho_pooled, days, train, n_boot, seed)
        chosen = min(stats, key=lambda o: stats[o]["mean"] if np.isfinite(stats[o]["mean"]) else np.inf)
        clearly_negative = bool(stats[chosen]["ci95"][1] < 0)
        spread_med = float(np.nanmedian(circ_std[fold != k]))
        decisions[name] = {
            "training_folds": [f for j, f in enumerate(FOLDS) if j != k],
            "training_timestamps": int(train.sum()),
            "offsets": {o: {"mean": s["mean"], "ci95": s["ci95"]} for o, s in stats.items()},
            "chosen_offset_deg": int(chosen),
            "clearly_negative": clearly_negative,
            "median_circular_std_deg": spread_med,
            "spread_ok": spread_med <= spread_max_deg,
            "f1b_admissible": clearly_negative and spread_med <= spread_max_deg,
        }
    return (
        {
            "timestamps": int(len(steps)),
            "spread": spread,
            "distribution": distribution,
            "frame_alignment": readings,
            "cross_fitted_decisions": decisions,
        },
        farm_phi,
    )


def _correlation_by_distance(
    table: SdwpfTable, rated: float, *, atom_lo: float, atom_hi_frac: float, n_bins: int
) -> dict[str, Any]:
    """Pairwise correlation across all timestamps: raw Patv, and its cross-sectional anomaly (farm mean removed)."""
    patv = table.patv.astype(np.float64)
    valid = ~np.isnan(patv)
    with np.errstate(invalid="ignore"):
        inside = valid & (patv > atom_lo) & (patv < atom_hi_frac * rated)
    strict = inside.sum(axis=1) == valid.sum(axis=1)
    anomaly = patv - np.nanmean(patv, axis=1, keepdims=True)
    raw_c = pd.DataFrame(patv).corr(min_periods=100).to_numpy()
    anom_c = pd.DataFrame(anomaly).corr(min_periods=100).to_numpy()
    anom_pl = pd.DataFrame(anomaly[strict]).corr(min_periods=100).to_numpy()
    iu = np.triu_indices(len(table.turb_ids), k=1)
    dist = squareform(pdist(np.stack([table.x, table.y], axis=1)))[iu]
    edges = np.linspace(dist.min(), dist.max() + 1e-9, n_bins + 1)
    which = np.digitize(dist, edges) - 1
    bins = []
    for b in range(n_bins):
        m = which == b
        if not m.any():
            continue
        ac = anom_c[iu][m]
        bins.append(
            {
                "d_lo_m": float(edges[b]),
                "d_hi_m": float(edges[b + 1]),
                "pairs": int(m.sum()),
                "raw": float(np.nanmean(raw_c[iu][m])),
                "anomaly": float(np.nanmean(ac)),
                "anomaly_frac_negative": float(np.nanmean(ac < 0)),
                "anomaly_partial_strict": float(np.nanmean(anom_pl[iu][m])),
            }
        )
    return {"partial_strict_timestamps": int(strict.sum()), "bins": bins}


def audit(
    table: SdwpfTable,
    *,
    n_query: int = 64,
    n_context: list[int] | None = None,
    atom_lo: float = 0.0,
    atom_hi_frac: float = 0.99,
    rated_quantile: float = 0.999,
    spread_max_deg: float = 30.0,
    distance_bins: int = 12,
    scan_step_deg: float = 10.0,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[dict[str, Any], np.ndarray]:
    """Every audit number as a JSON-ready dict, plus the hourly farm directions for the rose plot."""
    n_context = [32, 16, 64] if n_context is None else list(n_context)
    rated = estimate_rated_power(table, ALL_DAYS, rated_quantile)
    patv_all = table.patv[table.valid]
    direction, farm_phi = _direction_checks(
        table,
        rated,
        atom_lo=atom_lo,
        atom_hi_frac=atom_hi_frac,
        min_turbines=n_context[0] + n_query,
        spread_max_deg=spread_max_deg,
        scan_step_deg=scan_step_deg,
        n_boot=n_boot,
        seed=seed,
    )
    result: dict[str, Any] = {
        "data": {
            "n_records": table.n_raw,
            "n_turbines": len(table.turb_ids),
            "n_days": table.n_steps // STEPS_PER_DAY,
            "dropped": table.dropped,
            "flagged": table.flagged,
            "n_clipped_negative_patv": table.n_clipped,
            "n_valid": int(table.valid.sum()),
            "identity": table.identity,
        },
        "folds": {name: list(days) for name, days in FOLDS.items()},
        "rated_power": {
            "estimate_kw": rated,
            "quantile": rated_quantile,
            "max_patv_kw": float(patv_all.max()),
        },
        "atoms": _atoms(table, rated, atom_lo, atom_hi_frac),
        "eligibility": _eligibility(
            table, rated, atom_lo=atom_lo, atom_hi_frac=atom_hi_frac, n_context=n_context, n_query=n_query
        ),
        "geometry": _geometry(table),
        "direction": direction,
        "correlation_by_distance": _correlation_by_distance(
            table, rated, atom_lo=atom_lo, atom_hi_frac=atom_hi_frac, n_bins=distance_bins
        ),
    }
    return result, farm_phi


def _print_report(r: dict[str, Any]) -> None:
    d = r["data"]
    print("\n== Data ==")
    print(f"records {d['n_records']:,}  turbines {d['n_turbines']}  days {d['n_days']}  valid {d['n_valid']:,}")
    for rule in d["dropped"]:
        print(f"  {rule:9s} dropped {d['dropped'][rule]:>9,}  (matches alone {d['flagged'][rule]:>9,})")
    print(f"  negative Patv clipped to 0: {d['n_clipped_negative_patv']:,}")

    rp = r["rated_power"]
    print("\n== Rated power ==")
    print(f"q{rp['quantile']} of cleaned Patv, all days: {rp['estimate_kw']:.1f} kW (max {rp['max_patv_kw']:.1f})")

    a = r["atoms"]
    print(f"\n== Atoms (atom_hi = {a['atom_hi_kw']:.1f} kW), fraction of valid records ==")
    print(f"{'':8s} {'n':>10s} {'==0':>7s} {'(0,10]':>7s} {'>=hi':>7s} {'>=.95r':>7s} {'>rated':>7s}")
    for name in ("overall", *FOLDS):
        f = a[name]
        print(
            f"{name:8s} {f['n']:>10,} {f['eq_0']:>7.3f} {f['in_0_10kw']:>7.3f} {f['ge_atom_hi']:>7.3f} "
            f"{f['ge_0.95_rated']:>7.3f} {f['gt_rated']:>7.3f}"
        )
    for key in ("frac_zero", "frac_hi"):
        q = a["per_day_summary"][key]
        print(f"  per day {key}: " + "  ".join(f"{k} {v:.3f}" for k, v in q.items()))

    print("\n== Eligible hourly timestamps per fold (all / strict / pooled / episode rule, expected) ==")
    for name, rows in r["eligibility"].items():
        nv = rows["n_valid_per_timestamp"]
        print(
            f"{name}: {rows['hourly_timestamps']} hourly; valid turbines per timestamp "
            + "  ".join(f"{k} {v:.0f}" for k, v in nv.items())
        )
        for shape, cell in rows.items():
            if not shape.startswith("P"):
                continue
            print(
                f"  {shape:9s} all {cell['all']:>4}  strict {cell['partial_strict']:>4}  "
                f"pooled {cell['partial_pooled']:>4}  draw ~{cell['partial_draw_expected']:.0f}"
            )

    g = r["geometry"]
    pd_m = g["pair_distance_m"]
    print("\n== Geometry ==")
    print(f"pair distance m: min {pd_m['min']:.0f}  median {pd_m['median']:.0f}  max {pd_m['max']:.0f}")
    print("nearest neighbour m: " + "  ".join(f"{k} {v:.0f}" for k, v in g["nearest_neighbour_m"].items()))
    print(f"extent m: x {g['extent_m']['x']}  y {g['extent_m']['y']}")

    dr = r["direction"]
    print(f"\n== F1b 1: spread of phi_i across turbines per hourly timestamp ({dr['timestamps']} ts) ==")
    for name in FOLDS:
        s = dr["spread"][name]
        print(
            f"  {name}: n={s['timestamps']:>5}  circular std p50 {s['circular_std_deg']['p50']:.1f} "
            f"p95 {s['circular_std_deg']['p95']:.1f}  |  median |dev| p50 {s['median_abs_dev_deg']['p50']:.1f} "
            f"p95 {s['median_abs_dev_deg']['p95']:.1f} deg"
        )
    b = dr["spread"]["per_turbine_bias_deg"]
    print(
        f"  per-turbine |bias| vs farm mean: p50 {b['p50']:.1f}  p90 {b['p90']:.1f}  max {b['max']:.1f} deg, "
        f"{b['n_over_30']} turbines over 30 deg (worst {b['worst_turbines']})"
    )

    print("\n== F1b 2: farm direction phi (wind from), sector fractions % ==")
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    print("        " + " ".join(f"{n:>4s}" for n in names))
    for name in FOLDS:
        p = dr["distribution"][name]
        print(
            f"  {name} "
            + " ".join(f"{100 * f:>4.1f}" for f in p["sector_fraction"])
            + f"   mean {p['circular_mean_deg']:.0f} R {p['resultant_length']:.2f}"
        )

    print("\n== F1b 3: mean Spearman(u, Patv) across partial-load turbines, per fold [95% day-bootstrap CI] ==")
    for reading, per_fold in dr["frame_alignment"].items():
        print(f"  reading {reading}")
        for name, s in per_fold.items():
            cells = "  ".join(
                f"{o:>4s}: {v['mean']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]" for o, v in s["offsets"].items()
            )
            ba = s["best_axis_minus_phi"]
            print(
                f"    {name} n={s['timestamps']:>5}  {cells}  | best axis - phi {ba['mean_deg']:+.0f} R {ba['resultant']:.2f}"
            )

    print("\n== F1b 4: cross-fitted decisions (fold k decided on the other four, pooled reading) ==")
    for name, dcs in dr["cross_fitted_decisions"].items():
        o = dcs["offsets"][f"{dcs['chosen_offset_deg']:+d}"]
        print(
            f"  {name}: offset {dcs['chosen_offset_deg']:+4d}  mean {o['mean']:+.3f} "
            f"[{o['ci95'][0]:+.3f},{o['ci95'][1]:+.3f}]  clearly_negative={dcs['clearly_negative']}  "
            f"spread p50 {dcs['median_circular_std_deg']:.1f} deg  -> F1b admissible: {dcs['f1b_admissible']}"
        )

    cd = r["correlation_by_distance"]
    print(f"\n== Patv correlation by pair distance, all days (partial_strict: {cd['partial_strict_timestamps']} ts) ==")
    print(f"  {'d range m':>13s} {'pairs':>6s} {'raw':>6s} {'anom':>6s} {'anom<0':>7s} {'anom|PL':>8s}")
    for bn in cd["bins"]:
        print(
            f"  {bn['d_lo_m']:>6.0f}-{bn['d_hi_m']:<6.0f} {bn['pairs']:>6d} {bn['raw']:>6.3f} {bn['anomaly']:>6.3f} "
            f"{bn['anomaly_frac_negative']:>7.2f} {bn['anomaly_partial_strict']:>8.3f}"
        )


def _write_outputs(result: dict[str, Any], farm_phi: np.ndarray, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    per_day = result["atoms"].pop("per_day")
    with open(out_dir / "per_day_atoms.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["day", "n_valid", "frac_zero", "frac_hi"])
        writer.writerows(zip(per_day["day"], per_day["n_valid"], per_day["frac_zero"], per_day["frac_hi"]))
    with open(out_dir / "audit.json", "w") as fh:
        json.dump(result, fh, indent=2)
    fig = plt.figure(figsize=(5, 5))
    from matplotlib.projections.polar import PolarAxes

    ax = fig.add_subplot(projection="polar")
    assert isinstance(ax, PolarAxes)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.hist(np.deg2rad(farm_phi), bins=36)
    ax.set_title("SDWPF farm wind direction (from), hourly, all days")
    fig.savefig(out_dir / "farm_direction_rose.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_dir}/audit.json, per_day_atoms.csv, farm_direction_rose.png")


def run(args: SdwpfAuditSpec) -> None:
    table = load_sdwpf(args.data_dir)
    result, farm_phi = audit(
        table,
        n_query=args.n_query,
        n_context=args.n_context,
        atom_lo=args.atom_lo,
        atom_hi_frac=args.atom_hi_frac,
        rated_quantile=args.rated_quantile,
        spread_max_deg=args.spread_max_deg,
        distance_bins=args.distance_bins,
        scan_step_deg=args.scan_step_deg,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    _print_report(result)
    _write_outputs(result, farm_phi, Path(args.output_dir))


def validate_audit_spec(args: SdwpfAuditSpec) -> None:
    if not args.n_context or min(args.n_context) < 1 or args.n_query < 1:
        raise ValueError("n_context must list positive context sizes and n_query must be positive")
    if not 0.0 < args.atom_hi_frac <= 1.0:
        raise ValueError(f"atom_hi_frac={args.atom_hi_frac} must lie in (0, 1]")


main = hydra_entry("sdwpf_audit", SdwpfAuditSpec, run, validate=validate_audit_spec)

if __name__ == "__main__":
    main()
