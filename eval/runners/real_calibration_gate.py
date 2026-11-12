"""Marginal calibration gate for real-data runs: is the marginal's PIT z_test standard normal?

Usage:
    python -m eval.runners.real_calibration_gate dumps=[results/kaust/1a_2_random.json]

Reads eval_checkpoint output.dump_episodes files (real.dataset runs store each
episode's z_test) and prints, per dump and pooled over every test point:
mean and std of z_test, the fraction with |z| > 3, the Kolmogorov-Smirnov
statistic against N(0, 1), and the quantile ECE of the marginal
(copula_inter/marginal/calibration.py; y <= q_tau  <=>  Phi(z) <= tau, so the
ECE is computed on u = Phi(z) against the identity quantiles). A std outside
[std_lo, std_hi] is flagged: copula gains on top of a miscalibrated marginal are
confounded. Points of one episode are dependent, so the KS p-value is
indicative only; read the statistic.

The total-NLL means per method are printed alongside; for EMNIST also in
bits/pixel, (NLL + ln 256) / ln 2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from copula_inter.marginal.calibration import compute_quantile_ece
from eval.runners.hydra_cli import hydra_entry

__all__ = ["RealCalibrationGateSpec", "gate_stats"]

_LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 2)


@dataclass
class RealCalibrationGateSpec:
    """Calibration-gate statistics of the marginal PIT stored in real-data episode dumps."""

    # Dump files or globs (a glob pools the shards of one cell).
    dumps: list[str] = field(default_factory=list)
    # The z_test std band outside which the marginal is flagged as miscalibrated.
    std_lo: float = 0.8
    std_hi: float = 1.25
    out_json: str | None = None


def gate_stats(z: np.ndarray, std_lo: float = 0.8, std_hi: float = 1.25) -> dict[str, Any]:
    """Mean, std, tail fraction, KS against N(0, 1) and quantile ECE of pooled PIT z-scores."""
    from scipy.stats import kstest, norm

    z = np.asarray(z, dtype=np.float64)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return {"n": 0}
    u = norm.cdf(z)
    ece, _ = compute_quantile_ece(u, np.tile(_LEVELS, (len(u), 1)), _LEVELS)
    ks = kstest(z, "norm")
    std = float(z.std(ddof=1)) if z.size > 1 else float("nan")
    return {
        "n": int(z.size),
        "mean": float(z.mean()),
        "std": std,
        "frac_abs_gt3": float((np.abs(z) > 3).mean()),
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "quantile_ece": float(ece),
        "flag": not (std_lo <= std <= std_hi),
    }


def _total_means(episodes: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted(set().union(*(e.get("total_nlls", {}).keys() for e in episodes)))
    out: dict[str, float] = {}
    for k in keys:
        vals = [float((e["total_nlls"].get(k) or {}).get("total", np.nan)) for e in episodes]
        finite = [v for v in vals if np.isfinite(v)]
        if finite:
            out[k] = float(np.mean(finite))
    return out


def run(spec: RealCalibrationGateSpec) -> None:
    """Print the gate per dump and pooled."""
    from eval.data.real.emnist import bits_per_pixel

    results: dict[str, Any] = {}
    pooled: list[float] = []
    import glob

    for path in spec.dumps:
        files = sorted(glob.glob(path)) if any(c in path for c in "*?[") else [path]
        blob: dict[str, Any] = {}
        eps: list[dict[str, Any]] = []
        for f in files:
            with open(f) as fh:
                blob = json.load(fh)
            eps += blob["episodes"]
        z = [v for e in eps for v in e.get("z_test", [])]
        if not z:
            raise ValueError(f"{path} stores no z_test: dump a real.dataset run of this branch")
        pooled.extend(z)
        stats = gate_stats(np.array(z), spec.std_lo, spec.std_hi)
        means = _total_means(eps)
        results[path] = {"dataset": blob.get("real_dataset"), "gate": stats, "total_nll": means}
        flag = "  ** FLAG: std outside " + f"[{spec.std_lo}, {spec.std_hi}]" if stats["flag"] else ""
        print(f"\n{path}  ({blob.get('real_dataset')}, {len(eps)} episodes, {stats['n']} test points){flag}")
        print(
            f"  z_test mean {stats['mean']:+.3f}  std {stats['std']:.3f}  |z|>3 {stats['frac_abs_gt3']:.4f}  "
            f"KS {stats['ks_stat']:.4f} (p {stats['ks_p']:.2g}, dependent points)  quantile ECE {stats['quantile_ece']:.4f}"
        )
        emnist = blob.get("real_dataset") == "emnist"
        for k, v in means.items():
            extra = f"   {bits_per_pixel(v):.4f} bits/pixel" if emnist else ""
            print(f"    total NLL {k:<32}{v:+.4f}{extra}")
    if len(spec.dumps) > 1:
        results["pooled"] = gate_stats(np.array(pooled), spec.std_lo, spec.std_hi)
        print(f"\nPooled: {results['pooled']}")
    if spec.out_json:
        with open(spec.out_json, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"Wrote {spec.out_json}")


def _validate(spec: RealCalibrationGateSpec) -> None:
    if not spec.dumps:
        raise ValueError("dumps=[<dump.json>, ...] is required")
    if not 0 < spec.std_lo < 1 < spec.std_hi:
        raise ValueError("need 0 < std_lo < 1 < std_hi")


main = hydra_entry("real_calibration_gate", RealCalibrationGateSpec, run, validate=_validate)


if __name__ == "__main__":
    main()
