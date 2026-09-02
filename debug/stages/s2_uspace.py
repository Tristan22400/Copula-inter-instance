"""s2_uspace.py — PIT (u-space) audit + clamping census.

Works in u = F_hat(y) rather than z = Phi^-1(u): a well-calibrated marginal
gives u ~ Uniform[0,1], which is directly visible as a flat histogram and a
diagonal reliability curve, and the clamping question IS a question about
where u piles up. z is only produced from pit.py::run_pit_batched (it
returns z, not u); since u = 0.5*(1+erf(z/sqrt(2))) is the EXACT inverse of
pit.py::_probit's forward transform, converting back is lossless for every
point that didn't saturate, and recovers the exact clamp value (1e-6 or
1-1e-6) for every point that did -- which is exactly what the census below
needs, without touching pit.py.

Reports, per marginal backend (default: analytic GP-LOO vs. TabICL PIT,
paired same-seed via common.generate_paired_episodes):
  - u histogram + PIT calibration curve/ECE (reusing eval/spatial/
    calibration.py::compute_quantile_ece -- see _pit_ece's docstring for
    how a generic quantile-coverage function also computes PIT-uniformity
    ECE without modification)
  - KS statistic vs Uniform[0,1], pooled and per-episode
  - CLAMPING CENSUS: fraction of points at the hard _probit clamp
    (u <= 1e-6 or u >= 1-1e-6, source of the exactly-+-4.7534 z spike) and
    at TabICL's outermost spline knot (u <= 1e-3 or u >= 1-1e-3, beyond
    which its quantile head switches to exponential-tail extrapolation --
    see tabicl_upstream's quantile_dist.py), pooled AND per-episode, plus
    the count of episodes with >1% of points saturated
  - the fraction of episodes that would fail data_gen.py's own
    z_std in [0.1, 3.0] degeneracy filter if applied post-PIT (it is
    computed on the analytic residual only, data_gen.py:3429-3442, so a
    degenerate TabICL z_train is never rejected today)

Usage:
    python debug/run_debug.py s2
    python debug/stages/s2_uspace.py --n-episodes 100
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC, os.path.join(_REPO_ROOT, "debug")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common
from config import DebugConfig, add_common_args, build_config

U_HARD_CLAMP = 1e-6      # pit.py::_probit's clamp -- exactly |z| = 4.7534 beyond this
U_SPLINE_KNOT = 1e-3     # TabICL's outermost quantile knot (num_quantiles=999 -> alpha in [.001,.999])
Z_STD_DEGEN_LO, Z_STD_DEGEN_HI = 0.1, 3.0  # data_gen.py's own post-hoc degeneracy filter bounds


def u_from_z(z: torch.Tensor) -> np.ndarray:
    """Exact inverse of pit.py::_probit's forward transform (erfinv -> erf),
    lossless everywhere z didn't saturate the clamp, and returns exactly
    U_HARD_CLAMP / 1-U_HARD_CLAMP for every point that did."""
    u = 0.5 * (1.0 + torch.special.erf(z / math.sqrt(2.0)))
    return u.detach().cpu().numpy()


def _pit_ece(u: np.ndarray, n_levels: int = 19) -> tuple[float, np.ndarray, np.ndarray]:
    """PIT-uniformity ECE via eval/spatial/calibration.py::compute_quantile_ece,
    reused rather than reimplemented: that function's generic definition
    (mean |nominal_level - P(y_true <= predicted_quantile)|) computes exactly
    PIT calibration ECE when y_true=u and the "predicted quantile" at level
    alpha is the constant alpha itself, since u <= alpha IFF the true value
    falls at or below the alpha-quantile by u's own definition (u = F_hat(y)).
    Returns (ece, alpha_grid, empirical_coverage).
    """
    from eval.spatial.calibration import compute_quantile_ece

    alpha_grid = np.linspace(1.0 / (n_levels + 1), n_levels / (n_levels + 1), n_levels)
    y_pred_quantiles = np.tile(alpha_grid, (len(u), 1))
    ece, coverage = compute_quantile_ece(u, y_pred_quantiles, alpha_grid)
    return ece, alpha_grid, coverage


def _clamp_stats(u_per_episode: "list[np.ndarray]") -> dict:
    pooled = np.concatenate(u_per_episode) if u_per_episode else np.array([])
    per_ep_frac_spline = np.array([
        float(((e <= U_SPLINE_KNOT) | (e >= 1 - U_SPLINE_KNOT)).mean()) for e in u_per_episode
    ])
    return {
        "pooled_frac_hard_clamp": float(((pooled <= U_HARD_CLAMP) | (pooled >= 1 - U_HARD_CLAMP)).mean()) if pooled.size else None,
        "pooled_frac_spline_saturated": float(((pooled <= U_SPLINE_KNOT) | (pooled >= 1 - U_SPLINE_KNOT)).mean()) if pooled.size else None,
        "per_episode_frac_spline_saturated": {
            "mean": float(per_ep_frac_spline.mean()) if per_ep_frac_spline.size else None,
            "max": float(per_ep_frac_spline.max()) if per_ep_frac_spline.size else None,
        },
        "n_episodes_gt_1pct_saturated": int((per_ep_frac_spline > 0.01).sum()),
        "n_episodes_total": len(u_per_episode),
    }


def _audit_source(u_train_per_ep, u_test_per_ep) -> dict:
    from scipy.stats import kstest

    out = {}
    for name, per_ep in (("z_train", u_train_per_ep), ("z_test", u_test_per_ep)):
        pooled = np.concatenate(per_ep) if per_ep else np.array([])
        if pooled.size == 0:
            out[name] = {"error": "no points"}
            continue
        ks_pooled = kstest(pooled, "uniform")
        ks_per_ep = [kstest(e, "uniform").statistic for e in per_ep if e.size >= 2]
        ece, alpha_grid, coverage = _pit_ece(pooled)
        hist, edges = np.histogram(pooled, bins=20, range=(0, 1))
        out[name] = {
            "mean": float(pooled.mean()), "std": float(pooled.std()),
            "ks_statistic_pooled": float(ks_pooled.statistic), "ks_pvalue_pooled": float(ks_pooled.pvalue),
            "ks_statistic_per_episode_mean": float(np.mean(ks_per_ep)) if ks_per_ep else None,
            "ks_statistic_per_episode_max": float(np.max(ks_per_ep)) if ks_per_ep else None,
            "pit_ece": float(ece),
            "reliability_curve": {"alpha": alpha_grid.tolist(), "coverage": coverage.tolist()},
            "histogram": {"counts": hist.tolist(), "bin_edges": edges.tolist()},
            "clamping_census": _clamp_stats(per_ep),
        }
    return out


def _degeneracy_rate(z_train_per_ep: "list[np.ndarray]") -> float:
    """Fraction of episodes whose z_train std falls outside [0.1, 3.0] --
    data_gen.py's own analytic-residual degeneracy filter (data_gen.py:
    3429-3442), applied here post-hoc to the TabICL PIT output, which never
    goes through that filter today."""
    if not z_train_per_ep:
        return float("nan")
    stds = np.array([e.std() for e in z_train_per_ep if e.size >= 2])
    if stds.size == 0:
        return float("nan")
    return float(((stds < Z_STD_DEGEN_LO) | (stds > Z_STD_DEGEN_HI)).mean())


def run(dcfg: DebugConfig) -> dict:
    analytic_eps, tabicl_eps = common.generate_paired_episodes(dcfg, dcfg.n_episodes)

    def _z_arrays(episodes, key):
        return [ep[key].detach().cpu().numpy() for ep in episodes]

    result = {}
    for label, episodes in (("analytic", analytic_eps), ("tabicl", tabicl_eps)):
        z_train_np = _z_arrays(episodes, "z_train")
        z_test_np = _z_arrays(episodes, "z_test")
        u_train = [u_from_z(torch.from_numpy(z)) for z in z_train_np]
        u_test = [u_from_z(torch.from_numpy(z)) for z in z_test_np]
        result[label] = _audit_source(u_train, u_test)
        result[label]["z_train_degeneracy_rate_post_pit"] = _degeneracy_rate(z_train_np)

    result["n_episodes"] = len(analytic_eps)
    return result


def _print_summary(result: dict) -> None:
    for label in ("analytic", "tabicl"):
        r = result[label]
        print(f"\n=== {label} ===")
        for name in ("z_train", "z_test"):
            d = r[name]
            if "error" in d:
                print(f"  {name}: {d['error']}"); continue
            cc = d["clamping_census"]
            print(
                f"  {name:8s} mean={d['mean']:.4f} std={d['std']:.4f} "
                f"KS(pooled)={d['ks_statistic_pooled']:.4f} ECE={d['pit_ece']:.4f} | "
                f"hard_clamp={cc['pooled_frac_hard_clamp']:.5f} "
                f"spline_sat={cc['pooled_frac_spline_saturated']:.5f} "
                f"episodes>1%sat={cc['n_episodes_gt_1pct_saturated']}/{cc['n_episodes_total']}"
            )
        print(f"  z_train degeneracy rate (post-PIT, std outside [0.1,3.0]): {r['z_train_degeneracy_rate_post_pit']:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = p.parse_args()

    dcfg = build_config(
        overrides=args.override, model_preset=args.model, n_episodes=args.n_episodes,
        ckpt=args.ckpt, device=args.device, seed=args.seed, run_id=args.run_id,
    )
    result = run(dcfg)
    _print_summary(result)
    path = common.save_stage_result(dcfg, "s2_uspace", result)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
