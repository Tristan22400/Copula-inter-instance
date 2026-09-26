"""Debug stage S2: PIT audit in u-space and clamping census.

For each marginal (analytic vs TabICL, paired seeds): u histogram, PIT ECE,
KS against Uniform[0, 1], the fraction of points at the probit clamp
(u <= 1e-6 or >= 1 - 1e-6) and beyond TabICL's outer spline knots (1e-3),
and the fraction of episodes whose z_train std is outside [0.1, 3.0].

Usage:
    python debug/run_debug.py s2
    python debug/stages/s2_uspace.py --n-episodes 100
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from debug import common
from debug.config import DebugConfig, add_common_args, build_config

U_HARD_CLAMP = 1e-6  # pit.py::_probit's clamp -- exactly |z| = 4.7534 beyond this
U_SPLINE_KNOT = 1e-3  # TabICL's outermost quantile knot (num_quantiles=999 -> alpha in [.001,.999])
Z_STD_DEGEN_LO, Z_STD_DEGEN_HI = 0.1, 3.0  # data_gen.py's own post-hoc degeneracy filter bounds


def u_from_z(z: torch.Tensor) -> np.ndarray:
    """u = Phi(z), the exact inverse of pit._probit (clamped points map to the clamp values)."""
    u = 0.5 * (1.0 + torch.special.erf(z / math.sqrt(2.0)))
    return u.detach().cpu().numpy()


def _pit_ece(u: np.ndarray, n_levels: int = 19) -> tuple[float, np.ndarray, np.ndarray]:
    """PIT-uniformity ECE via calibration.compute_quantile_ece with y_true = u and quantile alpha = alpha.

    Returns (ece, alpha_grid, empirical_coverage).
    """
    from eval.spatial.calibration import compute_quantile_ece

    alpha_grid = np.linspace(1.0 / (n_levels + 1), n_levels / (n_levels + 1), n_levels)
    y_pred_quantiles = np.tile(alpha_grid, (len(u), 1))
    ece, coverage = compute_quantile_ece(u, y_pred_quantiles, alpha_grid)
    return ece, alpha_grid, coverage


def _clamp_stats(u_per_episode: "list[np.ndarray]") -> dict:
    pooled = np.concatenate(u_per_episode) if u_per_episode else np.array([])
    per_ep_frac_spline = np.array(
        [float(((e <= U_SPLINE_KNOT) | (e >= 1 - U_SPLINE_KNOT)).mean()) for e in u_per_episode]
    )
    return {
        "pooled_frac_hard_clamp": float(((pooled <= U_HARD_CLAMP) | (pooled >= 1 - U_HARD_CLAMP)).mean())
        if pooled.size
        else None,
        "pooled_frac_spline_saturated": float(((pooled <= U_SPLINE_KNOT) | (pooled >= 1 - U_SPLINE_KNOT)).mean())
        if pooled.size
        else None,
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
            "mean": float(pooled.mean()),
            "std": float(pooled.std()),
            "ks_statistic_pooled": float(ks_pooled.statistic),
            "ks_pvalue_pooled": float(ks_pooled.pvalue),
            "ks_statistic_per_episode_mean": float(np.mean(ks_per_ep)) if ks_per_ep else None,
            "ks_statistic_per_episode_max": float(np.max(ks_per_ep)) if ks_per_ep else None,
            "pit_ece": float(ece),
            "reliability_curve": {"alpha": alpha_grid.tolist(), "coverage": coverage.tolist()},
            "histogram": {"counts": hist.tolist(), "bin_edges": edges.tolist()},
            "clamping_census": _clamp_stats(per_ep),
        }
    return out


def _degeneracy_rate(z_train_per_ep: "list[np.ndarray]") -> float:
    """Fraction of episodes whose z_train std is outside [0.1, 3.0]."""
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
                print(f"  {name}: {d['error']}")
                continue
            cc = d["clamping_census"]
            print(
                f"  {name:8s} mean={d['mean']:.4f} std={d['std']:.4f} "
                f"KS(pooled)={d['ks_statistic_pooled']:.4f} ECE={d['pit_ece']:.4f} | "
                f"hard_clamp={cc['pooled_frac_hard_clamp']:.5f} "
                f"spline_sat={cc['pooled_frac_spline_saturated']:.5f} "
                f"episodes>1%sat={cc['n_episodes_gt_1pct_saturated']}/{cc['n_episodes_total']}"
            )
        print(
            f"  z_train degeneracy rate (post-PIT, std outside [0.1,3.0]): {r['z_train_degeneracy_rate_post_pit']:.4f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = p.parse_args()

    dcfg = build_config(
        overrides=args.override,
        model_preset=args.model,
        n_episodes=args.n_episodes,
        ckpt=args.ckpt,
        device=args.device,
        seed=args.seed,
        run_id=args.run_id,
    )
    result = run(dcfg)
    _print_summary(result)
    path = common.save_stage_result(dcfg, "s2_uspace", result)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
