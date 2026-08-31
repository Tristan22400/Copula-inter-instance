"""s0_signal.py — how much copula signal does the prior even offer?

No model, no training: for a sweep of context sizes P, draws episodes and
scores the exact GP posterior (pit.py::gp_analytical_posterior) against
itself. Reports per-point nll_post_copula / nll_post_marginal (the
Bayes-optimal floor a perfect model would score), the |R_post| off-diagonal
distribution, and the eigenspectrum of R_post - I (how much of the N x N
structure is actually low-rank).

This is the first thing to re-run after any conf/data/gp_tasks.yaml change
— it answers "is there anything here to learn" before asking whether the
model learned it.

Usage:
    python debug/run_debug.py s0
    python debug/stages/s0_signal.py --n-episodes 100 data.P_min=32 data.P_max=32
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "debug")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common
from config import DebugConfig, add_common_args, build_config

P_SWEEP_DEFAULT = [16, 32, 64, 128, 256]


def _offdiag(R: torch.Tensor) -> np.ndarray:
    n = R.shape[0]
    ri, ci = torch.triu_indices(n, n, offset=1)
    return R[ri, ci].cpu().numpy()


def run_one_P(dcfg: DebugConfig, P: int, n_episodes: int) -> dict:
    pairs = common.collect_posteriors(dcfg, n_episodes, P_override=P)
    if not pairs:
        return {"P": P, "n_episodes_scored": 0}

    copula_per_pt, marginal_per_pt = [], []
    offdiag_all = []
    eff_rank_90, eff_rank_99 = [], []
    for ep, post in pairs:
        n_test = int(ep["x_norm_test"].shape[0])
        copula_per_pt.append(float(post["nll_post_copula"]) / n_test)
        marginal_per_pt.append(float(post["nll_post_marginal"]) / n_test)
        R = post["R_post"].float().cpu()
        offdiag_all.append(_offdiag(R))
        eigs = torch.linalg.eigvalsh(R - torch.eye(n_test)).cpu().numpy()
        eigs = np.sort(eigs)[::-1]
        eigs_clipped = np.clip(eigs, 0, None)
        total = eigs_clipped.sum()
        if total > 1e-8:
            cum = np.cumsum(eigs_clipped) / total
            eff_rank_90.append(int(np.searchsorted(cum, 0.90) + 1))
            eff_rank_99.append(int(np.searchsorted(cum, 0.99) + 1))

    offdiag = np.concatenate(offdiag_all) if offdiag_all else np.array([])
    return {
        "P": P,
        "n_episodes_scored": len(pairs),
        "n_episodes_requested": n_episodes,
        "copula_nll_per_point": {
            "mean": float(np.mean(copula_per_pt)), "std": float(np.std(copula_per_pt)),
            "min": float(np.min(copula_per_pt)), "max": float(np.max(copula_per_pt)),
        },
        "marginal_nll_per_point": {
            "mean": float(np.mean(marginal_per_pt)), "std": float(np.std(marginal_per_pt)),
        },
        "offdiag_R_post": {
            "mean": float(offdiag.mean()) if offdiag.size else None,
            "abs_mean": float(np.abs(offdiag).mean()) if offdiag.size else None,
            "std": float(offdiag.std()) if offdiag.size else None,
            "frac_abs_gt_0.3": float((np.abs(offdiag) > 0.3).mean()) if offdiag.size else None,
            "frac_abs_gt_0.7": float((np.abs(offdiag) > 0.7).mean()) if offdiag.size else None,
        },
        "effective_rank": {
            "at_90pct_variance_mean": float(np.mean(eff_rank_90)) if eff_rank_90 else None,
            "at_99pct_variance_mean": float(np.mean(eff_rank_99)) if eff_rank_99 else None,
        },
    }


def run(dcfg: DebugConfig, P_sweep=None) -> dict:
    P_sweep = P_sweep or P_SWEEP_DEFAULT
    per_P = [run_one_P(dcfg, P, dcfg.n_episodes) for P in P_sweep]
    return {"P_sweep": P_sweep, "per_P": per_P}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--P-sweep", default=",".join(str(x) for x in P_SWEEP_DEFAULT),
                   help="Comma-separated P values to sweep (default: 16,32,64,128,256)")
    args = p.parse_args()

    dcfg = build_config(
        overrides=args.override, model_preset=args.model, n_episodes=args.n_episodes,
        ckpt=args.ckpt, device=args.device, seed=args.seed, run_id=args.run_id,
    )
    P_sweep = [int(x) for x in args.P_sweep.split(",")]
    result = run(dcfg, P_sweep=P_sweep)

    print(f"{'P':>6} {'n_scored':>9} {'copula/pt':>11} {'marginal/pt':>12} {'|R| abs_mean':>13} {'eff_rank@90%':>13}")
    for row in result["per_P"]:
        if row["n_episodes_scored"] == 0:
            print(f"{row['P']:>6}   (no episodes scored — all unsupported kernel schema)")
            continue
        print(
            f"{row['P']:>6} {row['n_episodes_scored']:>9} "
            f"{row['copula_nll_per_point']['mean']:>11.4f} "
            f"{row['marginal_nll_per_point']['mean']:>12.4f} "
            f"{row['offdiag_R_post']['abs_mean']:>13.4f} "
            f"{row['effective_rank']['at_90pct_variance_mean']:>13.2f}"
        )

    path = common.save_stage_result(dcfg, "s0_signal", result)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
