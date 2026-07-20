"""
diag_kernels.py — Quick sanity check: does each kernel produce meaningful R_star?

Generates N_PER_KERNEL tasks per kernel, prints per-task stats and a summary,
then (Stage 3) pools the off-diagonal R_star entries across N_STAGE3 tasks per
kernel to check the family isn't collapsing toward independence (screening
effect) or degenerating toward triviality (near-1 correlations everywhere).
Run from the project root:
    python src/diag_kernels.py
    python src/diag_kernels.py --n-stage3 200   # smaller/faster batch
    python src/diag_kernels.py --skip-stage3    # per-task checks only

DataCfg/Cfg/check_task/batch_off_diagonal_stats below are also imported
directly by tests/test_diag_kernels.py, which turns this same per-task/
Stage-3 health check into real pytest assertions (parametrized over
ALL_KERNELS) so a regression like a kernel family losing PSD-ness is caught
by the test suite instead of only a manually-run script. Everything below
this module's "Main" section only runs under `if __name__ == "__main__":`,
so importing this module (e.g. from a test) has no side effects.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass, field
from typing import List

import torch

sys.path.insert(0, "src")
from data_gen import generate_gp_task, ALL_KERNELS  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal cfg stub (mirrors gp_tasks.yaml defaults)
# ---------------------------------------------------------------------------

@dataclass
class DataCfg:
    d_features: int = 10
    inactive_frac_min: float = 0.6
    inactive_frac_max: float = 0.9
    P_min: int = 5
    P_max: int = 30
    N_min: int = 10
    N_max: int = 50
    l_min: float = 1.0
    l_max: float = 10.0
    alpha2_min: float = 0.5
    alpha2_max: float = 2.0
    dot_product_alpha2_min: float = 0.0   # must be >= 0 for PSD (see gp_tasks.yaml)
    dot_product_alpha2_max: float = 1.0
    nugget_min: float = 0.1
    nugget_max: float = 1.0
    period_min: float = 0.5
    period_max: float = 3.0
    rq_alpha_min: float = 0.1
    rq_alpha_max: float = 5.0
    # will be overridden per run
    kernel: str = "rbf"
    kernels: list = field(default_factory=list)


@dataclass
class Cfg:
    data: DataCfg = field(default_factory=DataCfg)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_task(task: dict, kernel_name: str, task_idx: int) -> dict:
    R = task["R_star"]        # (N, N)
    Sigma = None              # we don't return Sigma from generate_gp_task
    N = R.shape[0]
    issues: List[str] = []

    # --- 1. NaN / Inf ---
    if torch.isnan(R).any():
        issues.append("NaN in R_star")
    if torch.isinf(R).any():
        issues.append("Inf in R_star")

    if issues:  # can't do further numeric checks
        return {"ok": False, "issues": issues}

    # --- 2. Diagonal = 1 ---
    diag_err = (R.diagonal() - 1.0).abs().max().item()
    if diag_err > 1e-4:
        issues.append(f"diag not 1 (max_err={diag_err:.2e})")

    # --- 3. Values in [-1, 1] ---
    r_min, r_max = R.min().item(), R.max().item()
    if r_min < -1.01 or r_max > 1.01:
        issues.append(f"out-of-range values: [{r_min:.3f}, {r_max:.3f}]")

    # --- 4. Symmetry ---
    sym_err = (R - R.T).abs().max().item()
    if sym_err > 1e-5:
        issues.append(f"not symmetric (max_err={sym_err:.2e})")

    # --- 5. PSD: smallest eigenvalue ---
    eigvals = torch.linalg.eigvalsh(R)
    min_eig = eigvals.min().item()
    if min_eig < -1e-4:
        issues.append(f"not PSD (min_eigval={min_eig:.4f})")

    # --- 6. Correlation structure: off-diagonal stats ---
    mask = ~torch.eye(N, dtype=torch.bool)
    off_diag = R[mask]
    od_mean = off_diag.mean().item()
    od_std  = off_diag.std().item()
    od_abs  = off_diag.abs().mean().item()

    # "Meaningful" = not all near 0 (no structure) or all same value (trivial)
    if od_std < 1e-4:
        issues.append(f"trivial off-diagonal (std={od_std:.2e}, all vals ~same)")
    if od_abs < 1e-3:
        issues.append(f"near-zero correlations (mean|r|={od_abs:.4f})")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "diag_err": diag_err,
        "r_min": r_min,
        "r_max": r_max,
        "min_eig": min_eig,
        "od_mean": od_mean,
        "od_std": od_std,
        "od_abs_mean": od_abs,
        "N": N,
        "P": task["n_train"].item(),
        # scalar, unless the episode was generated ARD (cfg.data.ard=True),
        # in which case l is a per-dimension lengthscale vector (k,) —
        # report its mean here.
        "l": task["l"].mean().item() if task["l"].numel() > 1 else task["l"].item(),
        "alpha2": task["alpha2"].item(),
        "nugget": task["nugget"].item(),
    }


# ---------------------------------------------------------------------------
# Stage 3 — information-theoretic bounds on the off-diagonal distribution
# ---------------------------------------------------------------------------
# The task is only learnable if the training context P leaves a meaningful
# residual dependence in R_star at the test points N: too little (posterior
# collapses toward independence — the "screening effect") and the model just
# learns to predict I_N; too much (R_star saturates near +-1 everywhere) and
# the task is trivially degenerate (test instances are basically identical).

COLLAPSE_THRESHOLD   = 0.01  # E[|R*_offdiag|] below this -> screening effect
DEGENERATE_THRESHOLD = 0.95  # E[|R*_offdiag|] above this -> trivial task
HEALTHY_LOW, HEALTHY_HIGH = 0.1, 0.8  # target band for a "Goldilocks" mean|r|


def _ascii_histogram(values: torch.Tensor, n_bins: int = 20, width: int = 40) -> str:
    """Text histogram of |R*_offdiag| in [0, 1], for a quick look without a display."""
    counts = torch.histc(values.abs(), bins=n_bins, min=0.0, max=1.0)
    max_count = counts.max().item()
    lines = []
    for i, c in enumerate(counts.tolist()):
        lo, hi = i / n_bins, (i + 1) / n_bins
        bar_len = int(width * c / max_count) if max_count > 0 else 0
        lines.append(f"    |r| in [{lo:.2f},{hi:.2f})  {'#' * bar_len:<{width}}  {int(c)}")
    return "\n".join(lines)


def batch_off_diagonal_stats(kernel_name: str, cfg: "Cfg", n_tasks: int) -> dict:
    """Generate n_tasks GP tasks for kernel_name and pool all off-diagonal R_star entries."""
    cfg.data.kernel = kernel_name
    pooled: List[torch.Tensor] = []
    for _ in range(n_tasks):
        task = generate_gp_task(cfg)
        R = task["R_star"]
        N = R.shape[0]
        mask = ~torch.eye(N, dtype=torch.bool)
        pooled.append(R[mask])
    off_diag = torch.cat(pooled)
    abs_off_diag = off_diag.abs()

    mean_abs = abs_off_diag.mean().item()
    if mean_abs < COLLAPSE_THRESHOLD:
        verdict = "COLLAPSED (screening effect — model will just learn identity)"
    elif mean_abs > DEGENERATE_THRESHOLD:
        verdict = "DEGENERATE (trivially near-identical instances)"
    elif HEALTHY_LOW <= mean_abs <= HEALTHY_HIGH:
        verdict = "HEALTHY"
    else:
        verdict = "BORDERLINE (outside the [0.1, 0.8] Goldilocks band)"

    quantiles = torch.quantile(
        abs_off_diag.float(), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    ).tolist()

    return {
        "kernel": kernel_name,
        "n_tasks": n_tasks,
        "n_pairs": off_diag.numel(),
        "mean": off_diag.mean().item(),
        "mean_abs": mean_abs,
        "std": off_diag.std().item(),
        "p5": quantiles[0], "p25": quantiles[1], "p50": quantiles[2],
        "p75": quantiles[3], "p95": quantiles[4],
        "frac_collapsed": (abs_off_diag < COLLAPSE_THRESHOLD).float().mean().item(),
        "frac_degenerate": (abs_off_diag > DEGENERATE_THRESHOLD).float().mean().item(),
        "verdict": verdict,
        "off_diag": off_diag,
    }


def run_stage3(cfg: "Cfg", n_tasks: int) -> bool:
    """Runs the batch off-diagonal check for every kernel; returns True iff all HEALTHY."""
    print(f"\n{SEP}")
    print(f"  STAGE 3 — INFORMATION-THEORETIC BOUNDS (n_tasks={n_tasks} per kernel)")
    print(SEP)

    all_healthy = True
    fig_axes = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ncols = 3
        nrows = math.ceil(len(ALL_KERNELS) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        fig_axes = list(axes.flat)
        for ax in fig_axes[len(ALL_KERNELS):]:
            ax.set_visible(False)
    except ImportError:
        fig = None

    for i, kernel_name in enumerate(ALL_KERNELS):
        stats = batch_off_diagonal_stats(kernel_name, cfg, n_tasks)
        if stats["verdict"] != "HEALTHY":
            all_healthy = False

        print(f"\n  {kernel_name.upper()}  ({stats['n_pairs']:,} pooled off-diagonal entries)")
        print(
            f"    E[|R*_offdiag|]={stats['mean_abs']:.4f}  "
            f"mean={stats['mean']:+.4f}  std={stats['std']:.4f}"
        )
        print(
            f"    percentiles(|r|): p5={stats['p5']:.3f} p25={stats['p25']:.3f} "
            f"p50={stats['p50']:.3f} p75={stats['p75']:.3f} p95={stats['p95']:.3f}"
        )
        print(
            f"    frac(|r|<{COLLAPSE_THRESHOLD})={stats['frac_collapsed']*100:.2f}%   "
            f"frac(|r|>{DEGENERATE_THRESHOLD})={stats['frac_degenerate']*100:.2f}%"
        )
        print(f"    verdict: {stats['verdict']}")
        if stats["verdict"] == "COLLAPSED (screening effect — model will just learn identity)":
            print("    fix: increase l_min (or l_log_uniform range) so training context "
                  "explains less of the test set, or decrease nugget_max.")
        elif stats["verdict"] == "DEGENERATE (trivially near-identical instances)":
            print("    fix: decrease l_max so instances decorrelate faster, "
                  "or increase nugget_min.")
        print(_ascii_histogram(stats["off_diag"]))

        if fig is not None:
            ax = fig_axes[i]
            ax.hist(stats["off_diag"].numpy(), bins=40, range=(-1, 1), color="steelblue")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(f"{kernel_name}\nE[|r|]={stats['mean_abs']:.3f} ({stats['verdict'].split()[0]})",
                         fontsize=9)

    if fig is not None:
        fig.suptitle(f"Off-diagonal R_star distribution ({n_tasks} tasks/kernel)")
        fig.tight_layout()
        out_path = "plots/off_diag_distribution.png"
        import os
        os.makedirs("plots", exist_ok=True)
        fig.savefig(out_path, dpi=120)
        print(f"\n  Saved histogram grid to: {out_path}")

    print(f"\n  STAGE 3: {'ALL FAMILIES HEALTHY' if all_healthy else '*** SOME FAMILIES OUT OF BAND ***'}")
    return all_healthy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

N_PER_KERNEL = 5
SEED = 42


def main() -> bool:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-stage3", type=int, default=1000,
                         help="Tasks per kernel for the Stage 3 batch distribution check (default: 1000)")
    parser.add_argument("--skip-stage3", action="store_true",
                         help="Skip the Stage 3 batch distribution check")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    cfg = Cfg()
    # disable list-based selection
    cfg.data.kernels = []

    all_ok = True

    for kernel_name in ALL_KERNELS:
        cfg.data.kernel = kernel_name
        print(f"\n{SEP}")
        print(f"  KERNEL: {kernel_name.upper()}")
        print(SEP)

        kernel_ok = True
        for i in range(N_PER_KERNEL):
            task = generate_gp_task(cfg)
            result = check_task(task, kernel_name, i)

            status = "OK " if result["ok"] else "FAIL"
            if not result["ok"]:
                kernel_ok = False
                all_ok = False

            if result["ok"]:
                print(
                    f"  [{status}] task {i+1}  "
                    f"P={result['P']:2d} N={result['N']:2d}  "
                    f"l={result['l']:.2f} a2={result['alpha2']:.2f} nug={result['nugget']:.2f}  "
                    f"r:[{result['r_min']:+.3f},{result['r_max']:+.3f}]  "
                    f"od_mean={result['od_mean']:+.3f}  od_std={result['od_std']:.3f}  "
                    f"|r|_mean={result['od_abs_mean']:.3f}  "
                    f"min_eig={result['min_eig']:.4f}"
                )
            else:
                print(f"  [{status}] task {i+1}  ISSUES: {result['issues']}")
                # also print any numeric stats if they were computed
                for key in ("r_min", "r_max", "od_mean", "od_std", "od_abs_mean", "min_eig"):
                    if key in result:
                        print(f"           {key}={result[key]:.4f}")

        print(f"  {'ALL PASSED' if kernel_ok else '*** FAILURES DETECTED ***'}")

    print(f"\n{SEP}")
    print(f"  OVERALL: {'ALL KERNELS OK' if all_ok else 'SOME FAILURES — see above'}")
    print(SEP)

    if not args.skip_stage3:
        stage3_ok = run_stage3(cfg, args.n_stage3)
        all_ok = all_ok and stage3_ok

    return all_ok


SEP = "─" * 70

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
