"""
diag_kernels.py — Quick sanity check: does each kernel produce meaningful R_star?

Generates N_PER_KERNEL tasks per kernel, prints per-task stats and a summary.
Run from the project root:
    python src/diag_kernels.py
"""
from __future__ import annotations

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
    d_kernel_min: int = 1
    d_kernel_max: int = 4
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
        "l": task["l"].item(),
        "alpha2": task["alpha2"].item(),
        "nugget": task["nugget"].item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

N_PER_KERNEL = 5
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

cfg = Cfg()
# disable list-based selection
cfg.data.kernels = []

SEP = "─" * 70
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
