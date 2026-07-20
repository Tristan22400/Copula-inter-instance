"""
test_diag_kernels.py — Automated version of src/diag_kernels.py's health check.

diag_kernels.py is a manually-run script ("does each kernel produce a
meaningful, valid R_star?"); this file runs the same checks as real pytest
assertions, parametrized over every entry in ALL_KERNELS (27 base + composite
kernels), so a regression (e.g. a kernel family losing PSD-ness) is caught by
the test suite instead of requiring someone to notice a manual script run.

Uses diag_kernels.DataCfg/Cfg rather than conftest's small_cfg: DataCfg's
ranges mirror gp_tasks.yaml's production defaults (d_features=10, P up to 30,
N up to 50, inactive_frac_min/max=0.6/0.9), which is the regime that actually
exercises the numerically fragile composites (e.g. cosine+rational_quadratic)
— small_cfg's much narrower ranges (d_features=1, P<=10, N<=6) would not
reliably reproduce that.
"""

from __future__ import annotations

import random

import pytest
import torch

from data_gen import ALL_KERNELS, generate_gp_task
from diag_kernels import Cfg, batch_off_diagonal_stats, check_task

# ---------------------------------------------------------------------------
# Per-task structural checks (NaN/Inf, unit diagonal, [-1,1] range, symmetry,
# PSD, non-trivial off-diagonal structure) — mirrors diag_kernels.py's Stage 1/2.
# ---------------------------------------------------------------------------

N_TASKS_PER_KERNEL = 10
SEED = 42


@pytest.fixture
def cfg():
    c = Cfg()
    c.data.kernels = []  # force single-kernel selection via c.data.kernel
    return c


@pytest.mark.parametrize("kernel_name", ALL_KERNELS)
def test_kernel_produces_valid_r_star(cfg, kernel_name):
    """Every kernel family must produce NaN/Inf-free, unit-diagonal, PSD,
    non-trivial R_star across repeated sampling."""
    torch.manual_seed(SEED)
    random.seed(SEED)
    cfg.data.kernel = kernel_name

    failures = []
    for i in range(N_TASKS_PER_KERNEL):
        task = generate_gp_task(cfg)
        result = check_task(task, kernel_name, i)
        if not result["ok"]:
            failures.append((i, result["issues"]))

    assert not failures, (
        f"{kernel_name}: {len(failures)}/{N_TASKS_PER_KERNEL} tasks failed — {failures}"
    )


# ---------------------------------------------------------------------------
# Stage-3-style distributional check: off-diagonal R_star shouldn't collapse
# toward independence (screening effect) or degenerate toward triviality.
# Uses a smaller n_tasks than diag_kernels.py's script default (1000) to keep
# CI fast, and only fails on COLLAPSED/DEGENERATE (not "BORDERLINE" — a
# family sitting just outside the informal [0.1, 0.8] Goldilocks band is not
# itself a correctness bug, just worth a human glancing at the histogram).
# ---------------------------------------------------------------------------

N_TASKS_STAGE3 = 100


@pytest.mark.parametrize("kernel_name", ALL_KERNELS)
def test_kernel_off_diagonal_not_degenerate(cfg, kernel_name):
    torch.manual_seed(SEED)
    random.seed(SEED)
    cfg.data.kernel = kernel_name

    stats = batch_off_diagonal_stats(kernel_name, cfg, N_TASKS_STAGE3)
    assert not stats["verdict"].startswith("COLLAPSED"), (
        f"{kernel_name}: screening effect — E[|R*_offdiag|]={stats['mean_abs']:.4f} "
        f"({stats['n_pairs']} pooled pairs)"
    )
    assert not stats["verdict"].startswith("DEGENERATE"), (
        f"{kernel_name}: trivially near-identical instances — "
        f"E[|R*_offdiag|]={stats['mean_abs']:.4f} ({stats['n_pairs']} pooled pairs)"
    )
