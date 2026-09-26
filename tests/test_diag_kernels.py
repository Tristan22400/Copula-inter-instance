"""diag_kernels' R_star health checks as tests, for every kernel in ALL_KERNELS, with diag_kernels.DataCfg (production-like ranges)."""

from __future__ import annotations

import random

import pytest
import torch

from copula_inter.data_gen import ALL_KERNELS, generate_gp_task
from copula_inter.diag_kernels import Cfg, batch_off_diagonal_stats, check_task

# Per-task checks: finite, unit diagonal, range, symmetry, PSD, non-trivial.

N_TASKS_PER_KERNEL = 10
SEED = 42


@pytest.fixture
def cfg() -> Cfg:
    c = Cfg()
    c.data.kernels = []  # force single-kernel selection via c.data.kernel
    return c


@pytest.mark.parametrize("kernel_name", ALL_KERNELS)
def test_kernel_produces_valid_r_star(cfg: Cfg, kernel_name: str) -> None:
    """Every kernel gives finite, unit-diagonal, PSD, non-trivial R_star."""
    torch.manual_seed(SEED)
    random.seed(SEED)
    cfg.data.kernel = kernel_name

    failures = []
    for i in range(N_TASKS_PER_KERNEL):
        task = generate_gp_task(cfg)
        result = check_task(task, kernel_name, i)
        if not result["ok"]:
            failures.append((i, result["issues"]))

    assert not failures, f"{kernel_name}: {len(failures)}/{N_TASKS_PER_KERNEL} tasks failed — {failures}"


# Stage-3 check over fewer tasks: fail only on COLLAPSED or DEGENERATE.

N_TASKS_STAGE3 = 100


@pytest.mark.parametrize("kernel_name", ALL_KERNELS)
def test_kernel_off_diagonal_not_degenerate(cfg: Cfg, kernel_name: str) -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    cfg.data.kernel = kernel_name

    stats = batch_off_diagonal_stats(kernel_name, cfg, N_TASKS_STAGE3)
    assert not stats["verdict"].startswith("COLLAPSED"), (
        f"{kernel_name}: screening effect — E[|R*_offdiag|]={stats['mean_abs']:.4f} ({stats['n_pairs']} pooled pairs)"
    )
    assert not stats["verdict"].startswith("DEGENERATE"), (
        f"{kernel_name}: trivially near-identical instances — "
        f"E[|R*_offdiag|]={stats['mean_abs']:.4f} ({stats['n_pairs']} pooled pairs)"
    )
