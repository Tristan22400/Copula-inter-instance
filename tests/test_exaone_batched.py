"""test_exaone_batched.py — Regression test for
eval/spatial/exaone_batched.py::exaone_run_pit_batched.

Checks the batched multi-episode path against marginal_backends.py's
existing per-episode loo_pit/quantiles path (data.z_train_source=exaone's
production implementation before this module existed) on real, non-mocked
episodes -- exaone_batched.py's docstring claims they're equivalent, this
pins that claim down.

CUDA-gated, not CPU: a single EXAONE fit()+predict() call costs ~15-20s on
CPU (fixed per-call architecture cost, only weakly affected by context size
-- see eval/spatial/marginal_backends.py::make_regressor's "exaone" branch
docstring for the ~120x CPU-vs-CUDA gap measured on an RTX A5000), so a
K-fold comparison test (B*(K+1) reference calls alone) would cost minutes on
CPU -- too slow for a routine test run. On CUDA the same comparison takes
seconds (measured 2026-08-31: ~4s reference + ~2s batched at B=3,P=14,K=4).

Tolerance: NOT bit-exact on CUDA -- batched vs per-episode execution takes a
different path through cuDNN/SDPA kernels (different effective batch size),
which is a well-known source of float32 non-associativity, not a
correctness bug. Confirmed separately on CPU (deterministic, no kernel-order
effects): max abs diff was 2e-7/5e-7/3e-6 for z_train/z_test/log_pdf_test at
B=2,P=12,K=3 (2026-08-31, see this module's own dev notes) -- i.e. batched
and per-episode are mathematically identical, only float-order differs. The
CUDA tolerances below are set well above the ~1e-3-3e-2 noise band actually
observed on an RTX A5000, but far below the ~1.0 magnitude a real bug (e.g.
the contiguous-fold-vs-random-permutation mismatch this module's first draft
had) would produce -- tight enough to catch a real regression, loose enough
to not flake on ordinary kernel-selection noise.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("exaonetabular", reason="exaonetabular not installed")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="EXAONE CPU inference is ~120x slower (see module docstring) -- CUDA-only test",
)


@pytest.fixture(scope="module")
def regressor():
    from eval.spatial.marginal_backends import make_regressor

    return make_regressor("exaone", device="cuda")


def test_exaone_batched_matches_per_episode(regressor):
    from eval.metrics.joint_nll import compute_pit
    from eval.spatial.exaone_batched import exaone_run_pit_batched
    from eval.spatial.marginal_backends import loo_pit, quantiles

    rng = np.random.default_rng(0)
    B, P, N, p_x, K, probs_n = 3, 14, 6, 3, 4, 33
    X_train = rng.normal(size=(B, P, p_x)).astype(np.float32)
    true_w = rng.normal(size=(B, p_x)).astype(np.float32)
    Y_train = (
        np.einsum("bpi,bi->bp", X_train, true_w) + 0.2 * rng.normal(size=(B, P))
    ).astype(np.float32)
    X_test = rng.normal(size=(B, N, p_x)).astype(np.float32)
    Y_test = (
        np.einsum("bni,bi->bn", X_test, true_w) + 0.2 * rng.normal(size=(B, N))
    ).astype(np.float32)
    probs = np.linspace(1.0 / (probs_n + 1), probs_n / (probs_n + 1), probs_n)
    base_seed = 12345

    z_train_ref = np.empty((B, P), dtype=np.float32)
    z_test_ref = np.empty((B, N), dtype=np.float32)
    log_pdf_ref = np.empty((B, N), dtype=np.float32)
    for b in range(B):
        z_train_ref[b] = loo_pit(
            "exaone", regressor, X_train[b], Y_train[b], probs, k_folds=K, seed=base_seed + b
        )
        q_test = quantiles("exaone", regressor, X_train[b], Y_train[b], X_test[b], probs, seed=base_seed + b)
        z_b, lp_b = compute_pit(q_test, probs, Y_test[b])
        z_test_ref[b] = z_b
        log_pdf_ref[b] = lp_b

    out = exaone_run_pit_batched(
        regressor, X_train, Y_train, X_test, Y_test, k_folds=K, probs_n=probs_n, seed=base_seed
    )

    assert np.isfinite(out["z_train"]).all()
    assert np.isfinite(out["z_test"]).all()
    assert np.isfinite(out["log_pdf_test"]).all()
    assert np.max(np.abs(out["z_train"] - z_train_ref)) < 0.1
    assert np.max(np.abs(out["z_test"] - z_test_ref)) < 0.1
    assert np.max(np.abs(out["log_pdf_test"] - log_pdf_ref)) < 0.2
