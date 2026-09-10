"""test_tabldm_batched.py — Regression test for
eval/spatial/tabldm_batched.py::tabldm_run_pit_batched.

Checks the batched multi-episode path (episodes stacked along TabLDM's own
documented "number of tables" axis, see that module's docstring) against
marginal_backends.py's existing per-episode loo_pit/quantiles path on real,
non-mocked episodes. This is the check that makes tabldm_batched.py's safety
argument empirical rather than just a reading of TabLDM's source -- the same
standard test_exaone_batched.py holds exaone_batched.py to.

CPU-only and deliberately small (B=2, P=12, K=3): TabLDM costs ~1-5s per
fit+predict call on CPU, and this test issues (K+1) fused calls plus
B*(K+1) reference ones. Correctness, not throughput, is what's being
checked -- the batching win is a GPU-side property and is unrelated to
whether the numbers agree.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("tabldm", reason="Xiaomi-TabLDM not installed")


@pytest.fixture(scope="module")
def regressor():
    from eval.spatial.marginal_backends import make_regressor

    return make_regressor("tabldm", device="cpu")


def test_tabldm_batched_matches_per_episode(regressor):
    from eval.metrics.joint_nll import compute_pit
    from eval.spatial.marginal_backends import loo_pit, quantiles
    from eval.spatial.tabldm_batched import tabldm_run_pit_batched

    rng = np.random.default_rng(0)
    B, P, N, p_x, K, probs_n = 2, 12, 4, 3, 3, 21
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
            "tabldm", regressor, X_train[b], Y_train[b], probs, k_folds=K, seed=base_seed + b
        )
        q_test = quantiles("tabldm", regressor, X_train[b], Y_train[b], X_test[b], probs, seed=base_seed + b)
        z_b, lp_b = compute_pit(q_test, probs, Y_test[b])
        z_test_ref[b] = z_b
        log_pdf_ref[b] = lp_b

    out = tabldm_run_pit_batched(
        regressor, X_train, Y_train, X_test, Y_test, k_folds=K, probs_n=probs_n, seed=base_seed
    )

    assert out["z_train"].shape == (B, P)
    assert out["z_test"].shape == (B, N)
    assert out["log_pdf_test"].shape == (B, N)
    assert np.isfinite(out["z_train"]).all()
    assert np.isfinite(out["z_test"]).all()
    assert np.isfinite(out["log_pdf_test"]).all()
    np.testing.assert_allclose(out["z_train"], z_train_ref, atol=1e-3)
    np.testing.assert_allclose(out["z_test"], z_test_ref, atol=1e-3)
    np.testing.assert_allclose(out["log_pdf_test"], log_pdf_ref, atol=1e-2)


def test_tabldm_batched_rejects_unsupported_regressor_modes(regressor):
    """The two predict() branches this module deliberately does not mirror
    must fail loudly rather than silently fusing the wrong thing."""
    from eval.spatial.tabldm_batched import _episode_member_batch

    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 3)).astype(np.float32)
    y = rng.normal(size=10).astype(np.float32)

    original = regressor.enhance_candidates
    regressor.enhance_candidates = True
    try:
        with pytest.raises(RuntimeError, match="enhance_candidates"):
            _episode_member_batch(regressor, X[:8], y[:8], X[8:])
    finally:
        regressor.enhance_candidates = original
