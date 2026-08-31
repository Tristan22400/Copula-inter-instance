"""test_tabpfn_batched.py — Regression test for
eval/spatial/tabpfn_batched.py::tabpfn_run_pit_batched.

Checks the batched multi-episode path (TabPFNRegressor.predict_batched)
against marginal_backends.py's existing per-episode loo_pit/quantiles path
on real, non-mocked episodes.

Skipped without TABPFN_TOKEN (PriorLabs' own license gate, see
eval/spatial/marginal_backends.py::_require_tabpfn_token) -- this was the
case in the environment tabpfn_batched.py was written in, so unlike
test_exaone_batched.py this test has NEVER actually run end-to-end here.
Set TABPFN_TOKEN and run this before relying on data.z_train_source=tabpfn's
batched path for a real training run -- see tabpfn_batched.py's module
docstring for the same caveat.
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

pytest.importorskip("tabpfn", reason="tabpfn not installed")

pytestmark = pytest.mark.skipif(
    not os.environ.get("TABPFN_TOKEN"),
    reason="TabPFN v3 requires a one-time license acceptance + TABPFN_TOKEN, see module docstring",
)


@pytest.fixture(scope="module")
def regressor():
    from eval.spatial.marginal_backends import make_regressor

    return make_regressor("tabpfn", device="cpu")


def test_tabpfn_batched_matches_per_episode(regressor):
    from eval.metrics.joint_nll import compute_pit
    from eval.spatial.marginal_backends import loo_pit, quantiles
    from eval.spatial.tabpfn_batched import tabpfn_run_pit_batched

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
            "tabpfn", regressor, X_train[b], Y_train[b], probs, k_folds=K, seed=base_seed + b
        )
        q_test = quantiles("tabpfn", regressor, X_train[b], Y_train[b], X_test[b], probs, seed=base_seed + b)
        z_b, lp_b = compute_pit(q_test, probs, Y_test[b])
        z_test_ref[b] = z_b
        log_pdf_ref[b] = lp_b

    out = tabpfn_run_pit_batched(
        regressor, X_train, Y_train, X_test, Y_test, k_folds=K, probs_n=probs_n, seed=base_seed
    )

    assert np.isfinite(out["z_train"]).all()
    assert np.isfinite(out["z_test"]).all()
    assert np.isfinite(out["log_pdf_test"]).all()
    np.testing.assert_allclose(out["z_train"], z_train_ref, atol=1e-3)
    np.testing.assert_allclose(out["z_test"], z_test_ref, atol=1e-3)
    np.testing.assert_allclose(out["log_pdf_test"], log_pdf_ref, atol=1e-2)
