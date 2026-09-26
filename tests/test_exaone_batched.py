"""exaone_run_pit_batched vs the per-episode loo_pit/quantiles path on real episodes (CUDA only).

Not bit-exact on CUDA (different kernel paths); tolerances sit above the
observed float noise and far below the error of a real bug.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

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
