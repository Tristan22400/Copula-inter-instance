"""tabpfn_batched.py — genuine multi-episode batched PIT for TabPFN v3, the
"batch mode" analogue of pit.py::run_pit_batched for
data.z_train_source=tabpfn. Unlike marginal_backends.py's per-episode
quantiles()/loo_pit() (still used for exaone's fallback path/tabfm/tabm and
for a single ad-hoc call), this batches the expensive step -- the model
forward -- across every episode in a shard-generation call.

MUCH lower-risk than eval/spatial/exaone_batched.py's approach: TabPFN
ships this as a first-class PUBLIC, DOCUMENTED method --
TabPFNRegressor.predict_batched(X_train_list, y_train_list, X_test_list,
output_type=..., quantiles=...) -- "Predict for several independent datasets
in one pass... all datasets are stacked on the model's batch dimension and
scored with a single fused forward per estimator. Equivalent to fitting and
predicting each dataset independently." (see that method's own docstring).
No internal-API reverse-engineering needed here, unlike exaone_batched.py.

Requires every X_train_list/X_test_list entry to share one shape (TabPFN
rejects ragged batches rather than padding them) -- guaranteed by
data_gen.py's per-call P/N/feature-count homogeneity (generate_gp_batch's
module docstring), the same assumption pit.py::run_pit_batched already
relies on for TabICL.

Execution-verified: tests/test_tabpfn_batched.py passed end-to-end against a
real TabPFN v3 API call (TABPFN_TOKEN set), matching the per-episode
quantiles()/loo_pit() reference to the same tolerances used for
exaone_batched.py -- no fold-assignment or shape bug surfaced on the first
real run, unlike exaone_batched.py's internal-API path. Requires a
PriorLabs-issued TABPFN_TOKEN env var (see
marginal_backends.py::_require_tabpfn_token); the token itself must never be
committed to the repo -- export it as a shell/CI secret only.
"""

from __future__ import annotations

import numpy as np

__all__ = ["tabpfn_run_pit_batched"]


def _quantile_bank_batched(
    regressor, X_context: list, y_context: list, X_query: list, probs: np.ndarray,
) -> np.ndarray:
    """One fused predict_batched call for B episodes sharing
    support_rows/query_rows/feature_count (guaranteed by data_gen.py's
    per-call homogeneity -- see module docstring). Returns
    (B, query_rows, len(probs)) in RAW y-units -- unlike EXAONE's fixed
    999-level native grid, TabPFN's predict_batched decodes directly onto
    the requested `probs`, no interpolation step needed.
    """
    results = regressor.predict_batched(
        list(X_context), list(y_context), list(X_query),
        output_type="quantiles", quantiles=list(probs),
    )
    # Each entry mirrors predict()'s own quantiles contract (see
    # marginal_backends.py::quantiles' "tabpfn" branch): (n_quantiles,
    # n_query) -> transpose to (n_query, n_quantiles).
    return np.stack([np.asarray(r).T for r in results], axis=0)  # (B, query_rows, len(probs))


def tabpfn_run_pit_batched(
    regressor, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray,
    k_folds: int = 10, probs_n: int = 99, eps: float = 1e-6, seed: int = 0,
) -> dict:
    """``run_pit_batched``, TabPFN version -- K-fold PIT for z_train AND a
    single held-in-context pass for z_test/log_pdf_test, batched across every
    episode in the call via TabPFNRegressor.predict_batched (see module
    docstring).

    Args:
        X_train: (B, P, p_x)   X_test: (B, N, p_x)
        Y_train: (B, P)        Y_test: (B, N)      -- already y-scaled by the
            caller, same convention as pit.py::run_pit_batched.
        k_folds: clamped into [1, P] (matching eval/metrics/joint_nll.py::
            kfold_loo_pit's own `min(k_folds, n)`), shared across the batch
            since P is.
        probs_n: quantile grid size requested directly from TabPFN (no
            native-grid interpolation needed, unlike exaone_batched.py).
        seed: per-episode fold assignment uses
            np.random.default_rng(seed + b).permutation(P) % K -- the exact
            recipe eval/metrics/joint_nll.py::kfold_loo_pit uses (see
            exaone_batched.py::exaone_run_pit_batched's docstring for why
            fold SIZE, not membership, is guaranteed equal across episodes
            despite each using its own seed -- same argument applies here).

    Returns dict with z_train (B,P), z_test (B,N), log_pdf_test (B,N) --
    log_pdf_test is in the SAME (already-scaled) y-units as Y_test; callers
    apply their own Jacobian correction back to raw-y nats.
    """
    from eval.metrics.joint_nll import compute_pit

    B, P, _p_x = X_train.shape
    N = X_test.shape[1]
    K = min(int(k_folds), P)
    probs = np.linspace(1.0 / (probs_n + 1), probs_n / (probs_n + 1), probs_n)

    fold_ids = [np.random.default_rng(seed + b).permutation(P) % K for b in range(B)]

    z_train = np.empty((B, P), dtype=np.float32)
    for k in range(K):
        held = [fold_ids[b] == k for b in range(B)]
        qry_idx = [np.where(held[b])[0] for b in range(B)]
        ctx_idx = [np.where(~held[b])[0] for b in range(B)]
        if qry_idx[0].size == 0 or ctx_idx[0].size == 0:
            continue  # size is shared across b (see docstring); checking b=0 suffices
        X_ctx = [X_train[b][ctx_idx[b]] for b in range(B)]
        y_ctx = [Y_train[b][ctx_idx[b]] for b in range(B)]
        X_qry = [X_train[b][qry_idx[b]] for b in range(B)]
        bank = _quantile_bank_batched(regressor, X_ctx, y_ctx, X_qry, probs)  # (B, F, len(probs))
        for b in range(B):
            z_held, _ = compute_pit(bank[b], probs, Y_train[b][qry_idx[b]], eps)
            z_train[b, qry_idx[b]] = z_held

    bank_test = _quantile_bank_batched(
        regressor, list(X_train), list(Y_train), list(X_test), probs
    )  # (B, N, len(probs))
    z_test = np.empty((B, N), dtype=np.float32)
    log_pdf_test = np.empty((B, N), dtype=np.float32)
    for b in range(B):
        z_b, log_pdf_b = compute_pit(bank_test[b], probs, Y_test[b], eps)
        z_test[b] = z_b
        log_pdf_test[b] = log_pdf_b

    return {"z_train": z_train, "z_test": z_test, "log_pdf_test": log_pdf_test}
