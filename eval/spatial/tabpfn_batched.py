"""tabpfn_batched.py — genuine multi-episode batched PIT for TabPFN v3, the
"batch mode" analogue of pit.py::run_pit_batched for
data.z_train_source=tabpfn. Unlike marginal_backends.py's per-episode
quantiles()/loo_pit() (still used for exaone's fallback path and
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
    """``run_pit_batched``, TabPFN version. Signature, y-unit convention and
    return dict are identical to exaone_batched.py/tabldm_batched.py's -- the
    whole K-fold driver is shared (_batched_pit.py::run_kfold_pit_batched,
    which also documents the fold-assignment recipe and the equal-fold-size
    guarantee batching relies on); only the bank above is backend-specific.

    probs_n is requested directly from TabPFN's predict_batched, so unlike
    exaone_batched.py there is no native-grid interpolation step.
    """
    from eval.spatial._batched_pit import run_kfold_pit_batched

    return run_kfold_pit_batched(
        lambda X_ctx, y_ctx, X_qry, probs: _quantile_bank_batched(
            regressor, X_ctx, y_ctx, X_qry, probs
        ),
        X_train, Y_train, X_test, Y_test,
        k_folds=k_folds, probs_n=probs_n, eps=eps, seed=seed,
    )
