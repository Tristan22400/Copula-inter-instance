"""_batched_pit.py — the K-fold-PIT driver shared by every batched marginal
backend (exaone_batched.py, tabpfn_batched.py, tabldm_batched.py).

Those three modules differ only in HOW they turn a list of B episodes into a
quantile bank -- EXAONE fuses ensemble-member tensors along its executor's
chunkable batch axis, TabPFN calls its public ``predict_batched``, TabLDM
stacks episodes along ``predict_stats``' documented "number of tables" axis.
Everything downstream of that -- the fold assignment, which rows are held
out, the per-fold context/query split, the compute_pit calls, the output
dict -- is byte-for-byte the same work regardless of backend, and was
literally duplicated between exaone_batched.py and tabpfn_batched.py before
this module existed.

That duplication is the kind this repo explicitly avoids: the loop below
decides which y-value each z_train entry is scored against, so a divergence
between two copies wouldn't crash, it would silently train the copula head
on subtly wrong targets in one backend only. One tested implementation,
three thin callers.

The backend-specific part is passed in as ``bank_fn``:

    bank_fn(X_context: list[np.ndarray],   # B arrays, (n_context, p_x)
            y_context: list[np.ndarray],   # B arrays, (n_context,)
            X_query:   list[np.ndarray],   # B arrays, (n_query, p_x)
            probs:     np.ndarray,         # (Q,)
            ) -> np.ndarray                # (B, n_query, Q), RAW y-units

Note ``probs`` is the CALLER's grid, not the backend's native one: a backend
whose model emits a fixed native grid (EXAONE's 999 evenly spaced levels)
interpolates onto ``probs`` inside its own bank_fn, so this driver never
sees a grid mismatch and needs no per-backend interpolation branch.
"""

from __future__ import annotations

import numpy as np

__all__ = ["run_kfold_pit_batched"]


def run_kfold_pit_batched(
    bank_fn, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray,
    k_folds: int = 10, probs_n: int = 99, eps: float = 1e-6, seed: int = 0,
) -> dict:
    """``run_pit_batched`` for any batched marginal backend -- K-fold PIT for
    z_train AND a single held-in-context pass for z_test/log_pdf_test,
    batched across every episode in the call.

    Args:
        bank_fn: the backend's batched quantile-bank callable (see module
            docstring for its exact contract).
        X_train: (B, P, p_x)   X_test: (B, N, p_x)
        Y_train: (B, P)        Y_test: (B, N)      -- already y-scaled by the
            caller (data_gen.py z-scores y_train/y_test per episode before
            calling this, same convention as pit.py::run_pit_batched).
        k_folds: clamped into [1, P] (matching eval/metrics/joint_nll.py::
            kfold_loo_pit's own `min(k_folds, n)`, NOT pit.py::run_pit_
            batched's `max(2, ...)` floor), shared across the batch since P
            is -- see below for why fold SIZE is guaranteed equal across
            episodes even though fold MEMBERSHIP differs per episode.
        probs_n: size of the quantile grid handed to bank_fn.
        seed: per-episode fold assignment uses
            np.random.default_rng(seed + b).permutation(P) % K -- the exact
            recipe eval/metrics/joint_nll.py::kfold_loo_pit uses (bit-
            identical to marginal_backends.py::loo_pit's fold splits when
            called with matching per-episode seeds, e.g. data_gen.py's
            marginal_backend branch's `seed_b = (base_seed + b) %
            (2**31)`), NOT pit.py::run_pit_batched's shared contiguous-block
            split -- these backends' per-episode PIT paths already committed
            to the random-permutation convention, and batching exists to
            make them faster, not to change their semantics. Fold SIZE (not
            membership) only depends on P and K, both shared across the
            batch, so every episode's fold k has the same query-row count
            regardless of its own seed -- permutation preserves the multiset
            of residues {0..P-1} mod K, just reorders which original row
            index lands in which fold -- so batching per fold across
            episodes is still valid despite the differing seeds. That
            equal-size guarantee is what lets bank_fn stack episodes into
            one rectangular forward at all.

    Returns dict with z_train (B,P), z_test (B,N), log_pdf_test (B,N) --
    log_pdf_test is in the SAME (already-scaled) y-units as Y_test; callers
    apply their own Jacobian correction back to raw-y nats, matching every
    other backend's convention in this pipeline.
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
        bank = bank_fn(X_ctx, y_ctx, X_qry, probs)  # (B, F, Q)
        for b in range(B):
            z_held, _ = compute_pit(bank[b], probs, Y_train[b][qry_idx[b]], eps)
            z_train[b, qry_idx[b]] = z_held

    bank_test = bank_fn(
        [X_train[b] for b in range(B)], [Y_train[b] for b in range(B)],
        [X_test[b] for b in range(B)], probs,
    )  # (B, N, Q)
    z_test = np.empty((B, N), dtype=np.float32)
    log_pdf_test = np.empty((B, N), dtype=np.float32)
    for b in range(B):
        z_b, log_pdf_b = compute_pit(bank_test[b], probs, Y_test[b], eps)
        z_test[b] = z_b
        log_pdf_test[b] = log_pdf_b

    return {"z_train": z_train, "z_test": z_test, "log_pdf_test": log_pdf_test}
