"""exaone_batched.py — genuine multi-episode batched PIT for EXAONE-Tabular,
the "batch mode" analogue of pit.py::run_pit_batched for
data.z_train_source=exaone. Unlike marginal_backends.py's per-episode
_exaone_quantiles (still used for tabpfn/tabfm/tabm and for a single ad-hoc
call), this batches the one genuinely expensive step -- the model forward --
across every episode in a shard-generation call, the same trick TabICL's own
run_pit_batched uses.

WHY THIS IS SAFE (found by reading exaonetabular's own source, not assumed):
EXAONETabularRegressor.predict() already runs an internal "ensemble" of
manifest.runtime.ensemble_count x (1 or 2 svd passes) member views of ONE
fitted dataset, all stacked along dim 0 of a
(members, rows, features)-shaped tensor and passed through
_InferenceExecutor.forward(model, support_batch, label_batch, query_batch)
in one call. That executor (exaonetabular/_execution.py::forward /
_run_execution_plan) treats dim 0 purely as a VRAM-chunkable batch axis --
it slices ensemble_start:ensemble_stop and calls the underlying
ClassificationModel/RegressionModel forward on each chunk independently, with
no cross-item computation. Nothing about that requires every member to be a
view of the SAME dataset -- concatenating DIFFERENT episodes' member batches
along that same axis (support_rows/query_rows/feature_count are guaranteed
identical across every episode in one data_gen.py generate_gp_batch call, see
its module docstring) is processed identically to calling the executor once
per episode, just in fewer, larger calls. Confirmed empirically, not just
argued: see tests/test_exaone_batched.py, which checks this module's batched
output matches marginal_backends.py's per-episode path to float32 precision
on real (non-mocked) episodes.

REUSE, NOT REIMPLEMENTATION: every step that isn't the forward call goes
through the regressor's own real code --
  - regressor.fit(...) per episode (feature selection, row subsampling,
    center/scale, n_svd/svd_gate/svd_split resolution, NNLS-weight fit) is
    called UNMODIFIED; this module never re-derives any of that logic. It's
    also cheap (small numpy/CPU work for our small GP-episode contexts), so
    looping it per episode costs nothing worth batching away.
  - state["preprocessor"].transform(...) (the fitted Gaussianization/
    quantile-map) is called UNMODIFIED per episode for the query features.
  - build_ensemble_inputs(...) (member permutation/SVD augmentation) is
    called UNMODIFIED, with the exact (n_svd, seed) pairs regressor.fit()
    already resolved into state["passes"] -- this module never re-implements
    _ensemble_passes' svd-gate decision.
  - regressor._executor().forward(...) is called UNMODIFIED -- the same
    function predict() calls, just with a batch spanning multiple episodes'
    members instead of one episode's.
The only new code here is the orchestration: build each episode's member
tensors via the real building blocks above, concatenate across episodes,
make ONE forward call, then split/rescale/pool back into per-episode
results.

NOT valid when any episode's NNLS member-weighting fires (state["member_
weights"] is not None) -- predict()'s weighted-combine step is per-episode
by construction (a fitted weight vector over ONE dataset's members), and
mixing weighted and batched-uniform pooling would silently give the wrong
answer. Guarded by a RuntimeError below, same as
marginal_backends.py::_exaone_quantiles -- only matters above
nnls_min_validation_rows=2000 support rows, never true for GP-episode
context sizes this pipeline uses.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["exaone_run_pit_batched"]


def _episode_member_batch(regressor, x_support: np.ndarray, y_support: np.ndarray, x_query: np.ndarray):
    """fit() + build_ensemble_inputs() for ONE episode, via the regressor's
    own real code (see module docstring) -- returns, per (n_svd, seed) pass,
    the (support_batch, label_batch, query_batch) tensors ready to concat
    across episodes, plus this episode's (center, scale) for de-standardizing
    the pooled output later. Raises if this episode's NNLS weighting fired
    (see module docstring)."""
    from exaonetabular.ensemble import EnsemblePlan, build_ensemble_inputs

    regressor.fit(x_support, y_support)
    state = regressor._fitted_state
    if state["member_weights"] is not None:
        raise RuntimeError(
            "EXAONE NNLS member-weighting is active for this episode; batched "
            "uniform-mean pooling (exaone_batched.py) assumes every episode "
            "pools its members uniformly, same restriction as "
            "marginal_backends.py::_exaone_quantiles."
        )
    device = regressor.device
    support_x = torch.as_tensor(state["support_x"], dtype=torch.float32, device=device)
    support_y = torch.as_tensor(state["support_y"], dtype=torch.float32, device=device)
    query_np = state["preprocessor"].transform(x_query).values
    query_x = torch.as_tensor(query_np, dtype=torch.float32, device=device)

    passes = []
    for n_svd, seed in state["passes"]:
        plan = EnsemblePlan(members=regressor.manifest.runtime.ensemble_count, seed=seed, task="regression", n_svd=n_svd)
        batch_xs, batch_y, batch_xq, _fitted_plan = build_ensemble_inputs(support_x, support_y, query_x, plan)
        passes.append((batch_xs, batch_y, batch_xq))
    return passes, float(state["center"]), float(state["scale"])


def _quantile_bank_batched(regressor, X_context: list, y_context: list, X_query: list) -> np.ndarray:
    """Batched EXAONE quantile bank for B episodes sharing support_rows/
    query_rows/feature_count (guaranteed by data_gen.py's per-call
    homogeneity -- see this module's docstring). Returns
    (B, query_rows, quantile_count) in RAW y-units, uniformly mean-pooled
    over every ensemble member/pass -- the same pooling predict() does when
    member_weights is None.
    """
    B = len(X_context)
    per_episode = [
        _episode_member_batch(regressor, X_context[b], y_context[b], X_query[b]) for b in range(B)
    ]
    n_passes = len(per_episode[0][0])
    query_rows = X_query[0].shape[0]

    pass_outputs = []
    for p in range(n_passes):
        support_batch = torch.cat([per_episode[b][0][p][0] for b in range(B)], dim=0)
        label_batch = torch.cat([per_episode[b][0][p][1] for b in range(B)], dim=0)
        query_batch = torch.cat([per_episode[b][0][p][2] for b in range(B)], dim=0)
        members_per_episode = per_episode[0][0][p][0].shape[0]

        # eval() + inference_mode(): same as _member_points' own forward call
        # (regressor.py) -- without inference_mode the output tensor stays
        # grad-tracked (build_ensemble_inputs' tensors are plain floats, not
        # leaves under no_grad, so autograd would otherwise record the whole
        # forward for nothing) and .numpy() below fails.
        regressor.model.eval()
        with torch.inference_mode():
            raw = regressor._forward_chunked(support_batch, label_batch, query_batch)  # (B*members, query_rows, Q)
        expected = (B * members_per_episode, query_rows, regressor.manifest.output_width)
        if tuple(raw.shape) != expected or not bool(torch.isfinite(raw).all()):
            raise RuntimeError("exaone_batched: model returned invalid regression quantiles")
        pass_outputs.append(raw.float().reshape(B, members_per_episode, query_rows, -1))

    # torch.sort guards tau-crossing per member, mirroring
    # _exaone_capture_quantile_bank's per-member sort in marginal_backends.py
    # -- predict()'s own point-estimate path sorts too (_collapse_members'
    # "trimmed" branch), just after the reduction instead of before.
    pooled = torch.cat(pass_outputs, dim=1)              # (B, total_members, query_rows, Q)
    pooled = torch.sort(pooled, dim=-1).values.mean(dim=1)  # (B, query_rows, Q)

    center = torch.tensor([per_episode[b][1] for b in range(B)], device=pooled.device).view(B, 1, 1)
    scale = torch.tensor([per_episode[b][2] for b in range(B)], device=pooled.device).view(B, 1, 1)
    return (pooled * scale + center).cpu().numpy()


def exaone_run_pit_batched(
    regressor, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray,
    k_folds: int = 10, probs_n: int = 99, eps: float = 1e-6, seed: int = 0,
) -> dict:
    """``run_pit_batched``, EXAONE version -- K-fold PIT for z_train AND a
    single held-in-context pass for z_test/log_pdf_test, batched across every
    episode in the call (see module docstring for why this is safe).

    Args:
        X_train: (B, P, p_x)   X_test: (B, N, p_x)
        Y_train: (B, P)        Y_test: (B, N)      -- already y-scaled by the
            caller (data_gen.py z-scores y_train/y_test per episode before
            calling this, same convention as pit.py::run_pit_batched).
        k_folds: clamped into [1, P] (matching eval/metrics/joint_nll.py::
            kfold_loo_pit's own `min(k_folds, n)`, NOT pit.py::run_pit_
            batched's `max(2, ...)` floor), shared across the batch since P
            is -- see below for why fold SIZE is guaranteed equal across
            episodes even though fold MEMBERSHIP differs per episode.
        probs_n: quantile grid size EXAONE's native 999-level bank is
            interpolated onto (see marginal_backends.py::_exaone_quantiles).
        seed: per-episode fold assignment uses
            np.random.default_rng(seed + b).permutation(P) % K -- the exact
            recipe eval/metrics/joint_nll.py::kfold_loo_pit uses (bit-
            identical to marginal_backends.py::loo_pit's fold splits when
            called with matching per-episode seeds, e.g. data_gen.py's
            marginal_backend branch's `seed_b = (base_seed + b) %
            (2**31)`), NOT pit.py::run_pit_batched's shared contiguous-block
            split -- this backend's per-episode PIT path already committed
            to the random-permutation convention, and this module exists to
            batch it faster, not to change its semantics. Fold SIZE (not
            membership) only depends on P and K, both shared across the
            batch, so every episode's fold k has the same query-row count
            regardless of its own seed -- permutation preserves the multiset
            of residues {0..P-1} mod K, just reorders which original row
            index lands in which fold -- so batching per fold across
            episodes is still valid despite the differing seeds.

    Returns dict with z_train (B,P), z_test (B,N), log_pdf_test (B,N) --
    log_pdf_test is in the SAME (already-scaled) y-units as Y_test; callers
    apply their own Jacobian correction back to raw-y nats, matching every
    other backend's convention in this pipeline.
    """
    from eval.metrics.joint_nll import compute_pit

    B, P, _p_x = X_train.shape
    N = X_test.shape[1]
    K = min(int(k_folds), P)
    quantile_count = regressor.manifest.regression.quantile_count
    native_probs = np.linspace(1.0 / (quantile_count + 1), quantile_count / (quantile_count + 1), quantile_count)
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
        bank = _quantile_bank_batched(regressor, X_ctx, y_ctx, X_qry)  # (B, F, quantile_count)
        for b in range(B):
            F = qry_idx[b].size
            q_interp = np.empty((F, len(probs)))
            for i in range(F):
                q_interp[i] = np.interp(probs, native_probs, bank[b, i])
            z_held, _ = compute_pit(q_interp, probs, Y_train[b][qry_idx[b]], eps)
            z_train[b, qry_idx[b]] = z_held

    X_ctx_full = [X_train[b] for b in range(B)]
    y_ctx_full = [Y_train[b] for b in range(B)]
    X_qry_full = [X_test[b] for b in range(B)]
    bank_test = _quantile_bank_batched(regressor, X_ctx_full, y_ctx_full, X_qry_full)  # (B, N, quantile_count)
    z_test = np.empty((B, N), dtype=np.float32)
    log_pdf_test = np.empty((B, N), dtype=np.float32)
    for b in range(B):
        q_interp = np.empty((N, len(probs)))
        for i in range(N):
            q_interp[i] = np.interp(probs, native_probs, bank_test[b, i])
        z_b, log_pdf_b = compute_pit(q_interp, probs, Y_test[b], eps)
        z_test[b] = z_b
        log_pdf_test[b] = log_pdf_b

    return {"z_train": z_train, "z_test": z_test, "log_pdf_test": log_pdf_test}
