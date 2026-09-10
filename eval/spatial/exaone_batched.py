"""exaone_batched.py — genuine multi-episode batched PIT for EXAONE-Tabular,
the "batch mode" analogue of pit.py::run_pit_batched for
data.z_train_source=exaone. Unlike marginal_backends.py's per-episode
_exaone_quantiles (still used for tabpfn and for a single ad-hoc
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

import dataclasses
import logging
import numpy as np
import torch

__all__ = ["exaone_run_pit_batched"]

# Silence EXAONE's NNLS member-weighting fallback warning on small context sizes
logging.getLogger("exaonetabular.regressor").setLevel(logging.ERROR)


def _episode_member_batch(regressor, x_support: np.ndarray, y_support: np.ndarray, x_query: np.ndarray):
    """fit() + build_ensemble_inputs() for ONE episode, via the regressor's
    own real code (see module docstring) -- returns, per (n_svd, seed) pass,
    the (support_batch, label_batch, query_batch) tensors ready to concat
    across episodes, plus this episode's (center, scale) for de-standardizing
    the pooled output later. Raises if this episode's NNLS weighting fired
    (see module docstring)."""
    from exaonetabular.ensemble import EnsemblePlan, build_ensemble_inputs

    if getattr(getattr(regressor, "manifest", None), "regression", None) is not None:
        if regressor.manifest.regression.member_weighting != "uniform":
            regressor.manifest = dataclasses.replace(
                regressor.manifest,
                regression=dataclasses.replace(
                    regressor.manifest.regression, member_weighting="uniform"
                ),
            )

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


def _quantile_bank_on_probs(regressor, X_context: list, y_context: list, X_query: list,
                            probs: np.ndarray) -> np.ndarray:
    """_quantile_bank_batched, interpolated from EXAONE's fixed native grid
    onto the caller's ``probs``.

    EXAONE is the one backend whose model emits a grid it chose (999 evenly
    spaced levels, fixed by the released checkpoint) rather than the levels
    asked for, so the interpolation the shared driver must never have to know
    about lives here -- the same np.interp step
    marginal_backends.py::_exaone_quantiles does per episode.
    """
    quantile_count = regressor.manifest.regression.quantile_count
    native_probs = np.linspace(
        1.0 / (quantile_count + 1), quantile_count / (quantile_count + 1), quantile_count
    )
    bank = _quantile_bank_batched(regressor, X_context, y_context, X_query)  # (B, F, quantile_count)
    out = np.empty(bank.shape[:2] + (len(probs),), dtype=np.float64)
    for b in range(bank.shape[0]):
        for i in range(bank.shape[1]):
            out[b, i] = np.interp(probs, native_probs, bank[b, i])
    return out


def exaone_run_pit_batched(
    regressor, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray,
    k_folds: int = 10, probs_n: int = 99, eps: float = 1e-6, seed: int = 0,
) -> dict:
    """``run_pit_batched``, EXAONE version. Signature, y-unit convention and
    return dict are identical to tabpfn_batched.py/tabldm_batched.py's -- the
    whole K-fold driver is shared (_batched_pit.py::run_kfold_pit_batched,
    which also documents the fold-assignment recipe and the equal-fold-size
    guarantee batching relies on); only the bank above is backend-specific.

    probs_n is the grid EXAONE's native 999-level bank is interpolated onto
    (see _quantile_bank_on_probs).
    """
    from eval.spatial._batched_pit import run_kfold_pit_batched

    return run_kfold_pit_batched(
        lambda X_ctx, y_ctx, X_qry, probs: _quantile_bank_on_probs(
            regressor, X_ctx, y_ctx, X_qry, probs
        ),
        X_train, Y_train, X_test, Y_test,
        k_folds=k_folds, probs_n=probs_n, eps=eps, seed=seed,
    )
