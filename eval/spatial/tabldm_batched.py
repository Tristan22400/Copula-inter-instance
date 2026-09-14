"""tabldm_batched.py — genuine multi-episode batched PIT for Xiaomi-TabLDM,
the "batch mode" analogue of pit.py::run_pit_batched for
data.z_train_source=tabldm. Unlike marginal_backends.py's per-episode
_tabldm_quantiles (still used for single ad-hoc calls and for the debug
S7a/S7b comparison stages), this batches the one genuinely expensive step --
the model forward -- across every episode in a shard-generation call, the
same trick TabICL's own run_pit_batched uses.

WHY THIS IS SAFE (read off TabLDM's own source and docstrings, not assumed):
TabLDM's forward entry point is TabLDM.predict_stats(X, y_train, ...), whose
OWN docstring defines its leading axis as a multi-table axis, verbatim:

    X : Tensor
        Input tensor of shape (B, T, H) where:
         - B is the number of tables
         - T is the number of samples (rows)
         - H is the number of features (columns)
        The first train_size positions contain training samples, and the
        remaining positions contain test samples.

    Output shapes:
      - "quantiles": (B, test_size, len(alphas))

So this module is NOT reinterpreting some internal ensemble axis as a batch
axis (which is what exaone_batched.py has to argue at length, since EXAONE's
equivalent axis nominally indexes ensemble-member views of ONE dataset).
Stacking B independent episodes there is the axis's documented purpose. The
regressor-level wrapper agrees: TabLDMRegressor._batch_forward np.array_
split()s its input along axis 0 into self.batch_size_-sized chunks and calls
predict_stats on each chunk independently, with no cross-item computation --
i.e. it already treats that axis as a VRAM-chunkable batch of unrelated
tables.

What episodes must share to be stacked: T (context rows + query rows), H
(feature count), and train_size after preprocessing. Even equal-width raw
inputs can lose different constant columns in TabLDM's per-episode feature
filter. Episodes are therefore grouped by their preprocessed shapes, with
one fused forward per group and results restored to the original order.

REUSE, NOT REIMPLEMENTATION: every step that isn't the fused forward goes
through the regressor's own real code, in the exact order its own predict()
runs them (see TabLDMRegressor.predict's non-enhanced path) --
  - regressor.fit(...) per episode (encoder fit, y scaling, normalization/
    feature-selection ensemble construction) is called UNMODIFIED; this
    module never re-derives any of it. It is also cheap next to the forward
    for the small GP-episode contexts here, so looping it per episode costs
    nothing worth batching away.
  - sklearn_utils.validate_data(...) + regressor.X_encoder_.transform(...)
    on the query features -- UNMODIFIED, same two calls predict() makes.
  - regressor.ensemble_generator_.transform(X, mode="both") -- UNMODIFIED;
    this is what produces the (n_members, T, H) / (n_members, train_size)
    pair predict() forwards, with the context rows already prepended.
  - regressor._batch_forward(...) -- UNMODIFIED, the same method predict()
    calls, just with a leading axis spanning multiple episodes' members
    instead of one episode's.
  - regressor.y_scaler_.inverse_transform(...) then a mean over the member
    axis -- UNMODIFIED, predict()'s own de-standardization/pooling, applied
    per episode with THAT episode's own scaler (each fit() refits it).
The only new code here is the orchestration: capture each episode's member
batch, group compatible shapes, forward, split the result back per episode.

NOT SUPPORTED, and rejected loudly rather than silently mis-fused:
enhance_candidates=True (predict()'s "enhanced path" only supports
output_type="mean", so it has no quantiles to give us) and a fitted KV cache
(predict() then takes a different, cache-keyed branch). marginal_backends.py
::make_regressor("tabldm") constructs neither, so these guards are
defence-in-depth against a caller passing its own regressor.

Verified, not just argued: tests/test_tabldm_batched.py checks this module's
output matches marginal_backends.py's per-episode path to float32 precision
on real (non-mocked) episodes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["tabldm_run_pit_batched"]


def _episode_member_batch(regressor, x_support: np.ndarray, y_support: np.ndarray, x_query: np.ndarray):
    """Fit ``regressor`` on one episode's context and return that episode's
    (members, T, H) / (members, train_size) forward inputs plus the y-scaler
    fitted for it.

    Mirrors TabLDMRegressor.predict's non-enhanced, no-KV-cache path up to
    (but not including) its _batch_forward call -- see module docstring for
    the call-by-call correspondence.
    """
    from tabldm._sklearn.sklearn_utils import validate_data

    if not getattr(regressor, "_load_model_cached", False):
        orig_load = regressor._load_model

        def _cached_load():
            if getattr(regressor, "model_", None) is None:
                orig_load()

        regressor._load_model = _cached_load
        regressor._load_model_cached = True

    regressor.fit(x_support, y_support)

    if getattr(regressor, "enhance_candidates", False):
        raise RuntimeError(
            "tabldm_batched does not support enhance_candidates=True: TabLDM's "
            "enhanced predict path only supports output_type='mean' (see "
            "TabLDMRegressor.predict), so it exposes no quantiles to batch."
        )
    if getattr(regressor, "model_kv_cache_", None) is not None:
        raise RuntimeError(
            "tabldm_batched does not support a fitted KV cache: TabLDM's predict "
            "takes a separate cache-keyed branch (_batch_forward_with_cache) that "
            "this module does not mirror. Leave TabLDM's kv_cache at its default "
            "(off), as marginal_backends.py::make_regressor does."
        )

    xq = validate_data(regressor, x_query, reset=False, dtype=None, skip_check_array=True)
    xq = regressor.X_encoder_.transform(xq)
    data = regressor.ensemble_generator_.transform(xq, mode="both")

    # predict() concatenates its per-normalization-group outputs along the
    # member axis in dict order and means over the result; concatenating the
    # INPUTS in that same order and meaning over the same axis below is the
    # identical reduction, just fused.
    xs = np.concatenate([g[0] for g in data.values()], axis=0)  # (members, T, H)
    ys = np.concatenate([g[1] for g in data.values()], axis=0)  # (members, train_size)
    return xs, ys, regressor.y_scaler_


def _group_episode_batches(per_episode):
    """Group compatible preprocessed inputs without padding extra features."""
    groups = {}
    for b, (xs, ys, _) in enumerate(per_episode):
        groups.setdefault((xs.shape, ys.shape), []).append(b)
    return list(groups.values())


def _quantile_bank_batched(
    regressor, X_context: list, y_context: list, X_query: list, probs: np.ndarray,
) -> np.ndarray:
    """One fused _batch_forward per compatible preprocessed shape.

    Returns (B, n_query, len(probs)) in RAW y-units, already on the caller's
    own ``probs`` grid -- TabLDM evaluates the requested alphas inside its own
    spline/GPD quantile distribution, so unlike exaone_batched.py there is no
    fixed native grid to interpolate off.
    """
    B = len(X_context)
    per_episode = [
        _episode_member_batch(regressor, X_context[b], y_context[b], X_query[b]) for b in range(B)
    ]
    banks = [None] * B
    for indices in _group_episode_batches(per_episode):
        members = per_episode[indices[0]][0].shape[0]
        xs = np.concatenate([per_episode[b][0] for b in indices], axis=0)
        ys = np.concatenate([per_episode[b][1] for b in indices], axis=0)
        out = regressor._batch_forward(xs, ys, output_type="quantiles", alphas=list(probs))
        out = np.asarray(out).reshape(len(indices), members, -1, len(probs))
        for local, b in enumerate(indices):
            arr = per_episode[b][2].inverse_transform(out[local].reshape(-1, 1)).reshape(out[local].shape)
            banks[b] = arr.mean(axis=0)
    bank = np.stack(banks).astype(np.float64)
    return bank


def tabldm_run_pit_batched(
    regressor, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray,
    k_folds: int = 10, probs_n: int = 99, eps: float = 1e-6, seed: int = 0,
) -> dict:
    """``run_pit_batched``, TabLDM version. Signature, y-unit convention and
    return dict are identical to exaone_batched.py/tabpfn_batched.py's -- the
    whole K-fold driver is shared (_batched_pit.py::run_kfold_pit_batched);
    only the bank above is backend-specific.
    """
    from eval.spatial._batched_pit import run_kfold_pit_batched

    return run_kfold_pit_batched(
        lambda X_ctx, y_ctx, X_qry, probs: _quantile_bank_batched(
            regressor, X_ctx, y_ctx, X_qry, probs
        ),
        X_train, Y_train, X_test, Y_test,
        k_folds=k_folds, probs_n=probs_n, eps=eps, seed=seed,
    )
