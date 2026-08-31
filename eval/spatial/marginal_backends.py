"""marginal_backends.py — pluggable tabular-foundation-model backends for
K-fold PIT z_train estimation, so the "z_train gap vs. ground truth"
comparison (debug/stages/s7_backbone.py) can swap TabICLv2
for other tabular foundation models without touching the real-ERA5
pipeline (eval/spatial/diagnostics.py::extract_model_context_correlation,
sweep_core.py), which stays on TabICL only.

Every backend implements the same
    quantiles(regressor, X_context, y_context, X_query, probs) -> (n_query, Q)
contract already established by eval/tabicl_utils.py::tabicl_quantiles, so
they all plug into the same generic K-fold loop (``loo_pit`` below) and the
same eval/metrics/joint_nll.py::compute_pit finite-difference PIT recipe
downstream — no per-backend PIT math.

Registered backends:
  - "tabicl" : TabICL v2.0.3, pretrained (existing eval/tabicl_utils.py,
               unchanged). Native quantile output.
  - "tabpfn" : PriorLabs TabPFN v3 (pip `tabpfn`), pretrained. Native
               quantile output. Gated: requires a one-time license
               acceptance at https://ux.priorlabs.ai and a `TABPFN_TOKEN`
               env var (see _require_tabpfn_token) -- there is no
               programmatic way around this, it's PriorLabs' own license
               gate, not a bug here.
  - "exaone" : LG AI Research EXAONE-Tabular (pip `exaonetabular`),
               pretrained. `.predict()`'s PUBLIC surface is point-estimate
               only, but the model itself has a genuine native quantile head
               underneath: every forward pass produces a
               (ensemble_count, n_query, quantile_count) bank
               (quantile_count=999, evenly spaced -- see LG's own model
               card), which regressor.py::_collapse_members immediately
               reduces to one scalar (median or trimmed-mean) per row before
               .predict() ever returns. _exaone_quantiles below recovers
               that bank instead of approximating a distribution from
               residuals -- see its docstring for how (a monkeypatch on
               _collapse_members, not a private reimplementation of the
               forward pass).
  - "tabfm"  : Google Research TabFM (pip `tabfm[pytorch]`), pretrained,
               released 2026-06-30. Unlike "exaone", genuinely has NO native
               quantile function for regression: confirmed against tabfm
               1.0.1's own source (classifier_and_regressor.py::
               _check_regressor_output_dim asserts the model's output last
               axis is exactly 1 -- "produces scalar regression outputs" --
               immediately squeezed away). What it does have is a real
               32-member ensemble (`n_estimators`) whose UNAVERAGED
               per-member point predictions are exposed via the private
               `_predict_internal` (the same call `.predict()` itself
               makes before averaging over members) -- _tabfm_quantiles
               below takes the empirical quantile grid over that ensemble
               axis, the same idea "tabm" uses below (real per-query
               disagreement across members), not an invented constant-sigma
               spread. CPU by default (small fold contexts here, ~20-30
               rows; not forced like exaone's SDPA constraint below, just
               untested on GPU).
  - "tabm"   : Yandex Research TabM (pip `tabm`) -- NOT a pretrained
               foundation model (no downloadable weights, see its own
               README); trained from scratch on each fold's context with a
               short full-batch Adam loop, then its k parallel ensemble
               heads' raw (unaveraged) predictions give a heteroscedastic
               empirical quantile grid directly (see _tabm_quantiles) --
               no distributional assumption needed, unlike "exaone".
"""

from __future__ import annotations

import contextlib
import os

import numpy as np

__all__ = ["BACKEND_NAMES", "make_regressor", "quantiles", "loo_pit"]

BACKEND_NAMES = ["tabicl", "tabpfn", "exaone", "tabfm", "tabm"]


# ---------------------------------------------------------------------------
# Regressor construction — one instance reused across every fold/task, same
# rationale as eval/tabicl_utils.py::make_tabicl_regressor (avoid reloading
# backbone weights per .fit() call). "tabm" returns None: it has no
# pretrained weights to reuse, a fresh model is trained per quantiles() call.
# ---------------------------------------------------------------------------
def make_regressor(name: str, device: "str | None" = None):
    if name == "tabicl":
        from eval.tabicl_utils import make_tabicl_regressor

        return make_tabicl_regressor(device=device)
    if name == "tabpfn":
        _require_tabpfn_token()
        # tabpfn 8.3.0's own model_loading.load_model() keeps an in-memory
        # LRU of *built* models (architecture + loaded state dict), keyed by
        # checkpoint path+identity, but only consults it when
        # TABPFN_MODEL_CACHE_SIZE > 0 (env-gated, defaults to 0 = off) AND
        # cache_trainset_representation is False -- true for every fit_mode
        # we use ("fit_preprocessors", set by predict_batched and by
        # loo_pit's per-episode path alike). Without this, EVERY .fit() call
        # rebuilds the whole transformer from scratch (kaiming/uniform-init
        # ~2500 Linear layers, then immediately overwrites them via
        # load_state_dict) before running a single forward pass -- profiled
        # at ~660ms of a ~700ms .fit() call, i.e. the rebuild *is* the cost,
        # not preprocessing or the model forward. Setting this once (as
        # setdefault, so an operator's own value always wins) cut measured
        # batched-PIT throughput from ~5.0s/episode to ~1.0s/episode
        # (B=16,P=32,N=16,K=5, RTX A5000) -- pure caching, same weights,
        # bit-for-bit identical predictions, verified against
        # tests/test_tabpfn_batched.py. Size 2 is headroom, not a
        # requirement: this process only ever resolves one model_path
        # ("auto"), and the cache holds a reference to the already-loaded
        # nn.Module (no extra GPU memory per cache slot), not a copy.
        os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "2")
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(device=device or "cpu")
    if name == "exaone":
        from exaonetabular import EXAONETabularRegressor

        # exaonetabular's own attention.py hardcodes a single SDPA backend
        # per call (flash or mem-efficient, chosen by _select_sdpa_backend)
        # with NO math fallback -- on a CUDA device below sm80 (e.g. an
        # older Titan RTX/sm75 node this shared cluster can reassign a job
        # to mid-run) neither kernel is available and
        # F.scaled_dot_product_attention raises "No available kernel"
        # outright. Measured on an actual sm86 GPU (RTX A5000): CUDA is
        # ~120x faster than CPU (1.8s vs 219s for one fit+predict call, tiny
        # ~30-row context) -- CPU-only was leaving two orders of magnitude
        # on the table, not a conservative-but-harmless default. Falls back
        # to CPU (PyTorch's CPU build always has FLASH_ATTENTION available,
        # see _select_sdpa_backend) only when CUDA is unavailable or the
        # actual device is below sm80.
        import torch

        use_cuda = (
            (device or "").startswith("cuda")
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(device)[0] >= 8
        )
        return EXAONETabularRegressor.from_pretrained(device="cuda" if use_cuda else "cpu")
    if name == "tabfm":
        from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

        # tabfm_v1_0_0.load's own `device` kwarg defaults to "cpu" (see its
        # docstring); no sm80 constraint here like exaone above (a plain
        # torch.nn.Module forward, no hardcoded SDPA backend selection), so
        # just pass the requested device straight through when CUDA is
        # actually available.
        import torch

        load_device = device if (device and torch.cuda.is_available()) else "cpu"
        return TabFMRegressor(model=tabfm_v1_0_0.load(model_type="regression", device=load_device))
    if name == "tabm":
        return None
    raise ValueError(f"Unknown marginal backend '{name}', choose from {BACKEND_NAMES}.")


def _require_tabpfn_token() -> None:
    if not os.environ.get("TABPFN_TOKEN"):
        raise RuntimeError(
            "TabPFN v3 requires a one-time license acceptance: open "
            "https://ux.priorlabs.ai, log in, accept the license, copy your "
            "API key from the account page, then `export TABPFN_TOKEN=...` "
            "before running this backend."
        )


# ---------------------------------------------------------------------------
# quantiles(...) — the shared (X_context, y_context, X_query, probs) ->
# quantile_grid contract every K-fold loop below (and compute_pit downstream)
# expects, in RAW y-units.
# ---------------------------------------------------------------------------
def quantiles(
    name: str, regressor, X_context: np.ndarray, y_context: np.ndarray,
    X_query: np.ndarray, probs: np.ndarray, *, seed: int = 0,
) -> np.ndarray:
    if name == "tabicl":
        from eval.tabicl_utils import tabicl_quantiles

        return tabicl_quantiles(regressor, X_context, y_context, X_query, probs)
    if name == "tabpfn":
        regressor.fit(X_context, y_context)
        out = regressor.predict(X_query, output_type="quantiles", quantiles=list(probs))
        return np.asarray(out).T  # (n_quantiles, n_query) -> (n_query, n_quantiles)
    if name == "exaone":
        return _exaone_quantiles(regressor, X_context, y_context, X_query, probs, seed=seed)
    if name == "tabfm":
        return _tabfm_quantiles(regressor, X_context, y_context, X_query, probs, seed=seed)
    if name == "tabm":
        return _tabm_quantiles(X_context, y_context, X_query, probs, seed=seed)
    raise ValueError(f"Unknown marginal backend '{name}', choose from {BACKEND_NAMES}.")


@contextlib.contextmanager
def _exaone_capture_quantile_bank():
    """Temporarily disables EXAONETabularRegressor._collapse_members'
    reduction to a single point estimate, so .predict() returns the full
    (n_query, quantile_count) bank instead of one number per row.

    This is the only place regressor.py throws the model's real
    (ensemble_count, n_query, quantile_count) forward output away in favor
    of a single trimmed-mean/median scalar per row (see its docstring) --
    every other real step .fit()/.predict() run (preprocessing, SVD/ensemble
    passes, de-standardization, member weighting) is left exactly as
    production runs it; only that one reduction is skipped, replaced by a
    per-member sort (guards tau-crossing, same as the real "trimmed" branch)
    so the bank stays a valid quantile function per member before
    .predict()'s own member-averaging combines them.

    NOT valid when EXAONE's NNLS member-weighting is active: predict()'s
    weighted-combine step assumes a 2D (members, rows) tensor and would
    silently mis-broadcast against the 3D (members, rows, quantiles) bank
    this produces instead. Guarded by a RuntimeError in _exaone_quantiles
    below rather than raised here, since it only matters above
    nnls_min_validation_rows=2000 support rows -- never true for the small
    (~20-30 row) K-fold contexts this pipeline uses.
    """
    import torch
    from exaonetabular.regressor import EXAONETabularRegressor

    original = EXAONETabularRegressor._collapse_members

    def _passthrough(self, output, query_count):
        expected = (self.manifest.runtime.ensemble_count, query_count, self.manifest.output_width)
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != expected or not bool(torch.isfinite(output).all()):
            raise RuntimeError("model returned invalid regression quantiles")
        return torch.sort(output.float(), dim=-1).values

    EXAONETabularRegressor._collapse_members = _passthrough
    try:
        yield
    finally:
        EXAONETabularRegressor._collapse_members = original


def _exaone_quantiles(
    regressor, X_context: np.ndarray, y_context: np.ndarray, X_query: np.ndarray,
    probs: np.ndarray, *, seed: int,
) -> np.ndarray:
    """EXAONETabularRegressor's REAL native quantile grid (999 evenly spaced
    levels, fixed by the released checkpoint -- not the caller's `probs`),
    recovered via _exaone_capture_quantile_bank above and linearly
    interpolated onto whatever `probs` the caller asked for. `seed` is
    unused (EXAONE's forward pass is deterministic given its fitted state,
    unlike tabm's from-scratch training loop below) but kept for a uniform
    call signature across every backend's quantiles() dispatch.
    """
    regressor.fit(X_context, y_context)
    if regressor._fitted_state.get("member_weights") is not None:
        raise RuntimeError(
            "EXAONE NNLS member-weighting is active; native quantile capture "
            "assumes uniform member averaging (see _exaone_capture_quantile_bank)."
        )
    quantile_count = regressor.manifest.regression.quantile_count
    native_probs = np.linspace(1.0 / (quantile_count + 1), quantile_count / (quantile_count + 1), quantile_count)
    with _exaone_capture_quantile_bank():
        bank = np.asarray(regressor.predict(X_query))  # (n_query, quantile_count), raw y-units

    out = np.empty((bank.shape[0], len(probs)))
    for i in range(bank.shape[0]):
        out[i] = np.interp(probs, native_probs, bank[i])
    return out


def _tabfm_quantiles(
    regressor, X_context: np.ndarray, y_context: np.ndarray, X_query: np.ndarray,
    probs: np.ndarray, *, seed: int,
) -> np.ndarray:
    """Empirical quantile grid from TabFMRegressor's own ensemble members
    (default n_estimators=32), the same idea _tabm_quantiles below uses --
    NOT split-conformal, because unlike "exaone" there is no per-row quantile
    function to recover: confirmed against tabfm 1.0.1's own source
    (classifier_and_regressor.py::_check_regressor_output_dim asserts the
    model's regression output is a single scalar per row, per member --
    "produces scalar regression outputs"). What TabFM DOES have is real
    cross-member disagreement: `_predict_internal` (the same call
    `.predict()` itself makes, before averaging) exposes each of the 32
    members' UNAVERAGED point prediction. Each member is inverse-transformed
    individually (mirroring `_combine_predictions`'s own NNLS branch) before
    taking the empirical quantile over the member axis -- heteroscedastic
    across X_query for free, unlike a constant-sigma approximation.
    `seed` is unused (TabFM's ensemble diversity comes from its own fixed
    per-member preprocessing views, not a call-time RNG) but kept for a
    uniform call signature across every backend's quantiles() dispatch.
    """
    regressor.fit(X_context, y_context)
    member_preds_scaled = np.asarray(regressor._predict_internal(X_query))  # (n_estimators, n_query), scaled
    member_preds = np.stack(
        [regressor._inverse_transform_y(row) for row in member_preds_scaled], axis=0
    )  # (n_estimators, n_query), raw y-units
    return np.quantile(member_preds, probs, axis=0).T  # (n_query, Q)


def _tabm_quantiles(
    X_context: np.ndarray, y_context: np.ndarray, X_query: np.ndarray, probs: np.ndarray,
    *, seed: int, k: int = 16, n_steps: int = 200, lr: float = 1e-2,
    d_block: int = 32, n_blocks: int = 1,
) -> np.ndarray:
    """TabM has no pretrained weights (confirmed against its own README: "If
    you need zero-shot tabular capabilities ... TabM is not the right
    tool"), so a fresh k-head ensemble is trained on THIS fold's context
    alone via a short full-batch Adam loop, then every one of its k
    unaveraged member predictions at X_query becomes one empirical quantile
    sample -- np.quantile over the k axis, no Gaussian assumption, unlike
    "exaone" above (which has no ensemble to draw from). Deliberately small
    (k=16, n_steps=200, d_block=32, n_blocks=1): TabM.make()'s OWN defaults
    (d_block=512, n_blocks=3) are sized for real datasets with thousands of
    rows -- at those widths, 300 full-batch steps on a ~25-point fold
    context measured ~190s/call (see PR discussion), ~100x more than
    exaone/tabpfn's native-quantile calls on the same data. Shrinking the
    backbone to match the data scale isn't just a speed fix: a 512-wide
    3-block MLP ensemble on 19-27 points would be absurdly overparameterized
    regardless of runtime. A systematically underdispersed ensemble on this
    little data is itself part of the gap this comparison is meant to
    surface, not something to engineer away by over-provisioning capacity.
    """
    import torch
    from tabm import TabM

    torch.manual_seed(seed)
    y_mean, y_std = float(y_context.mean()), max(float(y_context.std()), 1e-6)
    Xt = torch.as_tensor(X_context, dtype=torch.float32)
    yt = torch.as_tensor((y_context - y_mean) / y_std, dtype=torch.float32)

    model = TabM.make(n_num_features=X_context.shape[1], d_out=1, k=k, d_block=d_block, n_blocks=n_blocks)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(n_steps):
        opt.zero_grad()
        pred = model(Xt)  # (n_context, k, 1)
        loss = ((pred.squeeze(-1) - yt.unsqueeze(1)) ** 2).mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        out = model(torch.as_tensor(X_query, dtype=torch.float32))  # (n_query, k, 1)
    ensemble = out.squeeze(-1).numpy() * y_std + y_mean  # (n_query, k), raw y-units
    return np.quantile(ensemble, probs, axis=1).T  # (n_query, Q)


# ---------------------------------------------------------------------------
# Generic K-fold leave-fold-out PIT — the fold-splitting/PIT recipe itself
# lives in eval/metrics/joint_nll.py::kfold_loo_pit (shared with
# eval/tabicl_utils.py::tabicl_loo_pit); this just plugs the quantiles()
# dispatch above in as the per-fold callback instead of being hardcoded to
# TabICL.
# ---------------------------------------------------------------------------
def loo_pit(
    name: str, regressor, X_train: np.ndarray, y_train: np.ndarray, probs: np.ndarray,
    k_folds: int = 10, eps: float = 1e-6, seed: int = 0,
) -> np.ndarray:
    from eval.metrics.joint_nll import kfold_loo_pit

    return kfold_loo_pit(
        lambda Xc, yc, Xq, k: quantiles(name, regressor, Xc, yc, Xq, probs, seed=seed * 1000 + k),
        X_train, y_train, probs, k_folds=k_folds, eps=eps, seed=seed,
    )
