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
"""

from __future__ import annotations

import contextlib
import os

import numpy as np

__all__ = ["BACKEND_NAMES", "make_regressor", "quantiles", "loo_pit"]

BACKEND_NAMES = ["tabicl", "tabpfn", "exaone"]


# ---------------------------------------------------------------------------
# Regressor construction — one instance reused across every fold/task, same
# rationale as eval/tabicl_utils.py::make_tabicl_regressor (avoid reloading
# backbone weights per .fit() call).
# ---------------------------------------------------------------------------
def make_regressor(name: str, device: "str | None" = None):
    if name == "tabicl":
        from eval.tabicl_utils import make_tabicl_regressor

        return make_tabicl_regressor(device=device)
    if name == "tabpfn":
        _require_tabpfn_token()
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
    unused (EXAONE's forward pass is deterministic given its fitted state)
    but kept for a uniform call signature across every backend's
    quantiles() dispatch.
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
