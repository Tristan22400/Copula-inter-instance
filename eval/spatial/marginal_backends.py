"""marginal_backends.py — pluggable tabular-foundation-model backends for
K-fold PIT z_train estimation, so the "z_train gap vs. ground truth"
comparison (eval/runners/compare_marginal_backbones.py) can swap TabICLv2
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
               pretrained. Point-prediction API ONLY (no native quantiles)
               -- approximated via homoscedastic split-conformal residual
               std per K-fold (see _exaone_quantiles): a held-out
               calibration slice of each fold's context estimates one
               constant sigma, reused for every query point in that fold.
  - "tabm"   : Yandex Research TabM (pip `tabm`) -- NOT a pretrained
               foundation model (no downloadable weights, see its own
               README); trained from scratch on each fold's context with a
               short full-batch Adam loop, then its k parallel ensemble
               heads' raw (unaveraged) predictions give a heteroscedastic
               empirical quantile grid directly (see _tabm_quantiles) --
               no distributional assumption needed, unlike "exaone".
"""

from __future__ import annotations

import os

import numpy as np

__all__ = ["BACKEND_NAMES", "make_regressor", "quantiles", "loo_pit"]

BACKEND_NAMES = ["tabicl", "tabpfn", "exaone", "tabm"]


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
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(device=device or "cpu")
    if name == "exaone":
        from exaonetabular import EXAONETabularRegressor

        # Always CPU, regardless of the requested device: exaonetabular's own
        # attention.py hardcodes a single SDPA backend per call (flash or
        # mem-efficient, chosen by _select_sdpa_backend) with NO math
        # fallback -- on GPUs below sm80 (e.g. an older Titan RTX/sm75 node
        # this shared cluster can reassign a job to mid-run) neither kernel
        # is available and F.scaled_dot_product_attention raises "No
        # available kernel" outright. Off-CUDA, _select_sdpa_backend always
        # returns FLASH_ATTENTION, which PyTorch's CPU build always has, so
        # CPU is the only device this library is guaranteed to run on
        # everywhere -- and fine here since fold contexts are ~20-30 rows.
        return EXAONETabularRegressor.from_pretrained(device="cpu")
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
    if name == "tabm":
        return _tabm_quantiles(X_context, y_context, X_query, probs, seed=seed)
    raise ValueError(f"Unknown marginal backend '{name}', choose from {BACKEND_NAMES}.")


def _exaone_quantiles(
    regressor, X_context: np.ndarray, y_context: np.ndarray, X_query: np.ndarray,
    probs: np.ndarray, *, seed: int, calib_frac: float = 0.3, min_calib: int = 3,
) -> np.ndarray:
    """Homoscedastic split-conformal quantile grid: EXAONETabularRegressor
    exposes .predict() as a point estimate only (no quantile/distribution
    API, confirmed against exaonetabular 1.0.0's public surface), so
    uncertainty has to come from held-out residuals instead. Splits off a
    calibration slice of the context, fits on the remainder, and uses the
    calibration residuals' std as one constant sigma for every X_query point
    in this fold -- a real approximation (no heteroscedasticity across
    X_query), acceptable here because the point of this comparison is
    exposing exactly this kind of backend-specific quality gap, not hiding
    it behind a more sophisticated calibration scheme.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    n = len(y_context)
    n_calib = max(min_calib, int(round(calib_frac * n)))
    n_calib = min(n_calib, n - 1) if n > 1 else 0

    if n_calib == 0:
        regressor.fit(X_context, y_context)
        mu = np.asarray(regressor.predict(X_query))
        sigma = max(float(y_context.std()), 1e-6)
    else:
        perm = rng.permutation(n)
        calib_idx, fit_idx = perm[:n_calib], perm[n_calib:]
        regressor.fit(X_context[fit_idx], y_context[fit_idx])
        calib_pred = np.asarray(regressor.predict(X_context[calib_idx]))
        sigma = max(float((y_context[calib_idx] - calib_pred).std()), 1e-6)
        # Refit on the full fold context so the query prediction itself
        # isn't left worse-off than it needs to be by the calibration split.
        regressor.fit(X_context, y_context)
        mu = np.asarray(regressor.predict(X_query))

    z = norm.ppf(np.clip(probs, 1e-6, 1.0 - 1e-6))
    return mu[:, None] + sigma * z[None, :]


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
