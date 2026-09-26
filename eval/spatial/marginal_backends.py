"""Tabular foundation models as marginal backends for K-fold PIT.

Every backend implements quantiles(name, regressor, X_context, y_context,
X_query, probs) -> (n_query, Q) in raw y units, which loo_pit plugs into the
generic K-fold loop (eval/metrics/joint_nll.kfold_loo_pit).

    tabicl: TabICL v2 (native quantiles).
    tabpfn: PriorLabs TabPFN v3 (native quantiles); needs TABPFN_TOKEN after
        accepting the licence at https://ux.priorlabs.ai.
    exaone: LG EXAONE-Tabular; its native 999-level quantile bank is recovered
        by bypassing the member reduction in .predict().
    tabldm: Xiaomi-TabLDM; predict(output_type="quantiles", alphas=probs).
        Weights come from HF (occams/Xiaomi-TabLDM) on first use.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import types

import numpy as np
from copula_inter.backend_registry import BACKENDS, require_capability

__all__ = ["BACKEND_NAMES", "make_regressor", "quantiles", "loo_pit"]

# Silence EXAONE's NNLS member-weighting fallback warning on small context sizes
logging.getLogger("exaonetabular.regressor").setLevel(logging.ERROR)

BACKEND_NAMES = list(BACKENDS)


# Regressor construction (one instance reused across folds and tasks).
def make_regressor(name: str, device: "str | None" = None, ckpt: "str | None" = None):
    """Build a backend regressor, optionally loading a Phase-A checkpoint (MarginalBackbone.save format) into its trainable module."""
    require_capability(name, "name")
    regressor = _make_pretrained_regressor(name, device)
    if ckpt:
        _load_finetuned_weights(name, regressor, ckpt, device)
    return regressor


def _load_finetuned_weights(name: str, regressor, ckpt: str, device: "str | None") -> None:
    import torch

    from copula_inter.marginal_backbones import _trainable_module

    payload = torch.load(ckpt, map_location=device or "cpu", weights_only=False)
    written_for = payload.get("backbone")
    if written_for not in (None, name):
        raise ValueError(
            f"marginal checkpoint {ckpt} was written for backbone {written_for!r}, not {name!r}."
        )
    _trainable_module(name, regressor).load_state_dict(payload["state_dict"], strict=True)


def _make_pretrained_regressor(name: str, device: "str | None" = None):
    if name == "tabicl":
        from eval.tabicl_utils import make_tabicl_regressor

        return make_tabicl_regressor(device=device)
    if name == "tabpfn":
        _require_tabpfn_token()
        # Enable tabpfn's in-memory model cache so each .fit() doesn't rebuild the model.
        os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "2")
        from tabpfn import TabPFNRegressor

        # n_estimators=1: only one quantile grid is read, so ensemble members are wasted work.
        return TabPFNRegressor(device=device or "cpu", n_estimators=1)
    if name == "exaone":
        from exaonetabular import EXAONETabularRegressor

        # CUDA only on sm80+ (exaone's attention has no fallback kernel below that); else CPU.
        import torch

        use_cuda = (
            (device or "").startswith("cuda")
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(device)[0] >= 8
        )
        # ensemble_count=1 (members are pooled into one bank anyway).
        reg = EXAONETabularRegressor.from_pretrained(
            device="cuda" if use_cuda else "cpu", ensemble_count=1
        )
        # Uniform member weighting (contexts are far below the NNLS threshold).
        if getattr(getattr(reg, "manifest", None), "regression", None) is not None:
            reg.manifest = dataclasses.replace(
                reg.manifest,
                regression=dataclasses.replace(
                    reg.manifest.regression, member_weighting="uniform"
                ),
            )
        return reg
    if name == "tabldm":
        from tabldm import TabLDMRegressor

        # n_estimators=1 (members are pooled into one grid). device=None lets TabLDM pick.
        # random_state is fixed at construction.
        reg = TabLDMRegressor(n_estimators=1, device=device, random_state=0)
        # Avoid reloading the ~300MB checkpoint from disk on every .fit() call.
        orig_load = reg._load_model

        def _cached_load():
            if getattr(reg, "model_", None) is None:
                orig_load()

        reg._load_model = _cached_load
        reg._load_model_cached = True
        return reg
    raise ValueError(f"Unknown marginal backend '{name}', choose from {BACKEND_NAMES}.")


def _require_tabpfn_token() -> None:
    if not os.environ.get("TABPFN_TOKEN"):
        raise RuntimeError(
            "TabPFN v3 requires a one-time license acceptance: open "
            "https://ux.priorlabs.ai, log in, accept the license, copy your "
            "API key from the account page, then `export TABPFN_TOKEN=...` "
            "before running this backend."
        )


# quantiles(): (X_context, y_context, X_query, probs) -> (n_query, Q) in raw y units.
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
    if name == "tabldm":
        return _tabldm_quantiles(regressor, X_context, y_context, X_query, probs, seed=seed)
    raise ValueError(f"Unknown marginal backend '{name}', choose from {BACKEND_NAMES}.")


@contextlib.contextmanager
def _exaone_capture_quantile_bank(regressor):
    """Context manager making EXAONE's .predict() return the full (n_query, quantile_count) bank.

    Replaces _collapse_members' per-row reduction with a per-member sort. Not
    valid with NNLS member weighting.
    """
    import torch
    original = regressor.__dict__.get("_collapse_members")
    had_override = "_collapse_members" in regressor.__dict__

    def _passthrough(self, output, query_count):
        expected = (self.manifest.runtime.ensemble_count, query_count, self.manifest.output_width)
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != expected or not bool(torch.isfinite(output).all()):
            raise RuntimeError("model returned invalid regression quantiles")
        return torch.sort(output.float(), dim=-1).values

    regressor._collapse_members = types.MethodType(_passthrough, regressor)
    try:
        yield
    finally:
        if had_override:
            regressor._collapse_members = original
        else:
            del regressor._collapse_members


def _exaone_quantiles(
    regressor, X_context: np.ndarray, y_context: np.ndarray, X_query: np.ndarray,
    probs: np.ndarray, *, seed: int,
) -> np.ndarray:
    """EXAONE's native 999-level quantile grid, linearly interpolated onto probs (seed unused)."""
    if getattr(getattr(regressor, "manifest", None), "regression", None) is not None:
        if regressor.manifest.regression.member_weighting != "uniform":
            regressor.manifest = dataclasses.replace(
                regressor.manifest,
                regression=dataclasses.replace(
                    regressor.manifest.regression, member_weighting="uniform"
                ),
            )
    regressor.fit(X_context, y_context)
    if regressor._fitted_state.get("member_weights") is not None:
        raise RuntimeError(
            "EXAONE NNLS member-weighting is active; native quantile capture "
            "assumes uniform member averaging (see _exaone_capture_quantile_bank)."
        )
    quantile_count = regressor.manifest.regression.quantile_count
    native_probs = np.linspace(1.0 / (quantile_count + 1), quantile_count / (quantile_count + 1), quantile_count)
    with _exaone_capture_quantile_bank(regressor):
        bank = np.asarray(regressor.predict(X_query))  # (n_query, quantile_count), raw y-units

    out = np.empty((bank.shape[0], len(probs)))
    for i in range(bank.shape[0]):
        out[i] = np.interp(probs, native_probs, bank[i])
    return out


def _tabldm_quantiles(
    regressor, X_context: np.ndarray, y_context: np.ndarray, X_query: np.ndarray,
    probs: np.ndarray, *, seed: int,
) -> np.ndarray:
    """TabLDM quantiles at probs via predict(output_type="quantiles") (seed unused)."""
    regressor.fit(X_context, y_context)
    q = np.asarray(regressor.predict(X_query, output_type="quantiles", alphas=list(probs)))
    return q  # (n_query, Q), already this module's orientation


# Generic K-fold PIT through quantiles().
def loo_pit(
    name: str, regressor, X_train: np.ndarray, y_train: np.ndarray, probs: np.ndarray,
    k_folds: int = 10, eps: float = 1e-6, seed: int = 0,
) -> np.ndarray:
    from eval.metrics.joint_nll import kfold_loo_pit

    return kfold_loo_pit(
        lambda Xc, yc, Xq, k: quantiles(name, regressor, Xc, yc, Xq, probs, seed=seed * 1000 + k),
        X_train, y_train, probs, k_folds=k_folds, eps=eps, seed=seed,
    )
