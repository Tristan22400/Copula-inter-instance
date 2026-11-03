"""Tabular foundation models as marginal backends for K-fold PIT.

Every backend implements quantiles(name, regressor, X_context, y_context,
X_query, probs) -> (n_query, Q) in raw y units, which loo_pit plugs into the
generic K-fold loop (eval/metrics/joint_nll.kfold_loo_pit).

    tabicl: TabICL v2 (native quantiles).
    tabpfn: PriorLabs TabPFN v3 (native quantiles); needs TABPFN_TOKEN after
        accepting the licence at https://ux.priorlabs.ai.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from copula_inter.backend_registry import MARGINAL_BACKENDS, get_backend

__all__ = ["make_regressor", "quantiles", "loo_pit"]


# Regressor construction (one instance reused across folds and tasks).
def make_regressor(name: str, device: "str | None" = None, ckpt: "str | None" = None) -> Any:
    """Build a backend regressor, optionally loading a Phase-A checkpoint (MarginalBackbone.save format) into its trainable module."""
    get_backend(name)
    regressor = _make_pretrained_regressor(name, device)
    if ckpt:
        _load_finetuned_weights(name, regressor, ckpt, device)
    return regressor


def _load_finetuned_weights(name: str, regressor: Any, ckpt: str, device: "str | None") -> None:
    import torch

    from copula_inter.marginal.finetune_adapters import _trainable_module

    payload = torch.load(ckpt, map_location=device or "cpu", weights_only=False)
    written_for = payload.get("backbone")
    if written_for not in (None, name):
        raise ValueError(f"marginal checkpoint {ckpt} was written for backbone {written_for!r}, not {name!r}.")
    _trainable_module(name, regressor).load_state_dict(payload["state_dict"], strict=True)


def _make_pretrained_regressor(name: str, device: "str | None" = None) -> Any:
    if name == "tabicl":
        from copula_inter.marginal.tabicl_regressor import make_tabicl_regressor

        return make_tabicl_regressor(device=device)
    if name == "tabpfn":
        _require_tabpfn_token()
        # Enable tabpfn's in-memory model cache so each .fit() doesn't rebuild the model.
        os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "2")
        from tabpfn import TabPFNRegressor

        # n_estimators=1: only one quantile grid is read, so ensemble members are wasted work.
        return TabPFNRegressor(device=device or "cpu", n_estimators=1)
    raise ValueError(f"Unknown marginal backend '{name}', choose from {list(MARGINAL_BACKENDS)}.")


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
    name: str,
    regressor: Any,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    probs: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    if name == "tabicl":
        from copula_inter.marginal.tabicl_regressor import tabicl_quantiles

        return tabicl_quantiles(regressor, X_context, y_context, X_query, probs)
    if name == "tabpfn":
        regressor.fit(X_context, y_context)
        out = regressor.predict(X_query, output_type="quantiles", quantiles=list(probs))
        return np.asarray(out).T  # (n_quantiles, n_query) -> (n_query, n_quantiles)
    raise ValueError(f"Unknown marginal backend '{name}', choose from {list(MARGINAL_BACKENDS)}.")


# Generic K-fold PIT through quantiles().
def loo_pit(
    name: str,
    regressor: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    probs: np.ndarray,
    k_folds: int = 10,
    eps: float = 1e-6,
    seed: int = 0,
) -> np.ndarray:
    from copula_inter.marginal.kfold import kfold_loo_pit

    return kfold_loo_pit(
        lambda Xc, yc, Xq, k: quantiles(name, regressor, Xc, yc, Xq, probs, seed=seed * 1000 + k),
        X_train,
        y_train,
        probs,
        k_folds=k_folds,
        eps=eps,
        seed=seed,
    )
