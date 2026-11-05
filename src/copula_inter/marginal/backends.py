"""Tabular foundation models as marginal backends for K-fold PIT.

Every backend implements quantiles(name, regressor, X_context, y_context,
X_query, probs) -> (n_query, Q) in raw y units, which loo_pit plugs into the
generic K-fold loop (eval/metrics/joint_nll.kfold_loo_pit).

    tabicl: TabICL v2 (native quantiles).
    tabpfn35_fast: PriorLabs TabPFN-3.5, Fast checkpoint (marginal/tabpfn_marginal.py);
        needs TABPFN_TOKEN with the TabPFN-3.5 licence accepted at https://ux.priorlabs.ai.
    kumo_small: NVIDIA Kumo Tabular, small (marginal/kumo_marginal.py).

The per-episode quantiles of tabpfn35_fast and kumo_small are their batched
quantile functions at batch size one, so the two paths agree by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from copula_inter.backend_registry import MARGINAL_BACKENDS, batched_pit, get_backend

__all__ = ["BackendMarginal", "make_regressor", "quantiles", "loo_pit"]


@dataclass(frozen=True)
class BackendMarginal:
    """A generic marginal backend with its regressor, accepted where a frozen TabICL marginal PITs probe episodes.

    pit.compute_context_z_train and training.probe_batches._tabicl_pit_batch
    take it in place of a TabICL module.
    """

    name: str
    regressor: Any
    probs_n: int = 99

    def pit(
        self,
        x_train: np.ndarray,
        y_train_scaled: np.ndarray,
        x_test: np.ndarray,
        y_test_scaled: np.ndarray,
        k_folds: int,
        seed: int = 0,
    ) -> dict[str, np.ndarray]:
        """K-fold PIT of one episode: z_train (P,), z_test (N,), log_pdf_test (N,) in the scaled y units."""
        out = batched_pit(self.name)(
            self.regressor,
            x_train[None],
            y_train_scaled[None],
            x_test[None],
            y_test_scaled[None],
            k_folds=k_folds,
            probs_n=self.probs_n,
            seed=seed,
        )
        return {key: value[0] for key, value in out.items()}


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
    if written_for != name:
        raise ValueError(f"marginal checkpoint {ckpt} was written for backbone {written_for!r}, not {name!r}.")
    _trainable_module(name, regressor).load_state_dict(payload["state_dict"], strict=True)


def _make_pretrained_regressor(name: str, device: "str | None" = None) -> Any:
    if name == "tabicl":
        from copula_inter.marginal.tabicl_regressor import make_tabicl_regressor

        return make_tabicl_regressor(device=device)
    if name == "tabpfn35_fast":
        from copula_inter.marginal.tabpfn_marginal import make_tabpfn_regressor

        return make_tabpfn_regressor(device=device)
    if name == "kumo_small":
        from copula_inter.marginal.kumo_marginal import make_kumo_regressor

        return make_kumo_regressor(device=device)
    raise ValueError(f"Unknown marginal backend '{name}', choose from {list(MARGINAL_BACKENDS)}.")


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
    if name in ("tabpfn35_fast", "kumo_small"):
        import torch

        from copula_inter.marginal.finetune_adapters import backend_quantiles

        with torch.no_grad():
            q = backend_quantiles(name, regressor, [X_context], [y_context], [X_query], probs)
        return q[0].float().cpu().numpy()
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
