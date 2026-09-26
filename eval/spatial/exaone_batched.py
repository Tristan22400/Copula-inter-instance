"""Batched PIT for EXAONE-Tabular across episodes (data.z_train_source=exaone).

Each episode is fitted with the regressor's own code (fit, preprocessor,
build_ensemble_inputs); the member batches of all episodes (same support,
query and feature counts) are concatenated along the executor's batch axis
for one forward call, then split, rescaled and mean-pooled per episode.
tests/test_exaone_batched.py checks it against the per-episode path. Not
valid when NNLS member weighting is active (raises).
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
    """Fit one episode and return, per (n_svd, seed) pass, its (support, label, query) member tensors plus its (center, scale)."""
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
    """(B, query_rows, quantile_count) quantile bank in raw y units, mean-pooled over members, from one forward."""
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

        # no_grad (not inference_mode: LoRA-parametrized weights need version counters).
        regressor.model.eval()
        with torch.no_grad():
            raw = regressor._forward_chunked(support_batch, label_batch, query_batch)  # (B*members, query_rows, Q)
        expected = (B * members_per_episode, query_rows, regressor.manifest.output_width)
        if tuple(raw.shape) != expected or not bool(torch.isfinite(raw).all()):
            raise RuntimeError("exaone_batched: model returned invalid regression quantiles")
        pass_outputs.append(raw.float().reshape(B, members_per_episode, query_rows, -1))

    # Sort each member's quantiles to remove crossings.
    pooled = torch.cat(pass_outputs, dim=1)              # (B, total_members, query_rows, Q)
    pooled = torch.sort(pooled, dim=-1).values.mean(dim=1)  # (B, query_rows, Q)

    center = torch.tensor([per_episode[b][1] for b in range(B)], device=pooled.device).view(B, 1, 1)
    scale = torch.tensor([per_episode[b][2] for b in range(B)], device=pooled.device).view(B, 1, 1)
    return (pooled * scale + center).cpu().numpy()


def _quantile_bank_on_probs(regressor, X_context: list, y_context: list, X_query: list,
                            probs: np.ndarray) -> np.ndarray:
    """_quantile_bank_batched interpolated from EXAONE's native 999-level grid onto probs."""
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
    """run_pit_batched for EXAONE, via the shared K-fold driver (_batched_pit.run_kfold_pit_batched)."""
    from eval.spatial._batched_pit import run_kfold_pit_batched

    return run_kfold_pit_batched(
        lambda X_ctx, y_ctx, X_qry, probs: _quantile_bank_on_probs(
            regressor, X_ctx, y_ctx, X_qry, probs
        ),
        X_train, Y_train, X_test, Y_test,
        k_folds=k_folds, probs_n=probs_n, eps=eps, seed=seed,
    )
