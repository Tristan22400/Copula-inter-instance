"""Diagnostic: how the electrode encoding seen by the copula changes its EEG joint log-likelihood.

The marginal always sees [t, one-hot electrode] (so indep is identical across rows); only the
copula's features change:
  onehot   [t, one-hot electrode] (the runner's default)
  ordinal  [t, electrode index]
  time     [t]  (same-time rows of different electrodes look identical to the copula)

Usage (node, main venv): python debug/eeg_copula_encoding.py <data_dir> [marginal] [n_trials]
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from eval.data.eeg_tasks import hassan_batches
from eval.runners.eeg_joint_eval import (
    EegJointEvalSpec,
    _copula_factor,
    _load_models,
    _normalize,
    _pit,
    _tensors,
    copula_scores,
    episode_scores,
)


def _copula_x(x: np.ndarray, how: str) -> np.ndarray:
    if how == "onehot":
        return x
    if how == "ordinal":
        return np.concatenate([x[..., :1], x[..., 1:].argmax(-1, keepdims=True).astype(float)], axis=-1)
    return x[..., :1]


def main() -> None:
    data_dir = sys.argv[1]
    marginal = sys.argv[2] if len(sys.argv) > 2 else "tabicl"
    n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    dev = torch.device("cuda")
    models = _load_models(EegJointEvalSpec(out_dir="-"), marginal, dev)
    for mode, n, m in (("for", 192, 16), ("for", 8, 16), ("int", 32, 16)):
        batch = next(hassan_batches(data_dir, mode, n, m, batch_size=n_trials, seed=0))
        x_tr, y_tr, x_te, y_te = _tensors(batch, dev)
        with torch.no_grad():
            pit = _pit(models, x_tr, y_tr, x_te, y_te, 5, seed=0)
        log_pdf = pit["log_pdf_test"].double().cpu().numpy()
        z = pit["z_test"].float()
        row = {}
        for how in ("onehot", "ordinal", "time"):
            ctr, cte = _normalize(_copula_x(batch.x_train, how), _copula_x(batch.x_test, how))
            ctr_t = torch.from_numpy(ctr.astype(np.float32)).to(dev)
            cte_t = torch.from_numpy(cte.astype(np.float32)).to(dev)
            with torch.no_grad():
                U, D = _copula_factor(models.copula, ctr_t, pit["z_train"], cte_t)
            sc = episode_scores(log_pdf, copula_scores(U, D, z))
            row[how] = float(np.mean(sc["ll_copula"]))
        print(
            f"{marginal} {mode} N={n} M={m}: indep {np.mean(sc['ll_indep']):.3f}  "
            + "  ".join(f"{k} {v:.3f}" for k, v in row.items()),
            flush=True,
        )


if __name__ == "__main__":
    main()
