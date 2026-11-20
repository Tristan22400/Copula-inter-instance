"""Diagnostic: is the EEG copula gap a prediction problem (Sigma) or a family problem (Gaussian copula)?

Hassan-style forecasting: every trial's targets are the M time points right after its N context
points, so target row k means the same (time offset, electrode) in every trial and z_test vectors
of different trials are comparable. Split the evaluation trials in two halves (even / odd index);
estimate R_hat = corr(z_test) on one half (shrunk 1% toward I), score the other half with it, and
vice versa. This "empirical Gaussian copula" is an oracle-ish reference (it sees other trials' test
values) for the best Gaussian copula on these marginals; compare it with the model's own copula.
Also prints mean correlations of two pair types, predicted vs empirical:
  adj   same electrode, consecutive time points
  xel   same time point, different electrodes

Usage (node, main venv): python debug/eeg_attainable_copula.py <data_dir> [marginal]
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
    _pit,
    _tensors,
    copula_scores,
)


def _dense_nll(R: np.ndarray, z: np.ndarray) -> np.ndarray:
    """(T,) summed Gaussian copula NLL of each row of z (T, N) under R."""
    _, logdet = np.linalg.slogdet(R)
    quad = np.einsum("tn,tn->t", z @ np.linalg.inv(R), z)
    return 0.5 * (logdet + quad - (z * z).sum(1))


def main() -> None:
    data_dir = sys.argv[1]
    marginal = sys.argv[2] if len(sys.argv) > 2 else "tabicl"
    dev = torch.device("cuda")
    models = _load_models(EegJointEvalSpec(out_dir="-"), marginal, dev)
    for n, m in ((8, 16), (64, 16), (192, 16)):
        zs, lps, cop, s_adj, s_xel = [], [], [], [], []
        for b_no, batch in enumerate(hassan_batches(data_dir, "for", n, m, batch_size=64, seed=0)):
            x_tr, y_tr, x_te, y_te = _tensors(batch, dev)
            with torch.no_grad():
                pit = _pit(models, x_tr, y_tr, x_te, y_te, 5, seed=b_no)
                U, D = _copula_factor(models.copula, x_tr, pit["z_train"], x_te)
            z = pit["z_test"].float()
            zs.append(z.double().cpu().numpy())
            lps.append(pit["log_pdf_test"].double().cpu().numpy())
            cop.append(copula_scores(U, D, z))
            S = (U @ U.transpose(1, 2)).double().cpu().numpy()  # off-diagonal of Sigma
            rows_t = np.rint(batch.x_test[0, :, 0] * 256).astype(int)
            rows_e = batch.x_test[0, :, 1:].argmax(1)
            adj = (rows_e[:, None] == rows_e[None]) & (np.abs(rows_t[:, None] - rows_t[None]) == 1)
            xel = (rows_t[:, None] == rows_t[None]) & (rows_e[:, None] != rows_e[None])
            s_adj.append(S[:, adj].mean(1))
            s_xel.append(S[:, xel].mean(1))
        z = np.concatenate(zs)
        N = z.shape[1]  # adj / xel: the pair masks of the (shared) target layout
        indep = np.concatenate(lps).sum(1) / N
        model = indep - np.concatenate(cop) / N
        halves = [np.arange(len(z)) % 2 == h for h in (0, 1)]
        emp = np.empty(len(z))
        for fit, score in (halves, halves[::-1]):
            R = np.corrcoef(z[fit].T)
            R = 0.99 * R + 0.01 * np.eye(N)
            emp[score] = indep[score] - _dense_nll(R, z[score]) / N
        R_all = np.corrcoef(z.T)
        print(
            f"{marginal} for N={n} M={m} (T={len(z)}): indep {indep.mean():.3f}  model copula {model.mean():.3f}  "
            f"empirical Gaussian copula (cross-fit) {emp.mean():.3f} | corr adj: model {np.mean(np.concatenate(s_adj)):.2f} "
            f"emp {R_all[adj].mean():.2f}; corr xel: model {np.mean(np.concatenate(s_xel)):.2f} emp {R_all[xel].mean():.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
