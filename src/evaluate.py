"""
evaluate.py — Evaluation metrics for the inter-instance Copula Transformer.

Metrics computed per task:
  1. copula_nll_pred   — NLL under predicted R̂ (main metric)
  2. copula_nll_oracle — NLL under oracle R* (lower bound)
  3. copula_nll_indep  — NLL under R=I (independence baseline, ≈ 0 by construction)
  4. oracle_frac       — pred / oracle (1.0 = perfect, higher = worse)
  5. frob_error        — ||R̂ - R*||_F / N
  6. offdiag_pearson   — Pearson r between off-diagonal entries of R̂ and R*

Usage:
    python src/evaluate.py --ckpt ./checkpoints/copula_transformer/step_0050000.pt \
                           --dataset_dir ./data/pit_episodes --n_tasks 500
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataset import CopulaDataset, collate_fn
from loss import copula_nll, oracle_copula_nll
from model import build_copula_transformer


def evaluate(
    model: torch.nn.Module,
    dataset_dir: str,
    device: str,
    batch_size: int = 16,
    n_tasks: int | None = None,
) -> dict:
    files = sorted(glob(os.path.join(dataset_dir, "*.pt")))
    if n_tasks is not None:
        files = files[:n_tasks]

    loader = DataLoader(
        CopulaDataset(file_list=files),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
    )

    results = {
        k: [] for k in ["nll_pred", "nll_oracle", "nll_indep", "frob", "pearson"]
    }

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            W_tilde = model(batch)

            nll_pred = copula_nll(W_tilde, batch["z_test"], batch["test_mask"])
            nll_oracle = oracle_copula_nll(
                batch["R_star"], batch["z_test"], batch["test_mask"]
            )

            # Independence baseline: R=I, eps=1 → R_eps = 2I
            # copula_nll with W_tilde=0 and eps→1 → L = 0.5*(N*log(1+0)+0 - z^Tz)/N ≠ 0
            # True independence NLL: R=I → L = 0 (by formula); approx via small eps
            B, N_max, r1 = W_tilde.shape
            W_zero = torch.zeros_like(W_tilde)
            nll_indep = copula_nll(W_zero, batch["z_test"], batch["test_mask"], eps=1.0)

            results["nll_pred"].append(nll_pred.item())
            results["nll_oracle"].append(nll_oracle.item())
            results["nll_indep"].append(nll_indep.item())

            # Per-task structural metrics
            R_hat = torch.bmm(W_tilde, W_tilde.transpose(-2, -1))
            for b in range(batch["x_train"].shape[0]):
                N = batch["n_test"][b].item()
                R_h = R_hat[b, :N, :N].cpu()
                R_s = batch["R_star"][b, :N, :N].cpu()

                results["frob"].append((R_h - R_s).norm(p="fro").item() / N)

                mask = ~torch.eye(N, dtype=torch.bool)
                rh_off = R_h[mask].numpy()
                rs_off = R_s[mask].numpy()
                if rh_off.std() > 1e-8 and rs_off.std() > 1e-8:
                    results["pearson"].append(float(np.corrcoef(rh_off, rs_off)[0, 1]))

    def mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    return {
        "copula_nll_pred": mean(results["nll_pred"]),
        "copula_nll_oracle": mean(results["nll_oracle"]),
        "copula_nll_indep": mean(results["nll_indep"]),
        "oracle_frac": mean(results["nll_pred"]) / (mean(results["nll_oracle"]) + 1e-8),
        "frob_error": mean(results["frob"]),
        "offdiag_pearson": mean(results["pearson"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Copula Transformer")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--dataset_dir", required=True, help="Directory of PIT episodes"
    )
    parser.add_argument(
        "--n_tasks", type=int, default=None, help="Max tasks to evaluate"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    from omegaconf import OmegaConf

    cfg = OmegaConf.create(ckpt["cfg"])
    model = build_copula_transformer(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    metrics = evaluate(model, args.dataset_dir, device, args.batch_size, args.n_tasks)

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}")


if __name__ == "__main__":
    main()
