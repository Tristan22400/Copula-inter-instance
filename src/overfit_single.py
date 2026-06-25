"""
overfit_single.py — Overfit on K synthetic realizations from one episode's R*.

To recover the correlation matrix from NLL alone you need multiple z_test samples
from N(0, R*) — a single sample gives the rank-1 MLE z*z^T, not R*.

This script:
  1. Loads one .pt episode and extracts its R* and fixed context (x_train, z_train, x_test).
  2. Draws K synthetic z_test ~ N(0, R*) while keeping the context fixed.
  3. Trains on those K realizations (cycling), so the expected gradient pushes R̂ → R*.
  4. Tracks convergence via ||R̂ - R*||_F and copula NLL vs oracle.

Healthy run: copula_gap → 0 and ||R̂ - R*||_F → 0.

Usage:
    python src/overfit_single.py
    python src/overfit_single.py --episode data/pit_episodes_cosine/task_000042.pt
    python src/overfit_single.py --k-realizations 500 --steps 5000 --lr 1e-3
    python src/overfit_single.py --freeze-backbone
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "tabicl_upstream", "src"))

from dataset import collate_fn
from loss import _safe_cholesky, oracle_copula_nll, y_space_nll
from model import build_copula_transformer, low_rank_correlation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--episode",
        default=None,
        help="Path to .pt episode file (default: task_000000.pt in pit_episodes_cosine)",
    )
    p.add_argument(
        "--k-realizations",
        type=int,
        default=200,
        help="Number of synthetic z_test samples drawn from N(0, R*) (default: 200)",
    )
    p.add_argument("--steps", type=int, default=3000, help="Gradient steps")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--plot", default="overfit_correlation.png", help="Output plot path")
    p.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze TabICL backbone (faster but may converge slower)",
    )
    return p.parse_args()


def plot_correlation_comparison(
    R_star: torch.Tensor,
    R_hat: torch.Tensor,
    frob: float,
    gap: float,
    out_path: str,
) -> None:
    """Save a 3-panel figure: R*, R̂, and |R̂ - R*|."""
    R_s = R_star.cpu().numpy()
    R_h = R_hat.cpu().detach().numpy()
    diff = abs(R_h - R_s)

    vmax = max(abs(R_s).max(), abs(R_h).max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, mat, title in zip(
        axes,
        [R_s, R_h, diff],
        [r"Oracle $R^*$", r"Predicted $\hat{R}$", r"$|\hat{R} - R^*|$"],
    ):
        if "diff" in title or "|" in title:
            im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=diff.max())
        else:
            im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("test instance")
        ax.set_ylabel("test instance")

    fig.suptitle(
        rf"Overfit sanity check — $\|R^* - \hat{{R}}\|_F = {frob:.4f}$,  copula gap $= {gap:.4f}$",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out_path}")


def build_synthetic_dataset(episode: dict, K: int) -> list[dict]:
    """Generate K synthetic episodes from the same task by sampling z_test ~ N(0, R*)."""
    n_test = int(episode["n_test"].item())
    R_star = episode["R_star"][:n_test, :n_test].float()

    # Cholesky-sample K vectors from N(0, R*)
    L = _safe_cholesky(R_star)
    eps = torch.randn(n_test, K)          # (n_test, K)
    z_samples = (L @ eps).T               # (K, n_test)

    realizations = []
    for k in range(K):
        ep = {key: val.clone() for key, val in episode.items()}
        ep["z_test"] = z_samples[k]
        # Marginal log_pdf_test is set to 0 — we only care about the copula term.
        ep["log_pdf_test"] = torch.zeros(n_test)
        realizations.append(ep)
    return realizations


def main() -> None:
    args = parse_args()

    # Build config from yaml files without Hydra.
    base_cfg = OmegaConf.load(os.path.join(_ROOT, "conf", "config.yaml"))
    model_cfg = OmegaConf.load(
        os.path.join(_ROOT, "conf", "model", "copula_transformer.yaml")
    )
    OmegaConf.set_struct(base_cfg, False)
    cfg = OmegaConf.merge(base_cfg, OmegaConf.create({"model": model_cfg}))
    if args.freeze_backbone:
        cfg.model.unfreeze_backbone = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load episode.
    if args.episode is None:
        episode_path = os.path.join(
            _ROOT, "data", "pit_episodes_cosine", "task_000000.pt"
        )
    else:
        episode_path = args.episode

    episode = torch.load(episode_path, map_location="cpu", weights_only=True)
    n_train = int(episode["n_train"].item())
    n_test = int(episode["n_test"].item())
    print(f"Episode : {os.path.basename(episode_path)}")
    print(f"  n_train={n_train}  n_test={n_test}  d_x={episode['x_norm_train'].shape[-1]}")
    print(f"  K realizations from N(0, R*): {args.k_realizations}")

    # Synthetic dataset: K different z_test from the same R*.
    realizations = build_synthetic_dataset(episode, args.k_realizations)

    # Build model.
    model = build_copula_transformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")
    print(f"Backbone frozen : {not bool(cfg.model.get('unfreeze_backbone', False))}")

    jitter = float(cfg.model.get("sigma_jitter", 1e-4))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
    )

    # Oracle NLL (true R*, averaged over the K synthetic z_test vectors).
    oracle_nll_vals = []
    for i in range(0, args.k_realizations, args.batch_size):
        chunk = realizations[i : i + args.batch_size]
        b = collate_fn(chunk)
        b = {k: v.to(device) for k, v in b.items()}
        with torch.no_grad():
            oracle_nll_vals.append(
                oracle_copula_nll(
                    b["R_star"].float(), b["z_test"].float(), b["test_mask"]
                ).item()
            )
    oracle_nll = sum(oracle_nll_vals) / len(oracle_nll_vals)

    R_star = episode["R_star"][:n_test, :n_test].float().to(device)

    print(f"\nOracle copula NLL = {oracle_nll:.6f}  (expected ≈ 0 for R* ≈ I, negative for structured R*)")
    header = (
        f"{'step':>6}  {'cop_nll':>10}  {'oracle':>10}  {'gap':>10}"
        f"  {'||R-R*||_F':>12}  {'||R-R*||_F/N²':>15}"
    )
    print(header)
    print("-" * len(header))

    model.train()
    step = 0
    while step <= args.steps:
        # Shuffle realizations each pass.
        indices = torch.randperm(len(realizations)).tolist()
        for i in range(0, len(realizations), args.batch_size):
            if step > args.steps:
                break
            chunk = [realizations[j] for j in indices[i : i + args.batch_size]]
            batch = collate_fn(chunk)
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(batch)
            Sigma = low_rank_correlation(
                out["W"].float(), out["s"].float(), batch["test_mask"], jitter=jitter
            )
            parts = y_space_nll(
                Sigma,
                batch["z_test"].float(),
                batch["log_pdf_test"].float(),
                batch["test_mask"],
            )
            loss = parts["copula"]  # ignore marginal (set to 0 in synthetic data)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            if step % args.log_every == 0:
                with torch.no_grad():
                    # Evaluate R̂ on the first batch (fixed input → same R̂ always).
                    eval_batch = collate_fn([realizations[0]])
                    eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                    out_eval = model(eval_batch)
                    Sigma_eval = low_rank_correlation(
                        out_eval["W"].float(),
                        out_eval["s"].float(),
                        eval_batch["test_mask"],
                        jitter=jitter,
                    )
                    R_hat = Sigma_eval[0, :n_test, :n_test]
                    frob = (R_hat - R_star).norm().item()
                    frob_per_n2 = frob / (n_test ** 2)
                print(
                    f"{step:>6}  {parts['copula'].item():>10.4f}  {oracle_nll:>10.4f}"
                    f"  {parts['copula'].item() - oracle_nll:>10.4f}"
                    f"  {frob:>12.4f}  {frob_per_n2:>15.6f}"
                )
            step += 1

    # Final visual: compare R̂ vs R*.
    with torch.no_grad():
        eval_batch = collate_fn([realizations[0]])
        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
        out_final = model(eval_batch)
        Sigma_final = low_rank_correlation(
            out_final["W"].float(),
            out_final["s"].float(),
            eval_batch["test_mask"],
            jitter=jitter,
        )
        R_hat_final = Sigma_final[0, :n_test, :n_test]

    show = min(n_test, 5)
    print(f"\nR* (top-left {show}×{show}):")
    print(R_star[:show, :show].cpu().numpy().round(3))
    print(f"\nR̂  (top-left {show}×{show}):")
    print(R_hat_final[:show, :show].cpu().detach().numpy().round(3))

    final_frob = (R_hat_final - R_star).norm().item()
    final_gap = parts["copula"].item() - oracle_nll
    print(f"\nFinal ||R̂ - R*||_F = {final_frob:.4f}")
    print(f"Final copula gap   = {final_gap:.4f}")

    plot_correlation_comparison(R_star, R_hat_final, final_frob, final_gap, args.plot)


if __name__ == "__main__":
    main()
