"""s5_kfold.py — impact of K-fold noise on z_train, frozen checkpoint.

K-folding only affects z_train (the model's INPUT context) -- z_test/
log_pdf_test always come from one non-folded forward regardless of K (see
pit.py::run_pit_batched's own docstring: "Test-set PIT stays a single
forward pass"). So this is a frozen-checkpoint probe, not a training
ablation: for a fixed batch of episodes (fixed context+test points, fixed
z_test), re-run TabICL's K-fold PIT at several K to get several z_train
variants, feed each through the SAME trained --ckpt, and score copula NLL /
correlation-vs-R_post.

Also scores an "oracle input" upper bound: the episode's own exact GP-LOO
z_train (data_gen.py's analytic z_train field, Rasmussen & Williams Eq.
5.12 -- computed for every episode regardless of z_train_source) paired
with the SAME TabICL-PIT z_test used above. This isolates how much of the
model's gap-to-oracle comes from z_train's K-fold noise specifically, vs.
everything else (rank, PIT distortion in z_test, optimization).

Usage:
    python debug/run_debug.py s5 --ckpt <name>
    python debug/stages/s5_kfold.py --ckpt kernel-sweep-all-tabicl-retrain-15k --n-episodes 50
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC, os.path.join(_REPO_ROOT, "debug")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common
from config import DebugConfig, add_common_args, build_config

K_SWEEP_DEFAULT = [2, 5, 10]  # "P" (true K-fold LOO through TabICL) is appended in run()


def _pit_at_k(tabicl_model, episodes: list[dict], k_folds: int, device: str):
    """run_pit_batched at a given K over one shared-P/N batch of episodes,
    replicating data_gen.py's own y-scaling convention exactly (per-episode
    y_train mean/std, y_test/log_pdf_test corrected the same way) so a
    frozen checkpoint sees the same input distribution it was trained on."""
    from pit import run_pit_batched

    x_train = torch.stack([ep["x_norm_train"] for ep in episodes]).to(device)
    x_test = torch.stack([ep["x_norm_test"] for ep in episodes]).to(device)
    y_train = torch.stack([ep["y_train"] for ep in episodes]).to(device)
    y_test = torch.stack([ep["y_test"] for ep in episodes]).to(device)

    y_mean = y_train.mean(dim=1, keepdim=True)
    y_std = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
    y_train_scaled = ((y_train - y_mean) / y_std).unsqueeze(-1)
    y_test_scaled = ((y_test - y_mean) / y_std).unsqueeze(-1)

    out = run_pit_batched(tabicl_model, x_train, y_train_scaled, x_test, y_test_scaled, k_folds=k_folds)
    z_train = out["z_train"].squeeze(-1)                              # (B, P)
    z_test = out["z_test"].squeeze(-1)                                # (B, N)
    log_pdf_test = out["log_pdf_test"].squeeze(-1) - y_std.log()  # (B, N) - (B, 1) broadcast, matches data_gen.py's own convention
    return z_train, z_test, log_pdf_test


def _score_variant(model, episodes: list[dict], z_train, z_test, log_pdf_test, posts, cfg, device):
    from dataset import collate_fn
    from loss import y_space_nll
    from model import build_sigma

    variant_eps = []
    for i, ep in enumerate(episodes):
        e = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in ep.items()}
        e["z_train"] = z_train[i].cpu()
        e["z_test"] = z_test[i].cpu()
        e["log_pdf_test"] = log_pdf_test[i].cpu()
        variant_eps.append(e)

    batch = {k: v.to(device) for k, v in collate_fn(variant_eps).items()}
    jitter = float(cfg.model.get("sigma_jitter", 1e-4))
    parametrization = str(cfg.model.get("correlation_parametrization", "covnorm"))
    with torch.no_grad():
        out = model(batch)
        Sigma = build_sigma(out, cfg, jitter=jitter, test_mask=batch["test_mask"])
        parts = y_space_nll(Sigma, batch["z_test"].float(), batch["log_pdf_test"].float(), batch["test_mask"])

    pearsons, maes = [], []
    for i, post in enumerate(posts):
        if post is None:
            continue
        n = int(episodes[i]["n_test"].item())
        R_pred = Sigma[i, :n, :n].float().cpu().numpy()
        R_post = post["R_post"].float().cpu().numpy()
        ri, ci = np.triu_indices(n, k=1)
        off_pred, off_post = R_pred[ri, ci], R_post[ri, ci]
        if len(off_pred) > 1:
            pearsons.append(float(np.corrcoef(off_pred, off_post)[0, 1]))
            maes.append(float(np.abs(off_pred - off_post).mean()))

    return {
        "copula_nll": float(parts["copula"].item()),
        "marginal_nll": float(parts["marginal"].item()),
        "total_nll": float(parts["total"].item()),
        "corr_pearson_vs_Rpost": float(np.mean(pearsons)) if pearsons else None,
        "corr_mae_vs_Rpost": float(np.mean(maes)) if maes else None,
    }


def run(dcfg: DebugConfig, k_sweep=None) -> dict:
    if dcfg.ckpt is None:
        return {"error": "S5 needs a trained --ckpt (frozen-checkpoint probe, no retraining here)."}

    k_sweep = list(k_sweep or K_SWEEP_DEFAULT)
    model = common.load_model(dcfg)
    tabicl_model = common.load_frozen_tabicl(dcfg)

    episodes = common.generate_episodes(dcfg, dcfg.n_episodes, tabicl_model=None, return_kernel_metadata=True)
    posts = [common.posterior_oracle(ep) for ep in episodes]
    P = int(episodes[0]["n_train"].item())
    k_sweep_eff = sorted(set(k_sweep) | {P})  # always include true K-fold LOO (K=P)

    results = {}
    for K in k_sweep_eff:
        z_train, z_test, log_pdf_test = _pit_at_k(tabicl_model, episodes, K, dcfg.device)
        label = "P (true LOO)" if K == P else str(K)
        results[label] = _score_variant(model, episodes, z_train, z_test, log_pdf_test, posts, dcfg.cfg, dcfg.device)

    # Oracle-input upper bound: exact analytic GP-LOO z_train (data_gen.py's
    # own field, untouched) + the SAME TabICL-PIT z_test/log_pdf_test from
    # the last K-fold call above (z_test is K-invariant, see module docstring).
    z_train_oracle = torch.stack([ep["z_train"] for ep in episodes]).to(dcfg.device)
    results["oracle_z_train"] = _score_variant(
        model, episodes, z_train_oracle, z_test, log_pdf_test, posts, dcfg.cfg, dcfg.device
    )

    return {"n_episodes": len(episodes), "P": P, "k_sweep": k_sweep_eff, "ckpt": dcfg.ckpt, "results": results}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--k-sweep", default=",".join(str(x) for x in K_SWEEP_DEFAULT))
    args = p.parse_args()

    dcfg = build_config(
        overrides=args.override, model_preset=args.model, n_episodes=args.n_episodes,
        ckpt=args.ckpt, device=args.device, seed=args.seed, run_id=args.run_id,
    )
    result = run(dcfg, k_sweep=[int(x) for x in args.k_sweep.split(",")])
    if "error" in result:
        print(result["error"])
        return

    print(f"Scored {result['n_episodes']} episodes (P={result['P']}), ckpt={result['ckpt']}\n")
    print(f"{'K':>14} {'copula NLL/pt':>14} {'total NLL/pt':>13} {'corr Pearson':>13} {'corr MAE':>10}")
    for label, r in result["results"].items():
        pear = f"{r['corr_pearson_vs_Rpost']:.4f}" if r["corr_pearson_vs_Rpost"] is not None else "n/a"
        mae = f"{r['corr_mae_vs_Rpost']:.4f}" if r["corr_mae_vs_Rpost"] is not None else "n/a"
        print(f"{label:>14} {r['copula_nll']:>14.4f} {r['total_nll']:>13.4f} {pear:>13} {mae:>10}")
    print(
        "\n'oracle_z_train' is the upper bound: exact GP-LOO z_train (no TabICL "
        "K-fold noise) + the same TabICL-PIT z_test every K row above uses. "
        "The gap between the best K row and oracle_z_train isolates z_train's "
        "K-fold-noise cost specifically."
    )

    path = common.save_stage_result(dcfg, "s5_kfold", result)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
