"""s1_rank_ceiling.py — the low-rank capacity ceiling. THE key debug stage.

For each episode's exact GP posterior correlation R_post (pit.py::
gp_analytical_posterior), fits the best rank-r factor model
Sigma = covnorm(W, s) = normalize(W W^T + diag(softplus(s))) — the SAME
parametrization the real model uses (model.py::low_rank_correlation,
imported directly, not reimplemented) — by minimizing the exact expected
copula NLL under z ~ N(0, R_post):

    E[copula_nll]/N = 0.5/N * ( log|Sigma| + tr(Sigma^-1 R_post) - tr(R_post) )

This is deterministic factor analysis on a known population covariance (no
sampling noise, no z draws needed): Adam directly minimizes the closed-form
expectation above. Answers "how much of R_post's structure CAN a rank-r
covnorm factor even represent", independent of the backbone/optimizer/data
distortion questions every other stage asks.

Model rank stays fixed at 32 (see debug/README.md) — this sweep measures
capacity, it isn't a proposal to raise it.

Validation: this stage's r=8 vs r=32 ceiling delta should track the
observed copula-gap delta from real training runs (0.234 @ r=8 vs 0.158 @
r=32, both at P=32/N=256 -- see debug/README.md's wandb table). If it does,
this is the yardstick every later stage measures against; if it doesn't,
rank isn't the story and S3/S6 take over.

Usage:
    python debug/run_debug.py s1
    python debug/stages/s1_rank_ceiling.py --n-episodes 64 --ranks 8,16,32,64
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

RANK_SWEEP_DEFAULT = [4, 8, 16, 32, 64, 128]


def fit_rank_ceiling(
    R_post: torch.Tensor, r: int, *, steps: int = 300, lr: float = 0.05,
    jitter: float = 1e-4, device: str = "cpu",
):
    """Fit a rank-r covnorm factor model to a batch of population
    covariances R_post (B, N, N) by minimizing the exact expected copula
    NLL (closed form, no sampling). Returns (loss_per_episode (B,), Sigma_hat).

    Reuses model.py::low_rank_correlation (the real model's exact
    parametrization) and loss.py::_safe_cholesky (the real model's exact
    numerical-safety path) rather than reimplementing either.
    """
    from loss import _safe_cholesky
    from model import low_rank_correlation

    B, N, _ = R_post.shape
    R_post = R_post.to(device)
    # Same init as CopulaTabICL.copula_head's output layer (model.py:196-198):
    # W ~ N(0, 0.02^2), s = 0 -- starts near Sigma ≈ I, same as the real model.
    W = (torch.randn(B, N, r, device=device) * 0.02).requires_grad_(True)
    s = torch.zeros(B, N, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, s], lr=lr)

    trace_R = R_post.diagonal(dim1=-2, dim2=-1).sum(-1)  # (B,) -- constant w.r.t. W/s

    def _loss(W_, s_):
        Sigma = low_rank_correlation(W_, s_, jitter=jitter, parametrization="covnorm")
        L = _safe_cholesky(Sigma)
        logdet = 2.0 * L.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).log().sum(-1)
        # Sigma^-1 R_post via two triangular solves (loss.py's own convention --
        # cholesky_solve is missing sm_75 kernels on some cluster nodes, see
        # loss.py::copula_nll's comment).
        tmp = torch.linalg.solve_triangular(L, R_post, upper=False)
        X = torch.linalg.solve_triangular(L.mT, tmp, upper=True)
        trace_term = X.diagonal(dim1=-2, dim2=-1).sum(-1)
        per_ep = 0.5 * (logdet + trace_term - trace_R) / N
        return per_ep, Sigma

    for _ in range(steps):
        opt.zero_grad()
        per_ep, _ = _loss(W, s)
        per_ep.mean().backward()
        opt.step()

    with torch.no_grad():
        per_ep_final, Sigma_final = _loss(W, s)
    return per_ep_final.detach().cpu(), Sigma_final.detach()


def run(dcfg: DebugConfig, ranks=None, steps: int = 300, lr: float = 0.05) -> dict:
    ranks = ranks or RANK_SWEEP_DEFAULT
    pairs = common.collect_posteriors(dcfg, dcfg.n_episodes)
    if not pairs:
        return {"ranks": ranks, "error": "no episodes scored (all unsupported kernel schema)"}

    # Group by N (test-set size) -- fixed in practice (N_min==N_max in
    # gp_tasks.yaml) but this stays correct if a caller varies N per episode.
    by_N: dict[int, list[torch.Tensor]] = {}
    for ep, post in pairs:
        n_test = int(ep["x_norm_test"].shape[0])
        by_N.setdefault(n_test, []).append(post["R_post"].float())

    jitter = float(dcfg.cfg.model.get("sigma_jitter", 1e-4))
    per_rank = []
    for r in ranks:
        losses_all = []
        for n_test, R_list in by_N.items():
            r_eff = min(r, n_test - 1)
            R_batch = torch.stack(R_list, dim=0)
            per_ep, _ = fit_rank_ceiling(R_batch, r_eff, steps=steps, lr=lr, jitter=jitter, device=dcfg.device)
            losses_all.append(per_ep.numpy())
        losses = np.concatenate(losses_all)
        per_rank.append({
            "rank": r,
            "ceiling_copula_nll_per_point": {
                "mean": float(losses.mean()), "std": float(losses.std()),
                "min": float(losses.min()), "max": float(losses.max()),
            },
        })

    return {
        "ranks": ranks, "n_episodes_scored": len(pairs), "steps": steps, "lr": lr,
        "per_rank": per_rank,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--ranks", default=",".join(str(x) for x in RANK_SWEEP_DEFAULT))
    p.add_argument("--steps", type=int, default=300, help="Adam steps per rank/N-group fit")
    p.add_argument("--lr", type=float, default=0.05)
    args = p.parse_args()

    dcfg = build_config(
        overrides=args.override, model_preset=args.model, n_episodes=args.n_episodes,
        ckpt=args.ckpt, device=args.device, seed=args.seed, run_id=args.run_id,
    )
    ranks = [int(x) for x in args.ranks.split(",")]
    result = run(dcfg, ranks=ranks, steps=args.steps, lr=args.lr)

    if "error" in result:
        print(result["error"])
        return

    print(f"Fit on {result['n_episodes_scored']} episodes, {args.steps} Adam steps/fit\n")
    print(f"{'rank':>6} {'ceiling copula NLL/pt':>22} {'std':>8}")
    for row in result["per_rank"]:
        c = row["ceiling_copula_nll_per_point"]
        print(f"{row['rank']:>6} {c['mean']:>22.4f} {c['std']:>8.4f}")
    print(
        "\nCompare against observed model copula NLL at matching rank (from "
        "train/val logs) -- if the model's gap-to-oracle tracks this ceiling, "
        "rank is the binding constraint; if the model is far ABOVE this "
        "ceiling, something else (S2 PIT distortion, S6 optimization) is."
    )

    path = common.save_stage_result(dcfg, "s1_rank_ceiling", result)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
