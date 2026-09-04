"""s1b_rank_gap_decomp.py — what rank COSTS, and whether a different basis
would cost less. The companion to s1: s1 reports the rank-r ceiling in
absolute copula-NLL units, this one reports it as a GAP to the exact GP
posterior, which is the number that answers "is rank the binding
constraint on this episode distribution?".

Three numbers per rank r, all copula-NLL nats/point on synthetic episodes:
  oracle   = 0.5*logdet(R_post)/N          (Bayes floor: Sigma == R_post)
  ceiling  = best rank-r covnorm fit       (s1_rank_ceiling.fit_rank_ceiling)
  gap      = ceiling - oracle              (what rank r forfeits)

`oracle` is where the copula NLL bottoms out because at Sigma == R_post the
trace term collapses (tr(R^-1 R) == N == tr(R), R_post being a correlation
matrix), leaving 0.5*logdet(R_post)/N. It is negative, and its magnitude is
the TOTAL copula signal available in the episode distribution -- worth
reading first, since it bounds what any correlation model can ever win here.

Also reports each rank's share of R_post's eigenvalue mass. Note that mass
and NLL gap are NOT interchangeable: covnorm's free diagonal absorbs the
residual, so eigenvalue mass badly overstates the damage of truncation
(measured here: 79% mass at r=128 yet only 0.3% of the oracle forfeited).
When the two disagree, the NLL gap is the one that means anything.

The decomposition variant fits Sigma = covnorm(K_theta - K_su (K_uu+D)^-1
K_us + diag) instead: an exact sparse-GP Schur complement (PSD by
construction) over m learned inducing points, with an ARD RBF+Matern32
mixture kernel. That is the "prior kernel minus rank-m correction" basis,
motivated by the GP posterior's own algebra -- K_post = K_ss - K_sf K_ff^-1
K_fs is exactly a full-rank prior minus a rank-P correction, so a global
low-rank factor is being asked to represent a shape it does not have.
Scored by the identical objective, so the columns are directly comparable;
compare on `n_params` too, since a head has to EMIT these per episode.

Caveat on the decomposition: its kernel is a fixed RBF+Matern32 mixture,
so it carries an irreducible misspecification floor on episodes from other
families (periodic, RQ, polynomial). It plateaus rather than going to zero,
and that plateau is the kernel's fault, not the basis's.

Usage:
    python debug/run_debug.py s1b
    python debug/stages/s1b_rank_gap_decomp.py --n-episodes 128 --ranks 32,64,128
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC, os.path.join(_REPO_ROOT, "debug"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common
from config import add_common_args, build_config
from s1_rank_ceiling import fit_rank_ceiling


def oracle_nll(R: torch.Tensor) -> torch.Tensor:
    """0.5*logdet(R)/N per episode -- the copula NLL at Sigma == R_post."""
    L = torch.linalg.cholesky(R.double())
    logdet = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    return (0.5 * logdet / R.shape[-1]).float().cpu()


def spectrum_fraction(R: torch.Tensor, ranks) -> dict:
    ev = torch.linalg.eigvalsh(R.double()).flip(-1).clamp_min(0)  # (B,N) descending
    cum = ev.cumsum(-1) / ev.sum(-1, keepdim=True)
    return {r: float(cum[:, min(r, cum.shape[1]) - 1].mean()) for r in ranks}


# ---------------------------------------------------------------------------
# Decomposition: covnorm( K_theta - K_su (K_uu + D)^-1 K_us + diag(softplus s) )
# ---------------------------------------------------------------------------
def _ard_kernel(a: torch.Tensor, b: torch.Tensor, log_ls: torch.Tensor,
                log_out: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    """Sum of ARD-RBF and ARD-Matern3/2 with learned positive weights."""
    ls = log_ls.exp().clamp_min(1e-3)[:, None, None, :]   # (B,1,1,d)
    d2 = ((a.unsqueeze(-2) / ls - b.unsqueeze(-3) / ls) ** 2).sum(-1).clamp_min(0)
    d1 = d2.clamp_min(1e-12).sqrt()
    w = torch.softmax(mix, dim=0)
    k = (w[0] * torch.exp(-0.5 * d2)
         + w[1] * (1 + np.sqrt(3.0) * d1) * torch.exp(-np.sqrt(3.0) * d1))
    return log_out.exp() * k


def fit_decomposition(R_post: torch.Tensor, X: torch.Tensor, m: int, *,
                      steps: int = 400, lr: float = 0.05, jitter: float = 1e-4,
                      device: str = "cpu"):
    """Fit the sparse-GP-Schur basis to R_post by the same exact expected
    copula NLL s1 minimizes. X: (B, N, d) test inputs."""
    from loss import _safe_cholesky

    B, N, _ = R_post.shape
    d = X.shape[-1]
    R_post, X = R_post.to(device), X.to(device)
    trace_R = R_post.diagonal(dim1=-2, dim2=-1).sum(-1)

    idx = torch.stack([torch.randperm(N, device=device)[:m] for _ in range(B)])
    U = torch.gather(X, 1, idx.unsqueeze(-1).expand(B, m, d)).clone().requires_grad_(True)
    log_ls = torch.zeros(B, d, device=device, requires_grad=True)
    log_out = torch.zeros(B, 1, 1, device=device, requires_grad=True)
    mix = torch.zeros(2, device=device, requires_grad=True)
    a = torch.full((B, m), -2.0, device=device, requires_grad=True)   # inducing noise
    s = torch.zeros(B, N, device=device, requires_grad=True)          # diagonal
    opt = torch.optim.Adam([U, log_ls, log_out, mix, a, s], lr=lr)
    eye_N = torch.eye(N, device=device)

    def _loss():
        Kss = _ard_kernel(X, X, log_ls, log_out, mix)
        Ksu = _ard_kernel(X, U, log_ls, log_out, mix)
        Kuu = _ard_kernel(U, U, log_ls, log_out, mix)
        Kuu = Kuu + (torch.nn.functional.softplus(a) + 1e-4).diag_embed()
        Lu = _safe_cholesky(Kuu)
        V = torch.linalg.solve_triangular(Lu, Ksu.mT, upper=False)      # (B,m,N)
        Sig = (Kss - V.mT @ V
               + torch.nn.functional.softplus(s).diag_embed() + jitter * eye_N)
        dg = Sig.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8).rsqrt()
        Sig = Sig * dg.unsqueeze(-1) * dg.unsqueeze(-2)
        L = _safe_cholesky(Sig)
        logdet = 2.0 * L.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).log().sum(-1)
        tmp = torch.linalg.solve_triangular(L, R_post, upper=False)
        Xs = torch.linalg.solve_triangular(L.mT, tmp, upper=True)
        return 0.5 * (logdet + Xs.diagonal(dim1=-2, dim2=-1).sum(-1) - trace_R) / N

    for _ in range(steps):
        opt.zero_grad()
        _loss().mean().backward()
        opt.step()
    with torch.no_grad():
        out = _loss().detach().cpu()
    n_params = m * d + d + 1 + 2 + m + N
    return out, n_params


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--ranks", default="8,16,32,64,128,255")
    p.add_argument("--inducing", default="8,16,32,64")
    p.add_argument("--steps", type=int, default=600, help="Adam steps per fit")
    p.add_argument("--out", default=None, help="extra JSON copy (results always land in run_dir)")
    p.add_argument("--skip-decomp", action="store_true")
    args = p.parse_args()

    ranks = [int(x) for x in args.ranks.split(",")]
    inducings = [int(x) for x in args.inducing.split(",")]
    dcfg = build_config(overrides=args.override, model_preset=args.model,
                        n_episodes=args.n_episodes, ckpt=args.ckpt,
                        device=args.device, seed=args.seed, run_id=args.run_id)
    print(f"device={dcfg.device}  P=[{dcfg.cfg.data.P_min},{dcfg.cfg.data.P_max}]  "
          f"N=[{dcfg.cfg.data.N_min},{dcfg.cfg.data.N_max}]", flush=True)

    pairs = common.collect_posteriors(dcfg, args.n_episodes)
    if not pairs:
        print("no episodes")
        return
    print(f"scored {len(pairs)} episodes", flush=True)

    # Rank fits need only R_post -> group by N. The decomposition also needs
    # the test inputs, and d_features varies per shard (gp_tasks.yaml's
    # d_features_lognormal_*), so that groups by (N, d).
    by_N: dict[int, list] = {}
    by_Nd: dict[tuple, list] = {}
    for ep, post in pairs:
        R = post["R_post"].float().cpu()
        X = torch.as_tensor(ep["x_norm_test"]).float().cpu()
        n_test, d_feat = int(X.shape[0]), int(X.shape[1])
        by_N.setdefault(n_test, []).append(R)
        by_Nd.setdefault((n_test, d_feat), []).append((R, X))
    jitter = float(dcfg.cfg.model.get("sigma_jitter", 1e-4))
    print("d_features groups: "
          + ", ".join(f"d={d}:{len(v)}ep" for (_, d), v in sorted(by_Nd.items())), flush=True)

    results = {"n_episodes": len(pairs), "steps": args.steps,
               "P": [int(dcfg.cfg.data.P_min), int(dcfg.cfg.data.P_max)],
               "rows": [], "decomp": []}

    for n_test, R_list in by_N.items():
        R_batch = torch.stack(R_list, 0)
        orc = oracle_nll(R_batch).numpy()
        spec = spectrum_fraction(R_batch, ranks)
        results["oracle_nll_per_point"] = float(orc.mean())
        results["spectrum_frac"] = spec
        results["N"] = n_test
        print(f"\nN={n_test}  n_ep={len(R_list)}  "
              f"oracle copula NLL/pt = {orc.mean():.4f}", flush=True)
        print(f"{'rank':>6} {'ceiling':>10} {'gap':>10} {'gap%':>8} {'eig frac':>10} {'params':>9}")
        for r in ranks:
            r_eff = min(r, n_test - 1)
            ceil_, _ = fit_rank_ceiling(R_batch, r_eff, steps=args.steps, lr=0.05,
                                        jitter=jitter, device=dcfg.device)
            c = ceil_.numpy()
            gap = float((c - orc).mean())
            denom = abs(float(orc.mean()))
            pct = 100.0 * gap / denom if denom > 0 else float("nan")
            npar = n_test * r_eff + n_test
            print(f"{r:>6} {c.mean():>10.4f} {gap:>10.4f} {pct:>7.1f}% "
                  f"{spec[r]:>10.3f} {npar:>9}", flush=True)
            results["rows"].append({"rank": r, "rank_eff": r_eff, "ceiling": float(c.mean()),
                                    "gap": gap, "gap_pct_of_oracle": pct,
                                    "eig_frac": spec[r], "n_params": npar})

    if not args.skip_decomp:
        print("\n  decomposition: covnorm(K_theta - rank-m correction), pooled over d groups")
        print(f"{'m':>6} {'ceiling':>10} {'gap':>10} {'params':>9}")
        for m in inducings:
            ceils, gaps, pars = [], [], []
            for (n_test, d_feat), items in sorted(by_Nd.items()):
                Rb = torch.stack([a for a, _ in items], 0)
                Xb = torch.stack([b for _, b in items], 0)
                dl, npar = fit_decomposition(Rb, Xb, m, steps=args.steps,
                                             jitter=jitter, device=dcfg.device)
                dnp = dl.numpy()
                ceils.append(dnp)
                gaps.append(dnp - oracle_nll(Rb).numpy())
                pars.append(npar)
            c_all = np.concatenate(ceils)
            g_all = np.concatenate(gaps)
            npar_mean = int(np.mean(pars))
            print(f"{m:>6} {c_all.mean():>10.4f} {g_all.mean():>10.4f} {npar_mean:>9}", flush=True)
            results["decomp"].append({"m": m, "ceiling": float(c_all.mean()),
                                      "gap": float(g_all.mean()), "n_params": npar_mean})

    path = common.save_stage_result(dcfg, "s1b_rank_gap_decomp", results)
    print(f"\nSaved -> {path}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
