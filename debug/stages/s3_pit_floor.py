"""Debug stage S3: attainable copula floor when TabICL's PIT is the marginal.

Per episode, draw M y_test from the exact GP posterior, PIT each through one
TabICL forward, and report: R_z (Ledoit-Wolf-shrunk correlation of the PIT z)
vs R_post; floor, the held-out copula NLL of R_z; the copula NLL of R_post on
the PIT z; and the rank-r ceiling fitted to R_z (population fit, so it can
look better than floor).

Usage:
    python debug/run_debug.py s3
    python debug/stages/s3_pit_floor.py --n-episodes 20 --m-samples 2048
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from debug import common
from debug.config import DebugConfig, add_common_args, build_config
from debug.stages.s1_rank_ceiling import fit_rank_ceiling


def _build_quantile_dist(tabicl_model, x_train: torch.Tensor, y_train_scaled: torch.Tensor, x_test: torch.Tensor):
    """One TabICL forward -> quantile distribution with batch_shape (N,)."""
    X_concat = torch.cat([x_train, x_test], dim=0).unsqueeze(0)  # (1, P+N, d_x)
    y_train_batch = y_train_scaled.unsqueeze(0)                   # (1, P)
    with torch.no_grad():
        logits = tabicl_model(X_concat, y_train_batch)            # (1, N, Q)
    logits = logits.to(x_test.device)
    N, Q = logits.shape[1], logits.shape[-1]
    return tabicl_model.quantile_dist(logits.reshape(N, Q))


def sample_and_pit(tabicl_model, episode: dict, post: dict, M: int, device: str) -> torch.Tensor:
    """z_samples (N, M): M posterior draws at the episode's test points, PIT'd through TabICL."""
    from copula_inter.loss import _safe_cholesky
    from copula_inter.pit import _probit

    x_train = episode["x_norm_train"].to(device)
    y_train = episode["y_train"].to(device)
    x_test = episode["x_norm_test"].to(device)
    y_mean = y_train.mean()
    y_std = y_train.std().clamp(min=1e-8)
    y_train_scaled = (y_train - y_mean) / y_std

    dist = _build_quantile_dist(tabicl_model, x_train, y_train_scaled, x_test)

    mu_post = post["mu_post"].to(device).float()
    Sigma_post = post["Sigma_post"].to(device).float()
    N = mu_post.shape[0]
    L = _safe_cholesky(Sigma_post.unsqueeze(0)).squeeze(0)
    eps = torch.randn(N, M, device=device)
    y_samples = mu_post.unsqueeze(-1) + L @ eps            # (N, M), raw y-space

    y_samples_scaled = (y_samples - y_mean) / y_std
    with torch.no_grad():
        u = dist.cdf(y_samples_scaled)                      # (N, M)
    return _probit(u)                                        # (N, M)


def _shrunk_correlation(z_fit: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Ledoit-Wolf-shrunk correlation of z_fit (N, m_fit); returns (R_z, shrinkage coefficient)."""
    from sklearn.covariance import LedoitWolf

    z_np = z_fit.detach().cpu().numpy().T  # (m_fit, N) -- sklearn's (n_samples, n_features)
    lw = LedoitWolf().fit(z_np)
    cov = lw.covariance_
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    R = cov / np.outer(d, d)
    return torch.from_numpy(R).to(z_fit.device, dtype=z_fit.dtype), float(lw.shrinkage_)


def run(dcfg: DebugConfig, M: int = 2048, m_fit: int = None, rank: int = 32) -> dict:
    from copula_inter.loss import oracle_copula_nll

    m_fit = m_fit or M // 2
    tabicl_model = common.load_frozen_tabicl(dcfg)
    pairs = common.collect_posteriors(dcfg, dcfg.n_episodes)
    if not pairs:
        return {"error": "no episodes scored (all unsupported kernel schema)"}

    jitter = float(dcfg.cfg.model.get("sigma_jitter", 1e-4))
    per_episode = []
    for ep, post in pairs:
        z_samples = sample_and_pit(tabicl_model, ep, post, M, dcfg.device)  # (N, M)
        N = z_samples.shape[0]
        z_fit, z_eval = z_samples[:, :m_fit], z_samples[:, m_fit:]
        M_eval = z_eval.shape[1]

        R_z, shrinkage = _shrunk_correlation(z_fit)  # (N, N)
        R_post = post["R_post"].float().to(dcfg.device)

        z_eval_t = z_eval.T.contiguous()  # (M_eval, N)
        test_mask = torch.ones(M_eval, N, dtype=torch.bool, device=dcfg.device)
        floor_nll = oracle_copula_nll(R_z.unsqueeze(0).expand(M_eval, -1, -1), z_eval_t, test_mask)
        r_post_on_pit_nll = oracle_copula_nll(R_post.unsqueeze(0).expand(M_eval, -1, -1), z_eval_t, test_mask)

        ri, ci = torch.triu_indices(N, N, offset=1)
        off_z = R_z[ri, ci].detach().cpu().numpy()
        off_post = R_post[ri, ci].detach().cpu().numpy()
        pearson = float(np.corrcoef(off_z, off_post)[0, 1]) if len(off_z) > 1 else float("nan")

        ceiling_per_ep, _ = fit_rank_ceiling(R_z.unsqueeze(0), min(rank, N - 1), jitter=jitter, device=dcfg.device)

        per_episode.append({
            "n_test": N,
            "floor_copula_nll_on_Rz": float(floor_nll.item()),
            "copula_nll_of_Rpost_on_pit_z": float(r_post_on_pit_nll.item()),
            "rank_ceiling_on_Rz": float(ceiling_per_ep.item()),
            "Rz_shrinkage_coefficient": shrinkage,
            "Rz_vs_Rpost_pearson": pearson,
            "Rz_vs_Rpost_mae": float(np.abs(off_z - off_post).mean()),
        })

    def _mean(key):
        return float(np.mean([e[key] for e in per_episode]))

    return {
        "M": M, "m_fit": m_fit, "rank": rank, "n_episodes_scored": len(per_episode),
        "summary": {
            "floor_copula_nll_on_Rz_mean": _mean("floor_copula_nll_on_Rz"),
            "copula_nll_of_Rpost_on_pit_z_mean": _mean("copula_nll_of_Rpost_on_pit_z"),
            "rank_ceiling_on_Rz_mean": _mean("rank_ceiling_on_Rz"),
            "Rz_shrinkage_coefficient_mean": _mean("Rz_shrinkage_coefficient"),
            "Rz_vs_Rpost_pearson_mean": _mean("Rz_vs_Rpost_pearson"),
            "Rz_vs_Rpost_mae_mean": _mean("Rz_vs_Rpost_mae"),
        },
        "per_episode": per_episode,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--m-samples", type=int, default=2048, help="Posterior draws per episode (default: 2048)")
    p.add_argument("--rank", type=int, default=32)
    args = p.parse_args()

    dcfg = build_config(
        overrides=args.override, model_preset=args.model, n_episodes=args.n_episodes,
        ckpt=args.ckpt, device=args.device, seed=args.seed, run_id=args.run_id,
    )
    result = run(dcfg, M=args.m_samples, rank=args.rank)
    if "error" in result:
        print(result["error"])
        return

    s = result["summary"]
    print(f"Scored {result['n_episodes_scored']} episodes, M={result['M']} draws/episode (fit={result['m_fit']})\n")
    print(f"  floor copula NLL/pt on R_z (held-out draws)      : {s['floor_copula_nll_on_Rz_mean']:.4f}")
    print(f"  copula NLL/pt of R_post scored on PIT z          : {s['copula_nll_of_Rpost_on_pit_z_mean']:.4f}")
    print(f"  rank-{result['rank']} ceiling recomputed on R_z          : {s['rank_ceiling_on_Rz_mean']:.4f}")
    print(f"  R_z Ledoit-Wolf shrinkage coefficient (mean)     : {s['Rz_shrinkage_coefficient_mean']:.4f}")
    print(f"  R_z vs R_post off-diag Pearson (mean over eps)   : {s['Rz_vs_Rpost_pearson_mean']:.4f}")
    print(f"  R_z vs R_post off-diag MAE (mean over eps)       : {s['Rz_vs_Rpost_mae_mean']:.4f}")
    print(
        "\nDecomposition: observed_gap ~= (rank-r ceiling on R_z - floor) "
        "  [capacity loss on the distorted target]"
        "\n             + (floor - S0's oracle_posterior_copula_nll)     "
        "  [PIT distortion loss]"
    )

    path = common.save_stage_result(dcfg, "s3_pit_floor", result)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
