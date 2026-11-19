"""Is there copula signal to learn? Per-episode dependence left in a TabICL marginal's z_test.

For each prior (GP; SCM coupled; SCM decoupled) and each frozen marginal (the config's PIT
checkpoint and the pretrained TabICLv2), draws EPISODES episodes with the marginal's PIT and reports

    delta = -copula_nll(z_test; R_post) / N   (nats per query point)

the gain over the independence copula of a copula that knew the conditional oracle's posterior
correlation R_post, scored on the marginal's own z_test. delta > 0 means the marginal leaves
dependence a copula can use; delta_oracle is the same on the analytic z (the ceiling when the
marginal is exact). Also prints I / N = -1/2 log det R_post / N, and, for SCM episodes, the same
by residual-share bin. Ends with a cProfile of CPU episode generation for each prior.

    python -m debug.scm_signal_check experiment=scm_residual model=copula_nano
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
from typing import Any

import torch
from omegaconf import DictConfig

from copula_inter.config_path import compose_config, config_dir, merge_configs
from copula_inter.data.gp_episodes import generate_gp_batch
from copula_inter.marginal.pit import gp_analytical_pit, gp_analytical_posterior, load_tabicl, resolve_pit_ckpt

EPISODES = 128
CALL_SIZE = 32
PRETRAINED_V2 = "tabicl-regressor-v2-20260212.ckpt"
SIGNAL_THRESHOLD = 0.01  # nats per query point
LAMBDA_BINS = ("lambda=0", "0<lambda<=0.5", "0.5<lambda<1", "lambda=1")


def _lambda_bin(lam: float) -> str:
    if lam <= 0.0:
        return LAMBDA_BINS[0]
    if lam >= 1.0:
        return LAMBDA_BINS[3]
    return LAMBDA_BINS[1] if lam <= 0.5 else LAMBDA_BINS[2]


def _copula_gain(R: torch.Tensor, z: torch.Tensor) -> float:
    """-copula NLL / N of z under correlation R: [z'z - log det R - z' R^-1 z] / (2N)."""
    R, z = R.double(), z.double()
    L = torch.linalg.cholesky(R + 1e-9 * torch.eye(R.shape[0], dtype=R.dtype))
    sol = torch.cholesky_solve(z.unsqueeze(-1), L).squeeze(-1)
    logdet = 2.0 * L.diagonal().log().sum()
    return float((z @ z - logdet - z @ sol) / (2 * z.shape[0]))


def _prior_cfgs(cfg: DictConfig) -> dict[str, DictConfig]:
    gp = merge_configs(cfg)
    gp.data.scm_residual.enabled = False
    out = {"GP prior": gp}
    for mode in ("coupled", "decoupled"):
        c = merge_configs(cfg)
        c.data.scm_residual.enabled = True
        c.data.scm_residual.prob = 1.0
        c.data.scm_residual.amplitude_mode = mode
        out[f"SCM {mode}"] = c
    return out


def _episodes(cfg: DictConfig, marginal: Any, device: str) -> list[dict]:
    k_folds = int(cfg.data.get("z_train_tabicl_k_folds", 5))
    base_seed = int(cfg.seed)
    eps: list[dict] = []
    call = 0
    while len(eps) < EPISODES:
        cfg.seed = base_seed + 7919 * call
        eps += generate_gp_batch(
            cfg,
            CALL_SIZE,
            "cpu",
            return_kernel_metadata=True,
            tabicl_model=marginal,
            tabicl_k_folds=k_folds,
            pit_device=device,
        )
        call += 1
    cfg.seed = base_seed
    return eps[:EPISODES]


def _summary(rows: list[dict[str, float]]) -> str:
    d = torch.tensor([r["delta"] for r in rows])
    d_or = torch.tensor([r["delta_oracle"] for r in rows])
    info = torch.tensor([r["info"] for r in rows])
    q = d.quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()
    return (
        f"{len(rows)} | {info.mean():.3f} | {d_or.mean():.3f} | {d.mean():.3f} | "
        f"{q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f} | {100 * (d > SIGNAL_THRESHOLD).float().mean():.0f}% | "
        f"{100 * (d > 0).float().mean():.0f}%"
    )


def main() -> None:
    cfg = compose_config(config_dir(__file__), "config", sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    marginals = {"config PIT": resolve_pit_ckpt(cfg), "pretrained v2": PRETRAINED_V2}
    priors = _prior_cfgs(cfg)
    header = (
        "| prior | marginal | episodes | I/N (oracle) | delta on analytic z | delta on marginal z | "
        "delta p10 / p50 / p90 | delta > 0.01 | delta > 0 |\n|---|---|---:|---:|---:|---:|---|---:|---:|"
    )
    lines, bin_lines = [header], []
    for m_name, ckpt in marginals.items():
        assert ckpt is not None
        marginal = load_tabicl(str(ckpt), device)
        for p_name, p_cfg in priors.items():
            rows = []
            for ep in _episodes(p_cfg, marginal, device):
                post = gp_analytical_posterior(ep)
                R = post["R_post"]
                N = R.shape[0]
                rows.append(
                    {
                        "delta": _copula_gain(R, ep["z_test"]),
                        "delta_oracle": _copula_gain(R, gp_analytical_pit(ep)["z_test"].reshape(-1)),
                        "info": -0.5 * float(torch.linalg.slogdet(R.double()).logabsdet) / N,
                        "lam": float(ep.get("lambda_resid", torch.tensor(1.0))),
                    }
                )
            lines.append(f"| {p_name} | {m_name} | {_summary(rows)} |")
            if p_name.startswith("SCM"):
                for label in LAMBDA_BINS:
                    sel = [r for r in rows if _lambda_bin(r["lam"]) == label]
                    if sel:
                        bin_lines.append(f"| {p_name} | {m_name} | {label} | {_summary(sel)} |")
            print(lines[-1], flush=True)
        del marginal
        torch.cuda.empty_cache()

    print("\n## Copula signal per prior and marginal\n")
    print("\n".join(lines))
    print("\n## By residual share (SCM priors)\n")
    print(
        "| prior | marginal | lambda bin | episodes | I/N | delta analytic | delta marginal | p10/p50/p90 | >0.01 | >0 |"
    )
    print("|---|---|---|---:|---:|---:|---:|---|---:|---:|")
    print("\n".join(bin_lines))

    print("\n## CPU generation profile (32 episodes, no marginal)\n")
    for p_name, p_cfg in priors.items():
        prof = cProfile.Profile()
        prof.enable()
        generate_gp_batch(p_cfg, CALL_SIZE, "cpu", return_kernel_metadata=False)
        prof.disable()
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(18)
        print(f"### {p_name}\n```\n{buf.getvalue()}\n```")


if __name__ == "__main__":
    main()
