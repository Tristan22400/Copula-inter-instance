"""Lambda sweep of the SCM + GP residual prior: dependence strength, discards and throughput.

For each residual share lambda in LAMBDAS, draws EPISODES SCM episodes from the composed
training config (conf/config.yaml + Hydra overrides from the command line) with lambda held
fixed, and prints a markdown table of

    I        = mean over episodes of -1/2 log det R*   (copula information, nats per episode)
    |offdiag| = mean |R*_ij|, i != j
    discard  = fraction of requested episodes the generator dropped (Cholesky, degenerate z,
               SCM tables that lost a feature)
    eps/s    = generation throughput (synthesis + analytic PIT; no marginal model)

with R* the conditional oracle's posterior correlation corr(Sigma_{Q|C}) (gp_analytical_posterior)
and, as a reference row, the current GP prior (scm_residual disabled) on the same shapes and CPU.

    python -m debug.scm_residual_sweep experiment=scm_residual
    python -m debug.scm_residual_sweep experiment=scm_residual data.N_min=64 data.N_max=64 seed=3
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any

import torch
from omegaconf import DictConfig

from copula_inter.config_path import compose_config, config_dir, merge_configs
from copula_inter.data.gp_episodes import _generate_gp_batch_raw
from copula_inter.marginal.pit import gp_analytical_posterior

LAMBDAS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
EPISODES = 200
CALL_SIZE = 50
# Beta(c lambda, c (1 - lambda)) with this c pins lambda to ~1e-4.
_PIN_CONCENTRATION = 1e8


def _lambda_overrides(lam: float) -> dict[str, float]:
    """scm_residual keys that make every episode's residual share equal lam."""
    if lam <= 0.0:
        return {"lambda_p0": 1.0, "lambda_p1": 0.0}
    if lam >= 1.0:
        return {"lambda_p0": 0.0, "lambda_p1": 1.0}
    return {
        "lambda_p0": 0.0,
        "lambda_p1": 0.0,
        "lambda_beta_a": _PIN_CONCENTRATION * lam,
        "lambda_beta_b": _PIN_CONCENTRATION * (1.0 - lam),
    }


def _sweep_row(cfg: DictConfig, base_seed: int) -> dict[str, Any]:
    """Generate EPISODES episodes (in CALL_SIZE calls) and summarize their R*."""
    infos, offdiags, shares = [], [], []
    requested = returned = 0
    elapsed = 0.0
    call = 0
    while returned < EPISODES and call < 4 * EPISODES // CALL_SIZE:
        cfg.seed = base_seed + 1000 * call
        start = time.perf_counter()
        episodes = _generate_gp_batch_raw(cfg, CALL_SIZE, "cpu", return_kernel_metadata=True)
        elapsed += time.perf_counter() - start
        requested += CALL_SIZE
        returned += len(episodes)
        call += 1
        for ep in episodes:
            R = gp_analytical_posterior(ep)["R_post"].double()
            N = R.shape[0]
            infos.append(-0.5 * float(torch.linalg.slogdet(R).logabsdet))
            if "lambda_resid" in ep:
                shares.append(float(ep["lambda_resid"]))
            offdiags.append(float((R - torch.eye(N, dtype=R.dtype)).abs().sum()) / max(N * (N - 1), 1))
    n = max(len(infos), 1)
    q = torch.tensor(shares).quantile(torch.tensor([0.1, 0.5, 0.9])).tolist() if shares else [math.nan] * 3
    return {
        "lam_q": q,
        "I": sum(infos) / n,
        "offdiag": sum(offdiags) / n,
        "discard": 1.0 - returned / max(requested, 1),
        "eps_per_s": returned / max(elapsed, 1e-9),
    }


def main() -> None:
    cfg = compose_config(config_dir(__file__), "config", sys.argv[1:])
    base_seed = int(cfg.seed)
    rows: list[tuple[str, dict[str, Any]]] = []

    gp_cfg = merge_configs(cfg)
    gp_cfg.data.scm_residual.enabled = False
    rows.append(("GP prior (current)", _sweep_row(gp_cfg, base_seed)))
    for lam in LAMBDAS:
        lam_cfg = merge_configs(cfg)
        lam_cfg.data.scm_residual.enabled = True
        lam_cfg.data.scm_residual.prob = 1.0
        lam_cfg.data.scm_residual.amplitude_mode = "coupled"  # lambda pinned by the coupled mixture
        for key, value in _lambda_overrides(lam).items():
            lam_cfg.data.scm_residual[key] = value
        rows.append((f"SCM lambda={lam:g}", _sweep_row(lam_cfg, base_seed)))
    for mode in ("coupled", "decoupled"):
        prior_cfg = merge_configs(cfg)
        prior_cfg.data.scm_residual.enabled = True
        prior_cfg.data.scm_residual.prob = 1.0
        prior_cfg.data.scm_residual.amplitude_mode = mode
        rows.append((f"SCM {mode} prior", _sweep_row(prior_cfg, base_seed)))

    P, N = int(cfg.data.P_max), int(cfg.data.N_max)
    print(
        f"\nP={P}, N={N}, prior_type={cfg.data.scm_residual.prior_type}, "
        f"{EPISODES} episodes per row, seed={base_seed}, torch threads={torch.get_num_threads()}\n"
    )
    print(
        "| prior | lambda p10 / p50 / p90 | I = -1/2 log det R* (nats) | mean abs offdiag R* | discarded | episodes/s |"
    )
    print("|---|---|---:|---:|---:|---:|")
    for name, r in rows:
        info = "nan" if math.isnan(r["I"]) else f"{r['I']:.2f}"
        shares = "-" if math.isnan(r["lam_q"][0]) else " / ".join(f"{v:.2f}" for v in r["lam_q"])
        print(f"| {name} | {shares} | {info} | {r['offdiag']:.4f} | {100 * r['discard']:.1f}% | {r['eps_per_s']:.1f} |")


if __name__ == "__main__":
    main()
