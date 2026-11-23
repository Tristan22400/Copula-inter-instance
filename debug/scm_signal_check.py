"""Is there copula signal to learn? Per-episode dependence left in a TabICL marginal's z_test.

For each prior (GP; SCM + GP residual) and each frozen marginal (the config's PIT
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


SHRINK_GRID = [i / 10 for i in range(11)]


def _standardize(z: torch.Tensor) -> torch.Tensor:
    """z with the episode's own mean and std removed (marginal bias and scale miscalibration)."""
    z = z.double()
    return (z - z.mean()) / z.std().clamp(min=1e-8)


def _best_shrink(R: torch.Tensor, z: torch.Tensor) -> tuple[float, float]:
    """max_t gain of R_t = (1 - t) I + t R on z, and the argmax t (R's shape with a learned strength)."""
    eye = torch.eye(R.shape[0], dtype=torch.float64)
    gains = [_copula_gain((1 - t) * eye + t * R.double(), z) for t in SHRINK_GRID]
    best = max(range(len(gains)), key=lambda i: gains[i])
    return gains[best], SHRINK_GRID[best]


def _nn_autocorr(x: torch.Tensor, z: torch.Tensor) -> float:
    """mean z_i z_nn(i) of standardized z over each row's nearest neighbour in x (0 under independence).

    Shape-agnostic: positive when nearby rows (in the features the copula sees) have similar z,
    whatever the dependence's shape. Its null sd is about 1 / sqrt(n).
    """
    zs = _standardize(z)
    dist = torch.cdist(x.double(), x.double())
    dist.fill_diagonal_(float("inf"))
    return float((zs * zs[dist.argmin(dim=1)]).mean())


def _prior_cfgs(cfg: DictConfig) -> dict[str, DictConfig]:
    gp = merge_configs(cfg)
    gp.data.scm_residual.enabled = False
    out = {"GP prior": gp}
    c = merge_configs(cfg)
    c.data.scm_residual.enabled = True
    c.data.scm_residual.prob = 1.0
    out["SCM + GP residual"] = c
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


def _summary(rows: list[dict[str, float]], n_test: int = 256) -> str:
    def col(key: str) -> torch.Tensor:
        return torch.tensor([r[key] for r in rows])

    nn_test = col("nn_test")
    null_sd = n_test**-0.5
    return (
        f"{len(rows)} | {col('info').mean():.3f} | {col('delta_oracle').mean():.3f} | {col('delta').mean():.3f} | "
        f"{col('delta_std').mean():.3f} | {col('shrink_gain').mean():.3f} (t*={col('shrink_t').mean():.2f}) | "
        f"{100 * (col('shrink_gain') > SIGNAL_THRESHOLD).float().mean():.0f}% | "
        f"{nn_test.mean():+.3f} | {100 * (nn_test > 2 * null_sd).float().mean():.0f}% | {col('nn_train').mean():+.3f}"
    )


def main() -> None:
    cfg = compose_config(config_dir(__file__), "config", sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    marginals = {"config PIT": resolve_pit_ckpt(cfg), "pretrained v2": PRETRAINED_V2}
    priors = _prior_cfgs(cfg)
    cols = (
        "episodes | I/N | gain analytic z | gain marginal z | gain std z | best shrink gain (t*) | shrink > 0.01 | "
        "nn-corr z_test | nn-corr > 2/sqrt(N) | nn-corr z_train"
    )
    header = f"| prior | marginal | {cols} |\n|---|---|" + "---:|" * 10
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
                z_std = _standardize(ep["z_test"])
                shrink_gain, shrink_t = _best_shrink(R, z_std)
                rows.append(
                    {
                        "delta": _copula_gain(R, ep["z_test"]),
                        "delta_std": _copula_gain(R, z_std),
                        "shrink_gain": shrink_gain,
                        "shrink_t": shrink_t,
                        "nn_test": _nn_autocorr(ep["x_norm_test"], ep["z_test"]),
                        "nn_train": _nn_autocorr(ep["x_norm_train"], ep["z_train"]),
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
    print(f"| prior | marginal | lambda bin | {cols} |")
    print("|---|---|---|" + "---:|" * 10)
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
