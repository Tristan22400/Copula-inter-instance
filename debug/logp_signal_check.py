"""Copula signal of the TabICLv2 log-uniform-P dataset, by context-size bin.

Part 1 draws fresh episodes from the generator config of data/tabiclv2_logp (same overrides,
kernel metadata kept, PIT by TabICLv2) and reports per P bin: I/N = -1/2 log det R_post / N,
the oracle copula gain on analytic z (H, the headroom), the same R_post scored on the TabICLv2
z_test, and the nearest-neighbour z_test correlation. Part 2 reads the saved shards (no oracle
stored) and reports the nearest-neighbour z_test correlation per P bin (null sd ~ 1/sqrt(N)).

    python -m debug.logp_signal_check experiment=scm_residual data.scm_residual.prior_P_min=10 ...
"""

from __future__ import annotations

import glob
import math
import os
import sys

import torch

from copula_inter.config_path import compose_config, config_dir
from copula_inter.data.gp_episodes import generate_gp_batch
from copula_inter.marginal.pit import gp_analytical_pit, gp_analytical_posterior, load_tabicl
from debug.scm_signal_check import _copula_gain, _nn_autocorr

CALLS = int(os.environ.get("LOGP_SIGNAL_CALLS", "64"))
CALL_SIZE = 4
P_BINS = ((10, 32), (32, 100), (100, 316), (316, 1001))
SHARD_DIR = "/home/trmartin/thoth_storage/copula-inter/data/tabiclv2_logp/pit"
SHARDS = 60


def _bin(P: int) -> str:
    for lo, hi in P_BINS:
        if lo <= P < hi:
            return f"{lo}-{hi - 1}"
    return "other"


def _mean(rows: list[dict[str, float]], key: str) -> float:
    vals = [r[key] for r in rows if math.isfinite(r[key])]
    return sum(vals) / max(1, len(vals))


def main() -> None:
    cfg = compose_config(config_dir(__file__), "config", sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    marginal = load_tabicl("tabicl-regressor-v2-20260212.ckpt", device)
    rows: list[dict[str, float]] = []
    base_seed = int(cfg.seed) + 31337
    for call in range(CALLS):
        cfg.seed = base_seed + 7919 * call
        for ep in generate_gp_batch(
            cfg,
            CALL_SIZE,
            "cpu",
            return_kernel_metadata=True,
            tabicl_model=marginal,
            tabicl_k_folds=5,
            pit_device=device,
        ):
            R = gp_analytical_posterior(ep)["R_post"]
            N = R.shape[0]
            rows.append(
                {
                    "P": float(ep["x_norm_train"].shape[0]),
                    "N": float(N),
                    "info": -0.5 * float(torch.linalg.slogdet(R.double()).logabsdet) / N,
                    "H": _copula_gain(R, gp_analytical_pit(ep)["z_test"].reshape(-1)),
                    "delta": _copula_gain(R, ep["z_test"]),
                    "nn": _nn_autocorr(ep["x_norm_test"], ep["z_test"]),
                    "lam": float(ep.get("lambda_resid", torch.tensor(1.0))),
                }
            )
        print(f"call {call + 1}/{CALLS}: {len(rows)} episodes", flush=True)

    print("\n## Fresh episodes from the generator config (oracle known)\n")
    print(
        "| P bin | episodes | mean N | I/N | oracle gain H (analytic z) | gain on TabICLv2 z | nn-corr z_test | H > 0.01 |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for lo, hi in (*P_BINS, (10, 1001)):
        sel = [r for r in rows if lo <= r["P"] < hi]
        if not sel:
            continue
        frac = 100 * sum(r["H"] > 0.01 for r in sel) / len(sel)
        label = "all" if (lo, hi) == (10, 1001) else f"{lo}-{hi - 1}"
        print(
            f"| {label} | {len(sel)} | {_mean(sel, 'N'):.0f} | {_mean(sel, 'info'):.3f} | {_mean(sel, 'H'):.3f} | "
            f"{_mean(sel, 'delta'):.3f} | {_mean(sel, 'nn'):+.3f} | {frac:.0f}% |"
        )

    print("\n## Saved shards (model-free: nearest-neighbour z_test correlation)\n")
    files = sorted(glob.glob(f"{SHARD_DIR}/shard_*.pt"))
    step = max(1, len(files) // SHARDS)
    srows: dict[str, list[float]] = {}
    shapes: dict[str, int] = {}
    for path in files[::step][:SHARDS]:
        episodes = torch.load(path, map_location="cpu", weights_only=False)
        P = int(episodes[0]["x_norm_train"].shape[0])
        assert len({(e["x_norm_train"].shape, e["x_norm_test"].shape) for e in episodes}) == 1, path
        key = _bin(P)
        shapes[key] = shapes.get(key, 0) + 1
        for e in episodes[:32]:
            srows.setdefault(key, []).append(_nn_autocorr(e["x_norm_test"], e["z_test"]))
    print("| P bin | shards | episodes | nn-corr z_test |")
    print("|---|---:|---:|---:|")
    for lo, hi in P_BINS:
        key = f"{lo}-{hi - 1}"
        if key in srows:
            v = srows[key]
            print(f"| {key} | {shapes[key]} | {len(v)} | {sum(v) / len(v):+.3f} |")


if __name__ == "__main__":
    main()
