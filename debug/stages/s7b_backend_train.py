"""s7b_backend_train.py — actually train with a different marginal backend.

s7_backbone.py (S7a) diagnoses the z_train gap across backends on a FROZEN,
already-trained copula head. This stage instead trains fresh models under
different marginal backends from scratch, to see whether the plateau is
specific to TabICL's own PIT or shows up with any real (non-oracle)
marginal.

Scoped as a debug-local comparison trainer, not a src/data_gen.py
production knob: TabPFN's API is per-task sklearn fit/predict (K-fold PIT
means K separate .fit()+.predict() calls per episode, see
eval/spatial/marginal_backends.py::quantiles), not a single batched GPU
forward like TabICL's -- wiring that into the production live-generation
path (train.py, live_dataset.py, era5_live_dataset.py, generate_pit_dataset.py
all call generate_gp_batch) would need a much larger refactor across every
caller for a backend that's fundamentally 10-100x slower per episode. This
stage instead runs a smaller, reduced-scale comparison good enough to read
gap TRAJECTORIES off, not to produce a checkpoint worth shipping. If the
comparison looks informative, promoting "tabicl"/"tabpfn" to a real
data.marginal_backend config knob (mirroring cfg.data.z_train_source) is
the natural next step -- deliberately not done here.

Both backends' models start from the SAME initialization (torch.manual_seed
reset before each build_copula_transformer call) for a fair comparison. The
plain AdamW loop here (no Muon, no AMP) matches s4_overfit.py's convention
-- this is an exploratory comparison, not a production-parity run.

Usage:
    python debug/run_debug.py s7b --backends tabicl,tabpfn --steps 200
    python debug/stages/s7b_backend_train.py --backends tabicl,tabpfn --steps 200 --batch-size 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC, os.path.join(_REPO_ROOT, "debug")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common
from config import DebugConfig, add_common_args, build_config

DEFAULT_PROBS_N = 99  # coarser than TabICL's 999 -- TabPFN's per-fold .fit()+.predict() dominates wall-clock


def _tabicl_pit_batch(tabicl_model, episodes: list[dict], k_folds: int, device: str):
    """Same convention as debug/stages/s5_kfold.py::_pit_at_k -- reused
    inline (not imported) since it's a five-line wrapper and s5's version
    is documented in terms of the K-fold sweep, not the training loop."""
    from pit import run_pit_batched

    x_train = torch.stack([ep["x_norm_train"] for ep in episodes]).to(device)
    x_test = torch.stack([ep["x_norm_test"] for ep in episodes]).to(device)
    y_train = torch.stack([ep["y_train"] for ep in episodes]).to(device)
    y_test = torch.stack([ep["y_test"] for ep in episodes]).to(device)
    y_mean = y_train.mean(dim=1, keepdim=True)
    y_std = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
    out = run_pit_batched(
        tabicl_model, x_train, ((y_train - y_mean) / y_std).unsqueeze(-1), x_test,
        ((y_test - y_mean) / y_std).unsqueeze(-1), k_folds=k_folds,
    )
    z_train = out["z_train"].squeeze(-1).cpu()
    z_test = out["z_test"].squeeze(-1).cpu()
    log_pdf_test = (out["log_pdf_test"].squeeze(-1) - y_std.log()).cpu()
    return z_train, z_test, log_pdf_test


def _tabpfn_pit_episode(regressor, ep: dict, k_folds: int, probs_n: int, seed: int):
    """Per-episode TabPFN PIT via eval/spatial/marginal_backends.{quantiles,
    loo_pit} + eval/metrics/joint_nll.compute_pit -- reused, not
    reimplemented. Same y-scaling convention as data_gen.py's tabicl branch."""
    from eval.metrics.joint_nll import compute_pit
    from eval.spatial.marginal_backends import loo_pit, quantiles

    x_train = ep["x_norm_train"].numpy()
    x_test = ep["x_norm_test"].numpy()
    y_train = ep["y_train"].numpy()
    y_test = ep["y_test"].numpy()
    y_mean, y_std = y_train.mean(), max(y_train.std(), 1e-8)
    y_train_s, y_test_s = (y_train - y_mean) / y_std, (y_test - y_mean) / y_std

    probs = np.linspace(1.0 / (probs_n + 1), probs_n / (probs_n + 1), probs_n)
    z_train = loo_pit("tabpfn", regressor, x_train, y_train_s, probs, k_folds=k_folds, seed=seed)
    q_test = quantiles("tabpfn", regressor, x_train, y_train_s, x_test, probs, seed=seed)
    z_test, log_pdf = compute_pit(q_test, probs, y_test_s)
    log_pdf = log_pdf - np.log(y_std)  # Jacobian back to raw-y nats, same convention as data_gen.py
    return torch.from_numpy(z_train).float(), torch.from_numpy(z_test).float(), torch.from_numpy(log_pdf).float()


def _build_batch_for_backend(dcfg: DebugConfig, backend: str, n: int, seed_offset: int, tabicl_model=None,
                              tabpfn_regressor=None, k_folds: int = 5, probs_n: int = DEFAULT_PROBS_N):
    from dataset import collate_fn

    episodes = common.generate_episodes(dcfg, n, tabicl_model=None, seed_offset=seed_offset)
    if backend == "tabicl":
        z_train, z_test, log_pdf_test = _tabicl_pit_batch(tabicl_model, episodes, k_folds, dcfg.device)
        for i, ep in enumerate(episodes):
            ep["z_train"], ep["z_test"], ep["log_pdf_test"] = z_train[i], z_test[i], log_pdf_test[i]
    elif backend == "tabpfn":
        for i, ep in enumerate(episodes):
            zt, zte, lp = _tabpfn_pit_episode(tabpfn_regressor, ep, k_folds, probs_n, seed=seed_offset + i)
            ep["z_train"], ep["z_test"], ep["log_pdf_test"] = zt, zte, lp
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return {k: v.to(dcfg.device) for k, v in collate_fn(episodes).items()}


def _train_one_backend(dcfg: DebugConfig, backend: str, steps: int, batch_size: int, lr: float,
                        eval_every: int, n_eval: int, k_folds: int, probs_n: int):
    from loss import y_space_nll
    from model import build_copula_transformer, build_sigma

    torch.manual_seed(dcfg.seed)  # same init across backends for a fair comparison
    model = build_copula_transformer(dcfg.cfg).to(dcfg.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    jitter = float(dcfg.cfg.model.get("sigma_jitter", 1e-4))
    parametrization = str(dcfg.cfg.model.get("correlation_parametrization", "covnorm"))

    tabicl_model = common.load_frozen_tabicl(dcfg) if backend == "tabicl" else None
    tabpfn_regressor = None
    if backend == "tabpfn":
        from eval.spatial.marginal_backends import make_regressor

        tabpfn_regressor = make_regressor("tabpfn", device=dcfg.device)

    # Fixed eval set: same episodes across both backends and every checkpoint
    # (seed_offset far from the training stream), re-PIT'd per backend.
    eval_batch = _build_batch_for_backend(
        dcfg, backend, n_eval, seed_offset=9_999_999, tabicl_model=tabicl_model,
        tabpfn_regressor=tabpfn_regressor, k_folds=k_folds, probs_n=probs_n,
    )

    history = []
    t0 = time.time()
    for step in range(1, steps + 1):
        batch = _build_batch_for_backend(
            dcfg, backend, batch_size, seed_offset=step * 104_729, tabicl_model=tabicl_model,
            tabpfn_regressor=tabpfn_regressor, k_folds=k_folds, probs_n=probs_n,
        )
        out = model(batch)
        Sigma = build_sigma(out, dcfg.cfg, jitter=jitter, test_mask=batch["test_mask"])
        parts = y_space_nll(Sigma, batch["z_test"].float(), batch["log_pdf_test"].float(), batch["test_mask"])
        loss = parts["total"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                out_e = model(eval_batch)
                Sigma_e = build_sigma(out_e, dcfg.cfg, jitter=jitter, test_mask=eval_batch["test_mask"])
                eval_parts = y_space_nll(
                    Sigma_e, eval_batch["z_test"].float(), eval_batch["log_pdf_test"].float(), eval_batch["test_mask"],
                )
            model.train()
            history.append({
                "step": step, "train_loss": float(loss.item()),
                "eval_total": float(eval_parts["total"].item()),
                "eval_copula": float(eval_parts["copula"].item()),
                "eval_marginal": float(eval_parts["marginal"].item()),
                "elapsed_s": time.time() - t0,
            })
            print(f"  [{backend:>7}] step {step:>5}/{steps}  train_loss={loss.item():.4f}  "
                  f"eval_total={eval_parts['total'].item():.4f}  eval_copula={eval_parts['copula'].item():.4f}")

    return history


def run(dcfg: DebugConfig, backends: list[str], steps: int, batch_size: int, lr: float,
        eval_every: int, n_eval: int, k_folds: int, probs_n: int) -> dict:
    result = {}
    for backend in backends:
        print(f"\n=== training backend={backend} ===")
        result[backend] = _train_one_backend(
            dcfg, backend, steps=steps, batch_size=batch_size, lr=lr,
            eval_every=eval_every, n_eval=n_eval, k_folds=k_folds, probs_n=probs_n,
        )
    return {"backends": backends, "steps": steps, "batch_size": batch_size, "history": result}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--backends", default="tabicl,tabpfn")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--n-eval", type=int, default=8, help="Fixed eval-set episode count")
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--probs-n", type=int, default=DEFAULT_PROBS_N, help="Quantile grid size for tabpfn (default 99; TabICL always uses its own 999)")
    args = p.parse_args()

    dcfg = build_config(
        overrides=args.override, model_preset=args.model, n_episodes=args.n_episodes,
        ckpt=args.ckpt, device=args.device, seed=args.seed, run_id=args.run_id,
    )
    backends = args.backends.split(",")
    result = run(
        dcfg, backends, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        eval_every=args.eval_every, n_eval=args.n_eval, k_folds=args.k_folds, probs_n=args.probs_n,
    )

    print("\n=== final eval_copula by backend ===")
    for backend in backends:
        last = result["history"][backend][-1]
        print(f"  {backend:>7}: eval_total={last['eval_total']:.4f} eval_copula={last['eval_copula']:.4f} "
              f"eval_marginal={last['eval_marginal']:.4f}  ({last['elapsed_s']:.0f}s)")

    path = common.save_stage_result(dcfg, "s7b_backend_train", result)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
