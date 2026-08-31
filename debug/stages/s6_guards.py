"""s6_guards.py — saturation-guard audit: covnorm escape ratio + Cholesky
jitter escalation / non-finite-input fallback counts.

Narrowed from the original debug list's broader "audit saturation guards":
the jitter ceiling (1/(1+sigma_jitter) ~= 0.9999 at the default 1e-4) and
rank's lack of a pairwise correlation ceiling are both closed-form facts,
not worth instrumenting (see debug/README.md). What's left to actually
MEASURE on a real model:

  - covnorm escape ratio ||W_i||^2 / softplus(s_i) per test point -- this
    ratio, not rank, sets how close a pair's correlation can get to +-1
    (Sigma_ij bounded by sqrt((1-D_i/C_i)(1-D_j/C_j)) before the jitter
    floor, model.py::low_rank_correlation's "covnorm" branch). At init
    (s=0, W~N(0,0.02^2)) this ratio starts near 0 -- reports how far a
    trained checkpoint has moved off that init.
  - loss.py::_safe_cholesky's two failure paths, counted via monkey-patching
    torch.linalg.cholesky and loss._safe_cholesky for the duration of one
    y_space_nll call (no src/ edits): (a) how many (episode, retry) pairs
    needed jitter escalation beyond the base 1e-6, (b) how many Sigma slices
    had non-finite entries and were silently replaced with identity before
    factorization. train/sigma_nonfinite_count already tracks (b) in
    production (reads 0 in every run inspected) -- this stage exists to
    confirm that at a chosen checkpoint/config rather than take it on faith,
    and to add (a), which nothing in the repo counts today.

Usage:
    python debug/run_debug.py s6 --ckpt <name>
    python debug/stages/s6_guards.py --n-episodes 50   # fresh (untrained) model
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC, os.path.join(_REPO_ROOT, "debug")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common
from config import DebugConfig, add_common_args, build_config


@contextlib.contextmanager
def _cholesky_instrumentation():
    """Monkey-patches torch.linalg.cholesky (counts calls per _safe_cholesky
    invocation -> jitter-escalation count) and loss._safe_cholesky (counts
    non-finite-input slices before it replaces them with identity) for the
    duration of the `with` block. Restores both on exit regardless of
    exceptions. No src/ files are edited."""
    import loss as loss_mod

    counters = {"cholesky_calls": 0, "safe_cholesky_calls": 0, "escalated_calls": 0, "nonfinite_slices": 0}
    orig_cholesky = torch.linalg.cholesky
    orig_safe_cholesky = loss_mod._safe_cholesky

    def counting_cholesky(K, *a, **kw):
        counters["cholesky_calls"] += 1
        return orig_cholesky(K, *a, **kw)

    def counting_safe_cholesky(K, *a, **kw):
        counters["safe_cholesky_calls"] += 1
        calls_before = counters["cholesky_calls"]
        finite = torch.isfinite(K).flatten(-2).all(-1)
        counters["nonfinite_slices"] += int((~finite).sum().item())
        out = orig_safe_cholesky(K, *a, **kw)
        if counters["cholesky_calls"] - calls_before > 1:
            counters["escalated_calls"] += 1
        return out

    torch.linalg.cholesky = counting_cholesky
    loss_mod._safe_cholesky = counting_safe_cholesky
    try:
        yield counters
    finally:
        torch.linalg.cholesky = orig_cholesky
        loss_mod._safe_cholesky = orig_safe_cholesky


def run(dcfg: DebugConfig) -> dict:
    from dataset import collate_fn
    from loss import y_space_nll
    from model import low_rank_correlation

    model = common.load_model(dcfg)
    episodes = common.generate_episodes(dcfg, dcfg.n_episodes, tabicl_model=None)
    batch = {k: v.to(dcfg.device) for k, v in collate_fn(episodes).items()}
    jitter = float(dcfg.cfg.model.get("sigma_jitter", 1e-4))
    parametrization = str(dcfg.cfg.model.get("correlation_parametrization", "covnorm"))

    with torch.no_grad():
        out = model(batch)
        W = out["W"].float()                          # (B, N_max, r)
        s = out.get("s")
        Sigma = low_rank_correlation(
            W, s.float() if s is not None else None, batch["test_mask"],
            jitter=jitter, parametrization=parametrization,
        )

    mask = batch["test_mask"]
    W_norm = W.norm(dim=-1)[mask].cpu().numpy()          # ||W_i|| per valid test point
    escape_ratio = None
    if parametrization == "covnorm" and s is not None:
        D = F.softplus(s.float())
        escape_ratio = ((W.pow(2).sum(-1)) / D.clamp_min(1e-12))[mask].cpu().numpy()

    mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    n = Sigma.shape[1]
    ri, ci = torch.triu_indices(n, n, offset=1)
    off_valid = mask_2d[:, ri, ci]
    offdiag = Sigma[:, ri, ci][off_valid].detach().float().cpu().numpy()

    with _cholesky_instrumentation() as counters:
        with torch.no_grad():
            _ = y_space_nll(Sigma, batch["z_test"].float(), batch["log_pdf_test"].float(), batch["test_mask"])

    return {
        "n_episodes": len(episodes),
        "parametrization": parametrization,
        "sigma_jitter": jitter,
        "W_norm": {"mean": float(W_norm.mean()), "std": float(W_norm.std()), "max": float(W_norm.max())},
        "escape_ratio": (
            {"mean": float(escape_ratio.mean()), "std": float(escape_ratio.std()),
             "max": float(escape_ratio.max()), "frac_gt_1": float((escape_ratio > 1).mean())}
            if escape_ratio is not None else None
        ),
        "sigma_offdiag": {"mean": float(offdiag.mean()), "abs_mean": float(np.abs(offdiag).mean()), "std": float(offdiag.std())},
        "cholesky_calls_total": counters["cholesky_calls"],
        "safe_cholesky_calls_total": counters["safe_cholesky_calls"],
        "safe_cholesky_calls_escalated": counters["escalated_calls"],
        "nonfinite_input_slices": counters["nonfinite_slices"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = p.parse_args()

    dcfg = build_config(
        overrides=args.override, model_preset=args.model, n_episodes=args.n_episodes,
        ckpt=args.ckpt, device=args.device, seed=args.seed, run_id=args.run_id,
    )
    result = run(dcfg)

    print(f"Scored {result['n_episodes']} episodes, parametrization={result['parametrization']}, ckpt={dcfg.ckpt}\n")
    w = result["W_norm"]
    print(f"  ||W_i||: mean={w['mean']:.4f} std={w['std']:.4f} max={w['max']:.4f}")
    if result["escape_ratio"] is not None:
        e = result["escape_ratio"]
        print(f"  escape ratio ||W||^2/softplus(s): mean={e['mean']:.4f} std={e['std']:.4f} max={e['max']:.4f} frac>1={e['frac_gt_1']:.4f}")
        print("    (near 0 = still near init-independence; >1 = the model has actively pushed toward strong correlations)")
    sd = result["sigma_offdiag"]
    print(f"  Sigma offdiag: mean={sd['mean']:+.4f} abs_mean={sd['abs_mean']:.4f} std={sd['std']:.4f}")
    print(f"\n  Cholesky calls (y_space_nll pass): {result['cholesky_calls_total']} over {result['safe_cholesky_calls_total']} _safe_cholesky invocation(s)")
    print(f"  Escalated (needed jitter > 1e-6): {result['safe_cholesky_calls_escalated']}")
    print(f"  Non-finite input slices (identity fallback fired): {result['nonfinite_input_slices']}")

    path = common.save_stage_result(dcfg, "s6_guards", result)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
