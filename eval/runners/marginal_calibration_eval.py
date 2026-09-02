"""marginal_calibration_eval.py — Deliverable 2: make the marginal's defect
MEASURABLE, before any weight moves.

    python eval/runners/marginal_calibration_eval.py                  # pretrained baseline
    python eval/runners/marginal_calibration_eval.py --ckpt ./checkpoints/marginal_finetune/step_0020000_final.pt
    python eval/runners/marginal_calibration_eval.py --p-values 32 --n-episodes 256

Why this runner exists
----------------------
No existing runner reports marginal calibration on the production PIT path.
``eval/spatial/calibration.py::compute_quantile_ece`` has been in the repo the
whole time but is orphaned — reachable only from
``tests/test_reliability_diagram.py``. So the size of the marginal's error, the
thing the whole Phase-A workstream is aimed at, was never a number anyone could
quote. This runner is that number, and it is deliberately zero-training: run it
once on the pretrained checkpoint to get the baseline row, run it again on a
Phase-A output, subtract.

What it reports, and why each one
---------------------------------
* ``nll`` vs ``nll_oracle`` -> ``gap``. **The headline.** ``y`` is a pure GP draw,
  so the exact marginal posterior predictive is known in closed form; the gap is
  how many nats/point the frozen marginal is above the analytic floor. This is
  exactly the part of ``loss.y_space_nll``'s marginal term that no copula run can
  ever improve, because that term contains no trainable parameters.
* ``ks`` / rank histogram — is ``u = F(y)`` actually Uniform(0,1)? KS gives a
  scalar; the rank histogram says *how* it fails (U-shaped = over-sharp,
  dome = under-sharp), which a scalar cannot.
* ``ece`` — ``compute_quantile_ece`` over TabICL's native 999-level grid. Reuses
  the orphan rather than reimplementing it.
* ``z_gap`` — ``mean|z_tabicl - z_analytic|``, the same quantity
  ``train.py::_compute_tabicl_z_train_gap`` tracks, against the "two independent
  standard normals" reference of ``2/sqrt(pi) ~ 1.128``. This is the number that
  matters to the *copula*: z-space distortion is why this repo forbids comparing
  a learned Sigma against the oracle R_star at all.
* ``probit_clamp`` / ``slope_clamp`` — **silent failures**, currently invisible.
  ``pit._probit``'s ``eps=1e-6`` hard-caps ``|z| <= 4.7534``, and
  ``QuantileDistribution``'s ``MIN_SLOPE/MAX_SLOPE = 1e-+6`` bounds ``|log f| <=
  13.8``. Both saturate silently and both kill the gradient there, so a nonzero
  fraction is a real constraint on what Phase A can even learn.

Everything is broken out per kernel family and per context size ``P`` — the two
axes the defect is expected to vary along, and the two the fix is expected to
vary along too.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"),
           os.path.join(_REPO_ROOT, "tabicl_upstream", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gen import generate_gp_batch  # noqa: E402
from eval.configs.checkpoints import resolve_marginal_checkpoint  # noqa: E402
from eval.spatial.calibration import compute_quantile_ece  # noqa: E402
from marginal_finetune import (  # noqa: E402
    analytic_marginal_targets,
    ks_uniform,
    oracle_marginal_nll,
    rank_histogram,
)
from pit import (  # noqa: E402
    DEFAULT_K_FOLDS,
    _kernel_fn_from_task,
    gp_analytical_pit,
    load_tabicl,
    normalize_targets,
    run_pit_batched,
)

# |log f| ceiling implied by QuantileDistribution's MIN_SLOPE/MAX_SLOPE = 1e-+6:
# the density is 1 / (dQ/dalpha), so clamping the slope to [1e-6, 1e6] clamps
# log f to [-13.8155, +13.8155]. Hitting it means the spline could not resolve
# the density at that point at all -- and the gradient there is exactly zero.
_LOG_F_CEILING = math.log(1e6)


def _episode_metrics(
    tabicl, episodes, k_folds: int, eps: float, device: str
) -> list[dict]:
    """Score one shared-(P, N) batch of GP episodes on the EXACT production PIT
    path, and return one record per episode.

    ``run_pit_batched`` (not a hand-rolled forward) so this measures what
    training and deployment actually compute, including the K-fold geometry and
    the ``normalize_targets`` scaling every real call site applies. Three
    different ``u = F(y)`` implementations coexist in this repo -- this one,
    ``era5_calibration_eval.py``'s 99-knot ``np.interp``, and
    ``joint_nll.py``'s ``np.interp`` + finite difference -- and they disagree in
    the tails. This deliberately uses ``pit.py``'s.
    """
    B = len(episodes)
    x_tr = torch.stack([e["x_norm_train"] for e in episodes]).to(device)
    y_tr = torch.stack([e["y_train"] for e in episodes]).to(device)
    x_te = torch.stack([e["x_norm_test"] for e in episodes]).to(device)
    y_te = torch.stack([e["y_test"] for e in episodes]).to(device)

    y_tr_s, y_te_s, stds = [], [], []
    for b in range(B):
        a, c, _m, sd = normalize_targets(y_tr[b], y_te[b])
        y_tr_s.append(a)
        y_te_s.append(c)
        stds.append(sd)
    y_tr_s = torch.stack(y_tr_s)
    y_te_s = torch.stack(y_te_s)
    std_t = torch.stack(stds)

    out = run_pit_batched(
        tabicl, x_tr, y_tr_s.unsqueeze(-1), x_te, y_te_s.unsqueeze(-1),
        k_folds=k_folds, eps=eps, return_quantiles=True,
    )
    q_test = out["q_test"].squeeze(2)                          # (B, N, Q)
    u_test = out["u_test"].squeeze(2)                          # (B, N)
    z_test = out["z_test"].squeeze(2)                          # (B, N)
    z_train = out["z_train"].squeeze(2)                        # (B, P)
    logp_scaled = out["log_pdf_test"].squeeze(2)               # (B, N)
    alpha = out["alpha_levels"].detach().cpu().numpy()

    records = []
    for b, ep in enumerate(episodes):
        sd = float(std_t[b])
        # Jacobian back to raw-y nats: log p_raw(y) = log p_scaled(y_s) - log sd.
        # Same convention as train.py::_tabicl_pit_batch and data_gen, so these
        # numbers are directly comparable to val/y_nll_marginal.
        nll_raw = float(-(logp_scaled[b] - math.log(sd)).mean())

        rec = {
            "kernel": str(ep.get("kernel", "unknown")),
            "P": int(x_tr.shape[1]),
            "N": int(x_te.shape[1]),
            "nll": nll_raw,
            "ks": ks_uniform(u_test[b].cpu().numpy()),
            "ece": float(
                compute_quantile_ece(
                    y_te_s[b].cpu().numpy(), q_test[b].cpu().numpy(), alpha
                )[0]
            ),
            "probit_clamp": float(
                ((u_test[b] <= eps) | (u_test[b] >= 1 - eps)).float().mean()
            ),
            "slope_clamp": float(
                (logp_scaled[b].abs() >= _LOG_F_CEILING - 1e-3).float().mean()
            ),
            "rank_hist": rank_histogram(u_test[b].cpu().numpy(), 20).tolist(),
        }

        # --- analytic references (skipped for kernel families this repo cannot
        # --- reconstruct; see gp_analytical_posterior's own callers).
        try:
            kfn, nugget = _kernel_fn_from_task(ep)
            mu, sigma = analytic_marginal_targets(
                ep, x_tr[b], y_tr[b], x_te[b], kernel_fn=kfn, nugget=nugget
            )
            rec["nll_oracle"] = oracle_marginal_nll(y_te[b], mu, sigma)
            rec["gap"] = rec["nll"] - rec["nll_oracle"]
        except (NotImplementedError, KeyError):
            rec["nll_oracle"] = float("nan")
            rec["gap"] = float("nan")

        try:
            ana = gp_analytical_pit(ep)
            rec["z_gap_train"] = float(
                (z_train[b].cpu() - ana["z_train"].reshape(-1).cpu()).abs().mean()
            )
            rec["z_gap_test"] = float(
                (z_test[b].cpu() - ana["z_test"].reshape(-1).cpu()).abs().mean()
            )
        except (NotImplementedError, KeyError):
            rec["z_gap_train"] = float("nan")
            rec["z_gap_test"] = float("nan")

        records.append(rec)
    return records


def _agg(records: list[dict], keys: list[str]) -> dict:
    return {k: float(np.nanmean([r[k] for r in records])) for k in keys}


_METRIC_KEYS = [
    "nll", "nll_oracle", "gap", "ks", "ece",
    "z_gap_train", "z_gap_test", "probit_clamp", "slope_clamp",
]


def _print_table(title: str, rows: list[tuple[str, int, dict]]) -> None:
    print(f"\n{title}")
    hdr = (
        f"{'group':<28} {'n':>5} {'nll':>8} {'oracle':>8} {'gap':>8} "
        f"{'ks':>7} {'ece':>7} {'z_gap_tr':>9} {'z_gap_te':>9} {'clampP':>7} {'clampS':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, n, m in rows:
        print(
            f"{name:<28} {n:>5} {m['nll']:>8.4f} {m['nll_oracle']:>8.4f} {m['gap']:>8.4f} "
            f"{m['ks']:>7.4f} {m['ece']:>7.4f} {m['z_gap_train']:>9.4f} "
            f"{m['z_gap_test']:>9.4f} {m['probit_clamp']:>7.4f} {m['slope_clamp']:>7.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Marginal calibration / analytic-headroom report for a TabICL marginal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--ckpt", default="tabicl-regressor-v2-20260212.ckpt",
        help="HF filename in jingang/TabICL, a local .pt/.ckpt path, or a name "
             "registered in eval/configs/checkpoints.py::MARGINAL_FAMILIES.",
    )
    ap.add_argument("--n-episodes", type=int, default=128,
                    help="Episodes per context size P.")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Episodes per generate_gp_batch / PIT call (they share P and N).")
    ap.add_argument("--p-values", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256],
                    help="Context sizes to sweep. The defect is expected to vary "
                         "along this axis, and so is the fix.")
    ap.add_argument("--n-test", type=int, default=128, help="Query rows per episode.")
    ap.add_argument("--k-folds", type=int, default=DEFAULT_K_FOLDS,
                    help="Must match deployment (tabicl.pit_k_folds).")
    ap.add_argument("--eps", type=float, default=1.0e-6, help="Probit clamp epsilon.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--out-json", default=None,
                    help="Write the full per-group record (incl. rank histograms) here.")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = resolve_marginal_checkpoint(args.ckpt)
    print(f"[marginal_calibration_eval] ckpt={ckpt} device={device} k_folds={args.k_folds}")
    tabicl = load_tabicl(ckpt, device)

    from hydra import compose, initialize_config_dir

    # Compose the SAME prior Phase A trains against, rather than re-declaring
    # one here -- a measurement taken on a different prior than the fine-tuning
    # would not be a before/after of anything.
    with initialize_config_dir(config_dir=os.path.join(_REPO_ROOT, "conf"), version_base=None):
        full = compose(config_name="finetune_marginal")

    from omegaconf import OmegaConf

    all_records: list[dict] = []
    for P in args.p_values:
        gp_cfg = OmegaConf.create(
            {"data": OmegaConf.to_container(full.data, resolve=True), "seed": args.seed}
        )
        gp_cfg.data.P_min = int(P)
        gp_cfg.data.P_max = int(P)
        gp_cfg.data.N_min = int(args.n_test)
        gp_cfg.data.N_max = int(args.n_test)

        done = 0
        batch_i = 0
        while done < args.n_episodes:
            B = min(args.batch_size, args.n_episodes - done)
            gp_cfg.seed = args.seed + 1_000_003 * P + batch_i
            episodes = generate_gp_batch(gp_cfg, B, device, return_kernel_metadata=True)
            all_records.extend(_episode_metrics(tabicl, episodes, args.k_folds, args.eps, device))
            done += B
            batch_i += 1
        print(f"  P={P:<4} done ({done} episodes)")

    overall = _agg(all_records, _METRIC_KEYS)
    _print_table("OVERALL", [("all", len(all_records), overall)])

    by_p = []
    for P in args.p_values:
        rs = [r for r in all_records if r["P"] == P]
        if rs:
            by_p.append((f"P={P}", len(rs), _agg(rs, _METRIC_KEYS)))
    _print_table("BY CONTEXT SIZE", by_p)

    by_k = []
    for kern in sorted({r["kernel"] for r in all_records}):
        rs = [r for r in all_records if r["kernel"] == kern]
        by_k.append((kern[:28], len(rs), _agg(rs, _METRIC_KEYS)))
    _print_table("BY KERNEL FAMILY", by_k)

    hist = np.mean([r["rank_hist"] for r in all_records], axis=0)
    print("\nRANK HISTOGRAM of u = F(y)  (flat 0.05 = calibrated; U = over-sharp, dome = under-sharp)")
    print("  " + " ".join(f"{v:.3f}" for v in hist))

    print(
        f"\nHEADLINE  marginal NLL gap to the analytic oracle: "
        f"{overall['gap']:.4f} nats/point over {len(all_records)} episodes.\n"
        f"          z-space distortion mean|z_tabicl - z_analytic|: "
        f"{overall['z_gap_train']:.4f} (train) / {overall['z_gap_test']:.4f} (test); "
        f"the 'two independent standard normals' reference is {2 / math.sqrt(math.pi):.4f}."
    )

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(
                {
                    "ckpt": ckpt,
                    "args": vars(args),
                    "overall": overall,
                    "by_p": {n: m for n, _c, m in by_p},
                    "by_kernel": {n: m for n, _c, m in by_k},
                    "rank_hist": hist.tolist(),
                    "records": all_records,
                },
                f, indent=2,
            )
        print(f"[out] {args.out_json}")


if __name__ == "__main__":
    main()
