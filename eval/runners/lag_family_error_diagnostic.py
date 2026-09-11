"""lag_family_error_diagnostic.py — decompose the copula head's correlation
error by pairwise lag (normalized by each episode's own lengthscale) and by
kernel family, to arbitrate between two hypotheses for why rank-128 hits a
ceiling on the marginal-NLL gap to the oracle:

  (H1) non-local folding: the head can't align w(x_i) with w(x_j) for
       DISTANT correlated points (periodic/cosine-style kernels) -- error
       should concentrate at LARGE normalized lag, and be much worse for
       periodic/cosine than rbf/matern.
  (H2) basis resolution: the head's implicit basis can't represent
       high-frequency structure at all -- error should concentrate at SMALL
       normalized lag (short-range curvature), roughly uniformly across
       families.

Ground truth is R_star (the prior/unconditional kernel correlation among
test points) -- NOT the Schur-complement-conditioned posterior. R_star is
what cfg.data.oracle_mode="prior" (the only mode data_gen.py supports) makes
the actual training target: z_test is the exact GP-LOO PIT residual, and
oracle NLL in eval_checkpoint.py is corr_nll_single(R_star, z_test). Scoring
the model's W against R_star elementwise therefore measures the same thing
training optimizes, not a different (posterior) quantity the model was
never asked to match -- see project_danp_gp_comparison.md's
oracle_mode=prior-vs-posterior note for why those two are NOT
interchangeable references.

Deliberate distribution overrides (all noted below) trade a bit of
train/eval-distribution fidelity for a lag axis that is actually
interpretable:
  - cfg.data.systematic_composition = False, cfg.data.kernel = <name>:
    forces one ELEMENTARY, non-composite kernel per family instead of a
    CauKer chain. For rbf/matern32/rational_quadratic this is a proper
    subset of training support (chain length 1 is always in-range). For
    periodic/cosine it is genuinely OUT of training support -- both are in
    conf/data/gp_tasks.yaml's composite_exclude_kernels, so the model has
    never seen them under the default systematic-composition config. Their
    numbers below measure zero-shot generalization to an excluded family,
    not in-distribution difficulty -- reported as a separate, explicitly
    labeled column, not pooled in with rbf/matern/RQ.
  - cfg.data.ard = False: isotropic-only. ARD would need a per-dimension
    Mahalanobis lag normalization instead of a scalar l; isotropic keeps
    "lag / l" a single well-defined number.
  - cfg.data.d_features = 2, lognormal keys unset, inactive_frac_{min,max}
    = 0: fixes d and removes inactive (noise) columns, so every input
    dimension is kernel-relevant and Euclidean distance in x-space isn't
    diluted by columns the kernel never looked at.

Everything else (P/N ranges, noise, hyperparameter priors) is left at
--config's own cfg.data.

Usage
-----
    python eval/runners/lag_family_error_diagnostic.py \\
        --ckpt checkpoints/copula_prod/canonical/kernel-sweep-classic-zcorrupt-noise-mild-bigN/step_0285000.pt \\
        --n_episodes 40 --out_dir eval/results/lag_family_diagnostic
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gen import generate_gp_batch  # noqa: E402
from inference.copula_inference import load_copula_model  # noqa: E402
from model import low_rank_correlation  # noqa: E402

from eval.baselines.classical import corr_nll_single  # noqa: E402
from eval.runners.eval_checkpoint import _load_full_config  # noqa: E402

# Families tested. rbf/matern32/rational_quadratic are IN systematic
# composition's support (chain length 1); periodic/cosine are excluded from
# it by conf/data/gp_tasks.yaml's composite_exclude_kernels default, so they
# are zero-shot for any checkpoint trained under that default -- kept
# separate in the report for exactly that reason.
_IN_SUPPORT_KERNELS = ["rbf", "matern32", "rational_quadratic"]
_OOD_KERNELS = ["periodic", "cosine"]

# Fixed bin edges in lag/lengthscale units -- physically meaningful (e.g.
# "within half a lengthscale" / "beyond 3 lengthscales") rather than
# quantile bins, which would shift per family and defeat cross-family
# comparison.
_LAG_BIN_EDGES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, float("inf")]


def _make_episode_cfg(base_cfg, kernel_name: str, seed: int):
    ep_cfg = copy.deepcopy(base_cfg)
    ep_cfg.data.systematic_composition = False
    ep_cfg.data.kernel = kernel_name
    ep_cfg.data.ard = False
    ep_cfg.data.d_features = 2
    ep_cfg.data.d_features_lognormal_loc = None
    ep_cfg.data.d_features_lognormal_scale = None
    ep_cfg.data.inactive_frac_min = 0.0
    ep_cfg.data.inactive_frac_max = 0.0
    ep_cfg.seed = seed
    return ep_cfg


def _predicted_R(icl_model, ep: dict, device: torch.device) -> torch.Tensor:
    X_train = ep["x_norm_train"].to(device)
    z_train = ep["z_train"].to(device)
    X_test = ep["x_norm_test"].to(device)
    P = X_train.shape[0]
    train_mask = torch.ones(1, P, dtype=torch.bool, device=device)
    batch = {
        "x_train": X_train.unsqueeze(0),
        "x_test": X_test.unsqueeze(0),
        "z_train": z_train.unsqueeze(0),
        "train_mask": train_mask,
    }
    with torch.no_grad():
        out = icl_model(batch)
        Sigma = low_rank_correlation(
            out["W"],
            out.get("s"),
            parametrization=getattr(icl_model, "correlation_parametrization", "covnorm"),
            lam=out.get("lam"),
        )
    N = X_test.shape[0]
    return Sigma[0, :N, :N]


def _episode_lengthscale(ep: dict, kernel_name: str) -> float:
    # "l" holds the isotropic lengthscale for every family here except
    # cosine, which reuses the same schema slot for period_length (see
    # data_gen.py's lengthscale_attr dispatch) -- either way it's the right
    # scalar decay/period parameter to normalize lag by.
    l = ep["l"]
    return float(l.reshape(-1)[0].item() if torch.is_tensor(l) else l)


def _accumulate(R_pred: torch.Tensor, R_star: torch.Tensor, x_test: torch.Tensor,
                 l: float, bin_sums: np.ndarray, bin_counts: np.ndarray) -> None:
    N = x_test.shape[0]
    iu = torch.triu_indices(N, N, offset=1)
    dist = torch.cdist(x_test.unsqueeze(0), x_test.unsqueeze(0))[0]
    lag = (dist[iu[0], iu[1]] / max(l, 1e-8)).cpu().numpy()
    err = (R_pred[iu[0], iu[1]] - R_star[iu[0], iu[1]]).abs().cpu().numpy()
    bin_idx = np.digitize(lag, _LAG_BIN_EDGES[1:-1])
    for b in range(len(_LAG_BIN_EDGES) - 1):
        m = bin_idx == b
        if m.any():
            bin_sums[b] += err[m].sum()
            bin_counts[b] += m.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(_REPO_ROOT, "conf", "config.yaml"))
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n_episodes", type=int, default=40, help="episodes per kernel family")
    parser.add_argument("--kernels", nargs="+", default=_IN_SUPPORT_KERNELS + _OOD_KERNELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out_dir", default=os.path.join(_REPO_ROOT, "eval", "results", "lag_family_diagnostic"))
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                           else (args.device if args.device != "auto" else "cpu"))
    print(f"Device: {device}")

    cfg = _load_full_config(args.config)
    print(f"Loading ICL checkpoint: {args.ckpt}")
    icl_model, _ = load_copula_model(args.ckpt, config_path=args.config, device=str(device))
    icl_model.eval()

    n_bins = len(_LAG_BIN_EDGES) - 1
    results: dict[str, dict] = {}

    for k_i, kernel_name in enumerate(args.kernels):
        print(f"\n=== {kernel_name} ===")
        ep_cfg = _make_episode_cfg(cfg, kernel_name, seed=args.seed + 1000 * k_i)
        episodes = generate_gp_batch(ep_cfg, args.n_episodes, device, return_kernel_metadata=True)

        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        nlls_icl, nlls_oracle = [], []

        for ep in episodes:
            R_pred = _predicted_R(icl_model, ep, device)
            R_star = ep["R_star"].to(device)
            x_test = ep["x_norm_test"].to(device)
            l = _episode_lengthscale(ep, kernel_name)
            _accumulate(R_pred, R_star, x_test, l, bin_sums, bin_counts)

            z_test = ep["z_test"].to(device)
            nlls_icl.append(corr_nll_single(R_pred, z_test))
            nlls_oracle.append(corr_nll_single(R_star, z_test))

        mean_err = np.divide(bin_sums, bin_counts, out=np.full(n_bins, np.nan), where=bin_counts > 0)
        results[kernel_name] = {
            "lag_bin_edges": _LAG_BIN_EDGES,
            "mean_abs_err_per_bin": mean_err.tolist(),
            "pair_count_per_bin": bin_counts.tolist(),
            "nll_icl_mean": float(np.nanmean(nlls_icl)),
            "nll_oracle_mean": float(np.nanmean(nlls_oracle)),
            "nll_gap_mean": float(np.nanmean(nlls_icl) - np.nanmean(nlls_oracle)),
            "n_episodes": len(episodes),
            "in_support": kernel_name in _IN_SUPPORT_KERNELS,
        }

        print(f"  NLL gap to oracle (z-space, nats/pt): {results[kernel_name]['nll_gap_mean']:.4f}")
        print(f"  mean |err| by lag bin:")
        for b in range(n_bins):
            lo, hi = _LAG_BIN_EDGES[b], _LAG_BIN_EDGES[b + 1]
            hi_s = f"{hi:.2f}" if hi != float("inf") else "inf"
            print(f"    [{lo:.2f}, {hi_s})  n={int(bin_counts[b]):>7}  mean|err|={mean_err[b]:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")

    # ---- Summary: low-lag vs high-lag error ratio per family, in/out of
    # support pooled separately, as the direct H1-vs-H2 readout. ----
    print(f"\n{'='*70}\nSummary: short-range vs long-range normalized-lag error\n{'='*70}")
    print(f"{'kernel':<20}{'support':<10}{'lag<0.5':>12}{'lag>2':>12}{'ratio(far/near)':>18}")
    for kernel_name, r in results.items():
        edges = r["lag_bin_edges"]
        errs = r["mean_abs_err_per_bin"]
        counts = r["pair_count_per_bin"]
        near = [e for e, lo, c in zip(errs, edges[:-1], counts) if lo < 0.5 and c > 0]
        far = [e for e, lo, c in zip(errs, edges[:-1], counts) if lo >= 2.0 and c > 0]
        near_m = float(np.mean(near)) if near else float("nan")
        far_m = float(np.mean(far)) if far else float("nan")
        ratio = far_m / near_m if near_m > 0 else float("nan")
        support = "in" if r["in_support"] else "OOD"
        print(f"{kernel_name:<20}{support:<10}{near_m:>12.4f}{far_m:>12.4f}{ratio:>18.2f}")


if __name__ == "__main__":
    main()
