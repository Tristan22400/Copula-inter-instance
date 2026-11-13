"""Well-specified GP-MLE diagnostics: does the baseline fit converge to the truth when its model is right?

D1 (consistency): episodes are drawn from the GP-MLE's own model -- one kernel family,
hyperparameters and noise from the fit's hyperpriors (sample_prior_gp), a constant mean
~ N(0, 1), x ~ N(0, I_d), no nuisance features or warps. The matching GP-MLE
(posterior, constant mean, the eval's optimizer settings) is scored against the
known-theta posterior on N query points:
    gap = total_NLL(theta_hat) - total_NLL(theta)   (nats per query point, >= 0 on average).
Pass: the gap shrinks toward 0 as the context size P grows; a plateau points to the optimizer.

D2 (optimizer): J(theta) = log p(y_C | theta) + log pi(theta) (gp_map_objective) is what the
fit maximizes and theta_true is in its search space, so a successful fit has
J(theta_hat) >= J(theta_true). failure_rate is the fraction of episodes with
J(theta_hat) < J(theta_true) - j_tol. Only for ard=false fits (ARD has no lengthscale prior,
so its J is a different objective, and it is selected on a held-out split).

Usage:
    python -m eval.runners.gp_wellspecified_diag out_json=eval/results/gp_diag/d1_d2.json
    python -m eval.runners.gp_wellspecified_diag kernels=[rbf] context_sizes=[10,30] n_episodes=5
"""

from __future__ import annotations

import copy
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from copula_inter.baselines.gp_mle import fit_and_eval_gpytorch, gp_map_objective, sample_prior_gp
from eval.runners.hydra_cli import check_choices, hydra_entry

__all__ = ["GpDiagSpec", "diagnose_episode", "run", "summarize"]

DIAG_KERNELS = ("rbf", "matern32", "rational_quadratic", "periodic")


@dataclass
class GpDiagSpec:
    """D1/D2 grid: kernels x context sizes x episodes, each fitted plain and (optionally) ARD."""

    kernels: list[str] = field(default_factory=lambda: ["rbf", "matern32", "rational_quadratic"])
    context_sizes: list[int] = field(default_factory=lambda: [10, 30, 100, 300])
    n_query: int = 64
    d_x: int = 2
    n_episodes: int = 50
    # Fit plain (false) and ARD (true, selected on a held-out split as in the eval) GP-MLE.
    ard: list[bool] = field(default_factory=lambda: [False, True])
    # The eval's GP-MLE optimizer settings (baselines.n_steps_mle / lr_mle / n_restarts_mle).
    n_steps: int = 1000
    lr: float = 0.05
    n_restarts: int = 5
    # D2 tolerance on J(theta_hat) - J(theta_true), in nats (whole context; above float32 rounding).
    j_tol: float = 0.05
    seed: int = 0
    # Worker processes (one CPU thread each); 0 = all cores.
    n_workers: int = 0
    out_json: str | None = None


def _mvn_nll_per_point(mean: torch.Tensor, Sigma: torch.Tensor, y: torch.Tensor) -> float:
    return -float(torch.distributions.MultivariateNormal(mean, covariance_matrix=Sigma).log_prob(y)) / y.shape[0]


def diagnose_episode(spec: GpDiagSpec, kernel: str, P: int, episode: int) -> list[dict[str, Any]]:
    """One well-specified episode: sample, score the known-theta posterior, fit each ARD setting."""
    seed = spec.seed * 1_000_003 + DIAG_KERNELS.index(kernel) * 100_003 + P * 1_009 + episode
    torch.manual_seed(seed)
    N = spec.n_query
    X = torch.randn(P + N, spec.d_x, dtype=torch.float64)
    true_model = sample_prior_gp(X, kernel, mean_constant=float(torch.randn(())))
    true_model.double()
    true_model.train()
    with torch.no_grad():
        y = true_model.likelihood(true_model(X)).sample()
    X_C, y_C, X_Q, y_Q = X[:P], y[:P], X[P:], y[P:]

    # In float32 like the fit's own J(theta_hat), so the D2 comparison has matched rounding.
    j_true = gp_map_objective(copy.deepcopy(true_model).float(), X_C.float(), y_C.float())
    # Condition the known-theta GP on the context (sample_prior_gp's training data are placeholders).
    true_model.set_train_data(inputs=X_C, targets=y_C, strict=False)
    true_model.eval()
    with torch.no_grad():
        pred = true_model.likelihood(true_model(X_Q))
        oracle = _mvn_nll_per_point(pred.mean, pred.covariance_matrix, y_Q)

    rows = []
    for ard in spec.ard:
        base = {"kernel": kernel, "P": P, "episode": episode, "ard": ard, "oracle_total": oracle}
        try:
            fit = fit_and_eval_gpytorch(
                X_C.float(),
                y_C.float(),
                X_Q.float(),
                kernel,
                n_steps=spec.n_steps,
                lr=spec.lr,
                ard=ard,
                oracle_mode="posterior",
                n_restarts=spec.n_restarts,
                val_select=ard,
                mean="constant",
            )
        except RuntimeError as exc:
            print(f"  [{kernel} P={P} ep={episode} ard={ard}] fit failed: {exc}")
            rows.append({**base, "fit_total": math.nan, "gap": math.nan, "j_fit": math.nan, "j_true": j_true})
            continue
        fit_total = _mvn_nll_per_point(fit["mean"].double(), fit["Sigma"].double(), y_Q)
        rows.append(
            {
                **base,
                "fit_total": fit_total,
                "gap": fit_total - oracle,
                "j_fit": float(fit["map_objective"]),
                "j_true": j_true,
            }
        )
    return rows


def _worker(args: tuple[GpDiagSpec, str, int, int]) -> list[dict[str, Any]]:
    torch.set_num_threads(1)
    return diagnose_episode(*args)


def summarize(rows: list[dict[str, Any]], j_tol: float) -> list[dict[str, Any]]:
    """Per (kernel, ard, P): D1 mean gap with a normal 95% CI and median; D2 failure rate (ard=false only)."""
    groups: dict[tuple[str, bool, int], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["kernel"], r["ard"], r["P"]), []).append(r)
    out = []
    for (kernel, ard, P), grp in sorted(groups.items()):
        gaps = np.array([r["gap"] for r in grp if math.isfinite(r["gap"])])
        n = gaps.size
        se = gaps.std(ddof=1) / math.sqrt(n) if n > 1 else math.nan
        summary: dict[str, Any] = {
            "kernel": kernel,
            "ard": ard,
            "P": P,
            "n": n,
            "n_failed_fits": len(grp) - n,
            "gap_mean": float(gaps.mean()) if n else math.nan,
            "gap_ci95": [float(gaps.mean() - 1.96 * se), float(gaps.mean() + 1.96 * se)] if n > 1 else None,
            "gap_median": float(np.median(gaps)) if n else math.nan,
        }
        if not ard:
            dj = np.array([r["j_fit"] - r["j_true"] for r in grp if math.isfinite(r["j_fit"])])
            summary["d2_failure_rate"] = float((dj < -j_tol).mean()) if dj.size else math.nan
            summary["d2_median_dj"] = float(np.median(dj)) if dj.size else math.nan
            summary["d2_worst_dj"] = float(dj.min()) if dj.size else math.nan
        out.append(summary)
    return out


def _print(summary: list[dict[str, Any]]) -> None:
    print(f"{'kernel':<20}{'ard':>5}{'P':>6}{'n':>5}{'gap mean':>11}{'95% CI':>22}{'median':>9}{'D2 fail':>9}")
    for s in summary:
        ci = f"[{s['gap_ci95'][0]:+.4f}, {s['gap_ci95'][1]:+.4f}]" if s["gap_ci95"] else "-"
        fail = f"{s['d2_failure_rate']:.2f}" if "d2_failure_rate" in s else "-"
        print(
            f"{s['kernel']:<20}{str(s['ard']):>5}{s['P']:>6}{s['n']:>5}{s['gap_mean']:>+11.4f}{ci:>22}"
            f"{s['gap_median']:>+9.4f}{fail:>9}"
        )


def validate(spec: GpDiagSpec) -> None:
    check_choices("kernels", spec.kernels, DIAG_KERNELS)
    if min(spec.context_sizes, default=0) < 2 or spec.n_query < 1 or spec.n_episodes < 1:
        raise ValueError("context_sizes must be >= 2, n_query and n_episodes >= 1")


def run(spec: GpDiagSpec) -> None:
    validate(spec)
    tasks = [(spec, k, P, e) for k in spec.kernels for P in spec.context_sizes for e in range(spec.n_episodes)]
    n_workers = spec.n_workers or os.cpu_count() or 1
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for i, episode_rows in enumerate(pool.map(_worker, tasks)):
            rows.extend(episode_rows)
            if (i + 1) % max(1, len(tasks) // 20) == 0:
                print(f"  {i + 1}/{len(tasks)} episodes", flush=True)
    summary = summarize(rows, spec.j_tol)
    _print(summary)
    if spec.out_json:
        os.makedirs(os.path.dirname(spec.out_json) or ".", exist_ok=True)
        with open(spec.out_json, "w") as fh:
            json.dump({"spec": asdict(spec), "summary": summary, "episodes": rows}, fh, indent=1)
        print(f"Wrote {spec.out_json}")


main = hydra_entry("gp_wellspecified_diag", GpDiagSpec, run, validate=validate)


if __name__ == "__main__":
    main()
