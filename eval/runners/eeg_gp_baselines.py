"""GP baselines on the EEG joint tasks of eval/runners/eeg_joint_eval.py, on the same episodes.

Usage:
    python -m eval.runners.eeg_gp_baselines out_dir=results/npbench/eeg

For each episode of a subset of every unit (the AR episodes of eeg_joint_eval, plus an even spread),
a GP is fitted on the context by MAP (copula_inter.baselines.gp_mle.fit_and_eval_gpytorch: ARD
kernel on the [t, one-hot electrode] features, constant mean, Gaussian noise, targets standardised by
the context) for each of `kernels`; the fit with the highest context log marginal likelihood is kept
(every kernel has the same parameter count, so this is the BIC rule of eval_checkpoint's
"GP-MLE (BIC-selected)" row). Its posterior at the targets, N(mu, Sigma) in raw units, gives:

- GP joint:    -log N(y | mu, Sigma) / n, split by loss.gp_oracle_y_nll into a marginal part
               (under diag Sigma) and a copula part (the rest);
- GP diagonal: the marginal part alone (the GP with independent targets);
- <marginal> + GP corr.: each TFM marginal (its own PIT z_test, as in eeg_joint_eval) with the GP's
               posterior correlation R = corr(Sigma) as the copula (eval_checkpoint's
               "Marginal + GP-MLE posterior corr." row, the matched competitor of the copula head).

All values are NLL per target value (lower is better). Outputs out_dir/<unit>/gp.npz, unit as in
eeg_joint_eval; a unit already written with the same settings is skipped.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omegaconf import MISSING

from copula_inter.artifacts import canonical_digest
from eval.runners.hydra_cli import check_choices, hydra_entry

if TYPE_CHECKING:
    from eval.data.eeg_tasks import EEGBatch
    from eval.runners.eeg_joint_eval import _Models

__all__ = ["EegGpBaselinesSpec", "correlation_copula_nll", "run"]

_MARGINALS = ("tabicl", "tabpfn35_fast", "kumo_small")


@dataclass
class EegGpBaselinesSpec:
    """GP baselines (joint, diagonal, TFM marginal + GP correlation) on a subset of the EEG episodes."""

    tasks: list[str] = field(
        default_factory=lambda: ["arcnp_interpolation", "arcnp_reconstruction", "hassan_int", "hassan_for"]
    )
    hassan_n_context: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 192])
    hassan_n_target: list[int] = field(default_factory=lambda: [16, 64])
    # Subset: Bruinsma, an even spread of arcnp_tasks tasks plus the AR batches (ar_arcnp_batches,
    # as eeg_joint_eval's ar.arcnp_batches); Hassan, hassan_trials trials plus the ar_hassan_trials AR trials.
    arcnp_tasks: int = 256
    ar_arcnp_batches: int = 4
    hassan_trials: int = 64
    ar_hassan_trials: int = 32
    kernels: list[str] = field(default_factory=lambda: ["rbf", "matern12", "matern52"])
    n_steps: int = 300
    lr: float = 0.05
    n_restarts: int = 1
    # TFM marginals for the "marginal + GP corr." rows (their eeg_joint_eval copulas are loaded but unused).
    marginals: list[str] = field(default_factory=lambda: list(_MARGINALS))
    tabicl_ckpt: str = "pretrained"
    k_folds: int = 5
    data_dir: str | None = None
    seed: int = 0
    device: str = "cuda"
    out_dir: str = MISSING


def _validate(spec: EegGpBaselinesSpec) -> None:
    from eval.data.eeg_tasks import task_names

    check_choices("tasks", spec.tasks, task_names())
    check_choices("marginals", spec.marginals, _MARGINALS)
    check_choices("kernels", spec.kernels, ["rbf", "matern12", "matern32", "matern52", "rational_quadratic"])


def correlation_copula_nll(R: torch.Tensor, z: torch.Tensor) -> float:
    """Summed Gaussian copula NLL 0.5 (log det R + z^T R^-1 z - z^T z) of z (N,) under a dense R, float64."""
    Rd, zd = R.double(), z.double()
    L = torch.linalg.cholesky(Rd)
    w = torch.cholesky_solve(zd[:, None], L)[:, 0]
    return float(0.5 * (2 * torch.log(torch.diagonal(L)).sum() + zd @ w - zd @ zd))


def _subset(spec: EegGpBaselinesSpec, data_dir: str, family: str, mode: str) -> np.ndarray:
    """Global episode indices of the subset (Bruinsma: task index = 16 * batch + position)."""
    from eval.data import eeg_tasks

    if family == "arcnp":
        n_batches = eeg_tasks.arcnp_n_batches(data_dir, mode)
        spread = eeg_tasks.spread(16 * n_batches, spec.arcnp_tasks)
        ar = np.concatenate([16 * b + np.arange(16) for b in eeg_tasks.spread(n_batches, spec.ar_arcnp_batches)])
        return np.union1d(spread, ar)
    T = eeg_tasks.n_eval_trials(data_dir)
    return np.union1d(eeg_tasks.spread(T, spec.hassan_trials), eeg_tasks.spread(T, spec.ar_hassan_trials))


def _batches(spec: EegGpBaselinesSpec, data_dir: str, family: str, mode: str, n: int, m: int, sel: np.ndarray) -> Any:
    from eval.data import eeg_tasks

    if family == "hassan":
        yield from eeg_tasks.hassan_batches(data_dir, mode, n, m, batch_size=16, seed=spec.seed, trials=sel.tolist())
        return
    keep = set(sel.tolist())
    for batch in eeg_tasks.arcnp_batches(data_dir, mode):
        rows = [i for i, j in enumerate(batch.index) if int(j) in keep]
        if rows:
            yield type(batch)(
                x_train=batch.x_train[rows],
                y_train=batch.y_train[rows],
                x_test=batch.x_test[rows],
                y_test=batch.y_test[rows],
                index=batch.index[rows],
                n_context=batch.n_context,
            )


def _fit_gp(spec: EegGpBaselinesSpec, x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, dev: torch.device) -> dict:
    """BIC-selected GP posterior at the targets, in raw units: mean (N,), Sigma (N, N), kernel index, log_ml per kernel."""
    from copula_inter.baselines.gp_mle import fit_and_eval_gpytorch

    mu, sd = float(y_tr.mean()), max(float(y_tr.std(ddof=1)), 1e-8)
    X = torch.as_tensor(x_tr, dtype=torch.float32, device=dev)
    Xt = torch.as_tensor(x_te, dtype=torch.float32, device=dev)
    Y = torch.as_tensor((y_tr - mu) / sd, dtype=torch.float32, device=dev)
    fits: list[dict[str, torch.Tensor] | None] = []
    lml: list[float] = []
    for k in spec.kernels:
        try:
            f = fit_and_eval_gpytorch(
                X,
                Y,
                Xt,
                k,
                n_steps=spec.n_steps,
                lr=spec.lr,
                ard=True,
                oracle_mode="posterior",
                n_restarts=spec.n_restarts,
                mean="constant",
            )
            fits.append(f)
            lml.append(float(f["log_ml"]))
        except RuntimeError as exc:
            print(f"  [gp {k}] failed: {exc}", flush=True)
            fits.append(None)
            lml.append(-np.inf)
    best = int(np.argmax(lml))
    f = fits[best]
    if f is None:
        raise RuntimeError("every GP kernel failed")
    return {"mean": f["mean"].double() * sd + mu, "Sigma": f["Sigma"].double() * sd**2, "kernel": best, "log_ml": lml}


def _gp_parts(mean: torch.Tensor, Sigma: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """NLL per target value of y under N(mean, Sigma): total, marginal (diag Sigma), copula = total - marginal."""
    from copula_inter.copula.loss import gp_oracle_y_nll

    n = y.shape[0]
    mask = torch.ones(1, n, dtype=torch.bool, device=y.device)
    out = gp_oracle_y_nll(Sigma[None], mean[None], y.double()[None], mask)
    return {k: float(v) for k, v in out.items()}


def _score_batch(
    spec: EegGpBaselinesSpec, batch: EEGBatch, models: dict[str, _Models], dev: torch.device
) -> dict[str, list]:
    from eval.runners.eeg_joint_eval import _normalize, _pit, _tensors

    rows: dict[str, list] = {
        "index": [],
        "kernel": [],
        "gp_total": [],
        "gp_marginal": [],
        "gp_copula": [],
        "seconds": [],
    }
    for m in models:
        rows[f"{m}_marginal"], rows[f"{m}_gpcorr_copula"] = [], []
    xtr_n, xte_n = _normalize(batch.x_train, batch.x_test)
    Rs = []
    for b in range(len(batch.index)):
        t0 = time.time()
        fit = _fit_gp(spec, xtr_n[b], batch.y_train[b], xte_n[b], dev)
        y = torch.as_tensor(batch.y_test[b], dtype=torch.float64, device=dev)
        parts = _gp_parts(fit["mean"], fit["Sigma"], y)
        d = torch.sqrt(torch.diagonal(fit["Sigma"]))
        Rs.append(fit["Sigma"] / (d[:, None] * d[None]))
        rows["index"].append(int(batch.index[b]))
        rows["kernel"].append(fit["kernel"])
        rows["gp_total"].append(parts["total"])
        rows["gp_marginal"].append(parts["marginal"])
        rows["gp_copula"].append(parts["copula"])
        rows["seconds"].append(time.time() - t0)
    x_tr, y_tr, x_te, y_te = _tensors(batch, dev)
    N = y_te.shape[1]
    for m, mod in models.items():
        with torch.no_grad():
            pit = _pit(mod, x_tr, y_tr, x_te, y_te, spec.k_folds, seed=spec.seed)
        for b in range(len(batch.index)):
            rows[f"{m}_marginal"].append(-float(pit["log_pdf_test"][b].double().mean()))
            rows[f"{m}_gpcorr_copula"].append(correlation_copula_nll(Rs[b], pit["z_test"][b]) / N)
    return rows


def run(spec: EegGpBaselinesSpec) -> None:
    """Fit and score the GP baselines unit by unit, skipping units already written with these settings."""
    from copula_inter.paths import NPBENCH_DATA_DIR
    from eval.runners.eeg_joint_eval import EegJointEvalSpec, _load_models

    dev = torch.device(spec.device if torch.cuda.is_available() or spec.device == "cpu" else "cpu")
    data_dir = spec.data_dir or str(NPBENCH_DATA_DIR)
    units: list[tuple[str, str, str, int, int]] = []
    for task in spec.tasks:
        family, mode = task.split("_", 1)
        if family == "arcnp":
            units.append((task, family, mode, 0, 0))
        else:
            units += [
                (f"{task}_N{n}_M{m}", family, mode, n, m) for m in spec.hassan_n_target for n in spec.hassan_n_context
            ]
    models: dict[str, _Models] | None = None
    for name, family, mode, n, m in units:
        path = os.path.join(spec.out_dir, name, "gp.npz")
        sel = _subset(spec, data_dir, family, mode)
        fp = canonical_digest(
            {"unit": name, "spec": asdict(spec) | {"tasks": None, "out_dir": None}, "sel": sel.tolist()}
        )
        if os.path.isfile(path):
            with np.load(path) as f:
                if str(f["fingerprint"]) == fp:
                    print(f"[{name} gp] done, skipping", flush=True)
                    continue
        if models is None:
            jspec = EegJointEvalSpec(out_dir="-", tabicl_ckpt=spec.tabicl_ckpt)
            models = {mg: _load_models(jspec, mg, dev) for mg in spec.marginals}
        t0 = time.time()
        cols: dict[str, list] = {}
        for batch in _batches(spec, data_dir, family, mode, n, m, sel):
            for k, v in _score_batch(spec, batch, models, dev).items():
                cols.setdefault(k, []).extend(v)
        arrays: dict[str, Any] = {k: np.asarray(v) for k, v in cols.items()}
        arrays["kernels"] = np.array(spec.kernels)
        arrays["fingerprint"] = np.array(fp)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
        summary = {
            k: float(np.mean(arrays[k])) for k in arrays if k not in ("index", "kernel", "kernels", "fingerprint")
        }
        print(f"[{name} gp] n={len(arrays['index'])} ({time.time() - t0:.0f} s) {json.dumps(summary)}", flush=True)


main = hydra_entry("eeg_gp_baselines", EegGpBaselinesSpec, run, validate=_validate)


if __name__ == "__main__":
    main()
