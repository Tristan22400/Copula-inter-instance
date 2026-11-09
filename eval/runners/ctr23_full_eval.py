"""OpenML-CTR23 on the full datasets: the task's official 10-fold CV, context = training folds, queries = test fold.

Usage:
    python -m eval.runners.ctr23_full_eval ckpt=<copula ckpt> tabicl_ckpt=<its marginal> data_id=44957 \\
        out_json=results/real/ctr23_full/44957.json

Per fold (eval/data/real/ctr23.openml_folds), with P = the training rows and
N = the test rows of that fold:

- Marginal: the frozen TabICL marginal PITs the context with its K-fold pass
  (pit_k_folds) and predicts the test rows from the whole context, as in
  eval_checkpoint. The context rows are shuffled first (default_rng([seed,
  data_id, fold])), since the K-fold pass cuts contiguous blocks. Reported in
  raw target units: marginal NLL per point, RMSE and R^2 of the predictive mean
  (the mean of the decoder quantiles), MAE of the median, CRPS on the quantile
  grid (2 x trapezoid of the pinball loss over the levels), z_test mean/std.
- Copula: the copula backbone conditions on the whole context's z_train and
  emits each test row's factor row (U_i, D_i) of Sigma = U U^T + diag(D). A test
  row attends only to the context, so the forward runs over query_batch test
  rows at a time and the rows are exact. The copula NLL of a query set S is
  0.5 (log det Sigma_SS + z_S^T Sigma_SS^{-1} z_S - z_S^T z_S); Sigma_SS is the
  sub-block, i.e. the copula's own joint over S. Scored over the whole test
  fold and over two partitions of it into chunks of chunk_size (random, and
  dyadic_knn: two kNN clusters per chunk), every test row in exactly one
  chunk. Per point: copula NLL summed over chunks / N; total = marginal +
  copula. Independence has copula NLL 0, so the copula NLL is also the copula
  minus independence; chunk_win_rate is the fraction of chunks where it is < 0.

No GP baselines and no autoregressive row (they do not scale to P ~ 6e4); the
comparison with other methods is on the marginal metrics (RMSE, R^2, CRPS),
which published CTR23 results report on the same splits. Each fold is written
as soon as it finishes; a rerun with the same settings skips the folds the
output already holds.
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

from copula_inter.artifacts import atomic_write, canonical_digest
from eval.runners.hydra_cli import hydra_entry

if TYPE_CHECKING:
    from copula_inter.copula.model import CopulaTabICL
    from copula_inter.marginal.pit import TabICLLike

__all__ = ["TARGETS", "Ctr23FullEvalSpec", "chunk_copula_nll", "compare_with_independence", "quantile_metrics"]


@dataclass
class Ctr23FullEvalSpec:
    """One CTR23 dataset on its official CV folds: the frozen marginal and the copula over the whole test fold."""

    # Copula checkpoint and the TabICL marginal it was trained with (never pair it with another marginal).
    ckpt: str = MISSING
    tabicl_ckpt: str = MISSING
    # OpenML data id (eval.data.real.ctr23.CTR23_TASKS).
    data_id: int = MISSING
    # sklearn data_home / split cache; default copula_inter.paths.CTR23_DATA_DIR.
    data_dir: str | None = None
    # The atom rule of the episode protocol: an excluded dataset raises.
    max_atom_frac: float = 0.05
    # Folds to run (empty = all of the task's folds).
    folds: list[int] = field(default_factory=list)
    # K of the marginal's K-fold PIT of the context (eval_checkpoint's default).
    pit_k_folds: int = 10
    # Query chunk size of the two partitions.
    chunk_size: int = 64
    # Test rows per copula forward (exact: a test row attends only to the context).
    query_batch: int = 2048
    seed: int = 42
    device: str = "cuda"
    # Frozen-marginal precision (float32 unless true), as marginal.tabicl_amp in eval_checkpoint.
    tabicl_amp: bool = False
    # Results JSON (a .npz of per-row arrays is written next to it); required.
    out_json: str = MISSING


def _validate(spec: Ctr23FullEvalSpec) -> None:
    from eval.data.real.ctr23 import CTR23_TASKS

    if int(spec.data_id) not in CTR23_TASKS:
        raise ValueError(f"data_id={spec.data_id} is not a CTR23 dataset ({sorted(CTR23_TASKS)})")
    if spec.chunk_size < 2 or spec.query_batch < 1 or spec.pit_k_folds < 2:
        raise ValueError("need chunk_size >= 2, query_batch >= 1, pit_k_folds >= 2")


def chunk_copula_nll(U: torch.Tensor, D: torch.Tensor, z: torch.Tensor) -> float:
    """Summed (not per-point) Gaussian copula NLL of z under Sigma = U U^T + diag(D), in float64 via Woodbury."""
    from copula_inter.copula.correlation import LowRankCorrelationFactor

    f = LowRankCorrelationFactor(U=U.double().unsqueeze(0), D=D.double().unsqueeze(0))
    zz = z.double().unsqueeze(0)
    return float(0.5 * (f.log_det() + f.quad_form(zz) - (zz * zz).sum(-1)))


def quantile_metrics(q: np.ndarray, levels: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """RMSE and R^2 of the quantile mean, MAE of the median, CRPS on the grid; q (N, Q) and y (N,) in raw units."""
    mean = q.mean(axis=1)
    med = np.array([np.interp(0.5, levels, row) for row in q])
    err = y[:, None] - q
    pinball = np.maximum(levels[None] * err, (levels[None] - 1.0) * err)
    crps = 2.0 * np.trapezoid(pinball, levels, axis=1)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt(((y - mean) ** 2).mean())),
        "r2": float(1.0 - ((y - mean) ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "mae_median": float(np.abs(y - med).mean()),
        "crps": float(crps.mean()),
    }


def _marginal(
    tab: TabICLLike, x_tr: torch.Tensor, y_tr: torch.Tensor, x_te: torch.Tensor, y_te: torch.Tensor, k_folds: int
) -> dict[str, torch.Tensor]:
    """z_train, z_test, raw-unit log_pdf_test and raw-unit test quantiles (eval_checkpoint's _pit_group scaling)."""
    from copula_inter.marginal.pit import _alpha_levels_of, run_pit_batched

    mean, std = y_tr.mean(), y_tr.std().clamp(min=1e-8)
    out = run_pit_batched(
        tab,
        x_tr[None],
        ((y_tr - mean) / std)[None, :, None],
        x_te[None],
        ((y_te - mean) / std)[None, :, None],
        k_folds=k_folds,
        return_quantiles=True,
        Y_train_raw=y_tr[None, :, None],
    )
    levels = _alpha_levels_of(tab, x_tr.device)
    if levels is None:
        raise ValueError("the marginal exposes no quantile levels")
    return {
        "z_train": out["z_train"][0, :, 0],
        "z_test": out["z_test"][0, :, 0],
        "log_pdf_test": out["log_pdf_test"][0, :, 0] - std.log(),
        "q_test": out["q_test"][0, :, 0] * std + mean,
        "clamp_frac_test": out["clamp_frac_test"].reshape(-1)[0],
        "levels": levels,
    }


def _copula_rows(
    cop: CopulaTabICL, x_tr: torch.Tensor, z_tr: torch.Tensor, x_te: torch.Tensor, batch: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(U (N, r), D (N,)) of the copula's Sigma for every test row, query_batch rows per forward."""
    from copula_inter.copula.model import low_rank_correlation_factor

    Us, Ds = [], []
    for s in range(0, x_te.shape[0], batch):
        out = cop({"x_train": x_tr[None], "x_test": x_te[None, s : s + batch], "z_train": z_tr[None]})
        sv = out.get("s")
        fac = low_rank_correlation_factor(
            out["W"].float(),
            None if sv is None else sv.float(),
            parametrization=cop.correlation_parametrization,
            lam=out.get("lam"),
        )
        Us.append(fac.U[0])
        Ds.append(fac.D[0])
    return torch.cat(Us), torch.cat(Ds)


def _partition_scores(
    U: torch.Tensor, D: torch.Tensor, z: torch.Tensor, chunks: list[np.ndarray]
) -> tuple[float, float, int]:
    """(copula NLL per point summed over chunks / N, chunk win rate, number of chunks)."""
    nlls = [chunk_copula_nll(U[c], D[c], z[c]) for c in (torch.as_tensor(c, device=U.device) for c in chunks)]
    return float(sum(nlls) / z.numel()), float(np.mean([v < 0 for v in nlls])), len(chunks)


def _fingerprint(spec: Ctr23FullEvalSpec, identity: dict[str, Any]) -> str:
    keep = ("ckpt", "tabicl_ckpt", "max_atom_frac", "pit_k_folds", "chunk_size", "seed", "tabicl_amp")
    return canonical_digest({"data": identity, **{k: getattr(spec, k) for k in keep}})


def _write(path: str, blob: dict[str, Any]) -> None:
    def _w(tmp: str) -> None:
        with open(tmp, "w") as fh:
            json.dump(blob, fh, indent=1)

    atomic_write(path, _w)


def _summary(folds: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Mean over the finished folds of every numeric per-fold field."""
    rows = list(folds.values())
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k != "fold"]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys} | {"n_folds": len(rows)}


def run(spec: Ctr23FullEvalSpec) -> None:
    """Score every requested fold and write out_json (and its .npz) after each."""
    from copula_inter.checkpoint_registry import resolve_marginal_checkpoint
    from copula_inter.marginal.pit import configure_tabicl_inference_amp, load_tabicl, normalize_features
    from eval.data.real import ctr23
    from eval.runners.eval_args import Ctr23Spec
    from inference.copula_inference import load_copula_model

    data = ctr23.load(Ctr23Spec(data_dir=spec.data_dir, data_id=int(spec.data_id), max_atom_frac=spec.max_atom_frac))
    splits = ctr23.openml_folds(data, spec.data_dir)
    wanted = list(spec.folds) or list(range(len(splits)))
    fp = _fingerprint(spec, data.identity)
    blob: dict[str, Any] = {}
    if os.path.isfile(spec.out_json):
        with open(spec.out_json) as fh:
            blob = json.load(fh)
        if blob.get("fingerprint") != fp:
            raise ValueError(f"{spec.out_json} holds results of other settings; remove it or change out_json")
    blob = {
        "fingerprint": fp,
        "spec": asdict(spec),
        "dataset": {
            "data_id": data.data_id,
            "name": data.name,
            "n_rows": len(data.y),
            "n_features": int(data.X.shape[1]),
            "n_missing_target": data.n_missing_target,
            "n_folds": len(splits),
        },
        "folds": blob.get("folds", {}),
    }
    npz_path = os.path.splitext(spec.out_json)[0] + ".npz"
    # Any: np.savez_compressed types **kwds against its allow_pickle flag.
    arrays: dict[str, Any] = dict(np.load(npz_path)) if os.path.isfile(npz_path) and blob["folds"] else {}
    todo = [k for k in wanted if str(k) not in blob["folds"]]
    print(f"{data.name}: {len(data.y)} rows x {data.X.shape[1]} features; folds to run {todo}", flush=True)
    if not todo:
        return

    dev = torch.device(spec.device)
    configure_tabicl_inference_amp(spec.tabicl_amp)
    tab = load_tabicl(resolve_marginal_checkpoint(spec.tabicl_ckpt), str(dev))
    cop, _ = load_copula_model(spec.ckpt, device=str(dev))

    for k in todo:
        rng = np.random.default_rng([int(spec.seed), int(spec.data_id), int(k)])
        train, test = splits[k]
        train = rng.permutation(train)  # the K-fold PIT cuts contiguous blocks of the context
        x_tr_np, x_te_np = normalize_features(data.X[train], data.X[test])
        x_tr = torch.from_numpy(x_tr_np.astype(np.float32)).to(dev)
        x_te = torch.from_numpy(x_te_np.astype(np.float32)).to(dev)
        y_tr = torch.from_numpy(data.y[train].astype(np.float32)).to(dev)
        y_te_np = data.y[test].astype(np.float64)
        y_te = torch.from_numpy(y_te_np.astype(np.float32)).to(dev)
        P, N = len(train), len(test)
        print(f"\nfold {k}: P={P} N={N}", flush=True)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        with torch.no_grad():
            m = _marginal(tab, x_tr, y_tr, x_te, y_te, spec.pit_k_folds)
        t_marg = time.time() - t0
        q = m["q_test"].double().cpu().numpy()
        levels = m["levels"].double().cpu().numpy()
        log_pdf = m["log_pdf_test"].double().cpu().numpy()
        z = m["z_test"].float()
        row: dict[str, Any] = {
            "fold": k,
            "P": P,
            "N": N,
            "marginal_nll": float(-log_pdf.mean()),
            **quantile_metrics(q, levels, y_te_np),
            "z_test_mean": float(z.mean()),
            "z_test_std": float(z.std()),
            "clamp_frac_test": float(m["clamp_frac_test"]),
        }

        t0 = time.time()
        with torch.no_grad():
            U, D = _copula_rows(cop, x_tr, m["z_train"].float(), x_te, spec.query_batch)
        t_cop = time.time() - t0
        row["copula_whole"] = chunk_copula_nll(U, D, z) / N
        parts = {
            "random": ctr23.random_partition(rng, np.arange(N), spec.chunk_size),
            "dyadic_knn": ctr23.dyadic_knn_partition(rng, data.Xs[test], np.arange(N), spec.chunk_size),
        }
        for name, chunks in parts.items():
            c, win, n_chunks = _partition_scores(U, D, z, chunks)
            row[f"copula_{name}"] = c
            row[f"chunk_win_rate_{name}"] = win
            row[f"n_chunks_{name}"] = n_chunks
        for name in ("whole", "random", "dyadic_knn"):
            row[f"total_{name}"] = row["marginal_nll"] + row[f"copula_{name}"]
        # Mean |Sigma_ij| over i != j (Sigma's off-diagonal is U U^T's).
        offdiag = (U @ U.T).abs().sum() - (U * U).sum(-1).sum() if N <= 20000 else torch.tensor(float("nan"))
        row["mean_abs_offdiag_corr"] = float(offdiag) / max(N * (N - 1), 1)
        row["seconds_marginal"] = t_marg
        row["seconds_copula"] = t_cop
        row["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
        print(
            "  "
            + "  ".join(
                f"{key} {row[key]:+.4f}"
                for key in ("marginal_nll", "copula_whole", "copula_random", "copula_dyadic_knn", "rmse", "crps")
            )
            + f"  ({t_marg:.0f}s + {t_cop:.0f}s, peak {row['peak_gpu_gib']:.1f} GiB)",
            flush=True,
        )

        blob["folds"][str(k)] = row
        blob["summary"] = _summary(blob["folds"])
        arrays |= {
            f"f{k}_test_rows": data.row_ids[test],
            f"f{k}_y": y_te_np,
            f"f{k}_pred_mean": q.mean(axis=1),
            f"f{k}_log_pdf": log_pdf,
            f"f{k}_z_test": z.cpu().numpy(),
            f"f{k}_U": U.cpu().numpy(),
            f"f{k}_D": D.cpu().numpy(),
        }

        def _save_npz(tmp: str) -> None:
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **arrays)

        atomic_write(npz_path, _save_npz)
        _write(spec.out_json, blob)
    s = blob["summary"]
    print(
        f"\n{data.name} over {s['n_folds']} folds: marginal NLL {s['marginal_nll']:+.4f}, copula whole "
        f"{s['copula_whole']:+.4f} / random {s['copula_random']:+.4f} / dyadic_knn {s['copula_dyadic_knn']:+.4f}, "
        f"RMSE {s['rmse']:.4g}, R2 {s['r2']:.4f}, CRPS {s['crps']:.4g}"
    )


# The query sets the joint NLL is scored over: the whole test fold, and its random / dyadic_knn chunks.
TARGETS = ("whole", "random", "dyadic_knn")


def compare_with_independence(blobs: list[dict[str, Any]], include_partial: bool = False) -> dict[str, Any]:
    """The copula against the independence model (same marginal, copula NLL 0) across datasets.

    blobs are ctr23_full_eval output JSONs, one per dataset; a dataset counts only
    once every fold is scored unless include_partial. Per dataset and target T:
    delta = mean over folds of copula_T (per point; = total_T(copula) - total(independence)),
    fold_win_rate = fraction of folds with copula_T < 0, and each method's rank
    (1 = lower mean total NLL; ties share 1.5). Across datasets, per target: mean and
    median delta, datasets where the copula wins, mean and median rank of each
    method, and the two-sided Wilcoxon signed-rank test of the per-dataset deltas.
    """
    from scipy.stats import wilcoxon

    rows: list[dict[str, Any]] = []
    for b in blobs:
        folds = list(b["folds"].values())
        if not folds or (len(folds) < b["dataset"]["n_folds"] and not include_partial):
            continue
        r: dict[str, Any] = {
            "data_id": b["dataset"]["data_id"],
            "name": b["dataset"]["name"],
            "n_rows": b["dataset"]["n_rows"],
            "n_folds": len(folds),
        }
        for key in ("marginal_nll", "rmse", "r2", "mae_median", "crps", "z_test_std", "mean_abs_offdiag_corr"):
            r[key] = float(np.mean([f[key] for f in folds]))
        for t in TARGETS:
            d = np.array([f[f"copula_{t}"] for f in folds])
            r[f"delta_{t}"] = float(d.mean())
            r[f"fold_win_rate_{t}"] = float((d < 0).mean())
            r[f"rank_copula_{t}"] = 1.0 if d.mean() < 0 else (2.0 if d.mean() > 0 else 1.5)
            r[f"rank_independence_{t}"] = 3.0 - r[f"rank_copula_{t}"]
        rows.append(r)
    summary: dict[str, dict[str, Any]] = {}
    for t in TARGETS:
        d = np.array([r[f"delta_{t}"] for r in rows])
        res: dict[str, Any] = {"n_datasets": int(d.size)}
        if d.size:
            res |= {
                "mean_delta": float(d.mean()),
                "median_delta": float(np.median(d)),
                "copula_wins": int((d < 0).sum()),
                "independence_wins": int((d > 0).sum()),
            }
            for m in ("copula", "independence"):
                ranks = np.array([r[f"rank_{m}_{t}"] for r in rows])
                res[f"mean_rank_{m}"] = float(ranks.mean())
                res[f"median_rank_{m}"] = float(np.median(ranks))
        if d.size >= 2 and np.any(d != 0):
            w = wilcoxon(d)
            res |= {"wilcoxon_stat": float(w.statistic), "p_value": float(w.pvalue)}
        summary[t] = res
    return {"per_dataset": rows, "summary": summary}


main = hydra_entry("ctr23_full_eval", Ctr23FullEvalSpec, run, validate=_validate)


if __name__ == "__main__":
    main()
