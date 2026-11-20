"""Joint log-likelihood on the EEG tasks of two neural-process benchmarks: each marginal alone, with its copula, and autoregressively.

Usage:
    python -m eval.runners.eeg_joint_eval out_dir=results/npbench/eeg                       # score
    python -m eval.runners.eeg_joint_eval command=report out_dir=results/npbench/eeg        # tables

Tasks (eval/data/eeg_tasks.py): Bruinsma et al. 2023 (arcnp_interpolation, arcnp_reconstruction;
exact regeneration) and Hassan et al. 2026 (hassan_int, hassan_for; reconstructed). Each task is a
table: rows are (time, electrode) pairs with features [t, one-hot electrode], the target is the EEG
value. For every marginal (TabICLv2, TabPFN-3.5 Fast, Kumo small, pretrained weights) the runner
scores, per episode and per target scalar (log q(y_T | ctx) / |T|, in nats):

- indep: the marginal with independent sampling, (1/N) sum_i log f_i(y_i);
- copula: the same marginal and its own copula (the copula trained on that marginal's pretrained
  PIT): indep - (1/N) * 0.5 (log det S + z^T S^{-1} z - z^T z), S the copula's correlation over all
  N target rows (low rank + diagonal, unit diagonal), z the targets' PIT. The marginal term is the
  same array as indep's, so the copula changes only the dependence.
- ar: the same marginal, exact chain rule with the true values fed back (eval/baselines/
  autoregressive.py), on the first ar.* episodes. Bruinsma: one random order (the paper's
  nps.ar_loglik). Hassan: log-mean-exp over ar.orders_hassan random orders (the paper uses 128).

The context PIT z_train comes from the marginal's K-fold pass (k_folds, default 5 as in the
copulas' training data). Marginal check: on the first check.episodes episodes, check.samples joint
draws z ~ N(0, S) give u = Phi(z); the mean KS distance of each target's u to U(0, 1) is compared
with independent draws (the copula must not change the marginals). Timing: wall-clock for joint
samples of the Hassan interpolation task through the copula, independently, and autoregressively.

Outputs: out_dir/<unit>/<marginal>.npz (per-episode arrays) and .json (settings, means), unit =
arcnp_<mode> or hassan_<mode>_N<N>_M<M>; out_dir/timing/<marginal>.json. A finished unit is skipped
on a rerun with the same settings.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from omegaconf import MISSING

from copula_inter.artifacts import atomic_write, canonical_digest
from eval.runners.hydra_cli import check_choice, check_choices, hydra_entry

if TYPE_CHECKING:
    from copula_inter.copula.model import CopulaTabICL
    from eval.data.eeg_tasks import EEGBatch

__all__ = [
    "EegJointEvalSpec",
    "copula_scores",
    "episode_scores",
    "logmeanexp",
    "marginal_ks",
    "run",
]

_MARGINALS = ("tabicl", "tabpfn35_fast", "kumo_small")
_COPULA_ROOT = "checkpoints/copula_nano"


def _default_copulas() -> dict[str, str]:
    return {m: f"{_COPULA_ROOT}/pretrained-marginal-{m}/step_0100000.pt" for m in _MARGINALS}


@dataclass
class HassanSpec:
    """Hassan et al.'s EEG grid: context sizes N, target sizes M (time points; x 7 electrodes in rows)."""

    n_context: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 192])
    n_target: list[int] = field(default_factory=lambda: [16, 64])
    # Trials per batch, and an optional cap on the 896-ish evaluation trials.
    batch_size: int = 32
    max_trials: int | None = None


@dataclass
class ArSpec:
    """The autoregressive row (same marginal, exact chain rule), on a subset of episodes."""

    enabled: bool = True
    # Bruinsma tasks: number of leading batches of 16 (its targets run to ~1,800 rows).
    arcnp_batches: int = 2
    # Hassan tasks: number of leading trials per (N, M), and random orders averaged in likelihood.
    hassan_trials: int = 64
    orders_hassan: int = 4
    # Episodes per chain call.
    batch_size: int = 16


@dataclass
class CheckSpec:
    """Marginal-preservation check: joint draws through the copula vs independent draws."""

    episodes: int = 32
    samples: int = 512


@dataclass
class TimingSpec:
    """Wall-clock of S joint samples for B Hassan-int episodes (N context, M target time points)."""

    enabled: bool = True
    n_context: int = 128
    n_target: list[int] = field(default_factory=lambda: [16, 64])
    episodes: int = 16
    samples: int = 16
    repeats: int = 3


@dataclass
class EegJointEvalSpec:
    """Score indep / copula / AR joint log-likelihoods on the EEG tasks, or render the tables."""

    # score: run the models; report: render the tables from out_dir.
    command: str = "score"
    tasks: list[str] = field(
        default_factory=lambda: ["arcnp_interpolation", "arcnp_reconstruction", "hassan_int", "hassan_for"]
    )
    marginals: list[str] = field(default_factory=lambda: list(_MARGINALS))
    # Copula checkpoint per marginal; each is scored only with the marginal it was trained on.
    copulas: dict[str, str] = field(default_factory=_default_copulas)
    # TabICL marginal weights (MARGINAL_FAMILIES name or path): the pretrained TabICLv2.
    tabicl_ckpt: str = "pretrained"
    # Exported tasks (scripts/npbench/export_eeg.py); default copula_inter.paths.NPBENCH_DATA_DIR.
    data_dir: str | None = None
    k_folds: int = 5
    # Bruinsma tasks: leading batches of 16 to score (null = all 256, i.e. the paper's 4096 tasks).
    arcnp_max_batches: int | None = None
    hassan: HassanSpec = field(default_factory=HassanSpec)
    ar: ArSpec = field(default_factory=ArSpec)
    check: CheckSpec = field(default_factory=CheckSpec)
    timing: TimingSpec = field(default_factory=TimingSpec)
    seed: int = 0
    device: str = "cuda"
    out_dir: str = MISSING


def _validate(spec: EegJointEvalSpec) -> None:
    from eval.data.eeg_tasks import task_names

    check_choice("command", spec.command, ("score", "report"))
    check_choices("tasks", spec.tasks, task_names())
    check_choices("marginals", spec.marginals, _MARGINALS)
    missing = [m for m in spec.marginals if m not in spec.copulas]
    if missing:
        raise ValueError(f"no copula checkpoint for marginals {missing}")
    if spec.k_folds < 2:
        raise ValueError("k_folds must be >= 2")


# ---------------------------------------------------------------------------------------------
# Scoring primitives (pure; unit-tested).


def logmeanexp(a: np.ndarray, axis: int = 0) -> np.ndarray:
    """log(mean(exp(a))) along axis, stably."""
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.mean(np.exp(a - m), axis=axis))


def copula_scores(U: torch.Tensor, D: torch.Tensor, z: torch.Tensor) -> np.ndarray:
    """(B,) summed Gaussian copula NLL of z (B, N) under S_b = U_b U_b^T + diag(D_b), in float64."""
    from eval.runners.ctr23_full_eval import chunk_copula_nll

    return np.array([chunk_copula_nll(U[b], D[b], z[b]) for b in range(z.shape[0])])


def episode_scores(log_pdf: np.ndarray, copula_nll: np.ndarray) -> dict[str, np.ndarray]:
    """Per-episode log-likelihood per target scalar: indep (marginal only) and copula; log_pdf (B, N)."""
    n = log_pdf.shape[1]
    indep = log_pdf.sum(axis=1) / n
    return {"ll_indep": indep, "ll_copula": indep - copula_nll / n, "copula_nll": copula_nll / n}


def marginal_ks(U: torch.Tensor, D: torch.Tensor, n_samples: int, gen: torch.Generator) -> tuple[float, float]:
    """(mean KS of Phi(z_i) to U(0,1) under the copula's draws, the same under independent draws), one episode."""
    from scipy.stats import kstest

    N, r = U.shape
    Ud, Dd = U.double().cpu(), D.double().cpu()
    e1 = torch.randn(n_samples, r, generator=gen, dtype=torch.float64)
    e2 = torch.randn(n_samples, N, generator=gen, dtype=torch.float64)
    z_cop = e1 @ Ud.T + e2 * Dd.clamp_min(0).sqrt()
    z_ind = torch.randn(n_samples, N, generator=gen, dtype=torch.float64)

    def mean_ks(z: torch.Tensor) -> float:
        return float(np.mean([kstest(z[:, i].numpy(), "norm").statistic for i in range(N)]))

    return mean_ks(z_cop), mean_ks(z_ind)


# ---------------------------------------------------------------------------------------------
# Models.


@dataclass
class _Models:
    marginal: str
    tabicl: Any
    regressor: Any
    copula: CopulaTabICL


def _load_models(spec: EegJointEvalSpec, marginal: str, dev: torch.device) -> _Models:
    from omegaconf import OmegaConf

    from copula_inter.checkpoint_registry import resolve_marginal_checkpoint
    from copula_inter.marginal.backends import make_regressor
    from copula_inter.marginal.pit import configure_tabicl_inference_amp, load_tabicl
    from inference.copula_inference import load_copula_model

    copula, cfg = load_copula_model(spec.copulas[marginal], device=str(dev))
    source = OmegaConf.select(cfg, "data.z_train_source", default="tabicl")
    trained_ckpt = OmegaConf.select(cfg, "data.z_train_marginal_ckpt", default=None)
    if source != marginal or (marginal != "tabicl" and trained_ckpt is not None):
        raise ValueError(
            f"copula {spec.copulas[marginal]} was trained on {source} ({trained_ckpt}), not pretrained {marginal}"
        )
    if marginal == "tabicl":
        configure_tabicl_inference_amp(False)
        tab = load_tabicl(resolve_marginal_checkpoint(spec.tabicl_ckpt), str(dev))
        pit_ckpt = OmegaConf.select(cfg, "tabicl.pit_ckpt", default=None)
        if pit_ckpt and os.path.basename(str(pit_ckpt)) != os.path.basename(
            resolve_marginal_checkpoint(spec.tabicl_ckpt)
        ):
            raise ValueError(f"copula trained with TabICL {pit_ckpt}, not {spec.tabicl_ckpt}")
        return _Models(marginal, tab, None, copula)
    return _Models(marginal, None, make_regressor(marginal, device=str(dev)), copula)


def _normalize(x_tr: np.ndarray, x_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """normalize_features per episode (jointly over its context and targets)."""
    from copula_inter.marginal.pit import normalize_features

    out = [normalize_features(a, b) for a, b in zip(x_tr, x_te)]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def _tensors(batch: EEGBatch, dev: torch.device) -> tuple[torch.Tensor, ...]:
    x_tr, x_te = _normalize(batch.x_train, batch.x_test)

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)

    return t(x_tr), t(batch.y_train), t(x_te), t(batch.y_test)


def _pit(
    models: _Models, x_tr: torch.Tensor, y_tr: torch.Tensor, x_te: torch.Tensor, y_te: torch.Tensor, k: int, seed: int
) -> dict:
    """Batched K-fold PIT through the marginal (eval_checkpoint's _pit_group): z_train, z_test, raw-unit log_pdf_test."""
    from copula_inter.data.era5_live_dataset import _pit_group

    generic = models.marginal != "tabicl"
    out = _pit_group(
        x_tr,
        y_tr,
        x_te,
        y_te,
        models.tabicl,
        k,
        marginal_backend=models.marginal if generic else None,
        marginal_regressor=models.regressor,
        seed=seed,
    )
    if out is None:
        raise ValueError("too little context for the marginal PIT")
    return {key: torch.as_tensor(v).to(x_tr.device) for key, v in out.items()}


def _copula_factor(
    cop: CopulaTabICL, x_tr: torch.Tensor, z_tr: torch.Tensor, x_te: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(U (B, N, r), D (B, N)) of the copula's correlation over the targets, one batched forward."""
    from copula_inter.copula.model import low_rank_correlation_factor

    out = cop({"x_train": x_tr, "x_test": x_te, "z_train": z_tr.float()})
    sv = out.get("s")
    fac = low_rank_correlation_factor(
        out["W"].float(),
        None if sv is None else sv.float(),
        parametrization=cop.correlation_parametrization,
        lam=out.get("lam"),
    )
    return fac.U, fac.D


def _ar_log_pdf(
    models: _Models,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    *,
    seed: int,
    index: list[int],
    conditioning: str = "teacher_forcing",
) -> torch.Tensor:
    from eval.baselines.autoregressive import autoregressive_log_pdf, backend_autoregressive_log_pdf

    if models.marginal == "tabicl":
        out = autoregressive_log_pdf(
            models.tabicl,
            x_tr,
            y_tr,
            x_te,
            y_te,
            order="random",
            conditioning=conditioning,
            seed=seed,
            episode_indices=index,
        )
    else:
        out = backend_autoregressive_log_pdf(
            models.marginal,
            models.regressor,
            x_tr,
            y_tr,
            x_te,
            y_te,
            order="random",
            conditioning=conditioning,
            seed=seed,
            episode_indices=index,
        )
    return out["log_pdf"].double().cpu()


# ---------------------------------------------------------------------------------------------
# Units of work.


@dataclass
class _Unit:
    name: str
    family: str  # arcnp / hassan
    batches: Callable[[], Any]
    n_context: int | None = None
    n_target: int | None = None
    # AR episodes, evenly spread over the task (its episodes are ordered by subject): arcnp, the
    # batch numbers whose 16 tasks get the chain; hassan, the trial indices, run in their own pass.
    ar_select: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    ar_batches: Callable[[], Any] | None = None


def _units(spec: EegJointEvalSpec, data_dir: str) -> list[_Unit]:
    from eval.data import eeg_tasks

    units: list[_Unit] = []
    for task in spec.tasks:
        family, mode = task.split("_", 1)
        if family == "arcnp":
            n_batches = eeg_tasks.arcnp_n_batches(data_dir, mode)
            if spec.arcnp_max_batches is not None:
                n_batches = min(n_batches, spec.arcnp_max_batches)
            units.append(
                _Unit(
                    task,
                    "arcnp",
                    partial(eeg_tasks.arcnp_batches, data_dir, mode, spec.arcnp_max_batches),
                    ar_select=eeg_tasks.spread(n_batches, spec.ar.arcnp_batches if spec.ar.enabled else 0),
                )
            )
            continue
        n_trials = eeg_tasks.n_eval_trials(data_dir)
        if spec.hassan.max_trials is not None:
            n_trials = min(n_trials, spec.hassan.max_trials)
        select = eeg_tasks.spread(n_trials, spec.ar.hassan_trials if spec.ar.enabled else 0)
        for m in spec.hassan.n_target:
            for n in spec.hassan.n_context:
                kw: dict[str, Any] = {"seed": spec.seed, "max_trials": spec.hassan.max_trials}
                units.append(
                    _Unit(
                        f"{task}_N{n}_M{m}",
                        "hassan",
                        partial(
                            eeg_tasks.hassan_batches, data_dir, mode, n, m, batch_size=spec.hassan.batch_size, **kw
                        ),
                        n,
                        m,
                        ar_select=select,
                        ar_batches=partial(
                            eeg_tasks.hassan_batches,
                            data_dir,
                            mode,
                            n,
                            m,
                            batch_size=spec.ar.batch_size,
                            trials=select.tolist(),
                            **kw,
                        ),
                    )
                )
    return units


def _fingerprint(spec: EegJointEvalSpec, unit: _Unit, marginal: str) -> str:
    keep: dict[str, Any] = {
        "unit": unit.name,
        "marginal": marginal,
        "copula": spec.copulas[marginal],
        "tabicl_ckpt": spec.tabicl_ckpt if marginal == "tabicl" else None,
        "k_folds": spec.k_folds,
        "seed": spec.seed,
        "ar": asdict(spec.ar),
        "ar_select": unit.ar_select.tolist(),
        "check": asdict(spec.check),
    }
    if unit.family == "arcnp":
        keep["arcnp_max_batches"] = spec.arcnp_max_batches
    else:
        keep |= {"hassan_batch": spec.hassan.batch_size, "hassan_max_trials": spec.hassan.max_trials}
    return canonical_digest(keep)


def _ar_chain_ll(
    spec: EegJointEvalSpec, models: _Models, batch: EEGBatch, n_orders: int, dev: torch.device
) -> np.ndarray:
    """(B,) AR log-likelihood per target scalar of a batch: log-mean-exp over n_orders random orders."""
    x_tr, y_tr, x_te, y_te = _tensors(batch, dev)
    out = np.empty(len(batch.index))
    for s in range(0, len(batch.index), spec.ar.batch_size):
        it = torch.arange(s, min(s + spec.ar.batch_size, len(batch.index)), device=dev)
        idx = [int(i) for i in batch.index[s : s + spec.ar.batch_size]]
        chains = np.stack(
            [
                _ar_log_pdf(models, x_tr[it], y_tr[it], x_te[it], y_te[it], seed=spec.seed + o, index=idx)
                .sum(-1)
                .numpy()
                for o in range(n_orders)
            ]
        )  # (orders, b)
        out[s : s + len(idx)] = logmeanexp(chains, axis=0) / y_te.shape[1]
    return out


def _score_unit(spec: EegJointEvalSpec, unit: _Unit, models: _Models, dev: torch.device) -> dict[str, np.ndarray]:
    cols: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "index",
            "n_context_rows",
            "n_target_rows",
            "ll_indep",
            "ll_copula",
            "copula_nll",
            "ll_ar",
            "z_mean",
            "z_std",
            "ks_copula",
            "ks_indep",
            "max_diag_err",
            "seconds",
        )
    }
    gen = torch.Generator().manual_seed(spec.seed)
    n_checked = 0
    for b_no, batch in enumerate(unit.batches()):
        t0 = time.time()
        x_tr, y_tr, x_te, y_te = _tensors(batch, dev)
        B, N = y_te.shape
        with torch.no_grad():
            pit = _pit(models, x_tr, y_tr, x_te, y_te, spec.k_folds, seed=spec.seed + b_no)
            U, D = _copula_factor(models.copula, x_tr, pit["z_train"], x_te)
        z = pit["z_test"].float()
        log_pdf = pit["log_pdf_test"].double().cpu().numpy()
        sc = episode_scores(log_pdf, copula_scores(U, D, z))
        ll_ar = np.full(B, np.nan)
        want = unit.family == "arcnp" and b_no in set(unit.ar_select.tolist())
        if want:  # Bruinsma: one random order, as nps.ar_loglik
            ll_ar = _ar_chain_ll(spec, models, batch, 1, dev)
        ks_c, ks_i = np.full(B, np.nan), np.full(B, np.nan)
        for b in range(B):
            if n_checked >= spec.check.episodes:
                break
            ks_c[b], ks_i[b] = marginal_ks(U[b], D[b], spec.check.samples, gen)
            n_checked += 1
        diag = (U * U).sum(-1) + D
        cols["index"].append(batch.index)
        cols["n_context_rows"].append(np.full(B, y_tr.shape[1]))
        cols["n_target_rows"].append(np.full(B, N))
        for k in ("ll_indep", "ll_copula", "copula_nll"):
            cols[k].append(sc[k])
        cols["ll_ar"].append(ll_ar)
        cols["z_mean"].append(z.mean(-1).cpu().numpy())
        cols["z_std"].append(z.std(-1).cpu().numpy())
        cols["ks_copula"].append(ks_c)
        cols["ks_indep"].append(ks_i)
        cols["max_diag_err"].append((diag - 1).abs().amax(-1).double().cpu().numpy())
        cols["seconds"].append(np.full(B, (time.time() - t0) / B))
        if b_no % 16 == 0:
            print(
                f"  [{unit.name} {models.marginal}] batch {b_no}: indep {np.mean(sc['ll_indep']):.3f} "
                f"copula {np.mean(sc['ll_copula']):.3f} ar {np.nanmean(ll_ar) if want else float('nan'):.3f} "
                f"({time.time() - t0:.1f} s)",
                flush=True,
            )
    out = {k: np.concatenate(v) for k, v in cols.items()}
    if unit.ar_batches is not None and unit.ar_select.size:
        # Hassan: the spread trials in their own batches, log-mean-exp over orders as the paper.
        pos = {int(j): i for i, j in enumerate(out["index"])}
        t0 = time.time()
        for batch in unit.ar_batches():
            ll = _ar_chain_ll(spec, models, batch, spec.ar.orders_hassan, dev)
            out["ll_ar"][[pos[int(j)] for j in batch.index]] = ll
        print(
            f"  [{unit.name} {models.marginal}] AR on {unit.ar_select.size} trials ({time.time() - t0:.0f} s)",
            flush=True,
        )
    return out


def _summary(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {"n_episodes": int(len(arrays["ll_indep"]))}
    for k in ("ll_indep", "ll_copula", "ll_ar", "ks_copula", "ks_indep", "max_diag_err", "seconds"):
        v = arrays[k][np.isfinite(arrays[k])]
        out[k] = float(v.mean()) if v.size else float("nan")
        out[f"{k}_n"] = int(v.size)
    gain = arrays["ll_copula"] - arrays["ll_indep"]
    out["copula_gain"] = float(gain.mean())
    out["copula_gain_se"] = float(gain.std(ddof=1) / np.sqrt(len(gain))) if len(gain) > 1 else float("nan")
    out["copula_win_rate"] = float((gain > 0).mean())
    return out


def _write_npz(path: str, arrays: dict[str, Any]) -> None:
    # Any: np.savez_compressed types **kwds against its allow_pickle flag.
    def _w(tmp: str) -> None:
        with open(tmp, "wb") as fh:  # a file handle: savez would append .npz to a str path
            np.savez_compressed(fh, **arrays)

    atomic_write(path, _w)


def _write_json(path: str, blob: dict[str, Any]) -> None:
    def _w(tmp: str) -> None:
        with open(tmp, "w") as fh:
            json.dump(blob, fh, indent=1)

    atomic_write(path, _w)


# ---------------------------------------------------------------------------------------------
# Timing.


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _quantiles(
    models: _Models, x_tr: torch.Tensor, y_tr: torch.Tensor, x_te: torch.Tensor
) -> tuple[torch.Tensor, np.ndarray]:
    """(B, N, Q) raw-unit test quantiles and their levels, one forward."""
    from copula_inter.marginal.finetune_adapters import _native_levels, backend_quantiles
    from copula_inter.marginal.pit import _alpha_levels_of, tabicl_forward

    mean = y_tr.mean(-1, keepdim=True)
    std = y_tr.std(-1, keepdim=True).clamp(min=1e-8)
    ys = (y_tr - mean) / std
    if models.marginal == "tabicl":
        logits = tabicl_forward(models.tabicl, torch.cat([x_tr, x_te], 1), ys).to(x_tr.device)
        q = models.tabicl.quantile_dist(logits).quantiles  # (B, N, Q)
        levels = _alpha_levels_of(models.tabicl, x_tr.device)
        assert levels is not None
        lv = levels.double().cpu().numpy()
    else:
        lv = _native_levels(models.marginal)
        q = backend_quantiles(
            models.marginal,
            models.regressor,
            list(x_tr.cpu().numpy()),
            list(ys.cpu().numpy()),
            list(x_te.cpu().numpy()),
            lv,
        ).to(x_tr.device)
    return q.double() * std.double().unsqueeze(-1) + mean.double().unsqueeze(-1), lv


def _inverse_cdf(q: torch.Tensor, levels: np.ndarray, u: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear F^{-1}: q (B, N, Q) at levels, u (S, B, N) -> (S, B, N)."""
    lv = torch.as_tensor(levels, dtype=q.dtype, device=q.device)
    u = u.clamp(float(lv[0]), float(lv[-1]))
    j = torch.searchsorted(lv, u.contiguous()).clamp(1, len(lv) - 1)
    qq = q.unsqueeze(0).expand(u.shape[0], -1, -1, -1)
    lo = qq.gather(-1, (j - 1).unsqueeze(-1)).squeeze(-1)
    hi = qq.gather(-1, j.unsqueeze(-1)).squeeze(-1)
    w = (u - lv[j - 1]) / (lv[j] - lv[j - 1])
    return lo + w * (hi - lo)


def _time(fn: Callable[[], Any], dev: torch.device, repeats: int) -> float:
    fn()  # warm-up
    best = float("inf")
    for _ in range(repeats):
        _sync(dev)
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        best = min(best, time.perf_counter() - t0)
    return best


def _timing(spec: EegJointEvalSpec, models: _Models, data_dir: str, dev: torch.device) -> dict[str, Any]:
    """Seconds to draw S joint samples of M target time points (x 7 electrodes) for B episodes."""
    from eval.data.eeg_tasks import hassan_batches

    ts = spec.timing
    rows: list[dict[str, Any]] = []
    for m in ts.n_target:
        batch = next(hassan_batches(data_dir, "int", ts.n_context, m, batch_size=ts.episodes, seed=spec.seed))
        x_tr, y_tr, x_te, y_te = _tensors(batch, dev)
        B, N = y_te.shape
        S = ts.samples

        def indep() -> torch.Tensor:
            with torch.no_grad():
                q, lv = _quantiles(models, x_tr, y_tr, x_te)
                u = torch.rand(S, B, N, device=dev, dtype=torch.float64)
                return _inverse_cdf(q, lv, u)

        def copula() -> torch.Tensor:
            with torch.no_grad():
                pit = _pit(models, x_tr, y_tr, x_te, y_te, spec.k_folds, seed=spec.seed)
                U, D = _copula_factor(models.copula, x_tr, pit["z_train"], x_te)
                q, lv = _quantiles(models, x_tr, y_tr, x_te)
                e1 = torch.randn(S, B, U.shape[-1], device=dev, dtype=U.dtype)
                e2 = torch.randn(S, B, N, device=dev, dtype=U.dtype)
                zs = torch.einsum("sbr,bnr->sbn", e1, U) + e2 * D.clamp_min(0).sqrt()
                return _inverse_cdf(q, lv, torch.special.ndtr(zs.double()))

        def ar() -> torch.Tensor:
            rep = torch.arange(B, device=dev).repeat(S)  # B*S chains, one per sample
            return _ar_log_pdf(
                models,
                x_tr[rep],
                y_tr[rep],
                x_te[rep],
                y_te[rep],
                seed=spec.seed,
                index=list(range(B * S)),
                conditioning="sample",
            )

        row: dict[str, Any] = {
            "n_context_time": ts.n_context,
            "n_target_time": m,
            "target_rows": N,
            "episodes": B,
            "samples": S,
        }
        row["seconds_indep"] = _time(indep, dev, ts.repeats)
        row["seconds_copula"] = _time(copula, dev, ts.repeats)
        row["seconds_ar"] = _time(ar, dev, 1)
        rows.append(row)
        print(f"  [timing {models.marginal}] {row}", flush=True)
    return {
        "marginal": models.marginal,
        "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "rows": rows,
    }


# ---------------------------------------------------------------------------------------------


def _data_dir(spec: EegJointEvalSpec) -> str:
    from copula_inter.paths import NPBENCH_DATA_DIR

    return spec.data_dir or str(NPBENCH_DATA_DIR)


def run(spec: EegJointEvalSpec) -> None:
    """Score every (unit, marginal) not already in out_dir, then the timing; or render the report."""
    if spec.command == "report":
        from eval.eeg_joint_report import write_report

        write_report(spec.out_dir)
        return
    dev = torch.device(spec.device if torch.cuda.is_available() or spec.device == "cpu" else "cpu")
    data_dir = _data_dir(spec)
    units = _units(spec, data_dir)
    for marginal in spec.marginals:
        models: _Models | None = None
        for unit in units:
            base = os.path.join(spec.out_dir, unit.name, marginal)
            fp = _fingerprint(spec, unit, marginal)
            if os.path.isfile(base + ".json"):
                with open(base + ".json") as fh:
                    if json.load(fh).get("fingerprint") == fp:
                        print(f"[{unit.name} {marginal}] done, skipping", flush=True)
                        continue
            if models is None:
                models = _load_models(spec, marginal, dev)
            print(f"[{unit.name} {marginal}] scoring", flush=True)
            t0 = time.time()
            arrays = _score_unit(spec, unit, models, dev)
            os.makedirs(os.path.dirname(base), exist_ok=True)
            _write_npz(base + ".npz", arrays)
            blob = {
                "fingerprint": fp,
                "unit": unit.name,
                "family": unit.family,
                "n_context": unit.n_context,
                "n_target": unit.n_target,
                "marginal": marginal,
                "spec": asdict(spec),
                "seconds": time.time() - t0,
                "summary": _summary(arrays),
            }
            _write_json(base + ".json", blob)
            print(f"[{unit.name} {marginal}] {blob['summary']}", flush=True)
        if spec.timing.enabled:
            path = os.path.join(spec.out_dir, "timing", f"{marginal}.json")
            if not os.path.isfile(path):
                if models is None:
                    models = _load_models(spec, marginal, dev)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                _write_json(path, _timing(spec, models, data_dir, dev))
        del models
        if dev.type == "cuda":
            torch.cuda.empty_cache()


main = hydra_entry("eeg_joint_eval", EegJointEvalSpec, run, validate=_validate)


if __name__ == "__main__":
    main()
