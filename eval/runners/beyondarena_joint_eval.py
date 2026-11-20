"""BeyondArena joint track: joint samples over blocks of test rows (docs/beyondarena/preregistration.md).

Usage:
    python -m eval.runners.beyondarena_joint_eval dataset=<name> method=tabicl-ft \\
        ckpt=<copula ckpt> tabicl_ckpt=<its marginal> out_dir=results/beyondarena/joint/tabicl-ft

Input: the splits scripts/beyondarena/export_splits.py wrote under data_dir/<dataset>/ (official
BeyondArena train/test splits after TabArena's model-agnostic preprocessing). Nothing is bagged.

Per split, with y scaled by s = std of the marginal context's y:
- Marginal context: at most p_marg training rows (seeded subsample). Test rows are predicted from
  all of it (ctr23_full_eval's test-only marginal calls), giving z_test, log f(y) and quantiles.
- Copula context: p_cop rows of the marginal context. Their z_train are out-of-fold PITs with
  k_folds folds over the marginal context: fold k's rows are predicted from the other folds.
  cross_fit=struct forms folds from whole groups (grouped tasks) or contiguous time ranges
  (temporal tasks); cross_fit=random forms them at random (IID tasks are always random).
- Blocks (primary B=block_size): one test group per block (capped at block_max rows), consecutive
  time windows, or random rows, by the task type. At most max_blocks blocks per split.
  block_sweep re-forms blocks of each size b for the coverage / runtime-vs-size figures.
- Methods on the same marginal and the same blocks, sharing the same normal draws:
  copula (Sigma_S = U_S U_S^T + diag D_S from the copula's factor rows, optionally truncated to
  rank r), indep (Sigma = I), exch (Sigma = (1-rho) I + rho 11^T, rho = one-way ANOVA ICC of the
  context PITs over context blocks formed by the same rule), and lmm on grouped tasks when
  lmm_dir holds scripts/beyondarena/fit_lmm.py's predictions (its own Gaussian marginal).
- Joint samples: z ~ N(0, Sigma_S), y = F^{-1}(Phi(z)) by linear interpolation of the marginal's
  quantile grid. Per block and method: energy score, variogram score (order vs_order), joint NLL
  per point (marginal + Gaussian-copula term), CRPS and central-90% coverage of the block mean,
  predicted mean off-diagonal correlation, sample CRPS of the rows (a check: it must match the
  grid CRPS, which is identical for every copula), and seconds.

Output: out_dir/<dataset>/<split>.csv.gz (one row per block x method x block size) and
out_dir/<dataset>.json (per-split summaries, the spec, its fingerprint). A rerun with the same
settings skips the splits already written.
"""

from __future__ import annotations

import json
import os
import time
import zlib
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omegaconf import MISSING

from copula_inter.artifacts import atomic_write, canonical_digest
from eval.runners.hydra_cli import hydra_entry

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "BeyondArenaJointSpec",
    "anova_icc",
    "block_metrics",
    "cross_fit_folds",
    "exchangeable",
    "keep_columns",
    "make_blocks",
    "sample_block",
    "truncate_rank",
]

REGIMES = {"random": "iid", "iid": "iid", "grouped": "grouped", "temporal": "temporal"}


@dataclass
class BeyondArenaJointSpec:
    """One BeyondArena regression dataset: joint samples over test blocks for one marginal + copula."""

    # Dataset name (data_dir/<dataset>/meta.json).
    dataset: str = MISSING
    # Method tag written to the tables (tabicl-ft, tabpfn-pre, ...).
    method: str = MISSING
    # Copula checkpoint; null runs indep / exch / lmm only.
    ckpt: str | None = None
    # Marginal backend and checkpoint, as in ctr23_full_eval (tabicl_ckpt for tabicl, marginal_ckpt otherwise).
    marginal: str = "tabicl"
    tabicl_ckpt: str | None = None
    marginal_ckpt: str | None = None
    # Exploratory transfer test: score the copula on a marginal it was not trained with.
    allow_unpaired: bool = False
    data_dir: str | None = None
    # Splits to run ("r<repeat>_f<fold>"); empty = every exported split.
    splits: list[str] = field(default_factory=list)
    # Ablations.
    cross_fit: str = "struct"  # struct | random
    drop_group: bool = False
    # Backend input cap (TabPFN-3.5's schema takes at most 500 columns): wider splits keep a seeded random
    # subset of columns, always with the group column; null = all columns.
    max_features: int | None = None
    rank: int | None = None  # post-hoc truncation of the copula factor; null = full rank
    p_cop: int = 1000
    # Frozen settings (preregistration.md).
    p_marg: int = 50_000
    k_folds: int = 10
    block_size: int = 32
    block_max: int = 128
    max_blocks: int = 200
    block_sweep: list[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64, 128])
    n_samples: int = 1000
    vs_order: float = 0.5
    # Fitted LMM predictions (fit_lmm.py); null = no lmm row.
    lmm_dir: str | None = None
    seed: int = 42
    device: str = "cuda"
    query_batch: int = 2048
    tabicl_amp: bool = False
    out_dir: str = MISSING


def _validate(spec: BeyondArenaJointSpec) -> None:
    from copula_inter.backend_registry import MARGINAL_BACKENDS

    if spec.marginal not in MARGINAL_BACKENDS:
        raise ValueError(f"marginal={spec.marginal!r} is not one of {list(MARGINAL_BACKENDS)}")
    if spec.marginal == "tabicl" and not spec.tabicl_ckpt:
        raise ValueError("marginal=tabicl needs tabicl_ckpt (the fine-tuned marginal, or pretrained)")
    if spec.cross_fit not in ("struct", "random"):
        raise ValueError(f"cross_fit={spec.cross_fit!r}: expected struct or random")
    if spec.block_size < 2 or spec.block_max < 2 or spec.k_folds < 2 or spec.p_cop < 2:
        raise ValueError("need block_size, block_max, k_folds, p_cop >= 2")
    if spec.rank is not None and spec.rank < 1:
        raise ValueError("rank must be >= 1 or null")


# ---------------------------------------------------------------- blocks and folds


def make_blocks(
    regime: str,
    idx: np.ndarray,
    g: np.ndarray,
    t: np.ndarray,
    size: int,
    cap: int,
    max_blocks: int,
    rng: np.random.Generator,
    exact: bool = False,
) -> list[np.ndarray]:
    """Blocks of positions idx: whole groups (capped at cap rows), consecutive time windows, or random rows.

    exact=True keeps only blocks of exactly `size` rows (grouped: a random size-row subset of every group
    with at least size rows), for the block-size sweep. At most max_blocks blocks, a seeded draw.
    """
    blocks: list[np.ndarray] = []
    if regime == "grouped":
        for gid in rng.permutation(np.unique(g[idx])):
            rows = idx[g[idx] == gid]
            want = size if exact else cap
            if len(rows) < (size if exact else 2):
                continue
            blocks.append(np.sort(rng.choice(rows, min(want, len(rows)), replace=False)))
    else:
        order = idx[np.lexsort((rng.random(len(idx)), t[idx]))] if regime == "temporal" else rng.permutation(idx)
        blocks = [order[s : s + size] for s in range(0, len(order), size)]
        blocks = [b for b in blocks if len(b) == size or (not exact and len(b) >= 2)]
    if len(blocks) > max_blocks:
        keep = rng.choice(len(blocks), max_blocks, replace=False)
        blocks = [blocks[i] for i in np.sort(keep)]
    return blocks


def cross_fit_folds(
    regime: str, mode: str, g: np.ndarray, t: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Fold id in [0, k) of each context row: by whole group, by contiguous time range, or at random."""
    n = len(g)
    if mode == "struct" and regime == "grouped":
        groups = rng.permutation(np.unique(g))
        sizes = np.array([(g == gid).sum() for gid in groups])
        # Greedy: each group goes to the currently smallest fold (balanced fold sizes, whole groups).
        load = np.zeros(k)
        fold_of: dict[Any, int] = {}
        for gid, sz in zip(groups, sizes):
            f = int(np.argmin(load))
            fold_of[gid] = f
            load[f] += sz
        return np.array([fold_of[x] for x in g])
    if mode == "struct" and regime == "temporal":
        order = np.lexsort((rng.random(n), t))
        folds = np.empty(n, dtype=np.int64)
        folds[order] = np.arange(n) * k // n
        return folds
    return rng.permutation(np.arange(n) % k)


def anova_icc(z: np.ndarray, labels: np.ndarray) -> float:
    """One-way ANOVA intraclass correlation of z over the groups in labels (groups of one row are ignored)."""
    uniq, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    ok = counts[inv] >= 2
    z, inv = z[ok], inv[ok]
    if z.size < 4:
        return float("nan")
    _, inv = np.unique(inv, return_inverse=True)
    n_k = np.bincount(inv).astype(np.float64)
    K, N = len(n_k), float(len(z))
    if K < 2 or N <= K:
        return float("nan")
    means = np.bincount(inv, weights=z) / n_k
    msb = float((n_k * (means - z.mean()) ** 2).sum() / (K - 1))
    msw = float(((z - means[inv]) ** 2).sum() / (N - K))
    n0 = (N - float((n_k**2).sum()) / N) / (K - 1)
    den = msb + (n0 - 1.0) * msw
    return float((msb - msw) / den) if den > 0 else float("nan")


def keep_columns(n_cols: int, cap: int, group_col: int, seed: int) -> np.ndarray:
    """Sorted indices of a seeded random subset of `cap` columns that always contains group_col (if >= 0).

    Uses its own generator, so the blocks and folds drawn afterwards do not change.
    """
    rng = np.random.default_rng([seed, n_cols, cap])
    others = np.setdiff1d(np.arange(n_cols), [group_col] if group_col >= 0 else [])
    take = rng.choice(others, cap - (group_col >= 0), replace=False)
    return np.sort(np.concatenate([take, [group_col]] if group_col >= 0 else [take]).astype(np.int64))


def exchangeable(n: int, rho: float) -> np.ndarray:
    """(n, n) exchangeable correlation (1 - rho) I + rho 11^T."""
    return (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))


def truncate_rank(U: np.ndarray, D: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep the top-r directions of U U^T and move the rest to the diagonal (the unit diagonal is kept)."""
    if r >= min(U.shape):
        return U, D
    u, s, _ = np.linalg.svd(U, full_matrices=False)
    Ur = u[:, :r] * s[:r]
    return Ur, D + (U * U).sum(1) - (Ur * Ur).sum(1)


# ---------------------------------------------------------------- sampling and metrics


def sample_block(
    sigma: np.ndarray, eps: np.ndarray, q: np.ndarray, levels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(z (M, b), y (M, b)): z = eps L^T with L L^T = sigma, y_i = F_i^{-1}(Phi(z_i)) on the quantile grid q (b, Q)."""
    from scipy.special import ndtr

    L = np.linalg.cholesky(sigma + 1e-9 * np.eye(len(sigma)))
    z = eps @ L.T
    u = ndtr(z)
    y = np.column_stack([np.interp(u[:, i], levels, q[i]) for i in range(q.shape[0])])
    return z, y


def _crps_samples(x: np.ndarray, obs: float) -> float:
    """Sample CRPS E|X - obs| - E|X - X'| / 2 (sorted-sample formula, O(M log M))."""
    xs = np.sort(x)
    m = len(xs)
    spread = float(((2.0 * np.arange(1, m + 1) - m - 1) * xs).sum() / (m * m))
    return float(np.abs(xs - obs).mean() - spread)


def block_metrics(y: np.ndarray, Y: np.ndarray, vs_order: float) -> dict[str, float]:
    """Energy score, variogram score, block-mean CRPS / 90% coverage, mean row CRPS of samples Y (M, b) at y (b,)."""
    from eval.metrics.energy_score import compute_energy_score

    b = len(y)
    iu = np.triu_indices(b, 1)
    vd_obs = np.abs(y[:, None] - y[None, :])[iu] ** vs_order
    vd_pred = (np.abs(Y[:, :, None] - Y[:, None, :]) ** vs_order).mean(0)[iu]
    mean_s = Y.mean(1)
    lo, hi = np.quantile(mean_s, [0.05, 0.95])
    ybar = float(y.mean())
    return {
        "es": compute_energy_score(Y, y),
        "vs": float(((vd_obs - vd_pred) ** 2).sum()),
        "mean_crps": _crps_samples(mean_s, ybar),
        "mean_cov90": float(lo <= ybar <= hi),
        "mean_width90": float(hi - lo),
        "row_crps_sample": float(np.mean([_crps_samples(Y[:, i], y[i]) for i in range(b)])),
    }


def _gauss_copula_nll(sigma: np.ndarray, z: np.ndarray) -> float:
    """Summed Gaussian-copula NLL 0.5 (log det sigma + z^T sigma^{-1} z - z^T z)."""
    L = np.linalg.cholesky(sigma + 1e-9 * np.eye(len(sigma)))
    w = np.linalg.solve(L, z)
    return float(0.5 * (2.0 * np.log(np.diag(L)).sum() + w @ w - z @ z))


# ---------------------------------------------------------------- marginal / copula calls


def _impute(X_tr: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-impute NaN with the training column means (0 for an all-NaN column)."""
    with np.errstate(all="ignore"):
        mu = np.nanmean(X_tr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return np.where(np.isnan(X_tr), mu, X_tr), np.where(np.isnan(X_te), mu, X_te)


def _marginal_test(
    spec: BeyondArenaJointSpec,
    models: dict[str, Any],
    x_c: torch.Tensor,
    y_c: torch.Tensor,
    x_q: torch.Tensor,
    y_q: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """z, raw-unit log-density and raw-unit quantiles of query rows given the context (ctr23 helpers)."""
    from eval.runners.ctr23_full_eval import _backend_test_only, _tabicl_test_only

    if spec.marginal == "tabicl":
        return _tabicl_test_only(models["tabicl"], x_c, y_c, x_q, y_q)
    return _backend_test_only(spec.marginal, models["regressor"], x_c, y_c, x_q, y_q, spec.query_batch)


def _marginal_z(
    spec: BeyondArenaJointSpec,
    models: dict[str, Any],
    x_c: torch.Tensor,
    y_c: torch.Tensor,
    x_q: torch.Tensor,
    y_q: torch.Tensor,
) -> torch.Tensor:
    """z of query rows given the context only (no quantile grid)."""
    if spec.marginal == "tabicl":
        return _marginal_test(spec, models, x_c, y_c, x_q, y_q)["z_test"].double().cpu()
    from copula_inter.marginal.finetune_adapters import backend_pit

    mean, std = y_c.mean(), y_c.std().clamp(min=1e-8)
    xc, yc = [x_c.cpu().numpy()], [((y_c - mean) / std).cpu().numpy()]
    us = []
    for s in range(0, x_q.shape[0], spec.query_batch):
        u, _ = backend_pit(
            spec.marginal,
            models["regressor"],
            xc,
            yc,
            [x_q[s : s + spec.query_batch].cpu().numpy()],
            [((y_q[s : s + spec.query_batch] - mean) / std).cpu().numpy()],
        )
        us.append(u[0].double().cpu())
    return torch.special.ndtri(torch.cat(us).clamp(1e-6, 1 - 1e-6))


def _load_models(spec: BeyondArenaJointSpec) -> dict[str, Any]:
    from copula_inter.checkpoint_registry import resolve_marginal_checkpoint
    from copula_inter.marginal.pit import configure_tabicl_inference_amp, load_tabicl
    from eval.runners.ctr23_full_eval import check_marginal_pairing

    models: dict[str, Any] = {}
    if spec.marginal == "tabicl":
        assert spec.tabicl_ckpt is not None  # _validate
        configure_tabicl_inference_amp(spec.tabicl_amp)
        models["tabicl"] = load_tabicl(resolve_marginal_checkpoint(spec.tabicl_ckpt), spec.device)
    else:
        from copula_inter.marginal.backends import make_regressor

        models["regressor"] = make_regressor(spec.marginal, device=spec.device, ckpt=spec.marginal_ckpt)
    if spec.ckpt:
        from inference.copula_inference import load_copula_model

        models["copula"], cop_cfg = load_copula_model(spec.ckpt, device=spec.device)
        if not spec.allow_unpaired:
            check_marginal_pairing(cop_cfg, spec)
    return models


# ---------------------------------------------------------------- one split


def _score_split(
    spec: BeyondArenaJointSpec,
    models: dict[str, Any],
    d: dict[str, np.ndarray],
    regime: str,
    rng: np.random.Generator,
    lmm: dict[str, np.ndarray] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(per-block rows, split summary)."""
    from copula_inter.marginal.pit import normalize_features
    from eval.runners.ctr23_full_eval import _copula_rows, quantile_metrics

    dev = torch.device(spec.device)
    X_tr, X_te = d["X_tr"].astype(np.float64), d["X_te"].astype(np.float64)
    gf = int(d["group_feature"])
    if spec.drop_group and gf >= 0:
        X_tr, X_te = np.delete(X_tr, gf, axis=1), np.delete(X_te, gf, axis=1)
        gf = -1
    if spec.max_features is not None and X_tr.shape[1] > spec.max_features:
        cols = keep_columns(X_tr.shape[1], spec.max_features, gf, spec.seed)
        X_tr, X_te = X_tr[:, cols], X_te[:, cols]
    y_tr, y_te = d["y_tr"], d["y_te"]
    g_tr, g_te, t_tr, t_te = d["g_tr"], d["g_te"], d["t_tr"], d["t_te"]

    # Marginal context (<= p_marg rows) and copula context (p_cop of them).
    m_idx = np.arange(len(y_tr))
    if len(m_idx) > spec.p_marg:
        m_idx = np.sort(rng.choice(m_idx, spec.p_marg, replace=False))
    X_m, X_te = _impute(X_tr[m_idx], X_te)
    X_m, X_te = normalize_features(X_m, X_te)
    y_m, g_m, t_m = y_tr[m_idx], g_tr[m_idx], t_tr[m_idx]
    s = float(np.std(y_m)) or 1.0
    P = len(m_idx)
    cop_pos = np.sort(rng.choice(P, min(spec.p_cop, P), replace=False))
    # Context blocks for the exchangeable rho, formed by the test-block rule.
    ctx_blocks = make_blocks(regime, np.arange(P), g_m, t_m, spec.block_size, spec.block_max, 10**9, rng)
    # Whole context blocks in random order (time windows would otherwise come from the earliest period).
    exch_pos: list[int] = []
    exch_ids: list[int] = []
    for i in rng.permutation(len(ctx_blocks)):
        if len(exch_pos) >= spec.p_cop:
            break
        exch_pos.extend(ctx_blocks[i].tolist())
        exch_ids.extend([int(i)] * len(ctx_blocks[i]))
    exch_arr = np.array(exch_pos, dtype=np.int64)
    exch_lab = np.array(exch_ids, dtype=np.int64)
    need = np.union1d(cop_pos, exch_arr)

    to = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).to(dev)  # noqa: E731
    xm, ym = to(X_m), to(y_m)
    folds = cross_fit_folds(regime, spec.cross_fit, g_m, t_m, spec.k_folds, rng)
    z_need = np.full(P, np.nan)
    t0 = time.time()
    with torch.no_grad():
        for k in range(spec.k_folds):
            q_pos = need[folds[need] == k]
            if len(q_pos) == 0:
                continue
            ctx = np.flatnonzero(folds != k)
            z_need[q_pos] = _marginal_z(spec, models, xm[ctx], ym[ctx], xm[q_pos], ym[q_pos]).numpy()
    t_crossfit = time.time() - t0
    rho_exch = anova_icc(z_need[exch_arr], exch_lab)
    rho_exch = float(np.clip(rho_exch, 0.0, 0.99)) if np.isfinite(rho_exch) else 0.0

    # Blocks of test rows: the primary partition and the size sweep.
    all_te = np.arange(len(y_te))
    parts: dict[int, list[np.ndarray]] = {
        0: make_blocks(regime, all_te, g_te, t_te, spec.block_size, spec.block_max, spec.max_blocks, rng)
    }
    for b in spec.block_sweep:
        parts[int(b)] = make_blocks(regime, all_te, g_te, t_te, int(b), int(b), spec.max_blocks, rng, exact=True)
    rows_used = np.unique(np.concatenate([np.concatenate(v) for v in parts.values() if v]))
    pos_of = {int(r): i for i, r in enumerate(rows_used)}

    t0 = time.time()
    with torch.no_grad():
        m = _marginal_test(spec, models, xm, ym, to(X_te[rows_used]), to(y_te[rows_used]))
    t_marg = time.time() - t0
    q = m["q_test"].double().cpu().numpy() / s
    levels = m["levels"].double().cpu().numpy()
    logf = m["log_pdf_test"].double().cpu().numpy() + np.log(s)
    z = m["z_test"].double().cpu().numpy()
    y_s = y_te[rows_used] / s

    U = Dg = None
    t_cop = 0.0
    if "copula" in models:
        t0 = time.time()
        with torch.no_grad():
            Ut, Dt = _copula_rows(
                models["copula"],
                xm[cop_pos],
                torch.from_numpy(z_need[cop_pos]).float().to(dev),
                to(X_te[rows_used]),
                spec.query_batch,
            )
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_cop = time.time() - t0
        U, Dg = Ut.double().cpu().numpy(), Dt.double().cpu().numpy()

    out: list[dict[str, Any]] = []
    for B, blocks in parts.items():
        for bi, blk in enumerate(blocks):
            p = np.array([pos_of[int(r)] for r in blk])
            b = len(p)
            eps = rng.standard_normal((spec.n_samples, b))
            sigmas: dict[str, np.ndarray] = {"indep": np.eye(b), "exch": exchangeable(b, rho_exch)}
            if U is not None and Dg is not None:
                Ub, Db = U[p], Dg[p]
                if spec.rank is not None:
                    Ub, Db = truncate_rank(Ub, Db, spec.rank)
                sigmas["copula"] = Ub @ Ub.T + np.diag(Db)
            marg_nll = float(-logf[p].sum())
            grid_crps = quantile_metrics(q[p], levels, y_s[p])["crps"]
            base = {"B": B, "block": bi, "n": b, "grid_crps": grid_crps}
            for name, sig in sigmas.items():
                t0 = time.time()
                _, Y = sample_block(sig, eps, q[p], levels)
                met = block_metrics(y_s[p], Y, spec.vs_order)
                off = (sig.sum() - np.trace(sig)) / max(b * (b - 1), 1)
                cop = 0.0 if name == "indep" else _gauss_copula_nll(sig, z[p])
                out.append(
                    base
                    | met
                    | {
                        "method": name,
                        "joint_nll": (marg_nll + cop) / b,
                        "copula_nll": cop / b,
                        "rho_pred": float(off),
                        "seconds_sample": time.time() - t0,
                    }
                )
            if lmm is not None and regime == "grouped":
                mu = lmm["mu_te"][blk] / s
                sig = (lmm["sigma_b2"] * np.ones((b, b)) + lmm["sigma2"] * np.eye(b)) / s**2
                L = np.linalg.cholesky(sig)
                Y = mu[None] + eps @ L.T
                met = block_metrics(y_s[p], Y, spec.vs_order)
                r = y_s[p] - mu
                w = np.linalg.solve(L, r)
                nll = 0.5 * (b * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum() + w @ w)
                from scipy.stats import norm

                sd = float(np.sqrt(sig[0, 0]))
                zz = r / sd
                crps = sd * (zz * (2 * norm.cdf(zz) - 1) + 2 * norm.pdf(zz) - 1 / np.sqrt(np.pi))
                out.append(
                    base
                    | met
                    | {
                        "method": "lmm",
                        "grid_crps": float(crps.mean()),
                        "joint_nll": float(nll / b),
                        "copula_nll": float("nan"),
                        "rho_pred": float(lmm["sigma_b2"] / (lmm["sigma_b2"] + lmm["sigma2"])),
                        "seconds_sample": 0.0,
                    }
                )

    prim = parts[0]
    lab = np.concatenate([np.full(len(b), i) for i, b in enumerate(prim)]) if prim else np.zeros(0)
    zt = np.concatenate([z[[pos_of[int(r)] for r in b]] for b in prim]) if prim else np.zeros(0)
    summary = {
        "regime": regime,
        "P_train": int(d["n_train_full"]) if "n_train_full" in d else len(y_tr),
        "P_marg": P,
        "P_cop": len(cop_pos),
        "N_test": len(y_te),
        "N_scored": len(rows_used),
        "n_blocks": len(prim),
        "y_scale": s,
        "rho_exch": rho_exch,
        "icc_test_z": anova_icc(zt, lab),
        "icc_ctx_z": anova_icc(z_need[exch_arr], exch_lab),
        "z_test_std": float(z.std()),
        "seconds_crossfit": t_crossfit,
        "seconds_marginal": t_marg,
        "seconds_copula": t_cop,
        "seconds_copula_per_row": t_cop / max(len(rows_used), 1),
    }
    return out, summary


def _fingerprint(spec: BeyondArenaJointSpec) -> str:
    drop = {"splits", "device", "query_batch", "out_dir", "data_dir"}
    if spec.max_features is None:  # added after the first runs: unset keeps their fingerprint
        drop.add("max_features")
    return canonical_digest({k: v for k, v in asdict(spec).items() if k not in drop})


def run(spec: BeyondArenaJointSpec) -> None:
    """Score every requested split; each split's table is written as soon as it finishes."""
    import pandas as pd

    from copula_inter.paths import BEYONDARENA_DATA_DIR

    ddir = os.path.join(spec.data_dir or str(BEYONDARENA_DATA_DIR), spec.dataset)
    with open(os.path.join(ddir, "meta.json")) as fh:
        meta = json.load(fh)
    regime = REGIMES[meta["task_type"]]
    keys = list(spec.splits) or [f"r{r}_f{f}" for r, f in meta["splits"]]
    out_json = os.path.join(spec.out_dir, f"{spec.dataset}.json")
    fp = _fingerprint(spec)
    blob: dict[str, Any] = {}
    if os.path.isfile(out_json):
        with open(out_json) as fh:
            blob = json.load(fh)
        if blob.get("fingerprint") != fp:
            raise ValueError(f"{out_json} holds results of other settings; remove it or change out_dir")
    blob = {
        "fingerprint": fp,
        "spec": asdict(spec),
        "meta": {k: meta[k] for k in ("dataset", "task_type", "group_on", "time_on")},
        "splits": blob.get("splits", {}),
    }
    todo = [k for k in keys if k not in blob["splits"]]
    print(f"{spec.dataset} ({regime}): splits to run {todo}", flush=True)
    if not todo:
        return
    models = _load_models(spec)
    for key in todo:
        d = dict(np.load(os.path.join(ddir, f"{key}.npz")))
        rep, fold = (int(x[1:]) for x in key.split("_"))
        rng = np.random.default_rng([spec.seed, zlib.crc32(spec.dataset.encode()), rep, fold])
        lmm = None
        if spec.lmm_dir and os.path.isfile(p := os.path.join(spec.lmm_dir, spec.dataset, f"{key}.npz")):
            lmm = dict(np.load(p))
        rows, summary = _score_split(spec, models, d, regime, rng, lmm)
        frame: pd.DataFrame = pd.DataFrame(rows).assign(
            dataset=spec.dataset, split=key, regime=regime, model=spec.method
        )
        csv = os.path.join(spec.out_dir, spec.dataset, f"{key}.csv.gz")
        atomic_write(csv, lambda tmp: frame.to_csv(tmp, index=False, compression="gzip"))
        prim = frame[frame.B == 0].groupby("method")[["es", "joint_nll", "mean_cov90"]].mean()
        print(
            f"  {key}: {summary['n_blocks']} blocks, rho_exch {summary['rho_exch']:.3f}, icc_test {summary['icc_test_z']:.3f}\n{prim}",
            flush=True,
        )
        blob["splits"][key] = summary

        def _w(tmp: str) -> None:
            with open(tmp, "w") as fh:
                json.dump(blob, fh, indent=1)

        atomic_write(out_json, _w)


main = hydra_entry("beyondarena_joint_eval", BeyondArenaJointSpec, run, validate=_validate)


if __name__ == "__main__":
    main()
