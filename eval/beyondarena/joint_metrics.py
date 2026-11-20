"""Joint scores of a Gaussian-copula predictive over a whole BeyondArena test fold.

A joint predictive of the N test rows is given by
- the marginal of each row: a quantile grid q (N, Q) at `levels` (Q,), the PIT z_i = Phi^{-1}(F_i(y_i))
  and the log-density log f_i(y_i) of the observed target;
- a correlation Sigma of the normal scores, one of
    indep:      I;
    lowrank:    U U^T + diag(D) (U (N, r)), the copula's output;
    blockexch:  block-diagonal exchangeable, (1 - rho) I + rho 11^T within each block of `block_ids`.
Joint samples: Z ~ N(0, Sigma), Y_i = F_i^{-1}(Phi(Z_i)) by linear interpolation of the grid.

Scores, all on the whole test fold (every metric is lower = better unless named coverage):
- energy score over the N-vector, with the Monte-Carlo estimator mean_m ||Y_m - y|| - 1/2 mean_m ||Y_m - Y_m+1||
  (unbiased; reported divided by sqrt(N) so that folds of different sizes are on one scale);
- variogram score of order p over a seeded uniform sample of pairs, and over the structured pairs
  (same group, or time-adjacent), per pair;
- joint NLL per point: -(sum_i log f_i(y_i) + log c(z)) / N with the Gaussian-copula log-density
  log c(z) = -1/2 (log det Sigma + z^T Sigma^{-1} z - z^T z), exact (Woodbury for lowrank);
- marginal CRPS from the quantile grid (identical for every Sigma) and from the samples (a check);
- aggregates computed from the same joint samples: the mean of every test group, of time windows /
  random blocks of each size in `agg_sizes`, and of the whole fold: their sample CRPS and the
  coverage / width of their central 90% interval;
- predicted mean within-block correlation against the empirical one (one-way ANOVA ICC of z).
y is divided by `scale` (the training targets' std) before scoring, so the scores are comparable across
datasets; the NLL adds log(scale) accordingly.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

__all__ = [
    "anova_icc",
    "block_ids_for",
    "crps_samples",
    "joint_scores",
    "keep_columns",
    "lowrank_copula_logdensity",
    "sample_normal_scores",
    "truncate_rank",
]

AGG_SIZES = (2, 4, 8, 16, 32, 64, 128)


# ---------------------------------------------------------------- helpers


def anova_icc(z: np.ndarray, labels: np.ndarray) -> float:
    """One-way ANOVA intraclass correlation of z over the groups in labels (groups of one row are ignored)."""
    z = np.asarray(z, dtype=np.float64)
    _, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
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
    """Sorted indices of a seeded random subset of `cap` columns that always contains group_col (if >= 0)."""
    rng = np.random.default_rng([seed, n_cols, cap])
    others = np.setdiff1d(np.arange(n_cols), [group_col] if group_col >= 0 else [])
    take = rng.choice(others, cap - (group_col >= 0), replace=False)
    return np.sort(np.concatenate([take, [group_col]] if group_col >= 0 else [take]).astype(np.int64))


def truncate_rank(U: np.ndarray, D: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep the top-r directions of U U^T and move the rest to the diagonal (the unit diagonal is kept)."""
    if r >= min(U.shape):
        return U, D
    u, s, _ = np.linalg.svd(U, full_matrices=False)
    Ur = u[:, :r] * s[:r]
    return Ur, D + (U * U).sum(1) - (Ur * Ur).sum(1)


def block_ids_for(
    regime: str, groups: np.ndarray | None, times: np.ndarray | None, size: int, seed: int, n: int | None = None
) -> np.ndarray:
    """Block label of each row: the group (grouped), consecutive time windows of `size` rows (temporal),
    or a seeded random partition into blocks of `size` rows (iid)."""
    if regime == "grouped" and groups is not None:
        return np.unique(np.asarray(groups).astype(str), return_inverse=True)[1]
    if n is None:
        ref = groups if groups is not None else times
        if ref is None:
            raise ValueError("block_ids_for needs groups, times or n")
        n = len(ref)
    rng = np.random.default_rng([seed, n, size])
    if regime == "temporal" and times is not None:
        order = np.lexsort((rng.random(n), np.asarray(times, dtype=np.float64)))
    else:
        order = rng.permutation(n)
    ids = np.empty(n, dtype=np.int64)
    ids[order] = np.arange(n) // size
    return ids


def crps_samples(samples: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Sample CRPS of each column: samples (M, K), obs (K,) -> (K,), E|X - y| - E|X - X'| / 2 (sorted formula)."""
    xs, _ = torch.sort(samples, dim=0)
    m = xs.shape[0]
    w = (2.0 * torch.arange(1, m + 1, device=xs.device, dtype=xs.dtype) - m - 1)[:, None]
    return (xs - obs[None]).abs().mean(0) - (w * xs).sum(0) / (m * m)


# ---------------------------------------------------------------- the copula


def lowrank_copula_logdensity(U: torch.Tensor, D: torch.Tensor, z: torch.Tensor) -> float:
    """log c(z) of the Gaussian copula with Sigma = U U^T + diag(D), exact in float64 (Woodbury, O(N r^2))."""
    from eval.runners.ctr23_full_eval import chunk_copula_nll

    return -chunk_copula_nll(U, D, z)


def _blockexch_logdensity(z: torch.Tensor, ids: torch.Tensor, rho: float) -> float:
    """log c(z) for block-diagonal exchangeable Sigma: closed form per block (eigenvalues 1 - rho and 1 + (n-1) rho)."""
    if rho <= 0:
        return 0.0
    z = z.double()
    n_k = torch.bincount(ids).double()
    s_k = torch.zeros_like(n_k).index_add_(0, ids, z)
    keep = n_k > 0
    n_k, s_k = n_k[keep], s_k[keep]
    lam = 1.0 + (n_k - 1.0) * rho
    logdet = ((n_k - 1.0) * math.log(1.0 - rho) + torch.log(lam)).sum()
    # z^T Sigma^{-1} z = (|z|^2 - rho/(1+(n-1)rho) * s^2) / (1 - rho), summed over blocks
    quad = ((z * z).sum() - (rho / lam * s_k * s_k).sum()) / (1.0 - rho)
    return float(-0.5 * (logdet + quad - (z * z).sum()))


def sample_normal_scores(
    kind: str,
    n: int,
    m: int,
    gen: torch.Generator,
    device: torch.device,
    *,
    U: torch.Tensor | None = None,
    D: torch.Tensor | None = None,
    ids: torch.Tensor | None = None,
    rho: float = 0.0,
) -> torch.Tensor:
    """(m, n) draws of Z ~ N(0, Sigma) for Sigma = I, U U^T + diag(D), or block-exchangeable(rho, ids); O(m n r)."""
    eps = torch.randn(m, n, generator=gen, device=device)
    if kind == "indep":
        return eps
    if kind == "lowrank":
        assert U is not None and D is not None
        e_r = torch.randn(m, U.shape[1], generator=gen, device=device)
        return e_r @ U.T.float() + eps * D.float().clamp(min=0).sqrt()[None]
    if kind == "blockexch":
        assert ids is not None
        eta = torch.randn(m, int(ids.max()) + 1, generator=gen, device=device)
        return math.sqrt(max(rho, 0.0)) * eta[:, ids] + math.sqrt(1.0 - max(rho, 0.0)) * eps
    raise ValueError(kind)


def _to_y(zs: torch.Tensor, q: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Y = F^{-1}(Phi(Z)) column-wise by linear interpolation of the quantile grid q (n, Q); outside it, the end quantiles."""
    u = torch.special.ndtr(zs).clamp(levels[0], levels[-1])
    hi = torch.searchsorted(levels, u.contiguous()).clamp(1, len(levels) - 1)
    lo = hi - 1
    l_lo, l_hi = levels[lo], levels[hi]
    w = (u - l_lo) / (l_hi - l_lo)
    cols = torch.arange(q.shape[0], device=q.device)[None].expand_as(hi)
    return q[cols, lo] + w * (q[cols, hi] - q[cols, lo])


# ---------------------------------------------------------------- the scores


def joint_scores(
    pred: dict[str, Any],
    y: np.ndarray,
    regime: str,
    groups: np.ndarray | None,
    times: np.ndarray | None,
    *,
    n_samples: int = 1000,
    max_cells: float = 2e8,
    vs_order: float = 0.5,
    n_pairs: int = 200_000,
    exch_block: int = 32,
    agg_sizes: tuple[int, ...] = AGG_SIZES,
    seed: int = 0,
    device: str = "cuda",
) -> dict[str, dict[str, float]]:
    """{variant: {score: value}} for every joint variant the predictive supports, on the whole test fold.

    pred (from a joint model's ``predict_joint``): q (N, Q), levels (Q,), z (N,), logf (N,) in y's units,
    scale, rho_exch, and optionally U (N, r), D (N,) (the copula) or kind="gaussian" (the LMM: its own
    Gaussian marginal, rho = its intraclass correlation within test groups).
    Variants: indep, exch (block-exchangeable with the context's rho), copula (if U), lmm (kind gaussian).
    The variants share the normal draws (common random numbers), so their differences carry little noise.
    """
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    s = float(pred["scale"])
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float64) / s, device=dev).float()
    N = len(y_t)
    q = torch.as_tensor(pred["q"], device=dev).float() / s
    levels = torch.as_tensor(pred["levels"], device=dev).float()
    z = torch.as_tensor(pred["z"], device=dev).double()
    logf_sum = float(np.sum(pred["logf"])) + N * math.log(s)
    m = int(max(100, min(n_samples, max_cells // max(N, 1))))
    ids_np = block_ids_for(regime, groups, times, exch_block, seed, n=N)
    ids = torch.as_tensor(ids_np, device=dev)

    variants: dict[str, dict[str, Any]] = {}
    if pred.get("kind") == "gaussian":
        variants["lmm"] = {"kind": "blockexch", "rho": float(pred["rho_exch"])}
    else:
        variants["indep"] = {"kind": "indep"}
        variants["exch"] = {"kind": "blockexch", "rho": float(pred["rho_exch"])}
        if pred.get("U") is not None:
            variants["copula"] = {
                "kind": "lowrank",
                "U": torch.as_tensor(pred["U"], device=dev),
                "D": torch.as_tensor(pred["D"], device=dev),
            }

    # Pairs for the variogram score: uniform, and structured (same group / time-adjacent).
    rng = np.random.default_rng([seed, N])
    n_all = N * (N - 1) // 2
    if n_all <= n_pairs:
        iu = np.triu_indices(N, 1)
        pairs_u = (iu[0], iu[1])
    else:
        a = rng.integers(0, N, n_pairs)
        b = (a + rng.integers(1, N, n_pairs)) % N
        pairs_u = (a, b)
    pairs_s: tuple[np.ndarray, np.ndarray] | None = None
    if regime == "temporal" and times is not None:
        order = np.lexsort((rng.random(N), np.asarray(times, dtype=np.float64)))
        pairs_s = (order[:-1], order[1:])
    elif regime == "grouped" and groups is not None:
        a_list, b_list = [], []
        for idx in np.split(np.argsort(ids_np, kind="stable"), np.cumsum(np.bincount(ids_np))[:-1]):
            if len(idx) >= 2:
                k = min(len(idx) * (len(idx) - 1) // 2, max(1, n_pairs * len(idx) // N))
                a_ = rng.choice(idx, k)
                b_ = rng.choice(idx, k)
                ok = a_ != b_
                a_list.append(a_[ok])
                b_list.append(b_[ok])
        if a_list:
            pairs_s = (np.concatenate(a_list), np.concatenate(b_list))

    # Aggregates: whole groups, windows / random blocks of each size, the whole fold.
    aggs: dict[str, np.ndarray] = {"fold": np.zeros(N, dtype=np.int64)}
    if regime == "grouped" and groups is not None:
        aggs["group"] = ids_np
    for b in agg_sizes:
        if b < N:
            lab = block_ids_for(
                "temporal" if regime == "temporal" else "iid",
                None,
                times if regime == "temporal" else np.zeros(N),
                b,
                seed + b,
            )
            full = np.bincount(lab)[lab] == b  # complete blocks only
            aggs[f"B{b}"] = np.where(full, lab, -1)

    gen0 = torch.Generator(device=dev)
    out: dict[str, dict[str, float]] = {}
    grid_crps = float(_grid_crps(q, levels, y_t).mean())
    for name, spec in variants.items():
        gen0.manual_seed(seed)  # common random numbers across variants
        if spec["kind"] == "lowrank":
            Z = sample_normal_scores("lowrank", N, m, gen0, dev, U=spec["U"], D=spec["D"])
        elif spec["kind"] == "blockexch":
            Z = sample_normal_scores("blockexch", N, m, gen0, dev, ids=ids, rho=spec["rho"])
        else:
            Z = sample_normal_scores("indep", N, m, gen0, dev)
        Y = _to_y(Z, q, levels)
        del Z
        r: dict[str, float] = {"n_test": float(N), "n_samples": float(m), "grid_crps": grid_crps}
        d1 = (Y - y_t[None]).norm(dim=1).mean()
        d2 = (Y - Y.roll(1, dims=0)).norm(dim=1).mean()
        r["energy"] = float(d1 - 0.5 * d2) / math.sqrt(N)
        for tag, pr in (("uniform", pairs_u), ("struct", pairs_s)):
            if pr is None or len(pr[0]) == 0:
                continue
            a_t = torch.as_tensor(pr[0], device=dev)
            b_t = torch.as_tensor(pr[1], device=dev)
            obs = (y_t[a_t] - y_t[b_t]).abs() ** vs_order
            prd = ((Y[:, a_t] - Y[:, b_t]).abs() ** vs_order).mean(0)
            r[f"variogram_{tag}"] = float(((obs - prd) ** 2).mean())
        if spec["kind"] == "lowrank":
            logc = lowrank_copula_logdensity(spec["U"], spec["D"], z)
        elif spec["kind"] == "blockexch":
            logc = _blockexch_logdensity(z, ids, spec["rho"])
        else:
            logc = 0.0
        r["joint_nll"] = -(logf_sum + logc) / N
        r["copula_logdensity_per_point"] = logc / N
        r["sample_crps"] = float(crps_samples(Y, y_t).mean())
        for agg, lab in aggs.items():
            sel = lab >= 0
            if sel.sum() < 2:
                continue
            lab_t = torch.as_tensor(lab[sel], device=dev)
            n_b = torch.bincount(lab_t).float()
            keep = n_b > 0
            sums = torch.zeros(m, len(n_b), device=dev).index_add_(
                1, lab_t, Y[:, torch.as_tensor(np.flatnonzero(sel), device=dev)]
            )
            means = (sums / n_b.clamp(min=1)[None])[:, keep]
            obs = (
                torch.zeros(len(n_b), device=dev).index_add_(
                    0, lab_t, y_t[torch.as_tensor(np.flatnonzero(sel), device=dev)]
                )
                / n_b.clamp(min=1)
            )[keep]
            lo, hi = torch.quantile(means, torch.tensor([0.05, 0.95], device=dev), dim=0)
            r[f"agg_{agg}_crps"] = float(crps_samples(means, obs).mean())
            r[f"agg_{agg}_cov90"] = float(((obs >= lo) & (obs <= hi)).float().mean())
            r[f"agg_{agg}_width90"] = float((hi - lo).mean())
            r[f"agg_{agg}_count"] = float(keep.sum())
        r["rho_pred"] = _mean_within_block_corr(spec, ids, N)
        r["icc_test_z"] = anova_icc(z.cpu().numpy(), ids_np)
        out[name] = r
        del Y
    return out


def _grid_crps(q: torch.Tensor, levels: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """CRPS of each row from its quantile grid: 2 x trapezoid of the pinball loss over the levels."""
    err = y[:, None] - q
    pin = torch.maximum(levels[None] * err, (levels[None] - 1.0) * err)
    return 2.0 * torch.trapezoid(pin, levels, dim=1)


def _mean_within_block_corr(spec: dict[str, Any], ids: torch.Tensor, N: int) -> float:
    """Mean predicted off-diagonal correlation over the within-block pairs (pair-weighted)."""
    n_k = torch.bincount(ids).double()
    pairs = float((n_k * (n_k - 1)).sum())
    if pairs == 0:
        return float("nan")
    if spec["kind"] == "blockexch":
        return float(spec["rho"])
    if spec["kind"] == "indep":
        return 0.0
    U = spec["U"].double()
    S = torch.zeros(len(n_k), U.shape[1], dtype=torch.float64, device=U.device).index_add_(0, ids, U)
    off = (S * S).sum() - (U * U).sum()  # sum over i != j in the same block of U_i . U_j
    return float(off / pairs)
