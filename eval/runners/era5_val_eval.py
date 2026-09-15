"""era5_val_eval.py — held-out-year real-ERA5 evaluation of one copula
checkpoint paired with one Phase-A fine-tuned marginal checkpoint.

WHY A SEPARATE RUNNER. The existing real-ERA5 scorer
(eval/spatial/sweep_core.py::run_real_config) is built around the curated
named regions in eval/configs/regions.py, re-fetched from ARCO-ERA5 per
call, and it takes its marginal from the copula checkpoint's own stored
`cfg.tabicl` — which pins whatever path that run was launched with. Neither
is what we want here:

  * TRAIN/VAL DISJOINTNESS. The Phase-A marginal was fine-tuned on
    eval/data/cache/era5_global_train (1990-01 .. 2022-12). The only corpus
    guaranteed disjoint from it is eval/data/cache/era5_global_val (2023),
    which is the default --corpus-dir here. Scoring on curated regions
    fetched at fetch_era5.py's own default start date would be a silent
    coin-flip on whether the marginal has seen those days.
  * EXPLICIT MARGINAL. --marginal-ckpt overrides the checkpoint's stored
    path, so a post-reorg checkpoint whose recorded `pit_ckpt` no longer
    resolves is still scorable, and so the marginal under test is stated at
    the call site rather than inherited.
  * PER-DAY RESOLUTION. Episodes carry the corpus day index they were drawn
    from, so the result is a distribution over many distinct days of the
    held-out year rather than a single scalar.

SCORING. Identical convention to run_real_config: OWN-MARGINAL Sklar
(see eval/metrics/joint_nll.py's "NAMING TRAP" note). Per held-out point,

    total = marginal + copula
    marginal = -mean log f_i(y_i)      (the marginal ckpt alone)
    copula   = 0.5(log|R| + z'R^-1 z - z'z) / N

The copula term is EXACTLY ZERO at R = I, so `copula` is already the
signed nats/point the learned correlation buys over an independence copula
using the identical marginal — no separate independence run is needed, and
`marginal` is itself the independence-copula total.

Usage:
    python eval/runners/era5_val_eval.py \\
        --copula-ckpt checkpoints/copula_nano/copula-finetune-marginal-float32/step_0630000.pt \\
        --marginal-ckpt checkpoints/marginal/ablations/marginal_finetune_era5_run1/step_0177600_final.pt
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from eval.configs.constants import NLL_PROBS, PIT_K_FOLDS  # noqa: E402
from eval.data.era5_global_corpus import GlobalERA5Corpus  # noqa: E402
from eval.metrics.joint_nll import compute_joint_nll, compute_pit  # noqa: E402
from eval.spatial.diagnostics import (  # noqa: E402
    _forward_correlation,
    compute_context_z_train,
    load_copula_model,
)
from eval.tabicl_utils import make_tabicl_regressor, tabicl_quantiles  # noqa: E402

_DEFAULT_VAL_CORPUS = os.path.join(_REPO_ROOT, "eval", "data", "cache", "era5_global_val")
_DEFAULT_TRAIN_CORPUS = os.path.join(_REPO_ROOT, "eval", "data", "cache", "era5_global_train")
_DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "eval", "reports", "era5_val_eval")


# ---------------------------------------------------------------------------
# Corpus day index -> calendar date
# ---------------------------------------------------------------------------
def build_day_calendar(corpus: GlobalERA5Corpus) -> list:
    """Map every global day index the corpus exposes to a (year, month, day)
    triple, by walking the sorted monthly file list the same way
    GlobalERA5Corpus._cum_days does. Needed because sample_episode returns a
    flat day index into the concatenated corpus, but a per-day/per-month
    breakdown of the results wants real dates."""
    if corpus._paths is None:
        raise ValueError("Corpus was attached from shared memory; no file paths to date from.")
    out = []
    for p in corpus._paths:
        m = re.search(r"era5_global_t2m_(\d{4})(\d{2})\.nc", os.path.basename(p))
        if not m:
            raise ValueError(f"Unparseable corpus filename: {p}")
        yy, mm = int(m.group(1)), int(m.group(2))
        for dd in range(1, calendar.monthrange(yy, mm)[1] + 1):
            out.append((yy, mm, dd))
    if len(out) != corpus.n_days_total:
        raise ValueError(f"Day calendar length {len(out)} != corpus.n_days_total {corpus.n_days_total}")
    return out


def corpus_year_range(cache_dir: str) -> str:
    months = sorted(
        re.search(r"(\d{6})\.nc", f).group(1)
        for f in os.listdir(cache_dir) if f.startswith("era5_global_t2m_")
    )
    return f"{months[0][:4]}-{months[0][4:]} .. {months[-1][:4]}-{months[-1][4:]}" if months else "empty"


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------
def gp_correlation(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
    kernel: str, device: str, n_steps: int, lr: float,
) -> np.ndarray | None:
    """Posterior correlation matrix of a classical GP fitted by MLE on the
    same context, at the same held-out points. Used ONLY as a shared-marginal
    anchor: scored with the identical Phase-A quantile grid the model is
    scored with, so the two copula terms differ purely in correlation
    structure. Answers the question a lone "copula gain = -0.02 nats" cannot
    — whether the held-out points genuinely carry little dependence, or
    whether the model is under-claiming it.

    y is z-scored before fitting because fit_and_eval_gpytorch's MAP priors
    are tuned to data_gen.py's unit-variance synthetic scale, not ERA5
    Kelvin (same reason sweep_core.py::_fit_gp_baseline_nll does it). The
    correlation matrix is scale-free, so nothing needs rescaling back.
    """
    from eval.baselines.classical import fit_and_eval_gpytorch

    mu = float(y_train.mean())
    sd = max(float(y_train.std(ddof=1)), 1e-6) if len(y_train) > 1 else 1.0
    Xtr = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    Xte = torch.as_tensor(x_test, dtype=torch.float32, device=device)
    ytr = torch.as_tensor((y_train - mu) / sd, dtype=torch.float32, device=device)
    try:
        fit = fit_and_eval_gpytorch(
            Xtr, ytr, Xte, kernel, n_steps=n_steps, lr=lr,
            oracle_mode="posterior", n_restarts=1,
        )
    except Exception:  # noqa: BLE001
        return None
    return fit["R"].detach().cpu().numpy().astype(np.float64)


def score_episode(
    ep: dict, model, marginal, regressor, device: str, probs: np.ndarray,
    max_test: int, k_folds: int, rng: np.random.Generator,
    gp_kernel: str | None = None, gp_n_steps: int = 300, gp_lr: float = 0.05,
) -> dict | None:
    """Score one held-out ERA5 episode: context-conditioned correlation from
    the copula model, marginal quantile grid from the Phase-A checkpoint,
    joint NLL of the true held-out temperatures under both."""
    from scipy.stats import norm as _norm

    x_train = ep["x_norm_train"].astype(np.float64)
    x_test_all = ep["x_norm_test"].astype(np.float64)
    y_train = ep["y_train"].astype(np.float64)
    y_test_all = ep["y_test"].astype(np.float64)

    # Context PIT (K-fold LOO) under the fine-tuned marginal -> z_train, the
    # copula model's conditioning signal. This is the real-data analogue of
    # the GP-oracle LOO PIT training episodes use.
    z_train = compute_context_z_train(x_train, y_train, marginal, device, k_folds)
    if not np.all(np.isfinite(z_train)):
        return None

    # Forward pass over EVERY held-out point at once (not just the scored
    # subset): the copula head attends across test points, so restricting
    # the input set would change the correlations it predicts. Subsample
    # only afterwards, matching sweep_core.py::run_real_config.
    R_full = _forward_correlation(model, device, x_train, z_train, x_test_all)

    n_all = x_test_all.shape[0]
    n_score = min(max_test, n_all)
    sel = rng.choice(n_all, size=n_score, replace=False)
    sel.sort()
    R = R_full[np.ix_(sel, sel)]
    x_test = x_test_all[sel]
    y_test = y_test_all[sel]

    qgrid = tabicl_quantiles(regressor, x_train, y_train, x_test, probs)
    if not np.all(np.isfinite(qgrid)):
        return None

    nll = compute_joint_nll(qgrid, probs, R, y_test)
    # PIT of the true held-out values under the marginal alone: uniform iff
    # the marginal is calibrated. u is what the rank histogram is built from.
    z_pit, _ = compute_pit(qgrid, probs, y_test)
    u = _norm.cdf(z_pit)

    # Shared-marginal GP anchor: same qgrid, same y_test, different R.
    gp_copula, gp_total, gp_mean_abs_r = float("nan"), float("nan"), float("nan")
    if gp_kernel is not None:
        R_gp = gp_correlation(x_train, y_train, x_test, gp_kernel, device, gp_n_steps, gp_lr)
        if R_gp is not None and np.all(np.isfinite(R_gp)):
            try:
                gp_nll = compute_joint_nll(qgrid, probs, R_gp, y_test)
                gp_copula, gp_total = gp_nll["copula"], gp_nll["total"]
                gp_mean_abs_r = float(np.mean(np.abs(R_gp[np.triu_indices(n_score, k=1)])))
            except Exception:  # noqa: BLE001
                pass

    # Off-diagonal correlation strength the model predicts, for context on
    # how much dependence it is claiming on this episode.
    iu = np.triu_indices(n_score, k=1)
    return {
        "day_idx": int(ep["day_idx"]),
        "grid_size": int(ep["grid_size"]),
        "n_context": int(x_train.shape[0]),
        "n_test_total": int(n_all),
        "n_test_scored": int(n_score),
        "lat_center": float(np.mean(ep["lat_bounds"])),
        "lon_center": float(np.mean(ep["lon_bounds"])),
        "box_deg": float(ep["lat_bounds"][1] - ep["lat_bounds"][0]),
        "nll_total": nll["total"],
        "nll_marginal": nll["marginal"],   # == independence-copula total
        "nll_copula": nll["copula"],       # == signed gain over independence
        "mean_abs_r": float(np.mean(np.abs(R[iu]))),
        "gp_nll_copula": gp_copula,
        "gp_nll_total": gp_total,
        "gp_mean_abs_r": gp_mean_abs_r,
        "y_std": float(np.std(y_test)),
        "u": u.tolist(),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(rows: list, out_dir: str, cal: list, meta: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    total = np.array([r["nll_total"] for r in rows])
    marg = np.array([r["nll_marginal"] for r in rows])
    cop = np.array([r["nll_copula"] for r in rows])
    doy = np.array([r["day_idx"] for r in rows])
    month = np.array([cal[i][1] for i in doy])
    gsz = np.array([r["grid_size"] for r in rows])
    ctxfrac = np.array([r["n_context"] / (r["n_context"] + r["n_test_total"]) for r in rows])
    lat = np.array([r["lat_center"] for r in rows])
    gp_cop = np.array([r.get("gp_nll_copula", np.nan) for r in rows])
    gp_tot = np.array([r.get("gp_nll_total", np.nan) for r in rows])
    gp_ok = np.isfinite(gp_cop)
    has_gp = bool(gp_ok.any())

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.28)
    C_TOT, C_MAR, C_COP, C_GP = "#1f4e79", "#c0504d", "#4f8a3d", "#7b5ea7"

    # (1) NLL across the held-out year, per episode + monthly mean.
    ax = fig.add_subplot(gs[0, :2])
    ax.scatter(doy, marg, s=16, c=C_MAR, alpha=.45, label="marginal only (= independence copula)")
    ax.scatter(doy, total, s=16, c=C_TOT, alpha=.65, label="model total (marginal + copula)")
    series = [(marg, C_MAR), (total, C_TOT)]
    if has_gp:
        series.append((gp_tot, C_GP))
        ax.plot([], [], "-o", color=C_GP, lw=2, ms=5, label=f"{meta.get('gp_kernel')} GP (same marginal)")
    for arr, c in series:
        mm = [np.nanmean(arr[month == m]) for m in range(1, 13) if (month == m).any()]
        mx = [np.mean(doy[month == m]) for m in range(1, 13) if (month == m).any()]
        ax.plot(mx, mm, "-o", color=c, lw=2, ms=5, mec="w")
    ax.set_xlabel(f"day index into held-out corpus ({meta['val_range']})")
    ax.set_ylabel("NLL (nats / held-out point)")
    ax.set_title("Joint NLL across the held-out year — lines are monthly means", fontweight="bold")
    ax.legend(fontsize=8, framealpha=.9)
    ax.grid(alpha=.25)

    # (2) The copula term is the gain over independence (exactly 0 at R=I).
    ax = fig.add_subplot(gs[0, 2])
    bins = np.histogram_bin_edges(np.concatenate([cop, gp_cop[gp_ok]]) if has_gp else cop, bins=28)
    ax.hist(cop, bins=bins, color=C_COP, edgecolor="w", label="model")
    if has_gp:
        ax.hist(gp_cop[gp_ok], bins=bins, histtype="step", color=C_GP, lw=2,
                label=f"{meta.get('gp_kernel')} GP")
        ax.axvline(np.nanmean(gp_cop), color=C_GP, lw=2, ls=":",
                   label=f"GP mean {np.nanmean(gp_cop):+.3f}")
    ax.axvline(0, color="k", lw=1.4, ls="--", label="independence copula")
    ax.axvline(cop.mean(), color=C_COP, lw=2, label=f"model mean {cop.mean():+.3f}")
    ax.set_xlabel("copula NLL term (nats/pt)\n< 0 = copula beats independence")
    ax.set_ylabel("episodes")
    ax.set_title(f"Copula gain\n{100*np.mean(cop < 0):.0f}% of episodes improved", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25, axis="y")

    # (3) Paired per-episode comparison — the honest read, since episodes
    # differ wildly in intrinsic difficulty (region, season, resolution).
    ax = fig.add_subplot(gs[1, 0])
    lim = [min(marg.min(), total.min()) - .2, max(marg.max(), total.max()) + .2]
    ax.plot(lim, lim, "k--", lw=1, zorder=1)
    ax.scatter(marg, total, s=22, c=np.where(cop < 0, C_COP, C_MAR), alpha=.75, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("independence-copula NLL")
    ax.set_ylabel("model total NLL")
    ax.set_title("Paired, per episode\n(below diagonal = copula helps)", fontweight="bold")
    ax.grid(alpha=.25)

    # (4) Marginal calibration: PIT of true y under the fine-tuned marginal.
    ax = fig.add_subplot(gs[1, 1])
    u_all = np.concatenate([np.asarray(r["u"]) for r in rows])
    ax.hist(u_all, bins=20, range=(0, 1), color=C_MAR, edgecolor="w", density=True)
    ax.axhline(1.0, color="k", ls="--", lw=1.4, label="calibrated (uniform)")
    ax.set_xlabel("PIT u = F(y_true)")
    ax.set_ylabel("density")
    ax.set_title(f"Marginal calibration on held-out data\n"
                 f"n={len(u_all)} points, KS={meta['ks_stat']:.3f}", fontweight="bold")
    ax.legend(fontsize=8)

    # (5) Does the copula gain depend on how much of the field is observed?
    ax = fig.add_subplot(gs[1, 2])
    sc = ax.scatter(ctxfrac, cop, c=gsz, s=26, cmap="viridis", alpha=.85)
    ax.axhline(0, color="k", ls="--", lw=1.2)
    plt.colorbar(sc, ax=ax, label="grid size (points/side)")
    ax.set_xlabel("context fraction")
    ax.set_ylabel("copula NLL term (nats/pt)")
    ax.set_title("Copula gain vs. context fraction", fontweight="bold")
    ax.grid(alpha=.25)

    # (6) Monthly means, the seasonal view.
    ax = fig.add_subplot(gs[2, 0])
    ms = [m for m in range(1, 13) if (month == m).any()]
    w = .38
    ax.bar([m - w / 2 for m in ms], [marg[month == m].mean() for m in ms], w, color=C_MAR, label="independence")
    ax.bar([m + w / 2 for m in ms], [total[month == m].mean() for m in ms], w, color=C_TOT, label="model total")
    ax.set_xticks(ms)
    ax.set_xticklabels([calendar.month_abbr[m] for m in ms], fontsize=8)
    ax.set_ylabel("NLL (nats/pt)")
    ax.set_title("Seasonal breakdown", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25, axis="y")

    # (7) How much of the achievable dependence does the model capture? The
    # GP is fitted on the same context and scored with the same marginal, so
    # the only difference is the correlation matrix.
    ax = fig.add_subplot(gs[2, 1])
    if has_gp:
        lo = min(np.nanmin(gp_cop), cop.min()) - .02
        hi = max(np.nanmax(gp_cop), cop.max()) + .02
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="parity", zorder=1)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.axvline(0, color="gray", ls=":", lw=1)
        ax.scatter(gp_cop[gp_ok], cop[gp_ok], s=24, c=C_GP, alpha=.75, zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"{meta.get('gp_kernel')} GP copula term (nats/pt)")
        ax.set_ylabel("model copula term (nats/pt)")
        ax.set_title("Model vs. GP correlation, shared marginal\n"
                     "(above diagonal = GP captures more)", fontweight="bold")
        ax.legend(fontsize=8)
    else:
        ax.scatter(lat, cop, s=22, c=C_COP, alpha=.75)
        ax.axhline(0, color="k", ls="--", lw=1.2)
        ax.set_xlabel("episode centre latitude (deg)")
        ax.set_ylabel("copula NLL term (nats/pt)")
        ax.set_title("Copula gain vs. latitude", fontweight="bold")
    ax.grid(alpha=.25)

    # (8) Summary panel.
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    txt = (
        f"HELD-OUT ERA5 EVALUATION\n"
        f"{'-'*42}\n"
        f"copula   {meta['copula_ckpt_short']}\n"
        f"marginal {meta['marginal_ckpt_short']}\n\n"
        f"val corpus  {meta['val_range']}\n"
        f"marg. train {meta['train_range']}  (disjoint)\n\n"
        f"{meta['n_episodes']} episodes / {meta['n_distinct_days']} distinct days\n"
        f"{meta['n_points']} held-out points scored\n"
        f"{'-'*42}\n"
        f"{'':<14}{'mean':>8}{'+-sem':>8}{'median':>9}\n"
        f"{'total':<14}{total.mean():>8.3f}{total.std(ddof=1)/np.sqrt(len(total)):>8.3f}{np.median(total):>9.3f}\n"
        f"{'marginal':<14}{marg.mean():>8.3f}{marg.std(ddof=1)/np.sqrt(len(marg)):>8.3f}{np.median(marg):>9.3f}\n"
        f"{'copula':<14}{cop.mean():>8.3f}{cop.std(ddof=1)/np.sqrt(len(cop)):>8.3f}{np.median(cop):>9.3f}\n"
        f"{'-'*42}\n"
        f"copula gain vs independence:\n"
        f"  {-cop.mean():+.3f} nats/pt  ({100*np.mean(cop<0):.0f}% of eps)\n"
        f"  paired t = {meta['t_stat']:.2f}\n"
    )
    if has_gp and "mean_gp_copula" in meta:
        txt += (
            f"{'-'*42}\n"
            f"{meta['gp_kernel']} GP anchor (same marginal)\n"
            f"  copula   {meta['mean_gp_copula']:+.3f} +-{meta['sem_gp_copula']:.3f}\n"
            f"  model captures {100*meta['copula_gain_fraction_of_gp']:.0f}% of it\n"
            f"  mean|r|  model {meta['mean_abs_r_model']:.3f} "
            f"/ GP {meta['mean_abs_r_gp']:.3f}\n"
        )
    ax.text(0, 1, txt, family="monospace", fontsize=9.5, va="top", ha="left")

    fig.suptitle(
        "Copula checkpoint on held-out ERA5 (validation year only, unseen by the marginal)",
        fontsize=14, fontweight="bold", y=.965,
    )
    path = os.path.join(out_dir, "era5_val_eval.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copula-ckpt", required=True)
    ap.add_argument("--marginal-ckpt", required=True,
                    help="Phase-A fine-tuned TabICL checkpoint; overrides the copula ckpt's stored pit_ckpt.")
    ap.add_argument("--corpus-dir", default=_DEFAULT_VAL_CORPUS,
                    help="Held-out corpus. Default: eval/data/cache/era5_global_val (2023).")
    ap.add_argument("--train-corpus-dir", default=_DEFAULT_TRAIN_CORPUS,
                    help="Only read to report the train date range, for the disjointness check.")
    ap.add_argument("--n-episodes", type=int, default=60)
    ap.add_argument("--max-test", type=int, default=128, help="Held-out points scored per episode.")
    ap.add_argument("--grid-min", type=int, default=8)
    ap.add_argument("--grid-max", type=int, default=28)
    ap.add_argument("--box-min", type=float, default=5.0)
    ap.add_argument("--box-max", type=float, default=25.0)
    ap.add_argument("--ctx-frac-min", type=float, default=0.05)
    ap.add_argument("--ctx-frac-max", type=float, default=0.4)
    ap.add_argument("--gp-kernel", default="matern32",
                    help="Shared-marginal classical-GP correlation anchor. 'none' to skip.")
    ap.add_argument("--gp-n-steps", type=int, default=300)
    ap.add_argument("--gp-lr", type=float, default=0.05)
    ap.add_argument("--k-folds", type=int, default=PIT_K_FOLDS)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    val_range = corpus_year_range(args.corpus_dir)
    train_range = (corpus_year_range(args.train_corpus_dir)
                   if os.path.isdir(args.train_corpus_dir) else "n/a")
    print(f"[era5_val_eval] device={device}")
    print(f"[era5_val_eval] val corpus   {args.corpus_dir}  ({val_range})")
    print(f"[era5_val_eval] marginal was trained on {train_range} — must be disjoint from the above")

    print(f"[era5_val_eval] loading copula checkpoint {args.copula_ckpt}")
    model, cfg, device = load_copula_model(args.copula_ckpt, device=device)
    print(f"[era5_val_eval] checkpoint's own recorded marginal: {cfg.tabicl.get('pit_ckpt', None)}")
    print(f"[era5_val_eval] OVERRIDING marginal with {args.marginal_ckpt}")

    from src.pit import load_tabicl
    marginal = load_tabicl(args.marginal_ckpt, device)                    # K-fold context PIT
    regressor = make_tabicl_regressor(args.marginal_ckpt, device=device)  # test-point quantiles

    corpus = GlobalERA5Corpus(args.corpus_dir)
    cal = build_day_calendar(corpus)
    print(f"[era5_val_eval] corpus holds {corpus.n_days_total} days; drawing {args.n_episodes} episodes")

    gp_kernel = None if args.gp_kernel.lower() == "none" else args.gp_kernel
    if gp_kernel:
        print(f"[era5_val_eval] shared-marginal GP anchor: {gp_kernel} "
              f"({args.gp_n_steps} MLE steps, posterior correlation)")

    rng = np.random.default_rng(args.seed)
    rows, attempts, t0 = [], 0, time.time()
    while len(rows) < args.n_episodes and attempts < args.n_episodes * 20:
        attempts += 1
        ep = corpus.sample_episode(
            rng, (args.grid_min, args.grid_max), (args.box_min, args.box_max),
            (args.ctx_frac_min, args.ctx_frac_max),
        )
        if ep is None:
            continue
        try:
            row = score_episode(ep, model, marginal, regressor, device, NLL_PROBS,
                                args.max_test, args.k_folds, rng,
                                gp_kernel=gp_kernel, gp_n_steps=args.gp_n_steps, gp_lr=args.gp_lr)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ep {len(rows)}] failed: {type(exc).__name__}: {exc}")
            continue
        if row is None:
            continue
        rows.append(row)
        y, m, d = cal[row["day_idx"]]
        print(f"  [{len(rows):>3}/{args.n_episodes}] {y}-{m:02d}-{d:02d} "
              f"g{row['grid_size']:>2} P={row['n_context']:>3} N={row['n_test_scored']:>3} "
              f"lat{row['lat_center']:+6.1f} | total {row['nll_total']:+7.3f} "
              f"= marg {row['nll_marginal']:+7.3f} + cop {row['nll_copula']:+7.3f} "
              f"| |r|={row['mean_abs_r']:.3f} | gp cop {row['gp_nll_copula']:+7.3f} "
              f"|r|={row['gp_mean_abs_r']:.3f}  ({time.time()-t0:.0f}s)")

    if not rows:
        raise SystemExit("No episodes scored.")

    from scipy.stats import kstest, ttest_1samp
    total = np.array([r["nll_total"] for r in rows])
    marg = np.array([r["nll_marginal"] for r in rows])
    cop = np.array([r["nll_copula"] for r in rows])
    u_all = np.concatenate([np.asarray(r["u"]) for r in rows])

    meta = {
        "copula_ckpt": args.copula_ckpt,
        "marginal_ckpt": args.marginal_ckpt,
        "copula_ckpt_short": "/".join(args.copula_ckpt.rstrip("/").split("/")[-2:]),
        "marginal_ckpt_short": "/".join(args.marginal_ckpt.rstrip("/").split("/")[-2:]),
        "corpus_dir": args.corpus_dir,
        "val_range": val_range,
        "train_range": train_range,
        "n_episodes": len(rows),
        "n_distinct_days": len({r["day_idx"] for r in rows}),
        "n_points": int(sum(r["n_test_scored"] for r in rows)),
        "max_test": args.max_test,
        "seed": args.seed,
        "ks_stat": float(kstest(u_all, "uniform").statistic),
        "ks_pvalue": float(kstest(u_all, "uniform").pvalue),
        "t_stat": float(ttest_1samp(cop, 0.0).statistic),
        "t_pvalue": float(ttest_1samp(cop, 0.0).pvalue),
        "mean_total": float(total.mean()), "sem_total": float(total.std(ddof=1) / np.sqrt(len(total))),
        "mean_marginal": float(marg.mean()), "sem_marginal": float(marg.std(ddof=1) / np.sqrt(len(marg))),
        "mean_copula": float(cop.mean()), "sem_copula": float(cop.std(ddof=1) / np.sqrt(len(cop))),
        "frac_episodes_copula_helps": float(np.mean(cop < 0)),
        "gp_kernel": gp_kernel,
    }

    gp_cop = np.array([r["gp_nll_copula"] for r in rows])
    gp_ok = np.isfinite(gp_cop)
    if gp_ok.any():
        gp_tot = np.array([r["gp_nll_total"] for r in rows])
        meta.update({
            "gp_n_scored": int(gp_ok.sum()),
            "mean_gp_copula": float(np.nanmean(gp_cop)),
            "sem_gp_copula": float(np.nanstd(gp_cop[gp_ok], ddof=1) / np.sqrt(gp_ok.sum())),
            "mean_gp_total": float(np.nanmean(gp_tot)),
            "mean_abs_r_model": float(np.mean([r["mean_abs_r"] for r in rows])),
            "mean_abs_r_gp": float(np.nanmean([r["gp_mean_abs_r"] for r in rows])),
            # Fraction of the achievable (GP) copula gain the model captures.
            "copula_gain_fraction_of_gp": (
                float(np.nanmean(cop[gp_ok]) / np.nanmean(gp_cop[gp_ok]))
                if np.nanmean(gp_cop[gp_ok]) < 0 else float("nan")
            ),
        })

    json_path = os.path.join(args.out_dir, "era5_val_eval.json")
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "episodes": [{k: v for k, v in r.items() if k != "u"} for r in rows]}, f, indent=2)

    png_path = make_plots(rows, args.out_dir, cal, meta)

    print("\n" + "=" * 78)
    print(f"HELD-OUT ERA5 ({val_range}) — {len(rows)} episodes / {meta['n_distinct_days']} distinct days")
    print(f"  marginal trained on {train_range} (disjoint)")
    print("=" * 78)
    print(f"{'component':<28}{'mean':>10}{'+-sem':>9}{'median':>10}")
    for name, arr in (("total (marginal+copula)", total), ("marginal / independence", marg),
                      ("copula term", cop)):
        print(f"{name:<28}{arr.mean():>10.4f}{arr.std(ddof=1)/np.sqrt(len(arr)):>9.4f}{np.median(arr):>10.4f}")
    print("-" * 78)
    print(f"copula gain vs. independence : {-cop.mean():+.4f} nats/pt "
          f"(paired t={meta['t_stat']:.2f}, p={meta['t_pvalue']:.2e})")
    print(f"episodes where copula helps  : {100*meta['frac_episodes_copula_helps']:.1f}%")
    if "mean_gp_copula" in meta:
        print(f"{gp_kernel} GP anchor (same marginal): copula {meta['mean_gp_copula']:+.4f} "
              f"+-{meta['sem_gp_copula']:.4f} nats/pt over {meta['gp_n_scored']} eps")
        print(f"  -> model captures {100*meta['copula_gain_fraction_of_gp']:.1f}% of the GP's copula gain")
        print(f"  -> mean |r| off-diag: model {meta['mean_abs_r_model']:.3f} vs GP {meta['mean_abs_r_gp']:.3f}")
    print(f"marginal PIT uniformity      : KS={meta['ks_stat']:.4f} (p={meta['ks_pvalue']:.2e})")
    print("=" * 78)
    print(f"\nfigure : {png_path}\nresults: {json_path}")


if __name__ == "__main__":
    main()
