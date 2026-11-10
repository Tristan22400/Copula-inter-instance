"""validate(): the training loop's validation pass, its metrics and figures."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import matplotlib

from copula_inter.training.era5_probes import _era5_marginal_variance_fig, _era5_viz_fig, _era5_z_samples_fig
from copula_inter.training.probe_batches import _corr_quality, _macro_average, size_grid_key

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from omegaconf import DictConfig

from copula_inter.copula.loss import y_space_nll
from copula_inter.copula.model import build_sigma
from copula_inter.marginal.pit import (
    gaussian_corr_kl,
    gp_analytical_posterior,
)


@dataclass
class _ValAccumulators:
    """Per-task/per-batch values collected over the validation batches."""

    cop_per_task: list[float] = field(default_factory=list)
    W_norms: list[float] = field(default_factory=list)
    s_vals: list[float] = field(default_factory=list)
    sigma_off: list[float] = field(default_factory=list)
    tabicl_total: list[float] = field(default_factory=list)
    tabicl_marginal: list[float] = field(default_factory=list)
    tabicl_copula: list[float] = field(default_factory=list)
    # Bayes-optimal ceiling, filled inline when val_episodes_meta is available,
    # else by _posterior_probe_pass.
    oracle_total: list[float] = field(default_factory=list)
    oracle_copula: list[float] = field(default_factory=list)
    nll_post: list[float] = field(default_factory=list)
    nll_post_copula: list[float] = field(default_factory=list)
    off_p_post: list[np.ndarray] = field(default_factory=list)
    off_o_post: list[np.ndarray] = field(default_factory=list)
    # KL(N(0,R_post) || N(0,Sigma))/n per episode -- see pit.gaussian_corr_kl.
    corr_kl: list[float] = field(default_factory=list)
    corr_kl_nonfinite: int = 0


def _extend_valid(dst: list[float], values: torch.Tensor, valid: torch.Tensor) -> None:
    dst.extend(values[valid].cpu().tolist())


def _sigma_diagnostics(
    acc: _ValAccumulators,
    out: dict,
    Sigma: torch.Tensor,
    z_test: torch.Tensor,
    test_mask: torch.Tensor,
    include_oracle_copula: bool = True,
) -> None:
    """Per-task copula NLL (when requested), W/s norms and Sigma entry statistics."""
    n_test = test_mask.sum(-1).float()  # (B,)
    valid = n_test >= 2
    if not valid.any():
        return
    mask_2d = test_mask.unsqueeze(-1) & test_mask.unsqueeze(-2)
    n_safe = n_test.clamp(min=1)
    N = Sigma.shape[1]

    if include_oracle_copula:
        eye = torch.eye(N, device=Sigma.device, dtype=Sigma.dtype).unsqueeze(0)
        S_safe = torch.where(mask_2d, Sigma, eye)
        L, info = torch.linalg.cholesky_ex(S_safe)
        if info.any():
            S_safe = S_safe + 1e-4 * eye
            L = torch.linalg.cholesky(S_safe)
        log_det = 2.0 * L.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).log().sum(-1)
        tmp = torch.linalg.solve_triangular(L, z_test.unsqueeze(-1), upper=False)
        S_inv_z = torch.linalg.solve_triangular(L.mT, tmp, upper=True).squeeze(-1)
        cop = 0.5 * (log_det + (z_test * S_inv_z).sum(-1) - (z_test**2).sum(-1)) / n_safe
        _extend_valid(acc.cop_per_task, cop, valid)

    # W row norms and s means over valid test rows (no s for tanhnorm).
    mask_f = test_mask.float()
    _extend_valid(acc.W_norms, (out["W"].float().norm(dim=-1) * mask_f).sum(-1) / n_safe, valid)
    s_raw = out.get("s")
    if s_raw is not None:
        _extend_valid(acc.s_vals, (s_raw.float() * mask_f).sum(-1) / n_safe, valid)

    ri, ci = torch.triu_indices(N, N, offset=1, device=Sigma.device)
    valid_off = mask_2d[:, ri, ci]  # (B, n_pairs) bool
    acc.sigma_off.extend(Sigma[:, ri, ci][valid_off].cpu().tolist())


def _single_episode_nll(
    model: nn.Module,
    cfg: DictConfig,
    jitter: float,
    device: str,
    batch: dict[str, torch.Tensor],
    b: int,
    n_tr: int,
    n_te: int,
    z_cache: dict,
) -> dict[str, torch.Tensor]:
    """Y-space NLL of episode b re-run on a cached PIT's z_train and scored under its z_test/log_pdf_test."""
    sub_batch = {
        "x_train": batch["x_train"][b : b + 1, :n_tr],
        "z_train": z_cache["z_train"][b, :n_tr].to(device).unsqueeze(0),
        "x_test": batch["x_test"][b : b + 1, :n_te],
    }
    Sigma = build_sigma(model(sub_batch), cfg, jitter=jitter)
    z_test = z_cache["z_test"][b, :n_te].to(device).unsqueeze(0)
    log_pdf = z_cache["log_pdf_test"][b, :n_te].to(device).unsqueeze(0)
    mask = torch.ones(1, n_te, dtype=torch.bool, device=device)
    return y_space_nll(Sigma, z_test, log_pdf, mask)


def _add_posterior_ceiling(acc: _ValAccumulators, episode: dict, Sigma_b: torch.Tensor, n: int) -> None:
    """Bayes-optimal ceiling for one episode from its kernel metadata (gp_analytical_posterior)."""
    try:
        post = gp_analytical_posterior(episode)
    except (KeyError, NotImplementedError):
        return  # rare unsupported kernel schema — see gp_analytical_posterior's docstring
    acc.nll_post.append(post["nll_post"] / n)  # raw-sum -> nats/point, matching y_space_nll
    acc.nll_post_copula.append(post["nll_post_copula"] / n)
    ri_p, ci_p = np.triu_indices(n, k=1)
    acc.off_p_post.append(Sigma_b[:n, :n].float().cpu().numpy()[ri_p, ci_p])
    acc.off_o_post.append(post["R_post"].cpu().numpy()[ri_p, ci_p])
    ckl = gaussian_corr_kl(Sigma_b[:n, :n].cpu(), post["R_post"].cpu())
    if math.isfinite(ckl):
        acc.corr_kl.append(ckl)
    else:
        acc.corr_kl_nonfinite += 1


def _score_val_batch(
    acc: _ValAccumulators,
    model: nn.Module,
    cfg: DictConfig,
    jitter: float,
    device: str,
    batch: dict[str, torch.Tensor],
    an_b: dict | None,
    z_cache_b: dict | None,
    eps_b: list[dict] | None,
    score_oracle: bool,
    score_copula_dispersion: bool = True,
) -> None:
    """Accumulate one validation batch's analytic-z, TabICL-marginal and ceiling scores."""
    out = model(batch)
    Sigma = build_sigma(out, cfg, jitter=jitter, test_mask=batch["test_mask"])

    # oracle_diag/* is computed in the analytic z-space: when the batch carries a
    # TabICL PIT, re-run the model on analytic_val_z's z_train and score against
    # its z_test. val/y_nll_* uses tabicl_val_z and is unaffected.
    if an_b is None:
        # data.z_train_source="analytic": the batch IS the analytic PIT.
        z_test_an = batch["z_test"].float()
        log_pdf_an = batch["log_pdf_test"].float()
    else:
        out = model({**batch, "z_train": an_b["z_train"].to(device)})
        Sigma = build_sigma(out, cfg, jitter=jitter, test_mask=batch["test_mask"])
        z_test_an = an_b["z_test"].to(device).float()
        log_pdf_an = an_b["log_pdf_test"].to(device).float()

    if score_oracle:
        parts_o = y_space_nll(Sigma, z_test_an, log_pdf_an, batch["test_mask"])
        acc.oracle_total.append(parts_o["total"].item())
        acc.oracle_copula.append(parts_o["copula"].item())

    _sigma_diagnostics(acc, out, Sigma, z_test_an, batch["test_mask"], include_oracle_copula=score_copula_dispersion)

    for b in range(Sigma.shape[0]):
        n = int(batch["test_mask"][b].sum())
        if n < 2:
            continue
        # TabICL-marginal NLL, conditioned on the precomputed TabICL z_train (feeds val/y_nll_*).
        if z_cache_b is not None:
            n_tr = int(batch["train_mask"][b].sum())
            if n_tr >= 2:
                parts = _single_episode_nll(model, cfg, jitter, device, batch, b, n_tr, n, z_cache_b)
                acc.tabicl_total.append(parts["total"].item())
                acc.tabicl_marginal.append(parts["marginal"].item())
                acc.tabicl_copula.append(parts["copula"].item())
        if score_oracle and eps_b is not None and b < len(eps_b):
            _add_posterior_ceiling(acc, eps_b[b], Sigma[b], n)


def _posterior_probe_pass(
    acc: _ValAccumulators, model: nn.Module, cfg: DictConfig, jitter: float, posterior_probe: dict
) -> None:
    """Fill the Bayes-optimal ceiling from the separately drawn posterior_probe."""
    pb = posterior_probe["batch"]
    Sigma_p = build_sigma(model(pb), cfg, jitter=jitter, test_mask=pb["test_mask"])
    parts_p = y_space_nll(Sigma_p, pb["z_test"].float(), pb["log_pdf_test"].float(), pb["test_mask"])
    acc.oracle_total.append(parts_p["total"].item())
    acc.oracle_copula.append(parts_p["copula"].item())
    for b, ep in enumerate(posterior_probe["episodes"]):
        n = int(ep["x_norm_test"].shape[0])
        if n < 1:
            continue
        try:
            post = gp_analytical_posterior(ep)
        except (KeyError, NotImplementedError):
            continue  # rare unsupported kernel schema — see gp_analytical_posterior's docstring
        acc.nll_post.append(post["nll_post"] / n)  # raw-sum -> nats/point, matching y_space_nll
        acc.nll_post_copula.append(post["nll_post_copula"] / n)
        if n >= 2:
            ri_p, ci_p = np.triu_indices(n, k=1)
            acc.off_p_post.append(Sigma_p[b, :n, :n].float().cpu().numpy()[ri_p, ci_p])
            acc.off_o_post.append(post["R_post"].cpu().numpy()[ri_p, ci_p])


def _sigma_metrics(acc: _ValAccumulators, *, include_oracle_diagnostics: bool, z_label: str) -> dict:
    """Copula NLL dispersion (when requested) and Sigma/W/s stats for the validation z-space."""
    metrics: dict = {}
    if include_oracle_diagnostics:
        metrics["oracle_diag/copula_nll_std"] = float(np.std(acc.cop_per_task)) if acc.cop_per_task else float("nan")
    if acc.sigma_off:
        off_arr = np.array(acc.sigma_off, dtype=np.float32)
        metrics[f"sigma_offdiag_mean_{z_label}"] = float(off_arr.mean())
        metrics[f"sigma_offdiag_std_{z_label}"] = float(off_arr.std())
        metrics[f"sigma_offdiag_abs_mean_{z_label}"] = float(np.abs(off_arr).mean())
    else:
        metrics[f"sigma_offdiag_mean_{z_label}"] = 0.0
        metrics[f"sigma_offdiag_std_{z_label}"] = 0.0
        metrics[f"sigma_offdiag_abs_mean_{z_label}"] = 0.0
    metrics[f"W_norm_mean_{z_label}"] = float(np.mean(acc.W_norms)) if acc.W_norms else 0.0
    metrics[f"s_mean_{z_label}"] = float(np.mean(acc.s_vals)) if acc.s_vals else 0.0
    return metrics


def _tabicl_metrics(acc: _ValAccumulators) -> dict:
    """TabICL-marginal total Y-space NLL, when available."""
    metrics: dict = {}
    if acc.tabicl_total:
        metrics["y_nll_total"] = float(np.mean(acc.tabicl_total))
        metrics["y_nll_marginal"] = float(np.mean(acc.tabicl_marginal))
        metrics["y_nll_copula"] = float(np.mean(acc.tabicl_copula))
    return metrics


def _oracle_metrics(acc: _ValAccumulators, *, include_oracle_diagnostics: bool) -> dict:
    """Model NLL in analytic z-space, the Bayes-optimal ceiling and their gaps."""
    if not include_oracle_diagnostics:
        return _tabicl_metrics(acc)

    metrics: dict = {}
    if acc.oracle_total:
        metrics["oracle_diag/copula_nll"] = float(np.mean(acc.oracle_copula))
        metrics["oracle_diag/total_nll"] = float(np.mean(acc.oracle_total))
    if acc.nll_post:
        oracle_posterior_nll = float(np.mean(acc.nll_post))
        metrics["y_nll_oracle_posterior"] = oracle_posterior_nll
        oracle_posterior_copula = float(np.mean(acc.nll_post_copula))
        if "oracle_diag/total_nll" in metrics:
            # gap_nll = total_nll - y_nll_oracle_posterior on the same episodes (>= 0 in expectation).
            metrics["oracle_diag/gap_nll"] = metrics["oracle_diag/total_nll"] - oracle_posterior_nll
            # gap_nll - copula_gap is the marginal gap: ~0 when both sides use the analytic marginal.
            metrics["oracle_diag/copula_gap"] = metrics["oracle_diag/copula_nll"] - oracle_posterior_copula
            # The total improvement correlation can give over independence.
            metrics["oracle_diag/copula_headroom"] = -oracle_posterior_copula
    if acc.off_p_post:
        cq_p = _corr_quality(np.concatenate(acc.off_p_post), np.concatenate(acc.off_o_post))
        metrics["oracle_diag/corr_pearson"] = cq_p["pearson"]
        metrics["oracle_diag/corr_mae"] = cq_p["mae"]
    if acc.corr_kl:
        metrics["oracle_diag/corr_kl"] = float(np.mean(acc.corr_kl))
        metrics["oracle_diag/corr_kl_p90"] = float(np.percentile(acc.corr_kl, 90))
    metrics["oracle_diag/corr_kl_nonfinite"] = float(acc.corr_kl_nonfinite)
    metrics.update(_tabicl_metrics(acc))
    return metrics


def _kernel_fit_metrics(
    model: nn.Module,
    cfg: DictConfig,
    jitter: float,
    device: str,
    family: str,
    probe_s: dict,
    z_cache_fam: dict | None,
) -> dict:
    """One kernel-family probe: analytic-z NLL, its ceiling and, if cached, the TabICL-marginal NLL."""
    metrics: dict = {}
    sbatch = probe_s["batch"]
    Sigma_s = build_sigma(model(sbatch), cfg, jitter=jitter, test_mask=sbatch["test_mask"])
    parts_s = y_space_nll(Sigma_s, sbatch["z_test"].float(), sbatch["log_pdf_test"].float(), sbatch["test_mask"])
    tot_s = parts_s["total"].item()
    metrics[f"oracle_diag/kernel_fit/{family}/copula_nll"] = parts_s["copula"].item()
    metrics[f"oracle_diag/kernel_fit/{family}/total_nll"] = tot_s

    nll_post_s: list[float] = []
    for ep in probe_s["episodes"]:
        n_s = int(ep["x_norm_test"].shape[0])
        if n_s < 1:
            continue
        try:
            post_s = gp_analytical_posterior(ep)
        except (KeyError, NotImplementedError):
            continue
        nll_post_s.append(post_s["nll_post"] / n_s)
    oracle_post_s = None
    if nll_post_s:
        oracle_post_s = float(np.mean(nll_post_s))
        metrics[f"oracle_diag/kernel_fit/{family}/gap_nll"] = tot_s - oracle_post_s

    # Same probes on TabICL's z_train, scored under TabICL's marginal (feeds adaptive_kernel_signal="tabicl").
    if z_cache_fam is None:
        return metrics
    n_train_s = sbatch["train_mask"].sum(-1)
    n_test_s = sbatch["test_mask"].sum(-1)
    tot: list[float] = []
    for b in range(sbatch["x_train"].shape[0]):
        n_tr = int(n_train_s[b])
        n_te = int(n_test_s[b])
        if n_tr < 2 or n_te < 1:
            continue
        parts = _single_episode_nll(model, cfg, jitter, device, sbatch, b, n_tr, n_te, z_cache_fam)
        tot.append(parts["total"].item())
    if tot:
        tot_s_tabicl = float(np.mean(tot))
        metrics[f"kernel_fit/{family}/total_nll_tabicl"] = tot_s_tabicl
        if oracle_post_s is not None:
            metrics[f"kernel_fit/{family}/gap_nll_tabicl"] = tot_s_tabicl - oracle_post_s
    return metrics


# Below this oracle copula gain (nats/point) copula_capture is a ratio of two
# numbers near 0 and is reported as NaN instead.
_SIZE_FIT_MIN_HEADROOM = 1e-3


def _nanmean_or_nan(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _size_fit_cell(model: nn.Module, cfg: DictConfig, jitter: float, cell: dict, chunk_size: int) -> dict[str, float]:
    """Model vs analytic-posterior NLL on one (P, N) cell, averaged over episodes with a finite oracle."""
    batch = cell["batch"]
    n_eps = int(batch["x_train"].shape[0])
    model_total = np.full(n_eps, np.nan)
    model_copula = np.full(n_eps, np.nan)
    for start in range(0, n_eps, chunk_size):
        sub = {k: v[start : start + chunk_size] for k, v in batch.items()}
        Sigma = build_sigma(model(sub), cfg, jitter=jitter, test_mask=sub["test_mask"])
        for i in range(Sigma.shape[0]):
            parts = y_space_nll(
                Sigma[i : i + 1],
                sub["z_test"][i : i + 1].float(),
                sub["log_pdf_test"][i : i + 1].float(),
                sub["test_mask"][i : i + 1],
            )
            model_total[start + i] = parts["total"].item()
            model_copula[start + i] = parts["copula"].item()
    ok = np.isfinite(cell["oracle_total"]) & np.isfinite(model_total)
    if not ok.any():
        return {}
    total = float(model_total[ok].mean())
    copula = float(model_copula[ok].mean())
    oracle_total = float(cell["oracle_total"][ok].mean())
    oracle_copula = float(cell["oracle_copula"][ok].mean())
    headroom = -oracle_copula
    return {
        "gap_nll": total - oracle_total,
        "copula_capture": copula / oracle_copula if headroom > _SIZE_FIT_MIN_HEADROOM else float("nan"),
    }


def _size_fit_metrics(
    model: nn.Module, cfg: DictConfig, jitter: float, size_grid_batches: dict[tuple[int, int], dict]
) -> dict[str, float]:
    """size_fit/* metrics: copula quality as a function of the context size P and the test-set size N.

    Per cell (P, N), in nats per test point on analytic z (so the marginal is
    exact on both sides and the gaps measure the copula only):
        gap_nll = NLL_model - NLL_oracle_posterior               (>= 0, 0 is optimal)
        copula_capture = copula_nll_model / copula_nll_oracle      (1 optimal, 0 independence, < 0 harmful)
    headroom shrinks as P grows (the posterior correlation weakens), so the raw
    copula NLL is not comparable across P; gap_nll and copula_capture are. Also
    the per-P (mean over N) and per-N (mean over P) marginals of gap_nll and
    copula_capture, their overall mean, and P_spread/N_spread = max - min of the
    per-P / per-N gap_nll (0 when the model is equally good at every size).
    """
    bcfg = cfg.get("baselines", {}) or {}
    chunk_size = int(bcfg.get("size_grid_batch_size", 8))
    cells = {pn: _size_fit_cell(model, cfg, jitter, cell, chunk_size) for pn, cell in size_grid_batches.items()}
    metrics: dict[str, float] = {}
    for (P, N), cell_metrics in cells.items():
        for name, value in cell_metrics.items():
            metrics[f"size_fit/{size_grid_key(P, N)}/{name}"] = value
    P_values = sorted({P for P, _ in cells})
    N_values = sorted({N for _, N in cells})
    for name in ("gap_nll", "copula_capture"):
        per_P = {
            P: _nanmean_or_nan([m.get(name, float("nan")) for (p, _), m in cells.items() if p == P]) for P in P_values
        }
        per_N = {
            N: _nanmean_or_nan([m.get(name, float("nan")) for (_, n), m in cells.items() if n == N]) for N in N_values
        }
        for P, value in per_P.items():
            metrics[f"size_fit/P{P}/{name}"] = value
        for N, value in per_N.items():
            metrics[f"size_fit/N{N}/{name}"] = value
        metrics[f"size_fit/mean_{name}"] = _nanmean_or_nan([m.get(name, float("nan")) for m in cells.values()])
        if name == "gap_nll":
            for axis, per in (("P", per_P), ("N", per_N)):
                finite = [v for v in per.values() if math.isfinite(v)]
                metrics[f"size_fit/{axis}_spread_gap_nll"] = (max(finite) - min(finite)) if finite else float("nan")
    return metrics


def _size_fit_figure(metrics: dict[str, float], size_grid_batches: dict[tuple[int, int], dict]) -> Figure | None:
    """Heatmaps of size_fit gap_nll and copula_capture over the (P, N) grid."""
    P_values = sorted({P for P, _ in size_grid_batches})
    N_values = sorted({N for _, N in size_grid_batches})
    if not P_values or not N_values:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name, cmap in zip(axes, ("gap_nll", "copula_capture"), ("viridis", "RdYlGn")):
        grid = np.array(
            [[metrics.get(f"size_fit/{size_grid_key(P, N)}/{name}", float("nan")) for P in P_values] for N in N_values]
        )
        if name == "copula_capture":
            im = ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=1.0, origin="lower", aspect="auto")
        else:
            im = ax.imshow(grid, cmap=cmap, origin="lower", aspect="auto")
        for j in range(len(N_values)):
            for i in range(len(P_values)):
                if math.isfinite(grid[j, i]):
                    ax.text(i, j, f"{grid[j, i]:.3f}", ha="center", va="center", fontsize=8, color="black")
        ax.set_xticks(range(len(P_values)), [str(P) for P in P_values])
        ax.set_yticks(range(len(N_values)), [str(N) for N in N_values])
        ax.set_xlabel("context size P")
        ax.set_ylabel("test-set size N")
        ax.set_title(name)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def _era5_fit_metrics(model: nn.Module, cfg: DictConfig, jitter: float, era5_val_batches: dict) -> dict:
    """Per-region Y-space NLL on each real-ERA5 probe's held-out points, GP baselines, macro averages."""
    metrics: dict = {}
    region_total: list[float] = []
    region_marginal: list[float] = []
    region_copula: list[float] = []
    region_gp_baseline_total: dict[str, list[float]] = {}
    for region, probe in era5_val_batches.items():
        Sigma_e = build_sigma(model(probe["batch"]), cfg, jitter=jitter, test_mask=probe["batch"]["test_mask"])
        if "nll_test_z" in probe:
            idx = torch.as_tensor(probe["nll_test_idx"], dtype=torch.long, device=Sigma_e.device)
            Sigma_nll = Sigma_e.index_select(1, idx).index_select(2, idx)
            z_nll, log_pdf_nll = probe["nll_test_z"], probe["nll_test_log_pdf"]
            parts_e = y_space_nll(Sigma_nll, z_nll, log_pdf_nll, torch.ones_like(z_nll, dtype=torch.bool))
            y_total = parts_e["total"].item()
            y_marginal = parts_e["marginal"].item()
            y_copula = parts_e["copula"].item()
            metrics[f"era5_fit/{region}/y_nll_total"] = y_total
            metrics[f"era5_fit/{region}/y_nll_copula"] = y_copula
            if not math.isnan(y_total):
                region_total.append(y_total)
                region_marginal.append(y_marginal)
                region_copula.append(y_copula)

        for kname, parts in probe.get("gp_baseline_nll", {}).items():
            metrics[f"era5_fit/{region}/gp_baseline_{kname}_nll_total"] = parts["total"]
            if not math.isnan(parts["total"]):
                region_gp_baseline_total.setdefault(kname, []).append(parts["total"])

    metrics["era5_fit/mean_y_nll_total"] = _macro_average(region_total)
    metrics["era5_fit/mean_y_nll_marginal"] = _macro_average(region_marginal)
    metrics["era5_fit/mean_y_nll_copula"] = _macro_average(region_copula)
    for kname, vals in region_gp_baseline_total.items():
        metrics[f"era5_fit/mean_gp_baseline_{kname}_nll_total"] = _macro_average(vals)
    return metrics


def _era5_figures(model: nn.Module, cfg: DictConfig, era5_viz_batch: dict, jitter: float, device: str) -> dict:
    """ERA5 prediction, residual, latent-sample and marginal-variance figures keyed by wandb name."""
    figs: dict = {}
    fig_pred, fig_resid = _era5_viz_fig(model, cfg, era5_viz_batch, jitter, device)
    figs["val/era5_predictions"] = fig_pred
    figs["val/era5_residuals"] = fig_resid
    figs["val/era5_predictions_z"] = _era5_z_samples_fig(model, cfg, era5_viz_batch, jitter, device)
    figs["val/era5_marginal_variance"] = _era5_marginal_variance_fig(era5_viz_batch)
    return {k: v for k, v in figs.items() if v is not None}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: Iterable[dict[str, torch.Tensor]],
    cfg: DictConfig,
    device: str,
    step: int = 0,
    do_plot: bool = False,
    synth_kernel_batches: dict | None = None,
    tabicl_val_z: dict | None = None,
    analytic_val_z: dict | None = None,
    tabicl_kernel_fit_z: dict | None = None,
    era5_val_batches: dict | None = None,
    era5_viz_batch: dict | None = None,
    posterior_probe: dict | None = None,
    val_episodes_meta: dict[int, list[dict]] | None = None,
    include_oracle_diagnostics: bool = True,
    size_grid_batches: dict[tuple[int, int], dict] | None = None,
) -> tuple[dict, dict]:
    """Score the model on the validation batches and the optional probes.

    Metric groups: oracle_diag/* scores the model against the analytic ground
    truth; the bare names hold TabICL-marginal scores and model-free references.

    Returns:
        (metrics, figures): metrics keyed "oracle_diag/..." or bare names (the
        caller adds "val/"), and {wandb key: matplotlib figure} when do_plot.
    """
    # No model.eval(): TabICL's eval-mode forward uses float16 autocast and can give NaN (the model has no dropout).
    jitter = float(cfg.model.get("sigma_jitter", 1e-4))

    acc = _ValAccumulators()
    for batch_idx, batch in enumerate(val_loader):
        _score_val_batch(
            acc,
            model,
            cfg,
            jitter,
            device,
            {k: v.to(device) for k, v in batch.items()},
            an_b=analytic_val_z.get(batch_idx) if analytic_val_z else None,
            z_cache_b=tabicl_val_z.get(batch_idx) if tabicl_val_z else None,
            eps_b=val_episodes_meta.get(batch_idx) if val_episodes_meta is not None else None,
            score_oracle=include_oracle_diagnostics and val_episodes_meta is not None,
            # Needs only Sigma and z_test, not the episode meta (on-disk / ERA5-live have none).
            score_copula_dispersion=include_oracle_diagnostics,
        )

    z_label = "analytic_z" if include_oracle_diagnostics else "val_z"
    metrics = _sigma_metrics(acc, include_oracle_diagnostics=include_oracle_diagnostics, z_label=z_label)
    if include_oracle_diagnostics and posterior_probe is not None and val_episodes_meta is None:
        _posterior_probe_pass(acc, model, cfg, jitter, posterior_probe)
    metrics.update(_oracle_metrics(acc, include_oracle_diagnostics=include_oracle_diagnostics))
    for family, probe_s in (synth_kernel_batches or {}).items():
        z_cache_fam = (tabicl_kernel_fit_z or {}).get(family)
        metrics.update(_kernel_fit_metrics(model, cfg, jitter, device, family, probe_s, z_cache_fam))
    metrics.update(_era5_fit_metrics(model, cfg, jitter, era5_val_batches or {}))
    if size_grid_batches:
        metrics.update(_size_fit_metrics(model, cfg, jitter, size_grid_batches))

    model.train()

    plot_figs: dict = {}
    if do_plot and era5_viz_batch is not None:
        plot_figs = _era5_figures(model, cfg, era5_viz_batch, jitter, device)
    if do_plot and size_grid_batches:
        size_fig = _size_fit_figure(metrics, size_grid_batches)
        if size_fig is not None:
            plot_figs["val/size_fit_grid"] = size_fig
    return metrics, plot_figs
