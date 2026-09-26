"""validate(): the training loop's validation pass, its metrics and figures."""

from __future__ import annotations

import math

from copula_inter.era5_probes import _era5_marginal_variance_fig, _era5_viz_fig, _era5_z_samples_fig
from copula_inter.probe_batches import _corr_quality, _macro_average


import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from copula_inter.loss import y_space_nll
from copula_inter.model import build_sigma
from copula_inter.pit import (
    gaussian_corr_kl,
    gp_analytical_posterior,
)

_PLOT_COLLECT_BATCHES = 5


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
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
) -> tuple[dict, dict]:
    """Score the model on the validation batches and the optional probes.

    Returns:
        (metrics, figures): metrics keyed "oracle_diag/..." or bare names (the
        caller adds "val/"), and {wandb key: matplotlib figure} when do_plot.
    """
    # No model.eval(): TabICL's eval-mode forward uses float16 autocast and can give NaN (the model has no dropout).
    jitter = float(cfg.model.get("sigma_jitter", 1e-4))

    cop_per_task: list[float] = []
    all_W_norms: list[float] = []
    all_s_vals: list[float] = []
    all_sigma_off: list[float] = []
    all_sigma_diag: list[float] = []
    all_tabicl_marginal_total: list[float] = []
    all_tabicl_marginal_marginal: list[float] = []
    all_tabicl_marginal_copula: list[float] = []

    # Bayes-optimal ceiling accumulators, filled inline when val_episodes_meta is
    # available, else by the posterior_probe pass after the loop.
    all_oracle_total: list[float] = []
    all_oracle_copula: list[float] = []
    nll_post_per_point: list[float] = []
    nll_post_marginal_per_point: list[float] = []
    nll_post_copula_per_point: list[float] = []
    off_p_post: list[np.ndarray] = []
    off_o_post: list[np.ndarray] = []
    # KL(N(0,R_post) || N(0,Sigma))/n per episode -- see pit.gaussian_corr_kl.
    corr_kl_vals: list[float] = []
    corr_kl_nonfinite = 0

    for batch_idx, batch in enumerate(val_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch)
        Sigma = build_sigma(out, cfg, jitter=jitter, test_mask=batch["test_mask"])

        # oracle_diag/* is computed in the analytic z-space: when the batch carries a
        # TabICL PIT, re-run the model on analytic_val_z's z_train and score against
        # its z_test. val/y_nll_* uses tabicl_val_z and is unaffected.
        an_b = analytic_val_z.get(batch_idx) if analytic_val_z else None
        if an_b is None:
            # data.z_train_source="analytic": the batch IS the analytic PIT.
            z_test_an = batch["z_test"].float()
            log_pdf_an = batch["log_pdf_test"].float()
        else:
            batch_an = dict(batch)
            batch_an["z_train"] = an_b["z_train"].to(device)
            with torch.no_grad():
                out = model(batch_an)
            Sigma = build_sigma(out, cfg, jitter=jitter, test_mask=batch["test_mask"])
            z_test_an = an_b["z_test"].to(device).float()
            log_pdf_an = an_b["log_pdf_test"].to(device).float()

        # Oracle-posterior total/copula NLL for this batch (when val_episodes_meta is available).
        eps_b = val_episodes_meta.get(batch_idx) if val_episodes_meta is not None else None
        if val_episodes_meta is not None:
            parts_o = y_space_nll(
                Sigma, z_test_an, log_pdf_an, batch["test_mask"]
            )
            all_oracle_total.append(parts_o["total"].item())
            all_oracle_copula.append(parts_o["copula"].item())

        # ---- Per-task diagnostics (vectorized — no Python loop over batch) ----
        n_test_cur = batch["test_mask"].sum(-1).float()   # (B,)
        valid_cur = n_test_cur >= 2

        if valid_cur.any():
            mask_2d_cur = batch["test_mask"].unsqueeze(-1) & batch["test_mask"].unsqueeze(-2)
            n_safe_cur = n_test_cur.clamp(min=1)
            N_cur = Sigma.shape[1]

            # Per-task copula NLL against the batch's analytic z_test (oracle_diag/copula_nll_std).
            eye_cur = torch.eye(N_cur, device=Sigma.device, dtype=Sigma.dtype).unsqueeze(0)
            S_safe_cur = torch.where(mask_2d_cur, Sigma, eye_cur)
            L_cur, info_cur = torch.linalg.cholesky_ex(S_safe_cur)
            if info_cur.any():
                S_safe_cur = S_safe_cur + 1e-4 * eye_cur
                L_cur = torch.linalg.cholesky(S_safe_cur)
            log_det_cur = 2.0 * L_cur.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).log().sum(-1)
            z_f = z_test_an
            tmp_cur = torch.linalg.solve_triangular(L_cur, z_f.unsqueeze(-1), upper=False)
            S_inv_z_cur = torch.linalg.solve_triangular(L_cur.mT, tmp_cur, upper=True).squeeze(-1)
            cop_cur = 0.5 * (log_det_cur + (z_f * S_inv_z_cur).sum(-1) - (z_f ** 2).sum(-1)) / n_safe_cur
            cop_per_task.extend(cop_cur[valid_cur].cpu().tolist())

            # W row norms and s means over valid test rows (no s for tanhnorm).
            W_f = out["W"].float()
            mask_f = batch["test_mask"].float()
            W_norm_cur = (W_f.norm(dim=-1) * mask_f).sum(-1) / n_safe_cur
            all_W_norms.extend(W_norm_cur[valid_cur].cpu().tolist())
            s_raw = out.get("s")
            if s_raw is not None:
                s_f = s_raw.float()
                s_mean_cur = (s_f * mask_f).sum(-1) / n_safe_cur
                all_s_vals.extend(s_mean_cur[valid_cur].cpu().tolist())

            # Off-diagonal and diagonal statistics (all valid entries in one shot)
            ri_cur, ci_cur = torch.triu_indices(N_cur, N_cur, offset=1, device=Sigma.device)
            valid_off_cur = mask_2d_cur[:, ri_cur, ci_cur]  # (B, n_pairs) bool
            off_vals_cur = Sigma[:, ri_cur, ci_cur][valid_off_cur]
            all_sigma_off.extend(off_vals_cur.cpu().tolist())
            all_sigma_diag.extend(Sigma.diagonal(dim1=-2, dim2=-1)[batch["test_mask"]].cpu().tolist())

        # TabICL-marginal NLL on every val batch (feeds val/y_nll_*).
        B = Sigma.shape[0]
        z_cache_b = tabicl_val_z.get(batch_idx) if tabicl_val_z else None
        for b in range(B):
            n = int(batch["test_mask"][b].sum())
            if n < 2:
                continue

            # Re-run the model on this batch conditioned on the precomputed TabICL z_train.
            if z_cache_b is not None:
                n_tr = int(batch["train_mask"][b].sum())
                if n_tr >= 2:
                    z_tabicl_b = z_cache_b["z_train"][b, :n_tr].to(device).unsqueeze(0)
                    sub_batch = {
                        "x_train": batch["x_train"][b : b + 1, :n_tr],
                        "z_train": z_tabicl_b,
                        "x_test":  batch["x_test"][b : b + 1, :n],
                    }
                    out_tabicl = model(sub_batch)
                    Sigma_tabicl = build_sigma(out_tabicl, cfg, jitter=jitter)

                    # Y-space NLL of the TabICL-conditioned Sigma under TabICL's own z_test and log_pdf_test.
                    z_test_tabicl_b = z_cache_b["z_test"][b, :n].to(device).unsqueeze(0)
                    log_pdf_tabicl_b = z_cache_b["log_pdf_test"][b, :n].to(device).unsqueeze(0)
                    mask_tabicl_b = torch.ones(1, n, dtype=torch.bool, device=device)
                    parts_tabicl_b = y_space_nll(
                        Sigma_tabicl, z_test_tabicl_b, log_pdf_tabicl_b, mask_tabicl_b
                    )
                    all_tabicl_marginal_total.append(parts_tabicl_b["total"].item())
                    all_tabicl_marginal_marginal.append(parts_tabicl_b["marginal"].item())
                    all_tabicl_marginal_copula.append(parts_tabicl_b["copula"].item())

            # Bayes-optimal ceiling per episode from its kernel metadata (gp_analytical_posterior).
            if eps_b is not None and b < len(eps_b):
                try:
                    post = gp_analytical_posterior(eps_b[b])
                except (KeyError, NotImplementedError):
                    pass  # rare unsupported kernel schema — see gp_analytical_posterior's docstring
                else:
                    nll_post_per_point.append(post["nll_post"] / n)  # raw-sum -> nats/point, matching y_space_nll
                    nll_post_marginal_per_point.append(post["nll_post_marginal"] / n)
                    nll_post_copula_per_point.append(post["nll_post_copula"] / n)
                    ri_p, ci_p = np.triu_indices(n, k=1)
                    off_p_post.append(Sigma[b, :n, :n].float().cpu().numpy()[ri_p, ci_p])
                    off_o_post.append(post["R_post"].cpu().numpy()[ri_p, ci_p])
                    # Correlation KL, a noise-free convergence signal (pit.gaussian_corr_kl).
                    ckl = gaussian_corr_kl(
                        Sigma[b, :n, :n].cpu(), post["R_post"].cpu()
                    )
                    if math.isfinite(ckl):
                        corr_kl_vals.append(ckl)
                    else:
                        corr_kl_nonfinite += 1

    # Metric groups: oracle_diag/* runs the model and scores it against the
    # analytic ground truth; val/* holds TabICL-marginal scores and model-free
    # reference numbers. val/y_nll_* is only set when a PIT checkpoint is configured.
    metrics: dict = {}

    # Std of per-task copula NLL against the analytic z_test.
    metrics["oracle_diag/copula_nll_std"] = float(np.std(cop_per_task)) if cop_per_task else float("nan")

    # Sigma statistics from the forward on val_loader's own z_train (suffix _analytic_z).
    if all_sigma_off:
        off_arr = np.array(all_sigma_off, dtype=np.float32)
        metrics["sigma_offdiag_mean_analytic_z"] = float(off_arr.mean())
        metrics["sigma_offdiag_std_analytic_z"]  = float(off_arr.std())
        metrics["sigma_offdiag_abs_mean_analytic_z"] = float(np.abs(off_arr).mean())
    else:
        metrics["sigma_offdiag_mean_analytic_z"] = metrics["sigma_offdiag_std_analytic_z"] = metrics["sigma_offdiag_abs_mean_analytic_z"] = 0.0
    metrics["sigma_diag_mean_analytic_z"] = float(np.mean(all_sigma_diag)) if all_sigma_diag else 1.0

    # Model output statistics (same analytic-z_train caveat as above)
    metrics["W_norm_mean_analytic_z"] = float(np.mean(all_W_norms)) if all_W_norms else 0.0
    metrics["s_mean_analytic_z"]      = float(np.mean(all_s_vals))  if all_s_vals  else 0.0

    # Bayes-optimal ceiling (gp_analytical_posterior). If the main loop could not
    # fill the accumulators, use the separately drawn posterior_probe.
    if posterior_probe is not None and val_episodes_meta is None:
        pb = posterior_probe["batch"]
        out_p = model(pb)
        Sigma_p = build_sigma(out_p, cfg, jitter=jitter, test_mask=pb["test_mask"])
        parts_p = y_space_nll(
            Sigma_p, pb["z_test"].float(), pb["log_pdf_test"].float(), pb["test_mask"]
        )
        all_oracle_total.append(parts_p["total"].item())
        all_oracle_copula.append(parts_p["copula"].item())
        for b, ep in enumerate(posterior_probe["episodes"]):
            n = int(ep["x_norm_test"].shape[0])
            if n < 1:
                continue
            try:
                post = gp_analytical_posterior(ep)
            except (KeyError, NotImplementedError):
                continue  # rare unsupported kernel schema — see gp_analytical_posterior's docstring
            nll_post_per_point.append(post["nll_post"] / n)  # raw-sum -> nats/point, matching y_space_nll
            # Sklar split of the ceiling, same normalization.
            nll_post_marginal_per_point.append(post["nll_post_marginal"] / n)
            nll_post_copula_per_point.append(post["nll_post_copula"] / n)
            if n >= 2:
                ri_p, ci_p = np.triu_indices(n, k=1)
                off_p_post.append(Sigma_p[b, :n, :n].float().cpu().numpy()[ri_p, ci_p])
                off_o_post.append(post["R_post"].cpu().numpy()[ri_p, ci_p])

    if all_oracle_total:
        metrics["oracle_diag/copula_nll"] = float(np.mean(all_oracle_copula))
        metrics["oracle_diag/total_nll"] = float(np.mean(all_oracle_total))
        # Marginal = total - copula.
        metrics["oracle_diag/marginal_nll"] = metrics["oracle_diag/total_nll"] - metrics["oracle_diag/copula_nll"]
    if nll_post_per_point:
        # gap_nll = total_nll - y_nll_oracle_posterior, on the same episodes (>= 0 in expectation).
        oracle_posterior_nll = float(np.mean(nll_post_per_point))
        metrics["y_nll_oracle_posterior"] = oracle_posterior_nll
        # Sklar split of y_nll_oracle_posterior.
        metrics["y_nll_oracle_posterior_marginal"] = float(np.mean(nll_post_marginal_per_point))
        metrics["y_nll_oracle_posterior_copula"] = float(np.mean(nll_post_copula_per_point))
        if "oracle_diag/total_nll" in metrics:
            metrics["oracle_diag/gap_nll"] = metrics["oracle_diag/total_nll"] - oracle_posterior_nll
            # copula_gap = model copula NLL - Bayes-optimal copula NLL (equals gap_nll).
            # copula_headroom = -y_nll_oracle_posterior_copula, the total improvement
            # correlation can give over independence; it shrinks quickly with context size.
            metrics["oracle_diag/copula_gap"] = (
                metrics["oracle_diag/copula_nll"] - metrics["y_nll_oracle_posterior_copula"]
            )
            metrics["oracle_diag/copula_headroom"] = -metrics["y_nll_oracle_posterior_copula"]
            # Sanity check: ~0 when both sides use the analytic marginal.
            metrics["oracle_diag/marginal_gap"] = (
                metrics["oracle_diag/total_nll"] - metrics["oracle_diag/copula_nll"]
            ) - metrics["y_nll_oracle_posterior_marginal"]
    if off_p_post:
        cq_p = _corr_quality(np.concatenate(off_p_post), np.concatenate(off_o_post))
        metrics["oracle_diag/corr_pearson"] = cq_p["pearson"]
        metrics["oracle_diag/corr_mae"] = cq_p["mae"]
    if corr_kl_vals:
        # Also log the p90 (corr_kl is heavy-tailed across episodes).
        metrics["oracle_diag/corr_kl"] = float(np.mean(corr_kl_vals))
        metrics["oracle_diag/corr_kl_p90"] = float(np.percentile(corr_kl_vals, 90))
    metrics["oracle_diag/corr_kl_nonfinite"] = float(corr_kl_nonfinite)

    # TabICL-marginal total Y-space NLL: val's headline, set only when a PIT checkpoint is configured.
    if all_tabicl_marginal_total:
        metrics["y_nll_total"] = float(np.mean(all_tabicl_marginal_total))
        # Sklar split: y_nll_marginal is the frozen TabICL marginal's NLL; y_nll_copula is the model's copula NLL on TabICL's z_test.
        metrics["y_nll_marginal"] = float(np.mean(all_tabicl_marginal_marginal))
        metrics["y_nll_copula"] = float(np.mean(all_tabicl_marginal_copula))

    # Per-kernel-family probes. Model-vs-ground-truth metrics go to oracle_diag/;
    # model-free references (marginal_nll, oracle_posterior_total_nll) to val/.
    for family, probe_s in (synth_kernel_batches or {}).items():
        sbatch = probe_s["batch"]
        out_s = model(sbatch)
        Sigma_s = build_sigma(out_s, cfg, jitter=jitter, test_mask=sbatch["test_mask"])
        parts_s = y_space_nll(
            Sigma_s, sbatch["z_test"].float(), sbatch["log_pdf_test"].float(), sbatch["test_mask"]
        )
        cop_s = parts_s["copula"].item()
        mar_s = parts_s["marginal"].item()
        tot_s = parts_s["total"].item()
        metrics[f"oracle_diag/kernel_fit/{family}/copula_nll"] = cop_s
        metrics[f"oracle_diag/kernel_fit/{family}/total_nll"]  = tot_s
        metrics[f"kernel_fit/{family}/marginal_nll"] = mar_s

        # Per-family Bayes-optimal ceiling.
        nll_post_per_point_s: list[float] = []
        for ep in probe_s["episodes"]:
            n_s = int(ep["x_norm_test"].shape[0])
            if n_s < 1:
                continue
            try:
                post_s = gp_analytical_posterior(ep)
            except (KeyError, NotImplementedError):
                continue
            nll_post_per_point_s.append(post_s["nll_post"] / n_s)
        oracle_post_s = None
        if nll_post_per_point_s:
            oracle_post_s = float(np.mean(nll_post_per_point_s))
            metrics[f"kernel_fit/{family}/oracle_posterior_total_nll"] = oracle_post_s
            metrics[f"oracle_diag/kernel_fit/{family}/gap_nll"] = tot_s - oracle_post_s

        # Same probes conditioned on TabICL's z_train and scored under TabICL's
        # marginal (val/, feeds adaptive_kernel_signal="tabicl").
        z_cache_fam = (tabicl_kernel_fit_z or {}).get(family)
        if z_cache_fam is not None:
            n_train_s = sbatch["train_mask"].sum(-1)
            n_test_s = sbatch["test_mask"].sum(-1)
            tot_tabicl_list: list[float] = []
            mar_tabicl_list: list[float] = []
            cop_tabicl_list: list[float] = []
            for b in range(sbatch["x_train"].shape[0]):
                n_tr = int(n_train_s[b])
                n_te = int(n_test_s[b])
                if n_tr < 2 or n_te < 1:
                    continue
                z_train_b = z_cache_fam["z_train"][b, :n_tr].to(device).unsqueeze(0)
                sub_batch = {
                    "x_train": sbatch["x_train"][b : b + 1, :n_tr],
                    "z_train": z_train_b,
                    "x_test": sbatch["x_test"][b : b + 1, :n_te],
                }
                out_tb = model(sub_batch)
                Sigma_tb = build_sigma(out_tb, cfg, jitter=jitter)
                z_test_b = z_cache_fam["z_test"][b, :n_te].to(device).unsqueeze(0)
                log_pdf_b = z_cache_fam["log_pdf_test"][b, :n_te].to(device).unsqueeze(0)
                mask_b = torch.ones(1, n_te, dtype=torch.bool, device=device)
                parts_tb = y_space_nll(Sigma_tb, z_test_b, log_pdf_b, mask_b)
                tot_tabicl_list.append(parts_tb["total"].item())
                mar_tabicl_list.append(parts_tb["marginal"].item())
                cop_tabicl_list.append(parts_tb["copula"].item())
            if tot_tabicl_list:
                tot_s_tabicl = float(np.mean(tot_tabicl_list))
                metrics[f"kernel_fit/{family}/total_nll_tabicl"] = tot_s_tabicl
                metrics[f"kernel_fit/{family}/marginal_nll_tabicl"] = float(np.mean(mar_tabicl_list))
                metrics[f"kernel_fit/{family}/copula_nll_tabicl"] = float(np.mean(cop_tabicl_list))
                if oracle_post_s is not None:
                    metrics[f"kernel_fit/{family}/gap_nll_tabicl"] = tot_s_tabicl - oracle_post_s

    # Real ERA5 probe per region: the model's Y-space NLL on each region's held-out points.
    region_y_nll_total: list[float] = []
    region_y_nll_marginal: list[float] = []
    region_y_nll_copula: list[float] = []
    region_gp_baseline_total: dict[str, list[float]] = {}
    for region, probe in (era5_val_batches or {}).items():
        out_e = model(probe["batch"])
        Sigma_e = build_sigma(out_e, cfg, jitter=jitter, test_mask=probe["batch"]["test_mask"])

        # Held-out-point NLL from the same forward, under TabICL's precomputed PIT (when configured).
        if "nll_test_z" in probe:
            idx = torch.as_tensor(probe["nll_test_idx"], dtype=torch.long, device=Sigma_e.device)
            Sigma_nll = Sigma_e.index_select(1, idx).index_select(2, idx)
            z_nll, log_pdf_nll = probe["nll_test_z"], probe["nll_test_log_pdf"]
            mask_nll = torch.ones_like(z_nll, dtype=torch.bool)
            parts_e = y_space_nll(Sigma_nll, z_nll, log_pdf_nll, mask_nll)
            y_nll_total = parts_e["total"].item()
            y_nll_marginal = parts_e["marginal"].item()
            y_nll_copula = parts_e["copula"].item()
            metrics[f"era5_fit/{region}/y_nll_total"] = y_nll_total
            metrics[f"era5_fit/{region}/y_nll_marginal"] = y_nll_marginal
            metrics[f"era5_fit/{region}/y_nll_copula"] = y_nll_copula
            if not math.isnan(y_nll_total):
                region_y_nll_total.append(y_nll_total)
                region_y_nll_marginal.append(y_nll_marginal)
                region_y_nll_copula.append(y_nll_copula)

        # Classical GP-MLE baseline NLL, fitted once per region at startup.
        if "gp_baseline_nll" in probe:
            for kname, parts in probe["gp_baseline_nll"].items():
                metrics[f"era5_fit/{region}/gp_baseline_{kname}_nll_total"] = parts["total"]
                metrics[f"era5_fit/{region}/gp_baseline_{kname}_nll_marginal"] = parts["marginal"]
                metrics[f"era5_fit/{region}/gp_baseline_{kname}_nll_copula"] = parts["copula"]
                if not math.isnan(parts["total"]):
                    region_gp_baseline_total.setdefault(kname, []).append(parts["total"])

    metrics["era5_fit/mean_y_nll_total"] = _macro_average(region_y_nll_total)
    metrics["era5_fit/mean_y_nll_marginal"] = _macro_average(region_y_nll_marginal)
    metrics["era5_fit/mean_y_nll_copula"] = _macro_average(region_y_nll_copula)
    for kname, vals in region_gp_baseline_total.items():
        metrics[f"era5_fit/mean_gp_baseline_{kname}_nll_total"] = _macro_average(vals)

    model.train()

    plot_figs: dict = {}
    if do_plot and era5_viz_batch is not None:
        # ERA5 predicted vs true temperature field from sparse context (_build_era5_viz_batch).
        fig_era5, fig_era5_resid = _era5_viz_fig(model, cfg, era5_viz_batch, jitter, device)
        if fig_era5 is not None:
            plot_figs["val/era5_predictions"] = fig_era5
        # Same field with each predictor's own mean removed.
        if fig_era5_resid is not None:
            plot_figs["val/era5_residuals"] = fig_era5_resid
        # Three samples of the latent z per predictor (independent / copula / GP).
        fig_era5_z = _era5_z_samples_fig(model, cfg, era5_viz_batch, jitter, device)
        if fig_era5_z is not None:
            plot_figs["val/era5_predictions_z"] = fig_era5_z
        # Predictive variance of the TabICL marginal vs the GP baseline against distance to context.
        fig_era5_var = _era5_marginal_variance_fig(era5_viz_batch)
        if fig_era5_var is not None:
            plot_figs["val/era5_marginal_variance"] = fig_era5_var

    return metrics, plot_figs
