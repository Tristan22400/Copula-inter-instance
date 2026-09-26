"""validate(): the copula training loop's validation pass and its logged metrics and figures."""

from __future__ import annotations

import math
import os

from copula_inter.era5_probes import _era5_marginal_variance_fig, _era5_viz_fig, _era5_z_samples_fig
from copula_inter.probe_batches import _corr_quality, _macro_average

# P/N (hence attention sequence length T=P+N) are sampled per-shard from a wide
# range (see conf/data/gp_tasks.yaml P_min/P_max, N_min/N_max), so batches vary
# a lot in size while batch_size stays fixed — some shards get much closer to
# the VRAM ceiling than others. When that happens, PyTorch's caching allocator
# can fail a small allocation despite reserved-but-unallocated memory being
# nominally sufficient, because it's fragmented into pieces too small to
# satisfy the request (see the OOM message's "reserved but unallocated"
# figure). expandable_segments avoids this by growing/shrinking allocations
# in-place instead of requiring a fresh contiguous chunk. Must be set before
# the CUDA caching allocator initializes (i.e. before any CUDA call), so this
# goes at the top of the file, before `import torch`. setdefault so an
# explicit environment override still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

# eval/ (regions.py, spatial-correlation probe helpers -- see
# _build_era5_val_batches below) lives at the repo root, not under src/.
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
) -> tuple[dict, list]:
    # Do NOT call model.eval() here: TabICL's eval mode triggers _inference_forward
    # which uses InferenceManager with its own float16 autocast on CUDA, producing
    # NaN for certain inputs. There is no dropout in this model so eval mode has no
    # benefit. Use torch.no_grad() for efficiency instead.
    jitter = float(cfg.model.get("sigma_jitter", 1e-4))

    cop_per_task: list[float] = []
    all_W_norms: list[float] = []
    all_s_vals: list[float] = []
    all_sigma_off: list[float] = []
    all_sigma_diag: list[float] = []
    all_tabicl_marginal_total: list[float] = []
    all_tabicl_marginal_marginal: list[float] = []
    all_tabicl_marginal_copula: list[float] = []

    # ---- True Bayes-optimal ceiling accumulators (pit.gp_analytical_posterior) ----
    # Filled either inline below (val_episodes_meta present -- live-generation
    # val_loader, which now carries kernel metadata) or, if that's absent, by
    # the posterior_probe fallback pass after the main loop (disk-mode /
    # real-ERA5 live_source). Same accumulators either way, so the metrics
    # block after the main loop needs only one code path regardless of which
    # source filled them -- see this function's oracle_diag/gap_nll comment
    # further down.
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

        # ---- Put every oracle_diag/* quantity in the ORACLE z-space --------
        # Everything below this point in the loop (all_oracle_*, cop_per_task,
        # the sigma_*_analytic_z / W_norm / s statistics, off_p_post, and the
        # corr_kl accumulation) is scored against, or compared to, the exact
        # analytic GP -- gp_analytical_posterior's R_post and its
        # nll_post/nll_post_copula ceiling. Under data.z_train_source="tabicl"
        # the batch itself carries TABICL's PIT instead (data_gen's tabicl
        # branch overwrites z_train, z_test AND log_pdf_test -- see
        # _build_analytic_val_z's docstring), so scoring those against the
        # GP-posterior ceiling would be a cross-z-space comparison: the two
        # Sklar splits are taken at different marginals, so their copula terms
        # live on different, non-additive scales and their difference is not a
        # gap. `analytic_val_z` supplies the missing operand.
        #
        # The model is re-conditioned on the analytic z_train as well, not just
        # re-scored on the analytic z_test: oracle_diag/ answers "how far is
        # this model from Bayes-optimal when handed the ORACLE marginal", which
        # needs oracle z on both sides. One extra forward pass per val batch
        # (not per episode); `out`/`Sigma` are rebound so every consumer below
        # picks it up. val/y_nll_* -- the real-deployment TabICL headline --
        # is built further down from tabicl_val_z and is untouched by this.
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

        # ---- Oracle-posterior batch-level total/copula NLL (vectorized) ----
        # Same y_space_nll(Sigma, z_test, log_pdf_test, test_mask) call the old
        # separate posterior_probe pass used, just run here on val_loader's own
        # Sigma/z_test instead -- one call per val batch, appended and averaged
        # across batches below, rather than the old single call over a whole
        # separately-drawn probe. Only when val_episodes_meta is available
        # (live-generation val_loader); otherwise the posterior_probe fallback
        # after the main loop fills the same accumulators.
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

            # Per-task copula NLL against batch["z_test"] (the exact
            # analytic-GP PIT this val set was generated with) -> feeds
            # oracle_diag/copula_nll_std below: this DOES test the trained
            # model (Sigma is the model's own output) against ground truth,
            # so it belongs in oracle_diag/, not among the Sigma-only stats
            # below (which don't reference z_test at all).
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

            # W row-norms and s means (masked mean over valid test instances).
            # "s" is absent for "tanhnorm" (see model.py's _NO_SCALAR_COLUMN /
            # build_sigma's out.get("s") pattern) — skip the s-diagnostic for
            # that parametrization instead of KeyError-ing.
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

        # ---- TabICL-marginal real (non-oracle) NLL scoring ----
        # Runs on EVERY val_loader batch every val_every step: this is what
        # feeds val/y_nll_total below, which is meant to be sized like the
        # rest of val/'s metrics (training.val_episodes), not a small
        # plot-sized sub-sample — tabicl_val_z now has an entry for every
        # batch (see _build_tabicl_val_z).
        B = Sigma.shape[0]
        z_cache_b = tabicl_val_z.get(batch_idx) if tabicl_val_z else None
        for b in range(B):
            n = int(batch["test_mask"][b].sum())
            if n < 2:
                continue

            # Re-run the model on this SAME episode (same x_train/
            # x_test) conditioned on TabICL's own K-fold PIT z_train,
            # precomputed once by _build_tabicl_val_z. Under
            # data.z_train_source="analytic" this is the sim-to-real
            # substitution (oracle GP-LOO z_train -> real TabICL z_train);
            # under "tabicl" the batch already carries a TabICL PIT, so it
            # is instead a second estimate at the same fold count -- see
            # _build_tabicl_val_z's docstring. Either way it is what
            # val/y_nll_* is built from, and the oracle-side numbers
            # (oracle_diag/*) come from `Sigma`/`z_test_an` above, which
            # are always in the exact-GP z-space.
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

                    # Genuine (non-oracle) total Y-space NLL: score this
                    # same TabICL-conditioned Sigma against TabICL's OWN
                    # marginal at the test points
                    # (z_cache_b["z_test"]/["log_pdf_test"], also from
                    # _build_tabicl_val_z), not the oracle's — the number
                    # that actually answers "is this checkpoint correct
                    # once a real (imperfect) marginal replaces the
                    # oracle one," which no z-space-only copula NLL can
                    # (see eval_checkpoint.py's _print_total_nll_table
                    # for why: two different marginals' z-transforms put
                    # z-space copula NLL on different, non-additive
                    # scales — only a same-basis Y-space total is
                    # comparable). This is what val/y_nll_total below is
                    # built from.
                    z_test_tabicl_b = z_cache_b["z_test"][b, :n].to(device).unsqueeze(0)
                    log_pdf_tabicl_b = z_cache_b["log_pdf_test"][b, :n].to(device).unsqueeze(0)
                    mask_tabicl_b = torch.ones(1, n, dtype=torch.bool, device=device)
                    parts_tabicl_b = y_space_nll(
                        Sigma_tabicl, z_test_tabicl_b, log_pdf_tabicl_b, mask_tabicl_b
                    )
                    all_tabicl_marginal_total.append(parts_tabicl_b["total"].item())
                    all_tabicl_marginal_marginal.append(parts_tabicl_b["marginal"].item())
                    all_tabicl_marginal_copula.append(parts_tabicl_b["copula"].item())

            # True Bayes-optimal ceiling (pit.gp_analytical_posterior), one
            # episode at a time (float64 eigendecomposition-based PSD repair
            # -- no batched implementation) using THIS val episode's own
            # kernel metadata (val_episodes_meta[batch_idx][b]) instead of a
            # separately-drawn probe. Independent of z_cache_b/do_plot above.
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
                    # Noise-free convergence signal: gap_nll is a one-draw
                    # Monte-Carlo estimate (its per-episode value is often
                    # negative), this is a functional of the two matrices
                    # alone. See pit.gaussian_corr_kl.
                    ckl = gaussian_corr_kl(
                        Sigma[b, :n, :n].cpu(), post["R_post"].cpu()
                    )
                    if math.isfinite(ckl):
                        corr_kl_vals.append(ckl)
                    else:
                        corr_kl_nonfinite += 1

    # metrics starts empty. The old y_nll_total/y_nll_copula here scored the
    # model against batch["z_test"]/["log_pdf_test"] — the exact analytic-GP
    # PIT this val set was generated with, i.e. a ground-truth marginal no
    # real deployment ever provides — so they moved to oracle_diag/ below (a
    # sibling of val/, not nested under it — see the wandb.log prefixing in
    # the training loop): oracle_diag/ holds every diagnostic that runs the
    # trained model and scores/compares its output against this ground-truth
    # z_test/z_train, nothing else. Pure reference numbers that don't
    # exercise the model at all (e.g. y_nll_oracle_posterior, kernel_fit's
    # marginal_nll/oracle_posterior_total_nll — gp_analytical_posterior's
    # ceiling and data_gen.py's oracle marginal are both independent of the
    # model) stay in val/ instead, even though they're also ground-truth-
    # scored, since they aren't testing the model. val/'s own headline NLL
    # numbers (y_nll_total/y_nll_marginal/y_nll_copula, set below from
    # all_tabicl_marginal_*) are scored against TabICL's own frozen PIT
    # instead — a real, imperfect marginal, the same kind deployment would
    # actually supply — so they only populate when a PIT checkpoint is
    # configured (tabicl_val_z non-empty; see resolve_pit_ckpt in train()).
    metrics: dict = {}

    # Per-task copula NLL std (against ground truth z_test) — high value
    # means unstable or heterogeneous tasks. Tests the trained model, so
    # lives in oracle_diag/, not val/.
    metrics["oracle_diag/copula_nll_std"] = float(np.std(cop_per_task)) if cop_per_task else float("nan")

    # Sigma statistics — offdiag_mean ≈ 0 means model outputs near-identity.
    # "_analytic_z" suffix: these come from the single model(batch) forward
    # above, which is conditioned on val_loader's own z_train — the exact
    # analytic GP-LOO PIT (data.z_train_source="analytic" by default), NOT
    # TabICL's K-fold PIT. Unlike y_nll_total/kernel_fit's *_tabicl metrics
    # below, there is no TabICL-conditioned counterpart for these, so the
    # suffix exists purely to stop them from being mistaken for one.
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

    # ---- True Bayes-optimal ceiling (pit.gp_analytical_posterior) --------
    # Replaces the old full-val-set oracle_gap/copula_gap/copula_improvement
    # (deleted above along with y_nll_oracle*): those were scored against
    # data_gen.py's oracle_mode="prior" R_star/Sigma_star, which is
    # context-blind by construction (never conditions on x_train/y_train —
    # see data_gen.py:3359-3382) and therefore NOT the Bayes-optimal lower
    # bound achievable given the context the model actually receives, only
    # a weaker, beatable one — a model that legitimately exploits context
    # could (and should) beat it, which the old "copula_improvement"
    # (0=identity, 1=oracle) had no way to express as anything but a
    # confusing ">1". gp_analytical_posterior computes the real Schur-
    # complement posterior instead, so oracle_gap_posterior >= 0 in
    # expectation is a genuine inequality (see its docstring), not a
    # convention that can be beaten by a better model.
    #
    # all_oracle_total/all_oracle_copula/nll_post_per_point/etc. were already
    # filled inline in the main loop above when val_episodes_meta was
    # available (today's live-generation default: val_loader's own episodes,
    # same Sigma the rest of this function scores). When it isn't (disk-mode
    # CopulaDataset, or real-ERA5 live_source — neither carries kernel
    # metadata), fall back to the separately-drawn posterior_probe here
    # instead, filling the exact same accumulators via one extra forward
    # pass, so the metrics block below needs only one path regardless of
    # which source supplied it.
    #
    # copula_nll/total_nll/gap_nll/corr_pearson/corr_mae below all run the
    # model and score its output against ground truth, so they're grouped
    # under the "oracle_diag/" key prefix (see this function's return + the
    # training loop's wandb.log call), a sibling of val/. y_nll_oracle_posterior
    # does NOT run the model at all — it's gp_analytical_posterior's ceiling,
    # a fixed property of the episodes alone — so it stays in val/ instead,
    # right beside gap_nll's other operand (oracle_diag/total_nll) for easy
    # side-by-side reading.
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
            # Sklar split of the same raw sum (nll_post = nll_post_marginal + nll_post_copula,
            # see gp_analytical_posterior's docstring) -- same /n normalization as the total above.
            nll_post_marginal_per_point.append(post["nll_post_marginal"] / n)
            nll_post_copula_per_point.append(post["nll_post_copula"] / n)
            if n >= 2:
                ri_p, ci_p = np.triu_indices(n, k=1)
                off_p_post.append(Sigma_p[b, :n, :n].float().cpu().numpy()[ri_p, ci_p])
                off_o_post.append(post["R_post"].cpu().numpy()[ri_p, ci_p])

    if all_oracle_total:
        metrics["oracle_diag/copula_nll"] = float(np.mean(all_oracle_copula))
        metrics["oracle_diag/total_nll"] = float(np.mean(all_oracle_total))
        # y_space_nll's "total" is copula+marginal exactly (loss.py:729-730),
        # and that identity survives averaging by linearity, so this is a
        # subtraction of two quantities already computed above, not new
        # compute — no separate all_oracle_marginal accumulator needed.
        metrics["oracle_diag/marginal_nll"] = metrics["oracle_diag/total_nll"] - metrics["oracle_diag/copula_nll"]
    if nll_post_per_point:
        # total_nll and y_nll_oracle_posterior are scored on the SAME
        # episode population (whichever source supplied it above) — gap_nll,
        # their difference, is therefore a valid same-population comparison:
        # >= 0 in expectation, and this pair can't drift apart the way two
        # different-population NLLs could.
        oracle_posterior_nll = float(np.mean(nll_post_per_point))
        metrics["y_nll_oracle_posterior"] = oracle_posterior_nll
        # Sklar split of y_nll_oracle_posterior, for side-by-side reading against
        # oracle_diag/copula_nll and oracle_diag/total_nll's own marginal component.
        metrics["y_nll_oracle_posterior_marginal"] = float(np.mean(nll_post_marginal_per_point))
        metrics["y_nll_oracle_posterior_copula"] = float(np.mean(nll_post_copula_per_point))
        if "oracle_diag/total_nll" in metrics:
            metrics["oracle_diag/gap_nll"] = metrics["oracle_diag/total_nll"] - oracle_posterior_nll
            # ---- The named limits ---------------------------------------
            # Both operands are now in the same (GP-posterior) z-space, so
            # the marginal terms cancel exactly and copula_gap == gap_nll to
            # float precision (asserted in tests/test_oracle_diag_analytic_z.py).
            # copula_gap is kept as the primary name because it says what the
            # quantity IS: the model's copula NLL minus the Bayes-optimal one.
            #
            # copula_headroom is the entire Bayes-optimal copula reward
            # (-y_nll_oracle_posterior_copula, i.e. how many nats/point the
            # perfect correlation buys over predicting independence). A
            # copula_gap ABOVE it means the model is worse than the identity.
            # It is logged beside the gap always, because a gap quoted alone
            # is unreadable: the headroom collapses steeply with context size,
            # since each added context point is one the posterior has already
            # explained away. Measured on RBF episodes at N_test=256, 24
            # episodes/row, l in [0.5, 1.5], noise in [0.05, 0.2]:
            # P=4 -> 0.836, P=8 -> 0.482, P=16 -> 0.240, P=32 -> 0.099,
            # P=64 -> 0.058, P=128 -> 0.024 nats/pt. The exact numbers move
            # with the generator config (lengthscale and noise ranges above
            # all), which is why this is logged per run rather than assumed --
            # but the ordering does not, so "0.05 nats off" is mediocre at
            # P=4 and hopeless at P=64.
            metrics["oracle_diag/copula_gap"] = (
                metrics["oracle_diag/copula_nll"] - metrics["y_nll_oracle_posterior_copula"]
            )
            metrics["oracle_diag/copula_headroom"] = -metrics["y_nll_oracle_posterior_copula"]
            # Sanity only: with both sides on the analytic marginal this is
            # ~0 by construction. A non-zero value means the oracle_diag
            # z-space routing above has broken -- e.g. the model is being
            # scored against TabICL's log_pdf_test again.
            metrics["oracle_diag/marginal_gap"] = (
                metrics["oracle_diag/total_nll"] - metrics["oracle_diag/copula_nll"]
            ) - metrics["y_nll_oracle_posterior_marginal"]
    if off_p_post:
        cq_p = _corr_quality(np.concatenate(off_p_post), np.concatenate(off_o_post))
        metrics["oracle_diag/corr_pearson"] = cq_p["pearson"]
        metrics["oracle_diag/corr_mae"] = cq_p["mae"]
    if corr_kl_vals:
        # p90 alongside the mean: corr_kl is heavy-tailed across episodes
        # (a handful of near-degenerate R_post dominate), so a falling mean
        # with a flat p90 is a real and different story from both falling.
        metrics["oracle_diag/corr_kl"] = float(np.mean(corr_kl_vals))
        metrics["oracle_diag/corr_kl_p90"] = float(np.percentile(corr_kl_vals, 90))
    metrics["oracle_diag/corr_kl_nonfinite"] = float(corr_kl_nonfinite)

    # Genuine (non-oracle) total Y-space NLL under TabICL's own frozen
    # marginal — see the loop above (all_tabicl_marginal_total), populated
    # every val_every step (not gated on do_plot; only the plot-only pieces
    # collected alongside it are). This is val/'s real headline NLL: unlike
    # the deleted ground-truth-z_test y_nll_total, it needs no reference
    # matrix and scores against a real, imperfect (TabICL) marginal — the
    # "does this checkpoint actually work once you plug in a real marginal
    # at deployment" number. Only populated when a PIT checkpoint is
    # configured (resolve_pit_ckpt(cfg) resolves -> tabicl_val_z non-empty);
    # otherwise val/ has no total-NLL headline, which is the honest outcome
    # rather than falling back to a ground-truth-scored substitute.
    if all_tabicl_marginal_total:
        metrics["y_nll_total"] = float(np.mean(all_tabicl_marginal_total))
        # Sklar split of the same total: y_nll_marginal is TabICL's own
        # frozen marginal NLL (moves only if the PIT checkpoint or these
        # episodes change, not with this run's training) while y_nll_copula
        # is the model's copula NLL evaluated against TabICL's z_test
        # instead of the oracle's — the piece that actually reflects whether
        # the model's Sigma is still well-calibrated once conditioned on a
        # real (imperfect) marginal.
        metrics["y_nll_marginal"] = float(np.mean(all_tabicl_marginal_marginal))
        metrics["y_nll_copula"] = float(np.mean(all_tabicl_marginal_copula))

    # Model-fit-to-classical-kernel metrics: runs the CURRENT model on a fixed
    # synthetic probe set per kernel family (see _build_synthetic_kernel_batches),
    # so these move with training progress (unlike a fixed data-only baseline).
    # copula_nll/total_nll run the model and score it against ground truth ->
    # oracle_diag/. marginal_nll (data_gen.py's oracle marginal) and
    # oracle_posterior_total_nll (gp_analytical_posterior's ceiling) don't
    # involve the model at all -> val/, same split as the top-level
    # posterior_probe block above.
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

        # True Bayes-optimal ceiling for this family, same construction as
        # the top-level y_nll_oracle_posterior but restricted to this
        # family's own probe episodes (needs return_kernel_metadata=True —
        # see _build_synthetic_kernel_batches).
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

        # Real (non-oracle) counterpart of the block above: re-run the model
        # on this SAME family's probe episodes but conditioned on TabICL's
        # own K-fold PIT z_train (_build_tabicl_kernel_fit_z, precomputed
        # once at startup) instead of the exact analytic one, and score
        # against TabICL's own z_test/log_pdf_test — the same substitution
        # the top-level y_nll_total/all_tabicl_marginal_* block above makes
        # for the general val set. Doesn't touch ground truth (TabICL's PIT
        # is a real, imperfect marginal, not the oracle), so -> val/, not
        # oracle_diag/, same reasoning as y_nll_total. Feeds
        # training.adaptive_kernel_signal="tabicl" (see
        # _update_adaptive_kernel_weights) — a curriculum signal driven by
        # how the model performs under a real deployment-like marginal
        # instead of the idealized analytic one.
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

    # Real-ERA5 spatial-correlation probe per region (see
    # _build_era5_val_batches / eval/spatial/sweep_core.py::build_era5_probe).
    # There is no GP oracle for real data, so — unlike kernel_fit/<family>'s
    # NLL gap against Sigma_star — this scores the CURRENT model's
    # real, non-oracle Y-space NLL against the region's frozen held-out
    # points (see below); the shape_corr/rmse/bias/model_r2 curve-shape
    # comparison against rho_emp eval/runners/spatial_correlation_eval.py's
    # real-mode sweep reports was dropped from the training-time loop —
    # too noisy at the loop's small per-region day count to track.
    region_y_nll_total: list[float] = []
    region_y_nll_marginal: list[float] = []
    region_y_nll_copula: list[float] = []
    region_gp_baseline_total: dict[str, list[float]] = {}
    for region, probe in (era5_val_batches or {}).items():
        out_e = model(probe["batch"])
        Sigma_e = build_sigma(out_e, cfg, jitter=jitter, test_mask=probe["batch"]["test_mask"])

        # Real, non-oracle Y-space NLL on this region's held-out
        # (never-in-context) points — the era5_fit analogue of
        # kernel_fit/<family>'s *_tabicl block, minus the gap (no GP oracle
        # for real data to gap against; see build_era5_probe's docstring).
        # Reuses the SAME Sigma_e forward pass above (it already covers the
        # full D-point grid, which nll_test_idx indexes into), just scored
        # against TabICL's own frozen PIT (nll_test_z/nll_test_log_pdf,
        # precomputed once in _build_era5_val_batches) — only present when a
        # PIT checkpoint was configured for the probe.
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

        # Classical-GP-MLE baseline, frozen once per region in
        # _build_era5_val_batches (see its docstring for why it's not
        # refit here) — logged alongside y_nll_total/marginal/copula above
        # for a live "is the model beating a classical spatial GP on this
        # region" comparison during training.
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
        # Real-ERA5 predicted-vs-ground-truth temperature field, sparse
        # context (< baselines.era5_viz_context_frac, default 5%, of the
        # grid — see _build_era5_viz_batch) on a handful of frozen days.
        # Replaces the old val/corr_density_analytic_z (hexbin of predicted
        # vs. oracle off-diagonal correlation) and val/corr_grid (oracle vs.
        # predicted correlation-matrix heatmaps): both compared the model's
        # Sigma directly against the GP's exact R_star, which this repo's
        # feedback_no_raw_correlation_vs_oracle_comparison note rules out as
        # a diagnostic once TabICL's PIT is in the loop (Sigma lives in
        # TabICL's own approximate z-space, not the GP's exact one) — an
        # ERA5 field reconstruction has no such mismatch: it's judged (by
        # eye) in the same real Y-space the model is actually deployed in.
        fig_era5, fig_era5_resid = _era5_viz_fig(model, cfg, era5_viz_batch, jitter, device)
        if fig_era5 is not None:
            plot_figs["val/era5_predictions"] = fig_era5
        # Companion figure, same probe/day/forward-pass/z_shared draws as
        # val/era5_predictions: each row's OWN per-location predictive mean
        # (frozen TabICL marginal, or the fitted GP's own posterior mean for
        # the GP row) subtracted off, isolating the cross-location
        # correlation structure from the smooth mean field that otherwise
        # dominates the raw-temperature panel (see _era5_viz_fig).
        if fig_era5_resid is not None:
            plot_figs["val/era5_residuals"] = fig_era5_resid
        # Companion figure, same probe, same fixed first day: 3 posterior
        # SAMPLES of the copula LATENT z itself (no marginal) so the three
        # predictors' correlation structures — independent / copula model /
        # GP baseline — are compared directly, isolated from any marginal
        # differences (see _era5_z_samples_fig).
        fig_era5_z = _era5_z_samples_fig(model, cfg, era5_viz_batch, jitter, device)
        if fig_era5_z is not None:
            plot_figs["val/era5_predictions_z"] = fig_era5_z
        # Companion figure, same probe, all days: per-location predictive
        # VARIANCE of the frozen TabICL marginal vs. the fitted-GP baseline
        # — no model forward pass, unaffected by training — to check
        # whether either predictor's uncertainty actually grows with
        # distance from the sparse context (see _era5_marginal_variance_fig).
        fig_era5_var = _era5_marginal_variance_fig(era5_viz_batch)
        if fig_era5_var is not None:
            plot_figs["val/era5_marginal_variance"] = fig_era5_var

    return metrics, plot_figs
