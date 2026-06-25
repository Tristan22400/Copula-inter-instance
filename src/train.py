"""
train.py — Train the Copula Transformer in Y-space NLL via Sklar's theorem.

Loss:  L = Copula_NLL(z_test; Σ̂) + Marginal_NLL(y_test; TabICL log-pdf)
Σ̂ is built by ``low_rank_correlation(W, s)`` from the model output.

Usage:
    python src/train.py
    python src/train.py training.steps=500 training.dataset_dir=./data/debug_latent
    WANDB_MODE=disabled python src/train.py training.steps=200
"""

from __future__ import annotations

import math
import os
import sys
from glob import glob
from itertools import cycle

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import wandb

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataset import CopulaDataset, collate_fn
from loss import _safe_cholesky, gp_oracle_y_nll, oracle_copula_nll, y_space_nll
from model import build_copula_transformer, low_rank_correlation

_MAX_PLOT_EPISODES = 8
_PLOT_COLLECT_BATCHES = 5


def _sigma_stats(Sigma: torch.Tensor, mask: torch.Tensor) -> dict:
    """Cheap off-diagonal and diagonal statistics over a batch of correlation matrices.

    Key diagnostic: if offdiag_mean ≈ 0, the model is outputting near-identity
    matrices and has not learned any inter-instance correlation structure.

    Args:
        Sigma : (B, N_max, N_max) float32 — predicted correlation matrices
        mask  : (B, N_max) bool           — True for valid (non-padded) instances

    Returns dict with float scalars: offdiag_mean, offdiag_std, diag_mean
    """
    B, N, _ = Sigma.shape
    ri, ci = torch.triu_indices(N, N, offset=1, device=Sigma.device)
    mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)  # (B, N, N)
    valid_off = mask_2d[:, ri, ci]                     # (B, n_pairs)
    off_vals = Sigma[:, ri, ci][valid_off]             # flat valid off-diagonal entries
    diag_vals = Sigma.diagonal(dim1=-2, dim2=-1)[mask] # flat valid diagonal entries
    if off_vals.numel() == 0:
        return {"offdiag_mean": 0.0, "offdiag_std": 0.0, "diag_mean": 1.0}
    return {
        "offdiag_mean": off_vals.mean().item(),
        "offdiag_std":  off_vals.std().item(),
        "diag_mean":    diag_vals.mean().item(),
    }


def _corr_quality(off_pred: np.ndarray, off_ora: np.ndarray) -> dict:
    """MSE, MAE, Pearson r, and signed bias between predicted and oracle off-diagonal values.

    Args:
        off_pred : 1-D float array — predicted off-diagonal correlations
        off_ora  : 1-D float array — oracle off-diagonal correlations (same length)

    Returns dict with float scalars: mse, mae, pearson, bias
    """
    diff = off_pred - off_ora
    mse  = float(np.mean(diff ** 2))
    mae  = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    std_p, std_o = off_pred.std(), off_ora.std()
    pearson = float(np.corrcoef(off_pred, off_ora)[0, 1]) if (std_p > 1e-12 and std_o > 1e-12) else 0.0
    return {"mse": mse, "mae": mae, "pearson": pearson, "bias": bias}


def _corr_grid_fig(plot_episodes: list[dict], step: int) -> plt.Figure:
    """2-row × n_ep subplot grid showing oracle (top) vs predicted (bottom) correlation matrices."""
    n_ep = len(plot_episodes)
    fig, axes = plt.subplots(2, n_ep, figsize=(max(n_ep * 1.8, 4), 4),
                             squeeze=False)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="lightgrey")

    for col, ep in enumerate(plot_episodes):
        R_pred = ep["R_pred"].copy()
        R_ora = ep["R_ora"].copy()
        n = R_pred.shape[0]
        diag = np.arange(n)

        # Compute per-episode upper-triangle MSE
        ri, ci = np.triu_indices(n, k=1)
        mse = float(np.mean((R_pred[ri, ci] - R_ora[ri, ci]) ** 2))

        # Blank diagonal so it doesn't dominate the colour scale
        R_pred[diag, diag] = np.nan
        R_ora[diag, diag] = np.nan

        for row, (mat, row_label) in enumerate([(R_ora, "Oracle"), (R_pred, "Pred")]):
            ax = axes[row, col]
            im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1,
                           interpolation="nearest", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_label, fontsize=7)
            if row == 0:
                ax.set_title(ep["label"], fontsize=6)
            if row == 1:
                ax.set_xlabel(f"MSE={mse:.3f}", fontsize=6)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02)
    fig.suptitle(f"step {step} — oracle vs predicted correlations", fontsize=8)
    fig.tight_layout()
    return fig


def cosine_lr_lambda(step: int, warmup: int, total: int, lr_min_frac: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min_frac + (1.0 - lr_min_frac) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    val_files: list,
    cfg: DictConfig,
    device: str,
    step: int = 0,
    do_plot: bool = False,
) -> tuple[dict, list]:
    # Do NOT call model.eval() here: TabICL's eval mode triggers _inference_forward
    # which uses InferenceManager with its own float16 autocast on CUDA, producing
    # NaN for certain inputs. There is no dropout in this model so eval mode has no
    # benefit. Use torch.no_grad() for efficiency instead.
    val_loader = DataLoader(
        CopulaDataset(file_list=val_files),
        batch_size=cfg.training.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    jitter = float(cfg.model.get("sigma_jitter", 1e-4))

    tot, cop, mar, ora, ora_cop, ora_mar, ora_cop_z = [], [], [], [], [], [], []
    cop_per_task: list[float] = []
    all_W_norms: list[float] = []
    all_s_vals: list[float] = []
    all_sigma_off: list[float] = []
    all_sigma_diag: list[float] = []
    all_off_pred_flat: list[np.ndarray] = []
    all_off_ora_flat: list[np.ndarray] = []
    all_off_pred: list[np.ndarray] = []
    all_off_ora: list[np.ndarray] = []
    plot_episodes: list[dict] = []

    for batch_idx, batch in enumerate(val_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch)
        Sigma = low_rank_correlation(
            out["W"].float(), out["s"].float(), batch["test_mask"], jitter=jitter
        )
        parts = y_space_nll(
            Sigma,
            batch["z_test"].float(),
            batch["log_pdf_test"].float(),
            batch["test_mask"],
        )
        oracle_parts = gp_oracle_y_nll(
            batch["Sigma_star"].float(),
            batch["mu_star"].float(),
            batch["y_test"].float(),
            batch["test_mask"],
        )
        ora_cop_z_val = oracle_copula_nll(
            batch["R_star"].float(),
            batch["z_test"].float(),
            batch["test_mask"],
        )
        tot.append(parts["total"].item())
        cop.append(parts["copula"].item())
        mar.append(parts["marginal"].item())
        ora.append(oracle_parts["total"].item())
        ora_cop.append(oracle_parts["copula"].item())
        ora_mar.append(oracle_parts["marginal"].item())
        ora_cop_z.append(ora_cop_z_val.item())

        # ---- Per-task diagnostics (vectorized — no Python loop over batch) ----
        n_test_cur = batch["test_mask"].sum(-1).float()   # (B,)
        valid_cur = n_test_cur >= 2

        if valid_cur.any():
            N_cur = Sigma.shape[1]
            mask_2d_cur = batch["test_mask"].unsqueeze(-1) & batch["test_mask"].unsqueeze(-2)
            eye_cur = torch.eye(N_cur, device=Sigma.device, dtype=Sigma.dtype).unsqueeze(0)
            S_safe_cur = torch.where(mask_2d_cur, Sigma, eye_cur)
            L_cur, info_cur = torch.linalg.cholesky_ex(S_safe_cur)
            if info_cur.any():
                S_safe_cur = S_safe_cur + 1e-4 * eye_cur
                L_cur = torch.linalg.cholesky(S_safe_cur)

            log_det_cur = 2.0 * L_cur.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).log().sum(-1)
            z_f = batch["z_test"].float()
            tmp_cur = torch.linalg.solve_triangular(L_cur, z_f.unsqueeze(-1), upper=False)
            S_inv_z_cur = torch.linalg.solve_triangular(L_cur.mT, tmp_cur, upper=True).squeeze(-1)
            n_safe_cur = n_test_cur.clamp(min=1)
            cop_cur = 0.5 * (log_det_cur + (z_f * S_inv_z_cur).sum(-1) - (z_f ** 2).sum(-1)) / n_safe_cur
            cop_per_task.extend(cop_cur[valid_cur].cpu().tolist())

            # W row-norms and s means (masked mean over valid test instances)
            W_f = out["W"].float()
            s_f = out["s"].float()
            mask_f = batch["test_mask"].float()
            W_norm_cur = (W_f.norm(dim=-1) * mask_f).sum(-1) / n_safe_cur
            s_mean_cur = (s_f * mask_f).sum(-1) / n_safe_cur
            all_W_norms.extend(W_norm_cur[valid_cur].cpu().tolist())
            all_s_vals.extend(s_mean_cur[valid_cur].cpu().tolist())

            # Off-diagonal and diagonal statistics (all valid entries in one shot)
            ri_cur, ci_cur = torch.triu_indices(N_cur, N_cur, offset=1, device=Sigma.device)
            valid_off_cur = mask_2d_cur[:, ri_cur, ci_cur]  # (B, n_pairs) bool
            off_vals_cur = Sigma[:, ri_cur, ci_cur][valid_off_cur]
            R_star_off_cur = batch["R_star"].float()[:, ri_cur, ci_cur][valid_off_cur]
            all_sigma_off.extend(off_vals_cur.cpu().tolist())
            all_sigma_diag.extend(Sigma.diagonal(dim1=-2, dim2=-1)[batch["test_mask"]].cpu().tolist())
            all_off_pred_flat.append(off_vals_cur.cpu().numpy())
            all_off_ora_flat.append(R_star_off_cur.cpu().numpy())

        # ---- Collect data for plots ----
        if do_plot and batch_idx < _PLOT_COLLECT_BATCHES:
            B = Sigma.shape[0]
            for b in range(B):
                n = int(batch["test_mask"][b].sum())
                if n < 2:
                    continue
                R_pred_b = Sigma[b, :n, :n].float().cpu().numpy()
                R_ora_b = batch["R_star"][b, :n, :n].float().cpu().numpy()
                ri, ci = np.triu_indices(n, k=1)
                all_off_pred.append(R_pred_b[ri, ci])
                all_off_ora.append(R_ora_b[ri, ci])
                if len(plot_episodes) < _MAX_PLOT_EPISODES:
                    plot_episodes.append({
                        "R_pred": R_pred_b,
                        "R_ora": R_ora_b,
                        "label": f"ep{batch_idx * B + b}\nN={n}",
                    })

    mean_cop       = sum(cop)     / len(cop)
    mean_ora_cop_z = sum(ora_cop_z) / len(ora_cop_z)

    metrics = {
        "y_nll_total":           sum(tot) / len(tot),
        "y_nll_copula":          mean_cop,
        "y_nll_marginal":        sum(mar) / len(mar),
        "y_nll_oracle":          sum(ora) / len(ora),
        "y_nll_oracle_copula":   sum(ora_cop) / len(ora_cop),
        "y_nll_oracle_marginal": sum(ora_mar) / len(ora_mar),
        "y_nll_oracle_copula_z": mean_ora_cop_z,
    }
    metrics["oracle_gap"] = metrics["y_nll_total"] - metrics["y_nll_oracle"]
    metrics["copula_gap"] = mean_cop - mean_ora_cop_z

    # Copula improvement fraction: 0 = identity baseline (R=I → NLL=0), 1 = oracle.
    # Negative means model is worse than outputting identity.
    metrics["copula_improvement"] = (
        mean_cop / mean_ora_cop_z if abs(mean_ora_cop_z) > 1e-12 else float("nan")
    )

    # Per-task copula NLL std — high value means unstable or heterogeneous tasks
    metrics["y_nll_copula_std"] = float(np.std(cop_per_task)) if cop_per_task else float("nan")

    # Sigma statistics — offdiag_mean ≈ 0 means model outputs near-identity
    if all_sigma_off:
        off_arr = np.array(all_sigma_off, dtype=np.float32)
        metrics["sigma_offdiag_mean"] = float(off_arr.mean())
        metrics["sigma_offdiag_std"]  = float(off_arr.std())
        metrics["sigma_offdiag_abs_mean"] = float(np.abs(off_arr).mean())
    else:
        metrics["sigma_offdiag_mean"] = metrics["sigma_offdiag_std"] = metrics["sigma_offdiag_abs_mean"] = 0.0
    metrics["sigma_diag_mean"] = float(np.mean(all_sigma_diag)) if all_sigma_diag else 1.0

    # Model output statistics
    metrics["W_norm_mean"] = float(np.mean(all_W_norms)) if all_W_norms else 0.0
    metrics["s_mean"]      = float(np.mean(all_s_vals))  if all_s_vals  else 0.0

    # Correlation quality vs oracle
    if all_off_pred_flat:
        off_p_all = np.concatenate(all_off_pred_flat)
        off_o_all = np.concatenate(all_off_ora_flat)
        cq = _corr_quality(off_p_all, off_o_all)
        metrics["corr_mse"]     = cq["mse"]
        metrics["corr_mae"]     = cq["mae"]
        metrics["corr_pearson"] = cq["pearson"]
        metrics["corr_bias"]    = cq["bias"]
    else:
        metrics["corr_mse"] = metrics["corr_mae"] = float("nan")
        metrics["corr_pearson"] = metrics["corr_bias"] = float("nan")

    model.train()

    plot_figs: list = []
    if do_plot:
        # — 2D hexbin density of off-diagonal correlations —
        if all_off_pred:
            off_p = np.concatenate(all_off_pred)
            off_o = np.concatenate(all_off_ora)
            lo = min(float(off_o.min()), float(off_p.min()))
            hi = max(float(off_o.max()), float(off_p.max()))
            fig_den, ax_den = plt.subplots(figsize=(5, 5))
            hb = ax_den.hexbin(off_o, off_p, gridsize=60, cmap="YlOrRd", mincnt=1, bins="log")
            fig_den.colorbar(hb, ax=ax_den, label="log10(count)")
            ax_den.plot([lo, hi], [lo, hi], "b--", lw=1)
            ax_den.set_xlabel("Oracle off-diag corr")
            ax_den.set_ylabel("Predicted off-diag corr")
            ax_den.set_title(f"step {step} — density ({len(off_p):,} values)")
            fig_den.tight_layout()
            plot_figs.append(fig_den)

        # — Oracle vs predicted correlation matrix grid —
        if plot_episodes:
            plot_figs.append(_corr_grid_fig(plot_episodes, step))

    return metrics, plot_figs


def save_checkpoint(model, optimizer, scheduler, cfg, step: int) -> None:
    os.makedirs(cfg.training.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.training.ckpt_dir, f"step_{step:07d}.pt")
    raw = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "step": step,
            "state_dict": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cfg": OmegaConf.to_container(cfg),
        },
        path,
    )


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = (
        "cuda" if cfg.training.device == "auto" and torch.cuda.is_available()
        else ("cpu" if cfg.training.device == "auto" else cfg.training.device)
    )

    t = cfg.training
    run_name = (
        f"copula_r={cfg.model.rank}"
        f"_lr={t.lr}_steps={t.steps}"
        f"_unfreeze={bool(cfg.model.get('unfreeze_backbone', False))}"
    )
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity if cfg.wandb.entity else None,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    all_files = sorted(glob(os.path.join(t.dataset_dir, "*.pt")))
    if not all_files:
        raise RuntimeError(
            f"No episode files in {t.dataset_dir}. Run generate_pit_dataset.py first."
        )
    n_val = min(49, max(1, int(len(all_files) * t.val_fraction)))
    val_files = all_files[:n_val]
    train_files = all_files[n_val:]
    print(f"Train: {len(train_files)} | Val: {len(val_files)} episodes")

    train_loader = DataLoader(
        CopulaDataset(file_list=train_files),
        batch_size=t.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
    )

    torch._dynamo.config.capture_scalar_outputs = True
    model = build_copula_transformer(cfg).to(device)
    model = torch.compile(model, dynamic=True)
    wandb.watch(model, log="gradients", log_freq=500)

    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_train_params:,}")
    wandb.config.update({"n_trainable_params": n_train_params})

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=t.lr, weight_decay=0.0)
    lr_min_frac = t.lr_min / t.lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_lr_lambda(s, t.warmup_steps, t.steps, lr_min_frac),
    )

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = GradScaler(device=device) if (use_amp and amp_dtype == torch.float16) else None

    jitter = float(cfg.model.get("sigma_jitter", 1e-4))

    model.train()
    data_iter = cycle(train_loader)
    loss_ema: float | None = None
    _EMA_ALPHA = 0.98

    for step in range(t.steps + 1):
        batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            out = model(batch)
        # Loss in float32 — Cholesky / log-det want full precision.
        Sigma = low_rank_correlation(
            out["W"].float(), out["s"].float(), batch["test_mask"], jitter=jitter
        )
        parts = y_space_nll(
            Sigma,
            batch["z_test"].float(),
            batch["log_pdf_test"].float(),
            batch["test_mask"],
        )
        loss = parts["total"]

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(trainable, t.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(trainable, t.clip_grad_norm)
            optimizer.step()
        grad_norm_val = float(grad_norm)

        scheduler.step()

        loss_val = loss.item()
        loss_ema = loss_val if loss_ema is None else _EMA_ALPHA * loss_ema + (1.0 - _EMA_ALPHA) * loss_val

        if step % t.log_every == 0:
            lr_now = scheduler.get_last_lr()[0]
            amp_scale = scaler.get_scale() if scaler is not None else 1.0
            cop_val = parts["copula"].item()
            mar_val = parts["marginal"].item()
            with torch.no_grad():
                w_norm_mean = float(out["W"].float().norm(dim=-1).mean().item())
                sig_stats = _sigma_stats(Sigma, batch["test_mask"])
            wandb.log(
                {
                    "train/y_nll_total":        loss_val,
                    "train/y_nll_copula":       cop_val,
                    "train/y_nll_marginal":     mar_val,
                    "train/lr":                 lr_now,
                    "train/grad_norm":          grad_norm_val,
                    "train/amp_scale":          amp_scale,
                    "train/loss_ema":           loss_ema,
                    "train/W_norm_mean":        w_norm_mean,
                    "train/sigma_offdiag_mean": sig_stats["offdiag_mean"],
                },
                step=step,
            )
            print(
                f"[{step:6d}] loss={loss_val:.4f} "
                f"(cop_nll={cop_val:.4f} ema_nll={loss_ema:.4f} mar_nll={mar_val:.4f}) "
                f"| grad_norm={grad_norm_val:.3f} "
                f"| od_μ={sig_stats['offdiag_mean']:+.4f} od_σ={sig_stats['offdiag_std']:.4f} "
                f"| lr={lr_now:.2e}"
            )

        if step % t.val_every == 0 and step > 0:
            plot_val_every = int(t.get("plot_val_every", 5000))
            do_plot = plot_val_every > 0 and step % plot_val_every == 0
            metrics, plot_figs = validate(
                model, val_files, cfg, device, step=step, do_plot=do_plot
            )
            log_dict = {f"val/{k}": v for k, v in metrics.items()}
            if plot_figs:
                log_dict["val/corr_density"] = wandb.Image(plot_figs[0])
                if len(plot_figs) > 1:
                    log_dict["val/corr_grid"] = wandb.Image(plot_figs[1])
                for f in plot_figs:
                    plt.close(f)
            wandb.log(log_dict, step=step)
            pearson = metrics["corr_pearson"]
            pearson_str = f"{pearson:.3f}" if math.isfinite(pearson) else "n/a"
            cop_nll = metrics["y_nll_copula"]
            cop_str = f"{cop_nll:.4f}" if math.isfinite(cop_nll) else "nan"
            print(
                f"[{step:6d}] VAL  "
                f"cop={cop_str}  "
                f"corr_r={pearson_str}  "
                f"corr_mse={metrics['corr_mse']:.4f}  "
                f"od_μ={metrics['sigma_offdiag_mean']:+.4f} od_σ={metrics['sigma_offdiag_std']:.4f} od_|r|={metrics['sigma_offdiag_abs_mean']:.4f}  "
                f"cop_std={metrics['y_nll_copula_std']:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if step % t.save_every == 0 and step > 0:
            save_checkpoint(model, optimizer, scheduler, cfg, step)

    save_checkpoint(model, optimizer, scheduler, cfg, t.steps)
    wandb.finish()


if __name__ == "__main__":
    main()
