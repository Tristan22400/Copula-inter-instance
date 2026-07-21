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

import sys
import time
import zlib
from glob import glob

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

import wandb

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from classical_kernels import DEFAULT_FAMILIES
from data_gen import KERNEL_REGISTRY, generate_gp_batch
from dataset import (
    CopulaDataset,
    ShardBlockSampler,
    ShardHomogeneousBatchSampler,
    collate_fn,
)
from loss import _safe_cholesky, gp_oracle_y_nll, oracle_copula_nll, y_space_nll
from model import build_copula_transformer, low_rank_correlation
from muon import Muon

_MAX_PLOT_EPISODES = 8
_PLOT_COLLECT_BATCHES = 5
_CORR_GRID_N_WRAP = 3  # stack corr_grid episodes across this many bands


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
    """Correlation-matrix grid: each estimator paired side-by-side with the oracle.

    One row per estimator — the model Pred. Each episode occupies *two adjacent
    columns*: the oracle ``R_star`` on the left and that row's prediction on the
    right, so every estimate sits right next to the ground truth it is compared
    against (no scanning to a distant oracle row). Each prediction cell is
    annotated with its per-episode upper-triangle MSE against the oracle.
    Episodes are wrapped across ``_CORR_GRID_N_WRAP`` stacked bands instead of
    one very wide row, so the figure stays a reasonable aspect ratio on screen.
    """
    n_ep = len(plot_episodes)

    # (row_label, lookup) for each *estimator*: Pred is a top-level episode key.
    # The oracle is no longer a row — it is the left cell of every episode pair.
    rows: list[tuple[str, str]] = [("Pred", "R_pred")]
    n_est = len(rows)

    n_wrap = max(1, min(_CORR_GRID_N_WRAP, n_ep))
    per_line = math.ceil(n_ep / n_wrap)
    n_col = 2 * per_line
    n_row = n_est * n_wrap

    fig, axes = plt.subplots(
        n_row, n_col, figsize=(max(n_col * 1.1, 4), max(n_row * 1.5, 4)),
        squeeze=False, constrained_layout=True,
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="lightgrey")

    def _draw(ax, mat):
        m = mat.copy()
        d = np.arange(m.shape[0])
        m[d, d] = np.nan  # blank diagonal so it doesn't dominate the colour scale
        return ax.imshow(m, cmap=cmap, vmin=-1, vmax=1,
                         interpolation="nearest", aspect="auto")

    im = None
    for idx, ep in enumerate(plot_episodes):
        line, col = divmod(idx, per_line)
        R_ora = ep["R_ora"]
        ri, ci = np.triu_indices(R_ora.shape[0], k=1)
        c_ora, c_est = 2 * col, 2 * col + 1

        for row_idx, (row_label, key) in enumerate(rows):
            row = line * n_est + row_idx
            mat = ep.get(key)  # top-level key: R_pred

            # Left cell: the oracle, redrawn beside every estimator as its reference.
            ax_o = axes[row, c_ora]
            im = _draw(ax_o, R_ora)
            ax_o.set_xticks([])
            ax_o.set_yticks([])
            if col == 0:
                ax_o.set_ylabel(row_label, fontsize=7)
            if row_idx == 0:
                ax_o.set_title(f"{ep['label']}\noracle", fontsize=6)

            # Right cell: this row's prediction, annotated with its MSE vs oracle.
            ax_e = axes[row, c_est]
            ax_e.set_xticks([])
            ax_e.set_yticks([])
            if row_idx == 0:
                ax_e.set_title("\nest", fontsize=6)
            if mat is None:
                ax_e.axis("off")
                continue
            im = _draw(ax_e, mat)
            mse = float(np.mean((mat[ri, ci] - R_ora[ri, ci]) ** 2))
            ax_e.set_xlabel(f"MSE={mse:.3f}", fontsize=6)

    # Blank out the trailing unused slots in the last (possibly partial) band.
    for idx in range(n_ep, per_line * n_wrap):
        line, col = divmod(idx, per_line)
        for row_idx in range(n_est):
            row = line * n_est + row_idx
            axes[row, 2 * col].axis("off")
            axes[row, 2 * col + 1].axis("off")

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.4, aspect=40, pad=0.02)
    fig.suptitle(
        f"step {step} — oracle (left) vs prediction (right), per episode", fontsize=8
    )
    return fig


def _build_synthetic_kernel_batches(cfg: DictConfig, device: str) -> dict[str, dict]:
    """Fixed per-kernel-family synthetic probe episodes for the
    ``kernel_fit/<family>`` validation metrics (see validate()).

    Generates B episodes per family via data_gen.generate_gp_batch — the same
    (x_train, z_train, x_test, R_star, ...) construction used for real
    training/val data, but with the generative kernel forced to one classical
    family instead of this run's usual composite/systematic mixture. Built
    once, with a fixed per-family seed, and reused every validation call, so
    kernel_fit/<family> only reflects the model's changing predictions on a
    frozen probe set — not resampling noise.
    """
    bcfg = cfg.get("baselines", {}) or {}
    families = list(bcfg.get("kernels") or DEFAULT_FAMILIES)
    n_episodes = int(bcfg.get("synth_n_episodes", 64))
    base_seed = int(bcfg.get("synth_seed", 20260718))

    batches: dict[str, dict] = {}
    for family in families:
        if family not in KERNEL_REGISTRY:
            continue  # not standalone-generatable (e.g. an unregistered composite)
        family_seed = base_seed + (zlib.crc32(family.encode()) % 10_000)
        synth_cfg = OmegaConf.merge(
            cfg,
            OmegaConf.create(
                {"seed": family_seed, "data": {"kernel": family, "systematic_composition": False}}
            ),
        )
        episodes = generate_gp_batch(synth_cfg, n_episodes, device="cpu")
        batch = collate_fn(episodes)
        batches[family] = {k: v.to(device) for k, v in batch.items()}
    return batches


def cosine_lr_lambda(step: int, warmup: int, total: int, lr_min_frac: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min_frac + (1.0 - lr_min_frac) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def _fmt_run_value(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "+".join(_fmt_run_value(v) for v in value)
    return str(value).replace(" ", "")


def _run_segments(cfg: DictConfig, prefix: str, keys: list[tuple[str, str]]) -> str:
    parts = []
    for cfg_key, label in keys:
        value = cfg.get(cfg_key, None)
        if value is not None:
            parts.append(f"_{prefix}{label}={_fmt_run_value(value)}")
    return "".join(parts)


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    cfg: DictConfig,
    device: str,
    step: int = 0,
    do_plot: bool = False,
    synth_kernel_batches: dict | None = None,
) -> tuple[dict, list]:
    # Do NOT call model.eval() here: TabICL's eval mode triggers _inference_forward
    # which uses InferenceManager with its own float16 autocast on CUDA, producing
    # NaN for certain inputs. There is no dropout in this model so eval mode has no
    # benefit. Use torch.no_grad() for efficiency instead.
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

    # Model-fit-to-classical-kernel metrics: runs the CURRENT model on a fixed
    # synthetic probe set per kernel family (see _build_synthetic_kernel_batches),
    # so these move with training progress (unlike a fixed data-only baseline).
    for family, sbatch in (synth_kernel_batches or {}).items():
        out_s = model(sbatch)
        Sigma_s = low_rank_correlation(
            out_s["W"].float(), out_s["s"].float(), sbatch["test_mask"], jitter=jitter
        )
        parts_s = y_space_nll(
            Sigma_s, sbatch["z_test"].float(), sbatch["log_pdf_test"].float(), sbatch["test_mask"]
        )
        N_s = Sigma_s.shape[1]
        ri_s, ci_s = torch.triu_indices(N_s, N_s, offset=1, device=Sigma_s.device)
        mask2d_s = sbatch["test_mask"].unsqueeze(-1) & sbatch["test_mask"].unsqueeze(-2)
        valid_s = mask2d_s[:, ri_s, ci_s]
        off_p_s = Sigma_s[:, ri_s, ci_s][valid_s].cpu().numpy()
        off_o_s = sbatch["R_star"].float()[:, ri_s, ci_s][valid_s].cpu().numpy()
        cq_s = _corr_quality(off_p_s, off_o_s)
        oracle_cop_s = oracle_copula_nll(
            sbatch["R_star"].float(), sbatch["z_test"].float(), sbatch["test_mask"]
        ).item()
        metrics[f"kernel_fit/{family}/copula_nll"]        = parts_s["copula"].item()
        metrics[f"kernel_fit/{family}/oracle_copula_nll"] = oracle_cop_s
        metrics[f"kernel_fit/{family}/corr_mse"]     = cq_s["mse"]
        metrics[f"kernel_fit/{family}/corr_mae"]     = cq_s["mae"]
        metrics[f"kernel_fit/{family}/corr_pearson"] = cq_s["pearson"]

        # One extra corr_grid column per kernel family: its own synthetic
        # episode's oracle beside the model's prediction on it — replaces the
        # old classical-kernel baseline rows (which used the real episodes'
        # oracle instead of a kernel-specific one).
        if do_plot:
            n_s = int(sbatch["test_mask"][0].sum())
            if n_s >= 2:
                plot_episodes.append({
                    "R_pred": Sigma_s[0, :n_s, :n_s].float().cpu().numpy(),
                    "R_ora":  sbatch["R_star"][0, :n_s, :n_s].float().cpu().numpy(),
                    "label":  f"kfit:{family}\nN={n_s}",
                })

    model.train()

    plot_figs: list = []
    if do_plot:
        # — 2D hexbin density of off-diagonal correlations —
        if all_off_pred:
            off_p = np.concatenate(all_off_pred)
            off_o = np.concatenate(all_off_ora)
            lo = min(float(off_o.min()), float(off_p.min()))
            hi = max(float(off_o.max()), float(off_p.max()))
            mse = float(np.mean((off_p - off_o) ** 2))
            fig_den, ax_den = plt.subplots(figsize=(5, 5))
            hb = ax_den.hexbin(off_o, off_p, gridsize=60, cmap="YlOrRd", mincnt=1, bins="log")
            fig_den.colorbar(hb, ax=ax_den, label="log10(count)")
            ax_den.plot([lo, hi], [lo, hi], "b--", lw=1)
            ax_den.set_xlabel("Oracle off-diag corr")
            ax_den.set_ylabel("Predicted off-diag corr")
            ax_den.set_title(f"step {step} — density ({len(off_p):,} values)  MSE={mse:.4f}")
            fig_den.tight_layout()
            plot_figs.append(fig_den)

        # — Oracle vs predicted correlation matrix grid —
        if plot_episodes:
            plot_figs.append(_corr_grid_fig(plot_episodes, step))

    return metrics, plot_figs


def save_checkpoint(model, optimizer, scheduler, cfg, step: int, scaler=None) -> None:
    if cfg.training.ckpt_dir is None:
        return
    os.makedirs(cfg.training.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.training.ckpt_dir, f"step_{step:07d}.pt")
    raw = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "step": step,
            "state_dict": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "cfg": OmegaConf.to_container(cfg),
        },
        path,
    )


def load_checkpoint(ckpt_path: str, model: nn.Module, optimizer, scheduler, device: str, scaler=None) -> int:
    """Restore model/optimizer/scheduler(/scaler) state from a checkpoint saved by save_checkpoint().

    Returns the step to resume from (checkpoint step + 1).
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"resume_ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    raw = getattr(model, "_orig_mod", model)
    raw.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["step"]) + 1


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = (
        "cuda" if cfg.training.device == "auto" and torch.cuda.is_available()
        else ("cpu" if cfg.training.device == "auto" else cfg.training.device)
    )

    t = cfg.training
    dataset_path = os.path.normpath(t.dataset_dir)
    dataset_parent, dataset_leaf = os.path.split(dataset_path)
    dataset_name = f"{os.path.basename(dataset_parent)}_{dataset_leaf}" if dataset_parent else dataset_leaf
    lora_cfg = cfg.get("lora", None)
    lora_enabled = bool(lora_cfg and lora_cfg.get("enabled", False))
    if lora_enabled:
        lora_stages = "+".join(lora_cfg.get("stages", ["icl"]))
        lora_str = f"_lora-r{lora_cfg.get('rank', 8)}-a{lora_cfg.get('alpha', 16.0)}-{lora_stages}"
    else:
        lora_str = "_nolora"
    unfreeze = bool(cfg.model.get("unfreeze_backbone", False))
    model_hparams = _run_segments(
        cfg.model,
        "m_",
        [
            ("rank", "r"),
            ("sigma_jitter", "jit"),
            ("d_model", "dm"),
            ("n_heads", "h"),
            ("n_layers_s1", "s1"),
            ("n_layers_s2", "s2"),
            ("n_layers_s3", "s3"),
            ("n_inducing", "ind"),
            ("n_cls", "cls"),
            ("p_max", "pmax"),
            ("d_max", "dmax"),
            ("dropout", "drop"),
        ],
    )
    training_hparams = _run_segments(
        t,
        "tr_",
        [
            ("batch_size", "bs"),
            ("steps", "steps"),
            ("warmup_steps", "wu"),
            ("muon_lr", "lr"),
            ("muon_lr_min", "lrmin"),
            ("muon_weight_decay", "wd"),
            ("muon_momentum", "mom"),
            ("muon_matched_adamw_rms", "rms"),
            ("muon_ns_steps", "ns"),
            ("clip_grad_norm", "clip"),
            ("nll_weight", "nll"),
            ("aux_mse_weight", "aux"),
            ("compile", "compile"),
        ],
    )
    resume_ckpt = t.get("resume_ckpt", None)
    resume_str = "_resumed" if resume_ckpt else ""
    run_name = (
        f"{dataset_name}"
        f"{model_hparams}"
        f"{training_hparams}"
        f"_unfreeze={unfreeze}"
        f"{lora_str}"
        f"{resume_str}"
    )
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity if cfg.wandb.entity else None,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    meta_path   = os.path.join(t.dataset_dir, "meta.pt")
    shard_files = sorted(glob(os.path.join(t.dataset_dir, "shard_*.pt")))

    train_sampler = None
    train_batch_sampler = None
    val_batch_sampler = None
    if shard_files and os.path.exists(meta_path):
        shard_block_shards = int(t.get("shard_block_shards", 16))
        # Cache must hold a full active block, or each worker still thrashes
        # against the block's shards one-by-one (+4 margin: workers process
        # batches round-robin, so a worker can straddle two blocks briefly).
        full_dataset = CopulaDataset(
            episode_dir=t.dataset_dir, shard_cache_size=shard_block_shards + 4
        )
        n = len(full_dataset)
        n_val = min(49, max(1, int(n * t.val_fraction)))
        # generate_gp_batch (data_gen.py) samples kernel_name/P/N/active_dims
        # once per shard call, shared by every episode in that shard — a
        # contiguous index block smaller than shard_size (as a plain
        # range(n_val) would be) pins validation to a single task shape
        # instead of sampling the full config distribution train sees. Stride
        # evenly across the whole dataset so val spans many shards/configs.
        val_indices = sorted(set(int(i) for i in torch.linspace(0, n - 1, n_val)))
        val_set = set(val_indices)
        train_indices = [i for i in range(n) if i not in val_set]
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset   = Subset(full_dataset, val_indices)

        # Detect per-shard-varying d_features. Such datasets store a different
        # feature count per shard (data_gen.py::_sample_d_features); a batch that
        # mixes shards then has mismatched feature columns and cannot be stacked
        # by collate_fn (TabICL consumes one (B, T, d_x) tensor; the row masks do
        # not cover the feature axis). Probe a handful of shards for varying d.
        shard_size = full_dataset.shard_size
        n_shards = (n + shard_size - 1) // shard_size
        probe_ids = torch.randperm(n_shards)[:8].tolist()
        d_seen = {
            int(full_dataset[min(sid * shard_size, n - 1)]["x_norm_train"].shape[-1])
            for sid in probe_ids
        }
        variable_d = len(d_seen) > 1

        if variable_d:
            # Batch strictly within one shard (train AND val) so every minibatch
            # is feature-homogeneous. A shard also shares one kernel/P/N/
            # active_dims, so these batches are single-task — the accepted price
            # of variable-d. shard_block_shards (cross-shard mixing) is moot here.
            print(
                "[train] per-shard-varying d_features detected "
                f"({sorted(d_seen)}...) → batching within single shards "
                "(single-task batches; shard_block_shards ignored)."
            )
            train_batch_sampler = ShardHomogeneousBatchSampler(
                train_dataset.indices,
                shard_size=shard_size,
                batch_size=t.batch_size,
                shuffle=True,
            )
            val_batch_sampler = ShardHomogeneousBatchSampler(
                val_dataset.indices,
                shard_size=shard_size,
                batch_size=t.batch_size,
                shuffle=False,
            )
        else:
            # Fixed-d: sharded datasets can span thousands of shards; a global
            # shuffle scatters each batch across dozens of them, thrashing the
            # shard LRU cache (dataset.py) with repeated full-shard reloads from
            # disk/NFS. Shuffle at shard-block granularity instead — still a true
            # per-epoch permutation (see ShardBlockSampler docstring), just with
            # locality-friendly ordering. Cross-shard mixing within a batch is
            # fine (and desirable) because every shard shares the same d.
            train_sampler = ShardBlockSampler(
                train_dataset.indices,
                shard_size=shard_size,
                block_shards=shard_block_shards,
            )
    else:
        all_files = sorted(glob(os.path.join(t.dataset_dir, "task_*.pt")))
        if not all_files:
            raise RuntimeError(
                f"No episode files in {t.dataset_dir}. Run generate_pit_dataset.py first."
            )
        n_val = min(49, max(1, int(len(all_files) * t.val_fraction)))
        train_dataset = CopulaDataset(file_list=all_files[n_val:])
        val_dataset   = CopulaDataset(file_list=all_files[:n_val])

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} episodes")

    # A batch_sampler (variable-d homogeneous batching) is mutually exclusive
    # with batch_size/sampler/shuffle, so pick one construction or the other.
    train_loader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
        prefetch_factor=4,
        **(
            {"batch_sampler": train_batch_sampler}
            if train_batch_sampler is not None
            else {
                "batch_size": t.batch_size,
                "sampler": train_sampler,
                "shuffle": (train_sampler is None),
            }
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
        prefetch_factor=4,
        **(
            {"batch_sampler": val_batch_sampler}
            if val_batch_sampler is not None
            else {"batch_size": t.batch_size, "shuffle": False}
        ),
    )

    baselines_on = bool(cfg.get("baselines", {}).get("enabled", True))
    synth_kernel_batches = _build_synthetic_kernel_batches(cfg, device) if baselines_on else {}

    model = build_copula_transformer(cfg).to(device)
    if bool(t.get("compile", False)):
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model, dynamic=True)
    wandb.watch(model, log="gradients", log_freq=5000)

    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_train_params:,}")
    wandb.config.update({"n_trainable_params": n_train_params})

    trainable = [p for p in model.parameters() if p.requires_grad]
    muon_params  = [p for p in trainable if p.ndim >= 2]
    adamw_params = [p for p in trainable if p.ndim < 2]
    optimizer = Muon(
        [
            {
                "params": muon_params,
                "use_muon": True,
                "lr": t.muon_lr,
                "weight_decay": t.muon_weight_decay,
                "momentum": t.muon_momentum,
                "matched_adamw_rms": t.muon_matched_adamw_rms,
                "ns_steps": t.muon_ns_steps,
                "nesterov": t.muon_nesterov,
                "adamw_betas": tuple(t.muon_adamw_betas),
                "adamw_eps": t.muon_adamw_eps,
            },
            {
                "params": adamw_params,
                "use_muon": False,
                "lr": t.muon_lr,
                "weight_decay": 0.0,
                "adamw_betas": tuple(t.muon_adamw_betas),
                "adamw_eps": t.muon_adamw_eps,
            },
        ]
    )
    lr_min_frac = t.muon_lr_min / t.muon_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_lr_lambda(s, t.warmup_steps, t.steps, lr_min_frac),
    )

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = GradScaler(device=device) if (use_amp and amp_dtype == torch.float16) else None

    start_step = 0
    if resume_ckpt:
        start_step = load_checkpoint(resume_ckpt, model, optimizer, scheduler, device, scaler=scaler)
        print(f"Resumed from {resume_ckpt} — continuing at step {start_step}")

    jitter = float(cfg.model.get("sigma_jitter", 1e-4))
    nll_weight = float(t.get("nll_weight", 1.0))
    aux_mse_weight = float(t.get("aux_mse_weight", 0.0))

    model.train()
    # NOT itertools.cycle(train_loader): cycle() caches every yielded batch
    # forever to replay on the next lap, which (a) freezes the sample order
    # after the first epoch — no reshuffling ever again — and (b) for a
    # multi-million-episode dataset means caching hundreds of GB of batch
    # tensors in RAM. Re-creating the iterator on StopIteration instead reuses
    # the persistent workers but calls the sampler fresh each epoch, so both
    # the plain RandomSampler and ShardBlockSampler reshuffle every pass.
    train_iter = iter(train_loader)
    loss_ema: float | None = None
    _EMA_ALPHA = 0.98
    _triu_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    # ---- Lightweight per-phase profiling -----------------------------------
    # GPU phases are timed with cuda.Event pairs (queued async, no sync cost);
    # they're only read out (which syncs) once per log_every window, matching
    # the existing "defer syncs to logging steps" pattern below. The data-fetch
    # phase is plain CPU wall time (waiting on the DataLoader iterator).
    _prof_phases = ("forward", "loss", "backward_step")
    _prof_ms = {k: 0.0 for k in ("data",) + _prof_phases}
    _prof_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
        {k: [] for k in _prof_phases} if device == "cuda" else {}
    )
    _prof_n = 0
    _last_log_wall = time.perf_counter()
    _last_log_step = 0

    def _phase_start():
        if device == "cuda":
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return ev
        return time.perf_counter()

    def _phase_end(name, start):
        if device == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            _prof_events[name].append((start, end))
        else:
            _prof_ms[name] += (time.perf_counter() - start) * 1000.0

    for step in range(start_step, t.steps + 1):
        _t_data0 = time.perf_counter()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        # non_blocking overlaps H→D transfer with previous GPU work (pin_memory=True)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        _prof_ms["data"] += (time.perf_counter() - _t_data0) * 1000.0

        optimizer.zero_grad(set_to_none=True)
        try:
            _ev_fwd0 = _phase_start()
            with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                out = model(batch)
            _phase_end("forward", _ev_fwd0)

            # Loss in float32 — Cholesky / log-det want full precision.
            _ev_loss0 = _phase_start()
            Sigma = low_rank_correlation(
                out["W"].float(), out["s"].float(), batch["test_mask"], jitter=jitter
            )
            parts = y_space_nll(
                Sigma,
                batch["z_test"].float(),
                batch["log_pdf_test"].float(),
                batch["test_mask"],
            )
            loss = nll_weight * parts["total"]

            # Auxiliary MSE on off-diagonal correlations vs oracle R_star.
            # Gives a direct gradient toward the oracle structure; weight=0 disables.
            aux_mse = Sigma.new_tensor(0.0)
            if aux_mse_weight > 0.0:
                N_t = Sigma.shape[1]
                mask_2d_t = batch["test_mask"].unsqueeze(-1) & batch["test_mask"].unsqueeze(-2)
                if N_t not in _triu_cache:
                    _triu_cache[N_t] = torch.triu_indices(N_t, N_t, offset=1, device=Sigma.device)
                ri_t, ci_t = _triu_cache[N_t]
                valid_off_t = mask_2d_t[:, ri_t, ci_t]  # (B, n_pairs)
                if valid_off_t.any():
                    pred_off = Sigma[:, ri_t, ci_t][valid_off_t]
                    ora_off = batch["R_star"].float()[:, ri_t, ci_t][valid_off_t]
                    aux_mse = ((pred_off - ora_off) ** 2).mean()
                loss = loss + aux_mse_weight * aux_mse
            _phase_end("loss", _ev_loss0)

            _ev_bwd0 = _phase_start()
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

            scheduler.step()
            _phase_end("backward_step", _ev_bwd0)
            _prof_n += 1
        except torch.cuda.OutOfMemoryError:
            # P/N (attention length T=P+N) vary a lot per shard (see comment at
            # top of file) while batch_size is fixed, so an occasional
            # oversized shard can exceed VRAM even though most batches fit
            # comfortably. Rather than let one bad shard kill a 500k-step run,
            # drop it and move on — one skipped step is noise at this scale.
            P_b, N_b = batch["x_train"].shape[1], batch["x_test"].shape[1]
            print(
                f"[{step:6d}] CUDA OOM on batch (B={batch['x_train'].shape[0]}, "
                f"P={P_b}, N={N_b}, T={P_b + N_b}) — skipping step."
            )
            optimizer.zero_grad(set_to_none=True)
            del batch
            torch.cuda.empty_cache()
            continue

        # Defer .item() / float() GPU syncs to logging steps — saves 2+ syncs/step
        if step % t.log_every == 0:
            loss_val = loss.item()
            loss_ema = loss_val if loss_ema is None else _EMA_ALPHA * loss_ema + (1.0 - _EMA_ALPHA) * loss_val
            grad_norm_val = float(grad_norm)
            lr_now = scheduler.get_last_lr()[0]
            amp_scale = scaler.get_scale() if scaler is not None else 1.0
            cop_val = parts["copula"].item()
            mar_val = parts["marginal"].item()
            aux_mse_val = aux_mse.item()
            with torch.no_grad():
                w_norm_mean = float(out["W"].float().norm(dim=-1).mean().item())
                sig_stats = _sigma_stats(Sigma, batch["test_mask"])
                # Diagnostic for the non-finite-slice masking in _safe_cholesky
                # (loss.py), which silently substitutes identity for any
                # corrupted episode rather than warning per-occurrence.
                sigma_nonfinite = int(
                    (~torch.isfinite(Sigma).flatten(1).all(-1)).sum().item()
                )

            # ---- Profiling readout (one sync here, piggy-backing on the ----
            # ---- syncs the .item() calls above already forced) ------------
            if device == "cuda" and _prof_n > 0:
                torch.cuda.synchronize()
                for name in _prof_phases:
                    _prof_ms[name] += sum(s.elapsed_time(e) for s, e in _prof_events[name])
                    _prof_events[name].clear()
            now = time.perf_counter()
            steps_done = max(step - _last_log_step, 1)
            step_ms = {k: v / _prof_n for k, v in _prof_ms.items()} if _prof_n else {k: 0.0 for k in _prof_ms}
            wall_step_ms = (now - _last_log_wall) / steps_done * 1000.0
            steps_per_sec = steps_done / max(now - _last_log_wall, 1e-9)
            _last_log_wall = now
            _last_log_step = step
            for k in _prof_ms:
                _prof_ms[k] = 0.0
            _prof_n = 0

            # ---- GPU memory share: fraction of device VRAM capacity held ----
            # (distinct from wandb's system "GPU Memory Access %" panel, which
            # is a time-based bandwidth-utilization metric, not a capacity share)
            if device == "cuda":
                _free_b, _total_b = torch.cuda.mem_get_info()
                mem_alloc_pct = 100.0 * torch.cuda.memory_allocated() / _total_b
                mem_reserved_pct = 100.0 * torch.cuda.memory_reserved() / _total_b
                # max_memory_allocated() is a lifetime high-water mark, not a
                # per-step reading — left un-reset it stays pinned near its
                # first spike and hides real step-to-step variance (this is
                # part of why the OOM at a data-dependent large-T shard came
                # as a surprise from the logs). Reset after each read so the
                # printed value is "peak since last log line".
                mem_peak_pct = 100.0 * torch.cuda.max_memory_allocated() / _total_b
                torch.cuda.reset_peak_memory_stats()
            else:
                mem_alloc_pct = mem_reserved_pct = mem_peak_pct = 0.0

            wandb.log(
                {
                    "train/y_nll_total":          loss_val,
                    "train/y_nll_copula":         cop_val,
                    "train/y_nll_marginal":       mar_val,
                    "train/aux_mse":              aux_mse_val,
                    "train/lr":                   lr_now,
                    "train/grad_norm":            grad_norm_val,
                    "train/amp_scale":            amp_scale,
                    "train/loss_ema":             loss_ema,
                    "train/W_norm_mean":          w_norm_mean,
                    "train/sigma_offdiag_mean":   sig_stats["offdiag_mean"],
                    "train/sigma_nonfinite_count": sigma_nonfinite,
                    "perf/step_ms":                wall_step_ms,
                    "perf/steps_per_sec":          steps_per_sec,
                    "perf/data_ms":                step_ms["data"],
                    "perf/forward_ms":             step_ms["forward"],
                    "perf/loss_ms":                step_ms["loss"],
                    "perf/backward_step_ms":       step_ms["backward_step"],
                    "perf/mem_allocated_pct":      mem_alloc_pct,
                    "perf/mem_reserved_pct":        mem_reserved_pct,
                    "perf/mem_peak_pct":           mem_peak_pct,
                },
                step=step,
            )
            aux_str = f" aux_mse={aux_mse_val:.4f}" if aux_mse_weight > 0.0 else ""
            nonfinite_str = f" | sigma_nonfinite={sigma_nonfinite}" if sigma_nonfinite else ""
            print(
                f"[{step:6d}] loss={loss_val:.4f} "
                f"(cop_nll={cop_val:.4f} ema_nll={loss_ema:.4f} mar_nll={mar_val:.4f}{aux_str}) "
                f"| grad_norm={grad_norm_val:.3f} "
                f"| od_μ={sig_stats['offdiag_mean']:+.4f} od_σ={sig_stats['offdiag_std']:.4f} "
                f"| lr={lr_now:.2e}{nonfinite_str}\n"
                f"         perf: step={wall_step_ms:.1f}ms ({steps_per_sec:.2f} it/s) "
                f"data={step_ms['data']:.1f} fwd={step_ms['forward']:.1f} "
                f"loss={step_ms['loss']:.1f} bwd+opt={step_ms['backward_step']:.1f} "
                f"mem={mem_alloc_pct:.1f}%/{mem_reserved_pct:.1f}% (peak {mem_peak_pct:.1f}%)"
            )

        if step % t.val_every == 0 and step > 0:
            plot_val_every = int(t.get("plot_val_every", 5000))
            do_plot = plot_val_every > 0 and step % plot_val_every == 0
            metrics, plot_figs = validate(
                model, val_loader, cfg, device, step=step, do_plot=do_plot,
                synth_kernel_batches=synth_kernel_batches,
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
            save_checkpoint(model, optimizer, scheduler, cfg, step, scaler=scaler)

    save_checkpoint(model, optimizer, scheduler, cfg, t.steps, scaler=scaler)
    wandb.finish()


if __name__ == "__main__":
    main()
