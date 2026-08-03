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

import gc
import math
import os
import traceback

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
from torch.utils.flop_counter import FlopCounterMode

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
from live_dataset import build_fixed_live_val_batches, build_live_train_loader
from loss import _safe_cholesky, gp_oracle_y_nll, oracle_copula_nll, y_space_nll
from pit import corrupt_z_train
from model import build_copula_transformer, low_rank_correlation
from muon import Muon

_MAX_PLOT_EPISODES = 8
_PLOT_COLLECT_BATCHES = 5
_CORR_GRID_N_WRAP = 3  # stack corr_grid episodes across this many bands

# Peak dense FP16/BF16 tensor-core throughput (TFLOPS) per NVIDIA datasheets.
# torch has no API to query this, so match torch.cuda.get_device_name() against
# these substrings for Model FLOPs Utilization (MFU) logging (see
# get_gpu_peak_flops). Ordered — first match wins, so keys that are a prefix
# of another entry's name (e.g. "H100" vs "H100 PCIE", "L40S" vs "L4") must
# come after it.
#
# Covers every GPU model currently in the Grid5000 Grenoble site's `oarnodes`
# inventory (vercors2/3/4/5/7/8/9/10/11/12/13/14/15/16/17/18, drac, kinovis,
# adonis), plus the common cloud/datacenter cards from the original request,
# so a job lands with an accurate MFU denominator on whichever cluster it's
# scheduled to. Pascal (P100/TITAN Xp/TITAN X) and older Fermi/Tesla-10-series
# (C1060/C2050) cards predate Tensor Cores entirely, so their entries are
# standard CUDA-core FP16/FP32 peak instead — MFU numbers on those nodes are a
# rough proxy, not a Tensor Core utilization figure. C1060/C2050 (adonis) are
# also old enough (CUDA compute capability 1.3/2.0) that current PyTorch likely
# can't run on them at all; included only for completeness.
_GPU_PEAK_TFLOPS: dict[str, float] = {
    "H100 PCIE": 756e12,
    "H100": 989e12,               # SXM/HBM3/bare "H100" — no PCIe suffix in the name
    "RTX PRO 6000 BLACKWELL": 1021e12,  # vercors18 — estimate, not verified against a datasheet
    "A100": 312e12,
    "V100": 125e12,
    "RTX 4090": 165e12,
    "RTX 3090": 142e12,
    "RTX A6000": 130e12,
    "RTX A5000": 111e12,          # vercors9/10
    "RTX 6000 ADA": 728e12,       # vercors14/15 — this training node's GPU
    "L40S": 362e12,               # kinovis, vercors17 — must precede "L4"
    "L4": 121e12,                 # vercors16
    "QUADRO RTX 8000": 130.5e12,  # vercors5/8/11
    "TITAN RTX": 130.5e12,        # vercors4/7/12 — same TU102 die as Quadro RTX 8000
    "P100": 21.2e12,              # drac — Pascal, no Tensor Cores: CUDA-core FP16 peak
    "TITAN XP": 12.15e12,         # vercors3 — Pascal, no Tensor Cores: CUDA-core FP32 peak
    "TITAN X (PASCAL)": 10.97e12, # vercors2 — Pascal, no Tensor Cores: CUDA-core FP32 peak
    "C2050": 1.03e12,             # adonis (Fermi) — no Tensor Cores: CUDA-core FP32 peak
    "C1060": 0.933e12,            # adonis (Tesla 10-series) — no Tensor Cores: CUDA-core FP32 peak
}
_GPU_PEAK_FLOPS_DEFAULT = 100e12


def get_gpu_peak_flops(device: int = 0) -> float:
    """Theoretical peak dense FP16/BF16 tensor-core FLOPs for the active GPU.

    PyTorch has no API for this, so match torch.cuda.get_device_name() against
    a hardcoded table of common datacenter/consumer cards (_GPU_PEAK_TFLOPS).
    Falls back to a conservative default (with a printed warning) for anything
    unrecognized, so MFU numbers on an unrecognized GPU are directional only.
    """
    if not torch.cuda.is_available():
        return _GPU_PEAK_FLOPS_DEFAULT
    name = torch.cuda.get_device_name(device).upper()
    for key, tflops in _GPU_PEAK_TFLOPS.items():
        if key in name:
            return tflops
    print(
        f"[train] WARNING: unrecognized GPU {name!r} — no entry in the MFU "
        f"peak-FLOPs table, falling back to {_GPU_PEAK_FLOPS_DEFAULT / 1e12:.0f} "
        "TFLOPS. MFU numbers will be approximate."
    )
    return _GPU_PEAK_FLOPS_DEFAULT


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


def _oracle_diagonal_order(R_ora: np.ndarray) -> np.ndarray:
    """Permutation that reorders R_ora's rows/cols so strongly-correlated test
    points sit next to each other, concentrating high |correlation| along the
    diagonal instead of scattered uniformly across the heatmap.

    Hierarchical-clustering seriation (average-linkage, optimal leaf ordering)
    on the distance ``1 - |R_ora|`` — points the oracle says are strongly
    (anti-)correlated get small distance and land close together in the
    leaf order. N < 3 has no meaningful ordering.
    """
    n = R_ora.shape[0]
    if n < 3:
        return np.arange(n)
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    dist = 1.0 - np.abs(R_ora)
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)  # guard against float32 asymmetry
    link = linkage(squareform(dist, checks=False), method="average", optimal_ordering=True)
    return np.asarray(leaves_list(link))


def _corr_grid_fig(plot_episodes: list[dict], step: int) -> plt.Figure:
    """Correlation-matrix grid: each estimator paired side-by-side with the oracle.

    One row per estimator — the model Pred. Each episode occupies *two adjacent
    columns*: the oracle ``R_star`` on the left and that row's prediction on the
    right, so every estimate sits right next to the ground truth it is compared
    against (no scanning to a distant oracle row). Each prediction cell is
    annotated with its per-episode upper-triangle MSE against the oracle.
    Episodes are wrapped across ``_CORR_GRID_N_WRAP`` stacked bands instead of
    one very wide row, so the figure stays a reasonable aspect ratio on screen.

    Rows/cols of both oracle and prediction are reordered per episode by
    ``_oracle_diagonal_order`` (derived from the oracle alone, then reused for
    the prediction) so the oracle's correlation structure is as diagonal-heavy
    as possible and the two panels stay directly comparable.
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
        order = _oracle_diagonal_order(ep["R_ora"])
        R_ora = ep["R_ora"][order][:, order]
        ri, ci = np.triu_indices(R_ora.shape[0], k=1)
        c_ora, c_est = 2 * col, 2 * col + 1

        for row_idx, (row_label, key) in enumerate(rows):
            row = line * n_est + row_idx
            mat = ep.get(key)  # top-level key: R_pred
            if mat is not None:
                mat = mat[order][:, order]  # same permutation as the oracle, for a fair comparison

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


def _live_data_segment(data_cfg: DictConfig) -> str:
    """Summarize which kind of data live generation is producing this run.

    Unlike disk mode (where dataset_dir's basename is a user-curated name),
    live generation reads cfg.data.* directly every step, so the run name
    needs its own summary of the composition/correlation/warp knobs that
    otherwise wouldn't show up anywhere but the full wandb config.
    """
    if bool(data_cfg.get("systematic_composition", False)):
        lo = data_cfg.get("composite_num_kernels_min", 1)
        hi = data_cfg.get("composite_num_kernels_max", 1)
        kernel_str = f"syscomp{lo}-{hi}"
    elif data_cfg.get("kernels", None) is not None:
        kernel_str = f"mix{len(data_cfg.kernels)}"
    else:
        kernel_str = str(data_cfg.get("kernel", "rbf"))

    dfeat_str = (
        "logN"
        if data_cfg.get("d_features_lognormal_loc", None) is not None
        else str(data_cfg.get("d_features", 10))
    )

    tags = []
    sign_comp = float(data_cfg.get("sign_modulation_component_prob", 0.0) or 0.0)
    sign_outer = float(data_cfg.get("sign_modulation_outer_prob", 0.0) or 0.0)
    if sign_comp > 0 or sign_outer > 0:
        tags.append("sgn")
    if bool(data_cfg.get("mlp_mixing_enabled", False)):
        tags.append("mlp")
    if bool(data_cfg.get("structural_warp_enabled", False)):
        tags.append("struct")
    if bool(data_cfg.get("mean_fn_enabled", False)):
        tags.append("mean")
    if bool(data_cfg.get("z_train_corruption_enabled", False)):
        tags.append("zcorrupt")
    oracle = data_cfg.get("oracle_mode", None)
    if oracle is not None and oracle != "prior":
        tags.append(f"oracle-{oracle}")

    parts = [kernel_str, dfeat_str] + tags
    return "_d_" + "-".join(parts)


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
            neg_frac = float(np.mean(off_o < 0))
            fig_den, ax_den = plt.subplots(figsize=(5, 5))
            hb = ax_den.hexbin(off_o, off_p, gridsize=60, cmap="YlOrRd", mincnt=1, bins="log")
            fig_den.colorbar(hb, ax=ax_den, label="log10(count)")
            ax_den.plot([lo, hi], [lo, hi], "b--", lw=1)

            # Median predicted corr per oracle-corr bin.
            n_bins = 40
            bin_edges = np.linspace(lo, hi, n_bins + 1)
            bin_idx = np.clip(np.digitize(off_o, bin_edges) - 1, 0, n_bins - 1)
            bin_centers, bin_medians = [], []
            for b in range(n_bins):
                sel = bin_idx == b
                if sel.any():
                    bin_centers.append(0.5 * (bin_edges[b] + bin_edges[b + 1]))
                    bin_medians.append(float(np.median(off_p[sel])))
            ax_den.plot(bin_centers, bin_medians, "g-", lw=1.5, label="median pred")
            ax_den.legend(loc="upper left", fontsize=8)

            ax_den.set_xlabel("Oracle off-diag corr")
            ax_den.set_ylabel("Predicted off-diag corr")
            ax_den.set_title(
                f"step {step} — density ({len(off_p):,} values)  "
                f"MSE={mse:.4f}  neg%={100 * neg_frac:.1f}"
            )
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


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
) -> None:
    """Restore model weights and optimizer/scaler state from a checkpoint.

    The scheduler and step count are intentionally NOT restored: resuming
    always restarts at step 0 with a fresh warmup/cosine schedule and the
    full step budget of this run's config, rather than continuing the
    previous run's schedule and step count. Optimizer moments (Adam/Muon)
    and the AMP grad scaler state are restored so the run doesn't have to
    relearn gradient statistics from scratch.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"resume_ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    raw = getattr(model, "_orig_mod", model)
    raw.load_state_dict(ckpt["state_dict"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])


def _forward_and_loss(
    *,
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    nll_weight: float,
    aux_mae_weight: float,
    jitter: float,
    triu_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
    phase_start=lambda: None,
    phase_end=lambda name, start: None,
):
    """Forward pass + NLL(+aux MAE) loss — shared by _run_train_step and the
    throwaway FLOP-measurement pass in _measure_step_flops (phase_start/
    phase_end default to no-ops there, since that pass must not pollute the
    fwd/loss/backward_step timers with a second, throwaway forward).
    """
    ev_fwd0 = phase_start()
    with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
        out = model(batch)
    phase_end("forward", ev_fwd0)

    # Loss in float32 — Cholesky / log-det want full precision.
    ev_loss0 = phase_start()
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

    # Auxiliary MAE (L1) on off-diagonal correlations vs oracle R_star.
    aux_mae = Sigma.new_tensor(0.0)
    if aux_mae_weight > 0.0:
        n_test = Sigma.shape[1]
        mask_2d = batch["test_mask"].unsqueeze(-1) & batch["test_mask"].unsqueeze(-2)
        if n_test not in triu_cache:
            triu_cache[n_test] = torch.triu_indices(
                n_test, n_test, offset=1, device=Sigma.device
            )
        ri, ci = triu_cache[n_test]
        valid_off = mask_2d[:, ri, ci]
        if valid_off.any():
            pred_off = Sigma[:, ri, ci][valid_off]
            oracle_off = batch["R_star"].float()[:, ri, ci][valid_off]
            aux_mae = (pred_off - oracle_off).abs().mean()
        loss = loss + aux_mae_weight * aux_mae
    phase_end("loss", ev_loss0)
    return out, Sigma, parts, loss, aux_mae


def _measure_step_flops(
    *,
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    nll_weight: float,
    aux_mae_weight: float,
    jitter: float,
    triu_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> float:
    """Throwaway forward+backward (no optimizer/scheduler step) under
    FlopCounterMode, to measure this step's real dispatched FLOPs for MFU.

    Deliberately run *after*, not instead of, the real timed training step
    (see the call site in main()): FlopCounterMode's per-op Python dispatch
    hook adds real wall-clock overhead on GPU (measured ~3x on an RTX A5000
    smoke test), so wrapping the actual training step in it would bias
    iter_time_sec high and mfu_pct/tokens_per_sec low. This redundant pass
    costs one extra forward+backward, but only at log_every steps.
    """
    with FlopCounterMode(display=False) as flop_ctr:
        _, _, _, loss, _ = _forward_and_loss(
            model=model,
            batch=batch,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            nll_weight=nll_weight,
            aux_mae_weight=aux_mae_weight,
            jitter=jitter,
            triu_cache=triu_cache,
        )
        loss.backward()
    model.zero_grad(set_to_none=True)
    return flop_ctr.get_total_flops()


def _run_train_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    trainable: list[nn.Parameter],
    batch: dict[str, torch.Tensor],
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler: GradScaler | None,
    clip_grad_norm: float,
    nll_weight: float,
    aux_mae_weight: float,
    jitter: float,
    triu_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
    phase_start,
    phase_end,
):
    """Execute one training step in a short-lived frame.

    This function deliberately owns all tensors which can be attached to the
    autograd graph.  If CUDA raises ``OutOfMemoryError`` anywhere in the step,
    the exception unwinds this frame before the caller starts the next batch.
    Keeping this boundary separate from the long-lived training loop is
    important: ``empty_cache()`` only releases unreferenced allocator blocks;
    it cannot release tensors still reachable from a loop local or traceback.
    """
    out, Sigma, parts, loss, aux_mae = _forward_and_loss(
        model=model,
        batch=batch,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        nll_weight=nll_weight,
        aux_mae_weight=aux_mae_weight,
        jitter=jitter,
        triu_cache=triu_cache,
        phase_start=phase_start,
        phase_end=phase_end,
    )
    grad_norm = None

    ev_bwd0 = phase_start()
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(trainable, clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(trainable, clip_grad_norm)
        optimizer.step()

    scheduler.step()
    phase_end("backward_step", ev_bwd0)
    return out, Sigma, parts, loss, aux_mae, grad_norm


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = (
        "cuda" if cfg.training.device == "auto" and torch.cuda.is_available()
        else ("cpu" if cfg.training.device == "auto" else cfg.training.device)
    )
    gpu_peak_flops = get_gpu_peak_flops() if device == "cuda" else None
    if device == "cuda":
        print(
            f"[train] GPU: {torch.cuda.get_device_name(0)} — assumed peak "
            f"{gpu_peak_flops / 1e12:.0f} TFLOPS (dense bf16/fp16 tensor core) for MFU"
        )

    t = cfg.training
    live_generation = bool(t.get("live_generation", False))
    if live_generation:
        # dataset_dir is ignored entirely in this mode (see below) — naming the
        # run after it would be misleading, so summarize cfg.data.* instead.
        # Also fold in ckpt_dir's basename since it's often the only
        # user-chosen, human-readable identifier for a live-generation run.
        ckpt_dir = t.get("ckpt_dir", None)
        ckpt_str = f"_ckpt-{os.path.basename(os.path.normpath(ckpt_dir))}" if ckpt_dir else ""
        dataset_name = "live" + _live_data_segment(cfg.data) + ckpt_str
    else:
        dataset_path = os.path.normpath(t.dataset_dir)
        # Include the parent folder so runs pointing at same-named shard dirs
        # under different parents (e.g. runA/shards vs runB/shards) stay distinct.
        parent_name = os.path.basename(os.path.dirname(dataset_path))
        shard_name = os.path.basename(dataset_path)
        dataset_name = f"{parent_name}/{shard_name}" if parent_name else shard_name
    lora_cfg = cfg.get("lora", None)
    lora_enabled = bool(lora_cfg and lora_cfg.get("enabled", False))
    if lora_enabled:
        lora_stages = "+".join(lora_cfg.get("stages", ["icl", "row", "col"]))
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
            ("aux_mae_weight", "aux"),
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

    if live_generation:
        # No on-disk dataset at all: episodes are generated on the fly by
        # DataLoader worker processes (see live_dataset.py). Temporary
        # substitute for the disk pipeline below — set training.live_generation
        # =false (the default) to fall back to it unchanged.
        print(
            "[train] live_generation=true — generating episodes on the fly, "
            f"no dataset_dir read ({t.dataset_dir!r} ignored). "
            f"ckpt_dir={t.get('ckpt_dir', None)!r}"
        )
        train_loader = build_live_train_loader(cfg, t, device)
        val_loader   = build_fixed_live_val_batches(cfg, t)
        print(f"Train: <live> | Val: {len(val_loader) * t.batch_size} episodes (fixed)")
    else:
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
            n_val = min(int(t.get("val_episodes", 500)), n)
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
            n_val = min(int(t.get("val_episodes", 500)), len(all_files))
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
        load_checkpoint(resume_ckpt, model, device, optimizer=optimizer, scaler=scaler)
        print(f"Resumed weights + optimizer/scaler state from {resume_ckpt} — restarting schedule at step 0")

    jitter = float(cfg.model.get("sigma_jitter", 1e-4))
    nll_weight = float(t.get("nll_weight", 1.0))
    aux_mae_weight = float(t.get("aux_mae_weight", 0.0))

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
    _prof_T_sum = 0  # sum of per-step sequence length T=P+N, for MFU's avg batch shape
    # This step's own (not window-averaged) phase ms — only meaningful on the
    # CPU path, where _phase_end has no per-event list to pull a single step's
    # timing back out of after the fact (see the "last_step_ms" readout below).
    _prof_last_ms = {k: 0.0 for k in _prof_phases}
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
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            _prof_ms[name] += elapsed_ms
            _prof_last_ms[name] = elapsed_ms

    for step in range(start_step, t.steps + 1):
        _t_data0 = time.perf_counter()
        # Pre-clear loop references.  The actual computation graph is owned by
        # _run_train_step, but these names are still used for logging after a
        # successful step and must be safe to clean up after an OOM.
        batch = None
        out = Sigma = parts = loss = aux_mae = grad_norm = None
        step_flops = None
        try:
            # Keep the CPU batch separate so an OOM during H→D transfer can be
            # recovered just like an OOM in the model step.
            raw_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            raw_batch = next(train_iter)

        optimizer.zero_grad(set_to_none=True)
        try:
            # non_blocking overlaps H→D transfer with previous GPU work
            # (pin_memory=True).
            batch = {k: v.to(device, non_blocking=True) for k, v in raw_batch.items()}
            # Robustness augmentation: corrupt z_train toward a noisier proxy
            # of the exact GP-LOO residual it's otherwise always trained on
            # (see pit.py::corrupt_z_train's docstring for why -- deployment
            # on real, non-GP data can only ever produce an approximate PIT).
            # Reads cfg.data (not cfg.training) -- a data-generation-time
            # modulation, same convention as sign_modulation_component_prob/
            # mlp_mixing_enabled -- so an oarsub override like
            # data.z_train_corruption_enabled=true applies with no separate
            # wiring. No-op unless data.z_train_corruption_enabled is set;
            # applies identically regardless of whether `batch` came from the
            # disk pipeline or live_generation (both produce the same
            # collated schema by this point).
            batch["z_train"] = corrupt_z_train(
                batch["z_train"], batch["train_mask"], cfg.data, step,
            )
            _prof_ms["data"] += (time.perf_counter() - _t_data0) * 1000.0
            _prof_T_sum += batch["x_train"].shape[1] + batch["x_test"].shape[1]

            # _run_train_step owns the graph-bearing locals.  If it raises,
            # its frame is released as the exception unwinds.
            out, Sigma, parts, loss, aux_mae, grad_norm = _run_train_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                trainable=trainable,
                batch=batch,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
                clip_grad_norm=t.clip_grad_norm,
                nll_weight=nll_weight,
                aux_mae_weight=aux_mae_weight,
                jitter=jitter,
                triu_cache=_triu_cache,
                phase_start=_phase_start,
                phase_end=_phase_end,
            )
            # At log steps only, run one throwaway forward+backward under
            # FlopCounterMode to measure this step's *actual* dispatched
            # FLOPs for MFU (see the "Model FLOPs Utilization" comment
            # further below) instead of estimating them analytically. An
            # analytic count (params * batch * seq_len, PaLM-style) is wrong
            # here on two counts: this model attends over both rows *and*
            # columns (model.py wraps TabICL's col_embedder -> row_interactor
            # -> icl_predictor stack, so backbone compute scales with
            # d_features too, not just P+N), and `n_train_params` undercounts
            # forward FLOPs whenever the backbone is frozen
            # (model.unfreeze_backbone=false, or LoRA-only training) since a
            # frozen module still runs a full forward pass. FlopCounterMode
            # counts real dispatched ops, so it's automatically correct for
            # both: it sees the true column/row op graph, and autograd's own
            # graph pruning means backward-through-frozen-only subtrees is
            # (correctly) never dispatched, hence never counted.
            #
            # This is a *separate* pass from the real step above rather than
            # wrapping the real step itself, because FlopCounterMode's per-op
            # dispatch hook has real wall-clock cost on GPU — wrapping the
            # timed step would inflate iter_time_sec and understate
            # mfu_pct/tokens_per_sec (confirmed ~3x on an RTX A5000 smoke
            # test). The throwaway pass's own grads are discarded and never
            # touch the optimizer/scheduler, so it doesn't affect training;
            # it costs one extra forward+backward, but only every log_every
            # steps.
            if step % t.log_every == 0:
                try:
                    step_flops = _measure_step_flops(
                        model=model,
                        batch=batch,
                        device=device,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        nll_weight=nll_weight,
                        aux_mae_weight=aux_mae_weight,
                        jitter=jitter,
                        triu_cache=_triu_cache,
                    )
                except torch.cuda.OutOfMemoryError:
                    # The real step above already completed and applied its
                    # optimizer update — this is only the throwaway FLOP
                    # measurement running out of headroom, not a failed
                    # training step, so just skip the FLOP count for this
                    # log line rather than falling into the OOM handler
                    # below (which assumes the whole step needs discarding).
                    step_flops = None
                    if device == "cuda":
                        torch.cuda.empty_cache()
            _prof_n += 1
        except torch.cuda.OutOfMemoryError as exc:
            # P/N (attention length T=P+N) vary a lot per shard (see comment at
            # top of file) while batch_size is fixed, so an occasional
            # oversized shard can exceed VRAM even though most batches fit
            # comfortably. Rather than let one bad shard kill a 500k-step run,
            # drop it and move on — one skipped step is noise at this scale.
            shape_batch = batch if batch is not None else raw_batch
            P_b, N_b = shape_batch["x_train"].shape[1], shape_batch["x_test"].shape[1]
            print(
                f"[{step:6d}] CUDA OOM on batch (B={shape_batch['x_train'].shape[0]}, "
                f"P={P_b}, N={N_b}, T={P_b + N_b}) — skipping step."
            )
            # The active exception traceback otherwise keeps the failed
            # _run_train_step frame alive until this handler exits.  Clear its
            # locals before empty_cache(), so the graph is truly unreachable
            # when the allocator is asked to release cached blocks.
            traceback.clear_frames(exc.__traceback__)
            del exc
            optimizer.zero_grad(set_to_none=True)
            # Do not retain CUDA events from a failed/incomplete phase.  They
            # do not own the graph, but can otherwise accumulate when OOMs are
            # frequent between log intervals.
            if device == "cuda":
                for events in _prof_events.values():
                    events.clear()
            del raw_batch, batch, shape_batch, out, Sigma, parts, loss, aux_mae, grad_norm
            # `del` above only drops the *names*.  A failed autograd graph is a
            # reference *cycle* (tensor -> grad_fn -> saved tensors -> ...), and
            # so is the exception/traceback/frame chain — CPython's refcount
            # cannot reclaim cycles, only the cyclic collector can.  Until the
            # cycle is broken the graph's CUDA storages keep refcount > 0, so
            # empty_cache() cannot return their blocks.  When OOMs arrive in
            # bursts the graphs pile up faster than the generational GC happens
            # to run, reserved VRAM ratchets up and never recovers, and every
            # subsequent step OOMs regardless of size.  gc.collect() forces the
            # cycles to be broken *now*, before we ask the allocator to release
            # cached blocks.  This is the actual fix; no amount of careful
            # `del`-ing works without it.
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # The CPU copy is no longer needed after the H→D transfer.
        del raw_batch

        # Defer .item() / float() GPU syncs to logging steps — saves 2+ syncs/step
        if step % t.log_every == 0:
            loss_val = loss.item()
            loss_ema = loss_val if loss_ema is None else _EMA_ALPHA * loss_ema + (1.0 - _EMA_ALPHA) * loss_val
            grad_norm_val = float(grad_norm)
            lr_now = scheduler.get_last_lr()[0]
            amp_scale = scaler.get_scale() if scaler is not None else 1.0
            cop_val = parts["copula"].item()
            mar_val = parts["marginal"].item()
            aux_mae_val = aux_mae.item()
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
            # Also captures this exact step's own forward/loss/backward_step
            # ms (last_step_ms) alongside the window-averaged step_ms below —
            # needed to pair with step_flops (measured for this step only via
            # FlopCounterMode), since batch shapes vary per shard and a
            # window-averaged time would be the wrong denominator for a
            # single step's exact FLOP count.
            last_step_ms = dict(_prof_last_ms)  # CPU fallback; overwritten below on CUDA
            if device == "cuda" and _prof_n > 0:
                torch.cuda.synchronize()
                for name in _prof_phases:
                    elapsed = [s.elapsed_time(e) for s, e in _prof_events[name]]
                    _prof_ms[name] += sum(elapsed)
                    last_step_ms[name] = elapsed[-1] if elapsed else 0.0
                    _prof_events[name].clear()
            now = time.perf_counter()
            steps_done = max(step - _last_log_step, 1)
            step_ms = {k: v / _prof_n for k, v in _prof_ms.items()} if _prof_n else {k: 0.0 for k in _prof_ms}
            avg_T = _prof_T_sum / _prof_n if _prof_n else 0
            wall_step_ms = (now - _last_log_wall) / steps_done * 1000.0
            steps_per_sec = steps_done / max(now - _last_log_wall, 1e-9)
            _last_log_wall = now
            _last_log_step = step
            for k in _prof_ms:
                _prof_ms[k] = 0.0
            _prof_n = 0
            _prof_T_sum = 0

            # ---- Model FLOPs Utilization (MFU) ----
            # flops_per_iter is step_flops: this exact step's dispatched FLOPs
            # as measured by FlopCounterMode above (not an analytic estimate —
            # see the comment at that call site for why an analytic PaLM-style
            # count would be wrong for this table-shaped, partially-frozen
            # model). iter_time_sec pairs it with THIS SAME step's own
            # forward+loss+backward_step time (last_step_ms, from the CUDA
            # events just read out above), not the window-averaged step_ms —
            # batch shapes vary per shard (see the OOM handler above), so
            # averaging would mismatch the single-step FLOP count. t.batch_size
            # is already the per-process micro-batch (single-GPU script, no
            # DDP/FSDP). avg_T (mean seq_len this window) is only used below
            # for tokens_per_sec, a throughput metric independent of the FLOP
            # count's accuracy.
            iter_time_sec = sum(last_step_ms.values()) / 1000.0
            flops_per_iter = step_flops or 0.0
            if iter_time_sec > 0:
                actual_flops_per_sec = flops_per_iter / iter_time_sec
                tokens_per_sec = (t.batch_size * avg_T) / iter_time_sec
            else:
                actual_flops_per_sec = 0.0
                tokens_per_sec = 0.0
            mfu_pct = (
                100.0 * actual_flops_per_sec / gpu_peak_flops
                if (gpu_peak_flops and iter_time_sec > 0) else 0.0
            )

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
                    "train/aux_mae":              aux_mae_val,
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
                    "perf/mfu_pct":                mfu_pct,
                    "perf/tokens_per_sec":         tokens_per_sec,
                    "perf/iter_time_sec":          iter_time_sec,
                },
                step=step,
            )
            aux_str = f" aux_mae={aux_mae_val:.4f}" if aux_mae_weight > 0.0 else ""
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
                f"mem={mem_alloc_pct:.1f}%/{mem_reserved_pct:.1f}% (peak {mem_peak_pct:.1f}%) "
                f"mfu={mfu_pct:.1f}% tok/s={tokens_per_sec:,.0f}"
            )

        # Release this step's autograd graph before validation / checkpointing.
        # Otherwise out/Sigma/parts/loss stay bound to these loop locals until
        # the top of the *next* iteration, so the full training graph is pinned
        # on top of validate()'s own forward passes — a needless peak on a card
        # that already runs near the VRAM ceiling.  These names are not read
        # again this iteration (the log block above already consumed them).
        out = Sigma = parts = loss = aux_mae = grad_norm = batch = None

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
