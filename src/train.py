"""
train.py — Train the Copula Transformer in Y-space NLL via Sklar's theorem.

Loss:  L = Copula_NLL(z_test; Σ̂) + Marginal_NLL(y_test; TabICL log-pdf)
Σ̂ is built by ``model.build_sigma(out, cfg)`` from the model output — the
correlation parametrization (covnorm/cossim/tanhnorm/sparse_covnorm) is
selected by ``cfg.model.correlation_parametrization`` (see
correlation_factory.py).

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
from typing import Optional

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

# eval/ (regions.py, spatial-correlation probe helpers -- see
# _build_era5_val_batches below) lives at the repo root, not under src/.
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from classical_kernels import DEFAULT_FAMILIES
from data_gen import _COMPOSABLE_KERNELS, KERNEL_REGISTRY, _generate_gp_batch_raw, generate_gp_batch
from dataset import (
    CopulaDataset,
    ShardBlockSampler,
    ShardHomogeneousBatchSampler,
    collate_fn,
)
from eval.configs.regions import REGIONS as ERA5_REGIONS
from era5_live_dataset import build_era5_fixed_val_batches, build_era5_train_loader
from eval.spatial.diagnostics import bin_correlation_by_distance
from eval.spatial.sweep_core import build_era5_probe, weighted_corr, weighted_r2, weighted_rmse_bias
from live_dataset import (
    _LIVE_TABICL_FLAT_HEADROOM_GB,
    _LIVE_TABICL_WORKER_HEADROOM_GB,
    build_fixed_live_val_batches,
    build_live_train_loader,
    limited_main_process_threads,
    resolve_live_tabicl_num_workers,
)
from loss import _safe_cholesky, y_space_nll
from model import build_copula_transformer, build_sigma, low_rank_correlation
from muon import Muon
from pit import (
    DEFAULT_K_FOLDS,
    gp_analytical_posterior,
    load_tabicl,
    normalize_targets,
    resolve_pit_ckpt,
    run_pit,
)

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
    "RTX 5000 ADA": 522.2e12,     # not yet confirmed on a specific Grid5000 node -- derived
                                   # from NVIDIA's official datasheet (1044.4 TFLOPS "Tensor
                                   # Performance", footnoted as effective FP8-with-sparsity) / 2,
                                   # matching this table's RTX 6000 ADA convention (1457.0 / 2 = 728.5)
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


def _reserve_gpu_headroom_for_live_tabicl(cfg: DictConfig, t: DictConfig, device: str) -> None:
    """Cap this (main) process's own CUDA memory fraction when live-generation
    will also run TabICL inference inside separate GPU DataLoader worker
    processes on the same card (live_dataset.py's data.z_train_tabicl_mix_* /
    data.z_train_source=tabicl* path).

    Root cause this works around: each such worker holds its own CUDA
    context, entirely separate from this process's caching allocator pool.
    Live-generation batches vary P/N a lot (see the top-of-file
    expandable_segments comment), so this process's own pool can ratchet up
    to a high "reserved" watermark and never give it back -- PyTorch's
    caching allocator only returns memory to the driver via empty_cache(),
    which nothing here calls outside the OOM handler below. Reserved memory
    is invisible to *other* processes even when this process isn't actively
    using most of it, so given enough steps this process's pool can starve a
    TabICL worker's small allocation of the little free VRAM the driver has
    left -- and unlike this process's own OOMs, a worker's OOM was previously
    completely uncaught (see next(train_iter) below), taking the whole run
    down instead of costing one skipped step.

    set_per_process_memory_fraction makes this process itself OOM (into the
    already-correct, already-tested gc.collect()-before-empty_cache() handler
    below) once it would otherwise have crowded out the workers' headroom,
    instead of silently starving them. It only caps future growth -- it does
    not reclaim memory already reserved -- so this must run before any
    training-loop allocation happens.
    """
    z_train_source = str(cfg.data.get("z_train_source", "analytic"))
    mix_enabled = bool(cfg.data.get("z_train_tabicl_mix_enabled", False))
    tabicl_live_enabled = mix_enabled or z_train_source in ("tabicl", "tabicl_split")
    if not tabicl_live_enabled or device != "cuda":
        return
    # Same resolution build_live_train_loader uses below (auto-sized from
    # currently-free GPU memory when training.live_tabicl_num_workers is
    # left unset) -- keeping both call sites on one resolver means the
    # headroom reserved here always matches the worker count actually
    # spawned, instead of drifting if only one of the two were updated.
    num_workers = resolve_live_tabicl_num_workers(t, device)
    if num_workers <= 0:
        return
    headroom_gb = num_workers * _LIVE_TABICL_WORKER_HEADROOM_GB + _LIVE_TABICL_FLAT_HEADROOM_GB
    total_b = torch.cuda.get_device_properties(0).total_memory
    fraction = 1.0 - (headroom_gb * 1e9) / total_b
    # Clamp: never below 0.5 (a misconfigured huge num_workers shouldn't starve
    # this process instead), never above 0.97 (always leave the OOM handler's
    # own gc.collect()/empty_cache() cycle some margin to actually help).
    fraction = max(0.5, min(fraction, 0.97))
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    print(
        f"[train] live-generation TabICL workers ({num_workers}) run their own "
        f"CUDA context on this GPU -- capping this process's own VRAM use to "
        f"{fraction * 100:.0f}% (~{headroom_gb:.1f} GB reserved as worker "
        "headroom) so its allocator can't silently starve them; any excess is "
        "handled by the existing per-step OOM-skip path."
    )


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

    A second row, "Pred (z_tabicl)", is added when any episode carries an
    "R_pred_tabicl" key — the same model forward pass, but conditioned on
    TabICL's own K-fold PIT z_train instead of the exact GP-LOO one (see
    _build_tabicl_val_z / the do_plot block in validate()). This is the
    sim-to-real check: does the model's correlation prediction hold up when
    fed the imperfect PIT it will actually get at real-data deployment time,
    not just the closed-form oracle one it's trained on almost everywhere
    else. Episodes without that key (e.g. the kernel_fit probes appended
    below) simply leave that row blank for that column.
    """
    n_ep = len(plot_episodes)

    # (row_label, lookup) for each *estimator*: Pred is a top-level episode key.
    # The oracle is no longer a row — it is the left cell of every episode pair.
    rows: list[tuple[str, str]] = [("Pred (z_exact)", "R_pred")]
    if any("R_pred_tabicl" in ep for ep in plot_episodes):
        rows.append(("Pred (z_tabicl)", "R_pred_tabicl"))
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


def _name_seed(base_seed: int, name: str) -> int:
    """Deterministic per-name seed offset from a run-level base seed, so each
    kernel family / ERA5 region gets its own fixed-but-different probe draw
    instead of all of them sharing one seed."""
    return base_seed + (zlib.crc32(name.encode()) % 10_000)


def _macro_average(values: list[float]) -> float:
    """Unweighted mean of a metric collected across kernel families /
    regions, NaN if none were finite — used for the kernel_fit/era5_fit
    mean_* cross-run-comparable scalars in validate()."""
    return float(np.mean(values)) if values else float("nan")


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

    P_min/P_max/N_min/N_max are pinned to baselines.probe_* (NOT read from
    cfg.data.*): this run's own gp_tasks.yaml can change its context/test-size
    ranges (it has, repeatedly) without silently reshaping the probe episodes
    underneath kernel_fit/<family> — otherwise two runs with different
    data.P_min/P_max would each get a "frozen" probe that's fixed-per-run but
    different-across-runs, defeating the entire point of a cross-run-
    comparable benchmark.

    return_kernel_metadata=True so validate() can also run
    pit.gp_analytical_posterior per episode (oracle_diag/kernel_fit/<family>/
    gap_nll) — the same true Bayes-optimal ceiling used by the top-level
    posterior_probe, but scored per kernel family instead of on the run's
    own composite mixture.
    """
    bcfg = cfg.get("baselines", {}) or {}
    families = list(bcfg.get("kernels") or DEFAULT_FAMILIES)
    n_episodes = int(bcfg.get("synth_n_episodes", 64))
    base_seed = int(bcfg.get("synth_seed", 20260718))
    probe_P_min = int(bcfg.get("probe_P_min", 32))
    probe_P_max = int(bcfg.get("probe_P_max", 512))
    probe_N_min = int(bcfg.get("probe_N_min", 8))
    probe_N_max = int(bcfg.get("probe_N_max", 1024))

    batches: dict[str, dict] = {}
    for family in families:
        if family not in KERNEL_REGISTRY:
            continue  # not standalone-generatable (e.g. an unregistered composite)
        family_seed = _name_seed(base_seed, family)
        synth_cfg = OmegaConf.merge(
            cfg,
            OmegaConf.create({
                "seed": family_seed,
                "data": {
                    "kernel": family,
                    "systematic_composition": False,
                    "P_min": probe_P_min,
                    "P_max": probe_P_max,
                    "N_min": probe_N_min,
                    "N_max": probe_N_max,
                },
            }),
        )
        episodes = generate_gp_batch(synth_cfg, n_episodes, device="cpu", return_kernel_metadata=True)
        batch = collate_fn(episodes)
        batches[family] = {"episodes": episodes, "batch": {k: v.to(device) for k, v in batch.items()}}
    return batches


def _build_posterior_probe_batches(cfg: DictConfig, device: str) -> dict:
    """Fallback probe set for the true-Bayes-optimal-ceiling validation
    metrics (see validate()'s oracle_diag/gap_nll / oracle_diag/corr_pearson)
    when val_loader itself can't supply the needed kernel metadata.

    data_gen.py's own oracle_mode="prior" R_star/Sigma_star (what every other
    "oracle" quantity in this file is scored against) is context-blind by
    construction — see data_gen.py:3359-3382 — so it is NOT the Bayes-optimal
    lower bound achievable given (x_train, y_train), only a weaker,
    beatable one. pit.gp_analytical_posterior computes the real one (Schur
    complement, float64, PSD-repaired), but it only runs one episode at a
    time and needs return_kernel_metadata=True episodes (kernel name +
    hyperparameters). The live-generation val_loader (train.py's
    build_fixed_live_val_batches) now requests exactly that, so validate()
    scores oracle_diag/gap_nll directly against val_loader's own episodes in
    that (default) case — see validate()'s val_episodes_meta parameter. This
    function only still runs as the fallback for the two cases where
    val_loader can't carry that metadata: on-disk training
    (training.live_generation=false, CopulaDataset's shards were never
    written with it) and the real-ERA5 live_source (no GP kernel to
    reconstruct a posterior from at all). This builds a fixed set of such
    episodes once at startup — unlike _build_synthetic_kernel_batches,
    cfg.data's own kernel mixture (systematic_composition etc.) is left
    untouched, since the point here is to measure the ceiling on the SAME
    kind of episode the model actually trains on, not an isolated classical
    kernel family.

    baselines.posterior_probe_n_episodes defaults (conf/config.yaml) to
    ${training.val_episodes} — same episode count as val_loader, drawn fresh
    from the same cfg.data distribution val_loader itself samples from, so
    oracle_diag/gap_nll is a same-size, same-distribution stand-in for "gap
    on the full validation set" in these fallback cases (not literally the
    same episodes as val_loader). Override the config key directly for a
    different size (e.g. smaller, for faster iteration).

    Returns {"episodes": [...] (CPU dicts, consumed by gp_analytical_posterior
    one at a time), "batch": {...} (device-resident, collated/padded,
    consumed by the model forward pass — same episodes, same order)}.
    """
    bcfg = cfg.get("baselines", {}) or {}
    n_episodes = int(bcfg.get("posterior_probe_n_episodes", 64))
    base_seed = int(bcfg.get("synth_seed", 20260718)) + 2  # +1 is _compute_tabicl_z_train_gap's
    probe_cfg = OmegaConf.merge(cfg, OmegaConf.create({"seed": base_seed}))
    episodes = generate_gp_batch(probe_cfg, n_episodes, device="cpu", return_kernel_metadata=True)
    batch = collate_fn(episodes)
    return {"episodes": episodes, "batch": {k: v.to(device) for k, v in batch.items()}}


def _build_era5_val_batches(cfg: DictConfig, tabicl_marginal, device: str) -> dict[str, dict]:
    """Fixed per-region real-ERA5 probes for the ``era5_fit/<region>``
    validation metrics (see validate()) — the real-data analogue of
    _build_synthetic_kernel_batches above.

    Unlike a kernel_fit/<family> synthetic probe, real ERA5 has no known GP
    oracle (no Sigma_star/R_star), so there is no NLL-gap metric to compute
    here. Instead, eval.spatial.sweep_core.build_era5_probe freezes a ground-
    truth correlation-vs-distance curve (empirical Pearson correlation, the
    same convention eval/runners/spatial_correlation_eval.py's real-mode
    sweep uses) plus a fixed real in-context sample (context coords/values,
    PIT'd once against `tabicl_marginal`) for a handful of ERA5 days per
    region. validate() re-runs only the CURRENT model's forward pass on this
    frozen input every call and scores the resulting correlogram against the
    frozen curve — the ERA5 fetch + PIT cost is paid once, here, not on the
    training loop's hot path.

    `tabicl_marginal` may be None (no PIT checkpoint configured): falls back
    to naive per-context standardization, same as
    eval.spatial.diagnostics.extract_model_context_correlation. In that case
    there is no real predictive density to score a Y-space NLL against, so
    the returned probe carries no "nll_test_z"/"nll_test_log_pdf" and
    validate()'s era5_fit/<region>/y_nll_total block is skipped for every
    region.

    When `tabicl_marginal` IS given, this also runs TabICL's own PIT
    (_tabicl_pit_batch) once on the probe's held-out (never-in-context)
    points (build_era5_probe's nll_test_idx/context_values_per_day/
    nll_test_values_per_day) — the same real-marginal z_test/log_pdf_test
    val/y_nll_total is scored against for the general val set, just frozen
    here alongside z_train since tabicl_marginal doesn't change during
    training either.
    """
    ecfg = cfg.get("baselines", {}) or {}
    region_names = list(ecfg.get("era5_regions") or list(ERA5_REGIONS.keys()))
    grid_size = int(ecfg.get("era5_grid_size", 10))
    n_days_fetch = int(ecfg.get("era5_n_days_fetch", 60))
    n_days_probe = int(ecfg.get("era5_n_days_probe", 3))
    n_context = int(ecfg.get("era5_n_context", 30))
    n_bins = int(ecfg.get("era5_n_bins", 12))
    base_seed = int(ecfg.get("era5_seed", 20260818))
    pit_k_folds = int(cfg.tabicl.get("pit_k_folds", DEFAULT_K_FOLDS))

    batches: dict[str, dict] = {}
    for region_name in region_names:
        if region_name not in ERA5_REGIONS:
            continue  # not a registered eval/configs/regions.py entry
        region_seed = _name_seed(base_seed, region_name)
        probe = build_era5_probe(
            region_name, grid_size, n_days_fetch, n_days_probe, n_context, n_bins,
            tabicl_marginal, device, seed=region_seed,
        )
        n_days_p = probe["z_train_per_day"].shape[0]
        x_train = torch.as_tensor(probe["x_train_norm"], dtype=torch.float32, device=device)
        x_test = torch.as_tensor(probe["x_test_norm"], dtype=torch.float32, device=device)
        z_train = torch.as_tensor(probe["z_train_per_day"], dtype=torch.float32, device=device)
        model_batch = {
            "x_train": x_train.unsqueeze(0).expand(n_days_p, -1, -1).contiguous(),
            "x_test": x_test.unsqueeze(0).expand(n_days_p, -1, -1).contiguous(),
            "z_train": z_train,
            "test_mask": torch.ones(n_days_p, probe["D"], dtype=torch.bool, device=device),
        }
        region_batch = {
            "batch": model_batch,
            "dist": probe["dist"],
            "bin_edges": probe["bin_edges"],
            "pair_counts": probe["pair_counts"],
            "rho_emp": probe["rho_emp"],
        }
        if tabicl_marginal is not None:
            n_nll = probe["x_nll_test_norm"].shape[0]
            x_nll_test = torch.as_tensor(probe["x_nll_test_norm"], dtype=torch.float32, device=device)
            nll_pit_batch = {
                "x_train": model_batch["x_train"],
                "y_train": torch.as_tensor(probe["context_values_per_day"], dtype=torch.float32, device=device),
                "train_mask": torch.ones(n_days_p, probe["n_context"], dtype=torch.bool, device=device),
                "x_test": x_nll_test.unsqueeze(0).expand(n_days_p, -1, -1).contiguous(),
                "y_test": torch.as_tensor(probe["nll_test_values_per_day"], dtype=torch.float32, device=device),
                "test_mask": torch.ones(n_days_p, n_nll, dtype=torch.bool, device=device),
            }
            nll_pit = _tabicl_pit_batch(nll_pit_batch, tabicl_marginal, pit_k_folds, device)
            region_batch["nll_test_idx"] = probe["nll_test_idx"]
            region_batch["nll_test_z"] = nll_pit["z_test"].to(device)
            region_batch["nll_test_log_pdf"] = nll_pit["log_pdf_test"].to(device)
        batches[region_name] = region_batch
    return batches


def _update_adaptive_kernel_weights(
    prev_weights: torch.Tensor, metrics: dict, lr: float, floor: float,
    exclude: Optional[set] = None, signal: str = "oracle",
) -> torch.Tensor:
    """DoReMi/GroupDRO-style exponentiated-gradient update of per-kernel-family
    live-generation sampling weights (see training.adaptive_kernel_sampling),
    ordered to match data_gen._COMPOSABLE_KERNELS.

    Signal is the per-family excess loss (regret) already computed by
    validate()'s kernel_fit/<family> probes, selected by `signal`
    (training.adaptive_kernel_signal):

      "oracle" (default) — oracle_diag/kernel_fit/<family>/gap_nll =
        total_nll - oracle_posterior_total_nll, both scored against the
        exact analytic-GP PIT (NLL is lower-is-better, and
        oracle_posterior_total_nll is pit.gp_analytical_posterior's true
        Schur-complement Bayes-optimal ceiling for that family's probe
        episodes — see validate()'s kernel_fit loop — so this is typically
        >=0, bigger when the model is further from the true posterior on
        that family = more room to improve).
      "tabicl" — kernel_fit/<family>/gap_nll_tabicl instead: the identical
        gap construction, but total_nll is scored against TabICL's own
        frozen K-fold PIT (a real, imperfect marginal) rather than the
        exact analytic one — see _build_tabicl_kernel_fit_z /
        validate()'s TabICL-conditioned kernel_fit block. Only present
        when a PIT checkpoint is configured (pit.py::resolve_pit_ckpt);
        falls back per-family to the oracle gap wherever it's missing (no
        PIT checkpoint at all, or that family's probe had no valid
        episodes), rather than silently zeroing the signal for every
        family the moment the run has no PIT checkpoint.

    Previously used copula_nll - oracle_copula_nll against data_gen.py's
    context-blind oracle_mode="prior" R_star, a weaker, beatable bound;
    gap_nll is in Y-space total-NLL units either way, so it stays a valid
    regret signal regardless of which marginal produced z_test, unlike a
    z-space-only copula gap. Families with no probe (metrics missing the key
    — e.g. not in cfg.baselines.kernels, or gp_analytical_posterior raised on
    every episode) get gap=0, i.e. no update pressure, only the floor's
    implicit pull toward uniform.

    exclude (optional): family names to hold out of the gap-driven update
    entirely (gap forced to 0), regardless of whether a kernel_fit probe
    exists for them. Meant for cfg.data.composite_exclude_kernels — those
    families are never in _sample_kernel_chain_structure's sampling pool
    (data_gen.py::_weights_for_pool already renormalizes over the
    post-exclude pool, so their tensor entry is inert either way), so
    driving their weight off model performance is just noise: it moves the
    number without moving anything the number controls.

    w' = prev_weights * exp(lr * gap), renormalized, then blended with a
    uniform floor: w = (1 - floor) * w' + floor * uniform — prevents any
    family's weight collapsing toward 0 and being effectively dropped from
    the curriculum. Pure function: caller is responsible for writing the
    result into the shared-memory tensor DataLoader workers read from
    (`kernel_weights_tensor.copy_(...)`, never rebind).
    """
    exclude = exclude or set()
    n = len(_COMPOSABLE_KERNELS)
    gaps = torch.zeros(n, dtype=torch.float32)
    for i, family in enumerate(_COMPOSABLE_KERNELS):
        if family in exclude:
            continue
        gap_nll = metrics.get(f"oracle_diag/kernel_fit/{family}/gap_nll")
        if signal == "tabicl":
            gap_nll_tabicl = metrics.get(f"kernel_fit/{family}/gap_nll_tabicl")
            if gap_nll_tabicl is not None:
                gap_nll = gap_nll_tabicl
        if gap_nll is not None and math.isfinite(gap_nll):
            gaps[i] = gap_nll
    # Clamp the exponent, not the gap itself, so a single wild probe can't
    # overflow exp() into inf and NaN out every family's weight via the
    # shared normalization below.
    exponent = torch.clamp(lr * gaps, min=-30.0, max=30.0)
    raw = prev_weights.float() * torch.exp(exponent)
    total = raw.sum()
    uniform = torch.full((n,), 1.0 / n, dtype=torch.float32)
    if not torch.isfinite(total) or total <= 0:
        raw = uniform.clone()
    else:
        raw = raw / total
    return (1.0 - floor) * raw + floor * uniform


@torch.no_grad()
def _compute_tabicl_z_train_gap(
    cfg: DictConfig, tabicl_marginal: nn.Module, k_folds: int, device: str = "cpu",
) -> dict[str, float]:
    """Measure, once per data_gen._COMPOSABLE_KERNELS family, how far the
    frozen TabICL marginal's own K-fold PIT diverges from the exact
    analytic GP-LOO z_train on the SAME episodes -- the signal
    data.z_train_tabicl_mix_* (conf/data/gp_tasks.yaml) uses to set each
    family's live-generation mixing fraction (see _tabicl_gap_to_mix_frac
    below).

    Calls _generate_gp_batch_raw directly (not the public generate_gp_batch
    top-up wrapper) TWICE per family with the identical cfg.seed -- once
    with tabicl_model=None (exact analytic z_train), once with
    tabicl_model=tabicl_marginal (real TabICL K-fold PIT) -- so both calls
    draw byte-identical kernel/hyperparameters/x/y (see
    _generate_gp_batch_raw's seeding-contract docstring) and differ ONLY in
    which z_train ends up in the returned episode dicts. The discard mask
    that determines which episodes survive is itself computed from the
    exact analytic residual before either call's z_train override runs
    (see _generate_gp_batch_raw's z_train-override comment), so it's
    identical across both calls too -- episode i in one list is the same
    episode as index i in the other, safe to pair up directly without
    needing generate_gp_batch's reseeding top-up loop (which would risk
    the two calls discarding different subsets on a retry round).

    This is a property of TabICL's frozen marginal-quantile approximation
    for that kernel family, not of the copula model being trained -- unlike
    train.py::_update_adaptive_kernel_weights's regret signal, which chases
    a moving target as the model trains, this doesn't move with the model,
    so by default it's computed once, up front (train.py's startup
    sequence, alongside _build_tabicl_val_z, before `tabicl_marginal` is
    freed) and not re-measured again. data.z_train_tabicl_mix_adaptive
    opts into periodic re-measurement anyway (see _refresh_tabicl_mix_weights,
    called on the training.save_every cadence), e.g. to track drift in
    TabICL's own approximation quality if the checkpoint backing it changes
    meaning over a long run -- either way this is never called from inside
    validate() itself, since it needs its own fresh episode generation, not
    validate()'s fixed probe batches.

    Uses cfg.baselines.synth_n_episodes/synth_seed/probe_P_*/probe_N_* (the
    same fixed-probe-set knobs _build_synthetic_kernel_batches uses) offset
    by +1 so this draws an independent episode stream from that function's
    own kernel_fit/<family> probes, rather than silently reusing the exact
    same seed for a different purpose.

    device: must match wherever tabicl_marginal itself lives (train.py's
    startup sequence passes its own `device`, typically "cuda") -- BOTH
    paired calls below run on this same device, not just the tabicl one.
    torch's CPU and CUDA generators are separate RNG streams that do not
    produce identical draws from the same torch.manual_seed/cuda.manual_seed
    even though _seed_everything seeds both every call (different underlying
    algorithms) -- running the analytic call on "cpu" while tabicl_marginal
    lives on "cuda" would silently break the byte-identical-pairing
    guarantee above (and, separately, crash outright once the override
    branch tries to mix cuda-resident TabICL weights with cpu-resident
    x_norm_train).

    Returns {family: mean |z_tabicl - z_analytic|} (CPU floats) for every
    family that produced at least one valid paired episode. Two independent
    standard normals have E|Z1-Z2| = 2/sqrt(pi) ~= 1.13, so this gap is
    typically O(0-1) in these units: near 0 means TabICL's PIT tracks the
    analytic residual closely for that family, growing toward ~1.1+ means
    it's close to uninformative.
    """
    bcfg = cfg.get("baselines", {}) or {}
    n_episodes = int(bcfg.get("synth_n_episodes", 64))
    base_seed = int(bcfg.get("synth_seed", 20260718)) + 1
    probe_P_min = int(bcfg.get("probe_P_min", 32))
    probe_P_max = int(bcfg.get("probe_P_max", 512))
    probe_N_min = int(bcfg.get("probe_N_min", 8))
    probe_N_max = int(bcfg.get("probe_N_max", 1024))

    gaps: dict[str, float] = {}
    with limited_main_process_threads():
        # This function is a main-process caller of _generate_gp_batch_raw
        # (two calls per _COMPOSABLE_KERNELS family), never a DataLoader
        # worker -- see limited_main_process_threads' docstring for why that
        # needs an explicit thread cap here (OS-default thread count causes
        # ~2-3x slowdown on generate_gp_batch's CPU-bound tensor ops).
        for family in _COMPOSABLE_KERNELS:
            family_seed = _name_seed(base_seed, family)
            probe_cfg = OmegaConf.merge(
                cfg,
                OmegaConf.create({
                    "seed": family_seed,
                    "data": {
                        "kernel": family,
                        "systematic_composition": False,
                        "P_min": probe_P_min, "P_max": probe_P_max,
                        "N_min": probe_N_min, "N_max": probe_N_max,
                    },
                }),
            )
            analytic_eps = _generate_gp_batch_raw(probe_cfg, n_episodes, device=device)
            probe_cfg.seed = family_seed  # _generate_gp_batch_raw mutates nothing, but stay explicit
            tabicl_eps = _generate_gp_batch_raw(
                probe_cfg, n_episodes, device=device,
                tabicl_model=tabicl_marginal, tabicl_k_folds=k_folds,
            )
            n = min(len(analytic_eps), len(tabicl_eps))
            if n == 0:
                continue
            diffs = [
                (tabicl_eps[i]["z_train"] - analytic_eps[i]["z_train"]).abs().mean().item()
                for i in range(n)
            ]
            gaps[family] = float(sum(diffs) / len(diffs))
    return gaps


def _tabicl_gap_to_mix_frac(
    gaps: dict[str, float], floor_frac: float, max_frac: float,
) -> torch.Tensor:
    """Map _compute_tabicl_z_train_gap's per-family gap to a
    `_COMPOSABLE_KERNELS`-ordered live-generation mixing-fraction tensor
    (see data.z_train_tabicl_mix_* in conf/data/gp_tasks.yaml).

    Min-max normalizes gaps across the families that were actually measured
    (missing/degenerate families -- e.g. a family with 0 valid probe
    episodes -- fall back to floor_frac, the same anti-starvation treatment
    _update_adaptive_kernel_weights's floor gives an unmeasured family), then
    linearly interpolates each family's normalized gap into
    [floor_frac, max_frac]. If every measured gap is equal (or only one
    family was measured), normalization is undefined -- every family gets
    floor_frac instead, since there's no relative signal to differentiate on.
    """
    n = len(_COMPOSABLE_KERNELS)
    frac = torch.full((n,), floor_frac, dtype=torch.float32)
    if len(gaps) < 2:
        return frac
    values = list(gaps.values())
    lo, hi = min(values), max(values)
    spread = hi - lo
    if spread <= 1e-12:
        return frac
    for i, family in enumerate(_COMPOSABLE_KERNELS):
        if family not in gaps:
            continue
        normalized = (gaps[family] - lo) / spread
        frac[i] = floor_frac + (max_frac - floor_frac) * normalized
    return frac


def _refresh_tabicl_mix_weights(
    cfg: DictConfig, pit_ckpt: str, tabicl_mix_weights: torch.Tensor, device: str,
) -> tuple[dict[str, float], torch.Tensor]:
    """data.z_train_tabicl_mix_adaptive's periodic analogue of the one-shot
    startup measurement above: reloads the frozen TabICL marginal, re-runs
    _compute_tabicl_z_train_gap / _tabicl_gap_to_mix_frac, and .copy_()'s the
    result into tabicl_mix_weights in place (same shared-memory-tensor
    convention as _update_adaptive_kernel_weights's own in-place update --
    rebinding the name would leave LiveGPDataset workers pointed at the old
    tensor).

    Unlike _update_adaptive_kernel_weights, which reuses metrics validate()
    already computed, there's no cheap reusable signal here: measuring the
    gap needs a live TabICL forward pass, so this loads tabicl_marginal fresh
    and frees it again around the measurement rather than keeping a second
    frozen TabICL resident for the whole run (this repo runs close to the
    VRAM ceiling -- see the comment above the training loop's autograd-graph
    release). That reload + ~1k-episode remeasurement is why callers gate
    this on training.save_every (already 10x rarer than training.val_every
    by default) rather than every validate() call.

    floor_frac == max_frac short-circuit: see the matching comment at this
    function's startup-time sibling call site in main() -- when the two
    fracs are equal, _tabicl_gap_to_mix_frac's interpolation collapses to
    floor_frac for every family regardless of the measured gap, so the
    reload + remeasurement below would be pure wasted work every
    training.save_every steps for the life of the run.
    """
    floor_frac = float(cfg.data.get("z_train_tabicl_mix_floor_frac", 0.05))
    max_frac = float(cfg.data.get("z_train_tabicl_mix_max_frac", 0.35))
    if math.isclose(floor_frac, max_frac, abs_tol=1e-12):
        new_mix_frac = torch.full(
            (len(_COMPOSABLE_KERNELS),), floor_frac, dtype=torch.float32
        )
        tabicl_mix_weights.copy_(new_mix_frac)
        return {}, new_mix_frac
    tabicl_marginal = load_tabicl(pit_ckpt, device)
    pit_k_folds = int(cfg.tabicl.get("pit_k_folds", DEFAULT_K_FOLDS))
    z_gap = _compute_tabicl_z_train_gap(cfg, tabicl_marginal, pit_k_folds, device)
    new_mix_frac = _tabicl_gap_to_mix_frac(z_gap, floor_frac, max_frac)
    tabicl_mix_weights.copy_(new_mix_frac)
    del tabicl_marginal
    if device == "cuda":
        # gc.collect() before empty_cache() (see this repo's OOM-handler
        # gotcha): del alone doesn't free CUDA storage until any reference
        # cycles in the eval-mode forward graph are collected.
        gc.collect()
        torch.cuda.empty_cache()
    return z_gap, new_mix_frac


@torch.no_grad()
def _tabicl_pit_batch(
    batch: dict, tabicl_marginal: nn.Module, k_folds: int, device: str,
) -> dict[str, torch.Tensor]:
    """Run TabICL's own K-fold PIT (pit.py::run_pit) once per episode in a
    single already-collated batch, returning the same real (non-oracle)
    z_train/z_test/log_pdf_test triple _build_tabicl_val_z caches for the
    main val loader. Factored out of that function so kernel_fit/<family>'s
    fixed probe batches (_build_synthetic_kernel_batches) can reuse the
    identical PIT/scaling logic instead of duplicating it — see
    _build_tabicl_val_z and _build_tabicl_kernel_fit_z below, the two
    callers.

    `batch` must carry x_train/y_train/train_mask/x_test/y_test/test_mask
    (collate_fn's schema); may live on any device, moved to `device` here.
    Returns CPU tensors {"z_train": (B, P_max), "z_test": (B, N_max),
    "log_pdf_test": (B, N_max)}, zero-padded outside each episode's true
    train/test length (matching train_mask/test_mask).

    y_train/y_test are z-scored via pit.normalize_targets (y_test scaled
    with y_train's own mean/std, never its own — see that function's
    docstring) before reaching the raw TabICL module: run_pit does no
    target scaling of its own (unlike tabicl.TabICLRegressor.fit(), which
    fits a fresh StandardScaler before ever calling this same underlying
    model). Every other run_pit call site in the repo
    (inference/copula_inference.py::loo_pit,
    eval_checkpoint.py::_tabicl_pit) goes through the same helper, so this
    conditioning input is computed identically everywhere. Episode y's
    scale is not fixed — outputscale is drawn from a GammaPrior
    (data_gen.py's generative process) — so an unscaled call risks
    saturating the pretrained quantile head's CDF into its extreme tail for
    every point alike on high-outputscale episodes, collapsing the PIT
    residuals' spread instead of reflecting the true per-point rank.

    log_pdf_test comes back in that same normalize_targets-scaled space —
    a Jacobian correction (log p_raw(y) = log p_scaled(y_scaled) -
    log(std), per normalize_targets' own docstring) is applied here so
    every caller of this cache's log_pdf_test gets raw-nats units, matching
    the oracle's log_pdf_test (data_gen.py's z_test/log_pdf_test are always
    raw-nats — see y_space_nll's Args).
    """
    x_train = batch["x_train"].to(device)
    y_train = batch["y_train"].to(device)
    x_test = batch["x_test"].to(device)
    y_test = batch["y_test"].to(device)
    train_mask = batch["train_mask"].to(device)
    test_mask = batch["test_mask"].to(device)
    B, P_max = y_train.shape
    N_max = y_test.shape[1]
    z_tabicl = torch.zeros(B, P_max, device=device)
    z_test_tabicl = torch.zeros(B, N_max, device=device)
    log_pdf_test_tabicl = torch.zeros(B, N_max, device=device)
    for b in range(B):
        n = int(train_mask[b].sum())
        n_te = int(test_mask[b].sum())
        if n < 2 or n_te < 1:
            continue  # run_pit's fold split needs >=2 context points
        X_b = x_train[b, :n]
        X_te = x_test[b, :n_te]
        y_b_scaled, y_te_scaled, _, std = normalize_targets(y_train[b, :n], y_test[b, :n_te])
        Y_b = y_b_scaled.unsqueeze(-1)
        Y_te = y_te_scaled.unsqueeze(-1)
        pit_out = run_pit(tabicl_marginal, X_b, Y_b, X_te, Y_te, k_folds=k_folds)
        z_tabicl[b, :n] = pit_out["z_train"].squeeze(-1)
        z_test_tabicl[b, :n_te] = pit_out["z_test"].squeeze(-1)
        log_pdf_test_tabicl[b, :n_te] = pit_out["log_pdf_test"].squeeze(-1) - std.log()
    return {
        "z_train": z_tabicl.cpu(),
        "z_test": z_test_tabicl.cpu(),
        "log_pdf_test": log_pdf_test_tabicl.cpu(),
    }


@torch.no_grad()
def _build_tabicl_val_z(
    val_loader, tabicl_marginal: nn.Module, k_folds: int, device: str,
) -> dict[int, dict[str, torch.Tensor]]:
    """Precompute the frozen TabICL marginal's K-fold PIT once per val_loader
    episode (see _tabicl_pit_batch / pit.py::run_pit), instead of re-running
    it every validate() call.

    tabicl_marginal never changes during training and val_loader itself is
    fixed across every call (live mode: build_fixed_live_val_batches
    generates it once up front; disk mode: val_dataset + shuffle=False
    iterate in the same order every time) — so this is the same value every
    validate() call and only needs computing once, here, before the training
    loop starts.

    Covers every batch in val_loader (not just the first _PLOT_COLLECT_BATCHES
    used for plotting) — val/y_nll_total below is meant to be the "real
    deployment" headline number, so it needs the same episode count as the
    rest of val/'s metrics (training.val_episodes), not a smaller plot-sized
    sub-sample. This is a one-time startup cost (not per-validate() call), so
    the extra episodes here are cheap relative to re-running it every
    validate() call would be.

    Queries each episode's REAL x_test/y_test (not a throwaway probe), so
    run_pit's test-side forward pass also returns a genuine TabICL marginal
    at the test points (z_test, log_pdf_test) — the missing ingredient for
    scoring the model's own total (marginal+copula) Y-space NLL under a
    real, non-oracle marginal (validate()'s val/y_nll_total), the same way
    eval_checkpoint.py::_tabicl_pit does for --z_train_source=tabicl. z_train
    alone still drives the sim-to-real correlation check (validate()'s
    do_plot block / corr_*_tabicl_z).

    Returns {batch_idx: {"z_train": (B, P_max), "z_test": (B, N_max),
    "log_pdf_test": (B, N_max)}}, CPU, zero-padded outside each episode's
    true train/test length (matching train_mask/test_mask) — moved to
    device and sliced per-episode inside validate().
    """
    cache: dict[int, dict[str, torch.Tensor]] = {}
    for batch_idx, batch in enumerate(val_loader):
        cache[batch_idx] = _tabicl_pit_batch(batch, tabicl_marginal, k_folds, device)
    return cache


@torch.no_grad()
def _build_tabicl_kernel_fit_z(
    synth_kernel_batches: dict, tabicl_marginal: nn.Module, k_folds: int, device: str,
) -> dict[str, dict[str, torch.Tensor]]:
    """Per-kernel-family analogue of _build_tabicl_val_z: TabICL's own
    K-fold PIT on each kernel_fit/<family> fixed probe set
    (_build_synthetic_kernel_batches), computed once at startup alongside
    it, on the SAME probe episodes oracle_diag/kernel_fit/<family>/total_nll
    scores against the exact analytic PIT.

    Feeds validate()'s kernel_fit/<family>/total_nll_tabicl and
    gap_nll_tabicl — the "how does this family perform once a real,
    imperfect (TabICL) marginal replaces the oracle one" numbers, the
    alternate training.adaptive_kernel_signal="tabicl" curriculum signal
    (see _update_adaptive_kernel_weights) exists to chase.

    Returns {family: {"z_train": (B, P_max), "z_test": (B, N_max),
    "log_pdf_test": (B, N_max)}}, same shapes/padding as
    _build_tabicl_val_z's per-batch entries.
    """
    return {
        family: _tabicl_pit_batch(probe["batch"], tabicl_marginal, k_folds, device)
        for family, probe in synth_kernel_batches.items()
    }


def cosine_lr_lambda(step: int, warmup: int, total: int, lr_min_frac: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    # Clamp progress to [0, 1]: with schedule-preserving resume (see
    # load_checkpoint/train), `step` can now start above 0 and, if a resumed
    # run's `training.steps` is set lower than the step it resumes from,
    # exceed `total` — without clamping, cos(pi * progress) would swing back
    # upward past progress=1 instead of holding at the min-LR floor.
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
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
    tabicl_val_z: dict | None = None,
    tabicl_kernel_fit_z: dict | None = None,
    era5_val_batches: dict | None = None,
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
    all_off_pred: list[np.ndarray] = []
    all_off_ora: list[np.ndarray] = []
    all_off_pred_tabicl: list[np.ndarray] = []
    all_off_ora_tabicl: list[np.ndarray] = []
    all_tabicl_marginal_total: list[float] = []
    all_tabicl_marginal_marginal: list[float] = []
    all_tabicl_marginal_copula: list[float] = []
    plot_episodes: list[dict] = []

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

    for batch_idx, batch in enumerate(val_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch)
        Sigma = build_sigma(out, cfg, jitter=jitter, test_mask=batch["test_mask"])

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
                Sigma, batch["z_test"].float(), batch["log_pdf_test"].float(), batch["test_mask"]
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
            z_f = batch["z_test"].float()
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

        # ---- TabICL-marginal real (non-oracle) NLL scoring + plot collection ----
        # Runs on EVERY val_loader batch every val_every step (not just
        # do_plot steps): this is what feeds val/y_nll_total below, which is
        # meant to be sized like the rest of val/'s metrics (training.
        # val_episodes), not a small plot-sized sub-sample — tabicl_val_z now
        # has an entry for every batch (see _build_tabicl_val_z). The
        # R_star/oracle-only plotting pieces (all_off_pred/all_off_ora,
        # corr_mse_tabicl_z/sim2real_gap, plot_episodes, R_pred_tabicl
        # overlay) only ever run on do_plot steps (sparser cadence), but
        # otherwise now cover the full val_loader too — no longer capped at
        # _PLOT_COLLECT_BATCHES batches, which was a small (~80-episode)
        # sub-sample for no reason tied to the plots themselves (the hexbin
        # density plot and corr_mse_tabicl_z/sim2real_gap's MSE comparison
        # are both fine, and more accurate, over the full val set; only
        # plot_episodes' individual grid subplots stay small, via their own
        # _MAX_PLOT_EPISODES cap below).
        B = Sigma.shape[0]
        z_cache_b = tabicl_val_z.get(batch_idx) if tabicl_val_z else None
        collect_plot = do_plot
        for b in range(B):
            n = int(batch["test_mask"][b].sum())
            if n < 2:
                continue

            ep_dict = None
            ri = ci = None
            R_ora_b = None
            if collect_plot:
                R_pred_b = Sigma[b, :n, :n].float().cpu().numpy()
                R_ora_b = batch["R_star"][b, :n, :n].float().cpu().numpy()
                ri, ci = np.triu_indices(n, k=1)
                all_off_pred.append(R_pred_b[ri, ci])
                all_off_ora.append(R_ora_b[ri, ci])
                if len(plot_episodes) < _MAX_PLOT_EPISODES:
                    ep_dict = {
                        "R_pred": R_pred_b,
                        "R_ora": R_ora_b,
                        "label": f"ep{batch_idx * B + b}\nN={n}",
                    }

            # Sim-to-real check: re-run the model on this SAME episode
            # (same x_train/x_test) but conditioned on TabICL's own
            # K-fold PIT z_train (precomputed once by
            # _build_tabicl_val_z) instead of the exact GP-LOO one.
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

                    if collect_plot and ep_dict is not None:
                        R_pred_tabicl_b = Sigma_tabicl[0].float().cpu().numpy()
                        ep_dict["R_pred_tabicl"] = R_pred_tabicl_b
                        all_off_pred_tabicl.append(R_pred_tabicl_b[ri, ci])
                        all_off_ora_tabicl.append(R_ora_b[ri, ci])

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

            if ep_dict is not None:
                plot_episodes.append(ep_dict)

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
    if off_p_post:
        cq_p = _corr_quality(np.concatenate(off_p_post), np.concatenate(off_o_post))
        metrics["oracle_diag/corr_pearson"] = cq_p["pearson"]
        metrics["oracle_diag/corr_mae"] = cq_p["mae"]

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

    # Real-ERA5 spatial-correlation probe per region (see
    # _build_era5_val_batches / eval/spatial/sweep_core.py::build_era5_probe).
    # There is no GP oracle for real data, so — unlike kernel_fit/<family>'s
    # NLL gap against Sigma_star — this scores the CURRENT model's
    # context-conditioned correlogram against the region's frozen EMPIRICAL
    # Pearson correlation curve (rho_emp), using the same weighted
    # shape_corr/rmse/bias/model_r2 convention
    # eval/runners/spatial_correlation_eval.py's real-mode sweep reports.
    region_shape_corr: list[float] = []
    region_model_r2: list[float] = []
    region_y_nll_total: list[float] = []
    region_y_nll_marginal: list[float] = []
    region_y_nll_copula: list[float] = []
    for region, probe in (era5_val_batches or {}).items():
        out_e = model(probe["batch"])
        Sigma_e = build_sigma(out_e, cfg, jitter=jitter, test_mask=probe["batch"]["test_mask"])
        # Averaging the predicted correlation matrix over the probe's few
        # fixed days before binning is equivalent to averaging the binned
        # curve over days (binning is a per-day-identical linear reduction
        # over (i, j) pairs, since `dist`/bin_edges don't depend on the day)
        # — cheaper than bin_correlation_by_distance once per day.
        R_mean = Sigma_e.float().mean(dim=0).detach().cpu().numpy()
        rho_context = bin_correlation_by_distance(R_mean, probe["dist"], probe["bin_edges"])
        pair_counts, rho_emp = probe["pair_counts"], probe["rho_emp"]
        shape_corr = weighted_corr(rho_context, rho_emp, pair_counts)
        rmse, bias = weighted_rmse_bias(rho_context, rho_emp, pair_counts)
        model_r2 = weighted_r2(rho_context, rho_emp, pair_counts)
        metrics[f"era5_fit/{region}/shape_corr"] = shape_corr
        metrics[f"era5_fit/{region}/rmse"] = rmse
        metrics[f"era5_fit/{region}/bias"] = bias
        metrics[f"era5_fit/{region}/model_r2"] = model_r2
        if not math.isnan(shape_corr):
            region_shape_corr.append(shape_corr)
        if not math.isnan(model_r2):
            region_model_r2.append(model_r2)

        # Real, non-oracle Y-space NLL on this region's held-out
        # (never-in-context) points — the era5_fit analogue of
        # kernel_fit/<family>'s *_tabicl block, minus the gap (no GP oracle
        # for real data to gap against; see build_era5_probe's docstring).
        # Reuses the SAME Sigma_e forward pass above (it already covers the
        # full D-point grid, which nll_test_idx indexes into), just scored
        # against TabICL's own frozen PIT (nll_test_z/nll_test_log_pdf,
        # precomputed once in _build_era5_val_batches) instead of rho_emp —
        # only present when a PIT checkpoint was configured for the probe.
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

    metrics["era5_fit/mean_shape_corr"] = _macro_average(region_shape_corr)
    metrics["era5_fit/mean_model_r2"] = _macro_average(region_model_r2)
    metrics["era5_fit/mean_y_nll_total"] = _macro_average(region_y_nll_total)
    metrics["era5_fit/mean_y_nll_marginal"] = _macro_average(region_y_nll_marginal)
    metrics["era5_fit/mean_y_nll_copula"] = _macro_average(region_y_nll_copula)

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

            # Sim-to-real gap: same plot-subset episodes, same oracle targets
            # (data_gen.py's context-blind R_star, not gp_analytical_
            # posterior's R_post — this stays a RELATIVE comparison against
            # the model's own oracle-context prediction on the identical
            # subset, so the reference's own imperfection cancels out even
            # though it isn't the true posterior), but off_p_t comes from
            # conditioning on TabICL's own K-fold PIT z_train instead of the
            # exact one (see the do_plot block above / _build_tabicl_val_z).
            # Compared against `mse` from the same subset so the gap isn't
            # confounded by the two metrics covering different episode sets.
            if all_off_pred_tabicl:
                off_p_t = np.concatenate(all_off_pred_tabicl)
                off_o_t = np.concatenate(all_off_ora_tabicl)
                cq_t = _corr_quality(off_p_t, off_o_t)
                metrics["corr_mse_tabicl_z"]     = cq_t["mse"]
                metrics["corr_mae_tabicl_z"]     = cq_t["mae"]
                # corr_pearson_tabicl_z dropped: same "value can shift with
                # whichever marginal produced z_test" issue as the deleted
                # top-level pooled corr_pearson — corr_mse_tabicl_z/
                # corr_mae_tabicl_z (and sim2real_gap, an MSE difference)
                # aren't scale-free the same way Pearson r superficially
                # looks like it should be, so they're kept.
                metrics["corr_bias_tabicl_z"]    = cq_t["bias"]
                metrics["sim2real_gap"] = cq_t["mse"] - mse

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
) -> int:
    """Restore model weights and optimizer/scaler state from a checkpoint.

    Optimizer moments (Adam/Muon) and the AMP grad scaler state are restored
    so the run doesn't have to relearn gradient statistics from scratch.
    Returns the step the checkpoint was saved at (0 for legacy checkpoints
    without a "step" key), which the caller uses to decide where the LR
    schedule resumes — see the `resume_reset_schedule` handling in train().
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
    return int(ckpt.get("step", 0))


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
    parametrization: str = "covnorm",
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
    s = out.get("s")
    lam = out.get("lam")
    Sigma = low_rank_correlation(
        out["W"].float(),
        s.float() if s is not None else None,
        batch["test_mask"],
        jitter=jitter,
        parametrization=parametrization,
        lam=lam.float() if lam is not None else None,
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
    parametrization: str = "covnorm",
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
            parametrization=parametrization,
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
    parametrization: str = "covnorm",
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
        parametrization=parametrization,
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
        # TF32 tensor-core matmul on Ampere+/Ada+/Hopper: NOT enabled by torch
        # by default, even though the model's own forward already runs under
        # bf16 autocast. What that autocast doesn't cover — Muon's Newton-
        # Schulz orthogonalization (src/muon.py, fp32 grad-derived matmuls,
        # confirmed the single most expensive part of each step: bwd+opt time
        # is several times forward time in profiling) and y_space_nll's
        # Cholesky/logdet path — still runs fp32 matmuls at full CUDA-core
        # precision without this. One-line, ~free win (negligible accuracy
        # cost, standard recommendation for Ampere+) that raises MFU's
        # numerator directly. See torch.set_float32_matmul_precision docs.
        torch.set_float32_matmul_precision("high")
        print(
            f"[train] GPU: {torch.cuda.get_device_name(0)} — assumed peak "
            f"{gpu_peak_flops / 1e12:.0f} TFLOPS (dense bf16/fp16 tensor core) for MFU"
        )

    t = cfg.training
    live_generation = bool(t.get("live_generation", False))
    live_source = str(t.get("live_source", "gp"))
    if live_generation and live_source == "era5" and float(t.get("aux_mae_weight", 0.0)) > 0.0:
        # No oracle R_star exists for real ERA5 (data_gen.py's kernel-generated
        # ground truth has no real-data analogue) -- see era5_live_dataset.py's
        # module docstring and _build_era5_val_batches' docstring for the same
        # constraint on the validation-probe side.
        print("[train] live_source=era5: forcing training.aux_mae_weight=0.0 (real data has no oracle R_star)")
        t.aux_mae_weight = 0.0
    if live_generation and live_source == "era5" and int(t.get("plot_val_every", 0)) > 0:
        # validate()'s do_plot path indexes batch["R_star"] (correlation
        # scatter/heatmap plots against the oracle) unconditionally when a
        # plot step lands -- era5_collate_fn's batches carry no R_star (no
        # oracle exists for real data), so a plot step would KeyError. No
        # oracle-vs-predicted plot is possible here by construction; disable
        # rather than special-case the plotting internals for a field that
        # can never exist under this source.
        print("[train] live_source=era5: forcing training.plot_val_every=0 (no oracle R_star to plot against)")
        t.plot_val_every = 0
    if live_generation:
        _reserve_gpu_headroom_for_live_tabicl(cfg, t, device)
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

    adaptive_kernel_weights = None  # set below only when live_generation + adaptive_kernel_sampling
    tabicl_mix_weights = None  # set below only when live_generation + data.z_train_tabicl_mix_enabled
    train_iter = None  # possibly kicked off early below (live_generation only) -- see there
    # Per-batch raw episodes (kernel metadata intact) for val_loader, keyed by
    # batch_idx -- only populated for the synthetic-GP live-generation path
    # (build_fixed_live_val_batches), which is the only val_loader source that
    # carries return_kernel_metadata=True. Real-ERA5 live_source has no GP
    # kernel to reconstruct a posterior from (same reasoning as era5_fit's own
    # lack of a GP oracle), and the on-disk CopulaDataset path never persisted
    # this metadata to shard files -- both leave this None, and validate()
    # falls back to the separate posterior_probe draw for oracle_diag/gap_nll
    # in those cases (see _build_posterior_probe_batches).
    val_episodes_meta: dict[int, list[dict]] | None = None
    if live_generation:
        # No on-disk dataset at all: episodes are generated on the fly by
        # DataLoader worker processes (see live_dataset.py). Temporary
        # substitute for the disk pipeline below — set training.live_generation
        # =false (the default) to fall back to it unchanged.
        print(
            "[train] live_generation=true — generating episodes on the fly, "
            f"no dataset_dir read ({t.dataset_dir!r} ignored). "
            f"ckpt_dir={t.get('ckpt_dir', None)!r} live_source={live_source!r}"
        )
        if live_source == "era5":
            # Real, worldwide ARCO-ERA5 episodes (era5_live_dataset.py) instead
            # of synthetic GP kernels — no adaptive-kernel-sampling / TabICL-
            # z_train-mix machinery applies here (both are GP-generation-only
            # features), so those two returns stay None.
            train_loader = build_era5_train_loader(cfg, t, device)
            val_loader = build_era5_fixed_val_batches(cfg, t, device)
        else:
            train_loader, adaptive_kernel_weights, tabicl_mix_weights = build_live_train_loader(cfg, t, device)
            val_loader, val_episodes_by_batch = build_fixed_live_val_batches(cfg, t, device)
            val_episodes_meta = dict(enumerate(val_episodes_by_batch))
        print(f"Train: <live> | Val: {len(val_loader) * t.batch_size} episodes (fixed)")
        # Kick off the persistent DataLoader workers now rather than waiting
        # until right before the training loop (the old location of this
        # iter() call). Constructing the iterator spawns the (num_workers)
        # worker processes and lets them start filling their prefetch queue
        # in the background -- each worker loads its own frozen TabICL copy
        # first (see live_dataset.py's "worker N: loading frozen TabICL
        # marginal" print), a ~5-10s cost that was previously paid fully
        # serially, showing up as step 0's outsized `data=` time. Everything
        # below this (the z_train sim-to-real diagnostic, kernel_fit_z,
        # era5 probes, model/optimizer construction) is independent of
        # train_loader, so it now overlaps with worker startup instead.
        # Only safe when tabicl_mix_weights is None: when data.
        # z_train_tabicl_mix_enabled=true, the gap-measurement pass below
        # does an in-place .copy_() into that same shared-memory tensor
        # before any worker may safely read it (see that section's own
        # torn-read comment) -- so in that case the kick is deferred to the
        # original, later spot instead.
        if tabicl_mix_weights is None:
            train_iter = iter(train_loader)
    else:
        meta_path   = os.path.join(t.dataset_dir, "meta.pt")
        shard_files = sorted(glob(os.path.join(t.dataset_dir, "shard_*.pt")))

        train_sampler = None
        train_batch_sampler = None
        val_batch_sampler = None
        variable_d = False
        loader_num_workers_override = t.get("loader_num_workers", None)
        loader_num_workers = (
            int(loader_num_workers_override) if loader_num_workers_override is not None else 4
        )
        # Batches queued ahead per worker. conf/config.yaml sets
        # training.prefetch_factor=8 by default (see its comment for the
        # RSS-vs-data-wait tradeoff this was tuned against); this fallback
        # only fires if that key is missing entirely (e.g. a hand-built cfg
        # that doesn't inherit config.yaml).
        prefetch_factor_override = t.get("prefetch_factor", None)
        prefetch_factor = (
            int(prefetch_factor_override) if prefetch_factor_override is not None else 8
        )
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
                #
                # full_dataset was constructed above with shard_cache_size=
                # shard_block_shards+4 (default 20), sized for ShardBlockSampler's
                # cross-shard blocking. ShardHomogeneousBatchSampler never blocks
                # across shards — its own docstring guarantees "at most one shard
                # is resident at a time" — so that 20-slot cache is dead weight
                # here: with num_workers=4 on train + 4 on val, 20 cached shards/
                # worker on datasets with multi-hundred-MB-to-multi-GB shards (e.g.
                # systematic-composition-all-base, up to ~1.8GB/shard) can push
                # aggregate resident memory into the tens-to-hundreds of GB and get
                # a DataLoader worker SIGKILLed by the OS OOM killer. Shrink to a
                # small constant — enough for the current shard plus one prefetch
                # margin at a shard boundary, not shard_block_shards-worth.
                #
                # That alone wasn't sufficient in practice (still OOM'd a GPU node on
                # systematic-composition-all-base with the real batch_size=32/
                # val_episodes=500 config): DataLoader's batch_sampler round-robin
                # hands consecutive batches to different workers, but
                # ShardHomogeneousBatchSampler emits every batch for one shard
                # consecutively before moving to the next — so with 4 workers, all 4
                # end up needing the SAME shard resident at once, each holding its own
                # independent ~1-1.8GB copy (worker processes don't share this cache;
                # only the returned batch tensors go through shared memory). Turned out
                # NOT to be caused by worker count, though (see the cache.clear() note
                # below for the real culprit) — verified empirically that num_workers=4
                # with the cache properly cleared stays bounded (~24GB peak on a 62GB
                # test node for this dataset's largest shards, vs ~16GB at
                # num_workers=2). If this ever needs to run on a smaller-RAM node,
                # dropping this back to 2 is the lever to pull.
                full_dataset._SHARD_CACHE_SIZE = min(full_dataset._SHARD_CACHE_SIZE, 2)
                # Lowering the cap alone doesn't shrink an already-oversized cache:
                # dataset.py's LRU only evicts one entry per one new insertion (never
                # evicts down to the new cap in one shot), so the d_seen probe just
                # above — which ran while the cache was still sized at
                # shard_block_shards+4, and can have touched up to 8 distinct random
                # shards — leaves _shard_cache stuck at ~8 resident entries forever
                # (each future access evicts 1 and inserts 1, net size unchanged).
                # That stale, oversized cache then gets inherited by every forked
                # DataLoader worker. Clear it now so the new cap actually applies.
                full_dataset._shard_cache.clear()
                # Was dropped to 2 on this ~31GB-cgroup-capped OAR job because
                # eager per-shard loading multiplied shard_cache_size x
                # num_workers x full shard size into RSS. dataset.py now loads
                # shards with mmap=True (near-zero per-shard RSS), which
                # removed that constraint — re-verified empirically
                # (2026-08-11, this same dataset/job): GPU duty cycle averaged
                # ~30% at num_workers=2 vs ~60-65% at 4/6/8 (nvidia-smi dmon,
                # 1s samples), with cgroup RSS flat (~4.9GB) across all of
                # them — a plateau, not a monotonic win, so 4 is the practical
                # default rather than pushing higher for no further gain.
                # Override via training.loader_num_workers to test further.
                loader_num_workers = (
                    int(loader_num_workers_override)
                    if loader_num_workers_override is not None
                    else 4
                )
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

        # collate_fn (dataset.py) also assembles an (B, N_max, N_max) R_prior
        # tensor for schema-complete consumers (eval/plotting scripts, per its
        # docstring) — but no training code path reads batch["R_prior"] (only
        # R_star/Sigma_star feed loss.py/model.py; grep-verified). On datasets
        # with large N this is a full extra big-matrix copy per batch (equal in
        # size to R_star/Sigma_star) purely to populate an unused key, and it's
        # redundant besides: dataset.py derives R_prior as a clone of R_star for
        # oracle_mode="prior" datasets (the only mode this repo writes), so it
        # never carries information collate_fn's R_star output doesn't already
        # have. Drop it before the shared collate_fn runs so its has_prior
        # branch (the actual allocate+copy cost) never fires for training.
        def _train_collate_fn(samples):
            for s in samples:
                s.pop("R_prior", None)
            return collate_fn(samples)

        # A batch_sampler (variable-d homogeneous batching) is mutually exclusive
        # with batch_size/sampler/shuffle, so pick one construction or the other.
        train_loader = DataLoader(
            train_dataset,
            collate_fn=_train_collate_fn,
            num_workers=loader_num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
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
            collate_fn=_train_collate_fn,
            num_workers=loader_num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            **(
                {"batch_sampler": val_batch_sampler}
                if val_batch_sampler is not None
                else {"batch_size": t.batch_size, "shuffle": False}
            ),
        )

    baselines_on = bool(cfg.get("baselines", {}).get("enabled", True))
    synth_kernel_batches = _build_synthetic_kernel_batches(cfg, device) if baselines_on else {}
    # Only needed as the oracle_diag/gap_nll fallback when val_loader itself
    # can't supply kernel metadata (disk-mode CopulaDataset, or the real-ERA5
    # live_source) -- see val_episodes_meta's own comment above and
    # validate()'s posterior_probe/val_episodes_meta handling.
    posterior_probe = (
        _build_posterior_probe_batches(cfg, device)
        if (baselines_on and val_episodes_meta is None)
        else None
    )

    # z_train sim-to-real diagnostic (see _build_tabicl_val_z / validate()'s
    # do_plot block): needs a second, frozen TabICL copy with its native
    # quantile head intact (unlike the copula model's backbone, which has it
    # stripped — see model.py:CopulaTabICL) to PIT the val episodes the same
    # way real (non-GP) deployment data would be. See pit.py::resolve_pit_ckpt
    # for which checkpoint (if any) that uses.
    pit_ckpt = resolve_pit_ckpt(cfg)
    tabicl_val_z: dict = {}
    tabicl_kernel_fit_z: dict = {}
    # Real-ERA5 spatial-correlation probes (see _build_era5_val_batches):
    # built here too, alongside tabicl_val_z, so the one-time PIT cost on the
    # frozen context sample is paid before `tabicl_marginal` is discarded.
    # Not strictly gated on pit_ckpt existing -- if pit_ckpt is None (e.g.
    # tabicl.pretrained=false with no explicit tabicl.pit_ckpt), the elif
    # branch below still tries tabicl.ckpt directly for era5_fit alone
    # before falling back to build_era5_probe's naive-standardization path.
    era5_val_batches: dict = {}
    era5_on = baselines_on and bool(cfg.get("baselines", {}).get("era5_enabled", True))
    # tabicl_mix_weights is not None only when live_generation + data.
    # z_train_tabicl_mix_enabled=true (see build_live_train_loader) -- needs
    # tabicl_marginal loaded below regardless of baselines_on, since the
    # gap measurement it drives is independent of the kernel_fit/<family>
    # baseline probes.
    if (baselines_on or tabicl_mix_weights is not None) and pit_ckpt:
        print("[train] Loading frozen TabICL marginal for the z_train sim-to-real diagnostic...")
        tabicl_marginal = load_tabicl(pit_ckpt, device)
        pit_k_folds = int(cfg.tabicl.get("pit_k_folds", DEFAULT_K_FOLDS))
        tabicl_val_z = _build_tabicl_val_z(val_loader, tabicl_marginal, pit_k_folds, device)
        if synth_kernel_batches:
            print(
                "[train] Building TabICL PIT cache for kernel_fit/<family> "
                "probes (feeds training.adaptive_kernel_signal='tabicl')..."
            )
            tabicl_kernel_fit_z = _build_tabicl_kernel_fit_z(
                synth_kernel_batches, tabicl_marginal, pit_k_folds, device
            )
        if tabicl_mix_weights is not None:
            floor_frac = float(cfg.data.get("z_train_tabicl_mix_floor_frac", 0.05))
            max_frac = float(cfg.data.get("z_train_tabicl_mix_max_frac", 0.35))
            if math.isclose(floor_frac, max_frac, abs_tol=1e-12):
                # _tabicl_gap_to_mix_frac(gaps, floor, max)'s interpolation
                # frac[i] = floor + (max - floor) * normalized collapses to
                # floor for every family whenever floor == max, independent
                # of the measured gap -- so the gap measurement below (a full
                # _generate_gp_batch_raw pass PER _COMPOSABLE_KERNELS family,
                # one of the two calls running real TabICL k-fold PIT) would
                # spend several minutes computing a value this run can never
                # use. tabicl_mix_weights is already initialized to
                # floor_frac uniformly by build_live_train_loader, so there's
                # nothing left to write here either.
                print(
                    f"[train] data.z_train_tabicl_mix_floor_frac == max_frac "
                    f"({floor_frac:.3f}) -- mix fraction is fixed regardless "
                    "of the TabICL-vs-analytic gap, skipping the gap "
                    "measurement pass."
                )
            else:
                print(
                    "[train] Measuring per-family TabICL-vs-analytic z_train gap "
                    "for data.z_train_tabicl_mix_* (this runs once, up front)..."
                )
                z_gap = _compute_tabicl_z_train_gap(cfg, tabicl_marginal, pit_k_folds, device)
                new_mix_frac = _tabicl_gap_to_mix_frac(z_gap, floor_frac, max_frac)
                # In-place: tabicl_mix_weights is the shared-memory tensor
                # LiveGPDataset workers already hold a reference to (built
                # before the DataLoader forks/spawns -- see build_live_train_
                # loader's docstring). Workers haven't started iterating yet at
                # this point in train.py's startup sequence, so there's no
                # torn-read race, but .copy_() (not rebind) is used anyway to
                # match kernel_weights's own update convention below.
                tabicl_mix_weights.copy_(new_mix_frac)
                for family, gap in sorted(z_gap.items(), key=lambda kv: -kv[1]):
                    idx = _COMPOSABLE_KERNELS.index(family)
                    print(
                        f"[train]   {family}: z_train_tabicl_gap={gap:.3f} "
                        f"-> mix_frac={float(new_mix_frac[idx]):.3f}"
                    )
                wandb.log(
                    {f"data/z_train_tabicl_gap/{f}": g for f, g in z_gap.items()}
                    | {
                        f"data/tabicl_mix_frac/{family}": float(new_mix_frac[i])
                        for i, family in enumerate(_COMPOSABLE_KERNELS)
                    },
                    step=0,
                )
        if era5_on:
            print("[train] Building frozen real-ERA5 spatial-correlation probes...")
            era5_val_batches = _build_era5_val_batches(cfg, tabicl_marginal, device)
        del tabicl_marginal  # only the caches built above are needed from here on
        if device == "cuda":
            # gc.collect() before empty_cache() (see this repo's OOM-handler
            # gotcha): del alone doesn't free CUDA storage until any
            # reference cycles in the eval-mode forward graph are collected.
            gc.collect()
            torch.cuda.empty_cache()
    elif era5_on:
        # No general pit_ckpt (e.g. tabicl.pretrained=false with no explicit
        # tabicl.pit_ckpt override -- see pit.py::resolve_pit_ckpt). era5_fit
        # doesn't care whether the run's OWN backbone is pretrained; it just
        # wants a real quantile-head marginal to PIT the ERA5 context with if
        # one is named, so it reuses tabicl.ckpt directly here rather than
        # going through resolve_pit_ckpt's pretrained-gated default.
        era5_ckpt = cfg.tabicl.get("ckpt", None)
        if era5_ckpt:
            print(
                f"[train] Loading frozen TabICL marginal ({era5_ckpt}) for "
                "era5_fit only (tabicl.pretrained="
                f"{bool(cfg.tabicl.get('pretrained', True))}, no general PIT "
                "checkpoint configured otherwise)..."
            )
            era5_tabicl_marginal = load_tabicl(era5_ckpt, device)
            era5_val_batches = _build_era5_val_batches(cfg, era5_tabicl_marginal, device)
            del era5_tabicl_marginal
            if device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
        else:
            print(
                "[train] Building frozen real-ERA5 spatial-correlation probes "
                "(no tabicl.ckpt configured -- context z_train falls back to "
                "naive standardization)..."
            )
            era5_val_batches = _build_era5_val_batches(cfg, None, device)

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

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = GradScaler(device=device) if (use_amp and amp_dtype == torch.float16) else None

    # Resume weights + optimizer/scaler state first (if requested) so we know
    # what step the LR schedule should continue from before building the
    # scheduler below. Default: continue the cosine schedule from the
    # checkpoint's step, instead of re-running warmup from a from-scratch
    # peak LR on top of already-warmed-up Adam/Muon moments — the two
    # combined were spiking effective step size right after resume.
    # `resume_reset_schedule=true` opts back into the old behavior, for the
    # deliberate "warm-start a new experiment from these weights" case where
    # a fresh warmup/cosine schedule (not a continuation) is actually wanted.
    start_step = 0
    if resume_ckpt:
        ckpt_step = load_checkpoint(resume_ckpt, model, device, optimizer=optimizer, scaler=scaler)
        if bool(t.get("resume_reset_schedule", False)):
            print(f"Resumed weights + optimizer/scaler state from {resume_ckpt} (step {ckpt_step}) — resetting to step 0 with a fresh warmup/cosine schedule (resume_reset_schedule=true)")
        else:
            start_step = ckpt_step
            print(f"Resumed weights + optimizer/scaler state from {resume_ckpt} — continuing cosine schedule from step {start_step}")

    if start_step > 0:
        # LambdaLR requires 'initial_lr' on each param group to resume at a
        # non-zero last_epoch. Rebase it to this run's configured base LR
        # (not whatever raw 'lr' the checkpoint's optimizer state restored,
        # which is the previous run's already-decayed value) so the cosine
        # curve below reflects this run's schedule at start_step.
        for group in optimizer.param_groups:
            group["initial_lr"] = t.muon_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_lr_lambda(s, t.warmup_steps, t.steps, lr_min_frac),
        last_epoch=start_step - 1 if start_step > 0 else -1,
    )

    jitter = float(cfg.model.get("sigma_jitter", 1e-4))
    parametrization = str(cfg.model.get("correlation_parametrization", "covnorm"))
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
    if train_iter is None:  # not already kicked off early above
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
        except torch.cuda.OutOfMemoryError as exc:
            # Live-generation's TabICL workers run on the GPU in separate
            # DataLoader worker processes (see live_dataset.py /
            # _reserve_gpu_headroom_for_live_tabicl above) — an OOM raised
            # there propagates through DataLoader's ExceptionWrapper.reraise()
            # same as any other worker exception. Unlike an OOM inside
            # _run_train_step below, this used to be completely uncaught here
            # and killed the whole run instead of costing one skipped step.
            # The dead worker's own per-process generator cannot resume after
            # raising (Python generators don't survive an unhandled
            # exception), so just retrying next(train_iter) would spin on a
            # now-empty source; re-creating the iterator makes
            # persistent_workers re-invoke LiveGPDataset.__iter__ in that
            # worker instead (reloading its frozen TabICL copy, ~5s, but only
            # on this rare recovery path).
            print(f"[{step:6d}] CUDA OOM in a live-generation DataLoader worker — recreating iterator, skipping step.")
            traceback.clear_frames(exc.__traceback__)
            del exc
            gc.collect()
            torch.cuda.empty_cache()
            train_iter = iter(train_loader)
            continue

        optimizer.zero_grad(set_to_none=True)
        try:
            # non_blocking overlaps H→D transfer with previous GPU work
            # (pin_memory=True).
            batch = {k: v.to(device, non_blocking=True) for k, v in raw_batch.items()}
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
                parametrization=parametrization,
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
                        parametrization=parametrization,
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
                tabicl_val_z=tabicl_val_z,
                tabicl_kernel_fit_z=tabicl_kernel_fit_z,
                era5_val_batches=era5_val_batches,
                posterior_probe=posterior_probe,
                val_episodes_meta=val_episodes_meta,
            )
            # oracle_diag/* keys are already fully qualified (a sibling
            # top-level wandb group, deliberately kept out of val/ — see
            # validate()'s ground-truth-z_test comment above the
            # posterior_probe block); everything else gets the usual val/
            # prefix.
            log_dict = {
                (k if k.startswith("oracle_diag/") else f"val/{k}"): v
                for k, v in metrics.items()
            }
            if adaptive_kernel_weights is not None:
                lr = float(t.get("adaptive_kernel_lr", 1.0))
                floor = float(t.get("adaptive_kernel_floor", 0.05))
                signal = str(t.get("adaptive_kernel_signal", "tabicl"))
                excluded_kernels = set(getattr(cfg.data, "composite_exclude_kernels", None) or [])
                new_kernel_weights = _update_adaptive_kernel_weights(
                    adaptive_kernel_weights, metrics, lr, floor,
                    exclude=excluded_kernels, signal=signal,
                )
                # In-place: adaptive_kernel_weights is the shared-memory
                # tensor LiveGPDataset workers read from (live_dataset.py) —
                # rebinding the name here would leave workers pointed at the
                # old tensor instead of picking up the update.
                adaptive_kernel_weights.copy_(new_kernel_weights)
                # Excluded families are never in _sample_kernel_chain_structure's
                # pool (data_gen.py::_weights_for_pool renormalizes over the
                # post-exclude pool), so their weight is inert -- skip logging
                # it to avoid implying it drives sampling.
                for i, family in enumerate(_COMPOSABLE_KERNELS):
                    if family in excluded_kernels:
                        continue
                    log_dict[f"val/kernel_sampling_weight/{family}"] = float(new_kernel_weights[i])
            if plot_figs:
                # plot_figs[0] is built from all_off_pred/all_off_ora — the
                # same analytic-z_train-conditioned Sigma as the
                # sigma_*_analytic_z scalars above, hence the matching suffix.
                log_dict["val/corr_density_analytic_z"] = wandb.Image(plot_figs[0])
                if len(plot_figs) > 1:
                    log_dict["val/corr_grid"] = wandb.Image(plot_figs[1])
                for f in plot_figs:
                    plt.close(f)
            wandb.log(log_dict, step=step)
            # Surfaces the real (TabICL-marginal) total NLL as the
            # live-monitoring headline (only available once a PIT checkpoint
            # is configured — see validate()'s y_nll_total comment; "n/a"
            # otherwise, not a ground-truth-scored substitute) alongside
            # oracle_diag/gap_nll (the true-Bayes-optimal-ceiling gap, a
            # same-population comparison — val_loader's own episodes when
            # val_episodes_meta is available, else the posterior_probe
            # fallback — see validate()) and oracle_diag/corr_pearson
            # (correlation VALUES against gp_analytical_posterior's true
            # R_post).
            total_nll = metrics.get("y_nll_total", float("nan"))
            total_str = f"{total_nll:.4f}" if math.isfinite(total_nll) else "n/a"
            gap = metrics.get("oracle_diag/gap_nll", float("nan"))
            gap_str = f"{gap:.4f}" if math.isfinite(gap) else "n/a"
            pearson = metrics.get("oracle_diag/corr_pearson", float("nan"))
            pearson_str = f"{pearson:.3f}" if math.isfinite(pearson) else "n/a"
            cop_std = metrics.get("oracle_diag/copula_nll_std", float("nan"))
            cop_std_str = f"{cop_std:.4f}" if math.isfinite(cop_std) else "n/a"
            print(
                f"[{step:6d}] VAL  "
                f"total={total_str}  "
                f"gap_post={gap_str}  "
                f"corr_r={pearson_str}  "
                f"od_μ={metrics['sigma_offdiag_mean_analytic_z']:+.4f} od_σ={metrics['sigma_offdiag_std_analytic_z']:.4f} od_|r|={metrics['sigma_offdiag_abs_mean_analytic_z']:.4f}  "
                f"cop_std={cop_std_str}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if step % t.save_every == 0 and step > 0:
            save_checkpoint(model, optimizer, scheduler, cfg, step, scaler=scaler)
            if (
                tabicl_mix_weights is not None and pit_ckpt
                and bool(cfg.data.get("z_train_tabicl_mix_adaptive", False))
            ):
                z_gap, new_mix_frac = _refresh_tabicl_mix_weights(
                    cfg, pit_ckpt, tabicl_mix_weights, device
                )
                print(f"[train][step {step}] Re-measured z_train_tabicl_mix_* (adaptive):")
                save_log = {}
                for i, family in enumerate(_COMPOSABLE_KERNELS):
                    if family in z_gap:
                        save_log[f"val/z_train_tabicl_gap/{family}"] = z_gap[family]
                        print(
                            f"[train]   {family}: z_train_tabicl_gap={z_gap[family]:.3f} "
                            f"-> mix_frac={float(new_mix_frac[i]):.3f}"
                        )
                    save_log[f"val/tabicl_mix_frac/{family}"] = float(new_mix_frac[i])
                wandb.log(save_log, step=step)

    save_checkpoint(model, optimizer, scheduler, cfg, t.steps, scaler=scaler)
    wandb.finish()


if __name__ == "__main__":
    main()
