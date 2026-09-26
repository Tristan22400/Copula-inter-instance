"""
train.py — Train the Copula Transformer in Y-space NLL via Sklar's theorem.

Loss:  L = Copula_NLL(z_test; Σ̂) + Marginal_NLL(y_test; TabICL log-pdf)
Σ̂ is built by ``model.build_sigma(out, cfg)`` from the model output — the
correlation parametrization (covnorm/cossim/tanhnorm/sparse_covnorm) is
selected by ``cfg.model.correlation_parametrization`` (see
correlation_factory.py).

Usage:
    python -m copula_inter.train
    python -m copula_inter.train training.steps=500 training.dataset_dir=./data/debug_latent
    WANDB_MODE=disabled python -m copula_inter.train training.steps=200
"""

from __future__ import annotations

import gc
import math
import os
import traceback

from copula_inter.adaptive_sampling import (
    _compute_tabicl_z_train_gap,
    _refresh_tabicl_mix_weights,
    _tabicl_gap_to_mix_frac,
    _update_adaptive_kernel_weights,
)
from copula_inter.checkpointing import load_checkpoint, save_checkpoint
from copula_inter.era5_probes import _build_era5_val_batches, _build_era5_viz_batch
from copula_inter.probe_batches import (
    _build_analytic_val_z,
    _build_posterior_probe_batches,
    _build_synthetic_kernel_batches,
    _build_tabicl_kernel_fit_z,
    _build_tabicl_val_z,
    _sigma_stats,
)
from copula_inter.validation import validate

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

import time
from glob import glob

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Subset

from copula_inter.backend_registry import TABICL_Z_TRAIN_SOURCES
from copula_inter.backend_registry import z_train_source as z_train_source_of

# eval/ (regions.py, spatial-correlation probe helpers -- see
# _build_era5_val_batches below) lives at the repo root, not under src/.
from copula_inter.config_path import config_dir
from copula_inter.data_gen import _COMPOSABLE_KERNELS
from copula_inter.dataset import (
    CopulaDataset,
    ShardBlockSampler,
    ShardHomogeneousBatchSampler,
    collate_fn,
)
from copula_inter.era5_live_dataset import build_era5_fixed_val_batches, build_era5_train_loader
from copula_inter.live_dataset import (
    _GENERIC_MARGINAL_BACKENDS,
    _LIVE_TABICL_FLAT_HEADROOM_GB,
    _LIVE_TABICL_WORKER_FIXED_OVERHEAD_GB,
    _LIVE_TABICL_WORKER_PER_EPISODE_GB,
    _validate_z_train_source,
    build_fixed_live_val_batches,
    build_live_train_loader,
    resolve_live_tabicl_num_workers,
)
from copula_inter.model import build_copula_transformer
from copula_inter.muon import Muon
from copula_inter.pit import (
    DEFAULT_K_FOLDS,
    configure_tabicl_inference_amp,
    load_tabicl,
    resolve_pit_ckpt,
)
from copula_inter.training_core import (
    _measure_step_flops,
    _run_train_step,
    cosine_lr_lambda,
)

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
    will also run marginal-backend inference inside separate GPU DataLoader
    worker processes on the same card (live_dataset.py's
    data.z_train_tabicl_mix_* / data.z_train_source=tabicl*/exaone/tabpfn
    path -- live_dataset.py::batched_marginal_worker_enabled, all sharing the
    same worker/spawn pattern now that exaone/tabpfn batch their own forward
    too, see eval/spatial/exaone_batched.py and tabpfn_batched.py).

    Root cause this works around: each such worker holds its own CUDA
    context, entirely separate from this process's caching allocator pool.
    Live-generation batches vary P/N a lot (see the top-of-file
    expandable_segments comment), so this process's own pool can ratchet up
    to a high "reserved" watermark and never give it back -- PyTorch's
    caching allocator only returns memory to the driver via empty_cache(),
    which nothing here calls outside the OOM handler below. Reserved memory
    is invisible to *other* processes even when this process isn't actively
    using most of it, so given enough steps this process's pool can starve a
    worker's small allocation of the little free VRAM the driver has left --
    and unlike this process's own OOMs, a worker's OOM was previously
    completely uncaught (see next(train_iter) below), taking the whole run
    down instead of costing one skipped step.

    set_per_process_memory_fraction makes this process itself OOM (into the
    already-correct, already-tested gc.collect()-before-empty_cache() handler
    below) once it would otherwise have crowded out the workers' headroom,
    instead of silently starving them. It only caps future growth -- it does
    not reclaim memory already reserved -- so this must run before any
    training-loop allocation happens.

    exaone/tabpfn workers share TabICL's own headroom formula below (see
    resolve_live_tabicl_num_workers' docstring for the caveat that its
    constants are calibrated against TabICL's per-worker VRAM specifically,
    not benchmarked per backend) -- a shared conservative estimate rather
    than no reservation at all, consistent with live_dataset.py's
    build_live_train_loader treating every batched-GPU-worker backend
    identically for worker/group-size sizing too.
    """
    z_train_source = z_train_source_of(cfg)
    _validate_z_train_source(z_train_source)
    mix_enabled = bool(cfg.data.get("z_train_tabicl_mix_enabled", False))
    batched_marginal_worker_enabled = (
        mix_enabled or z_train_source in TABICL_Z_TRAIN_SOURCES or z_train_source in _GENERIC_MARGINAL_BACKENDS
    )
    if not batched_marginal_worker_enabled or device != "cuda":
        return
    # Same resolution build_live_train_loader uses below (auto-sized from
    # currently-free GPU memory when training.live_tabicl_num_workers is
    # left unset) -- keeping both call sites on one resolver means the
    # headroom reserved here always matches the worker count actually
    # spawned, instead of drifting if only one of the two were updated.
    num_workers = resolve_live_tabicl_num_workers(t, device)
    if num_workers <= 0:
        return
    group_multiplier = max(1, int(t.get("live_tabicl_group_multiplier", 2)))
    group_size = int(t.batch_size) * group_multiplier
    per_worker_gb = _LIVE_TABICL_WORKER_FIXED_OVERHEAD_GB + _LIVE_TABICL_WORKER_PER_EPISODE_GB * group_size
    headroom_gb = num_workers * per_worker_gb + _LIVE_TABICL_FLAT_HEADROOM_GB
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


def _build_model_and_optimizer(cfg, device, resume_ckpt, t):
    """Build the copula model, Muon/AdamW optimizer, AMP scaler and LR schedule; resume from training.resume_ckpt if set."""
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

    use_amp = (device == "cuda") and bool(t.get("use_amp", True))
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
    return amp_dtype, model, optimizer, scaler, scheduler, start_step, trainable, use_amp


def _build_validation_probes(cfg, device, t, tabicl_mix_weights, val_episodes_meta, val_loader):
    """Build the fixed validation probes (synthetic kernel families, posterior probes, ERA5) and their TabICL/analytic z_train."""
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
    # Oracle z for every oracle_diag/* metric, independent of pit_ckpt and of
    # data.z_train_source. Only needed when the val batches are NOT already
    # analytic: under z_train_source="tabicl"/"tabicl_split",
    # build_fixed_live_val_batches routes generate_gp_batch through TabICL and
    # the batch's z_train/z_test/log_pdf_test are TabICL's, not the exact GP's
    # -- see _build_analytic_val_z's docstring for why comparing those against
    # gp_analytical_posterior's ceiling is a cross-z-space comparison rather
    # than a gap. Cheap: one triangular solve per episode, once, at startup,
    # reusing the _L_ff/_alpha factors the episodes already carry.
    analytic_val_z: dict = {}
    if val_episodes_meta is not None and z_train_source_of(cfg) != "analytic":
        print(
            "[train] Building the exact analytic-GP PIT cache for oracle_diag/* "
            f"(data.z_train_source={z_train_source_of(cfg)} puts the val "
            "batches in TabICL's z-space)..."
        )
        analytic_val_z = _build_analytic_val_z(val_loader, val_episodes_meta, device)
    # Real-ERA5 spatial-correlation probes (see _build_era5_val_batches):
    # built here too, alongside tabicl_val_z, so the one-time PIT cost on the
    # frozen context sample is paid before `tabicl_marginal` is discarded.
    # Not strictly gated on pit_ckpt existing -- if pit_ckpt is None (e.g.
    # tabicl.pretrained=false with no explicit tabicl.pit_ckpt), the elif
    # branch below still tries tabicl.ckpt directly for era5_fit alone
    # before falling back to build_era5_probe's naive-standardization path.
    era5_val_batches: dict = {}
    era5_viz_batch: "dict | None" = None
    era5_on = baselines_on and bool(cfg.get("baselines", {}).get("era5_enabled", True))
    # Only worth the fetch+PIT cost (paid once, here) if do_plot will ever
    # actually fire and consume it -- see validate()'s do_plot block /
    # _build_era5_viz_batch.
    era5_viz_on = era5_on and int(t.get("plot_val_every", 5000)) > 0
    # tabicl_mix_weights is not None only when live_generation + data.
    # z_train_tabicl_mix_enabled=true (see build_live_train_loader) -- needs
    # tabicl_marginal loaded below regardless of baselines_on, since the
    # gap measurement it drives is independent of the kernel_fit/<family>
    # baseline probes.
    if (baselines_on or tabicl_mix_weights is not None) and pit_ckpt:
        print("[train] Loading frozen TabICL marginal for the z_train sim-to-real diagnostic...")
        tabicl_marginal = load_tabicl(pit_ckpt, device)
        pit_k_folds = int(cfg.tabicl.get("pit_k_folds", DEFAULT_K_FOLDS))
        # val/y_nll_* must be scored against the SAME TabICL marginal the model
        # was conditioned on. Under data.z_train_source="tabicl"/"tabicl_split"
        # the val episodes were generated with data.z_train_tabicl_k_folds folds
        # (5 by default -- a documented throughput tradeoff in
        # conf/data/gp_tasks.yaml), while tabicl.pit_k_folds defaults to
        # DEFAULT_K_FOLDS=10. Recomputing the val PIT at 10 folds put
        # val/y_nll_total -- the real-deployment headline -- on a marginal the
        # model never saw, and the fold count is not a free parameter of a
        # comparison: a K-fold PIT's sharpness moves with K, so the total NLL
        # moves with it too. Follow the data's fold count in that case;
        # tabicl.pit_k_folds still governs everywhere the fold count is a free
        # choice (kernel_fit probes, the z_train mix-gap measurement).
        val_pit_k_folds = pit_k_folds
        if z_train_source_of(cfg) in ("tabicl", "tabicl_split"):
            val_pit_k_folds = int(cfg.data.get("z_train_tabicl_k_folds", pit_k_folds))
            if val_pit_k_folds != pit_k_folds:
                print(
                    f"[train] val PIT k_folds={val_pit_k_folds} (from "
                    f"data.z_train_tabicl_k_folds), matching how the val episodes "
                    f"were generated, not tabicl.pit_k_folds={pit_k_folds}."
                )
        tabicl_val_z = _build_tabicl_val_z(val_loader, tabicl_marginal, val_pit_k_folds, device)
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
        if era5_viz_on:
            print("[train] Building frozen real-ERA5 prediction-viz probe...")
            era5_viz_batch = _build_era5_viz_batch(cfg, tabicl_marginal, device)
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
            if era5_viz_on:
                era5_viz_batch = _build_era5_viz_batch(cfg, era5_tabicl_marginal, device)
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
            if era5_viz_on:
                era5_viz_batch = _build_era5_viz_batch(cfg, None, device)
    return analytic_val_z, era5_val_batches, era5_viz_batch, pit_ckpt, posterior_probe, synth_kernel_batches, tabicl_kernel_fit_z, tabicl_val_z


def _init_wandb_run(cfg, dataset_name, t):
    """Name the run from its model/training settings and start wandb."""
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
    return resume_ckpt


def _build_data_loaders(cfg, device, live_generation, live_source, t):
    """Build the training iterator and validation loader for live (GP or ERA5) or on-disk data."""
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
        # Reserve worker headroom only now, after build_fixed_live_val_batches'
        # own (uncapped-need) marginal-backend forward passes have already run
        # and freed their temporary model copy (see that function's docstring)
        # — capping this process's own VRAM fraction any earlier (the previous
        # location, right at live_generation's top) starved that one-time
        # setup call itself instead of protecting it: exaone/tabpfn's batched
        # forward at real (non-toy) P/N easily needs more than half the GPU
        # for a single call, well past this formula's own 0.5 floor, so the
        # main process was OOMing on its OWN val-batch construction before a
        # single worker even existed to protect (see git history for the
        # observed traceback). No iter(train_loader) call — the only thing
        # that actually spawns the persistent GPU workers this guards against
        # — happens before this point in any live_generation branch.
        _reserve_gpu_headroom_for_live_tabicl(cfg, t, device)
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
    return adaptive_kernel_weights, tabicl_mix_weights, train_iter, train_loader, val_episodes_meta, val_loader


@hydra.main(config_path=config_dir(__file__), config_name="config", version_base=None)
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
        # Schulz orthogonalization (src/copula_inter/muon.py, fp32 grad-derived matmuls,
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
    tabicl_amp = bool(t.get("tabicl_inference_amp", True))
    configure_tabicl_inference_amp(tabicl_amp)
    print(f"[train] frozen TabICL marginal inference AMP={'on' if tabicl_amp else 'off (float32)'}")
    live_generation = bool(t.get("live_generation", False))
    live_source = str(t.get("live_source", "gp"))
    if live_generation and live_source == "era5" and float(t.get("aux_mae_weight", 0.0)) > 0.0:
        # No oracle R_star exists for real ERA5 (data_gen.py's kernel-generated
        # ground truth has no real-data analogue) -- see era5_live_dataset.py's
        # module docstring and _build_era5_val_batches' docstring for the same
        # constraint on the validation-probe side.
        print("[train] live_source=era5: forcing training.aux_mae_weight=0.0 (real data has no oracle R_star)")
        t.aux_mae_weight = 0.0
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
    resume_ckpt = _init_wandb_run(cfg=cfg, dataset_name=dataset_name, t=t)

    (adaptive_kernel_weights, tabicl_mix_weights, train_iter, train_loader, val_episodes_meta, val_loader) = _build_data_loaders(cfg=cfg, device=device, live_generation=live_generation, live_source=live_source, t=t)

    (analytic_val_z, era5_val_batches, era5_viz_batch, pit_ckpt, posterior_probe, synth_kernel_batches, tabicl_kernel_fit_z, tabicl_val_z) = _build_validation_probes(cfg=cfg, device=device, t=t, tabicl_mix_weights=tabicl_mix_weights, val_episodes_meta=val_episodes_meta, val_loader=val_loader)

    (amp_dtype, model, optimizer, scaler, scheduler, start_step, trainable, use_amp) = _build_model_and_optimizer(cfg=cfg, device=device, resume_ckpt=resume_ckpt, t=t)

    jitter = float(cfg.model.get("sigma_jitter", 1e-4))
    parametrization = str(cfg.model.get("correlation_parametrization", "covnorm"))
    nll_weight = float(t.get("nll_weight", 1.0))
    aux_mae_weight = float(t.get("aux_mae_weight", 0.0))
    # Backbone's own MoE auxiliary loss weight (TabLDM only -- see
    # model.py's forward / copula_backbones.moe_aux_loss; a no-op for
    # backbones that don't emit "moe_aux_loss"). Defaults to 1.0, standard
    # MoE-training convention, since the term is already internally scaled
    # by the checkpoint's own router-loss coefficients.
    moe_aux_weight = float(t.get("moe_aux_weight", 1.0))

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
                moe_aux_weight=moe_aux_weight,
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
                        moe_aux_weight=moe_aux_weight,
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
                Sigma = Sigma.dense()
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
                analytic_val_z=analytic_val_z,
                tabicl_kernel_fit_z=tabicl_kernel_fit_z,
                era5_val_batches=era5_val_batches,
                era5_viz_batch=era5_viz_batch,
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
                # The real-ERA5 diagnostic figures (see validate()'s
                # do_plot block / _build_era5_viz_batch): predicted-vs-
                # ground-truth field, copula-latent-z predictor samples, and
                # marginal/GP predictive-variance-vs-context-distance — each
                # keyed by its own wandb panel name already.
                for key, fig in plot_figs.items():
                    log_dict[key] = wandb.Image(fig)
                    plt.close(fig)
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
            # TabICL-conditioned copula NLL (y_nll_copula, see its comment
            # above in validate()): the Sklar split of y_nll_total's copula
            # component, scored against TabICL's own z_test instead of the
            # oracle/analytic PIT used by oracle_diag/*. Side-by-side with
            # gap_post (oracle-PIT-conditioned) this isolates whether a
            # mismatch traces to the model's Sigma or to the oracle-vs-TabICL
            # PIT gap.
            cop_tabicl = metrics.get("y_nll_copula", float("nan"))
            cop_tabicl_str = f"{cop_tabicl:.4f}" if math.isfinite(cop_tabicl) else "n/a"
            # cop_gap={gap}/{headroom}: the copula gap against the ENTIRE
            # Bayes-optimal copula reward available on these episodes. The
            # denominator is not decoration -- it collapses with context size
            # and with the generator's lengthscale/noise ranges (see
            # oracle_diag/copula_headroom in validate() for a measured series),
            # so the numerator alone cannot be read as good or bad.
            # gap > headroom means the model is worse than independence.
            # corr_kl is the noise-free version of gap_post (see
            # pit.gaussian_corr_kl): a functional of Sigma and R_post alone,
            # with no one-draw Monte-Carlo term, so it is the signal to watch
            # for convergence.
            cop_gap = metrics.get("oracle_diag/copula_gap", float("nan"))
            headroom = metrics.get("oracle_diag/copula_headroom", float("nan"))
            cop_gap_str = (
                f"{cop_gap:.4f}/{headroom:.4f}"
                if math.isfinite(cop_gap) and math.isfinite(headroom) else "n/a"
            )
            ckl = metrics.get("oracle_diag/corr_kl", float("nan"))
            ckl_str = f"{ckl:.4f}" if math.isfinite(ckl) else "n/a"
            print(
                f"[{step:6d}] VAL  "
                f"total={total_str}  "
                f"gap_post={gap_str}  "
                f"corr_r={pearson_str}  "
                f"od_μ={metrics['sigma_offdiag_mean_analytic_z']:+.4f} od_σ={metrics['sigma_offdiag_std_analytic_z']:.4f} od_|r|={metrics['sigma_offdiag_abs_mean_analytic_z']:.4f}  "
                f"cop_std={cop_std_str}  "
                f"cop_tabicl={cop_tabicl_str}  "
                f"cop_gap={cop_gap_str}  "
                f"corr_kl={ckl_str}  "
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
