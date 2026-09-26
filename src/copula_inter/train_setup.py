"""Training-run setup for copula_inter.train: device and MFU peak, run naming, data loaders, validation probes, model and optimizer."""

from __future__ import annotations

import gc
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, TypeAlias, cast

import torch.nn as nn

from copula_inter.adaptive_sampling import (
    _compute_tabicl_z_train_gap,
    _tabicl_gap_to_mix_frac,
)
from copula_inter.checkpointing import load_checkpoint
from copula_inter.era5_probes import _build_era5_val_batches, _build_era5_viz_batch
from copula_inter.probe_batches import (
    _build_analytic_val_z,
    _build_posterior_probe_batches,
    _build_synthetic_kernel_batches,
    _build_tabicl_kernel_fit_z,
    _build_tabicl_val_z,
)

if TYPE_CHECKING:
    from copula_inter.model import CopulaTabICL


from glob import glob

import torch
import wandb
from omegaconf import DictConfig
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Subset

from copula_inter.backend_registry import TABICL_Z_TRAIN_SOURCES
from copula_inter.backend_registry import z_train_source as z_train_source_of
from copula_inter.config_path import config_dict
from copula_inter.dataset import (
    CopulaDataset,
    ShardBlockSampler,
    ShardHomogeneousBatchSampler,
    collate_fn,
)
from copula_inter.era5_live_dataset import build_era5_fixed_val_batches, build_era5_train_loader
from copula_inter.gp_kernels import _COMPOSABLE_KERNELS
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
    load_tabicl,
    resolve_pit_ckpt,
)
from copula_inter.training_core import (
    cosine_lr_lambda,
)

# Peak dense FP16/BF16 tensor-core TFLOPS by device-name substring, for MFU.
# First match wins, so longer names ("H100 PCIE", "L40S") precede their
# prefixes. Pre-Volta entries are CUDA-core peaks.
_GPU_PEAK_TFLOPS: dict[str, float] = {
    "H100 PCIE": 756e12,
    "H100": 989e12,  # SXM/HBM3/bare "H100" — no PCIe suffix in the name
    "RTX PRO 6000 BLACKWELL": 1021e12,  # vercors18 — estimate, not verified against a datasheet
    "A100": 312e12,
    "V100": 125e12,
    "RTX 4090": 165e12,
    "RTX 3090": 142e12,
    "RTX A6000": 130e12,
    "RTX A5000": 111e12,  # vercors9/10
    "RTX 6000 ADA": 728e12,  # vercors14/15 — this training node's GPU
    "RTX 5000 ADA": 522.2e12,  # not yet confirmed on a specific Grid5000 node -- derived
    # Datasheet 1044.4 / 2, same convention as RTX 6000 ADA.
    "L40S": 362e12,  # kinovis, vercors17 — must precede "L4"
    "L4": 121e12,  # vercors16
    "QUADRO RTX 8000": 130.5e12,  # vercors5/8/11
    "TITAN RTX": 130.5e12,  # vercors4/7/12 — same TU102 die as Quadro RTX 8000
    "P100": 21.2e12,  # drac — Pascal, no Tensor Cores: CUDA-core FP16 peak
    "TITAN XP": 12.15e12,  # vercors3 — Pascal, no Tensor Cores: CUDA-core FP32 peak
    "TITAN X (PASCAL)": 10.97e12,  # vercors2 — Pascal, no Tensor Cores: CUDA-core FP32 peak
    "C2050": 1.03e12,  # adonis (Fermi) — no Tensor Cores: CUDA-core FP32 peak
    "C1060": 0.933e12,  # adonis (Tesla 10-series) — no Tensor Cores: CUDA-core FP32 peak
}
_GPU_PEAK_FLOPS_DEFAULT = 100e12

# Fixed validation episodes: pre-built batches (live data) or a DataLoader over a dataset dir.
ValLoader: TypeAlias = "list[dict] | DataLoader"


@dataclass
class ModelBundle:
    """The copula model with its optimizer, LR schedule, AMP settings and resume step."""

    model: CopulaTabICL
    optimizer: Muon
    scheduler: torch.optim.lr_scheduler.LambdaLR
    scaler: GradScaler | None
    trainable: list[nn.Parameter]
    start_step: int
    use_amp: bool
    amp_dtype: torch.dtype


@dataclass
class TrainData:
    """Training iterator and validation loader, plus the shared tensors adaptive sampling updates in place."""

    train_loader: DataLoader
    # Already-started iterator (live generation kicks it off early), else None.
    train_iter: Iterator[Any] | None
    val_loader: ValLoader
    # Raw val episodes with kernel metadata, per batch; only for live GP generation.
    val_episodes_meta: dict[int, list[dict]] | None
    adaptive_kernel_weights: torch.Tensor | None
    tabicl_mix_weights: torch.Tensor | None


@dataclass
class ValidationProbes:
    """Fixed validation probes (synthetic kernel families, posterior probes, ERA5) and their precomputed z."""

    synth_kernel_batches: dict[str, dict]
    posterior_probe: dict | None
    pit_ckpt: str | None
    tabicl_val_z: dict
    analytic_val_z: dict
    tabicl_kernel_fit_z: dict
    era5_val_batches: dict
    era5_viz_batch: dict | None


def resolve_train_device(cfg: DictConfig) -> tuple[str, float | None]:
    """Resolve training.device; on CUDA enable TF32 matmuls and return the GPU's peak FLOPs for MFU."""
    device = (
        "cuda"
        if cfg.training.device == "auto" and torch.cuda.is_available()
        else ("cpu" if cfg.training.device == "auto" else cfg.training.device)
    )
    gpu_peak_flops: float | None = None
    if device == "cuda":
        gpu_peak_flops = get_gpu_peak_flops()
        # Allow TF32 matmuls (used by Muon's Newton-Schulz and the NLL's fp32 algebra).
        torch.set_float32_matmul_precision("high")
        print(
            f"[train] GPU: {torch.cuda.get_device_name(0)} — assumed peak "
            f"{gpu_peak_flops / 1e12:.0f} TFLOPS (dense bf16/fp16 tensor core) for MFU"
        )
    return device, gpu_peak_flops


def dataset_name(t: DictConfig, data_cfg: DictConfig, live_generation: bool) -> str:
    """Run-name prefix: live runs are named from cfg.data and ckpt_dir (dataset_dir is unused), disk runs from dataset_dir."""
    if live_generation:
        ckpt_dir = t.get("ckpt_dir", None)
        ckpt_str = f"_ckpt-{os.path.basename(os.path.normpath(ckpt_dir))}" if ckpt_dir else ""
        return "live" + _live_data_segment(data_cfg) + ckpt_str
    dataset_path = os.path.normpath(t.dataset_dir)
    # Include the parent folder in the dataset name.
    parent_name = os.path.basename(os.path.dirname(dataset_path))
    shard_name = os.path.basename(dataset_path)
    return f"{parent_name}/{shard_name}" if parent_name else shard_name


def get_gpu_peak_flops(device: int = 0) -> float:
    """Peak dense FP16/BF16 tensor-core FLOPs of the current GPU, from _GPU_PEAK_TFLOPS (default with a warning)."""
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
    """Cap this process's CUDA memory fraction when live generation runs GPU workers on the same card.

    Reserves the workers' estimated VRAM (resolve_live_tabicl_num_workers x the
    per-worker estimate) so this process OOMs into the training loop's handler
    instead of starving the workers. Must run before training allocations.
    """
    z_train_source = z_train_source_of(cfg)
    _validate_z_train_source(z_train_source)
    mix_enabled = bool(cfg.data.get("z_train_tabicl_mix_enabled", False))
    batched_marginal_worker_enabled = (
        mix_enabled or z_train_source in TABICL_Z_TRAIN_SOURCES or z_train_source in _GENERIC_MARGINAL_BACKENDS
    )
    if not batched_marginal_worker_enabled or device != "cuda":
        return
    # Same worker count as build_live_train_loader.
    num_workers = resolve_live_tabicl_num_workers(t, device)
    if num_workers <= 0:
        return
    group_multiplier = max(1, int(t.get("live_tabicl_group_multiplier", 2)))
    group_size = int(t.batch_size) * group_multiplier
    per_worker_gb = _LIVE_TABICL_WORKER_FIXED_OVERHEAD_GB + _LIVE_TABICL_WORKER_PER_EPISODE_GB * group_size
    headroom_gb = num_workers * per_worker_gb + _LIVE_TABICL_FLAT_HEADROOM_GB
    total_b = torch.cuda.get_device_properties(0).total_memory
    fraction = 1.0 - (headroom_gb * 1e9) / total_b
    # Clamp the fraction to [0.5, 0.97].
    fraction = max(0.5, min(fraction, 0.97))
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    print(
        f"[train] live-generation TabICL workers ({num_workers}) run their own "
        f"CUDA context on this GPU -- capping this process's own VRAM use to "
        f"{fraction * 100:.0f}% (~{headroom_gb:.1f} GB reserved as worker "
        "headroom) so its allocator can't silently starve them; any excess is "
        "handled by the existing per-step OOM-skip path."
    )


def _fmt_run_value(value: object) -> str:
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
    """Run-name segment summarizing cfg.data's composition, correlation and warp settings for live generation."""
    if bool(data_cfg.get("systematic_composition", False)):
        lo = data_cfg.get("composite_num_kernels_min", 1)
        hi = data_cfg.get("composite_num_kernels_max", 1)
        kernel_str = f"syscomp{lo}-{hi}"
    elif data_cfg.get("kernels", None) is not None:
        kernel_str = f"mix{len(data_cfg.kernels)}"
    else:
        kernel_str = str(data_cfg.get("kernel", "rbf"))

    dfeat_str = (
        "logN" if data_cfg.get("d_features_lognormal_loc", None) is not None else str(data_cfg.get("d_features", 10))
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


def build_model_and_optimizer(cfg: DictConfig, device: str, resume_ckpt: str | None, t: DictConfig) -> ModelBundle:
    """Build the copula model, Muon/AdamW optimizer, AMP scaler and LR schedule; resume from training.resume_ckpt if set."""
    model = build_copula_transformer(cfg).to(device)
    if bool(t.get("compile", False)):
        torch._dynamo.config.capture_scalar_outputs = True
        # The compiled wrapper forwards attribute access to the module.
        model = cast("CopulaTabICL", torch.compile(model, dynamic=True))
    wandb.watch(model, log="gradients", log_freq=5000)

    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_train_params:,}")
    wandb.config.update({"n_trainable_params": n_train_params})

    trainable = [p for p in model.parameters() if p.requires_grad]
    muon_params = [p for p in trainable if p.ndim >= 2]
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

    # Resume weights and optimizer/scaler state. By default the cosine schedule
    # continues from the checkpoint's step; training.resume_reset_schedule=true
    # restarts warmup at step 0.
    start_step = 0
    if resume_ckpt:
        ckpt_step = load_checkpoint(resume_ckpt, model, device, optimizer=optimizer, scaler=scaler)
        if bool(t.get("resume_reset_schedule", False)):
            print(
                f"Resumed weights + optimizer/scaler state from {resume_ckpt} (step {ckpt_step}) — resetting to step 0 with a fresh warmup/cosine schedule (resume_reset_schedule=true)"
            )
        else:
            start_step = ckpt_step
            print(
                f"Resumed weights + optimizer/scaler state from {resume_ckpt} — continuing cosine schedule from step {start_step}"
            )

    if start_step > 0:
        # LambdaLR needs initial_lr to resume at a non-zero step; use this run's base LR.
        for group in optimizer.param_groups:
            group["initial_lr"] = t.muon_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_lr_lambda(s, t.warmup_steps, t.steps, lr_min_frac),
        last_epoch=start_step - 1 if start_step > 0 else -1,
    )
    return ModelBundle(model, optimizer, scheduler, scaler, trainable, start_step, use_amp, amp_dtype)


def build_validation_probes(
    cfg: DictConfig,
    device: str,
    t: DictConfig,
    tabicl_mix_weights: torch.Tensor | None,
    val_episodes_meta: dict[int, list[dict]] | None,
    val_loader: ValLoader,
) -> ValidationProbes:
    """Build the fixed validation probes (synthetic kernel families, posterior probes, ERA5) and their TabICL/analytic z_train."""
    baselines_on = bool(cfg.get("baselines", {}).get("enabled", True))
    synth_kernel_batches = _build_synthetic_kernel_batches(cfg, device) if baselines_on else {}
    # Fallback posterior probe when the val loader has no kernel metadata (disk or ERA5).
    posterior_probe = (
        _build_posterior_probe_batches(cfg, device) if (baselines_on and val_episodes_meta is None) else None
    )

    # Frozen TabICL marginal (quantile head intact) for PIT-ing validation episodes; see pit.resolve_pit_ckpt.
    pit_ckpt = resolve_pit_ckpt(cfg)
    tabicl_val_z: dict = {}
    tabicl_kernel_fit_z: dict = {}
    # Analytic z for the oracle_diag/* metrics when the val batches use a TabICL PIT.
    analytic_val_z: dict = {}
    if val_episodes_meta is not None and z_train_source_of(cfg) != "analytic":
        print(
            "[train] Building the exact analytic-GP PIT cache for oracle_diag/* "
            f"(data.z_train_source={z_train_source_of(cfg)} puts the val "
            "batches in TabICL's z-space)..."
        )
        analytic_val_z = _build_analytic_val_z(val_loader, val_episodes_meta, device)
    # ERA5 probes, built while tabicl_marginal is loaded. Without pit_ckpt the
    # branch below uses tabicl.ckpt directly.
    era5_val_batches: dict = {}
    era5_viz_batch: "dict | None" = None
    era5_on = baselines_on and bool(cfg.get("baselines", {}).get("era5_enabled", True))
    # Only built when plotting is enabled.
    era5_viz_on = era5_on and int(t.get("plot_val_every", 5000)) > 0
    # The TabICL mix measurement needs tabicl_marginal even without baselines.
    if (baselines_on or tabicl_mix_weights is not None) and pit_ckpt:
        print("[train] Loading frozen TabICL marginal for the z_train sim-to-real diagnostic...")
        tabicl_marginal = load_tabicl(pit_ckpt, device)
        pit_k_folds = int(cfg.tabicl.get("pit_k_folds", DEFAULT_K_FOLDS))
        # Score val y_nll_* with the fold count the val episodes were generated with
        # (data.z_train_tabicl_k_folds) under a TabICL z_train source; otherwise
        # tabicl.pit_k_folds.
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
            tabicl_kernel_fit_z = _build_tabicl_kernel_fit_z(synth_kernel_batches, tabicl_marginal, pit_k_folds, device)
        if tabicl_mix_weights is not None:
            floor_frac = float(cfg.data.get("z_train_tabicl_mix_floor_frac", 0.05))
            max_frac = float(cfg.data.get("z_train_tabicl_mix_max_frac", 0.35))
            if math.isclose(floor_frac, max_frac, abs_tol=1e-12):
                # floor == max makes every family's fraction the floor; skip the measurement.
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
                # In-place update of the shared-memory tensor the workers read.
                tabicl_mix_weights.copy_(new_mix_frac)
                for family, gap in sorted(z_gap.items(), key=lambda kv: -kv[1]):
                    idx = _COMPOSABLE_KERNELS.index(family)
                    print(
                        f"[train]   {family}: z_train_tabicl_gap={gap:.3f} -> mix_frac={float(new_mix_frac[idx]):.3f}"
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
            # gc.collect() before empty_cache() so cyclic references release CUDA memory.
            gc.collect()
            torch.cuda.empty_cache()
    elif era5_on:
        # No pit_ckpt: use tabicl.ckpt for the ERA5 probes if set.
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
    return ValidationProbes(
        synth_kernel_batches=synth_kernel_batches,
        posterior_probe=posterior_probe,
        pit_ckpt=pit_ckpt,
        tabicl_val_z=tabicl_val_z,
        analytic_val_z=analytic_val_z,
        tabicl_kernel_fit_z=tabicl_kernel_fit_z,
        era5_val_batches=era5_val_batches,
        era5_viz_batch=era5_viz_batch,
    )


def init_wandb_run(cfg: DictConfig, dataset_name: str, t: DictConfig) -> str | None:
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
    run_name = f"{dataset_name}{model_hparams}{training_hparams}_unfreeze={unfreeze}{lora_str}{resume_str}"
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity if cfg.wandb.entity else None,
        name=run_name,
        config=config_dict(cfg),
    )
    return resume_ckpt


def build_data_loaders(
    cfg: DictConfig, device: str, live_generation: bool, live_source: str, t: DictConfig
) -> TrainData:
    """Build the training iterator and validation loader for live (GP or ERA5) or on-disk data."""
    adaptive_kernel_weights = None  # set below only when live_generation + adaptive_kernel_sampling
    tabicl_mix_weights = None  # set below only when live_generation + data.z_train_tabicl_mix_enabled
    train_iter = None  # possibly kicked off early below (live_generation only) -- see there
    # Raw val episodes with kernel metadata, per batch; only for live GP generation.
    val_episodes_meta: dict[int, list[dict]] | None = None
    val_loader: ValLoader
    train_dataset: Subset[Any] | CopulaDataset
    val_dataset: Subset[Any] | CopulaDataset
    if live_generation:
        # Live generation: episodes are generated by DataLoader workers (live_dataset.py).
        print(
            "[train] live_generation=true — generating episodes on the fly, "
            f"no dataset_dir read ({t.dataset_dir!r} ignored). "
            f"ckpt_dir={t.get('ckpt_dir', None)!r} live_source={live_source!r}"
        )
        if live_source == "era5":
            # Real ERA5 episodes; no adaptive kernel sampling or TabICL mix.
            train_loader = build_era5_train_loader(cfg, t, device)
            val_loader = build_era5_fixed_val_batches(cfg, t, device)
        else:
            train_loader, adaptive_kernel_weights, tabicl_mix_weights = build_live_train_loader(cfg, t, device)
            val_loader, val_episodes_by_batch = build_fixed_live_val_batches(cfg, t, device)
            val_episodes_meta = dict(enumerate(val_episodes_by_batch))
        # Reserve worker headroom after the val batches are built (their one-time
        # marginal forward needs the uncapped GPU) and before any worker starts.
        _reserve_gpu_headroom_for_live_tabicl(cfg, t, device)
        print(f"Train: <live> | Val: {len(val_loader) * t.batch_size} episodes (fixed)")
        # Start the workers now so their model loading overlaps the setup below,
        # unless the TabICL mix weights still have to be written first.
        if tabicl_mix_weights is None:
            train_iter = iter(train_loader)
    else:
        meta_path = os.path.join(t.dataset_dir, "meta.pt")
        shard_files = sorted(glob(os.path.join(t.dataset_dir, "shard_*.pt")))

        train_sampler = None
        train_batch_sampler = None
        val_batch_sampler = None
        variable_d = False
        loader_num_workers_override = t.get("loader_num_workers", None)
        loader_num_workers = int(loader_num_workers_override) if loader_num_workers_override is not None else 4
        # Batches queued per worker (conf/config.yaml sets 8).
        prefetch_factor_override = t.get("prefetch_factor", None)
        prefetch_factor = int(prefetch_factor_override) if prefetch_factor_override is not None else 8
        if shard_files and os.path.exists(meta_path):
            shard_block_shards = int(t.get("shard_block_shards", 16))
            # Cache a full shard block (+4 for workers straddling blocks).
            full_dataset = CopulaDataset(episode_dir=t.dataset_dir, shard_cache_size=shard_block_shards + 4)
            n = len(full_dataset)
            n_val = min(int(t.get("val_episodes", 500)), n)
            # Stride validation indices across the dataset so val spans many shards.
            val_indices = sorted(set(int(i) for i in torch.linspace(0, n - 1, n_val)))
            val_set = set(val_indices)
            train_indices = [i for i in range(n) if i not in val_set]
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)

            # Detect datasets whose d_features varies per shard.
            shard_size = full_dataset.shard_size
            n_shards = (n + shard_size - 1) // shard_size
            probe_ids = torch.randperm(n_shards)[:8].tolist()
            d_seen = {int(full_dataset[min(sid * shard_size, n - 1)]["x_norm_train"].shape[-1]) for sid in probe_ids}
            variable_d = len(d_seen) > 1

            if variable_d:
                # Variable d: batch within one shard (ShardHomogeneousBatchSampler), with a
                # small shard cache per worker.
                full_dataset._SHARD_CACHE_SIZE = min(full_dataset._SHARD_CACHE_SIZE, 2)
                # Clear the cache so the smaller cap applies (the LRU never shrinks on its own).
                full_dataset._shard_cache.clear()
                # Loader workers for variable-d data (training.loader_num_workers; shards are mmap-loaded).
                loader_num_workers = int(loader_num_workers_override) if loader_num_workers_override is not None else 4
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
                # Fixed d: shuffle at shard-block granularity (ShardBlockSampler) for cache locality.
                train_sampler = ShardBlockSampler(
                    train_dataset.indices,
                    shard_size=shard_size,
                    block_shards=shard_block_shards,
                )
        else:
            all_files = sorted(glob(os.path.join(t.dataset_dir, "task_*.pt")))
            if not all_files:
                raise RuntimeError(f"No episode files in {t.dataset_dir}. Run generate_pit_dataset.py first.")
            n_val = min(int(t.get("val_episodes", 500)), len(all_files))
            train_dataset = CopulaDataset(file_list=all_files[n_val:])
            val_dataset = CopulaDataset(file_list=all_files[:n_val])

        print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} episodes")

        # Training never reads R_prior; drop it before collate_fn to skip the copy.
        def _train_collate_fn(samples: list[dict]) -> dict:
            for s in samples:
                s.pop("R_prior", None)
            return collate_fn(samples)

        # batch_sampler is exclusive with batch_size/sampler/shuffle.
        train_batching: dict[str, Any] = (
            {"batch_sampler": train_batch_sampler}
            if train_batch_sampler is not None
            else {"batch_size": t.batch_size, "sampler": train_sampler, "shuffle": train_sampler is None}
        )
        val_batching: dict[str, Any] = (
            {"batch_sampler": val_batch_sampler}
            if val_batch_sampler is not None
            else {"batch_size": t.batch_size, "shuffle": False}
        )
        train_loader = DataLoader(
            train_dataset,
            collate_fn=_train_collate_fn,
            num_workers=loader_num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            **train_batching,
        )
        val_loader = DataLoader(
            val_dataset,
            collate_fn=_train_collate_fn,
            num_workers=loader_num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            **val_batching,
        )
    return TrainData(
        train_loader, train_iter, val_loader, val_episodes_meta, adaptive_kernel_weights, tabicl_mix_weights
    )
