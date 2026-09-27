"""Train the copula model on the Y-space NLL (Sklar's theorem).

    L = copula NLL(z_test; Sigma) + marginal NLL(y_test; marginal log-pdf)

Sigma is built by model.build_sigma from the model output.

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
from typing import TYPE_CHECKING, Any

from copula_inter.adaptive_sampling import (
    _refresh_tabicl_mix_weights,
    _update_adaptive_kernel_weights,
)
from copula_inter.checkpointing import save_checkpoint
from copula_inter.probe_batches import (
    _sigma_stats,
)
from copula_inter.train_setup import (
    ModelBundle,
    TrainData,
    ValidationProbes,
    build_data_loaders,
    build_model_and_optimizer,
    build_validation_probes,
    dataset_name,
    init_wandb_run,
    prepare_training_inputs,
    resolve_train_device,
)
from copula_inter.validation import validate

if TYPE_CHECKING:
    from copula_inter.model import CopulaTabICL

# Batch shapes vary with P and N; expandable segments reduce allocator
# fragmentation. Must be set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import wandb
from omegaconf import DictConfig

from copula_inter.config_path import config_dir
from copula_inter.gp_kernels import _COMPOSABLE_KERNELS
from copula_inter.pit import (
    configure_tabicl_inference_amp,
)
from copula_inter.training_core import (
    _measure_step_flops,
    _run_train_step,
)

_EMA_ALPHA = 0.98


class _PhaseTimer:
    """Per-phase step timings: CUDA events for GPU phases (read at log steps), wall time for data and on CPU."""

    PHASES = ("forward", "loss", "backward_step")

    def __init__(self, device: str) -> None:
        self.device = device
        self.ms = {k: 0.0 for k in ("data",) + self.PHASES}
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
            {k: [] for k in self.PHASES} if device == "cuda" else {}
        )
        self.n = 0
        self.T_sum = 0  # sum of per-step sequence length T=P+N, for MFU's avg batch shape
        # This step's phase times (CPU path only).
        self.last_ms = {k: 0.0 for k in self.PHASES}
        self.last_log_wall = time.perf_counter()
        self.last_log_step = 0

    def start(self) -> Any:
        if self.device == "cuda":
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return ev
        return time.perf_counter()

    def end(self, name: str, start: Any) -> None:
        if self.device == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.events[name].append((start, end))
        else:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.ms[name] += elapsed_ms
            self.last_ms[name] = elapsed_ms

    def drop_events(self) -> None:
        """Forget the CUDA events of a failed step."""
        for events in self.events.values():
            events.clear()

    def readout(self, step: int) -> dict[str, Any]:
        """Window averages since the previous readout plus this step's own phase times (one sync); resets the window."""
        last_step_ms = dict(self.last_ms)  # CPU fallback; overwritten below on CUDA
        if self.device == "cuda" and self.n > 0:
            torch.cuda.synchronize()
            for name in self.PHASES:
                elapsed = [s.elapsed_time(e) for s, e in self.events[name]]
                self.ms[name] += sum(elapsed)
                last_step_ms[name] = elapsed[-1] if elapsed else 0.0
                self.events[name].clear()
        now = time.perf_counter()
        steps_done = max(step - self.last_log_step, 1)
        out = {
            "last_step_ms": last_step_ms,
            "step_ms": {k: v / self.n for k, v in self.ms.items()} if self.n else {k: 0.0 for k in self.ms},
            "avg_T": self.T_sum / self.n if self.n else 0,
            "wall_step_ms": (now - self.last_log_wall) / steps_done * 1000.0,
            "steps_per_sec": steps_done / max(now - self.last_log_wall, 1e-9),
        }
        self.last_log_wall = now
        self.last_log_step = step
        for k in self.ms:
            self.ms[k] = 0.0
        self.n = 0
        self.T_sum = 0
        return out


def _memory_pcts(device: str) -> tuple[float, float, float]:
    """Allocated, reserved and peak-since-last-call GPU memory as a share of device capacity."""
    if device != "cuda":
        return 0.0, 0.0, 0.0
    _free_b, total_b = torch.cuda.mem_get_info()
    alloc = 100.0 * torch.cuda.memory_allocated() / total_b
    reserved = 100.0 * torch.cuda.memory_reserved() / total_b
    # Reset the peak so each log line shows the peak since the previous one.
    peak = 100.0 * torch.cuda.max_memory_allocated() / total_b
    torch.cuda.reset_peak_memory_stats()
    return alloc, reserved, peak


def _log_train_step(
    step: int,
    *,
    out: dict[str, torch.Tensor],
    Sigma: Any,
    parts: dict[str, torch.Tensor],
    loss: torch.Tensor,
    aux_mae: torch.Tensor,
    grad_norm: torch.Tensor,
    test_mask: torch.Tensor,
    loss_ema: float | None,
    bundle: ModelBundle,
    timer: _PhaseTimer,
    step_flops: float | None,
    gpu_peak_flops: float | None,
    batch_size: int,
    aux_mae_weight: float,
    device: str,
) -> float:
    """Log and print one training step's losses, Sigma statistics and performance; returns the updated loss EMA."""
    loss_val = loss.item()
    loss_ema = loss_val if loss_ema is None else _EMA_ALPHA * loss_ema + (1.0 - _EMA_ALPHA) * loss_val
    grad_norm_val = float(grad_norm)
    lr_now = bundle.scheduler.get_last_lr()[0]
    amp_scale = bundle.scaler.get_scale() if bundle.scaler is not None else 1.0
    cop_val = parts["copula"].item()
    mar_val = parts["marginal"].item()
    aux_mae_val = aux_mae.item()
    with torch.no_grad():
        w_norm_mean = float(out["W"].float().norm(dim=-1).mean().item())
        Sigma_dense = Sigma.dense()
        sig_stats = _sigma_stats(Sigma_dense, test_mask)
        # Count batches where _safe_cholesky replaced a non-finite slice.
        sigma_nonfinite = int((~torch.isfinite(Sigma_dense).flatten(1).all(-1)).sum().item())
        del Sigma_dense

    perf = timer.readout(step)
    step_ms = perf["step_ms"]
    # MFU: this step's measured FLOPs over this step's forward+loss+backward time.
    iter_time_sec = sum(perf["last_step_ms"].values()) / 1000.0
    flops_per_iter = step_flops or 0.0
    if iter_time_sec > 0:
        actual_flops_per_sec = flops_per_iter / iter_time_sec
        tokens_per_sec = (batch_size * perf["avg_T"]) / iter_time_sec
    else:
        actual_flops_per_sec = 0.0
        tokens_per_sec = 0.0
    mfu_pct = 100.0 * actual_flops_per_sec / gpu_peak_flops if (gpu_peak_flops and iter_time_sec > 0) else 0.0
    mem_alloc_pct, mem_reserved_pct, mem_peak_pct = _memory_pcts(device)

    wandb.log(
        {
            "train/y_nll_total": loss_val,
            "train/y_nll_copula": cop_val,
            "train/y_nll_marginal": mar_val,
            "train/aux_mae": aux_mae_val,
            "train/lr": lr_now,
            "train/grad_norm": grad_norm_val,
            "train/amp_scale": amp_scale,
            "train/loss_ema": loss_ema,
            "train/W_norm_mean": w_norm_mean,
            "train/sigma_offdiag_mean": sig_stats["offdiag_mean"],
            "train/sigma_nonfinite_count": sigma_nonfinite,
            "perf/step_ms": perf["wall_step_ms"],
            "perf/steps_per_sec": perf["steps_per_sec"],
            "perf/data_ms": step_ms["data"],
            "perf/forward_ms": step_ms["forward"],
            "perf/loss_ms": step_ms["loss"],
            "perf/backward_step_ms": step_ms["backward_step"],
            "perf/mem_allocated_pct": mem_alloc_pct,
            "perf/mem_reserved_pct": mem_reserved_pct,
            "perf/mem_peak_pct": mem_peak_pct,
            "perf/mfu_pct": mfu_pct,
            "perf/tokens_per_sec": tokens_per_sec,
            "perf/iter_time_sec": iter_time_sec,
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
        f"         perf: step={perf['wall_step_ms']:.1f}ms ({perf['steps_per_sec']:.2f} it/s) "
        f"data={step_ms['data']:.1f} fwd={step_ms['forward']:.1f} "
        f"loss={step_ms['loss']:.1f} bwd+opt={step_ms['backward_step']:.1f} "
        f"mem={mem_alloc_pct:.1f}%/{mem_reserved_pct:.1f}% (peak {mem_peak_pct:.1f}%) "
        f"mfu={mfu_pct:.1f}% tok/s={tokens_per_sec:,.0f}"
    )
    return loss_ema


def _fmt_metric(metrics: dict[str, float], key: str, spec: str) -> str:
    value = metrics.get(key, float("nan"))
    return format(value, spec) if math.isfinite(value) else "n/a"


def _update_kernel_sampling(
    cfg: DictConfig, adaptive_kernel_weights: torch.Tensor, metrics: dict[str, float], log_dict: dict[str, Any]
) -> None:
    """Move the adaptive kernel-family sampling weights toward the families validation scores worst."""
    t = cfg.training
    excluded_kernels = set(getattr(cfg.data, "composite_exclude_kernels", None) or [])
    new_kernel_weights = _update_adaptive_kernel_weights(
        adaptive_kernel_weights,
        metrics,
        float(t.get("adaptive_kernel_lr", 1.0)),
        float(t.get("adaptive_kernel_floor", 0.05)),
        exclude=excluded_kernels,
        signal=str(t.get("adaptive_kernel_signal", "tabicl")),
    )
    # In-place update of the shared-memory tensor the workers read.
    adaptive_kernel_weights.copy_(new_kernel_weights)
    # Excluded families are never sampled; don't log their weights.
    for i, family in enumerate(_COMPOSABLE_KERNELS):
        if family in excluded_kernels:
            continue
        log_dict[f"val/kernel_sampling_weight/{family}"] = float(new_kernel_weights[i])


def _validate_and_log(
    step: int, cfg: DictConfig, device: str, model: CopulaTabICL, data: TrainData, probes: ValidationProbes, lr: float
) -> None:
    """Run validation, adapt kernel sampling weights if enabled, and log/print the metrics."""
    plot_val_every = int(cfg.training.get("plot_val_every", 5000))
    do_plot = plot_val_every > 0 and step % plot_val_every == 0
    metrics, plot_figs = validate(
        model,
        data.val_loader,
        cfg,
        device,
        step=step,
        do_plot=do_plot,
        synth_kernel_batches=probes.synth_kernel_batches,
        tabicl_val_z=probes.tabicl_val_z,
        analytic_val_z=probes.analytic_val_z,
        tabicl_kernel_fit_z=probes.tabicl_kernel_fit_z,
        era5_val_batches=probes.era5_val_batches,
        era5_viz_batch=probes.era5_viz_batch,
        posterior_probe=probes.posterior_probe,
        val_episodes_meta=data.val_episodes_meta,
    )
    # oracle_diag/* keys are logged as-is; others get the val/ prefix.
    log_dict: dict[str, Any] = {(k if k.startswith("oracle_diag/") else f"val/{k}"): v for k, v in metrics.items()}
    if data.adaptive_kernel_weights is not None:
        _update_kernel_sampling(cfg, data.adaptive_kernel_weights, metrics, log_dict)
    if plot_figs:
        # ERA5 diagnostic figures, keyed by panel name.
        for key, fig in plot_figs.items():
            log_dict[key] = wandb.Image(fig)
            plt.close(fig)
    wandb.log(log_dict, step=step)
    # cop_gap = copula gap / total available copula improvement (gap > headroom
    # means worse than independence). corr_kl is the noise-free KL(R_post || Sigma).
    cop_gap = metrics.get("oracle_diag/copula_gap", float("nan"))
    headroom = metrics.get("oracle_diag/copula_headroom", float("nan"))
    cop_gap_str = f"{cop_gap:.4f}/{headroom:.4f}" if math.isfinite(cop_gap) and math.isfinite(headroom) else "n/a"
    print(
        f"[{step:6d}] VAL  "
        f"total={_fmt_metric(metrics, 'y_nll_total', '.4f')}  "
        f"gap_post={_fmt_metric(metrics, 'oracle_diag/gap_nll', '.4f')}  "
        f"corr_r={_fmt_metric(metrics, 'oracle_diag/corr_pearson', '.3f')}  "
        f"od_μ={metrics['sigma_offdiag_mean_analytic_z']:+.4f} od_σ={metrics['sigma_offdiag_std_analytic_z']:.4f} od_|r|={metrics['sigma_offdiag_abs_mean_analytic_z']:.4f}  "
        f"cop_std={_fmt_metric(metrics, 'oracle_diag/copula_nll_std', '.4f')}  "
        f"cop_tabicl={_fmt_metric(metrics, 'y_nll_copula', '.4f')}  "
        f"cop_gap={cop_gap_str}  "
        f"corr_kl={_fmt_metric(metrics, 'oracle_diag/corr_kl', '.4f')}  "
        f"lr={lr:.2e}"
    )


def _refresh_tabicl_mix(
    step: int, cfg: DictConfig, pit_ckpt: str, tabicl_mix_weights: torch.Tensor, device: str
) -> None:
    """Re-measure the per-family TabICL z_train gap and update the shared TabICL-mix fractions."""
    z_gap, new_mix_frac = _refresh_tabicl_mix_weights(cfg, pit_ckpt, tabicl_mix_weights, device)
    print(f"[train][step {step}] Re-measured z_train_tabicl_mix_* (adaptive):")
    save_log = {}
    for i, family in enumerate(_COMPOSABLE_KERNELS):
        if family in z_gap:
            save_log[f"val/z_train_tabicl_gap/{family}"] = z_gap[family]
            print(
                f"[train]   {family}: z_train_tabicl_gap={z_gap[family]:.3f} -> mix_frac={float(new_mix_frac[i]):.3f}"
            )
        save_log[f"val/tabicl_mix_frac/{family}"] = float(new_mix_frac[i])
    wandb.log(save_log, step=step)


@hydra.main(config_path=config_dir(__file__), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    resume_ckpt = prepare_training_inputs(cfg)
    torch.manual_seed(cfg.seed)
    device, gpu_peak_flops = resolve_train_device(cfg)

    t = cfg.training
    tabicl_amp = bool(t.get("tabicl_inference_amp", True))
    configure_tabicl_inference_amp(tabicl_amp)
    print(f"[train] frozen TabICL marginal inference AMP={'on' if tabicl_amp else 'off (float32)'}")
    live_generation = bool(t.get("live_generation", False))
    live_source = str(t.get("live_source", "gp"))
    if live_generation and live_source == "era5" and float(t.get("aux_mae_weight", 0.0)) > 0.0:
        # Real ERA5 has no oracle R_star.
        print("[train] live_source=era5: forcing training.aux_mae_weight=0.0 (real data has no oracle R_star)")
        t.aux_mae_weight = 0.0
    init_wandb_run(cfg=cfg, dataset_name=dataset_name(t, cfg.data, live_generation), t=t)

    data = build_data_loaders(cfg=cfg, device=device, live_generation=live_generation, live_source=live_source, t=t)
    probes = build_validation_probes(
        cfg=cfg,
        device=device,
        t=t,
        tabicl_mix_weights=data.tabicl_mix_weights,
        val_episodes_meta=data.val_episodes_meta,
        val_loader=data.val_loader,
    )
    bundle = build_model_and_optimizer(cfg=cfg, device=device, resume_ckpt=resume_ckpt, t=t)
    model, optimizer, scheduler, scaler = bundle.model, bundle.optimizer, bundle.scheduler, bundle.scaler

    jitter = float(cfg.model.get("sigma_jitter", 1e-4))
    parametrization = str(cfg.model.get("correlation_parametrization", "covnorm"))
    nll_weight = float(t.get("nll_weight", 1.0))
    aux_mae_weight = float(t.get("aux_mae_weight", 0.0))
    # Weight of the backbone's MoE auxiliary loss (TabLDM only).
    moe_aux_weight = float(t.get("moe_aux_weight", 1.0))
    adaptive_mix = bool(cfg.data.get("z_train_tabicl_mix_adaptive", False))

    model.train()
    # Recreate the iterator on StopIteration (not itertools.cycle) so every epoch reshuffles.
    train_iter = data.train_iter if data.train_iter is not None else iter(data.train_loader)
    loss_ema: float | None = None
    triu_cache: dict[int, torch.Tensor] = {}
    timer = _PhaseTimer(device)

    for step in range(bundle.start_step, t.steps + 1):
        t_data0 = time.perf_counter()
        # Pre-clear loop references so an OOM can release them.
        batch = None
        out = Sigma = parts = loss = aux_mae = grad_norm = None
        step_flops = None
        try:
            # Keep the CPU batch separate so a transfer OOM is recoverable.
            raw_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(data.train_loader)
            raw_batch = next(train_iter)
        except torch.cuda.OutOfMemoryError as exc:
            # An OOM in a GPU generation worker surfaces here; recreate the iterator so
            # the worker restarts, and skip the step.
            print(f"[{step:6d}] CUDA OOM in a live-generation DataLoader worker — recreating iterator, skipping step.")
            traceback.clear_frames(exc.__traceback__)
            del exc
            gc.collect()
            torch.cuda.empty_cache()
            train_iter = iter(data.train_loader)
            continue

        optimizer.zero_grad(set_to_none=True)
        try:
            # non_blocking transfer (pin_memory=True).
            batch = {k: v.to(device, non_blocking=True) for k, v in raw_batch.items()}
            timer.ms["data"] += (time.perf_counter() - t_data0) * 1000.0
            timer.T_sum += batch["x_train"].shape[1] + batch["x_test"].shape[1]

            out, Sigma, parts, loss, aux_mae, grad_norm = _run_train_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                trainable=bundle.trainable,
                batch=batch,
                device=device,
                use_amp=bundle.use_amp,
                amp_dtype=bundle.amp_dtype,
                scaler=scaler,
                clip_grad_norm=t.clip_grad_norm,
                nll_weight=nll_weight,
                aux_mae_weight=aux_mae_weight,
                jitter=jitter,
                triu_cache=triu_cache,
                phase_start=timer.start,
                phase_end=timer.end,
                parametrization=parametrization,
                moe_aux_weight=moe_aux_weight,
            )
            # At log steps, count this step's FLOPs with FlopCounterMode in a separate
            # throwaway forward+backward (grads discarded) for MFU.
            if step % t.log_every == 0:
                try:
                    step_flops = _measure_step_flops(
                        model=model,
                        batch=batch,
                        device=device,
                        use_amp=bundle.use_amp,
                        amp_dtype=bundle.amp_dtype,
                        nll_weight=nll_weight,
                        aux_mae_weight=aux_mae_weight,
                        jitter=jitter,
                        triu_cache=triu_cache,
                        parametrization=parametrization,
                        moe_aux_weight=moe_aux_weight,
                    )
                except torch.cuda.OutOfMemoryError:
                    # OOM during the FLOP measurement only: skip the count, keep the step.
                    step_flops = None
                    if device == "cuda":
                        torch.cuda.empty_cache()
            timer.n += 1
        except torch.cuda.OutOfMemoryError as exc:
            # Skip a batch that runs out of memory.
            shape_batch = batch if batch is not None else raw_batch
            P_b, N_b = shape_batch["x_train"].shape[1], shape_batch["x_test"].shape[1]
            print(
                f"[{step:6d}] CUDA OOM on batch (B={shape_batch['x_train'].shape[0]}, "
                f"P={P_b}, N={N_b}, T={P_b + N_b}) — skipping step."
            )
            # Clear the failed frame's locals before empty_cache().
            traceback.clear_frames(exc.__traceback__)
            del exc
            optimizer.zero_grad(set_to_none=True)
            timer.drop_events()
            del raw_batch, batch, shape_batch, out, Sigma, parts, loss, aux_mae, grad_norm
            # gc.collect() breaks the failed graph's reference cycles so empty_cache() can free its memory.
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # The CPU copy is no longer needed after the H→D transfer.
        del raw_batch

        # Defer .item() / float() GPU syncs to logging steps — saves 2+ syncs/step
        if step % t.log_every == 0:
            loss_ema = _log_train_step(
                step,
                out=out,
                Sigma=Sigma,
                parts=parts,
                loss=loss,
                aux_mae=aux_mae,
                grad_norm=grad_norm,
                test_mask=batch["test_mask"],
                loss_ema=loss_ema,
                bundle=bundle,
                timer=timer,
                step_flops=step_flops,
                gpu_peak_flops=gpu_peak_flops,
                batch_size=int(t.batch_size),
                aux_mae_weight=aux_mae_weight,
                device=device,
            )

        # Release this step's graph before validation and checkpointing.
        out = Sigma = parts = loss = aux_mae = grad_norm = batch = None

        if step % t.val_every == 0 and step > 0:
            _validate_and_log(step, cfg, device, model, data, probes, float(scheduler.get_last_lr()[0]))

        if step % t.save_every == 0 and step > 0:
            save_checkpoint(model, optimizer, scheduler, cfg, step, scaler=scaler)
            if data.tabicl_mix_weights is not None and probes.pit_ckpt and adaptive_mix:
                _refresh_tabicl_mix(step, cfg, probes.pit_ckpt, data.tabicl_mix_weights, device)

    save_checkpoint(model, optimizer, scheduler, cfg, t.steps, scaler=scaler)
    wandb.finish()


if __name__ == "__main__":
    main()
