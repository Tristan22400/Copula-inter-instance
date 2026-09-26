"""The training schedule and single-step execution used by all entrypoints."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.flop_counter import FlopCounterMode

from copula_inter.loss import y_space_nll
from copula_inter.model import low_rank_correlation_factor

def cosine_lr_lambda(step: int, warmup: int, total: int, lr_min_frac: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    # Clamp progress to [0, 1] (a resumed run can start past training.steps).
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return lr_min_frac + (1.0 - lr_min_frac) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


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
    moe_aux_weight: float = 1.0,
    phase_start=lambda: None,
    phase_end=lambda name, start: None,
):
    """Forward pass and loss: Y-space NLL, plus the optional aux MAE and MoE auxiliary terms.

    phase_start/phase_end time the phases (no-ops by default).
    """
    ev_fwd0 = phase_start()
    with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
        out = model(batch)
    phase_end("forward", ev_fwd0)

    # Loss in float32 — Cholesky / log-det want full precision.
    ev_loss0 = phase_start()
    s = out.get("s")
    lam = out.get("lam")
    # Sigma stays factored (U U^T + diag(D)) for the O(N r^2) NLL; call .dense() for the matrix.
    Sigma = low_rank_correlation_factor(
        out["W"].float(),
        s.float() if s is not None else None,
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
    aux_mae = parts["total"].new_tensor(0.0)
    if aux_mae_weight > 0.0:
        Sigma_dense = Sigma.dense()
        n_test = Sigma_dense.shape[1]
        mask_2d = batch["test_mask"].unsqueeze(-1) & batch["test_mask"].unsqueeze(-2)
        if n_test not in triu_cache:
            triu_cache[n_test] = torch.triu_indices(
                n_test, n_test, offset=1, device=Sigma_dense.device
            )
        ri, ci = triu_cache[n_test]
        valid_off = mask_2d[:, ri, ci]
        if valid_off.any():
            pred_off = Sigma_dense[:, ri, ci][valid_off]
            oracle_off = batch["R_star"].float()[:, ri, ci][valid_off]
            aux_mae = (pred_off - oracle_off).abs().mean()
        loss = loss + aux_mae_weight * aux_mae

    # Backbone auxiliary loss (TabLDM MoE only).
    moe_aux = out.get("moe_aux_loss")
    if moe_aux is not None:
        loss = loss + moe_aux_weight * moe_aux
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
    moe_aux_weight: float = 1.0,
) -> float:
    """FLOPs of one forward+backward under FlopCounterMode, without an optimizer step (for MFU)."""
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
            moe_aux_weight=moe_aux_weight,
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
    moe_aux_weight: float = 1.0,
):
    """Run one training step in its own frame, so an OOM releases every graph tensor when it unwinds."""
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
        moe_aux_weight=moe_aux_weight,
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

