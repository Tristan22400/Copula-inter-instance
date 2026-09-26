"""Probability integral transform (PIT) through a TabICL marginal, and the exact-GP equivalent.

TabICL's quantile head gives F(y | x, context); the PIT maps each target to
z = Phi^{-1}(F(y)). Training points are scored with K-fold partitioning (each
fold against the other K-1 as context); test points use the full training set
as context. run_pit_calib_split_batched instead scores a query set against a
separate, disjoint calibration set in one pass. gp_analytical_* compute the
same quantities in closed form from an episode's saved GP kernel.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Sequence

import torch
import torch.nn as nn


from copula_inter.data_gen import build_kernel_fn, _safe_cholesky, sigma_to_correlation  # noqa: E402
from tabicl._model.inference_config import InferenceConfig  # noqa: E402

DEFAULT_K_FOLDS = 10

# None keeps TabICL's default (AMP on CUDA); set once by the entrypoint and inherited by workers.
_TABICL_INFERENCE_CONFIG: Optional[InferenceConfig] = None


def configure_tabicl_inference_amp(use_amp: bool) -> None:
    """Set the process-global inference precision for every PIT TabICL forward (None = TabICL default)."""
    global _TABICL_INFERENCE_CONFIG
    _TABICL_INFERENCE_CONFIG = InferenceConfig(
        COL_CONFIG={"use_amp": bool(use_amp)},
        ROW_CONFIG={"use_amp": bool(use_amp)},
        ICL_CONFIG={"use_amp": bool(use_amp)},
    )


def tabicl_forward(tabicl: nn.Module, X: torch.Tensor, y_train: torch.Tensor, **kwargs) -> torch.Tensor:
    """Forward through TabICL using the configured marginal precision."""
    if _TABICL_INFERENCE_CONFIG is None:
        return tabicl(X, y_train, **kwargs)
    return tabicl(X, y_train, inference_config=_TABICL_INFERENCE_CONFIG, **kwargs)


def _optional_param(t: torch.Tensor):
    """None if every entry is the 0.0 "not applicable" sentinel, else a float (scalar) or the tensor (ARD vector)."""
    if torch.all(t == 0.0):
        return None
    return t.item() if t.numel() == 1 else t


def _mean_train_from_task(task: dict, x: torch.Tensor) -> torch.Tensor:
    """Evaluate the episode's saved mean function at x_train (all-zero when the task has no mean_* fields)."""
    zero = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    if not bool(task.get("mean_nonzero", torch.tensor(False)).item()):
        return zero

    def _on_x(name: str) -> torch.Tensor:
        return task[name].to(device=x.device, dtype=x.dtype)

    family = int(task["mean_family"].item())
    if family == 0:  # linear (incl. constant-only, when mean_weight is all-zero)
        return (x * _on_x("mean_weight")).sum(-1) + _on_x("mean_bias")
    elif family == 1:  # exponential
        proj = (x * _on_x("mean_exp_direction")).sum(-1)
        exponent = torch.clamp(_on_x("mean_exp_rate") * proj, min=-10.0, max=10.0)
        return _on_x("mean_exp_scale") * torch.exp(exponent)
    elif family == 2:  # sparse anomaly
        proj = (x * _on_x("mean_anomaly_direction")).sum(-1)
        hit = (proj > _on_x("mean_anomaly_threshold")).to(x.dtype)
        return hit * _on_x("mean_anomaly_magnitude")
    raise ValueError(f"Unknown mean_family {family}; expected 0, 1, or 2.")


def resolve_pit_ckpt(cfg) -> str | None:
    """The TabICL checkpoint to load as the frozen PIT marginal, or None.

    tabicl.pit_ckpt if set; otherwise tabicl.ckpt when tabicl.pretrained is true.
    """
    pit_ckpt = cfg.tabicl.get("pit_ckpt", None)
    if pit_ckpt is None and bool(cfg.tabicl.get("pretrained", True)):
        pit_ckpt = cfg.tabicl.get("ckpt", None)
    return pit_ckpt


def load_tabicl(
    ckpt_name: str, device: str, trainable: bool = False, return_config: bool = False
) -> nn.Module:
    """Load a TabICL regressor.

    Args:
        ckpt_name: a filename in the jingang/TabICL HF repo, or a local .ckpt/.pt
            path with the same {"config", "state_dict"} schema.
        device: torch device.
        trainable: keep gradients and train mode (for fine-tuning); otherwise
            frozen and in eval mode.
        return_config: also return the architecture config dict.
    """
    from tabicl._model.tabicl import TabICL  # type: ignore[import]

    if os.path.isfile(ckpt_name):
        ckpt_path = ckpt_name
    else:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(repo_id="jingang/TabICL", filename=ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    base = TabICL(**checkpoint["config"])
    base.load_state_dict(checkpoint["state_dict"])
    if trainable:
        base.train()
    else:
        for p in base.parameters():
            p.requires_grad_(False)
        base.eval()
    base.to(device)
    if return_config:
        return base, dict(checkpoint["config"])
    return base


def _probit(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Clamp u to (eps, 1-eps) and apply Phi^{-1}."""
    u = u.clamp(eps, 1.0 - eps)
    return torch.erfinv(2.0 * u - 1.0) * math.sqrt(2.0)


def normalize_targets(
    y_train: torch.Tensor, y_test: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Z-score targets with y_train's mean and std (y_test uses y_train's moments).

    Args:
        y_train: (P,) raw targets.
        y_test: optional (N,) raw targets.

    Returns:
        (y_train_scaled, y_test_scaled or None, mean, std). Log densities of the
        scaled call convert back as log p_raw(y) = log p_scaled(y_scaled) - log(std).
    """
    mean = y_train.mean()
    std = y_train.std().clamp(min=1e-8)
    y_train_scaled = (y_train - mean) / std
    y_test_scaled = (y_test - mean) / std if y_test is not None else None
    return y_train_scaled, y_test_scaled, mean, std


def _scale_fold_targets(y_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize one fold using only its context labels (unit scale for a single context row)."""
    mean = y_context.mean(dim=-1, keepdim=True)
    std = (
        y_context.std(dim=-1, keepdim=True).clamp(min=1e-8)
        if y_context.shape[-1] > 1 else torch.ones_like(mean)
    )
    return (y_context - mean) / std, mean, std


@torch.no_grad()
def run_pit(
    tabicl: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    k_folds: int = DEFAULT_K_FOLDS,
    eps: float = 1e-6,
    Y_train_raw: Optional[torch.Tensor] = None,
) -> dict:
    """PIT for one task.

    Args:
        tabicl: TabICL regressor (max_classes=0).
        X_train: (P, p_x).
        Y_train: (P, d) on the caller's full-context target scale; each fold is
            re-standardized on its own context labels.
        X_test: (N, p_x).
        Y_test: (N, d).
        k_folds: number of folds for the training-set PIT, clamped to [2, P].
        eps: probit clamp.
        Y_train_raw: optional (P, d) raw labels used for fold scaling.

    Returns:
        dict with z_train (P, d), z_test (N, d), log_pdf_test (N, d).
    """
    device = X_train.device
    P, p_x = X_train.shape
    N = X_test.shape[0]
    d = Y_train.shape[1]

    K = max(2, min(int(k_folds), P))
    fold_targets = Y_train if Y_train_raw is None else Y_train_raw

    # A) Test instances: one forward over the full train context, d targets on the batch axis.
    X_concat = torch.cat([X_train, X_test], dim=0)                       # (P+N, p_x)
    X_test_batch = X_concat.unsqueeze(0).expand(d, -1, -1).contiguous()  # (d, P+N, p_x)
    y_train_batch = Y_train.permute(1, 0).contiguous()                   # (d, P)

    logits = tabicl_forward(tabicl, X_test_batch, y_train_batch)         # (d, N, Q)
    # TabICL may return its output on CPU under memory pressure; move it back to `device`.
    logits = logits.to(device)
    Q = logits.shape[-1]
    dist = tabicl.quantile_dist(logits.reshape(d * N, Q))

    y_test_flat = Y_test.permute(1, 0).reshape(d * N)
    u_test = dist.cdf(y_test_flat).reshape(d, N).permute(1, 0)           # (N, d)
    log_pdf_test = dist.log_prob(y_test_flat).reshape(d, N).permute(1, 0)  # (N, d)

    # B) Training instances: K disjoint folds.
    fold_size = math.ceil(P / K)
    u_train = torch.empty(P, d, device=device, dtype=Y_train.dtype)
    indices = torch.arange(P, device=device)

    for k in range(K):
        start = k * fold_size
        end = min(start + fold_size, P)
        if start >= end:
            break

        qry_idx = indices[start:end]
        ctx_mask = torch.ones(P, dtype=torch.bool, device=device)
        ctx_mask[qry_idx] = False
        ctx_idx = indices[ctx_mask]
        F = qry_idx.numel()

        X_fold = torch.cat([X_train[ctx_idx], X_train[qry_idx]], dim=0)    # (P-F+F, p_x)
        X_fold_batch = X_fold.unsqueeze(0).expand(d, -1, -1).contiguous()
        y_context = fold_targets[ctx_idx].permute(1, 0)                  # (d, P-F)
        y_ctx_batch, fold_mean, fold_std = _scale_fold_targets(y_context)
        y_ctx_batch = y_ctx_batch.contiguous()

        logits_fold = tabicl_forward(tabicl, X_fold_batch, y_ctx_batch)    # (d, F, Q)
        logits_fold = logits_fold.to(device)  # see run_pit's offload-mode comment above
        dist_fold = tabicl.quantile_dist(logits_fold.reshape(d * F, Q))

        y_query = fold_targets[qry_idx].permute(1, 0)
        y_qry_flat = ((y_query - fold_mean) / fold_std).reshape(d * F)
        u_train[qry_idx, :] = (
            dist_fold.cdf(y_qry_flat).reshape(d, F).permute(1, 0)
        )

    z_train = _probit(u_train, eps)
    z_test = _probit(u_test, eps)

    return {
        "z_train": z_train,
        "z_test": z_test,
        "log_pdf_test": log_pdf_test,
    }


def _alpha_levels_of(tabicl: nn.Module, device) -> "torch.Tensor | None":
    """The quantile levels the model's decoder emits, or None if it exposes none."""
    levels = getattr(getattr(tabicl, "quantile_dist", None), "alpha_levels", None)
    return None if levels is None else levels.to(device)


class _train_mode:
    """Context manager that keeps module in train mode (TabICL's eval-mode forward uses float16 autocast)."""

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.was_training = module.training

    def __enter__(self) -> nn.Module:
        self.module.train()
        return self.module

    def __exit__(self, *exc) -> None:
        self.module.train(self.was_training)


def _run_pit_batched_impl(
    tabicl: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    k_folds: int,
    eps: float,
    *,
    return_quantiles: bool = False,
    fold_subset: "Sequence[int] | None" = None,
    compute_pit: bool = True,
    fuse_folds: bool = False,
    Y_train_raw: Optional[torch.Tensor] = None,
) -> dict:
    """Shared body of run_pit_batched and run_pit_batched_grad.

    fold_subset scores only the listed folds (same fold geometry); the result then
    carries train_query_idx and z_train/u_train/q_train are indexed by it.
    return_quantiles also returns the decoder quantiles (on the caller's target
    scale), the pre-probit CDF values and the probit-clamp saturation fractions.
    """
    if not compute_pit and not return_quantiles:
        raise ValueError("compute_pit=False requires return_quantiles=True")

    device = X_train.device
    B, P, p_x = X_train.shape
    N = X_test.shape[1]
    d = Y_train.shape[2]

    K = max(2, min(int(k_folds), P))
    fold_targets = Y_train if Y_train_raw is None else Y_train_raw
    if Y_train_raw is not None:
        full_mean = Y_train_raw.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        full_std = Y_train_raw.std(dim=1).clamp(min=1e-8).unsqueeze(-1).unsqueeze(-1)

    # A) Test instances: one forward, batch axis = B*d.
    X_concat = torch.cat([X_train, X_test], dim=1)                              # (B, P+N, p_x)
    X_test_batch = (
        X_concat.unsqueeze(1).expand(B, d, P + N, p_x).reshape(B * d, P + N, p_x).contiguous()
    )
    y_train_batch = Y_train.permute(0, 2, 1).reshape(B * d, P).contiguous()     # (B*d, P)

    logits = tabicl_forward(tabicl, X_test_batch, y_train_batch)                # (B*d, N, Q)
    # TabICL may return its output on CPU under memory pressure.
    logits = logits.to(device)
    Q = logits.shape[-1]
    if compute_pit:
        dist = tabicl.quantile_dist(logits.reshape(B * d * N, Q))
        y_test_flat = Y_test.permute(0, 2, 1).reshape(B * d * N)
        u_test = dist.cdf(y_test_flat).reshape(B, d, N).permute(0, 2, 1)       # (B, N, d)
        log_pdf_test = dist.log_prob(y_test_flat).reshape(B, d, N).permute(0, 2, 1)
    else:
        u_test = log_pdf_test = None

    q_test = None
    if return_quantiles:
        q_test = logits.reshape(B, d, N, Q).permute(0, 2, 1, 3)                 # (B, N, d, Q)

    # B) Training instances: K disjoint folds, batch axis = B*d.
    fold_size = math.ceil(P / K)
    u_train_parts: list = []
    q_train_parts: list = []
    indices = torch.arange(P, device=device)
    wanted_folds = range(K) if fold_subset is None else sorted(set(int(k) for k in fold_subset))

    fold_specs = []
    for k in wanted_folds:
        start = k * fold_size
        end = min(start + fold_size, P)
        if start >= end:
            continue  # empty trailing fold (K > P after clamping) -- not an
            # fold_subset folds need not be contiguous, so skip rather than break.

        qry_idx = indices[start:end]
        ctx_mask = torch.ones(P, dtype=torch.bool, device=device)
        ctx_mask[qry_idx] = False
        ctx_idx = indices[ctx_mask]
        F = qry_idx.numel()

        fold_specs.append((qry_idx, ctx_idx, F))

    # Folds with equal query size share shapes and can be fused into one launch.
    fold_groups: list[list[tuple[torch.Tensor, torch.Tensor, int]]] = []
    if fuse_folds:
        by_size: dict[int, list[tuple[torch.Tensor, torch.Tensor, int]]] = {}
        for spec in fold_specs:
            by_size.setdefault(spec[2], []).append(spec)
        fold_groups.extend(by_size.values())
    else:
        fold_groups.extend([[spec] for spec in fold_specs])

    for group in fold_groups:
        x_group = []
        y_group = []
        fold_scales = []
        for qry_idx, ctx_idx, F in group:
            X_fold = torch.cat([X_train[:, ctx_idx], X_train[:, qry_idx]], dim=1)
            x_group.append(
                X_fold.unsqueeze(1).expand(B, d, X_fold.shape[1], p_x)
                .reshape(B * d, X_fold.shape[1], p_x).contiguous()
            )
            y_context = fold_targets[:, ctx_idx].permute(0, 2, 1)  # (B, d, P-F)
            y_scaled, fold_mean, fold_std = _scale_fold_targets(y_context)
            y_group.append(y_scaled.reshape(B * d, P - F).contiguous())
            fold_scales.append((fold_mean, fold_std))

        logits_group = tabicl_forward(tabicl, torch.cat(x_group, dim=0), torch.cat(y_group, dim=0))
        logits_group = logits_group.to(device)

        for group_idx, (qry_idx, _ctx_idx, F) in enumerate(group):
            logits_fold = logits_group[group_idx * B * d:(group_idx + 1) * B * d]
            fold_mean, fold_std = fold_scales[group_idx]

            if compute_pit:
                dist_fold = tabicl.quantile_dist(logits_fold.reshape(B * d * F, Q))
                y_query = fold_targets[:, qry_idx].permute(0, 2, 1)
                y_qry_flat = ((y_query - fold_mean) / fold_std).reshape(B * d * F)
                u_fold = dist_fold.cdf(y_qry_flat).reshape(B, d, F).permute(0, 2, 1)
                u_train_parts.append((qry_idx, u_fold))
            if return_quantiles:
                q_fold = logits_fold.reshape(B, d, F, Q)
                q_fold = q_fold * fold_std.unsqueeze(-1) + fold_mean.unsqueeze(-1)
                if Y_train_raw is not None:
                    q_fold = (q_fold - full_mean) / full_std
                q_train_parts.append(
                    (qry_idx, q_fold.permute(0, 2, 1, 3))
                )

    no_fold_outputs = not (q_train_parts if return_quantiles else u_train_parts)
    if no_fold_outputs:
        if fold_subset is None or len(list(fold_subset)) > 0:
            raise ValueError(
                f"run_pit_batched: no folds were scored (K={K}, P={P}, "
                f"fold_subset={fold_subset}). Every requested fold was empty."
            )
        # fold_subset=[]: test rows only, one forward with the full P-row context.
        out = {"train_query_idx": torch.empty(0, dtype=torch.long, device=device)}
        if compute_pit:
            out.update({"z_test": _probit(u_test, eps), "log_pdf_test": log_pdf_test})
        if return_quantiles:
            out.update(
                {
                    "q_test": q_test,
                    **({
                        "u_test": u_test,
                        "clamp_frac_test": (
                            (u_test <= eps) | (u_test >= 1.0 - eps)
                        ).float().mean().detach(),
                    } if compute_pit else {}),
                }
            )
            levels = _alpha_levels_of(tabicl, device)
            if levels is not None:
                out["alpha_levels"] = levels                                   # (Q,)
        return out

    parts_for_order = q_train_parts if return_quantiles else u_train_parts
    order = torch.cat([qi for qi, _ in parts_for_order], dim=0)                # (P',)
    if compute_pit:
        u_train = torch.cat([uf for _, uf in u_train_parts], dim=1)            # (B, P', d)
    if fold_subset is None:
        # Full pass: restore the caller's row order.
        inv = torch.argsort(order)
        if compute_pit:
            u_train = u_train[:, inv, :]
    else:
        inv = None

    out = {}
    if compute_pit:
        out.update({
            "z_train": _probit(u_train, eps),
            "z_test": _probit(u_test, eps),
            "log_pdf_test": log_pdf_test,
        })
    if fold_subset is not None:
        out["train_query_idx"] = order
    if return_quantiles:
        q_train = torch.cat([qf for _, qf in q_train_parts], dim=1)
        if inv is not None:
            q_train = q_train[:, inv, :, :]
        out.update(
            {
                "q_train": q_train,                                            # (B, P', d, Q)
                "q_test": q_test,                                              # (B, N, d, Q)
                **({
                    "u_train": u_train,
                    "u_test": u_test,
                    # Silent-failure counters: _probit hard-caps |z| at 4.7534.
                    "clamp_frac_train": (
                        (u_train <= eps) | (u_train >= 1.0 - eps)
                    ).float().mean().detach(),
                    "clamp_frac_test": (
                        (u_test <= eps) | (u_test >= 1.0 - eps)
                    ).float().mean().detach(),
                } if compute_pit else {}),
            }
        )
        levels = _alpha_levels_of(tabicl, device)
        if levels is not None:
            out["alpha_levels"] = levels                                       # (Q,)
    return out


@torch.no_grad()
def run_pit_batched(
    tabicl: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    k_folds: int = DEFAULT_K_FOLDS,
    eps: float = 1e-6,
    return_quantiles: bool = False,
    Y_train_raw: Optional[torch.Tensor] = None,
) -> dict:
    """run_pit over a batch of episodes that share P and N.

    Args:
        tabicl: TabICL regressor (max_classes=0).
        X_train: (B, P, p_x).
        Y_train: (B, P, d) on the caller's full-context target scale.
        X_test: (B, N, p_x).
        Y_test: (B, N, d).
        k_folds: as in run_pit.
        eps: probit clamp.
        return_quantiles: also return decoder quantiles, CDF values and clamp
            fractions (see _run_pit_batched_impl).
        Y_train_raw: optional (B, P, d) raw labels for fold scaling.

    Returns:
        dict with z_train (B, P, d), z_test (B, N, d), log_pdf_test (B, N, d).
    """
    return _run_pit_batched_impl(
        tabicl, X_train, Y_train, X_test, Y_test, k_folds, eps,
        return_quantiles=return_quantiles,
        Y_train_raw=Y_train_raw,
    )


def run_pit_batched_grad(
    tabicl: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    k_folds: int = DEFAULT_K_FOLDS,
    eps: float = 1e-6,
    return_quantiles: bool = True,
    fold_subset: "Sequence[int] | None" = None,
    compute_pit: bool = True,
    fuse_folds: bool = False,
    Y_train_raw: Optional[torch.Tensor] = None,
) -> dict:
    """run_pit_batched with gradients enabled and the model in train mode; returns quantiles by default."""
    with _train_mode(tabicl):
        return _run_pit_batched_impl(
            tabicl, X_train, Y_train, X_test, Y_test, k_folds, eps,
            return_quantiles=return_quantiles, fold_subset=fold_subset,
            compute_pit=compute_pit, fuse_folds=fuse_folds,
            Y_train_raw=Y_train_raw,
        )


@torch.no_grad()
def run_pit_calib_split_batched(
    tabicl: nn.Module,
    X_query: torch.Tensor,
    Y_query: torch.Tensor,
    X_calib: torch.Tensor,
    Y_calib: torch.Tensor,
    eps: float = 1e-6,
    Y_query_raw: Optional[torch.Tensor] = None,
    Y_calib_raw: Optional[torch.Tensor] = None,
) -> dict:
    """PIT of a query set against a separate calibration set, in one forward pass.

    Args:
        tabicl: TabICL regressor (max_classes=0).
        X_query: (B, P_Q, p_x).
        Y_query: (B, P_Q, d), used only to evaluate the CDF.
        X_calib: (B, P_C, p_x) context, disjoint from the query set.
        Y_calib: (B, P_C, d).
        eps: probit clamp.
        Y_query_raw, Y_calib_raw: optional raw labels for context-only scaling.

    Returns:
        dict with z_train (B, P_Q, d).
    """
    device = X_query.device
    B, P_Q, p_x = X_query.shape
    P_C = X_calib.shape[1]
    d = Y_query.shape[2]

    X_concat = torch.cat([X_calib, X_query], dim=1)                          # (B, P_C+P_Q, p_x)
    X_batch = (
        X_concat.unsqueeze(1).expand(B, d, P_C + P_Q, p_x).reshape(B * d, P_C + P_Q, p_x).contiguous()
    )
    if (Y_query_raw is None) != (Y_calib_raw is None):
        raise ValueError("Y_query_raw and Y_calib_raw must be supplied together")
    calib_source = Y_calib if Y_calib_raw is None else Y_calib_raw
    query_source = Y_query if Y_query_raw is None else Y_query_raw
    y_calib_scaled, calib_mean, calib_std = _scale_fold_targets(calib_source.permute(0, 2, 1))
    y_calib_batch = y_calib_scaled.reshape(B * d, P_C).contiguous()          # (B*d, P_C)

    logits = tabicl_forward(tabicl, X_batch, y_calib_batch)                  # (B*d, P_Q, Q)
    # TabICL may return its output on CPU under memory pressure.
    logits = logits.to(device)
    Q = logits.shape[-1]
    dist = tabicl.quantile_dist(logits.reshape(B * d * P_Q, Q))

    y_query = query_source.permute(0, 2, 1)
    y_query_flat = ((y_query - calib_mean) / calib_std).reshape(B * d * P_Q)
    u_query = dist.cdf(y_query_flat).reshape(B, d, P_Q).permute(0, 2, 1)     # (B, P_Q, d)
    z_train = _probit(u_query, eps)

    return {"z_train": z_train}


def _sign_triple(d: dict, applied_key: str, w_key: str, b_key: str, a_key: str):
    """(sign_w, sign_b, sign_a) from a dict's sign-modulation fields, or (None, None, None) if not applied.

    sign_a may be None for datasets saved before the sharpness parameter existed.
    """
    applied = d.get(applied_key)
    if applied is None or applied.item() == 0.0:
        return None, None, None
    return d[w_key], d[b_key], d.get(a_key)


def _kernel_fn_from_chain_task(task: dict):
    """Rebuild (kernel_fn, nugget) for a systematic-composition chain episode.

    Each component comes from task["kernel_component_params"][i]; the dense
    kernels are combined left to right by task["kernel_ops"] ("+" or "*").
    Raises NotImplementedError when whole-chain (outer) sign modulation was applied.
    """
    names = task["kernel_components"]
    ops = task["kernel_ops"]
    comp_params = task["kernel_component_params"]
    nugget = task["nugget"].item()
    cols = task["kernel_feature_indices"].tolist()

    sign_w_outer, sign_b_outer, sign_a_outer = _sign_triple(
        task, "sign_applied_outer", "sign_w_outer", "sign_b_outer", "sign_a_outer"
    )
    if sign_w_outer is not None:
        raise NotImplementedError(
            "_kernel_fn_from_chain_task: whole-chain outer sign modulation "
            "(cfg.data.sign_modulation_outer_prob) is not supported for "
            "systematic-composition chain reconstruction."
        )

    component_fns = []
    for name, params in zip(names, comp_params):
        l_t = params["l"]
        l = l_t.item() if l_t.numel() == 1 else l_t
        alpha2 = params["alpha2"].item()
        period = _optional_param(params["period"]) if "period" in params else None
        rq_alpha = (
            params["rq_alpha"].item() if "rq_alpha" in params and params["rq_alpha"].item() != 0.0 else None
        )
        power = (
            params["power"].item() if "power" in params and params["power"].item() != 0.0 else None
        )
        sign_w, sign_b, sign_a = _sign_triple(params, "sign_applied", "sign_w", "sign_b", "sign_a")
        component_fns.append(build_kernel_fn(
            name, l, alpha2, period=period, rq_alpha=rq_alpha, power=power,
            active_dims=cols, sign_w=sign_w, sign_b=sign_b, sign_a=sign_a,
        ))

    def kernel_fn(X1, X2):
        K = component_fns[0](X1, X2)
        for op, fn in zip(ops, component_fns[1:]):
            Ki = fn(X1, X2)
            K = K + Ki if op == "+" else K * Ki
        return K

    return kernel_fn, nugget


def _kernel_fn_from_task(task: dict):
    """Rebuild (kernel_fn, nugget) from a task's saved kernel metadata (flat or chain schema)."""
    if "kernel_components" in task:
        return _kernel_fn_from_chain_task(task)

    kernel_name = task["kernel"]
    # Scalar, or a (k,) lengthscale vector for ARD episodes.
    l_tensor = task["l"]
    l      = l_tensor.item() if l_tensor.numel() == 1 else l_tensor
    alpha2 = task["alpha2"].item()
    nugget = task["nugget"].item()
    # 0.0 means not applicable; period is a vector under periodic+ARD.
    period   = _optional_param(task["period"])
    rq_alpha = task["rq_alpha"].item() if task["rq_alpha"].item() != 0.0 else None
    # Polynomial degree (0.0 = not applicable).
    power    = task["power"].item() if task["power"].item() != 0.0 else None
    # Second component of a composite ("A+B"/"A*B") kernel; may be ARD vectors.
    l_b        = _optional_param(task["l_b"])
    alpha2_b   = task["alpha2_b"].item() if task["alpha2_b"].item() != 0.0 else None
    period_b   = _optional_param(task["period_b"])
    rq_alpha_b = task["rq_alpha_b"].item() if task["rq_alpha_b"].item() != 0.0 else None
    power_b    = task["power_b"].item() if task["power_b"].item() != 0.0 else None

    sign_w, sign_b, sign_a = _sign_triple(task, "sign_applied", "sign_w", "sign_b", "sign_a")
    sign_w_b, sign_b_b, sign_a_b = _sign_triple(task, "sign_applied_b", "sign_w_b", "sign_b_b", "sign_a_b")
    sign_w_outer, sign_b_outer, sign_a_outer = _sign_triple(
        task, "sign_applied_outer", "sign_w_outer", "sign_b_outer", "sign_a_outer"
    )

    # active_dims lets kernel_fn select its columns from the full-width inputs.
    cols = task["kernel_feature_indices"].tolist()
    kernel_fn = build_kernel_fn(
        kernel_name, l, alpha2, period=period, rq_alpha=rq_alpha, power=power,
        l_b=l_b, alpha2_b=alpha2_b, period_b=period_b, rq_alpha_b=rq_alpha_b, power_b=power_b,
        active_dims=cols,
        sign_w=sign_w, sign_b=sign_b, sign_a=sign_a,
        sign_w_b=sign_w_b, sign_b_b=sign_b_b, sign_a_b=sign_a_b,
        sign_w_outer=sign_w_outer, sign_b_outer=sign_b_outer, sign_a_outer=sign_a_outer,
    )
    return kernel_fn, nugget


@torch.no_grad()
def gp_analytical_pit(task: dict, eps: float = 1e-6) -> dict:
    """Exact PIT from the episode's GP: LOO for training points, posterior marginals for test points.

    Test points:
        mu_post = mean(x_test) + K_sf K_ff^{-1} (y_train - mean_train)
        var_post = diag(K_ss - K_sf K_ff^{-1} K_fs)
        z_test = (y_test - mu_post) / sqrt(var_post)
    Training points (Rasmussen & Williams Eq. 5.12), with
    alpha = K_ff^{-1} (y_train - mean_train):
        z_train_i = alpha_i / sqrt([K_ff^{-1}]_ii)

    Args:
        task: task dict with kernel metadata (return_kernel_metadata=True),
            x_norm_train/test, y_train/test, mu_star and the mean_* fields.
        eps: unused.

    Returns:
        dict with z_train (P,), z_test (N,), log_pdf_test (N,).
    """
    kernel_fn, nugget = _kernel_fn_from_task(task)
    x_k_train = task.get("x_kernel_train", task["x_norm_train"])   # (P, d_features)
    x_k_test   = task.get("x_kernel_test", task["x_norm_test"])    # (N, d_features)
    y_train    = task["y_train"]                  # (P,)
    y_test     = task["y_test"]                   # (N,)
    mu_star    = task["mu_star"]                  # (N,) PRIOR mean at the test points

    # L_ff / alpha, shared by the test posterior and the train LOO; reuse cached factors when present.
    P = y_train.shape[0]
    if "_L_ff" in task and "_alpha" in task:
        L     = task["_L_ff"]
        alpha = task["_alpha"]
    else:
        K_ff       = kernel_fn(x_k_train, x_k_train) + nugget * torch.eye(P, device=y_train.device)
        L          = _safe_cholesky(K_ff)
        mean_train = _mean_train_from_task(task, x_k_train)
        alpha      = torch.cholesky_solve((y_train - mean_train).unsqueeze(-1), L).squeeze(-1)  # (P,)

    # Test: exact GP posterior marginals (K_sf noise-free, nugget on K_ss's diagonal).
    x_ref     = x_k_test.to(L.device)
    K_sf      = kernel_fn(x_ref, x_k_train.to(L.device))                       # (N, P)
    K_ss_diag = (
        kernel_fn(x_ref, x_ref).diagonal() + nugget
    )                                                                          # (N,)
    V_sf      = torch.linalg.solve_triangular(L, K_sf.T.to(L.dtype), upper=False)  # (P, N)
    mu_post   = mu_star.to(L.device) + K_sf @ alpha                            # (N,)
    var_post  = (K_ss_diag - (V_sf ** 2).sum(dim=0)).clamp(min=max(nugget, 1e-12))
    sig_clamped  = var_post.sqrt()
    z_test       = (y_test.to(L.device) - mu_post) / sig_clamped
    log_pdf_test = (
        -0.5 * math.log(2.0 * math.pi)
        - sig_clamped.log()
        - 0.5 * z_test**2
    )

    # Train: exact GP LOO; diag(K_ff^{-1}) is the column-wise squared norm of L^{-1}.
    L_inv      = torch.linalg.solve_triangular(
        L, torch.eye(P, device=L.device, dtype=L.dtype), upper=False
    )                                                                      # (P, P)
    K_inv_diag = (L_inv**2).sum(dim=0).clamp(min=1e-12)                   # (P,)
    z_train    = alpha * K_inv_diag.rsqrt()                               # alpha_i/√[K⁻¹]_ii

    return {"z_train": z_train, "z_test": z_test, "log_pdf_test": log_pdf_test}


def mvn_nll(y: torch.Tensor, mean: torch.Tensor, Sigma: torch.Tensor) -> float:
    """Negative log-likelihood of y under N(mean, Sigma), in raw-y units (float64 Cholesky)."""
    y, mean, Sigma = y.double(), mean.double(), Sigma.double()
    L = _safe_cholesky(Sigma)
    resid = (y - mean).unsqueeze(-1)
    sol = torch.cholesky_solve(resid, L)
    quad = (resid * sol).sum()
    log_det = 2.0 * torch.log(torch.diagonal(L)).sum()
    n = y.shape[0]
    return (0.5 * (n * math.log(2.0 * math.pi) + log_det + quad)).item()


def mvn_nll_parts(y: torch.Tensor, mean: torch.Tensor, Sigma: torch.Tensor) -> dict:
    """mvn_nll split into Sklar parts: dict(total, marginal, copula), unnormalized sums.

    marginal is the sum of univariate N(mean_i, Sigma_ii) NLLs; copula = total - marginal.
    """
    total = mvn_nll(y, mean, Sigma)
    y64, mean64 = y.double(), mean.double()
    std = Sigma.double().diagonal().clamp(min=1e-12).sqrt()
    z = (y64 - mean64) / std
    marginal = (0.5 * math.log(2.0 * math.pi) + std.log() + 0.5 * z ** 2).sum().item()
    return {"total": total, "marginal": marginal, "copula": total - marginal}


@torch.no_grad()
def gp_analytical_posterior(task: dict, eig_floor: float = 1e-6) -> dict:
    """Exact GP posterior over the test points, conditioned on (x_train, y_train).

    Sigma_post = K_ss - K_sf K_ff^{-1} K_fs (noise on K_ss's diagonal), computed in
    float64. Eigenvalues are floored at max(eig_floor * max|diag(Sigma_post)|,
    nugget) before converting to a correlation matrix. Raises NotImplementedError
    for chain episodes with outer sign modulation.

    Returns:
        dict with mu_post (N,), Sigma_post (N, N), R_post (N, N) (float32);
        min_eig (pre-repair) and repaired (bool); nll_prior and nll_post, the
        Y-space NLL of y_test under N(mean_test, K_ss) and N(mu_post, Sigma_post);
        and their marginal/copula splits nll_{prior,post}_{marginal,copula}.
    """
    kernel_fn, nugget = _kernel_fn_from_task(task)
    # Run on the device of the cached _L_ff/_alpha (x_train's device when absent).
    x_train_raw = task.get("x_kernel_train", task["x_norm_train"])
    x_test_raw  = task.get("x_kernel_test", task["x_norm_test"])
    ref_device = task["_L_ff"].device if "_L_ff" in task else x_train_raw.device
    x_train = x_train_raw.to(ref_device)   # (P, d), float32
    x_test  = x_test_raw.to(ref_device)    # (N, d), float32
    y_train = task["y_train"].to(ref_device)         # (P,)
    P, N = x_train.shape[0], x_test.shape[0]

    if "_L_ff" in task and "_alpha" in task:
        L_ff  = task["_L_ff"].double()
        alpha = task["_alpha"].double()
    else:
        K_ff       = kernel_fn(x_train, x_train) + nugget * torch.eye(P, device=x_train.device)
        L_ff       = _safe_cholesky(K_ff).double()
        mean_train = _mean_train_from_task(task, x_train)
        alpha      = torch.cholesky_solve(
            (y_train - mean_train).double().unsqueeze(-1), L_ff
        ).squeeze(-1)

    # K_sf has no noise term; K_ss has the nugget on its diagonal.
    K_sf = kernel_fn(x_test, x_train).double()                                              # (N, P)
    K_ss = (kernel_fn(x_test, x_test) + nugget * torch.eye(N, device=x_test.device)).double()  # (N, N)

    V = torch.linalg.solve_triangular(L_ff, K_sf.T, upper=False)   # (P, N)
    Sigma_post = K_ss - V.T @ V
    Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)

    # mu_star is the prior mean mean_module(x_test).
    mean_test = task["mu_star"].to(ref_device).double()
    mu_post = mean_test + K_sf @ alpha

    # Eigenvalue floor: relative to Sigma_post's diagonal scale, and never below the nugget.
    scale = Sigma_post.diagonal().abs().max().clamp(min=1e-12).item()
    eig_floor_eff = max(eig_floor * scale, nugget)
    eigvals = torch.linalg.eigvalsh(Sigma_post)
    min_eig = eigvals.min().item()
    repaired = min_eig < eig_floor_eff
    if repaired:
        eigvals_c, eigvecs = torch.linalg.eigh(Sigma_post)
        Sigma_post = eigvecs @ torch.diag(eigvals_c.clamp(min=eig_floor_eff)) @ eigvecs.T
        Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)

    R_post, _ = sigma_to_correlation(Sigma_post.float())

    # Total Y-space NLL under the prior and the posterior (full MVN density).
    y_test = task["y_test"].to(ref_device).double()

    K_ss_sym = 0.5 * (K_ss + K_ss.T)
    prior_parts = mvn_nll_parts(y_test, mean_test, K_ss_sym)
    post_parts  = mvn_nll_parts(y_test, mu_post, Sigma_post)

    return {
        "mu_post":    mu_post.float(),
        "Sigma_post": Sigma_post.float(),
        "R_post":     R_post,
        "min_eig":    min_eig,
        "repaired":   repaired,
        "nll_prior":  prior_parts["total"],
        "nll_post":   post_parts["total"],
        # Marginal/copula split of both totals.
        "nll_prior_marginal": prior_parts["marginal"],
        "nll_prior_copula":   prior_parts["copula"],
        "nll_post_marginal":  post_parts["marginal"],
        "nll_post_copula":    post_parts["copula"],
    }


def gaussian_corr_kl(R_model: torch.Tensor, R_post: torch.Tensor) -> float:
    """Per-point KL(N(0, R_post) || N(0, R_model)) between two correlation matrices.

        KL / n = 0.5 * [tr(R_model^{-1} R_post) - n + log|R_model| - log|R_post|] / n

    Zero iff R_model == R_post. Returns +inf if R_model is not positive definite.
    """
    A = R_model.double()
    B = R_post.double()
    n = A.shape[-1]
    try:
        L = torch.linalg.cholesky(A)
    except Exception:
        return float("inf")
    if not torch.isfinite(L).all():
        return float("inf")
    log_det_model = 2.0 * torch.log(torch.diagonal(L)).sum()
    trace = torch.diagonal(torch.cholesky_solve(B, L)).sum()
    sign, log_det_post = torch.linalg.slogdet(B)
    if sign.item() <= 0:
        return float("inf")
    val = 0.5 * (trace - n + log_det_model - log_det_post) / n
    return float(val.item())
