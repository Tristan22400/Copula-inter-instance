"""Phase A: fine-tune a standalone TabICL marginal (quantile decoder intact).

Trains the marginal's posterior predictive on GP episodes (optionally mixed
with real ERA5) and writes a TabICL-schema checkpoint usable as
tabicl.pit_ckpt:

    python -m copula_inter.train tabicl.pit_ckpt=<checkpoint>

Usage:
    python -m copula_inter.finetune_marginal
    python -m copula_inter.finetune_marginal marginal.tier=1 training.lr=2e-5
    python -m copula_inter.finetune_marginal wandb.mode=disabled training.steps=20
"""

from __future__ import annotations

import math
import os
import time
from typing import TYPE_CHECKING, Optional, Sequence, cast

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from copula_inter.artifacts import atomic_torch_save
from copula_inter.config_path import config_dict, config_dir
from copula_inter.lora import (
    merged_base_state_dict_any,
)
from copula_inter.marginal_backbones import (  # noqa: E402
    MarginalBackbone,  # noqa: E402
    kfold_quantiles_grad,
    load_backbone,
)
from copula_inter.marginal_data import (
    ERA5EpisodeSampler,
    _build_gp_val_batches,
    _generate_phase_a_gp_batch,
    _gp_cfg,
    build_era5_marginal_val_batches,
    stack_episodes,
)
from copula_inter.marginal_objective import (
    AnchorPenalty,
    MarginalLossWeights,
    analytic_marginal_targets,
    episode_fold_targets,
    marginal_metrics,
    marginal_objective,
    oracle_marginal_nll,
)
from copula_inter.marginal_tiers import apply_tier
from copula_inter.pit import (
    DEFAULT_K_FOLDS,
    _kernel_fn_from_task,
    load_tabicl,  # noqa: E402
    normalize_targets,
    run_pit_batched_grad,
)
from copula_inter.rng import seed_everything
from copula_inter.training_core import cosine_lr_lambda  # noqa: E402

if TYPE_CHECKING:
    from tabicl._model.tabicl import TabICL


def _tabicl_module(tabicl: TabICL | MarginalBackbone) -> TabICL:
    """The TabICL module itself (a tabicl MarginalBackbone wraps one)."""
    return cast("TabICL", tabicl.module) if isinstance(tabicl, MarginalBackbone) else tabicl


def phase_a_batch_loss(
    tabicl: "TabICL | MarginalBackbone",
    episodes: Sequence[dict],
    weights: MarginalLossWeights,
    *,
    k_folds: int = DEFAULT_K_FOLDS,
    folds_per_step: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: str | torch.device = "cuda",
    eps: float = 1e-6,
    timings: Optional[dict[str, float]] = None,
    marginal_probs_n: "int | None" = None,
) -> dict:
    """Forward and loss of one Phase-A step on a batch of GP episodes.

    Scores the N test rows against the full context and folds_per_step of the K
    training folds against their K-1-fold context (default all K). Episodes
    whose kernel cannot be rebuilt get no distillation target but still
    contribute sample-score terms. tabicl is a TabICL module or a
    MarginalBackbone (marginal_backbones.kfold_quantiles_grad, same fold geometry).
    """

    def _mark(name: str, started: float) -> float:
        if timings is not None:
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            timings[name] = timings.get(name, 0.0) + now - started
            return now
        return time.perf_counter()

    if timings is not None and torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    t_part = time.perf_counter()
    B = len(episodes)
    batch = stack_episodes(episodes, device)
    t_part = _mark("collate", t_part)
    P = batch["x_train"].shape[1]
    K = max(2, min(int(k_folds), P))

    # Sample only non-empty folds (for P < K some ceil(P/K) blocks are empty).
    fold_size = math.ceil(P / K)
    n_folds_eff = math.ceil(P / fold_size)
    if folds_per_step is None or folds_per_step >= n_folds_eff:
        fold_subset = None
    else:
        n_f = max(1, int(folds_per_step))
        perm = torch.randperm(n_folds_eff, generator=generator)[:n_f]
        fold_subset = sorted(perm.tolist())

    is_backbone = isinstance(tabicl, MarginalBackbone) and tabicl.name != "tabicl"
    if is_backbone:
        # None: score the model's native decoder grid (999 levels).
        probs = (
            None
            if marginal_probs_n is None
            else np.linspace(
                1.0 / (marginal_probs_n + 1),
                marginal_probs_n / (marginal_probs_n + 1),
                marginal_probs_n,
            )
        )
        assert isinstance(tabicl, MarginalBackbone)
        out = kfold_quantiles_grad(
            tabicl,
            batch["x_train"],
            batch["y_train_scaled"],
            batch["x_test"],
            batch["y_test_scaled"],
            k_folds=K,
            probs=probs,
            fold_subset=fold_subset,
        )
        quantile_dist = tabicl.quantile_dist_module(probs)
        q_test, q_train = out["q_test"], out["q_train"]
    else:
        module = _tabicl_module(tabicl)
        out = run_pit_batched_grad(
            module,
            batch["x_train"],
            batch["y_train_scaled"].unsqueeze(-1),
            batch["x_test"],
            batch["y_test_scaled"].unsqueeze(-1),
            k_folds=K,
            eps=eps,
            return_quantiles=True,
            fold_subset=fold_subset,
            compute_pit=False,
            fuse_folds=True,
            Y_train_raw=batch["y_train_raw"].unsqueeze(-1),
        )
        quantile_dist = module.quantile_dist
        q_test = out["q_test"].squeeze(2)  # (B, N, Q)
        q_train = out["q_train"].squeeze(2)  # (B, P', Q)
    t_part = _mark("tabicl_forward", t_part)
    if fold_subset is None:
        train_idx = torch.arange(P, device=q_train.device)
    else:
        train_idx = out["train_query_idx"]

    # --- analytic targets, per episode, in normalize_targets space ---------
    M = q_test.shape[1] + q_train.shape[1]
    mu_all = torch.zeros(B, M, device=q_test.device)
    sig_all = torch.ones(B, M, device=q_test.device)
    mask_all = torch.zeros(B, M, dtype=torch.bool, device=q_test.device)
    n_ok = 0
    # Compute analytic targets without autograd.
    with torch.no_grad():
        for b, ep in enumerate(episodes):
            try:
                kernel_fn, nugget = _kernel_fn_from_task(ep)
                mu_te, sig_te = analytic_marginal_targets(
                    ep,
                    batch["x_train"][b],
                    batch["y_train_raw"][b],
                    batch["x_test"][b],
                    kernel_fn=kernel_fn,
                    nugget=nugget,
                    use_cached_full_context=True,
                )
                mu_tr, sig_tr = episode_fold_targets(ep, train_idx, K, device=device)
            except (NotImplementedError, KeyError):
                continue  # configured sample scores may apply; target does not
            n_ok += 1
            m, sd = batch["y_mean"][b], batch["y_std"][b]
            mu_all[b] = torch.cat([(mu_te - m) / sd, (mu_tr - m) / sd])
            sig_all[b] = torch.cat([sig_te / sd, sig_tr / sd])
            mask_all[b] = True
    t_part = _mark("analytic_targets", t_part)

    q_all = torch.cat([q_test, q_train], dim=1)  # (B, N+P', Q)
    y_all = torch.cat([batch["y_test_scaled"], batch["y_train_scaled"][:, train_idx]], dim=1)  # (B, N+P')

    Q = q_all.shape[-1]
    q_flat = q_all.reshape(-1, Q)
    y_flat = y_all.reshape(-1)
    mu_flat = mu_all.reshape(-1)
    sig_flat = sig_all.reshape(-1)
    mask_flat = mask_all.reshape(-1)

    res = marginal_objective(
        q_flat,
        y_flat,
        quantile_dist,
        weights,
        mu=mu_flat,
        sigma=sig_flat,
        target_mask=mask_flat,
    )
    # Report the pre-sort quantile crossing rate (a decoder collapse shows up here).
    res["raw_crossing_frac"] = float((q_flat[:, 1:] < q_flat[:, :-1]).float().mean().detach())
    _mark("objective", t_part)
    res["n_episodes_with_target"] = n_ok
    res["oracle_nll"] = (
        oracle_marginal_nll(y_flat[mask_flat], mu_flat[mask_flat], sig_flat[mask_flat]) if n_ok else float("nan")
    )
    # Gap of the model's marginal NLL to the analytic floor on these rows.
    res["nll_gap_to_oracle"] = float(res["nll"].detach()) - res["oracle_nll"]
    return res


@torch.no_grad()
def validate_era5_marginal(
    tabicl: "TabICL | MarginalBackbone",
    batches: dict,
    *,
    eps: float = 1e-6,
    marginal_probs_n: "int | None" = None,
) -> dict:
    """Marginal metrics per ERA5 region and their means: val_marginal/<region>/{nll, crps, ece, ks, clamp_frac}, val_marginal/mean_*.

    Query points are outside the context, so one full-context forward
    (fold_subset=[]) is used.
    """
    per_region: dict[str, dict] = {}
    is_backbone = isinstance(tabicl, MarginalBackbone) and tabicl.name != "tabicl"
    if is_backbone:
        probs = (
            None
            if marginal_probs_n is None
            else np.linspace(1.0 / (marginal_probs_n + 1), marginal_probs_n / (marginal_probs_n + 1), marginal_probs_n)
        )
        assert isinstance(tabicl, MarginalBackbone)
        quantile_dist = tabicl.quantile_dist_module(probs)

    for region, b in batches.items():
        y_tr_list, y_te_list, std = [], [], []
        for d in range(b["y_train"].shape[0]):
            a, c, _, sd = normalize_targets(b["y_train"][d], b["y_test"][d])
            y_tr_list.append(a)
            y_te_list.append(c)
            std.append(sd)
        y_tr_s = torch.stack(y_tr_list)
        y_te_s = torch.stack(y_te_list)
        std_t = torch.stack(std)

        if is_backbone:
            assert isinstance(tabicl, MarginalBackbone)
            xtr = b["x_train"].detach().cpu().numpy()
            ytr = y_tr_s.detach().cpu().numpy()
            xte = b["x_test"].detach().cpu().numpy()
            days = b["y_train"].shape[0]
            q = tabicl.quantile_forward(
                [xtr[d] for d in range(days)],
                [ytr[d] for d in range(days)],
                [xte[d] for d in range(days)],
                probs,
            )
        else:
            module = _tabicl_module(tabicl)
            out = run_pit_batched_grad(
                module,
                b["x_train"],
                y_tr_s.unsqueeze(-1),
                b["x_test"],
                y_te_s.unsqueeze(-1),
                k_folds=2,
                eps=eps,
                return_quantiles=True,
                fold_subset=[],
                compute_pit=False,
                Y_train_raw=b["y_train"].unsqueeze(-1),
            )
            q = out["q_test"].squeeze(2)  # (days, N, Q)
            quantile_dist = module.quantile_dist

        # Per-day std for the raw-nats conversion.
        n_q = q.shape[1]
        log_std = std_t.log().unsqueeze(1).expand(-1, n_q).reshape(-1)
        y_std = std_t.unsqueeze(1).expand(-1, n_q).reshape(-1)
        per_region[region] = marginal_metrics(
            q.reshape(-1, q.shape[-1]),
            y_te_s.reshape(-1),
            quantile_dist,
            log_std=log_std,
            y_std=y_std,
            eps=eps,
        )

    metrics: dict[str, float] = {}
    for region, m in per_region.items():
        for k in ("nll", "crps", "ece", "ks", "clamp_frac"):
            metrics[f"val_marginal/{region}/{k}"] = m[k]
    if per_region:
        for k in ("nll", "crps", "ece", "ks", "clamp_frac"):
            metrics[f"val_marginal/mean_{k}"] = float(np.mean([m[k] for m in per_region.values()]))
    return metrics


@torch.no_grad()
def validate_synthetic_marginal(
    tabicl: "TabICL | MarginalBackbone",
    episode_batches: Sequence[Sequence[dict]],
    *,
    k_folds: int = DEFAULT_K_FOLDS,
    eps: float = 1e-6,
    device: str | torch.device = "cuda",
    marginal_probs_n: "int | None" = None,
) -> dict:
    """Marginal metrics on the fixed GP validation set: val_marginal/gp/nll, nll_oracle, nll_gap_to_oracle and the training objective."""
    # Also report the training objective on the validation episodes.
    metric_w = MarginalLossWeights(distill=1.0, nll=0.0, crps=0.0)
    nlls, crpss, distills, oracles, crossings = [], [], [], [], []
    for episodes in episode_batches:
        res = phase_a_batch_loss(
            tabicl,
            episodes,
            metric_w,
            k_folds=k_folds,
            folds_per_step=None,
            device=device,
            eps=eps,
            marginal_probs_n=marginal_probs_n,
        )
        nlls.append(float(res["nll"]))
        crpss.append(float(res["crps"]))
        distills.append(float(res["distill"]))
        oracles.append(res["oracle_nll"])
        crossings.append(res["raw_crossing_frac"])
    out = {
        "val_marginal/gp/nll": float(np.mean(nlls)) if nlls else float("nan"),
        "val_marginal/gp/crps": float(np.mean(crpss)) if crpss else float("nan"),
        "val_marginal/gp/distill": (float(np.mean(distills)) if distills else float("nan")),
        "val_marginal/gp/nll_oracle": float(np.nanmean(oracles)) if oracles else float("nan"),
        "val_marginal/gp/raw_crossing_frac": (float(np.mean(crossings)) if crossings else float("nan")),
    }
    out["val_marginal/gp/nll_gap_to_oracle"] = out["val_marginal/gp/nll"] - out["val_marginal/gp/nll_oracle"]
    return out


def save_marginal_checkpoint(
    path: str,
    backbone: nn.Module,
    tabicl_config: dict,
    *,
    step: int,
    cfg: DictConfig | None = None,
    extra: Optional[dict] = None,
) -> None:
    """Write a TabICL-schema checkpoint ({"config", "state_dict"}, LoRA merged) that pit.load_tabicl can read.

    step, cfg and extra are stored alongside.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "config": dict(tabicl_config),
        "state_dict": merged_base_state_dict_any(backbone),
        "step": int(step),
    }
    if cfg is not None:
        from omegaconf import OmegaConf

        payload["cfg"] = OmegaConf.to_container(cfg, resolve=True)
    if extra:
        payload.update(extra)
    atomic_torch_save(payload, path)


def _resolve_device(spec: str) -> str:
    if spec != "auto":
        return spec
    return "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path=config_dir(__file__), config_name="finetune_marginal", version_base=None)
def main(cfg: DictConfig) -> None:
    device = _resolve_device(str(cfg.training.device))
    torch.set_float32_matmul_precision(str(cfg.training.matmul_precision))
    seed_everything(int(cfg.seed))
    print(OmegaConf.to_yaml(cfg))

    # ---- model + tier routing -------------------------------------------
    backbone_name = str(cfg.marginal.get("backbone", "tabicl"))
    _probs_n_cfg = cfg.marginal.get("probs_n", None)
    marginal_probs_n = None if _probs_n_cfg is None else int(_probs_n_cfg)
    tabicl: TabICL | MarginalBackbone
    trainable_module: nn.Module
    if backbone_name == "tabicl":
        # Unchanged path: load_tabicl owns TabICL's own checkpoint schema.
        tabicl, tabicl_config = load_tabicl(str(cfg.marginal.ckpt), device, trainable=True, return_config=True)
        trainable_module = tabicl
    else:
        backbone_obj = load_backbone(backbone_name, ckpt=cfg.marginal.get("resume_ckpt", None), device=device)
        if backbone_name == "exaone":
            backbone_obj.exaone_chunk_size = int(cfg.marginal.exaone.chunk_size)
            if backbone_obj.exaone_chunk_size < 1:
                raise ValueError("marginal.exaone.chunk_size must be positive")
            backbone_obj.exaone_activation_checkpointing = bool(cfg.marginal.exaone.activation_checkpointing)
        tabicl, tabicl_config = backbone_obj, {}
        trainable_module = backbone_obj.module
        for p_ in trainable_module.parameters():
            p_.requires_grad_(True)
    report = apply_tier(
        trainable_module,
        int(cfg.marginal.tier),
        lora_rank=int(cfg.marginal.lora_rank),
        lora_alpha=float(cfg.marginal.lora_alpha),
        lora_target=str(cfg.marginal.lora_target),
        backbone_name=backbone_name,
        all_layers=bool(cfg.marginal.get("lora_all_layers", True)),
    )
    trainable_module.to(device)
    print(
        f"[{report.get('backbone', 'tabicl')} tier {report['tier']}] {report['tier_desc']}: "
        f"{report['n_trainable_params']:,} / {report['n_total_params']:,} trainable "
        f"({100 * report['trainable_frac']:.2f}%), "
        f"{report['lora_modules_replaced']} LoRA module(s) at rank {report['lora_rank']}"
    )

    weights = MarginalLossWeights(
        distill=float(cfg.marginal.loss.distill),
        nll=float(cfg.marginal.loss.nll),
        crps=float(cfg.marginal.loss.crps),
        pinball=float(cfg.marginal.loss.pinball),
        anchor=float(cfg.marginal.loss.anchor),
        huber_delta=float(cfg.marginal.loss.huber_delta),
        tail_power=float(cfg.marginal.loss.tail_power),
    )
    # Separate loss weights for synthetic and ERA5 batches.
    era5_loss_cfg = cfg.marginal.era5.loss
    era5_weights = MarginalLossWeights(
        distill=0.0,
        nll=float(era5_loss_cfg.nll),
        crps=float(era5_loss_cfg.crps),
        pinball=float(era5_loss_cfg.pinball),
        anchor=weights.anchor,
        huber_delta=weights.huber_delta,
        tail_power=weights.tail_power,
    )
    anchor = AnchorPenalty(trainable_module) if weights.anchor > 0 else None

    # AdamW over all trainable parameters in one group.
    params = [p for p in trainable_module.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("Tier routing left no trainable parameters.")
    adam_eps = 1e-4 if any(p.dtype == torch.float16 for p in params) else 1e-8
    opt = torch.optim.AdamW(
        params,
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        eps=adam_eps,
    )
    total_steps = int(cfg.training.steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: cosine_lr_lambda(
            s,
            int(cfg.training.warmup_steps),
            total_steps,
            float(cfg.training.lr_min_frac),
        ),
    )

    # ---- data ------------------------------------------------------------
    gp_cfg = _gp_cfg(cfg)
    eps = float(cfg.marginal.pit_eps)
    k_folds = int(cfg.marginal.k_folds)
    folds_per_step = cfg.marginal.folds_per_step
    folds_per_step = None if folds_per_step is None else int(folds_per_step)
    mix_frac = float(cfg.marginal.era5.mix_frac)
    if not 0.0 <= mix_frac <= 1.0:
        raise ValueError(f"marginal.era5.mix_frac must be in [0, 1], got {mix_frac}")

    def _has_sample_objective(w: MarginalLossWeights) -> bool:
        return any(value != 0.0 for value in (w.distill, w.nll, w.crps, w.pinball))

    # Fail if every loss weight is zero.
    if mix_frac < 1.0 and not _has_sample_objective(weights):
        raise ValueError("Synthetic batches have no non-zero marginal loss weight.")
    if mix_frac > 0.0 and not _has_sample_objective(era5_weights):
        raise ValueError("ERA5 batches have no non-zero marginal loss weight.")

    era5_sampler = None
    if mix_frac > 0:
        from eval.data.era5_global_corpus import GlobalERA5Corpus

        corpus = GlobalERA5Corpus(
            str(cfg.marginal.era5.corpus_dir),
            max_months=int(cfg.marginal.era5.max_months),
        )
        era5_sampler = ERA5EpisodeSampler(
            corpus,
            grid_size=int(cfg.marginal.era5.grid_size),
            n_context=int(cfg.marginal.era5.n_context),
            box_deg_range=(
                float(cfg.marginal.era5.box_deg_min),
                float(cfg.marginal.era5.box_deg_max),
            ),
            seed=int(cfg.seed) + 7717,
        )
        print(f"[era5] mixture on: {corpus.n_days_total} days loaded, mix_frac={mix_frac}")

    print("[val] building fixed validation sets (one-off ERA5 fetch/crop)...")
    era5_val = build_era5_marginal_val_batches(cfg.validation, device)
    gp_val = _build_gp_val_batches(cfg, device)
    print(f"[val] {len(era5_val)} ERA5 region(s), {len(gp_val)} synthetic GP batch(es)")

    # ---- wandb -----------------------------------------------------------
    run = None
    if str(cfg.wandb.mode) != "disabled":
        import wandb

        run = wandb.init(
            project=str(cfg.wandb.project),
            entity=cfg.wandb.entity,
            config=config_dict(cfg),
            mode=cfg.wandb.mode,
        )
        wandb.watch(trainable_module, log="gradients", log_freq=max(1, int(cfg.training.log_every)))
        wandb.log({f"model/{k}": v for k, v in report.items() if isinstance(v, (int, float))}, step=0)

    def _log(payload: dict, step: int) -> None:
        if run is not None:
            run.log(payload, step=step)

    def _validate(step: int) -> dict[str, float]:
        t0 = time.time()
        t_era5 = time.time()
        metrics = validate_era5_marginal(tabicl, era5_val, eps=eps, marginal_probs_n=marginal_probs_n)
        metrics["val_marginal/era5_seconds"] = time.time() - t_era5
        t_gp = time.time()
        metrics.update(
            validate_synthetic_marginal(
                tabicl,
                gp_val,
                k_folds=k_folds,
                eps=eps,
                device=device,
                marginal_probs_n=marginal_probs_n,
            )
        )
        metrics["val_marginal/gp_seconds"] = time.time() - t_gp
        metrics["val_marginal/seconds"] = time.time() - t0
        _log(metrics, step)
        print(
            f"[val step {step}] "
            f"era5 nll={metrics.get('val_marginal/mean_nll', float('nan')):.4f} "
            f"ece={metrics.get('val_marginal/mean_ece', float('nan')):.4f} "
            f"ks={metrics.get('val_marginal/mean_ks', float('nan')):.4f} | "
            f"gp nll={metrics.get('val_marginal/gp/nll', float('nan')):.4f} "
            f"distill={metrics.get('val_marginal/gp/distill', float('nan')):.4f} "
            f"oracle={metrics.get('val_marginal/gp/nll_oracle', float('nan')):.4f} "
            f"gap={metrics.get('val_marginal/gp/nll_gap_to_oracle', float('nan')):.4f} | "
            f"{metrics['val_marginal/seconds']:.2f}s "
            f"(era5 {metrics['val_marginal/era5_seconds']:.2f}s, "
            f"gp {metrics['val_marginal/gp_seconds']:.2f}s)"
        )
        return metrics

    def _save(step: int, tag: str = "") -> str | None:
        if cfg.training.ckpt_dir is None:
            return None
        name = f"step_{step:07d}{tag}.pt"
        path = os.path.join(str(cfg.training.ckpt_dir), name)
        tier_extra = {"tier_report": {k: v for k, v in report.items() if isinstance(v, (int, float, str))}}
        if isinstance(tabicl, MarginalBackbone):
            # Non-TabICL backbones write their own checkpoint format.
            tabicl.save(path, step=step, cfg=cfg, extra=tier_extra)
        else:
            save_marginal_checkpoint(
                path,
                tabicl,
                tabicl_config,
                step=step,
                cfg=cfg,
                extra=tier_extra,
            )
        print(f"[ckpt] {path}")
        return path

    # ---- train -----------------------------------------------------------
    initial_metrics = _validate(0)
    selection_metric = str(cfg.training.get("selection_metric", "val_marginal/mean_nll"))
    if selection_metric not in initial_metrics:
        raise KeyError(
            f"training.selection_metric={selection_metric!r} was not emitted by "
            f"validation. Available metrics: {sorted(initial_metrics)}"
        )
    best_value = float(initial_metrics[selection_metric])
    if not math.isfinite(best_value):
        raise RuntimeError(f"Initial selection metric {selection_metric} is non-finite: {best_value}")
    best_step = 0
    selection_min_delta = float(cfg.training.get("selection_min_delta", 0.0))

    def _snapshot_trainable() -> dict[str, torch.Tensor]:
        # Keep only the trainable tensors for best-checkpoint selection.
        return {name: p.detach().cpu().clone() for name, p in trainable_module.named_parameters() if p.requires_grad}

    def _restore_trainable(state: dict[str, torch.Tensor]) -> None:
        named = dict(trainable_module.named_parameters())
        with torch.no_grad():
            for name, value in state.items():
                named[name].copy_(value.to(device=named[name].device))

    best_state = _snapshot_trainable()

    def _consider_validation(step: int, metrics: dict[str, float]) -> None:
        nonlocal best_step, best_value, best_state
        value = float(metrics[selection_metric])
        if math.isfinite(value) and value < best_value - selection_min_delta:
            best_step = step
            best_value = value
            best_state = _snapshot_trainable()
            print(f"[selection] new best {selection_metric}={best_value:.6f} at step {best_step}")

    rng = np.random.default_rng(int(cfg.seed) + 991)
    gen = torch.Generator().manual_seed(int(cfg.seed) + 13)
    B = int(cfg.training.batch_size)
    t_last = time.time()
    profile_steps = int(cfg.training.get("profile_steps", 0))
    profile_totals: dict[str, float] = {}

    for step in range(1, total_steps + 1):
        profiling = step <= profile_steps
        if profiling and device.startswith("cuda"):
            torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        data_started = step_started
        use_era5 = era5_sampler is not None and rng.random() < mix_frac
        if use_era5:
            assert era5_sampler is not None
            episodes = era5_sampler.batch(B)
            episodes = [{k: v.to(device) for k, v in ep.items()} for ep in episodes]
            w = era5_weights
        else:
            gp_cfg.seed = int(cfg.seed) * 1_000_003 + step
            episodes = _generate_phase_a_gp_batch(gp_cfg, B, device)
            w = weights
        if profiling and device.startswith("cuda"):
            torch.cuda.synchronize(device)
        data_seconds = time.perf_counter() - data_started

        part_timings: dict[str, float] | None = {} if profiling else None
        res = phase_a_batch_loss(
            tabicl,
            episodes,
            w,
            k_folds=k_folds,
            folds_per_step=folds_per_step,
            generator=gen,
            device=device,
            eps=eps,
            timings=part_timings,
            marginal_probs_n=marginal_probs_n,
        )
        loss = res["loss"]
        anchor_val = 0.0
        if anchor is not None:
            a = anchor(trainable_module)
            loss = loss + weights.anchor * a
            anchor_val = a.detach().item()

        backward_started = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, float(cfg.training.clip_grad_norm))
        if profiling and device.startswith("cuda"):
            torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - backward_started
        optimizer_started = time.perf_counter()
        opt.step()
        sched.step()
        if profiling and device.startswith("cuda"):
            torch.cuda.synchronize(device)
        optimizer_seconds = time.perf_counter() - optimizer_started

        if profiling:
            measured = {
                "data": data_seconds,
                **(part_timings or {}),
                "backward_and_clip": backward_seconds,
                "optimizer": optimizer_seconds,
                "total": time.perf_counter() - step_started,
            }
            for key, value in measured.items():
                profile_totals[key] = profile_totals.get(key, 0.0) + value
            print("[profile step %d] %s" % (step, " ".join(f"{key}={value:.4f}s" for key, value in measured.items())))
            if step == profile_steps:
                means = {key: value / profile_steps for key, value in profile_totals.items()}
                print("[profile mean] " + " ".join(f"{key}={value:.4f}s" for key, value in means.items()))
                _log({f"profile/{key}_seconds": value for key, value in means.items()}, step)

        if step % int(cfg.training.log_every) == 0:
            dt = (time.time() - t_last) / int(cfg.training.log_every)
            t_last = time.time()
            payload = {
                "train/loss": loss.detach().item(),
                "train/nll": res["nll"].detach().item(),
                "train/crps": res["crps"].detach().item(),
                "train/pinball": res["pinball"].detach().item(),
                "train/distill": res["distill"].detach().item(),
                "train/raw_crossing_frac": res["raw_crossing_frac"],
                "train/anchor": anchor_val,
                "train/grad_norm": gnorm.detach().item(),
                "train/lr": sched.get_last_lr()[0],
                "train/sec_per_step": dt,
                "train/is_era5_batch": float(use_era5),
                "train/P": int(episodes[0]["x_norm_train"].shape[0]),
            }
            if not use_era5:
                payload["train/nll_oracle"] = res["oracle_nll"]
                payload["train/nll_gap_to_oracle"] = res["nll_gap_to_oracle"]
            _log(payload, step)
            print(
                f"step {step:>7} loss={loss.detach().item():.4f} "
                f"nll={res['nll'].detach().item():.4f} "
                f"distill={res['distill'].detach().item():.4f} "
                f"pinball={res['pinball'].detach().item():.4f} "
                f"cross={res['raw_crossing_frac']:.3%} "
                f"gap={res.get('nll_gap_to_oracle', float('nan')):.4f} "
                f"lr={sched.get_last_lr()[0]:.2e} {dt:.2f}s/step" + ("  [era5]" if use_era5 else "")
            )

        hooks_started = time.time()
        if step % int(cfg.training.val_every) == 0:
            _consider_validation(step, _validate(step))
        if step % int(cfg.training.save_every) == 0:
            _save(step)
        # Do not charge validation/checkpoint I/O to the next sec_per_step window.
        t_last += time.time() - hooks_started

    if total_steps % int(cfg.training.val_every) != 0:
        _consider_validation(total_steps, _validate(total_steps))

    if bool(cfg.training.get("restore_best", True)):
        _restore_trainable(best_state)
        print(f"[selection] restored step {best_step} with {selection_metric}={best_value:.6f} before final export")
        export_step = best_step
    else:
        export_step = total_steps
    final = _save(export_step, tag="_final")
    if final:
        print(
            "\nPhase A done. Use it as the copula run's marginal with:\n"
            f"    python -m copula_inter.train tabicl.pit_ckpt={os.path.abspath(final)}\n"
            "and measure it first with:\n"
            f"    python eval/runners/marginal_calibration_eval.py --ckpt {os.path.abspath(final)}"
        )
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
