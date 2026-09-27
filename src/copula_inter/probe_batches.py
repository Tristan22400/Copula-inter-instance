"""Fixed validation probe batches and their z_train/z_test: synthetic kernel families, posterior probes, TabICL and analytic PIT."""

from __future__ import annotations

import zlib
from typing import Iterable

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from copula_inter.classical_kernels import DEFAULT_FAMILIES
from copula_inter.data_gen import generate_gp_batch
from copula_inter.dataset import (
    collate_fn,
)
from copula_inter.gp_kernels import KERNEL_REGISTRY
from copula_inter.pit import (
    TabICLLike,
    gp_analytical_pit,
    normalize_targets,
    run_pit,
)


def _sigma_stats(Sigma: torch.Tensor, mask: torch.Tensor) -> dict:
    """Off-diagonal mean/std and diagonal mean over a batch of correlation matrices.

    Args:
        Sigma: (B, N_max, N_max).
        mask: (B, N_max) bool, True for valid rows.

    Returns:
        dict with offdiag_mean, offdiag_std, diag_mean.
    """
    B, N, _ = Sigma.shape
    ri, ci = torch.triu_indices(N, N, offset=1, device=Sigma.device)
    mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)  # (B, N, N)
    valid_off = mask_2d[:, ri, ci]  # (B, n_pairs)
    off_vals = Sigma[:, ri, ci][valid_off]  # flat valid off-diagonal entries
    diag_vals = Sigma.diagonal(dim1=-2, dim2=-1)[mask]  # flat valid diagonal entries
    if off_vals.numel() == 0:
        return {"offdiag_mean": 0.0, "offdiag_std": 0.0, "diag_mean": 1.0}
    return {
        "offdiag_mean": off_vals.mean().item(),
        "offdiag_std": off_vals.std().item(),
        "diag_mean": diag_vals.mean().item(),
    }


def _corr_quality(off_pred: np.ndarray, off_ora: np.ndarray) -> dict:
    """MSE, MAE, Pearson r and signed bias of predicted vs oracle off-diagonal correlations (1-D arrays)."""
    diff = off_pred - off_ora
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    std_p, std_o = off_pred.std(), off_ora.std()
    pearson = float(np.corrcoef(off_pred, off_ora)[0, 1]) if (std_p > 1e-12 and std_o > 1e-12) else 0.0
    return {"mse": mse, "mae": mae, "pearson": pearson, "bias": bias}


def _name_seed(base_seed: int, name: str) -> int:
    """Deterministic per-name seed derived from base_seed (zlib.crc32)."""
    return base_seed + (zlib.crc32(name.encode()) % 10_000)


def _macro_average(values: list[float]) -> float:
    """Unweighted mean of the finite values, NaN if none."""
    return float(np.mean(values)) if values else float("nan")


def _build_synthetic_kernel_batches(cfg: DictConfig, device: str) -> dict[str, dict]:
    """Fixed probe episodes per kernel family for the kernel_fit/<family> metrics.

    For each family in cfg.baselines.kernels, generate synth_n_episodes episodes
    with that kernel forced, a per-family fixed seed, and P/N ranges from
    baselines.probe_* (not cfg.data), with kernel metadata. Built once.
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
            OmegaConf.create(
                {
                    "seed": family_seed,
                    "data": {
                        "kernel": family,
                        "systematic_composition": False,
                        "P_min": probe_P_min,
                        "P_max": probe_P_max,
                        "N_min": probe_N_min,
                        "N_max": probe_N_max,
                    },
                }
            ),
        )
        episodes = generate_gp_batch(synth_cfg, n_episodes, device=device, return_kernel_metadata=True)
        batch = collate_fn(episodes)
        batches[family] = {"episodes": episodes, "batch": {k: v.to(device) for k, v in batch.items()}}
    return batches


def _build_posterior_probe_batches(cfg: DictConfig, device: str) -> dict:
    """Fixed probe set for the Bayes-optimal-ceiling metrics when val_loader has no kernel metadata.

    Used for on-disk training and real-ERA5 live data. Draws
    baselines.posterior_probe_n_episodes episodes from cfg.data with kernel
    metadata.

    Returns:
        {"episodes": [CPU episode dicts], "batch": collated batch on device}, same order.
    """
    bcfg = cfg.get("baselines", {}) or {}
    n_episodes = int(bcfg.get("posterior_probe_n_episodes", 64))
    base_seed = int(bcfg.get("synth_seed", 20260718)) + 2  # +1 is _compute_tabicl_z_train_gap's
    probe_cfg = OmegaConf.merge(cfg, OmegaConf.create({"seed": base_seed}))
    episodes = generate_gp_batch(probe_cfg, n_episodes, device=device, return_kernel_metadata=True)
    batch = collate_fn(episodes)
    return {"episodes": episodes, "batch": {k: v.to(device) for k, v in batch.items()}}


@torch.no_grad()
def _tabicl_pit_batch(
    batch: dict,
    tabicl_marginal: TabICLLike,
    k_folds: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """TabICL K-fold PIT (pit.run_pit) of each episode in a collated batch.

    Targets are z-scored with pit.normalize_targets; log_pdf_test is converted
    back to raw-y nats.

    Returns:
        CPU {"z_train": (B, P_max), "z_test": (B, N_max), "log_pdf_test":
        (B, N_max)}, zero outside each episode's valid rows.
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
        pit_out = run_pit(
            tabicl_marginal,
            X_b,
            Y_b,
            X_te,
            Y_te,
            k_folds=k_folds,
            Y_train_raw=y_train[b, :n].unsqueeze(-1),
        )
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
    val_loader: Iterable[dict[str, torch.Tensor]],
    tabicl_marginal: TabICLLike,
    k_folds: int,
    device: str,
) -> dict[int, dict[str, torch.Tensor]]:
    """TabICL PIT of every val_loader episode, computed once before training.

    Used for val/y_nll_*. Pass the val episodes' own fold count when they were
    generated with a TabICL z_train.

    Returns:
        {batch_idx: {"z_train", "z_test", "log_pdf_test"}} on CPU, zero-padded.
    """
    cache: dict[int, dict[str, torch.Tensor]] = {}
    for batch_idx, batch in enumerate(val_loader):
        cache[batch_idx] = _tabicl_pit_batch(batch, tabicl_marginal, k_folds, device)
    return cache


@torch.no_grad()
def _build_analytic_val_z(
    val_loader: Iterable[dict[str, torch.Tensor]],
    val_episodes_meta: dict[int, list[dict]],
    device: str,
) -> dict[int, dict[str, torch.Tensor]]:
    """Exact GP PIT (pit.gp_analytical_pit) of every val episode, computed once.

    Gives the oracle_diag/* metrics analytic z when the val batches carry a
    TabICL PIT. Uses val_episodes_meta's cached factors; device is unused.

    Returns:
        Same structure as _build_tabicl_val_z.
    """
    cache: dict[int, dict[str, torch.Tensor]] = {}
    for batch_idx, batch in enumerate(val_loader):
        eps_b = val_episodes_meta.get(batch_idx)
        if not eps_b:
            continue
        B, P_max = batch["y_train"].shape
        N_max = int(batch["y_test"].shape[1])
        z_train = torch.zeros(B, P_max)
        z_test = torch.zeros(B, N_max)
        log_pdf_test = torch.zeros(B, N_max)
        for b, ep in enumerate(eps_b[:B]):
            try:
                pit_out = gp_analytical_pit(ep)
            except (KeyError, NotImplementedError):
                continue  # rare unsupported kernel schema, as elsewhere
            zt = pit_out["z_train"].detach().float().cpu().reshape(-1)
            zs = pit_out["z_test"].detach().float().cpu().reshape(-1)
            lp = pit_out["log_pdf_test"].detach().float().cpu().reshape(-1)
            z_train[b, : zt.shape[0]] = zt
            z_test[b, : zs.shape[0]] = zs
            log_pdf_test[b, : lp.shape[0]] = lp
        cache[batch_idx] = {
            "z_train": z_train,
            "z_test": z_test,
            "log_pdf_test": log_pdf_test,
        }
    return cache


@torch.no_grad()
def _build_tabicl_kernel_fit_z(
    synth_kernel_batches: dict,
    tabicl_marginal: TabICLLike,
    k_folds: int,
    device: str,
) -> dict[str, dict[str, torch.Tensor]]:
    """TabICL PIT of each kernel_fit/<family> probe set, computed once.

    Returns:
        {family: {"z_train", "z_test", "log_pdf_test"}}, as _build_tabicl_val_z.
    """
    return {
        family: _tabicl_pit_batch(probe["batch"], tabicl_marginal, k_folds, device)
        for family, probe in synth_kernel_batches.items()
    }
