"""Adaptive kernel-family weights and the TabICL/analytic z_train mix fraction."""

from __future__ import annotations

import gc
import math
from typing import Optional

import matplotlib

from copula_inter.probe_batches import _name_seed

matplotlib.use("Agg")
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from copula_inter.data_gen import _COMPOSABLE_KERNELS, _generate_gp_batch_raw
from copula_inter.live_dataset import (
    limited_main_process_threads,
)
from copula_inter.pit import (
    DEFAULT_K_FOLDS,
    load_tabicl,
)


def _update_adaptive_kernel_weights(
    prev_weights: torch.Tensor, metrics: dict, lr: float, floor: float,
    exclude: Optional[set] = None, signal: str = "oracle",
) -> torch.Tensor:
    """Exponentiated-gradient update of per-family kernel sampling weights (_COMPOSABLE_KERNELS order).

    The per-family gap is oracle_diag/kernel_fit/<family>/gap_nll for
    signal="oracle", or kernel_fit/<family>/gap_nll_tabicl for signal="tabicl"
    (falling back to the oracle gap where missing). Missing families and
    families in exclude get gap 0.

        w' = prev * exp(lr * gap), normalized
        w = (1 - floor) * w' + floor / n_families

    Returns a new tensor; the caller copies it into the shared tensor.
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
    # Clamp the exponent so one extreme gap cannot overflow exp().
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
    """Per kernel family, mean |z_tabicl - z_analytic| of z_train on the same episodes.

    Calls _generate_gp_batch_raw twice per family with the same seed on the same
    device, with and without tabicl_marginal, so the episodes are identical. Uses
    the baselines.synth_* / probe_* settings with seed + 1.

    Returns:
        {family: gap} for families with at least one valid episode (two
        independent standard normals give about 1.13).
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
        # Cap threads for generation in the main process.
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
    """Map per-family gaps to mixing fractions in [floor_frac, max_frac] (_COMPOSABLE_KERNELS order).

    Gaps are min-max normalized over the measured families; unmeasured families,
    or all families when the gaps are equal, get floor_frac.
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
    """Reload the TabICL marginal, re-measure the gaps and copy new mix fractions into tabicl_mix_weights in place.

    No-op when floor_frac == max_frac.
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
        # gc.collect() before empty_cache() so cyclic references release CUDA memory.
        gc.collect()
        torch.cuda.empty_cache()
    return z_gap, new_mix_frac
