"""Adaptive kernel-family weights and the TabICL/analytic z_train mix fraction."""

from __future__ import annotations

import gc
import math
import os

from copula_inter.probe_batches import _name_seed

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

from typing import Optional

import matplotlib

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

# eval/ (regions.py, spatial-correlation probe helpers -- see
# _build_era5_val_batches below) lives at the repo root, not under src/.


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
