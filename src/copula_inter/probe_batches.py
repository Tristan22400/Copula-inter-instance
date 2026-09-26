"""Fixed validation probe batches and their z_train/z_test: synthetic kernel families, posterior probes, TabICL and analytic PIT."""

from __future__ import annotations

import os

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

import zlib

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

# eval/ (regions.py, spatial-correlation probe helpers -- see
# _build_era5_val_batches below) lives at the repo root, not under src/.
from copula_inter.classical_kernels import DEFAULT_FAMILIES
from copula_inter.data_gen import KERNEL_REGISTRY, generate_gp_batch
from copula_inter.dataset import (
    collate_fn,
)
from copula_inter.pit import (
    gp_analytical_pit,
    normalize_targets,
    run_pit,
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
        episodes = generate_gp_batch(synth_cfg, n_episodes, device=device, return_kernel_metadata=True)
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
    episodes = generate_gp_batch(probe_cfg, n_episodes, device=device, return_kernel_metadata=True)
    batch = collate_fn(episodes)
    return {"episodes": episodes, "batch": {k: v.to(device) for k, v in batch.items()}}


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
        pit_out = run_pit(
            tabicl_marginal, X_b, Y_b, X_te, Y_te, k_folds=k_folds,
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

    What this is a contrast AGAINST depends on data.z_train_source, and the
    distinction matters when reading val/y_nll_*:

      - "analytic": the batch carries the exact GP-LOO PIT, so this really is
        the sim-to-real substitution it was written for -- oracle marginal
        replaced by a real, imperfect one.
      - "tabicl"/"tabicl_split" (the production default): the batch ALREADY
        carries a TabICL PIT (live_dataset.build_fixed_live_val_batches passes
        the frozen TabICL into generate_gp_batch). This is then a second
        estimate of the same thing, not a contrast, and the two only agree if
        they use the same fold count -- hence `k_folds` is passed as
        data.z_train_tabicl_k_folds rather than tabicl.pit_k_folds in that case
        (see train()'s val_pit_k_folds). A K-fold PIT's sharpness moves with K,
        so scoring at a different K than the model was conditioned on shifts
        val/y_nll_total by an amount that has nothing to do with the model.

    The oracle-side counterpart is _build_analytic_val_z, which supplies the
    exact GP PIT for oracle_diag/* regardless of which of those two the batch
    happens to carry.

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
def _build_analytic_val_z(
    val_loader, val_episodes_meta: dict[int, list[dict]], device: str,
) -> dict[int, dict[str, torch.Tensor]]:
    """The EXACT analytic GP PIT (pit.gp_analytical_pit) for every val episode,
    cached once at startup -- the oracle counterpart of _build_tabicl_val_z.

    Why this exists at all, given val_loader already carries a z_train/z_test/
    log_pdf_test triple. Under data.z_train_source="tabicl" (the production
    default, and the regime this repo actually deploys in),
    live_dataset.build_fixed_live_val_batches hands the frozen TabICL to
    generate_gp_batch, and data_gen's TabICL branch overwrites z_train AND
    z_test/log_pdf_test with TabICL's K-fold PIT. So batch["z_test"] is in
    TABICL's z-space, while gp_analytical_posterior's ceiling
    (val/y_nll_oracle_posterior*) is in the exact GP-POSTERIOR z-space. Two
    Sklar splits taken at DIFFERENT marginals are not comparable term by term
    -- only a Y-space total is (see eval/metrics/joint_nll.py's module
    docstring) -- so scoring oracle_diag/gap_nll or the oracle_diag/corr_*
    statistics across that boundary compares two different quantities and the
    "gap" is not a bound. This cache restores the missing operand: the same
    episodes, standardized the way the ceiling is.

    No regeneration and no TabICL forward pass. val_episodes_meta's episode
    dicts are the ones val_loader's batches were built from, kernel metadata
    and cached _L_ff/_alpha intact, so gp_analytical_pit is one triangular
    solve per episode.

    Returns _build_tabicl_val_z's shape -- {batch_idx: {"z_train": (B, P_max),
    "z_test": (B, N_max), "log_pdf_test": (B, N_max)}}, CPU, zero-padded
    outside each episode's true train/test length (matching train_mask/
    test_mask) -- so validate() consumes the two caches identically.

    Callers should skip this entirely when data.z_train_source="analytic":
    the batch then already carries exactly these tensors, and paying for them
    twice buys nothing (see train()'s call site).

    `device` is unused -- the episodes' cached _L_ff/_alpha are CPU tensors
    (build_fixed_live_val_batches moves them there deliberately) and the
    result is cached on CPU like _build_tabicl_val_z's, so there is nothing
    to move. Kept in the signature for call-site symmetry with that function.
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
