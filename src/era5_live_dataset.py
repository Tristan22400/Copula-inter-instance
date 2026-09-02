"""era5_live_dataset.py — real-world analogue of live_dataset.py: an infinite
stream of training episodes drawn from real ARCO-ERA5 2m-temperature data
(eval/data/era5_global_corpus.py) instead of synthetic GP kernels
(data_gen.py), for finetuning a checkpoint on real worldwide spatial data
across many regions and grid resolutions.

Enabled via training.live_generation=true training.live_source=era5 (see
src/train.py) — everything else about the training loop (optimizer,
scheduler, AMP, logging, checkpointing, training.resume_ckpt) is unchanged;
only the DataLoader construction differs from live_dataset.py's GP path.

Unlike LiveGPDataset, there is no oracle Sigma_star/R_star for real data (no
known generative kernel), so episodes here carry only the ingredients
y_space_nll needs (z_train/z_test/log_pdf_test/masks) — training.
aux_mae_weight must be 0 for this source (src/train.py enforces this, same
constraint _build_era5_val_batches already documents for the era5_fit
validation probes). z_train/z_test/log_pdf_test come from the SAME frozen
TabICL K-fold PIT machinery (src/pit.py::run_pit) the data.z_train_source=
tabicl live-generation path already uses for synthetic data — there is no
"analytic" oracle-PIT option for real data, so a TabICL checkpoint is always
required here (unlike LiveGPDataset, where tabicl_device is optional).
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from live_dataset import resolve_live_tabicl_num_workers
from pit import load_tabicl, normalize_targets, resolve_pit_ckpt, run_pit, run_pit_batched

from eval.data.era5_global_corpus import GlobalERA5Corpus, load_shared_corpus_arrays

__all__ = ["build_era5_train_loader", "build_era5_fixed_val_batches", "era5_collate_fn"]


def _limit_worker_threads(_worker_id: int) -> None:
    # Same rationale as live_dataset.py's _limit_worker_threads: each worker
    # runs its own TabICL forward pass, no need for machine-wide BLAS fan-out.
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def era5_collate_fn(samples: List[dict]) -> dict:
    """Pad a batch of variable-P/N real-ERA5 episodes. Deliberately a
    stripped-down sibling of dataset.collate_fn: no R_star/Sigma_star/
    R_prior/mu_star/sigma_star (no oracle exists for real data). Still
    carries y_train/y_test (raw, unscaled) alongside the PIT'd z_train/
    z_test/log_pdf_test — _forward_and_loss (aux_mae_weight forced 0 for
    this source) never reads y_train/y_test, but validate()'s TabICL
    sim-to-real diagnostic (_build_tabicl_val_z -> _tabicl_pit_batch) runs
    unconditionally over every val_loader batch regardless of live_source
    and requires them; dropping them would KeyError the very first
    validate() call."""
    B = len(samples)
    d_x = samples[0]["x_norm_train"].shape[-1]
    P_list = [int(s["x_norm_train"].shape[0]) for s in samples]
    N_list = [int(s["x_norm_test"].shape[0]) for s in samples]
    P_max = max(P_list)
    N_max = max(N_list)

    x_train = torch.zeros(B, P_max, d_x)
    x_test = torch.zeros(B, N_max, d_x)
    y_train = torch.zeros(B, P_max)
    y_test = torch.zeros(B, N_max)
    z_train = torch.zeros(B, P_max)
    z_test = torch.zeros(B, N_max)
    log_pdf_test = torch.zeros(B, N_max)
    train_mask = torch.zeros(B, P_max, dtype=torch.bool)
    test_mask = torch.zeros(B, N_max, dtype=torch.bool)

    for b, s in enumerate(samples):
        P, N = P_list[b], N_list[b]
        x_train[b, :P] = s["x_norm_train"]
        x_test[b, :N] = s["x_norm_test"]
        y_train[b, :P] = s["y_train"]
        y_test[b, :N] = s["y_test"]
        z_train[b, :P] = s["z_train"]
        z_test[b, :N] = s["z_test"]
        log_pdf_test[b, :N] = s["log_pdf_test"]
        train_mask[b, :P] = True
        test_mask[b, :N] = True

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "z_train": z_train,
        "z_test": z_test,
        "log_pdf_test": log_pdf_test,
        "train_mask": train_mask,
        "test_mask": test_mask,
    }


def _pit_episode(
    x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor,
    tabicl_model, k_folds: int,
) -> Optional[dict]:
    """z_train/z_test/log_pdf_test for one (x_train, y_train, x_test, y_test)
    real-ERA5 episode, via the same run_pit + normalize_targets convention
    src/train.py::_tabicl_pit_batch uses per-episode inside its batch loop.
    Returns None if there's too little context for run_pit's fold split."""
    if x_train.shape[0] < 2 or x_test.shape[0] < 1:
        return None
    y_train_scaled, y_test_scaled, _, std = normalize_targets(y_train, y_test)
    Y_train = y_train_scaled.unsqueeze(-1)
    Y_test = y_test_scaled.unsqueeze(-1)
    pit_out = run_pit(tabicl_model, x_train, Y_train, x_test, Y_test, k_folds=k_folds)
    return {
        "z_train": pit_out["z_train"].squeeze(-1),
        "z_test": pit_out["z_test"].squeeze(-1),
        # Jacobian correction back to raw-nats units — see _tabicl_pit_batch's
        # docstring for why (log p_raw = log p_scaled - log(std)).
        "log_pdf_test": pit_out["log_pdf_test"].squeeze(-1) - std.log(),
    }


def _pit_group(
    x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor,
    tabicl_model, k_folds: int,
) -> Optional[dict]:
    """Batched sibling of `_pit_episode`: PITs a whole *group* of B episodes
    that all share P/N (see LiveERA5Dataset.__iter__'s grouped sampling) in
    ONE run_pit_batched call instead of B separate run_pit calls -- mirrors
    src/live_dataset.py::LiveGPDataset's group_size mechanism, cutting
    TabICL's own per-call Python/CUDA-launch overhead from B*(k_folds+1)
    invocations to k_folds+1 (see run_pit_batched's docstring).

    Unlike normalize_targets (single episode, unconditional .mean()/.std()
    reduction), each of the B episodes needs its OWN mean/std computed over
    its own P training points -- reduce over dim=-1 with keepdim instead.

    Args:
        x_train/y_train : (B, P, p_x) / (B, P)
        x_test/y_test   : (B, N, p_x) / (B, N)
    Returns dict of (B, P)/(B, N)/(B, N) z_train/z_test/log_pdf_test, or
    None if there's too little context for run_pit_batched's fold split.
    """
    if x_train.shape[1] < 2 or x_test.shape[1] < 1:
        return None
    mean = y_train.mean(dim=-1, keepdim=True)
    std = y_train.std(dim=-1, keepdim=True).clamp(min=1e-8)
    y_train_scaled = (y_train - mean) / std
    y_test_scaled = (y_test - mean) / std
    Y_train = y_train_scaled.unsqueeze(-1)
    Y_test = y_test_scaled.unsqueeze(-1)
    pit_out = run_pit_batched(tabicl_model, x_train, Y_train, x_test, Y_test, k_folds=k_folds)
    return {
        "z_train": pit_out["z_train"].squeeze(-1),
        "z_test": pit_out["z_test"].squeeze(-1),
        # Per-episode Jacobian correction -- see _pit_episode's docstring.
        "log_pdf_test": pit_out["log_pdf_test"].squeeze(-1) - std.log(),
    }


class LiveERA5Dataset(IterableDataset):
    """Infinite stream of real-ERA5 episodes: each worker attaches to a
    corpus another process already loaded into shared memory (see
    `shared_corpus`/era5_global_corpus.py::load_shared_corpus_arrays) and
    loads its own frozen TabICL copy on `tabicl_device` the first time
    __iter__ runs — mirrors LiveGPDataset's one-time-per-worker TabICL load.
    Requires tabicl_device="cuda" unconditionally (there is no CPU-worker
    path here, same throughput argument live_dataset.py's
    build_live_train_loader docstring makes for data.z_train_source=tabicl).

    `shared_corpus` MUST be built (load_shared_corpus_arrays) in the main
    process before this Dataset's DataLoader spawns workers -- same
    ordering constraint LiveGPDataset's kernel_weights/tabicl_mix_weights
    docstring describes for its own share_memory_() tensors. Every worker
    then attaches to that SAME physical memory (GlobalERA5Corpus.from_shared)
    instead of each re-reading and holding its own ~15GB private copy of the
    corpus, which is what used to force live_tabicl_num_workers down to a
    RAM-bound 1 on a system-RAM-constrained job even when the GPU had
    headroom for several — see build_era5_train_loader.

    Episodes are drawn in *groups* of `group_size` sharing one P/N each (see
    __iter__): every call rolls its own grid_size/n_context once, shared by
    the whole group, and draws region/day/box_deg independently per episode
    within it via GlobalERA5Corpus.sample_episode_fixed_shape — the same
    P/N-homogeneous-group, everything-else-free-to-vary trade LiveGPDataset's
    group_size already makes for synthetic data (see its docstring), applied
    here to amortize TabICL's per-call PIT overhead across the group via
    run_pit_batched instead of one run_pit call per episode (benchmarked
    ~40x fewer Python-level TabICL invocations per episode). d_x=2 (lon,
    lat) is fixed regardless of region/resolution, so — unlike the synthetic
    path, where group_size must also equal a multiple of d_features — P/N
    can still vary freely *across* groups/batches; era5_collate_fn pads
    exactly like dataset.collate_fn already does for that.
    """

    def __init__(
        self,
        shared_corpus: dict,
        tabicl_ckpt: str,
        tabicl_device: str,
        k_folds: int,
        grid_size_range: Tuple[int, int],
        box_deg_range: Tuple[float, float],
        n_context_frac_range: Tuple[float, float],
        base_seed: int,
        group_size: int = 1,
    ):
        self.shared_corpus = shared_corpus
        self.tabicl_ckpt = tabicl_ckpt
        self.tabicl_device = tabicl_device
        self.k_folds = k_folds
        self.grid_size_range = grid_size_range
        self.box_deg_range = box_deg_range
        self.n_context_frac_range = n_context_frac_range
        self.base_seed = base_seed
        self.group_size = group_size

    def _seed_for(self, worker_id: int, call_idx: int) -> int:
        raw = (self.base_seed + 1) * 1_000_003 + worker_id * 1_000_000_007 + call_idx
        return raw % (2**32)

    def __iter__(self):
        import numpy as np

        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        print(f"[era5_live_dataset] worker {worker_id}: attaching to shared global ERA5 corpus")
        corpus = GlobalERA5Corpus.from_shared(self.shared_corpus)
        print(f"[era5_live_dataset] worker {worker_id}: loading frozen TabICL marginal: {self.tabicl_ckpt}")
        tabicl_model = load_tabicl(self.tabicl_ckpt, self.tabicl_device)

        call_idx = 0
        while True:
            rng = np.random.default_rng(self._seed_for(worker_id, call_idx))
            call_idx += 1

            # Roll grid_size/n_context ONCE per group -- every episode in
            # the group shares P/N (region/day/box_deg still vary per draw),
            # required for the single run_pit_batched call below. See
            # sample_episode_fixed_shape's docstring.
            grid_size = int(rng.integers(self.grid_size_range[0], self.grid_size_range[1] + 1))
            D = grid_size * grid_size
            n_context_frac = float(rng.uniform(*self.n_context_frac_range))
            n_context = int(np.clip(round(n_context_frac * D), 1, D - 1))

            raw_eps: list = []
            attempts = 0
            max_attempts = self.group_size * 20
            while len(raw_eps) < self.group_size and attempts < max_attempts:
                attempts += 1
                ep = corpus.sample_episode_fixed_shape(rng, grid_size, self.box_deg_range, n_context)
                if ep is not None:
                    raw_eps.append(ep)
            if len(raw_eps) < self.group_size:
                # Pathological draw (e.g. grid_size too large for
                # box_deg_range to ever satisfy) -- redraw the whole group
                # with a fresh grid_size/n_context on the next call_idx.
                continue

            x_train = torch.as_tensor(
                np.stack([e["x_norm_train"] for e in raw_eps]), dtype=torch.float32, device=self.tabicl_device
            )
            x_test = torch.as_tensor(
                np.stack([e["x_norm_test"] for e in raw_eps]), dtype=torch.float32, device=self.tabicl_device
            )
            y_train = torch.as_tensor(
                np.stack([e["y_train"] for e in raw_eps]), dtype=torch.float32, device=self.tabicl_device
            )
            y_test = torch.as_tensor(
                np.stack([e["y_test"] for e in raw_eps]), dtype=torch.float32, device=self.tabicl_device
            )
            pit = _pit_group(x_train, y_train, x_test, y_test, tabicl_model, self.k_folds)
            if pit is None:
                continue
            for i in range(self.group_size):
                yield {
                    "x_norm_train": x_train[i].cpu(),
                    "x_norm_test": x_test[i].cpu(),
                    "y_train": y_train[i].cpu(),
                    "y_test": y_test[i].cpu(),
                    "z_train": pit["z_train"][i].cpu(),
                    "z_test": pit["z_test"][i].cpu(),
                    "log_pdf_test": pit["log_pdf_test"][i].cpu(),
                }


def _resolve_era5_cfg(cfg: DictConfig) -> dict:
    e = cfg.get("era5_live", {}) or {}
    val_corpus_dir = e.get("val_corpus_dir", None)
    return {
        "corpus_dir": str(e.get("corpus_dir", "./eval/data/cache/era5_global")),
        # Separate cache dir (e.g. a held-out year fetched into its own
        # directory via fetch_era5_global.py) for the fixed validation set
        # below. None (default) falls back to corpus_dir -- the pre-existing
        # behavior of validating on a different random seed's slice of the
        # SAME date range training draws from, not a temporally disjoint one.
        "val_corpus_dir": str(val_corpus_dir) if val_corpus_dir else None,
        "grid_size_range": (int(e.get("grid_size_min", 8)), int(e.get("grid_size_max", 28))),
        "box_deg_range": (float(e.get("box_deg_min", 5.0)), float(e.get("box_deg_max", 25.0))),
        "n_context_frac_range": (float(e.get("n_context_frac_min", 0.05)), float(e.get("n_context_frac_max", 0.4))),
        "val_episodes": int(e.get("val_episodes", 200)),
        "val_seed": int(e.get("val_seed", 20260823)),
    }


def build_era5_train_loader(cfg: DictConfig, t: DictConfig, device: str) -> DataLoader:
    """Training DataLoader backed by LiveERA5Dataset. Mirrors
    live_dataset.py::build_live_train_loader's tabicl-worker path exactly
    (spawn context, few GPU workers, per-worker thread pinning, GPU-headroom
    -bound worker count via resolve_live_tabicl_num_workers) — this source
    has no CPU-only path at all, so there's no analyticGP-style branch to
    pick between. Unlike that function's OWN docstring caveat for the
    synthetic-GP path (no on-disk corpus there), era5 DOES have one; loading
    it once here into shared memory (load_shared_corpus_arrays) BEFORE the
    DataLoader spawns workers is what makes it safe to size num_workers off
    GPU headroom alone instead of a separate system-RAM-aware cap — see
    LiveERA5Dataset's docstring.
    """
    if device != "cuda":
        raise ValueError(
            f"training.live_source=era5 requires device='cuda' (got {device!r}) — "
            "real-ERA5 episodes are PIT'd through a frozen TabICL model, and "
            "CPU-only TabICL inference was benchmarked and rejected as too slow "
            "for live generation (see live_dataset.py::build_live_train_loader)."
        )
    ckpt = resolve_pit_ckpt(cfg)
    if ckpt is None:
        raise ValueError(
            "training.live_source=era5 requires a resolvable TabICL checkpoint — "
            "set tabicl.ckpt (with tabicl.pretrained=true) or tabicl.pit_ckpt."
        )
    ecfg = _resolve_era5_cfg(cfg)
    k_folds = int(cfg.tabicl.get("pit_k_folds", 10))
    base_seed = int(getattr(cfg, "seed", None) or 0)
    # Reuse live_dataset.py's dedicated tabicl-GPU-worker group knob (default
    # 2, benchmarked there for the synthetic-GP path) rather than adding a
    # separate era5-specific one -- same TabICL-per-call-overhead problem,
    # same fix. See LiveERA5Dataset's docstring for what grouping buys here.
    group_multiplier = max(1, int(t.get("live_tabicl_group_multiplier", 2)))
    group_size = int(t.batch_size) * group_multiplier

    # Load once, HERE in the main process, before the DataLoader below spawns
    # any workers -- see LiveERA5Dataset's docstring for why this is what
    # lets every worker attach to one shared copy instead of holding its own.
    print(f"[era5_live_dataset] loading global ERA5 corpus into shared memory from {ecfg['corpus_dir']}")
    shared_corpus = load_shared_corpus_arrays(ecfg["corpus_dir"])

    live_ds = LiveERA5Dataset(
        shared_corpus=shared_corpus,
        tabicl_ckpt=ckpt,
        tabicl_device=device,
        k_folds=k_folds,
        grid_size_range=ecfg["grid_size_range"],
        box_deg_range=ecfg["box_deg_range"],
        n_context_frac_range=ecfg["n_context_frac_range"],
        base_seed=base_seed,
        group_size=group_size,
    )
    # GPU-headroom-bound only, same as the synthetic-GP tabicl path -- no
    # longer separately RAM-capped, since every worker shares one corpus
    # copy instead of holding its own (see LiveERA5Dataset's docstring).
    num_workers = resolve_live_tabicl_num_workers(t, device)
    loader = DataLoader(
        live_ds,
        batch_size=t.batch_size,
        collate_fn=era5_collate_fn,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_limit_worker_threads if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    return loader


def build_era5_fixed_val_batches(cfg: DictConfig, t: DictConfig, device: str = "cpu") -> List[dict]:
    """Fixed, once-generated real-ERA5 validation set — analogue of
    live_dataset.py::build_fixed_live_val_batches. Uses a val-specific seed
    (era5_live.val_seed) distinct from the training seed stream, computed
    once in the main process, so val/... tracks only model changes across
    training, same rationale as the GP path's fixed val batches.

    This is separate from — and complements — the era5_fit/<region>
    validation probes (src/train.py::_build_era5_val_batches, cfg.baselines.
    era5_*): those score a handful of fixed, curated named regions; this is
    the live-training-loop's own held-out slice of the worldwide random
    corpus distribution the model is being finetuned on.

    era5_live.val_corpus_dir (None by default) points this at a SEPARATE
    corpus directory instead of corpus_dir -- e.g. one calendar year fetched
    into its own cache via fetch_era5_global.py, disjoint from the years the
    training corpus covers -- so val/era5_live_* tracks genuine held-out-year
    generalization rather than a different random seed's slice of the same
    date range training already draws from. Falls back to corpus_dir when
    unset (the original same-range behavior).
    """
    import numpy as np

    ecfg = _resolve_era5_cfg(cfg)
    ckpt = resolve_pit_ckpt(cfg)
    if ckpt is None:
        raise ValueError(
            "training.live_source=era5 requires a resolvable TabICL checkpoint — "
            "set tabicl.ckpt (with tabicl.pretrained=true) or tabicl.pit_ckpt."
        )
    k_folds = int(cfg.tabicl.get("pit_k_folds", 10))
    print(f"[era5_live_dataset] Loading frozen TabICL marginal for fixed val batches: {ckpt}")
    tabicl_model = load_tabicl(ckpt, device)
    val_corpus_dir = ecfg["val_corpus_dir"] or ecfg["corpus_dir"]
    print(f"[era5_live_dataset] Building fixed val batches from corpus: {val_corpus_dir}")
    corpus = GlobalERA5Corpus(val_corpus_dir)

    batch_size = int(t.batch_size)
    n_val = ecfg["val_episodes"]
    n_batches = max(1, (n_val + batch_size - 1) // batch_size)
    rng = np.random.default_rng(ecfg["val_seed"])

    batches = []
    with torch.no_grad():
        for _ in range(n_batches):
            episodes = []
            attempts = 0
            while len(episodes) < batch_size and attempts < batch_size * 20:
                attempts += 1
                ep = corpus.sample_episode(
                    rng, ecfg["grid_size_range"], ecfg["box_deg_range"],
                    ecfg["n_context_frac_range"],
                )
                if ep is None:
                    continue
                x_train = torch.as_tensor(ep["x_norm_train"], dtype=torch.float32, device=device)
                x_test = torch.as_tensor(ep["x_norm_test"], dtype=torch.float32, device=device)
                y_train = torch.as_tensor(ep["y_train"], dtype=torch.float32, device=device)
                y_test = torch.as_tensor(ep["y_test"], dtype=torch.float32, device=device)
                pit = _pit_episode(x_train, y_train, x_test, y_test, tabicl_model, k_folds)
                if pit is None:
                    continue
                episodes.append({
                    "x_norm_train": x_train.cpu(), "x_norm_test": x_test.cpu(),
                    "y_train": y_train.cpu(), "y_test": y_test.cpu(),
                    "z_train": pit["z_train"].cpu(), "z_test": pit["z_test"].cpu(),
                    "log_pdf_test": pit["log_pdf_test"].cpu(),
                })
            if episodes:
                batches.append(era5_collate_fn(episodes))
    del tabicl_model
    return batches
