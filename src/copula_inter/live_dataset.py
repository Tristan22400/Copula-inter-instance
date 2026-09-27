"""Live GP episode generation for training without an on-disk dataset.

LiveGPDataset wraps data_gen.generate_gp_batch in an IterableDataset so
DataLoader workers generate episodes while the GPU trains. Enabled with
training.live_generation=true.
"""

from __future__ import annotations

import contextlib
import copy
import os
import warnings
from typing import Iterator, List, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from copula_inter.backend_registry import GENERIC_MARGINAL_BACKENDS, TABICL_Z_TRAIN_SOURCES, validate_z_train_source
from copula_inter.backend_registry import z_train_source as z_train_source_of
from copula_inter.data_gen import generate_gp_batch
from copula_inter.dataset import collate_fn
from copula_inter.gp_kernels import _COMPOSABLE_KERNELS
from copula_inter.pit import configure_tabicl_inference_amp, load_tabicl, resolve_pit_ckpt

# Thread count for generation in the main process (validation batches, gap
# diagnostic); small CPU ops are slow at high thread counts.
_MAIN_PROCESS_GEN_THREADS = 8


@contextlib.contextmanager
def limited_main_process_threads(n: int = _MAIN_PROCESS_GEN_THREADS) -> Iterator[None]:
    """Context manager capping torch's intra-op threads for generation in the main process."""
    prev = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


# z_train sources that need no marginal model ("y_train": the z-scored target, live generation only).
_RAW_Y_SOURCES = ("y_train",)


# Per-worker VRAM estimate for GPU generation workers:
# _LIVE_TABICL_WORKER_FIXED_OVERHEAD_GB + _LIVE_TABICL_WORKER_PER_EPISODE_GB *
# group_size. Calibrated at batch_size=16, group_size=48; larger group sizes
# are not validated.
_LIVE_TABICL_WORKER_FIXED_OVERHEAD_GB = 0.5
_LIVE_TABICL_WORKER_PER_EPISODE_GB = 0.02
# Per-worker CUDA context overhead plus margin.
_LIVE_TABICL_FLAT_HEADROOM_GB = 1.0
# Default VRAM reserved for the training process when auto-sizing workers
# (training.live_tabicl_main_process_reserve_gb overrides it).
_LIVE_TABICL_AUTO_MAIN_PROCESS_RESERVE_GB = 8.0
# Upper bound on auto-sized workers (throughput peak measured on an 8-core RTX A5000 node).
_LIVE_TABICL_AUTO_WORKERS_MIN = 1
_LIVE_TABICL_AUTO_WORKERS_MAX = 12


def resolve_live_tabicl_num_workers(t: DictConfig, device: str) -> int:
    """Number of GPU generation workers: training.live_tabicl_num_workers, or auto-sized.

    Auto-sizing divides the currently free GPU memory, minus
    training.live_tabicl_main_process_reserve_gb and a flat headroom, by the
    per-worker estimate for this group size, clamped to
    [_LIVE_TABICL_AUTO_WORKERS_MIN, _LIVE_TABICL_AUTO_WORKERS_MAX].
    """
    configured = t.get("live_tabicl_num_workers", None)
    if configured is not None:
        return int(configured)
    if device != "cuda" or not torch.cuda.is_available():
        return _LIVE_TABICL_AUTO_WORKERS_MIN
    main_process_reserve_gb = float(
        t.get("live_tabicl_main_process_reserve_gb", _LIVE_TABICL_AUTO_MAIN_PROCESS_RESERVE_GB)
    )
    group_multiplier = max(1, int(t.get("live_tabicl_group_multiplier", 2)))
    group_size = int(t.batch_size) * group_multiplier
    per_worker_gb = _LIVE_TABICL_WORKER_FIXED_OVERHEAD_GB + _LIVE_TABICL_WORKER_PER_EPISODE_GB * group_size
    free_b, _total_b = torch.cuda.mem_get_info()
    free_gb = free_b / 1e9
    available_gb = free_gb - main_process_reserve_gb - _LIVE_TABICL_FLAT_HEADROOM_GB
    n = int(available_gb // per_worker_gb) if available_gb > 0 else 0
    n = max(_LIVE_TABICL_AUTO_WORKERS_MIN, min(n, _LIVE_TABICL_AUTO_WORKERS_MAX))
    print(
        f"[live_dataset] training.live_tabicl_num_workers not set -- auto-detected "
        f"{n} worker(s) from {free_gb:.1f}GB currently free on this GPU (reserving "
        f"{main_process_reserve_gb:.1f}GB for the training process itself, "
        f"~{per_worker_gb:.2f}GB/worker at batch_size={int(t.batch_size)} x "
        f"group_multiplier={group_multiplier} = {group_size} episodes/call). Set "
        "training.live_tabicl_num_workers explicitly to override, e.g. if this node "
        "runs several such jobs concurrently on the same GPU."
    )
    return n


class LiveGPDataset(IterableDataset):
    """Infinite stream of GP episodes from generate_gp_batch(cfg, group_size, ...).

    group_size must be a multiple of the loader's batch_size (enforced by
    build_live_train_loader) so every batch comes from one call and shares d.
    Each (worker, call) pair gets its own seed (worker_seed). kernel_weights and
    tabicl_mix_weights are shared-memory tensors that the main process may update
    after the workers start. With device set, each worker lazily loads its own
    TabICL (or marginal backend) and generates on that device.
    """

    def __init__(
        self,
        cfg: DictConfig,
        group_size: int = 1,
        kernel_weights: Optional[torch.Tensor] = None,
        tabicl_device: Optional[str] = None,
        tabicl_mix_weights: Optional[torch.Tensor] = None,
        marginal_backend: Optional[str] = None,
        marginal_device: Optional[str] = None,
        marginal_probs_n: int = 99,
    ) -> None:
        # Deep copy so per-call seed changes don't touch the caller's cfg.
        self._cfg = copy.deepcopy(cfg)
        self._base_seed = int(getattr(cfg, "seed", None) or 0)
        self.group_size = group_size
        # Shared-memory tensor, not copied.
        self.kernel_weights = kernel_weights
        # None: generate on CPU. "cuda": each worker loads its own TabICL lazily.
        self.tabicl_device = tabicl_device
        # Shared-memory tensor, not copied (written once before workers start).
        self.tabicl_mix_weights = tabicl_mix_weights
        # Non-TabICL backend: each worker builds its own regressor lazily.
        self.marginal_backend = marginal_backend
        self.marginal_device = marginal_device
        self.marginal_probs_n = marginal_probs_n

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        cfg = copy.deepcopy(self._cfg)
        configure_tabicl_inference_amp(bool(cfg.training.get("tabicl_inference_amp", False)))
        call_idx = 0
        # Silence per-call degenerate-episode warnings.
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Load the marginal once per worker, before the generation loop.
        z_train_source = z_train_source_of(cfg)
        validate_z_train_source(z_train_source)
        raw_y_override = z_train_source in _RAW_Y_SOURCES
        tabicl_model = None
        gen_device = "cpu"
        tabicl_k_folds = int(cfg.data.get("z_train_tabicl_k_folds", 10))
        tabicl_split_calib_frac = (
            float(cfg.data.get("z_train_split_calib_frac", 1.0)) if z_train_source == "tabicl_split" else 0.0
        )
        mix_enabled = self.tabicl_mix_weights is not None
        if self.tabicl_device is not None:
            ckpt = resolve_pit_ckpt(cfg)
            if ckpt is None:
                reason = (
                    "data.z_train_tabicl_mix_enabled=true" if mix_enabled else f"data.z_train_source={z_train_source}"
                )
                raise ValueError(
                    f"training.live_generation with {reason} requires a resolvable "
                    "TabICL checkpoint -- set tabicl.ckpt (with tabicl.pretrained=true) "
                    "or tabicl.pit_ckpt."
                )
            reason = (
                f"data.z_train_tabicl_mix_enabled=true (z_train_source={z_train_source})"
                if mix_enabled
                else f"data.z_train_source={z_train_source}"
            )
            print(
                f"[live_dataset] worker {worker_id}: loading frozen TabICL marginal "
                f"for {reason} on {self.tabicl_device}: {ckpt}"
            )
            tabicl_model = load_tabicl(ckpt, self.tabicl_device)
            # Generation runs on the same device as the marginal.
            gen_device = self.tabicl_device

        marginal_regressor = None
        if self.marginal_backend is not None:
            from eval.spatial.marginal_backends import make_regressor

            print(
                f"[live_dataset] worker {worker_id}: loading {self.marginal_backend} marginal "
                f"for data.z_train_source={self.marginal_backend} on {self.marginal_device}"
            )
            marginal_regressor = make_regressor(self.marginal_backend, device=self.marginal_device)

        while True:
            cfg.seed = worker_seed(self._base_seed, worker_id, call_idx)
            call_idx += 1
            episodes = generate_gp_batch(
                cfg,
                self.group_size,
                device=gen_device,
                kernel_weights=self.kernel_weights,
                tabicl_model=tabicl_model,
                tabicl_k_folds=tabicl_k_folds,
                tabicl_split_calib_frac=tabicl_split_calib_frac,
                tabicl_mix_weights=self.tabicl_mix_weights,
                marginal_backend=self.marginal_backend,
                marginal_regressor=marginal_regressor,
                marginal_probs_n=self.marginal_probs_n,
                raw_y_override=raw_y_override,
            )
            for ep in episodes:
                yield ep


def worker_seed(base_seed: int, worker_id: int, call_idx: int) -> int:
    """Per-(worker, call) seed in [0, 2**32)."""
    raw = (base_seed + 1) * 1_000_003 + worker_id * 1_000_000_007 + call_idx
    return raw % (2**32)


def _limit_worker_threads(_worker_id: int) -> None:
    """DataLoader worker_init_fn: one intra-op thread per worker."""
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def build_live_train_loader(
    cfg: DictConfig, t: DictConfig, device: str
) -> Tuple[DataLoader, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Training DataLoader over LiveGPDataset.

    Analytic z_train: CPU workers (live_num_workers, capped by available CPUs).
    TabICL or batched-backend z_train: few CUDA workers (see
    resolve_live_tabicl_num_workers), each with its own model copy, using the
    "spawn" start method; device must be "cuda".

    Returns:
        (loader, kernel_weights, tabicl_mix_weights). kernel_weights is a
        shared-memory _COMPOSABLE_KERNELS-ordered tensor when
        training.adaptive_kernel_sampling is on, else None. tabicl_mix_weights is
        one initialized to the floor fraction when data.z_train_tabicl_mix_enabled,
        else None.
    """
    batch_size = int(t.batch_size)

    z_train_source = z_train_source_of(cfg)
    validate_z_train_source(z_train_source)
    mix_enabled = bool(cfg.data.get("z_train_tabicl_mix_enabled", False))
    tabicl_live_enabled = mix_enabled or z_train_source in TABICL_Z_TRAIN_SOURCES
    # Batched non-TabICL backends use the same GPU-worker setup as TabICL.
    generic_marginal_enabled = z_train_source in GENERIC_MARGINAL_BACKENDS
    batched_marginal_worker_enabled = tabicl_live_enabled or generic_marginal_enabled

    # Group-size multiplier for GPU workers (live_group_multiplier is for CPU workers).
    if batched_marginal_worker_enabled:
        group_multiplier = max(1, int(t.get("live_tabicl_group_multiplier", 2)))
    else:
        group_multiplier = max(1, int(t.get("live_group_multiplier", 1)))
    group_size = batch_size * group_multiplier

    if batched_marginal_worker_enabled and device != "cuda":
        reason = "data.z_train_tabicl_mix_enabled=true" if mix_enabled else f"data.z_train_source={z_train_source}"
        raise ValueError(
            f"training.live_generation with {reason} requires device='cuda' "
            f"(got {device!r}) -- CPU-only inference for this backend was "
            "benchmarked and rejected as too slow to keep the GPU fed (see "
            "this function's docstring, and eval/spatial/exaone_batched.py's "
            "for the exaone-specific ~120x CPU-vs-CUDA gap). Use "
            "data.z_train_source=analytic and "
            "data.z_train_tabicl_mix_enabled=false for CPU-only runs."
        )
    tabicl_device = device if tabicl_live_enabled else None
    marginal_device = device if generic_marginal_enabled else None

    if batched_marginal_worker_enabled:
        # GPU workers: count from resolve_live_tabicl_num_workers (TabICL-calibrated
        # for every backend).
        num_workers = resolve_live_tabicl_num_workers(t, device)
    else:
        num_workers = int(t.get("live_num_workers", 8))
        # Cap CPU workers at the CPUs available to this process.
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cpus = os.cpu_count() or num_workers
        if num_workers > available_cpus:
            print(
                f"[live_dataset] live_num_workers={num_workers} exceeds this "
                f"process's {available_cpus} available CPUs; clamping to "
                f"{available_cpus} to avoid oversubscription/OOM."
            )
            num_workers = available_cpus

    kernel_weights = None
    if bool(t.get("adaptive_kernel_sampling", False)):
        n = len(_COMPOSABLE_KERNELS)
        kernel_weights = torch.full((n,), 1.0 / n, dtype=torch.float32).share_memory_()
    tabicl_mix_weights = None
    if mix_enabled:
        n = len(_COMPOSABLE_KERNELS)
        # Floor fraction until train.py writes the measured values.
        floor_frac = float(cfg.data.get("z_train_tabicl_mix_floor_frac", 0.05))
        tabicl_mix_weights = torch.full((n,), floor_frac, dtype=torch.float32).share_memory_()
    marginal_probs_n = int(cfg.data.get("z_train_marginal_probs_n", 99))
    live_ds = LiveGPDataset(
        cfg,
        group_size=group_size,
        kernel_weights=kernel_weights,
        tabicl_device=tabicl_device,
        tabicl_mix_weights=tabicl_mix_weights,
        marginal_backend=z_train_source if generic_marginal_enabled else None,
        marginal_device=marginal_device,
        marginal_probs_n=marginal_probs_n,
    )
    # CUDA-resident worker models require spawn.
    loader = DataLoader(
        live_ds,
        batch_size=t.batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_limit_worker_threads if num_workers > 0 else None,
        multiprocessing_context="spawn" if (batched_marginal_worker_enabled and num_workers > 0) else None,
    )
    return loader, kernel_weights, tabicl_mix_weights


def build_fixed_live_val_batches(
    cfg: DictConfig,
    t: DictConfig,
    device: str = "cpu",
) -> Tuple[List[dict], List[List[dict]]]:
    """Fixed validation set for live-generation training.

    n_batches generate_gp_batch calls with fixed seeds (each batch is one kernel
    family and shape), collated on CPU. For TabICL / backend z_train sources a
    marginal is loaded once in the main process. Episodes keep kernel metadata
    (with _L_ff/_alpha moved to CPU) for posterior scoring in validate().

    Returns:
        (batches, episodes_by_batch): the collated batches, and for each the list
        of raw episode dicts.
    """
    n_val = int(t.get("val_episodes", 500))
    val_seed = int(t.get("live_val_seed", 20260723))
    batch_size = int(t.batch_size)
    n_batches = max(1, (n_val + batch_size - 1) // batch_size)

    z_train_source = z_train_source_of(cfg)
    validate_z_train_source(z_train_source)
    tabicl_live_enabled = z_train_source in TABICL_Z_TRAIN_SOURCES
    generic_marginal_enabled = z_train_source in GENERIC_MARGINAL_BACKENDS
    raw_y_override = z_train_source in _RAW_Y_SOURCES
    if (tabicl_live_enabled or generic_marginal_enabled) and device != "cuda":
        raise ValueError(
            f"training.live_generation with data.z_train_source={z_train_source} "
            f"requires device='cuda' (got {device!r}) -- see build_live_train_loader's "
            "batched_marginal_worker_enabled docstring for why CPU is rejected "
            "rather than silently slow for exaone/tabpfn too."
        )
    tabicl_model = None
    gen_device = "cpu"
    tabicl_k_folds = int(cfg.data.get("z_train_tabicl_k_folds", 10))
    tabicl_split_calib_frac = (
        float(cfg.data.get("z_train_split_calib_frac", 1.0)) if z_train_source == "tabicl_split" else 0.0
    )
    if tabicl_live_enabled:
        ckpt = resolve_pit_ckpt(cfg)
        if ckpt is None:
            raise ValueError(
                f"training.live_generation with data.z_train_source={z_train_source} "
                "requires a resolvable TabICL checkpoint -- set tabicl.ckpt (with "
                "tabicl.pretrained=true) or tabicl.pit_ckpt."
            )
        print(f"[live_dataset] Loading frozen TabICL marginal for fixed val batches: {ckpt}")
        tabicl_model = load_tabicl(ckpt, device)
        gen_device = device

    # Build the backend regressor once in the main process; GP synthesis stays on CPU.
    marginal_regressor = None
    marginal_probs_n = int(cfg.data.get("z_train_marginal_probs_n", 99))
    if generic_marginal_enabled:
        from eval.spatial.marginal_backends import make_regressor

        print(f"[live_dataset] Loading {z_train_source} marginal for fixed val batches on {device}")
        marginal_regressor = make_regressor(z_train_source, device=device)

    batches = []
    episodes_by_batch: List[List[dict]] = []
    with warnings.catch_warnings(), limited_main_process_threads():
        # Silence degenerate-episode warnings for this call.
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for i in range(n_batches):
            val_cfg = copy.deepcopy(cfg)
            val_cfg.seed = val_seed + i * 104_729  # distinct, fixed, reproducible per batch
            episodes = generate_gp_batch(
                val_cfg,
                batch_size,
                device=gen_device,
                tabicl_model=tabicl_model,
                tabicl_k_folds=tabicl_k_folds,
                tabicl_split_calib_frac=tabicl_split_calib_frac,
                return_kernel_metadata=True,
                marginal_backend=z_train_source if generic_marginal_enabled else None,
                marginal_regressor=marginal_regressor,
                marginal_probs_n=marginal_probs_n,
                raw_y_override=raw_y_override,
            )
            if gen_device == "cuda":
                for ep in episodes:
                    ep["_L_ff"] = ep["_L_ff"].cpu()
                    ep["_alpha"] = ep["_alpha"].cpu()
            batches.append(collate_fn(episodes))
            episodes_by_batch.append(episodes)

    if tabicl_model is not None or marginal_regressor is not None:
        del tabicl_model, marginal_regressor
        if device == "cuda":
            import gc

            gc.collect()
            torch.cuda.empty_cache()
    return batches, episodes_by_batch
