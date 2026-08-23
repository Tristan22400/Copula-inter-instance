"""
live_dataset.py — On-the-fly GP episode generation for training without a
pre-materialised on-disk dataset (see generate_pit_dataset.py / dataset.py
for the disk-backed path this substitutes for).

Wraps data_gen.generate_gp_batch — already used at train.py's synthetic-kernel
validation probes (_build_synthetic_kernel_batches), and identical to what
generate_pit_dataset.py writes to shard_*.pt — in a torch IterableDataset, so
DataLoader workers generate episodes in background processes while the GPU
trains on the previous batch. That's the same overlap num_workers/
prefetch_factor already gives the disk path for reading shards; here it hides
generation latency instead of I/O latency.

Temporary, easily-removable substitute for the disk pipeline (e.g. during a
storage-constrained period): enabled via training.live_generation=true in
train.py. Nothing in this module touches disk.
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

from data_gen import _COMPOSABLE_KERNELS, generate_gp_batch
from dataset import collate_fn
from pit import load_tabicl, resolve_pit_ckpt

# Thread count for generate_gp_batch calls made directly in the MAIN process
# (build_fixed_live_val_batches below, train.py's z_train-gap diagnostic) --
# these never go through a DataLoader's worker_init_fn, so unlike the
# persistent training workers (_limit_worker_threads below, torch.set_
# num_threads(1) each) they'd otherwise run at the OS-default thread count
# (e.g. 64 on a many-core node). Benchmarked: generate_gp_batch's structural-
# warp and final NaN/Inf-sweep code makes many small/medium CPU tensor ops
# whose thread-pool launch overhead dominates at high thread counts -- a
# single 12-tensor isfinite sweep measured 200ms at 64 threads vs 1.2ms at 16
# (2026-08-23). Unlike the many-worker case, there's only one caller here, so
# (unlike _limit_worker_threads's 1) a modest thread count is used instead of
# fully serializing -- 8 stays comfortably inside the fast, flat region the
# same benchmark found across 1-16 threads without needing per-node tuning.
_MAIN_PROCESS_GEN_THREADS = 8


@contextlib.contextmanager
def limited_main_process_threads(n: int = _MAIN_PROCESS_GEN_THREADS):
    """Temporarily cap torch's intra-op thread count, restored on exit.

    Wrap any generate_gp_batch/_generate_gp_batch_raw call made directly in
    the main training process (i.e. NOT inside a DataLoader worker, which
    already gets _limit_worker_threads via worker_init_fn) with this --
    see _MAIN_PROCESS_GEN_THREADS' docstring above for why.
    """
    prev = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


# ---------------------------------------------------------------------------
# Shared-GPU-aware sizing for live-generation's TabICL DataLoader workers
# ---------------------------------------------------------------------------
# Each live_tabicl_num_workers worker holds its own frozen-TabICL CUDA
# context, entirely separate from the training process's own allocator pool
# (see train.py::_reserve_gpu_headroom_for_live_tabicl, which imports these
# same constants so the two stay in sync). ~120MB resident / ~1.4GB peak per
# worker was measured at batch_size=32 on the cheaper tabicl_split path;
# z_train_source=tabicl's K-fold rotation does several forward passes per
# call instead of one, so budget above that measured peak rather than at it.
_LIVE_TABICL_WORKER_HEADROOM_GB = 2.5
# Flat allowance on top of the per-worker figure above, for each worker's own
# CUDA context overhead (not activation memory, so it doesn't scale with
# batch/fold count) plus a general safety margin.
_LIVE_TABICL_FLAT_HEADROOM_GB = 1.0
# Minimum VRAM reserved for the training process's OWN use (model, optimizer
# state, activations) when auto-sizing live_tabicl_num_workers below -- a
# floor, not a measurement of this run's actual footprint (unknowable before
# the model is even built), chosen conservatively so auto-sizing doesn't
# starve training on a small card.
_LIVE_TABICL_AUTO_MAIN_PROCESS_RESERVE_GB = 8.0
# Empirically validated range for the worker-count auto-sizing below
# (benchmarked on an RTX 6000 Ada, 2026-08-23): throughput plateaus by ~6-8
# concurrent GPU workers (they share one physical GPU and TabICL's forward
# pass is real compute, not just launch overhead that hides via overlap, so
# more workers past this range cost VRAM/CPU-thread overhead for no measured
# gain) and a single worker still gets the overlap-with-training benefit
# num_workers=0 would forgo entirely.
_LIVE_TABICL_AUTO_WORKERS_MIN = 1
_LIVE_TABICL_AUTO_WORKERS_MAX = 8


def resolve_live_tabicl_num_workers(t: DictConfig, device: str) -> int:
    """Resolve training.live_tabicl_num_workers, auto-detecting from
    CURRENTLY FREE (not total) GPU memory when left unset (null in config).

    Using free rather than total device memory is the important part: it's
    what makes this safe on a node where multiple jobs (this one plus
    others, possibly with no coordination between them) share one physical
    GPU. A worker count tuned/hardcoded against one card's FULL VRAM would
    either starve this run or risk OOM-ing everyone once another job is
    already holding memory on the same device -- reading free memory at
    call time naturally shrinks the auto-picked count when that's the case
    (and grows it back on an idle card), without needing to know anything
    about what else is running. It only reflects occupancy AT STARTUP,
    though -- a second job launched on the same GPU after this one has
    already sized itself is a real, unmitigated risk either way; explicit
    training.live_tabicl_num_workers (set deliberately low) is the correct
    tool for a node you know will run several such jobs concurrently.

    Also naturally adapts across GPU models with different VRAM (e.g. a
    32GB RTX 5000 Ada vs a 48GB RTX 6000 Ada) without per-node tuning, since
    it's driven by how much memory is actually free, not a name/model
    lookup.

    Explicit training.live_tabicl_num_workers in config always wins over
    auto-detection (e.g. to force a known-safe number by hand on a node
    running several such jobs by agreement).
    """
    configured = t.get("live_tabicl_num_workers", None)
    if configured is not None:
        return int(configured)
    if device != "cuda" or not torch.cuda.is_available():
        return _LIVE_TABICL_AUTO_WORKERS_MIN
    free_b, _total_b = torch.cuda.mem_get_info()
    free_gb = free_b / 1e9
    available_gb = free_gb - _LIVE_TABICL_AUTO_MAIN_PROCESS_RESERVE_GB - _LIVE_TABICL_FLAT_HEADROOM_GB
    n = int(available_gb // _LIVE_TABICL_WORKER_HEADROOM_GB) if available_gb > 0 else 0
    n = max(_LIVE_TABICL_AUTO_WORKERS_MIN, min(n, _LIVE_TABICL_AUTO_WORKERS_MAX))
    print(
        f"[live_dataset] training.live_tabicl_num_workers not set -- auto-detected "
        f"{n} worker(s) from {free_gb:.1f}GB currently free on this GPU (reserving "
        f"{_LIVE_TABICL_AUTO_MAIN_PROCESS_RESERVE_GB:.0f}GB for the training process "
        f"itself, ~{_LIVE_TABICL_WORKER_HEADROOM_GB:.1f}GB/worker). Set training."
        "live_tabicl_num_workers explicitly to override, e.g. if this node runs "
        "several such jobs concurrently on the same GPU."
    )
    return n


class LiveGPDataset(IterableDataset):
    """Infinite stream of GP episodes generated on the fly via generate_gp_batch.

    Each step of __iter__ advances a per-worker call counter and calls
    generate_gp_batch(cfg, group_size, device="cpu") — the exact generation
    code path the disk pipeline and the in-training kernel probes already use.

    IMPORTANT — group_size must be a multiple of the DataLoader's batch_size
    (build_live_train_loader enforces this; do not construct this class
    directly with an arbitrary group_size). generate_gp_batch samples
    kernel/P/N/active_dims *and, for variable-d configs, d_features itself*
    once per call, shared by every episode in that call — the same homogeneity
    a disk shard has. collate_fn cannot pad across a mismatched feature axis
    (unlike P/N), so if a batch straddled two groups with different sampled d,
    collation would fail exactly the way it does for the disk path's
    variable-d datasets without ShardHomogeneousBatchSampler. Keeping
    group_size a multiple of batch_size guarantees every batch comes from
    exactly one call, so this can't happen — the live-generation equivalent of
    that sampler, enforced structurally instead of by a separate sampler class.

    generate_gp_batch also has ~1s of fixed per-call overhead (kernel/
    hyperparameter sampling), so group_size == batch_size (benchmarked ~0.03
    s/episode at group=32) is both the correctness floor and close to the
    throughput ceiling — group_size=1 measured ~1.2s/episode, too slow to keep
    the GPU fed. Cross-batch diversity comes from multiple DataLoader workers
    each running independently-seeded streams — the live-generation analogue
    of ShardBlockSampler mixing across shards (traded off against the
    single-task-per-batch price the disk path already accepts for variable-d
    datasets — see ShardHomogeneousBatchSampler's docstring in dataset.py).

    Seeding: _generate_gp_batch_raw reseeds torch/numpy/random globally from
    cfg.seed on every call (data_gen.py), so two calls sharing a cfg.seed are
    byte-identical. Every (worker_id, call_idx) pair gets its own seed via a
    hash-like combination of base_seed/worker_id/call_idx, so distinct workers
    (separate processes — safe to mutate global RNG state independently) and
    distinct calls within one worker never repeat the same episodes.

    kernel_weights (optional): a `data_gen._COMPOSABLE_KERNELS`-ordered
    torch.Tensor created with .share_memory_() by build_live_train_loader
    (see training.adaptive_kernel_sampling). Passed straight through to every
    generate_gp_batch call — since it's shared memory allocated before the
    DataLoader forks its workers, the main training loop can mutate it in
    place (train.py, after each validate() call) and every worker picks up
    the update on its very next draw, with no explicit IPC. None (default)
    means unweighted/uniform sampling, unchanged from before this feature.
    """

    def __init__(
        self,
        cfg: DictConfig,
        group_size: int = 1,
        kernel_weights: Optional[torch.Tensor] = None,
        tabicl_device: Optional[str] = None,
        tabicl_mix_weights: Optional[torch.Tensor] = None,
    ):
        # Deep-copy so mutating .seed per call never touches the caller's cfg
        # (and so pickling this Dataset to worker processes doesn't drag along
        # anything unexpected the caller's cfg object might reference later).
        self._cfg = copy.deepcopy(cfg)
        self._base_seed = int(getattr(cfg, "seed", None) or 0)
        self.group_size = group_size
        # NOT deep-copied: must stay the same shared-memory tensor the main
        # process updates post-fork (see class docstring above).
        self.kernel_weights = kernel_weights
        # None (default): every episode generated on CPU, exactly as before
        # this knob existed -- data.z_train_source=analytic's only supported
        # live-generation path. A device string ("cuda"): data.z_train_source
        # is tabicl/tabicl_split, and build_live_train_loader has already
        # verified `device=="cuda"` and switched the DataLoader to
        # multiprocessing_context="spawn" (see its docstring for why fork
        # can't be used here). Each worker process loads its OWN frozen
        # TabICL copy onto this device, lazily, once, the first time __iter__
        # runs in that process -- a loaded CUDA model can't be handed to a
        # worker via fork's copy-on-write (no such mechanism exists across
        # process boundaries for CUDA memory) or safely pickled through
        # spawn's IPC without real complexity, so each worker pays its own
        # one-time checkpoint-load cost instead (~5s, benchmarked) rather
        # than sharing one instance.
        self.tabicl_device = tabicl_device
        # NOT deep-copied, same reasoning as kernel_weights above: a
        # `_COMPOSABLE_KERNELS`-ordered, .share_memory_()'d per-family
        # mixing-fraction tensor (see data.z_train_tabicl_mix_* in
        # conf/data/gp_tasks.yaml and train.py::_tabicl_gap_to_mix_frac),
        # forwarded straight through to every generate_gp_batch call below.
        # Unlike kernel_weights, this is set ONCE before the training loop
        # starts and never mutated again during training (the TabICL-vs-
        # analytic z_train gap it's derived from doesn't move with the
        # model) — but it still needs to be shared memory rather than
        # deep-copied, since it's written by the main process AFTER this
        # Dataset is constructed but BEFORE the DataLoader is first
        # iterated (workers fork/spawn lazily on first __iter__), same
        # ordering constraint kernel_weights's docstring above describes.
        # None (default) reproduces pre-feature behavior exactly (see
        # _tabicl_mix_prob_for_kernel's docstring in data_gen.py).
        self.tabicl_mix_weights = tabicl_mix_weights

    def _seed_for(self, worker_id: int, call_idx: int) -> int:
        # Not a cryptographic mix — just enough spread that (worker_id, call_idx)
        # collisions are astronomically unlikely over a training run. A
        # collision would only cost a moment of duplicated episodes, not
        # correctness, so this doesn't need to be bulletproof.
        # _seed_everything (data_gen.py) forwards this to np.random.seed, which
        # requires 0 <= seed < 2**32 — mod into that range, not a wider one.
        raw = (self._base_seed + 1) * 1_000_003 + worker_id * 1_000_000_007 + call_idx
        return raw % (2**32)

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        cfg = copy.deepcopy(self._cfg)
        call_idx = 0
        # data_gen.py warns (RuntimeWarning) on every degenerate-episode
        # discard — a routine, expected event at this call rate (every worker,
        # every call, for the whole training run), unlike the disk pipeline's
        # one-shot generate_pit_dataset.py run where the same warnings are
        # informative. Episodes are still discarded regardless; only the
        # console spam is silenced here.
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # z_train_source override setup — done ONCE per worker process,
        # before the infinite generation loop, not per-call: loading TabICL
        # (~5s, benchmarked) on every generate_gp_batch call would dominate
        # runtime. Mirrors generate_pit_dataset.py's "load once up front,
        # thread through every call" pattern, just scoped to a worker process
        # instead of the whole run.
        z_train_source = str(cfg.data.get("z_train_source", "analytic"))
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
                    "data.z_train_tabicl_mix_enabled=true" if mix_enabled
                    else f"data.z_train_source={z_train_source}"
                )
                raise ValueError(
                    f"training.live_generation with {reason} requires a resolvable "
                    "TabICL checkpoint -- set tabicl.ckpt (with tabicl.pretrained=true) "
                    "or tabicl.pit_ckpt."
                )
            reason = (
                f"data.z_train_tabicl_mix_enabled=true (z_train_source={z_train_source})" if mix_enabled
                else f"data.z_train_source={z_train_source}"
            )
            print(
                f"[live_dataset] worker {worker_id}: loading frozen TabICL marginal "
                f"for {reason} on {self.tabicl_device}: {ckpt}"
            )
            tabicl_model = load_tabicl(ckpt, self.tabicl_device)
            # Generation itself (kernel/Cholesky/feature warps, not just the
            # PIT override) also moves to this device: _generate_gp_batch_raw
            # takes one device for the whole call, and x_norm_train/
            # y_train_scaled must already be on tabicl_model's device before
            # the override call anyway. Measured faster than CPU generation
            # too (5.7ms/episode GPU vs 117ms/episode CPU at group_size=32),
            # so this is a net win, not just a requirement.
            gen_device = self.tabicl_device

        while True:
            cfg.seed = self._seed_for(worker_id, call_idx)
            call_idx += 1
            episodes = generate_gp_batch(
                cfg, self.group_size, device=gen_device, kernel_weights=self.kernel_weights,
                tabicl_model=tabicl_model, tabicl_k_folds=tabicl_k_folds,
                tabicl_split_calib_frac=tabicl_split_calib_frac,
                tabicl_mix_weights=self.tabicl_mix_weights,
            )
            for ep in episodes:
                yield ep


def _limit_worker_threads(_worker_id: int) -> None:
    """DataLoader worker_init_fn: pin each live-generation worker to a single
    CPU thread.

    generate_gp_batch runs GP kernel construction + Cholesky (LOO PIT) on CPU
    inside every worker (device="cpu"). Left unset, each worker process
    defaults to torch/BLAS intra-op parallelism sized to the *whole* machine's
    core count, so live_num_workers processes each fan out over every core --
    e.g. 8 workers x 20 threads on a 20-core node, ~8x oversubscription. That
    thrash is consistent with the highly variable step "data=" wait times
    observed in practice (a few ms up to several hundred ms on the same run):
    contention severity depends on which other workers happen to be doing CPU
    work at that instant. One thread per worker lets live_num_workers workers
    run truly in parallel across the available cores instead of fighting each
    other for them.
    """
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def build_live_train_loader(
    cfg: DictConfig, t: DictConfig, device: str
) -> Tuple[DataLoader, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Training DataLoader backed by LiveGPDataset instead of an on-disk
    CopulaDataset. Mirrors the disk path's DataLoader kwargs (train.py) so
    downstream code — batch dict shape, the non_blocking .to(device) call, the
    train_iter/StopIteration re-creation loop — is unaffected. LiveGPDataset
    never raises StopIteration, so that re-creation branch simply never fires.

    data.z_train_source (see conf/data/gp_tasks.yaml) analytic (default):
    every worker generates on CPU, exactly as before this knob existed.
    tabicl/tabicl_split: each worker instead generates on `device` (must be
    "cuda" -- there is no viable CPU path, see below) with its own
    lazily-loaded frozen TabICL copy (LiveGPDataset.__iter__), and the
    DataLoader is forced onto multiprocessing_context="spawn" instead of the
    platform default (fork on Linux).

    Why spawn is required for GPU workers: fork()'d worker processes inherit
    the parent's already-initialized CUDA context, which CUDA does not
    support re-using from a child process ("Cannot re-initialize CUDA in
    forked subprocess"). spawn re-executes this module fresh in each worker
    instead of copy-on-write duplicating the parent, so each worker
    initializes its own CUDA context cleanly -- standard PyTorch guidance for
    CUDA-using DataLoader workers. Requires train.py's entrypoint to sit
    behind `if __name__ == "__main__":` (verified -- see bottom of train.py);
    without that guard spawn would re-execute the whole training script in
    every worker.

    Why CPU-only TabICL workers were rejected (benchmarked, not assumed):
    even the cheaper tabicl_split path measured 766ms/episode on CPU with 1
    thread (matching _limit_worker_threads's constraint) vs 9.75ms/episode on
    GPU -- ~79x slower, easily enough to stall the GPU waiting on data. K-fold
    was 4381ms/episode, worse still. Separately, TabICL's own InferenceManager
    (tabicl_upstream/src/tabicl/_model/inference.py) auto-selects
    exe_device="cuda" at MODEL CONSTRUCTION time whenever CUDA is visible in
    the process, independent of the model's own .to(device) placement -- so a
    naive "just call load_tabicl(ckpt, 'cpu')" in a process that can still see
    a GPU silently mismatches internal buffers against CPU parameters and
    crashes; genuine CPU-only execution would additionally require hiding
    CUDA from the worker (os.environ["CUDA_VISIBLE_DEVICES"] = "") before
    ever constructing the model. Moot given the throughput gap above, so not
    implemented.

    VRAM: each GPU worker holds its own TabICL copy (~120MB resident,
    measured peak ~1.4GB during a tabicl_split call at batch_size=32) on the
    SAME physical GPU the training job itself uses -- unlike CPU workers,
    which only needed many of them to make up for slow single-threaded
    inference, GPU workers are individually fast enough that few are needed
    (see live_tabicl_num_workers below), so this stays a modest addition to
    the training job's own VRAM footprint rather than an unbounded one.

    Returns (loader, kernel_weights, tabicl_mix_weights): kernel_weights is a
    `_COMPOSABLE_KERNELS`-ordered, .share_memory_()'d tensor of per-family
    sampling weights when t.adaptive_kernel_sampling is true, else None.
    tabicl_mix_weights is a `_COMPOSABLE_KERNELS`-ordered, .share_memory_()'d
    tensor of per-family real-TabICL-z_train mixing fractions when
    data.z_train_tabicl_mix_enabled is true, else None (see
    data.z_train_tabicl_mix_* in conf/data/gp_tasks.yaml and train.py::
    _tabicl_gap_to_mix_frac -- unlike kernel_weights, its values are
    initialized to the floor fraction here and overwritten ONCE from the
    measured TabICL-vs-analytic z_train gap before training starts, not
    updated repeatedly during training). Both must be created before the
    loader is first iterated (i.e. before workers fork/spawn) for the
    shared-memory update path in train.py to reach the workers — see
    LiveGPDataset's docstring. share_memory_() tensors use a
    file-descriptor-based reduction (torch.multiprocessing), not fork's
    copy-on-write, so this still works unchanged under spawn.
    """
    batch_size = int(t.batch_size)

    z_train_source = str(cfg.data.get("z_train_source", "analytic"))
    mix_enabled = bool(cfg.data.get("z_train_tabicl_mix_enabled", False))
    tabicl_live_enabled = mix_enabled or z_train_source in ("tabicl", "tabicl_split")

    # Dedicated group-size knob for the tabicl GPU-worker path, distinct from
    # live_group_multiplier (which stays analytic-CPU-worker-only, unaffected
    # by this default): benchmarked 2026-08-23 that batching 2 training
    # batches' worth of episodes into one generate_gp_batch call meaningfully
    # helps the tabicl path amortize TabICL's own per-call cost, while the
    # analytic path's docstring already established group_size=1 is close to
    # ITS OWN throughput ceiling -- bumping the shared knob would help nothing
    # there and only add cross-batch-diversity risk (see LiveGPDataset's
    # docstring), so this stays a separate knob rather than reusing
    # live_group_multiplier for both paths.
    if tabicl_live_enabled:
        group_multiplier = max(1, int(t.get("live_tabicl_group_multiplier", 2)))
    else:
        group_multiplier = max(1, int(t.get("live_group_multiplier", 1)))
    group_size = batch_size * group_multiplier

    if tabicl_live_enabled and device != "cuda":
        reason = (
            "data.z_train_tabicl_mix_enabled=true" if mix_enabled
            else f"data.z_train_source={z_train_source}"
        )
        raise ValueError(
            f"training.live_generation with {reason} requires device='cuda' "
            f"(got {device!r}) -- CPU-only TabICL inference was benchmarked and "
            "rejected as too slow to keep the GPU fed (see this function's "
            "docstring). Use data.z_train_source=analytic and "
            "data.z_train_tabicl_mix_enabled=false for CPU-only runs."
        )
    tabicl_device = device if tabicl_live_enabled else None

    if tabicl_live_enabled:
        # Few, fast GPU workers instead of many, slow CPU ones -- see the
        # VRAM paragraph above. Auto-sized from currently-free GPU memory
        # when training.live_tabicl_num_workers is left unset (null) -- see
        # resolve_live_tabicl_num_workers's docstring for why free (not
        # total) memory matters on a node where this GPU might be shared
        # with other jobs, and how it adapts across GPU models/VRAM sizes.
        num_workers = resolve_live_tabicl_num_workers(t, device)
    else:
        num_workers = int(t.get("live_num_workers", 8))
        # live_num_workers=16 is tuned for a 44-core node (see conf/config.yaml);
        # on a smaller cpuset (e.g. an OAR/Grid5000 job allocated 8 cores) that
        # oversubscribes CPU 2x and has been observed to OOM-kill workers/the main
        # process outright — every job in a same-sized 12-job sweep died this way
        # on 2026-08-06 (one crashed inside a Muon torch.compile trace with a
        # confusing "RuntimeError when making fake tensor call" that was actually
        # a killed worker surfacing mid-compile). Clamp to what this process can
        # actually schedule onto. Not applied to the GPU-worker path above: those
        # workers are CPU-thread-light (only feature warps/Python-loop overhead
        # run on CPU, everything else offloaded to the GPU), so CPU oversubscription
        # isn't the binding constraint there — VRAM is, and num_workers is already
        # small by default for that reason.
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
        # Initialized to the floor fraction uniformly; train.py overwrites
        # this in place (.copy_()) with the measured per-family gap-driven
        # fraction once, before the training loop starts (see this
        # function's Returns docstring above) -- floor here is just a safe
        # value for the brief window before that happens (e.g. if a worker
        # somehow drew a batch before the main process finishes the
        # gap-measurement pass).
        floor_frac = float(cfg.data.get("z_train_tabicl_mix_floor_frac", 0.05))
        tabicl_mix_weights = torch.full((n,), floor_frac, dtype=torch.float32).share_memory_()
    live_ds = LiveGPDataset(
        cfg, group_size=group_size, kernel_weights=kernel_weights, tabicl_device=tabicl_device,
        tabicl_mix_weights=tabicl_mix_weights,
    )
    loader = DataLoader(
        live_ds,
        batch_size=t.batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_limit_worker_threads if num_workers > 0 else None,
        multiprocessing_context="spawn" if (tabicl_live_enabled and num_workers > 0) else None,
    )
    return loader, kernel_weights, tabicl_mix_weights


def build_fixed_live_val_batches(
    cfg: DictConfig, t: DictConfig, device: str = "cpu",
) -> Tuple[List[dict], List[List[dict]]]:
    """Fixed, once-generated validation set for live-generation training.

    Generated once here with fixed, deterministic seeds (distinct from the
    training seed stream), then cached as plain collated (CPU) batches. Every
    validate() call iterates this same list instead of resampling, so val
    metrics track only model changes across training — the live-mode analogue
    of the disk path's held-out val_indices (train.py) and the existing
    kernel_fit probes' fixed-seed generation (_build_synthetic_kernel_batches).

    One generate_gp_batch call per batch (not one big call for the whole val
    set): a single call shares kernel/P/N/active_dims/d_features across all its
    episodes (see LiveGPDataset), so one call for the full val set would make
    every validation episode the same task — a much narrower probe than the
    disk path's val_indices, which stride across many shards/configs. Per-
    batch calls with distinct fixed seeds keep each batch internally
    homogeneous (required for collate_fn) while spanning many different
    kernels/configs across the val set as a whole.

    data.z_train_source=tabicl/tabicl_split: loads its own frozen TabICL copy
    on `device` (must be "cuda"), used for every val batch, then freed before
    returning — this runs once in the MAIN process (not a worker), a bounded
    number of calls (n_batches, typically ~16), so unlike the training-stream
    workers above this needs no spawn/CPU-throughput considerations. A
    separate load from train.py's own "z_train sim-to-real diagnostic" copy
    (different purpose/lifetime) rather than a shared instance, trading one
    extra ~5s startup load for not threading a loaded model through an
    unrelated code path.

    Requests return_kernel_metadata=True from every generate_gp_batch call
    (cheap: extra per-episode fields + the _L_ff/_alpha Cholesky factors
    already computed as a byproduct of sampling y_train, not new compute) so
    validate() can run pit.gp_analytical_posterior directly on these SAME
    episodes instead of needing a separately-drawn probe
    (train.py::_build_posterior_probe_batches, still used as the disk-mode
    fallback since CopulaDataset's on-disk shards never carry this metadata).
    _L_ff/_alpha are moved to CPU before being cached here regardless of
    gen_device, mirroring _build_posterior_probe_batches's own
    always-device="cpu" choice — these episodes live for the entire training
    run, so any risk of them pinning persistent VRAM (when
    z_train_source=tabicl/tabicl_split puts gen_device="cuda") is worth
    avoiding even though the tensors themselves are small.

    Returns (batches, episodes_by_batch): batches is the plain list of
    collated CPU batch dicts (unchanged contract, see below);
    episodes_by_batch[i] is the raw per-episode list (kernel metadata intact)
    for batches[i], same order, for validate()'s oracle-posterior scoring.

    Returned as a plain list (not a DataLoader): validate() only ever does
    ``for batch_idx, batch in enumerate(val_loader)`` and moves each batch to
    device itself, so a list of CPU batch dicts satisfies that contract with
    no changes to validate() regardless of z_train_source.
    """
    n_val = int(t.get("val_episodes", 500))
    val_seed = int(t.get("live_val_seed", 20260723))
    batch_size = int(t.batch_size)
    n_batches = max(1, (n_val + batch_size - 1) // batch_size)

    z_train_source = str(cfg.data.get("z_train_source", "analytic"))
    tabicl_live_enabled = z_train_source in ("tabicl", "tabicl_split")
    if tabicl_live_enabled and device != "cuda":
        raise ValueError(
            f"training.live_generation with data.z_train_source={z_train_source} "
            f"requires device='cuda' (got {device!r})."
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

    batches = []
    episodes_by_batch: List[List[dict]] = []
    with warnings.catch_warnings(), limited_main_process_threads():
        # Same degenerate-episode discard warnings as LiveGPDataset.__iter__
        # above, silenced here too (scoped to this call, not process-global,
        # since this runs once in the main process rather than a dedicated
        # live-generation worker).
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for i in range(n_batches):
            val_cfg = copy.deepcopy(cfg)
            val_cfg.seed = val_seed + i * 104_729  # distinct, fixed, reproducible per batch
            episodes = generate_gp_batch(
                val_cfg, batch_size, device=gen_device,
                tabicl_model=tabicl_model, tabicl_k_folds=tabicl_k_folds,
                tabicl_split_calib_frac=tabicl_split_calib_frac,
                return_kernel_metadata=True,
            )
            if gen_device == "cuda":
                for ep in episodes:
                    ep["_L_ff"] = ep["_L_ff"].cpu()
                    ep["_alpha"] = ep["_alpha"].cpu()
            batches.append(collate_fn(episodes))
            episodes_by_batch.append(episodes)

    if tabicl_model is not None:
        del tabicl_model
        if device == "cuda":
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    return batches, episodes_by_batch
