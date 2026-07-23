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

import copy
from typing import Iterator, List

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from data_gen import generate_gp_batch
from dataset import collate_fn


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
    """

    def __init__(self, cfg: DictConfig, group_size: int = 1):
        # Deep-copy so mutating .seed per call never touches the caller's cfg
        # (and so pickling this Dataset to worker processes doesn't drag along
        # anything unexpected the caller's cfg object might reference later).
        self._cfg = copy.deepcopy(cfg)
        self._base_seed = int(getattr(cfg, "seed", None) or 0)
        self.group_size = group_size

    def _seed_for(self, worker_id: int, call_idx: int) -> int:
        # Not a cryptographic mix — just enough spread that (worker_id, call_idx)
        # collisions are astronomically unlikely over a training run. A
        # collision would only cost a moment of duplicated episodes, not
        # correctness, so this doesn't need to be bulletproof.
        raw = (self._base_seed + 1) * 1_000_003 + worker_id * 1_000_000_007 + call_idx
        return raw % (2**63 - 1)

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        cfg = copy.deepcopy(self._cfg)
        call_idx = 0
        while True:
            cfg.seed = self._seed_for(worker_id, call_idx)
            call_idx += 1
            episodes = generate_gp_batch(cfg, self.group_size, device="cpu")
            for ep in episodes:
                yield ep


def build_live_train_loader(cfg: DictConfig, t: DictConfig, device: str) -> DataLoader:
    """Training DataLoader backed by LiveGPDataset instead of an on-disk
    CopulaDataset. Mirrors the disk path's DataLoader kwargs (train.py) so
    downstream code — batch dict shape, the non_blocking .to(device) call, the
    train_iter/StopIteration re-creation loop — is unaffected. LiveGPDataset
    never raises StopIteration, so that re-creation branch simply never fires.
    """
    batch_size = int(t.batch_size)
    group_multiplier = max(1, int(t.get("live_group_multiplier", 1)))
    group_size = batch_size * group_multiplier
    num_workers = int(t.get("live_num_workers", 8))
    live_ds = LiveGPDataset(cfg, group_size=group_size)
    return DataLoader(
        live_ds,
        batch_size=t.batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def build_fixed_live_val_batches(cfg: DictConfig, t: DictConfig) -> List[dict]:
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

    Returned as a plain list (not a DataLoader): validate() only ever does
    ``for batch_idx, batch in enumerate(val_loader)`` and moves each batch to
    device itself, so a list of CPU batch dicts satisfies that contract with
    no changes to validate().
    """
    n_val = int(t.get("live_val_episodes", 500))
    val_seed = int(t.get("live_val_seed", 20260723))
    batch_size = int(t.batch_size)
    n_batches = max(1, (n_val + batch_size - 1) // batch_size)

    batches = []
    for i in range(n_batches):
        val_cfg = copy.deepcopy(cfg)
        val_cfg.seed = val_seed + i * 104_729  # distinct, fixed, reproducible per batch
        episodes = generate_gp_batch(val_cfg, batch_size, device="cpu")
        batches.append(collate_fn(episodes))
    return batches
