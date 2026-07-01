"""
dataset.py — CopulaDataset and collate_fn for inter-instance copula training.

Supports two on-disk layouts (auto-detected):

  Individual files   task_XXXXXX.pt   — one episode per file (legacy)
  Sharded files      shard_XXXXXX.pt  — list of B episodes per file (new)

The sharded layout is produced by generate_pit_dataset.py and is much faster
on NFS because it reduces file-metadata operations by a factor of B.
A small LRU shard cache (default 4 shards) keeps recently accessed shards
in memory to amortise repeated random accesses within a DataLoader.
"""

from __future__ import annotations

import os
import random
from collections import OrderedDict
from glob import glob
from typing import List, Optional, Sequence

import torch
from torch.utils.data import Dataset, Sampler


class CopulaDataset(Dataset):
    """Dataset of pre-computed PIT episodes.

    Auto-detects individual (task_*.pt) or sharded (shard_*.pt + meta.pt) layout.
    """

    _SHARD_CACHE_SIZE = 4   # default shards kept in memory per worker process

    def __init__(
        self,
        episode_dir: Optional[str] = None,
        file_list: Optional[List[str]] = None,
        shard_cache_size: Optional[int] = None,
    ):
        # Override the default cache size — needed so it can be sized to hold
        # a full ShardBlockSampler block (otherwise the 4-slot default
        # thrashes against a larger block, since each worker still touches
        # every shard in the active block).
        if shard_cache_size is not None:
            self._SHARD_CACHE_SIZE = shard_cache_size

        if file_list is not None:
            # Explicit list → individual-file mode (backward compat)
            self._init_individual(sorted(file_list))
            return

        if episode_dir is None:
            raise ValueError("Provide either episode_dir or file_list.")

        meta_path   = os.path.join(episode_dir, "meta.pt")
        shard_files = sorted(glob(os.path.join(episode_dir, "shard_*.pt")))

        if shard_files and os.path.exists(meta_path):
            self._init_sharded(shard_files, meta_path)
        else:
            indiv_files = sorted(glob(os.path.join(episode_dir, "task_*.pt")))
            if not indiv_files:
                raise RuntimeError(
                    f"No episode files found in {episode_dir}. "
                    "Expected shard_*.pt+meta.pt or task_*.pt files."
                )
            self._init_individual(indiv_files)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_individual(self, files: List[str]) -> None:
        self._mode  = "individual"
        existing = [f for f in files if os.path.isfile(f)]
        if len(existing) < len(files):
            import warnings
            warnings.warn(
                f"CopulaDataset: {len(files) - len(existing)} listed file(s) missing on disk."
            )
        if not existing:
            raise RuntimeError("No .pt files available.")
        self._files = existing

    def _init_sharded(self, shard_files: List[str], meta_path: str) -> None:
        self._mode         = "sharded"
        self._shard_files  = shard_files
        meta               = torch.load(meta_path, map_location="cpu", weights_only=True)
        self._n_total      = int(meta["n_total"])
        self._shard_size   = int(meta["shard_size"])
        self._shard_cache: OrderedDict[str, list] = OrderedDict()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    @property
    def shard_size(self) -> int:
        """Episodes per shard (only meaningful in sharded mode)."""
        if self._mode != "sharded":
            raise AttributeError("shard_size is only defined for sharded-layout datasets.")
        return self._shard_size

    def __len__(self) -> int:
        if self._mode == "individual":
            return len(self._files)
        return self._n_total

    def __getitem__(self, idx: int) -> dict:
        if self._mode == "individual":
            return self._get_individual(idx)
        return self._get_sharded(idx)

    # ------------------------------------------------------------------
    # Individual-file loading
    # ------------------------------------------------------------------

    def _get_individual(self, idx: int) -> dict:
        try:
            return torch.load(self._files[idx], map_location="cpu", weights_only=True)
        except FileNotFoundError:
            candidates = [i for i in range(len(self._files)) if i != idx]
            if not candidates:
                raise
            return torch.load(
                self._files[random.choice(candidates)], map_location="cpu", weights_only=True
            )

    # ------------------------------------------------------------------
    # Sharded loading with LRU cache
    # ------------------------------------------------------------------

    def _get_sharded(self, idx: int) -> dict:
        shard_idx  = min(idx // self._shard_size, len(self._shard_files) - 1)
        local_idx  = idx  - shard_idx * self._shard_size
        shard_path = self._shard_files[shard_idx]

        if shard_path not in self._shard_cache:
            if len(self._shard_cache) >= self._SHARD_CACHE_SIZE:
                self._shard_cache.popitem(last=False)   # evict LRU
            self._shard_cache[shard_path] = torch.load(
                shard_path, map_location="cpu", weights_only=False
            )
        else:
            # Move to end to mark as most-recently used
            self._shard_cache.move_to_end(shard_path)

        shard     = self._shard_cache[shard_path]
        local_idx = min(local_idx, len(shard) - 1)   # guard for last shard
        return shard[local_idx]


def collate_fn(samples: List[dict]) -> dict:
    """Pad a batch of variable-length tasks.

    Returns (all padded to batch-max P, N):
        x_train      : (B, P_max, d_x)
        x_test       : (B, N_max, d_x)
        y_train      : (B, P_max)
        y_test       : (B, N_max)
        z_train      : (B, P_max)
        z_test       : (B, N_max)
        log_pdf_test : (B, N_max)        (0 for padding → log(1)=0 contributes nothing)
        train_mask   : BoolTensor (B, P_max)
        test_mask    : BoolTensor (B, N_max)
        R_star       : (B, N_max, N_max)
        Sigma_star   : (B, N_max, N_max)
        mu_star      : (B, N_max)
        sigma_star   : (B, N_max)
        n_train      : LongTensor (B,)
        n_test       : LongTensor (B,)
    """
    B   = len(samples)
    d_x = samples[0]["x_norm_train"].shape[-1]

    P_list = [int(s["n_train"].item()) for s in samples]
    N_list = [int(s["n_test"].item())  for s in samples]
    P_max  = max(P_list)
    N_max  = max(N_list)

    x_train      = torch.zeros(B, P_max, d_x)
    x_test       = torch.zeros(B, N_max, d_x)
    y_train      = torch.zeros(B, P_max)
    y_test       = torch.zeros(B, N_max)
    z_train      = torch.zeros(B, P_max)
    z_test       = torch.zeros(B, N_max)
    log_pdf_test = torch.zeros(B, N_max)
    train_mask   = torch.zeros(B, P_max, dtype=torch.bool)
    test_mask    = torch.zeros(B, N_max, dtype=torch.bool)
    R_star       = torch.zeros(B, N_max, N_max)
    Sigma_star   = torch.zeros(B, N_max, N_max)
    mu_star      = torch.zeros(B, N_max)
    sigma_star   = torch.zeros(B, N_max)

    for b, s in enumerate(samples):
        P = P_list[b]
        N = N_list[b]

        x_train[b, :P]      = s["x_norm_train"]
        x_test[b,  :N]      = s["x_norm_test"]
        y_train[b, :P]      = s["y_train"]
        y_test[b,  :N]      = s["y_test"]
        z_train[b, :P]      = s["z_train"]
        z_test[b,  :N]      = s["z_test"]
        log_pdf_test[b, :N] = s["log_pdf_test"]
        train_mask[b, :P]   = True
        test_mask[b,  :N]   = True
        R_star[b,    :N, :N] = s["R_star"]
        Sigma_star[b, :N, :N] = s["Sigma_star"]
        mu_star[b,   :N]    = s["mu_star"]
        sigma_star[b, :N]   = s["sigma_star"]

    return {
        "x_train":      x_train,
        "x_test":       x_test,
        "y_train":      y_train,
        "y_test":       y_test,
        "z_train":      z_train,
        "z_test":       z_test,
        "log_pdf_test": log_pdf_test,
        "train_mask":   train_mask,
        "test_mask":    test_mask,
        "R_star":       R_star,
        "Sigma_star":   Sigma_star,
        "mu_star":      mu_star,
        "sigma_star":   sigma_star,
        "n_train":      torch.tensor(P_list, dtype=torch.long),
        "n_test":       torch.tensor(N_list, dtype=torch.long),
    }


class ShardBlockSampler(Sampler[int]):
    """Epoch sampler for sharded datasets: shuffles at shard-block granularity
    instead of globally, so at most ``block_shards`` shards need to be
    resident at once (avoids one-full-shard-load-per-sample thrashing on
    network storage when the dataset spans thousands of shards).

    Still yields a true permutation of ``range(len(subset_indices))`` each
    epoch — every position is produced exactly once, nothing is skipped or
    repeated — identical contract to ``shuffle=True``. Only the *order* is
    weaker: fully random within a block of ``block_shards`` shards, but not
    reshuffled across blocks.

    ``subset_indices`` maps each local position (what the sampler yields,
    i.e. what a wrapping ``Subset`` expects) to its *global* dataset index,
    used only to look up which shard that position lives in. Pass
    ``train_dataset.indices`` when wrapping a ``torch.utils.data.Subset``.
    """

    def __init__(self, subset_indices: Sequence[int], shard_size: int, block_shards: int = 16):
        self.subset_indices = list(subset_indices)
        self.shard_size = shard_size
        self.block_shards = block_shards

    def __len__(self) -> int:
        return len(self.subset_indices)

    def __iter__(self):
        groups: dict[int, list[int]] = {}
        for local_pos, global_idx in enumerate(self.subset_indices):
            groups.setdefault(global_idx // self.shard_size, []).append(local_pos)
        shard_ids = list(groups.keys())

        shard_order = [shard_ids[i] for i in torch.randperm(len(shard_ids)).tolist()]
        for start in range(0, len(shard_order), self.block_shards):
            block_positions: list[int] = []
            for sid in shard_order[start : start + self.block_shards]:
                block_positions.extend(groups[sid])
            for i in torch.randperm(len(block_positions)).tolist():
                yield block_positions[i]
