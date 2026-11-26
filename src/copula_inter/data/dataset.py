"""CopulaDataset and collate_fn for on-disk episodes.

Layouts (auto-detected): task_XXXXXX.pt (one episode per file) or
shard_XXXXXX.pt (a list of episodes per file, with meta.pt). Shards are loaded
with mmap into a small per-worker LRU cache.

A PIT overlay (copula_inter.repit_dataset) replaces z_train, z_test and
log_pdf_test of a sharded dataset by those of another marginal model: the
same tables, read through a different marginal. A dataset keeps its overlays
beside its tables, one per marginal, under a name:

    <dataset>/pit/                  tables + the PIT they were generated with
    <dataset>/overlays/<name>/      overlay.json + shard_XXXXXX.pt (one marginal)

The marginal's settings (backend, checkpoint, K folds, precision) are in
overlay.json, not in the name.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import OrderedDict
from glob import glob
from typing import Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from copula_inter.data.contracts import validate_episode

# Keys a PIT overlay replaces (copula_inter.repit_dataset), and its metadata file.
PIT_OVERLAY_KEYS = ("z_train", "z_test", "log_pdf_test")
PIT_OVERLAY_META = "overlay.json"
# Directory, beside a dataset's pit/, that holds its overlays.
PIT_OVERLAYS_DIR = "overlays"
_OVERLAY_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def pit_overlay_dir(pit_dir: str, name: str) -> str:
    """Directory of the overlay called name of the dataset whose tables are in pit_dir: <dataset>/overlays/<name>."""
    if not _OVERLAY_NAME.fullmatch(name):
        raise ValueError(f"PIT overlay name {name!r} must match {_OVERLAY_NAME.pattern} (a name, not a path)")
    return os.path.join(os.path.dirname(os.path.normpath(pit_dir)), PIT_OVERLAYS_DIR, name)


# Keys checked for NaN/Inf when an episode is loaded (older shards may contain some).
_FINITE_CHECK_KEYS = ("z_train", "z_test", "y_train", "y_test")


def has_varying_context_feature(x_train: torch.Tensor) -> bool:
    """Whether some feature column of a (P, d) context varies over its rows (and is not all NaN).

    An episode without one ("null" episode) gives the features no say: TabPFN and Kumo
    drop constant columns, so the generic marginals fall back to a Gaussian fitted on the
    context targets (finetune_adapters.backend_pit).
    """
    varies = (x_train != x_train[:1]).any(dim=0) & ~x_train.isnan().all(dim=0)
    return bool(varies.any())


def _episode_is_finite(ep: dict) -> bool:
    return all(torch.isfinite(ep[k]).all() for k in _FINITE_CHECK_KEYS if k in ep)


def _add_derived_fields(ep: dict) -> dict:
    """Add R_prior (= R_star) and Sigma_star (= R_star * sigma_star sigma_star^T) when a shard omits them.

    A shard written with data.store_r_star=false has no R_star, so neither is derived.
    """
    if "R_star" not in ep:
        return ep
    if "Sigma_star" not in ep:
        sigma = ep["sigma_star"]
        ep["Sigma_star"] = ep["R_star"] * sigma.unsqueeze(0) * sigma.unsqueeze(1)
    if "R_prior" not in ep:
        ep["R_prior"] = ep["R_star"].clone()
    return ep


class CopulaDataset(Dataset):
    """Dataset of saved PIT episodes (individual files or shards)."""

    _SHARD_CACHE_SIZE = 4  # default shards kept in memory per worker process
    # Sharded mode (set by _init_sharded).
    _shard_files: List[str]
    _shard_size: int
    _n_total: int

    def __init__(
        self,
        episode_dir: Optional[str] = None,
        file_list: Optional[List[str]] = None,
        shard_cache_size: Optional[int] = None,
        pit_overlay: Optional[str] = None,
        max_shards: Optional[int] = None,
    ) -> None:
        # Cache size (size it to hold a ShardBlockSampler block).
        if shard_cache_size is not None:
            self._SHARD_CACHE_SIZE = shard_cache_size

        if file_list is not None:
            # Explicit list → individual-file mode (backward compat)
            self._init_individual(sorted(file_list))
            return

        if episode_dir is None:
            raise ValueError("Provide either episode_dir or file_list.")

        meta_path = os.path.join(episode_dir, "meta.pt")
        shard_files = sorted(glob(os.path.join(episode_dir, "shard_*.pt")))

        if shard_files and os.path.exists(meta_path):
            self._init_sharded(shard_files, meta_path)
            if max_shards is not None:
                self._keep_first_shards(int(max_shards))
            if pit_overlay is not None:
                self._init_overlay(pit_overlay_dir(episode_dir, pit_overlay))
        else:
            if pit_overlay is not None or max_shards is not None:
                raise ValueError(
                    f"pit_overlay / max_shards need a sharded dataset; {episode_dir} has no shard_*.pt + meta.pt"
                )
            indiv_files = sorted(glob(os.path.join(episode_dir, "task_*.pt")))
            if not indiv_files:
                raise RuntimeError(
                    f"No episode files found in {episode_dir}. Expected shard_*.pt+meta.pt or task_*.pt files."
                )
            self._init_individual(indiv_files)

    _overlay_files: Optional[List[str]] = None

    def _keep_first_shards(self, max_shards: int) -> None:
        """Restrict the dataset to its first max_shards shards (a fixed subset, e.g. one pass in a short run)."""
        if max_shards < 1:
            raise ValueError(f"max_shards={max_shards} must be at least 1")
        if max_shards < len(self._shard_files):
            self._shard_files = self._shard_files[:max_shards]
            self._n_total = max_shards * self._shard_size

    def _init_overlay(self, overlay_dir: str) -> None:
        """Check that overlay_dir is a complete PIT overlay of this dataset's tables."""
        meta_path = os.path.join(overlay_dir, PIT_OVERLAY_META)
        if not os.path.isfile(meta_path):
            root = os.path.dirname(overlay_dir)
            names = sorted(os.listdir(root)) if os.path.isdir(root) else []
            raise ValueError(f"{overlay_dir} is not a PIT overlay (no {PIT_OVERLAY_META}); overlays here: {names}")
        with open(meta_path, encoding="utf-8") as source:
            meta = json.load(source)
        source_dir = os.path.dirname(self._shard_files[0])
        with open(os.path.join(source_dir, "manifest.json"), encoding="utf-8") as source:
            digest = json.load(source).get("digest")
        if meta.get("source_digest") != digest:
            raise ValueError(f"PIT overlay {overlay_dir} was computed from another dataset than {source_dir}")
        if int(meta["shard_size"]) != self._shard_size:
            raise ValueError(f"PIT overlay {overlay_dir} has shard_size {meta['shard_size']}, not {self._shard_size}")
        files = [os.path.join(overlay_dir, os.path.basename(path)) for path in self._shard_files]
        missing = [path for path in files if not os.path.isfile(path)]
        if missing:
            raise ValueError(
                f"PIT overlay {overlay_dir} lacks {len(missing)} of {len(files)} shards, e.g. {missing[0]}"
            )
        self._overlay_files = files

    def _init_individual(self, files: List[str]) -> None:
        self._mode = "individual"
        existing = [f for f in files if os.path.isfile(f)]
        if len(existing) < len(files):
            import warnings

            warnings.warn(f"CopulaDataset: {len(files) - len(existing)} listed file(s) missing on disk.")
        if not existing:
            raise RuntimeError("No .pt files available.")
        self._files = existing

    def _init_sharded(self, shard_files: List[str], meta_path: str) -> None:
        self._mode = "sharded"
        meta = torch.load(meta_path, map_location="cpu", weights_only=True)
        self._n_total = int(meta["n_total"])
        self._shard_size = int(meta["shard_size"])
        if self._n_total < 0 or self._shard_size <= 0:
            raise ValueError(f"invalid shard metadata in {meta_path}")
        needed = (self._n_total + self._shard_size - 1) // self._shard_size
        directory = os.path.dirname(meta_path)
        expected = [os.path.join(directory, f"shard_{i:06d}.pt") for i in range(needed)]
        if any(not os.path.isfile(path) for path in expected):
            raise ValueError(f"shard metadata in {meta_path} references missing or noncontiguous shards")
        self._shard_files = expected
        if digest := meta.get("manifest_digest"):
            manifest_path = os.path.join(directory, "manifest.json")
            with open(manifest_path, encoding="utf-8") as source:
                manifest = json.load(source)
            if manifest.get("digest") != digest:
                raise ValueError(f"manifest identity does not match {meta_path}")
            from copula_inter.data.manifest import shard_count_path

            counts = []
            for path in expected:
                sidecar = shard_count_path(path)
                if not sidecar.is_file():
                    raise ValueError(f"manifest dataset shard lacks count sidecar: {path}")
                with sidecar.open(encoding="utf-8") as source:
                    counts.append(json.load(source)["count"])
            if sum(counts) != self._n_total:
                raise ValueError(f"shard counts disagree with {meta_path}")
        self._shard_cache: OrderedDict[str, list] = OrderedDict()
        self._overlay_cache: OrderedDict[str, list] = OrderedDict()

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

    def get_exact(self, idx: int) -> dict:
        """Read this index without substituting another episode, for paired evaluations."""
        ep = self._load_individual(idx) if self._mode == "individual" else self._load_shard_entry(idx)
        if not _episode_is_finite(ep):
            raise ValueError(f"episode at idx {idx} has non-finite context or query values")
        return ep

    def _load_individual(self, idx: int) -> dict:
        ep = torch.load(self._files[idx], map_location="cpu", weights_only=True, mmap=True)
        return _add_derived_fields(ep)

    def _get_individual(self, idx: int) -> dict:
        try:
            return self._load_individual(idx)
        except FileNotFoundError:
            candidates = [i for i in range(len(self._files)) if i != idx]
            if not candidates:
                raise
            return self._load_individual(random.choice(candidates))

    _MAX_INVALID_RETRIES = 8

    def _load_shard_entry(self, idx: int) -> dict:
        if idx < 0 or idx >= self._n_total:
            raise IndexError(idx)
        shard_idx = idx // self._shard_size
        local_idx = idx - shard_idx * self._shard_size
        shard_path = self._shard_files[shard_idx]

        if shard_path not in self._shard_cache:
            if len(self._shard_cache) >= self._SHARD_CACHE_SIZE:
                self._shard_cache.popitem(last=False)  # evict LRU
            shard = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
            self._shard_cache[shard_path] = shard
        else:
            # Move to end to mark as most-recently used
            self._shard_cache.move_to_end(shard_path)

        shard = self._shard_cache[shard_path]
        if local_idx >= len(shard):
            raise ValueError(f"shard {shard_path} has fewer episodes than its metadata declares")
        # Derive R_prior/Sigma_star on a shallow copy so the cached shard stays mmap-only.
        ep = dict(shard[local_idx])
        if self._overlay_files is not None:
            ep.update(self._overlay_entry(shard_idx, local_idx, ep))
        return _add_derived_fields(ep)

    def _overlay_entry(self, shard_idx: int, local_idx: int, ep: dict) -> dict:
        """The PIT_OVERLAY_KEYS of one episode from the overlay, checked against its table shape."""
        assert self._overlay_files is not None
        path = self._overlay_files[shard_idx]
        if path not in self._overlay_cache:
            if len(self._overlay_cache) >= self._SHARD_CACHE_SIZE:
                self._overlay_cache.popitem(last=False)
            self._overlay_cache[path] = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        else:
            self._overlay_cache.move_to_end(path)
        overlay = self._overlay_cache[path]
        if local_idx >= len(overlay):
            raise ValueError(f"PIT overlay shard {path} has fewer episodes than its table shard")
        entry = {key: overlay[local_idx][key] for key in PIT_OVERLAY_KEYS}
        if entry["z_train"].shape != ep["z_train"].shape or entry["z_test"].shape != ep["z_test"].shape:
            raise ValueError(f"PIT overlay shard {path} episode {local_idx} does not match its table's (P, N)")
        return entry

    def _get_sharded(self, idx: int) -> dict:
        # Skip non-finite episodes, to the next episode in the same shard.
        shard_size = self._shard_size
        shard_start = (idx // shard_size) * shard_size
        shard_len = min(shard_size, self._n_total - shard_start)
        probe = idx
        for _ in range(self._MAX_INVALID_RETRIES):
            ep = self._load_shard_entry(probe)
            if _episode_is_finite(ep):
                return ep
            next_probe = shard_start + (probe - shard_start + 1) % shard_len
            import warnings

            warnings.warn(
                f"CopulaDataset: episode at idx {probe} has non-finite "
                f"z_train/y_train (stale degenerate episode); skipping to "
                f"idx {next_probe}.",
                RuntimeWarning,
            )
            probe = next_probe
        raise RuntimeError(
            f"CopulaDataset: {self._MAX_INVALID_RETRIES} consecutive non-finite "
            f"episodes starting at idx {idx} — dataset may need regeneration."
        )


def collate_fn(samples: List[dict]) -> dict:
    """Pad a batch of episodes to the batch's max P and N.

    Returns:
        x_train (B, P_max, d_x), x_test (B, N_max, d_x), y_train, z_train
        (B, P_max), y_test, z_test, log_pdf_test (B, N_max; 0 on padding),
        train_mask (B, P_max) and test_mask (B, N_max) bool, R_star and
        Sigma_star (B, N_max, N_max) when every episode has R_star, R_prior when present, mu_star and
        sigma_star (B, N_max), n_train and n_test (B,).
    """
    if not samples:
        raise ValueError("collate_fn requires at least one episode")
    for sample in samples:
        validate_episode(sample)
    B = len(samples)
    d_x = samples[0]["x_norm_train"].shape[-1]

    # Episodes with different d_features cannot be stacked (use ShardHomogeneousBatchSampler).
    if any(s["x_norm_train"].shape[-1] != d_x for s in samples):
        d_set = sorted({int(s["x_norm_train"].shape[-1]) for s in samples})
        raise RuntimeError(
            f"collate_fn received a batch with mixed feature counts {d_set}. "
            "This dataset has per-shard-varying d_features; batches must stay "
            "within one shard. Ensure train.py uses ShardHomogeneousBatchSampler "
            "(auto-enabled for variable-d datasets)."
        )

    P_list = [int(s["n_train"]) for s in samples]
    N_list = [int(s["n_test"]) for s in samples]
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
    # The oracle matrices are optional (data.store_r_star=false shards omit them).
    has_star = all("R_star" in s for s in samples)
    R_star = torch.zeros(B, N_max, N_max) if has_star else None
    Sigma_star = torch.zeros(B, N_max, N_max) if has_star else None
    mu_star = torch.zeros(B, N_max)
    sigma_star = torch.zeros(B, N_max)
    # R_prior is optional.
    has_prior = "R_prior" in samples[0]
    R_prior = torch.zeros(B, N_max, N_max) if has_prior else None

    for b, s in enumerate(samples):
        P = P_list[b]
        N = N_list[b]

        x_train[b, :P] = s["x_norm_train"]
        x_test[b, :N] = s["x_norm_test"]
        y_train[b, :P] = s["y_train"]
        y_test[b, :N] = s["y_test"]
        z_train[b, :P] = s["z_train"]
        z_test[b, :N] = s["z_test"]
        log_pdf_test[b, :N] = s["log_pdf_test"]
        train_mask[b, :P] = True
        test_mask[b, :N] = True
        if R_star is not None and Sigma_star is not None:
            R_star[b, :N, :N] = s["R_star"]
            Sigma_star[b, :N, :N] = s["Sigma_star"]
        mu_star[b, :N] = s["mu_star"]
        sigma_star[b, :N] = s["sigma_star"]
        if R_prior is not None:
            R_prior[b, :N, :N] = s["R_prior"]

    out = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "z_train": z_train,
        "z_test": z_test,
        "log_pdf_test": log_pdf_test,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "mu_star": mu_star,
        "sigma_star": sigma_star,
        "n_train": torch.tensor(P_list, dtype=torch.long),
        "n_test": torch.tensor(N_list, dtype=torch.long),
    }
    if R_star is not None and Sigma_star is not None:
        out["R_star"] = R_star
        out["Sigma_star"] = Sigma_star
    if R_prior is not None:
        out["R_prior"] = R_prior
    return out


class ShardBlockSampler(Sampler[int]):
    """Epoch sampler that shuffles at shard-block granularity (at most block_shards shards resident).

    Yields a permutation of range(len(subset_indices)); subset_indices maps
    positions to global dataset indices.
    """

    def __init__(self, subset_indices: Sequence[int], shard_size: int, block_shards: int = 16) -> None:
        self.subset_indices = list(subset_indices)
        self.shard_size = shard_size
        self.block_shards = block_shards

    def __len__(self) -> int:
        return len(self.subset_indices)

    def __iter__(self) -> Iterator:
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


class ShardHomogeneousBatchSampler(Sampler[List[int]]):
    """Batch sampler whose batches never span shards (needed when d_features varies per shard).

    Yields lists of local positions (subset_indices as for ShardBlockSampler),
    covering each once per epoch; shuffle randomizes shard and within-shard
    order. A shard's last batch may be short unless drop_last.
    """

    def __init__(
        self,
        subset_indices: Sequence[int],
        shard_size: int,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.subset_indices = list(subset_indices)
        self.shard_size = shard_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def _groups(self) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for local_pos, global_idx in enumerate(self.subset_indices):
            groups.setdefault(global_idx // self.shard_size, []).append(local_pos)
        return groups

    def __len__(self) -> int:
        total = 0
        for members in self._groups().values():
            if self.drop_last:
                total += len(members) // self.batch_size
            else:
                total += (len(members) + self.batch_size - 1) // self.batch_size
        return total

    def __iter__(self) -> Iterator:
        groups = self._groups()
        shard_ids = list(groups.keys())
        if self.shuffle:
            shard_ids = [shard_ids[i] for i in torch.randperm(len(shard_ids)).tolist()]
        for sid in shard_ids:
            members = groups[sid]
            if self.shuffle:
                members = [members[i] for i in torch.randperm(len(members)).tolist()]
            for start in range(0, len(members), self.batch_size):
                batch = members[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch
