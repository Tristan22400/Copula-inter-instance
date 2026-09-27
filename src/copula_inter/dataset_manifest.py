"""Identity and shard indexing for generated PIT datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from omegaconf import OmegaConf

from copula_inter.artifacts import artifact_identity, canonical_digest, file_digest, mkstemp_like_open
from copula_inter.backend_registry import TABICL_Z_TRAIN_SOURCES
from copula_inter.backend_registry import z_train_source as z_train_source_of
from copula_inter.config_path import config_dict

if TYPE_CHECKING:
    from omegaconf import DictConfig

SCHEMA = 1


def generation_spec(cfg: DictConfig, marginal_checkpoint: str | None) -> dict:
    data = config_dict(cfg.data)
    for output_key in ("resume", "dataset_dir", "pit_dir"):
        data.pop(output_key, None)
    return {
        "schema": SCHEMA,
        "seed": OmegaConf.select(cfg, "seed"),
        "data": data,
        "tabicl": (
            config_dict(cfg.tabicl) if z_train_source_of(cfg) in TABICL_Z_TRAIN_SOURCES and "tabicl" in cfg else None
        ),
        "marginal": artifact_identity(marginal_checkpoint),
    }


def stat_identity(path: str | os.PathLike[str]) -> dict[str, int]:
    """The stat fields a shard sidecar records; a same-path replacement changes ctime/inode."""
    st = os.stat(path)
    return {"size": st.st_size, "ctime_ns": st.st_ctime_ns, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino}


def _stat_token(path: Path) -> str:
    st = stat_identity(path)
    return f"stat:{st['size']}:{st['mtime_ns']}:{st['ctime_ns']}:{st['inode']}"


def ensure_manifest(directory: str | os.PathLike[str], spec: dict) -> dict:
    """Create once across workers; refuse to reinterpret existing shards."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    candidate = {"schema": SCHEMA, "spec": spec, "digest": canonical_digest(spec)}
    if not path.exists():
        if any(root.glob("shard_*.pt")):
            raise ValueError(f"{root} has shards but no manifest; use a new dataset directory")
        fd, temporary = mkstemp_like_open(prefix=".manifest.", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                json.dump(candidate, out, sort_keys=True)
            try:
                os.link(temporary, path)  # atomic create without replacing another worker's manifest
            except FileExistsError:
                pass
        finally:
            os.unlink(temporary)
    with path.open(encoding="utf-8") as source:
        existing = json.load(source)
    if existing != candidate:
        raise ValueError(
            f"dataset identity mismatch in {root}: existing={existing.get('digest')}, "
            f"requested={candidate['digest']}. Use a new directory."
        )
    return candidate


def shard_count_path(shard_path: str | os.PathLike[str]) -> Path:
    return Path(shard_path).with_suffix(".count.json")


def verified_shard_digest(
    shard_path: str | os.PathLike[str],
    *,
    require_match: bool = False,
) -> tuple[int, str]:
    """Trust write-time digest only while size and ctime match its sidecar."""
    path = Path(shard_path)
    sidecar = shard_count_path(path)
    if not sidecar.is_file():
        raise ValueError(f"cannot resume incomplete shard {path}: count sidecar missing")
    saved = json.loads(sidecar.read_text())
    count = saved["count"]
    digest = saved.get("sha256")
    if not digest or any(saved.get(key) != value for key, value in stat_identity(path).items()):
        actual = file_digest(path)
        if require_match and digest and actual != digest:
            raise ValueError(f"shard content differs from its sidecar: {path}")
        digest = actual
    return count, digest


def contiguous_shard_counts(directory: str | os.PathLike[str], n_shards: int, shard_size: int) -> list[int]:
    root = Path(directory)
    counts = []
    for idx in range(n_shards):
        shard = root / f"shard_{idx:06d}.pt"
        sidecar = shard_count_path(shard)
        if not shard.is_file() or not sidecar.is_file():
            break
        with sidecar.open(encoding="utf-8") as source:
            count = json.load(source)["count"]
        if not isinstance(count, int) or count <= 0 or count > shard_size:
            raise ValueError(f"invalid episode count for {shard}: {count!r}")
        if count != shard_size and idx != n_shards - 1:
            raise ValueError(f"non-final shard {shard} has {count} episodes; expected {shard_size}")
        counts.append(count)
    return counts


def dataset_identity(directory: str | os.PathLike[str]) -> dict:
    """Content identity for caches; reuse verified write-time shard digests."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    shards = []
    for path in sorted(root.glob("shard_*.pt")):
        if shard_count_path(path).is_file():
            shards.append((path.name, verified_shard_digest(path)[1]))
        else:
            # Pre-refactor shards have no digest sidecar. Hashing every one on every eval run is
            # prohibitive for large datasets, so identify them by the same stat fields the sidecar
            # check trusts (a same-path replacement changes ctime/inode even if mtime is reset).
            shards.append((path.name, _stat_token(path)))
    legacy = [(p.name, _stat_token(p)) for p in sorted(root.glob("task_*.pt"))]
    meta = root / "meta.pt"
    return {
        "manifest": manifest,
        "meta": torch.load(meta, map_location="cpu", weights_only=True) if meta.is_file() else None,
        "shards_sha256": canonical_digest(shards + legacy),
    }
