"""Identity and shard indexing for generated PIT datasets."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

from copula_inter.artifacts import artifact_identity, canonical_digest, file_digest
from copula_inter.backend_registry import z_train_source as z_train_source_of

SCHEMA = 1


def generation_spec(cfg, marginal_checkpoint: str | None) -> dict:
    data = OmegaConf.to_container(cfg.data, resolve=True)
    for output_key in ("resume", "dataset_dir", "pit_dir"):
        data.pop(output_key, None)
    return {
        "schema": SCHEMA,
        "seed": OmegaConf.select(cfg, "seed"),
        "data": data,
        "tabicl": (
            OmegaConf.to_container(cfg.tabicl, resolve=True)
            if z_train_source_of(cfg) in ("tabicl", "tabicl_split")
            and "tabicl" in cfg else None
        ),
        "marginal": artifact_identity(marginal_checkpoint),
    }


def ensure_manifest(directory: str | os.PathLike[str], spec: dict) -> dict:
    """Create once across workers; refuse to reinterpret existing shards."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    candidate = {"schema": SCHEMA, "spec": spec, "digest": canonical_digest(spec)}
    if not path.exists():
        if any(root.glob("shard_*.pt")) and not path.exists():
            raise ValueError(f"{root} has shards but no manifest; use a new dataset directory")
        fd, temporary = tempfile.mkstemp(prefix=".manifest.", dir=root)
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
    shard_path: str | os.PathLike[str], *, require_match: bool = False,
) -> tuple[int, str]:
    """Trust write-time digest only while size and ctime match its sidecar."""
    path = Path(shard_path)
    sidecar = shard_count_path(path)
    if not sidecar.is_file():
        raise ValueError(f"cannot resume incomplete shard {path}: count sidecar missing")
    saved = json.loads(sidecar.read_text())
    count = saved["count"]
    stat = path.stat()
    digest = saved.get("sha256")
    if (not digest or saved.get("size") != stat.st_size
            or saved.get("ctime_ns") != stat.st_ctime_ns
            or saved.get("mtime_ns") != stat.st_mtime_ns
            or saved.get("inode") != stat.st_ino):
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
        sidecar = shard_count_path(path)
        digest = verified_shard_digest(path)[1] if sidecar.is_file() else None
        # Replacing bytes at the same pathname changes ctime even if mtime is
        # reset. Legacy or modified shards are hashed directly.
        shards.append((path.name, digest or file_digest(path)))
    legacy = [(p.name, file_digest(p)) for p in sorted(root.glob("task_*.pt"))]
    meta = root / "meta.pt"
    return {
        "manifest": manifest,
        "meta": torch.load(meta, map_location="cpu", weights_only=True) if meta.is_file() else None,
        "shards_sha256": canonical_digest(shards + legacy),
    }
