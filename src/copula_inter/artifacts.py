"""Artifact identity and atomic publication shared by training and evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

import torch


def file_digest(path: str | os.PathLike[str]) -> str:
    """Hash bytes once per run; a pathname or mtime is not an artifact identity."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(reference: str | os.PathLike[str] | None) -> dict | None:
    if reference is None:
        return None
    path = Path(reference).expanduser()
    if path.is_file():
        return {"sha256": file_digest(path), "size": path.stat().st_size}
    # Hub names are immutable only if the caller pins a revision. Keep the
    # reference visible in the key instead of pretending a local file exists.
    return {"reference": str(reference)}


def canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_file_mode() -> int:
    """The mode a plain open() would create: 0666 minus the process umask."""
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def mkstemp_like_open(*, prefix: str, dir: str | Path, suffix: str = "") -> tuple[int, str]:
    """tempfile.mkstemp, but with open()'s permissions (mkstemp forces 0600, which
    os.replace/os.link then publish -- unreadable to group jobs on shared storage)."""
    fd, temporary = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=dir)
    os.chmod(temporary, default_file_mode())
    return fd, temporary


def atomic_write(path: str | os.PathLike[str], writer: Callable[[str], None]) -> None:
    """Publish a complete file on the destination filesystem or leave it absent."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = mkstemp_like_open(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    os.close(fd)
    try:
        writer(temporary)
        os.replace(temporary, dest)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(value: object, path: str | os.PathLike[str]) -> None:
    atomic_write(path, lambda temporary: torch.save(value, temporary))


def atomic_json_save(value: object, path: str | os.PathLike[str]) -> None:
    def write(temporary: str) -> None:
        with open(temporary, "w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True)

    atomic_write(path, write)
