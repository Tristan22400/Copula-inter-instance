"""Resolve Hydra configs from a checkout or an installed wheel."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def config_dir(caller_file: str) -> str:
    for parent in Path(caller_file).resolve().parents:
        if (parent / "conf" / "config.yaml").is_file():
            return str(parent / "conf")
    return str(files("conf"))
