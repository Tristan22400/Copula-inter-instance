"""Resolve Hydra configs from a checkout or an installed wheel."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def config_dir(caller_file: str) -> str:
    checkout = Path(caller_file).resolve().parent.parent / "conf"
    return str(checkout if checkout.is_dir() else files("conf"))
