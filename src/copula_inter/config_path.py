"""Resolve Hydra configs from a checkout or an installed wheel."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf


def config_dir(caller_file: str) -> str:
    for parent in Path(caller_file).resolve().parents:
        if (parent / "conf" / "config.yaml").is_file():
            return str(parent / "conf")
    return str(files("conf"))


def config_dict(cfg: DictConfig, resolve: bool = True) -> dict[str, Any]:
    """cfg as a plain dict, interpolations resolved unless resolve=False."""
    out = OmegaConf.to_container(cfg, resolve=resolve)
    if not isinstance(out, dict):
        raise TypeError(f"expected a mapping config, got {type(out).__name__}")
    return cast(dict[str, Any], out)


def merge_configs(*parts: Any) -> DictConfig:
    """OmegaConf.merge of mapping configs, as a DictConfig."""
    out = OmegaConf.merge(*parts)
    if not isinstance(out, DictConfig):
        raise TypeError(f"expected a mapping config, got {type(out).__name__}")
    return out
