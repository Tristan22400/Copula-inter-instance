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


def compose_config(conf_dir: str, config_name: str, overrides: list[str] | None = None) -> DictConfig:
    """Compose conf_dir/<config_name>.yaml outside a running Hydra app (resets Hydra's global state)."""
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=conf_dir, version_base=None):
        return hydra.compose(config_name=config_name, overrides=overrides or [])


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
