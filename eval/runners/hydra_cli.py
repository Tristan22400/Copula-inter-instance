"""Hydra entry points for the eval runners: one typed dataclass config per runner, key=value overrides.

Each runner registers its dataclass as `<name>_schema` and composes
conf/eval/<name>.yaml (the schema plus the shared conf/eval/_runner.yaml:
no output directory, no chdir, logging to stdout). `--cfg job` prints every
key with its current value.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Collection
from typing import Any, Callable, TypeVar

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue

from copula_inter.config_path import config_dict, config_dir

T = TypeVar("T")

EVAL_CONF_DIR = os.path.join(config_dir(__file__), "eval")

# Hydra's own command-line flags; any other --flag is a leftover argparse flag.
_HYDRA_FLAGS = {
    "--help",
    "--hydra-help",
    "--version",
    "--cfg",
    "--resolve",
    "--package",
    "--run",
    "--multirun",
    "--shell-completion",
    "--config-path",
    "--config-name",
    "--config-dir",
    "--experimental-rerun",
    "--info",
}


def _register(config_name: str, schema: type) -> None:
    ConfigStore.instance().store(name=f"{config_name}_schema", node=schema)


def _to_spec(cfg: DictConfig, schema: type[T]) -> T:
    spec = OmegaConf.to_object(cfg)
    if not isinstance(spec, schema):
        raise TypeError(f"expected a {schema.__name__} config, got {type(spec).__name__}")
    return spec


def check_choice(key: str, value: str | None, choices: Collection[str]) -> None:
    """Raise ValueError unless value is None or one of choices."""
    if value is not None and value not in choices:
        raise ValueError(f"{key}={value!r} is not one of {sorted(choices)}")


def check_choices(key: str, values: Collection[str], choices: Collection[str]) -> None:
    """Raise ValueError unless every value is one of choices."""
    for value in values:
        check_choice(key, value, choices)


def _flat_keys(node: dict[str, Any], prefix: str = "") -> list[str]:
    keys: list[str] = []
    for key, value in node.items():
        keys.extend(_flat_keys(value, f"{prefix}{key}.") if isinstance(value, dict) else [f"{prefix}{key}"])
    return keys


def _suggest(flag: str, keys: list[str], aliases: dict[str, str]) -> str | None:
    """The override an old --flag most likely maps to: an explicit alias, else a key matching its dotted or leaf name."""
    name = flag.lstrip("-").split("=", 1)[0].replace("-", "_")
    negated = name.startswith("no_") and name not in aliases
    if negated:
        name = name[3:]
    if name in aliases:
        target = aliases[name]
        return target if "=" in target else target + ("=false" if negated else "=<value>")
    for key in keys:
        if key.replace(".", "_") == name or key.rsplit(".", 1)[-1] == name or key == f"{name}.enabled":
            return key + ("=false" if negated else "=<value>")
    return None


def reject_argparse_flags(
    config_name: str, schema: type, argv: list[str], aliases: dict[str, str] | None = None
) -> None:
    """Exit with a pointer to the key=value form when argv carries an argparse-style --flag."""
    bad = [a for a in argv if a.startswith("--") and a.split("=", 1)[0] not in _HYDRA_FLAGS]
    if not bad:
        return
    keys = _flat_keys(config_dict(OmegaConf.structured(schema), resolve=False))
    lines = [f"{config_name}: takes Hydra overrides (key=value), not --flags."]
    for flag in bad:
        key = _suggest(flag, keys, aliases or {})
        lines.append(f"  {flag}: " + (f"use {key}" if key else "no matching key"))
    lines.append(f"Every key and its default: python -m eval.runners.{config_name} --cfg job")
    print("\n".join(lines), file=sys.stderr)
    raise SystemExit(2)


def compose_spec(config_name: str, schema: type[T], overrides: list[str] | None = None) -> T:
    """Compose conf/eval/<config_name>.yaml with overrides into a schema instance, without running anything."""
    _register(config_name, schema)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=EVAL_CONF_DIR, version_base=None):
        return _to_spec(hydra.compose(config_name=config_name, overrides=overrides or []), schema)


def hydra_entry(
    config_name: str,
    schema: type[T],
    run: Callable[[T], None],
    aliases: dict[str, str] | None = None,
    validate: Callable[[T], object] | None = None,
) -> Callable[[], None]:
    """A console entry point that composes conf/eval/<config_name>.yaml and calls run with the typed config.

    validate runs first; a ValueError it raises exits with its message instead of a traceback. aliases maps
    a retired argparse flag (underscored, without dashes) to the key that replaced it, for the error message
    a leftover --flag gets.
    """
    _register(config_name, schema)

    @hydra.main(config_path=EVAL_CONF_DIR, config_name=config_name, version_base=None)
    def _main(cfg: DictConfig) -> None:
        try:
            spec = _to_spec(cfg, schema)
        except MissingMandatoryValue as exc:
            raise SystemExit(f"{config_name}: {exc.full_key} is required (pass {exc.full_key}=<value>)") from None
        if validate is not None:
            try:
                validate(spec)
            except ValueError as exc:
                raise SystemExit(f"{config_name}: {exc}") from None
        run(spec)

    def main() -> None:
        reject_argparse_flags(config_name, schema, sys.argv[1:], aliases)
        _main()

    return main
