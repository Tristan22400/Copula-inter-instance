"""Capabilities of each supported marginal and copula backbone."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Backend:
    name: str
    max_tier: int
    copula_backbone: bool = False
    batched_pit: str | None = None
    autoregressive: bool = False


BACKENDS = {
    spec.name: spec for spec in (
        Backend("tabicl", 3, copula_backbone=True, autoregressive=True),
        Backend("exaone", 0, batched_pit="eval.spatial.exaone_batched:exaone_run_pit_batched"),
        Backend("tabpfn", 0, batched_pit="eval.spatial.tabpfn_batched:tabpfn_run_pit_batched"),
        Backend("tabldm", 3, copula_backbone=True,
                batched_pit="eval.spatial.tabldm_batched:tabldm_run_pit_batched"),
    )
}

MARGINAL_BACKENDS = tuple(BACKENDS)
GENERIC_MARGINAL_BACKENDS = tuple(name for name, spec in BACKENDS.items() if spec.batched_pit)
TABICL_Z_TRAIN_SOURCES = ("tabicl", "tabicl_split")
Z_TRAIN_SOURCES = ("analytic", "tabicl", "tabicl_split", *GENERIC_MARGINAL_BACKENDS, "y_train")
COPULA_BACKBONES = tuple(name for name, spec in BACKENDS.items() if spec.copula_backbone)
DEFAULT_Z_TRAIN_SOURCE = "tabicl"
# Eval-side name for the exact-GP ground truth that training calls "analytic".
EVAL_Z_TRAIN_SOURCES = ("oracle", *MARGINAL_BACKENDS)


def z_train_source(cfg) -> str:
    """The configured ``data.z_train_source``, defaulting to TabICL."""
    data = cfg.get("data") if hasattr(cfg, "get") else None
    return str((data or {}).get("z_train_source", DEFAULT_Z_TRAIN_SOURCE))


def require_capability(name: str, capability: str) -> Backend:
    try:
        spec = BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"unknown backend {name!r}; choose from {tuple(BACKENDS)}") from exc
    if not getattr(spec, capability):
        raise ValueError(f"backend {name!r} does not support {capability}")
    return spec


def batched_backend_factories() -> dict[str, Callable[[], Callable]]:
    def factory(path: str):
        module, symbol = path.split(":")
        return getattr(importlib.import_module(module), symbol)

    return {
        name: (lambda path=spec.batched_pit: factory(path))
        for name, spec in BACKENDS.items() if spec.batched_pit
    }
