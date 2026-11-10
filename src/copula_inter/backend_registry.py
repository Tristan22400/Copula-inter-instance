"""Capabilities of each supported marginal and copula backbone."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from copula_inter.type_aliases import HasDataConfig


@dataclass(frozen=True)
class Backend:
    name: str
    max_tier: int
    copula_backbone: bool = False
    batched_pit: str | None = None
    autoregressive: bool = False
    # Quantile levels the Phase-A forward emits when no grid is requested
    # (linspace(1/(n+1), n/(n+1), n)); TabICL reads its own from the decoder.
    native_quantiles: int | None = None


# The three marginals: TabICL v2, PriorLabs TabPFN-3.5 (Fast checkpoint) and
# NVIDIA Kumo Tabular (small). The weights variant is part of the name, so a
# z_train_source recorded in a manifest or results cache names exact weights.
BACKENDS = {
    spec.name: spec
    for spec in (
        Backend("tabicl", 3, copula_backbone=True, autoregressive=True),
        Backend(
            "tabpfn35_fast",
            0,
            batched_pit="copula_inter.marginal.tabpfn_marginal:tabpfn_run_pit_batched",
            native_quantiles=999,
        ),
        Backend(
            "kumo_small",
            0,
            batched_pit="copula_inter.marginal.kumo_marginal:kumo_run_pit_batched",
            native_quantiles=999,
        ),
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


def z_train_source(cfg: HasDataConfig) -> str:
    """The configured ``data.z_train_source``, defaulting to TabICL."""
    data = cfg.get("data") if hasattr(cfg, "get") else None
    return str((data or {}).get("z_train_source", DEFAULT_Z_TRAIN_SOURCE))


def z_train_marginal_ckpt(cfg: HasDataConfig) -> str | None:
    """The configured ``data.z_train_marginal_ckpt``: Phase-A weights for a generic marginal backend, or None."""
    data = cfg.get("data") if hasattr(cfg, "get") else None
    ckpt = (data or {}).get("z_train_marginal_ckpt", None)
    return None if ckpt is None else str(ckpt)


def validate_z_train_source(value: str) -> None:
    """Raise ValueError if data.z_train_source is not a recognized value."""
    if value not in Z_TRAIN_SOURCES:
        raise ValueError(
            f"Unknown data.z_train_source {value!r}; expected {', '.join(repr(v) for v in Z_TRAIN_SOURCES)}."
        )


def get_backend(name: str) -> Backend:
    try:
        return BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"unknown backend {name!r}; choose from {tuple(BACKENDS)}") from exc


def require_capability(name: str, capability: str) -> Backend:
    spec = get_backend(name)
    if not getattr(spec, capability):
        raise ValueError(f"backend {name!r} does not support {capability}")
    return spec


def batched_pit(name: str) -> Callable[..., Any]:
    """Import and return ``name``'s batched PIT function (lazy: heavy optional dependencies)."""
    path = require_capability(name, "batched_pit").batched_pit
    assert path is not None
    module, symbol = path.split(":")
    return getattr(importlib.import_module(module), symbol)
