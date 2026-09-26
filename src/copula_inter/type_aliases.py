"""Type aliases and protocols shared across modules."""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias

import torch

Device: TypeAlias = "str | torch.device"


class HasDataConfig(Protocol):
    """A config with a data section: a Hydra DictConfig or diag_kernels.Cfg."""

    @property
    def data(self) -> Any: ...
