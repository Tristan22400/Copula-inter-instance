"""independent.py — the R = I baseline: TabICL marginals with no modeled dependency."""

from __future__ import annotations

import numpy as np

__all__ = ["get_correlation"]


def get_correlation(n_test: int) -> np.ndarray:
    return np.eye(n_test)
