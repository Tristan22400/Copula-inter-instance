"""_synthetic.py — shared synthetic-GP-function helpers for experiments A and B.

Thin wrappers around ``src/data_gen.py``'s ``build_kernel_fn``/``gp_posterior``/
``sigma_to_correlation`` — no kernel math lives here, just the "draw one 1D
test function + pick sparse train points" bookkeeping both experiment scripts
need.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data_gen import _safe_cholesky, build_kernel_fn  # noqa: E402

OBS_NOISE_STD = 0.05


def sample_gp_function(kernel_name: str, lengthscale: float, n_test: int, rng: torch.Generator):
    """Draw one true function from a GP prior on a dense [0,1] grid.

    Returns:
        X_dense   : (n_test, 1) input grid.
        true_f    : (n_test,) sampled true function values.
        kernel_fn : callable(X1, X2) -> K, with `lengthscale` baked in.
    """
    X_dense = torch.linspace(0.0, 1.0, n_test).unsqueeze(-1)
    kernel_fn = build_kernel_fn(kernel_name, l=lengthscale, alpha2=1.0)
    K = kernel_fn(X_dense, X_dense) + 1e-5 * torch.eye(n_test)
    L = _safe_cholesky(K)
    z = torch.randn(n_test, generator=rng)
    true_f = (L @ z.unsqueeze(-1)).squeeze(-1)
    return X_dense, true_f, kernel_fn


def pick_train_indices(n_test: int, n_train: int, rng: np.random.Generator) -> np.ndarray:
    """Evenly-spaced-ish sparse train indices, jittered, always including the endpoints."""
    base = np.linspace(0, n_test - 1, n_train).round().astype(int)
    base = np.unique(base)
    if len(base) < n_train:
        extra = rng.choice(
            [i for i in range(n_test) if i not in base], size=n_train - len(base), replace=False
        )
        base = np.sort(np.concatenate([base, extra]))
    return base
