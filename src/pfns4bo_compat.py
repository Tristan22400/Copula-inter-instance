"""pfns4bo_compat.py — load pfns4bo_upstream/pfns4bo/priors/hebo_prior unmodified.

The `pfns4bo_upstream` submodule (github.com/automl/PFNs4BO, added 2023) is
kept pristine — no edits, same convention as `tabicl_upstream`. Its
`priors/hebo_prior.py` was written against an older gpytorch/botorch API and
fails to import as-is against the current pinned versions in this repo's env
(gpytorch 1.15.x / botorch 0.18.x). All breakage is *upstream botorch/gpytorch
renaming or moving names it still provides elsewhere* — nothing about the
prior's math changed. This module patches those names/call-shapes back in at
import time so `hebo_prior.get_model` runs unmodified:

  1. `botorch.models.gp_regression.MIN_INFERRED_NOISE_LEVEL` moved to
     `botorch.models.utils.gpytorch_modules` (same value, 1e-4).
  2. `botorch.fit.fit_gpytorch_model` was renamed `fit_gpytorch_mll`.
  3. `pfns4bo/priors/hebo_prior.py` does `from botorch.models.transforms.input
     import *` to pull in `List`/`Optional`/`Union`/`expand_and_copy_tensor`/
     `Kumaraswamy`, which that module no longer re-exports at top level.
  4. `botorch.models.SingleTaskGP`'s positional argument order changed —
     `hebo_prior.get_model` calls `SingleTaskGP(x, y, likelihood, covar_module=...,
     input_transform=...)` (likelihood 3rd positional, matching botorch
     circa 2023); current botorch inserts `train_Yvar` as the 3rd positional
     argument instead. Wrap the constructor so the old call shape still binds
     `likelihood` to the likelihood.

Verified end-to-end (2026-07-06) against gpytorch==1.15.2, botorch==0.18.1,
torch==2.11.0 in the `multivariate-icl` conda env.
"""

from __future__ import annotations

import os
import sys
import typing

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PFNS4BO_ROOT = os.path.join(os.path.dirname(_HERE), "pfns4bo_upstream")

_patched = False


def _apply_botorch_compat_shims() -> None:
    """Patch the four moved/renamed/reordered botorch APIs described above.

    Idempotent (guarded by _patched) and additive only — every patched name
    still resolves to real, currently-supported botorch/gpytorch code; this
    never changes behavior for anything except pfns4bo_upstream's own import.
    """
    global _patched
    if _patched:
        return
    _patched = True

    import botorch.fit as botorch_fit
    import botorch.models as botorch_models
    import botorch.models.gp_regression as gp_regression
    import botorch.models.transforms.input as input_tf
    from botorch.models.transforms.utils import expand_and_copy_tensor
    from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL

    gp_regression.MIN_INFERRED_NOISE_LEVEL = MIN_INFERRED_NOISE_LEVEL
    botorch_fit.fit_gpytorch_model = botorch_fit.fit_gpytorch_mll

    input_tf.List = typing.List
    input_tf.Optional = typing.Optional
    input_tf.Union = typing.Union
    input_tf.expand_and_copy_tensor = expand_and_copy_tensor
    input_tf.Kumaraswamy = torch.distributions.Kumaraswamy

    _RealSingleTaskGP = botorch_models.SingleTaskGP

    def _compat_single_task_gp(train_X, train_Y, likelihood=None, **kwargs):
        return _RealSingleTaskGP(train_X, train_Y, likelihood=likelihood, **kwargs)

    botorch_models.SingleTaskGP = _compat_single_task_gp


def get_hebo_prior_module():
    """Import and return pfns4bo.priors.hebo_prior, patching botorch first.

    Raises a clear ImportError if the submodule wasn't checked out or if
    gpytorch/botorch aren't installed, instead of a confusing traceback deep
    inside pfns4bo_upstream.
    """
    if not os.path.isdir(_PFNS4BO_ROOT):
        raise ImportError(
            f"PFNs4BO submodule not found at {_PFNS4BO_ROOT}. "
            "Run `git submodule update --init` from the repo root."
        )
    try:
        _apply_botorch_compat_shims()
    except ImportError as e:
        raise ImportError(
            "kernel='hebo' requires gpytorch and botorch. "
            "Install them in this env with `pip install gpytorch botorch`."
        ) from e

    if _PFNS4BO_ROOT not in sys.path:
        sys.path.insert(0, _PFNS4BO_ROOT)
    from pfns4bo.priors import hebo_prior

    return hebo_prior
