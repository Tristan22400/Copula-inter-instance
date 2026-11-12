"""Real-data benchmark episodes for eval_checkpoint (real.dataset=<name>): the registry and the generic builder.

Each dataset is a module eval/data/real/<name>.py meeting the contract in
eval/data/real/base.py. The builder draws episode ep_i from
default_rng([seed, ep_i]), z-scores its features with normalize_features, and
hands it to eval/data/episode_packaging.py for the marginal PIT, the
autoregressive chain and standardize_y, exactly as ERA5 does.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import numpy as np
import torch

from copula_inter.marginal.pit import normalize_features
from eval.data.episode_packaging import PitSettings, RawEpisode, check_pit_settings, package_pit_episodes

__all__ = ["REAL_DATASETS", "build_real_eval_episodes", "dataset_module", "real_episode_fingerprint"]

# real.dataset choices; each names eval/data/real/<name>.py and the real.<name> sub-config.
REAL_DATASETS = ("sdwpf", "kaust", "emnist", "california", "meuse", "ctr23")


def dataset_module(name: str) -> ModuleType:
    """The eval/data/real/<name>.py module."""
    if name not in REAL_DATASETS:
        raise ValueError(f"unknown real.dataset {name!r}; choose one of {REAL_DATASETS}")
    return importlib.import_module(f"eval.data.real.{name}")


def build_real_eval_episodes(
    name: str,
    cfg: Any,
    data: Any,
    n_episodes: int,
    settings: PitSettings,
    *,
    seed: int,
    offset: int = 0,
    pit_batch: int = 8,
    verbose: bool = True,
) -> list[dict]:
    """PIT'd episodes with global indices offset .. offset + n_episodes - 1; skipped draws are absent.

    Each episode dict is what build_era5_eval_episodes returns, with real_meta
    (ep_i, dataset, P, N and the module's draw meta) in place of era5_meta.
    """
    mod = dataset_module(name)
    check_pit_settings(settings, name)
    raw: list[RawEpisode] = []
    for local_i in range(n_episodes):
        ep_i = offset + local_i
        drawn = mod.draw(data, np.random.default_rng([int(seed), int(ep_i)]), cfg, seed=seed, ep_i=ep_i)
        if drawn is None:
            continue
        x_tr, x_te = normalize_features(drawn.x_train, drawn.x_test)
        raw.append(
            RawEpisode(
                ep_i=ep_i,
                x_norm_train=torch.from_numpy(x_tr.astype(np.float32)),
                x_norm_test=torch.from_numpy(x_te.astype(np.float32)),
                y_train=torch.from_numpy(np.asarray(drawn.y_train, dtype=np.float32)),
                y_test=torch.from_numpy(np.asarray(drawn.y_test, dtype=np.float32)),
                meta=drawn.meta,
            )
        )
    if verbose:
        print(f"  [{name}] drew {len(raw)}/{n_episodes} episodes")

    def _real_meta(r: RawEpisode, P: int, N: int) -> dict[str, Any]:
        return {"ep_i": r.ep_i, "dataset": name, "P": P, "N": N, **r.meta}

    return package_pit_episodes(
        raw,
        settings,
        seed=seed,
        group=pit_batch if mod.FIXED_SHAPE else 1,
        meta_key="real_meta",
        make_meta=_real_meta,
        tag=name,
        verbose=verbose,
    )


def real_episode_fingerprint(
    name: str, cfg: Any, data: Any, *, k_folds: int, marginal: str, standardize_y: bool
) -> dict[str, Any]:
    """Digest of what defines real-data episode k (dataset, draw settings, marginal) for the baseline cache."""
    return {
        "source": f"real-{name}",
        "dataset": dataset_module(name).fingerprint(data, cfg),
        "pit_k_folds": int(k_folds),
        "pit_marginal": marginal,
        "standardize_y": bool(standardize_y),
    }
