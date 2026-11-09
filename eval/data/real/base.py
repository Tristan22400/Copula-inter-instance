"""The contract every eval/data/real/<name>.py module meets, and the row-split helpers they share.

A dataset module exposes:
    NAME, CLEANING_VERSION, FIXED_SHAPE (P and N fixed by the config, so the PIT batches);
    load(cfg) -> data: read and verify the raw files, cache a cleaned .npz keyed
        by CLEANING_VERSION and the raw files' identity;
    draw(data, rng, cfg, *, seed, ep_i) -> RealDraw | None: one episode in raw
        units from rng = default_rng([seed, ep_i]); None skips the episode;
    fingerprint(data, cfg) -> dict: everything that defines the episodes;
    label(meta) -> str: a one-line episode description for the tables;
    validate(cfg): raise ValueError for an out-of-range real.<name> field;
    episode_shape(cfg) -> (P, N): the smallest context and query sizes an episode has.
Meta keys every module sets: "cluster" (the CI unit), "fold" (the grouped
cross-fitting fold, or None when the dataset needs no score-based choice).
HAS_CI is False only for a dataset whose episodes share rows so heavily that a
bootstrap CI is meaningless (Meuse); paired_compare then reports mean +- sd.

Rows and the context/query split are drawn before any feature is built, so the
feature-set variants of one dataset see identical episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["RealDraw", "block_split", "check_choice", "random_split"]


def check_choice(key: str, value: str | None, choices: tuple[str, ...]) -> None:
    """Raise ValueError unless value is one of choices (the message eval_args' check_choice gives)."""
    if value not in choices:
        raise ValueError(f"{key}={value!r} is not one of {sorted(choices)}")


@dataclass
class RealDraw:
    """One episode in raw units: features (P, d) / (N, d), targets (P,) / (N,), and JSON-able metadata.

    meta must identify the episode (rows, split, group) and nothing that varies
    across feature sets, so paired runs can assert identical meta.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    meta: dict[str, Any]


def random_split(
    rng: np.random.Generator, pool: np.ndarray, n_context: int, n_query: int
) -> tuple[np.ndarray, np.ndarray]:
    """(context, query) index arrays: n_context + n_query rows of pool drawn uniformly, split uniformly."""
    sel = rng.choice(pool, size=n_context + n_query, replace=False)
    return sel[:n_context], sel[n_context:]


def block_split(
    rng: np.random.Generator, pool: np.ndarray, coords: np.ndarray, n_context: int, n_query: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """(context, query, anchor): the n_query rows of pool nearest a uniform anchor are the queries.

    coords is (len(pool), k), aligned with pool. The context is drawn uniformly
    from the remaining rows. Ties in distance break by position in pool.
    """
    a = int(rng.integers(len(pool)))
    d2 = ((coords - coords[a]) ** 2).sum(axis=1)
    order = np.argsort(d2, kind="stable")
    query = pool[order[:n_query]]
    rest = pool[order[n_query:]]
    context = rng.choice(rest, size=n_context, replace=False)
    return context, query, int(pool[a])
