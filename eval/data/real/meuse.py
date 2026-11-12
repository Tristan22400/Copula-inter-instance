"""Meuse river flood plain (R package sp): 155 topsoil samples, target log(zinc ppm).

Data: meuse.csv exported from sp 2.2-4's meuse.rda (eval/data/cache/meuse/SOURCE.md),
placed by hand under MEUSE_DATA_DIR. x, y are Dutch RD coordinates (metres),
dist the normalised distance to the river, elev the relative elevation (m).

Episode contract
----------------
- Repeated K-fold (K = n_folds = 3, n_repeats = 67, so 201 episodes):
  episode ep_i is fold ep_i % K of repetition r = ep_i // K. Repetition r
  permutes the 155 points with default_rng([seed, 0x3E05E, r]) and cuts the
  permutation into K near-equal parts (np.array_split): the query set is part
  k (51 or 52 points), the context the other parts (103 or 104). Every point is
  queried exactly once per repetition, so equally often overall.
- Shapes vary by one row across folds, so the PIT runs per episode.
- Features: F0 = (x, y); F1 = (x, y, dist, elev). Target log(zinc), the
  standard choice, so the GP baselines are not handicapped by the skew.
- No CI: the episodes of a repetition partition the same 155 points, and
  repetitions reuse them, so no resampling unit is independent. Report the
  mean +- sd over episodes and say so (HAS_CI = False).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from copula_inter.artifacts import artifact_identity, atomic_write, canonical_digest
from copula_inter.paths import MEUSE_DATA_DIR
from eval.data.real.base import RealDraw, check_choice

if TYPE_CHECKING:
    from eval.runners.eval_args import MeuseSpec

__all__ = [
    "CLEANING_VERSION",
    "FEATURE_SETS",
    "N_POINTS",
    "MeuseData",
    "MeuseDataError",
    "draw",
    "episode_shape",
    "fingerprint",
    "label",
    "load",
    "repetition_parts",
    "validate",
]

NAME = "meuse"
CLEANING_VERSION = 1
FIXED_SHAPE = False
HAS_CI = False
FEATURE_SETS = ("f0", "f1")
N_POINTS = 155
_USED = ("x", "y", "zinc", "dist", "elev")


class MeuseDataError(ValueError):
    """meuse.csv is missing or not sp's meuse data frame."""


@dataclass(frozen=True)
class MeuseData:
    """The 155 points: columns x, y, dist, elev (raw units) and log_zinc."""

    x: np.ndarray
    y: np.ndarray
    dist: np.ndarray
    elev: np.ndarray
    log_zinc: np.ndarray
    identity: dict[str, Any]


def load(cfg: MeuseSpec, verbose: bool = True) -> MeuseData:
    """Verify meuse.csv and cache the used columns as .npz (CLEANING_VERSION + file identity)."""
    root = Path(cfg.data_dir) if cfg.data_dir else MEUSE_DATA_DIR
    path = root / "meuse.csv"
    if not path.is_file():
        raise MeuseDataError(f"{path} does not exist; see eval/data/cache/meuse/SOURCE.md")
    identity = {"cleaning_version": CLEANING_VERSION, "file": artifact_identity(path)}
    cache = root / f"meuse_clean_v{CLEANING_VERSION}_{canonical_digest(identity)[:16]}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as f:
            return MeuseData(f["x"], f["y"], f["dist"], f["elev"], f["log_zinc"], identity)
    frame = pd.read_csv(path)
    problems: list[str] = []
    missing = [c for c in _USED if c not in frame.columns]
    if missing:
        problems.append(f"missing columns {missing}")
    elif len(frame) != N_POINTS:
        problems.append(f"{len(frame)} rows, expected {N_POINTS}")
    elif frame[list(_USED)].isna().any().any():
        problems.append("missing values in " + ", ".join(_USED))
    elif (frame["zinc"] <= 0).any():
        problems.append("non-positive zinc")
    if problems:
        raise MeuseDataError("meuse.csv verification failed: " + "; ".join(problems))
    arrays = {c: frame[c].to_numpy(dtype=np.float64) for c in ("x", "y", "dist", "elev")}
    arrays["log_zinc"] = np.log(frame["zinc"].to_numpy(dtype=np.float64))

    def _write(tmp: str) -> None:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays, identity=np.array(json.dumps(identity)))

    atomic_write(cache, _write)
    if verbose:
        print(f"  [meuse] wrote {cache}")
    return MeuseData(arrays["x"], arrays["y"], arrays["dist"], arrays["elev"], arrays["log_zinc"], identity)


def repetition_parts(seed: int, rep: int, n_folds: int) -> list[np.ndarray]:
    """The n_folds query sets of repetition rep: a seeded permutation of the points cut into near-equal parts."""
    # The constant separates this stream from the per-episode default_rng([seed, ep_i]).
    perm = np.random.default_rng([int(seed), 0x3E05E, int(rep)]).permutation(N_POINTS)
    return list(np.array_split(perm, n_folds))


def draw(data: MeuseData, rng: np.random.Generator, cfg: MeuseSpec, *, seed: int, ep_i: int) -> RealDraw | None:
    """Episode ep_i: fold ep_i % n_folds of repetition ep_i // n_folds (rng is unused: the repetition fixes the split)."""
    if ep_i >= cfg.n_folds * cfg.n_repeats:
        print(f"  [meuse] ep {ep_i}: beyond {cfg.n_folds} folds x {cfg.n_repeats} repetitions; skipping")
        return None
    rep, k = divmod(int(ep_i), cfg.n_folds)
    parts = repetition_parts(seed, rep, cfg.n_folds)
    qry = parts[k]
    ctx = np.concatenate([p for j, p in enumerate(parts) if j != k])
    cols = [data.x, data.y] if cfg.features == "f0" else [data.x, data.y, data.dist, data.elev]
    feats = np.stack(cols, axis=1)
    meta = {
        "fold": None,
        "repetition": rep,
        "cv_fold": k,
        "query_points": qry.tolist(),
        "cluster": f"rep{rep}",
    }
    return RealDraw(feats[ctx], data.log_zinc[ctx], feats[qry], data.log_zinc[qry], meta)


def validate(cfg: MeuseSpec) -> None:
    """Raise ValueError for an out-of-range real.meuse field."""
    check_choice("real.meuse.features", cfg.features, FEATURE_SETS)
    if not 2 <= cfg.n_folds <= N_POINTS // 2:
        raise ValueError(f"real.meuse.n_folds must lie in [2, {N_POINTS // 2}]")
    if cfg.n_repeats < 1:
        raise ValueError("real.meuse.n_repeats must be at least 1")


def episode_shape(cfg: MeuseSpec) -> tuple[int, int]:
    """(smallest P, smallest N): the largest fold leaves the smallest context."""
    n_small = N_POINTS // cfg.n_folds
    n_large = -(-N_POINTS // cfg.n_folds)
    return N_POINTS - n_large, n_small


def fingerprint(data: MeuseData, cfg: MeuseSpec) -> dict[str, Any]:
    """Everything that defines Meuse episode k: the file identity, the target transform and the K-fold design."""
    return {
        "data": data.identity,
        "target": "log(zinc)",
        "features": cfg.features,
        "n_folds": int(cfg.n_folds),
        "n_repeats": int(cfg.n_repeats),
    }


def label(meta: dict[str, Any]) -> str:
    """One-line episode description for the per-episode print."""
    return f"Meuse rep{meta['repetition']} fold{meta['cv_fold']} P={meta['P']} N={meta['N']}"
