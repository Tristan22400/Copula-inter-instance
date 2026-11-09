"""California Housing (Pace & Barry 1997, via sklearn.datasets.fetch_california_housing): rows = census block groups.

Target: median house value in $100,000. Features: the 8 sklearn columns
(MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude,
Longitude). fetch_california_housing downloads sklearn's official copy into
CALIFORNIA_DATA_DIR once and reads it from there afterwards.

Episode contract
----------------
- Censoring: the survey top-codes the target at $500,001 (5.00001), a point
  mass. Every row at that top-code (the maximum, verified to be >= censor_cap
  = 5.0) is dropped once, at the dataset level, before any episode is drawn.
  Rows at exactly 5.0 ($500,000, a plausible real value just below the code)
  are kept; the audit reports their count. The evaluated population is
  therefore "block groups below the top-code".
- Rows and split are drawn from default_rng([seed, ep_i]) before features are
  built, so F0 and F1 see identical episodes. random: P + N rows uniformly
  over the state, split uniformly. block: the N rows nearest a uniform anchor
  in (Latitude, Longitude) degrees are the queries, the P context rows are
  drawn uniformly from the rest (eval/datasets/spatial_housing.py's split).
- Features: F0 = all 8 columns; F1 = without Latitude and Longitude (the
  ablation: the copula's gain should shrink without location). The block
  mask selects on location in both, so the episodes stay paired.
- CI cluster: the block anchor (block) or the episode (random). No
  score-based choice, so no folds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from copula_inter.artifacts import artifact_identity, atomic_write, canonical_digest
from copula_inter.paths import CALIFORNIA_DATA_DIR
from eval.data.real.base import RealDraw, block_split, check_choice, random_split

if TYPE_CHECKING:
    from eval.runners.eval_args import CaliforniaSpec

__all__ = [
    "CLEANING_VERSION",
    "FEATURE_NAMES",
    "FEATURE_SETS",
    "MASKS",
    "CaliforniaData",
    "CaliforniaDataError",
    "clean",
    "draw",
    "episode_shape",
    "fingerprint",
    "label",
    "load",
    "validate",
]

NAME = "california"
CLEANING_VERSION = 1
FIXED_SHAPE = True
HAS_CI = True
MASKS = ("random", "block")
FEATURE_SETS = ("f0", "f1")
FEATURE_NAMES = (
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
)
N_RAW = 20_640
_LAT, _LON = FEATURE_NAMES.index("Latitude"), FEATURE_NAMES.index("Longitude")


class CaliforniaDataError(ValueError):
    """The California Housing copy is not the one sklearn distributes, or its censoring is not a single cap."""


@dataclass(frozen=True)
class CaliforniaData:
    """The uncensored rows: features (n, 8) in FEATURE_NAMES order and target (n,), plus the cleaning counts."""

    X: np.ndarray
    y: np.ndarray
    n_raw: int
    n_censored: int
    cap_value: float
    identity: dict[str, Any]


def clean(X: np.ndarray, y: np.ndarray, censor_cap: float) -> tuple[np.ndarray, np.ndarray, int, float]:
    """(X, y, n_censored, top-code) without the rows at the top-code (the maximum); raises unless it is >= censor_cap."""
    cap = float(y.max())
    if cap < censor_cap:
        raise CaliforniaDataError(f"maximum target {cap} is below censor_cap={censor_cap}: no top-code found")
    censored = y == cap
    return X[~censored], y[~censored], int(censored.sum()), cap


def load(cfg: CaliforniaSpec, verbose: bool = True) -> CaliforniaData:
    """Fetch (once) and verify sklearn's copy, drop the censored rows, and cache the result as .npz."""
    from sklearn.datasets import fetch_california_housing

    root = Path(cfg.data_dir) if cfg.data_dir else CALIFORNIA_DATA_DIR
    root.mkdir(parents=True, exist_ok=True)
    bunch = fetch_california_housing(data_home=str(root))
    raw_files = sorted(p for p in root.glob("cal_housing*") if p.is_file())
    identity = {
        "cleaning_version": CLEANING_VERSION,
        "censor_cap": float(cfg.censor_cap),
        "files": {p.name: artifact_identity(p) for p in raw_files},
    }
    cache = root / f"california_clean_v{CLEANING_VERSION}_{canonical_digest(identity)[:16]}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as f:
            stats = json.loads(str(f["stats"]))
            return CaliforniaData(f["X"], f["y"], stats["n_raw"], stats["n_censored"], stats["cap_value"], identity)
    X = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target, dtype=np.float64)
    if tuple(bunch.feature_names) != FEATURE_NAMES or X.shape != (N_RAW, len(FEATURE_NAMES)):
        raise CaliforniaDataError(f"unexpected shape {X.shape} or columns {list(bunch.feature_names)}")
    Xc, yc, n_censored, cap_value = clean(X, y, cfg.censor_cap)
    stats = {"n_raw": len(y), "n_censored": n_censored, "cap_value": cap_value}

    def _write(tmp: str) -> None:
        with open(tmp, "wb") as fh:
            np.savez(fh, X=Xc, y=yc, stats=np.array(json.dumps(stats)))

    atomic_write(cache, _write)
    if verbose:
        print(f"  [california] {len(y)} rows, {n_censored} censored at {cap_value} dropped; wrote {cache}")
    return CaliforniaData(Xc, yc, len(y), n_censored, cap_value, identity)


def draw(
    data: CaliforniaData, rng: np.random.Generator, cfg: CaliforniaSpec, *, seed: int, ep_i: int
) -> RealDraw | None:
    """Episode ep_i: rows and split by the mask, then the feature set's columns."""
    pool = np.arange(len(data.y))
    anchor: int | None = None
    if cfg.mask == "random":
        ctx, qry = random_split(rng, pool, cfg.n_context, cfg.n_query)
    else:
        ctx, qry, anchor = block_split(rng, pool, data.X[:, [_LAT, _LON]], cfg.n_context, cfg.n_query)
    cols = list(range(len(FEATURE_NAMES))) if cfg.features == "f0" else [c for c in range(8) if c not in (_LAT, _LON)]
    meta = {
        "fold": None,
        "context_rows": ctx.tolist(),
        "query_rows": qry.tolist(),
        "anchor_row": anchor,
        "cluster": f"anchor{anchor}" if anchor is not None else f"ep{ep_i}",
    }
    return RealDraw(data.X[ctx][:, cols], data.y[ctx], data.X[qry][:, cols], data.y[qry], meta)


def validate(cfg: CaliforniaSpec) -> None:
    """Raise ValueError for an out-of-range real.california field."""
    check_choice("real.california.mask", cfg.mask, MASKS)
    check_choice("real.california.features", cfg.features, FEATURE_SETS)
    if not cfg.censor_cap > 0:
        raise ValueError("real.california.censor_cap must be positive")


def episode_shape(cfg: CaliforniaSpec) -> tuple[int, int]:
    """(P, N) of every episode."""
    return int(cfg.n_context), int(cfg.n_query)


def fingerprint(data: CaliforniaData, cfg: CaliforniaSpec) -> dict[str, Any]:
    """Everything that defines California episode k: the data identity, the censoring and every draw setting."""
    return {
        "data": data.identity,
        "n_rows": len(data.y),
        "n_context": int(cfg.n_context),
        "n_query": int(cfg.n_query),
        "mask": cfg.mask,
        "features": cfg.features,
    }


def label(meta: dict[str, Any]) -> str:
    """One-line episode description for the per-episode print."""
    where = f"anchor{meta['anchor_row']}" if meta.get("anchor_row") is not None else "random"
    return f"California {where} P={meta['P']} N={meta['N']}"
