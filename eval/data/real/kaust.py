"""KAUST 2022 spatial-statistics competition, sub-competitions 1a and 1b: square-window episodes on the unit square.

Data: Abdulah, Alamri, Ltaief, Sun, Keyes, Genton, "The Second Competition on
Spatial Statistics for Large Datasets", Journal of Data Science 20(4), 2022
(doi:10.6339/22-JDS1076); files from the KAUST repository record
doi:10.25781/KAUST-4ADYZ, placed by hand under KAUST_DATA_DIR. Only the
training files of 1a and 1b are read; 2a/2b (space-time), 3a/3b (bivariate)
and the competition's test files are never used. Held-out episodes are built
from the training files.

Variants (real.kaust.variant = "<sub-competition>_<dataset>"):
    1a_1, 1b_1  deterministic nonstationary mean plus iid N(0, tau^2) noise:
                once the marginal has the mean, residuals are independent, so
                this is the negative control (the copula must not hurt);
    1a_2, 1b_2  zero-mean GP with a nonstationary Matern covariance: the main
                result.
1a files hold 90,000 points, 1b files 900,000, as the paper states (1b_1_train
has no trailing newline, which is why a line count shows 899,999).

Episode contract
----------------
- Folds: five vertical strips x in [k/5, (k+1)/5). Episode ep_i belongs to
  strip ep_i % 5 and its window lies inside that strip. No score-based choice
  is made on KAUST, so the strips only guarantee the episodes of one fold never
  touch another strip.
- The window is a square of side 1 / window_cells at a uniform position inside
  the strip; window_tries positions are drawn from default_rng([seed, ep_i])
  until one holds at least P + N points, else the episode is skipped.
- CI cluster: the cell of the window_cells x window_cells grid holding the
  window's centre (window_cells is a multiple of 5, so cells never straddle a
  strip). Windows in one cell overlap; distinct cells overlap only at edges.
- Rows and split are drawn among the window's points before features are
  built. random: P + N points uniformly, split uniformly. block: the N points
  nearest an anchor drawn uniformly among the window's points are the queries,
  the P context points are drawn uniformly from the rest.
- Features: the coordinates (x, y); the files have no covariates.
- Target z is continuous: no atom rule.

Cleaning (CLEANING_VERSION): none beyond verification (header, row count,
unique ids, coordinates in [0, 1], finite targets); nothing is dropped.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from copula_inter.artifacts import artifact_identity, atomic_write, canonical_digest
from copula_inter.paths import KAUST_DATA_DIR
from eval.data.real.base import RealDraw, block_split, check_choice, random_split

if TYPE_CHECKING:
    from eval.runners.eval_args import KaustSpec

__all__ = [
    "CLEANING_VERSION",
    "EXPECTED_ROWS",
    "MASKS",
    "N_STRIPS",
    "VARIANTS",
    "KaustData",
    "KaustDataError",
    "draw",
    "episode_shape",
    "fingerprint",
    "label",
    "load",
    "load_kaust",
    "validate",
]

NAME = "kaust"
CLEANING_VERSION = 1
FIXED_SHAPE = True
HAS_CI = True
N_STRIPS = 5
MASKS = ("random", "block")
VARIANTS = ("1a_1", "1a_2", "1b_1", "1b_2")
# Data rows of each training file as distributed (all four match the paper).
EXPECTED_ROWS = {"1a_1": 90_000, "1a_2": 90_000, "1b_1": 900_000, "1b_2": 900_000}
COLUMNS = ("id_train", "x", "y", "z")


class KaustDataError(ValueError):
    """A KAUST training file is missing or not the file the competition distributed."""


@dataclass(frozen=True)
class KaustData:
    """One variant's training points, sorted by x so a strip or window is a contiguous slice."""

    variant: str
    ids: np.ndarray  # (n,) int64, id_train
    x: np.ndarray  # (n,) float64, ascending
    y: np.ndarray  # (n,) float64
    z: np.ndarray  # (n,) float64
    identity: dict[str, Any]

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    def window(self, x0: float, y0: float, side: float) -> np.ndarray:
        """Indices of the points in [x0, x0 + side) x [y0, y0 + side), ascending."""
        lo, hi = np.searchsorted(self.x, [x0, x0 + side], side="left")
        idx = np.arange(lo, hi)
        yy = self.y[idx]
        return idx[(yy >= y0) & (yy < y0 + side)]


def _data_dir(cfg: KaustSpec) -> Path:
    return Path(cfg.data_dir) if cfg.data_dir else KAUST_DATA_DIR


def _verify(frame: pd.DataFrame, variant: str, expected_rows: int) -> None:
    problems: list[str] = []
    if tuple(frame.columns) != COLUMNS:
        problems.append(f"columns {list(frame.columns)}, expected {list(COLUMNS)}")
    else:
        if len(frame) != expected_rows:
            problems.append(f"{len(frame)} rows, expected {expected_rows}")
        if frame["id_train"].duplicated().any():
            problems.append("duplicate id_train")
        xy = frame[["x", "y"]].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(xy)) or xy.min() < 0.0 or xy.max() > 1.0:
            problems.append("coordinates outside [0, 1] or non-finite")
        if not np.all(np.isfinite(frame["z"].to_numpy(dtype=np.float64))):
            problems.append("non-finite z")
    if problems:
        raise KaustDataError(f"KAUST {variant}_train.csv verification failed: " + "; ".join(problems))


def load_kaust(
    data_dir: str | os.PathLike[str], variant: str, *, expected_rows: int | None = None, verbose: bool = True
) -> KaustData:
    """Verify one <variant>_train.csv and cache it as .npz (keyed by CLEANING_VERSION and the file identity).

    expected_rows defaults to EXPECTED_ROWS[variant] (tests pass their own).
    """
    n_expected = EXPECTED_ROWS[variant] if expected_rows is None else int(expected_rows)
    root = Path(data_dir)
    path = root / f"{variant}_train.csv"
    if not path.is_file():
        raise KaustDataError(
            f"{path} does not exist. Place the KAUST 2022 training file {variant}_train.csv "
            "(columns id_train, x, y, z) there; see eval/data/cache/kaust2022/SOURCE.md."
        )
    identity = {
        "cleaning_version": CLEANING_VERSION,
        "variant": variant,
        "expected_rows": n_expected,
        "file": artifact_identity(path),
    }
    cache = root / f"kaust_{variant}_clean_v{CLEANING_VERSION}_{canonical_digest(identity)[:16]}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as f:
            return KaustData(variant, f["ids"], f["x"], f["y"], f["z"], identity)
    if verbose:
        print(f"  [kaust] reading {path}", flush=True)
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip() for c in frame.columns]
    _verify(frame, variant, n_expected)
    order = np.argsort(frame["x"].to_numpy(dtype=np.float64), kind="stable")
    arrays = {
        "ids": frame["id_train"].to_numpy(dtype=np.int64)[order],
        "x": frame["x"].to_numpy(dtype=np.float64)[order],
        "y": frame["y"].to_numpy(dtype=np.float64)[order],
        "z": frame["z"].to_numpy(dtype=np.float64)[order],
    }

    def _write(tmp: str) -> None:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays, identity=np.array(json.dumps(identity)))

    atomic_write(cache, _write)
    if verbose:
        print(f"  [kaust] wrote {cache}", flush=True)
    return KaustData(variant, arrays["ids"], arrays["x"], arrays["y"], arrays["z"], identity)


def load(cfg: KaustSpec, verbose: bool = True) -> KaustData:
    """The verified variant named by cfg."""
    data = load_kaust(_data_dir(cfg), cfg.variant, verbose=verbose)
    if verbose:
        side = 1.0 / cfg.window_cells
        print(
            f"  [kaust] {cfg.variant}: {data.n} points; window side {side:.4f} "
            f"(~{data.n * side * side:.0f} points per window, P + N = {cfg.n_context + cfg.n_query})"
        )
    return data


def strip_of(ep_i: int) -> int:
    """The strip (fold) episode ep_i is drawn from."""
    return int(ep_i) % N_STRIPS


def draw(data: KaustData, rng: np.random.Generator, cfg: KaustSpec, *, seed: int, ep_i: int) -> RealDraw | None:
    """Episode ep_i: a window inside strip ep_i % 5 holding P + N points, then the mask's split."""
    strip = strip_of(ep_i)
    side = 1.0 / cfg.window_cells
    lo = strip / N_STRIPS
    n_rows = cfg.n_context + cfg.n_query
    for attempt in range(cfg.window_tries):
        x0 = float(rng.uniform(lo, lo + 1.0 / N_STRIPS - side))
        y0 = float(rng.uniform(0.0, 1.0 - side))
        pool = data.window(x0, y0, side)
        if len(pool) < n_rows:
            continue
        anchor: int | None = None
        if cfg.mask == "random":
            ctx, qry = random_split(rng, pool, cfg.n_context, cfg.n_query)
        else:
            coords = np.stack([data.x[pool], data.y[pool]], axis=1)
            ctx, qry, anchor = block_split(rng, pool, coords, cfg.n_context, cfg.n_query)
        cx, cy = x0 + side / 2.0, y0 + side / 2.0
        cell = (int(cx * cfg.window_cells), int(cy * cfg.window_cells))
        feats = np.stack([data.x, data.y], axis=1)
        meta = {
            "variant": data.variant,
            "fold": strip,
            "window": [round(x0, 6), round(y0, 6), round(side, 6)],
            "n_window": int(len(pool)),
            "attempt": attempt,
            "context_ids": data.ids[ctx].tolist(),
            "query_ids": data.ids[qry].tolist(),
            "anchor_id": None if anchor is None else int(data.ids[anchor]),
            "cluster": f"cell{cell[0]}_{cell[1]}",
        }
        return RealDraw(feats[ctx], data.z[ctx], feats[qry], data.z[qry], meta)
    print(f"  [kaust] ep {ep_i}: no window of {cfg.window_tries} held {n_rows} points; skipping")
    return None


def validate(cfg: KaustSpec) -> None:
    """Raise ValueError for an out-of-range real.kaust field."""
    check_choice("real.kaust.variant", cfg.variant, VARIANTS)
    check_choice("real.kaust.mask", cfg.mask, MASKS)
    if cfg.window_cells < N_STRIPS or cfg.window_cells % N_STRIPS:
        raise ValueError(f"real.kaust.window_cells must be a positive multiple of {N_STRIPS} (the strips)")
    if cfg.window_tries < 1:
        raise ValueError("real.kaust.window_tries must be at least 1")


def episode_shape(cfg: KaustSpec) -> tuple[int, int]:
    """(P, N) of every episode."""
    return int(cfg.n_context), int(cfg.n_query)


def fingerprint(data: KaustData, cfg: KaustSpec) -> dict[str, Any]:
    """Everything that defines KAUST episode k: the file identity, the strips and every draw setting."""
    return {
        "data": data.identity,
        "n_strips": N_STRIPS,
        "n_context": int(cfg.n_context),
        "n_query": int(cfg.n_query),
        "mask": cfg.mask,
        "window_cells": int(cfg.window_cells),
        "window_tries": int(cfg.window_tries),
    }


def label(meta: dict[str, Any]) -> str:
    """One-line episode description for the per-episode print."""
    return f"KAUST {meta['variant']} strip{meta['fold']} {meta['cluster']} P={meta['P']} N={meta['N']}"
