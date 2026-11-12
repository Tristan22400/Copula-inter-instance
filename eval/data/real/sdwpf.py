"""SDWPF (KDD Cup 2022) wind-farm evaluation episodes: the raw-data loader, the cleaning, and the episode contract.

Data: Zhou et al., SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting
Challenge at KDD Cup 2022 (arXiv:2208.04360; Scientific Data 2024,
doi:10.1038/s41597-024-03427-5). 134 turbines of one farm, 245 days at
10-minute resolution, 4,727,520 SCADA records, and a file of relative turbine
positions (x, y in metres). load_sdwpf reads them from a local directory
(copula_inter.paths.SDWPF_DATA_DIR by default) and never downloads anything.

Episode contract
----------------
- An episode is one timestamp (Day, Tmstamp). Its rows are the turbines valid
  at that timestamp and its target is Patv (kW). Nothing from another timestamp
  enters an episode: no lags, no history, no time-of-day features.
- A query turbine is unobserved, so its features hold only what is known
  without its SCADA record:
    F0   (x, y), the relative position;
    F1b  (x, y, u, v), the position plus along-wind u and cross-wind v: the
         positions rotated by the farm direction phi of that timestamp. phi is
         the circular mean of the context turbines' absolute direction
         (Ndir + Wdir) mod 360, never of the query turbines'. Context and query
         rows share one phi.
  phi is one value per episode, so as a column it would be constant and
  normalize_features would zero it. It is informative only through its
  interaction with position, which is what u and v encode.
- No row's Wspd, Wdir, Etmp, Itmp, Ndir, Pab*, Prtv or Patv, nor anything
  derived from them, is a feature of any row. F1b's context-only phi is the one
  exception.
- Features are z-scored by normalize_features (joint train+test, the training
  convention). That is legitimate here because every row's position is known.
- Folds: five contiguous blocks of 49 days (FOLD_DAYS). Episode ep_i belongs
  to fold ep_i % 5 and its timestamp lies inside that fold's days. The folds
  carry every cross-fitted choice (direction offset, F0 vs F1b); the
  confidence-interval cluster is the day. Every episode is reported.
- Timestamps lie on the hourly grid (Tmstamp HH:00), so two episodes are at
  least 1 h apart. Each fold's hourly candidates (those with at least P + N
  valid turbines) are permuted once per seed, and episode j of the fold owns
  the disjoint slice [j * timestamp_tries, (j + 1) * timestamp_tries) of that
  permutation: no two episodes share a timestamp, and a shard reproduces the
  unsharded stream. The episode takes the first timestamp of its slice whose
  draw passes the regime; with none, it is skipped and reported.
- Rows and split are drawn before any feature is built, from
  default_rng([seed, ep_i]), so F0 and F1b see identical episodes.
  random: P + N valid turbines uniformly, split uniformly. block: the N valid
  turbines nearest an anchor drawn uniformly among the valid ones are the
  queries; the P context turbines are drawn uniformly from the rest.
- Regime partial_load keeps a timestamp only if every selected context and
  query turbine is strictly inside (0, atom_hi_frac * rated). It is decided per
  episode on the selected set, before the context/query split matters and
  identically for every method; no row is ever filtered on its own target.
  Regime all keeps every timestamp; its NLL is unreliable because Patv has
  atoms at 0 and at rated power, where a continuous density is ill-defined.
  rated is the rated_quantile (99.9 %) quantile of cleaned Patv over all days.

Cleaning (CLEANING_VERSION, part of the cache key and the fingerprint)
----------------------------------------------------------------------
The paper's rules, applied in this order: drop rows with a missing value; drop
"unknown" rows (Patv <= 0 and Wspd > 2.5, or any Pab* > 89 deg); drop
"abnormal" rows (|Ndir| > 720 or |Wdir| > 180); clip negative Patv to 0.
Dropped rows are neither context nor query. These filters read each row's own
SCADA, so the evaluated population is "turbines reporting normal operation",
not every turbine of the farm.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from copula_inter.artifacts import artifact_identity, atomic_write, canonical_digest
from copula_inter.paths import SDWPF_DATA_DIR
from eval.data.real.base import RealDraw, block_split, check_choice, random_split

if TYPE_CHECKING:
    from eval.runners.eval_args import SdwpfSpec

__all__ = [
    "ALL_DAYS",
    "CLEANING_VERSION",
    "DIRECTION_OFFSETS",
    "FEATURE_SETS",
    "FOLD_DAYS",
    "MASKS",
    "N_FOLDS",
    "REGIMES",
    "SDWPF_EXPECTED",
    "STEPS_PER_DAY",
    "SdwpfData",
    "SdwpfDataError",
    "SdwpfExpected",
    "SdwpfTable",
    "absolute_wind_direction",
    "circular_mean_deg",
    "draw",
    "episode_shape",
    "estimate_rated_power",
    "fingerprint",
    "fold_of",
    "label",
    "load",
    "load_sdwpf",
    "validate",
    "wind_aligned_axes",
]

NAME = "sdwpf"
# Bump on any change to the cleaning rules: it keys the cleaned cache and every SDWPF fingerprint.
CLEANING_VERSION = 1
STEPS_PER_DAY = 144  # 10-minute records
STEPS_PER_HOUR = 6
N_FOLDS = 5
_FOLD_LEN = 49
FOLD_DAYS = tuple((k * _FOLD_LEN + 1, (k + 1) * _FOLD_LEN) for k in range(N_FOLDS))
ALL_DAYS = (1, N_FOLDS * _FOLD_LEN)
MASKS = ("random", "block")
FEATURE_SETS = ("f0", "f1b")
REGIMES = ("partial_load", "all")
# Offsets added to (Ndir + Wdir) before the mod 360; the audit cross-fits one per fold.
DIRECTION_OFFSETS = (0, 90, 180, -90)
# Every episode has P and N fixed by the config, so the PIT can batch.
FIXED_SHAPE = True
# Episodes are distinct timestamps; the day-cluster bootstrap is valid.
HAS_CI = True

RAW_COLUMNS = (
    "TurbID",
    "Day",
    "Tmstamp",
    "Wspd",
    "Wdir",
    "Etmp",
    "Itmp",
    "Ndir",
    "Pab1",
    "Pab2",
    "Pab3",
    "Prtv",
    "Patv",
)
LOCATION_COLUMNS = ("TurbID", "x", "y")
CLEANING_RULES = ("missing", "unknown", "abnormal")
# The SCADA fields kept after cleaning: the target, the two angles of F1b's direction, and Wspd for the audit.
_KEPT_FIELDS = ("patv", "wspd", "wdir", "ndir")


class SdwpfDataError(ValueError):
    """The SDWPF files are missing, ambiguous, or not the dataset the paper describes."""


@dataclass(frozen=True)
class SdwpfExpected:
    """What load_sdwpf asserts about the raw files; the defaults are the paper's figures."""

    n_turbines: int = 134
    n_days: int = 245
    n_records: int = 4_727_520


SDWPF_EXPECTED = SdwpfExpected()


@dataclass(frozen=True)
class SdwpfTable:
    """The cleaned farm on a dense (time step, turbine) grid; a dropped record is NaN in every field.

    Step t is (Day - 1) * STEPS_PER_DAY + minutes(Tmstamp) // 10. Column j is
    turbine turb_ids[j], at (x[j], y[j]) metres.
    """

    turb_ids: np.ndarray  # (n_turbines,) int64
    x: np.ndarray  # (n_turbines,) float64
    y: np.ndarray  # (n_turbines,) float64
    patv: np.ndarray  # (n_steps, n_turbines) float32, kW, negatives clipped to 0
    wspd: np.ndarray  # (n_steps, n_turbines) float32, m/s
    wdir: np.ndarray  # (n_steps, n_turbines) float32, deg, relative to the nacelle
    ndir: np.ndarray  # (n_steps, n_turbines) float32, deg, cumulative nacelle yaw
    n_raw: int
    # Rows each rule removes when applied in CLEANING_RULES order.
    dropped: dict[str, int]
    # Rows each rule matches on its own; the rules overlap, so these need not sum to the dropped total.
    flagged: dict[str, int]
    n_clipped: int
    # Cleaning version and the raw files' identities: the data part of every SDWPF fingerprint.
    identity: dict[str, Any]

    @property
    def valid(self) -> np.ndarray:
        """(n_steps, n_turbines) bool: the record survived cleaning."""
        return ~np.isnan(self.patv)

    @property
    def n_steps(self) -> int:
        return int(self.patv.shape[0])

    def day_of_step(self) -> np.ndarray:
        """(n_steps,) the 1-based Day of each step."""
        return np.arange(self.n_steps) // STEPS_PER_DAY + 1

    def period_steps(self, days: tuple[int, int]) -> np.ndarray:
        """The steps whose Day lies in the inclusive range days."""
        day = self.day_of_step()
        return np.flatnonzero((day >= days[0]) & (day <= days[1]))


def _csv_header(path: Path) -> list[str]:
    with open(path, encoding="utf-8-sig") as fh:
        return [c.strip() for c in fh.readline().split(",")]


def find_raw_files(data_dir: str | os.PathLike[str]) -> tuple[Path, Path]:
    """The SCADA CSV and the turbine-location CSV in data_dir, recognized by their headers, not their names."""
    root = Path(data_dir)
    if not root.is_dir():
        raise SdwpfDataError(
            f"SDWPF directory {root} does not exist. Place the SCADA CSV (columns {', '.join(RAW_COLUMNS)}) "
            f"and the turbine-location CSV (columns {', '.join(LOCATION_COLUMNS)}) there."
        )
    csvs = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    scada = [p for p in csvs if set(RAW_COLUMNS) <= set(_csv_header(p))]
    locations = [p for p in csvs if set(LOCATION_COLUMNS) <= set(_csv_header(p)) and p not in scada]
    if len(scada) != 1 or len(locations) != 1:
        raise SdwpfDataError(
            f"{root} must hold exactly one SCADA CSV (columns {', '.join(RAW_COLUMNS)}) and one location CSV "
            f"(columns {', '.join(LOCATION_COLUMNS)}); found SCADA {[p.name for p in scada]}, "
            f"locations {[p.name for p in locations]} among {[p.name for p in csvs]}."
        )
    return scada[0], locations[0]


def _tmstamp_slot(tmstamp: pd.Series) -> np.ndarray:
    """'HH:MM' -> the 10-minute slot of the day, 0..143; raises on anything else."""
    parts = tmstamp.astype(str).str.strip().str.split(":", expand=True)
    if parts.shape[1] < 2:
        raise SdwpfDataError(f"Tmstamp is not HH:MM, e.g. {tmstamp.iloc[0]!r}")
    minutes = parts[0].astype(int).to_numpy() * 60 + parts[1].astype(int).to_numpy()
    if np.any(minutes % 10) or np.any((minutes < 0) | (minutes >= 24 * 60)):
        raise SdwpfDataError("Tmstamp holds times that are not on the 10-minute grid of one day")
    return minutes // 10


def _verify(raw: pd.DataFrame, slot: np.ndarray, loc: pd.DataFrame, expected: SdwpfExpected) -> None:
    """Assert the raw files are the paper's dataset; collect every failure before raising."""
    problems: list[str] = []
    if len(raw) != expected.n_records:
        problems.append(f"{len(raw)} records, expected {expected.n_records}")
    turbs = np.unique(raw["TurbID"].to_numpy())
    if len(turbs) != expected.n_turbines:
        problems.append(f"{len(turbs)} turbines, expected {expected.n_turbines}")
    days = np.unique(raw["Day"].to_numpy())
    if not np.array_equal(days, np.arange(1, expected.n_days + 1)):
        problems.append(f"days {days.min()}..{days.max()} ({len(days)} distinct), expected 1..{expected.n_days}")
    keys = pd.DataFrame({"t": raw["TurbID"].to_numpy(), "d": raw["Day"].to_numpy(), "s": slot})
    n_dup = int(keys.duplicated().sum())
    if n_dup:
        problems.append(f"{n_dup} duplicate (TurbID, Day, Tmstamp) records")
    loc_ids = loc["TurbID"].to_numpy()
    if len(np.unique(loc_ids)) != len(loc_ids):
        problems.append("the location file lists a TurbID twice")
    missing = sorted(set(turbs.tolist()) - set(loc_ids.tolist()))
    if missing:
        problems.append(f"no position for TurbID {missing}")
    if not np.all(np.isfinite(loc[["x", "y"]].to_numpy(dtype=float))):
        problems.append("non-finite turbine positions")
    if problems:
        raise SdwpfDataError("SDWPF verification failed: " + "; ".join(problems))


def _clean(raw: pd.DataFrame) -> tuple[np.ndarray, dict[str, int], dict[str, int], int]:
    """Keep mask, rows dropped per rule (sequential), rows flagged per rule (alone), and the clip count."""
    missing = raw[list(RAW_COLUMNS)].isna().any(axis=1).to_numpy()
    patv, wspd = raw["Patv"].to_numpy(), raw["Wspd"].to_numpy()
    pab = raw[["Pab1", "Pab2", "Pab3"]].to_numpy()
    # NaN compares False, so a missing row is flagged by "missing" alone.
    unknown = ((patv <= 0) & (wspd > 2.5)) | (pab > 89).any(axis=1)
    abnormal = (np.abs(raw["Ndir"].to_numpy()) > 720) | (np.abs(raw["Wdir"].to_numpy()) > 180)
    keep = np.ones(len(raw), dtype=bool)
    dropped: dict[str, int] = {}
    for name, rule in zip(CLEANING_RULES, (missing, unknown, abnormal)):
        dropped[name] = int((keep & rule).sum())
        keep &= ~rule
    flagged = {name: int(rule.sum()) for name, rule in zip(CLEANING_RULES, (missing, unknown, abnormal))}
    n_clipped = int((keep & (patv < 0)).sum())
    return keep, dropped, flagged, n_clipped


def _cache_path(root: Path, key: str) -> Path:
    return root / f"sdwpf_clean_v{CLEANING_VERSION}_{key[:16]}.npz"


def load_sdwpf(
    data_dir: str | os.PathLike[str] = SDWPF_DATA_DIR,
    *,
    expected: SdwpfExpected = SDWPF_EXPECTED,
    verbose: bool = True,
) -> SdwpfTable:
    """Verify, clean and grid the SDWPF files in data_dir, caching the result as .npz next to them.

    The cache is keyed by CLEANING_VERSION, the expected figures and both raw
    files' content identity, so new raw bytes or new rules rebuild it.
    Verification runs on the raw files before cleaning and raises
    SdwpfDataError on any mismatch with expected.
    """
    root = Path(data_dir)
    scada_path, loc_path = find_raw_files(root)
    if verbose:
        print(f"  [sdwpf] hashing {scada_path.name} and {loc_path.name}", flush=True)
    identity = {
        "cleaning_version": CLEANING_VERSION,
        "expected": asdict(expected),
        "scada": artifact_identity(scada_path),
        "locations": artifact_identity(loc_path),
    }
    cache = _cache_path(root, canonical_digest(identity))
    if cache.is_file():
        if verbose:
            print(f"  [sdwpf] cleaned cache {cache.name}", flush=True)
        with np.load(cache, allow_pickle=False) as z:
            stats = json.loads(str(z["stats"]))
            return SdwpfTable(
                turb_ids=z["turb_ids"],
                x=z["x"],
                y=z["y"],
                **{f: z[f] for f in _KEPT_FIELDS},
                n_raw=int(stats["n_raw"]),
                dropped=stats["dropped"],
                flagged=stats["flagged"],
                n_clipped=int(stats["n_clipped"]),
                identity=identity,
            )

    if verbose:
        print(f"  [sdwpf] reading {scada_path}", flush=True)
    raw = pd.read_csv(scada_path, usecols=list(RAW_COLUMNS), dtype={"Tmstamp": str})
    # The official location file starts with a byte-order mark.
    loc = pd.read_csv(loc_path, encoding="utf-8-sig", skipinitialspace=True)
    loc.columns = [c.strip() for c in loc.columns]
    loc = loc[list(LOCATION_COLUMNS)]
    slot = _tmstamp_slot(raw["Tmstamp"])
    _verify(raw, slot, loc, expected)
    keep, dropped, flagged, n_clipped = _clean(raw)

    loc = loc.sort_values("TurbID").reset_index(drop=True)
    turb_ids = loc["TurbID"].to_numpy(dtype=np.int64)
    # Positions of turbines absent from the SCADA file would be dead columns.
    turb_ids = turb_ids[np.isin(turb_ids, raw["TurbID"].to_numpy())]
    loc = loc.set_index("TurbID").loc[turb_ids]
    col = np.searchsorted(turb_ids, raw["TurbID"].to_numpy())
    step = (raw["Day"].to_numpy() - 1) * STEPS_PER_DAY + slot
    n_steps = expected.n_days * STEPS_PER_DAY
    grids: dict[str, np.ndarray] = {}
    for field, column in zip(_KEPT_FIELDS, ("Patv", "Wspd", "Wdir", "Ndir")):
        grid = np.full((n_steps, len(turb_ids)), np.nan, dtype=np.float32)
        values = raw[column].to_numpy(dtype=np.float64)
        if field == "patv":
            values = np.maximum(values, 0.0)  # the paper sets negative power to 0
        grid[step[keep], col[keep]] = values[keep]
        grids[field] = grid
    table = SdwpfTable(
        turb_ids=turb_ids,
        x=loc["x"].to_numpy(dtype=np.float64),
        y=loc["y"].to_numpy(dtype=np.float64),
        **grids,
        n_raw=len(raw),
        dropped=dropped,
        flagged=flagged,
        n_clipped=n_clipped,
        identity=identity,
    )
    stats = json.dumps({"n_raw": len(raw), "dropped": dropped, "flagged": flagged, "n_clipped": n_clipped})

    def _write(tmp: str) -> None:
        with open(tmp, "wb") as fh:
            arrays: dict[str, Any] = {
                "turb_ids": table.turb_ids,
                "x": table.x,
                "y": table.y,
                "stats": np.array(stats),
                **grids,
            }
            np.savez(fh, **arrays)

    atomic_write(cache, _write)
    if verbose:
        print(f"  [sdwpf] wrote {cache}", flush=True)
    return table


def estimate_rated_power(table: SdwpfTable, days: tuple[int, int] = ALL_DAYS, quantile: float = 0.999) -> float:
    """The turbines' rated power in kW, estimated as a high quantile of cleaned Patv over the given days.

    The paper does not state it. It is a dataset-level constant, not a score,
    so it uses every day.
    """
    patv = table.patv[table.period_steps(days)]
    return float(np.quantile(patv[~np.isnan(patv)], quantile))


def absolute_wind_direction(ndir: np.ndarray, wdir: np.ndarray) -> np.ndarray:
    """Per-turbine compass direction the wind blows from, degrees in [0, 360): (Ndir + Wdir) mod 360.

    Wdir is the wind's angle to the nacelle and Ndir the cumulative nacelle yaw,
    so their sum is an absolute angle only if Ndir is a compass heading; the
    audit checks that hypothesis. NaN stays NaN.
    """
    return np.mod(np.asarray(ndir, dtype=np.float64) + np.asarray(wdir, dtype=np.float64), 360.0)


def circular_mean_deg(angles_deg: np.ndarray, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """NaN-ignoring circular mean in [0, 360) and mean resultant length in [0, 1] along axis."""
    rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    s = np.nanmean(np.sin(rad), axis=axis)
    c = np.nanmean(np.cos(rad), axis=axis)
    return np.mod(np.rad2deg(np.arctan2(s, c)), 360.0), np.hypot(s, c)


def wind_aligned_axes(x: np.ndarray, y: np.ndarray, phi_deg: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate positions (x east, y north) into along-wind u (larger = further downwind) and cross-wind v.

    phi_deg is the compass direction the wind blows from, so the downwind unit
    vector is d = (-sin phi, -cos phi). The map is a proper rotation: distances
    in (u, v) equal those in (x, y). phi_deg may broadcast against x and y.
    """
    phi = np.deg2rad(np.asarray(phi_deg, dtype=np.float64))
    s, c = np.sin(phi), np.cos(phi)
    u = -np.asarray(x) * s - np.asarray(y) * c
    v = np.asarray(x) * c - np.asarray(y) * s
    return u, v


# ---------------------------------------------------------------------------
# Episodes (the eval/data/real contract)
# ---------------------------------------------------------------------------


def fold_of(ep_i: int) -> int:
    """The day-block fold episode ep_i is drawn from."""
    return int(ep_i) % N_FOLDS


@dataclass(frozen=True)
class SdwpfData:
    """The cleaned farm plus what the episode draw needs: the rated power and each fold's hourly candidates."""

    table: SdwpfTable
    rated: float
    # Per fold: hourly steps with at least P + N valid turbines, ascending.
    candidates: tuple[np.ndarray, ...]


def _data_dir(cfg: SdwpfSpec) -> Path:
    return Path(cfg.data_dir) if cfg.data_dir else SDWPF_DATA_DIR


def candidate_steps(table: SdwpfTable, days: tuple[int, int], n_rows: int) -> np.ndarray:
    """The hourly steps of the inclusive day range with at least n_rows valid turbines (validity is cleaning, not target)."""
    steps = table.period_steps(days)
    steps = steps[steps % STEPS_PER_HOUR == 0]
    return steps[table.valid[steps].sum(axis=1) >= n_rows]


def load(cfg: SdwpfSpec, verbose: bool = True) -> SdwpfData:
    """Verify, clean (cached) and index the farm for cfg's P + N."""
    table = load_sdwpf(_data_dir(cfg), verbose=verbose)
    rated = estimate_rated_power(table, quantile=cfg.rated_quantile)
    n_rows = cfg.n_context + cfg.n_query
    candidates = tuple(candidate_steps(table, days, n_rows) for days in FOLD_DAYS)
    if verbose:
        print(
            f"  [sdwpf] rated power (q{cfg.rated_quantile:g} of cleaned Patv, all days) = {rated:.1f} kW; "
            f"partial load = (0, {cfg.atom_hi_frac * rated:.1f}) kW"
        )
        print(
            "  [sdwpf] hourly candidates with >= "
            f"{n_rows} valid turbines per fold: " + ", ".join(str(len(c)) for c in candidates)
        )
    return SdwpfData(table=table, rated=rated, candidates=candidates)


def _fold_permutation(data: SdwpfData, seed: int, fold: int) -> np.ndarray:
    """The seed's fixed order of fold's candidate timestamps (independent of the episode index)."""
    # The constant separates this stream from the per-episode default_rng([seed, ep_i]).
    return np.random.default_rng([int(seed), 0x5D3F, int(fold)]).permutation(data.candidates[fold])


def _select_rows(
    rng: np.random.Generator, table: SdwpfTable, step: int, cfg: SdwpfSpec
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """(context, query, anchor) turbine columns at step, drawn among the valid turbines."""
    pool = np.flatnonzero(table.valid[step])
    if cfg.mask == "random":
        ctx, qry = random_split(rng, pool, cfg.n_context, cfg.n_query)
        return ctx, qry, None
    coords = np.stack([table.x[pool], table.y[pool]], axis=1)
    ctx, qry, anchor = block_split(rng, pool, coords, cfg.n_context, cfg.n_query)
    return ctx, qry, anchor


def farm_direction(table: SdwpfTable, step: int, columns: np.ndarray, offset_deg: float) -> float:
    """phi: circular mean over `columns` (the context turbines only) of (Ndir + Wdir + offset) mod 360."""
    phi_i = np.mod(absolute_wind_direction(table.ndir[step, columns], table.wdir[step, columns]) + offset_deg, 360.0)
    mean, _ = circular_mean_deg(phi_i)
    return float(mean)


def features(
    table: SdwpfTable, step: int, ctx: np.ndarray, qry: np.ndarray, cfg: SdwpfSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Raw (P, d), (N, d) features: F0 (x, y) or F1b (x, y, u, v) with phi from the context rows only."""
    cols = np.concatenate([ctx, qry])
    x, y = table.x[cols], table.y[cols]
    if cfg.features == "f0":
        feats = np.stack([x, y], axis=1)
    else:
        phi = farm_direction(table, step, ctx, float(cfg.direction_offset_deg))
        u, v = wind_aligned_axes(x, y, phi)
        feats = np.stack([x, y, u, v], axis=1)
    return feats[: len(ctx)], feats[len(ctx) :]


def draw(data: SdwpfData, rng: np.random.Generator, cfg: SdwpfSpec, *, seed: int, ep_i: int) -> RealDraw | None:
    """Episode ep_i: the first timestamp of its slice of the fold's permutation whose draw passes the regime."""
    table = data.table
    fold = fold_of(ep_i)
    j = int(ep_i) // N_FOLDS
    tries = int(cfg.timestamp_tries)
    slice_ = _fold_permutation(data, seed, fold)[j * tries : (j + 1) * tries]
    hi = cfg.atom_hi_frac * data.rated
    for attempt, step in enumerate(slice_):
        step = int(step)
        ctx, qry, anchor = _select_rows(rng, table, step, cfg)
        cols = np.concatenate([ctx, qry])
        patv = table.patv[step, cols].astype(np.float64)
        if cfg.regime == "partial_load" and not bool(np.all((patv > 0.0) & (patv < hi))):
            continue
        x_tr, x_te = features(table, step, ctx, qry, cfg)
        slot = step % STEPS_PER_DAY
        meta = {
            "fold": fold,
            "day": int(step // STEPS_PER_DAY + 1),
            "tmstamp": f"{slot // 6:02d}:{(slot % 6) * 10:02d}",
            "step": step,
            "context_turbines": table.turb_ids[ctx].tolist(),
            "query_turbines": table.turb_ids[qry].tolist(),
            "anchor_turbine": None if anchor is None else int(table.turb_ids[anchor]),
            "attempt": attempt,
            # The CI cluster (day-block bootstrap).
            "cluster": f"day{int(step // STEPS_PER_DAY + 1)}",
        }
        return RealDraw(x_tr, patv[: len(ctx)], x_te, patv[len(ctx) :], meta)
    if len(slice_) == 0:
        print(
            f"  [sdwpf] ep {ep_i}: fold {fold} has only {len(data.candidates[fold])} candidate timestamps "
            f"({len(data.candidates[fold]) // tries} episodes at timestamp_tries={tries}); skipping"
        )
    else:
        print(f"  [sdwpf] ep {ep_i}: no timestamp of its {len(slice_)} passed regime={cfg.regime}; skipping")
    return None


def validate(cfg: SdwpfSpec) -> None:
    """Raise ValueError for an out-of-range real.sdwpf field."""
    check_choice("real.sdwpf.mask", cfg.mask, MASKS)
    check_choice("real.sdwpf.features", cfg.features, FEATURE_SETS)
    check_choice("real.sdwpf.regime", cfg.regime, REGIMES)
    check_choice(
        "real.sdwpf.direction_offset_deg", str(cfg.direction_offset_deg), tuple(str(o) for o in DIRECTION_OFFSETS)
    )
    if not 0.0 < cfg.atom_hi_frac <= 1.0:
        raise ValueError("real.sdwpf.atom_hi_frac must lie in (0, 1]")
    if not 0.5 < cfg.rated_quantile <= 1.0:
        raise ValueError("real.sdwpf.rated_quantile must lie in (0.5, 1]")
    if cfg.timestamp_tries < 1:
        raise ValueError("real.sdwpf.timestamp_tries must be at least 1")


def episode_shape(cfg: SdwpfSpec) -> tuple[int, int]:
    """(P, N) of every episode."""
    return int(cfg.n_context), int(cfg.n_query)


def fingerprint(data: SdwpfData, cfg: SdwpfSpec) -> dict[str, Any]:
    """Everything that defines SDWPF episode k: cleaned data identity, rated power, folds, and every draw setting."""
    return {
        "data": data.table.identity,
        "rated": round(float(data.rated), 6),
        "fold_days": [list(d) for d in FOLD_DAYS],
        "n_context": int(cfg.n_context),
        "n_query": int(cfg.n_query),
        "mask": cfg.mask,
        "features": cfg.features,
        "regime": cfg.regime,
        "direction_offset_deg": int(cfg.direction_offset_deg) if cfg.features == "f1b" else None,
        "atom_hi_frac": float(cfg.atom_hi_frac),
        "rated_quantile": float(cfg.rated_quantile),
        "timestamp_tries": int(cfg.timestamp_tries),
    }


def label(meta: dict[str, Any]) -> str:
    """One-line episode description for the per-episode print."""
    return f"SDWPF fold{meta['fold']} day{meta['day']} {meta['tmstamp']} P={meta['P']} N={meta['N']}"
