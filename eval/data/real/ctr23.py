"""OpenML-CTR23 (suite 353; Fischer, Feurer, Bischl 2023): curated tabular regression, one dataset per run.

Joint log-loss protocol adapted from The Neural Testbed (Osband et al.,
NeurIPS 2022): the dyadic_knn mask makes the query set two tight clusters in
feature space, where a joint predictive can beat independent marginals; the
random mask is a negative control where the copula must not hurt.

Data: sklearn.datasets.fetch_openml(data_id=...) into CTR23_DATA_DIR (official
openml.org; downloaded once, then read from that cache). CTR23_TASKS pins the
suite's 35 tasks (task id, data id, target) as listed by the OpenML API on
2026-11-05; the audit re-checks them against the API.

Episode contract
----------------
- Preprocessing, as TabICL's own (tabicl/_sklearn/preprocessing.py):
  categorical/object/bool columns are ordinal-encoded (sorted categories,
  missing = -1), numeric columns mean-imputed. Both use features only, over
  the whole dataset. Rows with a missing target are dropped.
- Atoms: a dataset whose most common target value holds more than
  max_atom_frac (5 %) of its rows is excluded before any episode (a continuous
  NLL rewards a density spike on a point mass, so a quantile marginal would
  win spuriously). The rule tests for a real atom: targets that are merely
  rounded (prices to $100) repeat values but carry no large atom, and stay;
  discrete targets (wine quality, abalone rings) have one and go. It replaces
  the first draft's "> 5 % of rows tied with another row", which excluded 28
  of 35 datasets on rounding alone (decided with the user on 2026-11-05,
  before any CTR23 score was seen). load raises Ctr23Excluded; the audit
  lists every excluded dataset with both fractions.
- Rows and split from default_rng([seed, ep_i]). dyadic_knn: two anchors; the
  ceil(N/2) rows nearest anchor 1 in standardized feature space are queries,
  then the floor(N/2) rows nearest anchor 2 among the rest (anchor 2 is drawn
  outside the first cluster); the P context rows are drawn uniformly from the
  remaining rows. random: P + N rows uniformly, split uniformly.
- Features: all columns after preprocessing (z-scored per episode by the
  generic builder). CI cluster: the dataset (per-dataset delta, then a
  Wilcoxon signed-rank test across datasets in paired_compare across=[...]);
  within a dataset, episodes are the resampling unit. No folds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from copula_inter.artifacts import atomic_write, canonical_digest
from copula_inter.paths import CTR23_DATA_DIR
from eval.data.real.base import RealDraw, check_choice, random_split

if TYPE_CHECKING:
    from eval.runners.eval_args import Ctr23Spec

__all__ = [
    "CLEANING_VERSION",
    "CTR23_TASKS",
    "MASKS",
    "Ctr23Data",
    "Ctr23DataError",
    "Ctr23Excluded",
    "draw",
    "dyadic_knn_partition",
    "dyadic_knn_split",
    "encode_features",
    "episode_shape",
    "fingerprint",
    "label",
    "load",
    "openml_folds",
    "parse_split_arff",
    "random_partition",
    "atom_fraction",
    "tie_fraction",
    "validate",
]

NAME = "ctr23"
CLEANING_VERSION = 1
FIXED_SHAPE = True
HAS_CI = True
MASKS = ("dyadic_knn", "random")
SUITE_ID = 353
# data_id -> (task_id, target column), OpenML suite 353 as of 2026-11-05.
CTR23_TASKS: dict[int, tuple[int, str]] = {
    44956: (361234, "rings"),
    44957: (361235, "sound_pressure"),
    44958: (361236, "verification.time"),
    44959: (361237, "strength"),
    44963: (361241, "RMSD"),
    44964: (361242, "critical_temp"),
    44965: (361243, "latitude"),
    44966: (361244, "c_class_flares"),
    44969: (361247, "gt_compressor_decay_state_coefficient"),
    44971: (361249, "quality"),
    44972: (361250, "quality"),
    44973: (361251, "stab"),
    44974: (361252, "utime"),
    44975: (361253, "energy_total"),
    44976: (361254, "V22"),
    44977: (361255, "medianHouseValue"),
    44978: (361256, "usr"),
    44979: (361257, "price"),
    44980: (361258, "y"),
    44981: (361259, "thetadd6"),
    44983: (361260, "SALE_PRC"),
    44984: (361261, "wage"),
    44987: (361264, "counts_for_sons_current_occupation"),
    44989: (361266, "price"),
    44990: (361267, "total"),
    44992: (361268, "FPS"),
    44993: (361269, "whrswk"),
    45012: (361272, "wage_eur"),
    41021: (361616, "RS"),
    44960: (361617, "heating_load"),
    44962: (361618, "area"),
    44967: (361619, "G3"),
    44970: (361621, "LC50"),
    44994: (361622, "Price"),
    45402: (361623, "ln_votes_pop"),
}


class Ctr23DataError(ValueError):
    """A CTR23 dataset cannot be fetched or does not have the task's target."""


class Ctr23Excluded(Ctr23DataError):
    """The dataset fails the pre-declared atom rule and is excluded."""


@dataclass(frozen=True)
class Ctr23Data:
    """One dataset after preprocessing: features X (n, d), standardized Xs for the kNN, target y (n,)."""

    data_id: int
    name: str
    X: np.ndarray
    Xs: np.ndarray
    y: np.ndarray
    n_raw: int
    n_missing_target: int
    tie_frac: float
    atom_frac: float
    identity: dict[str, Any]
    # OpenML row index of each kept row (rows with a missing target are absent); maps the task splits.
    row_ids: np.ndarray


def atom_fraction(y: np.ndarray) -> float:
    """Fraction of rows holding the most common target value (the largest point mass)."""
    if not len(y):
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / len(y))


def tie_fraction(y: np.ndarray) -> float:
    """Fraction of rows whose target value is shared with at least one other row."""
    _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    return float((counts[inverse] > 1).mean()) if len(y) else 0.0


def encode_features(frame: pd.DataFrame) -> np.ndarray:
    """TabICL's preprocessing: ordinal-encode categorical columns (missing = -1), mean-impute numeric ones."""
    cols: list[np.ndarray] = []
    for name in frame.columns:
        col = frame[name]
        if isinstance(col.dtype, pd.CategoricalDtype) or col.dtype == object or col.dtype == bool:
            values = col.astype("object")
            cats = sorted({str(v) for v in values.dropna()})
            lookup = {c: float(i) for i, c in enumerate(cats)}
            cols.append(np.array([-1.0 if pd.isna(v) else lookup[str(v)] for v in values], dtype=np.float64))
        else:
            num = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64)
            fill = float(np.nanmean(num)) if np.isfinite(num).any() else 0.0
            cols.append(np.where(np.isnan(num), fill, num))
    return np.stack(cols, axis=1) if cols else np.zeros((len(frame), 0))


def _standardize(X: np.ndarray) -> np.ndarray:
    std = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(std > 1e-12, std, 1.0)


def load(cfg: Ctr23Spec, verbose: bool = True) -> Ctr23Data:
    """Fetch (once), preprocess and atom-check one CTR23 dataset; raises Ctr23Excluded when the atom rule fails."""
    from sklearn.datasets import fetch_openml

    data_id = int(cfg.data_id)
    _, target = CTR23_TASKS[data_id]
    root = Path(cfg.data_dir) if cfg.data_dir else CTR23_DATA_DIR
    root.mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml(data_id=data_id, data_home=str(root), as_frame=True, target_column=None, parser="auto")
    frame: pd.DataFrame = bunch.frame
    if target not in frame.columns:
        raise Ctr23DataError(f"data_id {data_id}: target {target!r} not among columns {list(frame.columns)[:10]}...")
    y_all = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=np.float64)
    keep = np.isfinite(y_all)
    feats = frame.drop(columns=[target]).loc[keep].reset_index(drop=True)
    y = y_all[keep]
    X = encode_features(feats)
    digest = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
    name = str(bunch.details.get("name", data_id))
    identity = {
        "cleaning_version": CLEANING_VERSION,
        "data_id": data_id,
        "target": target,
        "name": name,
        "version": str(bunch.details.get("version", "")),
        "content_sha256": digest,
    }
    tie, atom = tie_fraction(y), atom_fraction(y)
    data = Ctr23Data(
        data_id, name, X, _standardize(X), y, len(y_all), int((~keep).sum()), tie, atom, identity, np.flatnonzero(keep)
    )
    cache = root / f"ctr23_{data_id}_clean_v{CLEANING_VERSION}_{canonical_digest(identity)[:16]}.npz"
    if not cache.is_file():

        def _write(tmp: str) -> None:
            with open(tmp, "wb") as fh:
                np.savez(fh, X=X, y=y, identity=np.array(json.dumps(identity)))

        atomic_write(cache, _write)
    if verbose:
        print(
            f"  [ctr23] {data.name} (data_id {data_id}): {len(y)} rows x {X.shape[1]} features, "
            f"{data.n_missing_target} missing targets dropped, largest atom {atom:.3f} (tie fraction {tie:.3f})"
        )
    if atom > cfg.max_atom_frac:
        raise Ctr23Excluded(
            f"CTR23 {data.name} (data_id {data_id}) is excluded: its most common target value holds "
            f"{atom:.1%} of the rows (> max_atom_frac = {cfg.max_atom_frac:.0%})"
        )
    return data


def dyadic_knn_split(
    rng: np.random.Generator, Xs: np.ndarray, n_context: int, n_query: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """(context, query, anchors): two kNN clusters of ceil(N/2) and floor(N/2) rows, context uniform from the rest."""
    n = Xs.shape[0]
    n1 = -(-n_query // 2)
    a1 = int(rng.integers(n))
    d1 = ((Xs - Xs[a1]) ** 2).sum(axis=1)
    q1 = np.argsort(d1, kind="stable")[:n1]
    taken = np.zeros(n, dtype=bool)
    taken[q1] = True
    free = np.flatnonzero(~taken)
    a2 = int(free[rng.integers(len(free))])
    d2 = ((Xs - Xs[a2]) ** 2).sum(axis=1)
    d2[taken] = np.inf
    q2 = np.argsort(d2, kind="stable")[: n_query - n1]
    query = np.concatenate([q1, q2])
    taken[q2] = True
    context = rng.choice(np.flatnonzero(~taken), size=n_context, replace=False)
    return context, query, (a1, a2)


def draw(data: Ctr23Data, rng: np.random.Generator, cfg: Ctr23Spec, *, seed: int, ep_i: int) -> RealDraw | None:
    """Episode ep_i: rows and split by the mask."""
    if len(data.y) < cfg.n_context + cfg.n_query:
        print(f"  [ctr23] {data.name}: {len(data.y)} rows < P + N; skipping")
        return None
    anchors: tuple[int, int] | None = None
    if cfg.mask == "random":
        ctx, qry = random_split(rng, np.arange(len(data.y)), cfg.n_context, cfg.n_query)
    else:
        ctx, qry, anchors = dyadic_knn_split(rng, data.Xs, cfg.n_context, cfg.n_query)
    meta = {
        "data_id": data.data_id,
        "name": data.name,
        "fold": None,
        "context_rows": ctx.tolist(),
        "query_rows": qry.tolist(),
        "anchor_rows": None if anchors is None else list(anchors),
        "cluster": f"ep{ep_i}",
    }
    return RealDraw(data.X[ctx], data.y[ctx], data.X[qry], data.y[qry], meta)


def validate(cfg: Ctr23Spec) -> None:
    """Raise ValueError for an out-of-range real.ctr23 field."""
    check_choice("real.ctr23.mask", cfg.mask, MASKS)
    check_choice("real.ctr23.data_id", str(cfg.data_id), tuple(str(k) for k in CTR23_TASKS))
    if not 0.0 <= cfg.max_atom_frac < 1.0:
        raise ValueError("real.ctr23.max_atom_frac must lie in [0, 1)")


def episode_shape(cfg: Ctr23Spec) -> tuple[int, int]:
    """(P, N) of every episode."""
    return int(cfg.n_context), int(cfg.n_query)


def fingerprint(data: Ctr23Data, cfg: Ctr23Spec) -> dict[str, Any]:
    """Everything that defines CTR23 episode k: the dataset content, the atom rule and every draw setting."""
    return {
        "data": data.identity,
        "max_atom_frac": float(cfg.max_atom_frac),
        "n_context": int(cfg.n_context),
        "n_query": int(cfg.n_query),
        "mask": cfg.mask,
    }


def label(meta: dict[str, Any]) -> str:
    """One-line episode description for the per-episode print."""
    return f"CTR23 {meta['name']} P={meta['P']} N={meta['N']}"


# ---------------------------------------------------------------------------
# Full-dataset protocol (eval/runners/ctr23_full_eval.py): the task's official
# 10-fold cross-validation, context = the training folds, queries = the test fold.
# ---------------------------------------------------------------------------

SPLITS_URL = "https://www.openml.org/api_splits/get/{task}/Task_{task}_splits.arff"


def parse_split_arff(text: str) -> list[tuple[str, int, int, int]]:
    """(type, rowid, repeat, fold) rows of an OpenML task split ARFF (type is TRAIN or TEST)."""
    rows: list[tuple[str, int, int, int]] = []
    in_data = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if not in_data:
            in_data = line.lower() == "@data"
            continue
        kind, rowid, repeat, fold = (v.strip() for v in line.split(","))
        rows.append((kind, int(float(rowid)), int(float(repeat)), int(float(fold))))
    if not rows:
        raise Ctr23DataError("split ARFF holds no @data rows")
    return rows


def openml_folds(data: Ctr23Data, data_dir: str | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(train, test)] per fold of the task's official split (repeat 0), as indices into data's kept rows.

    The ARFF is fetched once from openml.org into the CTR23 data dir. Rows whose
    target is missing (absent from data) are dropped from both sides. Raises
    unless every fold's train and test are disjoint and the test folds cover
    every kept row exactly once.
    """
    task, _ = CTR23_TASKS[data.data_id]
    root = Path(data_dir) if data_dir else CTR23_DATA_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"ctr23_{data.data_id}_task{task}_splits.arff"
    if not path.is_file():
        from urllib.request import urlopen

        with urlopen(SPLITS_URL.format(task=task), timeout=120) as resp:
            body: bytes = resp.read()

        def _write(tmp: str) -> None:
            with open(tmp, "wb") as fh:
                fh.write(body)

        atomic_write(path, _write)
    rows = [r for r in parse_split_arff(path.read_text()) if r[2] == 0]
    pos = {int(r): i for i, r in enumerate(data.row_ids)}
    n_folds = 1 + max(r[3] for r in rows)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    covered = np.zeros(len(data.y), dtype=np.int64)
    for k in range(n_folds):
        train = np.array(sorted(pos[r[1]] for r in rows if r[3] == k and r[0] == "TRAIN" and r[1] in pos))
        test = np.array(sorted(pos[r[1]] for r in rows if r[3] == k and r[0] == "TEST" and r[1] in pos))
        if not len(train) or not len(test) or np.intersect1d(train, test).size:
            raise Ctr23DataError(f"{data.name}: fold {k} of task {task} is empty or overlaps")
        covered[test] += 1
        folds.append((train, test))
    if not np.all(covered == 1):
        raise Ctr23DataError(f"{data.name}: the test folds of task {task} do not cover every row exactly once")
    return folds


def random_partition(rng: np.random.Generator, rows: np.ndarray, size: int) -> list[np.ndarray]:
    """rows shuffled and cut into ceil(len / size) near-equal chunks (sizes differ by at most one)."""
    n_chunks = max(1, -(-len(rows) // size))
    return [c for c in np.array_split(rng.permutation(rows), n_chunks) if len(c)]


def dyadic_knn_partition(rng: np.random.Generator, Xs: np.ndarray, rows: np.ndarray, size: int) -> list[np.ndarray]:
    """rows cut into chunks of two kNN clusters each (dyadic_knn_split's geometry), every row in exactly one chunk.

    Repeatedly: a uniform anchor among the rows left, its ceil(size/2) nearest
    rows left (standardized features Xs, indexed by row), then a second anchor
    among the rows left and its floor(size/2) nearest. The last chunk takes
    whatever remains once at most `size` rows are left.
    """
    left = np.asarray(rows).copy()
    chunks: list[np.ndarray] = []
    half = -(-size // 2)
    while len(left) > size:
        parts = []
        for k in (half, size - half):
            a = left[int(rng.integers(len(left)))]
            d2 = ((Xs[left] - Xs[a]) ** 2).sum(axis=1)
            take = np.argsort(d2, kind="stable")[:k]
            parts.append(left[take])
            left = np.delete(left, take)
        chunks.append(np.concatenate(parts))
    if len(left):
        chunks.append(left)
    return chunks
