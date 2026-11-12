"""EMNIST balanced: one handwritten character per episode, rows = pixels, target = dequantized intensity.

Data: Cohen, Afshar, Tapson, van Schaik, "EMNIST: an extension of MNIST to
handwritten letters" (arXiv:1702.05373); the NIST idx files of the balanced
split (112,800 train / 18,800 test images, 47 classes), placed by hand under
EMNIST_DATA_DIR and parsed with numpy. The stored images are transposed; the
loader flips them upright.

Episode contract
----------------
- Episode ep_i is image perm[ep_i] of the split, perm a fixed permutation per
  seed and split, so every episode is a different image (ep_i beyond the split
  is skipped). Reported runs use the test split; debugging uses train.
- Rows are the 784 pixels; features (row, col). random: P + N pixels
  uniformly, split uniformly. block: the N pixels nearest an anchor pixel are
  the queries, the P context pixels are drawn uniformly from the rest.
- Atoms: pixels are integers 0..255 with a large atom at 0, so targets are
  uniformly dequantized, y = (pixel + U(0, 1)) / 256 in (0, 1), with U drawn
  from the episode's own default_rng([seed, ep_i]) after the split. The NLL of
  y bounds the discrete NLL: bits/pixel = (NLL_y + ln 256) / ln 2 (bits_per_pixel).
- CI cluster: the image (one per episode). No score-based choice, so no folds.

Cleaning (CLEANING_VERSION): none beyond verification (idx magic numbers, image
and label counts, 28 x 28, 47 balanced classes).
"""

from __future__ import annotations

import gzip
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from copula_inter.artifacts import artifact_identity, atomic_write, canonical_digest
from copula_inter.paths import EMNIST_DATA_DIR
from eval.data.real.base import RealDraw, block_split, check_choice, random_split

if TYPE_CHECKING:
    from eval.runners.eval_args import EmnistSpec

__all__ = [
    "CLEANING_VERSION",
    "EXPECTED",
    "MASKS",
    "SPLITS",
    "EmnistData",
    "EmnistDataError",
    "bits_per_pixel",
    "dequantize",
    "draw",
    "episode_shape",
    "fingerprint",
    "label",
    "load",
    "load_emnist",
    "read_idx",
    "validate",
]

NAME = "emnist"
CLEANING_VERSION = 1
FIXED_SHAPE = True
HAS_CI = True
MASKS = ("random", "block")
SPLITS = ("test", "train")
SIDE = 28
N_CLASSES = 47
# Images per split in the balanced release.
EXPECTED = {"train": 112_800, "test": 18_800}
_IMAGE_MAGIC, _LABEL_MAGIC = 2051, 2049


class EmnistDataError(ValueError):
    """An EMNIST idx file is missing or not the balanced release."""


@dataclass(frozen=True)
class EmnistData:
    """One split: upright uint8 images (n, 28, 28), labels (n,) and the label -> character map."""

    split: str
    images: np.ndarray
    labels: np.ndarray
    chars: tuple[str, ...]
    identity: dict[str, Any]


def read_idx(path: Path, magic: int) -> np.ndarray:
    """A gzipped big-endian idx file as a uint8 array shaped by its header; raises on a wrong magic number."""
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    head = np.frombuffer(raw[:4], dtype=">u4")[0]
    if int(head) != magic:
        raise EmnistDataError(f"{path.name}: magic {int(head)}, expected {magic}")
    ndim = magic & 0xFF
    dims = np.frombuffer(raw[4 : 4 + 4 * ndim], dtype=">u4").astype(np.int64)
    data = np.frombuffer(raw[4 + 4 * ndim :], dtype=np.uint8)
    if data.size != int(np.prod(dims)):
        raise EmnistDataError(f"{path.name}: {data.size} bytes of data, header says {dims.tolist()}")
    return data.reshape(tuple(int(d) for d in dims))


def _paths(root: Path, split: str) -> tuple[Path, Path, Path]:
    return (
        root / f"emnist-balanced-{split}-images-idx3-ubyte.gz",
        root / f"emnist-balanced-{split}-labels-idx1-ubyte.gz",
        root / "emnist-balanced-mapping.txt",
    )


def load_emnist(
    data_dir: str | os.PathLike[str], split: str, *, expected_n: int | None = None, verbose: bool = True
) -> EmnistData:
    """Verify one split's idx files and cache the upright images as .npz (CLEANING_VERSION + file identities).

    expected_n defaults to EXPECTED[split] (tests pass their own).
    """
    n_expected = EXPECTED[split] if expected_n is None else int(expected_n)
    root = Path(data_dir)
    img_path, lab_path, map_path = _paths(root, split)
    for p in (img_path, lab_path, map_path):
        if not p.is_file():
            raise EmnistDataError(f"{p} does not exist; see eval/data/cache/emnist/SOURCE.md")
    identity = {
        "cleaning_version": CLEANING_VERSION,
        "split": split,
        "expected_n": n_expected,
        "images": artifact_identity(img_path),
        "labels": artifact_identity(lab_path),
        "mapping": artifact_identity(map_path),
    }
    mapping = [line.split() for line in map_path.read_text().splitlines() if line.strip()]
    chars = tuple(chr(int(code)) for _, code in sorted(mapping, key=lambda m: int(m[0])))
    cache = root / f"emnist_{split}_clean_v{CLEANING_VERSION}_{canonical_digest(identity)[:16]}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as f:
            return EmnistData(split, f["images"], f["labels"], chars, identity)
    images = read_idx(img_path, _IMAGE_MAGIC)
    labels = read_idx(lab_path, _LABEL_MAGIC)
    problems: list[str] = []
    if images.shape != (n_expected, SIDE, SIDE):
        problems.append(f"images {images.shape}, expected {(n_expected, SIDE, SIDE)}")
    if labels.shape != (n_expected,):
        problems.append(f"labels {labels.shape}, expected {(n_expected,)}")
    counts = np.bincount(labels, minlength=N_CLASSES)
    if len(counts) != N_CLASSES or len(set(counts.tolist())) != 1:
        problems.append(f"labels are not {N_CLASSES} balanced classes: counts {counts.tolist()}")
    if len(chars) != N_CLASSES:
        problems.append(f"mapping has {len(chars)} classes, expected {N_CLASSES}")
    if problems:
        raise EmnistDataError(f"EMNIST balanced {split} verification failed: " + "; ".join(problems))
    # Stored transposed: swap row and column to get upright characters.
    upright = np.ascontiguousarray(np.transpose(images, (0, 2, 1)))

    def _write(tmp: str) -> None:
        with open(tmp, "wb") as fh:
            np.savez(fh, images=upright, labels=labels, identity=np.array(json.dumps(identity)))

    atomic_write(cache, _write)
    if verbose:
        print(f"  [emnist] wrote {cache}", flush=True)
    return EmnistData(split, upright, labels, chars, identity)


def load(cfg: EmnistSpec, verbose: bool = True) -> EmnistData:
    """The verified split named by cfg."""
    data = load_emnist(Path(cfg.data_dir) if cfg.data_dir else EMNIST_DATA_DIR, cfg.split, verbose=verbose)
    if verbose:
        print(f"  [emnist] {cfg.split}: {len(data.images)} images")
    return data


def dequantize(pixels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(pixel + U(0, 1)) / 256: a continuous target in (0, 1) whose density bounds the pixel pmf."""
    u = rng.uniform(0.0, 1.0, size=pixels.shape)
    # U(0, 1) can return exactly 0; keep y strictly inside (0, 1).
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return (pixels.astype(np.float64) + u) / 256.0


def bits_per_pixel(nll_nats: float) -> float:
    """Discrete bits/pixel bound from a per-point NLL (nats) of the dequantized target: (NLL + ln 256) / ln 2."""
    return (float(nll_nats) + math.log(256.0)) / math.log(2.0)


_COORDS = np.stack(np.divmod(np.arange(SIDE * SIDE), SIDE), axis=1).astype(np.float64)  # (784, 2): (row, col)


def _image_order(n: int, seed: int, split: str) -> np.ndarray:
    # The constant separates this stream from the per-episode default_rng([seed, ep_i]).
    return np.random.default_rng([int(seed), 0xE3A1, SPLITS.index(split)]).permutation(n)


def draw(data: EmnistData, rng: np.random.Generator, cfg: EmnistSpec, *, seed: int, ep_i: int) -> RealDraw | None:
    """Episode ep_i: image perm[ep_i]; pixels split by the mask, then dequantized."""
    n = len(data.images)
    if ep_i >= n:
        print(f"  [emnist] ep {ep_i}: the {data.split} split has only {n} images; skipping")
        return None
    img = int(_image_order(n, seed, data.split)[ep_i])
    pool = np.arange(SIDE * SIDE)
    anchor: int | None = None
    if cfg.mask == "random":
        ctx, qry = random_split(rng, pool, cfg.n_context, cfg.n_query)
    else:
        ctx, qry, anchor = block_split(rng, pool, _COORDS, cfg.n_context, cfg.n_query)
    cols = np.concatenate([ctx, qry])
    y = dequantize(data.images[img].reshape(-1)[cols], rng)
    meta = {
        "split": data.split,
        "image": img,
        "char": data.chars[int(data.labels[img])],
        "fold": None,
        "context_pixels": ctx.tolist(),
        "query_pixels": qry.tolist(),
        "anchor_pixel": anchor,
        "cluster": f"img{img}",
    }
    return RealDraw(_COORDS[ctx], y[: len(ctx)], _COORDS[qry], y[len(ctx) :], meta)


def validate(cfg: EmnistSpec) -> None:
    """Raise ValueError for an out-of-range real.emnist field."""
    check_choice("real.emnist.split", cfg.split, SPLITS)
    check_choice("real.emnist.mask", cfg.mask, MASKS)
    if cfg.n_context + cfg.n_query > SIDE * SIDE:
        raise ValueError(f"real.emnist: n_context + n_query exceeds the {SIDE * SIDE} pixels of an image")


def episode_shape(cfg: EmnistSpec) -> tuple[int, int]:
    """(P, N) of every episode."""
    return int(cfg.n_context), int(cfg.n_query)


def fingerprint(data: EmnistData, cfg: EmnistSpec) -> dict[str, Any]:
    """Everything that defines EMNIST episode k: the split's file identities and every draw setting."""
    return {
        "data": data.identity,
        "n_context": int(cfg.n_context),
        "n_query": int(cfg.n_query),
        "mask": cfg.mask,
        "dequantization": "uniform/256",
    }


def label(meta: dict[str, Any]) -> str:
    """One-line episode description for the per-episode print."""
    return f"EMNIST {meta['split']} img{meta['image']} {meta['char']!r} P={meta['P']} N={meta['N']}"
