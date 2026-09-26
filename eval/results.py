"""Result summaries, coverage rules, cache I/O, and offline rendering."""

from __future__ import annotations

import math
import json
import os
import argparse
from collections.abc import Mapping, Sequence

import numpy as np
from torch import Tensor
from copula_inter.artifacts import atomic_json_save

NAN_PARTS: dict[str, float] = {"total": float("nan"), "marginal": float("nan"), "copula": float("nan")}


def numeric_summary(values: Sequence[float]) -> tuple[float, float, int]:
    finite = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if not len(finite):
        return float("nan"), float("nan"), 0
    return float(finite.mean()), float(finite.std()), len(finite)


def score_summary(episodes: Sequence[Mapping[str, float]], key: str) -> tuple[float, float, int]:
    return numeric_summary([ep.get(key, float("nan")) for ep in episodes])


def competition_ranks(
    episodes: Sequence[Mapping[str, float]], keys: Sequence[str],
) -> dict[str, list[int]]:
    """Equal scores share a rank; the next rank skips the tied positions."""
    ranks: dict[str, list[int]] = {key: [] for key in keys}
    for episode in episodes:
        scores = [(key, float(episode.get(key, float("nan")))) for key in keys]
        valid = sorted(
            ((key, value) for key, value in scores if math.isfinite(value)),
            key=lambda pair: pair[1],
        )
        prior_value = None
        rank = 0
        for position, (key, value) in enumerate(valid, start=1):
            if prior_value is None or value != prior_value:
                rank = position
            ranks[key].append(rank)
            prior_value = value
    return ranks


def require_coverage(valid: int, attempted: int, minimum_fraction: float) -> None:
    if not 0 <= minimum_fraction <= 1:
        raise ValueError("minimum coverage must lie in [0, 1]")
    if attempted == 0 or valid / attempted < minimum_fraction:
        raise RuntimeError(
            f"evaluation coverage {valid}/{attempted} is below {minimum_fraction:.0%}"
        )


def jsonable(obj):
    """Convert nested numeric results, including scalar tensors, to JSON values."""
    if isinstance(obj, dict):
        return {key: jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(value) for value in obj]
    if isinstance(obj, Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return float(obj)


def load_results_cache(path: str, fingerprint: dict) -> dict[str, dict]:
    """Return only scored episodes with a matching result fingerprint."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as source:
            blob = json.load(source)
    except (OSError, ValueError) as exc:
        print(f"  [results_cache] failed to read {path}: {exc} — starting fresh")
        return {}
    if blob.get("fingerprint") != fingerprint:
        print(f"  [results_cache] {path} was produced under different settings — ignoring it")
        return {}
    entries = blob.get("episodes", {})
    print(f"  [results_cache] resuming with {len(entries)} already-scored episode(s) from {path}")
    return entries


def save_results_cache(path: str, fingerprint: dict, entries: dict[str, dict]) -> None:
    """Publish completed scored episodes atomically after every episode."""
    atomic_json_save({"fingerprint": fingerprint, "episodes": entries}, path)


def render_saved_totals(report: Mapping) -> str:
    """Summarize a --dump_episodes file without constructing a model."""
    episodes = report["episodes"]
    methods = sorted({name for episode in episodes for name in episode.get("total_nlls", {})})
    lines = ["| Method | Mean total NLL | Std | Valid/All |", "|---|---:|---:|---:|"]
    for method in methods:
        values = [episode.get("total_nlls", {}).get(method, {}).get("total", float("nan"))
                  for episode in episodes]
        mean, std, count = numeric_summary(values)
        lines.append(f"| {method} | {mean:.4f} | {std:.4f} | {count}/{len(episodes)} |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render saved evaluation totals without loading a model")
    parser.add_argument("dump", help="JSON created by eval_checkpoint.py --dump_episodes")
    args = parser.parse_args()
    with open(args.dump, encoding="utf-8") as source:
        print(render_saved_totals(json.load(source)))
