"""io.py — result serialization + reporting shared by every benchmark runner."""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import norm

__all__ = ["gp_to_quantile_and_R", "save_results_json", "print_markdown_summary"]

_METRIC_COLUMNS = ["energy_score", "nll_total", "nll_copula", "nll_marginal", "corr_frob"]


def gp_to_quantile_and_R(
    mean: np.ndarray, cov: np.ndarray, probs: np.ndarray, jitter: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a joint Gaussian (mean, cov) into the (quantile_grid, probs, R)
    representation shared by every method in this suite. Each marginal of a
    joint Gaussian is exactly Gaussian, so this is an exact conversion, not
    an approximation.

    Returns:
        quantile_grid: (N, Q) — quantile_grid[i, j] = mean[i] + sqrt(var[i]) * Phi^-1(probs[j])
        probs        : (Q,)   — unchanged, returned for interface symmetry
        R            : (N, N) — correlation matrix, symmetrized, unit diagonal, clipped to [-1, 1]
    """
    var = np.clip(np.diag(cov), jitter, None)
    z = norm.ppf(probs)
    quantile_grid = mean[:, None] + np.sqrt(var)[:, None] * z[None, :]

    outer_std = np.sqrt(np.outer(var, var))
    R = cov / outer_std
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    R = np.clip(R, -1.0, 1.0)
    return quantile_grid, probs, R


def save_results_json(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def print_markdown_summary(results: list[dict]) -> None:
    """Print a markdown table of mean +/- std per (benchmark, method, metric)."""
    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r["benchmark"], r["method"])
        for col in _METRIC_COLUMNS:
            val = r.get(col)
            if val is not None and np.isfinite(val):
                groups[key][col].append(val)

    header = ["benchmark", "method"] + _METRIC_COLUMNS
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for (benchmark, method), metrics in sorted(groups.items()):
        row = [benchmark, method]
        for col in _METRIC_COLUMNS:
            vals = metrics.get(col, [])
            if not vals:
                row.append("n/a")
            else:
                arr = np.asarray(vals)
                row.append(f"{arr.mean():.4f} +/- {arr.std():.4f}")
        print("| " + " | ".join(row) + " |")
