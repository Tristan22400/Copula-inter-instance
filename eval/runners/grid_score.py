"""Score copula checkpoints on saved synthetic episodes with a stored GP posterior oracle.

The grid (generate_pit_dataset with data.store_posterior_oracle=true and data.query_rows)
is a directory of cells <grid_dir>/<group>/<cell>/pit, each holding episodes with the
marginal's PIT (z_train, z_test, log_pdf_test) and the exact posterior of the query
rows given the context (mu_post, sigma_post, R_post). Per checkpoint, cell and episode,
in nats per query row (N rows):

    total     = marginal + copula            (y_space_nll of the model's Sigma)
    marginal  = -(1/N) sum_i log p(y_i)      (= the independence copula's total)
    copula    = (1/N) [ 1/2 log|Sigma| + 1/2 z^T (Sigma^{-1} - I) z ]
    oracle    = (1/N) [ 1/2 log|S| + 1/2 r^T S^{-1} r + N/2 log 2 pi ],
                S = D R_post D, D = diag(sigma_post), r = y - mu_post
    oracle_marginal = the same with R_post replaced by I (the exact marginals alone)

with gain = marginal - total (the copula's improvement over independence),
gap = total - oracle, split into marginal_gap = marginal - oracle_marginal and
copula_gap = gap - marginal_gap (= copula - the oracle's copula term, each under its
own PIT). Checkpoints after the first also get the paired per-episode
difference to the first (same episodes), with its standard error.

A copula trained on another marginal's PIT reads the grid through that marginal's
PIT overlay (copula_inter.repit_dataset), with the same relative cell paths as the grid.

Usage:
    python -m eval.runners.grid_score grid_dir=<data>/eval_grid_v1 ckpts=[<a.pt>,<b.pt>] \\
        out_json=results/grid/a_vs_b.json
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from omegaconf import MISSING

from copula_inter.artifacts import atomic_write
from copula_inter.copula.loss import y_space_nll
from copula_inter.copula.model import build_sigma
from copula_inter.data.dataset import CopulaDataset, collate_fn, has_varying_context_feature
from copula_inter.data.gp_episodes import subsample_query_rows
from eval.runners.hydra_cli import hydra_entry
from inference.copula_inference import load_copula_model

__all__ = ["GridScoreSpec", "grid_cells", "oracle_nll", "score_cell"]

TERMS = ("total", "marginal", "copula", "oracle", "oracle_marginal", "gain", "gap", "marginal_gap", "copula_gap")


@dataclass
class GridScoreSpec:
    """Checkpoints to score on the grid cells under grid_dir."""

    grid_dir: str = MISSING
    ckpts: list[str] = field(default_factory=list)
    # Cells to score, as "<group>/<cell>" (empty: every cell under grid_dir).
    cells: list[str] = field(default_factory=list)
    # Shared PIT overlay root, with the grid's relative cell paths. Null uses the stored PIT.
    pit_overlay_root: str | None = None
    # Optional root per checkpoint in ckpts order. A null entry uses the stored PIT.
    pit_overlay_roots: list[str | None] = field(default_factory=list)
    # Score nested query subsets on episodes large enough for the largest size. Empty scores all rows.
    query_sizes: list[int] = field(default_factory=list)
    # Leave out the episodes whose context features are all constant (scored by TabPFN / Kumo
    # through a feature-free Gaussian fallback).
    drop_constant_context: bool = False
    batch_size: int = 32
    device: str = "cuda"
    out_json: str | None = None


def _cell_sort_key(cell: str) -> tuple[str, float, str]:
    group, name = os.path.split(cell)
    size = int(name[1:]) if name.startswith("P") and name[1:].isdecimal() else math.inf
    return group, size, name


def grid_cells(grid_dir: str, cells: list[str]) -> list[str]:
    """Find fixed-size and variable-size cells under the grid root."""
    found = sorted(
        (
            os.path.relpath(os.path.dirname(path), grid_dir)
            for path in glob.glob(os.path.join(grid_dir, "*", "*", "pit"))
            if os.path.isdir(path)
        ),
        key=_cell_sort_key,
    )
    if not found:
        raise ValueError(f"no episode cells (<group>/<cell>/pit) under {grid_dir}")
    if not cells:
        return found
    missing = [c for c in cells if c not in found]
    if missing:
        raise ValueError(f"cells {missing} not under {grid_dir} (found {found})")
    return list(cells)


def oracle_nll(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Per-row Gaussian NLL of y under N(mu, D R D), D = diag(sigma): (B, N) inputs, (B,) output."""
    y, mu, sigma, R = (t.double() for t in (y, mu, sigma, R))
    S = sigma.unsqueeze(-1) * R * sigma.unsqueeze(-2)
    L = torch.linalg.cholesky(S + 1e-10 * torch.eye(S.shape[-1], dtype=S.dtype, device=S.device))
    r = (y - mu).unsqueeze(-1)
    alpha = torch.linalg.solve_triangular(L, r, upper=False).squeeze(-1)
    log_det = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    N = y.shape[-1]
    return (0.5 * log_det + 0.5 * (alpha**2).sum(-1) + 0.5 * N * math.log(2 * math.pi)) / N


@torch.no_grad()
def score_cell(
    model: torch.nn.Module,
    cfg: Any,
    dataset: CopulaDataset,
    device: str,
    batch_size: int,
    indices: list[int] | None = None,
    query_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Per-episode TERMS of one cell (arrays over its episodes, or those in indices, in dataset order)."""
    jitter = float(cfg.model.get("sigma_jitter", 1e-4))
    order = range(len(dataset)) if indices is None else indices
    out = {term: np.empty(len(order), dtype=np.float64) for term in TERMS}
    for start in range(0, len(order), batch_size):
        groups: dict[tuple[int, ...], list[tuple[int, dict]]] = {}
        for position, episode_idx in enumerate(order[start : start + batch_size], start=start):
            ep = dataset.get_exact(episode_idx)
            if query_size is not None:
                # Resetting the seed across sizes takes prefixes of one permutation per episode.
                ep = subsample_query_rows(ep, query_size, torch.Generator().manual_seed(episode_idx))
            shape = (*ep["x_norm_train"].shape, int(ep["x_norm_test"].shape[0]))
            groups.setdefault(shape, []).append((position, ep))
        for entries in groups.values():
            group = [ep for _, ep in entries]
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in collate_fn(group).items()}
            Sigma = build_sigma(model(batch), cfg, jitter=jitter, test_mask=batch["test_mask"])
            for b, (position, ep) in enumerate(entries):
                parts = y_space_nll(
                    Sigma[b : b + 1].float(),
                    batch["z_test"][b : b + 1].float(),
                    batch["log_pdf_test"][b : b + 1].float(),
                    batch["test_mask"][b : b + 1],
                )
                total, marginal = float(parts["total"]), float(parts["marginal"])
                y, mu, sd, R = (ep[k][None] for k in ("y_test", "mu_post", "sigma_post", "R_post"))
                orc = float(oracle_nll(y, mu, sd, R))
                orc_marginal = float(oracle_nll(y, mu, sd, torch.eye(R.shape[-1], dtype=R.dtype)[None]))
                out["total"][position] = total
                out["marginal"][position] = marginal
                out["copula"][position] = float(parts["copula"])
                out["oracle"][position] = orc
                out["oracle_marginal"][position] = orc_marginal
                out["gain"][position] = marginal - total
                out["gap"][position] = total - orc
                out["marginal_gap"][position] = marginal - orc_marginal
                out["copula_gap"][position] = (total - orc) - (marginal - orc_marginal)
    return out


def _eligible_indices(dataset: CopulaDataset, *, min_queries: int, drop_constant_context: bool) -> list[int] | None:
    if not min_queries and not drop_constant_context:
        return None
    indices = []
    for index in range(len(dataset)):
        ep = dataset.get_exact(index)
        if int(ep["n_test"]) >= min_queries and (
            not drop_constant_context or has_varying_context_feature(ep["x_norm_train"])
        ):
            indices.append(index)
    return indices


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    n = len(values)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")


def run(spec: GridScoreSpec) -> None:
    if not spec.ckpts:
        raise SystemExit("grid_score: ckpts=[...] names no checkpoint")
    if len(set(spec.ckpts)) != len(spec.ckpts) or len(set(spec.cells)) != len(spec.cells):
        raise ValueError("ckpts and cells must contain distinct entries")
    if spec.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if any(n < 1 for n in spec.query_sizes) or len(set(spec.query_sizes)) != len(spec.query_sizes):
        raise ValueError("query_sizes must contain distinct positive integers")
    if spec.pit_overlay_roots and (spec.pit_overlay_root is not None or len(spec.pit_overlay_roots) != len(spec.ckpts)):
        raise ValueError("pit_overlay_roots must have one entry per checkpoint and excludes pit_overlay_root")
    device = spec.device if torch.cuda.is_available() else "cpu"
    cells = grid_cells(spec.grid_dir, spec.cells)
    query_sizes = sorted(spec.query_sizes)
    sizes: list[int | None] = list(query_sizes) if query_sizes else [None]
    score_cells = [cell if size is None else f"{cell}/N{size}" for cell in cells for size in sizes]
    roots = spec.pit_overlay_roots or [spec.pit_overlay_root] * len(spec.ckpts)
    per_episode: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    eligible: dict[str, list[int] | None] = {}
    for ckpt, root in zip(spec.ckpts, roots, strict=True):
        model, cfg = load_copula_model(ckpt, device=device)
        per_episode[ckpt] = {}
        for cell in cells:
            overlay = None if root is None else os.path.join(root, cell, "pit")
            dataset = CopulaDataset(episode_dir=os.path.join(spec.grid_dir, cell, "pit"), pit_overlay_dir=overlay)
            if not len(dataset):
                raise ValueError(f"{cell} has no episodes")
            if "R_post" not in dataset.get_exact(0):
                raise ValueError(
                    f"{cell} has no stored posterior oracle (generate it with data.store_posterior_oracle=true)"
                )
            if cell not in eligible:
                eligible[cell] = _eligible_indices(
                    dataset, min_queries=max(query_sizes, default=0), drop_constant_context=spec.drop_constant_context
                )
                if eligible[cell] == []:
                    raise ValueError(
                        f"{cell} has no episodes matching the context filter and query sizes {query_sizes}"
                    )
            for size in sizes:
                key = cell if size is None else f"{cell}/N{size}"
                per_episode[ckpt][key] = score_cell(
                    model, cfg, dataset, device, spec.batch_size, eligible[cell], query_size=size
                )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    reference = spec.ckpts[0]
    result: dict[str, Any] = {
        "grid_dir": spec.grid_dir,
        "pit_overlay_root": spec.pit_overlay_root,
        "drop_constant_context": spec.drop_constant_context,
        "cells": {},
    }
    if query_sizes:
        result["query_sizes"] = query_sizes
    if spec.pit_overlay_roots:
        result["pit_overlay_roots"] = dict(zip(spec.ckpts, spec.pit_overlay_roots, strict=True))
    for cell in score_cells:
        row: dict[str, Any] = {}
        for ckpt in spec.ckpts:
            scores = per_episode[ckpt][cell]
            entry: dict[str, Any] = {"n": len(scores["total"])}
            for term in TERMS:
                entry[term], entry[f"{term}_se"] = _mean_se(scores[term])
            if ckpt != reference:
                diff = scores["total"] - per_episode[reference][cell]["total"]
                entry["total_minus_ref"], entry["total_minus_ref_se"] = _mean_se(diff)
            row[ckpt] = entry
        result["cells"][cell] = row

    names = {ckpt: f"m{i}" for i, ckpt in enumerate(spec.ckpts)}
    for ckpt, name in names.items():
        print(f"{name} = {ckpt}")
    cols = ("total", "gain", "gap", "marginal_gap", "copula_gap", "oracle")
    print(f"\n{'cell':24s} {'model':5s} " + " ".join(f"{c:>12s}" for c in cols) + f" {'Δ total vs m0':>16s}")
    for cell in score_cells:
        for ckpt in spec.ckpts:
            e = result["cells"][cell][ckpt]
            delta = f"{e['total_minus_ref']:+.4f}±{e['total_minus_ref_se']:.4f}" if "total_minus_ref" in e else ""
            print(f"{cell:24s} {names[ckpt]:5s} " + " ".join(f"{e[c]:12.4f}" for c in cols) + f" {delta:>16s}")
    if spec.out_json:

        def write(path: str) -> None:
            with open(path, "w", encoding="utf-8") as sink:
                json.dump(result, sink, indent=1)

        atomic_write(spec.out_json, write)
        print(f"\nwrote {spec.out_json}")


main = hydra_entry("grid_score", GridScoreSpec, run)


if __name__ == "__main__":
    main()
