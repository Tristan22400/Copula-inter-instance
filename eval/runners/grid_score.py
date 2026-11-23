"""Score copula checkpoints on a fixed evaluation grid against its stored GP posterior oracle.

The grid (generate_pit_dataset with data.store_posterior_oracle=true and data.query_rows)
is a directory of cells <grid_dir>/<group>/P<P>/pit, each holding episodes with the
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
PIT overlay (copula_inter.repit_dataset), one per cell at <pit_overlay_root>/<group>/P<P>/pit.

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

from copula_inter.copula.loss import y_space_nll
from copula_inter.copula.model import build_sigma
from copula_inter.data.dataset import CopulaDataset, collate_fn, has_varying_context_feature
from eval.runners.hydra_cli import hydra_entry
from inference.copula_inference import load_copula_model

__all__ = ["GridScoreSpec", "grid_cells", "oracle_nll", "score_cell"]

TERMS = ("total", "marginal", "copula", "oracle", "oracle_marginal", "gain", "gap", "marginal_gap", "copula_gap")


@dataclass
class GridScoreSpec:
    """Checkpoints to score on the grid cells under grid_dir."""

    grid_dir: str = MISSING
    ckpts: list[str] = field(default_factory=list)
    # Cells to score, as "<group>/P<P>" (empty: every cell under grid_dir).
    cells: list[str] = field(default_factory=list)
    # PIT overlays of the grid (one per cell at <root>/<group>/P<P>/pit); null reads the grid's own PIT.
    pit_overlay_root: str | None = None
    # Leave out the episodes whose context features are all constant (scored by TabPFN / Kumo
    # through a feature-free Gaussian fallback).
    drop_constant_context: bool = False
    batch_size: int = 32
    device: str = "cuda"
    out_json: str | None = None


def grid_cells(grid_dir: str, cells: list[str]) -> list[str]:
    """The "<group>/P<P>" cells under grid_dir (those named in cells, or all), in a stable order."""
    found = sorted(
        os.path.relpath(os.path.dirname(path), grid_dir)
        for path in glob.glob(os.path.join(grid_dir, "*", "P*", "pit"))
        if os.path.isdir(path)
    )
    if not cells:
        return sorted(found, key=lambda c: (c.split("/")[0], int(c.split("/P")[1])))
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
) -> dict[str, np.ndarray]:
    """Per-episode TERMS of one cell (arrays over its episodes, or those in indices, in dataset order)."""
    jitter = float(cfg.model.get("sigma_jitter", 1e-4))
    out: dict[str, list[float]] = {term: [] for term in TERMS}
    order = list(range(len(dataset))) if indices is None else indices
    for start in range(0, len(order), batch_size):
        episodes = [dataset[i] for i in order[start : start + batch_size]]
        # Episodes of one batch must share (P, N, d): a cell's shards each hold one shape.
        groups: dict[tuple[int, ...], list[dict]] = {}
        for ep in episodes:
            groups.setdefault((*ep["x_norm_train"].shape, int(ep["x_norm_test"].shape[0])), []).append(ep)
        for group in groups.values():
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in collate_fn(group).items()}
            Sigma = build_sigma(model(batch), cfg, jitter=jitter, test_mask=batch["test_mask"])
            for b, ep in enumerate(group):
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
                out["total"].append(total)
                out["marginal"].append(marginal)
                out["copula"].append(float(parts["copula"]))
                out["oracle"].append(orc)
                out["oracle_marginal"].append(orc_marginal)
                out["gain"].append(marginal - total)
                out["gap"].append(total - orc)
                out["marginal_gap"].append(marginal - orc_marginal)
                out["copula_gap"].append((total - orc) - (marginal - orc_marginal))
    return {term: np.asarray(values) for term, values in out.items()}


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    n = len(values)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")


def run(spec: GridScoreSpec) -> None:
    if not spec.ckpts:
        raise SystemExit("grid_score: ckpts=[...] names no checkpoint")
    device = spec.device if torch.cuda.is_available() else "cpu"
    cells = grid_cells(spec.grid_dir, spec.cells)
    per_episode: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for ckpt in spec.ckpts:
        model, cfg = load_copula_model(ckpt, device=device)
        per_episode[ckpt] = {}
        for cell in cells:
            overlay = None if spec.pit_overlay_root is None else os.path.join(spec.pit_overlay_root, cell, "pit")
            dataset = CopulaDataset(episode_dir=os.path.join(spec.grid_dir, cell, "pit"), pit_overlay_dir=overlay)
            if "R_post" not in dataset[0]:
                raise ValueError(
                    f"{cell} has no stored posterior oracle (generate it with data.store_posterior_oracle=true)"
                )
            indices = None
            if spec.drop_constant_context:
                indices = [i for i in range(len(dataset)) if has_varying_context_feature(dataset[i]["x_norm_train"])]
            per_episode[ckpt][cell] = score_cell(model, cfg, dataset, device, spec.batch_size, indices)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    reference = spec.ckpts[0]
    result: dict[str, Any] = {"grid_dir": spec.grid_dir, "pit_overlay_root": spec.pit_overlay_root, "cells": {}}
    for cell in cells:
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
    for cell in cells:
        for ckpt in spec.ckpts:
            e = result["cells"][cell][ckpt]
            delta = f"{e['total_minus_ref']:+.4f}±{e['total_minus_ref_se']:.4f}" if "total_minus_ref" in e else ""
            print(f"{cell:24s} {names[ckpt]:5s} " + " ".join(f"{e[c]:12.4f}" for c in cols) + f" {delta:>16s}")
    if spec.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(spec.out_json)), exist_ok=True)
        with open(spec.out_json, "w", encoding="utf-8") as sink:
            json.dump(result, sink, indent=1)
        print(f"\nwrote {spec.out_json}")


main = hydra_entry("grid_score", GridScoreSpec, run)


if __name__ == "__main__":
    main()
