"""Recompute the PIT of an on-disk dataset with another frozen marginal model (a PIT overlay).

The tables of a sharded dataset (x_norm_train, y_train, x_norm_test, y_test) stay
as they are; only the marginal changes. For each source shard this writes an
overlay shard of the same name, in <dataset>/overlays/<data.repit_overlay>, holding, per episode, z_train (the K-fold PIT of the
context rows), z_test and log_pdf_test (the query rows given every context row)
under the marginal set by data.z_train_source (+ tabicl.pit_ckpt or
data.z_train_marginal_ckpt), computed exactly as generate_pit_dataset computes
them at generation (gp_episodes.marginal_pit_tables). Training reads it with
training.pit_overlay=<name> (CopulaDataset), so copulas trained on the PIT of
different marginals see the same tables. Name an overlay after its marginal
(tabpfn35, kumo_small_era5ft); its settings go in overlay.json.

overlay.json records the source dataset's manifest digest and the marginal; a
rerun with another marginal under the same name is refused. Shards are
split among workers by worker_id / num_workers and resumed from their count
sidecars, as in generate_pit_dataset (scripts/generate_dataset.sh runs either
module through GEN_MODULE).

Usage:
    python -m copula_inter.repit_dataset data.repit_source_dir=<dataset>/pit \\
        data.repit_overlay=tabpfn35 data.z_train_source=tabpfn35_fast
"""

from __future__ import annotations

import fcntl
import gc
import json
import os
from typing import Any

# Set before torch initializes CUDA (reduces allocator fragmentation).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from copula_inter.artifacts import atomic_json_save, atomic_torch_save
from copula_inter.backend_registry import (
    GENERIC_MARGINAL_BACKENDS,
    TABICL_Z_TRAIN_SOURCES,
    validate_z_train_source,
    z_train_marginal_ckpt,
)
from copula_inter.backend_registry import z_train_source as z_train_source_of
from copula_inter.config_path import config_dir
from copula_inter.data.dataset import PIT_OVERLAY_KEYS, PIT_OVERLAY_META, pit_overlay_dir
from copula_inter.data.gp_episodes import marginal_pit_tables
from copula_inter.data.manifest import shard_count_path

__all__ = ["main", "overlay_spec", "repit_shard"]


def overlay_spec(cfg: DictConfig, source_dir: str, ckpt: str | None) -> dict[str, Any]:
    """overlay.json of a re-PIT of source_dir: its manifest digest, shard layout and the marginal."""
    with open(os.path.join(source_dir, "manifest.json"), encoding="utf-8") as source:
        manifest = json.load(source)
    data = manifest["spec"]["data"]
    return {
        "source_dir": os.path.normpath(source_dir),
        "source_digest": manifest["digest"],
        "n_tasks": int(data["n_tasks"]),
        "shard_size": int(data["shard_size"]),
        "z_train_source": z_train_source_of(cfg),
        "marginal_ckpt": None if ckpt is None else str(ckpt),
        "k_folds": int(cfg.data.get("z_train_tabicl_k_folds", 10)),
        # Recorded only when on, so float32 overlays written before the key existed keep their identity.
        **({"tabicl_inference_amp": True} if _tabicl_amp(cfg) else {}),
        **({"marginal_inference_amp": True} if _marginal_amp(cfg) else {}),
    }


def _tabicl_amp(cfg: DictConfig) -> bool:
    """training.tabicl_inference_amp: mixed-precision TabICL forwards (TabICL marginals only)."""
    return z_train_source_of(cfg) == "tabicl" and bool(cfg.training.get("tabicl_inference_amp", False))


def _marginal_amp(cfg: DictConfig) -> bool:
    """training.marginal_inference_amp: mixed-precision forwards of a generic marginal (tabpfn35_fast, kumo_small)."""
    return z_train_source_of(cfg) in GENERIC_MARGINAL_BACKENDS and bool(
        cfg.training.get("marginal_inference_amp", False)
    )


def _ensure_overlay_meta(out_dir: str, spec: dict[str, Any]) -> None:
    """Write overlay.json, or check that the one already there describes the same re-PIT."""
    path = os.path.join(out_dir, PIT_OVERLAY_META)
    with open(os.path.join(out_dir, "overlay.lock"), "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as source:
                    existing = json.load(source)
                if existing != spec:
                    raise ValueError(f"{path} describes another re-PIT: {existing} != {spec}")
            else:
                atomic_json_save(spec, path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def repit_shard(
    episodes: list[dict[str, Any]],
    device: str,
    *,
    seed: int,
    k_folds: int,
    tabicl_model: Any = None,
    marginal_backend: str | None = None,
    marginal_regressor: Any = None,
    max_chunk: int | None = None,
) -> list[dict[str, torch.Tensor]]:
    """PIT_OVERLAY_KEYS of every episode of one shard under the given marginal.

    Episodes of equal (P, N, d) are batched, max_chunk at a time, halving the
    chunk on CUDA OOM. Chunk c of a shape group gets the fold seed seed + c.
    """
    out: list[dict[str, torch.Tensor] | None] = [None] * len(episodes)
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, ep in enumerate(episodes):
        key = (*ep["x_norm_train"].shape, int(ep["x_norm_test"].shape[0]))
        groups.setdefault(key, []).append(i)
    chunk_seed = seed
    for members in groups.values():
        chunk = min(len(members), max_chunk) if max_chunk else len(members)
        start = 0
        while start < len(members):
            idx = members[start : start + chunk]
            try:
                z_train, z_test, log_pdf = marginal_pit_tables(
                    torch.stack([episodes[i]["x_norm_train"] for i in idx]).float().to(device),
                    torch.stack([episodes[i]["y_train"] for i in idx]).float().to(device),
                    torch.stack([episodes[i]["x_norm_test"] for i in idx]).float().to(device),
                    torch.stack([episodes[i]["y_test"] for i in idx]).float().to(device),
                    device,
                    seed=chunk_seed,
                    k_folds=k_folds,
                    tabicl_model=tabicl_model,
                    marginal_backend=marginal_backend,
                    marginal_regressor=marginal_regressor,
                )
            except torch.cuda.OutOfMemoryError:
                if chunk == 1:
                    raise
                gc.collect()
                torch.cuda.empty_cache()
                chunk = max(1, chunk // 2)
                print(f"  [repit] CUDA OOM; chunk -> {chunk}")
                continue
            for j, i in enumerate(idx):
                out[i] = {
                    "z_train": z_train[j].float().cpu().clone(),
                    "z_test": z_test[j].float().cpu().clone(),
                    "log_pdf_test": log_pdf[j].float().cpu().clone(),
                }
            start += len(idx)
            chunk_seed += 1
            if device == "cuda":
                torch.cuda.empty_cache()
    assert all(entry is not None for entry in out)
    return [entry for entry in out if entry is not None]


def _overlay_is_done(out_path: str, expected: int) -> bool:
    sidecar = shard_count_path(out_path)
    if not (os.path.isfile(out_path) and sidecar.is_file()):
        return False
    with sidecar.open(encoding="utf-8") as source:
        return int(json.load(source)["count"]) == expected


@hydra.main(config_path=config_dir(__file__), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source_dir = cfg.data.get("repit_source_dir", None)
    if not source_dir:
        raise ValueError("data.repit_source_dir must name the pit/ directory of the dataset to re-PIT.")
    source_dir = str(source_dir)
    name = cfg.data.get("repit_overlay", None)
    if not name:
        raise ValueError("data.repit_overlay must name the overlay, after its marginal (e.g. tabpfn35).")
    out_dir = pit_overlay_dir(source_dir, str(name))
    os.makedirs(out_dir, exist_ok=True)
    worker_id = int(getattr(cfg, "worker_id", 0))
    num_workers = int(getattr(cfg, "num_workers", 1))
    if not (0 <= worker_id < num_workers):
        raise ValueError(f"worker_id={worker_id} must be in [0, num_workers={num_workers})")

    z_train_source = z_train_source_of(cfg)
    validate_z_train_source(z_train_source)
    tabicl_model = None
    marginal_regressor = None
    marginal_backend = z_train_source if z_train_source in GENERIC_MARGINAL_BACKENDS else None
    ckpt: str | None
    if z_train_source == "tabicl":
        from copula_inter.marginal.pit import configure_tabicl_inference_amp, load_tabicl, resolve_pit_ckpt

        configure_tabicl_inference_amp(_tabicl_amp(cfg))

        ckpt = resolve_pit_ckpt(cfg)
        if ckpt is None:
            raise ValueError("data.z_train_source=tabicl needs tabicl.pit_ckpt (or tabicl.ckpt).")
        print(f"Loading frozen TabICL marginal: {ckpt}")
        tabicl_model = load_tabicl(ckpt, device)
    elif marginal_backend is not None:
        from copula_inter.marginal.backends import configure_marginal_inference_amp, make_regressor

        configure_marginal_inference_amp(_marginal_amp(cfg))
        ckpt = z_train_marginal_ckpt(cfg)
        print(f"Building {marginal_backend} marginal on {device}" + (f" from {ckpt}" if ckpt else ""))
        marginal_regressor = make_regressor(marginal_backend, device=device, ckpt=ckpt)
    else:
        supported = ["tabicl", *GENERIC_MARGINAL_BACKENDS]
        raise ValueError(
            f"data.z_train_source={z_train_source} has no K-fold re-PIT (supported: {supported}; "
            f"{sorted(set(TABICL_Z_TRAIN_SOURCES) - {'tabicl'})} are not K-fold)."
        )

    spec = overlay_spec(cfg, source_dir, None if ckpt is None else str(ckpt))
    _ensure_overlay_meta(out_dir, spec)
    n_shards = (spec["n_tasks"] + spec["shard_size"] - 1) // spec["shard_size"]
    # data.repit_max_shards: re-PIT only the first shards (the subset a training.max_shards run reads).
    max_shards = cfg.data.get("repit_max_shards", None)
    if max_shards is not None:
        n_shards = min(n_shards, int(max_shards))
    owned = range(worker_id, n_shards, num_workers)
    print(
        f"Re-PIT {source_dir} -> {out_dir} with {z_train_source}: worker {worker_id}/{num_workers}, {len(owned)} shards"
    )

    max_chunk = cfg.data.get("generate_chunk_size", None)
    k_folds = int(spec["k_folds"])
    skipped_missing = 0
    for shard_idx in tqdm(owned, desc=f"repit[w{worker_id}]", unit="shard"):
        name = f"shard_{shard_idx:06d}.pt"
        src_path = os.path.join(source_dir, name)
        out_path = os.path.join(out_dir, name)
        src_sidecar = shard_count_path(src_path)
        if not (os.path.isfile(src_path) and src_sidecar.is_file()):
            skipped_missing += 1
            continue
        with src_sidecar.open(encoding="utf-8") as source:
            count = int(json.load(source)["count"])
        if _overlay_is_done(out_path, count):
            continue
        episodes = torch.load(src_path, map_location="cpu", weights_only=False, mmap=True)
        overlay = repit_shard(
            episodes,
            device,
            seed=shard_idx,
            k_folds=k_folds,
            tabicl_model=tabicl_model,
            marginal_backend=marginal_backend,
            marginal_regressor=marginal_regressor,
            max_chunk=None if max_chunk is None else int(max_chunk),
        )
        assert set(overlay[0]) == set(PIT_OVERLAY_KEYS)
        atomic_torch_save(overlay, out_path)
        atomic_json_save({"count": len(overlay)}, shard_count_path(out_path))
        del episodes
        if device == "cuda" and shard_idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    if skipped_missing:
        print(f"Skipped {skipped_missing} source shards not written yet (rerun to fill them).")
    print(f"Done. Worker {worker_id}/{num_workers}.")


if __name__ == "__main__":
    main()
