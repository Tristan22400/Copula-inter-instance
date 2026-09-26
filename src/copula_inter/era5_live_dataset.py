"""Live training episodes from real ARCO-ERA5 2 m temperature (eval/data/era5_global_corpus.py).

Enabled with training.live_generation=true training.live_source=era5. There
is no oracle correlation, so episodes carry only what y_space_nll needs
(training.aux_mae_weight must be 0). z_train/z_test/log_pdf_test come from a
frozen TabICL (or another marginal backend) PIT, so a checkpoint is required.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from copula_inter.backend_registry import z_train_source as z_train_source_of
from copula_inter.live_dataset import _limit_worker_threads, resolve_live_tabicl_num_workers, worker_seed
from copula_inter.pit import (
    configure_tabicl_inference_amp,
    load_tabicl,
    normalize_targets,
    resolve_pit_ckpt,
    run_pit,
    run_pit_batched,
)
from eval.data.era5_global_corpus import GlobalERA5Corpus, load_shared_corpus_arrays

__all__ = ["build_era5_train_loader", "build_era5_fixed_val_batches", "era5_collate_fn"]


def era5_collate_fn(samples: List[dict]) -> dict:
    """Pad a batch of variable-P/N ERA5 episodes: x, y (raw), z_train, z_test, log_pdf_test and masks (no oracle fields)."""
    B = len(samples)
    d_x = samples[0]["x_norm_train"].shape[-1]
    P_list = [int(s["x_norm_train"].shape[0]) for s in samples]
    N_list = [int(s["x_norm_test"].shape[0]) for s in samples]
    P_max = max(P_list)
    N_max = max(N_list)

    x_train = torch.zeros(B, P_max, d_x)
    x_test = torch.zeros(B, N_max, d_x)
    y_train = torch.zeros(B, P_max)
    y_test = torch.zeros(B, N_max)
    z_train = torch.zeros(B, P_max)
    z_test = torch.zeros(B, N_max)
    log_pdf_test = torch.zeros(B, N_max)
    train_mask = torch.zeros(B, P_max, dtype=torch.bool)
    test_mask = torch.zeros(B, N_max, dtype=torch.bool)

    for b, s in enumerate(samples):
        P, N = P_list[b], N_list[b]
        x_train[b, :P] = s["x_norm_train"]
        x_test[b, :N] = s["x_norm_test"]
        y_train[b, :P] = s["y_train"]
        y_test[b, :N] = s["y_test"]
        z_train[b, :P] = s["z_train"]
        z_test[b, :N] = s["z_test"]
        log_pdf_test[b, :N] = s["log_pdf_test"]
        train_mask[b, :P] = True
        test_mask[b, :N] = True

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "z_train": z_train,
        "z_test": z_test,
        "log_pdf_test": log_pdf_test,
        "train_mask": train_mask,
        "test_mask": test_mask,
    }


def _resolve_marginal(cfg) -> Tuple[Optional[str], int]:
    """(marginal_backend, marginal_probs_n) from data.z_train_source; backend None means TabICL ("analytic" is treated as TabICL)."""
    from copula_inter.live_dataset import _GENERIC_MARGINAL_BACKENDS, _validate_z_train_source

    z_train_source = z_train_source_of(cfg)
    _validate_z_train_source(z_train_source)
    backend = z_train_source if z_train_source in _GENERIC_MARGINAL_BACKENDS else None
    probs_n = int(cfg.data.get("z_train_marginal_probs_n", 99)) if "data" in cfg else 99
    return backend, probs_n


def _backend_pit_batched(
    x_train: torch.Tensor,
    y_train_scaled: torch.Tensor,
    x_test: torch.Tensor,
    y_test_scaled: torch.Tensor,
    *,
    backend: str,
    regressor,
    k_folds: int,
    probs_n: int,
    seed: int,
) -> dict:
    """Batched PIT of a group of ERA5 episodes through a non-TabICL backend; returns tensors on x_train's device."""
    from copula_inter.data_gen import _BATCHED_MARGINAL_BACKENDS

    run_batched = _BATCHED_MARGINAL_BACKENDS[backend]()
    out = run_batched(
        regressor,
        x_train.detach().cpu().numpy(),
        y_train_scaled.detach().cpu().numpy(),
        x_test.detach().cpu().numpy(),
        y_test_scaled.detach().cpu().numpy(),
        k_folds=k_folds,
        probs_n=probs_n,
        seed=seed,
    )
    dev = x_train.device
    return {k: torch.as_tensor(v, dtype=torch.float32, device=dev) for k, v in out.items()}


def _pit_episode(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    tabicl_model,
    k_folds: int,
    *,
    marginal_backend: Optional[str] = None,
    marginal_regressor=None,
    marginal_probs_n: int = 99,
    seed: int = 0,
) -> Optional[dict]:
    """PIT of one ERA5 episode (z_train, z_test, log_pdf_test in raw nats), or None with too little context."""
    if x_train.shape[0] < 2 or x_test.shape[0] < 1:
        return None
    y_train_scaled, y_test_scaled, _, std = normalize_targets(y_train, y_test)
    if marginal_backend is not None:
        # One episode through the batched path with a leading singleton axis.
        out = _backend_pit_batched(
            x_train.unsqueeze(0),
            y_train_scaled.unsqueeze(0),
            x_test.unsqueeze(0),
            y_test_scaled.unsqueeze(0),
            backend=marginal_backend,
            regressor=marginal_regressor,
            k_folds=k_folds,
            probs_n=marginal_probs_n,
            seed=seed,
        )
        return {
            "z_train": out["z_train"][0],
            "z_test": out["z_test"][0],
            "log_pdf_test": out["log_pdf_test"][0] - std.log(),
        }
    Y_train = y_train_scaled.unsqueeze(-1)
    Y_test = y_test_scaled.unsqueeze(-1)
    pit_out = run_pit(
        tabicl_model,
        x_train,
        Y_train,
        x_test,
        Y_test,
        k_folds=k_folds,
        Y_train_raw=y_train.unsqueeze(-1),
    )
    return {
        "z_train": pit_out["z_train"].squeeze(-1),
        "z_test": pit_out["z_test"].squeeze(-1),
        # Jacobian back to raw nats.
        "log_pdf_test": pit_out["log_pdf_test"].squeeze(-1) - std.log(),
    }


def _pit_group(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    tabicl_model,
    k_folds: int,
    *,
    marginal_backend: Optional[str] = None,
    marginal_regressor=None,
    marginal_probs_n: int = 99,
    seed: int = 0,
) -> Optional[dict]:
    """Batched PIT of B episodes that share P and N, each normalized by its own training targets.

    Args:
        x_train, y_train: (B, P, p_x), (B, P).
        x_test, y_test: (B, N, p_x), (B, N).

    Returns:
        dict of z_train (B, P), z_test (B, N), log_pdf_test (B, N), or None with
        too little context.
    """
    if x_train.shape[1] < 2 or x_test.shape[1] < 1:
        return None
    mean = y_train.mean(dim=-1, keepdim=True)
    std = y_train.std(dim=-1, keepdim=True).clamp(min=1e-8)
    y_train_scaled = (y_train - mean) / std
    y_test_scaled = (y_test - mean) / std
    if marginal_backend is not None:
        out = _backend_pit_batched(
            x_train,
            y_train_scaled,
            x_test,
            y_test_scaled,
            backend=marginal_backend,
            regressor=marginal_regressor,
            k_folds=k_folds,
            probs_n=marginal_probs_n,
            seed=seed,
        )
        return {
            "z_train": out["z_train"],
            "z_test": out["z_test"],
            "log_pdf_test": out["log_pdf_test"] - std.log(),
        }
    Y_train = y_train_scaled.unsqueeze(-1)
    Y_test = y_test_scaled.unsqueeze(-1)
    pit_out = run_pit_batched(
        tabicl_model,
        x_train,
        Y_train,
        x_test,
        Y_test,
        k_folds=k_folds,
        Y_train_raw=y_train.unsqueeze(-1),
    )
    return {
        "z_train": pit_out["z_train"].squeeze(-1),
        "z_test": pit_out["z_test"].squeeze(-1),
        # Per-episode Jacobian correction -- see _pit_episode's docstring.
        "log_pdf_test": pit_out["log_pdf_test"].squeeze(-1) - std.log(),
    }


class LiveERA5Dataset(IterableDataset):
    """Infinite stream of real-ERA5 episodes.

    Workers attach to a corpus loaded into shared memory by the main process
    (shared_corpus) and each load their own marginal on tabicl_device ("cuda").
    Episodes come in groups of group_size that share one grid size / context
    size (region, day and box vary), PIT'd in one batched call.
    """

    def __init__(
        self,
        shared_corpus: dict,
        tabicl_ckpt: str,
        tabicl_device: str,
        k_folds: int,
        grid_size_range: Tuple[int, int],
        box_deg_range: Tuple[float, float],
        n_context_frac_range: Tuple[float, float],
        base_seed: int,
        group_size: int = 1,
        tabicl_inference_amp: bool = True,
        marginal_backend: Optional[str] = None,
        marginal_probs_n: int = 99,
    ) -> None:
        self.shared_corpus = shared_corpus
        self.tabicl_ckpt = tabicl_ckpt
        self.tabicl_device = tabicl_device
        self.k_folds = k_folds
        self.grid_size_range = grid_size_range
        self.box_deg_range = box_deg_range
        self.n_context_frac_range = n_context_frac_range
        self.base_seed = base_seed
        self.group_size = group_size
        self.tabicl_inference_amp = bool(tabicl_inference_amp)
        # None: TabICL run_pit; otherwise a generic backend built per worker.
        self.marginal_backend = marginal_backend
        self.marginal_probs_n = int(marginal_probs_n)

    def __iter__(self) -> Iterator:
        import numpy as np

        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        configure_tabicl_inference_amp(self.tabicl_inference_amp)
        print(f"[era5_live_dataset] worker {worker_id}: attaching to shared global ERA5 corpus")
        corpus = GlobalERA5Corpus.from_shared(self.shared_corpus)
        tabicl_model = None
        marginal_regressor = None
        if self.marginal_backend is None:
            print(f"[era5_live_dataset] worker {worker_id}: loading frozen TabICL marginal: {self.tabicl_ckpt}")
            tabicl_model = load_tabicl(self.tabicl_ckpt, self.tabicl_device)
        else:
            from eval.spatial.marginal_backends import make_regressor

            print(
                f"[era5_live_dataset] worker {worker_id}: building {self.marginal_backend} marginal "
                f"on {self.tabicl_device}"
            )
            marginal_regressor = make_regressor(self.marginal_backend, device=self.tabicl_device)

        call_idx = 0
        while True:
            rng = np.random.default_rng(worker_seed(self.base_seed, worker_id, call_idx))
            call_idx += 1

            # One grid size / context size per group.
            grid_size = int(rng.integers(self.grid_size_range[0], self.grid_size_range[1] + 1))
            D = grid_size * grid_size
            n_context_frac = float(rng.uniform(*self.n_context_frac_range))
            n_context = int(np.clip(round(n_context_frac * D), 1, D - 1))

            raw_eps: list = []
            attempts = 0
            max_attempts = self.group_size * 20
            while len(raw_eps) < self.group_size and attempts < max_attempts:
                attempts += 1
                ep = corpus.sample_episode_fixed_shape(rng, grid_size, self.box_deg_range, n_context)
                if ep is not None:
                    raw_eps.append(ep)
            if len(raw_eps) < self.group_size:
                # No valid box for this shape: redraw the group next call.
                continue

            x_train = torch.as_tensor(
                np.stack([e["x_norm_train"] for e in raw_eps]), dtype=torch.float32, device=self.tabicl_device
            )
            x_test = torch.as_tensor(
                np.stack([e["x_norm_test"] for e in raw_eps]), dtype=torch.float32, device=self.tabicl_device
            )
            y_train = torch.as_tensor(
                np.stack([e["y_train"] for e in raw_eps]), dtype=torch.float32, device=self.tabicl_device
            )
            y_test = torch.as_tensor(
                np.stack([e["y_test"] for e in raw_eps]), dtype=torch.float32, device=self.tabicl_device
            )
            pit = _pit_group(
                x_train,
                y_train,
                x_test,
                y_test,
                tabicl_model,
                self.k_folds,
                marginal_backend=self.marginal_backend,
                marginal_regressor=marginal_regressor,
                marginal_probs_n=self.marginal_probs_n,
                seed=worker_seed(self.base_seed, worker_id, call_idx),
            )
            if pit is None:
                continue
            for i in range(self.group_size):
                yield {
                    "x_norm_train": x_train[i].cpu(),
                    "x_norm_test": x_test[i].cpu(),
                    "y_train": y_train[i].cpu(),
                    "y_test": y_test[i].cpu(),
                    "z_train": pit["z_train"][i].cpu(),
                    "z_test": pit["z_test"][i].cpu(),
                    "log_pdf_test": pit["log_pdf_test"][i].cpu(),
                }


def _resolve_era5_cfg(cfg: DictConfig) -> dict:
    e = cfg.get("era5_live", {}) or {}
    val_corpus_dir = e.get("val_corpus_dir", None)
    return {
        "corpus_dir": str(e.get("corpus_dir", "./eval/data/cache/era5_global")),
        # Optional separate corpus for the fixed validation set.
        "val_corpus_dir": str(val_corpus_dir) if val_corpus_dir else None,
        "grid_size_range": (int(e.get("grid_size_min", 8)), int(e.get("grid_size_max", 28))),
        "box_deg_range": (float(e.get("box_deg_min", 5.0)), float(e.get("box_deg_max", 25.0))),
        "n_context_frac_range": (float(e.get("n_context_frac_min", 0.05)), float(e.get("n_context_frac_max", 0.4))),
        "val_episodes": int(e.get("val_episodes", 200)),
        "val_seed": int(e.get("val_seed", 20260823)),
    }


def build_era5_train_loader(cfg: DictConfig, t: DictConfig, device: str) -> DataLoader:
    """Training DataLoader over LiveERA5Dataset: shared-memory corpus loaded here, spawn start method, GPU workers sized by resolve_live_tabicl_num_workers."""
    if device != "cuda":
        raise ValueError(
            f"training.live_source=era5 requires device='cuda' (got {device!r}) — "
            "real-ERA5 episodes are PIT'd through a frozen TabICL model, and "
            "CPU-only TabICL inference was benchmarked and rejected as too slow "
            "for live generation (see live_dataset.py::build_live_train_loader)."
        )
    marginal_backend, marginal_probs_n = _resolve_marginal(cfg)
    ckpt = resolve_pit_ckpt(cfg)
    if ckpt is None and marginal_backend is None:
        raise ValueError(
            "training.live_source=era5 requires a resolvable TabICL checkpoint — "
            "set tabicl.ckpt (with tabicl.pretrained=true) or tabicl.pit_ckpt, "
            "or select a non-TabICL marginal with data.z_train_source="
            "exaone/tabpfn/tabldm."
        )
    ecfg = _resolve_era5_cfg(cfg)
    k_folds = int(cfg.tabicl.get("pit_k_folds", 10))
    base_seed = int(getattr(cfg, "seed", None) or 0)
    # Group size from training.live_tabicl_group_multiplier.
    group_multiplier = max(1, int(t.get("live_tabicl_group_multiplier", 2)))
    group_size = int(t.batch_size) * group_multiplier

    # Load the corpus into shared memory before workers spawn.
    print(f"[era5_live_dataset] loading global ERA5 corpus into shared memory from {ecfg['corpus_dir']}")
    shared_corpus = load_shared_corpus_arrays(ecfg["corpus_dir"])

    live_ds = LiveERA5Dataset(
        shared_corpus=shared_corpus,
        tabicl_ckpt=ckpt,
        tabicl_device=device,
        k_folds=k_folds,
        grid_size_range=ecfg["grid_size_range"],
        box_deg_range=ecfg["box_deg_range"],
        n_context_frac_range=ecfg["n_context_frac_range"],
        base_seed=base_seed,
        group_size=group_size,
        tabicl_inference_amp=bool(t.get("tabicl_inference_amp", True)),
        marginal_backend=marginal_backend,
        marginal_probs_n=marginal_probs_n,
    )
    # Worker count bound by GPU headroom (the corpus is shared).
    num_workers = resolve_live_tabicl_num_workers(t, device)
    loader = DataLoader(
        live_ds,
        batch_size=t.batch_size,
        collate_fn=era5_collate_fn,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_limit_worker_threads if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    return loader


def build_era5_fixed_val_batches(cfg: DictConfig, t: DictConfig, device: str = "cpu") -> List[dict]:
    """Fixed ERA5 validation batches, drawn once with era5_live.val_seed.

    Uses era5_live.val_corpus_dir (e.g. a held-out year) when set, else corpus_dir.
    """
    import numpy as np

    ecfg = _resolve_era5_cfg(cfg)
    marginal_backend, marginal_probs_n = _resolve_marginal(cfg)
    ckpt = resolve_pit_ckpt(cfg)
    if ckpt is None and marginal_backend is None:
        raise ValueError(
            "training.live_source=era5 requires a resolvable TabICL checkpoint — "
            "set tabicl.ckpt (with tabicl.pretrained=true) or tabicl.pit_ckpt, "
            "or select a non-TabICL marginal with data.z_train_source="
            "exaone/tabpfn/tabldm."
        )
    k_folds = int(cfg.tabicl.get("pit_k_folds", 10))
    tabicl_model = None
    marginal_regressor = None
    if marginal_backend is None:
        print(f"[era5_live_dataset] Loading frozen TabICL marginal for fixed val batches: {ckpt}")
        tabicl_model = load_tabicl(ckpt, device)
    else:
        from eval.spatial.marginal_backends import make_regressor

        print(f"[era5_live_dataset] Building {marginal_backend} marginal for fixed val batches on {device}")
        marginal_regressor = make_regressor(marginal_backend, device=device)
    val_corpus_dir = ecfg["val_corpus_dir"] or ecfg["corpus_dir"]
    print(f"[era5_live_dataset] Building fixed val batches from corpus: {val_corpus_dir}")
    corpus = GlobalERA5Corpus(val_corpus_dir)

    batch_size = int(t.batch_size)
    n_val = ecfg["val_episodes"]
    n_batches = max(1, (n_val + batch_size - 1) // batch_size)
    rng = np.random.default_rng(ecfg["val_seed"])

    batches = []
    with torch.no_grad():
        for _ in range(n_batches):
            episodes = []
            attempts = 0
            while len(episodes) < batch_size and attempts < batch_size * 20:
                attempts += 1
                ep = corpus.sample_episode(
                    rng,
                    ecfg["grid_size_range"],
                    ecfg["box_deg_range"],
                    ecfg["n_context_frac_range"],
                )
                if ep is None:
                    continue
                x_train = torch.as_tensor(ep["x_norm_train"], dtype=torch.float32, device=device)
                x_test = torch.as_tensor(ep["x_norm_test"], dtype=torch.float32, device=device)
                y_train = torch.as_tensor(ep["y_train"], dtype=torch.float32, device=device)
                y_test = torch.as_tensor(ep["y_test"], dtype=torch.float32, device=device)
                pit = _pit_episode(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    tabicl_model,
                    k_folds,
                    marginal_backend=marginal_backend,
                    marginal_regressor=marginal_regressor,
                    marginal_probs_n=marginal_probs_n,
                    seed=int(ecfg["val_seed"]),
                )
                if pit is None:
                    continue
                episodes.append(
                    {
                        "x_norm_train": x_train.cpu(),
                        "x_norm_test": x_test.cpu(),
                        "y_train": y_train.cpu(),
                        "y_test": y_test.cpu(),
                        "z_train": pit["z_train"].cpu(),
                        "z_test": pit["z_test"].cpu(),
                        "log_pdf_test": pit["log_pdf_test"].cpu(),
                    }
                )
            if episodes:
                batches.append(era5_collate_fn(episodes))
    del tabicl_model
    return batches
