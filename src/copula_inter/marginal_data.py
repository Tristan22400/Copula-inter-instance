"""Phase-A episodes: synthetic GP batches with a pinned (P, N, d), the ERA5 sampler and the fixed validation batches."""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from copula_inter.data_gen import generate_gp_batch  # noqa: E402
from copula_inter.pit import (
    normalize_targets,
)

if TYPE_CHECKING:
    from eval.data.era5_global_corpus import GlobalERA5Corpus


def stack_episodes(episodes: Sequence[dict], device: str | torch.device) -> dict:
    """Stack same-shape generate_gp_batch episodes into (B, ...) tensors plus each episode's normalize_targets scale."""
    x_train = torch.stack([e["x_norm_train"] for e in episodes]).to(device)  # (B,P,d)
    y_train = torch.stack([e["y_train"] for e in episodes]).to(device)  # (B,P)
    x_test = torch.stack([e["x_norm_test"] for e in episodes]).to(device)  # (B,N,d)
    y_test = torch.stack([e["y_test"] for e in episodes]).to(device)  # (B,N)

    y_tr_s, y_te_s, means, stds = [], [], [], []
    for b in range(len(episodes)):
        a, c, m, s = normalize_targets(y_train[b], y_test[b])
        y_tr_s.append(a)
        y_te_s.append(c)
        means.append(m)
        stds.append(s)
    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train_raw": y_train,
        "y_test_raw": y_test,
        "y_train_scaled": torch.stack(y_tr_s),
        "y_test_scaled": torch.stack(y_te_s),
        "y_mean": torch.stack(means),
        "y_std": torch.stack(stds),
    }


class ERA5EpisodeSampler:
    """Batches of real-ERA5 episodes with one shared P and N (sample_episode_fixed_shape); region, day and box vary."""

    def __init__(
        self, corpus: GlobalERA5Corpus, *, grid_size: int, n_context: int, box_deg_range: tuple[float, float], seed: int
    ) -> None:
        self.corpus = corpus
        self.grid_size = int(grid_size)
        self.n_context = int(n_context)
        self.box_deg_range = box_deg_range
        self.rng = np.random.default_rng(seed)

    def batch(self, B: int, max_tries: int = 200) -> list[dict]:
        out: list[dict] = []
        tries = 0
        while len(out) < B and tries < max_tries * B:
            tries += 1
            ep = self.corpus.sample_episode_fixed_shape(self.rng, self.grid_size, self.box_deg_range, self.n_context)
            if ep is None:
                continue
            out.append(
                {
                    "x_norm_train": torch.as_tensor(ep["x_norm_train"]),
                    "y_train": torch.as_tensor(ep["y_train"]),
                    "x_norm_test": torch.as_tensor(ep["x_norm_test"]),
                    "y_test": torch.as_tensor(ep["y_test"]),
                }
            )
        if len(out) < B:
            raise RuntimeError(
                f"ERA5 sampler produced {len(out)}/{B} episodes in {tries} draws at "
                f"grid_size={self.grid_size}, n_context={self.n_context}, "
                f"box_deg_range={self.box_deg_range}. Widen box_deg_max or lower "
                f"grid_size."
            )
        return out


def _gp_cfg(cfg: DictConfig) -> DictConfig:
    """A config with the data group and a seed, as generate_gp_batch expects."""
    return OmegaConf.create({"data": OmegaConf.to_container(cfg.data, resolve=True), "seed": int(cfg.seed)})


def _generate_phase_a_gp_batch(gp_cfg: DictConfig, batch_size: int, device: str, *, max_rounds: int = 20) -> list[dict]:
    """generate_gp_batch with P, N and d pinned after the first call, so every episode has the same shape."""
    episodes = generate_gp_batch(gp_cfg, batch_size, device, return_kernel_metadata=True)
    P = int(episodes[0]["x_norm_train"].shape[0])
    N = int(episodes[0]["x_norm_test"].shape[0])
    d = int(episodes[0]["x_norm_train"].shape[1])
    out = [ep for ep in episodes if ep["x_norm_train"].shape == (P, d) and ep["x_norm_test"].shape == (N, d)]
    if len(out) == batch_size:
        return out

    fixed = OmegaConf.create(OmegaConf.to_container(gp_cfg, resolve=True))
    fixed.data.P_min = fixed.data.P_max = P
    fixed.data.N_min = fixed.data.N_max = N
    base_seed = int(gp_cfg.seed)
    for round_idx in range(1, max_rounds + 1):
        fixed.seed = base_seed + round_idx * 1_000_003
        out.extend(
            generate_gp_batch(
                fixed,
                batch_size - len(out),
                device,
                return_kernel_metadata=True,
                d_override=d,
            )
        )
        if len(out) >= batch_size:
            return out[:batch_size]
    raise RuntimeError(
        f"Phase-A GP generator produced only {len(out)}/{batch_size} episodes "
        f"with fixed shape P={P}, N={N}, d={d} after {max_rounds} retries."
    )


def _build_gp_val_batches(cfg: DictConfig, device: str) -> list[list[dict]]:
    """Fixed synthetic GP validation batches, drawn once with their own seed."""
    gp_cfg = _gp_cfg(cfg)
    batches = []
    for i in range(int(cfg.validation.gp_n_batches)):
        gp_cfg.seed = int(cfg.validation.gp_seed) + i
        batches.append(
            _generate_phase_a_gp_batch(
                gp_cfg,
                int(cfg.validation.gp_batch_size),
                device,
            )
        )
    return batches


def build_era5_marginal_val_batches(vcfg: DictConfig, device: str | torch.device) -> dict:
    """Fixed per-region real-ERA5 probes for Phase-A validation, holding raw (x, y).

    Same geometry and seeds as the copula run's ERA5 probes
    (sweep_core.build_era5_probe with tabicl_marginal=None), without correlation
    or GP-baseline fields. Data comes from the held-out 2023 period.
    """
    from eval.configs.regions import REGIONS as ERA5_REGIONS
    from eval.spatial.sweep_core import build_era5_probe

    def _g(key: str, default: Any) -> Any:
        return vcfg.get(key, default) if hasattr(vcfg, "get") else getattr(vcfg, key, default)

    region_names = list(_g("era5_regions", []) or list(ERA5_REGIONS.keys()))
    grid_size = int(_g("era5_grid_size", 24))
    n_days_fetch = int(_g("era5_n_days_fetch", 60))
    n_days_probe = int(_g("era5_n_days_probe", 3))
    n_context = int(_g("era5_n_context", 30))
    base_seed = int(_g("era5_seed", 20260818))

    batches: dict[str, dict] = {}
    for region in region_names:
        if region not in ERA5_REGIONS:
            continue  # not a registered eval/configs/regions.py entry
        # zlib.crc32 seed, same as probe_batches._name_seed, so both phases use the same points.
        seed = base_seed + (zlib.crc32(region.encode()) % 10_000)
        probe = build_era5_probe(
            region,
            grid_size,
            n_days_fetch,
            n_days_probe,
            n_context,
            n_bins=12,
            tabicl_marginal=None,
            device=str(device),
            seed=seed,
        )
        n_days = probe["context_values_per_day"].shape[0]
        x_tr = torch.as_tensor(probe["x_train_norm"], dtype=torch.float32, device=device)
        x_te = torch.as_tensor(probe["x_nll_test_norm"], dtype=torch.float32, device=device)
        batches[region] = {
            "x_train": x_tr.unsqueeze(0).expand(n_days, -1, -1).contiguous(),
            "x_test": x_te.unsqueeze(0).expand(n_days, -1, -1).contiguous(),
            "y_train": torch.as_tensor(probe["context_values_per_day"], dtype=torch.float32, device=device),
            "y_test": torch.as_tensor(probe["nll_test_values_per_day"], dtype=torch.float32, device=device),
        }
    return batches
