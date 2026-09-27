"""eval_checkpoint inputs.

The episode-generating config, the copula and marginal models, and the
evaluation episodes (ERA5, live-generated or on disk).
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from omegaconf import DictConfig, OmegaConf

from eval.baselines.prefit import (
    _baseline_fit_seed,
)

if TYPE_CHECKING:
    from copula_inter.model import CopulaTabICL
    from copula_inter.type_aliases import Device
    from eval.runners.eval_args import EvalSpec

from copula_inter.backend_registry import GENERIC_MARGINAL_BACKENDS
from copula_inter.config_path import compose_config
from copula_inter.config_path import config_dir as project_config_dir
from copula_inter.data_gen import generate_gp_batch
from copula_inter.dataset import CopulaDataset
from copula_inter.gp_kernels import _parse_composite
from copula_inter.pit import (
    DEFAULT_K_FOLDS,
    TabICLLike,
    configure_tabicl_inference_amp,
    load_tabicl,
)
from eval.baselines.classical import (
    episode_cache_key,
)
from eval.configs.checkpoints import (
    DEFAULT_MARGINAL_FAMILY,
    resolve_marginal_checkpoint,
)
from eval.data.era5_episodes import (
    build_era5_eval_episodes,
)
from inference.copula_inference import load_copula_model


def resolve_config_path(config_path: str) -> str:
    """The default relative "conf/config.yaml" names the project config wherever the runner is launched from."""
    if config_path == "conf/config.yaml" and not os.path.isfile(config_path):
        return os.path.join(project_config_dir(__file__), "config.yaml")
    return config_path


def _load_full_config(config_path: str) -> DictConfig:
    """Compose the episode config through Hydra's defaults list (model and data groups), independent of any checkpoint."""
    config_path = os.path.abspath(resolve_config_path(config_path))
    return compose_config(os.path.dirname(config_path), os.path.splitext(os.path.basename(config_path))[0])


def _dataset_dir_for_eval(spec: EvalSpec, cfg: DictConfig, live_generate: bool, era5: bool) -> str | None:
    """The on-disk dataset directory used for loading and cache keys."""
    return None if live_generate or era5 else str(spec.dataset_dir or cfg.training.dataset_dir)


def _live_generate_alternating(
    gen_cfg: DictConfig,
    n_ep: int,
    device: Device,
    seed: int,
    offset: int = 0,
    alternate_noncomposite: bool = True,
) -> list[dict]:
    """Live-generate n_ep episodes, one generate_gp_batch call each with seed + global index.

    With alternate_noncomposite, even global indices use a single elementary
    kernel. The global index is offset + local index, so sharded runs
    (episode_offset) produce the same episodes.
    """
    episodes: list[dict] = []
    for local_i in range(n_ep):
        global_i = offset + local_i
        ep_cfg = copy.deepcopy(gen_cfg)
        ep_cfg.seed = seed + global_i
        if alternate_noncomposite and global_i % 2 == 0:
            # Force a non-composite kernel for both kernel-selection modes.
            if bool(getattr(ep_cfg.data, "systematic_composition", False)):
                ep_cfg.data.composite_num_kernels_min = 1
                ep_cfg.data.composite_num_kernels_max = 1
            else:
                fixed = getattr(ep_cfg.data, "kernel", None)
                if fixed:
                    composite = _parse_composite(str(fixed))
                    if composite is not None:
                        ep_cfg.data.kernel = composite[0]
                elif getattr(ep_cfg.data, "kernels", None):
                    pool = [k for k in ep_cfg.data.kernels if _parse_composite(k) is None]
                    if not pool:
                        raise ValueError(
                            f"cfg.data.kernels={list(ep_cfg.data.kernels)} contains only "
                            "composite kernels; cannot force a non-composite episode."
                        )
                    ep_cfg.data.kernels = pool
                # else: _resolve_kernel_name's own "rbf" default, already non-composite.
        episodes.extend(generate_gp_batch(ep_cfg, 1, device, return_kernel_metadata=True))
    return episodes


def _load_episodes(
    spec: EvalSpec,
    cfg: DictConfig,
    dataset_dir: str | None,
    device: torch.device,
    era5: bool,
    live_generate: bool,
    marginal_backend: str | None,
    marginal_regressor: Any,
    min_test_points: int,
    n_ep: int,
    tabicl_marginal: TabICLLike | None,
    tabicl_pit_k_folds: int,
) -> tuple[CopulaDataset | None, dict[str, Any] | None, list[dict] | None, Any, int | None, TabICLLike | None]:
    """Load the evaluation episodes: fixed-geometry ERA5, live-generated GP, or an on-disk dataset."""
    dataset = None
    era5_geometry: dict[str, Any] | None = None
    live_episodes = None
    n_available = None
    if era5:
        print(
            f"\nBuilding {n_ep} REAL ARCO-ERA5 episodes, seed={spec.seed}, "
            f"global indices {spec.episode_offset}..{spec.episode_offset + n_ep - 1}"
        )
        print(f"  corpus={spec.era5.corpus_dir}")
        if spec.era5.vary_geometry:
            print("  geometry: per-episode grid_size/context fraction (era5_live ranges)")
        else:
            print(
                f"  geometry: fixed grid={spec.era5.grid_size} "
                f"(D={spec.era5.grid_size**2}), P={spec.era5.n_context}, "
                f"N={spec.era5.grid_size**2 - spec.era5.n_context}, "
                f"box {spec.era5.box_deg_min}..{spec.era5.box_deg_max} deg"
            )
        # One geometry dict shared by the episode builder and the cache fingerprint.
        era5_geometry = dict(
            grid_size=spec.era5.grid_size,
            n_context=spec.era5.n_context,
            box_deg_range=(spec.era5.box_deg_min, spec.era5.box_deg_max),
            vary_geometry=spec.era5.vary_geometry,
            grid_size_range=(
                int(OmegaConf.select(cfg, "era5_live.grid_size_min", default=8)),
                int(OmegaConf.select(cfg, "era5_live.grid_size_max", default=28)),
            ),
            n_context_frac_range=(
                float(OmegaConf.select(cfg, "era5_live.n_context_frac_min", default=0.05)),
                float(OmegaConf.select(cfg, "era5_live.n_context_frac_max", default=0.4)),
            ),
            max_months=spec.era5.max_months,
            standardize_y=spec.era5.standardize_y,
        )
        live_episodes = build_era5_eval_episodes(
            spec.era5.corpus_dir,
            n_ep,
            seed=spec.seed,
            offset=spec.episode_offset,
            tabicl_model=tabicl_marginal,
            k_folds=tabicl_pit_k_folds,
            device=device,
            pit_group_size=spec.era5.pit_batch,
            marginal_backend=marginal_backend,
            marginal_regressor=marginal_regressor,
            marginal_probs_n=spec.marginal.probs_n,
            autoregressive=spec.autoregressive.enabled,
            ar_order=spec.autoregressive.order,
            ar_conditioning=spec.autoregressive.conditioning,
            ar_max_context=spec.autoregressive.max_context,
            ar_n_episodes=spec.autoregressive.n_episodes,
            **era5_geometry,
        )
        # Free the marginal after the PIT.
        tabicl_marginal = None
        marginal_regressor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif live_generate:
        # Episodes come from cfg (the eval config), not the checkpoint's config.
        if cfg.data.N_min < min_test_points:
            print(
                f"Raising eval episode N_min {cfg.data.N_min} -> {min_test_points} "
                f"(selection.min_test_points) for this run only — training's own "
                "conf/data/gp_tasks.yaml N_min is untouched"
            )
            cfg.data.N_min = min_test_points
            if cfg.data.N_max < cfg.data.N_min:
                cfg.data.N_max = cfg.data.N_min
        print(
            f"\nLive-generating {n_ep} episodes via generate_gp_batch "
            f"(return_kernel_metadata=True), seed={spec.seed}, "
            f"global indices {spec.episode_offset}..{spec.episode_offset + n_ep - 1}, "
            "alternating every-other episode to a non-composite kernel"
        )
        live_episodes = _live_generate_alternating(
            cfg,
            n_ep,
            device,
            spec.seed,
            offset=spec.episode_offset,
            alternate_noncomposite=spec.alternate_noncomposite,
        )
    else:
        dataset = CopulaDataset(episode_dir=dataset_dir)
        n_available = len(dataset)
        print(f"\nEvaluating {n_ep} episodes from {dataset_dir} (start={spec.episode_idx})")
        print(f"  Dataset size: {n_available} episodes")
    return dataset, era5_geometry, live_episodes, marginal_regressor, n_available, tabicl_marginal


def _load_models(
    spec: EvalSpec, cfg: DictConfig, device: torch.device
) -> tuple[CopulaTabICL, int, str | None, Any, str | None, TabICLLike | None, int]:
    """Load the copula checkpoint and the marginal that PITs each episode's z_train."""
    tabicl_ckpt = None
    # ---- Load ICL model ----
    print(f"\nLoading ICL checkpoint: {spec.ckpt}")
    icl_model, icl_cfg = load_copula_model(spec.ckpt, config_path=spec.config, device=str(device))
    icl_rank = int(icl_cfg.model.rank)
    n_params = sum(p.numel() for p in icl_model.parameters())
    print(f"ICL model parameters: {n_params:,}  rank={icl_rank}")

    tabicl_marginal: TabICLLike | None = None
    marginal_backend: str | None = (
        spec.marginal.z_train_source if spec.marginal.z_train_source in GENERIC_MARGINAL_BACKENDS else None
    )
    marginal_regressor = None
    tabicl_pit_k_folds = DEFAULT_K_FOLDS
    configured_k_folds = spec.marginal.tabicl_pit_k_folds or int(
        OmegaConf.select(cfg, "tabicl.pit_k_folds", default=DEFAULT_K_FOLDS)
    )
    if marginal_backend is not None:
        from eval.spatial.marginal_backends import make_regressor

        tabicl_pit_k_folds = configured_k_folds
        print(
            f"\nBuilding {marginal_backend} marginal for marginal.z_train_source={marginal_backend} "
            f"(k_folds={tabicl_pit_k_folds}, probs_n={spec.marginal.probs_n})"
        )
        marginal_regressor = make_regressor(marginal_backend, device=str(device))
    elif spec.marginal.z_train_source == "tabicl":
        # marginal.tabicl_ckpt, then cfg.tabicl.ckpt, then DEFAULT_MARGINAL_FAMILY.
        tabicl_ckpt = (
            spec.marginal.tabicl_ckpt or OmegaConf.select(cfg, "tabicl.ckpt", default=None) or DEFAULT_MARGINAL_FAMILY
        )
        tabicl_ckpt = resolve_marginal_checkpoint(str(tabicl_ckpt))
        if not tabicl_ckpt:
            raise ValueError(
                "marginal.z_train_source=tabicl requires a TabICL checkpoint: set "
                "marginal.tabicl_ckpt or tabicl.ckpt in the episode config."
            )
        tabicl_pit_k_folds = configured_k_folds
        print(
            f"\nLoading frozen TabICL marginal for marginal.z_train_source=tabicl: {tabicl_ckpt} "
            f"(k_folds={tabicl_pit_k_folds})"
        )
        tabicl_marginal = load_tabicl(tabicl_ckpt, str(device))
        configure_tabicl_inference_amp(spec.marginal.tabicl_amp)
        print(f"Frozen TabICL marginal inference AMP={'on' if spec.marginal.tabicl_amp else 'off (float32)'}")
    return icl_model, icl_rank, marginal_backend, marginal_regressor, tabicl_ckpt, tabicl_marginal, tabicl_pit_k_folds


def _resolve_episode_source(spec: EvalSpec, cfg: DictConfig) -> tuple[bool, bool, str | None]:
    """(era5, live_generate, dataset_dir) from era5.enabled, dataset_dir and live_generate."""
    era5 = bool(spec.era5.enabled)
    if era5:
        live_generate = False
    else:
        live_generate = spec.live_generate if spec.live_generate is not None else (spec.dataset_dir is None)
    return era5, live_generate, _dataset_dir_for_eval(spec, cfg, live_generate, era5)


class _PlannedEpisode(NamedTuple):
    local_i: int
    ep_i: int
    cache_key: str
    ep: dict
    fit_seed: int


def _plan_episodes(
    spec: EvalSpec,
    *,
    era5: bool,
    live_generate: bool,
    dataset_dir: str | None,
    dataset: CopulaDataset | None,
    live_episodes: list[dict] | None,
    n_available: int | None,
    min_test_points: int,
) -> list[_PlannedEpisode]:
    """Every episode to evaluate, with its cache key; skips out-of-range or too-small dataset episodes."""
    episode_plan: list[_PlannedEpisode] = []
    for local_i in range(spec.n_episodes):
        if era5:
            assert live_episodes is not None
            # ERA5 may return fewer episodes; use each episode's own global index.
            if local_i >= len(live_episodes):
                continue
            ep = live_episodes[local_i]
            ep_i = int(ep["era5_meta"]["ep_i"])
        elif live_generate:
            assert live_episodes is not None
            # Global index (offset + local index).
            ep_i = spec.episode_offset + local_i
            ep = live_episodes[local_i]
        else:
            assert dataset is not None and n_available is not None
            ep_i = spec.episode_idx + local_i
            if ep_i >= n_available:
                print(f"  [ep {ep_i}] index out of range ({n_available} available), skipping")
                continue
            ep = dataset[ep_i]
            n_test = ep["z_test"].shape[0]
            if n_test < min_test_points:
                print(
                    f"  [ep {ep_i}] only {n_test} test points (< selection.min_test_points="
                    f"{min_test_points}), skipping — best_baseline needs enough for "
                    ">=2 nested-CV folds"
                )
                continue
        cache_key = episode_cache_key(
            live_generate,
            dataset_dir,
            spec.seed,
            ep_i,
            source="era5" if era5 else None,
        )
        episode_plan.append(_PlannedEpisode(local_i, ep_i, cache_key, ep, _baseline_fit_seed(spec.seed, cache_key)))
    return episode_plan
