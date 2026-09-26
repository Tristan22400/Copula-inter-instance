"""Evaluate a copula checkpoint against the classical baselines and the oracle.

Usage:
    python eval/runners/eval_checkpoint.py --ckpt kernel-sweep-all-tabicl-retrain-15k
    python eval/runners/eval_checkpoint.py --era5 --n_episodes 400 --ckpt <ckpt>

See --help for every flag. Episodes are generated from --config's data config
(not the checkpoint's), so the same --config and --seed give the same
episodes for any checkpoint. Baseline fits (classical.py) do not depend on
the checkpoint and are cached per episode in --baseline_cache; scored results
are cached per checkpoint in --results_cache. Baselines are fitted on CPU
(--baseline_device) across --baseline_workers processes, one per physical
core by default.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from eval.baselines.prefit import (
    _NON_FITTED_EXCLUDED,
    _baseline_fit_seed,
    _count_physical_cores,
    _eval_zero_mean_gp_baselines,
    _prefit_baselines_parallel,
    _select_best_baseline_cv,
    _valid_cached_entry,
)
from eval.runners.eval_tables import (
    _GP_TOTAL_KEYS,
    _METHOD_LABELS,
    _RANK_KEYS,
    _TOTAL_RANK_ORDER,
    _ar_note,
    _kernel_composition_label,
    _print_rank_table,
    _print_table,
    _print_total_nll_table,
    _print_y_space_oracle,
)

if TYPE_CHECKING:
    from copula_inter.model import CopulaTabICL
    from copula_inter.type_aliases import Device

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

from copula_inter.artifacts import artifact_identity, atomic_json_save  # noqa: E402
from copula_inter.backend_registry import GENERIC_MARGINAL_BACKENDS  # noqa: E402
from copula_inter.config_path import config_dict  # noqa: E402
from copula_inter.config_path import config_dir as project_config_dir  # noqa: E402
from copula_inter.data_gen import generate_gp_batch  # noqa: E402
from copula_inter.dataset import CopulaDataset  # noqa: E402
from copula_inter.gp_kernels import _parse_composite  # noqa: E402
from copula_inter.loss import y_space_nll  # noqa: E402
from copula_inter.model import low_rank_correlation  # noqa: E402
from copula_inter.pit import (  # noqa: E402
    DEFAULT_K_FOLDS,
    TabICLLike,
    configure_tabicl_inference_amp,
    gp_analytical_posterior,
    load_tabicl,
    normalize_targets,
    run_pit,
)
from eval.baselines.autoregressive import (  # noqa: E402
    ar_parts_from_log_pdf,
    autoregressive_log_pdf,
)
from eval.baselines.classical import (  # noqa: E402
    assert_shared_z_test,
    baseline_fingerprint,
    corr_nll_single,
    episode_cache_key,
    eval_baselines_episode,
    load_baseline_cache,
    save_baseline_entry,
)
from eval.configs.checkpoints import (  # noqa: E402
    DEFAULT_MARGINAL_FAMILY,
    resolve_marginal_checkpoint,
)
from eval.data.era5_episodes import (
    build_era5_eval_episodes,
    era5_episode_fingerprint,
)
from eval.results import (
    NAN_PARTS as _NAN_PARTS,
)
from eval.results import (
    jsonable as _jsonable,
)
from eval.results import (
    load_results_cache as _load_results_cache,
)
from eval.results import (  # noqa: E402
    require_coverage,
)
from eval.results import (
    save_results_cache as _save_results_cache,
)
from eval.runners.eval_args import _validate_eval_spec, parse_eval_spec  # noqa: E402
from eval.viz.correlation_plots import plot_corr_grid  # noqa: E402
from inference.copula_inference import load_copula_model  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def _snapshot_rng_and_threads(seed: int | None = None, cpu_threads: int | None = None) -> Iterator[None]:
    """Context manager: save and restore the torch RNG; optionally reseed and cap intra-op threads meanwhile."""
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    prev_threads = torch.get_num_threads() if cpu_threads is not None else None
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    if seed is not None:
        torch.manual_seed(seed)
    try:
        yield
    finally:
        if prev_threads is not None:
            torch.set_num_threads(prev_threads)
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)


def _load_full_config(config_path: str) -> DictConfig:
    """Compose --config through Hydra's defaults list (model and data groups), independent of any checkpoint."""
    if config_path == "conf/config.yaml" and not os.path.isfile(config_path):
        config_path = os.path.join(project_config_dir(__file__), "config.yaml")
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        return hydra.compose(config_name=config_name)


def _dataset_dir_for_eval(args: argparse.Namespace, cfg: DictConfig, live_generate: bool, era5: bool) -> str | None:
    """The on-disk dataset directory used for loading and cache keys."""
    return None if live_generate or era5 else str(args.dataset_dir or cfg.training.dataset_dir)


def _eval_icl_episode(
    ep: dict,
    icl_model: nn.Module,
    device: torch.device,
    marginal_pit: dict[str, Tensor] | None = None,
) -> tuple[dict[str, float], dict[str, Tensor], Tensor | None, dict[str, dict[str, float]], dict[str, float]]:
    """Score the ICL model and the oracle on one episode.

    marginal_pit, when given, replaces the episode's z_train as the model input
    and supplies the marginal (z_test, log_pdf_test) for the model's Y-space NLL.

    Returns:
        (nlls, R_dict, R_oracle, y_space_nlls, icl_y_parts): nlls holds the
        shared-z_test copula NLLs of icl and oracle; y_space_nlls the analytic GP
        prior/posterior Y-space NLLs ({total, marginal, copula}, NaN when
        unavailable); icl_y_parts the model's own Y-space split (NaN without
        marginal_pit).
    """
    X_train = ep["x_norm_train"].to(device)  # (P, d_x)
    z_train = (
        marginal_pit["z_train"].to(device) if marginal_pit is not None else ep["z_train"].to(device)
    )  # (P,)  ICL's conditioning input — oracle LOO-PIT residual by default
    X_test = ep["x_norm_test"].to(device)  # (N, d_x)
    z_test = ep["z_test"].to(device)  # (N,)
    assert_shared_z_test(z_test, ep)
    # Real-ERA5 episodes have no generating kernel: skip the oracle part.
    R_oracle = ep["R_star"].to(device) if "R_star" in ep else None  # (N, N) or None

    P, N = X_train.shape[0], X_test.shape[0]
    nlls: dict[str, float] = {}
    R_dict: dict[str, Tensor] = {}
    R_I = torch.eye(N, dtype=X_train.dtype, device=device)
    icl_y_parts = _NAN_PARTS.copy()

    try:
        train_mask = torch.ones(1, P, dtype=torch.bool, device=device)
        batch = {
            "x_train": X_train.unsqueeze(0),
            "x_test": X_test.unsqueeze(0),
            "z_train": z_train.unsqueeze(0),
            "train_mask": train_mask,
        }
        with torch.no_grad():
            out = icl_model(batch)
            Sigma_icl = low_rank_correlation(
                out["W"],
                out.get("s"),
                parametrization=getattr(icl_model, "correlation_parametrization", "covnorm"),
                lam=out.get("lam"),
            )  # (1, N, N)
        R_icl = Sigma_icl[0, :N, :N]
        nlls["icl"] = corr_nll_single(R_icl, z_test)
        R_dict["icl"] = R_icl
        if marginal_pit is not None:
            test_mask = torch.ones(1, N, dtype=torch.bool, device=device)
            icl_parts = y_space_nll(
                Sigma_icl[:, :N, :N],
                marginal_pit["z_test"].to(device).unsqueeze(0),
                marginal_pit["log_pdf_test"].to(device).unsqueeze(0),
                test_mask,
            )
            icl_y_parts = {k: v.item() for k, v in icl_parts.items()}
    except Exception as exc:
        print(f"  [icl] failed: {exc}")
        nlls["icl"] = float("nan")
        R_dict["icl"] = R_I.clone()

    if R_oracle is not None:
        nlls["oracle"] = corr_nll_single(R_oracle, z_test)
        R_dict["oracle"] = R_oracle

    # "oracle" is the prior reference (R_star). The posterior Y-space reference
    # comes from gp_analytical_posterior; R_post is only kept for plotting.
    y_space_nlls = {"prior": _NAN_PARTS.copy(), "posterior": _NAN_PARTS.copy()}
    try:
        if R_oracle is None:
            raise NotImplementedError(
                "no generating kernel (real-data episode) -- no analytic GP prior/posterior reference exists"
            )
        post = gp_analytical_posterior(ep)
        R_dict["oracle_posterior"] = post["R_post"].to(device)
        y_space_nlls = {
            "prior": {
                "total": post["nll_prior"],
                "marginal": post["nll_prior_marginal"],
                "copula": post["nll_prior_copula"],
            },
            "posterior": {
                "total": post["nll_post"],
                "marginal": post["nll_post_marginal"],
                "copula": post["nll_post_copula"],
            },
        }
        if post["repaired"]:
            print(f"    [oracle_posterior] PSD repair fired (min_eig={post['min_eig']:.2e})")
    except (KeyError, NotImplementedError) as exc:
        R_dict["oracle_posterior"] = R_I.clone()
        print(f"  [oracle_posterior] unavailable: {exc}")

    return nlls, R_dict, R_oracle, y_space_nlls, icl_y_parts


def _marginal_pit(
    ep: dict,
    tabicl_marginal: TabICLLike | None,
    k_folds: int,
    device: torch.device,
    marginal_backend: str | None = None,
    marginal_regressor: Any = None,
    marginal_probs_n: int = 99,
    seed: int = 0,
) -> dict[str, Tensor] | None:
    """K-fold PIT of one episode through a real marginal (TabICL or a batched backend).

    Returns {z_train, z_test, log_pdf_test} with log_pdf_test in raw nats, or None
    for fewer than 2 training points.
    """
    X_train = ep["x_norm_train"].to(device)  # (P, d_x)
    y_train = ep["y_train"].to(device)  # (P,)
    X_test = ep["x_norm_test"].to(device)  # (N, d_x)
    y_test = ep["y_test"].to(device)  # (N,)
    P = X_train.shape[0]
    if P < 2:
        return None
    y_train_scaled, y_test_scaled, _, std = normalize_targets(y_train, y_test)
    if marginal_backend is not None:
        # Batched backend PIT with a singleton episode axis.
        from copula_inter.data_gen import _BATCHED_MARGINAL_BACKENDS

        run_batched = _BATCHED_MARGINAL_BACKENDS[marginal_backend]()
        out = run_batched(
            marginal_regressor,
            X_train.unsqueeze(0).cpu().numpy(),
            y_train_scaled.unsqueeze(0).cpu().numpy(),
            X_test.unsqueeze(0).cpu().numpy(),
            y_test_scaled.unsqueeze(0).cpu().numpy(),
            k_folds=k_folds,
            probs_n=marginal_probs_n,
            seed=seed,
        )
        as_t = lambda a: torch.as_tensor(a[0], dtype=torch.float32, device=device)  # noqa: E731
        return {
            "z_train": as_t(out["z_train"]),
            "z_test": as_t(out["z_test"]),
            "log_pdf_test": as_t(out["log_pdf_test"]) - std.log(),
        }
    Y_train = y_train_scaled.unsqueeze(-1)  # (P, 1)
    Y_test = y_test_scaled.unsqueeze(-1)  # (N, 1)
    assert tabicl_marginal is not None
    pit_out = run_pit(
        tabicl_marginal,
        X_train,
        Y_train,
        X_test,
        Y_test,
        k_folds=k_folds,
        Y_train_raw=y_train.unsqueeze(-1),
    )
    return {
        "z_train": pit_out["z_train"].squeeze(-1),  # (P,)
        "z_test": pit_out["z_test"].squeeze(-1),  # (N,)
        "log_pdf_test": pit_out["log_pdf_test"].squeeze(-1) - std.log(),  # (N,) raw-nats
    }


def _results_fingerprint(
    baseline_fp: dict,
    args: argparse.Namespace,
    tabicl_pit_k_folds: int,
    resolved_marginal: str | None = None,
) -> dict:
    """Digest of everything that determines an episode's scored results: the baseline fingerprint plus the checkpoint, marginal and CV settings."""
    ckpt = os.path.abspath(args.ckpt) if args.ckpt else None
    return {
        "baseline": baseline_fp,
        "ckpt_identity": artifact_identity(ckpt),
        "z_train_source": args.z_train_source,
        "marginal_identity": artifact_identity(
            resolved_marginal if resolved_marginal is not None else args.tabicl_ckpt
        ),
        "tabicl_pit_k_folds": tabicl_pit_k_folds,
        "tabicl_amp": args.tabicl_amp,
        "marginal_probs_n": args.marginal_probs_n,
        "n_folds": args.n_folds,
        "min_fold_size": args.min_fold_size,
        "seed": args.seed,
        "zeromean_gp": args.zeromean_gp,
        "n_steps_zeromean_gp": args.n_steps_zeromean_gp,
        "lr_zeromean_gp": args.lr_zeromean_gp,
        "n_restarts_zeromean_gp": args.n_restarts_zeromean_gp,
        # Bump when the derived total-NLL or rank rows change.
        "result_schema": 3,
        "autoregressive": bool(args.autoregressive),
        "ar_order": args.ar_order if args.autoregressive else None,
        "ar_conditioning": args.ar_conditioning if args.autoregressive else None,
        "ar_max_context": args.ar_max_context if args.autoregressive else None,
        "ar_n_episodes": args.ar_n_episodes if args.autoregressive else None,
    }


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
    (--episode_offset) produce the same episodes.
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


def _report_results(
    all_episode_meta: list[dict],
    all_nlls: list[dict[str, float]],
    all_total_nlls: list[dict[str, dict[str, float]]],
    all_y_space_nlls: list[dict[str, dict[str, float]]],
    args: argparse.Namespace,
    dataset_dir: str | None,
    era5: bool,
    live_generate: bool,
    n_ep: int,
    plot_R_dict: dict[str, Tensor] | None,
    plot_R_oracle: Tensor | None,
    plot_best_R: Tensor | None,
    plot_best_key: str | None,
) -> None:
    """Print the result tables, enforce --min_icl_coverage, dump per-episode scores, and plot correlation grids."""
    if not all_nlls:
        raise RuntimeError(f"no episodes evaluated successfully out of {n_ep} requested")

    _print_table(all_nlls, z_train_source=args.z_train_source, era5=era5, attempted=n_ep)
    if era5:
        print(
            "GP oracle total NLL (Y-space): unavailable on real ERA5 — that table is "
            "the analytic prior/posterior of the GP that generated the episode, and "
            "no such GP exists here.\n"
        )
    else:
        _print_y_space_oracle(all_y_space_nlls)
    _print_total_nll_table(
        all_total_nlls,
        z_train_source=args.z_train_source,
        era5=era5,
        attempted=n_ep,
        ar_note=_ar_note(all_total_nlls, args.ar_order, args.ar_conditioning, args.ar_max_context),
    )
    _print_rank_table(
        all_nlls,
        _RANK_KEYS,
        title="Method rank — z-space copula NLL (shared ground-truth marginal)",
        z_train_source=args.z_train_source,
    )
    total_only = [
        {k: m.get(k, _NAN_PARTS).get("total", float("nan")) for k, _ in _TOTAL_RANK_ORDER} for m in all_total_nlls
    ]
    _print_rank_table(
        total_only,
        _TOTAL_RANK_ORDER,
        title="Method rank — total NLL, Y-space (own marginal per method)",
        z_train_source=args.z_train_source,
    )

    if args.min_icl_coverage:
        valid_icl = sum(
            bool(np.isfinite(row.get("icl", _NAN_PARTS).get("total", float("nan")))) for row in all_total_nlls
        )
        require_coverage(valid_icl, n_ep, args.min_icl_coverage)

    if args.dump_episodes:
        dump = {
            "ckpt": args.ckpt,
            "config": args.config,
            "seed": args.seed,
            "live_generate": live_generate,
            "dataset_dir": dataset_dir,
            "z_train_source": args.z_train_source,
            "episodes": [
                {**meta, "nlls": nlls, "total_nlls": total_nlls}
                for meta, nlls, total_nlls in zip(all_episode_meta, all_nlls, all_total_nlls)
            ],
        }
        atomic_json_save(dump, args.dump_episodes)
        print(f"Dumped {len(dump['episodes'])} per-episode NLLs to: {args.dump_episodes}")

    # ---- Correlation heatmap ----
    if plot_R_dict is not None and plot_R_oracle is not None:
        import matplotlib

        matplotlib.use("Agg")

        os.makedirs(args.out_dir, exist_ok=True)
        # Plot order: fitted baselines, then oracle, best baseline and icl side by side.
        estimators = {k: v for k, v in plot_R_dict.items() if k != "oracle"}
        icl_panel = estimators.pop("icl", None)
        if plot_best_key is not None and plot_best_R is not None:
            best_label = f"best_baseline ({_METHOD_LABELS.get(plot_best_key, plot_best_key)})"
            estimators[best_label] = plot_best_R
        if icl_panel is not None:
            estimators["icl"] = icl_panel
        fig = plot_corr_grid(
            estimators=estimators,
            oracle_R=plot_R_oracle,
            title=f"Correlation estimators — episode {args.episode_idx + args.plot_episode}",
        )
        out_path = os.path.join(args.out_dir, f"corr_grid_ep{args.plot_episode}.png")
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        print(f"Saved corr_grid to: {out_path}")

    print("Done.")


def _load_episodes(
    args: argparse.Namespace,
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
    era5_geometry = None
    live_episodes = None
    n_available = None
    if era5:
        print(
            f"\nBuilding {n_ep} REAL ARCO-ERA5 episodes, seed={args.seed}, "
            f"global indices {args.episode_offset}..{args.episode_offset + n_ep - 1}"
        )
        print(f"  corpus={args.era5_corpus_dir}")
        if args.era5_vary_geometry:
            print("  geometry: per-episode grid_size/context fraction (era5_live ranges)")
        else:
            print(
                f"  geometry: fixed grid={args.era5_grid_size} "
                f"(D={args.era5_grid_size**2}), P={args.era5_n_context}, "
                f"N={args.era5_grid_size**2 - args.era5_n_context}, "
                f"box {args.era5_box_deg_min}..{args.era5_box_deg_max} deg"
            )
        # One geometry dict shared by the episode builder and the cache fingerprint.
        era5_geometry = dict(
            grid_size=args.era5_grid_size,
            n_context=args.era5_n_context,
            box_deg_range=(args.era5_box_deg_min, args.era5_box_deg_max),
            vary_geometry=args.era5_vary_geometry,
            grid_size_range=(
                int(OmegaConf.select(cfg, "era5_live.grid_size_min", default=8)),
                int(OmegaConf.select(cfg, "era5_live.grid_size_max", default=28)),
            ),
            n_context_frac_range=(
                float(OmegaConf.select(cfg, "era5_live.n_context_frac_min", default=0.05)),
                float(OmegaConf.select(cfg, "era5_live.n_context_frac_max", default=0.4)),
            ),
            max_months=args.era5_max_months,
            standardize_y=args.era5_standardize_y,
        )
        live_episodes = build_era5_eval_episodes(
            args.era5_corpus_dir,
            n_ep,
            seed=args.seed,
            offset=args.episode_offset,
            tabicl_model=tabicl_marginal,
            k_folds=tabicl_pit_k_folds,
            device=device,
            pit_group_size=args.era5_pit_batch,
            marginal_backend=marginal_backend,
            marginal_regressor=marginal_regressor,
            marginal_probs_n=args.marginal_probs_n,
            autoregressive=args.autoregressive,
            ar_order=args.ar_order,
            ar_conditioning=args.ar_conditioning,
            ar_max_context=args.ar_max_context,
            ar_n_episodes=args.ar_n_episodes,
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
                f"(--min_test_points) for this run only — training's own "
                "conf/data/gp_tasks.yaml N_min is untouched"
            )
            cfg.data.N_min = min_test_points
            if cfg.data.N_max < cfg.data.N_min:
                cfg.data.N_max = cfg.data.N_min
        print(
            f"\nLive-generating {n_ep} episodes via generate_gp_batch "
            f"(return_kernel_metadata=True), seed={args.seed}, "
            f"global indices {args.episode_offset}..{args.episode_offset + n_ep - 1}, "
            "alternating every-other episode to a non-composite kernel"
        )
        live_episodes = _live_generate_alternating(
            cfg,
            n_ep,
            device,
            args.seed,
            offset=args.episode_offset,
            alternate_noncomposite=args.alternate_noncomposite,
        )
    else:
        dataset = CopulaDataset(episode_dir=dataset_dir)
        n_available = len(dataset)
        print(f"\nEvaluating {n_ep} episodes from {dataset_dir} (start={args.episode_idx})")
        print(f"  Dataset size: {n_available} episodes")
    return dataset, era5_geometry, live_episodes, marginal_regressor, n_available, tabicl_marginal


def _load_models(
    args: argparse.Namespace, cfg: DictConfig, device: torch.device
) -> tuple[CopulaTabICL, int, str | None, Any, str | None, TabICLLike | None, int]:
    """Load the copula checkpoint and the marginal that PITs each episode's z_train."""
    tabicl_ckpt = None
    # ---- Load ICL model ----
    print(f"\nLoading ICL checkpoint: {args.ckpt}")
    icl_model, icl_cfg = load_copula_model(args.ckpt, config_path=args.config, device=str(device))
    icl_rank = int(icl_cfg.model.rank)
    n_params = sum(p.numel() for p in icl_model.parameters())
    print(f"ICL model parameters: {n_params:,}  rank={icl_rank}")

    tabicl_marginal: TabICLLike | None = None
    marginal_backend: str | None = args.z_train_source if args.z_train_source in GENERIC_MARGINAL_BACKENDS else None
    marginal_regressor = None
    tabicl_pit_k_folds = DEFAULT_K_FOLDS
    if marginal_backend is not None:
        from eval.spatial.marginal_backends import make_regressor

        tabicl_pit_k_folds = args.tabicl_pit_k_folds or int(
            OmegaConf.select(cfg, "tabicl.pit_k_folds", default=DEFAULT_K_FOLDS)
        )
        print(
            f"\nBuilding {marginal_backend} marginal for --z_train_source={marginal_backend} "
            f"(k_folds={tabicl_pit_k_folds}, probs_n={args.marginal_probs_n})"
        )
        marginal_regressor = make_regressor(marginal_backend, device=str(device))
    elif args.z_train_source == "tabicl":
        # --tabicl_ckpt, then cfg.tabicl.ckpt, then DEFAULT_MARGINAL_FAMILY.
        tabicl_ckpt = args.tabicl_ckpt or OmegaConf.select(cfg, "tabicl.ckpt", default=None) or DEFAULT_MARGINAL_FAMILY
        tabicl_ckpt = resolve_marginal_checkpoint(str(tabicl_ckpt))
        if not tabicl_ckpt:
            raise ValueError(
                "--z_train_source=tabicl requires a TabICL checkpoint: pass --tabicl_ckpt "
                "or set cfg.tabicl.ckpt in --config."
            )
        tabicl_pit_k_folds = args.tabicl_pit_k_folds or int(
            OmegaConf.select(cfg, "tabicl.pit_k_folds", default=DEFAULT_K_FOLDS)
        )
        print(
            f"\nLoading frozen TabICL marginal for --z_train_source=tabicl: {tabicl_ckpt} "
            f"(k_folds={tabicl_pit_k_folds})"
        )
        tabicl_marginal = load_tabicl(tabicl_ckpt, str(device))
        configure_tabicl_inference_amp(args.tabicl_amp)
        print(f"Frozen TabICL marginal inference AMP={'on' if args.tabicl_amp else 'off (float32)'}")
    return icl_model, icl_rank, marginal_backend, marginal_regressor, tabicl_ckpt, tabicl_marginal, tabicl_pit_k_folds


def _resolve_episode_source(args: argparse.Namespace, cfg: DictConfig) -> tuple[bool, bool, str | None]:
    """(era5, live_generate, dataset_dir) from --era5, --dataset_dir and --live_generate."""
    era5 = bool(args.era5)
    if era5:
        if args.dataset_dir is not None:
            raise ValueError("--era5 and --dataset_dir are mutually exclusive episode sources.")
        if args.live_generate:
            raise ValueError("--era5 and --live_generate are mutually exclusive episode sources.")
        if args.z_train_source == "oracle":
            raise ValueError(
                "--era5 has no oracle marginal: real ERA5 has no generating GP to take "
                "an exact LOO-PIT residual from. Use --z_train_source=tabicl (default) "
                "or one of exaone/tabpfn/tabldm."
            )
        live_generate = False
    else:
        live_generate = args.live_generate if args.live_generate is not None else (args.dataset_dir is None)
    return era5, live_generate, _dataset_dir_for_eval(args, cfg, live_generate, era5)


# (local index, global episode index, baseline cache key, episode, baseline fit seed)
_PlannedEpisode = tuple[int, int, str, dict, int]


def _plan_episodes(
    args: argparse.Namespace,
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
    for local_i in range(args.n_episodes):
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
            ep_i = args.episode_offset + local_i
            ep = live_episodes[local_i]
        else:
            assert dataset is not None and n_available is not None
            ep_i = args.episode_idx + local_i
            if ep_i >= n_available:
                print(f"  [ep {ep_i}] index out of range ({n_available} available), skipping")
                continue
            ep = dataset[ep_i]
            n_test = ep["z_test"].shape[0]
            if n_test < min_test_points:
                print(
                    f"  [ep {ep_i}] only {n_test} test points (< --min_test_points="
                    f"{min_test_points}), skipping — best_baseline needs enough for "
                    ">=2 nested-CV folds"
                )
                continue
        cache_key = episode_cache_key(
            live_generate,
            dataset_dir,
            args.seed,
            ep_i,
            source="era5" if era5 else None,
        )
        episode_plan.append((local_i, ep_i, cache_key, ep, _baseline_fit_seed(args.seed, cache_key)))
    return episode_plan


def _baseline_worker_count(args: argparse.Namespace, baseline_device: torch.device) -> int:
    """--baseline_workers, defaulting to one per physical core (capped at 32); 1 on a GPU."""
    n_workers = args.baseline_workers
    try:
        _aff = os.sched_getaffinity(0)
    except AttributeError:  # pragma: no cover - non-Linux
        _aff = set(range(os.cpu_count() or 1))
    n_physical = _count_physical_cores(_aff)
    if n_workers <= 0:
        n_workers = max(1, min(32, n_physical or len(_aff)))
    if n_physical and n_physical < len(_aff):
        print(
            f"  [prefit] note: the {len(_aff)} allocated logical CPUs are only "
            f"{n_physical} physical core(s) ({len(_aff) // n_physical} threads each) — "
            "expect scaling closer to the physical count; request more cores "
            "from the scheduler for a proportionally faster run"
        )
    if baseline_device.type != "cpu" and n_workers > 1:
        print(
            f"  [prefit] --baseline_device={baseline_device.type}: forcing "
            "--baseline_workers=1 (parallel processes would just contend for one GPU)"
        )
        n_workers = 1
    return n_workers


@dataclass
class _EvalContext:
    """Settings, models and cache identities shared by every scored episode."""

    args: argparse.Namespace
    device: torch.device
    baseline_device: torch.device
    icl_model: CopulaTabICL
    tabicl_marginal: TabICLLike | None
    marginal_backend: str | None
    marginal_regressor: Any
    tabicl_pit_k_folds: int
    oracle_mode: str
    prior_cfg: dict
    fit_kwargs: dict[str, Any]
    use_cache: bool
    fingerprint: dict


def _prefit_baselines(ctx: _EvalContext, episode_plan: list[_PlannedEpisode], cache_entries: dict) -> dict[str, dict]:
    """Baselines available before scoring: valid cache entries, plus parallel fits of the rest."""
    args = ctx.args
    n_workers = _baseline_worker_count(args, ctx.baseline_device)
    fitted: dict[str, dict] = {}
    if ctx.use_cache and not args.refresh_baselines:
        for _, ep_i, cache_key, _, _ in episode_plan:
            entry = _valid_cached_entry(cache_entries, cache_key, ep_i)
            if entry is not None:
                fitted[cache_key] = entry

    pending = [(cache_key, fit_seed, ep) for _, _, cache_key, ep, fit_seed in episode_plan if cache_key not in fitted]
    print(
        f"\nBaselines: {len(fitted)} episode(s) reused from cache, "
        f"{len(pending)} to fit on {ctx.baseline_device.type}"
        + (f" across {n_workers} worker process(es)" if n_workers > 1 else " serially")
    )
    if pending and n_workers > 1:
        _prefit_baselines_parallel(
            pending,
            ctx.fit_kwargs,
            n_workers,
            args.baseline_cache,
            ctx.fingerprint,
            fitted,
            ctx.use_cache,
        )
    return fitted


# (copula NLL per method, correlation matrix per method, Y-space parts per method)
_BaselineResults = tuple[dict[str, float], dict[str, Tensor], dict[str, dict[str, float]]]


def _episode_baselines(
    ctx: _EvalContext, entry: dict | None, ep: dict, cache_key: str, fit_seed: int
) -> _BaselineResults:
    """The episode's baseline results: from its prefit entry, or fitted here and cached."""
    device, baseline_device = ctx.device, ctx.baseline_device
    if entry is not None:
        return entry["nlls"], {k: v.to(device) for k, v in entry["R_dict"].items()}, entry["y_nlls"]
    # Serial fitting (one worker or a GPU baseline device): results moved to the
    # evaluation device; RNG saved/restored and one BLAS thread, matching the pool.
    with _snapshot_rng_and_threads(cpu_threads=1 if baseline_device.type == "cpu" else None):
        baseline_nlls, baseline_R, baseline_y_nlls = eval_baselines_episode(
            ep={k: (v.to(baseline_device) if isinstance(v, Tensor) else v) for k, v in ep.items()},
            device=baseline_device,
            fit_seed=fit_seed,
            **ctx.fit_kwargs,
        )
    baseline_R = {k: v.to(device) for k, v in baseline_R.items()}
    if ctx.use_cache:
        save_baseline_entry(
            ctx.args.baseline_cache,
            ctx.fingerprint,
            cache_key,
            {
                "nlls": baseline_nlls,
                "R_dict": {k: v.cpu() for k, v in baseline_R.items()},
                "y_nlls": baseline_y_nlls,
            },
        )
    return baseline_nlls, baseline_R, baseline_y_nlls


def _episode_marginal_pit(ctx: _EvalContext, ep: dict, ep_i: int) -> dict[str, Tensor] | None:
    """The episode's own PIT (ERA5), else the loaded marginal's; None falls back to the oracle z_train."""
    marginal_pit = ep.get("marginal_pit")
    if marginal_pit is None and (ctx.tabicl_marginal is not None or ctx.marginal_regressor is not None):
        marginal_pit = _marginal_pit(
            ep=ep,
            tabicl_marginal=ctx.tabicl_marginal,
            k_folds=ctx.tabicl_pit_k_folds,
            device=ctx.device,
            marginal_backend=ctx.marginal_backend,
            marginal_regressor=ctx.marginal_regressor,
            marginal_probs_n=ctx.args.marginal_probs_n,
            seed=ep_i,
        )
        if marginal_pit is None:
            print(f"  [ep {ep_i}] fewer than 2 training points — falling back to oracle z_train for this episode")
    return marginal_pit


def _add_autoregressive_log_pdf(ctx: _EvalContext, ep: dict, local_i: int, ep_i: int) -> None:
    """Store the teacher-forced autoregressive chain's log-densities in ep (ERA5 precomputes them)."""
    args = ctx.args
    if not args.autoregressive or "ar_log_pdf" in ep:
        return
    if args.ar_n_episodes is not None and local_i >= args.ar_n_episodes:
        return
    if ctx.tabicl_marginal is None:
        raise RuntimeError("autoregressive scoring needs the loaded TabICL marginal")
    device = ctx.device
    ep["ar_log_pdf"] = (
        autoregressive_log_pdf(
            ctx.tabicl_marginal,
            ep["x_norm_train"].to(device).unsqueeze(0),
            ep["y_train"].to(device).unsqueeze(0),
            ep["x_norm_test"].to(device).unsqueeze(0),
            ep["y_test"].to(device).unsqueeze(0),
            order=args.ar_order,
            conditioning=args.ar_conditioning,
            max_context=args.ar_max_context,
            seed=args.seed,
            episode_indices=[ep_i],
        )["log_pdf"][0]
        .detach()
        .cpu()
    )


def _with_zero_mean_gp(
    ctx: _EvalContext, ep: dict, marginal_pit: dict[str, Tensor], fit_seed: int, baselines: _BaselineResults
) -> _BaselineResults:
    """Add the zero-mean GP baselines on the marginal's z_train, fitted here (not cached).

    Seeded from fit_seed with the RNG restored and one thread.
    """
    args = ctx.args
    with _snapshot_rng_and_threads(
        seed=fit_seed + 1,
        cpu_threads=1 if ctx.baseline_device.type == "cpu" else None,
    ):
        zm_nlls, zm_R, zm_y_nlls = _eval_zero_mean_gp_baselines(
            ep=ep,
            marginal_pit=marginal_pit,
            device=ctx.baseline_device,
            n_steps=args.n_steps_zeromean_gp,
            lr=args.lr_zeromean_gp,
            n_restarts=args.n_restarts_zeromean_gp,
            oracle_mode=ctx.oracle_mode,
            prior_cfg=ctx.prior_cfg,
        )
    baseline_nlls, baseline_R, baseline_y_nlls = baselines
    return (
        {**baseline_nlls, **zm_nlls},
        {**baseline_R, **{k: v.to(ctx.device) for k, v in zm_R.items()}},
        {**baseline_y_nlls, **zm_y_nlls},
    )


def _total_nll_rows(
    ep: dict,
    baseline_y_nlls: dict[str, dict[str, float]],
    icl_y_parts: dict[str, float],
    y_space_nlls: dict[str, dict[str, float]],
    marginal_pit: dict[str, Tensor] | None,
) -> dict[str, dict[str, float]]:
    """Per-point total/marginal/copula Y-space NLL for every row of the TOTAL table."""
    n_test = ep["z_test"].shape[0]
    # Episodes with rescaled targets carry a per-point Jacobian shift, applied to the
    # baselines' marginal and total (not the copula).
    _shift = float(ep.get("y_log_std", 0.0))
    if _shift:
        baseline_y_nlls = {
            k: {
                "total": v["total"] + _shift,
                "marginal": v["marginal"] + _shift,
                "copula": v["copula"],
            }
            for k, v in baseline_y_nlls.items()
        }
    # Autoregressive row (already in raw nats).
    ar_log_pdf = ep.get("ar_log_pdf")
    ar_parts = (
        ar_parts_from_log_pdf(ar_log_pdf, marginal_pit["log_pdf_test"].cpu())
        if ar_log_pdf is not None and marginal_pit is not None
        else _NAN_PARTS.copy()
    )
    marginal_only = {
        "total": icl_y_parts["marginal"],
        "marginal": icl_y_parts["marginal"],
        "copula": 0.0 if not np.isnan(icl_y_parts["marginal"]) else float("nan"),
    }
    total_nlls = {
        **baseline_y_nlls,
        "icl": icl_y_parts,
        "independence_marginal": marginal_only,
        "autoregressive": ar_parts,
        "oracle_prior": {k: v / n_test for k, v in y_space_nlls["prior"].items()},
        "oracle_posterior": {k: v / n_test for k, v in y_space_nlls["posterior"].items()},
    }
    gp_totals = [
        total_nlls.get(k, _NAN_PARTS)["total"]
        for k in _GP_TOTAL_KEYS
        if not np.isnan(total_nlls.get(k, _NAN_PARTS)["total"])
    ]
    total_nlls["best_gp_total"] = {
        "total": min(gp_totals) if gp_totals else float("nan"),
        "marginal": float("nan"),
        "copula": float("nan"),
    }
    return total_nlls


def _print_episode(
    ep: dict,
    ep_i: int,
    nlls: dict[str, float],
    total_nlls: dict[str, dict[str, float]],
    best_nll: float,
    fold_details: list[dict],
) -> None:
    """Per-episode summary: shared-z_test copula NLLs, the TOTAL rows, best baseline and top 5."""
    icl_nll = nlls.get("icl", float("nan"))
    ora_nll = nlls.get("oracle", float("nan"))
    ranked_baselines = sorted(
        ((k, v) for k, v in nlls.items() if k not in _NON_FITTED_EXCLUDED),
        key=lambda kv: kv[1],
    )
    top5 = ranked_baselines[:5]
    print(f"  ep {ep_i:04d}: kernel={_kernel_composition_label(ep)}")
    # Per-point numbers against the shared z_test.
    print(
        f"    icl(shared_marginal)={icl_nll:.4f}  "
        f"oracle_prior(shared_marginal, z-space copula)={ora_nll:.4f}  "
        f"GP-oracle-y-space-total(prior={total_nlls['oracle_prior']['total']:.4f}, "
        f"posterior={total_nlls['oracle_posterior']['total']:.4f})"
    )
    print(
        f"    total Y-space NLL, own marginal (total = marginal + copula): "
        f"icl=(cop={total_nlls['icl']['copula']:.4f}, "
        f"marg={total_nlls['icl']['marginal']:.4f}, "
        f"tot={total_nlls['icl']['total']:.4f})  "
        f"oracle_prior=(cop={total_nlls['oracle_prior']['copula']:.4f}, "
        f"marg={total_nlls['oracle_prior']['marginal']:.4f}, "
        f"tot={total_nlls['oracle_prior']['total']:.4f})  "
        f"oracle_posterior=(cop={total_nlls['oracle_posterior']['copula']:.4f}, "
        f"marg={total_nlls['oracle_posterior']['marginal']:.4f}, "
        f"tot={total_nlls['oracle_posterior']['total']:.4f})"
    )
    if fold_details:
        fold_summary = ", ".join(
            f"fold{fd['fold']}={_METHOD_LABELS.get(fd['selected'], fd['selected'])}" for fd in fold_details
        )
        print(f"    best_baseline (nested {len(fold_details)}-fold CV, pooled test NLL)={best_nll:.4f}")
        print(f"      per-fold picks: {fold_summary}")
    else:
        print("    best_baseline: unavailable (too few test points for nested CV)")
    print(
        "    top-5 baselines (ranked by shared-ground-truth-marginal copula "
        "NLL, diagnostic only — not the selection used for best_baseline "
        "above; 'own' columns are this baseline's OWN fitted marginal, "
        "NOT the same copula quantity as the ranking column — see "
        "eval_baselines_episode's docstring):"
    )
    for key, val in top5:
        own = total_nlls.get(key, _NAN_PARTS)
        print(
            f"      {_METHOD_LABELS.get(key, key):<28}"
            f"shared_copula={val:.4f}  "
            f"own(cop={own['copula']:.4f}, marg={own['marginal']:.4f}, "
            f"tot={own['total']:.4f})"
        )


@dataclass
class _EpisodeScore:
    nlls: dict[str, float]  # copula NLL per method (shared z_test), incl. best_baseline
    total_nlls: dict[str, dict[str, float]]  # TOTAL-table rows
    y_space_nlls: dict[str, dict[str, float]]  # oracle Y-space parts
    meta: dict
    R_dict: dict[str, Tensor]  # correlation per method
    R_oracle: Tensor | None
    best_key: str | None  # method nested CV picked (for the plot)


def _score_episode(
    ctx: _EvalContext,
    ep: dict,
    local_i: int,
    ep_i: int,
    fit_seed: int,
    baselines: _BaselineResults,
) -> _EpisodeScore:
    """Score one episode: the ICL model, the oracle and every baseline, then the nested-CV best baseline."""
    args, device = ctx.args, ctx.device
    marginal_pit = _episode_marginal_pit(ctx, ep, ep_i)
    _add_autoregressive_log_pdf(ctx, ep, local_i, ep_i)
    if args.zeromean_gp and marginal_pit is not None:
        baselines = _with_zero_mean_gp(ctx, ep, marginal_pit, fit_seed, baselines)
    baseline_nlls, baseline_R, baseline_y_nlls = baselines

    icl_nlls, icl_R, R_oracle, y_space_nlls, icl_y_parts = _eval_icl_episode(
        ep=ep,
        icl_model=ctx.icl_model,
        device=device,
        marginal_pit=marginal_pit,
    )
    total_nlls = _total_nll_rows(ep, baseline_y_nlls, icl_y_parts, y_space_nlls, marginal_pit)
    meta = {
        "ep_i": ep_i,
        "n_test": ep["z_test"].shape[0],
        "kernel": _kernel_composition_label(ep),
    }
    nlls = {**baseline_nlls, **icl_nlls}
    R_dict = {**baseline_R, **icl_R}

    # Best fitted baseline per episode, selected by nested CV over the test points.
    holdout_seed = (args.seed * 1_000_003 + ep_i) % (2**31 - 1)
    best_nll, mode_key, fold_details = _select_best_baseline_cv(
        baseline_R,
        ep["z_test"].to(device),
        args.n_folds,
        args.min_fold_size,
        holdout_seed,
    )
    _print_episode(ep, ep_i, nlls, total_nlls, best_nll, fold_details)
    nlls["best_baseline"] = best_nll
    return _EpisodeScore(nlls, total_nlls, y_space_nlls, meta, R_dict, R_oracle, mode_key)


def run_evaluation(args: argparse.Namespace) -> None:
    """Score one validated evaluation specification."""
    _validate_eval_spec(args)
    if args.config == "conf/config.yaml" and not os.path.isfile(args.config):
        args.config = os.path.join(project_config_dir(__file__), "config.yaml")
    _set_seed(args.seed)

    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    print(f"Device: {device}")

    # cfg: the episode-generating config from --config (not the checkpoint's).
    cfg = _load_full_config(args.config)

    (icl_model, icl_rank, marginal_backend, marginal_regressor, tabicl_ckpt, tabicl_marginal, tabicl_pit_k_folds) = (
        _load_models(args=args, cfg=cfg, device=device)
    )
    print(f"z_train source (ICL conditioning input): {args.z_train_source}")

    # Baselines score against the same oracle_mode as R_star (default "prior").
    oracle_mode = args.oracle_mode or OmegaConf.select(cfg, "data.oracle_mode", default="prior")
    print(f"Oracle mode: {oracle_mode}")

    # Baseline hyperpriors from cfg.data, falling back to classical._DEFAULT_PRIOR_CFG.
    data_cfg = OmegaConf.select(cfg, "data", default=None)
    prior_cfg = config_dict(data_cfg, resolve=False) if data_cfg is not None else {}
    print(f"GP-MLE restarts: {args.n_restarts_mle}")
    print(f"DKL restarts: {args.n_restarts_dkl}")

    era5, live_generate, dataset_dir = _resolve_episode_source(args, cfg)

    # Minimum test points (default 2 * --min_fold_size).
    min_test_points = args.min_test_points if args.min_test_points is not None else 2 * args.min_fold_size

    (dataset, era5_geometry, live_episodes, marginal_regressor, n_available, tabicl_marginal) = _load_episodes(
        args=args,
        cfg=cfg,
        dataset_dir=dataset_dir,
        device=device,
        era5=era5,
        live_generate=live_generate,
        marginal_backend=marginal_backend,
        marginal_regressor=marginal_regressor,
        min_test_points=min_test_points,
        n_ep=args.n_episodes,
        tabicl_marginal=tabicl_marginal,
        tabicl_pit_k_folds=tabicl_pit_k_folds,
    )

    print(
        f"  GP MLE: {args.n_steps_mle} steps | DKL: {args.n_steps_dkl} steps | "
        f"PerEp: {args.n_steps_per_ep} steps (patience={args.patience_per_ep})"
    )
    print(
        "  [per-episode print legend] 'shared_marginal'/'shared_copula' values "
        "score every method's own correlation matrix against the SAME "
        "ground-truth-standardized z_test, so they rank correlation-structure "
        "quality alone. 'own(...)'/'own_marginal' values instead use each "
        "method's OWN fitted marginal, so its own copula/marginal split is "
        "NOT comparable across methods (a method's own marginal can score "
        "better OR worse than another's regardless of which has the better "
        "overall fit) — only 'total' is a proper scoring rule comparable "
        "across methods with different marginals."
    )

    # Baseline cache.
    use_cache = not args.no_baseline_cache
    fingerprint = baseline_fingerprint(
        cfg,
        live_generate,
        dataset_dir,
        args.seed,
        icl_rank,
        oracle_mode,
        args.n_steps_mle,
        args.lr_mle,
        args.n_restarts_mle,
        args.n_steps_dkl,
        args.lr_dkl,
        args.n_steps_per_ep,
        args.patience_per_ep,
        gp_val_select=args.gp_val_select,
        n_restarts_dkl=args.n_restarts_dkl,
    )
    if era5:
        assert era5_geometry is not None
        # ERA5 fingerprint also includes the corpus, geometry and marginal.
        fingerprint["era5"] = era5_episode_fingerprint(
            args.era5_corpus_dir,
            k_folds=tabicl_pit_k_folds,
            marginal=args.z_train_source,
            **era5_geometry,
        )
    cache_entries = load_baseline_cache(args.baseline_cache, fingerprint) if use_cache else {}

    # ---- Scored-results cache: the checkpoint-DEPENDENT half of a resume ----
    use_results_cache = not args.no_results_cache
    results_fp = _results_fingerprint(
        fingerprint,
        args,
        tabicl_pit_k_folds,
        resolved_marginal=tabicl_ckpt if args.z_train_source == "tabicl" else None,
    )
    results_entries = _load_results_cache(args.results_cache, results_fp) if use_results_cache else {}
    if use_results_cache and args.refresh_baselines:
        # Refitting also rescores.
        results_entries = {}

    ctx = _EvalContext(
        args=args,
        device=device,
        baseline_device=torch.device(str(device) if args.baseline_device == "auto" else args.baseline_device),
        icl_model=icl_model,
        tabicl_marginal=tabicl_marginal,
        marginal_backend=marginal_backend,
        marginal_regressor=marginal_regressor,
        tabicl_pit_k_folds=tabicl_pit_k_folds,
        oracle_mode=oracle_mode,
        prior_cfg=prior_cfg,
        fit_kwargs=dict(
            icl_rank=icl_rank,
            n_steps_mle=args.n_steps_mle,
            lr_mle=args.lr_mle,
            n_steps_dkl=args.n_steps_dkl,
            lr_dkl=args.lr_dkl,
            n_steps_per_ep=args.n_steps_per_ep,
            patience_per_ep=args.patience_per_ep,
            oracle_mode=oracle_mode,
            prior_cfg=prior_cfg,
            n_restarts_mle=args.n_restarts_mle,
            n_restarts_dkl=args.n_restarts_dkl,
            gp_val_select=args.gp_val_select,
        ),
        use_cache=use_cache,
        fingerprint=fingerprint,
    )
    episode_plan = _plan_episodes(
        args,
        era5=era5,
        live_generate=live_generate,
        dataset_dir=dataset_dir,
        dataset=dataset,
        live_episodes=live_episodes,
        n_available=n_available,
        min_test_points=min_test_points,
    )
    fitted = _prefit_baselines(ctx, episode_plan, cache_entries)

    all_nlls: list[dict[str, float]] = []
    all_y_space_nlls: list[dict[str, dict[str, float]]] = []
    all_total_nlls: list[dict[str, dict[str, float]]] = []
    # Per-episode metadata for --dump_episodes.
    all_episode_meta: list[dict] = []
    plot_R_dict: dict[str, Tensor] | None = None
    plot_R_oracle: Tensor | None = None
    plot_best_key: str | None = None
    plot_best_R: Tensor | None = None
    for local_i, ep_i, cache_key, ep, fit_seed in episode_plan:
        # pop: each entry holds large N x N matrices and is used once.
        entry = fitted.pop(cache_key, None)
        cache_entries.pop(cache_key, None)
        baselines = _episode_baselines(ctx, entry, ep, cache_key, fit_seed)

        # Reuse an episode's cached scored results; the plotted episode is always
        # rescored (the cache keeps no R matrices).
        want_plot = local_i == args.plot_episode
        res_cached = results_entries.get(str(ep_i)) if use_results_cache else None
        if res_cached is not None and not want_plot:
            all_y_space_nlls.append(res_cached["y_space_nlls"])
            all_total_nlls.append(res_cached["total_nlls"])
            all_episode_meta.append(res_cached["meta"])
            all_nlls.append(res_cached["nlls"])
            print(f"  ep {ep_i:04d}: reusing scored results (--results_cache)")
            continue

        score = _score_episode(ctx, ep, local_i, ep_i, fit_seed, baselines)
        all_y_space_nlls.append(score.y_space_nlls)
        all_total_nlls.append(score.total_nlls)
        all_episode_meta.append(score.meta)
        all_nlls.append(score.nlls)
        if use_results_cache:
            results_entries[str(ep_i)] = _jsonable(
                {
                    "nlls": score.nlls,
                    "total_nlls": score.total_nlls,
                    "y_space_nlls": score.y_space_nlls,
                    "meta": score.meta,
                }
            )
            _save_results_cache(args.results_cache, results_fp, results_entries)
        if want_plot:
            plot_R_dict = score.R_dict
            plot_R_oracle = score.R_oracle
            if score.best_key is not None:
                plot_best_key = score.best_key
                plot_best_R = score.R_dict[score.best_key]

    _report_results(
        all_episode_meta=all_episode_meta,
        all_nlls=all_nlls,
        all_total_nlls=all_total_nlls,
        all_y_space_nlls=all_y_space_nlls,
        args=args,
        dataset_dir=dataset_dir,
        era5=era5,
        live_generate=live_generate,
        n_ep=args.n_episodes,
        plot_R_dict=plot_R_dict,
        plot_R_oracle=plot_R_oracle,
        plot_best_R=plot_best_R,
        plot_best_key=plot_best_key,
    )


def main() -> None:
    run_evaluation(parse_eval_spec())


if __name__ == "__main__":
    main()
