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
from copula_inter.backend_registry import EVAL_Z_TRAIN_SOURCES, GENERIC_MARGINAL_BACKENDS  # noqa: E402
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
    AR_CONDITIONINGS,
    AR_ORDERS,
    ar_parts_from_log_pdf,
    autoregressive_log_pdf,
)
from eval.baselines.classical import (  # noqa: E402
    GP_VAL_SELECT_MODES,
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
    resolve_checkpoint,
    resolve_marginal_checkpoint,
)
from eval.configs.constants import N_CONTEXT  # noqa: E402
from eval.data.era5_episodes import (  # noqa: E402
    DEFAULT_CORPUS_DIR as ERA5_DEFAULT_CORPUS_DIR,
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


def parse_eval_spec(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate CLI settings without loading checkpoints or data."""
    parser = argparse.ArgumentParser(
        description="Evaluate ICL checkpoint vs baselines on inter-instance copula episodes"
    )
    parser.add_argument(
        "--config",
        default="conf/config.yaml",
        help="Hydra config defining the eval-episode-generating "
        "distribution (cfg.data) for --live_generate, "
        "resolved through its own defaults list — "
        "independent of --ckpt's saved training cfg. "
        "Keeping this fixed is what lets the baseline "
        "cache survive switching checkpoints.",
    )
    parser.add_argument("--ckpt", required=True, help="Checkpoint path, or a CHECKPOINT_FAMILIES name[:step].")
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Episode directory to evaluate on (overrides "
        "training.dataset_dir from --config). Passing "
        "this disables --live_generate by default.",
    )
    parser.add_argument(
        "--live_generate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Generate evaluation episodes on the fly via "
        "data_gen.generate_gp_batch(..., return_kernel_metadata=True) "
        "instead of loading a pre-built PIT dataset directory. Default: "
        "True unless --dataset_dir is given. --episode_idx is ignored "
        "in this mode (episodes are freshly sampled, not indexed).",
    )
    parser.add_argument(
        "--alternate_noncomposite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force every even live synthetic episode to a single elementary "
        "kernel for mixed-distribution coverage. Disable for a pure "
        "systematic-composition benchmark.",
    )
    # Real-ERA5 episodes (--era5): same baselines and tables, no oracle rows;
    # the shared z_test is the TabICL PIT. --z_train_source=oracle is rejected.
    parser.add_argument(
        "--era5",
        action="store_true",
        help="Evaluate on real ARCO-ERA5 episodes instead of synthetic "
        "GP draws. Mutually exclusive with --dataset_dir; forces "
        "--live_generate off.",
    )
    parser.add_argument(
        "--era5_corpus_dir",
        default=ERA5_DEFAULT_CORPUS_DIR,
        help="Cached global-ERA5 monthly NetCDF directory (populate with "
        "eval/data/fetch_era5_global.py). Defaults to the held-out "
        "era5_global_val corpus, deliberately disjoint from the "
        "era5_global_train years finetuning draws from.",
    )
    parser.add_argument(
        "--era5_grid_size",
        type=int,
        default=24,
        help="Points per side of each sampled region (D = grid_size^2 "
        "total). 24 matches conf/config.yaml's baselines.era5_grid_size "
        "and eval/configs/regions.py's grid_resolution convention.",
    )
    parser.add_argument(
        "--era5_n_context",
        type=int,
        default=N_CONTEXT,
        help="In-context points P per episode (eval.configs.constants."
        "N_CONTEXT). The remaining grid_size^2 - P points are all "
        "held-out targets, so N=546 at the defaults.",
    )
    parser.add_argument("--era5_box_deg_min", type=float, default=5.0)
    parser.add_argument(
        "--era5_box_deg_max",
        type=float,
        default=25.0,
        help="Region box width in degrees, drawn per episode. Boxes too "
        "small to hold a full grid_size^2 block of native 0.25deg "
        "points are redrawn, not clipped.",
    )
    parser.add_argument(
        "--era5_vary_geometry",
        action="store_true",
        help="Draw grid_size/context-fraction per episode from era5_live's "
        "training ranges instead of the fixed --era5_grid_size/"
        "--era5_n_context. Costs the batched PIT (P/N stop being "
        "homogeneous) and makes per-episode NLLs N-heterogeneous.",
    )
    parser.add_argument(
        "--era5_pit_batch",
        type=int,
        default=8,
        help="Episodes PIT'd per run_pit_batched call (fixed geometry "
        "only). Higher is faster until TabICL's activations stop "
        "fitting in VRAM.",
    )
    parser.add_argument(
        "--era5_standardize_y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Z-score each episode's ERA5 target by its own training "
        "mean/std before fitting the classical baselines, then add "
        "log(std) per point back to their marginal/total Y-space "
        "NLLs so the printed table stays in raw Kelvin nats. On by "
        "default: ERA5 targets are absolute Kelvin (~280) while "
        "eval/baselines/classical.py's GP hyperpriors are calibrated "
        "for data_gen.py's O(1) draws, and the ICL model under test "
        "normalizes internally via its TabICL PIT — so leaving the "
        "baselines on raw y handicaps them on units alone. Pass "
        "--no-era5_standardize_y to fit on raw Kelvin instead.",
    )
    # ---- Autoregressive (chain-rule) marginal row --------------------------
    parser.add_argument(
        "--autoregressive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add the 'Autoregressive marginal (chain rule)' row to the "
        "total Y-space NLL table: the SAME marginal, but revealing "
        "the test points one at a time so each prediction conditions "
        "on the ones already revealed (eval/baselines/"
        "autoregressive.py). An exact factorization of a joint "
        "density, so it is directly comparable to every other row, "
        "and it is the copula-free reference the copula head has to "
        "beat. On by default for the TabICL marginal on every episode "
        "source; it costs one marginal forward pass per test point.",
    )
    parser.add_argument(
        "--ar_order",
        default="random",
        choices=list(AR_ORDERS),
        help="Order the chain reveals test points in. An in-context "
        "learner is not a coherent joint, so the chain-rule total "
        "really does depend on this. 'random' (default) is a seeded "
        "per-episode permutation; 'natural' is the grid's row-major "
        "order, which hands almost every step a just-revealed "
        "neighbour and reads as a best case rather than a typical one.",
    )
    parser.add_argument(
        "--ar_conditioning",
        default="teacher_forcing",
        choices=list(AR_CONDITIONINGS),
        help="What to append to the context at each step. "
        "'teacher_forcing' (default) appends the TRUE y, which is "
        "what makes the printed number an exact joint log-density "
        "and a proper scoring rule. 'sample' appends a draw from "
        "that step's predictive instead — ancestral sampling from "
        "the model's implied joint, useful for generating fields, "
        "but its log-density sum is then taken along a SAMPLED "
        "conditioning path and is NOT a density of y_test. Do not "
        "compare a 'sample' number to the other rows.",
    )
    parser.add_argument(
        "--ar_max_context",
        type=int,
        default=None,
        help="Cap the chain's context at this many rows (the episode's "
        "own P are always kept; the oldest revealed points are "
        "dropped first). None (default) keeps every revealed point, "
        "which at the --era5 defaults grows the context 30 -> 576.",
    )
    parser.add_argument(
        "--ar_n_episodes",
        type=int,
        default=None,
        help="Run the chain on only the first N episodes (None = all). "
        "The row then averages over those episodes and the rest "
        "contribute nan, same convention as every other partially "
        "available row.",
    )
    parser.add_argument(
        "--era5_max_months",
        type=int,
        default=None,
        help="Cap the corpus to its most recent N monthly files (smaller "
        "date range). None reads everything cached. NOTE this is not "
        "a RAM knob and can cost RAM: GlobalERA5Corpus picks "
        "memory-mapped reading only when it sees >60 files, and the "
        "cap is applied first, so capping a 396-month corpus to 60 "
        "makes it load ~7.5 GB eagerly instead of ~100 MB lazily.",
    )
    parser.add_argument("--n_episodes", type=int, default=30)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument(
        "--n_steps_mle",
        type=int,
        default=1000,
        help="Adam steps for GP kernel MLE fitting (also used for ARD variants)",
    )
    parser.add_argument("--lr_mle", type=float, default=0.05, help="Learning rate for GP MLE Adam")
    parser.add_argument(
        "--n_restarts_mle",
        type=int,
        default=5,
        help="Independent random restarts per GP-MLE kernel fit (each "
        "initialised by sampling from the same LogNormal/Gamma "
        "hyperpriors data_gen.py's generative process uses); keeps "
        "whichever restart reaches the best final training loss.",
    )
    parser.add_argument(
        "--zeromean_gp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit the 'Marginal + Zero Mean GP (RBF/Matern32)' baselines "
        "directly on marginal_pit['z_train'] -- the same real "
        "(non-oracle) marginal input the ICL model conditions on -- "
        "instead of the raw y-space GP-MLE/DKL baselines above (see "
        "eval.baselines.classical.fit_zero_mean_gp_on_marginal's "
        "module docstring). Only 2 kernels (RBF, Matern32) by design, "
        "to keep this cheap. Automatically a no-op for episodes with "
        "no real marginal available (--z_train_source=oracle, or "
        "fewer than 2 training points).",
    )
    parser.add_argument(
        "--n_steps_zeromean_gp",
        type=int,
        default=500,
        help="Adam steps for each Zero-Mean GP z-space fit. Needs far "
        "fewer than --n_steps_mle's 1000: profiled on a real episode "
        "(P=32/N=256/d_x=11, RBF/Matern32, 1 CPU thread), the fit "
        "NLL plateaus by step 500 (rbf -0.0252, matern32 -0.0386) and "
        "barely moves by step 1000 (-0.0251/-0.0386) -- unlike raw "
        "y-space GP-MLE, there is no separate mean/scale to also "
        "discover here, just 2-3 kernel hyperparameters against an "
        "already marginally-standardized target, so it converges much "
        "faster. Chosen on that convergence plateau, not trimmed for "
        "speed (see the module docstring's Runtime note on why "
        "--n_steps_mle itself must never be tuned that way).",
    )
    parser.add_argument(
        "--lr_zeromean_gp", type=float, default=0.05, help="Learning rate for the Zero-Mean GP Adam fits"
    )
    parser.add_argument(
        "--n_restarts_zeromean_gp",
        type=int,
        default=2,
        help="Random restarts per Zero-Mean GP kernel fit -- fewer than "
        "--n_restarts_mle's 5 by design. Profiled on the same episode: "
        "restart 2 clearly beats restart 1 (rbf -0.0301 -> -0.0251, "
        "matern32 -0.0442 -> -0.0386) but a 3rd restart finds nothing "
        "further (identical NLL to 2), so 2 is the measured sweet spot "
        "rather than a guess.",
    )
    parser.add_argument(
        "--n_steps_dkl", type=int, default=5000, help="Adam steps for Deep Kernel Learning (MLP+GP) fitting"
    )
    parser.add_argument("--lr_dkl", type=float, default=0.01, help="Learning rate for DKL Adam")
    parser.add_argument(
        "--n_restarts_dkl",
        type=int,
        default=2,
        help="Independent random restarts per DKL kernel fit -- each "
        "restart gets a FRESH feature-extractor MLP (not the same "
        "instance re-trained further), since a shared instance would "
        "carry its already-updated weights from one restart into the "
        "next and silently defeat the point of restarting (see "
        "fit_and_eval_gpytorch's docstring). Was hardcoded to 1 "
        "(no restart loop at all) until this option existed: DKL's "
        "joint MLP+kernel landscape is harder and more init-sensitive "
        "than plain GP-MLE's, so a single unlucky random MLP init had "
        "no chance to recover, and diagnostics traced z-space copula "
        "NLL swinging from ~5 to >100 nats/pt across episodes purely "
        "on that one seed's luck. Measured on 4 held-out episodes "
        "(dkl_rbf/dkl_dot_product, n_steps=2000): restart 2 roughly "
        "halves the median z-space copula NLL vs. restart 1 (rbf "
        "45.5 -> 18.9, dot_product 69.5 -> 55.9 nats/pt), a 3rd "
        "restart found nothing further in every one of the 8 "
        "(kernel, episode) combinations tried -- same restart-2-"
        "suffices pattern as --n_restarts_zeromean_gp. DKL still ends "
        "up well behind GP-MLE/Marginal+ZeroMean-GP even at restart 2: "
        "unlike those, its kernel operates on a LEARNED feature space, "
        "so the same rescale-to-defeat-the-lengthscale-prior "
        "identifiability issue the module docstring already describes "
        "for the held-out-NLL guard applies to the restarts here too "
        "-- restarts pick a better basin, they do not fix the "
        "underlying identifiability gap. (A LayerNorm on the MLP "
        "output was tried as a fix and made things WORSE -- median "
        "NLL 18.9 -> 57.0 for rbf, 55.9 -> 341.7 for dot_product on "
        "the same episodes -- because per-sample normalisation erases "
        "exactly the relative-magnitude information RBF/dot_product "
        "need between different points; not applied.)",
    )
    parser.add_argument("--n_steps_per_ep", type=int, default=5000, help="Training steps for PerEpisodeTransformer")
    parser.add_argument(
        "--patience_per_ep", type=int, default=500, help="Early stopping patience for PerEpisodeTransformer"
    )
    parser.add_argument(
        "--z_train_source",
        default="tabicl",
        choices=EVAL_Z_TRAIN_SOURCES,
        help="What the ICL model conditions on for each episode's z_train. "
        "'tabicl' (default): a K-fold cross-fitted PIT estimate from the "
        "frozen TabICL marginal (pit.py::run_pit) — the same proxy "
        "real-world deployment is stuck with, so this is the setting "
        "that makes _print_total_nll_table's icl row (the genuinely "
        "cross-method-comparable total marginal+copula NLL) populate; "
        "adds the cost of one extra frozen TabICL forward pass per fold "
        "per episode. 'oracle': the episode's exact GP-LOO PIT residual "
        "(R&W Eq. 5.12) computed from the true generating kernel — "
        "unavailable on real data, an idealized upper bound only; under "
        "this mode the icl row of _print_total_nll_table is NaN (no "
        "learned marginal to score a total NLL against). icl_nll under "
        "either source is unaffected in every other respect: z_test, "
        "R_oracle, and every baseline still score/fit against the "
        "episode's true values — use both to measure the sim-to-real "
        "gap. 'exaone'/'tabpfn'/'tabldm': the same non-oracle idea with "
        "a different tabular foundation model supplying the marginal "
        "(eval/spatial/marginal_backends.py), through the identical "
        "batched PIT module the training pipelines use — so an eval "
        "scores exactly the marginal a run trained against. These need "
        "no --tabicl_ckpt; --tabicl_pit_k_folds still sets K, and "
        "--marginal_probs_n sets their quantile-grid size.",
    )
    parser.add_argument(
        "--marginal_probs_n",
        type=int,
        default=99,
        help="Quantile grid size for --z_train_source=exaone/tabpfn/tabldm "
        "(ignored for oracle/tabicl, which use their own native grids). "
        "Mirrors data.z_train_marginal_probs_n in conf/data/gp_tasks.yaml; "
        "a proportional lever on those backends' per-episode cost.",
    )
    parser.add_argument(
        "--tabicl_ckpt",
        default=None,  # path OR a MARGINAL_FAMILIES name
        help="TabICL checkpoint filename for --z_train_source=tabicl. Default: read from --config's cfg.tabicl.ckpt.",
    )
    parser.add_argument(
        "--tabicl_pit_k_folds",
        type=int,
        default=None,
        help="K-fold count for --z_train_source=tabicl's run_pit call. "
        f"Default: cfg.tabicl.pit_k_folds, falling back to "
        f"pit.DEFAULT_K_FOLDS ({DEFAULT_K_FOLDS}).",
    )
    parser.add_argument(
        "--tabicl_amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="AMP (float16 autocast) for the frozen TabICL marginal's "
        "forward passes under --z_train_source=tabicl (pit.py::"
        "configure_tabicl_inference_amp). Disabled by default for "
        "float32 quantile-grid precision; pass --tabicl_amp to enable "
        "AMP, or --no-tabicl_amp to keep it disabled. This matters "
        "more for eval's log_pdf_test/marginal-NLL fidelity than for "
        "live-generation throughput.",
    )
    parser.add_argument(
        "--plot_episode", type=int, default=0, help="Local episode index to generate the corr_grid plot for"
    )
    parser.add_argument(
        "--out_dir", default=os.path.join(_REPO_ROOT, "eval", "results"), help="Directory for saved corr_grid figure"
    )
    parser.add_argument(
        "--dump_episodes",
        default=None,
        help="Write per-episode NLLs (all_nlls + all_total_nlls, keyed by "
        "ep_i, plus each episode's kernel label) to this JSON path. "
        "Two runs (e.g. different --ckpt) sharing --config/--seed/"
        "--live_generate see identical episodes (see "
        "_live_generate_alternating), so their dumps can be joined on "
        "ep_i for a PAIRED per-episode comparison — far lower variance "
        "than comparing the two runs' printed Mean/Std NLL as independent "
        "samples, since episode-to-episode difficulty (kernel, "
        "lengthscale, N) cancels in the per-episode difference instead "
        "of inflating each run's own across-episode std.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds for nested (leave-one-fold-out) "
        "cross-validation of the per-episode best_baseline pick: "
        "each fold's held-out NLL is scored using a baseline "
        "selected (argmin NLL) on the other K-1 folds only, and "
        "the K fold NLLs are pooled — instead of argmin-ing "
        "directly over z_test, which lets the winner peek at the "
        "very data it's scored on (see _select_best_baseline_cv). "
        "Actual K is capped at n_test // 3 per episode, so small-N "
        "episodes automatically use fewer folds. The fold split is "
        "deterministic per episode (seeded from --seed + episode "
        "index) and never touches icl/oracle or any individual "
        "baseline's own diagnostic row, which keep scoring against "
        "the full test set as before.",
    )
    parser.add_argument(
        "--min_fold_size",
        type=int,
        default=20,
        help="Minimum points required on both sides of every "
        "leave-one-fold-out split (a fold's own size, and its "
        "(K-1)-fold val complement) before a candidate baseline's "
        "NLL there is trusted enough to base a selection on. Below "
        "this, ranking ~12 candidate kernels by NLL on a handful of "
        "points is closer to a coin flip than a real comparison, and "
        "an occasional spurious pick of a blow-up-prone candidate "
        "(e.g. dot_product/polynomial kernels, whose NLL is "
        "unbounded above when misspecified) can drag the averaged "
        "best_baseline mean above even a single steady fixed kernel "
        "used for every episode. Effective K per episode is "
        "min(--n_folds, n_test // min_fold_size); episodes where "
        "that's < 2 report best_baseline=nan (excluded from its "
        "mean/std, not silently zero) instead of forcing a low-"
        "confidence selection.",
    )
    parser.add_argument(
        "--min_test_points",
        type=int,
        default=None,
        help="Floor on eval episodes' test-point count N, so every episode "
        "is valid for the nested-CV best_baseline selection (see "
        "--min_fold_size: _select_best_baseline_cv needs n_test // "
        "min_fold_size >= 2 folds, i.e. n_test >= 2 * min_fold_size, "
        "or it reports best_baseline=nan for that episode — silently "
        "excluding it from the summary table's best_baseline mean while "
        "every other row's mean still includes it). Default: "
        "2 * --min_fold_size (40 under the argparse defaults). For "
        "--live_generate (the default), this raises cfg.data.N_min for "
        "this run only — conf/data/gp_tasks.yaml's own N_min=8, and "
        "hence training's distribution, is untouched. For --dataset_dir, "
        "episode sizes are fixed by the pre-built dataset and can't be "
        "regenerated, so episodes below this floor are skipped instead.",
    )
    parser.add_argument(
        "--oracle_mode",
        default=None,
        choices=["prior", "posterior"],
        help="How R_star was built for this dataset. Determines whether "
        "GP-MLE/DKL score the fitted kernel's posterior (conditioned "
        "on X_train) or its raw prior covariance at X_test. Default: "
        "read from the checkpoint's own saved training config "
        "(cfg.data.oracle_mode), falling back to 'prior' if absent.",
    )
    parser.add_argument(
        "--baseline_cache",
        default="./baseline_cache.pt",
        help="Path to a cache file storing every classical baseline's fitted "
        "NLL/correlation results, keyed per-episode. These are the "
        "expensive, checkpoint-independent part of the comparison; the "
        "ICL model + oracle are always recomputed fresh since they're "
        "what actually changes between runs. A cache entry is only "
        "reused when the episode-generating config and every baseline-"
        "fitting hyperparameter below match exactly what produced it "
        "(see eval.baselines.classical.baseline_fingerprint) — otherwise "
        "it's recomputed and the cache updated in place.",
    )
    parser.add_argument(
        "--no_baseline_cache",
        action="store_true",
        help="Disable baseline caching entirely: always recompute, never read or write --baseline_cache.",
    )
    parser.add_argument(
        "--refresh_baselines",
        action="store_true",
        help="Recompute every baseline even if a matching cache entry "
        "exists, overwriting it (still writes --baseline_cache unless "
        "--no_baseline_cache is also given).",
    )
    parser.add_argument(
        "--results_cache",
        default="./eval_results_partial.json",
        help="Per-episode SCORED results (every table this script "
        "prints), written after each episode and reused on a "
        "restart. --baseline_cache only spares the classical "
        "fitting; the ICL forward pass, the marginal PIT and the "
        "nested-CV best_baseline pick lived only in memory until "
        "the summary tables, so an interrupted run used to re-score "
        "every episode. Unlike --baseline_cache this key INCLUDES "
        "the checkpoint and every marginal/CV setting (results "
        "depend on --ckpt; baseline fits deliberately do not), so "
        "pointing two different checkpoints at one path makes the "
        "second ignore the first's entries rather than report them "
        "— give concurrent runs distinct paths.",
    )
    parser.add_argument(
        "--no_results_cache",
        action="store_true",
        help="Disable the scored-results cache: always re-score every episode and never read or write --results_cache.",
    )
    parser.add_argument(
        "--gp_val_select",
        choices=GP_VAL_SELECT_MODES,
        default="ard",
        help="Which GP-MLE kernels pick their fit on a held-out 20%% "
        "split of the episode's training points -- both which step "
        "within a restart and which of --n_restarts_mle restarts -- "
        "instead of running to --n_steps_mle and keeping the lowest "
        "TRAINING loss. 'ard' (default) applies it to the ARD "
        "kernels only, 'always' to every kernel, 'never' restores "
        "the pre-v3 behaviour. ARD needs it: it fits one lengthscale "
        "per input dimension (9 from P=32 points), the LogNormal "
        "prior does not hold them, and they run away unbounded -- "
        "ard_rbf ends up +6.06 nats/point worse at 1000 steps than "
        "at its own optimum (10 steps). The non-ARD kernels do not: "
        "with 2-3 hyperparameters the prior regularises them "
        "adequately, so the split is pure data loss (dot_product, a "
        "linear kernel with nothing to overfit, measures +0.36 "
        "nats/point WORSE under 'always'). See "
        "eval.baselines.classical._resolve_val_select for the "
        "per-kernel numbers.",
    )
    parser.add_argument(
        "--baseline_device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device for fitting the classical baselines (GP-MLE/DKL/"
        "per_ep_transformer) only — the ICL model, TabICL marginal "
        "and oracle always run on --device. Defaults to cpu because "
        "it is measurably FASTER here: episodes have P=32 training "
        "points, so each of the ~85,000 Adam steps per episode is a "
        "32x32 Cholesky whose launch latency dwarfs its arithmetic. "
        "Measured same-seed, same-steps against a TITAN-RTX-class "
        "GPU, one CPU thread runs GP-MLE rbf in 2.98 s vs 7.50 s and "
        "DKL rbf in 20.23 s vs 38.54 s, for identical NLLs. 'auto' "
        "follows --device; pass cuda to reproduce the old behaviour.",
    )
    parser.add_argument(
        "--baseline_workers",
        type=int,
        default=0,
        help="Worker processes for fitting baselines, which are perfectly "
        "independent across episodes. 0 (default) = auto: one per "
        "PHYSICAL core this process is allowed to use "
        "(os.sched_getaffinity, so an OAR allocation is respected; "
        "hyperthread siblings are not counted twice because they add "
        "well under a core here), capped at 32. 1 fits serially "
        "in-process, as before. Only "
        "used with --baseline_device=cpu: spreading GPU fits across "
        "processes just contends for one device. Combined with the "
        "cpu default this is the main speedup — ~2x from the device, "
        "the rest from cores. Scaling tracks PHYSICAL cores: measured "
        "78.6 s/episode vs 649 s on GPU (~8x) where the 8 allocated "
        "logical CPUs were only 4 physical ones, so ask the scheduler "
        "for real cores, not threads.",
    )
    parser.add_argument(
        "--episode_offset",
        type=int,
        default=0,
        help="Global index of the first live-generated episode (ignored "
        "unless --live_generate). Lets one episode stream be split "
        "across OAR array jobs: --n_episodes 100 with "
        "--episode_offset 0/100/200/300 evaluates the same 400 "
        "episodes as a single --n_episodes 400 run, bit-identically "
        "(generating seed, non-composite parity, cache key and "
        "nested-CV holdout seed all key off the global index — see "
        "_live_generate_alternating). Give each shard its own "
        "--baseline_cache; the resulting files share a fingerprint "
        "and can be merged by concatenating their 'entries' dicts.",
    )
    parser.add_argument(
        "--min_icl_coverage",
        type=float,
        default=0.0,
        help="Fail after scoring unless this fraction of attempted episodes has a finite ICL total NLL.",
    )
    args = parser.parse_args(argv)
    args.ckpt = resolve_checkpoint(args.ckpt)
    try:
        _validate_eval_spec(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _validate_eval_spec(args: argparse.Namespace) -> None:
    if not 0 <= args.min_icl_coverage <= 1:
        raise ValueError("--min_icl_coverage must lie in [0, 1]")
    if args.min_icl_coverage and args.z_train_source == "oracle":
        raise ValueError("--min_icl_coverage requires a learned marginal")

    # The autoregressive chain needs the TabICL marginal.
    if args.autoregressive and args.z_train_source != "tabicl":
        raise ValueError(
            "--autoregressive requires --z_train_source=tabicl: the chain needs "
            "a marginal callable with a growing context. Re-run with "
            "--no-autoregressive for this marginal backend."
        )


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

    # Episode source: --era5, --dataset_dir, or live generation (default).
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
    dataset_dir = _dataset_dir_for_eval(args, cfg, live_generate, era5)

    n_ep = args.n_episodes
    all_nlls: list[dict[str, float]] = []
    all_y_space_nlls: list[dict[str, dict[str, float]]] = []
    all_total_nlls: list[dict[str, dict[str, float]]] = []
    # Per-episode metadata for --dump_episodes.
    all_episode_meta: list[dict] = []
    plot_R_dict: dict[str, Tensor] | None = None
    plot_R_oracle: Tensor | None = None
    plot_best_key: str | None = None
    plot_best_R: Tensor | None = None

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
        n_ep=n_ep,
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

    baseline_device = torch.device(str(device) if args.baseline_device == "auto" else args.baseline_device)
    fit_kwargs = dict(
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
    )

    # Episode plan: every evaluated episode and its cache key.
    episode_plan: list[tuple[int, int, str, dict, int]] = []
    for local_i in range(n_ep):
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

    # Fit the uncached baselines in parallel before the scoring loop.
    n_workers = args.baseline_workers
    try:
        _aff = os.sched_getaffinity(0)
    except AttributeError:  # pragma: no cover - non-Linux
        _aff = set(range(os.cpu_count() or 1))
    n_physical = _count_physical_cores(_aff)
    # Default workers: one per physical core (capped); override with --baseline_workers.
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

    # Baselines available for this run: cached entries plus pool fits.
    fitted: dict[str, dict] = {}
    if use_cache and not args.refresh_baselines:
        for _, ep_i, cache_key, _, _ in episode_plan:
            entry = _valid_cached_entry(cache_entries, cache_key, ep_i)
            if entry is not None:
                fitted[cache_key] = entry

    pending = [(cache_key, fit_seed, ep) for _, _, cache_key, ep, fit_seed in episode_plan if cache_key not in fitted]
    print(
        f"\nBaselines: {len(fitted)} episode(s) reused from cache, "
        f"{len(pending)} to fit on {baseline_device.type}"
        + (f" across {n_workers} worker process(es)" if n_workers > 1 else " serially")
    )
    if pending and n_workers > 1:
        _prefit_baselines_parallel(
            pending,
            fit_kwargs,
            n_workers,
            args.baseline_cache,
            fingerprint,
            fitted,
            use_cache,
        )

    for local_i, ep_i, cache_key, ep, fit_seed in episode_plan:
        # pop: each entry holds large N x N matrices and is used once.
        entry = fitted.pop(cache_key, None)
        cache_entries.pop(cache_key, None)
        if entry is not None:
            baseline_nlls = entry["nlls"]
            baseline_R = {k: v.to(device) for k, v in entry["R_dict"].items()}
            baseline_y_nlls = entry["y_nlls"]
        else:
            # Serial fitting (one worker or a GPU baseline device): results moved to the
            # evaluation device; RNG saved/restored and one BLAS thread, matching the pool.
            with _snapshot_rng_and_threads(cpu_threads=1 if baseline_device.type == "cpu" else None):
                baseline_nlls, baseline_R, baseline_y_nlls = eval_baselines_episode(
                    ep={k: (v.to(baseline_device) if isinstance(v, Tensor) else v) for k, v in ep.items()},
                    device=baseline_device,
                    fit_seed=fit_seed,
                    **fit_kwargs,
                )
            baseline_R = {k: v.to(device) for k, v in baseline_R.items()}
            if use_cache:
                save_baseline_entry(
                    args.baseline_cache,
                    fingerprint,
                    cache_key,
                    {
                        "nlls": baseline_nlls,
                        "R_dict": {k: v.cpu() for k, v in baseline_R.items()},
                        "y_nlls": baseline_y_nlls,
                    },
                )

        # Reuse an episode's cached scored results.
        want_plot = local_i == args.plot_episode
        res_cached = results_entries.get(str(ep_i)) if use_results_cache else None
        if res_cached is not None and not want_plot:
            # The plotted episode is always rescored (the cache keeps no R matrices).
            all_y_space_nlls.append(res_cached["y_space_nlls"])
            all_total_nlls.append(res_cached["total_nlls"])
            all_episode_meta.append(res_cached["meta"])
            all_nlls.append(res_cached["nlls"])
            print(f"  ep {ep_i:04d}: reusing scored results (--results_cache)")
            continue

        # Use the episode's own PIT when it has one (ERA5).
        marginal_pit = ep.get("marginal_pit")
        if marginal_pit is None and (tabicl_marginal is not None or marginal_regressor is not None):
            marginal_pit = _marginal_pit(
                ep=ep,
                tabicl_marginal=tabicl_marginal,
                k_folds=tabicl_pit_k_folds,
                device=device,
                marginal_backend=marginal_backend,
                marginal_regressor=marginal_regressor,
                marginal_probs_n=args.marginal_probs_n,
                seed=ep_i,
            )
            if marginal_pit is None:
                print(f"  [ep {ep_i}] fewer than 2 training points — falling back to oracle z_train for this episode")

        # Teacher-forced autoregressive chain (ERA5 precomputes it).
        if (
            args.autoregressive
            and "ar_log_pdf" not in ep
            and (args.ar_n_episodes is None or local_i < args.ar_n_episodes)
        ):
            if tabicl_marginal is None:
                raise RuntimeError("autoregressive scoring needs the loaded TabICL marginal")
            ep["ar_log_pdf"] = (
                autoregressive_log_pdf(
                    tabicl_marginal,
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

        # Zero-mean GP baselines on the marginal's z_train, fitted here (not cached),
        # seeded from fit_seed with the RNG restored and one thread.
        if args.zeromean_gp and marginal_pit is not None:
            with _snapshot_rng_and_threads(
                seed=fit_seed + 1,
                cpu_threads=1 if baseline_device.type == "cpu" else None,
            ):
                zm_nlls, zm_R, zm_y_nlls = _eval_zero_mean_gp_baselines(
                    ep=ep,
                    marginal_pit=marginal_pit,
                    device=baseline_device,
                    n_steps=args.n_steps_zeromean_gp,
                    lr=args.lr_zeromean_gp,
                    n_restarts=args.n_restarts_zeromean_gp,
                    oracle_mode=oracle_mode,
                    prior_cfg=prior_cfg,
                )
            baseline_nlls = {**baseline_nlls, **zm_nlls}
            baseline_R = {**baseline_R, **{k: v.to(device) for k, v in zm_R.items()}}
            baseline_y_nlls = {**baseline_y_nlls, **zm_y_nlls}

        icl_nlls, icl_R, R_oracle, y_space_nlls, icl_y_parts = _eval_icl_episode(
            ep=ep,
            icl_model=icl_model,
            device=device,
            marginal_pit=marginal_pit,
        )
        all_y_space_nlls.append(y_space_nlls)

        n_test = ep["z_test"].shape[0]
        # Per-point oracle rows. Episodes with rescaled targets carry a per-point
        # Jacobian shift, applied to the baselines' marginal and total (not the copula).
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
        all_total_nlls.append(total_nlls)
        all_episode_meta.append(
            {
                "ep_i": ep_i,
                "n_test": n_test,
                "kernel": _kernel_composition_label(ep),
            }
        )

        nlls = {**baseline_nlls, **icl_nlls}
        R_dict = {**baseline_R, **icl_R}

        icl_nll = nlls.get("icl", float("nan"))
        ora_nll = nlls.get("oracle", float("nan"))
        ranked_baselines = sorted(
            ((k, v) for k, v in nlls.items() if k not in _NON_FITTED_EXCLUDED),
            key=lambda kv: kv[1],
        )
        top5 = ranked_baselines[:5]
        # Best fitted baseline per episode, selected by nested CV over the test points.
        holdout_seed = (args.seed * 1_000_003 + ep_i) % (2**31 - 1)
        best_nll, mode_key, fold_details = _select_best_baseline_cv(
            baseline_R,
            ep["z_test"].to(device),
            args.n_folds,
            args.min_fold_size,
            holdout_seed,
        )
        nlls["best_baseline"] = best_nll
        all_nlls.append(nlls)

        # Persist this episode's scored results.
        if use_results_cache:
            results_entries[str(ep_i)] = _jsonable(
                {
                    "nlls": nlls,
                    "total_nlls": total_nlls,
                    "y_space_nlls": y_space_nlls,
                    "meta": all_episode_meta[-1],
                }
            )
            _save_results_cache(args.results_cache, results_fp, results_entries)

        if local_i == args.plot_episode:
            plot_R_dict = R_dict
            plot_R_oracle = R_oracle
            if mode_key is not None:
                plot_best_key = mode_key
                plot_best_R = R_dict[mode_key]

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

    _report_results(
        all_episode_meta=all_episode_meta,
        all_nlls=all_nlls,
        all_total_nlls=all_total_nlls,
        all_y_space_nlls=all_y_space_nlls,
        args=args,
        dataset_dir=dataset_dir,
        era5=era5,
        live_generate=live_generate,
        n_ep=n_ep,
        plot_R_dict=plot_R_dict,
        plot_R_oracle=plot_R_oracle,
        plot_best_R=plot_best_R,
        plot_best_key=plot_best_key,
    )


def main() -> None:
    run_evaluation(parse_eval_spec())


if __name__ == "__main__":
    main()
