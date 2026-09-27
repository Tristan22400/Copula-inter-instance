"""Evaluate a copula checkpoint against the classical baselines and the oracle.

Usage:
    python -m eval.runners.eval_checkpoint ckpt=kernel-sweep-all-tabicl-retrain-15k
    python -m eval.runners.eval_checkpoint ckpt=<ckpt> era5.enabled=true n_episodes=400

Settings are Hydra overrides on eval_args.EvalSpec; `--cfg job` prints them all.
Episodes are generated from the `config` data config (not the checkpoint's),
so the same config and seed give the same episodes for any checkpoint.
Baseline fits (classical.py) do not depend on the checkpoint and are cached
per episode in baselines.cache; scored results are cached per checkpoint in
output.results_cache. Baselines are fitted on CPU (baselines.device) across
baselines.workers processes, one per physical core by default.
"""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor

from eval.runners.episode_scoring import _episode_baselines, _EvalContext, _prefit_baselines, _score_episode
from eval.runners.eval_inputs import (
    _load_episodes,
    _load_full_config,
    _load_models,
    _plan_episodes,
    _resolve_episode_source,
)
from eval.runners.eval_tables import (
    _METHOD_LABELS,
    _RANK_KEYS,
    _TOTAL_RANK_ORDER,
    _ar_note,
    _print_rank_table,
    _print_table,
    _print_total_nll_table,
    _print_y_space_oracle,
)

if TYPE_CHECKING:
    pass

from copula_inter.artifacts import artifact_identity, atomic_json_save
from copula_inter.config_path import config_dict
from copula_inter.config_path import config_dir as project_config_dir
from eval.baselines.classical import (
    baseline_fingerprint,
    load_baseline_cache,
)
from eval.data.era5_episodes import (
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
from eval.results import (
    require_coverage,
)
from eval.results import (
    save_results_cache as _save_results_cache,
)
from eval.runners.eval_args import EvalSpec, prepare_eval_spec, validate_eval_spec
from eval.runners.hydra_cli import hydra_entry
from eval.viz.correlation_plots import plot_corr_grid


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _results_fingerprint(
    baseline_fp: dict,
    spec: EvalSpec,
    tabicl_pit_k_folds: int,
    resolved_marginal: str | None = None,
) -> dict:
    """Digest of everything that determines an episode's scored results: the baseline fingerprint plus the checkpoint, marginal and CV settings."""
    ckpt = os.path.abspath(spec.ckpt) if spec.ckpt else None
    return {
        "baseline": baseline_fp,
        "ckpt_identity": artifact_identity(ckpt),
        "z_train_source": spec.marginal.z_train_source,
        "marginal_identity": artifact_identity(
            resolved_marginal if resolved_marginal is not None else spec.marginal.tabicl_ckpt
        ),
        "tabicl_pit_k_folds": tabicl_pit_k_folds,
        "tabicl_amp": spec.marginal.tabicl_amp,
        "marginal_probs_n": spec.marginal.probs_n,
        "n_folds": spec.selection.n_folds,
        "min_fold_size": spec.selection.min_fold_size,
        "seed": spec.seed,
        "zeromean_gp": spec.baselines.zeromean_gp,
        "n_steps_zeromean_gp": spec.baselines.n_steps_zeromean_gp,
        "lr_zeromean_gp": spec.baselines.lr_zeromean_gp,
        "n_restarts_zeromean_gp": spec.baselines.n_restarts_zeromean_gp,
        # Bump when the derived total-NLL or rank rows change.
        "result_schema": 3,
        "autoregressive": bool(spec.autoregressive.enabled),
        "ar_order": spec.autoregressive.order if spec.autoregressive.enabled else None,
        "ar_conditioning": spec.autoregressive.conditioning if spec.autoregressive.enabled else None,
        "ar_max_context": spec.autoregressive.max_context if spec.autoregressive.enabled else None,
        "ar_n_episodes": spec.autoregressive.n_episodes if spec.autoregressive.enabled else None,
    }


def _report_results(
    all_episode_meta: list[dict],
    all_nlls: list[dict[str, float]],
    all_total_nlls: list[dict[str, dict[str, float]]],
    all_y_space_nlls: list[dict[str, dict[str, float]]],
    spec: EvalSpec,
    dataset_dir: str | None,
    era5: bool,
    live_generate: bool,
    n_ep: int,
    plot_R_dict: dict[str, Tensor] | None,
    plot_R_oracle: Tensor | None,
    plot_best_R: Tensor | None,
    plot_best_key: str | None,
) -> None:
    """Print the result tables, enforce min_icl_coverage, dump per-episode scores, and plot correlation grids."""
    if not all_nlls:
        raise RuntimeError(f"no episodes evaluated successfully out of {n_ep} requested")

    _print_table(all_nlls, z_train_source=spec.marginal.z_train_source, era5=era5, attempted=n_ep)
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
        z_train_source=spec.marginal.z_train_source,
        era5=era5,
        attempted=n_ep,
        ar_note=_ar_note(
            all_total_nlls, spec.autoregressive.order, spec.autoregressive.conditioning, spec.autoregressive.max_context
        ),
    )
    _print_rank_table(
        all_nlls,
        _RANK_KEYS,
        title="Method rank — z-space copula NLL (shared ground-truth marginal)",
        z_train_source=spec.marginal.z_train_source,
    )
    total_only = [
        {k: m.get(k, _NAN_PARTS).get("total", float("nan")) for k, _ in _TOTAL_RANK_ORDER} for m in all_total_nlls
    ]
    _print_rank_table(
        total_only,
        _TOTAL_RANK_ORDER,
        title="Method rank — total NLL, Y-space (own marginal per method)",
        z_train_source=spec.marginal.z_train_source,
    )

    if spec.min_icl_coverage:
        valid_icl = sum(
            bool(np.isfinite(row.get("icl", _NAN_PARTS).get("total", float("nan")))) for row in all_total_nlls
        )
        require_coverage(valid_icl, n_ep, spec.min_icl_coverage)

    if spec.output.dump_episodes:
        dump: dict[str, Any] = {
            "ckpt": spec.ckpt,
            "config": spec.config,
            "seed": spec.seed,
            "live_generate": live_generate,
            "dataset_dir": dataset_dir,
            "z_train_source": spec.marginal.z_train_source,
            "episodes": [
                {**meta, "nlls": nlls, "total_nlls": total_nlls}
                for meta, nlls, total_nlls in zip(all_episode_meta, all_nlls, all_total_nlls)
            ],
        }
        atomic_json_save(dump, spec.output.dump_episodes)
        print(f"Dumped {len(dump['episodes'])} per-episode NLLs to: {spec.output.dump_episodes}")

    # ---- Correlation heatmap ----
    if plot_R_dict is not None and plot_R_oracle is not None:
        import matplotlib

        matplotlib.use("Agg")

        os.makedirs(spec.output.out_dir, exist_ok=True)
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
            title=f"Correlation estimators — episode {spec.episode_idx + spec.output.plot_episode}",
        )
        out_path = os.path.join(spec.output.out_dir, f"corr_grid_ep{spec.output.plot_episode}.png")
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        print(f"Saved corr_grid to: {out_path}")

    print("Done.")


def run_evaluation(spec: EvalSpec) -> None:
    """Score one validated evaluation specification."""
    validate_eval_spec(spec)
    if spec.config == "conf/config.yaml" and not os.path.isfile(spec.config):
        spec.config = os.path.join(project_config_dir(__file__), "config.yaml")
    _set_seed(spec.seed)

    device = torch.device(
        "cuda"
        if (spec.device == "auto" and torch.cuda.is_available())
        else (spec.device if spec.device != "auto" else "cpu")
    )
    print(f"Device: {device}")

    # cfg: the episode-generating config from spec.config (not the checkpoint's).
    cfg = _load_full_config(spec.config)

    (icl_model, icl_rank, marginal_backend, marginal_regressor, tabicl_ckpt, tabicl_marginal, tabicl_pit_k_folds) = (
        _load_models(spec=spec, cfg=cfg, device=device)
    )
    print(f"z_train source (ICL conditioning input): {spec.marginal.z_train_source}")

    # Baselines score against the same oracle_mode as R_star (default "prior").
    oracle_mode = spec.selection.oracle_mode or OmegaConf.select(cfg, "data.oracle_mode", default="prior")
    print(f"Oracle mode: {oracle_mode}")

    # Baseline hyperpriors from cfg.data, falling back to classical._DEFAULT_PRIOR_CFG.
    data_cfg = OmegaConf.select(cfg, "data", default=None)
    prior_cfg = config_dict(data_cfg, resolve=False) if data_cfg is not None else {}
    print(f"GP-MLE restarts: {spec.baselines.n_restarts_mle}")
    print(f"DKL restarts: {spec.baselines.n_restarts_dkl}")

    era5, live_generate, dataset_dir = _resolve_episode_source(spec, cfg)

    # Minimum test points (default 2 * selection.min_fold_size).
    min_test_points = (
        spec.selection.min_test_points
        if spec.selection.min_test_points is not None
        else 2 * spec.selection.min_fold_size
    )

    (dataset, era5_geometry, live_episodes, marginal_regressor, n_available, tabicl_marginal) = _load_episodes(
        spec=spec,
        cfg=cfg,
        dataset_dir=dataset_dir,
        device=device,
        era5=era5,
        live_generate=live_generate,
        marginal_backend=marginal_backend,
        marginal_regressor=marginal_regressor,
        min_test_points=min_test_points,
        n_ep=spec.n_episodes,
        tabicl_marginal=tabicl_marginal,
        tabicl_pit_k_folds=tabicl_pit_k_folds,
    )

    print(
        f"  GP MLE: {spec.baselines.n_steps_mle} steps | DKL: {spec.baselines.n_steps_dkl} steps | "
        f"PerEp: {spec.baselines.n_steps_per_ep} steps (patience={spec.baselines.patience_per_ep})"
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
    baseline_cache = spec.baselines.cache
    fingerprint = baseline_fingerprint(
        cfg,
        live_generate,
        dataset_dir,
        spec.seed,
        icl_rank,
        oracle_mode,
        spec.baselines.n_steps_mle,
        spec.baselines.lr_mle,
        spec.baselines.n_restarts_mle,
        spec.baselines.n_steps_dkl,
        spec.baselines.lr_dkl,
        spec.baselines.n_steps_per_ep,
        spec.baselines.patience_per_ep,
        gp_val_select=spec.baselines.gp_val_select,
        n_restarts_dkl=spec.baselines.n_restarts_dkl,
    )
    if era5:
        assert era5_geometry is not None
        # ERA5 fingerprint also includes the corpus, geometry and marginal.
        fingerprint["era5"] = era5_episode_fingerprint(
            spec.era5.corpus_dir,
            k_folds=tabicl_pit_k_folds,
            marginal=spec.marginal.z_train_source,
            **era5_geometry,
        )
    cache_entries = load_baseline_cache(baseline_cache, fingerprint) if baseline_cache is not None else {}

    # ---- Scored-results cache: the checkpoint-DEPENDENT half of a resume ----
    results_cache = spec.output.results_cache
    results_fp = _results_fingerprint(
        fingerprint,
        spec,
        tabicl_pit_k_folds,
        resolved_marginal=tabicl_ckpt if spec.marginal.z_train_source == "tabicl" else None,
    )
    results_entries = _load_results_cache(results_cache, results_fp) if results_cache is not None else {}
    if spec.baselines.refresh:
        # Refitting also rescores.
        results_entries = {}

    ctx = _EvalContext(
        spec=spec,
        device=device,
        baseline_device=torch.device(str(device) if spec.baselines.device == "auto" else spec.baselines.device),
        icl_model=icl_model,
        tabicl_marginal=tabicl_marginal,
        marginal_backend=marginal_backend,
        marginal_regressor=marginal_regressor,
        tabicl_pit_k_folds=tabicl_pit_k_folds,
        oracle_mode=oracle_mode,
        prior_cfg=prior_cfg,
        fit_kwargs=dict(
            icl_rank=icl_rank,
            n_steps_mle=spec.baselines.n_steps_mle,
            lr_mle=spec.baselines.lr_mle,
            n_steps_dkl=spec.baselines.n_steps_dkl,
            lr_dkl=spec.baselines.lr_dkl,
            n_steps_per_ep=spec.baselines.n_steps_per_ep,
            patience_per_ep=spec.baselines.patience_per_ep,
            oracle_mode=oracle_mode,
            prior_cfg=prior_cfg,
            n_restarts_mle=spec.baselines.n_restarts_mle,
            n_restarts_dkl=spec.baselines.n_restarts_dkl,
            gp_val_select=spec.baselines.gp_val_select,
        ),
        baseline_cache=baseline_cache,
        fingerprint=fingerprint,
    )
    episode_plan = _plan_episodes(
        spec,
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
    # Per-episode metadata for output.dump_episodes.
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
        want_plot = local_i == spec.output.plot_episode
        res_cached = results_entries.get(str(ep_i)) if results_cache is not None else None
        if res_cached is not None and not want_plot:
            all_y_space_nlls.append(res_cached["y_space_nlls"])
            all_total_nlls.append(res_cached["total_nlls"])
            all_episode_meta.append(res_cached["meta"])
            all_nlls.append(res_cached["nlls"])
            print(f"  ep {ep_i:04d}: reusing scored results (output.results_cache)")
            continue

        score = _score_episode(ctx, ep, local_i, ep_i, fit_seed, baselines)
        all_y_space_nlls.append(score.y_space_nlls)
        all_total_nlls.append(score.total_nlls)
        all_episode_meta.append(score.meta)
        all_nlls.append(score.nlls)
        if results_cache is not None:
            results_entries[str(ep_i)] = _jsonable(
                {
                    "nlls": score.nlls,
                    "total_nlls": score.total_nlls,
                    "y_space_nlls": score.y_space_nlls,
                    "meta": score.meta,
                }
            )
            _save_results_cache(results_cache, results_fp, results_entries)
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
        spec=spec,
        dataset_dir=dataset_dir,
        era5=era5,
        live_generate=live_generate,
        n_ep=spec.n_episodes,
        plot_R_dict=plot_R_dict,
        plot_R_oracle=plot_R_oracle,
        plot_best_R=plot_best_R,
        plot_best_key=plot_best_key,
    )


# Retired argparse flags whose key is not their own name.
_FLAG_ALIASES = {
    "ar_order": "autoregressive.order",
    "ar_conditioning": "autoregressive.conditioning",
    "ar_max_context": "autoregressive.max_context",
    "ar_n_episodes": "autoregressive.n_episodes",
    "baseline_device": "baselines.device",
    "baseline_workers": "baselines.workers",
    "baseline_cache": "baselines.cache",
    "refresh_baselines": "baselines.refresh",
    "no_baseline_cache": "baselines.cache=null",
    "no_results_cache": "output.results_cache=null",
    "marginal_probs_n": "marginal.probs_n",
}

main = hydra_entry("eval_checkpoint", EvalSpec, run_evaluation, _FLAG_ALIASES, validate=prepare_eval_spec)


if __name__ == "__main__":
    main()
