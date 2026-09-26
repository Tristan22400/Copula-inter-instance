"""eval_checkpoint command line: argument groups, parsing and validation (no checkpoint or data loading)."""

from __future__ import annotations

import argparse
import os

from copula_inter.backend_registry import EVAL_Z_TRAIN_SOURCES
from copula_inter.pit import DEFAULT_K_FOLDS
from eval.baselines.autoregressive import AR_CONDITIONINGS, AR_ORDERS
from eval.baselines.classical import GP_VAL_SELECT_MODES
from eval.configs.checkpoints import resolve_checkpoint
from eval.configs.constants import N_CONTEXT
from eval.data.era5_episodes import DEFAULT_CORPUS_DIR as ERA5_DEFAULT_CORPUS_DIR

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    """Episode source: config, checkpoint, dataset or live generation."""
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


def _add_era5_args(parser: argparse.ArgumentParser) -> None:
    """Real ARCO-ERA5 episodes."""
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


def _add_autoregressive_args(parser: argparse.ArgumentParser) -> None:
    """The autoregressive (copula-free) chain row."""
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


def _add_baseline_fit_args(parser: argparse.ArgumentParser) -> None:
    """Episode count and the classical baselines' optimisation settings."""
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


def _add_marginal_args(parser: argparse.ArgumentParser) -> None:
    """The marginal whose PIT conditions the model."""
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


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    """Plots, dumps, device and seed."""
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


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    """Nested-CV best-of-baselines selection and episode filters."""
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


def _add_cache_and_runtime_args(parser: argparse.ArgumentParser) -> None:
    """Result caches, baseline workers and coverage checks."""
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


def parse_eval_spec(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate CLI settings without loading checkpoints or data."""
    parser = argparse.ArgumentParser(
        description="Evaluate ICL checkpoint vs baselines on inter-instance copula episodes"
    )
    _add_source_args(parser)
    _add_era5_args(parser)
    _add_autoregressive_args(parser)
    _add_baseline_fit_args(parser)
    _add_marginal_args(parser)
    _add_output_args(parser)
    _add_selection_args(parser)
    _add_cache_and_runtime_args(parser)
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
