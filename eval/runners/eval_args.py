"""eval_checkpoint configuration: the typed Hydra schema and its validation (no checkpoint or data loading).

Every field is a Hydra override, e.g.
    python -m eval.runners.eval_checkpoint ckpt=<ckpt> era5.enabled=true n_episodes=400
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from omegaconf import MISSING

from copula_inter.backend_registry import EVAL_Z_TRAIN_SOURCES, get_backend
from eval.baselines.autoregressive import AR_CONDITIONINGS, AR_ORDERS
from eval.baselines.classical import GP_VAL_SELECT_MODES
from eval.configs.checkpoints import resolve_checkpoint
from eval.configs.constants import N_CONTEXT
from eval.data.era5_episodes import DEFAULT_CORPUS_DIR as ERA5_DEFAULT_CORPUS_DIR
from eval.runners.hydra_cli import check_choice, compose_spec

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Era5Spec:
    """Real ARCO-ERA5 episodes: same baselines and tables, no oracle rows; the shared z_test is the marginal's PIT."""

    # Evaluate on real ERA5 instead of synthetic GP draws; excludes dataset_dir and live_generate.
    enabled: bool = False
    # Cached monthly NetCDF corpus (eval/data/fetch_era5_global.py); defaults to the held-out val years.
    corpus_dir: str = ERA5_DEFAULT_CORPUS_DIR
    # Points per side of each region (D = grid_size^2).
    grid_size: int = 24
    # In-context points P per episode; the other grid_size^2 - P points are targets.
    n_context: int = N_CONTEXT
    # Region box width in degrees, drawn per episode; boxes too small for the grid are redrawn.
    box_deg_min: float = 5.0
    box_deg_max: float = 25.0
    # Draw grid_size and context fraction per episode from era5_live's training ranges.
    vary_geometry: bool = False
    # Episodes per batched PIT call (fixed geometry only).
    pit_batch: int = 8
    # Z-score each episode's target before fitting the baselines and add log(std) back, so the
    # tables stay in raw Kelvin nats; the GP hyperpriors assume O(1) targets.
    standardize_y: bool = True
    # Keep only the N most recent monthly files. Capping below 60 files switches the corpus
    # from memory-mapped to eager loading, so this can cost RAM.
    max_months: int | None = None


@dataclass
class AutoregressiveSpec:
    """The chain-rule row: the same marginal revealing test points one at a time (eval/baselines/autoregressive.py)."""

    enabled: bool = True
    # "random": seeded per-episode permutation; "natural": grid order (a best case).
    order: str = "random"
    # "teacher_forcing" appends the true y (a joint density of y_test); "sample" appends a draw,
    # which is NOT comparable to the other rows.
    conditioning: str = "teacher_forcing"
    # Cap on the chain's context (the episode's own P are always kept); None keeps every revealed point.
    max_context: int | None = None
    # Run the chain on the first N episodes only; the rest report nan.
    n_episodes: int | None = None


@dataclass
class BaselineSpec:
    """Classical baseline fits (eval/baselines/classical.py) and their cache."""

    # GP-MLE (and ARD) Adam steps, learning rate and random restarts.
    n_steps_mle: int = 1000
    lr_mle: float = 0.05
    n_restarts_mle: int = 5
    # Zero-mean GP (RBF, Matern32) fitted on the learned marginal's z_train; a no-op under oracle.
    zeromean_gp: bool = True
    n_steps_zeromean_gp: int = 500
    lr_zeromean_gp: float = 0.05
    n_restarts_zeromean_gp: int = 2
    # Deep kernel learning; each restart gets a fresh feature extractor.
    n_steps_dkl: int = 5000
    lr_dkl: float = 0.01
    n_restarts_dkl: int = 2
    # PerEpisodeTransformer steps and early-stopping patience.
    n_steps_per_ep: int = 5000
    patience_per_ep: int = 500
    # GP-MLE kernels that select their fit on a held-out 20% split: "ard", "always" or "never".
    gp_val_select: str = "ard"
    # "cpu" (faster for these 32-point fits), "cuda", or "auto" to follow the top-level device.
    device: str = "cpu"
    # Fitting processes; 0 = one per physical core (capped at 32), 1 = serial. CPU only.
    workers: int = 0
    # Per-episode fitted-baseline cache, keyed on the episode config and every fit setting; null disables it.
    cache: str | None = "./baseline_cache.pt"
    # Refit every baseline (and rescore every episode) even when a cache entry matches.
    refresh: bool = False


@dataclass
class MarginalSpec:
    """The marginal whose PIT gives the copula model its z_train."""

    # "tabicl": K-fold PIT from the frozen TabICL marginal (the deployment setting);
    # "oracle": exact GP-LOO PIT (synthetic only, no total-NLL row); exaone/tabpfn/tabldm:
    # another foundation model through the same batched PIT.
    z_train_source: str = "tabicl"
    # Quantile-grid size for exaone/tabpfn/tabldm.
    probs_n: int = 99
    # TabICL checkpoint (path or MARGINAL_FAMILIES name); default: the episode config's tabicl.ckpt.
    tabicl_ckpt: str | None = None
    # PIT fold count; default: the episode config's tabicl.pit_k_folds, then pit.DEFAULT_K_FOLDS.
    tabicl_pit_k_folds: int | None = None
    # float16 autocast for the frozen TabICL forward passes; off keeps float32 quantiles.
    tabicl_amp: bool = False


@dataclass
class SelectionSpec:
    """Nested-CV best-of-baselines selection."""

    # Leave-one-fold-out folds for the per-episode best_baseline pick (capped at n_test // min_fold_size).
    n_folds: int = 5
    # Minimum points on each side of a split; episodes with fewer than 2 such folds report nan.
    min_fold_size: int = 20
    # Test-point floor per episode; default 2 * min_fold_size. Raises data.N_min for live
    # generation, skips smaller episodes from a dataset.
    min_test_points: int | None = None
    # "prior" or "posterior" R_star; default: the checkpoint config's data.oracle_mode, else "prior".
    oracle_mode: str | None = None


@dataclass
class OutputSpec:
    """Plots, per-episode dumps and the scored-results cache."""

    # Directory for the corr_grid figure.
    out_dir: str = os.path.join(_REPO_ROOT, "eval", "results")
    # Local index of the episode whose correlation grid is plotted.
    plot_episode: int = 0
    # Write per-episode NLLs to this JSON; runs sharing config/seed/live_generate see the same episodes.
    dump_episodes: str | None = None
    # Per-episode scored results, reused on restart; keyed on the checkpoint, so give
    # concurrent runs distinct paths. null disables it.
    results_cache: str | None = "./eval_results_partial.json"


@dataclass
class EvalSpec:
    """Evaluate a copula checkpoint against the classical baselines and the oracle."""

    # Checkpoint path, or a CHECKPOINT_FAMILIES name[:step].
    ckpt: str = MISSING
    # Hydra config defining the episode distribution, independent of the checkpoint's own
    # config, so the baseline cache survives switching checkpoints.
    config: str = "conf/config.yaml"
    # Episode directory (overrides the config's training.dataset_dir); disables live generation by default.
    dataset_dir: str | None = None
    # Generate episodes on the fly; default: true unless dataset_dir is set.
    live_generate: bool | None = None
    # Force every even live episode to one elementary kernel.
    alternate_noncomposite: bool = True
    n_episodes: int = 30
    # First dataset episode (dataset_dir only).
    episode_idx: int = 0
    # Global index of the first live/ERA5 episode, so shards reproduce one stream bit-identically.
    episode_offset: int = 0
    device: str = "auto"
    seed: int = 42
    # Fail after scoring unless this fraction of attempted episodes has a finite ICL total NLL.
    min_icl_coverage: float = 0.0
    era5: Era5Spec = field(default_factory=Era5Spec)
    autoregressive: AutoregressiveSpec = field(default_factory=AutoregressiveSpec)
    baselines: BaselineSpec = field(default_factory=BaselineSpec)
    marginal: MarginalSpec = field(default_factory=MarginalSpec)
    selection: SelectionSpec = field(default_factory=SelectionSpec)
    output: OutputSpec = field(default_factory=OutputSpec)


def validate_eval_spec(spec: EvalSpec) -> None:
    """Raise ValueError for an invalid or inconsistent specification."""
    check_choice("marginal.z_train_source", spec.marginal.z_train_source, EVAL_Z_TRAIN_SOURCES)
    check_choice("autoregressive.order", spec.autoregressive.order, AR_ORDERS)
    check_choice("autoregressive.conditioning", spec.autoregressive.conditioning, AR_CONDITIONINGS)
    check_choice("baselines.gp_val_select", spec.baselines.gp_val_select, GP_VAL_SELECT_MODES)
    check_choice("baselines.device", spec.baselines.device, ("cpu", "cuda", "auto"))
    check_choice("selection.oracle_mode", spec.selection.oracle_mode, ("prior", "posterior"))
    if not 0 <= spec.min_icl_coverage <= 1:
        raise ValueError("min_icl_coverage must lie in [0, 1]")
    if spec.min_icl_coverage and spec.marginal.z_train_source == "oracle":
        raise ValueError("min_icl_coverage requires a learned marginal")
    # The autoregressive chain needs the TabICL marginal.
    z_src = spec.marginal.z_train_source
    if spec.autoregressive.enabled and (z_src == "oracle" or not get_backend(z_src).autoregressive):
        raise ValueError(
            "autoregressive.enabled requires marginal.z_train_source=tabicl: the chain needs "
            "a marginal callable with a growing context. Re-run with "
            "autoregressive.enabled=false for this marginal backend."
        )


def prepare_eval_spec(spec: EvalSpec) -> EvalSpec:
    """Resolve the checkpoint name and validate; returns the same object."""
    spec.ckpt = resolve_checkpoint(spec.ckpt)
    if os.path.isdir(spec.ckpt):
        # Pin a run directory to the step file it loads *now*, so the results cache is keyed on
        # that file's bytes (a later step in the same directory must not reuse cached scores).
        from inference.copula_inference import _resolve_copula_checkpoint

        spec.ckpt = _resolve_copula_checkpoint(spec.ckpt)
    validate_eval_spec(spec)
    return spec


def compose_eval_spec(overrides: list[str] | None = None) -> EvalSpec:
    """The validated EvalSpec for these Hydra overrides, without loading checkpoints or data."""
    return prepare_eval_spec(compose_spec("eval_checkpoint", EvalSpec, overrides))
