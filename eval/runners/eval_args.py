"""eval_checkpoint configuration: the typed Hydra schema and its validation (no checkpoint or data loading).

Every field is a Hydra override, e.g.
    python -m eval.runners.eval_checkpoint ckpt=<ckpt> era5.enabled=true n_episodes=400
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from omegaconf import MISSING

from copula_inter.backend_registry import EVAL_Z_TRAIN_SOURCES, get_backend
from copula_inter.checkpoint_registry import resolve_checkpoint
from eval.configs.constants import (
    AR_CONDITIONINGS,
    AR_ORDERS,
    ERA5_DEFAULT_CORPUS_DIR,
    GP_MEANS,
    GP_PREDICT_MODES,
    GP_VAL_SELECT_MODES,
    N_CONTEXT,
)
from eval.runners.hydra_cli import check_choice, compose_spec

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Era5Spec:
    """Real ARCO-ERA5 episodes: same baselines and tables, no oracle rows; the shared z_test is the marginal's PIT."""

    # Evaluate on real ERA5 instead of synthetic GP draws; excludes dataset_dir and live_generate.
    enabled: bool = False
    # Cached monthly NetCDF corpus (src/copula_inter/era5/fetch_global.py); defaults to the held-out val years.
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
class SdwpfSpec:
    """SDWPF wind farm (eval/data/real/sdwpf.py): one timestamp per episode, rows = turbines, target Patv (kW)."""

    # Directory with the SCADA and location CSVs; default copula_inter.paths.SDWPF_DATA_DIR.
    data_dir: str | None = None
    n_context: int = 32
    n_query: int = 64
    # "random" or "block" (the n_query turbines nearest a random anchor are the queries).
    mask: str = "random"
    # "f0" = (x, y); "f1b" = (x, y, u, v) rotated by the context turbines' mean wind direction.
    features: str = "f0"
    # "partial_load": keep a timestamp only if every selected turbine is in (0, atom_hi_frac * rated);
    # "all": no filter (NLL unreliable: Patv has atoms at 0 and rated).
    regime: str = "partial_load"
    # Added to (Ndir + Wdir) before the mod 360 (F1b only): one of 0, 90, 180, -90, cross-fitted by the audit.
    direction_offset_deg: int = 0
    atom_hi_frac: float = 0.99
    # Rated power = this quantile of cleaned Patv over all days.
    rated_quantile: float = 0.999
    # Candidate timestamps each episode owns; it takes the first that passes the regime.
    timestamp_tries: int = 8


@dataclass
class KaustSpec:
    """KAUST 2022 sub-competitions 1a/1b (eval/data/real/kaust.py): one square window per episode, target z."""

    # Directory with the four <variant>_train.csv files; default copula_inter.paths.KAUST_DATA_DIR.
    data_dir: str | None = None
    # "<sub-competition>_<dataset>": 1a_* = 90K training points, 1b_* = 900K; *_1 = mean + iid noise
    # (the negative control), *_2 = zero-mean nonstationary Matern GP (the main result).
    variant: str = "1a_2"
    n_context: int = 128
    n_query: int = 64
    # "random" or "block" (the n_query window points nearest a random anchor are the queries).
    mask: str = "random"
    # The unit square is cut into window_cells x window_cells cells, a multiple of the 5 strips; the
    # window side is 1 / window_cells and the CI cluster is the cell holding the window centre.
    window_cells: int = 15
    # Window positions drawn per episode until one holds n_context + n_query points.
    window_tries: int = 20


@dataclass
class EmnistSpec:
    """EMNIST balanced (eval/data/real/emnist.py): one image per episode, rows = pixels, dequantized intensity."""

    # Directory with the four emnist-balanced-*-idx*-ubyte.gz files; default copula_inter.paths.EMNIST_DATA_DIR.
    data_dir: str | None = None
    # "test" for reported runs, "train" for debugging.
    split: str = "test"
    n_context: int = 50
    n_query: int = 64
    # "random" or "block" (the n_query pixels nearest a random anchor pixel are the queries).
    mask: str = "random"


@dataclass
class CaliforniaSpec:
    """California Housing (eval/data/real/california.py): rows = block groups, target median value ($100k)."""

    # sklearn data_home for fetch_california_housing; default copula_inter.paths.CALIFORNIA_DATA_DIR.
    data_dir: str | None = None
    n_context: int = 128
    n_query: int = 64
    # "random" or "block" (the n_query rows nearest a random anchor in (lat, lon) are the queries).
    mask: str = "random"
    # "f0" = all 8 features; "f1" = without Latitude and Longitude (ablation).
    features: str = "f0"
    # The target's maximum (the survey's top-code, 5.00001) must be >= this; rows at the top-code are dropped.
    censor_cap: float = 5.0


@dataclass
class MeuseSpec:
    """Meuse zinc (eval/data/real/meuse.py): repeated K-fold over the 155 points, target log(zinc)."""

    # Directory with meuse.csv; default copula_inter.paths.MEUSE_DATA_DIR.
    data_dir: str | None = None
    # "f0" = (x, y); "f1" = (x, y, dist, elev).
    features: str = "f0"
    # Episode ep_i is fold ep_i % n_folds of repetition ep_i // n_folds.
    n_folds: int = 3
    n_repeats: int = 67


@dataclass
class Ctr23Spec:
    """OpenML-CTR23 (suite 353; eval/data/real/ctr23.py): one dataset per run, rows = instances."""

    # sklearn data_home for fetch_openml; default copula_inter.paths.CTR23_DATA_DIR.
    data_dir: str | None = None
    # OpenML data id of one CTR23 dataset (eval.data.real.ctr23.CTR23_TASKS); required.
    data_id: int = 0
    n_context: int = 128
    n_query: int = 64
    # "dyadic_knn": the n_query / 2 nearest neighbours of each of two anchors are the queries;
    # "random": uniform query set (the negative control).
    mask: str = "dyadic_knn"
    # A dataset whose most common target value holds more than this fraction of rows is excluded.
    max_atom_frac: float = 0.05


@dataclass
class RealSpec:
    """Real-data benchmark episodes (eval/data/real): same baselines and tables, no oracle rows."""

    # Dataset name (eval.data.real.REAL_DATASETS); null = off. Excludes era5.enabled, dataset_dir, live_generate.
    dataset: str | None = None
    # Episodes per batched PIT call (fixed-shape datasets only).
    pit_batch: int = 8
    # Z-score each episode's target before fitting the baselines and add log(std) back (as era5.standardize_y).
    standardize_y: bool = True
    sdwpf: SdwpfSpec = field(default_factory=SdwpfSpec)
    kaust: KaustSpec = field(default_factory=KaustSpec)
    emnist: EmnistSpec = field(default_factory=EmnistSpec)
    california: CaliforniaSpec = field(default_factory=CaliforniaSpec)
    meuse: MeuseSpec = field(default_factory=MeuseSpec)
    ctr23: Ctr23Spec = field(default_factory=Ctr23Spec)


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
    # What the GP-MLE, DKL and zero-mean GP rows predict at the queries: "posterior" (conditioned
    # on the context, the competitor) or "prior" (ignores the context; reproduces tables made
    # before 2026-11-13, when the baselines followed data.oracle_mode=prior).
    gp_predict: str = "posterior"
    # Mean function of the GP-MLE and DKL fits: "zero", "constant" or "linear" (affine in x).
    gp_mean: str = "constant"
    # Laplace-evidence kernel selection rows (gp_laplace*). false skips them and reuses caches
    # written without them (e.g. to add the attainable copulas to an existing run).
    laplace: bool = True
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
    # "oracle": exact GP-LOO PIT (synthetic only, no total-NLL row);
    # "tabpfn35_fast" / "kumo_small": that backend through the same batched PIT.
    z_train_source: str = "tabicl"
    # Phase-A weights for tabpfn35_fast / kumo_small; default: the copula
    # checkpoint's own data.z_train_marginal_ckpt (the marginal it was trained on).
    backend_ckpt: str | None = None
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


@dataclass
class AttainableSpec:
    """Attainable copulas under the frozen marginals (synthetic episodes, TabICL marginal only).

    Tracks the best Gaussian copula (full rank), the best rank-r Gaussian copula per rank, and a
    lower bound on the best copula of any family (exact for calibrated marginals).
    """

    # Score the attainable copulas and print the copula-efficiency table.
    enabled: bool = True
    # Draws of y_Q from the known-theta GP posterior for M = E[z z'] (entry SE ~ 1/sqrt(n)).
    n_samples: int = 8192
    # Extra ranks r for the best rank-r Gaussian copula (the head's own rank is always included).
    ranks: list[int] = field(default_factory=lambda: [1, 2, 4, 16])
    # L-BFGS iterations of each R_dagger fit.
    max_iter: int = 500


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
    real: RealSpec = field(default_factory=RealSpec)
    autoregressive: AutoregressiveSpec = field(default_factory=AutoregressiveSpec)
    baselines: BaselineSpec = field(default_factory=BaselineSpec)
    marginal: MarginalSpec = field(default_factory=MarginalSpec)
    selection: SelectionSpec = field(default_factory=SelectionSpec)
    attainable: AttainableSpec = field(default_factory=AttainableSpec)
    output: OutputSpec = field(default_factory=OutputSpec)


RealDatasetSpec = SdwpfSpec | KaustSpec | EmnistSpec | CaliforniaSpec | MeuseSpec | Ctr23Spec


def real_dataset_spec_of(real: RealSpec) -> RealDatasetSpec:
    """The <name> sub-config of real.dataset."""
    assert real.dataset is not None
    sub: RealDatasetSpec = getattr(real, real.dataset)
    return sub


def real_dataset_spec(spec: EvalSpec) -> RealDatasetSpec:
    """The real.<name> sub-config of the selected real.dataset."""
    return real_dataset_spec_of(spec.real)


def _validate_real(spec: EvalSpec) -> None:
    """real.dataset: exclusive source, learned marginal, enough queries, and every choice field in range."""
    from eval.data.real import REAL_DATASETS, dataset_module

    check_choice("real.dataset", spec.real.dataset, REAL_DATASETS)
    if spec.era5.enabled:
        raise ValueError("real.dataset and era5.enabled are mutually exclusive episode sources.")
    if spec.dataset_dir is not None:
        raise ValueError("real.dataset and dataset_dir are mutually exclusive episode sources.")
    if spec.live_generate:
        raise ValueError("real.dataset and live_generate are mutually exclusive episode sources.")
    if spec.marginal.z_train_source == "oracle":
        raise ValueError(
            "real.dataset has no oracle marginal: real data has no generating GP to take an exact "
            "LOO-PIT residual from. Use marginal.z_train_source=tabicl (default) or tabpfn."
        )
    name = spec.real.dataset
    assert name is not None
    sub = real_dataset_spec(spec)
    # Dataset-specific choice fields and ranges first, so a bad choice is named before the shape checks.
    dataset_module(name).validate(sub)
    n_context, n_query = dataset_module(name).episode_shape(sub)
    min_test = (
        spec.selection.min_test_points
        if spec.selection.min_test_points is not None
        else 2 * spec.selection.min_fold_size
    )
    if n_query < min_test:
        raise ValueError(
            f"real.{name}.n_query={n_query} is below selection.min_test_points={min_test}: "
            "best_baseline needs at least two nested-CV folds of test points."
        )
    if n_context < 2:
        raise ValueError(f"real.{name}.n_context must be at least 2 for the K-fold PIT")


def validate_eval_spec(spec: EvalSpec) -> None:
    """Raise ValueError for an invalid or inconsistent specification."""
    check_choice("marginal.z_train_source", spec.marginal.z_train_source, EVAL_Z_TRAIN_SOURCES)
    check_choice("autoregressive.order", spec.autoregressive.order, AR_ORDERS)
    check_choice("autoregressive.conditioning", spec.autoregressive.conditioning, AR_CONDITIONINGS)
    check_choice("baselines.gp_val_select", spec.baselines.gp_val_select, GP_VAL_SELECT_MODES)
    check_choice("baselines.device", spec.baselines.device, ("cpu", "cuda", "auto"))
    check_choice("baselines.gp_predict", spec.baselines.gp_predict, GP_PREDICT_MODES)
    check_choice("baselines.gp_mean", spec.baselines.gp_mean, GP_MEANS)
    if spec.attainable.n_samples < 2 or spec.attainable.max_iter < 1 or min(spec.attainable.ranks, default=1) < 1:
        raise ValueError("attainable.n_samples must be >= 2, attainable.max_iter and every attainable.ranks >= 1")
    if spec.era5.enabled:
        if spec.dataset_dir is not None:
            raise ValueError("era5.enabled and dataset_dir are mutually exclusive episode sources.")
        if spec.live_generate:
            raise ValueError("era5.enabled and live_generate are mutually exclusive episode sources.")
        if spec.marginal.z_train_source == "oracle":
            raise ValueError(
                "era5.enabled has no oracle marginal: real ERA5 has no generating GP to take "
                "an exact LOO-PIT residual from. Use marginal.z_train_source=tabicl (default), "
                "tabpfn35_fast or kumo_small."
            )
    if spec.real.dataset is not None:
        _validate_real(spec)
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
