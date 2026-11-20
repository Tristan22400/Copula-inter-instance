"""TabICLv2 graph-SCM prior adapter and the settings of the SCM + GP residual prior.

The SCM + GP residual prior (cfg.data.scm_residual) draws one episode as

    y = a_g * g~(x) + r(x) + eps,   r ~ GP(0, c K),   eps ~ N(0, sigma_n^2 I),

where (X, g) is a table from TabICLv2's graph SCM prior (the prior its regressor was
pretrained on), g~ its target standardized over the episode, K the episode's GP kernel at its
own output scale, and a_g^2 ~ LogNormal(loc, scale^2), c = 1 drawn independently
(sample_amplitudes); the residual share lambda = mean_i K_ii / (a_g^2 + mean_i K_ii) is induced.
a_g = 0 is the zero-mean GP prior, c = 0 TabICLv2's prior (rows i.i.d. given the graph). gp_episodes builds the target; this
module owns the settings, the amplitude draw and every use of TabICL's private prior
modules, so a TabICL upgrade touches one file.

Where y sits in the graph. The graph SCM writes the target as one column of some node of a
random DAG; that node may have descendants, and feature columns of those descendants are
functions of the target value the graph wrote. g is that written value, so the graph (X, g)
is a complete, acyclic SCM. The observed target y = a_g g~ + r(X) + eps is a new *leaf* node
whose parents are g and all of X: nothing downstream sees r, so r is independent of X a
priori, its conditional given X is the GP prior, and the conditional oracle (which knows g)
stays exact. Injecting r into g's node before propagation would need r to depend only on
the non-descendants of that node (else the graph has a cycle) and would make the descendant
features informative about r, so p(r | X) would no longer be the GP and no oracle would be
available.
"""

from __future__ import annotations

import importlib
import types
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from copula_inter.type_aliases import Device, HasDataConfig

# TabICLv2's graph SCM only; TabICLv1's mlp / tree / mix SCMs are not offered.
SCM_PRIOR_TYPES = ("graph_scm",)
# The coupled mode (a_g^2 = 1 - lambda, normalized kernel) was removed; configs that still ask
# for it are refused rather than silently switched.
AMPLITUDE_MODES = ("decoupled",)
# scm_prior_type codes saved per episode (-1: a GP-prior episode of an SCM-enabled run).
# 0 and 1 were TabICLv1's mlp_scm and tree_scm in earlier datasets; they are not reused.
SCM_PRIOR_CODES = {"graph_scm": 2}
GP_EPISODE_PRIOR_CODE = -1
# mean_family of an SCM episode (the mean bank's codes are 0 linear, 1 exponential, 2 anomaly):
# its mean is the saved mean_train / mean_test vectors, not a mean-bank function.
MEAN_FAMILY_SCM = 3
# Learned marginals the SCM prior can train with; "analytic" would leak g through m.
_ANALYTIC_SOURCES = ("analytic",)
# Redraws of one episode whose table loses a feature (constant column), is degenerate or is
# filtered as unpredictable, before it is marked invalid and left to generate_gp_batch's top-up.
_MAX_TABLE_ATTEMPTS = 8
# graph_lib PriorConfig of TabICLv2's regressor pretraining, identical in its three stages
# (tabicl_upstream/scripts/train_v2_reg_stage{1,2,3}.sh); other fields keep PriorConfig's defaults.
# Feature count of TabICLv2's regressor pretraining (--min_features 1 --max_features 100,
# log_n_features=False): d = round(Uniform(1, 100)), shared by a group of datasets (here, a call).
V2_MIN_FEATURES = 1
V2_MAX_FEATURES = 100
# Rows and train split of TabICLv2's regressor stage 1 (500K of its 550K pretraining steps):
# --max_seq_len 1024 (no min: fixed), --min_train_size 0.3 --max_train_size 0.9, so
# P = int(1024 * Uniform(0.3, 0.9)) context rows and N = 1024 - P query rows.
V2_SEQ_LEN = 1024
V2_TRAIN_FRAC_MIN = 0.3
V2_TRAIN_FRAC_MAX = 0.9
V2_REGRESSOR_PRIOR_CONFIG: dict[str, Any] = {
    "add_gaussian_noise": False,  # --graph_noise False
    "filter_unpredictable_graphs": True,
    "filter_unpredictable_datasets": True,
    "allow_act_warping": False,
    "min_n_nodes": 2,
    "max_n_nodes": 32,
    "cauchy_dag_offset": 0.0,
}


@dataclass(frozen=True)
class ScmResidualSettings:
    """cfg.data.scm_residual, with the defaults of conf/data/gp_tasks.yaml."""

    enabled: bool = False
    prob: float = 1.0
    prior_type: str = "graph_scm"
    lambda_p0: float = 0.2
    lambda_p1: float = 0.2
    allow_analytic_pit: bool = False
    amplitude_mode: str = "decoupled"
    # Draw the episode shape (d, P, N) from TabICLv2's prior; false keeps data.d_features* and
    # data.P_*/N_* (small shapes for tests and diagnostics).
    prior_shapes: bool = True
    # Context rows P of a prior-shaped call (rows stay V2_SEQ_LEN, N = V2_SEQ_LEN - P). None/None
    # keeps TabICLv2's split P = int(1024 Uniform(0.3, 0.9)); otherwise P in [prior_P_min,
    # prior_P_max], log-uniform or uniform (prior_P_dist). The rows are exchangeable given the
    # graph, so the split changes which rows are context, not the table.
    prior_P_min: int | None = None
    prior_P_max: int | None = None
    prior_P_dist: str = "loguniform"
    mean_amp_lognormal_loc: float = 0.0
    mean_amp_lognormal_scale: float = 1.5

    @classmethod
    def from_cfg(cls, cfg: HasDataConfig) -> ScmResidualSettings:
        """Read cfg.data.scm_residual; a config without the block (older checkpoints) is disabled."""
        block = getattr(cfg.data, "scm_residual", None)
        if block is None:
            return cls()

        def get(key: str, default: Any) -> Any:
            value = block.get(key, default) if hasattr(block, "get") else getattr(block, key, default)
            return default if value is None else value

        return cls(
            enabled=bool(get("enabled", cls.enabled)),
            prob=float(get("prob", cls.prob)),
            prior_type=str(get("prior_type", cls.prior_type)),
            lambda_p0=float(get("lambda_p0", cls.lambda_p0)),
            lambda_p1=float(get("lambda_p1", cls.lambda_p1)),
            allow_analytic_pit=bool(get("allow_analytic_pit", cls.allow_analytic_pit)),
            amplitude_mode=str(get("amplitude_mode", cls.amplitude_mode)),
            prior_shapes=bool(get("prior_shapes", cls.prior_shapes)),
            prior_P_min=None if get("prior_P_min", None) is None else int(get("prior_P_min", None)),
            prior_P_max=None if get("prior_P_max", None) is None else int(get("prior_P_max", None)),
            prior_P_dist=str(get("prior_P_dist", cls.prior_P_dist)),
            mean_amp_lognormal_loc=float(get("mean_amp_lognormal_loc", cls.mean_amp_lognormal_loc)),
            mean_amp_lognormal_scale=float(get("mean_amp_lognormal_scale", cls.mean_amp_lognormal_scale)),
        )


def is_scm_episode(task: Mapping[str, Any]) -> bool:
    """Whether a saved episode came from the SCM branch (its mean is a saved vector, not the mean bank)."""
    family = task.get("mean_family")
    return family is not None and int(family) == MEAN_FAMILY_SCM


def saved_mean_at(task: Mapping[str, Any], x: Tensor) -> Tensor:
    """An episode's saved mean m at the kernel-space rows x (rows of x_kernel_train / x_kernel_test).

    The SCM mean depends on the SCM table, not on x through a known function, so it is looked
    up: x equal to the saved train or test block returns that block's vector; any other x
    (a fold subset) is matched row by row against both blocks. Raises KeyError when a row is
    not one of the episode's, the same signal callers already treat as "oracle unavailable".
    """
    if "mean_train" not in task or "mean_test" not in task:
        raise KeyError("SCM episode without saved mean_train / mean_test")
    x_train = task.get("x_kernel_train", task["x_norm_train"]).to(device=x.device, dtype=x.dtype)
    x_test = task.get("x_kernel_test", task["x_norm_test"]).to(device=x.device, dtype=x.dtype)
    m_train = task["mean_train"].to(device=x.device, dtype=x.dtype)
    m_test = task["mean_test"].to(device=x.device, dtype=x.dtype)
    if x.shape == x_train.shape and torch.equal(x, x_train):
        return m_train
    if x.shape == x_test.shape and torch.equal(x, x_test):
        return m_test
    x_ref = torch.cat([x_train, x_test])
    dist = torch.cdist(x, x_ref)  # (M, P + N)
    best, idx = dist.min(dim=1)
    tol = 1e-5 * (1.0 + float(x_ref.abs().max()))
    if bool((best > tol).any()):
        raise KeyError("rows not in the episode: the SCM mean is only known at the episode's own rows")
    return torch.cat([m_train, m_test])[idx]


def validate_scm_settings(cfg: HasDataConfig, settings: ScmResidualSettings) -> None:
    """Refuse configurations the SCM prior cannot train with (only called when it is enabled).

    The analytic PIT standardizes by m = a_g g~, i.e. it hands the copula the
    SCM mechanism no model sees; it is allowed only for diagnostics (allow_analytic_pit).
    """
    if settings.prior_type not in SCM_PRIOR_TYPES:
        raise ValueError(f"data.scm_residual.prior_type={settings.prior_type!r}; expected one of {SCM_PRIOR_TYPES}.")
    if not 0.0 <= settings.prob <= 1.0:
        raise ValueError(f"data.scm_residual.prob={settings.prob} must lie in [0, 1].")
    p0, p1 = settings.lambda_p0, settings.lambda_p1
    if p0 < 0.0 or p1 < 0.0 or p0 + p1 > 1.0:
        raise ValueError(f"data.scm_residual.lambda_p0={p0}, lambda_p1={p1}: need p0, p1 >= 0 and p0 + p1 <= 1.")
    if settings.amplitude_mode not in AMPLITUDE_MODES:
        raise ValueError(
            f"data.scm_residual.amplitude_mode={settings.amplitude_mode!r}; expected one of {AMPLITUDE_MODES} "
            "(the coupled mode was removed)."
        )
    if settings.mean_amp_lognormal_scale < 0.0:
        raise ValueError("data.scm_residual.mean_amp_lognormal_scale must be non-negative.")
    source = str(getattr(cfg.data, "z_train_source", "analytic"))
    if source in _ANALYTIC_SOURCES and not settings.allow_analytic_pit:
        raise ValueError(
            "data.scm_residual.enabled=true with data.z_train_source=analytic: the analytic PIT "
            "of an SCM episode uses the SCM mechanism g (z = (y - m) / sigma with m = a_g g~), "
            "which leaks the target's mean into z_train. Train with a learned marginal "
            "(data.z_train_source=tabicl, tabicl_split, tabpfn35_fast or kumo_small), or set "
            "data.scm_residual.allow_analytic_pit=true for diagnostics only."
        )


@dataclass
class Amplitudes:
    """Per-episode amplitudes of y = a_g g~ + r + eps, r ~ GP(0, kernel_factor * K)."""

    mean_amp2: Tensor  # (B,) a_g^2, the SCM mean's variance (g~ is standardized)
    kernel_factor: Tensor  # (B,) 1, or 0 for an i.i.d. TabICLv2 episode


def sample_amplitudes(settings: ScmResidualSettings, B: int, device: Device) -> Amplitudes:
    """Draw the amplitudes of the SCM mean and the GP residual independently.

    a_g^2 ~ LogNormal(mean_amp_lognormal_loc, mean_amp_lognormal_scale^2) and r ~ GP(0, K) with the
    kernel's own output scale, so the share lambda = mean_i K_ii / (a_g^2 + mean_i K_ii) and the
    relative noise sigma_n^2 / (a_g^2 + mean_i K_ii) are induced. With probability lambda_p0 the
    residual is dropped (TabICLv2's prior, a_g = 1) and with lambda_p1 the mean (the zero-mean GP
    prior, kernel amplitude and noise included). Scale 0 pins a_g^2 = exp(loc).
    """
    u = torch.rand(B, device=device)
    log_amp2 = settings.mean_amp_lognormal_loc + settings.mean_amp_lognormal_scale * torch.randn(B, device=device)
    mean_amp2 = torch.exp(log_amp2)
    iid = u < settings.lambda_p0
    gp_only = (u >= settings.lambda_p0) & (u < settings.lambda_p0 + settings.lambda_p1)
    mean_amp2 = torch.where(
        iid, torch.ones_like(mean_amp2), torch.where(gp_only, torch.zeros_like(mean_amp2), mean_amp2)
    )
    kernel_factor = torch.where(iid, torch.zeros_like(mean_amp2), torch.ones_like(mean_amp2))
    return Amplitudes(mean_amp2=mean_amp2, kernel_factor=kernel_factor)


def sample_v2_num_features() -> int:
    """A call's feature count d ~ round(Uniform(1, 100)), as GraphPrior.get_batch draws it."""
    return int(round(np.random.uniform(V2_MIN_FEATURES, V2_MAX_FEATURES)))


P_DISTS = ("loguniform", "uniform")


def sample_v2_train_test_sizes(
    P_min: int | None = None, P_max: int | None = None, dist: str = "loguniform"
) -> tuple[int, int]:
    """A call's (P, N) with N = 1024 - P.

    Without bounds, TabICLv2's split: P = int(1024 u), u ~ Uniform(0.3, 0.9) (Prior.sample_train_size).
    With bounds, P = round(exp(Uniform(log P_min, log P_max))) ("loguniform") or
    round(Uniform(P_min, P_max)) ("uniform"), clipped to [P_min, P_max].
    """
    if P_min is None and P_max is None:
        P = int(V2_SEQ_LEN * np.random.uniform(V2_TRAIN_FRAC_MIN, V2_TRAIN_FRAC_MAX))
        return P, V2_SEQ_LEN - P
    if P_min is None or P_max is None or not 2 <= P_min <= P_max <= V2_SEQ_LEN - 2:
        raise ValueError(
            f"data.scm_residual.prior_P_min/prior_P_max = {P_min}/{P_max}; set both, with "
            f"2 <= prior_P_min <= prior_P_max <= {V2_SEQ_LEN - 2}."
        )
    if dist == "loguniform":
        u = float(np.exp(np.random.uniform(np.log(P_min), np.log(P_max))))
    elif dist == "uniform":
        u = float(np.random.uniform(P_min, P_max))
    else:
        raise ValueError(f"data.scm_residual.prior_P_dist={dist!r}; expected one of {P_DISTS}.")
    P = min(max(int(round(u)), P_min), P_max)
    return P, V2_SEQ_LEN - P


@lru_cache(maxsize=None)
def _prior_module(name: str) -> types.ModuleType:
    """tabicl.prior.<name> (the package imports xgboost through TabICLv1's tree SCM: xgboost-cpu)."""
    return importlib.import_module(f"tabicl.prior.{name}")


@dataclass
class ScmDraw:
    """B graph-SCM tables of T rows and d features (float32 on the requested device)."""

    X: Tensor  # (B, T, d)
    g: Tensor  # (B, T) SCM target, standardized over the T rows
    valid: Tensor  # (B,) bool: every feature non-constant, finite, g non-constant, predictable
    prior_type: Tensor  # (B,) long, SCM_PRIOR_CODES


@lru_cache(maxsize=1)
def _v2_prior_config() -> Any:
    """graph_lib PriorConfig with TabICLv2's regressor settings (no random draws)."""
    return _prior_module("graph_lib._config").PriorConfig(**V2_REGRESSOR_PRIOR_CONFIG)


def _draw_table(T: int, d: int) -> tuple[Tensor, Tensor, bool]:
    """One (X (T, d), y (T,)) graph-SCM table and whether TabICLv2's generator would keep it.

    Mirrors GraphPrior.generate_dataset for one regression dataset: GraphSCM (outlier removal
    and standard scaling of X and y inside), then should_filter (an ExtraTrees whose OOB error
    does not beat the mean is unpredictable). TabICLv2 drops constant columns instead; here a
    dropped column would change d, which every episode of a call shares, so the table is redrawn.
    """
    config = _v2_prior_config()
    scm = _prior_module("_graph_scm").GraphSCM(
        regression=True,
        seq_len=T,
        num_features=d,
        max_features=d,
        num_classes=0,
        permute_features=True,
        config=config,
        device="cpu",
    )
    X, y = scm()
    X, y = X.float(), y.float()
    finite = bool(torch.isfinite(X).all()) and bool(torch.isfinite(y).all())
    all_features = all(int(torch.unique(X[:, j]).numel()) > 1 for j in range(d))
    if not (finite and all_features and float(y.std()) > 1e-6):
        return X, y, False
    predictable = not _prior_module("_dataset").should_filter(X, y, config, is_classif=False)
    return X, y, predictable


@torch.no_grad()
def sample_scm_tables(B: int, T: int, d: int, prior_type: str, *, device: Device) -> ScmDraw:
    """Draw B regression tables from TabICLv2's graph SCM prior with exactly d features and T rows.

    Every episode draws its own DAG, node functions and feature/target placement (TabICLv2
    groups datasets only by shape). Generation is sequential on the CPU (it runs inside
    DataLoader workers) from the global numpy / torch RNGs, so seed_everything fixes it. A
    rejected table is redrawn up to _MAX_TABLE_ATTEMPTS times, then returned with valid=False.
    """
    if prior_type not in SCM_PRIOR_TYPES:
        raise ValueError(f"prior_type={prior_type!r}; expected one of {SCM_PRIOR_TYPES} (TabICLv2's prior)")
    X = torch.zeros(B, T, d, dtype=torch.float32)
    g = torch.zeros(B, T, dtype=torch.float32)
    valid = torch.zeros(B, dtype=torch.bool)
    # Fixed evaluation sets synthesize under a float64 default (generate_fixed_gp_batch); the
    # graph prior is written for float32, so pin it.
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        for b in range(B):
            for _ in range(_MAX_TABLE_ATTEMPTS):
                Xb, yb, ok = _draw_table(T, d)
                if ok:
                    X[b], g[b], valid[b] = Xb, yb, True
                    break
    finally:
        torch.set_default_dtype(previous_dtype)
    # Invalid episodes get finite placeholders so the batched GP algebra stays finite; they are discarded.
    if not bool(valid.all()):
        bad = ~valid
        X[bad] = torch.randn(int(bad.sum()), T, d, dtype=torch.float32)
        g[bad] = torch.randn(int(bad.sum()), T, dtype=torch.float32)
    g = (g - g.mean(1, keepdim=True)) / g.std(1, keepdim=True).clamp(min=1e-8)
    codes = torch.full((B,), SCM_PRIOR_CODES[prior_type], dtype=torch.long)
    return ScmDraw(X=X.to(device), g=g.to(device), valid=valid.to(device), prior_type=codes.to(device))
