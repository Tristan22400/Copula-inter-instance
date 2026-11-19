"""TabICL SCM prior adapter and the settings of the SCM + GP residual prior.

The SCM + GP residual prior (cfg.data.scm_residual) draws one episode as

    y = sqrt(1 - lambda) * g~(x) + L_all eps,   K_all = lambda * K~ + sigma_n^2 I,

where (X, g) is a TabICL SCM table, g~ its target standardized over the episode, K~ the
episode's GP kernel normalized to unit mean prior variance and lambda ~ p0 delta_0 +
p1 delta_1 + (1 - p0 - p1) Beta(a, b) the residual share. lambda = 0 is the TabICL prior
(rows i.i.d. given the SCM), lambda = 1 the zero-mean GP prior. gp_episodes builds the
target; this module owns the settings, the lambda draw and every use of TabICL's
private prior modules, so a TabICL upgrade touches one file.

TabICL's prior package imports xgboost at import time (its tree SCM), and the package
__init__ pulls that in for every submodule. The MLP SCM does not need xgboost, so when
xgboost is absent the submodules are loaded without running the package __init__.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from copula_inter.type_aliases import Device, HasDataConfig

SCM_PRIOR_TYPES = ("mlp_scm", "tree_scm", "mix_scm")
AMPLITUDE_MODES = ("coupled", "decoupled")
# scm_prior_type codes saved per episode (-1: a GP-prior episode of an SCM-enabled run).
SCM_PRIOR_CODES = {"mlp_scm": 0, "tree_scm": 1}
GP_EPISODE_PRIOR_CODE = -1
# mean_family of an SCM episode (the mean bank's codes are 0 linear, 1 exponential, 2 anomaly):
# its mean is the saved mean_train / mean_test vectors, not a mean-bank function.
MEAN_FAMILY_SCM = 3
# Learned marginals the SCM prior can train with; "analytic" would leak g through m.
_ANALYTIC_SOURCES = ("analytic",)
# Redraws of one episode whose SCM table loses a feature (constant column) or is degenerate
# before it is marked invalid and left to generate_gp_batch's top-up.
_MAX_TABLE_ATTEMPTS = 4


@dataclass(frozen=True)
class ScmResidualSettings:
    """cfg.data.scm_residual, with the defaults of conf/data/gp_tasks.yaml."""

    enabled: bool = False
    prob: float = 1.0
    prior_type: str = "mix_scm"
    lambda_p0: float = 0.2
    lambda_p1: float = 0.2
    lambda_beta_a: float = 2.0
    lambda_beta_b: float = 2.0
    allow_analytic_pit: bool = False
    apply_feature_warps: bool = False
    # Absent from configs written before the decoupled mode existed: they keep the coupled one.
    amplitude_mode: str = "coupled"
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
            lambda_beta_a=float(get("lambda_beta_a", cls.lambda_beta_a)),
            lambda_beta_b=float(get("lambda_beta_b", cls.lambda_beta_b)),
            allow_analytic_pit=bool(get("allow_analytic_pit", cls.allow_analytic_pit)),
            apply_feature_warps=bool(get("apply_feature_warps", cls.apply_feature_warps)),
            amplitude_mode=str(get("amplitude_mode", cls.amplitude_mode)),
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

    The analytic PIT standardizes by m = sqrt(1 - lambda) g~, i.e. it hands the copula the
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
            f"data.scm_residual.amplitude_mode={settings.amplitude_mode!r}; expected one of {AMPLITUDE_MODES}."
        )
    if settings.mean_amp_lognormal_scale < 0.0:
        raise ValueError("data.scm_residual.mean_amp_lognormal_scale must be non-negative.")
    if settings.lambda_beta_a <= 0.0 or settings.lambda_beta_b <= 0.0:
        raise ValueError("data.scm_residual.lambda_beta_a and lambda_beta_b must be positive.")
    source = str(getattr(cfg.data, "z_train_source", "analytic"))
    if source in _ANALYTIC_SOURCES and not settings.allow_analytic_pit:
        raise ValueError(
            "data.scm_residual.enabled=true with data.z_train_source=analytic: the analytic PIT "
            "of an SCM episode uses the SCM mechanism g (z = (y - m) / sigma with m = sqrt(1 - lambda) g~), "
            "which leaks the target's mean into z_train. Train with a learned marginal "
            "(data.z_train_source=tabicl, tabicl_split, tabpfn35_fast or kumo_small), or set "
            "data.scm_residual.allow_analytic_pit=true for diagnostics only."
        )
    if settings.prior_type in ("tree_scm", "mix_scm") and not xgboost_available():
        raise ImportError(
            f"data.scm_residual.prior_type={settings.prior_type} needs xgboost (TabICL's tree SCM), "
            "which is not installed. Use data.scm_residual.prior_type=mlp_scm."
        )


def sample_lambda(settings: ScmResidualSettings, B: int, device: Device) -> Tensor:
    """(B,) residual shares lambda ~ p0 delta_0 + p1 delta_1 + (1 - p0 - p1) Beta(a, b)."""
    u = torch.rand(B, device=device)
    beta = torch.distributions.Beta(
        torch.tensor(settings.lambda_beta_a, device=device), torch.tensor(settings.lambda_beta_b, device=device)
    ).sample((B,))
    lam = torch.where(u < settings.lambda_p0, torch.zeros_like(beta), beta)
    p01 = settings.lambda_p0 + settings.lambda_p1
    lam = torch.where((u >= settings.lambda_p0) & (u < p01), torch.ones_like(beta), lam)
    return lam.clamp(0.0, 1.0)


@dataclass
class Amplitudes:
    """Per-episode amplitudes of y = a_g g~ + r + eps, r ~ GP(0, kernel_factor * K[~])."""

    mean_amp2: Tensor  # (B,) a_g^2, the SCM mean's variance (g~ is standardized)
    kernel_factor: Tensor  # (B,) multiplies K (decoupled) or K~ = K / mean_i K_ii (coupled)
    normalize_kernel: bool


def sample_amplitudes(settings: ScmResidualSettings, B: int, device: Device) -> Amplitudes:
    """Draw the amplitudes of the SCM mean and the GP residual.

    coupled:   a_g^2 = 1 - lambda and r ~ GP(0, lambda K~), lambda from sample_lambda, so the
               signal variance is 1 and the relative noise sigma_n^2 does not depend on lambda.
    decoupled: a_g^2 ~ LogNormal(0, mean_amp_lognormal_scale^2) and r ~ GP(0, K) with the kernel's
               own output scale, independently; the share lambda = mean_i K_ii / (a_g^2 + mean_i K_ii)
               and the relative noise sigma_n^2 / (a_g^2 + mean_i K_ii) are then induced. With
               probability lambda_p0 the residual is dropped (TabICL's prior) and with lambda_p1 the
               mean (the zero-mean GP prior, kernel amplitude and noise included); the Beta keys are unused.
    """
    if settings.amplitude_mode == "coupled":
        lam = sample_lambda(settings, B, device)
        return Amplitudes(mean_amp2=1.0 - lam, kernel_factor=lam, normalize_kernel=True)
    u = torch.rand(B, device=device)
    mean_amp2 = torch.exp(settings.mean_amp_lognormal_scale * torch.randn(B, device=device))
    iid = u < settings.lambda_p0
    gp_only = (u >= settings.lambda_p0) & (u < settings.lambda_p0 + settings.lambda_p1)
    mean_amp2 = torch.where(
        iid, torch.ones_like(mean_amp2), torch.where(gp_only, torch.zeros_like(mean_amp2), mean_amp2)
    )
    kernel_factor = torch.where(iid, torch.zeros_like(mean_amp2), torch.ones_like(mean_amp2))
    return Amplitudes(mean_amp2=mean_amp2, kernel_factor=kernel_factor, normalize_kernel=False)


def xgboost_available() -> bool:
    """Whether TabICL's tree SCM can be imported."""
    return importlib.util.find_spec("xgboost") is not None


@lru_cache(maxsize=None)
def _prior_module(name: str) -> types.ModuleType:
    """tabicl.prior.<name>, loaded without the package __init__ when xgboost is missing."""
    full = f"tabicl.prior.{name}"
    if full in sys.modules:
        return sys.modules[full]
    if xgboost_available():
        return importlib.import_module(full)
    if name == "_tree_scm":
        raise ImportError("TabICL's tree SCM needs xgboost, which is not installed. Use prior_type=mlp_scm.")
    spec = importlib.util.find_spec("tabicl")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("cannot locate the vendored tabicl package")
    # A stand-in parent package so the relative imports inside the prior modules resolve; it is
    # removed again so a later `import tabicl.prior` runs the real __init__.
    stub = types.ModuleType("tabicl.prior")
    stub.__path__ = [str(Path(next(iter(spec.submodule_search_locations))) / "prior")]
    stub.__package__ = "tabicl.prior"
    previous = sys.modules.get("tabicl.prior")
    sys.modules["tabicl.prior"] = stub
    try:
        return importlib.import_module(full)
    finally:
        if previous is None:
            sys.modules.pop("tabicl.prior", None)
        else:
            sys.modules["tabicl.prior"] = previous


@dataclass
class ScmDraw:
    """B SCM tables of T rows and d features (float32 on the requested device)."""

    X: Tensor  # (B, T, d)
    g: Tensor  # (B, T) SCM target, standardized over the T rows
    valid: Tensor  # (B,) bool: every feature non-constant, finite, g non-constant
    prior_type: Tensor  # (B,) long, SCM_PRIOR_CODES


@lru_cache(maxsize=1)
def _hp_sampler() -> Any:
    """TabICL's hyperparameter sampler, built once per process.

    Building it draws no random numbers (its samplers are closures that draw when called), so
    reusing it keeps every seed's episodes identical; rebuilding it per episode was half the
    SCM generation time.
    """
    config = _prior_module("_prior_config")
    return _prior_module("_hp_sampling").HpSamplerList(config.DEFAULT_SAMPLED_HP, device="cpu")


def _episode_params(prior_type: str, T: int, d: int) -> dict[str, Any]:
    """One dataset's parameters, as SCMPrior.get_batch builds them for one regression dataset."""
    config = _prior_module("_prior_config")
    sampler = _hp_sampler()
    sampled = {k: v() if callable(v) else v for k, v in sampler.sample().items()}
    return {
        "regression": True,
        "device": "cpu",
        **config.DEFAULT_FIXED_HP,
        "seq_len": T,
        "train_size": T,  # unused for regression (no class-balance check)
        "max_features": d,
        **sampled,
        "prior_type": prior_type,
        "num_features": d,
        "num_classes": None,
    }


def _draw_table(prior_type: str, T: int, d: int) -> tuple[Tensor, Tensor, bool]:
    """One (X (T, d), y (T,)) table and whether it kept all d features and a usable target."""
    params = _episode_params(prior_type, T, d)
    scm_cls = _prior_module("_mlp_scm").MLPSCM if prior_type == "mlp_scm" else _prior_module("_tree_scm").TreeSCM
    X, y = scm_cls(**params)()
    X, y = _prior_module("_reg2cls").Reg2Cls(params)(X, y)
    X, y = X.float(), y.float()
    finite = bool(torch.isfinite(X).all()) and bool(torch.isfinite(y).all())
    # TabICL's delete_unique_features drops constant columns; here a dropped column would
    # change d, which every episode of a call shares, so the table is redrawn instead.
    all_features = all(int(torch.unique(X[:, j]).numel()) > 1 for j in range(d))
    target_ok = finite and float(y.std()) > 1e-6
    return X, y, finite and all_features and target_ok


def _resolve_prior_types(prior_type: str, B: int) -> list[str]:
    """Each episode's SCM family; mix_scm draws it with TabICL's mix_probs."""
    if prior_type != "mix_scm":
        return [prior_type] * B
    probs = _prior_module("_prior_config").DEFAULT_FIXED_HP.get("mix_probs", (0.7, 0.3))
    return [str(t) for t in np.random.choice(["mlp_scm", "tree_scm"], size=B, p=list(probs))]


@torch.no_grad()
def sample_scm_tables(B: int, T: int, d: int, prior_type: str, *, device: Device) -> ScmDraw:
    """Draw B regression tables from TabICL's SCM prior with exactly d features and T rows.

    Every episode draws its own SCM hyperparameters (TabICL groups datasets by shared
    hyperparameters; here episodes are independent, as the GP prior's are). Generation is
    sequential on the CPU (n_jobs=1, it runs inside DataLoader workers) from the global
    random / numpy / torch RNGs, so seed_everything fixes it. A table that loses a feature
    or has a degenerate target is redrawn up to _MAX_TABLE_ATTEMPTS times, then returned
    with valid=False.
    """
    if prior_type not in SCM_PRIOR_TYPES:
        raise ValueError(f"prior_type={prior_type!r}; expected one of {SCM_PRIOR_TYPES}")
    types_per_ep = _resolve_prior_types(prior_type, B)
    X = torch.zeros(B, T, d, dtype=torch.float32)
    g = torch.zeros(B, T, dtype=torch.float32)
    valid = torch.zeros(B, dtype=torch.bool)
    # TabICL's SCMs build their layers in the default dtype but sample inputs in float32; fixed
    # evaluation sets synthesize under a float64 default (generate_fixed_gp_batch), so pin float32.
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        for b, kind in enumerate(types_per_ep):
            for _ in range(_MAX_TABLE_ATTEMPTS):
                Xb, yb, ok = _draw_table(kind, T, d)
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
    codes = torch.tensor([SCM_PRIOR_CODES[t] for t in types_per_ep], dtype=torch.long)
    return ScmDraw(X=X.to(device), g=g.to(device), valid=valid.to(device), prior_type=codes.to(device))
