"""
data_gen.py — Stage A: GP task generation for inter-instance copula.

Each task samples a random GP with a configurable PSD kernel, draws P+N
instances, normalises features over the full P+N set, samples targets jointly
from the GP, computes the analytical correlation matrix R* at the test points
(from either the GP posterior conditioned on training data or the raw GP
prior, per cfg.data.oracle_mode), and saves all required tensors.

Kernels are built from gpytorch.kernels (RBFKernel, MaternKernel,
PeriodicKernel, RQKernel, CosineKernel, LinearKernel) wrapped in ScaleKernel
(LinearKernel is the one exception — see "dot_product" below); hyperparameters
(lengthscale, outputscale, nugget, period, rq_alpha) are sampled from
gpytorch.priors (LogNormalPrior / GammaPrior) rather than the uniform ranges
an earlier version of this file used — see _kernel_prior_spec / _nugget_prior
for the exact distributions (all cfg-overridable, same getattr-with-default
convention as before). Kernel composition (sums/products of two base kernels)
uses gpytorch's native `+`/`*` operator overloading on Kernel objects.

Supported kernels
-----------------
  rbf                 — Squared Exponential / RBF
  matern32            — Matérn ν=3/2
  cosine              — Cosine (spectral): k(r) = alpha2 * cos(2π r / l)
  periodic            — Periodic: k(r) = alpha2 * exp(-2 sin²(π r / period) / l²)
  rational_quadratic  — Rational Quadratic: k(r) = alpha2 * (1 + r²/(2α l²))^{-α}
  dot_product         — Linear (dot product): k(x1,x2) = alpha2 * x1ᵀx2, via
                         gpytorch.kernels.LinearKernel. Its `variance` plays
                         exactly the role `alpha2` (outputscale) plays for
                         every other kernel here — sampled from the same
                         alpha2 ~ Gamma(alpha2_gamma_concentration,
                         alpha2_gamma_rate) prior directly (see
                         _sample_episode_kernel), not wrapped in a separate
                         outer ScaleKernel (that would just be a second,
                         redundant alpha2). No lengthscale — geometry is
                         determined entirely by the feature space.
  hebo                — ARD Matérn ν=3/2 + outputscale, the tuned "HEBO+"
                         prior from the PFN4BO paper (github.com/automl/
                         PFNs4BO), Appendix B.1: lengthscale/outputscale ~
                         Gamma, noise ~ LogNormal — see
                         HEBO_PLUS_HYPERPARAMETERS for the exact tuned
                         constants and their source. Sampled with the same
                         gpytorch.priors machinery as every other kernel here
                         — no runtime dependency on pfns4bo_upstream any more
                         (its Gamma/LogNormal hyperpriors don't depend on the
                         input points, unlike upstream's SingleTaskGP-based
                         `hebo_prior.get_model` scaffolding).

ARD (cfg.data.ard)
-------------------
  When cfg.data.ard is True, rbf/matern32/periodic/rational_quadratic sample
  one independent lengthscale per active kernel dimension (ard_num_dims=k)
  instead of one isotropic scalar shared across all k dims — same mechanism
  "hebo" already always uses. periodic's period also becomes per-dimension
  (gpytorch.kernels.PeriodicKernel ties period_length's ard_num_dims to the
  same kwarg as lengthscale). Default False (isotropic), preserving prior
  dataset-generation behaviour. Not possible for "cosine": gpytorch's
  CosineKernel hardcodes period_length to a single scalar regardless of
  ard_num_dims — no per-dimension formula exists. Not applicable to
  "dot_product" (no lengthscale) or "hebo" (already unconditionally ARD via
  its own tuned prior, independent of this flag). See _ARD_ELIGIBLE_KERNELS.
  "periodic" is additionally always capped to k=1 active dims (independent
  of this flag) — see generate_gp_batch's kernel_cols selection.

  cfg.data.isotropic_ratio (default 0.0): even when a kernel would otherwise
  be ARD (cfg.data.ard=True for an ARD-eligible kernel, or "hebo" which is
  always ARD), each episode independently has probability isotropic_ratio of
  having its lengthscale (and periodic's period) collapsed to one shared
  value across all active dims instead of one independent value per dim —
  i.e. an isotropic kernel in effect, still stored in the ARD-shaped (k,)
  tensor (so "l"/"period" numel doesn't change, only whether the k values
  are equal). A no-op when the kernel isn't ARD in the first place. See
  _build_scaled_kernel.

Composite kernels ("A+B" / "A*B")
---------------------------------
  Sums and products of PSD kernels are PSD, so every pair drawn from
  {rbf, matern32, cosine, periodic, rational_quadratic} is auto-registered
  under both operators via gpytorch's `+`/`*` kernel composition, e.g.
  "rbf+periodic" (locally periodic: smooth decay times exact periodicity) or
  "matern32*cosine" (spectral windowing). See COMPOSITE_KERNELS for the full
  list. dot_product and hebo are not composable (irregular hyperparameter
  signatures — no lengthscale, or ARD-only). cfg.data.ard applies
  independently to each ARD-eligible component of a composite.

  Systematic composition (cfg.data.systematic_composition, CauKer-style —
  github.com/ShifengXIE/CauKer): an alternative, opt-in generative mode that
  samples a random chain length M ~ Uniform[composite_num_kernels_min,
  composite_num_kernels_max], draws M elementary kernels with replacement
  from {rbf, matern32, cosine, periodic, rational_quadratic}, and combines
  them left-to-right with independently-sampled +/* operators (see
  _sample_kernel_chain_structure / _build_kernel_chain), instead of the
  static enumerated 2-way COMPOSITE_KERNELS list. Produces chain names like
  "rbf+cosine*periodic" that are NOT registered in ALL_KERNELS/
  KERNEL_REGISTRY (unbounded cardinality) and are not reconstructible via
  build_kernel_fn — see generate_gp_batch's return_kernel_metadata handling
  for the separate kernel_components/kernel_ops/kernel_component_params
  schema this mode uses instead of the flat l/alpha2/l_b/alpha2_b keys.

Kernel selection (cfg.data.kernel / cfg.data.kernels)
------------------------------------------------------
  cfg.data.kernel   : str          → use this single kernel for every task
                                     (any entry in ALL_KERNELS, including composites)
  cfg.data.kernels  : list[str]    → sample uniformly at task generation time
  cfg.data.systematic_composition : bool → if True, ignore cfg.data.kernel/
                                     kernels entirely and sample a fresh
                                     random-length kernel chain per
                                     batch/task call instead (see
                                     "Systematic composition" above).
                                     Default False.
  If both kernel/kernels are absent (and systematic_composition is False)
  the default is "rbf".

Total feature count (cfg.data.d_features / d_features_lognormal_loc/scale)
----------------------------------------------------------------------------
  d (total feature columns, of which _sample_active_dims picks a subset as
  the kernel's active_dims) is normally the fixed cfg.data.d_features. If
  cfg.data.d_features_lognormal_loc/scale are both set instead, d ~
  round(LogNormal(...)) clipped to a minimum of 2, sampled once per
  generate_gp_batch call (i.e. once per shard in generate_pit_dataset.py —
  see _sample_d_features). Every episode within one shard shares the same
  d; different shards can differ. Since dataset.py's collate_fn stacks a
  training minibatch's x_train/x_test into one (B, *, d) tensor using the
  first sample's d, a minibatch that spans shards with different d will
  crash — when this mode is enabled, set training.shard_block_shards=1 and
  choose training.batch_size to evenly divide data.shard_size so every
  minibatch stays within a single shard (see conf/data/gp_tasks.yaml).
"""

from __future__ import annotations

import functools
import itertools
import math
import random
import re
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import gpytorch
import numpy as np
import torch
from gpytorch.priors import GammaPrior, LogNormalPrior, Prior
from torch import Tensor

from loss import _safe_cholesky

# gpytorch's own solver (via linear_operator) only guarantees an EXACT
# Cholesky solve for matrices up to gpytorch.settings.max_cholesky_size
# (default 800); above that it silently switches to an approximate
# Lanczos/CG solve. conf/data/gp_tasks.yaml allows P up to 1024 and N up to
# 128 (T = P+N up to 1152), so every gpytorch call in this file that touches
# a full (P+N, P+N) or (P, P) covariance is wrapped in
# `with gpytorch.settings.max_cholesky_size(_MAX_CHOLESKY):` — generous
# headroom over any realistic T, cheap to raise further if P_max/N_max grow.
_MAX_CHOLESKY = 8192


def _seed_everything(seed: int) -> None:
    """Seed python/numpy/torch RNGs for reproducible data generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe even with a single GPU / no GPU


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------


def _sq_dist(X1: Tensor, X2: Tensor) -> Tensor:
    """Squared Euclidean distance matrix (n1, n2)."""
    diff = X1.unsqueeze(1) - X2.unsqueeze(0)  # (n1, n2, d)
    return (diff**2).sum(-1)


def _dist(X1: Tensor, X2: Tensor) -> Tensor:
    """Euclidean distance matrix (n1, n2)."""
    return _sq_dist(X1, X2).clamp(min=0.0).sqrt()


# ---------------------------------------------------------------------------
# gpytorch-backed kernel construction (sampling + reconstruction)
# ---------------------------------------------------------------------------
# Two entry points share the machinery below:
#   - _sample_episode_kernel: draws fresh hyperparameters from gpytorch
#     LogNormal/Gamma priors for B episodes at once. Its kernel object feeds
#     BOTH of the following:
#       - generate_gp_batch / generate_gp_task's own sampling and posterior
#         conditioning, done via gpytorch's native GaussianLikelihood +
#         ExactGP (_build_likelihood, _GeneratorGP below) rather than
#         hand-rolled Gram-matrix + noise math.
#       - build_kernel_fn (below), for reconstructing a kernel from already-
#         known concrete hyperparameter values.
#   - build_kernel_fn: builds a kernel(X1, X2) -> K callable from CONCRETE,
#     already-known hyperparameter values (used by pit.py::gp_analytical_pit
#     and tests to reconstruct the kernel a saved episode was drawn from).
#     Materialises the Gram matrix via `.to_dense()` and hands off to this
#     file's own torch.linalg-based Cholesky/solve code (_safe_cholesky) —
#     there is no live train/test split at that point to condition an
#     ExactGP on, just a Gram matrix to factor.
#
# gpytorch's own ExactGP/lazy-tensor solve machinery silently switches to an
# approximate CG solve for matrices larger than
# gpytorch.settings.max_cholesky_size (default 800, see _MAX_CHOLESKY),
# which would silently diverge from the exact-Cholesky-based invariants this
# repo's test suite checks (well-conditioned floor, unit-diagonal tolerance)
# — this repo's episode sizes (P up to 1024, N up to 128, see
# conf/data/gp_tasks.yaml) regularly exceed that default. Every gpytorch
# call in this file that sees a full (P+N, P+N) or (P, P) covariance is
# therefore wrapped in `with gpytorch.settings.max_cholesky_size(_MAX_CHOLESKY):`
# to force exact solves — verified empirically to agree with the previous
# hand-rolled Cholesky implementation to ~1e-6 max abs difference.

_BASE_GPYTORCH_KERNEL_CLS: Dict[str, Callable[..., gpytorch.kernels.Kernel]] = {
    "rbf": gpytorch.kernels.RBFKernel,
    "matern12": functools.partial(gpytorch.kernels.MaternKernel, nu=0.5),
    "matern32": functools.partial(gpytorch.kernels.MaternKernel, nu=1.5),
    "matern52": functools.partial(gpytorch.kernels.MaternKernel, nu=2.5),
    "cosine": gpytorch.kernels.CosineKernel,
    "periodic": gpytorch.kernels.PeriodicKernel,
    "rational_quadratic": gpytorch.kernels.RQKernel,
    "hebo": functools.partial(gpytorch.kernels.MaternKernel, nu=1.5),
}

# Maps an output-dict/schema parameter name to the gpytorch attribute that
# holds it, for the "extra" (non-lengthscale, non-outputscale) parameters.
_EXTRA_PARAM_TO_ATTR: Dict[str, str] = {"period": "period_length", "rq_alpha": "alpha"}


@dataclass
class KernelPriorSpec:
    """Hyperprior distributions for one base kernel family.

    lengthscale_prior(k) returns the Prior over the kernel's shape parameter
    (its `.lengthscale`, or `.period_length` for cosine — see
    lengthscale_attr) given k active input dimensions; ard=True samples one
    independent value per dimension instead of one isotropic scalar shared
    across all k dims. Always True for "hebo" (the tuned HEBO+ prior is
    ARD by definition); for rbf/matern32/periodic/rational_quadratic it
    follows cfg.data.ard (see _kernel_prior_spec / _ARD_ELIGIBLE_KERNELS).
    "cosine" is never ARD — gpytorch.kernels.CosineKernel's period_length is
    a single scalar regardless of ard_num_dims (no per-dimension formula
    exists to opt into).
    """

    lengthscale_prior: Callable[[int], Prior]
    outputscale_prior: Prior
    lengthscale_attr: str = "lengthscale"
    extra_priors: Dict[str, Prior] = field(default_factory=dict)
    ard: bool = False
    # Per-episode probability of collapsing an otherwise-ARD lengthscale/
    # period to a single shared value across dims (cfg.data.isotropic_ratio,
    # see the module docstring's ARD section and _build_scaled_kernel).
    # No-op when ard=False.
    isotropic_ratio: float = 0.0


# Base kernel families whose lengthscale can be made ARD (one value per active
# input dimension) via cfg.data.ard. "cosine" is excluded: gpytorch's
# CosineKernel hardcodes period_length to shape (*batch_shape, 1, 1) — it
# ignores ard_num_dims entirely, so there's no per-dimension formula to opt
# into. "dot_product" has no lengthscale at all (see its docstring) and
# "hebo" is unconditionally ARD already (its tuned prior, not this flag).
_ARD_ELIGIBLE_KERNELS = frozenset(
    {"rbf", "matern12", "matern32", "matern52", "periodic", "rational_quadratic"}
)


HEBO_PLUS_HYPERPARAMETERS: Dict[str, object] = {
    # Tuned "HEBO+" prior constants — the exact config that trained
    # pfns4bo_upstream's released `hebo_plus_model` checkpoint. Source:
    # pfns4bo_upstream/pfns4bo/priors/hebo_prior.py::get_model +
    # Tutorial_Training_for_BO.ipynb + PFN4BO.pdf Appendix B.1. Used directly
    # by _kernel_prior_spec / _nugget_prior's "hebo" branches below — this
    # file reproduces the exact sampled distributions via
    # gpytorch.priors.GammaPrior/LogNormalPrior instead of calling upstream
    # at runtime (see module docstring).
    "lengthscale_concentration": 1.2106559584074301,
    "lengthscale_rate": 1.5212245992840594,
    "outputscale_concentration": 0.8452312502679863,
    "outputscale_rate": 0.3993553245745406,
    "hebo_noise_logmean": -4.63,
    "hebo_noise_std": 0.5,
    "add_linear_kernel": False,  # tuned HEBO+ drops the linear-kernel term — not implemented here
    "hebo_warping": False,  # tuned HEBO+ drops input warping — not implemented here
}


def _kernel_prior_spec(cfg, kernel_name: str) -> KernelPriorSpec:
    """Build the LogNormal/Gamma hyperprior spec for one base kernel family.

    Every numeric constant is overridable via cfg.data (getattr-defaulted,
    same convention the old l_min/l_max/alpha2_min/alpha2_max ranges used),
    except "hebo"'s, which are the fixed tuned HEBO+ constants above.
    """
    isotropic_ratio = float(getattr(cfg.data, "isotropic_ratio", 0.0))

    if kernel_name == "hebo":
        return KernelPriorSpec(
            lengthscale_prior=lambda k: GammaPrior(
                HEBO_PLUS_HYPERPARAMETERS["lengthscale_concentration"],
                HEBO_PLUS_HYPERPARAMETERS["lengthscale_rate"],
            ),
            outputscale_prior=GammaPrior(
                HEBO_PLUS_HYPERPARAMETERS["outputscale_concentration"],
                HEBO_PLUS_HYPERPARAMETERS["outputscale_rate"],
            ),
            ard=True,
            isotropic_ratio=isotropic_ratio,
        )

    l_loc = float(getattr(cfg.data, "l_lognormal_loc", 0.0))
    l_scale = float(getattr(cfg.data, "l_lognormal_scale", 0.7))
    a_conc = float(getattr(cfg.data, "alpha2_gamma_concentration", 4.0))
    a_rate = float(getattr(cfg.data, "alpha2_gamma_rate", 3.0))
    ard = bool(getattr(cfg.data, "ard", False)) and kernel_name in _ARD_ELIGIBLE_KERNELS

    def lengthscale_prior(k: int) -> LogNormalPrior:
        # loc is constant in k (active kernel dims) — no sqrt(k)/log(k) shift.
        return LogNormalPrior(l_loc, l_scale)

    # cosine has no `.lengthscale` attribute — its one shape parameter is
    # `.period_length`, playing the same role "l" does in cosine_kernel's
    # formula (NOT the same as periodic's separate `period` parameter below).
    lengthscale_attr = "period_length" if kernel_name == "cosine" else "lengthscale"

    extra_priors: Dict[str, Prior] = {}
    if kernel_name == "periodic":
        p_loc = float(getattr(cfg.data, "period_lognormal_loc", math.log(1.2)))
        p_scale = float(getattr(cfg.data, "period_lognormal_scale", 0.4))
        extra_priors["period"] = LogNormalPrior(p_loc, p_scale)
    elif kernel_name == "rational_quadratic":
        rq_conc = float(getattr(cfg.data, "rq_alpha_gamma_concentration", 2.0))
        rq_rate = float(getattr(cfg.data, "rq_alpha_gamma_rate", 1.0))
        extra_priors["rq_alpha"] = GammaPrior(rq_conc, rq_rate)

    return KernelPriorSpec(
        lengthscale_prior=lengthscale_prior,
        outputscale_prior=GammaPrior(a_conc, a_rate),
        lengthscale_attr=lengthscale_attr,
        extra_priors=extra_priors,
        ard=ard,
        isotropic_ratio=isotropic_ratio,
    )


def _nugget_prior(cfg, kernel_name: str) -> LogNormalPrior:
    """Diagonal regulariser prior, shared by every kernel (including "hebo" — this
    is HEBO+'s tuned noise prior, LogNormal(-4.63, 0.5), now the default noise
    floor for all kernels; see HEBO_PLUS_HYPERPARAMETERS for provenance)."""
    loc = float(getattr(cfg.data, "nugget_lognormal_loc", -4.63))
    scale = float(getattr(cfg.data, "nugget_lognormal_scale", 0.5))
    return LogNormalPrior(loc, scale)


def _build_likelihood(cfg, kernel_name: str, B: int, device) -> gpytorch.likelihoods.GaussianLikelihood:
    """Sample B episodes' diagonal noise (same _nugget_prior every kernel
    family already used) and hand it back wrapped in a GaussianLikelihood,
    so the rest of this file adds noise via gpytorch's own
    `likelihood(mvn)` instead of a hand-added `nugget * torch.eye(...)`.
    `.noise` is the "nugget" name used everywhere else in this file — same
    quantity, just gpytorch's own container for it."""
    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([B])).to(device)
    likelihood.noise = _nugget_prior(cfg, kernel_name).sample(torch.Size([B])).to(device)
    return likelihood


def _collapse_isotropic(sample: Tensor, iso_mask: Optional[Tensor]) -> Tensor:
    """Force the last (ard_num_dims) axis of an ARD-shaped `sample` to a
    single shared value, per episode, for every episode flagged True in
    iso_mask (B,) — i.e. that episode's kernel is isotropic in effect even
    though its lengthscale/period tensor keeps the ARD (k,)-per-episode
    shape. The shared value is simply the first of the already-sampled k
    values (still a valid draw from the same per-dim prior — see
    _kernel_prior_spec's lengthscale_prior, whose distribution doesn't
    depend on k), so this needs no extra sampling call.

    No-op when iso_mask is None (isotropic_ratio<=0 or spec.ard=False) or
    `sample`'s last axis already has size 1 (nothing to collapse)."""
    if iso_mask is None or sample.shape[-1] == 1:
        return sample
    collapsed = sample[..., :1].expand_as(sample)
    mask = iso_mask.view(-1, *([1] * (sample.dim() - 1)))
    return torch.where(mask, collapsed, sample)


def _build_scaled_kernel(
    name: str, spec: KernelPriorSpec, k: int, B: int, device, active_dims: Optional[List[int]] = None
) -> tuple[gpytorch.kernels.Kernel, Dict[str, Tensor]]:
    """Sample B episodes' hyperparameters for one base kernel and return the
    resulting ScaleKernel(base)(batch_shape=[B]) object plus a dict of the
    sampled values (keyed by the output-dict schema names: l, alpha2, and
    any of spec.extra_priors' keys).

    active_dims (gpytorch.kernels.Kernel's own constructor kwarg) makes the
    base kernel select its k active columns out of the caller's full-width
    input at call time (Kernel.__call__ index_selects them internally) —
    callers pass the full d_features tensor straight through instead of
    pre-slicing a (..., k) sub-matrix themselves."""
    batch_shape = torch.Size([B])
    kernel_kwargs: Dict = {"batch_shape": batch_shape}
    if active_dims is not None:
        kernel_kwargs["active_dims"] = active_dims
    if spec.ard:
        kernel_kwargs["ard_num_dims"] = k
    # gpytorch kernel modules default to CPU-resident parameters regardless
    # of `device`; move before assigning sampled values so the in-place
    # `self.initialize(...)` used by the `.lengthscale =` / `.outputscale =`
    # setters below copies into device-resident storage, not CPU storage.
    base = _BASE_GPYTORCH_KERNEL_CLS[name](**kernel_kwargs).to(device)

    # One shared per-episode isotropic-override coin flip, reused below for
    # both the lengthscale and (for "periodic") period — see
    # _collapse_isotropic / cfg.data.isotropic_ratio in the module docstring.
    iso_mask = (
        torch.rand(B, device=device) < spec.isotropic_ratio
        if spec.ard and spec.isotropic_ratio > 0.0
        else None
    )

    l_attr = getattr(base, spec.lengthscale_attr)
    l_sample = spec.lengthscale_prior(k).sample(l_attr.shape).to(device)
    l_sample = _collapse_isotropic(l_sample, iso_mask)
    setattr(base, spec.lengthscale_attr, l_sample)

    scaled = gpytorch.kernels.ScaleKernel(base, batch_shape=batch_shape).to(device)
    a_sample = spec.outputscale_prior.sample(scaled.outputscale.shape).to(device)
    scaled.outputscale = a_sample

    l_flat = l_sample.reshape(B, -1)
    params: Dict[str, Tensor] = {
        "l": l_flat.squeeze(-1) if l_flat.shape[-1] == 1 else l_flat,
        "alpha2": a_sample.reshape(B),
    }
    for schema_name, prior in spec.extra_priors.items():
        attr_name = _EXTRA_PARAM_TO_ATTR[schema_name]
        attr = getattr(base, attr_name)
        sample = prior.sample(attr.shape).to(device)
        # "period" is ARD-vector-shaped too when spec.ard (gpytorch's
        # PeriodicKernel ties period_length's ard_num_dims to the same
        # kwarg as lengthscale), so it collapses under the same iso_mask
        # used for "l" above (keeps both isotropic together, per episode).
        # "rq_alpha" is never ARD (RQKernel.alpha has no ard_num_dims), so
        # this is a no-op collapse/reshape/squeeze for it.
        sample = _collapse_isotropic(sample, iso_mask)
        setattr(base, attr_name, sample)
        sample_flat = sample.reshape(B, -1)
        params[schema_name] = sample_flat.squeeze(-1) if sample_flat.shape[-1] == 1 else sample_flat

    return scaled, params


def _sample_episode_kernel(
    cfg, kernel_name: str, k: int, B: int, device, active_dims: Optional[List[int]] = None
) -> tuple[gpytorch.kernels.Kernel, Dict[str, Tensor]]:
    """Sample B episodes' hyperparameters for kernel_name (base or "A+B"/"A*B"
    composite) and return (gpytorch Kernel with batch_shape=[B], params dict).

    "dot_product" has no lengthscale, but its LinearKernel `variance` is
    sampled from the same alpha2 ~ Gamma(alpha2_gamma_concentration,
    alpha2_gamma_rate) prior every other kernel's outputscale uses (see
    dot_product_kernel's docstring). params keys match the output-dict schema
    (l, alpha2, period, rq_alpha, l_b, alpha2_b, period_b, rq_alpha_b);
    not-applicable entries are filled with a 0.0 sentinel (the convention
    pit.py::gp_analytical_pit relies on).

    active_dims: column indices (out of the caller's full d_features input)
    this kernel is active on — None means every column. Forwarded to
    gpytorch's own active_dims kwarg (see _build_scaled_kernel), so callers
    pass the full-width input straight through instead of pre-slicing it.
    """
    composite = _parse_composite(kernel_name)
    if kernel_name == "dot_product":
        batch_shape = torch.Size([B])
        kernel_kwargs: Dict = {"batch_shape": batch_shape}
        if active_dims is not None:
            kernel_kwargs["active_dims"] = active_dims
        kernel = gpytorch.kernels.LinearKernel(**kernel_kwargs).to(device)
        a_conc = float(getattr(cfg.data, "alpha2_gamma_concentration", 4.0))
        a_rate = float(getattr(cfg.data, "alpha2_gamma_rate", 3.0))
        a_sample = GammaPrior(a_conc, a_rate).sample(kernel.variance.shape).to(device)
        kernel.variance = a_sample
        params: Dict[str, Tensor] = {
            "l": torch.zeros(B, device=device),
            "alpha2": a_sample.reshape(B),
        }
    elif composite is None:
        spec = _kernel_prior_spec(cfg, kernel_name)
        kernel, params = _build_scaled_kernel(kernel_name, spec, k, B, device, active_dims=active_dims)
    else:
        name_a, op, name_b = composite
        kernel_a, params_a = _build_scaled_kernel(
            name_a, _kernel_prior_spec(cfg, name_a), k, B, device, active_dims=active_dims
        )
        kernel_b, params_b = _build_scaled_kernel(
            name_b, _kernel_prior_spec(cfg, name_b), k, B, device, active_dims=active_dims
        )
        kernel = kernel_a + kernel_b if op == "+" else kernel_a * kernel_b
        params = dict(params_a)
        for key, val in params_b.items():
            params[f"{key}_b"] = val

    for key in ("period", "rq_alpha", "l_b", "alpha2_b", "period_b", "rq_alpha_b"):
        params.setdefault(key, torch.zeros(B, device=device))

    return kernel, params


def _build_concrete_kernel(
    name: str, l, alpha2, *, period=None, rq_alpha=None, active_dims: Optional[List[int]] = None
) -> gpytorch.kernels.Kernel:
    """Construct a non-batched gpytorch Kernel with CONCRETE hyperparameter
    values assigned — reconstruction (given known values), not sampling.
    Used by build_kernel_fn.

    "dot_product" returns the bare LinearKernel (no ScaleKernel wrapper):
    its `variance` already plays the role `alpha2` plays for every other
    kernel, so wrapping it would just be a second, redundant alpha2. "l" is
    ignored — no lengthscale, geometry comes entirely from the feature space.

    active_dims: column indices this kernel reads out of the caller's
    full-width input (gpytorch's own kwarg — see _build_scaled_kernel);
    None means every column.
    """
    if name == "dot_product":
        kernel_kwargs: Dict = {}
        if active_dims is not None:
            kernel_kwargs["active_dims"] = active_dims
        kernel = gpytorch.kernels.LinearKernel(**kernel_kwargs)
        kernel.variance = torch.as_tensor(alpha2, dtype=torch.get_default_dtype()).reshape(kernel.variance.shape)
        return kernel

    l_t = l if torch.is_tensor(l) else torch.as_tensor(l, dtype=torch.get_default_dtype())
    # l having more than one element means this episode was generated ARD
    # (cfg.data.ard=True for rbf/matern32/periodic/rational_quadratic, or
    # always for "hebo") — gpytorch needs ard_num_dims at construction time
    # to size .lengthscale (and, for "periodic", .period_length — see the
    # reshape below) correctly before values can be assigned into it.
    kernel_kwargs = {"ard_num_dims": l_t.numel()} if l_t.numel() > 1 else {}
    if active_dims is not None:
        kernel_kwargs["active_dims"] = active_dims
    base = _BASE_GPYTORCH_KERNEL_CLS[name](**kernel_kwargs)
    attr = "period_length" if name == "cosine" else "lengthscale"
    setattr(base, attr, l_t.reshape(getattr(base, attr).shape))
    if name == "periodic" and period is not None:
        period_t = period if torch.is_tensor(period) else torch.as_tensor(float(period))
        base.period_length = period_t.reshape(base.period_length.shape)
    if name == "rational_quadratic" and rq_alpha is not None:
        base.alpha = torch.as_tensor(float(rq_alpha)).reshape(base.alpha.shape)

    scale = gpytorch.kernels.ScaleKernel(base)
    scale.outputscale = torch.as_tensor(alpha2, dtype=torch.get_default_dtype()).reshape(scale.outputscale.shape)
    return scale


def build_kernel_fn(
    kernel_name: str,
    l,
    alpha2,
    *,
    period: Optional[float | Tensor] = None,
    rq_alpha: Optional[float] = None,
    l_b=None,
    alpha2_b=None,
    period_b: Optional[float | Tensor] = None,
    rq_alpha_b: Optional[float] = None,
    active_dims: Optional[List[int]] = None,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Return a kernel(X1, X2) -> K callable with hyperparameters baked in.

    l_b/alpha2_b/period_b/rq_alpha_b are the second component's hyperparameters
    for composite ("A+B" / "A*B") kernels. l/l_b/period/period_b may be an
    ARD per-dimension vector (Tensor) instead of a scalar when the episode
    was generated with cfg.data.ard=True (or "hebo", always ARD).

    active_dims: column indices this kernel is active on (both components of
    a composite share the same active columns — see generate_gp_task/
    generate_gp_batch, which sample one column subset per task/batch). The
    caller passes its full-width X1/X2 straight through; gpytorch's own
    active_dims kwarg selects the columns internally. None means every
    column (e.g. "dot_product" tasks that draw on all d_features).
    """
    composite = _parse_composite(kernel_name)
    if composite is None:
        kernel = _build_concrete_kernel(kernel_name, l, alpha2, period=period, rq_alpha=rq_alpha, active_dims=active_dims)
    else:
        name_a, op, name_b = composite
        kernel_a = _build_concrete_kernel(name_a, l, alpha2, period=period, rq_alpha=rq_alpha, active_dims=active_dims)
        kernel_b = _build_concrete_kernel(
            name_b, l_b, alpha2_b, period=period_b, rq_alpha=rq_alpha_b, active_dims=active_dims
        )
        kernel = kernel_a + kernel_b if op == "+" else kernel_a * kernel_b

    # kernel's parameters are CPU-resident regardless of the device l/alpha2
    # were on (see _build_scaled_kernel's docstring note) — move lazily to
    # X1's device at call time, since X1 isn't known yet at construction time.
    return lambda X1, X2: kernel.to(X1.device)(X1, X2).to_dense()


# ---------------------------------------------------------------------------
# Kernel registry (names + free-function dispatch, e.g. for
# scripts/visualize_kernel.py's membership checks and ALL_KERNELS)
#
# Every base kernel below evaluates the real gpytorch kernel object via
# build_kernel_fn — single source of truth for the math, no hand-rolled
# formula to drift out of sync with the gpytorch-backed episode-generation
# path above. NOT usable for backprop into l/alpha2/period/rq_alpha:
# build_kernel_fn assigns hyperparameters through gpytorch's `.lengthscale =`
# / `.outputscale =` setters, which do an in-place `.initialize()` copy that
# breaks autograd back to those inputs (the forward pass on X1/X2 itself is
# still differentiable). src/evaluate_baselines.py's from-scratch GP-MLE fit
# needs exactly that backprop, so it keeps its own local plain-torch copies of
# rbf_kernel/periodic_kernel/rational_quadratic_kernel instead of importing
# these — see its module comment.
# ---------------------------------------------------------------------------


def rbf_kernel(X1: Tensor, X2: Tensor, *, l, alpha2, **_) -> Tensor:
    """Squared Exponential (RBF), via gpytorch.kernels.RBFKernel."""
    return build_kernel_fn("rbf", l, alpha2)(X1, X2)


def matern12_kernel(X1: Tensor, X2: Tensor, *, l, alpha2, **_) -> Tensor:
    """Matérn ν=1/2, via gpytorch.kernels.MaternKernel(nu=0.5)."""
    return build_kernel_fn("matern12", l, alpha2)(X1, X2)


def matern32_kernel(X1: Tensor, X2: Tensor, *, l, alpha2, **_) -> Tensor:
    """Matérn ν=3/2, via gpytorch.kernels.MaternKernel(nu=1.5)."""
    return build_kernel_fn("matern32", l, alpha2)(X1, X2)


def matern52_kernel(X1: Tensor, X2: Tensor, *, l, alpha2, **_) -> Tensor:
    """Matérn ν=5/2, via gpytorch.kernels.MaternKernel(nu=2.5)."""
    return build_kernel_fn("matern52", l, alpha2)(X1, X2)


def cosine_kernel(X1: Tensor, X2: Tensor, *, l, alpha2, **_) -> Tensor:
    """Cosine (spectral), via gpytorch.kernels.CosineKernel."""
    return build_kernel_fn("cosine", l, alpha2)(X1, X2)


def periodic_kernel(X1: Tensor, X2: Tensor, *, l, alpha2, period: float = 1.0, **_) -> Tensor:
    """Periodic, via gpytorch.kernels.PeriodicKernel."""
    return build_kernel_fn("periodic", l, alpha2, period=period)(X1, X2)


def rational_quadratic_kernel(X1: Tensor, X2: Tensor, *, l, alpha2, rq_alpha: float = 1.0, **_) -> Tensor:
    """Rational Quadratic, via gpytorch.kernels.RQKernel."""
    return build_kernel_fn("rational_quadratic", l, alpha2, rq_alpha=rq_alpha)(X1, X2)


def dot_product_kernel(X1: Tensor, X2: Tensor, *, alpha2: float = 1.0, **_) -> Tensor:
    """Linear (dot product): alpha2 * X1 @ X2ᵀ, via gpytorch.kernels.LinearKernel.

    PSD because K = XᵀX is PSD. No lengthscale hyperparameter — "l" is
    ignored (build_kernel_fn's dot_product branch doesn't use it).
    """
    return build_kernel_fn("dot_product", 0.0, alpha2)(X1, X2)


def _hebo_kernel_dispatch(X1: Tensor, X2: Tensor, *, l, alpha2, **_) -> Tensor:
    return build_kernel_fn("hebo", l, alpha2)(X1, X2)


KERNEL_REGISTRY: Dict[str, Callable[..., Tensor]] = {
    "rbf": rbf_kernel,
    "matern12": matern12_kernel,
    "matern32": matern32_kernel,
    "matern52": matern52_kernel,
    "cosine": cosine_kernel,
    "periodic": periodic_kernel,
    "rational_quadratic": rational_quadratic_kernel,
    "dot_product": dot_product_kernel,
    "hebo": _hebo_kernel_dispatch,
}


# ---------------------------------------------------------------------------
# Composite kernels: sum / product of two base kernels
# ---------------------------------------------------------------------------
# Sums and products of PSD kernels are PSD, so "rbf+periodic" (locally
# periodic — smooth decay times exact periodicity) or "matern32*cosine"
# (spectral windowing) are valid kernels without any new math. Restricted to
# the kernels below because they share one calling convention (l, alpha2,
# plus an optional named extra); dot_product and hebo have irregular
# signatures (no lengthscale / ARD-only) and are left out of composites.
_COMPOSABLE_KERNELS: List[str] = ["rbf", "matern32", "cosine", "periodic", "rational_quadratic"]

# Kernels whose PSD guarantee only holds for scalar (1D) inputs — composites
# that include one of these must also cap the active kernel dimensionality
# to k=1 (see generate_gp_task / generate_gp_batch). Verified empirically:
# CosineKernel used isotropically is not PSD for k>=2 (Bochner/Schoenberg —
# an isotropic cos(||x||) is not a valid Mercer kernel for d>1). "periodic"
# is NOT in this set: gpytorch's ARD PeriodicKernel (ard_num_dims=k) is
# independently PSD for k>1 (per-dimension lengthscale/period, product-
# combined), so it uses the same _sample_active_dims / ARD path as
# rbf/matern32/rational_quadratic — see _ARD_ELIGIBLE_KERNELS. (It's still
# forced to k=1 in generate_gp_batch, but for identifiability, not PSD —
# see the kernel_cols selection there.)
_SCALAR_ONLY_KERNELS = {"cosine"}


def _parse_composite(name: str) -> Optional[tuple]:
    """Split "A+B" / "A*B" into (name_a, op, name_b), or None if not composite."""
    for op in ("+", "*"):
        if op in name:
            a, _, b = name.partition(op)
            if a in _COMPOSABLE_KERNELS and b in _COMPOSABLE_KERNELS:
                return a, op, b
    return None


def _kernel_needs_scalar_input(kernel_name: str) -> bool:
    """True if this kernel (or any component of a composite/chain) requires
    k=1 input dims. Uses a generic re.split rather than _parse_composite
    (which only handles exactly 2 parts via .partition) so this is correct
    for both the legacy 2-way composites and arbitrary-length systematic
    chains (cfg.data.systematic_composition) alike — e.g. a 3-way chain like
    "rbf+cosine*periodic" must still be detected as scalar-only because of
    the cosine component, even though it isn't a name _parse_composite
    recognizes."""
    return any(part in _SCALAR_ONLY_KERNELS for part in re.split(r"[+*]", kernel_name))


def _composite_kernel(
    X1: Tensor,
    X2: Tensor,
    *,
    kernel_name: str,
    l: float,
    alpha2: float,
    l_b: Optional[float] = None,
    alpha2_b: Optional[float] = None,
    period: Optional[float] = None,
    period_b: Optional[float] = None,
    rq_alpha: Optional[float] = None,
    rq_alpha_b: Optional[float] = None,
    **_,
) -> Tensor:
    """Evaluate a registered "A+B" / "A*B" composite kernel (KERNEL_REGISTRY
    dispatch convention) by delegating to build_kernel_fn."""
    fn = build_kernel_fn(
        kernel_name, l, alpha2, period=period, rq_alpha=rq_alpha,
        l_b=l_b, alpha2_b=alpha2_b, period_b=period_b, rq_alpha_b=rq_alpha_b,
    )
    return fn(X1, X2)


COMPOSITE_KERNELS: List[str] = []
for _name_a, _name_b in itertools.combinations(_COMPOSABLE_KERNELS, 2):
    for _op in ("+", "*"):
        _combo_name = f"{_name_a}{_op}{_name_b}"
        KERNEL_REGISTRY[_combo_name] = functools.partial(_composite_kernel, kernel_name=_combo_name)
        COMPOSITE_KERNELS.append(_combo_name)
del _name_a, _name_b, _op, _combo_name

ALL_KERNELS: List[str] = list(KERNEL_REGISTRY.keys())


def _sample_d_features(cfg) -> int:
    """Return the total feature-column count d for this batch/shard.

    If cfg.data.d_features_lognormal_loc/scale are both set, d ~
    round(LogNormal(d_features_lognormal_loc, d_features_lognormal_scale)),
    clipped to a minimum of 2 (a single-feature task is degenerate — see
    _sample_active_dims, which already enforces the same floor on the
    *active* subset of d). Sampled once per generate_gp_batch call, i.e.
    once per shard in generate_pit_dataset.py, matching the granularity
    kernel_name/P/N/active_dims already use — every episode within one
    shard shares the same d. Falls back to the fixed cfg.data.d_features
    when the lognormal keys are absent (backward compat with old configs;
    also what every unit test that pins an exact d relies on).
    """
    loc = getattr(cfg.data, "d_features_lognormal_loc", None)
    scale = getattr(cfg.data, "d_features_lognormal_scale", None)
    if loc is None or scale is None:
        return int(cfg.data.d_features)
    return max(2, round(random.lognormvariate(float(loc), float(scale))))


def _sample_active_dims(d_total: int, cfg) -> List[int]:
    """Return a sorted list of column indices that the kernel will use.

    A fraction of the d_total feature columns ~ Uniform[inactive_frac_min,
    inactive_frac_max] is left inactive (irrelevant noise the model must
    learn to ignore); the remaining columns are the kernel's active_dims.
    Falls back to using every column when the config keys are absent
    (backward compat with old episode files / configs).
    """
    frac_min = float(getattr(cfg.data, "inactive_frac_min", 0.0))
    frac_max = float(getattr(cfg.data, "inactive_frac_max", 0.0))
    frac = random.uniform(frac_min, frac_max)
    k = d_total - round(frac * d_total)
    k = max(1, min(k, d_total))
    return sorted(random.sample(range(d_total), k))


def _resolve_kernel_name(cfg) -> str:
    """Pick which kernel to use for one task based on config."""
    data = cfg.data
    if hasattr(data, "kernel") and data.kernel:
        name = str(data.kernel)
        if name not in KERNEL_REGISTRY:
            raise ValueError(f"Unknown kernel '{name}'. Choose from {ALL_KERNELS}.")
        return name
    if hasattr(data, "kernels") and data.kernels:
        pool = list(data.kernels)
        for k in pool:
            if k not in KERNEL_REGISTRY:
                raise ValueError(f"Unknown kernel '{k}'. Choose from {ALL_KERNELS}.")
        return random.choice(pool)
    return "rbf"


def _sample_kernel_chain_structure(cfg) -> tuple[List[str], List[str], str]:
    """CauKer-style composition (github.com/ShifengXIE/CauKer): sample a
    random component COUNT m ~ Uniform[composite_num_kernels_min,
    composite_num_kernels_max], then a length-m list of elementary kernels
    (with replacement) from _COMPOSABLE_KERNELS, then m-1 independently
    sampled +/* operators to combine them left-to-right (functools.reduce,
    see _build_kernel_chain) — instead of picking from the fixed 20-entry
    COMPOSITE_KERNELS pool. Active only when cfg.data.systematic_composition
    is True (see _resolve_kernel_name's docstring for the non-systematic
    path). cfg.data.composite_exclude_kernels (optional list, default empty)
    drops named elementary kernels from the sampling pool without touching
    _COMPOSABLE_KERNELS itself — that constant also seeds the static 20-entry
    COMPOSITE_KERNELS/KERNEL_REGISTRY at import time (module-level loop
    above), which must stay unfiltered for the non-systematic path. Returns
    (names, ops, chain_name) where chain_name is the same "A+B*C"-style
    left-to-right string the static composites already use (m=1 degenerates
    to a bare base-kernel name, no ops)."""
    exclude = set(getattr(cfg.data, "composite_exclude_kernels", None) or [])
    pool = [k for k in _COMPOSABLE_KERNELS if k not in exclude]
    if not pool:
        raise ValueError(
            f"composite_exclude_kernels={sorted(exclude)} excludes every kernel "
            f"in _COMPOSABLE_KERNELS={_COMPOSABLE_KERNELS}"
        )
    lo = int(getattr(cfg.data, "composite_num_kernels_min", 1))
    hi = int(getattr(cfg.data, "composite_num_kernels_max", 4))
    m = random.randint(lo, hi)
    names = random.choices(pool, k=m)
    ops = [random.choice(("+", "*")) for _ in range(m - 1)]
    chain_name = names[0] + "".join(f"{op}{name}" for op, name in zip(ops, names[1:]))
    return names, ops, chain_name


def _build_kernel_chain(
    cfg, names: List[str], ops: List[str], k: int, B: int, device, active_dims: Optional[List[int]] = None
) -> tuple[gpytorch.kernels.Kernel, List[Dict[str, Tensor]]]:
    """Sample B episodes' hyperparameters for each component in `names` (via
    the same _build_scaled_kernel/_kernel_prior_spec machinery
    _sample_episode_kernel's composite branch already uses — no new
    hyperparameter-sampling logic) and combine the resulting kernel objects
    left-to-right per `ops`. Returns the combined Kernel plus one params
    dict per component (in `names` order), UNFLATTENED — not coerced into
    the legacy l_b/alpha2_b-style schema, since component count is variable
    here (see generate_gp_batch's return_kernel_metadata handling)."""
    built = [
        _build_scaled_kernel(name, _kernel_prior_spec(cfg, name), k, B, device, active_dims=active_dims)
        for name in names
    ]
    kernel = built[0][0]
    for op, (comp_kernel, _) in zip(ops, built[1:]):
        kernel = kernel + comp_kernel if op == "+" else kernel * comp_kernel
    component_params = [params for _, params in built]
    return kernel, component_params


# ---------------------------------------------------------------------------
# GP posterior (kernel-agnostic)
# ---------------------------------------------------------------------------


def gp_posterior(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    kernel_fn: Callable[[Tensor, Tensor], Tensor],
    noise: float,
    *,
    latent: bool = True,
    return_factors: bool = False,
) -> tuple:
    """Analytical GP posterior for an arbitrary stationary kernel.

    Args:
        latent: if True, return posterior over f* (latent GP), not noisy y*.
                K_ss excludes the noise term so that R* reflects kernel structure
                rather than being diluted by σ² in the diagonal.
        return_factors: if True, also return (L_ff, alpha) so the caller can
                reuse them for the LOO PIT without a second Cholesky.

    Returns:
        mu_star   : (N,)   — posterior mean at test points
        Sigma_star: (N, N) — posterior covariance at test points
        L_ff      : (P, P) — Cholesky of K_ff  (only if return_factors=True)
        alpha     : (P,)   — K_ff^{-1} y_train (only if return_factors=True)
    """
    P, N = x_train.shape[0], x_test.shape[0]

    K_ff = kernel_fn(x_train, x_train) + noise * torch.eye(P, device=x_train.device)
    K_sf = kernel_fn(x_test, x_train)   # (N, P)
    K_ss = kernel_fn(x_test, x_test)
    if not latent:
        K_ss = K_ss + noise * torch.eye(N, device=x_test.device)

    L_ff = _safe_cholesky(K_ff, max_attempts=12)
    alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L_ff).squeeze(-1)  # (P,)

    mu_star = K_sf @ alpha  # (N,)

    V = torch.linalg.solve_triangular(L_ff, K_sf.T, upper=False)  # (P, N)
    Sigma_star = K_ss - V.T @ V  # (N, N)
    Sigma_star = 0.5 * (Sigma_star + Sigma_star.T)
    if return_factors:
        return mu_star, Sigma_star, L_ff, alpha
    return mu_star, Sigma_star


class _GeneratorGP(gpytorch.models.ExactGP):
    """Thin ExactGP wrapper so oracle_mode="posterior" conditioning goes
    through gpytorch's own exact-inference machinery (`model(x_test)` in
    eval mode) instead of the hand-rolled K_ss - K_sf K_ff^-1 K_fs formula
    gp_posterior used. `kernel` is the already-sampled (batch_shape=[B])
    Kernel object from _sample_episode_kernel — one instance, no per-family
    branching needed here. Verified numerically equivalent (~1e-6 max abs
    diff) to the old manual computation, given max_cholesky_size forced
    high enough (see _MAX_CHOLESKY) and fast_pred_var disabled at call time."""

    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood, kernel: gpytorch.kernels.Kernel, batch_shape: torch.Size):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
        self.covar_module = kernel

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def sigma_to_correlation(Sigma: Tensor) -> tuple[Tensor, Tensor]:
    """Convert covariance matrix to correlation matrix and marginal std."""
    sigma = Sigma.diagonal().clamp(min=1e-10).sqrt()  # (N,)
    D_inv = torch.diag(1.0 / sigma)
    R = D_inv @ Sigma @ D_inv
    # One-shot re-normalization using the original sigma (symmetric in i,j).
    # D_inv @ Sigma @ D_inv already gives diagonal=1 for PSD Sigma; this just
    # corrects any float32 rounding drift without introducing asymmetry.
    d = R.diagonal().clamp(min=1e-10).sqrt()
    R = R / (d.unsqueeze(0) * d.unsqueeze(1))
    return R, sigma


# ---------------------------------------------------------------------------
# Task generator
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_gp_task(cfg) -> Dict[str, Tensor]:
    """Sample one GP task and return a dict of tensors.

    Thin wrapper around generate_gp_batch(cfg, 1, "cpu",
    return_kernel_metadata=True) — a single episode is just a batch of one,
    and the two functions used to duplicate ~150 lines of kernel/column
    selection, hyperparameter sampling, feature warp, y sampling, and
    oracle-mode branching that's now implemented once. See
    generate_gp_batch's docstring for the full behaviour (kernel/active_dims
    selection, ARD, nugget, oracle_mode, seeding); this only documents the
    single-episode-specific return schema.

    Known behaviour change from the pre-dedup version: kernel column
    selection now goes through generate_gp_batch's vectorised active_dims
    sampling (including dot_product always using every column) instead of a
    separate single-episode code path — both are uniform draws over the same
    distribution, so this doesn't change task statistics, but a fixed
    cfg.seed will no longer reproduce the exact old column indices bit-for-bit.

    Keys returned (see generate_gp_batch for shapes/semantics):
        x_norm_train, y_train, x_norm_test, y_test,
        z_train, z_test, log_pdf_test,
        R_star, Sigma_star, mu_star, sigma_star, n_train, n_test,
        l, alpha2, nugget, kernel, period, rq_alpha,
        l_b, alpha2_b, period_b, rq_alpha_b, kernel_feature_indices,
        _L_ff, _alpha  (ephemeral Cholesky factors, consumed by
                        pit.py::gp_analytical_pit, not saved to disk)
    """
    return generate_gp_batch(cfg, 1, "cpu", return_kernel_metadata=True)[0]


# ---------------------------------------------------------------------------
# Batched generation (C: vectorised over B episodes simultaneously)
# ---------------------------------------------------------------------------


def _batched_cholesky(K: Tensor) -> Tensor:
    """Batched Cholesky (B, N, N) → (B, N, N) with automatic jitter for failures."""
    L, info = torch.linalg.cholesky_ex(K)
    failed = info.ne(0)
    if not failed.any():
        return L
    eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
    for jitter in (1e-5, 1e-4, 1e-3, 1e-2, 0.1):
        if not failed.any():
            break
        K = K.clone()
        K[failed] = K[failed] + jitter * eye
        L_new, info_new = torch.linalg.cholesky_ex(K)
        L[failed] = L_new[failed]
        failed = info_new.ne(0)
    if failed.any():
        # Last resort: replace with identity so the episode is invalid but non-crashing.
        L[failed] = eye.unsqueeze(0).expand_as(L[failed])
    return L


def tabiclv2_warp_features(x: Tensor, seed: Optional[int] = None) -> Tensor:
    """Warp each feature column with one of 8 random marginal transforms.

    Simulates the extreme marginal heterogeneity of real tabular data
    (TabICLv2): heavy tails, power laws, ordinal steps, bimodal mixtures,
    periodicity, and Cauchy outliers, applied on top of a Standard Normal
    baseline. Intended to run before any per-episode mean/std normalisation,
    so downstream kernel/covariance code keeps operating on calibrated,
    unit-scale features while the model still sees the warped shape.

    Args:
        x: (B, T, d) or (T, d) tensor of Standard Normal features.
        seed: if given, seeds python/numpy/torch RNGs so the warp choice and
            all sampled transform parameters are reproducible. Leave None
            when called from generate_gp_task/generate_gp_batch — those
            already seed globally before calling this, so reseeding here
            would just restart the same streams.

    Returns:
        Tensor of the same shape as `x`, with each (episode, column) warped
        independently by a randomly chosen transform.
    """
    if seed is not None:
        _seed_everything(seed)

    added_batch_dim = x.dim() == 2
    if added_batch_dim:
        x = x.unsqueeze(0)

    B, T, d = x.shape
    warped_x = x.clone()
    choices = torch.randint(0, 8, (B, d), device=x.device)

    for b in range(B):
        for col in range(d):
            c = choices[b, col].item()
            col_data = warped_x[b, :, col]

            if c == 0:  # Identity — Standard Normal baseline
                continue
            elif c == 1:  # Signed-square — mild heavy tails
                warped_x[b, :, col] = torch.sign(col_data) * (col_data ** 2)
            elif c == 2:  # Cube — Student-T-like heavy tails
                warped_x[b, :, col] = col_data ** 3
            elif c == 3:  # Log-normal / exponential — right-skewed power law
                # Clamp before exp() to avoid float overflow.
                warped_x[b, :, col] = torch.exp(col_data.clamp(min=-5.0, max=4.0))
            elif c == 4:  # Quantization — ordinal / discrete steps
                warped_x[b, :, col] = torch.round(col_data * 2.0) / 2.0
            elif c == 5:  # Bimodal mixture — mixed populations
                mask = torch.rand_like(col_data) > 0.5
                shift = torch.randn(1, device=x.device).item() * 4.0
                col_data[mask] += shift
            elif c == 6:  # Cyclic — seasonal / periodic features
                freq = torch.rand(1, device=x.device).item() * 3.0 + 0.5
                warped_x[b, :, col] = torch.sin(col_data * freq)
            elif c == 7:  # Cauchy — extreme heavy tails, undefined variance
                u = torch.erf(col_data / math.sqrt(2.0))
                # Scale by 0.95 to keep tan() away from its asymptotes.
                warped_x[b, :, col] = torch.tan(u * (math.pi / 2.0 * 0.95))

    if added_batch_dim:
        warped_x = warped_x.squeeze(0)
    return warped_x


# Activation bank for MLP feature mixing (adapted from CauKer's SCM activation
# set, applied here to GP *input coordinates* rather than sampled *outputs* —
# see apply_mlp_feature_mixing's docstring for why this preserves exact
# analytic Gaussianity while CauKer's approach would not).
_MLP_MIX_ACTIVATIONS: List[str] = ["linear", "relu", "sigmoid", "sin", "mod", "leaky_relu"]


def _apply_mlp_activation(x: Tensor, name: str) -> Tensor:
    """Elementwise nonlinearity for one MLP-mixing layer. `x` is any shape."""
    if name == "linear":
        return x
    if name == "relu":
        return torch.relu(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "sin":
        return torch.sin(x)
    if name == "mod":
        # Remainder by a fixed period (not data-dependent) keeps this a pure
        # deterministic function of x alone -> still a valid PSD-preserving
        # feature map; 2*pi period avoids introducing a new magic-number
        # scale unrelated to the 'sin' branch above.
        return torch.remainder(x, 2 * math.pi)
    if name == "leaky_relu":
        return torch.nn.functional.leaky_relu(x, negative_slope=0.1)
    raise ValueError(f"Unknown MLP-mixing activation '{name}'")


def apply_mlp_feature_mixing(x: Tensor, cfg, device) -> Tensor:
    """Randomly mix the GP's input feature columns through a small stack of
    dense affine + nonlinearity layers, applied to input coordinates x (never
    to sampled outputs y) so k(f(x_i), f(x_j)) remains a valid PSD kernel for
    the fixed deterministic map f = this mixing stack composed with
    tabiclv2_warp_features -- preserving EXACT analytic Gaussianity (closed-
    form GP posterior/Cholesky oracle), unlike CauKer's SCM approach of mixing
    sampled *outputs* through a random DAG (which would force Monte Carlo).

    Structure mirrors the rest of this file's "shared structure across the
    batch, independent per-episode parameters" convention (kernel_name,
    active_dims, tabiclv2_warp_features's per-(episode,column) transform
    choice): the number of layers L and each layer's activation name are
    sampled ONCE per batch call (shared across all B episodes); each layer's
    weight matrix and bias are sampled independently PER EPISODE and applied
    via a batched einsum (no Python loop over B).

    Note on active_dims/ARD semantics: this is a DENSE mix (every output
    column is a combination of every input column), so it partially subverts
    the "inactive_frac_min/max leaves some columns as pure noise" contract
    downstream in _sample_active_dims -- post-mixing, no column is purely
    irrelevant anymore. This is an accepted trade-off for increased task
    diversity, not a bug.

    Args:
        x: (B, T, d) tensor, already warped by tabiclv2_warp_features, NOT
            yet z-normalised (this runs before the existing per-episode
            mean/std normalisation step).
        cfg: Hydra config; reads cfg.data.mlp_mixing_* keys (see
            conf/data/gp_tasks.yaml), all optional/backward-compatible via
            getattr defaults (mlp_mixing_enabled defaults False -> exact
            no-op, byte-for-byte, for every existing config/dataset).
        device: torch device string, threaded through for the new W_l/b_l
            parameter tensors (same convention as the rest of this file).

    Returns:
        (B, T, d) tensor, same shape/dtype as x. Episodes not selected by the
        per-episode Bernoulli gate (mlp_mixing_prob) are returned unchanged.
    """
    if not bool(getattr(cfg.data, "mlp_mixing_enabled", False)):
        return x

    mixing_prob = float(getattr(cfg.data, "mlp_mixing_prob", 0.3))
    if mixing_prob <= 0.0:
        return x

    L_min = int(getattr(cfg.data, "mlp_num_layers_min", 1))
    L_max = int(getattr(cfg.data, "mlp_num_layers_max", 2))
    w_std = float(getattr(cfg.data, "mlp_mix_weight_std", 1.0))

    B, T, d = x.shape
    L = random.randint(L_min, L_max)
    # Activation sequence shared across the whole batch call (same granularity
    # as kernel_name/P/N/active_dims above) -- NOT per-episode, so every mixed
    # episode in this batch call shares one topology, differing only in the
    # sampled W_l/b_l weight values.
    activations = [random.choice(_MLP_MIX_ACTIVATIONS) for _ in range(L)]

    x_mixed = x
    for act_name in activations:
        # 1/sqrt(d) fan-in scaling keeps the pre-activation roughly variance-
        # preserving (same purpose as Xavier/He init) -- an empirically-tuned
        # default rather than an analytically-guaranteed bound; validated by
        # test_mlp_mixing_goldilocks_and_psd in tests/test_data.py.
        W_l = torch.randn(B, d, d, device=device) * (w_std / math.sqrt(d))
        b_l = torch.randn(B, 1, d, device=device) * w_std
        x_mixed = torch.einsum("btd,bde->bte", x_mixed, W_l) + b_l
        x_mixed = _apply_mlp_activation(x_mixed, act_name)

    gate = (torch.rand(B, device=device) < mixing_prob)[:, None, None]  # (B,1,1)
    # (B,1,1) is required for correct broadcast against (B,T,d); a bare (B,)
    # shape misaligns on the trailing (T, d) dims instead of the batch dim.
    return torch.where(gate, x_mixed, x)


@torch.no_grad()
def generate_gp_batch(
    cfg, B: int, device: str = "cpu", *, return_kernel_metadata: bool = False
) -> List[Dict[str, Tensor]]:
    """Generate B GP episodes in a single vectorised call.

    All B episodes share one kernel type, one (P, N) size, and one set of
    active_dims/k (all sampled once per call) but have independent
    hyperparameters and feature draws — gpytorch's `batch_shape=[B]` kernels
    draw B independent hyperparameter sets in one call (see
    _sample_episode_kernel), and a batch_shape=[B] GaussianLikelihood
    (_build_likelihood) draws B independent noise values the same way.
    Sampling and GP-posterior conditioning both go through gpytorch's own
    MultivariateNormal/ExactGP machinery (see _GeneratorGP and the
    max_cholesky_size discussion in the module docstring) rather than
    hand-rolled Gram-matrix + Cholesky code, evaluated once for all B
    episodes at once. This removes the Python-loop overhead of B separate
    generate_gp_task calls and enables GPU or CPU-SIMD acceleration for the
    linear-algebra steps.

    The returned dicts have the same schema as the episodes saved by
    generate_pit_dataset.py (no kernel metadata), unless
    return_kernel_metadata=True (see Args) — used by generate_gp_task, which
    delegates here with B=1.

    If cfg.seed is set, it seeds python/numpy/torch RNGs, making the
    kernel/shape choice (kernel_name, P, N, k), hyperparameters (l, nugget,
    alpha2, ...), feature sampling/warp, and y sampling all reproducible.
    Note that calling this repeatedly with the same cfg.seed (e.g. once per
    shard in generate_pit_dataset.py) restarts every RNG at the same point
    every call — vary cfg.seed per call (e.g. `cfg.seed + shard_idx`) if you
    need distinct shards.

    Args:
        cfg    : Hydra config (same as generate_gp_task).
        B      : number of episodes to generate in this batch.
        device : torch device string ("cpu" or "cuda").
        return_kernel_metadata: if True, also pack each episode's kernel
            name, hyperparameters (l, alpha2, nugget, period, rq_alpha,
            l_b, alpha2_b, period_b, rq_alpha_b), kernel_feature_indices, and
            the ephemeral _L_ff/_alpha Cholesky factors — the schema
            generate_gp_task/pit.py::gp_analytical_pit/diag_kernels.py need.
            Off by default so the production shard schema
            (generate_pit_dataset.py) is unaffected.

    Returns:
        list of B episode dicts ready for torch.save.
    """
    seed = getattr(cfg, "seed", None)
    if seed is not None:
        _seed_everything(seed)

    d = _sample_d_features(cfg)

    # --- Shared settings for this batch ---
    # systematic_composition (CauKer-style, see _sample_kernel_chain_structure)
    # bypasses cfg.data.kernel/kernels entirely and samples a fresh
    # variable-length kernel chain instead — resolved here (before the
    # kernel_cols/k decision below) since chain components are always drawn
    # from _COMPOSABLE_KERNELS, so _kernel_needs_scalar_input still applies
    # and the "dot_product" branch below is simply never taken.
    systematic = bool(getattr(cfg.data, "systematic_composition", False))
    if systematic:
        chain_names, chain_ops, kernel_name = _sample_kernel_chain_structure(cfg)
    else:
        kernel_name = _resolve_kernel_name(cfg)
    P = random.randint(cfg.data.P_min, cfg.data.P_max)
    N = random.randint(cfg.data.N_min, cfg.data.N_max)
    T = P + N
    batch_shape = torch.Size([B])

    # active_dims (and hence k) is sampled once per batch call and shared by
    # all B episodes — same granularity as kernel_name/P/N above. gpytorch's
    # active_dims kernel kwarg is a single fixed column spec per Kernel
    # instance, so it can't vary per-episode within one batched kernel call
    # the way the old per-row torch.gather selection did. Note: when
    # apply_mlp_feature_mixing is enabled, every output column is a dense mix
    # of all d input columns, so "inactive" columns selected here are no
    # longer purely irrelevant noise — see apply_mlp_feature_mixing's docstring.
    # "periodic" (bare or as a composite/chain component) is also capped to
    # k=1: it never decays with r, so at k>1 the period becomes unrecoverable
    # from a finite point cloud (aliasing) well before k=3-4.
    if _kernel_needs_scalar_input(kernel_name) or "periodic" in kernel_name:
        kernel_cols = [random.randint(0, d - 1)]
    elif kernel_name == "dot_product":
        # Every dot product can draw on all d columns (no lengthscale to
        # dilute with irrelevant dims, unlike rbf/matern32/rational_quadratic).
        kernel_cols = None
    else:
        kernel_cols = _sample_active_dims(d, cfg)
    k = d if kernel_cols is None else len(kernel_cols)

    # --- Per-episode hyperparameters + noise (B independent draws in one call) ---
    if systematic:
        kernel_obj, component_params = _build_kernel_chain(
            cfg, chain_names, chain_ops, k, B, device, active_dims=kernel_cols
        )
        # Legacy zero-sentinel schema (see _sample_episode_kernel's docstring)
        # is kept populated so the rest of this function — nugget sampling,
        # GP prior/posterior machinery, metadata packing loop below — is
        # untouched regardless of mode; the real per-component values live in
        # component_params instead (see the return_kernel_metadata block).
        params = {
            key: torch.zeros(B, device=device)
            for key in ("l", "alpha2", "period", "rq_alpha", "l_b", "alpha2_b", "period_b", "rq_alpha_b")
        }
    else:
        kernel_obj, params = _sample_episode_kernel(cfg, kernel_name, k, B, device, active_dims=kernel_cols)
    likelihood = _build_likelihood(cfg, kernel_name, B, device)
    nugget = likelihood.noise.reshape(B)  # "nugget" name kept for the saved-metadata schema

    # --- Features (B, T, d) ~ N(0, 1), warped, normalised per episode ---
    x_raw = torch.randn(B, T, d, device=device)
    x_raw = tabiclv2_warp_features(x_raw)
    x_raw = apply_mlp_feature_mixing(x_raw, cfg, device)
    x_norm = (x_raw - x_raw.mean(1, keepdim=True)) / x_raw.std(1, keepdim=True).clamp(min=1e-8)

    # HEBO+'s Gamma-distributed lengthscale is calibrated for x in [0,1]^k
    # (paper Appendix D), unlike this file's usual ~N(0,1)-standardised x_norm.
    # Safe to map every column (not just the active ones): kernel_obj only
    # ever reads its active_dims columns internally (gpytorch.kernels.Kernel
    # .__call__ index_selects them before forward).
    x_kernel_input = torch.special.ndtr(x_norm) if kernel_name == "hebo" else x_norm

    # --- Joint prior sample + noisy covariance (B, T, T), via gpytorch's own
    # GaussianLikelihood(MultivariateNormal) — replaces the old manual
    # `kernel_obj(...).to_dense() + nugget*eye` Gram-matrix assembly and
    # `_batched_cholesky(K_all) @ randn` sampling. max_cholesky_size is
    # forced high (see module docstring / _MAX_CHOLESKY) so this is an exact
    # Cholesky-based sample, not gpytorch's approximate CG/Lanczos fallback.
    with gpytorch.settings.max_cholesky_size(_MAX_CHOLESKY):
        prior_dist = gpytorch.distributions.MultivariateNormal(
            torch.zeros(B, T, device=device), kernel_obj(x_kernel_input)
        )
        noisy_dist = likelihood(prior_dist)
        y_all = noisy_dist.rsample()                 # (B, T)
        K_all = noisy_dist.covariance_matrix          # (B, T, T), nugget already on diagonal
    K_all = 0.5 * (K_all + K_all.permute(0, 2, 1))    # symmetrize float32 drift

    x_norm_train = x_norm[:, :P]   # (B, P, d)
    x_norm_test  = x_norm[:, P:]   # (B, N, d)
    y_train      = y_all[:,  :P]   # (B, P)
    y_test       = y_all[:,  P:]   # (B, N)

    # --- Sub-matrices of K_all (nugget already on diagonal) ---
    K_ff = K_all[:, :P, :P]   # (B, P, P)
    K_ss = K_all[:, P:, P:]   # (B, N, N)

    # --- LOO PIT always needs L_ff/alpha from K_ff (R&W Eq. 5.12), regardless
    # of which oracle drives the test-side R_star/mu_star/sigma_star below.
    # No clean gpytorch public API exposes diag(K_ff^-1) for an ExactGP, so
    # this stays hand-rolled (_batched_cholesky), just sourced from the
    # gpytorch-native K_all above instead of a separately hand-added nugget.
    L_ff  = _batched_cholesky(K_ff)
    alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L_ff).squeeze(-1)          # (B, P)

    oracle_mode = getattr(cfg.data, "oracle_mode", "posterior")
    if oracle_mode == "prior":
        # Prior oracle: ignore training conditioning — R_star reflects the raw
        # kernel structure among test points; mu_star is the GP prior mean (0).
        # No conditioning needed, so this branch never touches _GeneratorGP.
        mu_star    = torch.zeros(B, N, device=device)
        Sigma_star = K_ss
    elif oracle_mode == "posterior":
        # Posterior oracle: condition on (x_train, y_train) via gpytorch's own
        # exact-inference ExactGP.__call__ instead of the hand-rolled
        # K_ss - K_sf K_ff^-1 K_fs formula gp_posterior used — same likelihood
        # object as the joint sample above, so the noise model matches.
        # fast_pred_var(False) disables gpytorch's LOVE variance shortcut (a
        # separate approximation from the Cholesky-size one).
        #
        # Done in float64: the Schur-complement subtraction K_ss - V^T V that
        # ExactGP performs internally is a cancellation between two close
        # quantities, and in float32 this measurably breaks PSD-ness for
        # composite kernels combining a heavy-tailed component (e.g.
        # rational_quadratic, whose small-alpha tail makes K_ff ill-
        # conditioned) with an oscillatory one (cosine/periodic) — observed
        # min eigenvalues down to -1.1e-3 across repeated sampling. float64
        # brings the worst case to ~5e-13 (machine-epsilon noise), confirming
        # this is precision, not a genuine non-PSD kernel. kernel_obj/
        # likelihood are mutated in place by .double() (nn.Module convention)
        # but are not read again after this branch, so that's safe; casting
        # back to float32 keeps the returned schema consistent with every
        # other tensor in this function.
        x_kernel_train = x_kernel_input[:, :P]
        x_kernel_test  = x_kernel_input[:, P:]
        out_dtype = x_kernel_input.dtype
        with gpytorch.settings.max_cholesky_size(_MAX_CHOLESKY), gpytorch.settings.fast_pred_var(False):
            post_model = _GeneratorGP(
                x_kernel_train.double(), y_train.double(),
                likelihood.double(), kernel_obj.double(), batch_shape,
            ).eval()
            post = post_model(x_kernel_test.double())
            mu_star    = post.mean.to(out_dtype)               # (B, N)
            Sigma_star = post.covariance_matrix.to(out_dtype)  # (B, N, N)
    else:
        raise ValueError(f"Unknown data.oracle_mode '{oracle_mode}'; expected 'prior' or 'posterior'.")
    Sigma_star = 0.5 * (Sigma_star + Sigma_star.permute(0, 2, 1))

    # sigma_to_correlation (batched)
    var_diag   = Sigma_star.diagonal(dim1=1, dim2=2).clamp(min=1e-10)              # (B, N)
    sigma_star = var_diag.sqrt()
    inv_s      = var_diag.rsqrt()
    R_star     = Sigma_star * inv_s.unsqueeze(1) * inv_s.unsqueeze(2)             # (B, N, N)
    d_diag     = R_star.diagonal(dim1=1, dim2=2).clamp(min=1e-10).sqrt()
    R_star     = R_star / (d_diag.unsqueeze(1) * d_diag.unsqueeze(2))

    # --- Prior correlation among the test points -------------------------------
    # K_ss is the joint-prior test block (nugget already on the diagonal) that
    # y_test was actually drawn from (noisy_dist above). Its correlation is the
    # "sampling prior": identical to R_star when oracle_mode="prior", but a
    # distinct, informative reference under oracle_mode="posterior" — it shows the
    # raw kernel structure before conditioning on the training points. Stored per
    # episode so downstream plots can compare prior vs oracle vs prediction.
    # Same batched D^{-1/2} K D^{-1/2} normalization used for R_star just above.
    prior_var  = K_ss.diagonal(dim1=1, dim2=2).clamp(min=1e-10)                   # (B, N)
    prior_inv  = prior_var.rsqrt()
    R_prior    = K_ss * prior_inv.unsqueeze(1) * prior_inv.unsqueeze(2)           # (B, N, N)
    pd_diag    = R_prior.diagonal(dim1=1, dim2=2).clamp(min=1e-10).sqrt()
    R_prior    = R_prior / (pd_diag.unsqueeze(1) * pd_diag.unsqueeze(2))

    # --- LOO PIT for z_train (R&W Eq. 5.12, batched) ---
    # diag(K_ff^{-1}) = column-squared-norm of L_ff^{-1}
    eye_P      = torch.eye(P, device=device)
    L_inv      = torch.linalg.solve_triangular(
        L_ff, eye_P.unsqueeze(0).expand(B, -1, -1), upper=False
    )                                                                               # (B, P, P)
    K_inv_diag = (L_inv ** 2).sum(dim=1).clamp(min=1e-12)                         # (B, P)
    z_train    = alpha * K_inv_diag.rsqrt()                                       # (B, P)

    # --- Posterior PIT for z_test ---
    sig_c        = sigma_star.clamp(min=1e-8)
    z_test       = (y_test - mu_star) / sig_c                                      # (B, N)
    log_pdf_test = (
        -0.5 * math.log(2.0 * math.pi) - sig_c.log() - 0.5 * z_test ** 2
    )                                                                               # (B, N)

    # LOO residuals are N(0,1) by construction (R&W Eq. 5.12); no empirical
    # rescaling needed.  Filter degenerate episodes instead.
    z_std = z_train.std(dim=1)
    degen = (z_std < 0.1) | (z_std > 3.0)
    if degen.any():
        warnings.warn(
            f"generate_gp_batch: {int(degen.sum())}/{B} episodes have degenerate LOO z.",
            RuntimeWarning,
        )

    # Reconstruct full posterior covariance (for Y-space oracle)
    Sigma_full = R_star * sigma_star.unsqueeze(1) * sigma_star.unsqueeze(2)       # (B, N, N)

    # --- Pack into list of dicts (single D→H transfer) ---
    tensors = {
        "x_norm_train": x_norm_train.cpu(),
        "x_norm_test":  x_norm_test.cpu(),
        "y_train":      y_train.cpu(),
        "y_test":       y_test.cpu(),
        "z_train":      z_train.cpu(),
        "z_test":       z_test.cpu(),
        "log_pdf_test": log_pdf_test.cpu(),
        "R_star":       R_star.cpu(),
        "R_prior":      R_prior.cpu(),
        "Sigma_star":   Sigma_full.cpu(),
        "mu_star":      mu_star.cpu(),
        "sigma_star":   sigma_star.cpu(),
    }
    n_tr = torch.tensor(P)
    n_te = torch.tensor(N)
    extra: Dict[str, object] = {"n_train": n_tr, "n_test": n_te}

    if return_kernel_metadata:
        # Per-episode (sliceable via val[b]) hyperparameters/factors, plus
        # the batch-shared kernel name and active_dims — the schema
        # generate_gp_task / pit.py::gp_analytical_pit / diag_kernels.py need.
        for key in ("l", "alpha2", "period", "rq_alpha", "l_b", "alpha2_b", "period_b", "rq_alpha_b"):
            tensors[key] = params[key].cpu()
        tensors["nugget"] = nugget.cpu()
        tensors["_L_ff"] = L_ff
        tensors["_alpha"] = alpha
        extra["kernel"] = kernel_name
        extra["kernel_feature_indices"] = torch.tensor(
            kernel_cols if kernel_cols is not None else list(range(d)), dtype=torch.long
        )

    episodes = [
        {key: val[b] for key, val in tensors.items()} | extra
        for b in range(B)
    ]

    if return_kernel_metadata and systematic:
        # Systematic-composition chains have a variable component count, so
        # their per-component hyperparameters don't fit the legacy flat
        # l/alpha2/l_b/alpha2_b schema populated with zero-sentinels above.
        # kernel_components/kernel_ops are shared across the batch (like
        # extra["kernel"] already is); kernel_component_params is a plain
        # per-episode Python list (not a stacked tensor) so heterogeneous
        # ARD-vector-vs-scalar "l" shapes across components don't need
        # padding. Not reconstructible via build_kernel_fn — see module
        # docstring's "Systematic composition" section.
        for b in range(B):
            episodes[b]["kernel_components"] = chain_names
            episodes[b]["kernel_ops"] = chain_ops
            episodes[b]["kernel_component_params"] = [
                {pk: pv[b].cpu() for pk, pv in comp.items()} for comp in component_params
            ]

    return episodes
