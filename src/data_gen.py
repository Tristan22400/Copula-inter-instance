"""
data_gen.py — Stage A: GP task generation for inter-instance copula.

Each task samples a random GP with a configurable PSD kernel, draws P+N
instances, normalises features over the full P+N set, samples targets jointly
from the GP, computes the analytical correlation matrix R* at the test points
from the raw GP prior (cfg.data.oracle_mode == "prior", the only supported
mode), and saves all required tensors.

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
  polynomial          — Polynomial: k(x1,x2) = alpha2 * (x1ᵀx2 + c)^d, via
                         gpytorch.kernels.PolynomialKernel wrapped in
                         ScaleKernel. c (the offset) ~
                         Gamma(poly_offset_gamma_concentration,
                         poly_offset_gamma_rate) and is stored in the "l"
                         schema slot — the same reuse convention cosine's
                         period_length already relies on (see
                         _kernel_prior_spec), since polynomial has no
                         lengthscale either. d (the integer power/degree) ~
                         Uniform{poly_power_min, ..., poly_power_max}
                         (default 2..4), sampled ONCE per generate_gp_batch
                         call — same granularity as kernel_name/P/N/
                         active_dims below, NOT per-episode like l/alpha2:
                         gpytorch.kernels.PolynomialKernel raises if given
                         more than one distinct power value, so every
                         episode in one batch call shares the same degree.
                         Saved/reconstructed via the new "power"/"power_b"
                         schema keys (same 0.0-sentinel convention as
                         period/rq_alpha — see build_kernel_fn).

ARD (cfg.data.ard)
-------------------
  When cfg.data.ard is True, rbf/matern32/periodic/rational_quadratic sample
  one independent lengthscale per active kernel dimension (ard_num_dims=k)
  instead of one isotropic scalar shared across all k dims. periodic's
  period also becomes per-dimension (gpytorch.kernels.PeriodicKernel ties
  period_length's ard_num_dims to the same kwarg as lengthscale). Default
  False (isotropic), preserving prior dataset-generation behaviour. Not
  possible for "cosine": gpytorch's CosineKernel hardcodes period_length to
  a single scalar regardless of ard_num_dims — no per-dimension formula
  exists. Not applicable to "dot_product" (no lengthscale). See
  _ARD_ELIGIBLE_KERNELS. "periodic" is additionally always capped to k=1
  active dims (independent of this flag) — see generate_gp_batch's
  kernel_cols selection.

  cfg.data.isotropic_ratio (default 0.0): even when a kernel would otherwise
  be ARD (cfg.data.ard=True for an ARD-eligible kernel), each episode
  independently has probability isotropic_ratio of
  having its lengthscale (and periodic's period) collapsed to one shared
  value across all active dims instead of one independent value per dim —
  i.e. an isotropic kernel in effect, still stored in the ARD-shaped (k,)
  tensor (so "l"/"period" numel doesn't change, only whether the k values
  are equal). A no-op when the kernel isn't ARD in the first place. See
  _build_scaled_kernel.

Composite kernels ("A+B" / "A*B")
---------------------------------
  Sums and products of PSD kernels are PSD, so every pair drawn from
  _COMPOSABLE_KERNELS (every base kernel, including dot_product) is
  auto-registered under both operators via gpytorch's `+`/`*` kernel
  composition, e.g. "rbf+periodic" (locally periodic: smooth decay times
  exact periodicity), "matern32*cosine" (spectral windowing), or
  "dot_product+rbf" (linear trend plus smooth deviation — dot_product has no
  lengthscale, so it contributes only its LinearKernel term, and always over
  every feature column regardless of the other component's active_dims
  subset — see _build_kernel_component's docstring for why that matters).
  See COMPOSITE_KERNELS for the full list. cfg.data.ard applies independently
  to each ARD-eligible component of a composite. cfg.data.composite_exclude_kernels
  prunes elementary kernels from the systematic-composition sampling pool at
  run time (see below) without touching _COMPOSABLE_KERNELS itself.

  Systematic composition (cfg.data.systematic_composition, CauKer-style —
  github.com/ShifengXIE/CauKer): an alternative, opt-in generative mode that
  samples a random chain length M ~ round(LogNormal(
  composite_num_kernels_lognormal_loc, _scale)), clipped to
  [composite_num_kernels_min, composite_num_kernels_max], draws M elementary
  kernels with replacement
  from _COMPOSABLE_KERNELS (minus cfg.data.composite_exclude_kernels), and
  combines them left-to-right with independently-sampled +/* operators (see
  _sample_kernel_chain_structure / _build_kernel_chain), instead of the
  static enumerated 2-way COMPOSITE_KERNELS list. Produces chain names like
  "rbf+cosine*periodic" that are NOT registered in ALL_KERNELS/
  KERNEL_REGISTRY (unbounded cardinality) and are not reconstructible via
  build_kernel_fn — see generate_gp_batch's return_kernel_metadata handling
  for the separate kernel_components/kernel_ops/kernel_component_params
  schema this mode uses instead of the flat l/alpha2/l_b/alpha2_b keys.

Sign modulation (cfg.data.sign_modulation_component_prob / _outer_prob)
-------------------------------------------------------------------------
  An optional Schur-product wrapper (SignModulatedKernel) that injects
  negative pairwise correlation into R_star without any new positivity
  argument: K'(x1, x2) = K(x1, x2) * s(x1) * s(x2), where
  s(x) = tanh(a * (w.x[active]+b)) in [-1, 1] is a smooth soft-sign split of
  the wrapped kernel's own active-column subspace along a random affine
  hyperplane, one independent draw per episode (w ~ N(0, I_k)/sqrt(k) so the
  raw margin z = w.x+b has O(1) scale regardless of k, b ~ N(0, 1), and a
  positive sharpness a ~ LogNormal(sign_modulation_sharpness_lognormal_loc/
  _scale) controlling how closely s approximates a hard sign flip). PSD
  holds for ANY real-valued s(x), not just +-1: the outer product s s^T is
  always rank-1 PSD, and an elementwise product of two PSD matrices is PSD
  (Schur product theorem) — this was true before the tanh replacement too,
  and remains the whole positivity argument.

  This replaces an earlier bare torch.sign(w.x+b) (a hard +-1 flip): a
  fresh (w, b) is drawn per episode with no cross-episode transfer, so the
  model has to infer the hyperplane from the correlation pattern in y alone
  and extrapolate it to unseen query points — under a hard sign, every point
  strictly on one side of the (a priori unknown) hyperplane is
  indistinguishable from every other point on that side, and the boundary
  carries a discontinuous jump with zero local gradient anywhere except
  exactly on it, which empirically left the model unable to learn the
  correlation sign at all (near-chance-or-below cross-hyperplane sign
  agreement, flat across context size and training step). tanh(a*z) keeps
  the same random-hyperplane structure but turns that jump into a smooth
  ramp of width ~1/a in raw z units, giving the model a local gradient to
  climb near the boundary while a controls how much of that softening
  bleeds into the bulk of the distribution (z's spread is ~sqrt(2)
  regardless of k, given the w normalisation above): large a recovers the
  original hard-sign behaviour (and its learnability problem) almost
  everywhere except a shrinking boundary strip; small a smooths the
  transition over a wide region at the cost of attenuating correlation
  magnitude there too. The default LogNormal (median a=3) sits with most of
  its mass already near-saturated (tanh(3*1) ~= 0.995) at one z-std out,
  so the softening is concentrated near the boundary rather than smeared
  across the whole distribution.

  One consequence of replacing sign() with tanh(): the hard-sign version was
  exactly diagonal-invariant (s(x)^2 == 1 a.e., so K'(x,x) == K(x,x)); tanh
  breaks this (s(x)^2 < 1 away from saturation), so K'(x,x) = K(x,x)*s(x)^2
  <= K(x,x) — points near the hyperplane get a (mild, a-dependent) marginal
  variance shrinkage in addition to the correlation-sign/magnitude effect.

  Two independently Bernoulli-per-batch-call-gated injection points (same
  granularity as mlp_mixing_enabled/mlp_mixing_prob — if the coin flip
  fires for a given generate_gp_batch call, every episode in that call gets
  its own independent (w, b, a), same "shared gate, independent draw"
  convention used throughout this file):
    - cfg.data.sign_modulation_component_prob: per elementary component,
      wired into the shared _build_kernel_component choke point — covers
      bare kernels, both sides of a static "A+B"/"A*B" composite (via
      _sample_episode_kernel), and every link of a systematic_composition
      chain (via _build_kernel_chain), independently per component, with no
      extra plumbing needed at any of those three call sites.
    - cfg.data.sign_modulation_outer_prob: applied once more, independently,
      to the fully composed kernel (whichever of the three modes above
      produced it) — see the end of _sample_episode_kernel / the end of
      _build_kernel_chain.
  Both default to 0.0 (off), so existing datasets/behaviour are unaffected
  until explicitly turned on.

  Saved/reconstructed via new sign_applied[_b|_outer] (0.0/1.0 float
  sentinel — same "0 means N/A" convention dot_product's l=0 already uses)
  and sign_w[_b|_outer]/sign_b[_b|_outer]/sign_a[_b|_outer] schema keys (see
  generate_gp_batch's return_kernel_metadata handling and build_kernel_fn's
  signature), following the same flat-schema/zero-sentinel pattern as
  l/alpha2/period/rq_alpha/power. Systematic-composition chains instead
  carry their per-component sign fields inside each entry of
  kernel_component_params (same non-reconstructible-via-build_kernel_fn
  caveat as every other systematic-composition hyperparameter — see above).
  pit.py::gp_analytical_pit falls back to a very large `a` (numerically
  recovering the old hard sign()) when replaying sign_w/sign_b saved by a
  pre-tanh dataset that has no sign_a field, so older on-disk episodes still
  reconstruct their original (already-baked) z_train/z_test exactly.

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
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.utils.errors import NanError, NotPSDError
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
#       - generate_gp_batch / generate_gp_task's own sampling, done via
#         gpytorch's native GaussianLikelihood (_build_likelihood) rather
#         than hand-rolled Gram-matrix + noise math.
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
    across all k dims, following cfg.data.ard for
    rbf/matern32/periodic/rational_quadratic (see _kernel_prior_spec /
    _ARD_ELIGIBLE_KERNELS). "cosine" is never ARD — gpytorch.kernels.
    CosineKernel's period_length is a single scalar regardless of
    ard_num_dims (no per-dimension formula exists to opt into).
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
# into. "dot_product" has no lengthscale at all (see its docstring).
_ARD_ELIGIBLE_KERNELS = frozenset(
    {"rbf", "matern12", "matern32", "matern52", "periodic", "rational_quadratic"}
)


def _kernel_prior_spec(cfg, kernel_name: str) -> KernelPriorSpec:
    """Build the LogNormal/Gamma hyperprior spec for one base kernel family.

    Every numeric constant is overridable via cfg.data (getattr-defaulted,
    same convention the old l_min/l_max/alpha2_min/alpha2_max ranges used).
    """
    isotropic_ratio = float(getattr(cfg.data, "isotropic_ratio", 0.0))

    l_loc = float(getattr(cfg.data, "l_lognormal_loc", 0.0))
    l_scale = float(getattr(cfg.data, "l_lognormal_scale", 0.7))
    a_conc = float(getattr(cfg.data, "alpha2_gamma_concentration", 4.0))
    a_rate = float(getattr(cfg.data, "alpha2_gamma_rate", 3.0))
    ard = bool(getattr(cfg.data, "ard", False)) and kernel_name in _ARD_ELIGIBLE_KERNELS

    # Without a k-dependent shift, a k-dims-summed stationary kernel's squared
    # distance grows ~linearly in k (active kernel dims) for standardized iid
    # inputs, so a fixed-in-k lengthscale collapses R_star toward the identity
    # as k grows -- k=16-19 (reachable via this file's d_features/inactive_frac
    # priors) gives near-zero correlation regardless of the "interesting"
    # lengthscale draw, independent of ard (a single shared isotropic
    # lengthscale summed over k dims collapses the same way an ARD one does).
    # A prior version of this file had a full sqrt(k) shift (0.5*log(k)) here
    # and dropped it in f72a3d2 ("remove sqrt(k) lengthscale shift") because,
    # combined with this file's now much tighter HEBO+-derived nugget floor,
    # it pushed correlations toward a uniform/degenerate regime. 0.25*log(k)
    # (capped before the log so the rare d_features tail doesn't drag the
    # shift past what was validated) was tuned against the CURRENT nugget
    # floor and tests/test_dataset_corr_uniform.py's abs(mean)<0.30 bound:
    # full sqrt(k) still overshoots that bound (measured mean ~0.35 on the
    # systematic_composition mix), 0.25*log(k) does not (~0.24).
    k_exponent = float(getattr(cfg.data, "l_lognormal_k_exponent", 0.25))
    k_cap = float(getattr(cfg.data, "l_lognormal_k_cap", 15))

    def lengthscale_prior(k: int) -> LogNormalPrior:
        shift = k_exponent * math.log(max(min(k, k_cap), 1))
        return LogNormalPrior(l_loc + shift, l_scale)

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
    """Diagonal regulariser prior, shared by every kernel — defaults to the
    tuned "HEBO+" noise prior from the PFN4BO paper (github.com/automl/
    PFNs4BO, Appendix B.1), LogNormal(-4.63, 0.5), used as the default noise
    floor for all kernels here (not specific to any particular kernel
    family)."""
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


class SignModulatedKernel(gpytorch.kernels.Kernel):
    """Schur-product sign modulation: K'(x1, x2) = K(x1, x2) * s(x1) * s(x2),
    where s(x) = tanh(a * (w . x[active_cols] + b)) in [-1, +1] is a smooth
    soft-sign split of the (active-column subspace of the) input space along
    a random affine hyperplane, one independent (w, b, a) draw per episode
    (batch_shape=[B], mirroring ScaleKernel/_build_scaled_kernel above).

    PSD rationale: s(x1)*s(x2) is the outer product of a real-valued vector
    with itself, i.e. a rank-1 PSD matrix (s s^T) for ANY s (not just +-1),
    and the elementwise (Schur/Hadamard) product of two PSD matrices is PSD
    (Schur product theorem), so K' is PSD whenever K is -- no new positivity
    argument needed beyond what every existing kernel in this file already
    relies on, and none was needed for the tanh replacement either.

    `active_cols` intentionally reuses whatever column subset the wrapped
    base_kernel itself was built with (the caller passes the same
    `active_dims` list used for the kernel being wrapped -- see
    generate_gp_batch's `kernel_cols` and _build_kernel_component/
    _sample_episode_kernel/_build_kernel_chain below) rather than drawing a
    second, independent active-dims subset: the hyperplane should live in the
    same feature subspace the kernel actually sees, not an unrelated one.
    None means every column (matching gpytorch's own active_dims convention
    elsewhere in this file).

    Diagonal invariance (lost vs. the old hard sign()): torch.sign(z)^2 == 1
    a.e., so the old K'(x,x) == K(x,x) exactly; tanh(a*z)^2 < 1 away from
    saturation, so K'(x,x) = K(x,x)*s(x)^2 <= K(x,x) -- points near the
    hyperplane get a mild, a-dependent marginal-variance shrinkage on top of
    the correlation-sign effect. See the module docstring's "Sign
    modulation" section for the full tradeoff this tanh replaces a bare
    sign() to address (identifiability/learnability of a fresh per-episode
    hyperplane) and why a is drawn from a LogNormal prior rather than fixed.
    """

    def __init__(
        self,
        base_kernel: gpytorch.kernels.Kernel,
        w: Tensor,
        b: Tensor,
        a: Tensor,
        active_dims: Optional[List[int]] = None,
        **kwargs,
    ):
        # batch_shape is inferred from w's leading dim (B,), matching how
        # ScaleKernel infers it from outputscale's shape.
        super().__init__(batch_shape=torch.Size([w.shape[0]]), **kwargs)
        self.base_kernel = base_kernel
        self.register_buffer("w", w)   # (B, k)
        self.register_buffer("b", b)   # (B,)
        self.register_buffer("a", a)   # (B,) sharpness, > 0
        self.active_cols = list(active_dims) if active_dims is not None else None

    def _signs(self, x: Tensor) -> Tensor:
        """tanh(a * (w . x[..., active_cols] + b)), shape (..., n) for x of
        shape (..., n, d) -- same batch/broadcast convention gpytorch kernels
        use for their own forward() (x's leading dims are batch dims, its
        last two are (n, d)).

        w has shape (B, k), b/a have shape (B,) -- all three are unsqueezed
        with one extra axis right before their last dim (giving (B, 1, k) /
        (B, 1) / (B, 1)) so they broadcast against x_active's (..., n, k) /
        (..., n) from the right, the same way e.g. gpytorch's own
        outputscale (B,) broadcasts against a (B, n1, n2) covariance
        elsewhere in this file. Any further leading dims gpytorch's own
        kernel machinery adds (e.g. an extra singleton batch dim for some
        composition paths) broadcast for free via ordinary right-aligned
        torch broadcasting -- no manual padding needed for those.
        """
        cols = self.active_cols
        x_active = x[..., cols] if cols is not None else x
        w = self.w.unsqueeze(-2)   # (B, 1, k)
        b = self.b.unsqueeze(-1)   # (B, 1)
        a = self.a.unsqueeze(-1)   # (B, 1)
        z = (x_active * w).sum(-1) + b
        return torch.tanh(a * z)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        K = self.base_kernel(x1, x2, diag=diag, **params)
        K = K.to_dense() if hasattr(K, "to_dense") else K
        s1 = self._signs(x1)
        s2 = self._signs(x2)
        if diag:
            return K * s1 * s2
        return K * s1.unsqueeze(-1) * s2.unsqueeze(-2)


class _DenseComposedKernel(gpytorch.kernels.Kernel):
    """Combine two kernels via plain dense tensor +/*, INSTEAD of gpytorch's
    own Kernel.__add__/__mul__ (which builds an AdditiveKernel/ProductKernel
    that composes LinearOperators, not tensors) — the composite-chain
    counterpart of SignModulatedKernel.forward's `K.to_dense() if
    hasattr(K, "to_dense") else K` line just above, same rationale.

    Why this matters: gpytorch's LinearOperator.__add__ special-cases an
    operand that exposes a low-rank `.root` (exactly what LinearKernel's Gram
    matrix is -- x @ x.T is a RootLinearOperator, see LinearKernel.forward)
    by dispatching to add_low_rank, which EAGERLY computes a
    root/root-inverse decomposition (Cholesky, falling through to eigh if
    that fails) of the *other*, already-summed operand -- for the WHOLE
    batch, before _psd_safe_batch ever gets a chance to isolate and repair
    individual episodes. dot_product/polynomial (both LinearKernel/
    PolynomialKernel-backed, both finite-dimensional-feature-map kernels
    with rank <= d_features, often << T=P+N -- see composite_exclude_kernels'
    docstring in conf/data/gp_tasks.yaml) are exactly the kernels most likely
    to make that intermediate sum near-singular, so composite chains
    involving them hit this eager factorization constantly, raising
    NotPSDError (or, when the eigh fallback's LAPACK routine fails to
    converge, torch.linalg.LinAlgError) straight out of kernel evaluation --
    discarding the ENTIRE B-episode batch (see _generate_gp_batch_raw's
    K_all_raw construction comment) rather than the individual bad
    episode(s), by far the most expensive failure mode this pipeline has.

    Every caller of a composed kernel_obj in this file immediately calls
    .to_dense() on its output anyway (there's no downstream code that
    benefits from a lazy/structured LinearOperator), so there is no
    structure being thrown away by converting each side to dense before
    combining: plain torch.Tensor +/- can never raise a linear-algebra
    error, so this sidesteps add_low_rank entirely rather than merely
    handling its fallout. Recursive composition (chains longer than one
    +/* op) nests these, so only the innermost pair's kernels are ever
    "real" gpytorch kernels -- exactly mirroring how _build_kernel_chain
    already folds left-to-right with Python's own +/* before this class
    existed.
    """

    def __init__(self, kernel_a: gpytorch.kernels.Kernel, op: str, kernel_b: gpytorch.kernels.Kernel, **kwargs):
        super().__init__(**kwargs)
        assert op in ("+", "*"), f"op must be '+' or '*', got {op!r}"
        self.kernel_a = kernel_a
        self.op = op
        self.kernel_b = kernel_b

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        a = self.kernel_a(x1, x2, diag=diag, **params)
        b = self.kernel_b(x1, x2, diag=diag, **params)
        a = a.to_dense() if hasattr(a, "to_dense") else a
        b = b.to_dense() if hasattr(b, "to_dense") else b
        return a + b if self.op == "+" else a * b


def _sample_sign_modulation(
    cfg, k: int, B: int, device
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample one (w, b, a) hyperplane per episode: w ~ N(0, I_k)/sqrt(k) (so
    the raw margin z = w.x[active]+b has O(1) scale regardless of k -- the
    hard sign() this replaced was scale-invariant so w's magnitude never
    mattered, but tanh(a*z)'s effective sharpness is a*||w||, so w needs a
    fixed scale for `a`'s prior to mean the same thing across episodes with
    different active-dims counts k), b ~ N(0, 1), and a positive sharpness
    a ~ LogNormal(sign_modulation_sharpness_lognormal_loc/_scale) -- reused
    for both the per-component and post-composition SignModulatedKernel
    injection points (see the module docstring's "Sign modulation"
    section)."""
    w = torch.randn(B, k, device=device) / math.sqrt(max(k, 1))
    b = torch.randn(B, device=device)
    a_loc = float(getattr(cfg.data, "sign_modulation_sharpness_lognormal_loc", math.log(3.0)))
    a_scale = float(getattr(cfg.data, "sign_modulation_sharpness_lognormal_scale", 0.5))
    a = LogNormalPrior(a_loc, a_scale).sample(torch.Size([B])).to(device)
    return w, b, a


def _maybe_wrap_sign_modulated(
    cfg,
    kernel: gpytorch.kernels.Kernel,
    prob: float,
    k: int,
    B: int,
    device,
    active_dims: Optional[List[int]],
    param_suffix: str = "",
) -> tuple[gpytorch.kernels.Kernel, Dict[str, Tensor]]:
    """Bernoulli(prob)-per-call gate (same per-batch-call granularity as
    mlp_mixing_enabled/mlp_mixing_prob's own gate -- see
    apply_mlp_feature_mixing) deciding whether to wrap `kernel` in a
    SignModulatedKernel at all for this generate_gp_batch call. When gated
    on, EVERY episode in the batch gets its own independent (w, b, a) draw
    (the B-batched SignModulatedKernel itself), matching how every other
    batched hyperparameter in this file (l, alpha2, ...) is drawn once per
    call but independently per episode within it.

    Returns (possibly-wrapped kernel, params) where params has
    "sign_applied{suffix}" (0.0/1.0 float sentinel, same "0 means N/A"
    convention dot_product's l=0 already uses), "sign_w{suffix}" (B, k),
    "sign_b{suffix}" (B,) and "sign_a{suffix}" (B,) -- zero-filled/no-op when
    not applied, so the output schema always has these keys regardless of
    the coin flip.
    """
    if prob > 0.0 and random.random() < prob:
        w, b, a = _sample_sign_modulation(cfg, k, B, device)
        wrapped = SignModulatedKernel(kernel, w, b, a, active_dims=active_dims)
        params = {
            f"sign_applied{param_suffix}": torch.ones(B, device=device),
            f"sign_w{param_suffix}": w,
            f"sign_b{param_suffix}": b,
            f"sign_a{param_suffix}": a,
        }
        return wrapped, params
    params = {
        f"sign_applied{param_suffix}": torch.zeros(B, device=device),
        f"sign_w{param_suffix}": torch.zeros(B, max(k, 1), device=device),
        f"sign_b{param_suffix}": torch.zeros(B, device=device),
        f"sign_a{param_suffix}": torch.zeros(B, device=device),
    }
    return kernel, params


def _build_kernel_component(
    cfg, name: str, k: int, B: int, device, active_dims: Optional[List[int]] = None,
    d_total: Optional[int] = None,
) -> tuple[gpytorch.kernels.Kernel, Dict[str, Tensor]]:
    """Build one elementary (non-composite) kernel + its sampled hyperparameter
    dict — the unit _sample_episode_kernel calls once for a bare kernel or
    twice (component A, component B) for a composite.

    Also the shared choke point for the PER-COMPONENT sign-modulation
    injection point (cfg.data.sign_modulation_component_prob — see the
    module docstring's "Sign modulation" section and SignModulatedKernel):
    every branch below sits behind one `_maybe_wrap_sign_modulated` call
    right before it returns, so bare kernels, both components of a static
    "A+B"/"A*B" composite (called from _sample_episode_kernel), and every
    link of a systematic chain (called from _build_kernel_chain) are all
    covered with no extra plumbing at those call sites.

    d_total: total feature-column count d (generate_gp_batch's `d`), used
    ONLY to size the sign-modulation hyperplane for "dot_product" components,
    which ignore `active_dims`/`k` for the base kernel itself (see the
    active_dims paragraph below) but must still size `w` correctly — reusing
    `k` there would draw a hyperplane over the wrong (smaller) subspace.
    Defaults to `k` when omitted (every other kernel name uses `k` as-is).

    "dot_product" has no lengthscale, so it bypasses _kernel_prior_spec/
    _build_scaled_kernel entirely: a bare LinearKernel (no ScaleKernel
    wrapper — its `variance` already plays the alpha2 role, see
    dot_product_kernel's docstring) whose variance is sampled from the same
    alpha2 ~ Gamma(alpha2_gamma_concentration, alpha2_gamma_rate) prior every
    other kernel's outputscale uses. Every other kernel goes through
    _build_scaled_kernel (ScaleKernel-wrapped, real lengthscale prior).

    `active_dims` is deliberately IGNORED for "dot_product": unlike every
    stationary kernel here, its diagonal k(x,x) = alpha2 * x@x depends on the
    actual point (not just alpha2), so restricting it to a small column
    subset (e.g. k=1, forced when its composite partner is cosine/periodic)
    makes k(x,x)==0 a real, non-negligible event whenever that one column's
    per-episode-standardized value lands on ~0 for some point — which zeroes
    the WHOLE diagonal for a "*" (product) composite, breaking R_star's
    unit-diagonal invariant (empirically ~1% of episodes under forced MLP
    mixing for e.g. "matern32*dot_product" before this override). Always
    using every column (same as the bare "dot_product" kernel already did —
    see generate_gp_batch's kernel_cols selection) makes that coordinate-wise
    coincidence require ALL d columns to vanish simultaneously instead of
    just one, which the standalone kernel already relies on (0/3000 in an
    empirical sweep) and composites now share. gpytorch kernel `+`/`*`
    composition evaluates each side on the full-width input independently, so
    this doesn't require the other component to match active_dims.
    """
    sign_prob = float(getattr(cfg.data, "sign_modulation_component_prob", 0.0))

    if name == "dot_product":
        kernel = gpytorch.kernels.LinearKernel(batch_shape=torch.Size([B])).to(device)
        a_conc = float(getattr(cfg.data, "alpha2_gamma_concentration", 4.0))
        a_rate = float(getattr(cfg.data, "alpha2_gamma_rate", 3.0))
        a_sample = GammaPrior(a_conc, a_rate).sample(kernel.variance.shape).to(device)
        kernel.variance = a_sample
        params: Dict[str, Tensor] = {
            "l": torch.zeros(B, device=device),
            "alpha2": a_sample.reshape(B),
        }
        # dot_product ignores active_dims/k for the base kernel itself (see
        # this function's docstring) — its sign hyperplane must match, i.e.
        # span every column (d_total, defaulting to k), not the caller's
        # (possibly smaller) active_dims subset.
        kernel, sign_params = _maybe_wrap_sign_modulated(
            cfg, kernel, sign_prob, d_total if d_total is not None else k, B, device, active_dims=None
        )
        params.update(sign_params)
        return kernel, params
    if name == "polynomial":
        # power is a single Python int shared by every episode in this
        # batch/task call (gpytorch.kernels.PolynomialKernel raises if given
        # more than one distinct value) — sampled at the same granularity as
        # kernel_name/P/N/active_dims in generate_gp_batch, not per-episode
        # like l/alpha2 below. See the module docstring's "polynomial" entry.
        power_min = int(getattr(cfg.data, "poly_power_min", 2))
        power_max = int(getattr(cfg.data, "poly_power_max", 4))
        power = random.randint(power_min, power_max)
        kernel_kwargs: Dict = {"power": power, "batch_shape": torch.Size([B])}
        if active_dims is not None:
            kernel_kwargs["active_dims"] = active_dims
        base = gpytorch.kernels.PolynomialKernel(**kernel_kwargs).to(device)
        o_conc = float(getattr(cfg.data, "poly_offset_gamma_concentration", 2.0))
        o_rate = float(getattr(cfg.data, "poly_offset_gamma_rate", 1.0))
        o_sample = GammaPrior(o_conc, o_rate).sample(base.offset.shape).to(device)
        base.offset = o_sample

        scaled = gpytorch.kernels.ScaleKernel(base, batch_shape=torch.Size([B])).to(device)
        a_conc = float(getattr(cfg.data, "alpha2_gamma_concentration", 4.0))
        a_rate = float(getattr(cfg.data, "alpha2_gamma_rate", 3.0))
        a_sample = GammaPrior(a_conc, a_rate).sample(scaled.outputscale.shape).to(device)
        scaled.outputscale = a_sample

        params = {
            # Offset reuses the "l" schema slot (cosine's period_length
            # already does the same — see _kernel_prior_spec).
            "l": o_sample.reshape(B),
            "alpha2": a_sample.reshape(B),
            "power": torch.full((B,), float(power), device=device),
        }
        scaled, sign_params = _maybe_wrap_sign_modulated(
            cfg, scaled, sign_prob, k, B, device, active_dims=active_dims
        )
        params.update(sign_params)
        return scaled, params
    spec = _kernel_prior_spec(cfg, name)
    scaled, params = _build_scaled_kernel(name, spec, k, B, device, active_dims=active_dims)
    scaled, sign_params = _maybe_wrap_sign_modulated(
        cfg, scaled, sign_prob, k, B, device, active_dims=active_dims
    )
    params.update(sign_params)
    return scaled, params


def _sample_episode_kernel(
    cfg, kernel_name: str, k: int, B: int, device, active_dims: Optional[List[int]] = None,
    d_total: Optional[int] = None,
) -> tuple[gpytorch.kernels.Kernel, Dict[str, Tensor]]:
    """Sample B episodes' hyperparameters for kernel_name (base or "A+B"/"A*B"
    composite, either component of which may be "dot_product" — see
    _build_kernel_component) and return (gpytorch Kernel with
    batch_shape=[B], params dict).

    params keys match the output-dict schema (l, alpha2, period, rq_alpha,
    power, l_b, alpha2_b, period_b, rq_alpha_b, power_b, sign_applied,
    sign_w, sign_b, sign_applied_b, sign_w_b, sign_b_b, sign_applied_outer,
    sign_w_outer, sign_b_outer); not-applicable entries (including
    "dot_product"'s "l"/"l_b") are filled with a 0.0 sentinel (the
    convention pit.py::gp_analytical_pit relies on).

    active_dims: column indices (out of the caller's full d_features input)
    this kernel is active on — None means every column. Forwarded to
    gpytorch's own active_dims kwarg (see _build_scaled_kernel), so callers
    pass the full-width input straight through instead of pre-slicing it.

    Also the POST-COMPOSITION sign-modulation injection point
    (cfg.data.sign_modulation_outer_prob — see the module docstring's "Sign
    modulation" section): applied once more, independently of the
    per-component gate inside _build_kernel_component, to the fully composed
    kernel object (or the bare kernel, for a non-composite kernel_name).
    """
    d_total = d_total if d_total is not None else k
    composite = _parse_composite(kernel_name)
    if composite is None:
        kernel, params = _build_kernel_component(
            cfg, kernel_name, k, B, device, active_dims=active_dims, d_total=d_total
        )
    else:
        name_a, op, name_b = composite
        kernel_a, params_a = _build_kernel_component(
            cfg, name_a, k, B, device, active_dims=active_dims, d_total=d_total
        )
        kernel_b, params_b = _build_kernel_component(
            cfg, name_b, k, B, device, active_dims=active_dims, d_total=d_total
        )
        kernel = _DenseComposedKernel(kernel_a, op, kernel_b)
        params = dict(params_a)
        for key, val in params_b.items():
            params[f"{key}_b"] = val

    for key in ("period", "rq_alpha", "power", "l_b", "alpha2_b", "period_b", "rq_alpha_b", "power_b"):
        params.setdefault(key, torch.zeros(B, device=device))

    outer_prob = float(getattr(cfg.data, "sign_modulation_outer_prob", 0.0))
    kernel, outer_params = _maybe_wrap_sign_modulated(
        cfg, kernel, outer_prob, k, B, device, active_dims=active_dims, param_suffix="_outer"
    )
    params.update(outer_params)

    return kernel, params


def _wrap_concrete_sign_modulated(
    kernel: gpytorch.kernels.Kernel, sign_w, sign_b, sign_a, active_dims: Optional[List[int]]
) -> gpytorch.kernels.Kernel:
    """Wrap a non-batched, already-built concrete `kernel` in
    SignModulatedKernel using CONCRETE (already-known) sign_w/sign_b/sign_a
    values — the _build_concrete_kernel-side counterpart of
    _maybe_wrap_sign_modulated (which samples fresh w/b/a; this reconstructs
    from saved ones). No-op (returns `kernel` unchanged) when sign_w/sign_b
    are None (the "not applied" case — callers check the sign_applied*
    0.0/1.0 sentinel before calling this, same convention _optional_param
    callers use for period/rq_alpha/power elsewhere in this file).

    sign_a=None (sign_w/sign_b present) means a dataset saved before this
    sharpness parameter existed -- a very large sharpness is substituted so
    tanh(a*z) numerically recovers the hard sign() that dataset was actually
    generated with (see the module docstring's "Sign modulation" section and
    pit.py::gp_analytical_pit's `_sign_pair`, which is the caller that
    resolves this fallback).

    sign_w/sign_b/sign_a are reshaped to a (1, k)/(1,)/(1,) leading "batch"
    axis: SignModulatedKernel is written batched (mirroring ScaleKernel), and
    gpytorch kernels with batch_shape=[1] broadcast fine against the
    non-batched (n, d) X1/X2 build_kernel_fn's callers pass in — consistent
    with how every other concrete kernel built here has no explicit
    batch_shape either.
    """
    if sign_w is None or sign_b is None:
        return kernel
    w_t = sign_w if torch.is_tensor(sign_w) else torch.as_tensor(sign_w, dtype=torch.get_default_dtype())
    b_t = sign_b if torch.is_tensor(sign_b) else torch.as_tensor(sign_b, dtype=torch.get_default_dtype())
    if sign_a is None:
        a_t = torch.full_like(b_t, 1e6)
    else:
        a_t = sign_a if torch.is_tensor(sign_a) else torch.as_tensor(sign_a, dtype=torch.get_default_dtype())
    w_t = w_t.reshape(1, -1)
    b_t = b_t.reshape(1)
    a_t = a_t.reshape(1)
    return SignModulatedKernel(kernel, w_t, b_t, a_t, active_dims=active_dims)


def _build_concrete_kernel(
    name: str, l, alpha2, *, period=None, rq_alpha=None, power=None, active_dims: Optional[List[int]] = None,
    sign_w=None, sign_b=None, sign_a=None,
) -> gpytorch.kernels.Kernel:
    """Construct a non-batched gpytorch Kernel with CONCRETE hyperparameter
    values assigned — reconstruction (given known values), not sampling.
    Used by build_kernel_fn.

    "dot_product" returns the bare LinearKernel (no ScaleKernel wrapper):
    its `variance` already plays the role `alpha2` plays for every other
    kernel, so wrapping it would just be a second, redundant alpha2. "l" is
    ignored — no lengthscale, geometry comes entirely from the feature space.
    `active_dims` is likewise ignored for "dot_product" — see
    _build_kernel_component's docstring for why (always full columns,
    matching how it was actually sampled, including as a composite
    component).

    "polynomial" reads its offset out of "l" (same reuse convention cosine's
    period_length uses — see _build_kernel_component) and its integer
    power/degree out of `power` (defaults to 2 if not given, matching
    gpytorch.kernels.PolynomialKernel's own default).

    active_dims: column indices this kernel reads out of the caller's
    full-width input (gpytorch's own kwarg — see _build_scaled_kernel);
    None means every column.

    sign_w/sign_b/sign_a: CONCRETE per-component sign-modulation hyperplane
    values (see SignModulatedKernel / _maybe_wrap_sign_modulated), applied
    via _wrap_concrete_sign_modulated right before returning. None (the
    default) means "not applied" -- a no-op, matching the sign_applied 0.0
    sentinel convention build_kernel_fn's caller checks. Ignored (forced to
    active_dims=None) for "dot_product", same override the sampling-time
    _build_kernel_component uses -- the hyperplane must span every column,
    matching how it was actually sampled.
    """
    if name == "dot_product":
        kernel = gpytorch.kernels.LinearKernel()
        kernel.variance = torch.as_tensor(alpha2, dtype=torch.get_default_dtype()).reshape(kernel.variance.shape)
        return _wrap_concrete_sign_modulated(kernel, sign_w, sign_b, sign_a, active_dims=None)

    if name == "polynomial":
        power_int = int(round(float(power))) if power is not None else 2
        kernel_kwargs = {"power": power_int}
        if active_dims is not None:
            kernel_kwargs["active_dims"] = active_dims
        base = gpytorch.kernels.PolynomialKernel(**kernel_kwargs)
        offset_t = l if torch.is_tensor(l) else torch.as_tensor(l, dtype=torch.get_default_dtype())
        base.offset = offset_t.reshape(base.offset.shape)
        scale = gpytorch.kernels.ScaleKernel(base)
        scale.outputscale = torch.as_tensor(alpha2, dtype=torch.get_default_dtype()).reshape(scale.outputscale.shape)
        return _wrap_concrete_sign_modulated(scale, sign_w, sign_b, sign_a, active_dims=active_dims)

    l_t = l if torch.is_tensor(l) else torch.as_tensor(l, dtype=torch.get_default_dtype())
    # l having more than one element means this episode was generated ARD
    # (cfg.data.ard=True for rbf/matern32/periodic/rational_quadratic) —
    # gpytorch needs ard_num_dims at construction time to size .lengthscale
    # (and, for "periodic", .period_length — see the reshape below)
    # correctly before values can be assigned into it.
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
    return _wrap_concrete_sign_modulated(scale, sign_w, sign_b, sign_a, active_dims=active_dims)


def build_kernel_fn(
    kernel_name: str,
    l,
    alpha2,
    *,
    period: Optional[float | Tensor] = None,
    rq_alpha: Optional[float] = None,
    power: Optional[float | int] = None,
    l_b=None,
    alpha2_b=None,
    period_b: Optional[float | Tensor] = None,
    rq_alpha_b: Optional[float] = None,
    power_b: Optional[float | int] = None,
    active_dims: Optional[List[int]] = None,
    sign_w=None,
    sign_b=None,
    sign_a=None,
    sign_w_b=None,
    sign_b_b=None,
    sign_a_b=None,
    sign_w_outer=None,
    sign_b_outer=None,
    sign_a_outer=None,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Return a kernel(X1, X2) -> K callable with hyperparameters baked in.

    l_b/alpha2_b/period_b/rq_alpha_b/power_b are the second component's
    hyperparameters for composite ("A+B" / "A*B") kernels. l/l_b/period/
    period_b may be an ARD per-dimension vector (Tensor) instead of a scalar
    when the episode was generated with cfg.data.ard=True. power/power_b is
    "polynomial"'s integer degree (see _build_concrete_kernel); ignored for
    every other kernel name.

    active_dims: column indices this kernel is active on (both components of
    a composite share the same active columns — see generate_gp_task/
    generate_gp_batch, which sample one column subset per task/batch). The
    caller passes its full-width X1/X2 straight through; gpytorch's own
    active_dims kwarg selects the columns internally. None means every
    column (e.g. "dot_product" tasks that draw on all d_features).

    sign_w/sign_b/sign_a (component A) and sign_w_b/sign_b_b/sign_a_b
    (component B, composite only) are the PER-COMPONENT sign-modulation
    hyperplanes (see SignModulatedKernel / cfg.data.sign_modulation_component_prob);
    None (the default) means "not applied" for that component -- callers
    should pass None (not the saved 0.0-filled tensor) whenever that
    episode's sign_applied[_b] sentinel is 0.0, same pattern _optional_param
    already uses for period/rq_alpha/power/l_b (see
    pit.py::gp_analytical_pit). sign_w_outer/sign_b_outer/sign_a_outer is the
    POST-COMPOSITION hyperplane (cfg.data. sign_modulation_outer_prob),
    applied LAST -- after A/B are built and combined -- wrapping the whole
    (possibly composite) kernel, mirroring the order
    _sample_episode_kernel/_build_kernel_chain apply it at generation time
    (per-component first, then once more on the composed result). sign_a[_b|
    _outer]=None while its paired sign_w[_b|_outer] is not None falls back to
    a very large sharpness (recovering the pre-tanh hard sign()) -- see
    _wrap_concrete_sign_modulated's docstring.
    """
    composite = _parse_composite(kernel_name)
    if composite is None:
        kernel = _build_concrete_kernel(
            kernel_name, l, alpha2, period=period, rq_alpha=rq_alpha, power=power, active_dims=active_dims,
            sign_w=sign_w, sign_b=sign_b, sign_a=sign_a,
        )
    else:
        name_a, op, name_b = composite
        kernel_a = _build_concrete_kernel(
            name_a, l, alpha2, period=period, rq_alpha=rq_alpha, power=power, active_dims=active_dims,
            sign_w=sign_w, sign_b=sign_b, sign_a=sign_a,
        )
        kernel_b = _build_concrete_kernel(
            name_b, l_b, alpha2_b, period=period_b, rq_alpha=rq_alpha_b, power=power_b, active_dims=active_dims,
            sign_w=sign_w_b, sign_b=sign_b_b, sign_a=sign_a_b,
        )
        kernel = _DenseComposedKernel(kernel_a, op, kernel_b)

    kernel = _wrap_concrete_sign_modulated(kernel, sign_w_outer, sign_b_outer, sign_a_outer, active_dims=active_dims)

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
# still differentiable). eval/baselines/classical.py's GP-MLE fit needs
# exactly that backprop, so it sidesteps the issue entirely by constructing
# gpytorch kernel objects (RBFKernel/MaternKernel/PeriodicKernel/RQKernel/
# LinearKernel) directly and optimizing their own native trainable
# parameters, instead of importing these free functions.
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


def polynomial_kernel(X1: Tensor, X2: Tensor, *, l, alpha2, power: float = 2.0, **_) -> Tensor:
    """Polynomial: alpha2 * (x1ᵀx2 + c)^d, via gpytorch.kernels.PolynomialKernel.

    "l" holds the offset c (same schema-slot reuse cosine's period_length
    already uses); `power` is the integer degree d.
    """
    return build_kernel_fn("polynomial", l, alpha2, power=power)(X1, X2)


KERNEL_REGISTRY: Dict[str, Callable[..., Tensor]] = {
    "rbf": rbf_kernel,
    "matern12": matern12_kernel,
    "matern32": matern32_kernel,
    "matern52": matern52_kernel,
    "cosine": cosine_kernel,
    "periodic": periodic_kernel,
    "rational_quadratic": rational_quadratic_kernel,
    "dot_product": dot_product_kernel,
    "polynomial": polynomial_kernel,
}


# ---------------------------------------------------------------------------
# Composite kernels: sum / product of two base kernels
# ---------------------------------------------------------------------------
# Sums and products of PSD kernels are PSD, so "rbf+periodic" (locally
# periodic — smooth decay times exact periodicity) or "matern32*cosine"
# (spectral windowing) are valid kernels without any new math. Includes every
# base kernel in KERNEL_REGISTRY — "dot_product" has no lengthscale (see its
# docstring) but _build_kernel_component/_build_concrete_kernel both special-
# case it (bare LinearKernel, `l`/`l_b` ignored) so it composes fine, e.g.
# "dot_product+rbf" (linear trend plus smooth deviation). Per-run pruning
# (e.g. dropping periodic/cosine) goes through cfg.data.composite_exclude_kernels
# instead of hardcoding a subset here — see _sample_kernel_chain_structure.
_COMPOSABLE_KERNELS: List[str] = [
    "rbf", "matern12", "matern32", "matern52", "cosine", "periodic",
    "rational_quadratic", "dot_product", "polynomial",
]

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
    power: Optional[float] = None,
    power_b: Optional[float] = None,
    **_,
) -> Tensor:
    """Evaluate a registered "A+B" / "A*B" composite kernel (KERNEL_REGISTRY
    dispatch convention) by delegating to build_kernel_fn."""
    fn = build_kernel_fn(
        kernel_name, l, alpha2, period=period, rq_alpha=rq_alpha, power=power,
        l_b=l_b, alpha2_b=alpha2_b, period_b=period_b, rq_alpha_b=rq_alpha_b, power_b=power_b,
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


def _weights_for_pool(pool: List[str], kernel_weights: Optional[Tensor]) -> Optional[List[float]]:
    """Map a `_COMPOSABLE_KERNELS`-ordered weight tensor onto `pool` (a
    filtered subset/reordering of it), renormalized over just that subset.

    Returns None (meaning "uniform", i.e. random.choices' own default) when
    kernel_weights is None, so every caller's unweighted behavior is
    reproduced exactly when adaptive sampling is off.
    """
    if kernel_weights is None:
        return None
    idx = [_COMPOSABLE_KERNELS.index(name) for name in pool]
    sub = [float(kernel_weights[i]) for i in idx]
    total = sum(sub)
    if total <= 0:
        return None
    return [w / total for w in sub]


def _tabicl_mix_prob_for_kernel(kernel_name: str, tabicl_mix_weights: Optional[Tensor]) -> float:
    """Per-call probability of substituting real-TabICL-PIT z_train for the
    exact analytic GP-LOO residual, given the kernel this call sampled — see
    data.z_train_tabicl_mix_* in conf/data/gp_tasks.yaml.

    kernel_name may be a bare _COMPOSABLE_KERNELS entry or a composite/chain
    string ("A+B*C", the same left-to-right format _sample_kernel_chain_
    structure and the static composite pool both use). Parsed by splitting
    on '+'/'*' (kernel names never contain either character); the MAX weight
    across every component family is used, not the mean or a component-count
    average — the family TabICL approximates worst is what should drive how
    often the whole composite gets real-signal exposure, since a chain is
    only as realistic as its weakest-approximated component.

    tabicl_mix_weights: a `_COMPOSABLE_KERNELS`-ordered tensor of per-family
    mixing fractions (see train.py::_tabicl_gap_to_mix_frac — set once at
    training startup from the measured TabICL-vs-analytic z_train gap, not
    adapted during training). None (the default, and every caller before
    this feature existed) means "unconditionally use tabicl_model whenever
    given" — returns 1.0, reproducing the legacy always-on tabicl/
    tabicl_split full-override behavior exactly.
    """
    if tabicl_mix_weights is None:
        return 1.0
    members = [n for n in re.split(r"[+*]", kernel_name) if n in _COMPOSABLE_KERNELS]
    if not members:
        return 0.0
    idx = [_COMPOSABLE_KERNELS.index(n) for n in members]
    return float(max(tabicl_mix_weights[i] for i in idx))


def _resolve_kernel_name(cfg, kernel_weights: Optional[Tensor] = None) -> str:
    """Pick which kernel to use for one task based on config.

    kernel_weights (optional): a `_COMPOSABLE_KERNELS`-ordered tensor of
    sampling weights (see _sample_kernel_chain_structure) — used to bias the
    `data.kernels` pool branch below; the fixed `data.kernel` branch has
    nothing to weight since it's a single deterministic choice.
    """
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
        weights = None
        if all(k in _COMPOSABLE_KERNELS for k in pool):
            weights = _weights_for_pool(pool, kernel_weights)
        return random.choices(pool, weights=weights, k=1)[0] if weights else random.choice(pool)
    return "rbf"


def _sample_kernel_chain_structure(
    cfg, kernel_weights: Optional[Tensor] = None
) -> tuple[List[str], List[str], str]:
    """CauKer-style composition (github.com/ShifengXIE/CauKer): sample a
    random component COUNT m ~ round(LogNormal(composite_num_kernels_lognormal_loc,
    _scale)), clipped to [composite_num_kernels_min, composite_num_kernels_max],
    then a length-m list of elementary kernels (with replacement) from
    _COMPOSABLE_KERNELS, then m-1 independently sampled +/* operators to
    combine them left-to-right (functools.reduce, see _build_kernel_chain) —
    instead of picking from the fixed 56-entry COMPOSITE_KERNELS pool. Active
    only when cfg.data.systematic_composition is True (see
    _resolve_kernel_name's docstring for the non-systematic path).
    cfg.data.composite_exclude_kernels (optional list, default empty) drops
    named elementary kernels from the sampling pool without touching
    _COMPOSABLE_KERNELS itself — that constant also seeds the static 56-entry
    COMPOSITE_KERNELS/KERNEL_REGISTRY at import time (module-level loop
    above), which must stay unfiltered for the non-systematic path. Returns
    (names, ops, chain_name) where chain_name is the same "A+B*C"-style
    left-to-right string the static composites already use (m=1 degenerates
    to a bare base-kernel name, no ops).

    LogNormal (replacing an earlier Uniform[min, max]) concentrates most
    episodes on short chains -- long "+"/"*" chains empirically shrink
    |R_star| toward 0 (each "*" link damps magnitude, each "+" link averages
    toward the population mean via a CLT-like effect), which was the
    dominant reason extreme (near +-1) correlations were rare under the old
    uniform sampling. Default loc/scale (0.55, 1.05) give P(m=1)=0.45,
    P(m=2)=0.19, P(m=3)=0.11 (75% of episodes at m<=3, preserving roughly
    the old relative 60:25:15 shape among those three) and a decaying tail
    out to composite_num_kernels_max -- validated via many independent
    small-B generate_gp_batch calls (see the batch-sampling gotcha in this
    file's module docstring) against tests/test_dataset_corr_uniform.py:
    mean +0.166->+0.264 (still inside the abs(mean)<0.30 bound), frac(R>0.7)
    0.093->0.179. Re-validate if composite_exclude_kernels or the nugget
    prior change.

    kernel_weights (optional): a `_COMPOSABLE_KERNELS`-ordered tensor of
    per-family sampling weights (see train.py's adaptive_kernel_sampling —
    updated from a per-family model-vs-oracle performance gap so weaker
    families get drawn more often). Renormalized over `pool` (post-exclude)
    via _weights_for_pool; None (the default, and every non-adaptive caller)
    reproduces today's uniform random.choices exactly."""
    exclude = set(getattr(cfg.data, "composite_exclude_kernels", None) or [])
    pool = [k for k in _COMPOSABLE_KERNELS if k not in exclude]
    if not pool:
        raise ValueError(
            f"composite_exclude_kernels={sorted(exclude)} excludes every kernel "
            f"in _COMPOSABLE_KERNELS={_COMPOSABLE_KERNELS}"
        )
    lo = int(getattr(cfg.data, "composite_num_kernels_min", 1))
    hi = int(getattr(cfg.data, "composite_num_kernels_max", 4))
    m_loc = float(getattr(cfg.data, "composite_num_kernels_lognormal_loc", 0.55))
    m_scale = float(getattr(cfg.data, "composite_num_kernels_lognormal_scale", 1.05))
    m = min(max(round(random.lognormvariate(m_loc, m_scale)), lo), hi)
    names = random.choices(pool, weights=_weights_for_pool(pool, kernel_weights), k=m)
    ops = [random.choice(("+", "*")) for _ in range(m - 1)]
    chain_name = names[0] + "".join(f"{op}{name}" for op, name in zip(ops, names[1:]))
    return names, ops, chain_name


def _build_kernel_chain(
    cfg, names: List[str], ops: List[str], k: int, B: int, device, active_dims: Optional[List[int]] = None,
    d_total: Optional[int] = None,
) -> tuple[gpytorch.kernels.Kernel, List[Dict[str, Tensor]], Dict[str, Tensor]]:
    """Sample B episodes' hyperparameters for each component in `names` (via
    the same _build_kernel_component machinery _sample_episode_kernel's
    composite branch already uses — no new hyperparameter-sampling logic,
    and "dot_product" components dispatch to the bare-LinearKernel path
    just like the static composite path does) and combine the resulting
    kernel objects left-to-right per `ops`. Returns (combined Kernel,
    per-component params list, outer sign-modulation params dict).

    component_params is one dict per component (in `names` order),
    UNFLATTENED — not coerced into the legacy l_b/alpha2_b-style schema,
    since component count is variable here (see generate_gp_batch's
    return_kernel_metadata handling). Each component dict already carries
    its own sign_applied/sign_w/sign_b (per-component injection point, via
    _build_kernel_component — cfg.data.sign_modulation_component_prob,
    independently gated per link in the chain).

    The returned outer_params dict (sign_applied_outer/sign_w_outer/
    sign_b_outer) is the POST-FOLD injection point (cfg.data.
    sign_modulation_outer_prob) applied once to the fully-combined chain
    kernel, mirroring _sample_episode_kernel's own post-composition wrap —
    kept separate from component_params (rather than a synthetic extra
    "component") since it isn't a component, it wraps the whole chain."""
    built = [
        _build_kernel_component(cfg, name, k, B, device, active_dims=active_dims, d_total=d_total)
        for name in names
    ]
    kernel = built[0][0]
    for op, (comp_kernel, _) in zip(ops, built[1:]):
        kernel = _DenseComposedKernel(kernel, op, comp_kernel)
    component_params = [params for _, params in built]

    outer_prob = float(getattr(cfg.data, "sign_modulation_outer_prob", 0.0))
    kernel, outer_params = _maybe_wrap_sign_modulated(
        cfg, kernel, outer_prob, k, B, device, active_dims=active_dims, param_suffix="_outer"
    )

    return kernel, component_params, outer_params


class _MeanFunctionBank(gpytorch.means.Mean):
    """Full CauKer-style mean bank (github.com/ShifengXIE/CauKer): each
    episode gets exactly one of {Linear (incl. constant-only), Exponential,
    Sparse-Anomaly}, chosen by a per-episode categorical draw, or exact zero
    (handled upstream in _sample_mean_module by never constructing this
    class). All three non-zero families are deterministic functions of x
    alone (never of row order/index -- this repo's rows are i.i.d. tabular
    instances, not a time series, so mu_star must be reproducible from
    x_test regardless of which rows land in train vs test at call time).

    Linear reuses the per-feature coefficient vector `weight` (one
    coefficient per input dim, matching CauKer's per-feature trend and
    gpytorch.means.LinearMean's own convention). Exponential and Anomaly
    instead each need a single scalar "progression" axis, not one
    coefficient per feature -- exp()/a threshold only stay well-behaved
    if their input is O(1) regardless of d, so both project x onto a
    *unit-norm* random direction (`exp_direction`/`anomaly_direction`)
    rather than sampling per-feature coefficients the way `weight` does.
    Given the per-episode z-normalised features this file always produces
    (see generate_gp_batch), a unit-direction projection is approximately
    N(0, 1) by the CLT regardless of d, which is what lets
    `anomaly_threshold` below be calibrated from a target sparsity
    fraction via the inverse normal CDF.

    The exponential exponent is clamped to [-10, 10] before `exp()`: without
    this, a rare large sample of `exp_rate * exp_proj` could overflow to inf,
    and `inf * 0` is nan -- silently poisoning the *other* families' rows
    through the one-hot sum below even though their own term is finite.
    """

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor,
        exp_direction: Tensor,
        exp_rate: Tensor,
        exp_scale: Tensor,
        anomaly_direction: Tensor,
        anomaly_threshold: Tensor,
        anomaly_magnitude: Tensor,
        family_onehot: Tensor,
    ):
        super().__init__()
        self.register_buffer("weight", weight)  # (B, d)
        self.register_buffer("bias", bias)  # (B,)
        self.register_buffer("exp_direction", exp_direction)  # (B, d), unit norm
        self.register_buffer("exp_rate", exp_rate)  # (B,)
        self.register_buffer("exp_scale", exp_scale)  # (B,)
        self.register_buffer("anomaly_direction", anomaly_direction)  # (B, d), unit norm
        self.register_buffer("anomaly_threshold", anomaly_threshold)  # (B,)
        self.register_buffer("anomaly_magnitude", anomaly_magnitude)  # (B,)
        self.register_buffer("family_onehot", family_onehot)  # (B, 3): [linear, exponential, anomaly]

    def forward(self, x: Tensor) -> Tensor:  # x: (B, n, d)
        linear_val = (x * self.weight.unsqueeze(1)).sum(-1) + self.bias.unsqueeze(-1)

        exp_proj = (x * self.exp_direction.unsqueeze(1)).sum(-1)
        exponent = torch.clamp(self.exp_rate.unsqueeze(-1) * exp_proj, min=-10.0, max=10.0)
        exp_val = self.exp_scale.unsqueeze(-1) * torch.exp(exponent)

        anomaly_proj = (x * self.anomaly_direction.unsqueeze(1)).sum(-1)
        anomaly_hit = (anomaly_proj > self.anomaly_threshold.unsqueeze(-1)).to(x.dtype)
        anomaly_val = anomaly_hit * self.anomaly_magnitude.unsqueeze(-1)

        stacked = torch.stack([linear_val, exp_val, anomaly_val], dim=-1)  # (B, n, 3)
        return (stacked * self.family_onehot.unsqueeze(1)).sum(-1)


def _sample_mean_module(cfg, d: int, B: int, device) -> tuple[gpytorch.means.Mean, Dict[str, Tensor]]:
    """Non-zero GP mean bank (CauKer-inspired, github.com/ShifengXIE/CauKer:
    their Table 1 shows "Mean+KernelSynth" — adding a non-zero mean function
    — clearly beats zero-mean KernelSynth). Covers all four of CauKer's own
    mean families: Zero, Linear, Exponential, Sparse Anomalies (see
    _MeanFunctionBank for the Exponential/Anomaly formulas).

    Mathematically inert for R_star/Sigma_star/sigma_star regardless of
    which family fires: a GP's posterior *covariance* never depends on its
    mean function (only mu_star does — see R&W §2.7), so this only
    diversifies mu_star/z_train/z_test's realism and can never perturb the
    correlation structure this pipeline exists to report. It DOES need
    mu_star/z_train/z_test to be computed against this same mean (see
    _generate_gp_batch_raw's oracle_mode branch) — adding a mean to y_all
    without also updating those would silently miscalibrate the PIT.

    Per-episode gating (batched, no Python loop over B):
      - nonzero_mask ~ Bernoulli(mean_fn_prob): does this episode get any
        non-zero mean at all? (else exact ZeroMean)
      - family_idx ~ Categorical(mean_fn_family_probs) over
        {linear, exponential, anomaly}, AND'd with nonzero_mask via
        family_onehot (a fully-zeroed one-hot row makes _MeanFunctionBank's
        forward() return exactly 0 for that episode, i.e. ZeroMean).
      - Within the linear family: linear_mask ~
        Bernoulli(mean_fn_linear_prob) decides whether it carries a trend
        (a random direction across the full d-dimensional feature space) or
        is constant-only (weight forced to 0, bias kept).
    cfg.data.mean_fn_enabled defaults to False (byte-for-byte no-op — same
    convention as mlp_mixing_enabled) — every existing config/dataset is
    unaffected until this is explicitly turned on, and the disabled path
    returns a plain gpytorch.means.ZeroMean (no RNG draws at all, so no
    save/restore of RNG state is needed there).

    Returns (mean_module, params) where params (for return_kernel_metadata)
    holds "mean_weight" (B, d), "mean_bias" (B,), "mean_nonzero" (B,) bool,
    "mean_family" (B,) long in {0=linear, 1=exponential, 2=anomaly} (only
    meaningful where mean_nonzero is True), "mean_linear" (B,) bool, plus
    the exponential/anomaly families' own params ("mean_exp_direction" (B,
    d), "mean_exp_rate" (B,), "mean_exp_scale" (B,), "mean_anomaly_direction"
    (B, d), "mean_anomaly_threshold" (B,), "mean_anomaly_magnitude" (B,)) --
    every family's params are always present (0.0-sentinel convention, same
    as period/rq_alpha/power elsewhere in this file) so pit.py::gp_analytical_pit
    can reconstruct mean_module for *any* family, not just linear.
    """
    batch_shape = torch.Size([B])

    if not bool(getattr(cfg.data, "mean_fn_enabled", False)):
        mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape).to(device)
        params = {
            "mean_weight": torch.zeros(B, d, device=device),
            "mean_bias": torch.zeros(B, device=device),
            "mean_nonzero": torch.zeros(B, dtype=torch.bool, device=device),
            "mean_family": torch.zeros(B, dtype=torch.long, device=device),
            "mean_linear": torch.zeros(B, dtype=torch.bool, device=device),
            "mean_exp_direction": torch.zeros(B, d, device=device),
            "mean_exp_rate": torch.zeros(B, device=device),
            "mean_exp_scale": torch.zeros(B, device=device),
            "mean_anomaly_direction": torch.zeros(B, d, device=device),
            "mean_anomaly_threshold": torch.zeros(B, device=device),
            "mean_anomaly_magnitude": torch.zeros(B, device=device),
        }
        return mean_module, params

    prob_nonzero = float(getattr(cfg.data, "mean_fn_prob", 0.5))
    prob_linear = float(getattr(cfg.data, "mean_fn_linear_prob", 0.5))
    weight_std = float(getattr(cfg.data, "mean_fn_weight_std", 0.5))
    bias_std = float(getattr(cfg.data, "mean_fn_bias_std", 1.0))
    family_probs = list(getattr(cfg.data, "mean_fn_family_probs", [0.5, 0.25, 0.25]))
    exp_rate_std = float(getattr(cfg.data, "mean_fn_exp_rate_std", 0.5))
    exp_scale_std = float(getattr(cfg.data, "mean_fn_exp_scale_std", 1.0))
    anomaly_frac = float(getattr(cfg.data, "mean_fn_anomaly_frac", 0.1))
    anomaly_magnitude_std = float(getattr(cfg.data, "mean_fn_anomaly_magnitude_std", 2.0))

    nonzero_mask = torch.rand(B, device=device) < prob_nonzero

    family_weights = torch.tensor(family_probs, device=device, dtype=torch.float32)
    family_idx = torch.multinomial(family_weights.expand(B, -1), 1, replacement=True).squeeze(-1)  # (B,) in {0,1,2}
    family_onehot = torch.zeros(B, 3, device=device)
    family_onehot.scatter_(1, family_idx.unsqueeze(-1), 1.0)
    family_onehot = family_onehot * nonzero_mask.unsqueeze(-1)  # zero out episodes with no mean at all

    is_linear = nonzero_mask & (family_idx == 0)
    linear_mask = is_linear & (torch.rand(B, device=device) < prob_linear)
    weight = torch.randn(B, d, device=device) * weight_std * linear_mask.unsqueeze(-1)
    bias = torch.randn(B, device=device) * bias_std * is_linear

    exp_direction = torch.randn(B, d, device=device)
    exp_direction = exp_direction / exp_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    exp_rate = torch.randn(B, device=device) * exp_rate_std
    exp_scale = torch.randn(B, device=device) * exp_scale_std

    anomaly_direction = torch.randn(B, d, device=device)
    anomaly_direction = anomaly_direction / anomaly_direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # Projection onto a unit direction of per-episode z-normalised features is
    # approximately N(0, 1) (CLT), so the inverse normal CDF of (1 -
    # anomaly_frac) gives a threshold that fires on approximately
    # anomaly_frac of instances, independent of d.
    anomaly_threshold_value = math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * (1.0 - anomaly_frac) - 1.0))
    anomaly_threshold = anomaly_threshold_value.to(device).expand(B).clone()
    anomaly_magnitude = torch.randn(B, device=device) * anomaly_magnitude_std

    mean_module = _MeanFunctionBank(
        weight=weight,
        bias=bias,
        exp_direction=exp_direction,
        exp_rate=exp_rate,
        exp_scale=exp_scale,
        anomaly_direction=anomaly_direction,
        anomaly_threshold=anomaly_threshold,
        anomaly_magnitude=anomaly_magnitude,
        family_onehot=family_onehot,
    ).to(device)

    params = {
        "mean_weight": weight,
        "mean_bias": bias,
        "mean_nonzero": nonzero_mask,
        "mean_family": family_idx,
        "mean_linear": linear_mask,
        "mean_exp_direction": exp_direction,
        "mean_exp_rate": exp_rate,
        "mean_exp_scale": exp_scale,
        "mean_anomaly_direction": anomaly_direction,
        "mean_anomaly_threshold": anomaly_threshold,
        "mean_anomaly_magnitude": anomaly_magnitude,
    }
    return mean_module, params


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
        l, alpha2, nugget, kernel, period, rq_alpha, power,
        l_b, alpha2_b, period_b, rq_alpha_b, power_b, kernel_feature_indices,
        _L_ff, _alpha  (ephemeral Cholesky factors, consumed by
                        pit.py::gp_analytical_pit, not saved to disk)
    """
    return generate_gp_batch(cfg, 1, "cpu", return_kernel_metadata=True)[0]


# ---------------------------------------------------------------------------
# Batched generation (C: vectorised over B episodes simultaneously)
# ---------------------------------------------------------------------------


def _gathered_psd_safe_cholesky(K: Tensor, label: str, max_tries: int = 6) -> tuple[Tensor, Tensor]:
    """Shared engine for _batched_cholesky/_psd_safe_batch below: batched
    (B, N, N) -> (L, failed) Cholesky factor. failed (B,) bool marks episodes
    where even gpytorch's psd_safe_cholesky's max jitter couldn't recover a
    PSD matrix; L is an identity placeholder for those, and the caller is
    expected to drop them rather than save a degenerate episode.

    Tries a single plain cholesky_ex on the FULL batch first (cheap, and
    correct for the common well-conditioned case). For whichever episodes
    fail that, gathers ONLY those into a smaller (n_failed, N, N) tensor and
    hands that off to gpytorch's own psd_safe_cholesky — same escalating-
    jitter mechanism gpytorch falls back to internally (starting at
    gpytorch.settings.cholesky_jitter, x10 per retry, matching
    max_tries=6's ~1e-6*10^5=0.1 ceiling) — instead of calling it on the
    whole batch. A single batched LAPACK call can't skip elements, so
    psd_safe_cholesky's own escalation loop redoes ALL of its input every
    retry round; calling it on the whole B-sized batch means every well-
    conditioned episode pays for the escalation rounds too, even though only
    the failing few need them. Gathering first keeps that cost proportional
    to n_failed instead of B.

    For composite kernels with a finite-rank component (dot_product/
    polynomial — see composite_exclude_kernels' docstring in
    conf/data/gp_tasks.yaml) T commonly exceeds the feature-map rank, so a
    handful of episodes per batch routinely need the full escalation while
    the rest of the batch is fine on the first try — the common case this
    gathering optimizes for, not a rare edge case.

    psd_safe_cholesky raises NotPSDError (rather than returning a partial
    result) if ANY element of its input is still not PSD after max_tries —
    so on that exception (now scoped to just the n_failed subset, not the
    whole batch) we re-derive exactly which of THOSE are still bad at the
    same maximum jitter and fall back to identity only for them, logging the
    discard rate (not silent, so a run-wide rate can be monitored).

    It also raises NanError immediately (before any jitter escalation) if
    ANY element of its input is NaN — unlike a plain cholesky_ex, which
    _batched_cholesky's pre-refactor hand-rolled loop relied on quietly
    returning a nonzero info code for a NaN row (never actually recovering,
    just wasting max_tries rounds before the caller's existing fallback
    kicked in). Caught alongside NotPSDError and handled the same way: NaN
    rows are identified explicitly (jitter can't fix them — NaN + jitter is
    still NaN) and forced into the discarded set regardless of what
    cholesky_ex reports for them.
    """
    L, info = torch.linalg.cholesky_ex(K)
    failed = info.ne(0)
    if not failed.any():
        return L, failed
    idx = failed.nonzero(as_tuple=True)[0]
    eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
    try:
        L[idx] = psd_safe_cholesky(K[idx], max_tries=max_tries)
        return L, torch.zeros_like(failed)
    except (NotPSDError, NanError):
        jitter0 = gpytorch.settings.cholesky_jitter.value(K.dtype)
        max_jitter = jitter0 * (10 ** (max_tries - 1))
        K_boosted = K[idx] + max_jitter * eye
        nan_mask = torch.isnan(K_boosted).any(dim=-1).any(dim=-1)
        if nan_mask.any():
            K_boosted = K_boosted.clone()
            K_boosted[nan_mask] = eye
        L_sub, info_sub = torch.linalg.cholesky_ex(K_boosted)
        L[idx] = L_sub
        still_failed = torch.zeros_like(failed)
        still_failed[idx] = info_sub.ne(0) | nan_mask
        warnings.warn(
            f"{label}: {int(still_failed.sum())}/{K.shape[0]} episodes fell back "
            f"to an identity Cholesky factor (unrecoverable even at jitter="
            f"{max_jitter:.1e}, or had NaN entries) and will be discarded.",
            RuntimeWarning,
        )
        L[still_failed] = eye.unsqueeze(0).expand_as(L[still_failed])
        return L, still_failed


def _batched_cholesky(K: Tensor) -> tuple[Tensor, Tensor]:
    """Batched Cholesky (B, N, N) → (L, failed) for K_ff/LOO PIT — see
    _gathered_psd_safe_cholesky's docstring for the escalation/gathering
    mechanics. The caller (generate_gp_batch) discards episodes marked
    failed rather than saving a degenerate K_ff-derived episode."""
    return _gathered_psd_safe_cholesky(K, label="_batched_cholesky (K_ff)")


def _psd_safe_batch(K: Tensor, max_tries: int = 6) -> tuple[Tensor, Tensor]:
    """Batched (B, N, N) -> (L, failed) Cholesky factor for K_all, used to
    GUARANTEE Sigma_star/R_star are PSD rather than just symmetric (see
    generate_gp_batch's joint-sample block) — see _gathered_psd_safe_cholesky's
    docstring for the escalation/gathering mechanics shared with
    _batched_cholesky above.

    Separately, an occasional extreme composite-kernel hyperparameter draw
    makes gpytorch's own float32 kernel evaluation produce actual NaN
    entries (not just a slightly negative eigenvalue) in K_all_raw.
    psd_safe_cholesky checks for NaN and raises NanError immediately, before
    its jitter escalation ever runs — so even the other, perfectly fine
    episodes gathered into the same failing-subset call would be blocked.
    NaN episodes are therefore replaced with an identity placeholder up
    front (trivially PSD, so they never enter the failing subset at all —
    they're merged into the returned `failed` unconditionally below instead)
    and reported separately.
    """
    nan_failed = torch.isnan(K).any(dim=-1).any(dim=-1)
    if nan_failed.any():
        eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
        K = K.clone()
        K[nan_failed] = eye
        warnings.warn(
            f"_psd_safe_batch: {int(nan_failed.sum())}/{K.shape[0]} episodes had "
            f"NaN entries in K_all (kernel evaluation produced NaN) and will be "
            f"discarded.",
            RuntimeWarning,
        )
    L, failed = _gathered_psd_safe_cholesky(K, label="_psd_safe_batch (K_all)", max_tries=max_tries)
    return L, failed | nan_failed


def tabiclv2_warp_features(x: Tensor, seed: Optional[int] = None) -> Tensor:
    """Warp each feature column with one of 11 random marginal transforms.

    Simulates the extreme marginal heterogeneity of real tabular data
    (TabICLv2): heavy tails, power laws, ordinal steps, bimodal mixtures,
    periodicity, Cauchy outliers, zero-inflation, bounded/proportion-like
    ranges, and left skew, applied on top of a Standard Normal baseline.
    Intended to run before any per-episode mean/std normalisation, so
    downstream kernel/covariance code keeps operating on calibrated,
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
    choices = torch.randint(0, 11, (B, d), device=x.device)

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
            elif c == 8:  # Zero-inflation — point mass at 0 mixed with a continuous tail
                spike_frac = float(torch.empty(1).uniform_(0.2, 0.6))
                mask = torch.rand_like(col_data) < spike_frac
                warped_x[b, :, col] = torch.where(mask, torch.zeros_like(col_data), col_data)
            elif c == 9:  # Bounded / sigmoid squash — proportions, percentages, probabilities
                scale = float(torch.empty(1).uniform_(0.5, 3.0))
                warped_x[b, :, col] = torch.sigmoid(col_data * scale)
            elif c == 10:  # Left-skew — mirror of the log-normal/exponential (c == 3) above
                warped_x[b, :, col] = -torch.exp((-col_data).clamp(min=-5.0, max=4.0))

    if added_batch_dim:
        warped_x = warped_x.squeeze(0)
    return warped_x


# Structural feature-warp categories/ops, ported from TempoPFN's
# (github.com/automl/TempoPFN) offline per-series augmentor
# (UnivariateOfflineAugmentor.apply, offline_per_sample_iid_augmentations.py):
# TempoPFN applies these post-hoc to sampled *outputs*, which isn't valid
# here — our oracle target (R_star/Sigma_star) is derived analytically from
# the kernel evaluated on x, so any transform of the sampled y that isn't
# itself expressible as a kernel change invalidates the label. Instead these
# run on x, alongside tabiclv2_warp_features/apply_mlp_feature_mixing: a
# fixed, deterministic feature map f still yields a valid kernel
# k(f(x_i), f(x_j)), so R_star computed downstream stays exact, regardless of
# how many ops are composed or in what order.
#
# TempoPFN's own orchestration is two-level: sample 2-6 CATEGORIES (out of 6)
# without replacement, weighted, then apply at most ONE op per chosen
# category, always in the fixed category order below. Mirrored exactly here,
# except:
#   - "seasonality" drops the calendar-injection op (real day-of-week/
#     month-end effects need actual pd.Timestamps, which GP points don't
#     have) and keeps only amplitude_modulation, redefined on the value-rank
#     pseudo-time axis (see _structural_warp_column) instead of a real
#     position axis, per the same reasoning as regime_change/shock_recovery.
#   - "analytic" (TempoPFN's DifferentialAugmenter: gaussian-smooth / sobel /
#     laplace / integral / 3rd-/4th-derivative, one chosen uniformly) is
#     ported as a 4-way choice {smooth, first-derivative, second-derivative,
#     cumulative integral} -- 3rd/4th derivatives amplify noise enough on
#     already-warped Gaussian data to risk destabilizing R_star's Cholesky
#     (see the NotPSDError handling in _generate_gp_batch_raw), so they're
#     dropped rather than risk an unrecoverable draw for marginal benefit.
#   - cross-episode mixup has no analogue here — it combines *episodes*,
#     which would need a corresponding kernel combination to stay
#     label-safe (that's what the composite/sum-product kernels already do).
_STRUCTURAL_CATEGORIES: List[str] = [
    "invariances",
    "structure",
    "seasonality",
    "artifacts",
    "analytic",
    "discrete",
]

_CATEGORY_OPS: Dict[str, List[str]] = {
    "invariances": ["yflip", "time_flip"],
    "structure": ["regime_change", "shock_recovery"],
    "seasonality": ["amplitude_modulation"],
    "artifacts": ["resample_artifact"],
    "analytic": ["differential"],
    "discrete": ["quantize", "censor"],
}

# TempoPFN's own default category weights (offline_per_sample_iid_augmentations.py).
_DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = {
    "invariances": 0.6,
    "structure": 0.6,
    "seasonality": 0.5,
    "artifacts": 0.3,
    "analytic": 0.4,
    "discrete": 0.6,
}

# Sub-op weights within a category, for categories with >1 op. "discrete"
# matches TempoPFN's own quantize/censor split; "invariances" and "structure"
# are uniform in TempoPFN (plain rng.choice over the enabled ops).
_CATEGORY_SUB_OP_WEIGHTS: Dict[str, Dict[str, float]] = {
    "discrete": {"quantize": 0.6, "censor": 0.4},
}


def _structural_warp_column(col_data: Tensor, op: str, use_index_axis: bool = False) -> Tensor:
    """Apply one structural transform to a single (episode, column) feature
    vector of shape (T,).

    GP episodes have no time axis, unlike TempoPFN's series. By default
    (use_index_axis=False) "order" here is each point's own VALUE-RANK within
    this column -- decoupled from the train/test split (a plain index slice,
    see generate_gp_batch), so an order-dependent op can't systematically land
    on one side of the split. Setting use_index_axis=True instead uses the
    literal row position 0..T-1 (the same axis the split is defined on) as
    TempoPFN's real series would -- a more faithful port, but one that *can*
    introduce a fixed train/test coupling (see apply_structural_feature_warp's
    structural_warp_index_axis_ratio). Either way this stays a fixed,
    deterministic function of col_data alone (independent of the other d-1
    columns), keeping the overall column map a valid deterministic feature
    map (see the module-level comment above `_STRUCTURAL_CATEGORIES`).
    """
    T = col_data.shape[0]
    device = col_data.device
    std = col_data.std()
    if not torch.isfinite(std) or std <= 0:
        std = torch.ones((), device=device)

    # Ops that don't need either pseudo-time axis at all.
    if op == "yflip":
        return -col_data

    if op == "censor":
        # Elementwise clip between two random quantiles — no ordering needed.
        q_low, q_high = float(torch.rand(1)), float(torch.rand(1))
        q_low, q_high = min(q_low, q_high), max(q_low, q_high)
        sorted_vals = torch.sort(col_data).values
        lo = sorted_vals[int(q_low * (T - 1))]
        hi = sorted_vals[int(q_high * (T - 1))]
        # q_low/q_high round to the same discrete index fairly often for
        # small T (int(q*(T-1)) collapses a whole range of q onto one
        # index) -- when that happens lo == hi and clamp(min=lo, max=hi)
        # silently flattens the ENTIRE column to one constant, not just its
        # tails (same failure shape as quantize's lo==hi guard below, which
        # this mirrors). A column that degenerate feeding a kernel capped to
        # k=1 active dims (periodic/cosine, see generate_gp_batch) zeroes
        # out r for every pair, making that episode's whole covariance
        # structure a constant -- so no-op here instead of collapsing.
        if not torch.isfinite(hi - lo) or (hi - lo).item() <= 0:
            return col_data
        return col_data.clamp(min=lo.item(), max=hi.item())

    if op == "quantize":
        # Non-equidistant quantization (TempoPFN's QuantizationAugmenter):
        # snap each value to its nearest of n_levels levels, where levels are
        # {min, max} plus (n_levels-2) random interior points -- uniform
        # random interior points here instead of TempoPFN's Sobol sequence
        # (same idea: non-equidistant levels, no new dependency).
        lo, hi = col_data.min(), col_data.max()
        if not torch.isfinite(hi - lo) or (hi - lo).item() <= 0:
            return col_data
        n_levels = int(torch.randint(3, 11, (1,)).item())
        n_interior = max(0, n_levels - 2)
        interior = lo + (hi - lo) * torch.rand(n_interior, device=device)
        levels = torch.sort(torch.cat([lo.view(1), hi.view(1), interior])).values
        idx = torch.argmin((col_data.unsqueeze(1) - levels.unsqueeze(0)).abs(), dim=1)
        return levels[idx]

    # Remaining ops act on a pseudo-time axis: value-rank (default) or, if
    # use_index_axis, the raw row position (sort_idx is then the identity
    # permutation, so "sorted_vals" is just col_data itself and the final
    # scatter-back is a no-op reassignment).
    if use_index_axis:
        sort_idx = torch.arange(T, device=device)
    else:
        sort_idx = torch.argsort(col_data)
    sorted_vals = col_data[sort_idx]
    rank = torch.arange(T, device=device, dtype=torch.float32)

    if op == "time_flip":
        # Whole-axis reversal (TempoPFN's TimeFlipAugmenter) along whichever
        # pseudo-time axis is active: reverses raw row order if
        # use_index_axis, otherwise reverses the value-rank order (the point
        # holding the smallest value swaps with the one holding the largest,
        # etc.) -- either way a fixed permutation, still a valid deterministic
        # feature map.
        transformed = sorted_vals.flip(dims=[0])

    elif op == "regime_change":
        min_seg = max(4, T // 16)
        valid_hi = T - min_seg
        if valid_hi <= min_seg:
            transformed = sorted_vals
        else:
            num_cp = int(torch.randint(1, 4, (1,)).item())
            valid = torch.arange(min_seg, valid_hi, device=device)
            num_cp = min(num_cp, valid.numel())
            cp = torch.sort(valid[torch.randperm(valid.numel(), device=device)[:num_cp]]).values
            boundaries = torch.cat(
                [
                    torch.zeros(1, device=device, dtype=cp.dtype),
                    cp,
                    torch.full((1,), T, device=device, dtype=cp.dtype),
                ]
            )
            transformed = sorted_vals.clone()
            for i in range(boundaries.numel() - 1):
                s, e = int(boundaries[i]), int(boundaries[i + 1])
                if e <= s:
                    continue
                seg = sorted_vals[s:e]
                scale = float(torch.empty(1).uniform_(0.8, 1.25))
                shift = float(torch.randn(1)) * 0.15 * std.item()
                seg_mean = seg.mean()
                transformed[s:e] = (seg - seg_mean) * scale + seg_mean + shift

    elif op == "shock_recovery":
        lo = max(1, T // 16)
        hi = max(lo + 1, T - T // 16)
        t0 = int(torch.randint(lo, hi, (1,)).item())
        mag = float(torch.empty(1).uniform_(0.5, 2.0)) * std.item()
        if torch.rand(1).item() < 0.5:
            mag = -mag
        half_life = max(1.0, float(torch.empty(1).uniform_(0.05, 0.3)) * T)
        decay = torch.exp(-(rank - t0).clamp(min=0) / half_life)
        transformed = sorted_vals + mag * decay

    elif op == "amplitude_modulation":
        # Rescale one contiguous rank-window's amplitude around its local
        # mean (TempoPFN's _apply_seasonality_amplitude_modulation, redefined
        # on the rank axis -- no shift, unlike regime_change, and only ONE
        # window rather than partitioning the whole column).
        min_w = max(4, T // 16)
        max_w = max(min_w + 1, T // 2)
        win = int(torch.randint(min_w, max_w + 1, (1,)).item())
        start = int(torch.randint(0, max(1, T - win) + 1, (1,)).item())
        end = min(T, start + win)
        transformed = sorted_vals.clone()
        seg = sorted_vals[start:end]
        if seg.numel() > 0:
            seg_mean = seg.mean()
            amp = float(torch.empty(1).uniform_(0.5, 1.8))
            transformed[start:end] = (seg - seg_mean) * amp + seg_mean

    elif op == "differential":
        # TempoPFN's DifferentialAugmenter, reduced to a 4-way choice (see
        # module comment): gaussian-smooth, 1st derivative (Sobel-like),
        # 2nd derivative (Laplace-like), or a cumulative integral -- always
        # applied on top of a box-smoothed version, then rescaled back into
        # the original value range (matches TempoPFN's _rescale_signal).
        k = max(3, T // 32)
        k = k + 1 if k % 2 == 0 else k
        box = torch.ones(k, device=device) / k
        padded = torch.nn.functional.pad(sorted_vals.view(1, 1, -1), (k // 2, k // 2), mode="reflect")
        smoothed = torch.nn.functional.conv1d(padded, box.view(1, 1, -1)).view(-1)

        sub_op = int(torch.randint(0, 4, (1,)).item())
        if sub_op == 0:
            raw = smoothed
        elif sub_op == 1:  # first derivative
            sk = torch.tensor([-1.0, 0.0, 1.0], device=device)
            p = torch.nn.functional.pad(smoothed.view(1, 1, -1), (1, 1), mode="reflect")
            raw = torch.nn.functional.conv1d(p, sk.view(1, 1, -1)).view(-1)
        elif sub_op == 2:  # second derivative
            sk = torch.tensor([1.0, -2.0, 1.0], device=device)
            p = torch.nn.functional.pad(smoothed.view(1, 1, -1), (1, 1), mode="reflect")
            raw = torch.nn.functional.conv1d(p, sk.view(1, 1, -1)).view(-1)
        else:  # cumulative integral, running from the left or right
            if torch.rand(1).item() < 0.5:
                raw = torch.cumsum(smoothed, dim=0)
            else:
                raw = torch.flip(torch.cumsum(torch.flip(smoothed, dims=[0]), dim=0), dims=[0])

        r_min, r_max = raw.min(), raw.max()
        s_min, s_max = sorted_vals.min(), sorted_vals.max()
        # If raw comes out perfectly flat (e.g. a linear column's 2nd
        # derivative, or a constant-slope column's 1st derivative), the old
        # unconditional rescale ((raw - r_min) / clamp(denom, 1e-8) * range +
        # s_min) divides a numerically-zero numerator by a floor-clamped
        # denominator, which is finite but still collapses transformed to the
        # single constant s_min for every point -- the same "whole column
        # flattens to one value" failure shape as the censor lo==hi bug this
        # mirrors. No-op instead of collapsing.
        if not torch.isfinite(r_max - r_min) or (r_max - r_min).item() <= 1e-8:
            transformed = sorted_vals
        else:
            denom = r_max - r_min
            transformed = (raw - r_min) / denom * (s_max - s_min) + s_min

    elif op == "resample_artifact":
        # Downsample (with a random phase offset) then upsample back via one
        # of 3 modes (TempoPFN: linear interp / step-hold / linear+smooth).
        max_factor = max(2, min(8, T // 32))
        factor = int(torch.randint(2, max_factor + 1, (1,)).item())
        offset = int(torch.randint(0, factor, (1,)).item())
        ds_idx = torch.arange(offset, T, factor, device=device)
        if ds_idx.numel() < 3:
            transformed = sorted_vals
        else:
            ds_vals_np = sorted_vals[ds_idx].detach().cpu().numpy()
            ds_idx_np = ds_idx.detach().cpu().numpy().astype(np.float64)
            rank_np = rank.detach().cpu().numpy()
            mode_idx = int(torch.multinomial(torch.tensor([0.5, 0.2, 0.3]), 1).item())
            if mode_idx == 0:  # linear
                us_np = np.interp(rank_np, ds_idx_np, ds_vals_np)
            elif mode_idx == 1:  # step-hold: forward-fill from the last downsampled point
                us_np = ds_vals_np[np.searchsorted(ds_idx_np, rank_np, side="right") - 1]
            else:  # linear + light smoothing
                us_np = np.interp(rank_np, ds_idx_np, ds_vals_np)
                sm_k = max(3, T // 128)
                sm_kernel = np.ones(sm_k) / sm_k
                us_np = np.convolve(us_np, sm_kernel, mode="same")
            transformed = torch.from_numpy(us_np).to(device=device, dtype=sorted_vals.dtype)

    else:
        raise ValueError(f"Unknown structural warp op '{op}'")

    warped = torch.empty_like(col_data)
    warped[sort_idx] = transformed
    return warped


def _sample_structural_ops(category_weights: Dict[str, float], num_ops_min: int, num_ops_max: int) -> List[str]:
    """Sample 1 op per chosen category, mirroring TempoPFN's own two-level
    orchestration (UnivariateOfflineAugmentor.apply): first draw 2..6
    categories WITHOUT replacement (weighted by category_weights, zero-weight
    categories excluded), then always in the FIXED canonical category order
    (_STRUCTURAL_CATEGORIES) -- never in draw order -- pick exactly one op
    per chosen category (weighted by _CATEGORY_SUB_OP_WEIGHTS if the category
    has more than one op).

    Uses torch's RNG throughout (torch.multinomial/randint), not numpy's, so
    this is governed by the same torch.manual_seed/_seed_everything calls as
    every other draw in this file.
    """
    eligible = [c for c in _STRUCTURAL_CATEGORIES if category_weights.get(c, 0.0) > 0.0]
    if not eligible:
        return []
    k = min(int(torch.randint(num_ops_min, num_ops_max + 1, (1,)).item()), len(eligible))
    weights = torch.tensor([category_weights[c] for c in eligible], dtype=torch.float32)
    weights = weights / weights.sum()
    idx = torch.multinomial(weights, k, replacement=False)
    chosen_categories = {eligible[i] for i in idx.tolist()}

    ops: List[str] = []
    for category in _STRUCTURAL_CATEGORIES:  # fixed canonical order, not draw order
        if category not in chosen_categories:
            continue
        candidates = _CATEGORY_OPS[category]
        if len(candidates) == 1:
            ops.append(candidates[0])
            continue
        sub_weights_map = _CATEGORY_SUB_OP_WEIGHTS.get(category)
        if sub_weights_map is None:
            sub_weights = torch.ones(len(candidates))
        else:
            sub_weights = torch.tensor([sub_weights_map[c] for c in candidates], dtype=torch.float32)
        sub_weights = sub_weights / sub_weights.sum()
        pick = int(torch.multinomial(sub_weights, 1).item())
        ops.append(candidates[pick])
    return ops


def _sample_structural_category_mask(
    M: int, category_weights: Dict[str, float], num_ops_min: int, num_ops_max: int, device,
) -> Tuple[Tensor, List[str]]:
    """Batched equivalent of _sample_structural_ops's category-selection step
    for M independent draws at once: for each of the M rows, choose a random
    NUMBER of categories k in [num_ops_min, num_ops_max] (capped to the
    number of eligible -- nonzero-weight -- categories), then choose k
    categories from those eligible WITHOUT replacement, weighted by
    category_weights. Sub-op-within-category selection (for categories with
    >1 op) is handled separately by the caller, since it only needs to run
    over the (typically much smaller) subset of rows that picked that
    specific category.

    Implemented via the Gumbel-top-k trick (argsort of log-weight + i.i.d.
    Gumbel noise): this produces exactly a Plackett-Luce random ranking of
    the eligible categories per row -- the same sequential
    sample-then-renormalize distribution torch.multinomial(weights, k,
    replacement=False) draws from -- so taking the first k of that ranking
    per row is distributionally identical to _sample_structural_ops's
    original per-column torch.multinomial call, just computed for all M rows
    in one vectorised pass instead of a Python loop over M columns.

    Returns (chosen_mask, eligible): chosen_mask is (M, len(eligible)) bool
    (chosen_mask[m, i] = True iff row m selected eligible[i]); eligible is
    the same zero-weight-excluded category list _sample_structural_ops uses
    (fixed for the whole call, since category_weights doesn't vary per row).
    """
    eligible = [c for c in _STRUCTURAL_CATEGORIES if category_weights.get(c, 0.0) > 0.0]
    n_elig = len(eligible)
    if n_elig == 0:
        return torch.zeros(M, 0, dtype=torch.bool, device=device), eligible

    weights = torch.tensor([category_weights[c] for c in eligible], dtype=torch.float32, device=device)
    log_w = torch.log(weights / weights.sum())
    u = torch.rand(M, n_elig, device=device).clamp_min(1e-12)
    gumbel = -torch.log((-torch.log(u)).clamp_min(1e-12))
    scores = log_w.unsqueeze(0) + gumbel
    # rank[m, i] = position of category i in row m's Gumbel-perturbed order
    # (0 = first/most-preferred draw for that row).
    rank = torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1)

    k = torch.randint(num_ops_min, num_ops_max + 1, (M,), device=device)
    k = torch.clamp(k, max=n_elig)
    chosen_mask = rank < k.unsqueeze(1)
    return chosen_mask, eligible


def _structural_warp_batch(col_data: Tensor, op: str, use_index_axis: Tensor) -> Tensor:
    """Batched equivalent of _structural_warp_column: applies `op` to every
    row of col_data (M, T) independently and simultaneously (each row is one
    gated (episode, feature-column) pair's T-length pseudo-series), matching
    _structural_warp_column's per-row math and edge-case handling exactly,
    just without a Python loop over M. use_index_axis: (M,) bool, one choice
    per row (mirrors _structural_warp_column's own flag, chosen once per
    gated column upstream in apply_structural_feature_warp).

    "resample_artifact" is the one op NOT vectorised here: its numpy
    interp/searchsorted path has a genuinely variable-length downsample index
    per row (the factor/offset that determine ds_idx are themselves random
    per row), which doesn't reduce to a clean batched form the way the other
    ops' fixed-size (M, T) tensor ops do. It falls back to a bounded loop
    over the M rows selecting this category, reusing _structural_warp_column
    (the original, already-tested single-column implementation) unchanged
    rather than risking a fresh reimplementation of that numpy path.
    """
    M, T = col_data.shape
    device = col_data.device

    if op == "resample_artifact":
        out = torch.empty_like(col_data)
        for i in range(M):
            out[i] = _structural_warp_column(col_data[i], op, use_index_axis=bool(use_index_axis[i]))
        return out

    std = col_data.std(dim=1)
    std = torch.where(torch.isfinite(std) & (std > 0), std, torch.ones_like(std))

    if op == "yflip":
        return -col_data

    if op == "censor":
        q = torch.rand(M, 2, device=device)
        q_low = q.min(dim=1).values
        q_high = q.max(dim=1).values
        sorted_vals, _ = torch.sort(col_data, dim=1)
        lo_idx = (q_low * (T - 1)).long().clamp(0, T - 1)
        hi_idx = (q_high * (T - 1)).long().clamp(0, T - 1)
        lo = torch.gather(sorted_vals, 1, lo_idx.unsqueeze(1)).squeeze(1)
        hi = torch.gather(sorted_vals, 1, hi_idx.unsqueeze(1)).squeeze(1)
        # See _structural_warp_column's "censor" branch: q_low/q_high can
        # round to the same discrete index for small T, which would silently
        # flatten the whole column via clamp(min=lo, max=hi) with lo==hi.
        # +-inf bounds make the clamp a genuine no-op for those rows instead.
        degenerate = ~torch.isfinite(hi - lo) | ((hi - lo) <= 0)
        lo_eff = torch.where(degenerate, torch.full_like(lo, float("-inf")), lo)
        hi_eff = torch.where(degenerate, torch.full_like(hi, float("inf")), hi)
        return torch.clamp(col_data, min=lo_eff.unsqueeze(1), max=hi_eff.unsqueeze(1))

    if op == "quantize":
        lo = col_data.min(dim=1).values
        hi = col_data.max(dim=1).values
        degenerate = ~torch.isfinite(hi - lo) | ((hi - lo) <= 0)
        max_interior = 8  # n_levels in [3,10] -> n_interior in [1,8]
        n_levels = torch.randint(3, 11, (M,), device=device)
        n_interior = (n_levels - 2).clamp(min=0)
        interior_raw = torch.rand(M, max_interior, device=device)
        interior = lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * interior_raw
        slot_idx = torch.arange(max_interior, device=device).unsqueeze(0)
        interior_mask = slot_idx < n_interior.unsqueeze(1)
        # Padding slots beyond this row's n_interior are set to +inf so they
        # never win the nearest-level argmin below (real values are finite).
        interior = torch.where(interior_mask, interior, torch.full_like(interior, float("inf")))
        levels = torch.cat([lo.unsqueeze(1), hi.unsqueeze(1), interior], dim=1)
        levels, _ = torch.sort(levels, dim=1)
        diff = (col_data.unsqueeze(2) - levels.unsqueeze(1)).abs()
        idx = diff.argmin(dim=2)
        quantized = torch.gather(levels, 1, idx)
        return torch.where(degenerate.unsqueeze(1), col_data, quantized)

    # Remaining ops act on a pseudo-time axis: value-rank (default) or, if
    # use_index_axis, the raw row position -- see _structural_warp_column's
    # docstring. sort_idx/sorted_vals/rank mirror that function's per-row
    # construction, batched across all M rows.
    argsort_idx = torch.argsort(col_data, dim=1)
    index_idx = torch.arange(T, device=device).unsqueeze(0).expand(M, T)
    sort_idx = torch.where(use_index_axis.unsqueeze(1), index_idx, argsort_idx)
    sorted_vals = torch.gather(col_data, 1, sort_idx)
    rank = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).expand(M, T)

    if op == "time_flip":
        transformed = sorted_vals.flip(dims=[1])

    elif op == "regime_change":
        min_seg = max(4, T // 16)
        valid_hi = T - min_seg
        if valid_hi <= min_seg:
            transformed = sorted_vals
        else:
            # min_seg/valid_hi/valid are scalars (T is shared by every row in
            # this batch), so the only per-row randomness is WHICH num_cp
            # changepoints from the shared candidate range `valid` each row
            # draws -- a random permutation of `valid`'s indices per row
            # (argsort of iid random keys), taking the first (up to) 3.
            valid = torch.arange(min_seg, valid_hi, device=device)
            n_valid = valid.numel()
            max_cp = 3
            num_cp = torch.randint(1, 4, (M,), device=device).clamp(max=n_valid)
            keys = torch.rand(M, n_valid, device=device)
            take = min(max_cp, n_valid)
            order = torch.argsort(keys, dim=1)[:, :take]  # (M, take)
            if take < max_cp:
                # n_valid < max_cp (small-T edge case): pad with an arbitrary
                # in-bounds index (0) -- these extra slots always end up
                # masked out below since num_cp <= n_valid = take in this
                # branch, so slot positions >= take are always >= num_cp too.
                pad = torch.zeros(M, max_cp - take, dtype=order.dtype, device=device)
                order = torch.cat([order, pad], dim=1)
            chosen_pos = valid[order]  # (M, max_cp)
            slot_idx = torch.arange(max_cp, device=device).unsqueeze(0)
            valid_mask = slot_idx < num_cp.unsqueeze(1)
            # Slots beyond this row's num_cp are padded with T -- sorts to
            # the tail (real cp values are all < valid_hi < T) and produces
            # an empty (no-op) trailing segment in the loop below, exactly
            # equivalent to the original's shorter per-row boundaries list.
            chosen_pos = torch.where(valid_mask, chosen_pos, torch.full_like(chosen_pos, T))
            cp_sorted, _ = torch.sort(chosen_pos, dim=1)
            boundaries = torch.cat(
                [
                    torch.zeros(M, 1, dtype=cp_sorted.dtype, device=device),
                    cp_sorted,
                    torch.full((M, 1), T, dtype=cp_sorted.dtype, device=device),
                ],
                dim=1,
            )
            pos = torch.arange(T, device=device).unsqueeze(0)
            transformed = sorted_vals.clone()
            for i in range(boundaries.shape[1] - 1):
                s = boundaries[:, i].unsqueeze(1)
                e = boundaries[:, i + 1].unsqueeze(1)
                in_seg = (pos >= s) & (pos < e)
                seg_count = in_seg.sum(dim=1).clamp(min=1)
                seg_mean = (sorted_vals * in_seg).sum(dim=1) / seg_count
                scale = torch.empty(M, device=device).uniform_(0.8, 1.25)
                shift = torch.randn(M, device=device) * 0.15 * std
                new_vals = (
                    (sorted_vals - seg_mean.unsqueeze(1)) * scale.unsqueeze(1)
                    + seg_mean.unsqueeze(1)
                    + shift.unsqueeze(1)
                )
                transformed = torch.where(in_seg, new_vals, transformed)

    elif op == "shock_recovery":
        lo_t = max(1, T // 16)
        hi_t = max(lo_t + 1, T - T // 16)
        t0 = torch.randint(lo_t, hi_t, (M,), device=device).float()
        mag_abs = torch.empty(M, device=device).uniform_(0.5, 2.0) * std
        sign = torch.where(torch.rand(M, device=device) < 0.5, -1.0, 1.0)
        mag = mag_abs * sign
        half_life = torch.empty(M, device=device).uniform_(0.05, 0.3) * T
        half_life = half_life.clamp(min=1.0)
        decay = torch.exp(-(rank - t0.unsqueeze(1)).clamp(min=0) / half_life.unsqueeze(1))
        transformed = sorted_vals + mag.unsqueeze(1) * decay

    elif op == "amplitude_modulation":
        min_w = max(4, T // 16)
        max_w = max(min_w + 1, T // 2)
        win = torch.randint(min_w, max_w + 1, (M,), device=device)
        span = torch.clamp(T - win, min=1)  # mirrors max(1, T - win)
        start = (torch.rand(M, device=device) * (span + 1).float()).floor().long().clamp(max=span)
        end = (start + win).clamp(max=T)
        pos = torch.arange(T, device=device).unsqueeze(0)
        in_seg = (pos >= start.unsqueeze(1)) & (pos < end.unsqueeze(1))
        seg_count = in_seg.sum(dim=1).clamp(min=1)
        seg_mean = (sorted_vals * in_seg).sum(dim=1) / seg_count
        amp = torch.empty(M, device=device).uniform_(0.5, 1.8)
        new_vals = (sorted_vals - seg_mean.unsqueeze(1)) * amp.unsqueeze(1) + seg_mean.unsqueeze(1)
        transformed = torch.where(in_seg, new_vals, sorted_vals)

    elif op == "differential":
        # Box-average and the two 3-tap derivative kernels are all fixed,
        # tiny (in_channels=out_channels=1) filters -- torch.conv1d hits a
        # surprisingly slow path for that shape on this GPU (~77ms/call,
        # profiled: 24 conv1d calls dominating 1.86s of a 2.3s run), so all
        # 3 are computed via direct slicing/cumsum instead. Mathematically
        # identical to the conv1d formulation (conv1d is cross-correlation,
        # not flipped convolution, so e.g. sk1=[-1,0,1] against a VALID
        # window starting at t is -p[t]+p[t+2] = p[t+2]-p[t]; box-average is
        # a k-wide windowed sum via a padded cumsum difference) -- just
        # without the conv1d dispatch overhead.
        k = max(3, T // 32)
        k = k + 1 if k % 2 == 0 else k
        pad_k = k // 2
        padded = torch.nn.functional.pad(sorted_vals.unsqueeze(1), (pad_k, pad_k), mode="reflect").squeeze(1)
        csum = torch.nn.functional.pad(padded.cumsum(dim=1), (1, 0))  # csum[:,0] = 0
        smoothed = (csum[:, k:] - csum[:, :-k]) / k  # k-wide windowed mean, (M, T)

        # All 4 sub-op candidates are computed for the whole batch (the
        # smoothing/derivative windows are shared across rows since k
        # depends only on T), then gathered per-row by the random sub_op
        # choice -- cheaper and far simpler than masking/grouping rows by
        # sub-op for 4 small per-row computations.
        p1 = torch.nn.functional.pad(smoothed.unsqueeze(1), (1, 1), mode="reflect").squeeze(1)  # (M, T+2)
        raw_d1 = p1[:, 2:] - p1[:, :-2]
        raw_d2 = p1[:, :-2] - 2 * p1[:, 1:-1] + p1[:, 2:]
        int_fwd = torch.cumsum(smoothed, dim=1)
        int_bwd = torch.flip(torch.cumsum(torch.flip(smoothed, dims=[1]), dim=1), dims=[1])
        int_dir = torch.rand(M, 1, device=device) < 0.5
        raw_int = torch.where(int_dir, int_fwd, int_bwd)

        candidates = torch.stack([smoothed, raw_d1, raw_d2, raw_int], dim=1)  # (M, 4, T)
        sub_op = torch.randint(0, 4, (M,), device=device)
        raw = torch.gather(candidates, 1, sub_op.view(M, 1, 1).expand(-1, 1, T)).squeeze(1)

        r_min = raw.min(dim=1).values
        r_max = raw.max(dim=1).values
        s_min = sorted_vals.min(dim=1).values
        s_max = sorted_vals.max(dim=1).values
        degenerate = ~torch.isfinite(r_max - r_min) | ((r_max - r_min) <= 1e-8)
        denom = torch.where(degenerate, torch.ones_like(r_max), r_max - r_min)
        rescaled = (raw - r_min.unsqueeze(1)) / denom.unsqueeze(1) * (s_max - s_min).unsqueeze(1) + s_min.unsqueeze(1)
        transformed = torch.where(degenerate.unsqueeze(1), sorted_vals, rescaled)

    else:
        raise ValueError(f"Unknown structural warp op '{op}'")

    warped = torch.empty_like(col_data)
    warped.scatter_(1, sort_idx, transformed)
    return warped


def apply_structural_feature_warp(x: Tensor, cfg, device) -> Tensor:
    """Optionally apply TempoPFN-inspired transforms — invariances (yflip/
    time_flip), regime changes/shock-recovery, amplitude modulation,
    resample artifacts, differential (smoothing/derivative/integral), and
    discretization (quantize/censor) — to each feature column independently,
    per episode.

    Runs on the GP's INPUT coordinates x (never on sampled outputs y), same
    place and same reasoning as tabiclv2_warp_features/apply_mlp_feature_mixing:
    a deterministic feature map keeps k(f(x_i), f(x_j)) a valid kernel, so
    R_star/Sigma_star computed downstream stay exact. See the module-level
    comment above `_STRUCTURAL_CATEGORIES` for why this differs from
    TempoPFN's own (output-side) augmentor.

    Composition: once a column is gated in (structural_warp_prob), 1 op per
    chosen category is drawn, categories chosen WITHOUT replacement
    (structural_warp_num_ops_min/max out of the 6 categories, weighted by
    structural_warp_category_weights) and applied in the fixed category
    order — mirroring TempoPFN's own num_ops = randint(2,6) category
    composition exactly, same as _sample_structural_ops/_structural_warp_column
    (kept as the reference single-column implementation, still directly used
    by tests and by this function's resample_artifact fallback), just
    computed batched across every gated (episode, column) pair at once
    instead of a Python loop over each one -- see
    _sample_structural_category_mask/_structural_warp_batch. The batched and
    per-column implementations draw from the RNG in a different order/count,
    so outputs at a fixed seed differ between them, but both realize the
    exact same per-row distributions (gate rate, category-selection law,
    sub-op weights, per-op parameter distributions).

    Pseudo-time axis: every order-dependent op in the composition for a given
    column uses value-rank by default (decoupled from the train/test split),
    or the raw row-index axis with probability structural_warp_index_axis_ratio
    when structural_warp_index_axis_enabled is True — see
    _structural_warp_column's docstring. The choice is made ONCE per gated
    column (not per op), so a column's whole composed transform is consistent
    about which axis it treats as "time."

    Args:
        x: (B, T, d) tensor. This repo calls it between tabiclv2_warp_features
            and apply_mlp_feature_mixing; either order is valid.
        cfg: Hydra config; reads cfg.data.structural_warp_* keys, all optional
            via getattr (structural_warp_enabled defaults False -> exact
            no-op for every existing config/dataset).
        device: unused; kept for signature symmetry with
            apply_mlp_feature_mixing so call sites don't special-case this.

    Returns:
        (B, T, d) tensor, same shape/dtype as x.
    """
    if not bool(getattr(cfg.data, "structural_warp_enabled", False)):
        return x

    prob = float(getattr(cfg.data, "structural_warp_prob", 0.3))
    if prob <= 0.0:
        return x

    category_weights = dict(getattr(cfg.data, "structural_warp_category_weights", _DEFAULT_CATEGORY_WEIGHTS))

    num_ops_max = int(getattr(cfg.data, "structural_warp_num_ops_max", 6))
    num_ops_max = max(1, min(num_ops_max, len(_STRUCTURAL_CATEGORIES)))
    num_ops_min = int(getattr(cfg.data, "structural_warp_num_ops_min", 2))
    num_ops_min = max(1, min(num_ops_min, num_ops_max))

    index_axis_enabled = bool(getattr(cfg.data, "structural_warp_index_axis_enabled", False))
    index_axis_ratio = float(getattr(cfg.data, "structural_warp_index_axis_ratio", 0.0))

    B, T, d = x.shape
    dev = x.device

    # Per-(episode, column) Bernoulli(prob) gate, batched in one draw instead
    # of B*d individual torch.rand(1).item() calls (the dominant cost this
    # function used to pay -- see the profiling that motivated this rewrite).
    gate = torch.rand(B, d, device=dev) < prob
    gated_idx = gate.reshape(-1).nonzero(as_tuple=True)[0]
    if gated_idx.numel() == 0:
        return x.clone()  # matches the original's unconditional x.clone() up front

    if index_axis_enabled:
        axis_gate = torch.rand(B, d, device=dev) < index_axis_ratio
    else:
        axis_gate = torch.zeros(B, d, dtype=torch.bool, device=dev)

    # (B, T, d) -> (B*d, T): row r = b*d + col holds column (b, col)'s
    # T-length pseudo-series, matching gate.reshape(-1)'s (b, col) -> b*d+col
    # flattening. .reshape (not .view) since the permute makes this
    # non-contiguous; .clone() makes `flat` independent of `x`'s storage so
    # mutating it below can never alias the input.
    flat = x.permute(0, 2, 1).reshape(B * d, T).clone()
    gated_cols = flat[gated_idx]
    use_index_axis = axis_gate.reshape(-1)[gated_idx]

    M = gated_idx.numel()
    chosen_mask, eligible = _sample_structural_category_mask(
        M, category_weights, num_ops_min, num_ops_max, dev
    )

    # Fixed, small (<=6) loop over categories -- not over the (up to B*d)
    # gated columns -- applying each category's op(s) batched across
    # whichever subset of the M gated columns selected it this call.
    for category in _STRUCTURAL_CATEGORIES:
        if category not in eligible:
            continue
        cat_col = eligible.index(category)
        cat_mask = chosen_mask[:, cat_col]
        if not bool(cat_mask.any()):
            continue
        sel = cat_mask.nonzero(as_tuple=True)[0]
        candidates = _CATEGORY_OPS[category]
        if len(candidates) == 1:
            op = candidates[0]
            gated_cols[sel] = _structural_warp_batch(gated_cols[sel], op, use_index_axis[sel])
        else:
            sub_weights_map = _CATEGORY_SUB_OP_WEIGHTS.get(category)
            if sub_weights_map is None:
                sub_w = torch.ones(len(candidates))
            else:
                sub_w = torch.tensor([sub_weights_map[c] for c in candidates], dtype=torch.float32)
            sub_w = sub_w / sub_w.sum()
            picks = torch.multinomial(sub_w, sel.numel(), replacement=True)
            for k_op, op in enumerate(candidates):
                op_sel = sel[picks == k_op]
                if op_sel.numel() == 0:
                    continue
                gated_cols[op_sel] = _structural_warp_batch(gated_cols[op_sel], op, use_index_axis[op_sel])

    flat[gated_idx] = gated_cols
    warped_x = flat.reshape(B, d, T).permute(0, 2, 1).contiguous()
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


def apply_mlp_feature_mixing(
    x: Tensor, cfg, device, *, return_gate: bool = False
) -> Tensor | tuple[Tensor, Tensor]:
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
        return_gate: if True, also return the (B,) bool tensor recording
            which episodes were actually mixed (used by generate_gp_batch's
            return_kernel_metadata=True path to report mlp-mixing usage per
            episode). Default False preserves the original single-tensor
            return type/behavior for every existing call site.

    Returns:
        (B, T, d) tensor, same shape/dtype as x. Episodes not selected by the
        per-episode Bernoulli gate (mlp_mixing_prob) are returned unchanged.
        If return_gate=True, returns (x, gate) instead, where gate is a (B,)
        bool tensor (all False when mixing is disabled/no-op).
    """
    if not bool(getattr(cfg.data, "mlp_mixing_enabled", False)):
        if return_gate:
            return x, torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        return x

    mixing_prob = float(getattr(cfg.data, "mlp_mixing_prob", 0.3))
    if mixing_prob <= 0.0:
        if return_gate:
            return x, torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
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

    gate_1d = torch.rand(B, device=device) < mixing_prob  # (B,)
    gate = gate_1d[:, None, None]  # (B,1,1)
    # (B,1,1) is required for correct broadcast against (B,T,d); a bare (B,)
    # shape misaligns on the trailing (T, d) dims instead of the batch dim.
    x_out = torch.where(gate, x_mixed, x)
    if return_gate:
        return x_out, gate_1d
    return x_out


# ---------------------------------------------------------------------------
# z_train corruption (robustness augmentation, see conf/data/gp_tasks.yaml)
# ---------------------------------------------------------------------------
#
# Motivation (see plots/plot_spatial_correlation_diagnostics.py's real-mode
# diagnostic): CopulaTabICL is trained exclusively on the EXACT closed-form
# GP-LOO whitened residual computed just below, but at deployment on any
# dataset without a known generating kernel (e.g. real ERA5), z_train can
# only be estimated via src/pit.py::run_pit's K-fold TabICL-marginal quantile
# PIT -- measured to recover only a fraction of the correlation with the
# true whitened residual, flat across k_folds (not a fold-size artifact). The trained model turned out to have essentially zero
# tolerance for this: real-context predictions collapsed toward ~0
# correlation (near-neighbour truth 0.97 -> predicted 0.05-0.13). This is a
# train/deploy distribution-shift problem, not a lengthscale-prior or
# evaluation-methodology problem (both were ruled out).
#
# Fix: blend z_train toward i.i.d. N(0, 1) noise at generation time so the
# model never gets to rely on it being perfectly whitened -- standard
# input-noise-augmentation logic, with the blend strength (see
# DEFAULT_Z_CORRUPTION_RHO_BETA_A/B below) calibrated to the measured
# real-world signal correlation above. Applied here (inside batch
# generation) rather than in train.py: it's a modulation of z_train read
# fresh from cfg.data on every call, the same convention as
# sign_modulation_component_prob/mlp_mixing_enabled/etc, so it applies
# uniformly and invisibly to every caller of generate_gp_batch (the disk
# pipeline, live_generation, and validation/kernel-probe batches alike) with
# no separate wiring in train.py. This is safe here specifically because
# cfg.data.oracle_mode="prior" decouples the training TARGET
# (R_star = kernel(x_test, x_test)) from z_train's realized values entirely,
# so corrupting z_train never requires inventing a different loss target --
# the task stays "predict the same R_star", just from a noisier context
# signal.
DEFAULT_Z_CORRUPTION_RHO_BETA_A = 2.0
DEFAULT_Z_CORRUPTION_RHO_BETA_B = 3.0


def corrupt_z_train(z_train: Tensor, data_cfg) -> Tensor:
    """Randomly corrupt a batch of exact GP-LOO z_train toward i.i.d. N(0, 1)
    noise, per data_cfg.z_train_corruption_* knobs (see
    conf/data/gp_tasks.yaml). No-op (returns z_train unchanged) unless
    data_cfg.z_train_corruption_enabled is True.

    Per corrupted episode:
        z_corrupted = sqrt(rho) * z_train + sqrt(1 - rho) * N(0, 1)
    where rho ~ Beta(a, b) is the per-episode "signal fraction" (rho=1 leaves
    z_train untouched; rho=0 fully replaces it with pure noise). Since
    z_train and the i.i.d. noise term are independent and both unit-variance,
    this construction gives corr(z_train, z_corrupted) = sqrt(rho) in
    expectation, which is what the default Beta(2, 3) shape (E[sqrt(rho)]
    ~= 0.61, p10~=0.14, p90~=0.68) is calibrated against -- centered near the
    measured TabICL-marginal-PIT signal correlation, with spread toward both
    a near-clean and a more severely corrupted regime.

    Called on the batched, unpadded (B, P) z_train inside
    _generate_gp_batch_raw -- every episode in one such call shares the same
    P (see that function's docstring), so there is no padding to mask around
    at this stage (unlike a post-collation z_train, which pads across
    episodes of different P).

    Args:
        z_train  : (B, P) exact GP-LOO whitened residual for this call's B
                   episodes (all sharing this call's sampled P).
        data_cfg : cfg.data (Hydra DictConfig) -- see conf/data/gp_tasks.yaml
                   for the z_train_corruption_* keys this reads.

    Returns:
        (B, P) corrupted z_train, same dtype/device as the input.
    """
    # getattr, not data_cfg.get(...): matches every other optional-modulation
    # flag's access pattern in this file (mlp_mixing_enabled,
    # structural_warp_enabled, mean_fn_enabled, sign_modulation_component_prob
    # above) so this also works with the plain (non-OmegaConf) dataclass cfg
    # objects some tests/scripts pass to generate_gp_task/generate_gp_batch
    # (e.g. diag_kernels.py::DataCfg), not just Hydra's DictConfig.
    if not bool(getattr(data_cfg, "z_train_corruption_enabled", False)):
        return z_train

    prob = float(getattr(data_cfg, "z_train_corruption_prob", 0.5))
    if prob <= 0.0:
        return z_train

    beta_a = float(getattr(data_cfg, "z_train_corruption_rho_beta_a", DEFAULT_Z_CORRUPTION_RHO_BETA_A))
    beta_b = float(getattr(data_cfg, "z_train_corruption_rho_beta_b", DEFAULT_Z_CORRUPTION_RHO_BETA_B))

    B, P = z_train.shape
    device = z_train.device

    apply_ep = torch.rand(B, device=device) < prob  # (B,) which episodes get corrupted at all
    if not bool(apply_ep.any()):
        return z_train

    rho = torch.distributions.Beta(beta_a, beta_b).sample((B,)).to(device=device, dtype=z_train.dtype)
    noise = torch.randn(B, P, device=device, dtype=z_train.dtype)

    sqrt_rho = rho.clamp(0.0, 1.0).sqrt().unsqueeze(-1)
    sqrt_1m_rho = (1.0 - rho).clamp(0.0, 1.0).sqrt().unsqueeze(-1)
    z_blend = sqrt_rho * z_train + sqrt_1m_rho * noise

    return torch.where(apply_ep.unsqueeze(-1), z_blend, z_train)


def _max_batch_for_context(B: int, T: int, device: str) -> int:
    """Cap the per-call episode batch at B episodes given context length
    T=P+N, to avoid CUDA OOM in _generate_gp_batch_raw below.

    P/N (and hence T) are resampled per call from wide, independent ranges
    (see conf/data/gp_tasks.yaml's P_max/N_max, currently up to 512/1024),
    while B defaults to cfg.data.shard_size (256) regardless of T -- several
    (B, T, T) float32 buffers are live at once around the K_all_raw -> L_all
    -> K_all Cholesky/reconstruction (see _generate_gp_batch_raw), so peak
    VRAM scales like B*T^2. Most draws keep T small, but an occasional big
    P+N draw at the configured shard_size can exceed available memory even
    though the vast majority of shards generate fine -- this is what
    produced a mid-run `torch.OutOfMemoryError` in _generate_gp_batch_raw's
    `K_all = L_all @ L_all.mT` line after ~600 successful shards.

    Uses live free memory (torch.cuda.mem_get_info) rather than a static
    threshold so it adapts to whatever else is already resident -- notably
    the frozen TabICL marginal + its K-fold forward passes when
    cfg.data.z_train_source="tabicl" is active, which eats a large,
    P-dependent chunk of VRAM on top of the GP machinery here. Returning
    less than the requested B is safe and already part of
    _generate_gp_batch_raw's contract ("may return FEWER than B episodes");
    generate_gp_batch's top-up loop resamples a fresh (smaller, with high
    probability) T and tops up the shortfall.
    """
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return B
    free_bytes, _ = torch.cuda.mem_get_info(device)
    # Empirical: ~6 live (B,T,T) float32 buffers at peak (K_all_raw, L_all,
    # K_all, plus gpytorch/linear_operator's internal copies during
    # covariance_matrix materialization and psd_safe_cholesky's jitter
    # retries); budget only half of free memory as headroom for the
    # TabICL marginal and allocator fragmentation.
    bytes_per_episode = 6 * T * T * 4
    budget = 0.5 * free_bytes
    return max(1, min(B, int(budget // bytes_per_episode)))


def _evaluate_kernel_dense(kernel_obj: gpytorch.kernels.Kernel, x_norm: Tensor) -> Tensor:
    """Evaluate kernel_obj(x_norm) as a dense (B, T, T) tensor. Split out of
    _generate_gp_batch_raw as its own function purely as a stable
    monkeypatch seam: tests can replace this exact call to inject a
    synthetic NotPSDError/LinAlgError and verify the whole-batch
    discard-and-resample fallback in _generate_gp_batch_raw still works
    (see test_generate_gp_batch_raw_discards_batch_on_linalg_error), without
    needing to actually construct a kernel/hyperparameter draw pathological
    enough to trigger one for real."""
    return kernel_obj(x_norm).to_dense()


@torch.no_grad()
def _generate_gp_batch_raw(
    cfg, B: int, device: str = "cpu", *, return_kernel_metadata: bool = False,
    d_override: Optional[int] = None,
    tabicl_model: Optional[torch.nn.Module] = None,
    tabicl_k_folds: int = 10,
    tabicl_split_calib_frac: float = 0.0,
    kernel_weights: Optional[Tensor] = None,
    tabicl_mix_weights: Optional[Tensor] = None,
    marginal_backend: Optional[str] = None,
    marginal_regressor=None,
    marginal_probs_n: int = 99,
) -> List[Dict[str, Tensor]]:
    """Generate up to B GP episodes in a single vectorised call — the
    "raw" worker generate_gp_batch (below) wraps: may return FEWER than B
    episodes, since any episode whose K_all/K_ff Cholesky repair bottomed
    out at an identity placeholder (see _psd_safe_batch/_batched_cholesky's
    `discard` above) is dropped before returning, rather than saved as a
    degenerate placeholder. Call generate_gp_batch instead of this function
    directly unless you specifically want the possibly-short, unpadded
    result.

    All B episodes share one kernel type, one (P, N) size, and one set of
    active_dims/k (all sampled once per call) but have independent
    hyperparameters and feature draws — gpytorch's `batch_shape=[B]` kernels
    draw B independent hyperparameter sets in one call (see
    _sample_episode_kernel), and a batch_shape=[B] GaussianLikelihood

    All B episodes share one kernel type, one (P, N) size, and one set of
    active_dims/k (all sampled once per call) but have independent
    hyperparameters and feature draws — gpytorch's `batch_shape=[B]` kernels
    draw B independent hyperparameter sets in one call (see
    _sample_episode_kernel), and a batch_shape=[B] GaussianLikelihood
    (_build_likelihood) draws B independent noise values the same way.
    Sampling goes through gpytorch's own MultivariateNormal machinery (see
    the max_cholesky_size discussion in the module docstring) rather than
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
            name, hyperparameters (l, alpha2, nugget, period, rq_alpha, power,
            l_b, alpha2_b, period_b, rq_alpha_b, power_b), kernel_feature_indices,
            mlp_mixed (bool — whether the mlp-mixing gate fired for that
            episode; see apply_mlp_feature_mixing), and the ephemeral
            _L_ff/_alpha Cholesky factors — the schema
            generate_gp_task/pit.py::gp_analytical_pit/diag_kernels.py need.
            Off by default so the production shard schema
            (generate_pit_dataset.py) is unaffected.
        tabicl_model: if given, z_train is overridden with this frozen
            TabICL's own K-fold PIT (pit.py::run_pit_batched) instead of the
            exact analytic GP-LOO residual — see cfg.data.z_train_source in
            conf/data/gp_tasks.yaml. z_test/log_pdf_test are untouched
            (still the oracle values); only the ICL conditioning input
            z_train changes. None (default) preserves the exact analytic
            pipeline.
        tabicl_k_folds: K-fold count for tabicl_model's PIT, ignored when
            tabicl_model is None or tabicl_split_calib_frac > 0.
        tabicl_split_calib_frac: if > 0 (and tabicl_model is given), replaces
            the K-fold PIT above with pit.py::run_pit_calib_split_batched's
            one-pass calibration-split PIT instead: an extra pool of
            round(tabicl_split_calib_frac * P) points is drawn from the same
            episode purely to serve as TabICL's context (see
            cfg.data.z_train_split_calib_frac in conf/data/gp_tasks.yaml),
            scores every training point in a single forward pass, and is
            discarded before the episode is packed. 0.0 (default) preserves
            the K-fold behaviour above.
            Unlike the K-fold override, this is NOT a surgical "only z_train
            changes" swap: the per-episode x_norm z-score (below) is
            deliberately computed jointly over train+test+calibration (so
            TabICL sees the same well-scaled input it would in any other
            call here, benefiting from the larger sample when
            tabicl_split_calib_frac is large), which means x_norm_train/
            x_norm_test/K_ff/K_ss/mu_star/sigma_star/R_star/z_test/
            log_pdf_test/the pre-override analytic z_train all pick up a
            small, deliberate dependence on tabicl_split_calib_frac too —
            still a fully valid, internally self-consistent GP episode
            (x_norm and every value derived from it come from the exact same
            single normalization pass), just not byte-identical to a
            tabicl_split_calib_frac=0 run at the same seed. Renormalizing
            x_norm_train/x_norm_test a second time after dropping the
            calibration block would "fix" that byte-identity at the cost of
            decoupling the saved coordinates from the already-computed
            oracle/K_ff (which were derived from the first, calibration-
            inclusive scale) — a real inconsistency, so this deliberately
            does NOT do that.
        kernel_weights: optional `_COMPOSABLE_KERNELS`-ordered sampling-weight
            tensor forwarded to _sample_kernel_chain_structure/
            _resolve_kernel_name (see their docstrings) — None reproduces
            today's uniform sampling exactly.
        tabicl_mix_weights: optional `_COMPOSABLE_KERNELS`-ordered per-family
            probability tensor (see _tabicl_mix_prob_for_kernel and
            data.z_train_tabicl_mix_* in conf/data/gp_tasks.yaml) gating
            whether THIS call's z_train override actually fires, instead of
            unconditionally substituting tabicl_model's PIT whenever it's
            given. None (default) preserves the legacy always-on behavior of
            tabicl_model/tabicl_split_calib_frac above exactly.
        marginal_backend: if given (and not "tabicl"), overrides z_train
            AND z_test/log_pdf_test using eval/spatial/marginal_backends.py's
            generic {loo_pit, quantiles} contract instead of tabicl_model's
            batched pit.py path -- currently "exaone"/"tabpfn" (see
            data.z_train_source in conf/data/gp_tasks.yaml). Mutually
            exclusive with tabicl_model/tabicl_split_calib_frac (callers set
            at most one of the two override mechanisms). None (default)
            preserves the exact analytic pipeline.
        marginal_regressor: the backend's fitted-per-call regressor object
            (eval/spatial/marginal_backends.py::make_regressor(marginal_backend,
            device=...)), reused across every episode/fold in this call —
            ignored unless marginal_backend is given.
        marginal_probs_n: quantile grid size for marginal_backend's PIT
            (ignored for "tabicl"/None, which always use TabICL's own native
            999-level grid). Default 99 matches
            debug/stages/s7b_backend_train.py's DEFAULT_PROBS_N.

    Returns:
        list of B episode dicts ready for torch.save.
    """
    seed = getattr(cfg, "seed", None)
    if seed is not None:
        _seed_everything(seed)

    # d_override lets generate_gp_batch's top-up rounds (which reseed with a
    # different cfg.seed to escape a degenerate draw) pin d to the value the
    # shard's first round already committed to. Unlike kernel_name/P/N —
    # which vary freely across rounds because collate_fn pads them — d is a
    # hard tensor axis with no padding, so every episode in a shard must
    # share it (see ShardHomogeneousBatchSampler / _sample_d_features).
    d = d_override if d_override is not None else _sample_d_features(cfg)

    # --- Shared settings for this batch ---
    # systematic_composition (CauKer-style, see _sample_kernel_chain_structure)
    # bypasses cfg.data.kernel/kernels entirely and samples a fresh
    # variable-length kernel chain instead — resolved here (before the
    # kernel_cols/k decision below) since chain components are always drawn
    # from _COMPOSABLE_KERNELS, so _kernel_needs_scalar_input still applies.
    # The "dot_product" branch below still only fires for the degenerate
    # m=1 chain (kernel_name == "dot_product" exactly, no operator) — a
    # multi-component chain that merely includes dot_product falls through
    # to _sample_active_dims like any other composite, since its components
    # must share one active_dims subset (see _build_kernel_chain).
    systematic = bool(getattr(cfg.data, "systematic_composition", False))
    if systematic:
        chain_names, chain_ops, kernel_name = _sample_kernel_chain_structure(
            cfg, kernel_weights=kernel_weights
        )
    else:
        kernel_name = _resolve_kernel_name(cfg, kernel_weights=kernel_weights)
    P = random.randint(cfg.data.P_min, cfg.data.P_max)
    N = random.randint(cfg.data.N_min, cfg.data.N_max)
    # Extra calibration-only pool for tabicl_split_calib_frac > 0 (see this
    # function's docstring) -- drawn jointly with the P train / N test points
    # below (same GP sample, same kernel draw), sliced out as x_norm_calib/
    # y_calib after the joint sample, used once as run_pit_calib_split_batched's
    # TabICL context, and discarded before packing. 0 (default) is a no-op:
    # T falls back to the original P + N and every P_C-indexed slice below is
    # empty.
    #
    # Placed LAST in the point ordering (train, test, then calibration) --
    # NOT because it changes x_norm's values (the per-episode z-score below
    # is a mean/std over dim=1, which is permutation-invariant, so ordering
    # never changes what TabICL/the kernel actually sees) but because of
    # Cholesky's nested-block property: L_all's leading (P+N)x(P+N) block is
    # exactly the Cholesky factor of K_all's own leading (P+N)x(P+N) block,
    # so y_all[:, :P+N] = L_all[:, :P+N, :P+N] @ noise[:, :P+N] depends only
    # on the train+test sub-block, never on the calibration block, as long as
    # calibration sits after train+test rather than between them. Had
    # calibration been sandwiched between train and test, y_test's realized
    # sample would additionally pick up cross-terms against the calibration
    # block's own noise draws through L_all's lower-triangular structure.
    P_C = max(1, round(tabicl_split_calib_frac * P)) if tabicl_split_calib_frac > 0 else 0
    T = P + N + P_C
    # Cap B for this call so the (B, T, T) buffers below fit in free VRAM --
    # see _max_batch_for_context's docstring. Safe to shrink B here: this
    # function is already documented to possibly return fewer than the
    # requested B episodes, and generate_gp_batch's top-up loop assembles
    # the shortfall via additional calls.
    B = _max_batch_for_context(B, T, device)
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
        kernel_obj, component_params, outer_sign_params = _build_kernel_chain(
            cfg, chain_names, chain_ops, k, B, device, active_dims=kernel_cols, d_total=d
        )
        # Legacy zero-sentinel schema (see _sample_episode_kernel's docstring)
        # is kept populated so the rest of this function — nugget sampling,
        # GP prior/posterior machinery, metadata packing loop below — is
        # untouched regardless of mode; the real per-component values live in
        # component_params instead (see the return_kernel_metadata block).
        # The post-fold (outer) sign-modulation params DO belong in this flat
        # schema, same as the non-systematic branch below — they wrap the
        # whole chain, not any one component.
        params = {
            key: torch.zeros(B, device=device)
            for key in (
                "l", "alpha2", "period", "rq_alpha", "power",
                "l_b", "alpha2_b", "period_b", "rq_alpha_b", "power_b",
            )
        }
        params.update(outer_sign_params)
    else:
        kernel_obj, params = _sample_episode_kernel(
            cfg, kernel_name, k, B, device, active_dims=kernel_cols, d_total=d
        )
    likelihood = _build_likelihood(cfg, kernel_name, B, device)
    nugget = likelihood.noise.reshape(B)  # "nugget" name kept for the saved-metadata schema

    # --- Non-zero mean bank (CauKer-inspired, see _sample_mean_module) ---
    # Built once per batch call, independent hyperparameters per episode —
    # same granularity as kernel_obj/likelihood above.
    mean_module, mean_params = _sample_mean_module(cfg, d, B, device)

    # --- Features (B, T, d) ~ N(0, 1), warped, normalised per episode ---
    x_raw = torch.randn(B, T, d, device=device)
    x_raw = tabiclv2_warp_features(x_raw)
    x_raw = apply_structural_feature_warp(x_raw, cfg, device)
    if return_kernel_metadata:
        x_raw, mlp_mixed = apply_mlp_feature_mixing(x_raw, cfg, device, return_gate=True)
    else:
        x_raw = apply_mlp_feature_mixing(x_raw, cfg, device)
    x_norm = (x_raw - x_raw.mean(1, keepdim=True)) / x_raw.std(1, keepdim=True).clamp(min=1e-8)

    # --- Joint prior sample + noisy covariance (B, T, T) ---
    # Built as plain dense kernel + diagonal nugget (kernel_obj(x_norm).to_dense()
    # + nugget*eye), NOT via gpytorch's GaussianLikelihood(MultivariateNormal)
    # .covariance_matrix property this used to go through. That property
    # computes `covar + noise_covar` (_GaussianLikelihoodBase.marginal) by
    # building an AddedDiagLinearOperator (kernel LazyTensor + DiagLinearOperator)
    # — and gpytorch's own __add__/to_dense() for that specific combination
    # eagerly attempts a low-rank Cholesky factorization (add_low_rank ->
    # root_inv_decomposition) of an intermediate partial-sum LinearOperator,
    # for the WHOLE batch at once, before _psd_safe_batch below ever gets a
    # chance to repair anything. dot_product/polynomial have a finite-
    # dimensional feature map (rank <= d_features, often << T = P+N — see
    # composite_exclude_kernels' docstring in conf/data/gp_tasks.yaml), so
    # composite chains built from them used to hit that eager factorization's
    # near-singularity constantly; so could sign modulation and heavily-
    # composed structural warps pushing feature geometry to an extreme (see
    # apply_structural_feature_warp). This raised NotPSDError (or, when
    # root_decomposition falls through to _symeig -> torch.linalg.eigh and
    # THAT LAPACK routine fails to converge, torch.linalg.LinAlgError —
    # observed killing whole worker processes in production runs, job
    # 3000710) straight out of kernel evaluation, discarding the ENTIRE
    # B-episode batch (every other, perfectly fine episode's progress along
    # with it) and forcing generate_gp_batch's top-up loop to resample a
    # fresh kernel/hyperparameter draw from scratch — by far the most
    # expensive of this pipeline's failure modes.
    #
    # _build_kernel_chain/_sample_episode_kernel now combine composite
    # components via _DenseComposedKernel (plain dense tensor +/*, converting
    # each side with .to_dense() before combining — the same pattern
    # SignModulatedKernel.forward already used for its own base_kernel call)
    # instead of gpytorch's Kernel.__add__/__mul__, so add_low_rank's
    # RootLinearOperator trigger (LinearKernel/dot_product's Gram matrix is
    # exactly that) is no longer reachable from any composition point this
    # pipeline builds — the known cause of the job 3000710 crash is
    # structurally eliminated, not just caught. _evaluate_kernel_dense is
    # still wrapped in a try/except as defence in depth (a future kernel
    # addition, or a bug in _DenseComposedKernel itself, could still raise
    # here) — plain dense tensor addition below can't raise either error, so
    # only the to_dense() kernel evaluation itself needs the guard; no
    # factorization happens until we explicitly call _psd_safe_batch just
    # below, which isolates and repairs (or discards) failures per-episode
    # instead of per-batch. max_cholesky_size is still forced high (see
    # module docstring / _MAX_CHOLESKY) for the to_dense() kernel evaluation
    # itself, matching every other gpytorch call in this file that sees a
    # full (T, T) matrix.
    with gpytorch.settings.max_cholesky_size(_MAX_CHOLESKY):
        try:
            K_full_dense = _evaluate_kernel_dense(kernel_obj, x_norm)  # (B, T, T), no nugget yet
        except (NotPSDError, torch.linalg.LinAlgError):
            warnings.warn(
                f"_generate_gp_batch_raw: kernel evaluation for this "
                f"{B}-episode batch (kernel={kernel_name!r}) raised NotPSDError "
                f"or LinAlgError; discarding the whole batch and resampling.",
                RuntimeWarning,
            )
            return []
    nugget_eye = torch.eye(T, device=device, dtype=K_full_dense.dtype).expand(B, T, T)
    K_all_raw = K_full_dense + likelihood.noise.reshape(B, 1, 1) * nugget_eye

    # No explicit symmetrization needed here: torch.linalg.cholesky_ex (used
    # by psd_safe_cholesky below) only ever reads the lower triangle of its
    # input and ignores the upper triangle entirely (verified — corrupting
    # the upper triangle changes nothing about the result), and the K_all we
    # actually use downstream is reconstructed as L_all @ L_all.mT, which is
    # exactly symmetric by construction regardless of what K_all_raw's upper
    # triangle looked like. K_all_raw is NOT guaranteed PSD though: gpytorch's
    # own float32 kernel evaluation accumulates enough rounding error across
    # a long composite/systematic chain (worse once SignModulatedKernel's
    # elementwise +-1 factor is in the mix) to occasionally leave a slightly
    # negative eigenvalue. psd_safe_cholesky is gpytorch/linear_operator's
    # own canonical PSD-repair tool — the same escalating-jitter mechanism
    # gpytorch falls back to internally (see the "added jitter... Using
    # symeig method" warning this pipeline already emits for marginal
    # episodes), applied here explicitly, ONCE, so that y_all (the actual
    # sample) and K_all/K_ss/K_ff/R_star (the reported covariance/oracle)
    # are both derived from the exact same, provably-PSD matrix rather than
    # two independently-reconstructed quantities that could disagree at the
    # float32 rounding level. Replaces the old `noisy_dist.rsample()`, which
    # offered no such guarantee for the *reported* K_all.
    L_all, failed_all = _psd_safe_batch(K_all_raw)
    K_all = L_all @ L_all.mT                          # (B, T, T), PSD by construction
    y_all = (L_all @ torch.randn(B, T, 1, device=device)).squeeze(-1)  # zero-mean GP sample
    # Add the (possibly all-zero) mean bank on top — a deterministic function
    # of x_norm alone, so this stays an exact GP(mean_module, K_all) sample;
    # oracle_mode branches below read mu_star off this same mean_module, not
    # off a hardcoded zero, so z_train/z_test stay correctly calibrated (see
    # _sample_mean_module's docstring).
    y_all = y_all + mean_module(x_norm)

    x_norm_train = x_norm[:, :P]                     # (B, P, d)
    x_norm_test  = x_norm[:, P:P + N]                # (B, N, d)
    x_norm_calib = x_norm[:, P + N:]                 # (B, P_C, d) -- tabicl_split PIT context only
    y_train      = y_all[:,  :P]                     # (B, P)
    y_test       = y_all[:,  P:P + N]                # (B, N)
    y_calib      = y_all[:,  P + N:]                 # (B, P_C)

    # --- Sub-matrices of K_all (nugget already on diagonal) ---
    K_ff = K_all[:, :P, :P]         # (B, P, P) -- P_C never enters K_ff/LOO/oracle
    K_ss = K_all[:, P:P + N, P:P + N]  # (B, N, N)

    # --- LOO PIT always needs L_ff/alpha from K_ff (R&W Eq. 5.12), regardless
    # of which oracle drives the test-side R_star/mu_star/sigma_star below.
    # No clean gpytorch public API exposes diag(K_ff^-1) for an ExactGP, so
    # this stays hand-rolled (_batched_cholesky), just sourced from the
    # gpytorch-native K_all above instead of a separately hand-added nugget.
    # Eq. 5.12 is derived for a zero-mean joint Gaussian: alpha must be
    # K_ff^-1 @ (y_train - mean_train), not K_ff^-1 @ y_train directly, or
    # the LOO residual silently carries a leftover K_ff^-1 @ mean_train term
    # whenever mean_module is non-zero (see _sample_mean_module's docstring —
    # z_test already goes through mean_module via mu_star, so this keeps
    # z_train consistent with it).
    L_ff, failed_ff = _batched_cholesky(K_ff)
    mean_train = mean_module(x_norm_train)                                        # (B, P)
    alpha = torch.cholesky_solve((y_train - mean_train).unsqueeze(-1), L_ff).squeeze(-1)  # (B, P)

    # Episodes where either Cholesky repair above bottomed out at an
    # identity placeholder are not valid GP episodes (K_all/K_ff no longer
    # reflect the sampled kernel at all) — dropped entirely at the end of
    # this function (see the "Discard degenerate episodes" block below)
    # rather than saved with a degenerate placeholder. generate_gp_batch
    # (the public wrapper) tops up the shortfall so callers still get
    # exactly B valid episodes.
    discard = failed_all | failed_ff

    oracle_mode = getattr(cfg.data, "oracle_mode", "prior")
    if oracle_mode == "prior":
        # Prior oracle: ignore training conditioning — R_star reflects the raw
        # kernel structure among test points; mu_star is the GP prior mean (0).
        # No conditioning needed.
        # Sigma_star = K_ss is already guaranteed PSD here: K_ss is a
        # principal submatrix of K_all, which is constructed above as
        # L_all @ L_all.mT (PSD by construction, via psd_safe_cholesky) —
        # not a slice of an unprotected raw materialization. A principal
        # submatrix of a PSD matrix is itself PSD, so no further repair is
        # needed at this point.
        mu_star    = mean_module(x_norm_test)
        Sigma_star = K_ss
    else:
        # oracle_mode="posterior" (GP posterior conditioned on x_train/y_train
        # via a Schur complement) was removed: the float64 Schur-complement
        # computation followed by a cast back to float32 could still leave
        # R_star's minimum eigenvalue below the well-conditioned/PSD floors
        # for composite "systematic composition" kernels (observed min eig as
        # low as -1.8e-5 in data/systematic_composition-k5-posterior/pit),
        # and _generate_gp_batch_raw's discard mask never checked Sigma_star/
        # R_star's own eigenvalues to catch it. Only 'prior' is supported for
        # now — see git history for the removed implementation.
        raise ValueError(f"Unknown data.oracle_mode '{oracle_mode}'; only 'prior' is supported.")
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
    # y_test was actually drawn from (noisy_dist above). With oracle_mode="prior"
    # the only supported mode, this is always identical to R_star; kept as its
    # own field for schema stability (downstream plots/tests read R_prior).
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
    #
    # z_std itself is NaN whenever z_train contains a NaN/Inf (e.g. alpha or
    # K_inv_diag blowing up on a near-singular K_ff that jitter didn't fully
    # fix), and NaN comparisons are always False in PyTorch — so the std
    # threshold check below silently misses exactly the episodes it most
    # needs to catch unless non-finite z_train is checked for explicitly.
    non_finite = ~torch.isfinite(z_train).all(dim=1)
    z_std = z_train.std(dim=1)
    degen = non_finite | (z_std < 0.1) | (z_std > 3.0)
    if degen.any():
        warnings.warn(
            f"generate_gp_batch: {int(degen.sum())}/{B} episodes have degenerate LOO z "
            f"({int(non_finite.sum())} non-finite) and will be discarded.",
            RuntimeWarning,
        )
    # Fold into the same discard mask as the Cholesky-failure episodes above
    # (see `discard`/`keep` below) — previously computed but never applied,
    # so degenerate/NaN z_train episodes were saved to disk and only
    # surfaced much later as a training-time crash.
    discard = discard | degen

    # If every active dimension a kernel actually reads collapses to a
    # near-constant value, the kernel's r=0 for every point pair and R_star
    # silently becomes a constant, uninformative correlation matrix instead
    # of reflecting the sampled kernel at all. This has been observed via
    # more than one independent upstream cause (a structural-warp "censor"
    # quantile-index collision — now guarded in _structural_warp_column —
    # and mlp_mixing's ReLU/sigmoid saturating a unit to one value for every
    # point, which is ordinary "dead ReLU" behaviour, not a bug, and can't be
    # prevented at the source), so this is checked post-hoc here rather than
    # patched at each possible cause. x_norm is per-episode z-normalised
    # already (see above), so a healthy column has std ~= 1.0 and a
    # collapsed one has std ~= 0.0 -- no ambiguous middle ground to threshold
    # carefully.
    #
    # Only ALL-active-dims-collapsed counts as degenerate here: kernels
    # capped to k=1 (periodic/cosine) have no other dimension to fall back
    # on, so any collapse is total. Kernels with k>1 active dims (including
    # kernel_cols=None, meaning full ARD over every one of the d columns)
    # still vary through their other active dims if only some collapse —
    # that's reduced effective dimensionality, not a broken episode, so only
    # the all-collapsed case (the true multi-dim analogue of the k=1 bug) is
    # discarded.
    active_cols = kernel_cols if kernel_cols is not None else list(range(d))
    active_stds = x_norm[:, :, active_cols].std(dim=1)             # (B, len(active_cols))
    degenerate_active_col = (active_stds.max(dim=1).values) < 1e-4
    if degenerate_active_col.any():
        warnings.warn(
            f"generate_gp_batch: {int(degenerate_active_col.sum())}/{B} episodes have a "
            f"degenerate (near-constant) active kernel column and will be discarded.",
            RuntimeWarning,
        )
    discard = discard | degenerate_active_col

    # Reconstruct full posterior covariance (for Y-space oracle)
    Sigma_full = R_star * sigma_star.unsqueeze(1) * sigma_star.unsqueeze(2)       # (B, N, N)

    # z_train source override (cfg.data.z_train_source="tabicl"/"tabicl_split"):
    # replace the exact analytic GP-LOO residual with the real frozen
    # TabICL's own PIT -- trains/evaluates against the same approximate
    # marginal real (non-GP) deployment data would produce, instead of the
    # closed-form oracle. Deliberately AFTER the degenerate-episode z_std
    # check above (discard decisions are always made on the exact residual,
    # never on the substituted one) and BEFORE corrupt_z_train, which is
    # free to further blend either source toward noise. y_train (and,
    # for tabicl_split, y_calib) is z-scored per episode first -- run_pit_*
    # (like every other run_pit call site) does no target scaling of its
    # own, and episode y_train's scale is unconstrained (outputscale ~
    # GammaPrior), so an unscaled call risks saturating the pretrained
    # quantile head's CDF into its extreme tail.
    # Per-call real-TabICL gate (data.z_train_tabicl_mix_* — see
    # _tabicl_mix_prob_for_kernel's docstring): tabicl_mix_weights is None
    # for every caller before this feature existed, so apply_tabicl reduces
    # to the legacy "unconditionally use tabicl_model whenever given" check
    # with NO extra random.random() draw in that case — keeps the RNG stream
    # (and hence every existing z_train_source=tabicl/tabicl_split caller's
    # exact reproducibility at a fixed seed) byte-identical to before this
    # feature existed.
    if tabicl_mix_weights is not None:
        apply_tabicl = tabicl_model is not None and (
            random.random() < _tabicl_mix_prob_for_kernel(kernel_name, tabicl_mix_weights)
        )
    else:
        apply_tabicl = tabicl_model is not None

    if marginal_backend in ("exaone", "tabpfn"):
        # Genuine multi-episode BATCHED PIT -- the whole B-episode call
        # becomes (k_folds+1) fused forwards total, not B*(k_folds+1)
        # separate ones (see eval/spatial/exaone_batched.py /
        # tabpfn_batched.py's module docstrings for how each backend exposes
        # a batchable forward, and how this was verified equivalent to the
        # per-episode fallback below). y_train/y_test are z-scored per
        # episode first, same reasoning as the tabicl branch below.
        y_mean = y_train.mean(dim=1, keepdim=True)
        y_std = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
        y_train_s = ((y_train - y_mean) / y_std).detach().cpu().numpy()
        y_test_s = ((y_test - y_mean) / y_std).detach().cpu().numpy()
        x_train_np = x_norm_train.detach().cpu().numpy()
        x_test_np = x_norm_test.detach().cpu().numpy()
        base_seed = int(getattr(cfg, "seed", None) or 0)
        if marginal_backend == "exaone":
            from eval.spatial.exaone_batched import exaone_run_pit_batched as _run_batched
        else:
            from eval.spatial.tabpfn_batched import tabpfn_run_pit_batched as _run_batched
        out = _run_batched(
            marginal_regressor, x_train_np, y_train_s, x_test_np, y_test_s,
            k_folds=tabicl_k_folds, probs_n=marginal_probs_n, seed=base_seed,
        )
        z_train = torch.from_numpy(out["z_train"]).to(device=device)
        z_test = torch.from_numpy(out["z_test"]).to(device=device)
        # Jacobian correction back to raw-y-space nats, same convention as
        # the tabicl branch below.
        log_pdf_test = torch.from_numpy(out["log_pdf_test"]).to(device=device) - y_std.log()  # (B,N) - (B,1) broadcast
    elif marginal_backend not in (None, "tabicl"):
        # Generic per-episode K-fold PIT fallback for any
        # eval/spatial/marginal_backends entry with no dedicated batched
        # module above (currently "tabfm"/"tabm" -- neither exposes a
        # batchable multi-dataset forward the way exaone/tabpfn do, see
        # their marginal_backends.py docstrings). Unlike the batched branch
        # above, this is a Python loop over the B episodes in THIS call,
        # each costing (k_folds+1) fit/predict calls. Reuses
        # debug/stages/s7b_backend_train.py's validated _generic_pit_episode
        # recipe (loo_pit for z_train, quantiles+compute_pit for
        # z_test/log_pdf_test), inlined here so every generate_gp_batch
        # caller (not just that debug comparison trainer) can reach it via
        # data.z_train_source. Orders of magnitude slower per episode than
        # either batched branch -- see live_dataset.py's LiveGPDataset
        # docstring for measured numbers before using this for a full-scale
        # run.
        from eval.metrics.joint_nll import compute_pit
        from eval.spatial.marginal_backends import loo_pit as _backend_loo_pit
        from eval.spatial.marginal_backends import quantiles as _backend_quantiles

        y_mean = y_train.mean(dim=1, keepdim=True)
        y_std = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
        probs = np.linspace(
            1.0 / (marginal_probs_n + 1), marginal_probs_n / (marginal_probs_n + 1), marginal_probs_n
        )
        base_seed = int(getattr(cfg, "seed", None) or 0)
        z_train_np = np.empty((B, P), dtype=np.float32)
        z_test_np = np.empty((B, N), dtype=np.float32)
        log_pdf_np = np.empty((B, N), dtype=np.float32)
        for b in range(B):
            xc = x_norm_train[b].detach().cpu().numpy()
            xq = x_norm_test[b].detach().cpu().numpy()
            y_std_b = float(y_std[b])
            yc = ((y_train[b] - y_mean[b]) / y_std[b]).detach().cpu().numpy()
            yq = ((y_test[b] - y_mean[b]) / y_std[b]).detach().cpu().numpy()
            seed_b = (base_seed + b) % (2**31)
            z_train_np[b] = _backend_loo_pit(
                marginal_backend, marginal_regressor, xc, yc, probs,
                k_folds=tabicl_k_folds, seed=seed_b,
            )
            q_test = _backend_quantiles(
                marginal_backend, marginal_regressor, xc, yc, xq, probs, seed=seed_b
            )
            z_test_b, log_pdf_b = compute_pit(q_test, probs, yq)
            z_test_np[b] = z_test_b
            # Jacobian correction back to raw-y-space nats, same convention as
            # the tabicl branch below.
            log_pdf_np[b] = log_pdf_b - math.log(y_std_b)
        z_train = torch.from_numpy(z_train_np).to(device=device)
        z_test = torch.from_numpy(z_test_np).to(device=device)
        log_pdf_test = torch.from_numpy(log_pdf_np).to(device=device)
    elif apply_tabicl and tabicl_split_calib_frac > 0:
        # "tabicl_split": one forward pass, x_norm_calib/y_calib (P_C points,
        # never part of the official P-point train set) as TabICL's context
        # -- see pit.py::run_pit_calib_split_batched's docstring. Scaled with
        # y_train's own mean/std so context and query share one scale.
        # x_norm_calib/y_calib are never referenced again after this call,
        # so they're discarded (not packed into `tensors` below) simply by
        # falling out of scope.
        from pit import run_pit_calib_split_batched  # local: pit.py imports from this module

        y_mean = y_train.mean(dim=1, keepdim=True)
        y_std  = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
        y_train_scaled = ((y_train - y_mean) / y_std).unsqueeze(-1)   # (B, P, 1)
        y_calib_scaled = ((y_calib - y_mean) / y_std).unsqueeze(-1)   # (B, P_C, 1)
        split_pit = run_pit_calib_split_batched(
            tabicl_model, x_norm_train, y_train_scaled,
            x_norm_calib, y_calib_scaled,
        )
        z_train = split_pit["z_train"].squeeze(-1)                    # (B, P)
    elif apply_tabicl:
        # "tabicl": K-fold PIT (pit.py::run_pit_batched), scored against the
        # episode's REAL x_norm_test/y_test -- z_train AND z_test/
        # log_pdf_test all come from TabICL's own PIT, not just z_train.
        #
        # Root cause of the val/y_nll_copula positive-NLL investigation
        # (2026-08-24): an earlier version of this branch passed a dummy
        # 1-point X_test/Y_test slice here and kept z_test/log_pdf_test at
        # the oracle values computed above -- training against the exact
        # oracle z_test while conditioning on noisy TabICL z_train taught
        # the model an overconfident Sigma that was well-calibrated for the
        # oracle target but poorly calibrated once scored against TabICL's
        # own (noisier) PIT at validation time. era5_live_dataset.py never
        # had this problem: there's no oracle for real ERA5 data, so it has
        # always PIT-ed z_train AND z_test through the same TabICL call --
        # and ERA5-finetuned checkpoints were the only ones observed with
        # negative (well-calibrated) val/y_nll_copula, confirmed via an A/B
        # copula_nano run (val/y_nll_copula: persistently +0.09..+0.31 with
        # the dummy-test-slice version, vs. persistently ~-0.001..-0.009
        # with this real-test-points version, at every validation
        # checkpoint). Unconditional now (previously gated behind a
        # data.z_train_matched_test flag defaulting to False, which would
        # have reproduced the bug for anyone who forgot to opt in).
        from pit import run_pit_batched  # local: pit.py imports from this module

        y_mean = y_train.mean(dim=1, keepdim=True)
        y_std  = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
        y_train_scaled = ((y_train - y_mean) / y_std).unsqueeze(-1)   # (B, P, 1)
        y_test_scaled = ((y_test - y_mean) / y_std).unsqueeze(-1)     # (B, N, 1)
        tabicl_pit = run_pit_batched(
            tabicl_model, x_norm_train, y_train_scaled,
            x_norm_test, y_test_scaled,
            k_folds=tabicl_k_folds,
        )
        z_train = tabicl_pit["z_train"].squeeze(-1)                   # (B, P)
        z_test = tabicl_pit["z_test"].squeeze(-1)                     # (B, N)
        # Jacobian correction back to raw-y-space nats (log p_raw =
        # log p_scaled - log(std)) -- same convention as
        # era5_live_dataset.py::_pit_episode/_pit_group.
        log_pdf_test = tabicl_pit["log_pdf_test"].squeeze(-1) - y_std.log()  # (B, N) - (B, 1) broadcast

    # Robustness augmentation (opt-in, off by default -- see corrupt_z_train's
    # docstring above): applied AFTER the degenerate-episode z_std check above
    # so discard decisions are always made on the exact GP-LOO residual, never
    # on a corrupted one. Skipped when THIS call's z_train already came from
    # the adaptive real-TabICL mix path (tabicl_mix_weights is not None and
    # apply_tabicl fired): corrupting an already-real approximate signal
    # toward synthetic noise defeats the point of training against the real
    # one -- corrupt_z_train is itself a synthetic proxy for this exact gap
    # (see its docstring's "measured TabICL-marginal-PIT signal correlation"
    # calibration). The legacy always-on z_train_source=tabicl/tabicl_split
    # full-override path (tabicl_mix_weights=None) keeps corrupting on top,
    # unchanged from before this feature existed.
    if not (tabicl_mix_weights is not None and apply_tabicl):
        z_train = corrupt_z_train(z_train, cfg.data)

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

    # Belt-and-braces: any saved field containing NaN/Inf means a numerically
    # degenerate episode slipped past the checks above (e.g. R_star/Sigma_star
    # blowing up from a near-singular posterior covariance — a different
    # failure mode than the LOO z_train check, since it comes from the K_ss/
    # K_st/K_ff test-side conditioning rather than K_ff's LOO diag). NaN/Inf
    # is never a legitimate value regardless of kernel, so this is safe to
    # enforce unconditionally (unlike a min-eigenvalue floor, which would also
    # reject legitimate near-singular-but-finite draws).
    non_finite = torch.zeros(B, dtype=torch.bool)
    for _t in tensors.values():
        non_finite = non_finite | ~_t.reshape(_t.shape[0], -1).isfinite().all(dim=1)
    if non_finite.any():
        warnings.warn(
            f"generate_gp_batch: {int(non_finite.sum())}/{B} episodes contain "
            f"NaN/Inf in a saved field and will be discarded.",
            RuntimeWarning,
        )
    discard = discard | non_finite.to(discard.device)

    n_tr = torch.tensor(P)
    n_te = torch.tensor(N)
    extra: Dict[str, object] = {"n_train": n_tr, "n_test": n_te}

    if return_kernel_metadata:
        # Per-episode (sliceable via val[b]) hyperparameters/factors, plus
        # the batch-shared kernel name and active_dims — the schema
        # generate_gp_task / pit.py::gp_analytical_pit / diag_kernels.py need.
        # sign_applied_outer/sign_w_outer/sign_b_outer/sign_a_outer
        # (post-composition sign modulation — see
        # _sample_episode_kernel/_build_kernel_chain) are always present in
        # `params` (zero-filled when not applied), same 0.0-sentinel
        # convention as period/rq_alpha/power above. The per-component
        # sign_applied/sign_w/sign_b/sign_a (bare or non-systematic composite
        # kernel_name) are likewise always in `params` for the non-systematic
        # path -- systematic chains instead carry their per-component sign
        # fields inside kernel_component_params below.
        flat_keys = [
            "l", "alpha2", "period", "rq_alpha", "power",
            "l_b", "alpha2_b", "period_b", "rq_alpha_b", "power_b",
            "sign_applied_outer", "sign_w_outer", "sign_b_outer", "sign_a_outer",
        ]
        if not systematic:
            flat_keys += ["sign_applied", "sign_w", "sign_b", "sign_a"]
            if _parse_composite(kernel_name) is not None:
                flat_keys += ["sign_applied_b", "sign_w_b", "sign_b_b", "sign_a_b"]
        for key in flat_keys:
            tensors[key] = params[key].cpu()
        tensors["nugget"] = nugget.cpu()
        tensors["mlp_mixed"] = mlp_mixed.cpu()
        for key in (
            "mean_weight", "mean_bias", "mean_nonzero", "mean_family", "mean_linear",
            "mean_exp_direction", "mean_exp_rate", "mean_exp_scale",
            "mean_anomaly_direction", "mean_anomaly_threshold", "mean_anomaly_magnitude",
        ):
            tensors[key] = mean_params[key].cpu()
        tensors["_L_ff"] = L_ff
        tensors["_alpha"] = alpha
        extra["kernel"] = kernel_name
        extra["kernel_feature_indices"] = torch.tensor(
            kernel_cols if kernel_cols is not None else list(range(d)), dtype=torch.long
        )

    # --- Discard degenerate episodes (see `discard` above) rather than
    # saving an identity-placeholder K_all/K_ff/R_star. n_train/n_test/
    # kernel/kernel_feature_indices in `extra` are batch-shared (same P, N,
    # kernel_name, active_dims for every episode in this call — see the top
    # of this function), so they need no filtering; only the per-episode
    # `tensors` (and, for systematic chains, `component_params`) do.
    keep = ~discard
    B_kept = int(keep.sum())
    # tensors dict mixes CPU (most fields, .cpu()'d above) and device-resident
    # (_L_ff/_alpha, kept on `device` for reuse elsewhere) tensors, so index
    # each with a copy of `keep`/`discard`'s boolean mask moved to its own
    # device rather than a single fixed-device index tensor.
    tensors = {key: val[keep.to(val.device)] for key, val in tensors.items()}
    if return_kernel_metadata and systematic:
        component_params = [
            {pk: pv[keep.to(pv.device)] for pk, pv in comp.items()} for comp in component_params
        ]

    episodes = [
        {key: val[b] for key, val in tensors.items()} | extra
        for b in range(B_kept)
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
        for b in range(B_kept):
            episodes[b]["kernel_components"] = chain_names
            episodes[b]["kernel_ops"] = chain_ops
            episodes[b]["kernel_component_params"] = [
                {pk: pv[b].cpu() for pk, pv in comp.items()} for comp in component_params
            ]

    return episodes


def generate_gp_batch(
    cfg, B: int, device: str = "cpu", *, return_kernel_metadata: bool = False,
    tabicl_model: Optional[torch.nn.Module] = None,
    tabicl_k_folds: int = 10,
    tabicl_split_calib_frac: float = 0.0,
    d_override: Optional[int] = None,
    kernel_weights: Optional[Tensor] = None,
    tabicl_mix_weights: Optional[Tensor] = None,
    marginal_backend: Optional[str] = None,
    marginal_regressor=None,
    marginal_probs_n: int = 99,
) -> List[Dict[str, Tensor]]:
    """Generate exactly B GP episodes, discarding and regenerating any that
    turn out degenerate (see _generate_gp_batch_raw's `discard` — an
    unrecoverable K_all/K_ff Cholesky, i.e. even psd_safe_cholesky/
    _batched_cholesky's escalating jitter bottomed out at an identity
    placeholder) instead of saving a placeholder episode.

    Every caller (generate_pit_dataset.py, train.py, tests, ...) relies on
    getting exactly B episodes back: dataset.py's CopulaDataset._get_sharded
    indexes shards with a fixed stride (idx // shard_size), so a shard
    silently written with fewer than shard_size episodes would corrupt
    indexing for every shard after it. This wrapper preserves that
    invariant by topping up the shortfall with fresh top-up calls (which
    resample their own kernel/P/N/active_dims independently, same as any
    other call — collate_fn pads over P/N and doesn't care about kernel)
    until exactly B valid episodes are assembled. d_features is the one
    exception: it is pinned to the first round's value (see d_override)
    because it's an unpadded tensor axis, not row-masked like P/N — a
    top-up round resampling its own d would silently mix feature counts
    within one shard, which ShardHomogeneousBatchSampler/collate_fn assume
    can't happen.

    d_override lets a caller that itself assembles one shard across multiple
    generate_gp_batch calls (generate_pit_dataset.py's
    _generate_shard_with_oom_retry, which splits into smaller chunks on CUDA
    OOM) pin every chunk to the same d — without it, each chunk call would
    independently sample its own d via _sample_d_features and the shard could
    end up with the same kind of internally-mixed feature counts this
    function already prevents across its own top-up rounds.

    In practice this loop almost never repeats more than once: the discard
    rate is astronomically rare (an episode has to defeat escalating jitter
    up to ~0.1 — see _psd_safe_batch/_batched_cholesky). max_rounds bounds
    the retries so a pathological config (e.g. one that's non-PSD by
    construction regardless of jitter) fails loudly instead of hanging.
    """
    base_seed = getattr(cfg, "seed", None)
    episodes = _generate_gp_batch_raw(
        cfg, B, device, return_kernel_metadata=return_kernel_metadata,
        d_override=d_override,
        tabicl_model=tabicl_model, tabicl_k_folds=tabicl_k_folds,
        tabicl_split_calib_frac=tabicl_split_calib_frac,
        kernel_weights=kernel_weights, tabicl_mix_weights=tabicl_mix_weights,
        marginal_backend=marginal_backend, marginal_regressor=marginal_regressor,
        marginal_probs_n=marginal_probs_n,
    )
    # Pin every top-up round to the first round's d_features (or the
    # caller-supplied d_override, if the first round came up empty). Top-up
    # rounds reseed with a different cfg.seed (below), which would otherwise
    # re-sample d independently (variable-d datasets, see _sample_d_features)
    # and silently mix feature counts within one shard — collate_fn cannot
    # pad across the feature axis, unlike P/N, so this must stay fixed.
    if episodes:
        d_fixed = int(episodes[0]["x_norm_train"].shape[-1])
    else:
        d_fixed = d_override
    max_rounds = 20
    for round_idx in range(1, max_rounds + 1):
        if len(episodes) >= B:
            break
        shortfall = B - len(episodes)
        if base_seed is not None:
            # _generate_gp_batch_raw reseeds every RNG from cfg.seed on each
            # call, so retrying with the same cfg.seed would deterministically
            # redraw the identical (failing) kernel/hyperparameters every
            # round. Offset by a large prime per round so top-up retries
            # actually sample a fresh draw instead of repeating the failure.
            cfg.seed = base_seed + round_idx * 104_729
        new_episodes = _generate_gp_batch_raw(
            cfg, shortfall, device, return_kernel_metadata=return_kernel_metadata,
            d_override=d_fixed,
            tabicl_model=tabicl_model, tabicl_k_folds=tabicl_k_folds,
            tabicl_split_calib_frac=tabicl_split_calib_frac,
            kernel_weights=kernel_weights, tabicl_mix_weights=tabicl_mix_weights,
            marginal_backend=marginal_backend, marginal_regressor=marginal_regressor,
            marginal_probs_n=marginal_probs_n,
        )
        if d_fixed is None and new_episodes:
            d_fixed = int(new_episodes[0]["x_norm_train"].shape[-1])
        episodes += new_episodes
    if base_seed is not None:
        cfg.seed = base_seed
    if len(episodes) < B:
        raise RuntimeError(
            f"generate_gp_batch: could not assemble {B} valid episodes after "
            f"{max_rounds} top-up rounds ({len(episodes)} obtained) — the kernel/config "
            f"combination in this call appears to be persistently non-PSD."
        )
    return episodes[:B]
