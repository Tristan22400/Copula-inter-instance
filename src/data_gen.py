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
  samples a random chain length M ~ Uniform[composite_num_kernels_min,
  composite_num_kernels_max], draws M elementary kernels with replacement
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
        kernel = kernel_a + kernel_b if op == "+" else kernel_a * kernel_b
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
        kernel = kernel_a + kernel_b if op == "+" else kernel_a * kernel_b

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
    see _build_kernel_chain) — instead of picking from the fixed 56-entry
    COMPOSITE_KERNELS pool. Active only when cfg.data.systematic_composition
    is True (see _resolve_kernel_name's docstring for the non-systematic
    path). cfg.data.composite_exclude_kernels (optional list, default empty)
    drops named elementary kernels from the sampling pool without touching
    _COMPOSABLE_KERNELS itself — that constant also seeds the static 56-entry
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
        kernel = kernel + comp_kernel if op == "+" else kernel * comp_kernel
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
    _generate_gp_batch_raw's oracle_mode branches and _GeneratorGP) — adding
    a mean to y_all without also updating those would silently miscalibrate
    the PIT.

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
    meaningful where mean_nonzero is True), "mean_linear" (B,) bool.
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


class _GeneratorGP(gpytorch.models.ExactGP):
    """Thin ExactGP wrapper so oracle_mode="posterior" conditioning goes
    through gpytorch's own exact-inference machinery (`model(x_test)` in
    eval mode) instead of the hand-rolled K_ss - K_sf K_ff^-1 K_fs formula
    gp_posterior used. `kernel` is the already-sampled (batch_shape=[B])
    Kernel object from _sample_episode_kernel — one instance, no per-family
    branching needed here. Verified numerically equivalent (~1e-6 max abs
    diff) to the old manual computation, given max_cholesky_size forced
    high enough (see _MAX_CHOLESKY) and fast_pred_var disabled at call time."""

    def __init__(
        self, train_x: Tensor, train_y: Tensor, likelihood, kernel: gpytorch.kernels.Kernel,
        batch_shape: torch.Size, mean_module: Optional[gpytorch.means.Mean] = None,
    ):
        super().__init__(train_x, train_y, likelihood)
        # Defaults to ZeroMean for any other caller (none currently exist)
        # that doesn't pass one — _generate_gp_batch_raw always passes its
        # own _sample_mean_module result (an exact ZeroMean when
        # cfg.data.mean_fn_enabled is False, else a _MeanFunctionBank).
        self.mean_module = mean_module if mean_module is not None else gpytorch.means.ZeroMean(batch_shape=batch_shape)
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
        l, alpha2, nugget, kernel, period, rq_alpha, power,
        l_b, alpha2_b, period_b, rq_alpha_b, power_b, kernel_feature_indices,
        _L_ff, _alpha  (ephemeral Cholesky factors, consumed by
                        pit.py::gp_analytical_pit, not saved to disk)
    """
    return generate_gp_batch(cfg, 1, "cpu", return_kernel_metadata=True)[0]


# ---------------------------------------------------------------------------
# Batched generation (C: vectorised over B episodes simultaneously)
# ---------------------------------------------------------------------------


def _batched_cholesky(K: Tensor) -> tuple[Tensor, Tensor]:
    """Batched Cholesky (B, N, N) → (L, failed): L is (B, N, N) with automatic
    jitter for failures; failed (B,) bool marks episodes where even the
    maximum jitter (0.1) couldn't recover a PSD matrix — L is an identity
    placeholder for those, and the caller (generate_gp_batch) is expected to
    drop them rather than save a degenerate episode."""
    L, info = torch.linalg.cholesky_ex(K)
    failed = info.ne(0)
    if not failed.any():
        return L, failed
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
        # Last resort: replace with identity so this call doesn't crash.
        # Logged (not silent) so a run-wide rate can be monitored — this
        # means K_ff/LOO PIT (z_train) is degenerate for these episodes even
        # after the maximum jitter escalation above. The caller discards
        # these episodes entirely (see generate_gp_batch) rather than saving
        # this identity placeholder.
        warnings.warn(
            f"_batched_cholesky: {int(failed.sum())}/{K.shape[0]} episodes fell back "
            f"to an identity K_ff Cholesky factor (unrecoverable even at jitter=0.1) "
            f"and will be discarded.",
            RuntimeWarning,
        )
        L[failed] = eye.unsqueeze(0).expand_as(L[failed])
    return L, failed


def _psd_safe_batch(K: Tensor, max_tries: int = 6) -> tuple[Tensor, Tensor]:
    """Batched (B, N, N) -> (L, failed) Cholesky factor via gpytorch's own
    psd_safe_cholesky, instead of a hand-rolled retry loop (see
    _batched_cholesky above, kept as-is for K_ff/LOO PIT — this is the
    equivalent tool for K_all, used to GUARANTEE Sigma_star/R_star are PSD
    rather than just symmetric, see generate_gp_batch's joint-sample block).
    failed (B,) bool marks episodes where even the maximum jitter couldn't
    recover a PSD matrix; the caller is expected to drop them.

    psd_safe_cholesky adds escalating diagonal jitter (starting at
    gpytorch.settings.cholesky_jitter, x10 per retry — the exact same
    mechanism gpytorch itself falls back to internally, e.g. the "added
    jitter... Using symeig method" warning this pipeline already emits for
    marginal episodes) ONLY to the batch elements that actually fail a
    Cholesky attempt, leaving every well-conditioned episode's matrix
    untouched. max_tries=6 reaches a jitter ceiling of ~1e-6*10^5=0.1,
    matching _batched_cholesky's own escalation ceiling above.

    psd_safe_cholesky raises (rather than returning a partial result) if
    ANY batch element is still not PSD after max_tries — so on that
    exception we re-derive exactly which elements are still bad at the same
    maximum jitter (rather than discarding the whole batch's progress) and
    fall back to identity (same "mark the episode invalid, don't crash
    generation" convention _batched_cholesky uses) for only those, in the
    astronomically unlikely case max_tries isn't enough. Logged (not
    silent) so a run-wide rate can be monitored.

    Separately, an occasional extreme composite-kernel hyperparameter draw
    makes gpytorch's own float32 kernel evaluation produce actual NaN
    entries (not just a slightly negative eigenvalue) in K_all_raw.
    psd_safe_cholesky checks for NaN over the WHOLE batch tensor and raises
    NanError immediately if any element is NaN, before its jitter
    escalation ever runs — so even the other, perfectly fine episodes in
    this batch would be blocked. NaN episodes are therefore replaced with
    identity up front so jitter escalation still gets a chance to run on
    the rest of the batch, and are folded into `failed` like any other
    unrecoverable episode.
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
    try:
        L = psd_safe_cholesky(K, max_tries=max_tries)
        return L, nan_failed
    except NotPSDError:
        jitter0 = gpytorch.settings.cholesky_jitter.value(K.dtype)
        max_jitter = jitter0 * (10 ** (max_tries - 1))
        eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
        L, info = torch.linalg.cholesky_ex(K + max_jitter * eye)
        failed = info.ne(0) | nan_failed
        warnings.warn(
            f"_psd_safe_batch: {int(failed.sum())}/{K.shape[0]} episodes fell back "
            f"to an identity K_all (unrecoverable even at jitter={max_jitter:.1e}) "
            f"and will be discarded.",
            RuntimeWarning,
        )
        L[failed] = eye.unsqueeze(0).expand_as(L[failed])
        return L, failed


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


@torch.no_grad()
def _generate_gp_batch_raw(
    cfg, B: int, device: str = "cpu", *, return_kernel_metadata: bool = False,
    d_override: Optional[int] = None,
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
            name, hyperparameters (l, alpha2, nugget, period, rq_alpha, power,
            l_b, alpha2_b, period_b, rq_alpha_b, power_b), kernel_feature_indices,
            mlp_mixed (bool — whether the mlp-mixing gate fired for that
            episode; see apply_mlp_feature_mixing), and the ephemeral
            _L_ff/_alpha Cholesky factors — the schema
            generate_gp_task/pit.py::gp_analytical_pit/diag_kernels.py need.
            Off by default so the production shard schema
            (generate_pit_dataset.py) is unaffected.

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
    if return_kernel_metadata:
        x_raw, mlp_mixed = apply_mlp_feature_mixing(x_raw, cfg, device, return_gate=True)
    else:
        x_raw = apply_mlp_feature_mixing(x_raw, cfg, device)
    x_norm = (x_raw - x_raw.mean(1, keepdim=True)) / x_raw.std(1, keepdim=True).clamp(min=1e-8)

    # --- Joint prior sample + noisy covariance (B, T, T), via gpytorch's own
    # GaussianLikelihood(MultivariateNormal) — replaces the old manual
    # `kernel_obj(...).to_dense() + nugget*eye` Gram-matrix assembly.
    # max_cholesky_size is forced high (see module docstring / _MAX_CHOLESKY)
    # so the covariance_matrix materialization below is exact, not gpytorch's
    # approximate CG/Lanczos fallback.
    with gpytorch.settings.max_cholesky_size(_MAX_CHOLESKY):
        prior_dist = gpytorch.distributions.MultivariateNormal(
            torch.zeros(B, T, device=device), kernel_obj(x_norm)
        )
        noisy_dist = likelihood(prior_dist)
        try:
            K_all_raw = noisy_dist.covariance_matrix  # (B, T, T), nugget already on diagonal
        except NotPSDError:
            # Some composite/systematic kernel chains (esp. under sign
            # modulation — see _build_kernel_chain) leave an intermediate
            # partial-sum LinearOperator that gpytorch's own low-rank
            # __add__ path (add_low_rank -> root_inv_decomposition) can't
            # Cholesky-factor, even before _psd_safe_batch below gets a
            # chance to repair the final K_all. This raises straight out of
            # gpytorch's kernel evaluation, for the whole batch at once, so
            # the individual bad episode(s) can't be isolated the way
            # _psd_safe_batch/_batched_cholesky isolate per-row failures
            # below. Discard the whole B-episode batch (this call's kernel/
            # hyperparameter draw) and let generate_gp_batch's top-up loop
            # resample a fresh one, rather than crashing the whole run.
            warnings.warn(
                f"_generate_gp_batch_raw: kernel evaluation for this "
                f"{B}-episode batch (kernel={kernel_name!r}) raised NotPSDError "
                f"before psd_safe_cholesky repair could run; discarding the "
                f"whole batch and resampling.",
                RuntimeWarning,
            )
            return []

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
    L_ff, failed_ff = _batched_cholesky(K_ff)
    alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L_ff).squeeze(-1)          # (B, P)

    # Episodes where either Cholesky repair above bottomed out at an
    # identity placeholder are not valid GP episodes (K_all/K_ff no longer
    # reflect the sampled kernel at all) — dropped entirely at the end of
    # this function (see the "Discard degenerate episodes" block below)
    # rather than saved with a degenerate placeholder. generate_gp_batch
    # (the public wrapper) tops up the shortfall so callers still get
    # exactly B valid episodes.
    discard = failed_all | failed_ff

    oracle_mode = getattr(cfg.data, "oracle_mode", "posterior")
    if oracle_mode == "prior":
        # Prior oracle: ignore training conditioning — R_star reflects the raw
        # kernel structure among test points; mu_star is the GP prior mean (0).
        # No conditioning needed, so this branch never touches _GeneratorGP.
        # Sigma_star = K_ss is already guaranteed PSD here: K_ss is a
        # principal submatrix of K_all, which is constructed above as
        # L_all @ L_all.mT (PSD by construction, via psd_safe_cholesky) —
        # not a slice of an unprotected raw materialization. A principal
        # submatrix of a PSD matrix is itself PSD, so no further repair is
        # needed at this point.
        mu_star    = mean_module(x_norm_test)
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
        x_kernel_train = x_norm[:, :P]
        x_kernel_test  = x_norm[:, P:]
        out_dtype = x_norm.dtype
        with gpytorch.settings.max_cholesky_size(_MAX_CHOLESKY), gpytorch.settings.fast_pred_var(False):
            post_model = _GeneratorGP(
                x_kernel_train.double(), y_train.double(),
                likelihood.double(), kernel_obj.double(), batch_shape,
                mean_module.double(),
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

    discard = discard | degen

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
        for key in ("mean_weight", "mean_bias", "mean_nonzero", "mean_family", "mean_linear"):
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
    cfg, B: int, device: str = "cpu", *, return_kernel_metadata: bool = False
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

    In practice this loop almost never repeats more than once: the discard
    rate is astronomically rare (an episode has to defeat escalating jitter
    up to ~0.1 — see _psd_safe_batch/_batched_cholesky). max_rounds bounds
    the retries so a pathological config (e.g. one that's non-PSD by
    construction regardless of jitter) fails loudly instead of hanging.
    """
    base_seed = getattr(cfg, "seed", None)
    episodes = _generate_gp_batch_raw(cfg, B, device, return_kernel_metadata=return_kernel_metadata)
    # Pin every top-up round to the first round's d_features. Top-up rounds
    # reseed with a different cfg.seed (below), which would otherwise
    # re-sample d independently (variable-d datasets, see _sample_d_features)
    # and silently mix feature counts within one shard — collate_fn cannot
    # pad across the feature axis, unlike P/N, so this must stay fixed.
    d_fixed = int(episodes[0]["x_norm_train"].shape[-1]) if episodes else None
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
