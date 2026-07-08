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

Composite kernels ("A+B" / "A*B")
---------------------------------
  Sums and products of PSD kernels are PSD, so every pair drawn from
  {rbf, matern32, cosine, periodic, rational_quadratic} is auto-registered
  under both operators via gpytorch's `+`/`*` kernel composition, e.g.
  "rbf+periodic" (locally periodic: smooth decay times exact periodicity) or
  "matern32*cosine" (spectral windowing). See COMPOSITE_KERNELS for the full
  list. dot_product and hebo are not composable (irregular hyperparameter
  signatures — no lengthscale, or ARD-only).

Kernel selection (cfg.data.kernel / cfg.data.kernels)
------------------------------------------------------
  cfg.data.kernel   : str          → use this single kernel for every task
                                     (any entry in ALL_KERNELS, including composites)
  cfg.data.kernels  : list[str]    → sample uniformly at task generation time
  If both are absent the default is "rbf".
"""

from __future__ import annotations

import functools
import itertools
import math
import random
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import gpytorch
import numpy as np
import torch
from gpytorch.priors import GammaPrior, LogNormalPrior, Prior
from torch import Tensor

from loss import _safe_cholesky


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
#     LogNormal/Gamma priors for B episodes at once (used by
#     generate_gp_task / generate_gp_batch to generate new data).
#   - build_kernel_fn: builds a kernel(X1, X2) -> K callable from CONCRETE,
#     already-known hyperparameter values (used by pit.py::gp_analytical_pit
#     and tests to reconstruct the kernel a saved episode was drawn from).
#
# Both always materialise the Gram matrix immediately via `.to_dense()` and
# hand off to this file's own torch.linalg-based Cholesky/solve code
# (_safe_cholesky / _batched_cholesky) rather than gpytorch's own
# ExactGP/lazy-tensor solve machinery — gpytorch silently switches to an
# approximate CG solve for matrices larger than
# gpytorch.settings.max_cholesky_size (default 800), which would silently
# diverge from the exact-Cholesky-based invariants this repo's test suite
# checks (well-conditioned floor, unit-diagonal tolerance).

_BASE_GPYTORCH_KERNEL_CLS: Dict[str, Callable[..., gpytorch.kernels.Kernel]] = {
    "rbf": gpytorch.kernels.RBFKernel,
    "matern32": functools.partial(gpytorch.kernels.MaternKernel, nu=1.5),
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
    lengthscale_attr) given k active input dimensions; ard=True (currently
    only "hebo") samples one independent value per dimension instead of one
    isotropic scalar shared across all k dims.
    """

    lengthscale_prior: Callable[[int], Prior]
    outputscale_prior: Prior
    lengthscale_attr: str = "lengthscale"
    extra_priors: Dict[str, Prior] = field(default_factory=dict)
    ard: bool = False


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
        )

    l_loc = float(getattr(cfg.data, "l_lognormal_loc", math.log(3.0)))
    l_scale = float(getattr(cfg.data, "l_lognormal_scale", 0.7))
    scale_by_sqrt_k = bool(getattr(cfg.data, "l_scale_by_sqrt_k", True))
    a_conc = float(getattr(cfg.data, "alpha2_gamma_concentration", 4.0))
    a_rate = float(getattr(cfg.data, "alpha2_gamma_rate", 3.0))

    def lengthscale_prior(k: int) -> LogNormalPrior:
        # Squared distance sum_{i=1}^k (x1_i - x2_i)^2 grows ~linearly with k
        # (active kernel dims), so without this shift, k=1 and k=4 tasks
        # sample from very different effective-correlation regimes even
        # though l comes from the same distribution — this is the LogNormal
        # equivalent of the old l_scale_by_sqrt_k range-multiplier (if
        # l ~ LogNormal(loc, scale) then c*l ~ LogNormal(loc + log(c), scale)).
        loc = l_loc + (0.5 * math.log(max(k, 1)) if scale_by_sqrt_k else 0.0)
        return LogNormalPrior(loc, l_scale)

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
    )


def _nugget_prior(cfg, kernel_name: str) -> LogNormalPrior:
    """Diagonal regulariser prior — same role the old nugget_min/max range played."""
    if kernel_name == "hebo":
        return LogNormalPrior(
            HEBO_PLUS_HYPERPARAMETERS["hebo_noise_logmean"],
            HEBO_PLUS_HYPERPARAMETERS["hebo_noise_std"],
        )
    loc = float(getattr(cfg.data, "nugget_lognormal_loc", math.log(0.15)))
    scale = float(getattr(cfg.data, "nugget_lognormal_scale", 0.4))
    return LogNormalPrior(loc, scale)


def _build_scaled_kernel(
    name: str, spec: KernelPriorSpec, k: int, B: int, device
) -> tuple[gpytorch.kernels.Kernel, Dict[str, Tensor]]:
    """Sample B episodes' hyperparameters for one base kernel and return the
    resulting ScaleKernel(base)(batch_shape=[B]) object plus a dict of the
    sampled values (keyed by the output-dict schema names: l, alpha2, and
    any of spec.extra_priors' keys)."""
    batch_shape = torch.Size([B])
    kernel_kwargs: Dict = {"batch_shape": batch_shape}
    if spec.ard:
        kernel_kwargs["ard_num_dims"] = k
    # gpytorch kernel modules default to CPU-resident parameters regardless
    # of `device`; move before assigning sampled values so the in-place
    # `self.initialize(...)` used by the `.lengthscale =` / `.outputscale =`
    # setters below copies into device-resident storage, not CPU storage.
    base = _BASE_GPYTORCH_KERNEL_CLS[name](**kernel_kwargs).to(device)

    l_attr = getattr(base, spec.lengthscale_attr)
    l_sample = spec.lengthscale_prior(k).sample(l_attr.shape).to(device)
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
        setattr(base, attr_name, sample)
        params[schema_name] = sample.reshape(B)

    return scaled, params


def _sample_episode_kernel(
    cfg, kernel_name: str, k: int, B: int, device
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
    """
    composite = _parse_composite(kernel_name)
    if kernel_name == "dot_product":
        batch_shape = torch.Size([B])
        kernel = gpytorch.kernels.LinearKernel(batch_shape=batch_shape).to(device)
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
        kernel, params = _build_scaled_kernel(kernel_name, spec, k, B, device)
    else:
        name_a, op, name_b = composite
        kernel_a, params_a = _build_scaled_kernel(name_a, _kernel_prior_spec(cfg, name_a), k, B, device)
        kernel_b, params_b = _build_scaled_kernel(name_b, _kernel_prior_spec(cfg, name_b), k, B, device)
        kernel = kernel_a + kernel_b if op == "+" else kernel_a * kernel_b
        params = dict(params_a)
        for key, val in params_b.items():
            params[f"{key}_b"] = val

    for key in ("period", "rq_alpha", "l_b", "alpha2_b", "period_b", "rq_alpha_b"):
        params.setdefault(key, torch.zeros(B, device=device))

    return kernel, params


def _build_concrete_kernel(
    name: str, l, alpha2, *, period=None, rq_alpha=None
) -> gpytorch.kernels.Kernel:
    """Construct a non-batched gpytorch Kernel with CONCRETE hyperparameter
    values assigned — reconstruction (given known values), not sampling.
    Used by build_kernel_fn.

    "dot_product" returns the bare LinearKernel (no ScaleKernel wrapper):
    its `variance` already plays the role `alpha2` plays for every other
    kernel, so wrapping it would just be a second, redundant alpha2. "l" is
    ignored — no lengthscale, geometry comes entirely from the feature space.
    """
    if name == "dot_product":
        kernel = gpytorch.kernels.LinearKernel()
        kernel.variance = torch.as_tensor(alpha2, dtype=torch.get_default_dtype()).reshape(kernel.variance.shape)
        return kernel
    if name == "hebo":
        l_t = l if torch.is_tensor(l) else torch.tensor([float(l)])
        base = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=l_t.numel())
        base.lengthscale = l_t.reshape(base.lengthscale.shape)
    else:
        base = _BASE_GPYTORCH_KERNEL_CLS[name]()
        attr = "period_length" if name == "cosine" else "lengthscale"
        l_t = torch.as_tensor(l, dtype=torch.get_default_dtype())
        setattr(base, attr, l_t.reshape(getattr(base, attr).shape))
        if name == "periodic" and period is not None:
            base.period_length = torch.as_tensor(float(period)).reshape(base.period_length.shape)
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
    period: Optional[float] = None,
    rq_alpha: Optional[float] = None,
    l_b=None,
    alpha2_b=None,
    period_b: Optional[float] = None,
    rq_alpha_b: Optional[float] = None,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Return a kernel(X1, X2) -> K callable with hyperparameters baked in.

    l_b/alpha2_b/period_b/rq_alpha_b are the second component's hyperparameters
    for composite ("A+B" / "A*B") kernels.
    """
    composite = _parse_composite(kernel_name)
    if composite is None:
        kernel = _build_concrete_kernel(kernel_name, l, alpha2, period=period, rq_alpha=rq_alpha)
    else:
        name_a, op, name_b = composite
        kernel_a = _build_concrete_kernel(name_a, l, alpha2, period=period, rq_alpha=rq_alpha)
        kernel_b = _build_concrete_kernel(name_b, l_b, alpha2_b, period=period_b, rq_alpha=rq_alpha_b)
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


def matern32_kernel(X1: Tensor, X2: Tensor, *, l, alpha2, **_) -> Tensor:
    """Matérn ν=3/2, via gpytorch.kernels.MaternKernel(nu=1.5)."""
    return build_kernel_fn("matern32", l, alpha2)(X1, X2)


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
    "matern32": matern32_kernel,
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
# an isotropic cos(||x||) is not a valid Mercer kernel for d>1); periodic is
# conservatively capped the same way for behavioural parity even though
# gpytorch's ARD PeriodicKernel is independently PSD for k>1.
_SCALAR_ONLY_KERNELS = {"cosine", "periodic"}


def _parse_composite(name: str) -> Optional[tuple]:
    """Split "A+B" / "A*B" into (name_a, op, name_b), or None if not composite."""
    for op in ("+", "*"):
        if op in name:
            a, _, b = name.partition(op)
            if a in _COMPOSABLE_KERNELS and b in _COMPOSABLE_KERNELS:
                return a, op, b
    return None


def _kernel_needs_scalar_input(kernel_name: str) -> bool:
    """True if this kernel (or either half of a composite) requires k=1 input dims."""
    composite = _parse_composite(kernel_name)
    if composite is not None:
        name_a, _, name_b = composite
        return name_a in _SCALAR_ONLY_KERNELS or name_b in _SCALAR_ONLY_KERNELS
    return kernel_name in _SCALAR_ONLY_KERNELS


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


def _sample_kernel_cols(d_total: int, cfg) -> List[int]:
    """Return a sorted list of column indices that the kernel will use.

    k ~ Uniform[d_kernel_min, d_kernel_max]; falls back to all columns when
    the config keys are absent (backward compat with old episode files).
    """
    d_min = int(getattr(cfg.data, "d_kernel_min", d_total))
    d_max = int(getattr(cfg.data, "d_kernel_max", d_total))
    d_min = min(d_min, d_total)
    d_max = min(d_max, d_total)
    k = random.randint(d_min, d_max)
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
    latent: bool = False,
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

    The kernel operates on a random subset of k columns (k ~ Uniform[d_kernel_min,
    d_kernel_max]); the full d_features columns are returned so the model must
    identify which features drive the correlations.

    cosine and periodic kernels are capped to k=1 (they are PSD only for scalar inputs).

    A nugget ~ LogNormal(...) (see _nugget_prior) is added to the diagonal for
    guaranteed PSD and controls posterior tightness (replaces the former
    separate noise parameter).

    If cfg.seed is set (same field train.py uses via `torch.manual_seed(cfg.seed)`),
    it seeds python/numpy/torch RNGs, making the kernel/hyperparameter choice
    (kernel_name, P, N, l, alpha2, nugget, ...), the feature warp, and y
    sampling all reproducible. Calling this repeatedly with the same cfg.seed
    restarts every RNG at the same point every time.

    Keys returned:
        x_norm_train          : (P, d_features)  normalised train features (full)
        y_train               : (P,)             train targets
        x_norm_test           : (N, d_features)  normalised test features (full)
        y_test                : (N,)             test targets
        R_star                : (N, N)           ground-truth test correlation matrix
                                                  (posterior or prior, per cfg.data.oracle_mode)
        mu_star               : (N,)             mean at test points (posterior mean, or 0 for prior)
        sigma_star            : (N,)             marginal std at test points (posterior or prior)
        n_train               : int              P (as 0-dim tensor)
        n_test                : int              N (as 0-dim tensor)
        l                     : float            kernel length scale (scalar tensor)
        alpha2                : float            kernel variance / bias (scalar tensor)
        nugget                : float            diagonal regulariser (LogNormal-sampled)
        kernel                : str              name of the kernel used
        period                : float            period param (periodic kernel only, else 0.0)
        rq_alpha              : float            alpha param (rational_quadratic only, else 0.0)
        kernel_feature_indices: (k,)             column indices used by the kernel (metadata)
    """
    seed = getattr(cfg, "seed", None)
    if seed is not None:
        _seed_everything(seed)

    d = cfg.data.d_features

    # 1. Choose kernel and active columns.
    # cosine and periodic (and any composite containing one of them) are PSD
    # only for scalar (1D) inputs; cap to k=1
    kernel_name = _resolve_kernel_name(cfg)
    if _kernel_needs_scalar_input(kernel_name):
        kernel_cols = [random.randint(0, d - 1)]
    else:
        kernel_cols = _sample_kernel_cols(d, cfg)
    k = len(kernel_cols)

    # 2. Sample dataset sizes
    P = random.randint(cfg.data.P_min, cfg.data.P_max)
    N = random.randint(cfg.data.N_min, cfg.data.N_max)

    # 3. Sample GP hyperparameters via gpytorch LogNormal/Gamma hyperpriors
    # (see _kernel_prior_spec / _nugget_prior). Unlike the pre-gpytorch
    # implementation, "hebo" no longer needs to defer sampling until x_k
    # exists — its Gamma priors don't depend on the input points.
    nugget = _nugget_prior(cfg, kernel_name).sample(torch.Size([1])).item()
    kernel_obj, params = _sample_episode_kernel(cfg, kernel_name, k, 1, "cpu")
    l, alpha2, period, rq_alpha = (params[key][0] for key in ("l", "alpha2", "period", "rq_alpha"))
    l_b, alpha2_b, period_b, rq_alpha_b = (
        params[key][0] for key in ("l_b", "alpha2_b", "period_b", "rq_alpha_b")
    )

    def kernel_fn(X1: Tensor, X2: Tensor) -> Tensor:
        return kernel_obj(X1.unsqueeze(0), X2.unsqueeze(0)).to_dense()[0]

    # 4. Sample features x ~ N(0, 1)^d, warp marginals (TabICLv2-style feature
    # heterogeneity), normalise over all P+N instances
    x_raw = torch.randn(P + N, d)
    x_raw = tabiclv2_warp_features(x_raw)
    mu_x = x_raw.mean(0)
    std_x = x_raw.std(0).clamp(min=1e-8)
    x_norm = (x_raw - mu_x) / std_x                    # full (P+N, d_features)
    x_k = x_norm[:, kernel_cols]                        # kernel sub-matrix (P+N, k)

    # HEBO+'s Gamma-distributed lengthscale is calibrated for x in [0,1]^k
    # (paper Appendix D), unlike this file's usual ~N(0,1)-standardised x_k.
    x_k_kernel = torch.special.ndtr(x_k) if kernel_name == "hebo" else x_k

    # 5. Sample y ~ GP(0, K + nugget·I) jointly.
    K_all = kernel_fn(x_k_kernel, x_k_kernel) + nugget * torch.eye(P + N)
    y_all = _safe_cholesky(K_all, max_attempts=12) @ torch.randn(P + N)

    # 6. Split into train / test (full features returned to model)
    x_norm_train = x_norm[:P]
    y_train = y_all[:P]
    x_norm_test = x_norm[P:]
    y_test = y_all[P:]

    # 7. Compute R_star at test points, from either the GP posterior or the
    # GP prior, per cfg.data.oracle_mode (default "posterior" — unchanged
    # behaviour).
    # posterior: Sigma_star = K_ss + nugget·I − K_st K_ff⁻¹ K_ts accounts for
    #   what the training context already explains — off-diagonal entries
    #   shrink relative to the prior, giving the oracle for residual
    #   dependence after conditioning on training data.
    # prior: Sigma_star = K_ss + nugget·I, ignoring the training context —
    #   the oracle is the raw kernel structure among test points.
    x_k_train = x_k_kernel[:P]
    x_k_test = x_k_kernel[P:]
    oracle_mode = getattr(cfg.data, "oracle_mode", "posterior")
    if oracle_mode == "prior":
        # z_train's LOO PIT still needs L_ff/alpha from K_ff regardless of
        # which oracle is used for the test-side R_star/mu_star/sigma_star.
        K_ff = kernel_fn(x_k_train, x_k_train) + nugget * torch.eye(P, device=x_k_train.device)
        L_ff = _safe_cholesky(K_ff, max_attempts=12)
        alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L_ff).squeeze(-1)
        Sigma_star = kernel_fn(x_k_test, x_k_test) + nugget * torch.eye(N, device=x_k_test.device)
        mu_star = torch.zeros(N, device=x_k_test.device)
    elif oracle_mode == "posterior":
        mu_star, Sigma_star, L_ff, alpha = gp_posterior(
            x_k_train, y_train, x_k_test, kernel_fn, nugget, return_factors=True
        )
    else:
        raise ValueError(f"Unknown data.oracle_mode '{oracle_mode}'; expected 'prior' or 'posterior'.")
    R_star, sigma_star = sigma_to_correlation(Sigma_star)

    return {
        "x_norm_train": x_norm_train,                           # (P, d_features)
        "y_train": y_train,                                      # (P,)
        "x_norm_test": x_norm_test,                             # (N, d_features)
        "y_test": y_test,                                        # (N,)
        "R_star": R_star,                                        # (N, N)
        "mu_star": mu_star,                                      # (N,)
        "sigma_star": sigma_star,                                # (N,)
        "n_train": torch.tensor(P),
        "n_test": torch.tensor(N),
        # scalar for every kernel except "hebo", where l is an ARD
        # per-dimension lengthscale vector (k,) — see _build_scaled_kernel.
        "l": l if torch.is_tensor(l) else torch.tensor(l),
        "alpha2": torch.as_tensor(alpha2),
        "nugget": torch.tensor(nugget),
        "kernel": kernel_name,
        "period": torch.as_tensor(period),
        "rq_alpha": torch.as_tensor(rq_alpha),
        "l_b": torch.as_tensor(l_b),
        "alpha2_b": torch.as_tensor(alpha2_b),
        "period_b": torch.as_tensor(period_b),
        "rq_alpha_b": torch.as_tensor(rq_alpha_b),
        "kernel_feature_indices": torch.tensor(kernel_cols, dtype=torch.long),
        # Ephemeral Cholesky factors — consumed by gp_analytical_pit, not saved to disk.
        "_L_ff": L_ff,    # (P, P) Cholesky of K_ff
        "_alpha": alpha,  # (P,)   K_ff^{-1} y_train
    }


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


@torch.no_grad()
def generate_gp_batch(cfg, B: int, device: str = "cpu") -> List[Dict[str, Tensor]]:
    """Generate B GP episodes in a single vectorised call.

    All B episodes share one kernel type and one (P, N) size (both sampled
    once per call) but have independent hyperparameters and feature draws —
    gpytorch's `batch_shape=[B]` kernels draw B independent hyperparameter
    sets in one call (see _sample_episode_kernel), and evaluate all B Gram
    matrices in one vectorised call to the kernel object. This removes the
    Python-loop overhead of B separate generate_gp_task calls and enables GPU
    or CPU-SIMD acceleration for the linear-algebra steps.

    The returned dicts have the same schema as the episodes saved by
    generate_pit_dataset.py (no kernel metadata).

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

    Returns:
        list of B episode dicts ready for torch.save.
    """
    seed = getattr(cfg, "seed", None)
    if seed is not None:
        _seed_everything(seed)

    d = cfg.data.d_features

    # --- Shared settings for this batch ---
    kernel_name = _resolve_kernel_name(cfg)
    P = random.randint(cfg.data.P_min, cfg.data.P_max)
    N = random.randint(cfg.data.N_min, cfg.data.N_max)
    T = P + N

    if _kernel_needs_scalar_input(kernel_name):
        k = 1
    elif kernel_name == "dot_product":
        # Every dot product can draw on all d columns (no lengthscale to
        # dilute with irrelevant dims, unlike rbf/matern32/rational_quadratic).
        k = d
    else:
        k = random.randint(
            int(getattr(cfg.data, "d_kernel_min", 1)),
            min(int(getattr(cfg.data, "d_kernel_max", 4)), d),
        )

    # --- Per-episode hyperparameters (B independent draws in one call) ---
    nugget = _nugget_prior(cfg, kernel_name).sample(torch.Size([B])).to(device)
    kernel_obj, params = _sample_episode_kernel(cfg, kernel_name, k, B, device)

    # --- Features (B, T, d) ~ N(0, 1), warped, normalised per episode ---
    x_raw = torch.randn(B, T, d, device=device)
    x_raw = tabiclv2_warp_features(x_raw)
    x_norm = (x_raw - x_raw.mean(1, keepdim=True)) / x_raw.std(1, keepdim=True).clamp(min=1e-8)

    # --- Select kernel columns ---
    if kernel_name == "dot_product":
        x_k = x_norm                                                                # (B, T, d)
    elif _kernel_needs_scalar_input(kernel_name):
        col = torch.randint(0, d, (B,), device=device)                             # (B,)
        x_k = x_norm.gather(2, col.view(B, 1, 1).expand(B, T, 1))                 # (B, T, 1)
    else:
        # Vectorised random column selection: argsort of uniform noise picks k cols
        col_idx = torch.rand(B, d, device=device).argsort(dim=1)[:, :k]           # (B, k)
        x_k = x_norm.gather(2, col_idx.unsqueeze(1).expand(B, T, k))              # (B, T, k)

    # HEBO+'s Gamma-distributed lengthscale is calibrated for x in [0,1]^k
    # (paper Appendix D), unlike this file's usual ~N(0,1)-standardised x_k.
    x_k_kernel = torch.special.ndtr(x_k) if kernel_name == "hebo" else x_k

    # --- Build K_all (B, T, T) and sample y jointly ---
    K_all = kernel_obj(x_k_kernel).to_dense()
    K_all = K_all + nugget.view(B, 1, 1) * torch.eye(T, device=device)
    K_all = 0.5 * (K_all + K_all.permute(0, 2, 1))               # symmetrize float32 drift

    L_all = _batched_cholesky(K_all)                              # (B, T, T)
    y_all = (L_all @ torch.randn(B, T, 1, device=device)).squeeze(-1)  # (B, T)

    x_norm_train = x_norm[:, :P]   # (B, P, d)
    x_norm_test  = x_norm[:, P:]   # (B, N, d)
    y_train      = y_all[:,  :P]   # (B, P)
    y_test       = y_all[:,  P:]   # (B, N)

    # --- Sub-matrices of K_all (nugget already on diagonal) ---
    K_ff = K_all[:, :P, :P]   # (B, P, P)
    K_sf = K_all[:, P:, :P]   # (B, N, P)
    K_ss = K_all[:, P:, P:]   # (B, N, N)

    # --- GP posterior/prior (batched) ---
    # z_train's LOO PIT always needs L_ff/alpha from K_ff, regardless of which
    # oracle drives the test-side R_star/mu_star/sigma_star below.
    L_ff  = _batched_cholesky(K_ff)
    alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L_ff).squeeze(-1)          # (B, P)

    oracle_mode = getattr(cfg.data, "oracle_mode", "posterior")
    if oracle_mode == "prior":
        # Prior oracle: ignore training conditioning — R_star reflects the raw
        # kernel structure among test points; mu_star is the GP prior mean (0).
        mu_star    = torch.zeros(B, N, device=device)
        Sigma_star = K_ss
    elif oracle_mode == "posterior":
        mu_star    = (K_sf @ alpha.unsqueeze(-1)).squeeze(-1)                             # (B, N)
        V          = torch.linalg.solve_triangular(L_ff, K_sf.permute(0, 2, 1), upper=False)  # (B, P, N)
        Sigma_star = K_ss - V.permute(0, 2, 1) @ V                                     # (B, N, N)
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
        "Sigma_star":   Sigma_full.cpu(),
        "mu_star":      mu_star.cpu(),
        "sigma_star":   sigma_star.cpu(),
    }
    n_tr = torch.tensor(P)
    n_te = torch.tensor(N)
    return [
        {key: val[b] for key, val in tensors.items()} | {"n_train": n_tr, "n_test": n_te}
        for b in range(B)
    ]
