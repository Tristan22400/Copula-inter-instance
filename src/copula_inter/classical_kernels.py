"""Bank of classical GP kernels evaluated as fixed baselines in the validation loop.

Each kernel turns an episode's *test* features ``x_test`` (B, N, d) directly into an
inter-instance correlation matrix (B, N, N) — the same object the transformer
predicts as ``Sigma``. Scoring these fixed kernels against the oracle ``R_star``
(and via the copula NLL on ``z_test``) tells us how much of the inter-instance
structure is recoverable by a plain distance kernel on the features, i.e. how much
the learned model actually buys over a classical baseline.

Kernel families (each available with and without ARD):
    dot_product         — linear kernel <x_i, x_j>, normalised to cosine similarity
    rbf                 — squared exponential  exp(-r²/2)
    matern12            — Matérn ν=1/2         exp(-r)
    matern32            — Matérn ν=3/2         (1+√3 r) exp(-√3 r)
    matern52            — Matérn ν=5/2         (1+√5 r + 5r²/3) exp(-√5 r)
    rational_quadratic  — (1 + r²/(2α))^(−α)

Lengthscales are *fixed* (not learned) via the median heuristic, computed per
episode so the baseline adapts to each task's feature scale without any training:
    isotropic : ℓ  = median pairwise Euclidean distance over the valid points
    ARD       : ℓ_d = median pairwise |Δ| along dimension d (one per feature)

The kernels themselves are the real gpytorch kernel objects (``RBFKernel``,
``MaternKernel(nu=…)``, ``RQKernel``, ``LinearKernel``) — the same families
``data_gen.py`` uses to *generate* the data — so the baseline math stays a single
source of truth with the generating side. Each family is built batched
(``batch_shape=[B]``) so every episode gets its own median-heuristic lengthscale
in one ``kernel(x).to_dense()`` call. The resulting Gram matrix K is normalised to
a correlation matrix ``D^{-1/2} K D^{-1/2}`` so it has a unit diagonal and is a
drop-in for ``R_star`` / ``Sigma`` (a no-op for the unit-variance stationary
kernels; meaningful for dot_product).
"""

from __future__ import annotations

from collections import OrderedDict

import gpytorch
import torch
from torch import Tensor

# gpytorch MaternKernel supports exactly these three smoothness values.
_MATERN_NU = {"matern12": 0.5, "matern32": 1.5, "matern52": 2.5}

# Lengthscale floor: guards against a degenerate zero median (e.g. duplicate
# points or a single valid pair) producing a divide-by-zero.
_LS_FLOOR = 1e-6

# Default bank: every family, isotropic and ARD.
DEFAULT_FAMILIES = (
    "dot_product",
    "rbf",
    "matern12",
    "matern32",
    "matern52",
    "rational_quadratic",
)
DEFAULT_ARD = (False, True)


def kernel_name(family: str, ard: bool) -> str:
    return f"{family}_ard" if ard else family


def default_kernel_names() -> list[str]:
    return [kernel_name(f, a) for f in DEFAULT_FAMILIES for a in DEFAULT_ARD]


def _offdiag_pair_mask(mask: Tensor) -> Tensor:
    """(B, N) validity mask → (B, N, N) mask of valid *off-diagonal* pairs."""
    m2 = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    n = mask.shape[1]
    eye = torch.eye(n, dtype=torch.bool, device=mask.device)
    return m2 & ~eye


def _median_isotropic(dist: Tensor, pair_mask: Tensor, scale: float) -> Tensor:
    """Global median-heuristic lengthscale per episode. dist (B,N,N) → ls (B,)."""
    B = dist.shape[0]
    out = dist.new_ones(B)
    for b in range(B):
        vals = dist[b][pair_mask[b]]
        if vals.numel() > 0:
            out[b] = torch.median(vals).clamp_min(_LS_FLOOR)
    return out * scale


def _median_ard(absdiff: Tensor, pair_mask: Tensor, scale: float) -> Tensor:
    """Per-dimension median-heuristic lengthscale. absdiff (B,N,N,d) → ls (B,d)."""
    B, _, _, d = absdiff.shape
    out = absdiff.new_ones(B, d)
    for b in range(B):
        pm = pair_mask[b]
        if pm.any():
            vals = absdiff[b][pm]  # (n_pairs, d)
            out[b] = torch.median(vals, dim=0).values.clamp_min(_LS_FLOOR)
    return out * scale


def _stationary_gram(family: str, x: Tensor, ls: Tensor, ard: bool, rq_alpha: float) -> Tensor:
    """Batched gpytorch Gram matrix (B, N, N) with a per-episode lengthscale.

    ``ls`` is (B,) isotropic or (B, d) ARD; it is written straight into the
    kernel's batched ``lengthscale`` so each episode gets its own scale in one
    ``kernel(x).to_dense()`` call.
    """
    B, _, d = x.shape
    kw = {"batch_shape": torch.Size([B])}
    if ard:
        kw["ard_num_dims"] = d
    if family == "rbf":
        kernel = gpytorch.kernels.RBFKernel(**kw)
    elif family in _MATERN_NU:
        kernel = gpytorch.kernels.MaternKernel(nu=_MATERN_NU[family], **kw)
    elif family == "rational_quadratic":
        kernel = gpytorch.kernels.RQKernel(**kw)
    else:
        raise ValueError(f"unknown stationary kernel family: {family!r}")
    kernel = kernel.to(x.device)
    kernel.lengthscale = ls.reshape(kernel.lengthscale.shape)
    if family == "rational_quadratic":
        kernel.alpha = torch.full_like(kernel.alpha, float(rq_alpha))
    return kernel(x).to_dense()


def _to_correlation(K: Tensor, mask: Tensor) -> Tensor:
    """Normalise a kernel Gram matrix to a correlation matrix (unit diagonal).

    ``D^{-1/2} K D^{-1/2}`` with padded rows/cols forced to the identity so the
    result is a valid, Cholesky-able correlation matrix for every episode.
    """
    diag = K.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)  # (B, N)
    denom = torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2))
    C = K / denom
    C = 0.5 * (C + C.mT)  # symmetrise away round-off

    n = C.shape[1]
    eye = torch.eye(n, device=K.device, dtype=K.dtype)
    C = C * (1.0 - eye) + eye  # exact unit diagonal
    m2 = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    return torch.where(m2, C, eye.unsqueeze(0))


@torch.no_grad()
def compute_kernel_bank(
    x: Tensor,
    mask: Tensor,
    families=DEFAULT_FAMILIES,
    ard_flags=DEFAULT_ARD,
    rq_alpha: float = 1.0,
    lengthscale_scale: float = 1.0,
) -> "OrderedDict[str, Tensor]":
    """Correlation matrices for the full kernel bank.

    Args:
        x     : (B, N, d) test features (already normalised x_norm)
        mask  : (B, N) bool — valid (non-padded) test instances
        families / ard_flags : which kernels to build (Cartesian product)
        rq_alpha : α of the Rational Quadratic kernel
        lengthscale_scale : multiplier on the median-heuristic lengthscales

    Returns an OrderedDict name → (B, N, N) correlation matrix.
    """
    pair_mask = _offdiag_pair_mask(mask)
    need_iso = any(not a for a in ard_flags)
    need_ard = any(a for a in ard_flags)
    ls_iso = (
        _median_isotropic(torch.cdist(x, x), pair_mask, lengthscale_scale)
        if need_iso else None
    )
    ls_ard = (
        _median_ard((x.unsqueeze(2) - x.unsqueeze(1)).abs(), pair_mask, lengthscale_scale)
        if need_ard else None
    )

    bank: "OrderedDict[str, Tensor]" = OrderedDict()
    for family in families:
        for ard in ard_flags:
            if family == "dot_product":
                # LinearKernel has no lengthscale; fold the ARD scale into the
                # inputs (weighted inner product), then normalise to a cosine.
                xw = x / ls_ard.unsqueeze(1) if ard else x
                K = gpytorch.kernels.LinearKernel().to(x.device)(xw).to_dense()
            else:
                ls = ls_ard if ard else ls_iso
                K = _stationary_gram(family, x, ls, ard, rq_alpha)
            bank[kernel_name(family, ard)] = _to_correlation(K, mask)
    return bank
