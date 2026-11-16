"""The attainable Gaussian copula under fixed (possibly wrong) marginals, and the expected copula loss.

With frozen marginal CDFs F_j and queries y_Q ~ p(y_Q | context), the copula sees
z_j = probit(F_j(y_j)). Its expected per-point NLL under a correlation R depends on
z only through the second moment M = E[z z']:
    l(R) = E[-log c_R(z)] / N = (log det R + tr((R^-1 - I) M)) / (2 N),
which is 0 for R = I (independence) and negative when R beats it. The best Gaussian
copula in a family of correlation matrices is
    R_dagger = argmin_R  log det R + tr(R^-1 M),
the yardstick for a copula head trained on these marginals: under exact marginals
M is the posterior correlation R_po, so R_dagger = R_po; with imperfect marginals it
also absorbs whatever compensation for marginal errors is optimal.

No copula of any family can do better than the density of U = F(y_Q) itself:
    l_any >= E[-log p_U(U)] / N = E[-log p(y_Q) + sum_j log f_j(y_j)] / N   (any_copula_bound),
with equality when the marginals are calibrated (each U_j uniform), since p_U is then
a copula. Ordering: bound <= best copula <= l(R_dagger) <= l(R_dagger of rank r).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from copula_inter.copula.model import low_rank_correlation

__all__ = ["any_copula_bound", "attainable_correlation", "expected_copula_nll", "pit_moments", "pit_second_moment"]


def _normalize(S: Tensor) -> Tensor:
    inv_sqrt = S.diagonal().clamp_min(1e-12).rsqrt()
    return S * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)


def _expected_copula_nll_t(R: Tensor, M: Tensor) -> Tensor:
    N = R.shape[-1]
    L = torch.linalg.cholesky(R)
    logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
    trace_rinv_m = torch.trace(torch.cholesky_solve(M, L))
    return (logdet + trace_rinv_m - torch.trace(M)) / (2 * N)


def expected_copula_nll(R: Tensor, M: Tensor) -> float:
    """Per-point expected Gaussian-copula NLL of correlation R (N, N) for z with second moment M (N, N).

    For M = z z' this is exactly the copula NLL of that z (oracle_copula_nll).
    """
    return float(_expected_copula_nll_t(R.double(), M.double()))


def pit_moments(
    cdf: Callable[[Tensor], Tensor], samples: Tensor, eps: float = 1e-6, chunk: int = 256
) -> tuple[Tensor, Tensor]:
    """(m, M) = (mean_l z_l, mean_l z_l z_l') with z_l = probit(cdf(samples[l])), in float64.

    samples: (L, N) draws of y_Q on the marginal's scale. cdf maps a (C, N) chunk of
    rows to (C, N) uniforms (query j's own CDF in column j). eps clamps u to
    [eps, 1 - eps] before the probit, as the eval's PIT does. M - m m' is the covariance
    of z; m != 0 is the marginals' bias at these queries, which M (and so R_dagger) absorbs.
    """
    L, N = samples.shape
    m = torch.zeros(N, dtype=torch.float64, device=samples.device)
    M = torch.zeros(N, N, dtype=torch.float64, device=samples.device)
    for start in range(0, L, chunk):
        u = cdf(samples[start : start + chunk]).double()
        z = torch.special.ndtri(u.clamp(eps, 1.0 - eps))
        m += z.sum(0)
        M += z.T @ z
    return m / L, M / L


def pit_second_moment(cdf: Callable[[Tensor], Tensor], samples: Tensor, eps: float = 1e-6, chunk: int = 256) -> Tensor:
    """M = mean_l z_l z_l' (see pit_moments)."""
    return pit_moments(cdf, samples, eps, chunk)[1]


def attainable_correlation(
    M: Tensor, rank: int | None = None, max_iter: int = 500, parametrization: str = "covnorm"
) -> Tensor:
    """R_dagger = argmin over correlation matrices R of log det R + tr(R^-1 M) (float64).

    rank=None: every correlation matrix, R = normalize(B B') with a full (N, N) B.
    rank=r: the copula head's family, R = low_rank_correlation(W, s) with W (N, r)
    (covnorm: normalize(W W' + diag(softplus(s))), plus the head's jitter).
    Initialized at normalize(M) (its top-r eigenpairs for a rank r) and refined by L-BFGS.
    """
    if parametrization != "covnorm":
        raise ValueError(f"attainable_correlation supports parametrization='covnorm', not {parametrization!r}")
    M64 = M.double()
    N = M64.shape[0]
    eye = torch.eye(N, dtype=torch.float64, device=M64.device)
    R0 = _normalize(M64) + 1e-6 * eye

    if rank is None:
        B = torch.linalg.cholesky(R0).contiguous().requires_grad_(True)
        params = [B]

        def corr() -> Tensor:
            return _normalize(B @ B.T + 1e-10 * eye)

    else:
        r = min(rank, N)
        evals, evecs = torch.linalg.eigh(R0)
        top_vals, top_vecs = evals[-r:].clamp_min(0.0), evecs[:, -r:]
        W0 = top_vecs * top_vals.sqrt()
        resid = (1.0 - (W0**2).sum(-1)).clamp_min(1e-3)
        W = W0.contiguous().requires_grad_(True)
        s = torch.log(torch.expm1(resid)).requires_grad_(True)  # softplus^-1
        params = [W, s]

        def corr() -> Tensor:
            return low_rank_correlation(W.unsqueeze(0), s.unsqueeze(0), parametrization=parametrization)[0]

    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        opt.zero_grad()
        loss = _expected_copula_nll_t(corr(), M64)
        loss.backward()
        return loss

    with torch.enable_grad():  # callers may score under no_grad
        opt.step(closure)
    with torch.no_grad():
        R = corr()
    return 0.5 * (R + R.T)


def any_copula_bound(log_joint: Tensor, log_marginals: Tensor, n_points: int) -> float:
    """E[-log p_U(U)] / N from draws y^(l) ~ p(y_Q): mean_l(sum_j log f_j(y_j^(l)) - log p(y^(l))) / N.

    log_joint: (L,) log p(y^(l)) under the true conditional; log_marginals: (L,) sum_j log f_j(y_j^(l))
    under the frozen marginals, both on the same y scale.
    """
    return float((log_marginals.double() - log_joint.double()).mean()) / n_points
