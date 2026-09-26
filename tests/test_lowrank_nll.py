"""Low-rank (Matrix Determinant Lemma + Woodbury) y_space_nll vs the dense path: same Sigma, values and gradients."""


import pytest
import torch

from copula_inter.loss import y_space_nll  # noqa: E402
from copula_inter.model import low_rank_correlation, low_rank_correlation_factor  # noqa: E402

PARAMS = ["covnorm", "cossim", "tanhnorm", "sparse_covnorm"]


def _exact_copula_nll(Sigma, z, mask):
    """Jitter-free float64 reference (slogdet + solve per episode)."""
    vals = []
    for b in range(Sigma.shape[0]):
        n = int(mask[b].sum())
        if n == 0:
            continue
        S, zb = Sigma[b, :n, :n].double(), z[b, :n].double()
        quad = zb @ torch.linalg.solve(S, zb)
        vals.append(0.5 * (torch.linalg.slogdet(S)[1] + quad - zb @ zb) / n)
    return torch.stack(vals).mean()


def _inputs(B=3, N=17, r=5, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(B, N, r, generator=g, dtype=dtype)
    s = torch.randn(B, N, generator=g, dtype=dtype)
    lam = torch.full((1,), -1.0, dtype=dtype)
    z = torch.randn(B, N, generator=g, dtype=dtype)
    log_pdf = torch.randn(B, N, generator=g, dtype=dtype)
    # Ragged episodes, one fully empty -- exercises padding and the valid mask.
    n_valid = torch.tensor([N, N - 6, 0])[:B]
    mask = torch.arange(N).unsqueeze(0) < n_valid.unsqueeze(1)
    z = z * mask
    log_pdf = log_pdf * mask
    return W, s, lam, z, log_pdf, mask


@pytest.mark.parametrize("param", PARAMS)
def test_factor_dense_matches_low_rank_correlation(param):
    W, s, lam, *_ = _inputs()
    dense = low_rank_correlation(W, s, jitter=1e-4, parametrization=param, lam=lam)
    factor = low_rank_correlation_factor(W, s, jitter=1e-4, parametrization=param, lam=lam)
    torch.testing.assert_close(factor.dense(), dense, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        factor.dense().diagonal(dim1=-2, dim2=-1), torch.ones_like(s), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("param", PARAMS)
def test_lowrank_nll_matches_dense(param):
    W, s, lam, z, log_pdf, mask = _inputs()
    dense = low_rank_correlation(W, s, jitter=1e-4, parametrization=param, lam=lam)
    factor = low_rank_correlation_factor(W, s, jitter=1e-4, parametrization=param, lam=lam)
    ref = y_space_nll(dense, z, log_pdf, mask)
    got = y_space_nll(factor, z, log_pdf, mask)
    torch.testing.assert_close(got["copula"], _exact_copula_nll(dense, z, mask), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(got["marginal"], ref["marginal"])
    # vs the dense path only up to its own 1e-6 Cholesky-jitter bias.
    torch.testing.assert_close(got["total"], ref["total"], rtol=1e-3, atol=1e-3)


def test_lowrank_nll_gradients_match_dense():
    W, s, _, z, log_pdf, mask = _inputs(seed=1)
    grads = []
    for build in (low_rank_correlation, low_rank_correlation_factor):
        Wg, sg = W.clone().requires_grad_(), s.clone().requires_grad_()
        y_space_nll(build(Wg, sg, jitter=1e-4), z, log_pdf, mask)["total"].backward()
        grads.append((Wg.grad, sg.grad))
    torch.testing.assert_close(grads[1][0], grads[0][0], rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(grads[1][1], grads[0][1], rtol=1e-4, atol=1e-6)


def test_lowrank_nll_fp32_near_singular():
    """fp32 inputs with a ~jitter diagonal still match a float64 dense reference."""
    W, s, _, z, log_pdf, mask = _inputs(N=60, r=4, seed=2)
    s = torch.full_like(s, -30.0)  # softplus ≈ 1e-13: Σ ≈ rank-4 + 1e-4 I
    ref = _exact_copula_nll(low_rank_correlation(W, s, jitter=1e-4), z, mask)
    factor32 = low_rank_correlation_factor(W.float(), s.float(), jitter=1e-4)
    got = y_space_nll(factor32, z.float(), log_pdf.float(), mask)
    assert got["copula"].dtype == torch.float32
    torch.testing.assert_close(got["copula"].double(), ref, rtol=1e-4, atol=1e-3)


def test_lowrank_nll_all_padding_is_zero_and_differentiable():
    W, s, _, z, log_pdf, mask = _inputs()
    mask = torch.zeros_like(mask)
    Wg = W.clone().requires_grad_()
    out = y_space_nll(low_rank_correlation_factor(Wg, s), z, log_pdf, mask)
    assert out["total"].item() == 0.0
    out["total"].backward()
