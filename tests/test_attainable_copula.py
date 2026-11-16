"""The attainable copula R_dagger, the expected copula loss, the frozen-marginal CDF, and their eval rows."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from copula_inter.copula.attainable import (
    any_copula_bound,
    attainable_correlation,
    expected_copula_nll,
    pit_moments,
    pit_second_moment,
)
from copula_inter.data.gp_episodes import generate_gp_batch
from copula_inter.marginal.pit import gp_analytical_posterior, query_marginal, run_pit
from eval.baselines.classical import corr_nll_single
from eval.runners.episode_scoring import _attainable_rows, _marginal_pit
from eval.runners.eval_args import AttainableSpec


def _random_corr(n: int, rank: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(n, rank, generator=g, dtype=torch.float64)
    S = W @ W.T + 0.3 * torch.eye(n, dtype=torch.float64)
    d = S.diagonal().rsqrt()
    return S * d[:, None] * d[None, :]


def _std_normal_cdf(values: torch.Tensor) -> torch.Tensor:
    return torch.special.ndtr(values)


def test_expected_loss_is_the_copula_nll_for_one_draw_and_zero_for_independence() -> None:
    R = _random_corr(5, 2, 0)
    z = torch.randn(5, dtype=torch.float64)
    assert expected_copula_nll(R, torch.outer(z, z)) == pytest.approx(corr_nll_single(R.float(), z.float()), abs=1e-4)
    assert expected_copula_nll(torch.eye(5), R) == pytest.approx(0.0, abs=1e-12)


def test_exact_marginals_recover_the_true_correlation() -> None:
    """With exact N(0, 1) marginals, M -> R_true, so R_dagger (full rank) ~ R_true and beats every other R."""
    n = 6
    R_true = _random_corr(n, 2, 1)
    g = torch.Generator().manual_seed(2)
    y = torch.randn(200_000, n, generator=g, dtype=torch.float64) @ torch.linalg.cholesky(R_true).T
    M = pit_second_moment(_std_normal_cdf, y)
    assert torch.allclose(M, R_true, atol=0.02)
    R_dag = attainable_correlation(M)
    assert torch.allclose(R_dag, R_true, atol=0.02)
    assert torch.allclose(R_dag.diagonal(), torch.ones(n, dtype=torch.float64), atol=1e-8)
    loss = expected_copula_nll(R_dag, M)
    assert loss <= expected_copula_nll(R_true, M) + 1e-9
    assert loss < 0.0


def test_any_copula_bound_equals_the_gaussian_optimum_for_calibrated_marginals() -> None:
    """Exact N(0, 1) marginals and a Gaussian truth: p_U is the Gaussian copula of R_true, so bound = l(R_dagger)."""
    n = 6
    R_true = _random_corr(n, 2, 5)
    g = torch.Generator().manual_seed(6)
    y = torch.randn(100_000, n, generator=g, dtype=torch.float64) @ torch.linalg.cholesky(R_true).T
    log_joint = torch.distributions.MultivariateNormal(torch.zeros(n, dtype=torch.float64), R_true).log_prob(y)
    log_marg = torch.distributions.Normal(0.0, 1.0).log_prob(y).sum(-1)
    bound = any_copula_bound(log_joint, log_marg, n)
    M = pit_second_moment(_std_normal_cdf, y)
    assert bound == pytest.approx(expected_copula_nll(attainable_correlation(M), M), abs=5e-3)
    # In expectation: (log det R_true) / (2 n).
    assert bound == pytest.approx(float(torch.logdet(R_true)) / (2 * n), abs=5e-3)


def test_a_shared_marginal_bias_is_what_r_dagger_exploits_beyond_the_covariance() -> None:
    """z ~ N(b, I): independent spread, shared bias b. Fitted to Cov(z) = I the copula is I (l = 0);
    on M = I + b b' R_dagger gains, by (b'R^-1 b - |b|^2)/2N + log det R / 2N < 0."""
    n = 8
    b = torch.linspace(-1.0, 1.0, n, dtype=torch.float64) + 0.5
    g = torch.Generator().manual_seed(7)
    y = b + torch.randn(200_000, n, generator=g, dtype=torch.float64)
    m, M = pit_moments(_std_normal_cdf, y)
    assert torch.allclose(m, b, atol=0.01)
    assert torch.allclose(M, pit_second_moment(_std_normal_cdf, y))
    R_centered = attainable_correlation(M - torch.outer(m, m))
    assert torch.allclose(R_centered, torch.eye(n, dtype=torch.float64), atol=0.02)
    assert expected_copula_nll(R_centered, M) == pytest.approx(0.0, abs=5e-3)
    assert expected_copula_nll(attainable_correlation(M), M) < -0.05


def test_lowrank_family_is_no_better_than_full_rank_and_fits_a_lowrank_truth() -> None:
    """The head's rank-2 family contains (up to its jitter) a rank-2-plus-diagonal truth."""
    n = 8
    R_true = _random_corr(n, 2, 3)
    M = R_true.clone()
    full = expected_copula_nll(attainable_correlation(M), M)
    low = expected_copula_nll(attainable_correlation(M, rank=2), M)
    assert full <= low + 1e-9
    assert low == pytest.approx(full, abs=2e-3)
    rank1 = expected_copula_nll(attainable_correlation(M, rank=1), M)
    assert rank1 >= low - 1e-9


def test_miscalibrated_marginals_change_the_attainable_copula() -> None:
    """Marginals twice too wide shrink z (diag M = 1/4); R_dagger is still a correlation matrix and optimal."""
    n = 5
    R_true = _random_corr(n, 2, 4)
    M = R_true / 4.0
    R_dag = attainable_correlation(M)
    assert torch.allclose(R_dag.diagonal(), torch.ones(n, dtype=torch.float64), atol=1e-8)
    assert expected_copula_nll(R_dag, M) <= expected_copula_nll(R_true, M) + 1e-9


class _GaussianFakeTabICL(torch.nn.Module):
    """Stands in for TabICL: N(mean of the context labels, 0.8^2) at every query row."""

    def forward(self, X: torch.Tensor, y: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        batch, T, _ = X.shape
        n = T - y.shape[1]
        loc = y.mean(dim=1, keepdim=True).expand(batch, n)
        scale_raw = torch.full_like(loc, math.log(math.expm1(0.8)))
        return torch.stack([loc, scale_raw], dim=-1)

    def quantile_dist(self, logits_flat: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(logits_flat[:, 0], torch.nn.functional.softplus(logits_flat[:, 1]))


def test_query_marginal_matches_run_pit() -> None:
    torch.manual_seed(0)
    X_train, X_test = torch.randn(7, 2), torch.randn(4, 2)
    y_train, y_test = torch.randn(7), torch.randn(4)
    tabicl = _GaussianFakeTabICL()
    out = run_pit(tabicl, X_train, y_train.unsqueeze(-1), X_test, y_test.unsqueeze(-1), k_folds=2)
    marg = query_marginal(tabicl, X_train, y_train, X_test)
    u = marg.cdf(y_test.unsqueeze(0).repeat(3, 1))
    assert u.shape == (3, 4)
    z = torch.special.ndtri(u[1].clamp(1e-6, 1 - 1e-6))
    assert torch.allclose(z, out["z_test"].squeeze(-1), atol=1e-5)
    log_pdf = marg.log_prob(y_test.unsqueeze(0).repeat(2, 1))[0]
    assert torch.allclose(log_pdf, out["log_pdf_test"].squeeze(-1), atol=1e-5)


def test_attainable_rows_on_a_live_episode() -> None:
    """End to end on a tiny GP episode: finite rows, total = marginal + copula, and R_dagger beats R_po and I."""
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "data": {"d_features": 1, "P_min": 6, "P_max": 8, "N_min": 5, "N_max": 6, "n_tasks": 4, "l_min": 0.5},
        }
    )
    torch.manual_seed(0)
    ep = generate_gp_batch(cfg, B=1, device="cpu", return_kernel_metadata=True)[0]
    tabicl = _GaussianFakeTabICL()
    device = torch.device("cpu")
    marginal_pit = _marginal_pit(ep, tabicl, k_folds=2, device=device)
    ctx: Any = SimpleNamespace(
        spec=SimpleNamespace(
            attainable=AttainableSpec(n_samples=4096, max_iter=200, ranks=[1], bayes_draws=2048, bayes_keep=128), seed=0
        ),
        prior_cfg=OmegaConf.to_container(cfg.data),
        tabicl_marginal=tabicl,
        marginal_backend=None,
        device=device,
        fit_kwargs={"icl_rank": 2},
        icl_model=SimpleNamespace(correlation_parametrization="covnorm"),
    )
    n = ep["z_test"].shape[0]
    R_po = gp_analytical_posterior(ep)["R_post"].float()
    rows, eff = _attainable_rows(
        ctx, ep, 0, marginal_pit, {"oracle": R_po, "icl": torch.eye(n)}, {"oracle": 0.0, "icl": 0.0}
    )
    for key in ("marg_attainable", "marg_attainable_lowrank", "marg_bayes_attainable", "oracle_bayes"):
        parts = rows[key]
        assert all(math.isfinite(v) for v in parts.values()), key
        assert parts["total"] == pytest.approx(parts["marginal"] + parts["copula"], abs=1e-4)
    # Best Gaussian (full) <= rank 2 (the head's) <= rank 1, and R_dagger beats R_po on its own M.
    assert eff["attainable"] <= eff["attainable_r2"] + 1e-6 <= eff["attainable_r1"] + 2e-6
    assert eff["attainable"] <= eff["oracle_posterior"] + 1e-6
    assert eff["head_rank"] == 2
    # Any copula's bound is below the best Gaussian copula (in expectation; small Monte Carlo slack).
    assert eff["any_copula_bound"] <= eff["attainable"] + 0.02
    assert eff["icl"] == pytest.approx(0.0, abs=1e-9)  # R = I
    assert eff["m_diag"] > 0
    # R_bayes is fit to M_bayes, so it cannot beat R_dagger on the true M; on its own M it beats I.
    assert eff["attainable"] <= eff["bayes_attainable"] + 1e-6
    assert eff["bayes_attainable_self"] <= 1e-9 and eff["bayes_ess"] >= 1.0
    # R_dagger is optimal on M, so it beats the copulas fitted to Cov(z) and every Bayes rank.
    for key in ("attainable_centered", "bayes_attainable_centered", "bayes_attainable_r1", "bayes_attainable_r2"):
        assert eff["attainable"] <= eff[key] + 1e-6, key
    assert eff["attainable"] <= eff["corr_z"] + 1e-6 and eff["attainable"] <= eff["bayes_corr_z"] + 1e-6
    assert eff["mean_bias_sq"] >= 0 and -1.0 - 1e-9 <= eff["bayes_bias_cos"] <= 1.0 + 1e-9
    assert "icl_bayes_offdiag_corr" not in eff  # R_icl = I has no off-diagonal pattern
    assert eff["icl_offdiag_abs"] == pytest.approx(0.0, abs=1e-9)

    # Without a TabICL marginal (or a known theta) nothing is computed.
    ctx.tabicl_marginal = None
    rows, eff = _attainable_rows(ctx, ep, 0, marginal_pit, {}, {})
    assert eff == {} and math.isnan(rows["marg_attainable"]["total"])


def _bayes_cfg() -> Any:
    return OmegaConf.create(
        {
            "seed": 3,
            "data": {"d_features": 1, "P_min": 12, "P_max": 12, "N_min": 6, "N_max": 6, "n_tasks": 4, "l_min": 0.5},
        }
    )


def test_bayes_predictive_is_a_normalised_mixture_between_prior_and_known_theta() -> None:
    """Weights sum to 1, RNG states are restored, and on average the Bayes NLL lies between the known-theta
    posterior's (a lower bound in expectation) and the prior's."""
    from copula_inter.data.gp_bayes import bayes_predictive

    cfg = _bayes_cfg()
    torch.manual_seed(0)
    nll_post, nll_bayes, nll_prior = [], [], []
    for seed in range(12):
        cfg.seed = seed
        ep = generate_gp_batch(cfg, B=1, device="cpu", return_kernel_metadata=True)[0]
        state = torch.get_rng_state()
        bp = bayes_predictive(cfg.data, ep, 4096, torch.device("cpu"), seed=100 + seed, max_keep=256)
        assert torch.equal(state, torch.get_rng_state())
        assert bp.log_weights.exp().sum().item() == pytest.approx(1.0, abs=1e-9)
        assert bp.mu.shape[0] <= 256 and bp.ess >= 1.0
        y = ep["y_test"].double().unsqueeze(0)
        n = y.shape[1]
        joint = bp.log_prob(y).item()
        assert bp.marginal_log_prob(y).item() >= joint - 50 * n  # finite and of the same scale
        post = gp_analytical_posterior(ep)
        nll_post.append(post["nll_post"] / n)
        nll_bayes.append(-joint / n)
        nll_prior.append(post["nll_prior"] / n)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    assert mean(nll_post) <= mean(nll_bayes) + 0.05
    assert mean(nll_bayes) <= mean(nll_prior) + 0.05


def test_bayes_predictive_with_one_component_is_that_gaussian() -> None:
    from copula_inter.data.gp_bayes import bayes_predictive

    cfg = _bayes_cfg()
    ep = generate_gp_batch(cfg, B=1, device="cpu", return_kernel_metadata=True)[0]
    bp = bayes_predictive(cfg.data, ep, 64, torch.device("cpu"), seed=1, max_keep=1)
    assert bp.mu.shape[0] == 1 and bp.log_weights.item() == pytest.approx(0.0)
    y = torch.randn(5, bp.mu.shape[1], dtype=torch.float64)
    cov = bp.chol[0] @ bp.chol[0].T
    ref = torch.distributions.MultivariateNormal(bp.mu[0], cov).log_prob(y)
    assert torch.allclose(bp.log_prob(y), ref, atol=1e-8)
    ref_marg = torch.distributions.Normal(bp.mu[0], cov.diagonal().sqrt()).log_prob(y).sum(-1)
    assert torch.allclose(bp.marginal_log_prob(y), ref_marg, atol=1e-8)
    draws = bp.sample(20_000, torch.Generator().manual_seed(0))
    assert torch.allclose(draws.mean(0), bp.mu[0], atol=0.1 * cov.diagonal().sqrt().max().item())
