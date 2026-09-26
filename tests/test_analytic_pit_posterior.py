"""Tests that the analytic PIT standardizes z_test by the GP posterior (not the prior).

Per-episode moments of z_test, agreement between data_gen, gp_analytical_pit
and gp_analytical_posterior, and the consequence that the copula optimum is
R_post.
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from copula_inter.data_gen import generate_gp_batch
from copula_inter.pit import gp_analytical_pit, gp_analytical_posterior


def _episodes(small_cfg, b=24, seed=0):
    """RBF episodes with enough test points for per-episode moments."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.P_min = cfg.data.P_max = 16
    cfg.data.N_min = cfg.data.N_max = 96
    cfg.data.kernel = "rbf"
    torch.manual_seed(seed)
    return generate_gp_batch(cfg, b, "cpu", return_kernel_metadata=True)


def test_z_test_has_unit_variance_conditional_on_the_context(small_cfg):
    """Var(z_test | context) = 1 per episode (prior standardization gives ~0.3)."""
    eps = _episodes(small_cfg, b=32)
    per_ep = np.array([float(ep["z_test"].double().var()) for ep in eps])
    assert 0.85 < per_ep.mean() < 1.15, (
        f"mean per-episode Var(z_test | context) = {per_ep.mean():.4f}; "
        "expected ~1. A value near 0.3 means z_test is standardized by the "
        "PRIOR sigma (data_gen's oracle_mode='prior' mu_star/sigma_star) "
        "instead of the posterior marginals."
    )


def test_z_test_is_centred_conditional_on_the_context(small_cfg):
    """E[z_test | context] = 0 per episode."""
    eps = _episodes(small_cfg, b=32)
    per_ep = np.array([float(ep["z_test"].double().mean()) for ep in eps])
    assert abs(per_ep.mean()) < 0.15, f"mean per-episode E[z_test | context] = {per_ep.mean():+.4f}; expected ~0."


def test_analytic_pit_matches_gp_analytical_posterior_marginals(small_cfg):
    """gp_analytical_pit's implied (mu, sigma) are gp_analytical_posterior's mu_post and sqrt(diag(Sigma_post))."""
    for ep in _episodes(small_cfg, b=12):
        post = gp_analytical_posterior(ep)
        rec = gp_analytical_pit(ep)
        z = rec["z_test"].double()
        # log_pdf = -0.5*log(2pi) - log(sigma) - 0.5*z^2  =>  recover sigma
        sigma = torch.exp(-0.5 * float(np.log(2.0 * np.pi)) - 0.5 * z**2 - rec["log_pdf_test"].double())
        mu = ep["y_test"].double() - z * sigma
        assert torch.allclose(mu, post["mu_post"].double(), atol=1e-3), "mu != mu_post"
        assert torch.allclose(sigma, post["Sigma_post"].double().diagonal().sqrt(), atol=1e-3), (
            "sigma != sqrt(diag(Sigma_post))"
        )


def test_batched_generator_matches_single_episode_pit(small_cfg):
    """data_gen's batched path and gp_analytical_pit give the same z_test/log_pdf_test."""
    for ep in _episodes(small_cfg, b=12):
        rec = gp_analytical_pit(ep)
        assert torch.allclose(rec["z_test"], ep["z_test"], atol=1e-3)
        assert torch.allclose(rec["log_pdf_test"], ep["log_pdf_test"], atol=1e-3)


def test_analytic_pit_is_not_the_prior_standardisation(small_cfg):
    """z_test is not (y_test - mu_star) / sigma_star."""
    max_dev = 0.0
    for ep in _episodes(small_cfg, b=16):
        prior_z = (ep["y_test"].double() - ep["mu_star"].double()) / ep["sigma_star"].double().clamp(min=1e-8)
        max_dev = max(max_dev, float((prior_z - ep["z_test"].double()).abs().max()))
    assert max_dev > 1e-3, (
        "z_test equals the PRIOR standardisation (y - mu_star)/sigma_star; "
        "the analytic PIT has regressed to prior marginals."
    )


def _copula_nll(R, M, n):
    """Expected copula NLL 0.5 (log|R| + tr(R^{-1} M) - tr M) / n at second moment M."""
    L = torch.linalg.cholesky(R)
    logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
    trace = torch.diagonal(torch.cholesky_solve(M, L)).sum()
    return float(0.5 * (logdet + trace - torch.diagonal(M).sum()) / n)


def _second_moment_of_emitted_z(ep, post):
    """E[z z^T | context] for the standardization the episode used:

        M = D_u^{-1} Sigma_post D_u^{-1} + delta delta^T,  delta = D_u^{-1} (mu_post - mu_used)

    with (mu_used, sigma_used) recovered from the emitted z_test/log_pdf_test.
    """
    z = ep["z_test"].double()
    sigma_u = torch.exp(-0.5 * float(np.log(2.0 * np.pi)) - 0.5 * z**2 - ep["log_pdf_test"].double())
    mu_u = ep["y_test"].double() - z * sigma_u
    Dinv = torch.diag(1.0 / sigma_u)
    delta = Dinv @ (post["mu_post"].double() - mu_u)
    return Dinv @ post["Sigma_post"].double() @ Dinv + torch.outer(delta, delta)


def test_copula_optimum_is_the_posterior_correlation(small_cfg):
    """At the emitted z's second moment, R_post scores better than R_star."""
    wins = total = 0
    for ep in _episodes(small_cfg, b=24):
        post = gp_analytical_posterior(ep)
        n = int(ep["x_norm_test"].shape[0])
        M = _second_moment_of_emitted_z(ep, post)
        jit = 1e-8 * torch.eye(n, dtype=torch.float64)
        c_post = _copula_nll(post["R_post"].double() + jit, M, n)
        c_star = _copula_nll(ep["R_star"].double() + jit, M, n)
        c_indep = _copula_nll(torch.eye(n, dtype=torch.float64), M, n)
        total += 1
        wins += int(c_post <= c_star and c_post <= c_indep)
    assert wins == total, (
        f"R_post was the copula optimum in only {wins}/{total} episodes; "
        "if R_star or independence wins, z_test is prior-standardized again."
    )


def test_conditional_second_moment_is_a_correlation_matrix(small_cfg):
    """diag(E[z z^T | context]) = 1 for the emitted z."""
    diags = []
    worst = 0.0
    for ep in _episodes(small_cfg, b=12):
        post = gp_analytical_posterior(ep)
        d = _second_moment_of_emitted_z(ep, post).diagonal()
        diags.append(float(d.mean()))
        worst = max(worst, float((d - 1.0).abs().max()))
    assert worst < 1e-3, (
        f"max |diag(M_c) - 1| = {worst:.4f} across episodes "
        f"(per-episode means {min(diags):.4f}..{max(diags):.4f}); expected exactly 1. "
        "A diagonal != 1 means z_test carries a mean shift and/or a variance "
        "ratio, i.e. it is prior-standardized."
    )
