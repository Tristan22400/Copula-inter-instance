"""test_analytic_pit_posterior.py — pins the test-side marginal of the exact
analytic PIT to the GP POSTERIOR, not the prior.

Why this file exists. ``pit.gp_analytical_pit`` is the exact-GP stand-in for
what a frozen TabICL supplies, and ``pit.run_pit`` calls TabICL WITH the
episode's context labels in-context (``tabicl(X_test_batch, y_train_batch)``),
so its ``dist`` -- hence ``u_test`` / ``log_pdf_test`` -- is a POSTERIOR
PREDICTIVE marginal. The analytic path must mirror that, or
``data.z_train_source`` "analytic" and "tabicl" are two different problems
rather than two estimates of the same z.

It silently stopped mirroring it. The function's docstring said "posterior
(test) marginals" in four places while reading ``task["mu_star"]`` /
``["sigma_star"]``, which ``data_gen``'s surviving ``oracle_mode="prior"``
branch fills with the PRIOR mean and std. Nothing in the suite failed, because
nothing asserted the contract -- so this file asserts it, from two independent
directions:

  1. Distributional (cases 1-2). If z_test is standardized by the posterior
     then ``Var(z_test | context) == 1`` for EVERY episode. Under the prior it
     was ~0.30, because conditioning shrinks the variance while the divisor
     stays at the prior sigma. Marginalizing over y_train hides this -- pooled
     variance is ~1 either way -- so the check has to be per-episode.

  2. Cross-implementation (cases 3-5). The same marginal is computed in three
     places: ``data_gen._generate_gp_batch_raw`` (batched, feeds training),
     ``pit.gp_analytical_pit`` (single episode / disk reconstruction), and
     ``pit.gp_analytical_posterior`` (the validation ceiling). They must agree,
     and agree on the POSTERIOR value specifically.

Why it matters beyond naming: the copula objective's optimum is the conditional
second moment ``M_c = E[z z^T | context]``. Prior standardization leaves a
non-zero mean shift inside z, giving ``M_c = D^-1 Sigma_post D^-1 + m m^T``
whose y_train-average is the context-blind PRIOR correlation R_star (law of
total variance). Posterior standardization sets m = 0, so ``M_c = R_post``
exactly -- the conditional correlation the validation ceiling scores against.
Cases 6-7 pin that consequence directly.

No GPU, no network, no TabICL checkpoint.
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from data_gen import generate_gp_batch
from pit import gp_analytical_pit, gp_analytical_posterior


def _episodes(small_cfg, b=24, seed=0):
    """A batch of RBF episodes with enough test points for per-episode moments."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.P_min = cfg.data.P_max = 16
    cfg.data.N_min = cfg.data.N_max = 96
    cfg.data.kernel = "rbf"
    torch.manual_seed(seed)
    return generate_gp_batch(cfg, b, "cpu", return_kernel_metadata=True)


# ---------------------------------------------------------------------------
# 1-2. Distributional: z_test is standardized by the posterior
# ---------------------------------------------------------------------------

def test_z_test_has_unit_variance_conditional_on_the_context(small_cfg):
    """Var(z_test | context) == 1 per episode.

    This is THE regression guard. Prior standardization gives ~0.30 here,
    because Var(z | D) = diag(Sigma_post)/sigma_prior^2 < 1 once the context is
    conditioned on. Averaged over y_train the prior version still looks fine,
    which is exactly how the bug survived -- so this averages the PER-EPISODE
    variances, which the prior version cannot pass.
    """
    eps = _episodes(small_cfg, b=32)
    per_ep = np.array([float(ep["z_test"].double().var()) for ep in eps])
    assert 0.85 < per_ep.mean() < 1.15, (
        f"mean per-episode Var(z_test | context) = {per_ep.mean():.4f}; "
        "expected ~1. A value near 0.3 means z_test is standardized by the "
        "PRIOR sigma (data_gen's oracle_mode='prior' mu_star/sigma_star) "
        "instead of the posterior marginals."
    )


def test_z_test_is_centred_conditional_on_the_context(small_cfg):
    """E[z_test | context] == 0 per episode.

    Under prior standardization the conditional mean is
    m_c = (mu_post - mu_pr)/sigma_pr, a smooth non-zero field -- and that
    rank-one term is what carried the correlation target to R_star.
    """
    eps = _episodes(small_cfg, b=32)
    per_ep = np.array([float(ep["z_test"].double().mean()) for ep in eps])
    assert abs(per_ep.mean()) < 0.15, (
        f"mean per-episode E[z_test | context] = {per_ep.mean():+.4f}; expected ~0."
    )


# ---------------------------------------------------------------------------
# 3-5. Cross-implementation agreement, on the posterior value specifically
# ---------------------------------------------------------------------------

def test_analytic_pit_matches_gp_analytical_posterior_marginals(small_cfg):
    """gp_analytical_pit's implied (mu, sigma) are gp_analytical_posterior's
    mu_post and sqrt(diag(Sigma_post)) -- one quantity, two call sites.

    Recovered from the emitted z_test/log_pdf_test rather than read off an
    internal variable, so this pins the values callers actually receive.
    """
    for ep in _episodes(small_cfg, b=12):
        post = gp_analytical_posterior(ep)
        rec = gp_analytical_pit(ep)
        z = rec["z_test"].double()
        # log_pdf = -0.5*log(2pi) - log(sigma) - 0.5*z^2  =>  recover sigma
        sigma = torch.exp(
            -0.5 * float(np.log(2.0 * np.pi)) - 0.5 * z**2 - rec["log_pdf_test"].double()
        )
        mu = ep["y_test"].double() - z * sigma
        assert torch.allclose(mu, post["mu_post"].double(), atol=1e-3), "mu != mu_post"
        assert torch.allclose(
            sigma, post["Sigma_post"].double().diagonal().sqrt(), atol=1e-3
        ), "sigma != sqrt(diag(Sigma_post))"


def test_batched_generator_matches_single_episode_pit(small_cfg):
    """The batched path that feeds training and the single-episode path used
    for disk reconstruction must produce the same z_test/log_pdf_test."""
    for ep in _episodes(small_cfg, b=12):
        rec = gp_analytical_pit(ep)
        assert torch.allclose(rec["z_test"], ep["z_test"], atol=1e-3)
        assert torch.allclose(rec["log_pdf_test"], ep["log_pdf_test"], atol=1e-3)


def test_analytic_pit_is_not_the_prior_standardisation(small_cfg):
    """Explicitly reject the old behaviour.

    mu_star/sigma_star are still on every episode -- R_star/Sigma_star are
    built from them, and gp_analytical_posterior reads mu_star as its prior
    mean term -- so a well-meaning "simplification" back to
    (y_test - mu_star)/sigma_star is one line away.
    """
    max_dev = 0.0
    for ep in _episodes(small_cfg, b=16):
        prior_z = (ep["y_test"].double() - ep["mu_star"].double()) / ep[
            "sigma_star"
        ].double().clamp(min=1e-8)
        max_dev = max(max_dev, float((prior_z - ep["z_test"].double()).abs().max()))
    assert max_dev > 1e-3, (
        "z_test equals the PRIOR standardisation (y - mu_star)/sigma_star; "
        "the analytic PIT has regressed to prior marginals."
    )


# ---------------------------------------------------------------------------
# 6-7. The consequence: the copula target is R_post, not R_star
# ---------------------------------------------------------------------------

def _copula_nll(R, M, n):
    """0.5*(log|R| + tr(R^-1 M) - tr M)/n -- loss.y_space_nll's copula term
    with a fixed R, and the exact conditional second moment M in place of
    z z^T (so this is the expected value, with no sampling noise)."""
    L = torch.linalg.cholesky(R)
    logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
    trace = torch.diagonal(torch.cholesky_solve(M, L)).sum()
    return float(0.5 * (logdet + trace - torch.diagonal(M).sum()) / n)


def _second_moment_of_emitted_z(ep, post):
    """E[z z^T | context] for the standardisation the episode ACTUALLY used.

    Recovers (mu_used, sigma_used) from the emitted z_test/log_pdf_test rather
    than assuming them, so these tests exercise the pipeline instead of
    restating the algebra. With y | context ~ N(mu_post, Sigma_post) and
    z = D_u^-1 (y - mu_used):

        M = D_u^-1 Sigma_post D_u^-1 + delta delta^T,
        delta = D_u^-1 (mu_post - mu_used)

    Posterior standardisation makes mu_used = mu_post and D_u^2 =
    diag(Sigma_post), so delta = 0 and M = R_post. Prior standardisation
    leaves the rank-one delta term, which is what dragged the optimum to
    R_star.
    """
    z = ep["z_test"].double()
    sigma_u = torch.exp(
        -0.5 * float(np.log(2.0 * np.pi)) - 0.5 * z**2 - ep["log_pdf_test"].double()
    )
    mu_u = ep["y_test"].double() - z * sigma_u
    Dinv = torch.diag(1.0 / sigma_u)
    delta = Dinv @ (post["mu_post"].double() - mu_u)
    return Dinv @ post["Sigma_post"].double() @ Dinv + torch.outer(delta, delta)


def test_copula_optimum_is_the_posterior_correlation(small_cfg):
    """The copula objective's minimiser over correlation matrices is R_post.

    Scored at the exact conditional second moment of the z the generator
    actually emitted, so this fails if the standardisation regresses. Under
    prior standardisation R_star wins this comparison by ~0.7 nats/pt, which is
    what made the copula head chase a context-blind matrix.
    """
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
    """diag(E[z z^T | context]) == 1 for the emitted z.

    This is what makes R_post *feasible* for the model's unit-diagonal
    parametrisation, hence actually attainable. Under prior standardisation
    this diagonal is ~0.3 + the squared mean shift, so the unconstrained
    optimum sat outside the model family entirely.
    """
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
