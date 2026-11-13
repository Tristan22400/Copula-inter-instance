"""GP baselines condition on the context, BIC / Laplace selection from the context alone, and the marginal plug-in rows."""

from __future__ import annotations

import math

import gpytorch
import pytest
import torch
from gpytorch.priors import GammaPrior

from copula_inter.baselines.gp_mle import (
    _ExactGPModel,
    _period_prior_quantiles,
    fit_and_eval_gpytorch,
    laplace_log_evidence,
)
from eval.baselines.classical import _evidence_rows, _laplace_rows
from eval.runners.episode_scoring import _marginal_plugin_rows
from eval.runners.eval_tables import _METHOD_LABELS, _TOTAL_NLL_ORDER, _TOTAL_RANK_ORDER


def _smooth_1d(n: int = 12, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.linspace(-2.0, 2.0, n).unsqueeze(-1)
    y = 3.0 + torch.sin(2.0 * x.squeeze(-1)) + 0.05 * torch.randn(n, generator=gen)
    return x, y


def _fit(mode: str, mean: str) -> dict[str, torch.Tensor]:
    x, y = _smooth_1d()
    torch.manual_seed(0)
    return fit_and_eval_gpytorch(x, y, x, "rbf", n_steps=150, lr=0.05, oracle_mode=mode, mean=mean, n_restarts=1)


def test_posterior_mode_conditions_on_the_context() -> None:
    """At the context inputs, the posterior mean tracks y; the prior mean ignores it (a constant)."""
    _, y = _smooth_1d()
    post = _fit("posterior", "constant")
    prior = _fit("prior", "constant")
    assert torch.corrcoef(torch.stack([post["mean"], y]))[0, 1] > 0.95
    assert torch.allclose(prior["mean"], prior["mean"][0].expand_as(prior["mean"]))
    # Conditioning shrinks the predictive variance below the prior's.
    assert (post["Sigma"].diagonal() < prior["Sigma"].diagonal()).all()


@pytest.mark.parametrize(("mean", "n_hparams"), [("zero", 3), ("constant", 4), ("linear", 5)])
def test_log_ml_and_hparam_count(mean: str, n_hparams: int) -> None:
    """RBF: lengthscale, outputscale, noise, plus the mean's parameters (constant: 1; linear in 1-D: 2)."""
    fit = _fit("posterior", mean)
    assert math.isfinite(float(fit["log_ml"]))
    assert int(fit["n_hparams"]) == n_hparams


def test_unknown_mean_raises() -> None:
    x, y = _smooth_1d()
    with pytest.raises(ValueError, match="mean"):
        fit_and_eval_gpytorch(x, y, x, "rbf", n_steps=1, lr=0.05, mean="quadratic")


def _fake_fit(log_ml: float, n_hparams: int, n: int) -> dict[str, torch.Tensor]:
    return {"R": torch.eye(n), "log_ml": torch.tensor(log_ml), "n_hparams": torch.tensor(float(n_hparams))}


def test_evidence_rows_select_by_bic_and_mix() -> None:
    """BIC = log_ml - k/2 log P picks the selection; the mixture is -log sum_k w_k exp(-N total_k) / N."""
    n, P = 4, 16
    z = torch.zeros(n)
    fits = {
        "gp_mle_rbf": _fake_fit(-10.0, 3, n),  # BIC = -10 - 1.5 log 16
        "gp_mle_ard_rbf": _fake_fit(-9.0, 6, n),  # higher log_ml, but BIC = -9 - 3 log 16 is lower
    }
    y = {
        "gp_mle_rbf": {"total": 1.0, "marginal": 0.8, "copula": 0.2},
        "gp_mle_ard_rbf": {"total": 2.0, "marginal": 1.5, "copula": 0.5},
    }
    nlls, R, y_rows, selected = _evidence_rows(fits, y, z, P)

    assert selected == "gp_mle_rbf"
    assert y_rows["gp_evidence"] == y["gp_mle_rbf"]
    assert nlls["gp_evidence"] == pytest.approx(0.0, abs=1e-5)
    assert torch.equal(R["gp_evidence"], fits["gp_mle_rbf"]["R"])
    bic = torch.tensor([-10.0 - 1.5 * math.log(P), -9.0 - 3.0 * math.log(P)], dtype=torch.float64)
    w = torch.softmax(bic, 0)
    expected = -math.log(w[0] * math.exp(-n * 1.0) + w[1] * math.exp(-n * 2.0)) / n
    assert y_rows["gp_evidence_mix"]["total"] == pytest.approx(expected)
    assert math.isnan(y_rows["gp_evidence_mix"]["marginal"])


def test_evidence_rows_skip_failed_fits() -> None:
    """A fit with a non-finite total cannot be selected; with none left, every row is nan."""
    n = 3
    fits = {"gp_mle_rbf": _fake_fit(-5.0, 3, n)}
    y = {"gp_mle_rbf": {"total": float("nan"), "marginal": float("nan"), "copula": float("nan")}}
    nlls, _, y_rows, selected = _evidence_rows(fits, y, torch.zeros(n), 8)
    assert selected is None
    assert math.isnan(nlls["gp_evidence"])
    assert math.isnan(y_rows["gp_evidence_mix"]["total"])


def _fixed_rbf_model(
    x: torch.Tensor, y: torch.Tensor, outputscale_prior: GammaPrior | None
) -> tuple[_ExactGPModel, torch.Tensor]:
    """Constant-mean RBF GP with fixed lengthscale and noise (float64); returns it and K = k(x, x) + noise I."""
    x, y = x.double(), y.double()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kp = {"outputscale_prior": outputscale_prior} if outputscale_prior is not None else {}
    model = _ExactGPModel(x, y, likelihood, "rbf", kernel_priors=kp, mean="constant").double()
    model.covar_module.base_kernel.lengthscale = 0.7
    model.covar_module.outputscale = 1.3
    likelihood.noise = 0.05
    for name, param in model.named_parameters():
        param.requires_grad_(name == "mean_module.raw_constant" or name == "covar_module.raw_outputscale")
    model.covar_module.raw_outputscale.requires_grad_(outputscale_prior is not None)
    with torch.no_grad():
        K = model.covar_module(x).to_dense() + 0.05 * torch.eye(x.shape[0], dtype=torch.float64)
    return model, K


def test_laplace_is_exact_for_a_flat_constant_mean() -> None:
    """Only the constant mean c is free (flat prior): the log joint is quadratic in c, so Laplace is exact.

    int N(y; c 1, K) dc = N(y; 0, K) exp(b^2 / (2 a)) sqrt(2 pi / a),  a = 1'K^-1 1,  b = 1'K^-1 y.
    """
    x, y = _smooth_1d()
    model, K = _fixed_rbf_model(x, y, None)
    y64, one = y.double(), torch.ones(x.shape[0], dtype=torch.float64)
    a = one @ torch.linalg.solve(K, one)
    b = one @ torch.linalg.solve(K, y64)
    log_n0 = torch.distributions.MultivariateNormal(torch.zeros_like(y64), K).log_prob(y64)
    expected = float(log_n0 + b**2 / (2 * a) + 0.5 * torch.log(2 * math.pi / a))
    assert laplace_log_evidence(model, x, y) == pytest.approx(expected, abs=1e-6)


def test_laplace_matches_a_numerical_laplace_in_raw_coordinates() -> None:
    """Free outputscale s = softplus(r) (Gamma prior), mean fixed at 0: compare with the same integral on a grid.

    The grid integrand in r is p(y | s) pi(s) sigmoid(r) (sigmoid = ds/dr, the Jacobian). Its numerical
    Laplace (mode + finite-difference curvature) must match laplace_log_evidence; the exact integral
    differs by the Laplace error itself (~0.06 nats here: the posterior on r is wide and skewed).
    """
    x, y = _smooth_1d()
    y = y - y.mean()
    prior = GammaPrior(2.0, 1.0).double()
    model, _ = _fixed_rbf_model(x, y, prior)
    model.mean_module.raw_constant.requires_grad_(False)
    with torch.no_grad():
        model.mean_module.raw_constant.zero_()
    lap = laplace_log_evidence(model, x, y)

    x64, y64 = x.double(), y.double()
    eye = torch.eye(x.shape[0], dtype=torch.float64)
    with torch.no_grad():
        Kx = model.covar_module.base_kernel(x64).to_dense()
        r = torch.linspace(-6.0, 6.0, 2401, dtype=torch.float64)
        s = torch.nn.functional.softplus(r)
        log_lik = torch.stack(
            [
                torch.distributions.MultivariateNormal(torch.zeros_like(y64), si * Kx + 0.05 * eye).log_prob(y64)
                for si in s
            ]
        )
        f = log_lik + prior.log_prob(s) + torch.log(torch.sigmoid(r))
        h = float(r[1] - r[0])
        i = int(f.argmax())
        curvature = -float(f[i + 1] - 2 * f[i] + f[i - 1]) / h**2
        numerical_laplace = float(f[i]) + 0.5 * math.log(2 * math.pi / curvature)
        exact = float(torch.logsumexp(f, 0)) + math.log(h)
    assert lap == pytest.approx(numerical_laplace, abs=5e-3)
    assert lap == pytest.approx(exact, abs=0.15)


@pytest.mark.parametrize(("kernel", "ard"), [("rbf", False), ("rbf", True), ("dot_product", False)])
def test_fit_returns_log_evidence_only_when_asked(kernel: str, ard: bool) -> None:
    """laplace=True: a finite evidence below the MAP log likelihood's scale; laplace=False: nan."""
    x, y = _smooth_1d(n=16)
    x = torch.cat([x, 0.3 * x**2], dim=-1)
    torch.manual_seed(0)
    fit = fit_and_eval_gpytorch(x, y, x, kernel, n_steps=150, lr=0.05, ard=ard, mean="constant", laplace=True)
    assert math.isfinite(float(fit["log_evidence"]))
    # The evidence averages the likelihood over the prior, so it cannot beat the best fit by much.
    assert float(fit["log_evidence"]) < float(fit["log_ml"]) + 1.0
    torch.manual_seed(0)
    off = fit_and_eval_gpytorch(x, y, x, kernel, n_steps=5, lr=0.05, ard=ard, mean="constant")
    assert math.isnan(float(off["log_evidence"]))


def test_laplace_rows_weight_by_log_evidence() -> None:
    """gp_laplace picks the highest log evidence (not log_ml); nan evidences are dropped."""
    n = 4
    z = torch.zeros(n)
    fits = {
        "gp_mle_rbf": {**_fake_fit(-9.0, 3, n), "log_evidence": torch.tensor(-14.0)},
        "gp_mle_rq": {**_fake_fit(-10.0, 4, n), "log_evidence": torch.tensor(-12.0)},
        "gp_mle_polynomial": {**_fake_fit(-1.0, 3, n), "log_evidence": torch.tensor(float("nan"))},
    }
    y = {k: {"total": t, "marginal": t, "copula": 0.0} for k, t in zip(fits, (1.0, 2.0, 0.5))}
    _, _, y_rows, selected = _laplace_rows(fits, y, z)
    assert selected == "gp_mle_rq"
    w = torch.softmax(torch.tensor([-14.0, -12.0], dtype=torch.float64), 0)
    expected = -math.log(w[0] * math.exp(-n * 1.0) + w[1] * math.exp(-n * 2.0)) / n
    assert y_rows["gp_laplace_mix"]["total"] == pytest.approx(expected)


def test_marginal_plugin_rows() -> None:
    """Learned marginal + plug-in R: total = marginal + copula; a failed selection or no oracle gives nan."""
    n = 3
    z = torch.tensor([0.5, -0.2, 1.0])
    pit = {"z_train": torch.zeros(5), "z_test": z, "log_pdf_test": -0.5 * (z**2 + math.log(2 * math.pi))}
    R = torch.tensor([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cpu = torch.device("cpu")

    rows = _marginal_plugin_rows(pit, {"gp_evidence": 0.1}, {"gp_evidence": R}, None, cpu)
    parts = rows["marg_gp_evidence"]
    assert parts["total"] == pytest.approx(parts["marginal"] + parts["copula"], abs=1e-5)
    assert parts["copula"] != pytest.approx(0.0)
    assert math.isnan(rows["marg_oracle_corr"]["total"])

    failed = _marginal_plugin_rows(pit, {"gp_evidence": float("nan")}, {"gp_evidence": torch.eye(n)}, R, cpu)
    assert math.isnan(failed["marg_gp_evidence"]["total"])
    assert failed["marg_oracle_corr"]["total"] == pytest.approx(parts["total"])
    assert all(math.isnan(v["total"]) for v in _marginal_plugin_rows(None, {}, {}, R, cpu).values())
    assert math.isnan(rows["marg_gp_laplace"]["total"])  # gp_laplace not given


def test_tables_rank_only_context_only_selections() -> None:
    """gp_evidence (a copy of a ranked row) is in the total table but not the rank; references never ranked."""
    total, rank = dict(_TOTAL_NLL_ORDER), dict(_TOTAL_RANK_ORDER)
    for sel in ("gp_evidence", "gp_laplace"):
        for key in (sel, f"{sel}_mix", f"marg_{sel}"):
            assert key in total
        assert sel not in rank
        assert f"{sel}_mix" in rank and f"marg_{sel}" in rank
    assert "marg_oracle_corr" in total
    assert "marg_oracle_corr" not in rank and "oracle_posterior" not in rank
    assert "oracle_prior" in _METHOD_LABELS


def test_periodic_restarts_start_at_stratified_period_quantiles() -> None:
    """Increasing, symmetric in log space around the prior median exp(loc) = 1.2 (n odd: the median itself)."""
    q = _period_prior_quantiles({}, 9)
    assert q == sorted(q) and len(set(q)) == 9
    assert q[4] == pytest.approx(1.2)
    assert math.log(q[0]) + math.log(q[-1]) == pytest.approx(2 * math.log(1.2))
