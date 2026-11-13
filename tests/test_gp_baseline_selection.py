"""GP baselines condition on the context, BIC selection from the context alone, and the marginal plug-in rows."""

from __future__ import annotations

import math

import pytest
import torch

from copula_inter.baselines.gp_mle import fit_and_eval_gpytorch
from eval.baselines.classical import _evidence_rows
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


def test_tables_rank_only_context_only_selections() -> None:
    """gp_evidence (a copy of a ranked row) is in the total table but not the rank; references never ranked."""
    total, rank = dict(_TOTAL_NLL_ORDER), dict(_TOTAL_RANK_ORDER)
    for key in ("gp_evidence", "gp_evidence_mix", "marg_gp_evidence", "marg_oracle_corr"):
        assert key in total
    assert "gp_evidence" not in rank
    assert "gp_evidence_mix" in rank and "marg_gp_evidence" in rank
    assert "marg_oracle_corr" not in rank and "oracle_posterior" not in rank
    assert "oracle_prior" in _METHOD_LABELS
