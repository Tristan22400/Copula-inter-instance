"""
test_eval_checkpoint.py — regression tests for eval/baselines/classical.py
and eval/runners/eval_checkpoint.py.

No live checkpoint or network access required: episodes are tiny
live-generated GP draws (via data_gen.generate_gp_batch), and the ICL model
under test is a fake nn.Module matching CopulaTabICL's forward(batch) ->
{"W": ..., "s": ...} contract (same pattern as test_copula_inference.py's
_FakeCopulaModel) rather than a real TabICL backbone.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
from omegaconf import OmegaConf

_TESTS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS)
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gen import generate_gp_batch  # noqa: E402

from eval.baselines.classical import (  # noqa: E402
    baseline_fingerprint,
    episode_cache_key,
    eval_baselines_episode,
    load_baseline_cache,
    save_baseline_cache,
)
from eval.runners.eval_checkpoint import _eval_icl_episode  # noqa: E402
from pit import gp_analytical_posterior  # noqa: E402

_TINY_DATA_CFG = {
    "d_features": 1,
    "P_min": 5, "P_max": 8,
    "N_min": 4, "N_max": 6,
    "n_tasks": 4,
    "l_min": 0.5, "l_max": 1.5,
    "alpha2_min": 0.5, "alpha2_max": 1.5,
    "noise_min": 0.05, "noise_max": 0.2,
}


@pytest.fixture(scope="module")
def tiny_episode():
    cfg = OmegaConf.create({"seed": 0, "data": dict(_TINY_DATA_CFG)})
    torch.manual_seed(0)
    return generate_gp_batch(cfg, B=1, device="cpu", return_kernel_metadata=True)[0]


class _FakeICLModel(torch.nn.Module):
    """Stands in for CopulaTabICL: forward(batch) -> {"W": ..., "s": ...},
    ignoring the batch contents, so _eval_icl_episode can be exercised
    without a real TabICL backbone."""

    def __init__(self, n_test: int, rank: int):
        super().__init__()
        self.W = torch.randn(1, n_test, rank) * 0.3
        self.s = torch.randn(1, n_test)
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict) -> dict:
        return {"W": self.W, "s": self.s}


def _assert_valid_correlation(R: torch.Tensor, n: int, atol: float = 1e-3):
    assert R.shape == (n, n)
    assert torch.allclose(R, R.T, atol=atol)
    assert torch.allclose(R.diagonal(), torch.ones(n), atol=1e-2)


def test_eval_baselines_episode_runs_and_returns_valid_correlations(tiny_episode):
    """Every baseline in classical.py should fit (or safely fall back) on a
    tiny episode with minimal fitting steps, returning finite NLLs and
    well-formed correlation matrices — the "all baselines, correctly,
    together" contract eval_checkpoint.py relies on."""
    n_test = tiny_episode["x_norm_test"].shape[0]

    nlls, R_dict, y_space_nlls = eval_baselines_episode(
        ep=tiny_episode,
        icl_rank=2,
        n_steps_mle=3,
        lr_mle=0.1,
        n_steps_dkl=3,
        lr_dkl=0.1,
        n_steps_per_ep=3,
        patience_per_ep=2,
        device=torch.device("cpu"),
        oracle_mode="prior",
        n_restarts_mle=1,
    )

    expected_keys = {
        "independence", "gp_prior_rbf",
        "gp_mle_rbf", "gp_mle_ard_rbf", "gp_mle_matern32", "gp_mle_ard_matern32",
        "gp_mle_periodic", "gp_mle_ard_periodic", "gp_mle_rq", "gp_mle_ard_rq",
        "gp_mle_dot_product", "gp_mle_polynomial",
        "dkl_rbf", "dkl_matern32", "dkl_rq", "dkl_dot_product",
        "per_ep_transformer",
    }
    # y_space_nlls excludes independence/gp_prior_rbf (unfit references, no
    # genuine marginal — see eval_baselines_episode's docstring).
    expected_y_keys = expected_keys - {"independence", "gp_prior_rbf"}
    assert expected_keys <= nlls.keys()
    assert expected_keys <= R_dict.keys()
    assert expected_y_keys <= y_space_nlls.keys()

    assert abs(nlls["independence"]) < 1e-3
    _assert_valid_correlation(R_dict["independence"], n_test)

    # Every method must produce a finite NLL and a well-formed R, even the
    # ones expected to fit poorly at 3 Adam steps on a tiny episode — a NaN
    # or malformed matrix here means the fit-or-fallback path silently broke.
    for method in expected_keys:
        assert torch.isfinite(torch.tensor(nlls[method])), f"{method} produced a non-finite NLL"
        _assert_valid_correlation(R_dict[method], n_test)
    for method in expected_y_keys:
        parts = y_space_nlls[method]
        assert set(parts.keys()) == {"total", "marginal", "copula"}
        for part_name, val in parts.items():
            assert torch.isfinite(torch.tensor(val)), \
                f"{method}'s {part_name} Y-space NLL is non-finite"
        # Sklar's theorem: total = marginal + copula exactly, by construction
        # (see classical.py's _nll_parts / gp_oracle_y_nll), not just approximately.
        assert parts["total"] == pytest.approx(parts["marginal"] + parts["copula"], abs=1e-3)


def test_eval_icl_episode_scores_against_oracle(tiny_episode):
    n_test = tiny_episode["x_norm_test"].shape[0]
    fake_model = _FakeICLModel(n_test=n_test, rank=2)

    nlls, R_dict, R_oracle, y_space_nlls, icl_y_parts = _eval_icl_episode(
        ep=tiny_episode, icl_model=fake_model, device=torch.device("cpu"),
    )

    assert set(nlls.keys()) == {"icl", "oracle"}
    assert torch.isfinite(torch.tensor(nlls["icl"]))
    assert torch.isfinite(torch.tensor(nlls["oracle"]))
    _assert_valid_correlation(R_dict["icl"], n_test)
    assert torch.equal(R_oracle, tiny_episode["R_star"])
    # No tabicl_pit given (oracle z_train mode, the default) -> no learned
    # ICL marginal to score a total Y-space NLL against.
    assert set(icl_y_parts.keys()) == {"total", "marginal", "copula"}
    for val in icl_y_parts.values():
        assert torch.isnan(torch.tensor(val))
    # gp_analytical_posterior's prior/posterior are always available for this
    # (elementary-kernel) tiny episode -- both come back as a genuine
    # total/marginal/copula split, exactly consistent by construction.
    for key in ("prior", "posterior"):
        parts = y_space_nlls[key]
        assert set(parts.keys()) == {"total", "marginal", "copula"}
        for val in parts.values():
            assert torch.isfinite(torch.tensor(val))
        assert parts["total"] == pytest.approx(parts["marginal"] + parts["copula"], abs=1e-3)


def test_eval_icl_episode_with_tabicl_pit_populates_total_nll(tiny_episode):
    """With a (fake, standard-normal) tabicl_pit supplied -- standing in for
    --z_train_source=tabicl's real TabICL marginal -- icl_y_parts should be
    finite and internally consistent (total = marginal + copula), the same
    contract as every fitted baseline's own y_space_nlls entry."""
    n_train = tiny_episode["x_norm_train"].shape[0]
    n_test = tiny_episode["x_norm_test"].shape[0]
    fake_model = _FakeICLModel(n_test=n_test, rank=2)

    z_test = torch.randn(n_test)
    tabicl_pit = {
        "z_train": torch.randn(n_train),
        "z_test": z_test,
        # Standard-normal log-density at z_test -- a made-up but valid
        # "marginal" for this smoke test (matches what run_pit would return
        # under a marginal that reproduces the standard normal exactly).
        "log_pdf_test": -0.5 * (z_test ** 2 + math.log(2 * math.pi)),
    }

    _, _, _, _, icl_y_parts = _eval_icl_episode(
        ep=tiny_episode, icl_model=fake_model, device=torch.device("cpu"),
        tabicl_pit=tabicl_pit,
    )

    assert set(icl_y_parts.keys()) == {"total", "marginal", "copula"}
    for val in icl_y_parts.values():
        assert torch.isfinite(torch.tensor(val))
    assert icl_y_parts["total"] == pytest.approx(
        icl_y_parts["marginal"] + icl_y_parts["copula"], abs=1e-3,
    )


def test_gp_oracle_posterior_total_nll_bayes_optimal(tiny_episode):
    """The Y-space total-NLL table's oracle_prior/oracle_posterior rows
    (see _print_total_nll_table) are gp_analytical_posterior's own
    nll_prior/nll_post divided by N — dividing by a positive constant N
    can't flip the Bayes-optimality guarantee posterior <= prior
    (see gp_analytical_posterior's docstring), so it must still hold here."""
    post = gp_analytical_posterior(tiny_episode)
    assert post["nll_post"] <= post["nll_prior"] + 1e-6


def _near_duplicate_rbf_task(alpha2: float) -> dict:
    """Minimal, fully deterministic (no generate_gp_batch randomness) RBF
    task with two near-duplicate test points assigned different y values --
    forces Sigma_post's Schur complement indefinite along their difference
    direction regardless of alpha2 (outputscale), so gp_analytical_posterior's
    PSD-repair eigenvalue floor always fires here. Varying alpha2 varies
    Sigma_post's own natural scale, which is exactly the axis the eig_floor
    regression below needs."""
    zero = torch.zeros(1)
    x_train = torch.tensor([[-1.0], [0.0], [1.0]])
    x_test = torch.tensor([[0.50000], [0.50001], [-0.7]])  # first two are near-duplicates
    return {
        "kernel": "rbf", "l": torch.tensor([0.3]), "alpha2": torch.tensor([alpha2]),
        "nugget": torch.tensor([1e-4]),
        "period": zero, "rq_alpha": zero, "power": zero,
        "l_b": zero, "alpha2_b": zero, "period_b": zero,
        "rq_alpha_b": zero, "power_b": zero,
        "kernel_feature_indices": torch.tensor([0]),
        "x_norm_train": x_train, "x_norm_test": x_test,
        "y_train": torch.tensor([0.5, -0.3, 0.8]),
        "y_test": torch.tensor([1.0, -1.0, 0.5]),  # near-duplicates disagree by 2.0
        "mu_star": torch.zeros(3),
    }


def test_gp_analytical_posterior_eig_floor_scale_invariant():
    """Regression test for a real training incident: gp_analytical_posterior's
    PSD-repair eigenvalue floor (see its docstring) used to be a fixed
    absolute constant (1e-6) rather than relative to Sigma_post's own scale.
    That's fine for an O(1)-scale RBF/Matern posterior, but for a kernel
    whose Sigma_post legitimately has O(1e8)+ scale (large outputscale, or
    a high-degree "polynomial" kernel), flooring a repaired eigenvalue down
    to an absolute 1e-6 manufactures a residual^2/(2*1e-6) blowup out of an
    otherwise perfectly ordinary residual. In production this showed up as
    val/y_nll_oracle_posterior ~110 nats/point (vs <1 for a normal episode)
    and oracle_diag/gap_nll ~-109 (should be >=0 in expectation, since the
    Bayes-optimal posterior can't be beaten -- see
    test_gp_oracle_posterior_total_nll_bayes_optimal above).

    The near-duplicate-test-point construction above reliably drives
    Sigma_post indefinite (repaired=True) regardless of alpha2 -- confirmed
    empirically: at alpha2=1e8 the OLD absolute-1e-6-floor code scored this
    exact task at ~166,931 nats/point; the fix (eig_floor scaled by
    Sigma_post's own diagonal magnitude) brings it down to ~7.8."""
    task = _near_duplicate_rbf_task(alpha2=1e8)
    post = gp_analytical_posterior(task)
    n = task["x_norm_test"].shape[0]

    assert post["min_eig"] < 0, "test construction should force an indefinite Sigma_post"
    assert post["repaired"], "eigenvalue floor should have fired"
    # Old absolute-floor behavior scored this same task at ~166,931 nats/point;
    # the true value (measured against the fix) is ~7.8 -- 50 is a generous
    # margin that still catches any regression back toward an absolute floor.
    assert post["nll_post"] / n < 50.0


def test_baseline_cache_round_trip(tiny_episode, tmp_path):
    """save_baseline_cache/load_baseline_cache should reproduce exactly what
    was written when the fingerprint matches, and miss cleanly when it
    doesn't — this is the mechanism eval_checkpoint.py relies on to skip
    re-fitting GP-MLE/DKL/per_ep_transformer across repeated runs."""
    cache_path = str(tmp_path / "baseline_cache.pt")

    fingerprint = baseline_fingerprint(
        OmegaConf.create({"data": dict(_TINY_DATA_CFG)}),
        live_generate=True, dataset_dir=None, seed=0, icl_rank=2, oracle_mode="prior",
        n_steps_mle=3, lr_mle=0.1, n_restarts_mle=1,
        n_steps_dkl=3, lr_dkl=0.1, n_steps_per_ep=3, patience_per_ep=2,
    )

    nlls, R_dict, y_space_nlls = eval_baselines_episode(
        ep=tiny_episode, icl_rank=2, n_steps_mle=3, lr_mle=0.1, n_steps_dkl=3, lr_dkl=0.1,
        n_steps_per_ep=3, patience_per_ep=2, device=torch.device("cpu"), oracle_mode="prior", n_restarts_mle=1,
    )
    key = episode_cache_key(live_generate=True, dataset_dir=None, seed=0, local_i=0, ep_i=0)
    save_baseline_cache(
        cache_path, fingerprint,
        {key: {"nlls": nlls, "R_dict": R_dict, "y_nlls": y_space_nlls}},
    )

    reloaded = load_baseline_cache(cache_path, fingerprint)
    assert key in reloaded
    assert reloaded[key]["nlls"] == nlls
    assert reloaded[key]["y_nlls"] == y_space_nlls
    for method, R in R_dict.items():
        assert torch.equal(reloaded[key]["R_dict"][method], R)

    # A different fingerprint (e.g. changed n_steps_mle) must miss entirely.
    other_fingerprint = baseline_fingerprint(
        OmegaConf.create({"data": dict(_TINY_DATA_CFG)}),
        live_generate=True, dataset_dir=None, seed=0, icl_rank=2, oracle_mode="prior",
        n_steps_mle=99, lr_mle=0.1, n_restarts_mle=1,
        n_steps_dkl=3, lr_dkl=0.1, n_steps_per_ep=3, patience_per_ep=2,
    )
    assert load_baseline_cache(cache_path, other_fingerprint) == {}
