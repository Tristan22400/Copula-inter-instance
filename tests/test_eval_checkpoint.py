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

    nlls, R_dict = eval_baselines_episode(
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
        "gp_mle_dot_product",
        "dkl_rbf", "dkl_matern32", "dkl_rq", "dkl_dot_product",
        "per_ep_transformer",
    }
    assert expected_keys <= nlls.keys()
    assert expected_keys <= R_dict.keys()

    assert abs(nlls["independence"]) < 1e-3
    _assert_valid_correlation(R_dict["independence"], n_test)

    # Every method must produce a finite NLL and a well-formed R, even the
    # ones expected to fit poorly at 3 Adam steps on a tiny episode — a NaN
    # or malformed matrix here means the fit-or-fallback path silently broke.
    for method in expected_keys:
        assert torch.isfinite(torch.tensor(nlls[method])), f"{method} produced a non-finite NLL"
        _assert_valid_correlation(R_dict[method], n_test)


def test_eval_icl_episode_scores_against_oracle(tiny_episode):
    n_test = tiny_episode["x_norm_test"].shape[0]
    fake_model = _FakeICLModel(n_test=n_test, rank=2)

    nlls, R_dict, R_oracle = _eval_icl_episode(ep=tiny_episode, icl_model=fake_model, device=torch.device("cpu"))

    assert set(nlls.keys()) == {"icl", "oracle"}
    assert torch.isfinite(torch.tensor(nlls["icl"]))
    assert torch.isfinite(torch.tensor(nlls["oracle"]))
    _assert_valid_correlation(R_dict["icl"], n_test)
    assert torch.equal(R_oracle, tiny_episode["R_star"])


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

    nlls, R_dict = eval_baselines_episode(
        ep=tiny_episode, icl_rank=2, n_steps_mle=3, lr_mle=0.1, n_steps_dkl=3, lr_dkl=0.1,
        n_steps_per_ep=3, patience_per_ep=2, device=torch.device("cpu"), oracle_mode="prior", n_restarts_mle=1,
    )
    key = episode_cache_key(live_generate=True, dataset_dir=None, seed=0, local_i=0, ep_i=0)
    save_baseline_cache(cache_path, fingerprint, {key: {"nlls": nlls, "R_dict": R_dict}})

    reloaded = load_baseline_cache(cache_path, fingerprint)
    assert key in reloaded
    assert reloaded[key]["nlls"] == nlls
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
