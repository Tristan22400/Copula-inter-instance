"""Tests for eval/baselines/classical.py and eval/runners/eval_checkpoint.py on tiny live GP episodes with a fake ICL model."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from pytest import MonkeyPatch

_TESTS = os.path.dirname(os.path.abspath(__file__))

from copula_inter.data_gen import generate_gp_batch  # noqa: E402
from copula_inter.pit import gp_analytical_posterior  # noqa: E402
from eval.baselines.classical import (  # noqa: E402
    baseline_fingerprint,
    episode_cache_key,
    eval_baselines_episode,
    load_baseline_cache,
    save_baseline_cache,
)
from eval.baselines.prefit import (  # noqa: E402
    _episode_to_pool_payload,
    _pool_decode_tensors,
    _PoolTensor,
    _prefit_baselines_parallel,
)
from eval.runners.episode_scoring import _eval_icl_episode  # noqa: E402
from eval.runners.eval_args import EvalSpec

_TINY_DATA_CFG = {
    "d_features": 1,
    "P_min": 5,
    "P_max": 8,
    "N_min": 4,
    "N_max": 6,
    "n_tasks": 4,
    "l_min": 0.5,
    "l_max": 1.5,
    "alpha2_min": 0.5,
    "alpha2_max": 1.5,
    "noise_min": 0.05,
    "noise_max": 0.2,
}


@pytest.fixture(scope="module")
def tiny_episode() -> dict[str, torch.Tensor]:
    cfg = OmegaConf.create({"seed": 0, "data": dict(_TINY_DATA_CFG)})
    torch.manual_seed(0)
    return generate_gp_batch(cfg, B=1, device="cpu", return_kernel_metadata=True)[0]


class _FakeICLModel(torch.nn.Module):
    """Stands in for CopulaTabICL: forward(batch) -> {"W", "s"}, ignoring the batch."""

    def __init__(self, n_test: int, rank: int) -> None:
        super().__init__()
        self.W = torch.randn(1, n_test, rank) * 0.3
        self.s = torch.randn(1, n_test)
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict) -> dict:
        return {"W": self.W, "s": self.s}


def _assert_valid_correlation(R: torch.Tensor, n: int, atol: float = 1e-3) -> None:
    assert R.shape == (n, n)
    assert torch.allclose(R, R.T, atol=atol)
    assert torch.allclose(R.diagonal(), torch.ones(n), atol=1e-2)


def _contains_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def test_pool_episode_payload_encodes_nested_metadata_tensors() -> None:
    """Pool payloads contain no tensors, including inside kernel_component_params."""
    episode: dict[str, Any] = {
        "x_norm_train": torch.tensor([[1.0]]),
        "kernel_component_params": [
            {"l": torch.tensor([0.5]), "nested": (torch.tensor([2.0]),)},
        ],
    }

    payload = _episode_to_pool_payload(episode)

    assert not _contains_tensor(payload)
    assert isinstance(payload["x_norm_train"], _PoolTensor)
    assert isinstance(payload["kernel_component_params"][0]["l"], _PoolTensor)

    restored = _pool_decode_tensors(payload)
    assert _contains_tensor(restored)
    assert torch.equal(restored["x_norm_train"], episode["x_norm_train"])
    assert torch.equal(
        restored["kernel_component_params"][0]["nested"][0],
        episode["kernel_component_params"][0]["nested"][0],
    )


def test_parallel_prefit_accepts_nested_tensor_metadata(tiny_episode: dict[str, torch.Tensor], tmp_path: Path) -> None:
    """The spawned pool transports episodes with nested tensor metadata."""
    episode: dict[str, Any] = dict(tiny_episode)
    episode["kernel_component_params"] = [{"l": torch.tensor([0.5])}]
    fitted: dict[str, Any] = {}
    _prefit_baselines_parallel(
        pending=[("nested-metadata", 7, episode)],
        fit_kwargs={
            "icl_rank": 2,
            "n_steps_mle": 1,
            "lr_mle": 0.1,
            "n_steps_dkl": 1,
            "lr_dkl": 0.1,
            "n_steps_per_ep": 1,
            "patience_per_ep": 1,
            "oracle_mode": "prior",
            "n_restarts_mle": 1,
            "n_restarts_dkl": 1,
        },
        n_workers=1,
        cache_path=None,
        fingerprint={},
        fitted=fitted,
    )

    assert "nested-metadata" in fitted
    assert all(isinstance(R, torch.Tensor) for R in fitted["nested-metadata"]["R_dict"].values())


def test_eval_baselines_episode_runs_and_returns_valid_correlations(tiny_episode: dict[str, torch.Tensor]) -> None:
    """Every baseline fits (or falls back) on a tiny episode with finite NLLs and valid correlation matrices."""
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
        "independence",
        "gp_prior_rbf",
        "gp_mle_rbf",
        "gp_mle_ard_rbf",
        "gp_mle_matern32",
        "gp_mle_ard_matern32",
        "gp_mle_periodic",
        "gp_mle_ard_periodic",
        "gp_mle_rq",
        "gp_mle_ard_rq",
        "gp_mle_dot_product",
        "gp_mle_polynomial",
        "dkl_rbf",
        "dkl_matern32",
        "dkl_rq",
        "dkl_dot_product",
        "per_ep_transformer",
    }
    # y_space_nlls excludes the unfitted references.
    expected_y_keys = expected_keys - {"independence", "gp_prior_rbf"}
    assert expected_keys <= nlls.keys()
    assert expected_keys <= R_dict.keys()
    assert expected_y_keys <= y_space_nlls.keys()

    assert abs(nlls["independence"]) < 1e-3
    _assert_valid_correlation(R_dict["independence"], n_test)

    # Every method gives a finite NLL and a valid R.
    for method in expected_keys:
        assert torch.isfinite(torch.tensor(nlls[method])), f"{method} produced a non-finite NLL"
        _assert_valid_correlation(R_dict[method], n_test)
    for method in expected_y_keys:
        parts = y_space_nlls[method]
        assert set(parts.keys()) == {"total", "marginal", "copula"}
        for part_name, val in parts.items():
            assert torch.isfinite(torch.tensor(val)), f"{method}'s {part_name} Y-space NLL is non-finite"
        # total = marginal + copula exactly.
        assert parts["total"] == pytest.approx(parts["marginal"] + parts["copula"], abs=1e-3)


def test_eval_icl_episode_scores_against_oracle(tiny_episode: dict[str, torch.Tensor]) -> None:
    n_test = tiny_episode["x_norm_test"].shape[0]
    fake_model = _FakeICLModel(n_test=n_test, rank=2)

    nlls, R_dict, R_oracle, y_space_nlls, icl_y_parts = _eval_icl_episode(
        ep=tiny_episode,
        icl_model=fake_model,
        device=torch.device("cpu"),
    )

    assert set(nlls.keys()) == {"icl", "oracle"}
    assert torch.isfinite(torch.tensor(nlls["icl"]))
    assert torch.isfinite(torch.tensor(nlls["oracle"]))
    _assert_valid_correlation(R_dict["icl"], n_test)
    assert R_oracle is not None
    assert torch.equal(R_oracle, tiny_episode["R_star"])
    # Without a marginal PIT there is no ICL Y-space NLL.
    assert set(icl_y_parts.keys()) == {"total", "marginal", "copula"}
    for val in icl_y_parts.values():
        assert torch.isnan(torch.tensor(val))
    # Oracle prior/posterior splits are available and consistent.
    for key in ("prior", "posterior"):
        parts = y_space_nlls[key]
        assert set(parts.keys()) == {"total", "marginal", "copula"}
        for val in parts.values():
            assert torch.isfinite(torch.tensor(val))
        assert parts["total"] == pytest.approx(parts["marginal"] + parts["copula"], abs=1e-3)


def test_eval_icl_episode_with_tabicl_pit_populates_total_nll(tiny_episode: dict[str, torch.Tensor]) -> None:
    """With a marginal PIT, icl_y_parts is finite and total = marginal + copula."""
    n_train = tiny_episode["x_norm_train"].shape[0]
    n_test = tiny_episode["x_norm_test"].shape[0]
    fake_model = _FakeICLModel(n_test=n_test, rank=2)

    z_test = torch.randn(n_test)
    tabicl_pit = {
        "z_train": torch.randn(n_train),
        "z_test": z_test,
        # Standard-normal log-density as a stand-in marginal.
        "log_pdf_test": -0.5 * (z_test**2 + math.log(2 * math.pi)),
    }

    _, _, _, _, icl_y_parts = _eval_icl_episode(
        ep=tiny_episode,
        icl_model=fake_model,
        device=torch.device("cpu"),
        marginal_pit=tabicl_pit,
    )

    assert set(icl_y_parts.keys()) == {"total", "marginal", "copula"}
    for val in icl_y_parts.values():
        assert torch.isfinite(torch.tensor(val))
    assert icl_y_parts["total"] == pytest.approx(
        icl_y_parts["marginal"] + icl_y_parts["copula"],
        abs=1e-3,
    )


def test_gp_oracle_posterior_total_nll_bayes_optimal(tiny_episode: dict[str, torch.Tensor]) -> None:
    """The per-point oracle rows keep posterior <= prior."""
    post = gp_analytical_posterior(tiny_episode)
    assert post["nll_post"] <= post["nll_prior"] + 1e-6


def _near_duplicate_rbf_task(alpha2: float) -> dict:
    """Deterministic RBF task with two near-duplicate test points with different y, so Sigma_post needs the eigenvalue repair for any alpha2."""
    zero = torch.zeros(1)
    x_train = torch.tensor([[-1.0], [0.0], [1.0]])
    x_test = torch.tensor([[0.50000], [0.50001], [-0.7]])  # first two are near-duplicates
    return {
        "kernel": "rbf",
        "l": torch.tensor([0.3]),
        "alpha2": torch.tensor([alpha2]),
        "nugget": torch.tensor([1e-4]),
        "period": zero,
        "rq_alpha": zero,
        "power": zero,
        "l_b": zero,
        "alpha2_b": zero,
        "period_b": zero,
        "rq_alpha_b": zero,
        "power_b": zero,
        "kernel_feature_indices": torch.tensor([0]),
        "x_norm_train": x_train,
        "x_norm_test": x_test,
        "y_train": torch.tensor([0.5, -0.3, 0.8]),
        "y_test": torch.tensor([1.0, -1.0, 0.5]),  # near-duplicates disagree by 2.0
        "mu_star": torch.zeros(3),
    }


def test_gp_analytical_posterior_eig_floor_scale_invariant() -> None:
    """The eigenvalue floor scales with Sigma_post, so a large-alpha2 repaired episode gets a sensible NLL."""
    task = _near_duplicate_rbf_task(alpha2=1e8)
    post = gp_analytical_posterior(task)
    n = task["x_norm_test"].shape[0]

    assert post["min_eig"] < 0, "test construction should force an indefinite Sigma_post"
    assert post["repaired"], "eigenvalue floor should have fired"
    # The scale-invariant floor gives ~7.8 nats/point here; 50 catches a regression.
    assert post["nll_post"] / n < 50.0


def test_gp_analytical_posterior_eig_floor_nugget_bound() -> None:
    """The eigenvalue floor is never below the nugget (small-scale Sigma_post, alpha2=1e-2)."""
    task = _near_duplicate_rbf_task(alpha2=1e-2)
    post = gp_analytical_posterior(task)

    assert post["min_eig"] < 1e-4, (
        "test construction should produce a measured eigenvalue below the "
        "mathematically-guaranteed nugget floor (a numerical artifact)"
    )
    assert post["repaired"], "nugget floor should have fired even though Sigma_post's own scale is tiny"

    Sigma_post = post["Sigma_post"].double()
    repaired_min_eig = torch.linalg.eigvalsh(Sigma_post).min().item()
    assert repaired_min_eig >= 1e-4 - 1e-9, "post-repair eigenvalues must respect the nugget lower bound"


def test_baseline_cache_round_trip(tiny_episode: dict[str, torch.Tensor], tmp_path: Path) -> None:
    """save_baseline_cache / load_baseline_cache round-trip on a matching fingerprint and miss otherwise."""
    cache_path = str(tmp_path / "baseline_cache.pt")

    fingerprint = baseline_fingerprint(
        OmegaConf.create({"data": dict(_TINY_DATA_CFG)}),
        live_generate=True,
        dataset_dir=None,
        seed=0,
        icl_rank=2,
        oracle_mode="prior",
        n_steps_mle=3,
        lr_mle=0.1,
        n_restarts_mle=1,
        n_steps_dkl=3,
        lr_dkl=0.1,
        n_steps_per_ep=3,
        patience_per_ep=2,
    )

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
    key = episode_cache_key(live_generate=True, dataset_dir=None, seed=0, ep_i=0)
    save_baseline_cache(
        cache_path,
        fingerprint,
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
        live_generate=True,
        dataset_dir=None,
        seed=0,
        icl_rank=2,
        oracle_mode="prior",
        n_steps_mle=99,
        lr_mle=0.1,
        n_restarts_mle=1,
        n_steps_dkl=3,
        lr_dkl=0.1,
        n_steps_per_ep=3,
        patience_per_ep=2,
    )
    assert load_baseline_cache(cache_path, other_fingerprint) == {}


def test_failed_baseline_fit_still_yields_nan_parts_dict(
    tiny_episode: dict[str, torch.Tensor], monkeypatch: MonkeyPatch
) -> None:
    """A baseline whose fit raises records a {total, marginal, copula} NaN dict, not a bare float."""
    import eval.baselines.classical as classical

    real_fit = classical.fit_and_eval_gpytorch

    def fail_dkl_only(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        # Only DKL fits fail; GP-MLE still runs.
        if kwargs.get("feature_extractor_factory") is not None:
            raise RuntimeError("synthetic DKL failure")
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(classical, "fit_and_eval_gpytorch", fail_dkl_only)

    _, _, y_space_nlls = eval_baselines_episode(
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

    dkl_labels = [k for k in y_space_nlls if k.startswith("dkl_")]
    assert dkl_labels, "expected DKL baselines to be present in y_nlls"
    for label, parts in y_space_nlls.items():
        assert isinstance(parts, dict), f"{label} stored {type(parts).__name__}, not a dict"
        assert {"total", "marginal", "copula"} <= parts.keys(), label
    for label in dkl_labels:
        assert all(math.isnan(v) for v in y_space_nlls[label].values()), label


# The exact key set 42c0402's baseline_fingerprint produced. Live / ERA5 runs must keep it so the
# baseline caches written before the refactor (hours of fitting) stay valid.
_PRE_REFACTOR_FINGERPRINT_KEYS = {
    "algo_version",
    "data_cfg",
    "icl_rank",
    "live_generate",
    "dataset_dir",
    "seed",
    "oracle_mode",
    "n_steps_mle",
    "lr_mle",
    "n_restarts_mle",
    "n_steps_dkl",
    "lr_dkl",
    "n_restarts_dkl",
    "n_steps_per_ep",
    "patience_per_ep",
    "gp_val_select",
}


def _fingerprint(live_generate: bool, dataset_dir: str | None) -> dict:
    return baseline_fingerprint(
        OmegaConf.create({"data": dict(_TINY_DATA_CFG)}),
        live_generate=live_generate,
        dataset_dir=dataset_dir,
        seed=0,
        icl_rank=2,
        oracle_mode="prior",
        n_steps_mle=3,
        lr_mle=0.1,
        n_restarts_mle=1,
        n_steps_dkl=3,
        lr_dkl=0.1,
        n_steps_per_ep=3,
        patience_per_ep=2,
    )


def test_live_baseline_fingerprint_matches_pre_refactor_caches() -> None:
    fp = _fingerprint(live_generate=True, dataset_dir=None)
    assert set(fp) == _PRE_REFACTOR_FINGERPRINT_KEYS
    assert fp["algo_version"] == 4


def test_on_disk_baseline_fingerprint_tracks_dataset_content(tmp_path: Path) -> None:
    fp = _fingerprint(live_generate=False, dataset_dir=str(tmp_path))
    assert set(fp) == _PRE_REFACTOR_FINGERPRINT_KEYS | {"dataset_identity"}


def _eval_spec(tmp_path: Path, *overrides: str) -> EvalSpec:
    from eval.runners.eval_args import compose_eval_spec

    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"not loaded")
    return compose_eval_spec([f"ckpt={ckpt}", *overrides])


def test_ckpt_directory_is_pinned_to_its_latest_step(tmp_path: Path) -> None:
    """A run-directory ckpt must key caches on the step file it loads, not on the directory name."""
    from eval.runners.eval_args import compose_eval_spec

    run = tmp_path / "run"
    run.mkdir()
    (run / "step_0000010.pt").write_bytes(b"a")
    assert compose_eval_spec([f"ckpt={run}"]).ckpt == str(run / "step_0000010.pt")
    (run / "step_0000020.pt").write_bytes(b"b")
    assert compose_eval_spec([f"ckpt={run}"]).ckpt == str(run / "step_0000020.pt")


def _cache_fp(
    spec: EvalSpec,
    *,
    live_generate: bool = True,
    era5_geometry: dict | None = None,
    tabicl_ckpt: str | None = None,
) -> dict:
    from eval.runners.eval_checkpoint import _baseline_cache_fingerprint

    return _baseline_cache_fingerprint(
        spec,
        OmegaConf.create({"data": dict(_TINY_DATA_CFG)}),
        live_generate=live_generate,
        dataset_dir=None,
        icl_rank=2,
        oracle_mode="prior",
        era5_geometry=era5_geometry,
        tabicl_pit_k_folds=5,
        tabicl_ckpt=tabicl_ckpt,
    )


def test_alternate_noncomposite_is_part_of_the_live_cache_identity(tmp_path: Path) -> None:
    default = _cache_fp(_eval_spec(tmp_path))
    assert set(default) == _PRE_REFACTOR_FINGERPRINT_KEYS  # old caches still match by default
    assert _cache_fp(_eval_spec(tmp_path, "alternate_noncomposite=false")) != default


_ERA5_GEOMETRY = {
    "grid_size": 24,
    "n_context": 30,
    "box_deg_range": (5.0, 25.0),
    "vary_geometry": False,
    "grid_size_range": (8, 28),
    "n_context_frac_range": (0.05, 0.4),
    "max_months": None,
    "standardize_y": True,
}


def test_era5_cache_identity_records_the_pit_marginal(tmp_path: Path) -> None:
    from eval.configs.checkpoints import DEFAULT_MARGINAL_FAMILY, resolve_marginal_checkpoint

    default_ckpt = resolve_marginal_checkpoint(DEFAULT_MARGINAL_FAMILY)
    other = tmp_path / "other_marginal.pt"
    other.write_bytes(b"other")

    def era5_fp(ckpt: str, amp: bool) -> dict:
        spec = _eval_spec(tmp_path, "era5.enabled=true", f"marginal.tabicl_amp={str(amp).lower()}")
        return _cache_fp(spec, live_generate=False, era5_geometry=_ERA5_GEOMETRY, tabicl_ckpt=ckpt)["era5"]

    # The implicit pre-refactor setup (default marginal, AMP on) keeps the old key set.
    assert "pit_marginal_identity" not in era5_fp(default_ckpt, amp=True)
    assert era5_fp(default_ckpt, amp=False)["pit_marginal_identity"]["amp"] is False
    assert era5_fp(str(other), amp=True) != era5_fp(default_ckpt, amp=True)
