"""Boundary contracts for dataset identity, atomic files and saved results."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from pytest import MonkeyPatch

from copula_inter.artifacts import atomic_json_save, atomic_torch_save
from copula_inter.backend_registry import BACKENDS, COPULA_BACKBONES, GENERIC_MARGINAL_BACKENDS, require_capability
from copula_inter.dataset import CopulaDataset, collate_fn
from copula_inter.dataset_manifest import dataset_identity, ensure_manifest, generation_spec, verified_shard_digest
from copula_inter.episode_contracts import assemble_episodes, validate_episode
from copula_inter.generate_pit_dataset import _refresh_meta, _save_shard_atomic
from copula_inter.marginal_backbones import TIER0_PATTERNS
from eval.results import (
    competition_ranks,
    load_results_cache,
    render_saved_totals,
    require_coverage,
    save_results_cache,
)
from eval.runners.eval_args import (
    AutoregressiveSpec,
    BaselineSpec,
    EvalSpec,
    MarginalSpec,
    SelectionSpec,
    compose_eval_spec,
)
from eval.runners.eval_checkpoint import _results_fingerprint
from eval.runners.eval_inputs import _dataset_dir_for_eval, _load_full_config
from eval.spatial.marginal_backends import _exaone_capture_quantile_bank


def _episode(p: int = 3, n: int = 2, d: int = 4) -> dict:
    return {
        "x_norm_train": torch.randn(p, d),
        "x_norm_test": torch.randn(n, d),
        "y_train": torch.randn(p),
        "y_test": torch.randn(n),
        "z_train": torch.randn(p),
        "z_test": torch.randn(n),
        "log_pdf_test": torch.randn(n),
        "R_star": torch.eye(n),
        "Sigma_star": torch.eye(n),
        "mu_star": torch.zeros(n),
        "sigma_star": torch.ones(n),
        "n_train": torch.tensor(p),
        "n_test": torch.tensor(n),
    }


def test_episode_contract_and_padding() -> None:
    episodes = [_episode(3, 2), _episode(5, 4)]
    for episode in episodes:
        validate_episode(episode)
    batch = collate_fn(episodes)
    assert batch["train_mask"].sum(dim=1).tolist() == [3, 5]
    assert batch["test_mask"].sum(dim=1).tolist() == [2, 4]
    assert torch.equal(batch["z_test"][0, 2:], torch.zeros(2))

    broken = dict(episodes[0], log_pdf_test=torch.zeros(3))
    with pytest.raises(ValueError, match="log_pdf_test"):
        collate_fn([broken])
    broken = dict(episodes[0], x_kernel_train=torch.zeros(3, 7))
    with pytest.raises(ValueError, match="together"):
        validate_episode(broken)


def test_manifest_rejects_changed_settings_and_checkpoint_bytes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "marginal.pt"
    checkpoint.write_bytes(b"weights A")
    cfg = OmegaConf.create(
        {
            "seed": 7,
            "data": {
                "n_tasks": 2,
                "shard_size": 2,
                "z_train_source": "tabicl",
                "dataset_dir": str(tmp_path),
                "pit_dir": str(tmp_path / "pit"),
                "resume": False,
            },
            "tabicl": {"ckpt": str(checkpoint)},
        }
    )
    directory = tmp_path / "pit"
    spec = generation_spec(cfg, str(checkpoint))
    first = ensure_manifest(directory, spec)
    assert ensure_manifest(directory, spec) == first
    cfg.data.resume = True
    assert ensure_manifest(directory, generation_spec(cfg, str(checkpoint))) == first
    cfg.seed = 8
    with pytest.raises(ValueError, match="identity mismatch"):
        ensure_manifest(directory, generation_spec(cfg, str(checkpoint)))
    cfg.seed = 7
    checkpoint.write_bytes(b"weights B")
    with pytest.raises(ValueError, match="identity mismatch"):
        ensure_manifest(directory, generation_spec(cfg, str(checkpoint)))


def test_dataset_identity_changes_for_same_path_replacement(tmp_path: Path) -> None:
    shard = tmp_path / "shard_000000.pt"
    _save_shard_atomic([_episode()], str(shard))
    first = dataset_identity(tmp_path)
    assert dataset_identity(tmp_path) == first
    sidecar = json.loads((tmp_path / "shard_000000.count.json").read_text())
    assert sidecar["sha256"]
    # The old sidecar stays in place: ctime validation forces a fresh hash.
    atomic_torch_save([_episode()], shard)
    assert dataset_identity(tmp_path) != first
    with pytest.raises(ValueError, match="content differs"):
        verified_shard_digest(shard, require_match=True)


def test_assembly_keeps_metadata_aligned_after_discard() -> None:
    tensors = {"z_test": torch.tensor([[1.0], [2.0], [3.0]])}
    components = [{"lengthscale": torch.tensor([10.0, 20.0, 30.0])}]
    result = assemble_episodes(
        tensors,
        {"kernel": "rbf+matern32"},
        torch.tensor([False, True, False]),
        (["rbf", "matern32"], ["+"], components),
    )
    assert [float(ep["z_test"][0]) for ep in result] == [1.0, 3.0]
    assert [float(ep["kernel_component_params"][0]["lengthscale"]) for ep in result] == [10.0, 30.0]


def test_manifest_dataset_rejects_stale_meta_count(tmp_path: Path) -> None:
    _save_shard_atomic([_episode(), _episode()], str(tmp_path / "shard_000000.pt"))
    (tmp_path / "manifest.json").write_text(json.dumps({"digest": "dataset A"}))
    torch.save({"n_total": 1, "shard_size": 2, "manifest_digest": "dataset A"}, tmp_path / "meta.pt")
    with pytest.raises(ValueError, match="counts disagree"):
        CopulaDataset(episode_dir=str(tmp_path))


def test_meta_exposes_only_completed_contiguous_shards(tmp_path: Path) -> None:
    manifest = ensure_manifest(tmp_path, {"seed": 7})
    _save_shard_atomic([_episode(), _episode()], str(tmp_path / "shard_000000.pt"))
    torch.save([_episode(), _episode()], tmp_path / "shard_000001.pt")  # crash before sidecar
    _refresh_meta(str(tmp_path), 4, 2, 2, manifest["digest"])
    assert len(CopulaDataset(episode_dir=str(tmp_path))) == 2
    _save_shard_atomic([_episode(), _episode()], str(tmp_path / "shard_000001.pt"))
    _refresh_meta(str(tmp_path), 4, 2, 2, manifest["digest"])
    assert len(CopulaDataset(episode_dir=str(tmp_path))) == 4


def test_atomic_save_keeps_old_file_on_failure(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"step": 1}, path)
    real_save = torch.save

    def fail_after_partial_write(value: Any, destination: Any) -> None:
        with open(destination, "wb") as output:
            output.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="disk full"):
        atomic_torch_save({"step": 2}, path)
    monkeypatch.setattr(torch, "save", real_save)
    assert torch.load(path, weights_only=True) == {"step": 1}
    assert sorted(item.name for item in tmp_path.iterdir()) == ["checkpoint.pt"]


@pytest.mark.parametrize("umask", [0o022, 0o002, 0o077])
def test_published_files_get_umask_permissions_like_open(tmp_path: Path, umask: int) -> None:
    """mkstemp creates 0600; published artifacts must be readable like a plain open() on shared storage."""
    old = os.umask(umask)
    try:
        with open(tmp_path / "plain.txt", "w") as out:
            out.write("x")
        atomic_torch_save({"step": 1}, tmp_path / "checkpoint.pt")
        atomic_json_save({"a": 1}, tmp_path / "results.json")
        ensure_manifest(tmp_path / "ds", {"k": 1})
    finally:
        os.umask(old)
    want = (tmp_path / "plain.txt").stat().st_mode & 0o777
    assert want == 0o666 & ~umask
    for name in ("checkpoint.pt", "results.json", "ds/manifest.json"):
        assert (tmp_path / name).stat().st_mode & 0o777 == want, name


def test_scored_fingerprint_tracks_checkpoint_and_resolved_marginal(tmp_path: Path) -> None:
    checkpoint = tmp_path / "copula.pt"
    marginal = tmp_path / "marginal.pt"
    checkpoint.write_bytes(b"model A")
    marginal.write_bytes(b"marginal A")
    args = EvalSpec(ckpt=str(checkpoint), marginal=MarginalSpec(tabicl_ckpt=None, z_train_source="tabicl"))
    args.selection = SelectionSpec(n_folds=2, min_fold_size=2)
    args.seed = 1
    args.baselines = BaselineSpec(
        zeromean_gp=False, n_steps_zeromean_gp=1, lr_zeromean_gp=0.01, n_restarts_zeromean_gp=1
    )
    args.autoregressive = AutoregressiveSpec(enabled=False)
    first = _results_fingerprint({}, args, 5, resolved_marginal=str(marginal))
    copied = tmp_path / "same-model.pt"
    copied.write_bytes(checkpoint.read_bytes())
    args.ckpt = str(copied)
    assert _results_fingerprint({}, args, 5, resolved_marginal=str(marginal)) == first
    args.ckpt = str(checkpoint)
    checkpoint.write_bytes(b"model B")
    assert _results_fingerprint({}, args, 5, resolved_marginal=str(marginal)) != first
    checkpoint.write_bytes(b"model A")
    marginal.write_bytes(b"marginal B")
    assert _results_fingerprint({}, args, 5, resolved_marginal=str(marginal)) != first


def test_result_ties_and_coverage_are_explicit() -> None:
    ranks = competition_ranks([{"a": 1.0, "b": 1.0, "c": 3.0, "d": float("nan")}], ["a", "b", "c", "d"])
    assert ranks == {"a": [1], "b": [1], "c": [3], "d": []}
    interleaved = competition_ranks([{"a": 2.0, "b": float("nan"), "c": 1.0}], ["a", "b", "c"])
    assert interleaved == {"a": [2], "b": [], "c": [1]}
    with pytest.raises(RuntimeError, match="coverage 1/2"):
        require_coverage(1, 2, 0.75)
    table = render_saved_totals(
        {
            "episodes": [
                {"total_nlls": {"icl": {"total": 1.0}}},
                {"total_nlls": {"icl": {"total": float("nan")}}},
            ]
        }
    )
    assert "1/2" in table


def test_results_cache_reuses_only_matching_artifact_identity(tmp_path: Path) -> None:
    path = str(tmp_path / "scores.json")
    save_results_cache(path, {"checkpoint": "A"}, {"7": {"icl": 0.5}})
    assert load_results_cache(path, {"checkpoint": "A"}) == {"7": {"icl": 0.5}}
    assert load_results_cache(path, {"checkpoint": "B"}) == {}


def test_eval_spec_parses_without_loading_models(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    spec = compose_eval_spec([f"ckpt={checkpoint}", "autoregressive.enabled=false", "n_episodes=2"])
    assert spec.ckpt == str(checkpoint)
    assert spec.n_episodes == 2
    assert not spec.autoregressive.enabled


def test_default_eval_config_resolves_outside_checkout(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert _load_full_config("conf/config.yaml").data.n_tasks > 0


def test_eval_uses_configured_dataset_directory_in_cache_key() -> None:
    args = EvalSpec(ckpt="unused.pt")
    cfg = OmegaConf.create({"training": {"dataset_dir": "./generated/pit"}})
    assert _dataset_dir_for_eval(args, cfg, False, False) == "./generated/pit"
    assert _dataset_dir_for_eval(args, cfg, True, False) is None
    assert _dataset_dir_for_eval(args, cfg, False, True) is None


def test_backend_registry_capabilities() -> None:
    assert set(COPULA_BACKBONES) == {"tabicl", "tabldm"}
    assert set(GENERIC_MARGINAL_BACKENDS) == {"exaone", "tabpfn", "tabldm"}
    assert all(name == backend.name for name, backend in BACKENDS.items())
    assert set(BACKENDS) == set(TIER0_PATTERNS)
    with pytest.raises(ValueError, match="does not support"):
        require_capability("exaone", "copula_backbone")


def test_training_core_import_does_not_load_reporting_or_era5() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, copula_inter.training_core; "
            "assert not any(name == 'wandb' or name.startswith(('matplotlib', 'eval.data.era5')) "
            "for name in sys.modules)",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )


def test_exaone_adapter_overrides_only_one_instance() -> None:
    class FakeRegressor:
        manifest = SimpleNamespace(runtime=SimpleNamespace(ensemble_count=1), output_width=3)

        def _collapse_members(self, output: Any, query_count: int) -> Any:
            return output.sum(dim=-1)

    first, second = FakeRegressor(), FakeRegressor()
    output = torch.tensor([[[3.0, 1.0, 2.0]]])
    original = second._collapse_members(output, 1)
    with _exaone_capture_quantile_bank(first):
        torch.testing.assert_close(first._collapse_members(output, 1), torch.tensor([[[1.0, 2.0, 3.0]]]))
        torch.testing.assert_close(second._collapse_members(output, 1), original)
    assert "_collapse_members" not in first.__dict__
    torch.testing.assert_close(first._collapse_members(output, 1), original)


def test_legacy_shards_are_identified_without_rehashing(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Pre-refactor shards have no digest sidecar; identity must not sha256 every shard on every eval."""
    import copula_inter.dataset_manifest as dm

    shard = tmp_path / "shard_000000.pt"
    torch.save([_episode()], shard)  # legacy: no .count.json sidecar

    def no_hashing(path: object) -> str:
        raise AssertionError(f"re-hashed legacy shard {path}")

    monkeypatch.setattr(dm, "file_digest", no_hashing)
    first = dataset_identity(tmp_path)
    assert dataset_identity(tmp_path) == first
    os.utime(shard, ns=(1, 1))  # a rewritten / replaced shard changes its stat
    assert dataset_identity(tmp_path) != first


def test_diverged_episodes_count_against_a_method_like_pre_refactor_main() -> None:
    """Only NaN (not scored) is excluded; +inf (diverged) stays in the mean and ranks last, as np.nanmean did."""
    from eval.results import numeric_summary

    mean, _std, count = numeric_summary([1.0, float("inf"), float("nan")])
    assert mean == float("inf") and count == 2
    assert numeric_summary([1.0, 3.0, float("nan")])[:1] == (2.0,)
    ranks = competition_ranks([{"a": float("inf"), "b": 5.0, "c": float("nan")}], ["a", "b", "c"])
    assert ranks == {"a": [2], "b": [1], "c": []}
