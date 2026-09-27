"""The eval runners' Hydra configs compose, reject argparse flags, and every documented command still composes."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra

from copula_inter.config_path import config_dir
from copula_inter.train_setup import prepare_training_inputs
from eval.runners.autoregressive_baseline_eval import AutoregressiveBaselineSpec
from eval.runners.era5_calibration_eval import Era5CalibrationSpec
from eval.runners.eval_args import EvalSpec
from eval.runners.hydra_cli import compose_spec, reject_argparse_flags
from eval.runners.marginal_calibration_eval import MarginalCalibrationSpec
from eval.runners.run_benchmarks import BenchmarkSpec
from eval.runners.spatial_correlation_eval import (
    AllSpec,
    DiagnoseSpec,
    SpatialSpec,
    _checkpoint_tokens,
    validate_spatial_spec,
)
from eval.runners.summarize_results_cache import SummarizeSpec

REPO = Path(__file__).resolve().parents[1]

# Runner module name -> its schema. Required keys get a placeholder so the defaults compose.
RUNNERS: dict[str, tuple[type, list[str]]] = {
    "eval_checkpoint": (EvalSpec, ["ckpt=x.pt"]),
    "spatial_correlation_eval": (SpatialSpec, []),
    "run_benchmarks": (BenchmarkSpec, []),
    "marginal_calibration_eval": (MarginalCalibrationSpec, []),
    "era5_calibration_eval": (Era5CalibrationSpec, []),
    "autoregressive_baseline_eval": (AutoregressiveBaselineSpec, []),
    "summarize_results_cache": (SummarizeSpec, ["caches=[a.json]"]),
}


@pytest.mark.parametrize("name", sorted(RUNNERS))
def test_runner_defaults_compose_into_their_schema(name: str) -> None:
    schema, required = RUNNERS[name]
    assert isinstance(compose_spec(name, schema, required), schema)


def test_eval_spec_overrides_reach_nested_groups() -> None:
    spec = compose_spec(
        "eval_checkpoint",
        EvalSpec,
        ["ckpt=x.pt", "era5.enabled=true", "era5.grid_size=16", "autoregressive.order=natural", "baselines.cache=null"],
    )
    assert spec.era5.enabled and spec.era5.grid_size == 16
    assert spec.autoregressive.order == "natural"
    assert spec.baselines.cache is None


def test_spatial_command_group_selects_the_command_schema() -> None:
    assert isinstance(compose_spec("spatial_correlation_eval", SpatialSpec).command, AllSpec)
    spec = compose_spec(
        "spatial_correlation_eval", SpatialSpec, ["command=diagnose", "command.ckpt=[a,b:3]", "command.grid_size=8"]
    )
    assert isinstance(spec.command, DiagnoseSpec)
    assert _checkpoint_tokens(spec.command.ckpt) == ["a", "b:3"]
    assert _checkpoint_tokens("a, b") == ["a", "b"]
    spec.command.region = "atlantis"
    with pytest.raises(ValueError, match="command.region"):
        validate_spatial_spec(spec)


def test_argparse_flags_exit_with_the_replacement_key(capsys: pytest.CaptureFixture[str]) -> None:
    aliases = {"ar_order": "autoregressive.order", "no_baseline_cache": "baselines.cache=null"}
    argv = ["--era5_grid_size", "12", "--ar_order", "natural", "--no-autoregressive", "--no_baseline_cache", "--cfg"]
    with pytest.raises(SystemExit):
        reject_argparse_flags("eval_checkpoint", EvalSpec, argv, aliases)
    err = capsys.readouterr().err
    assert "--era5_grid_size: use era5.grid_size=<value>" in err
    assert "--ar_order: use autoregressive.order=<value>" in err
    assert "--no-autoregressive: use autoregressive.enabled=false" in err
    assert "--no_baseline_cache: use baselines.cache=null" in err
    assert "--cfg" not in err.split("not --flags.")[1].split("Every key")[0]
    reject_argparse_flags("eval_checkpoint", EvalSpec, ["ckpt=x", "--cfg", "job"], aliases)


def _compose_train(overrides: list[str]) -> Any:
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=config_dir(__file__), version_base=None):
        return hydra.compose(config_name="config", overrides=overrides)


def test_finetune_era5_preset_requires_a_checkpoint() -> None:
    cfg = _compose_train(["experiment=finetune_era5"])
    assert cfg.training.live_source == "era5" and cfg.training.live_generation
    assert cfg.training.resume_required and cfg.training.resume_ckpt is None
    assert cfg.training.aux_mae_weight == 0.0
    assert not _compose_train([]).training.resume_required
    with pytest.raises(ValueError, match="training.resume_ckpt is required"):
        prepare_training_inputs(cfg)


def test_finetune_inputs_are_checked_before_startup(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cfg = _compose_train(
        [
            "experiment=finetune_era5",
            f"training.resume_ckpt={checkpoint}",
            f"era5_live.corpus_dir={corpus}",
        ]
    )
    with pytest.raises(FileNotFoundError, match="era5_live.corpus_dir is empty"):
        prepare_training_inputs(cfg)
    (corpus / "2022-01.nc").touch()
    assert prepare_training_inputs(cfg) == str(checkpoint)
    assert cfg.training.resume_ckpt == str(checkpoint)
    assert "era5-finetune-source-" in cfg.training.ckpt_dir


def _documented_commands() -> list[str]:
    """Every `python -m <module> ...` line in the docs and job scripts, continuation lines joined."""
    commands = []
    for path in [REPO / "CLAUDE.md", REPO / "README.md", REPO / "AGENTS.md", *sorted((REPO / "scripts").glob("*.sh"))]:
        text = re.sub(r"\\\n\s*", " ", path.read_text())
        for line in text.splitlines():
            line = line.strip().lstrip("#").strip()
            line = line.split("  #")[0].replace("python -u -m", "python -m")
            if re.match(r"python -m (eval\.runners|copula_inter\.train)\b", line):
                commands.append(line)
    return commands


def _overrides(command: str) -> tuple[str, list[str]]:
    command = re.sub(r"<[^>]*>", "x", command)
    tokens = shlex.split(command)
    module = tokens[2]
    rest = tokens[3:]
    assert not [t for t in rest if t.startswith("--") and t not in ("--cfg", "--resolve")], command
    return module, [t for t in rest if "=" in t and not t.startswith("-") and "$" not in t]


def test_documented_commands_compose() -> None:
    commands = _documented_commands()
    assert any("eval.runners.eval_checkpoint" in c for c in commands)
    for command in commands:
        module, overrides = _overrides(command)
        if module == "copula_inter.train":
            _compose_train(overrides)
            continue
        name = module.rsplit(".", 1)[1]
        schema, required = RUNNERS[name]
        keys = {o.split("=", 1)[0] for o in overrides}
        compose_spec(name, schema, [r for r in required if r.split("=", 1)[0] not in keys] + overrides)


def test_no_doc_invokes_a_runner_with_argparse_flags() -> None:
    pattern = re.compile(r"eval/runners/\w+\.py\s+--|copula_inter\.finetune_era5")
    for path in [REPO / "CLAUDE.md", REPO / "README.md", *sorted((REPO / "scripts").glob("*.sh"))]:
        assert not pattern.search(path.read_text()), path


def test_run_benchmarks_tolerates_failed_episodes_by_default() -> None:
    """Pre-refactor run_benchmarks logged a failed episode and continued; one failure must not fail the run."""
    from eval.results import require_coverage

    spec = BenchmarkSpec()
    require_coverage(99, 100, 1 - spec.max_failed_fraction)  # must not raise
