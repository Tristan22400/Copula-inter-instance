"""The log sigma_y Jacobian shift reaches each TOTAL-table row exactly once (eval/runners/episode_scoring.py)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from pytest import MonkeyPatch
from torch import Tensor

import eval.runners.episode_scoring as scoring

_N = 4
_LOG_STD = 2.5  # log sigma_y, per-point nats
_STD_ROW = {"total": 1.0, "marginal": 0.6, "copula": 0.4}  # fitted on standardized y
_RAW_ROW = {"total": 3.0, "marginal": 2.2, "copula": 0.8}  # scored on marginal_pit: raw nats


def _stub_scoring(monkeypatch: MonkeyPatch) -> None:
    """Replace every fitter in _score_episode by a constant, keeping the shift/merge logic real."""
    pit = {"z_train": torch.zeros(3), "z_test": torch.zeros(_N), "log_pdf_test": torch.zeros(_N)}
    eye = torch.eye(_N)
    zero_parts = {"total": 0.0, "marginal": 0.0, "copula": 0.0}

    def zero_mean_gp(**_: Any) -> tuple[dict[str, float], dict[str, Tensor], dict[str, dict[str, float]]]:
        return {"gp_zeromean_rbf": 0.0}, {"gp_zeromean_rbf": eye}, {"gp_zeromean_rbf": dict(_RAW_ROW)}

    def icl(**_: Any) -> tuple[Any, ...]:
        icl_parts = {"total": 0.5, "marginal": 0.3, "copula": 0.2}
        return {"icl": 0.0}, {"icl": eye}, None, {"prior": zero_parts, "posterior": zero_parts}, icl_parts

    monkeypatch.setattr(scoring, "_episode_marginal_pit", lambda ctx, ep, ep_i: pit)
    monkeypatch.setattr(scoring, "_add_autoregressive_log_pdf", lambda ctx, ep, local_i, ep_i: None)
    monkeypatch.setattr(scoring, "_eval_zero_mean_gp_baselines", zero_mean_gp)
    monkeypatch.setattr(scoring, "_eval_icl_episode", icl)
    monkeypatch.setattr(scoring, "_select_best_baseline_cv", lambda *a: (0.0, None, []))
    monkeypatch.setattr(scoring, "_print_episode", lambda *a: None)
    monkeypatch.setattr(scoring, "_kernel_composition_label", lambda ep: "stub")


def _ctx(zeromean_gp: bool) -> Any:
    baselines = SimpleNamespace(
        zeromean_gp=zeromean_gp, n_steps_zeromean_gp=1, lr_zeromean_gp=0.1, n_restarts_zeromean_gp=1
    )
    spec = SimpleNamespace(baselines=baselines, seed=0, selection=SimpleNamespace(n_folds=2, min_fold_size=1))
    cpu = torch.device("cpu")
    return SimpleNamespace(
        spec=spec, device=cpu, baseline_device=cpu, icl_model=None, oracle_mode="prior", prior_cfg={}
    )


@pytest.mark.parametrize("y_log_std", [0.0, _LOG_STD])
def test_each_row_gets_the_jacobian_once(monkeypatch: MonkeyPatch, y_log_std: float) -> None:
    _stub_scoring(monkeypatch)
    ep = {"z_test": torch.zeros(_N), "y_log_std": y_log_std}
    baselines = ({"gp_rbf": 0.0}, {"gp_rbf": torch.eye(_N)}, {"gp_rbf": dict(_STD_ROW)})

    rows = scoring._score_episode(_ctx(zeromean_gp=True), ep, 0, 0, 0, baselines).total_nlls

    # Standardized-y baseline: marginal and total shifted by log sigma_y, copula unchanged.
    assert rows["gp_rbf"] == pytest.approx({"total": 1.0 + y_log_std, "marginal": 0.6 + y_log_std, "copula": 0.4})
    # Zero-mean GP row: already in raw nats, never shifted.
    assert rows["gp_zeromean_rbf"] == pytest.approx(_RAW_ROW)
    # ICL row: raw nats, never shifted.
    assert rows["icl"]["total"] == pytest.approx(0.5)


@pytest.mark.parametrize("y_log_std", [0.0, _LOG_STD])
def test_gp_selection_rows_shift_once(monkeypatch: MonkeyPatch, y_log_std: float) -> None:
    """gp_evidence / gp_evidence_mix are fitted rows (shifted); the plug-in marginal row is raw nats (not shifted)."""
    _stub_scoring(monkeypatch)
    ep = {"z_test": torch.zeros(_N), "y_log_std": y_log_std}
    mix = {"total": 1.5, "marginal": float("nan"), "copula": float("nan")}
    baselines = (
        {"gp_evidence": 0.0},
        {"gp_evidence": torch.eye(_N)},
        {"gp_evidence": dict(_STD_ROW), "gp_evidence_mix": dict(mix)},
    )

    rows = scoring._score_episode(_ctx(zeromean_gp=False), ep, 0, 0, 0, baselines).total_nlls

    assert rows["gp_evidence"]["total"] == pytest.approx(1.0 + y_log_std)
    assert rows["gp_evidence_mix"]["total"] == pytest.approx(1.5 + y_log_std)
    # Stub marginal: z_test = 0 and log_pdf = 0, so with R = I every part is 0, whatever y_log_std.
    assert rows["marg_gp_evidence"] == pytest.approx({"total": 0.0, "marginal": 0.0, "copula": 0.0}, abs=1e-5)


def test_shift_does_not_mutate_cached_rows(monkeypatch: MonkeyPatch) -> None:
    """The baseline cache entry passed in keeps its standardized-y values."""
    _stub_scoring(monkeypatch)
    y_nlls = {"gp_rbf": dict(_STD_ROW)}
    ep = {"z_test": torch.zeros(_N), "y_log_std": _LOG_STD}
    scoring._score_episode(_ctx(zeromean_gp=False), ep, 0, 0, 0, ({"gp_rbf": 0.0}, {"gp_rbf": torch.eye(_N)}, y_nlls))
    assert y_nlls["gp_rbf"] == _STD_ROW
