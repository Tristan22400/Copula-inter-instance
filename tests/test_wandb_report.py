"""eval.runners.wandb_report: the NLL decomposition, the size-grid copula and the rendering, on fake runs (no network)."""

from __future__ import annotations

import math

import pytest

from eval.runners.wandb_report import (
    VAL_KEYS,
    RunData,
    decompose,
    gap_key,
    grid_copula,
    marginal_setup,
    render,
    summarize_run,
)


def _val_row(step: int, full: float, marginal: float, copula_exact: float) -> dict[str, float]:
    return {
        "_step": step,
        VAL_KEYS["full"]: full,
        VAL_KEYS["marginal"]: marginal,
        VAL_KEYS["copula"]: full - marginal,
        VAL_KEYS["copula_exact"]: copula_exact,
        VAL_KEYS["oracle_full"]: -0.014,
        VAL_KEYS["headroom"]: 0.104,
    }


HEADROOM = {(10, 100): 0.5, (10, 1000): 0.8, (100, 100): 0.04, (100, 1000): 0.35}


def _grid_row(step: int, copula: dict[tuple[int, int], float]) -> dict[str, float]:
    """A size-grid row whose cells hold the given exact-z copula NLL (gap = copula + headroom)."""
    return {"_step": step, **{gap_key(*pn): c + HEADROOM[pn] for pn, c in copula.items()}}


def _run(label: str, full: float, setup: str = "z_train_source=tabicl") -> RunData:
    return RunData(
        label=label,
        run_id=label.lower(),
        state="finished",
        marginal_setup=setup,
        val_rows=[
            _val_row(0, 6.0, 0.8, 5.9),
            _val_row(5000, full - 0.01, 0.8, -0.004),
            _val_row(10000, full, 0.8, -0.003),
        ],
        headroom=HEADROOM,
        grid_rows=[
            _grid_row(0, dict.fromkeys(HEADROOM, 5.0)),
            _grid_row(10000, {(10, 100): -0.03, (10, 1000): -0.14, (100, 100): 0.005, (100, 1000): -0.03}),
        ],
    )


def test_decompose_splits_run_and_oracle_into_marginal_plus_copula() -> None:
    d = decompose(_val_row(7, full=0.768, marginal=0.785, copula_exact=-0.003))
    assert d["step"] == 7
    assert d["copula"] == pytest.approx(-0.017)
    assert d["oracle_copula"] == pytest.approx(-0.104)
    assert d["oracle_marginal"] == pytest.approx(0.090)
    assert d["excess"] == pytest.approx(d["excess_marginal"] + d["excess_copula"])
    assert d["excess_marginal"] == pytest.approx(0.785 - 0.090)


def test_grid_copula_is_gap_minus_headroom_with_per_P_and_oracle_means() -> None:
    g = grid_copula(_grid_row(3, {(10, 100): -0.03, (10, 1000): -0.14, (100, 100): 0.0, (100, 1000): -0.03}), HEADROOM)
    assert g["cells"]["P10_N1000"] == pytest.approx(-0.14)
    assert g["per_P"][10] == pytest.approx(-0.085)
    assert g["oracle_per_P"][100] == pytest.approx(-(0.04 + 0.35) / 2)
    assert g["mean"] == pytest.approx(-0.05)
    assert g["oracle_mean"] == pytest.approx(-(0.5 + 0.8 + 0.04 + 0.35) / 4)
    assert g["captured"] == pytest.approx(g["mean"] / g["oracle_mean"])
    assert grid_copula({"_step": 1}, HEADROOM) == {}


def test_summarize_run_reports_last_best_and_grid_endpoints() -> None:
    rec = summarize_run(_run("A", 0.70))
    assert rec is not None
    assert rec["last"]["step"] == 10000
    assert rec["best_full"]["step"] == 5000
    assert rec["best_copula_exact"]["step"] == 5000
    assert rec["grid_first"]["mean"] == pytest.approx(5.0)
    assert rec["grid_last"]["step"] == 10000
    empty = RunData("B", "b", "running", "", val_rows=[{"_step": 0}])
    assert summarize_run(empty) is None


def test_render_groups_by_validation_set_and_lists_marginal_setups() -> None:
    records = [r for r in (summarize_run(_run("A", 0.70)), summarize_run(_run("B", 0.65, "x"))) if r]
    text = render(records)
    assert text.count("## Validation set") == 1
    assert "**+0.7000**" in text and "**+0.6500**" in text
    assert "## Size grid" in text and "**oracle**" in text
    assert "- B (b, finished): x" in text


def test_marginal_setup_names_the_marginal_and_the_resume_point() -> None:
    cfg = {
        "data": {"z_train_source": "kumo_small", "z_train_marginal_ckpt": "a/b/step_1.pt", "P_min": 10, "P_max": 1000},
        "training": {"resume_ckpt": "x/F_nano_kumo_A/step_1000000.pt"},
    }
    s = marginal_setup(cfg)
    assert "marginal_ckpt=step_1.pt" in s and "P=[10,1000]" in s
    assert "resumed from F_nano_kumo_A/step_1000000.pt" in s
    assert "pit_ckpt=ckpt.pt" in marginal_setup({"tabicl": {"pit_ckpt": "c/ckpt.pt"}})
    assert math.isnan(decompose({})["full"])
