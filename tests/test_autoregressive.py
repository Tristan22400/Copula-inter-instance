"""Tests for eval/baselines/autoregressive.py and its place in eval_checkpoint's tables.

  1. Step 0 reproduces the one-shot PIT's log_pdf_test.
  2. Teacher forcing appends the true values in visit order.
  3. conditioning="sample" appends draws, reproducibly for a seed.
  4. Orderings are permutations seeded by global index; "natural" is identity.
  5. max_context caps the context rows and keeps the P context points.
  6. ar_parts_from_log_pdf: total = marginal + copula.
  7. "autoregressive" is in _TOTAL_NLL_ORDER but not _METHOD_ORDER.
"""

from __future__ import annotations

import math

import pytest
import torch

from copula_inter.era5_live_dataset import _pit_group  # noqa: E402
from eval.baselines.autoregressive import (  # noqa: E402
    _orderings,
    ar_parts_from_log_pdf,
    autoregressive_log_pdf,
)
from tests.test_pit_batched import RowIndependentFakeTabICL  # noqa: E402


class RecordingFakeTabICL(RowIndependentFakeTabICL):
    """RowIndependentFakeTabICL that records the shape of every table it is called with."""

    def __init__(self, q: int = 3):
        super().__init__(q)
        self.calls: list[tuple[int, int]] = []   # (n_context, n_query)

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.calls.append((y.shape[1], X.shape[1] - y.shape[1]))
        return super().forward(X, y)


def _episodes(B=2, P=6, N=5, d_x=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(B, P, d_x, generator=g),
        torch.randn(B, P, generator=g) * 3.0 + 280.0,   # absolute-scale, like ERA5 Kelvin
        torch.randn(B, N, d_x, generator=g),
        torch.randn(B, N, generator=g) * 3.0 + 280.0,
    )


def test_ar_step0_matches_the_one_shot_marginal():
    torch.manual_seed(0)
    tabicl = RowIndependentFakeTabICL()
    x_tr, y_tr, x_te, y_te = _episodes()

    one_shot = _pit_group(x_tr, y_tr, x_te, y_te, tabicl, k_folds=3)
    ar = autoregressive_log_pdf(
        tabicl, x_tr, y_tr, x_te, y_te, order="natural",
        conditioning="teacher_forcing", seed=0,
    )

    # Natural order visits test index 0 first.
    assert torch.allclose(
        ar["log_pdf"][:, 0], one_shot["log_pdf_test"][:, 0], atol=1e-5
    )
    # Later points differ from the one-shot values.
    assert not torch.allclose(
        ar["log_pdf"][:, 1:], one_shot["log_pdf_test"][:, 1:], atol=1e-5
    )


def test_teacher_forcing_appends_the_truth_in_visit_order():
    tabicl = RowIndependentFakeTabICL()
    x_tr, y_tr, x_te, y_te = _episodes(seed=1)

    ar = autoregressive_log_pdf(
        tabicl, x_tr, y_tr, x_te, y_te, order="random",
        conditioning="teacher_forcing", seed=7,
    )
    expected = y_te.gather(1, ar["order"])
    assert torch.allclose(ar["appended"], expected, atol=1e-4)


def test_sample_conditioning_departs_from_the_truth_but_is_reproducible():
    tabicl = RowIndependentFakeTabICL()
    x_tr, y_tr, x_te, y_te = _episodes(seed=2)

    kw = dict(order="natural", conditioning="sample", seed=11)
    a = autoregressive_log_pdf(tabicl, x_tr, y_tr, x_te, y_te, **kw)
    b = autoregressive_log_pdf(tabicl, x_tr, y_tr, x_te, y_te, **kw)

    assert torch.allclose(a["appended"], b["appended"], atol=0)
    assert torch.allclose(a["log_pdf"], b["log_pdf"], atol=0)
    assert not torch.allclose(a["appended"], y_te.gather(1, a["order"]), atol=1e-3)


def test_orderings_are_permutations_keyed_on_the_global_episode_index():
    B, N = 3, 9
    o = _orderings(B, N, "random", seed=5, episode_indices=[100, 101, 102])
    for b in range(B):
        assert sorted(o[b].tolist()) == list(range(N))
    # Same global indices -> same orders, whatever position they sit at.
    again = _orderings(B, N, "random", seed=5, episode_indices=[100, 101, 102])
    assert torch.equal(o, again)
    shifted = _orderings(2, N, "random", seed=5, episode_indices=[101, 102])
    assert torch.equal(shifted, o[1:])
    # A different global index gives a different order.
    assert not torch.equal(
        _orderings(1, N, "random", seed=5, episode_indices=[100])[0],
        _orderings(1, N, "random", seed=5, episode_indices=[999])[0],
    )
    assert torch.equal(
        _orderings(B, N, "natural", seed=5, episode_indices=None)[0],
        torch.arange(N),
    )


def test_unknown_order_and_conditioning_are_rejected():
    tabicl = RowIndependentFakeTabICL()
    x_tr, y_tr, x_te, y_te = _episodes(seed=3)
    with pytest.raises(ValueError, match="order must be one of"):
        autoregressive_log_pdf(tabicl, x_tr, y_tr, x_te, y_te, order="spiral")
    with pytest.raises(ValueError, match="conditioning must be one of"):
        autoregressive_log_pdf(tabicl, x_tr, y_tr, x_te, y_te, conditioning="beam")


def test_max_context_caps_the_table_and_keeps_the_episodes_own_context():
    P, N = 6, 5
    x_tr, y_tr, x_te, y_te = _episodes(B=1, P=P, N=N, seed=4)

    uncapped = RecordingFakeTabICL()
    autoregressive_log_pdf(uncapped, x_tr, y_tr, x_te, y_te, order="natural")
    assert [c[0] for c in uncapped.calls] == [P + i for i in range(N)]
    # Constant table: context + query is always P + N, every step.
    assert all(c[0] + c[1] == P + N for c in uncapped.calls)

    cap = P + 2
    capped = RecordingFakeTabICL()
    autoregressive_log_pdf(
        capped, x_tr, y_tr, x_te, y_te, order="natural", max_context=cap,
    )
    assert [c[0] for c in capped.calls] == [min(P + i, cap) for i in range(N)]
    assert max(c[0] for c in capped.calls) == cap


def test_ar_parts_split_is_exact_and_independence_is_copula_zero():
    ar = torch.tensor([-1.0, -2.0, -3.0])
    marg = torch.tensor([-1.5, -1.5, -1.5])

    parts = ar_parts_from_log_pdf(ar, marg)
    assert math.isclose(parts["total"], 2.0, rel_tol=1e-9)
    assert math.isclose(parts["marginal"], 1.5, rel_tol=1e-9)
    assert math.isclose(parts["copula"], parts["total"] - parts["marginal"], rel_tol=1e-9)

    same = ar_parts_from_log_pdf(marg, marg)
    assert math.isclose(same["copula"], 0.0, abs_tol=1e-12)


def test_autoregressive_is_a_total_table_row_only():
    from eval.runners.eval_tables import (
        _METHOD_ORDER,
        _TOTAL_NLL_ORDER,
        _TOTAL_RANK_ORDER,
    )

    assert "autoregressive" in dict(_TOTAL_NLL_ORDER)
    # No correlation matrix: not in the z-space table or best-baseline ranking.
    assert "autoregressive" not in dict(_METHOD_ORDER)
    # Y-space ranks include independence and the best GP, which are not z-space competitors.
    assert "independence_marginal" in dict(_TOTAL_RANK_ORDER)
    assert "best_gp_total" in dict(_TOTAL_RANK_ORDER)


def test_ar_note_warns_on_sampled_conditioning():
    from eval.runners.eval_tables import _NAN_PARTS, _ar_note

    rows = [{"autoregressive": {"total": 0.5, "marginal": 1.0, "copula": -0.5}},
            {"autoregressive": _NAN_PARTS.copy()}]
    forced = _ar_note(rows, "random", "teacher_forcing", None)
    assert "1/2 episodes" in forced and "WARNING" not in forced
    sampled = _ar_note(rows, "random", "sample", 64)
    assert "WARNING" in sampled and "max_context=64" in sampled
    assert _ar_note([{"autoregressive": _NAN_PARTS.copy()}], "random",
                    "teacher_forcing", None) is None
