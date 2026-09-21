"""
test_autoregressive.py — Tests for eval/baselines/autoregressive.py, the
chain-rule ("autoregressive") joint density built from the marginal alone,
and for its wiring into eval_checkpoint.py's total Y-space NLL table.

Tests verify:
  1. STEP 0 IS THE ONE-SHOT MARGINAL. The chain's first prediction, with the
     full test block still present in the table, reproduces the ordinary
     K-fold PIT's log_pdf_test at that point. This is the identity the whole
     marginal/copula split rests on -- "marginal" in the printed row is the
     one-shot NLL and "copula" is total minus it, which is only meaningful if
     the chain really does start from the one-shot model. It is also what the
     module's "constant table, moving split" design exists to buy: dropping
     the unrevealed queries would change the model's input distribution and
     break this.
  2. Teacher forcing appends the TRUTH, in visit order -- the property that
     makes the summed log-density an exact joint density rather than a
     sampled path.
  3. --ar_conditioning=sample appends something OTHER than the truth, and
     does so reproducibly for a fixed seed (a diagnostic nobody can rerun is
     not a diagnostic).
  4. Orderings are genuine permutations, are seeded from the episode's GLOBAL
     index rather than its position in the batch (so shard k of a run agrees
     with a single long run), and "natural" really is identity order.
  5. max_context caps the number of context rows the model is actually
     called with, and always keeps the episode's own P context points.
  6. ar_parts_from_log_pdf's split is exactly total = marginal + copula, with
     copula == 0 recovering independence -- the same sign and normalization
     convention as loss.y_space_nll, since the two share a table.
  7. "autoregressive" is in _TOTAL_NLL_ORDER but NOT in _METHOD_ORDER: it has
     no correlation matrix, so it must not reach the z-space table or the
     best-of-baselines ranking.
  8. attach_autoregressive runs on any episode source: it batches only
     episodes of matching (P, N) and leaves each episode's result in raw
     nats, undoing a standardized episode's y_log_std so the row is
     differenceable against that episode's one-shot log_pdf_test.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from era5_live_dataset import _pit_group  # noqa: E402
from eval.baselines.autoregressive import (  # noqa: E402
    _orderings,
    ar_parts_from_log_pdf,
    attach_autoregressive,
    autoregressive_log_pdf,
)
from tests.test_pit_batched import RowIndependentFakeTabICL  # noqa: E402


class RecordingFakeTabICL(RowIndependentFakeTabICL):
    """RowIndependentFakeTabICL that remembers the shape of every table it was
    called with — the only way to assert on max_context, whose effect is on
    what the model SEES and not on any returned value."""

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

    # "natural" order visits test index 0 first, so that column is the one the
    # chain evaluated before revealing anything. Both are in raw nats.
    assert torch.allclose(
        ar["log_pdf"][:, 0], one_shot["log_pdf_test"][:, 0], atol=1e-5
    )
    # ...and only that column: by index 1 the chain has conditioning the
    # one-shot pass never had, so the two must have genuinely diverged
    # (otherwise the chain is silently a no-op).
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
    # Different global index -> different order (this is what makes an episode
    # reproducible across shards rather than across batch slots).
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
    from eval.runners.eval_checkpoint import _METHOD_ORDER, _TOTAL_NLL_ORDER

    assert "autoregressive" in dict(_TOTAL_NLL_ORDER)
    # No correlation matrix exists for it, so it must never reach the z-space
    # copula table or the best-of-baselines ranking, both of which are driven
    # by _METHOD_ORDER.
    assert "autoregressive" not in dict(_METHOD_ORDER)


def test_ar_note_warns_on_sampled_conditioning():
    from eval.runners.eval_checkpoint import _NAN_PARTS, _ar_note

    rows = [{"autoregressive": {"total": 0.5, "marginal": 1.0, "copula": -0.5}},
            {"autoregressive": _NAN_PARTS.copy()}]
    forced = _ar_note(rows, "random", "teacher_forcing", None)
    assert "1/2 episodes" in forced and "WARNING" not in forced
    sampled = _ar_note(rows, "random", "sample", 64)
    assert "WARNING" in sampled and "max_context=64" in sampled
    assert _ar_note([{"autoregressive": _NAN_PARTS.copy()}], "random",
                    "teacher_forcing", None) is None


class SmoothFakeTabICL(nn.Module):
    """Fake marginal whose predictive is a SMOOTH function of the table: the
    context mean as location, a fixed scale.

    RowIndependentFakeTabICL hashes the table's float sum into an RNG seed, so
    a 1e-7 difference in the input gives a completely different distribution —
    fine for the equivalence tests it was written for, useless for checking a
    numerical identity that only ever holds up to round-off.
    """

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = X.shape[1] - y.shape[1]
        loc = y.mean(dim=1, keepdim=True).expand(-1, n)
        return torch.stack([loc, torch.zeros_like(loc), torch.zeros_like(loc)], dim=-1)

    def quantile_dist(self, logits_flat: torch.Tensor):
        return RowIndependentFakeTabICL.quantile_dist(self, logits_flat)


def _episode_dicts(shapes, seed=0):
    """One episode dict per (P, N, d_x) in `shapes`, carrying exactly the fields
    attach_autoregressive is allowed to read — the fields every episode source
    in this repo populates."""
    eps = []
    for i, (P, N, d_x) in enumerate(shapes):
        x_tr, y_tr, x_te, y_te = _episodes(B=1, P=P, N=N, d_x=d_x, seed=seed + i)
        eps.append({
            "x_norm_train": x_tr[0], "y_train": y_tr[0],
            "x_norm_test": x_te[0], "y_test": y_te[0],
        })
    return eps


def test_attach_batches_matching_shapes_and_isolates_mismatched_ones():
    tabicl = RecordingFakeTabICL()
    # Two episodes of one geometry (batchable), then one differing in N and
    # one differing only in d_x — live-generated synthetic episodes vary in
    # both, and either alone makes the tables unstackable.
    shapes = [(6, 5, 3), (6, 5, 3), (7, 4, 3), (6, 5, 2)]
    eps = _episode_dicts(shapes)

    n = attach_autoregressive(
        list(enumerate(eps)), tabicl, order="natural", seed=0, batch_size=8,
        verbose=False,
    )

    assert n == 4
    for ep, (_, N, _) in zip(eps, shapes):
        assert ep["ar_log_pdf"].shape == (N,)
    # 5 chain steps for the batched pair, plus 4 and 5 for the two that could
    # not join it — the pair shared its forwards rather than each paying its own.
    assert len(tabicl.calls) == 5 + 4 + 5


def test_attach_undoes_a_standardized_episodes_jacobian():
    """An episode stored in standardized y (ERA5's --era5_standardize_y)
    carries log(std) as y_log_std; the attached row must come back in the same
    RAW nats as an episode stored raw, or it is not differenceable against
    log_pdf_test."""
    raw = _episode_dicts([(6, 5, 3)])[0]
    mean, std = raw["y_train"].mean(), raw["y_train"].std()
    scaled = dict(
        raw,
        y_train=(raw["y_train"] - mean) / std,
        y_test=(raw["y_test"] - mean) / std,
        y_log_std=float(std.log()),
    )

    tabicl = SmoothFakeTabICL()
    attach_autoregressive([(0, raw)], tabicl, order="natural", seed=0, verbose=False)
    attach_autoregressive([(0, scaled)], tabicl, order="natural", seed=0, verbose=False)

    assert torch.allclose(raw["ar_log_pdf"], scaled["ar_log_pdf"], atol=1e-4)
    # The correction is the Jacobian, not a no-op: without it the standardized
    # episode's row would sit log(std) away from the raw one.
    assert not torch.allclose(
        raw["ar_log_pdf"], scaled["ar_log_pdf"] + float(std.log()), atol=1e-4
    )


def test_attach_keys_the_ordering_on_the_given_global_index():
    """The (index, episode) pairs carry GLOBAL indices, so an episode gets the
    same visit order whether it was scored in one long run or in a shard."""
    tabicl = RowIndependentFakeTabICL()
    a, b = _episode_dicts([(6, 5, 3)]), _episode_dicts([(6, 5, 3)])

    attach_autoregressive([(7, a[0])], tabicl, seed=0, verbose=False)
    alone = a[0]["ar_log_pdf"].clone()
    # Same episode, same global index 7, but now batched behind another one.
    attach_autoregressive([(0, b[0]), (7, a[0])], tabicl, seed=0, verbose=False)

    assert torch.allclose(alone, a[0]["ar_log_pdf"], atol=1e-5)
