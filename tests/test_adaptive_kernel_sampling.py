"""test_adaptive_kernel_sampling.py — Regression tests for
training.adaptive_kernel_sampling: the DoReMi/GroupDRO-style
exponentiated-gradient reweighting of live-generation kernel-family sampling
(train.py::_update_adaptive_kernel_weights, data_gen.py's
_sample_kernel_chain_structure/_resolve_kernel_name kernel_weights param).

Two things matter here: (1) the pure weight-update function biases toward
worse-fit families as intended, respects the anti-starvation floor, and
degrades gracefully on missing/NaN signal; (2) kernel_weights=None (every
existing call site, and every family without a probe) reproduces today's
uniform random.choices/random.choice behavior exactly — this feature must be
a no-op unless explicitly opted into.
"""

from __future__ import annotations

import random
from collections import Counter

import torch
from omegaconf import OmegaConf

from data_gen import _COMPOSABLE_KERNELS, _sample_kernel_chain_structure, _weights_for_pool
import train


def _uniform_weights() -> torch.Tensor:
    n = len(_COMPOSABLE_KERNELS)
    return torch.full((n,), 1.0 / n, dtype=torch.float32)


# ---------------------------------------------------------------------------
# _update_adaptive_kernel_weights
# ---------------------------------------------------------------------------


def test_update_weights_sums_to_one():
    prev = _uniform_weights()
    metrics = {
        "oracle_diag/kernel_fit/rbf/gap_nll": 0.4,
        "oracle_diag/kernel_fit/matern32/gap_nll": 0.05,
    }
    out = train._update_adaptive_kernel_weights(prev, metrics, lr=1.0, floor=0.05)
    assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-5)


def test_update_weights_biases_toward_worse_family():
    """rbf has a much bigger posterior gap (0.45) than matern32 (0.05) ->
    rbf should end up with strictly more weight."""
    prev = _uniform_weights()
    metrics = {
        "oracle_diag/kernel_fit/rbf/gap_nll": 0.45,
        "oracle_diag/kernel_fit/matern32/gap_nll": 0.05,
    }
    out = train._update_adaptive_kernel_weights(prev, metrics, lr=1.0, floor=0.05)
    i_rbf = _COMPOSABLE_KERNELS.index("rbf")
    i_mat = _COMPOSABLE_KERNELS.index("matern32")
    assert out[i_rbf] > out[i_mat]
    assert out[i_rbf] > prev[i_rbf]  # rbf's share grew from uniform


def test_update_weights_respects_floor():
    """A family with a catastrophically negative gap (looks perfect, better
    than the posterior ceiling) must not be driven below floor/N --
    anti-starvation."""
    n = len(_COMPOSABLE_KERNELS)
    prev = _uniform_weights()
    metrics = {
        "oracle_diag/kernel_fit/rbf/gap_nll": 100.0,  # huge positive gap elsewhere
    }
    floor = 0.2
    out = train._update_adaptive_kernel_weights(prev, metrics, lr=1.0, floor=floor)
    min_allowed = floor * (1.0 / n)
    assert (out >= min_allowed - 1e-6).all()


def test_update_weights_missing_or_nan_signal_is_neutral():
    """Families absent from metrics (no probe) or reporting NaN get gap=0 —
    no directional pressure, only the floor's pull toward uniform."""
    prev = _uniform_weights()
    metrics = {
        "oracle_diag/kernel_fit/rbf/gap_nll": float("nan"),
    }
    out = train._update_adaptive_kernel_weights(prev, metrics, lr=1.0, floor=0.05)
    assert torch.isfinite(out).all()
    assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-5)


def test_update_weights_extreme_gap_does_not_overflow():
    prev = _uniform_weights()
    metrics = {
        "oracle_diag/kernel_fit/rbf/gap_nll": -2e9,
    }
    out = train._update_adaptive_kernel_weights(prev, metrics, lr=1.0, floor=0.05)
    assert torch.isfinite(out).all()
    assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-5)


def test_update_weights_ignores_excluded_family_gap():
    """A family in `exclude` must not move off-uniform even with a huge
    posterior gap and a probe present -- it's never in _sample_kernel_chain_
    structure's post-exclude pool, so letting its weight track performance
    is pure noise (see composite_exclude_kernels docs in gp_tasks.yaml)."""
    prev = _uniform_weights()
    metrics = {
        "oracle_diag/kernel_fit/periodic/gap_nll": 100.0,
        "oracle_diag/kernel_fit/rbf/gap_nll": 0.4,
    }
    out_excluded = train._update_adaptive_kernel_weights(
        prev, metrics, lr=1.0, floor=0.05, exclude={"periodic"}
    )
    out_included = train._update_adaptive_kernel_weights(
        prev, metrics, lr=1.0, floor=0.05, exclude=None
    )
    i_periodic = _COMPOSABLE_KERNELS.index("periodic")
    # Excluding periodic's gap keeps its weight far below what the same huge
    # gap would otherwise drive it to.
    assert out_excluded[i_periodic] < out_included[i_periodic]
    assert torch.isclose(out_excluded.sum(), torch.tensor(1.0), atol=1e-5)


# ---------------------------------------------------------------------------
# signal="tabicl" (training.adaptive_kernel_signal)
# ---------------------------------------------------------------------------


def test_signal_tabicl_uses_tabicl_gap_not_oracle_gap():
    """With signal='tabicl', a family whose oracle gap is small but whose
    TabICL-marginal gap is large must still gain weight -- the tabicl key,
    not the oracle_diag one, drives the update."""
    prev = _uniform_weights()
    metrics = {
        "oracle_diag/kernel_fit/rbf/gap_nll": 0.01,       # tiny under the oracle marginal
        "kernel_fit/rbf/gap_nll_tabicl": 0.45,            # large under TabICL's real PIT
        "oracle_diag/kernel_fit/matern32/gap_nll": 0.01,
        "kernel_fit/matern32/gap_nll_tabicl": 0.01,
    }
    out = train._update_adaptive_kernel_weights(
        prev, metrics, lr=1.0, floor=0.05, signal="tabicl"
    )
    i_rbf = _COMPOSABLE_KERNELS.index("rbf")
    i_mat = _COMPOSABLE_KERNELS.index("matern32")
    assert out[i_rbf] > out[i_mat]
    assert out[i_rbf] > prev[i_rbf]


def test_signal_tabicl_falls_back_to_oracle_gap_when_missing():
    """A family with no kernel_fit/<family>/gap_nll_tabicl (e.g. no PIT
    checkpoint configured this run) must fall back to its oracle_diag gap
    instead of silently getting gap=0 -- signal='tabicl' shouldn't disable
    the curriculum entirely just because the tabicl cache is empty."""
    prev = _uniform_weights()
    metrics_tabicl_missing = {
        "oracle_diag/kernel_fit/rbf/gap_nll": 0.45,
        "oracle_diag/kernel_fit/matern32/gap_nll": 0.05,
    }
    out_tabicl = train._update_adaptive_kernel_weights(
        prev, metrics_tabicl_missing, lr=1.0, floor=0.05, signal="tabicl"
    )
    out_oracle = train._update_adaptive_kernel_weights(
        prev, metrics_tabicl_missing, lr=1.0, floor=0.05, signal="oracle"
    )
    assert torch.equal(out_tabicl, out_oracle)


def test_signal_default_is_oracle():
    """The `signal` kwarg defaults to 'oracle' -- unchanged behavior for
    every existing call site that doesn't pass it explicitly."""
    prev = _uniform_weights()
    metrics = {
        "oracle_diag/kernel_fit/rbf/gap_nll": 0.4,
        "kernel_fit/rbf/gap_nll_tabicl": 999.0,  # must be ignored by default
    }
    out_default = train._update_adaptive_kernel_weights(prev, metrics, lr=1.0, floor=0.05)
    out_explicit_oracle = train._update_adaptive_kernel_weights(
        prev, metrics, lr=1.0, floor=0.05, signal="oracle"
    )
    assert torch.equal(out_default, out_explicit_oracle)


# ---------------------------------------------------------------------------
# _sample_kernel_chain_structure weighting + no-op default
# ---------------------------------------------------------------------------


def _base_cfg(**data_overrides):
    data = {
        "composite_num_kernels_min": 1,
        "composite_num_kernels_max": 1,  # pin m=1 so names==[single kernel], no chain noise
    }
    data.update(data_overrides)
    return OmegaConf.create({"data": data})


def test_sample_kernel_chain_none_reproduces_uniform_given_same_seed():
    cfg_a = _base_cfg()
    cfg_b = _base_cfg()
    random.seed(12345)
    names_a, ops_a, chain_a = _sample_kernel_chain_structure(cfg_a, kernel_weights=None)
    random.seed(12345)
    names_b, ops_b, chain_b = _sample_kernel_chain_structure(cfg_b, kernel_weights=None)
    assert (names_a, ops_a, chain_a) == (names_b, ops_b, chain_b)


def test_sample_kernel_chain_skewed_weights_shift_frequency():
    n = len(_COMPOSABLE_KERNELS)
    weights = torch.full((n,), 0.01, dtype=torch.float32)
    target = "matern52"
    weights[_COMPOSABLE_KERNELS.index(target)] = 1.0  # dominant weight
    cfg = _base_cfg()
    random.seed(0)
    counts = Counter()
    for _ in range(500):
        names, _, _ = _sample_kernel_chain_structure(cfg, kernel_weights=weights)
        counts[names[0]] += 1
    # With m pinned to 1, names[0] is the only draw per call -> the
    # overwhelmingly-favored family should dominate the empirical frequency.
    assert counts[target] > 400


def test_weights_for_pool_none_passthrough():
    assert _weights_for_pool(["rbf", "matern32"], None) is None


def test_weights_for_pool_renormalizes_over_subset():
    n = len(_COMPOSABLE_KERNELS)
    weights = torch.zeros(n, dtype=torch.float32)
    weights[_COMPOSABLE_KERNELS.index("rbf")] = 3.0
    weights[_COMPOSABLE_KERNELS.index("matern32")] = 1.0
    out = _weights_for_pool(["rbf", "matern32"], weights)
    assert out == [0.75, 0.25]
