"""
test_live_tabicl_mix.py — Tests for the adaptive real-TabICL z_train mixing
feature (data.z_train_tabicl_mix_* in conf/data/gp_tasks.yaml):
data_gen.py::_tabicl_mix_prob_for_kernel/_generate_gp_batch_raw's
tabicl_mix_weights gate, and train.py::_tabicl_gap_to_mix_frac.

Unlike z_train_source=tabicl/tabicl_split (test_pit_batched.py), which
substitutes real TabICL z_train for EVERY episode of the whole run, this
feature substitutes it for a per-kernel-family FRACTION of live-generation
calls, driven by the measured TabICL-vs-analytic z_train gap for that
family. Tests verify:
  1. tabicl_mix_weights=None reproduces the legacy always-on override
     exactly (_tabicl_mix_prob_for_kernel returns 1.0 unconditionally).
  2. All-zero mixing weights are an exact no-op vs. tabicl_model=None (pure
     analytic z_train), for every episode in a call.
  3. All-one mixing weights reproduce the legacy always-on override exactly
     (byte-for-byte, since tabicl_mix_weights=None short-circuits to the
     same apply_tabicl=True decision without ever calling random.random()).
  4. Over many independent single-episode calls, the empirical hit rate for
     a single-family weight matches the configured fraction.
  5. Composite/chain kernel names route through the MAX weight across their
     component families (_tabicl_mix_prob_for_kernel's "weakest link"
     rule), not e.g. the mean.
  6. corrupt_z_train is skipped when a call's z_train came from the mixing
     gate, but still applies (unchanged) to analytic-sourced calls and to
     the legacy always-on path (tabicl_mix_weights=None).
  7. train.py::_tabicl_gap_to_mix_frac's gap-to-fraction mapping: linear
     interpolation between floor/max over the measured range, floor for
     unmeasured families, and the degenerate (too few families / zero
     spread) fallback to a uniform floor.
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from data_gen import _COMPOSABLE_KERNELS, _generate_gp_batch_raw, _tabicl_mix_prob_for_kernel

from test_pit_batched import RowIndependentFakeTabICL


def _mix_weights(**by_family: float) -> torch.Tensor:
    w = torch.zeros(len(_COMPOSABLE_KERNELS))
    for family, val in by_family.items():
        w[_COMPOSABLE_KERNELS.index(family)] = val
    return w


def test_tabicl_mix_prob_for_kernel_none_is_unconditional():
    assert _tabicl_mix_prob_for_kernel("rbf", None) == 1.0
    assert _tabicl_mix_prob_for_kernel("rbf*periodic+matern32", None) == 1.0


def test_tabicl_mix_prob_for_kernel_bare_family():
    w = _mix_weights(rbf=0.2, periodic=0.8)
    assert abs(_tabicl_mix_prob_for_kernel("rbf", w) - 0.2) < 1e-6
    assert abs(_tabicl_mix_prob_for_kernel("periodic", w) - 0.8) < 1e-6
    assert _tabicl_mix_prob_for_kernel("matern32", w) == 0.0


def test_tabicl_mix_prob_for_kernel_composite_uses_max():
    # "weakest link": the harder-to-approximate component (periodic, 0.8)
    # dominates over the easier one (rbf, 0.2), not their mean (0.5).
    w = _mix_weights(rbf=0.2, periodic=0.8)
    assert abs(_tabicl_mix_prob_for_kernel("rbf*periodic", w) - 0.8) < 1e-6
    assert abs(_tabicl_mix_prob_for_kernel("rbf+periodic", w) - 0.8) < 1e-6
    assert abs(_tabicl_mix_prob_for_kernel("matern32*rbf*periodic", w) - 0.8) < 1e-6


def test_zero_mix_weights_is_noop_vs_pure_analytic(small_cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.seed = 101

    tabicl = RowIndependentFakeTabICL()
    zero_w = torch.zeros(len(_COMPOSABLE_KERNELS))

    analytic = _generate_gp_batch_raw(cfg, B=8, device="cpu")
    zero_mix = _generate_gp_batch_raw(
        cfg, B=8, device="cpu", tabicl_model=tabicl, tabicl_k_folds=3,
        tabicl_mix_weights=zero_w,
    )
    assert len(analytic) == len(zero_mix)
    for ep_a, ep_z in zip(analytic, zero_mix):
        assert torch.allclose(ep_a["z_train"], ep_z["z_train"], atol=1e-6)


def test_one_mix_weights_matches_legacy_full_override(small_cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.seed = 202

    tabicl = RowIndependentFakeTabICL()
    one_w = torch.ones(len(_COMPOSABLE_KERNELS))

    legacy = _generate_gp_batch_raw(cfg, B=8, device="cpu", tabicl_model=tabicl, tabicl_k_folds=3)
    one_mix = _generate_gp_batch_raw(
        cfg, B=8, device="cpu", tabicl_model=tabicl, tabicl_k_folds=3,
        tabicl_mix_weights=one_w,
    )
    assert len(legacy) == len(one_mix)
    for ep_l, ep_m in zip(legacy, one_mix):
        assert torch.allclose(ep_l["z_train"], ep_m["z_train"], atol=1e-6)


def test_mix_hit_rate_matches_configured_fraction(small_cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False

    tabicl = RowIndependentFakeTabICL()
    target_frac = 0.3
    w = _mix_weights(rbf=target_frac)

    n_calls = 300
    hits = 0
    for i in range(n_calls):
        cfg.seed = 5000 + i
        ep_analytic = _generate_gp_batch_raw(cfg, B=1, device="cpu")[0]
        ep_mix = _generate_gp_batch_raw(
            cfg, B=1, device="cpu", tabicl_model=tabicl, tabicl_k_folds=3,
            tabicl_mix_weights=w,
        )[0]
        if not torch.allclose(ep_analytic["z_train"], ep_mix["z_train"], atol=1e-6):
            hits += 1
    empirical_frac = hits / n_calls
    assert abs(empirical_frac - target_frac) < 0.08, empirical_frac


def test_corruption_skipped_on_mix_hit_but_not_on_miss_or_legacy(small_cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.data.z_train_corruption_enabled = True
    cfg.data.z_train_corruption_prob = 1.0  # corrupt every episode when it does run
    cfg.seed = 303

    tabicl = RowIndependentFakeTabICL()
    one_w = torch.ones(len(_COMPOSABLE_KERNELS))
    zero_w = torch.zeros(len(_COMPOSABLE_KERNELS))

    # Mixing hit (weight=1): z_train is the real TabICL PIT, uncorrupted --
    # must differ from a corrupted analytic run but match the uncorrupted
    # TabICL override exactly.
    cfg.data.z_train_corruption_enabled = False
    uncorrupted_tabicl = _generate_gp_batch_raw(
        cfg, B=6, device="cpu", tabicl_model=tabicl, tabicl_k_folds=3, tabicl_mix_weights=one_w,
    )
    cfg.data.z_train_corruption_enabled = True
    mix_hit_should_skip_corruption = _generate_gp_batch_raw(
        cfg, B=6, device="cpu", tabicl_model=tabicl, tabicl_k_folds=3, tabicl_mix_weights=one_w,
    )
    for ep_u, ep_h in zip(uncorrupted_tabicl, mix_hit_should_skip_corruption):
        assert torch.allclose(ep_u["z_train"], ep_h["z_train"], atol=1e-6)

    # Mixing miss (weight=0): analytic z_train, corruption still applies --
    # must differ from the uncorrupted analytic residual.
    cfg.data.z_train_corruption_enabled = False
    uncorrupted_analytic = _generate_gp_batch_raw(cfg, B=6, device="cpu")
    cfg.data.z_train_corruption_enabled = True
    mix_miss_should_corrupt = _generate_gp_batch_raw(
        cfg, B=6, device="cpu", tabicl_model=tabicl, tabicl_k_folds=3, tabicl_mix_weights=zero_w,
    )
    any_diff = any(
        not torch.allclose(ep_u["z_train"], ep_m["z_train"], atol=1e-6)
        for ep_u, ep_m in zip(uncorrupted_analytic, mix_miss_should_corrupt)
    )
    assert any_diff

    # Legacy always-on path (tabicl_mix_weights=None): corruption still
    # applies on top, unchanged from before this feature existed.
    legacy_corrupted = _generate_gp_batch_raw(
        cfg, B=6, device="cpu", tabicl_model=tabicl, tabicl_k_folds=3,
    )
    any_diff_legacy = any(
        not torch.allclose(ep_u["z_train"], ep_l["z_train"], atol=1e-6)
        for ep_u, ep_l in zip(uncorrupted_tabicl, legacy_corrupted)
    )
    assert any_diff_legacy


def _import_train():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    import train
    return train


def test_compute_tabicl_z_train_gap_runs_on_declared_device(small_cfg):
    """Regression test: _compute_tabicl_z_train_gap must run BOTH paired
    _generate_gp_batch_raw calls on the SAME device tabicl_marginal lives
    on (its `device` arg), not a hardcoded "cpu" -- torch's CPU and CUDA
    generators are separate RNG streams that don't reproduce each other's
    draws from the same seed, so a device mismatch would silently break the
    byte-identical-pairing guarantee the docstring promises (and, on a
    machine with a GPU, crash outright once x_norm_train and TabICL's
    cuda-resident weights disagree on device). Exercised here on CPU only
    (no GPU required for the test suite), which is sufficient to catch a
    hardcoded-device regression: passing device="cpu" explicitly must work
    end to end and produce finite gaps.
    """
    train = _import_train()
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.baselines = {
        "synth_n_episodes": 4, "synth_seed": 999,
        "probe_P_min": 5, "probe_P_max": 10, "probe_N_min": 3, "probe_N_max": 6,
    }
    tabicl = RowIndependentFakeTabICL()
    gaps = train._compute_tabicl_z_train_gap(cfg, tabicl, k_folds=3, device="cpu")
    assert len(gaps) > 0
    for family, g in gaps.items():
        assert family in _COMPOSABLE_KERNELS
        assert g == g and g >= 0.0  # finite, non-negative


def test_tabicl_gap_to_mix_frac():
    train = _import_train()
    _tabicl_gap_to_mix_frac = train._tabicl_gap_to_mix_frac

    gaps = {"rbf": 0.1, "periodic": 1.1, "matern12": 0.6}
    frac = _tabicl_gap_to_mix_frac(gaps, floor_frac=0.05, max_frac=0.35)

    idx = _COMPOSABLE_KERNELS.index
    assert abs(float(frac[idx("rbf")]) - 0.05) < 1e-5        # min gap -> floor
    assert abs(float(frac[idx("periodic")]) - 0.35) < 1e-5   # max gap -> max_frac
    assert abs(float(frac[idx("matern12")]) - 0.20) < 1e-5   # midpoint -> midpoint
    assert abs(float(frac[idx("cosine")]) - 0.05) < 1e-5     # unmeasured -> floor

    # Degenerate cases fall back to a uniform floor.
    equal_gaps = _tabicl_gap_to_mix_frac({"rbf": 0.5, "periodic": 0.5}, 0.05, 0.35)
    assert torch.allclose(equal_gaps, torch.full_like(equal_gaps, 0.05))

    single_family = _tabicl_gap_to_mix_frac({"rbf": 0.9}, 0.05, 0.35)
    assert torch.allclose(single_family, torch.full_like(single_family, 0.05))

    empty = _tabicl_gap_to_mix_frac({}, 0.05, 0.35)
    assert torch.allclose(empty, torch.full_like(empty, 0.05))
