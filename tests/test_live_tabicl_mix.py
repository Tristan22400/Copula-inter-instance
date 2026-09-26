"""Tests for the per-kernel-family TabICL z_train mix (data.z_train_tabicl_mix_*).

1. tabicl_mix_weights=None always applies the override.
2. All-zero weights equal tabicl_model=None.
3. All-one weights equal the always-on override exactly.
4. A single-family weight gives the configured hit rate.
5. Composite kernels use their components' maximum weight.
6. corrupt_z_train is skipped only for mix-sourced z_train.
7. _tabicl_gap_to_mix_frac: linear between floor and max, floor for
   unmeasured families and for degenerate gaps.
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf
from test_pit_batched import RowIndependentFakeTabICL

from copula_inter.data_gen import _COMPOSABLE_KERNELS, _generate_gp_batch_raw, _tabicl_mix_prob_for_kernel


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
    # Maximum component weight (periodic 0.8), not the mean.
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
        cfg,
        B=8,
        device="cpu",
        tabicl_model=tabicl,
        tabicl_k_folds=3,
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
        cfg,
        B=8,
        device="cpu",
        tabicl_model=tabicl,
        tabicl_k_folds=3,
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
            cfg,
            B=1,
            device="cpu",
            tabicl_model=tabicl,
            tabicl_k_folds=3,
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

    # Mix hit: TabICL z_train, not corrupted.
    cfg.data.z_train_corruption_enabled = False
    uncorrupted_tabicl = _generate_gp_batch_raw(
        cfg,
        B=6,
        device="cpu",
        tabicl_model=tabicl,
        tabicl_k_folds=3,
        tabicl_mix_weights=one_w,
    )
    cfg.data.z_train_corruption_enabled = True
    mix_hit_should_skip_corruption = _generate_gp_batch_raw(
        cfg,
        B=6,
        device="cpu",
        tabicl_model=tabicl,
        tabicl_k_folds=3,
        tabicl_mix_weights=one_w,
    )
    for ep_u, ep_h in zip(uncorrupted_tabicl, mix_hit_should_skip_corruption):
        assert torch.allclose(ep_u["z_train"], ep_h["z_train"], atol=1e-6)

    # Mix miss: analytic z_train, corrupted.
    cfg.data.z_train_corruption_enabled = False
    uncorrupted_analytic = _generate_gp_batch_raw(cfg, B=6, device="cpu")
    cfg.data.z_train_corruption_enabled = True
    mix_miss_should_corrupt = _generate_gp_batch_raw(
        cfg,
        B=6,
        device="cpu",
        tabicl_model=tabicl,
        tabicl_k_folds=3,
        tabicl_mix_weights=zero_w,
    )
    any_diff = any(
        not torch.allclose(ep_u["z_train"], ep_m["z_train"], atol=1e-6)
        for ep_u, ep_m in zip(uncorrupted_analytic, mix_miss_should_corrupt)
    )
    assert any_diff

    # Always-on path (weights None): corrupted.
    legacy_corrupted = _generate_gp_batch_raw(
        cfg,
        B=6,
        device="cpu",
        tabicl_model=tabicl,
        tabicl_k_folds=3,
    )
    any_diff_legacy = any(
        not torch.allclose(ep_u["z_train"], ep_l["z_train"], atol=1e-6)
        for ep_u, ep_l in zip(uncorrupted_tabicl, legacy_corrupted)
    )
    assert any_diff_legacy


def _import_train():
    from copula_inter import adaptive_sampling as train

    return train


def test_compute_tabicl_z_train_gap_runs_on_declared_device(small_cfg):
    """_compute_tabicl_z_train_gap runs both paired calls on the given device (CPU here) with finite gaps."""
    train = _import_train()
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.baselines = {
        "synth_n_episodes": 4,
        "synth_seed": 999,
        "probe_P_min": 5,
        "probe_P_max": 10,
        "probe_N_min": 3,
        "probe_N_max": 6,
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
    assert abs(float(frac[idx("rbf")]) - 0.05) < 1e-5  # min gap -> floor
    assert abs(float(frac[idx("periodic")]) - 0.35) < 1e-5  # max gap -> max_frac
    assert abs(float(frac[idx("matern12")]) - 0.20) < 1e-5  # midpoint -> midpoint
    assert abs(float(frac[idx("cosine")]) - 0.05) < 1e-5  # unmeasured -> floor

    # Degenerate cases fall back to a uniform floor.
    equal_gaps = _tabicl_gap_to_mix_frac({"rbf": 0.5, "periodic": 0.5}, 0.05, 0.35)
    assert torch.allclose(equal_gaps, torch.full_like(equal_gaps, 0.05))

    single_family = _tabicl_gap_to_mix_frac({"rbf": 0.9}, 0.05, 0.35)
    assert torch.allclose(single_family, torch.full_like(single_family, 0.05))

    empty = _tabicl_gap_to_mix_frac({}, 0.05, 0.35)
    assert torch.allclose(empty, torch.full_like(empty, 0.05))
