"""Tests that validate()'s oracle_diag/* metrics use the exact-GP z-space even when the val batches carry another marginal's PIT.

The other marginal is simulated by re-standardizing episodes with the GP
prior (no TabICL needed).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from copula_inter.data_gen import generate_gp_batch
from copula_inter.dataset import collate_fn
from copula_inter.model import build_copula_transformer
from copula_inter.pit import gp_analytical_pit
from copula_inter.probe_batches import _build_analytic_val_z
from copula_inter.validation import validate


def _cfg(small_cfg, small_model_cfg):
    """small_cfg's data block with small_model_cfg's scratch backbone."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.model = OmegaConf.create(OmegaConf.to_container(small_model_cfg.model, resolve=True))
    cfg.tabicl = OmegaConf.create(OmegaConf.to_container(small_model_cfg.tabicl, resolve=True))
    cfg.data.P_min = cfg.data.P_max = 12
    cfg.data.N_min = cfg.data.N_max = 24
    cfg.data.kernel = "rbf"
    return cfg


def _episodes(cfg, b=4, seed=0):
    torch.manual_seed(seed)
    return generate_gp_batch(cfg, b, "cpu", return_kernel_metadata=True)


def _reprior_standardize(episodes):
    """Replace each episode's PIT with one under the GP prior (in place), like data_gen's TabICL branch."""
    for ep in episodes:
        sig = ep["sigma_star"].double().clamp_min(1e-8)
        z = (ep["y_test"].double() - ep["mu_star"].double()) / sig
        ep["z_test"] = z.float()
        ep["log_pdf_test"] = (-0.5 * math.log(2.0 * math.pi) - sig.log() - 0.5 * z**2).float()
        # Replace z_train too, as the TabICL branch does.
        ep["z_train"] = (ep["z_train"].double() * 0.5).float()
    return episodes


def _run_validate(cfg, model, batches, episodes_by_batch, analytic_val_z):
    return validate(
        model,
        batches,
        cfg,
        "cpu",
        step=0,
        do_plot=False,
        val_episodes_meta=dict(enumerate(episodes_by_batch)),
        analytic_val_z=analytic_val_z,
    )[0]


def test_build_analytic_val_z_recovers_the_exact_gp_pit(small_cfg, small_model_cfg):
    """_build_analytic_val_z recomputes gp_analytical_pit, ignoring the batch's PIT."""
    cfg = _cfg(small_cfg, small_model_cfg)
    eps = _episodes(cfg, b=4)
    truth = [gp_analytical_pit(ep) for ep in eps]  # BEFORE the overwrite
    _reprior_standardize(eps)
    batches = [collate_fn(eps)]

    cache = _build_analytic_val_z(batches, {0: eps}, "cpu")
    assert set(cache) == {0}
    for b, want in enumerate(truth):
        for key in ("z_train", "z_test", "log_pdf_test"):
            got = cache[0][key][b, : want[key].shape[0]]
            assert torch.allclose(got, want[key].float(), atol=1e-5), (key, b)

    # The cache differs from what the batch carries.
    assert not torch.allclose(cache[0]["z_test"], batches[0]["z_test"].float(), atol=1e-3)


def test_oracle_diag_marginal_follows_the_analytic_pit(small_cfg, small_model_cfg):
    """oracle_diag total - copula equals -mean(analytic log_pdf_test), not the batch's marginal."""
    cfg = _cfg(small_cfg, small_model_cfg)
    eps = _episodes(cfg, b=4, seed=1)
    analytic_marginal = float(np.mean([-float(gp_analytical_pit(ep)["log_pdf_test"].double().mean()) for ep in eps]))
    _reprior_standardize(eps)
    batch_marginal = float(np.mean([-float(ep["log_pdf_test"].double().mean()) for ep in eps]))
    assert abs(analytic_marginal - batch_marginal) > 1e-2, (
        "the stand-in did not actually change the marginal; the rest of this test would then be vacuous"
    )

    batches = [collate_fn(eps)]
    model = build_copula_transformer(cfg)
    cache = _build_analytic_val_z(batches, {0: eps}, "cpu")
    m = _run_validate(cfg, model, batches, [eps], cache)

    got = m["oracle_diag/total_nll"] - m["oracle_diag/copula_nll"]
    assert got == pytest.approx(analytic_marginal, abs=1e-3)
    assert abs(got - batch_marginal) > 1e-2


def test_copula_gap_equals_gap_nll_and_marginal_gap_vanishes(small_cfg, small_model_cfg):
    """With both operands in the posterior z-space, gap_nll == copula_gap and marginal_gap == 0."""
    cfg = _cfg(small_cfg, small_model_cfg)
    eps = _reprior_standardize(_episodes(cfg, b=4, seed=2))
    batches = [collate_fn(eps)]
    model = build_copula_transformer(cfg)
    cache = _build_analytic_val_z(batches, {0: eps}, "cpu")
    m = _run_validate(cfg, model, batches, [eps], cache)

    assert m["oracle_diag/marginal_gap"] == pytest.approx(0.0, abs=1e-3)
    assert m["oracle_diag/copula_gap"] == pytest.approx(m["oracle_diag/gap_nll"], abs=1e-4)
    # copula_headroom is positive.
    assert m["oracle_diag/copula_headroom"] > 0.0
    assert m["oracle_diag/copula_headroom"] == pytest.approx(-m["y_nll_oracle_posterior_copula"], abs=1e-9)


def test_without_the_cache_the_marginal_gap_is_nonzero(small_cfg, small_model_cfg):
    """With analytic_val_z=None, marginal_gap is clearly non-zero."""
    cfg = _cfg(small_cfg, small_model_cfg)
    eps = _reprior_standardize(_episodes(cfg, b=4, seed=3))
    batches = [collate_fn(eps)]
    model = build_copula_transformer(cfg)
    m = _run_validate(cfg, model, batches, [eps], None)
    assert abs(m["oracle_diag/marginal_gap"]) > 1e-2


def test_corr_kl_is_emitted_and_nonnegative(small_cfg, small_model_cfg):
    cfg = _cfg(small_cfg, small_model_cfg)
    eps = _reprior_standardize(_episodes(cfg, b=4, seed=4))
    batches = [collate_fn(eps)]
    model = build_copula_transformer(cfg)
    cache = _build_analytic_val_z(batches, {0: eps}, "cpu")
    m = _run_validate(cfg, model, batches, [eps], cache)

    assert m["oracle_diag/corr_kl"] >= 0.0
    assert math.isfinite(m["oracle_diag/corr_kl"])
    assert math.isfinite(m["oracle_diag/corr_kl_p90"])
    assert m["oracle_diag/corr_kl_nonfinite"] == 0.0


def test_kernel_hidden_warp_oracle_diag_marginal_gap_vanishes(small_cfg, small_model_cfg):
    """With kernel_hidden_enabled, marginal_gap == 0 and copula_gap == gap_nll."""
    cfg = _cfg(small_cfg, small_model_cfg)
    cfg.data.kernel_hidden_enabled = True
    cfg.data.kernel_hidden_prob = 1.0
    eps = _episodes(cfg, b=4, seed=5)
    batches = [collate_fn(eps)]
    model = build_copula_transformer(cfg)
    cache = _build_analytic_val_z(batches, {0: eps}, "cpu")
    m = _run_validate(cfg, model, batches, [eps], cache)

    assert m["oracle_diag/marginal_gap"] == pytest.approx(0.0, abs=1e-3)
    assert m["oracle_diag/copula_gap"] == pytest.approx(m["oracle_diag/gap_nll"], abs=1e-4)
    assert m["oracle_diag/copula_headroom"] > 0.0
