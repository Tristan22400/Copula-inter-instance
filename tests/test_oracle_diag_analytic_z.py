"""test_oracle_diag_analytic_z.py — pins validate()'s ``oracle_diag/*`` block to
the EXACT-GP z-space, whatever z-space the val batches themselves are in.

The defect this guards. ``data.z_train_source=tabicl`` is the production
default and the regime the model actually deploys in. In that regime
``live_dataset.build_fixed_live_val_batches`` hands the frozen TabICL to
``generate_gp_batch``, and ``data_gen``'s TabICL branch overwrites ``z_train``
**and** ``z_test`` / ``log_pdf_test`` with TabICL's K-fold PIT. But
``validate()`` compares the model's copula NLL, measured on those TabICL z,
against ``pit.gp_analytical_posterior``'s ceiling, which is in the exact
GP-POSTERIOR z-space. Two Sklar splits taken at DIFFERENT marginals are not
comparable term by term -- only a Y-space total is (see
``eval/metrics/joint_nll.py``'s module docstring) -- so ``oracle_diag/gap_nll``
was not a bound and ``oracle_diag/corr_*`` were raw-correlation comparisons
across a distorted z-space.

``train._build_analytic_val_z`` restores the missing operand, and ``validate()``
re-conditions the model on the analytic ``z_train`` for that scoring pass (the
oracle question -- "how far from Bayes-optimal when handed the ORACLE marginal"
-- needs oracle z on both sides, not just on the scoring side).

The tests below stand in for TabICL rather than loading it: TabICL needs a GPU
and a checkpoint, and nothing here depends on *which* other marginal the batch
carries -- only that it is a different one. Re-standardizing by the GP PRIOR
gives exactly that, and is the historically realistic wrong marginal besides
(it is what ``gp_analytical_pit`` itself used to emit).

No GPU, no network, no TabICL checkpoint.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from data_gen import generate_gp_batch
from dataset import collate_fn
from model import build_copula_transformer
from pit import gp_analytical_pit
from train import _build_analytic_val_z, validate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _cfg(small_cfg, small_model_cfg):
    """small_cfg's data block + small_model_cfg's real (scratch) backbone."""
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
    """Stand-in for data_gen's TabICL branch: replace each episode's PIT with
    one taken at a DIFFERENT marginal (the GP prior).

    Mutates in place and returns the episodes, exactly as the TabICL branch
    does -- so the collated batch and the cached episode dicts disagree with
    ``gp_analytical_pit(ep)`` in precisely the way the production tabicl path
    makes them disagree.
    """
    for ep in episodes:
        sig = ep["sigma_star"].double().clamp_min(1e-8)
        z = (ep["y_test"].double() - ep["mu_star"].double()) / sig
        ep["z_test"] = z.float()
        ep["log_pdf_test"] = (
            -0.5 * math.log(2.0 * math.pi) - sig.log() - 0.5 * z ** 2
        ).float()
        # z_train too: the TabICL branch replaces the conditioning input as
        # well, which is what makes the un-fixed oracle_diag/ numbers score a
        # model conditioned on one z-space against a ceiling in another.
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


# ---------------------------------------------------------------------------
# 1. The cache itself is the exact analytic PIT, recomputed from the episode
# ---------------------------------------------------------------------------
def test_build_analytic_val_z_recovers_the_exact_gp_pit(small_cfg, small_model_cfg):
    """_build_analytic_val_z ignores whatever PIT the batch carries and
    recomputes gp_analytical_pit from the episode's kernel metadata."""
    cfg = _cfg(small_cfg, small_model_cfg)
    eps = _episodes(cfg, b=4)
    truth = [gp_analytical_pit(ep) for ep in eps]     # BEFORE the overwrite
    _reprior_standardize(eps)
    batches = [collate_fn(eps)]

    cache = _build_analytic_val_z(batches, {0: eps}, "cpu")
    assert set(cache) == {0}
    for b, want in enumerate(truth):
        for key in ("z_train", "z_test", "log_pdf_test"):
            got = cache[0][key][b, : want[key].shape[0]]
            assert torch.allclose(got, want[key].float(), atol=1e-5), (key, b)

    # And it really differs from what the batch carries -- otherwise this test
    # would pass trivially even if the cache were just batch[...] passed through.
    assert not torch.allclose(
        cache[0]["z_test"], batches[0]["z_test"].float(), atol=1e-3
    )


# ---------------------------------------------------------------------------
# 2. validate()'s oracle_diag marginal is the ANALYTIC one, not the batch's
# ---------------------------------------------------------------------------
def test_oracle_diag_marginal_follows_the_analytic_pit(small_cfg, small_model_cfg):
    """oracle_diag/(total - copula) == -mean(analytic log_pdf_test), NOT the
    batch's own marginal.

    This is the load-bearing assertion: the marginal term is what identifies
    which z-space the Sklar split was taken in, and it is the term that must
    cancel against y_nll_oracle_posterior_marginal for gap_nll to be a gap.
    """
    cfg = _cfg(small_cfg, small_model_cfg)
    eps = _episodes(cfg, b=4, seed=1)
    analytic_marginal = float(
        np.mean([-float(gp_analytical_pit(ep)["log_pdf_test"].double().mean()) for ep in eps])
    )
    _reprior_standardize(eps)
    batch_marginal = float(
        np.mean([-float(ep["log_pdf_test"].double().mean()) for ep in eps])
    )
    assert abs(analytic_marginal - batch_marginal) > 1e-2, (
        "the stand-in did not actually change the marginal; the rest of this "
        "test would then be vacuous"
    )

    batches = [collate_fn(eps)]
    model = build_copula_transformer(cfg)
    cache = _build_analytic_val_z(batches, {0: eps}, "cpu")
    m = _run_validate(cfg, model, batches, [eps], cache)

    got = m["oracle_diag/total_nll"] - m["oracle_diag/copula_nll"]
    assert got == pytest.approx(analytic_marginal, abs=1e-3)
    assert abs(got - batch_marginal) > 1e-2


# ---------------------------------------------------------------------------
# 3. The marginal cancels, so gap_nll IS the copula gap
# ---------------------------------------------------------------------------
def test_copula_gap_equals_gap_nll_and_marginal_gap_vanishes(small_cfg, small_model_cfg):
    """With both operands in the GP-posterior z-space the marginal terms are
    the same number, so gap_nll == copula_gap and marginal_gap == 0.

    Asserted rather than assumed: the identity is what lets the stdout VAL
    line report a single gap, and it silently stops holding the moment the
    oracle_diag routing regresses to the batch's own z.
    """
    cfg = _cfg(small_cfg, small_model_cfg)
    eps = _reprior_standardize(_episodes(cfg, b=4, seed=2))
    batches = [collate_fn(eps)]
    model = build_copula_transformer(cfg)
    cache = _build_analytic_val_z(batches, {0: eps}, "cpu")
    m = _run_validate(cfg, model, batches, [eps], cache)

    assert m["oracle_diag/marginal_gap"] == pytest.approx(0.0, abs=1e-3)
    assert m["oracle_diag/copula_gap"] == pytest.approx(m["oracle_diag/gap_nll"], abs=1e-4)
    # copula_headroom is the whole Bayes-optimal copula reward: positive
    # (correlation always helps against independence on a GP episode) and the
    # scale the gap has to be read against.
    assert m["oracle_diag/copula_headroom"] > 0.0
    assert m["oracle_diag/copula_headroom"] == pytest.approx(
        -m["y_nll_oracle_posterior_copula"], abs=1e-9
    )


# ---------------------------------------------------------------------------
# 4. Without the cache the mismatch is visible — i.e. the fix does something
# ---------------------------------------------------------------------------
def test_without_the_cache_the_marginal_gap_is_nonzero(small_cfg, small_model_cfg):
    """The regression direction. Passing analytic_val_z=None reproduces the
    pre-fix behaviour (score against the batch's own z), and marginal_gap --
    the sanity term that must be ~0 -- comes out materially non-zero. If this
    ever passes at ~0 the stand-in has stopped standing in for anything.
    """
    cfg = _cfg(small_cfg, small_model_cfg)
    eps = _reprior_standardize(_episodes(cfg, b=4, seed=3))
    batches = [collate_fn(eps)]
    model = build_copula_transformer(cfg)
    m = _run_validate(cfg, model, batches, [eps], None)
    assert abs(m["oracle_diag/marginal_gap"]) > 1e-2


# ---------------------------------------------------------------------------
# 5. corr_kl is emitted, finite, and >= 0
# ---------------------------------------------------------------------------
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
