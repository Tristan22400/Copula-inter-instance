"""
test_pit_batched.py — Tests for pit.py::run_pit_batched/
run_pit_calib_split_batched and their wiring into
data_gen.py::_generate_gp_batch_raw / generate_gp_batch (data.z_train_source
= "tabicl"/"tabicl_split" in conf/data/gp_tasks.yaml).

Tests verify:
  1. run_pit_batched(B=1) matches the existing single-episode run_pit exactly.
  2. run_pit_batched(B>1) matches looping run_pit per episode -- the whole
     point of the batched version is to fold episodes into TabICL's own
     batch axis instead of a Python loop, so this must be bit-identical.
  3. Passing tabicl_model into _generate_gp_batch_raw overrides z_train only
     -- z_test/log_pdf_test and every other field stay exactly the oracle
     values, and the override actually changes z_train's values (not a
     silently inert no-op).
  4. run_pit_calib_split_batched is exactly run_pit_batched's test-side (part
     A) computation with the context/query roles renamed -- calling it with
     (query=X_test-role, calib=X_train-role) must reproduce run_pit_batched's
     z_test bit-for-bit.
  5. Passing tabicl_split_calib_frac > 0 into _generate_gp_batch_raw overrides
     z_train only, same guarantee as tabicl_k_folds's override in (3), and
     never perturbs n_train.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from data_gen import _generate_gp_batch_raw
from pit import run_pit, run_pit_batched, run_pit_calib_split_batched


class RowIndependentFakeTabICL(nn.Module):
    """A FakeTabICL whose output for one (episode, target-dim) row depends
    ONLY on that row's own (X, y) values -- unlike tests/test_tabicl_z_
    diagnostic.py's FakeTabICL, which seeds off X.sum() over the WHOLE
    batch axis (fine for that file's single-call-per-episode usage, but
    wrong here: it would make a combined B*d-batched forward call produce
    different results than B separate single-episode calls, which is
    exactly the equivalence these tests check)."""

    def __init__(self, q: int = 3):
        super().__init__()
        self.q = q

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch, T, _ = X.shape
        P = y.shape[1]
        n = T - P
        out = torch.empty(batch, n, self.q)
        for i in range(batch):
            seed = int((X[i].sum() * 1000 + y[i].sum() * 7).item() * 1000) % (2**31)
            g = torch.Generator().manual_seed(seed)
            out[i] = torch.randn(n, self.q, generator=g)
        return out

    def quantile_dist(self, logits_flat: torch.Tensor):
        loc = logits_flat[:, 0]
        scale = torch.nn.functional.softplus(logits_flat[:, 1]) + 1e-3
        return torch.distributions.Normal(loc, scale)


def test_run_pit_batched_b1_matches_run_pit():
    torch.manual_seed(0)
    tabicl = RowIndependentFakeTabICL()
    P, N, p_x, d = 7, 3, 2, 2
    X_train = torch.randn(P, p_x)
    Y_train = torch.randn(P, d)
    X_test = torch.randn(N, p_x)
    Y_test = torch.randn(N, d)

    single = run_pit(tabicl, X_train, Y_train, X_test, Y_test, k_folds=3)
    batched = run_pit_batched(
        tabicl, X_train.unsqueeze(0), Y_train.unsqueeze(0),
        X_test.unsqueeze(0), Y_test.unsqueeze(0), k_folds=3,
    )

    assert torch.allclose(batched["z_train"].squeeze(0), single["z_train"], atol=1e-5)
    assert torch.allclose(batched["z_test"].squeeze(0), single["z_test"], atol=1e-5)
    assert torch.allclose(batched["log_pdf_test"].squeeze(0), single["log_pdf_test"], atol=1e-5)


def test_run_pit_batched_matches_looped_run_pit():
    torch.manual_seed(1)
    tabicl = RowIndependentFakeTabICL()
    B, P, N, p_x, d = 4, 9, 5, 3, 2
    X_train = torch.randn(B, P, p_x)
    Y_train = torch.randn(B, P, d)
    X_test = torch.randn(B, N, p_x)
    Y_test = torch.randn(B, N, d)

    batched = run_pit_batched(tabicl, X_train, Y_train, X_test, Y_test, k_folds=4)

    for b in range(B):
        single = run_pit(tabicl, X_train[b], Y_train[b], X_test[b], Y_test[b], k_folds=4)
        assert torch.allclose(batched["z_train"][b], single["z_train"], atol=1e-5)
        assert torch.allclose(batched["z_test"][b], single["z_test"], atol=1e-5)
        assert torch.allclose(batched["log_pdf_test"][b], single["log_pdf_test"], atol=1e-5)


def test_run_pit_calib_split_batched_matches_run_pit_batched_test_side():
    torch.manual_seed(2)
    tabicl = RowIndependentFakeTabICL()
    B, P_C, P_Q, p_x, d = 3, 6, 4, 2, 2
    X_calib = torch.randn(B, P_C, p_x)
    Y_calib = torch.randn(B, P_C, d)
    X_query = torch.randn(B, P_Q, p_x)
    Y_query = torch.randn(B, P_Q, d)

    # run_pit_batched's part A (test-side PIT) already scores its X_test
    # against the full X_train context in one forward pass -- with the
    # context/query roles renamed (X_train -> X_calib, X_test -> X_query),
    # that is exactly what run_pit_calib_split_batched computes. k_folds is
    # irrelevant here (only used for run_pit_batched's train-side, K-fold
    # PIT), so any valid value works.
    reference = run_pit_batched(tabicl, X_calib, Y_calib, X_query, Y_query, k_folds=3)
    split = run_pit_calib_split_batched(tabicl, X_query, Y_query, X_calib, Y_calib)

    assert split["z_train"].shape == (B, P_Q, d)
    assert torch.allclose(split["z_train"], reference["z_test"], atol=1e-6)


def test_run_pit_calib_split_batched_finite():
    torch.manual_seed(3)
    tabicl = RowIndependentFakeTabICL()
    B, P_C, P_Q, p_x, d = 2, 5, 7, 3, 1
    X_calib = torch.randn(B, P_C, p_x)
    Y_calib = torch.randn(B, P_C, d)
    X_query = torch.randn(B, P_Q, p_x)
    Y_query = torch.randn(B, P_Q, d)

    out = run_pit_calib_split_batched(tabicl, X_query, Y_query, X_calib, Y_calib)
    assert torch.isfinite(out["z_train"]).all()


def test_generate_gp_batch_raw_tabicl_split_z_train_override(small_cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.seed = 42

    tabicl = RowIndependentFakeTabICL()

    # Control uses the SAME tabicl_split_calib_frac (hence the same T = P +
    # P_C + N, and the same per-episode x_norm normalisation, which is
    # computed jointly over all T points -- see _generate_gp_batch_raw) but
    # tabicl_model=None, so the override branch never fires and z_train stays
    # the analytic LOO residual. This isolates the override's effect: unlike
    # comparing against a tabicl_split_calib_frac=0 baseline (a different T,
    # hence a different shared normalisation constant -- NOT a byte-for-byte
    # comparable episode), every other field must now match exactly, since
    # x_norm/K_all/oracle only depend on P_C's value, never on tabicl_model.
    analytic = _generate_gp_batch_raw(cfg, B=6, device="cpu", tabicl_split_calib_frac=1.0)
    with_split = _generate_gp_batch_raw(
        cfg, B=6, device="cpu", tabicl_model=tabicl, tabicl_split_calib_frac=1.0,
    )

    assert len(analytic) == len(with_split)
    for ep_a, ep_t in zip(analytic, with_split):
        assert ep_a["z_train"].shape == ep_t["z_train"].shape
        assert ep_t["n_train"] == ep_a["n_train"]
        # The override must actually change z_train's values...
        assert not torch.allclose(ep_a["z_train"], ep_t["z_train"])
        # ...but leave everything else -- the test-side oracle fields in
        # particular -- exactly as the analytic pipeline computed them (the
        # calibration-only pool must never perturb x_norm_train/y_train or
        # the test-side oracle, only feed the one-pass PIT call for z_train).
        for key in ("x_norm_train", "x_norm_test", "y_train", "y_test",
                    "z_test", "log_pdf_test", "R_star", "Sigma_star",
                    "mu_star", "sigma_star"):
            assert torch.allclose(ep_a[key], ep_t[key], atol=1e-6), key


def test_generate_gp_batch_raw_tabicl_split_calib_frac_can_exceed_one(small_cfg):
    """z_train_split_calib_frac > 1.0 (calibration pool bigger than the
    training set) must work -- not capped anywhere in the pipeline."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.seed = 7

    tabicl = RowIndependentFakeTabICL()
    episodes = _generate_gp_batch_raw(
        cfg, B=4, device="cpu", tabicl_model=tabicl, tabicl_split_calib_frac=2.5,
    )
    assert len(episodes) > 0
    for ep in episodes:
        assert torch.isfinite(ep["z_train"]).all()
        assert ep["z_train"].shape == ep["y_train"].shape


def test_generate_gp_batch_raw_tabicl_split_calib_frac_zero_is_noop(small_cfg):
    """tabicl_split_calib_frac=0.0 (the default) must fall back to the
    K-fold path when tabicl_model is given, not silently do nothing."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.seed = 11

    tabicl = RowIndependentFakeTabICL()
    kfold = _generate_gp_batch_raw(
        cfg, B=4, device="cpu", tabicl_model=tabicl, tabicl_k_folds=3,
        tabicl_split_calib_frac=0.0,
    )
    cfg.seed = 11
    kfold_again = _generate_gp_batch_raw(
        cfg, B=4, device="cpu", tabicl_model=tabicl, tabicl_k_folds=3,
    )
    assert len(kfold) == len(kfold_again)
    for ep_a, ep_b in zip(kfold, kfold_again):
        assert torch.allclose(ep_a["z_train"], ep_b["z_train"], atol=1e-6)


def test_generate_gp_batch_raw_tabicl_z_train_override(small_cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.seed = 42

    tabicl = RowIndependentFakeTabICL()

    analytic = _generate_gp_batch_raw(cfg, B=6, device="cpu")
    with_tabicl = _generate_gp_batch_raw(
        cfg, B=6, device="cpu", tabicl_model=tabicl, tabicl_k_folds=3,
    )

    assert len(analytic) == len(with_tabicl)
    for ep_a, ep_t in zip(analytic, with_tabicl):
        assert ep_a["z_train"].shape == ep_t["z_train"].shape
        # The override must actually change z_train's values...
        assert not torch.allclose(ep_a["z_train"], ep_t["z_train"])
        # ...but leave everything else -- the test-side oracle fields in
        # particular -- exactly as the analytic pipeline computed them.
        for key in ("x_norm_train", "x_norm_test", "y_train", "y_test",
                    "z_test", "log_pdf_test", "R_star", "Sigma_star",
                    "mu_star", "sigma_star"):
            assert torch.allclose(ep_a[key], ep_t[key], atol=1e-6), key


# ---------------------------------------------------------------------------
# data.z_train_matched_test (see conf/data/gp_tasks.yaml): when True, the
# plain "tabicl" K-fold branch above also overrides z_test/log_pdf_test with
# TabICL's own PIT at the real x_norm_test/y_test, instead of leaving them
# at the oracle values the "leave everything else exactly as the analytic
# pipeline computed them" assertion above checks for. Root-caused
# 2026-08-24: training against the oracle z_test while conditioning on
# TabICL's noisy z_train teaches an overconfident Sigma that scores badly
# (positive copula NLL) once actually validated against TabICL's own,
# noisier z_test -- exactly the train/val marginal mismatch this flag
# closes, matching how era5_live_dataset.py already has to PIT both z_train
# AND z_test through TabICL (no oracle exists for real ERA5 data).
# ---------------------------------------------------------------------------


def test_generate_gp_batch_raw_tabicl_matched_test_overrides_z_test(small_cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.data.z_train_matched_test = True
    cfg.seed = 43

    tabicl = RowIndependentFakeTabICL()

    analytic = _generate_gp_batch_raw(cfg, B=6, device="cpu")
    matched = _generate_gp_batch_raw(
        cfg, B=6, device="cpu", tabicl_model=tabicl, tabicl_k_folds=3,
    )

    assert len(analytic) == len(matched)
    for ep_a, ep_m in zip(analytic, matched):
        # z_train changes, same as the plain override...
        assert not torch.allclose(ep_a["z_train"], ep_m["z_train"])
        # ...but now z_test/log_pdf_test change too -- the whole point of
        # this flag: the training target now comes from TabICL's own PIT at
        # the real test points, not the oracle.
        assert ep_m["z_test"].shape == ep_a["z_test"].shape
        assert not torch.allclose(ep_a["z_test"], ep_m["z_test"])
        assert not torch.allclose(ep_a["log_pdf_test"], ep_m["log_pdf_test"])
        assert torch.isfinite(ep_m["z_test"]).all()
        assert torch.isfinite(ep_m["log_pdf_test"]).all()
        # Every field that's purely a property of the underlying GP episode
        # (not the PIT target) must still be untouched.
        for key in ("x_norm_train", "x_norm_test", "y_train", "y_test",
                    "R_star", "Sigma_star", "mu_star", "sigma_star"):
            assert torch.allclose(ep_a[key], ep_m[key], atol=1e-6), key


def test_generate_gp_batch_raw_tabicl_matched_test_default_false_is_legacy_behaviour(small_cfg):
    """z_train_matched_test defaults to False when unset -- must reproduce
    the pre-existing "only z_train changes" override exactly."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.seed = 44

    tabicl = RowIndependentFakeTabICL()

    unset = _generate_gp_batch_raw(cfg, B=4, device="cpu", tabicl_model=tabicl, tabicl_k_folds=3)
    cfg.seed = 44
    cfg.data.z_train_matched_test = False
    explicit_false = _generate_gp_batch_raw(cfg, B=4, device="cpu", tabicl_model=tabicl, tabicl_k_folds=3)

    assert len(unset) == len(explicit_false)
    for ep_u, ep_f in zip(unset, explicit_false):
        for key in ("z_train", "z_test", "log_pdf_test"):
            assert torch.allclose(ep_u[key], ep_f[key], atol=1e-6), key


def test_generate_gp_batch_raw_tabicl_matched_test_matches_direct_run_pit_batched(small_cfg):
    """The matched-test override's z_test/log_pdf_test must equal calling
    run_pit_batched directly on the episode's own (oracle-computed)
    x_norm_train/y_train/x_norm_test/y_test with the same train-only
    y_mean/y_std scaling and Jacobian correction -- pins down the exact
    formula, not just "it changed something."."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.data.z_train_matched_test = True
    cfg.seed = 45

    tabicl = RowIndependentFakeTabICL()
    matched = _generate_gp_batch_raw(cfg, B=3, device="cpu", tabicl_model=tabicl, tabicl_k_folds=3)
    assert len(matched) > 0

    for ep in matched:
        x_train = ep["x_norm_train"].unsqueeze(0)
        x_test = ep["x_norm_test"].unsqueeze(0)
        y_train = ep["y_train"].unsqueeze(0)
        y_test = ep["y_test"].unsqueeze(0)

        y_mean = y_train.mean(dim=1, keepdim=True)
        y_std = y_train.std(dim=1, keepdim=True).clamp(min=1e-8)
        y_train_scaled = ((y_train - y_mean) / y_std).unsqueeze(-1)
        y_test_scaled = ((y_test - y_mean) / y_std).unsqueeze(-1)

        expected = run_pit_batched(
            tabicl, x_train, y_train_scaled, x_test, y_test_scaled, k_folds=3,
        )
        expected_log_pdf = expected["log_pdf_test"].squeeze(-1) - y_std.log()

        assert torch.allclose(ep["z_test"], expected["z_test"].squeeze(0).squeeze(-1), atol=1e-5)
        assert torch.allclose(ep["log_pdf_test"], expected_log_pdf.squeeze(0), atol=1e-5)


def test_generate_gp_batch_raw_tabicl_matched_test_noop_without_tabicl_model(small_cfg):
    """z_train_matched_test=True with tabicl_model=None must not do
    anything -- apply_tabicl still gates on tabicl_model being given
    first, same as the plain override."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.data.z_train_matched_test = True
    cfg.seed = 46

    plain = _generate_gp_batch_raw(cfg, B=4, device="cpu")
    cfg.seed = 46
    matched_no_model = _generate_gp_batch_raw(cfg, B=4, device="cpu")

    assert len(plain) == len(matched_no_model)
    for ep_p, ep_m in zip(plain, matched_no_model):
        for key in ("z_train", "z_test", "log_pdf_test"):
            assert torch.allclose(ep_p[key], ep_m[key], atol=1e-6), key


def test_generate_gp_batch_raw_tabicl_matched_test_ignored_for_tabicl_split(small_cfg):
    """z_train_matched_test only affects the plain "tabicl" K-fold branch --
    tabicl_split_calib_frac > 0 must keep the legacy oracle z_test/
    log_pdf_test regardless of this flag."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.data.z_train_matched_test = True
    cfg.seed = 47

    tabicl = RowIndependentFakeTabICL()
    analytic = _generate_gp_batch_raw(cfg, B=4, device="cpu", tabicl_split_calib_frac=1.0)
    cfg.seed = 47
    split = _generate_gp_batch_raw(
        cfg, B=4, device="cpu", tabicl_model=tabicl, tabicl_split_calib_frac=1.0,
    )

    assert len(analytic) == len(split)
    for ep_a, ep_s in zip(analytic, split):
        assert not torch.allclose(ep_a["z_train"], ep_s["z_train"])
        for key in ("z_test", "log_pdf_test"):
            assert torch.allclose(ep_a[key], ep_s[key], atol=1e-6), key
