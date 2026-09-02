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
  3. Passing tabicl_model into _generate_gp_batch_raw with the plain "tabicl"
     (K-fold) path overrides z_train AND z_test/log_pdf_test (scored against
     TabICL's own PIT at the real x_norm_test/y_test, matching what
     train.py::validate's sim-to-real diagnostic later scores it against --
     see conf/data/gp_tasks.yaml's z_train_source docstring for why, root-
     caused 2026-08-24) -- every other, purely-GP-episode field stays
     exactly the oracle/analytic values, and the override actually changes
     z_train/z_test's values (not a silently inert no-op).
  4. run_pit_calib_split_batched is exactly run_pit_batched's test-side (part
     A) computation with the context/query roles renamed -- calling it with
     (query=X_test-role, calib=X_train-role) must reproduce run_pit_batched's
     z_test bit-for-bit.
  5. Passing tabicl_split_calib_frac > 0 into _generate_gp_batch_raw (the
     "tabicl_split" path) overrides z_train only -- z_test/log_pdf_test stay
     the oracle values, unlike the plain "tabicl" path in (3) -- same
     never-perturbs-n_train guarantee as tabicl_k_folds's override.
  6. run_pit_batched_grad (the Phase-A marginal-finetuning entry point, see
     src/marginal_finetune.py) is numerically identical to run_pit_batched --
     they share one private body precisely so they cannot drift, and this is
     what proves the sharing actually holds.
  7. return_quantiles=True is purely additive: it does not perturb z_train/
     z_test/log_pdf_test, and the quantiles it returns are the same tensors
     the returned CDF values were computed from.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from data_gen import _generate_gp_batch_raw
from pit import (
    _run_pit_batched_impl,
    run_pit,
    run_pit_batched,
    run_pit_batched_grad,
    run_pit_calib_split_batched,
)


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
    """Plain "tabicl" (K-fold) override: z_train AND z_test/log_pdf_test are
    replaced with TabICL's own PIT (at the real x_norm_test/y_test), while
    every purely-GP-episode field stays exactly the analytic/oracle values.

    Root-caused 2026-08-24: an earlier version of this override left z_test/
    log_pdf_test at the oracle values (only z_train changed) -- training
    against that clean target while conditioning on noisy TabICL z_train
    taught an overconfident Sigma that scored badly (positive, worse-than-
    independence copula NLL) once train.py::validate's sim-to-real
    diagnostic (val/y_nll_copula) scored that same Sigma against TabICL's
    own noisier z_test instead. See conf/data/gp_tasks.yaml's z_train_source
    docstring for the full writeup and the A/B copula_nano confirmation.
    """
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
        assert ep_a["z_test"].shape == ep_t["z_test"].shape
        # The override must actually change z_train AND z_test/log_pdf_test...
        assert not torch.allclose(ep_a["z_train"], ep_t["z_train"])
        assert not torch.allclose(ep_a["z_test"], ep_t["z_test"])
        assert not torch.allclose(ep_a["log_pdf_test"], ep_t["log_pdf_test"])
        assert torch.isfinite(ep_t["z_test"]).all()
        assert torch.isfinite(ep_t["log_pdf_test"]).all()
        # ...but leave every purely-GP-episode field -- not derived from the
        # PIT target -- exactly as the analytic pipeline computed it.
        for key in ("x_norm_train", "x_norm_test", "y_train", "y_test",
                    "R_star", "Sigma_star", "mu_star", "sigma_star"):
            assert torch.allclose(ep_a[key], ep_t[key], atol=1e-6), key


def test_generate_gp_batch_raw_tabicl_z_test_matches_direct_run_pit_batched(small_cfg):
    """The override's z_test/log_pdf_test must equal calling run_pit_batched
    directly on the episode's own (oracle-computed) x_norm_train/y_train/
    x_norm_test/y_test with the same train-only y_mean/y_std scaling and
    Jacobian correction -- pins down the exact formula, not just "it changed
    something."."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.seed = 45

    tabicl = RowIndependentFakeTabICL()
    episodes = _generate_gp_batch_raw(cfg, B=3, device="cpu", tabicl_model=tabicl, tabicl_k_folds=3)
    assert len(episodes) > 0

    for ep in episodes:
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


def test_generate_gp_batch_raw_tabicl_noop_without_tabicl_model(small_cfg):
    """tabicl_model=None must not touch z_train, z_test, or log_pdf_test --
    apply_tabicl gates the whole override on tabicl_model being given."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.seed = 46

    plain = _generate_gp_batch_raw(cfg, B=4, device="cpu")
    cfg.seed = 46
    plain_again = _generate_gp_batch_raw(cfg, B=4, device="cpu")

    assert len(plain) == len(plain_again)
    for ep_p, ep_a in zip(plain, plain_again):
        for key in ("z_train", "z_test", "log_pdf_test"):
            assert torch.allclose(ep_p[key], ep_a[key], atol=1e-6), key


def test_generate_gp_batch_raw_tabicl_split_keeps_oracle_z_test(small_cfg):
    """Unlike the plain "tabicl" K-fold path above, "tabicl_split"
    (tabicl_split_calib_frac > 0) only overrides z_train -- z_test/
    log_pdf_test stay the oracle values (this is a separate, narrower
    override; see conf/data/gp_tasks.yaml's z_train_source docstring)."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
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


# ---------------------------------------------------------------------------
# 6-7. Grad-enabled variant and the quantile extras
# ---------------------------------------------------------------------------


def _pit_inputs(B=2, P=9, N=4, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(B, P, 3), torch.randn(B, P, 1),
        torch.randn(B, N, 3), torch.randn(B, N, 1),
    )


def test_run_pit_batched_grad_matches_the_no_grad_version():
    """The two entry points exist only to differ in gradient policy. Any
    numerical difference means the shared _run_pit_batched_impl stopped being
    shared -- which would silently let Phase A optimize a slightly different
    PIT than the one deployment runs."""
    tabicl = RowIndependentFakeTabICL()
    Xtr, Ytr, Xte, Yte = _pit_inputs()

    ref = run_pit_batched(tabicl, Xtr, Ytr, Xte, Yte, k_folds=3)
    got = run_pit_batched_grad(tabicl, Xtr, Ytr, Xte, Yte, k_folds=3, return_quantiles=False)

    for key in ("z_train", "z_test", "log_pdf_test"):
        assert torch.allclose(ref[key], got[key], atol=0), key


class GradProbeFakeTabICL(nn.Module):
    """A differentiable fake that records the grad-enabled state and the
    train/eval mode it was called under.

    RowIndependentFakeTabICL above builds its output with a torch.Generator and
    torch.empty, so nothing downstream of it can require grad no matter what
    policy the caller set -- it cannot distinguish the two entry points. This
    one produces its output from an actual Parameter, so requires_grad on the
    result is a real signal.
    """

    def __init__(self, q: int = 3):
        super().__init__()
        self.q = q
        self.w = nn.Parameter(torch.randn(q))
        self.saw_grad_enabled: list[bool] = []
        self.saw_training: list[bool] = []

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.saw_grad_enabled.append(torch.is_grad_enabled())
        self.saw_training.append(self.training)
        batch, T, _ = X.shape
        n = T - y.shape[1]
        base = X[:, -n:, :1].mean(-1, keepdim=True)          # (batch, n, 1)
        return base + self.w.view(1, 1, self.q)

    def quantile_dist(self, logits_flat: torch.Tensor):
        loc = logits_flat[:, 0]
        scale = torch.nn.functional.softplus(logits_flat[:, 1]) + 1e-3
        return torch.distributions.Normal(loc, scale)


def test_run_pit_batched_grad_builds_a_graph_and_the_public_one_does_not():
    """The whole point of the split: run_pit_batched is hard-decorated
    @torch.no_grad() for its (many) inference callers, so a shared flag would
    have silently made every one of them build an autograd graph."""
    Xtr, Ytr, Xte, Yte = _pit_inputs()

    frozen = GradProbeFakeTabICL()
    out_nograd = run_pit_batched(frozen, Xtr, Ytr, Xte, Yte, k_folds=3)
    assert not out_nograd["z_test"].requires_grad
    assert not any(frozen.saw_grad_enabled)

    live = GradProbeFakeTabICL()
    out_grad = run_pit_batched_grad(live, Xtr, Ytr, Xte, Yte, k_folds=3, return_quantiles=False)
    assert out_grad["z_test"].requires_grad
    assert out_grad["log_pdf_test"].requires_grad
    assert all(live.saw_grad_enabled)
    out_grad["log_pdf_test"].mean().backward()
    assert live.w.grad is not None and torch.isfinite(live.w.grad).all()


def test_grad_pit_forces_train_mode_and_restores_it():
    """TabICL's .eval() routes into _inference_forward/InferenceManager, whose
    own fp16 autocast produces NaN for this codebase's inputs -- the documented
    reason train.py::validate() never calls model.eval(). The grad path must
    therefore force train mode, and must put it back so it is not a hidden
    side effect on the caller's module."""
    Xtr, Ytr, Xte, Yte = _pit_inputs()
    probe = GradProbeFakeTabICL()
    probe.eval()

    run_pit_batched_grad(probe, Xtr, Ytr, Xte, Yte, k_folds=3, return_quantiles=False)

    assert all(probe.saw_training), "grad path must call the module in train mode"
    assert not probe.training, "grad path must restore the caller's original mode"


def test_return_quantiles_is_additive_and_self_consistent():
    tabicl = RowIndependentFakeTabICL(q=7)
    Xtr, Ytr, Xte, Yte = _pit_inputs(seed=3)

    plain = run_pit_batched(tabicl, Xtr, Ytr, Xte, Yte, k_folds=3)
    extra = run_pit_batched(tabicl, Xtr, Ytr, Xte, Yte, k_folds=3, return_quantiles=True)

    for key in ("z_train", "z_test", "log_pdf_test"):
        assert torch.allclose(plain[key], extra[key], atol=0), key

    B, P, _ = Ytr.shape
    N = Yte.shape[1]
    assert extra["q_train"].shape == (B, P, 1, 7)
    assert extra["q_test"].shape == (B, N, 1, 7)
    # u_test is the CDF the returned quantiles imply, so probit(u) must be the
    # z_test that came back alongside them.
    from pit import _probit

    assert torch.allclose(_probit(extra["u_test"], 1e-6), extra["z_test"], atol=0)
    assert torch.allclose(_probit(extra["u_train"], 1e-6), extra["z_train"], atol=0)


def test_fold_subset_scores_only_the_requested_folds():
    """Phase A never needs a complete z_train, so it subsets the fold rotation
    to cut the per-step forward cost by ~K. The rows it does score must be
    bit-identical to those rows of a full pass -- otherwise the fold geometry
    changed and the training conditioning no longer matches deployment's."""
    tabicl = RowIndependentFakeTabICL()
    Xtr, Ytr, Xte, Yte = _pit_inputs(B=2, P=12, N=3, seed=5)
    K = 4

    full = run_pit_batched(tabicl, Xtr, Ytr, Xte, Yte, k_folds=K)
    for subset in ([0], [1, 2], [0, 1, 2, 3]):
        sub = _run_pit_batched_impl(tabicl, Xtr, Ytr, Xte, Yte, K, 1e-6, fold_subset=subset)
        rows = sub["train_query_idx"]
        assert rows.numel() == sum(
            min((k + 1) * 3, 12) - k * 3 for k in subset
        ), subset
        assert torch.allclose(sub["z_train"], full["z_train"][:, rows, :], atol=0), subset
