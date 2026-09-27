"""Tests for pit.run_pit_batched / run_pit_calib_split_batched / run_pit_batched_grad and their use in data_gen.

1. run_pit_batched with B=1 matches run_pit.
2. B > 1 matches run_pit per episode.
3. The "tabicl" override replaces z_train, z_test and log_pdf_test and
   leaves every other field unchanged.
4. run_pit_calib_split_batched scales on calibration labels only.
5. The "tabicl_split" override replaces z_train only.
6. run_pit_batched_grad matches run_pit_batched.
7. return_quantiles does not change z_train/z_test/log_pdf_test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from copula_inter.data_gen import _generate_gp_batch_raw
from copula_inter.pit import (
    _run_pit_batched_impl,
    normalize_targets,
    run_pit,
    run_pit_batched,
    run_pit_batched_grad,
    run_pit_calib_split_batched,
)
from inference.copula_inference import loo_pit

if TYPE_CHECKING:
    from omegaconf import DictConfig


class RowIndependentFakeTabICL(nn.Module):
    """Fake TabICL whose output for each (episode, target) row depends only on that row's (X, y)."""

    def __init__(self, q: int = 3) -> None:
        super().__init__()
        self.q = q

    def forward(
        self, X: torch.Tensor, y: torch.Tensor, **_kwargs: object
    ) -> torch.Tensor:  # accepts inference_config like TabICL
        batch, T, _ = X.shape
        P = y.shape[1]
        n = T - P
        out = torch.empty(batch, n, self.q)
        for i in range(batch):
            seed = int((X[i].sum() * 1000 + y[i].sum() * 7).item() * 1000) % (2**31)
            g = torch.Generator().manual_seed(seed)
            out[i] = torch.randn(n, self.q, generator=g)
        return out

    def quantile_dist(self, logits_flat: torch.Tensor) -> torch.distributions.Normal:
        loc = logits_flat[:, 0]
        scale = torch.nn.functional.softplus(logits_flat[:, 1]) + 1e-3
        return torch.distributions.Normal(loc, scale)


class FoldScaleProbe(nn.Module):
    """A marginal whose prediction is sensitive to the context label scale."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(
        self, X: torch.Tensor, y: torch.Tensor, **_kwargs: object
    ) -> torch.Tensor:  # accepts inference_config like TabICL
        n_query = X.shape[1] - y.shape[1]
        loc = y.pow(3).mean(dim=1, keepdim=True) + self.anchor
        return loc[:, None, :].expand(-1, n_query, -1)

    def quantile_dist(self, logits_flat: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(logits_flat[:, 0], torch.ones_like(logits_flat[:, 0]))


def test_fold_target_scaling_uses_only_context_labels() -> None:
    model: Any = FoldScaleProbe()  # stands in for TabICL
    x = torch.arange(6, dtype=torch.float32)[:, None]
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    changed = y.clone()
    changed[1] = 40.0  # same held-out fold as row 0; row 0's label is fixed

    z = loo_pit(model, x.numpy(), y.numpy(), k_folds=3)
    z_changed = loo_pit(model, x.numpy(), changed.numpy(), k_folds=3)
    assert abs(z[0] - z_changed[0]) < 1e-5

    # B=1 uses the fold-local scale even though the input was scaled with all P labels.
    for labels, expected in ((y, z), (changed, z_changed)):
        scaled, _, _, _ = normalize_targets(labels)
        out = run_pit_batched(
            model,
            x[None],
            scaled[None, :, None],
            x[:1][None],
            scaled[:1][None, :, None],
            k_folds=3,
            Y_train_raw=labels[None, :, None],
        )
        assert torch.allclose(out["z_train"][0, :, 0], torch.tensor(expected), atol=1e-5)


def test_fold_quantiles_return_on_callers_scale() -> None:
    model = FoldScaleProbe()
    x = torch.arange(6, dtype=torch.float32)[:, None]
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    scaled, _, _, _ = normalize_targets(y)
    out = run_pit_batched(
        model,
        x[None],
        scaled[None, :, None],
        x[:1][None],
        scaled[:1][None, :, None],
        k_folds=3,
        return_quantiles=True,
        Y_train_raw=y[None, :, None],
    )
    context = y[2:]
    fold_mean, fold_std = context.mean(), context.std()
    local = (context - fold_mean) / fold_std
    expected_raw = local.pow(3).mean() * fold_std + fold_mean
    expected = (expected_raw - y.mean()) / y.std()
    assert torch.allclose(out["q_train"][0, :2, 0, 0], expected.expand(2), atol=1e-6)


def test_single_context_fold_uses_raw_label_units() -> None:
    model = FoldScaleProbe()
    x = torch.tensor([[0.0], [1.0]])
    y = torch.tensor([1.0, 2.0])
    scaled, _, _, _ = normalize_targets(y)
    out = run_pit(
        model,
        x,
        scaled[:, None],
        x[:1],
        scaled[:1, None],
        k_folds=2,
        Y_train_raw=y[:, None],
    )
    # With one context label the fold scale is one raw unit.
    assert torch.allclose(out["z_train"][0, 0], torch.tensor(-1.0), atol=1e-5)


def test_fused_grad_folds_preserve_raw_fold_scaling() -> None:
    model = FoldScaleProbe()
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8, 1)
    raw = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [4.0, 1.0, 6.0, 2.0, 9.0, 3.0, 8.0, 5.0]])
    mean = raw.mean(dim=1, keepdim=True)
    std = raw.std(dim=1, keepdim=True)
    scaled = (raw - mean) / std
    args = (model, x, scaled[:, :, None], x[:, :1], scaled[:, :1, None])
    plain = run_pit_batched_grad(
        *args,
        k_folds=4,
        compute_pit=False,
        Y_train_raw=raw[:, :, None],
    )
    fused = run_pit_batched_grad(
        *args,
        k_folds=4,
        compute_pit=False,
        fuse_folds=True,
        Y_train_raw=raw[:, :, None],
    )
    assert torch.allclose(plain["q_train"], fused["q_train"], atol=1e-6)
    assert torch.allclose(plain["q_test"], fused["q_test"], atol=0)


def test_run_pit_batched_b1_matches_run_pit() -> None:
    torch.manual_seed(0)
    tabicl = RowIndependentFakeTabICL()
    P, N, p_x, d = 7, 3, 2, 2
    X_train = torch.randn(P, p_x)
    Y_train = torch.randn(P, d)
    X_test = torch.randn(N, p_x)
    Y_test = torch.randn(N, d)

    single = run_pit(tabicl, X_train, Y_train, X_test, Y_test, k_folds=3)
    batched = run_pit_batched(
        tabicl,
        X_train.unsqueeze(0),
        Y_train.unsqueeze(0),
        X_test.unsqueeze(0),
        Y_test.unsqueeze(0),
        k_folds=3,
    )

    assert torch.allclose(batched["z_train"].squeeze(0), single["z_train"], atol=1e-5)
    assert torch.allclose(batched["z_test"].squeeze(0), single["z_test"], atol=1e-5)
    assert torch.allclose(batched["log_pdf_test"].squeeze(0), single["log_pdf_test"], atol=1e-5)


def test_run_pit_batched_matches_looped_run_pit() -> None:
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


def test_run_pit_calib_split_batched_matches_run_pit_batched_test_side() -> None:
    torch.manual_seed(2)
    tabicl = RowIndependentFakeTabICL()
    B, P_C, P_Q, p_x, d = 3, 6, 4, 2, 2
    X_calib = torch.randn(B, P_C, p_x)
    Y_calib = torch.randn(B, P_C, d)
    X_query = torch.randn(B, P_Q, p_x)
    Y_query = torch.randn(B, P_Q, d)

    mean = Y_calib.mean(dim=1, keepdim=True)
    std = Y_calib.std(dim=1, keepdim=True).clamp(min=1e-8)
    reference = run_pit_batched(
        tabicl,
        X_calib,
        (Y_calib - mean) / std,
        X_query,
        (Y_query - mean) / std,
        k_folds=3,
    )
    split = run_pit_calib_split_batched(tabicl, X_query, Y_query, X_calib, Y_calib)

    assert split["z_train"].shape == (B, P_Q, d)
    assert torch.allclose(split["z_train"], reference["z_test"], atol=1e-6)


def test_calibration_split_query_labels_do_not_scale_context() -> None:
    model = FoldScaleProbe()
    x_calib = torch.arange(3, dtype=torch.float32)[None, :, None]
    y_calib = torch.tensor([3.0, 4.0, 8.0])[None, :, None]
    x_query = torch.arange(3, 5, dtype=torch.float32)[None, :, None]
    y_query = torch.tensor([2.0, 5.0])[None, :, None]
    changed = y_query.clone()
    changed[0, 1, 0] = 100.0
    out = run_pit_calib_split_batched(model, x_query, y_query, x_calib, y_calib)
    alt = run_pit_calib_split_batched(model, x_query, changed, x_calib, y_calib)
    assert torch.allclose(out["z_train"][0, 0], alt["z_train"][0, 0], atol=0)

    # Raw labels remove the dependency on the query pool's moments.
    one_calib = y_calib[:, :1]
    for query in (y_query, changed):
        mean = query.mean(dim=1, keepdim=True)
        std = query.std(dim=1, keepdim=True)
        scaled_out = run_pit_calib_split_batched(
            model,
            x_query,
            (query - mean) / std,
            x_calib[:, :1],
            (one_calib - mean) / std,
            Y_query_raw=query,
            Y_calib_raw=one_calib,
        )
        assert torch.allclose(scaled_out["z_train"][0, 0], torch.tensor(-1.0), atol=1e-5)


def test_run_pit_calib_split_batched_finite() -> None:
    torch.manual_seed(3)
    tabicl = RowIndependentFakeTabICL()
    B, P_C, P_Q, p_x, d = 2, 5, 7, 3, 1
    X_calib = torch.randn(B, P_C, p_x)
    Y_calib = torch.randn(B, P_C, d)
    X_query = torch.randn(B, P_Q, p_x)
    Y_query = torch.randn(B, P_Q, d)

    out = run_pit_calib_split_batched(tabicl, X_query, Y_query, X_calib, Y_calib)
    assert torch.isfinite(out["z_train"]).all()


def test_generate_gp_batch_raw_tabicl_split_z_train_override(small_cfg: DictConfig) -> None:
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.seed = 42

    tabicl = RowIndependentFakeTabICL()

    # Control: same calibration fraction (same T and normalization) without a model.
    analytic = _generate_gp_batch_raw(cfg, B=6, device="cpu", tabicl_split_calib_frac=1.0)
    with_split = _generate_gp_batch_raw(
        cfg,
        B=6,
        device="cpu",
        tabicl_model=tabicl,
        tabicl_split_calib_frac=1.0,
    )

    assert len(analytic) == len(with_split)
    for ep_a, ep_t in zip(analytic, with_split):
        assert ep_a["z_train"].shape == ep_t["z_train"].shape
        assert ep_t["n_train"] == ep_a["n_train"]
        # The override must actually change z_train's values...
        assert not torch.allclose(ep_a["z_train"], ep_t["z_train"])
        # Everything else is unchanged.
        for key in (
            "x_norm_train",
            "x_norm_test",
            "y_train",
            "y_test",
            "z_test",
            "log_pdf_test",
            "R_star",
            "Sigma_star",
            "mu_star",
            "sigma_star",
        ):
            assert torch.allclose(ep_a[key], ep_t[key], atol=1e-6), key


def test_generate_gp_batch_raw_tabicl_split_calib_frac_can_exceed_one(small_cfg: DictConfig) -> None:
    """z_train_split_calib_frac > 1 works."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.seed = 7

    tabicl = RowIndependentFakeTabICL()
    episodes = _generate_gp_batch_raw(
        cfg,
        B=4,
        device="cpu",
        tabicl_model=tabicl,
        tabicl_split_calib_frac=2.5,
    )
    assert len(episodes) > 0
    for ep in episodes:
        assert torch.isfinite(ep["z_train"]).all()
        assert ep["z_train"].shape == ep["y_train"].shape


def test_generate_gp_batch_raw_tabicl_split_calib_frac_zero_is_noop(small_cfg: DictConfig) -> None:
    """tabicl_split_calib_frac=0 with a model uses the K-fold path."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.seed = 11

    tabicl = RowIndependentFakeTabICL()
    kfold = _generate_gp_batch_raw(
        cfg,
        B=4,
        device="cpu",
        tabicl_model=tabicl,
        tabicl_k_folds=3,
        tabicl_split_calib_frac=0.0,
    )
    cfg.seed = 11
    kfold_again = _generate_gp_batch_raw(
        cfg,
        B=4,
        device="cpu",
        tabicl_model=tabicl,
        tabicl_k_folds=3,
    )
    assert len(kfold) == len(kfold_again)
    for ep_a, ep_b in zip(kfold, kfold_again):
        assert torch.allclose(ep_a["z_train"], ep_b["z_train"], atol=1e-6)


def test_generate_gp_batch_raw_tabicl_z_train_override(small_cfg: DictConfig) -> None:
    """The "tabicl" override replaces z_train, z_test and log_pdf_test with TabICL's PIT and leaves every other field unchanged."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.seed = 42

    tabicl = RowIndependentFakeTabICL()

    analytic = _generate_gp_batch_raw(cfg, B=6, device="cpu")
    with_tabicl = _generate_gp_batch_raw(
        cfg,
        B=6,
        device="cpu",
        tabicl_model=tabicl,
        tabicl_k_folds=3,
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
        # Every other field is unchanged.
        for key in (
            "x_norm_train",
            "x_norm_test",
            "y_train",
            "y_test",
            "R_star",
            "Sigma_star",
            "mu_star",
            "sigma_star",
        ):
            assert torch.allclose(ep_a[key], ep_t[key], atol=1e-6), key


def test_generate_gp_batch_raw_tabicl_z_test_matches_direct_run_pit_batched(small_cfg: DictConfig) -> None:
    """The override's z_test/log_pdf_test equal run_pit_batched on the episode with train-only scaling and the Jacobian."""
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
            tabicl,
            x_train,
            y_train_scaled,
            x_test,
            y_test_scaled,
            k_folds=3,
        )
        expected_log_pdf = expected["log_pdf_test"].squeeze(-1) - y_std.log()

        assert torch.allclose(ep["z_test"], expected["z_test"].squeeze(0).squeeze(-1), atol=1e-5)
        assert torch.allclose(ep["log_pdf_test"], expected_log_pdf.squeeze(0), atol=1e-5)


def test_generate_gp_batch_raw_tabicl_noop_without_tabicl_model(small_cfg: DictConfig) -> None:
    """Without tabicl_model, z_train/z_test/log_pdf_test are unchanged."""
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


def test_generate_gp_batch_raw_tabicl_split_keeps_oracle_z_test(small_cfg: DictConfig) -> None:
    """ "tabicl_split" replaces z_train only; z_test/log_pdf_test stay analytic."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.seed = 47

    tabicl = RowIndependentFakeTabICL()
    analytic = _generate_gp_batch_raw(cfg, B=4, device="cpu", tabicl_split_calib_frac=1.0)
    cfg.seed = 47
    split = _generate_gp_batch_raw(
        cfg,
        B=4,
        device="cpu",
        tabicl_model=tabicl,
        tabicl_split_calib_frac=1.0,
    )

    assert len(analytic) == len(split)
    for ep_a, ep_s in zip(analytic, split):
        assert not torch.allclose(ep_a["z_train"], ep_s["z_train"])
        for key in ("z_test", "log_pdf_test"):
            assert torch.allclose(ep_a[key], ep_s[key], atol=1e-6), key


def _pit_inputs(
    B: int = 2, P: int = 9, N: int = 4, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    return (
        torch.randn(B, P, 3),
        torch.randn(B, P, 1),
        torch.randn(B, N, 3),
        torch.randn(B, N, 1),
    )


def test_run_pit_batched_grad_matches_the_no_grad_version() -> None:
    """run_pit_batched_grad and run_pit_batched give identical results."""
    tabicl = RowIndependentFakeTabICL()
    Xtr, Ytr, Xte, Yte = _pit_inputs()

    ref = run_pit_batched(tabicl, Xtr, Ytr, Xte, Yte, k_folds=3)
    got = run_pit_batched_grad(tabicl, Xtr, Ytr, Xte, Yte, k_folds=3, return_quantiles=False)

    for key in ("z_train", "z_test", "log_pdf_test"):
        assert torch.allclose(ref[key], got[key], atol=0), key


class GradProbeFakeTabICL(nn.Module):
    """Differentiable fake (output from a Parameter) that records grad mode and train/eval mode."""

    def __init__(self, q: int = 3) -> None:
        super().__init__()
        self.q = q
        self.w = nn.Parameter(torch.randn(q))
        self.saw_grad_enabled: list[bool] = []
        self.saw_training: list[bool] = []

    def forward(
        self, X: torch.Tensor, y: torch.Tensor, **_kwargs: object
    ) -> torch.Tensor:  # accepts inference_config like TabICL
        self.saw_grad_enabled.append(torch.is_grad_enabled())
        self.saw_training.append(self.training)
        batch, T, _ = X.shape
        n = T - y.shape[1]
        base = X[:, -n:, :1].mean(-1, keepdim=True)  # (batch, n, 1)
        return base + self.w.view(1, 1, self.q)

    def quantile_dist(self, logits_flat: torch.Tensor) -> torch.distributions.Normal:
        loc = logits_flat[:, 0]
        scale = torch.nn.functional.softplus(logits_flat[:, 1]) + 1e-3
        return torch.distributions.Normal(loc, scale)


def test_run_pit_batched_grad_builds_a_graph_and_the_public_one_does_not() -> None:
    """Only run_pit_batched_grad builds an autograd graph."""
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


def test_grad_pit_forces_train_mode_and_restores_it() -> None:
    """The grad path runs in train mode and restores the module's mode."""
    Xtr, Ytr, Xte, Yte = _pit_inputs()
    probe = GradProbeFakeTabICL()
    probe.eval()

    run_pit_batched_grad(probe, Xtr, Ytr, Xte, Yte, k_folds=3, return_quantiles=False)

    assert all(probe.saw_training), "grad path must call the module in train mode"
    assert not probe.training, "grad path must restore the caller's original mode"


def test_return_quantiles_is_additive_and_self_consistent() -> None:
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
    # probit(u_test) equals the returned z_test.
    from copula_inter.pit import _probit

    assert torch.allclose(_probit(extra["u_test"], 1e-6), extra["z_test"], atol=0)
    assert torch.allclose(_probit(extra["u_train"], 1e-6), extra["z_train"], atol=0)


def test_fold_subset_scores_only_the_requested_folds() -> None:
    """fold_subset scores only its folds, bit-identical to the same rows of a full pass."""
    tabicl = RowIndependentFakeTabICL()
    Xtr, Ytr, Xte, Yte = _pit_inputs(B=2, P=12, N=3, seed=5)
    K = 4

    full = run_pit_batched(tabicl, Xtr, Ytr, Xte, Yte, k_folds=K)
    for subset in ([0], [1, 2], [0, 1, 2, 3]):
        sub = _run_pit_batched_impl(tabicl, Xtr, Ytr, Xte, Yte, K, 1e-6, fold_subset=subset)
        rows = sub["train_query_idx"]
        assert rows.numel() == sum(min((k + 1) * 3, 12) - k * 3 for k in subset), subset
        assert torch.allclose(sub["z_train"], full["z_train"][:, rows, :], atol=0), subset
