"""Tests for _build_tabicl_val_z and resolve_pit_ckpt.

1. One (B, P_max) tensor per batch, zero beyond each episode's P.
2. Repeated calls give identical output.
3. Episodes with fewer than 2 training points stay zero.
4. resolve_pit_ckpt across the tabicl.pretrained / ckpt / pit_ckpt
   combinations the presets use.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from copula_inter.pit import resolve_pit_ckpt as _resolve_pit_ckpt
from copula_inter.probe_batches import _build_tabicl_val_z


class FakeTabICL(nn.Module):
    """Deterministic stand-in for run_pit's interface: forward(X, y) -> logits (d, N, Q), quantile_dist(logits) -> distribution with cdf/log_prob."""

    def __init__(self, q: int = 2) -> None:
        super().__init__()
        self.q = q

    def forward(
        self, X: torch.Tensor, y: torch.Tensor, **_kwargs: object
    ) -> torch.Tensor:  # accepts inference_config like TabICL
        d, T, _ = X.shape
        P = y.shape[1]
        n = T - P
        g = torch.Generator().manual_seed(int(X.sum().item() * 1000) % 2**31)
        return torch.randn(d, n, self.q, generator=g)

    def quantile_dist(self, logits_flat: torch.Tensor) -> torch.distributions.Normal:
        loc = logits_flat[:, 0]
        scale = torch.nn.functional.softplus(logits_flat[:, 1]) + 1e-3
        return torch.distributions.Normal(loc, scale)


def make_val_batch(
    B: int,
    P_max: int,
    d_x: int = 2,
    n_train: "list[int] | None" = None,
    N_max: int = 4,
    n_test: "list[int] | None" = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x_train = torch.randn(B, P_max, d_x, generator=g)
    y_train = torch.randn(B, P_max, generator=g)
    n_train = n_train or [P_max] * B
    train_mask = torch.zeros(B, P_max, dtype=torch.bool)
    for b, n in enumerate(n_train):
        train_mask[b, :n] = True
    x_test = torch.randn(B, N_max, d_x, generator=g)
    y_test = torch.randn(B, N_max, generator=g)
    n_test = n_test or [N_max] * B
    test_mask = torch.zeros(B, N_max, dtype=torch.bool)
    for b, n in enumerate(n_test):
        test_mask[b, :n] = True
    return {
        "x_train": x_train,
        "y_train": y_train,
        "train_mask": train_mask,
        "x_test": x_test,
        "y_test": y_test,
        "test_mask": test_mask,
    }


def test_cache_shape_and_padding() -> None:
    tabicl = FakeTabICL()
    batch = make_val_batch(B=3, P_max=6, n_train=[6, 4, 2])
    cache = _build_tabicl_val_z([batch], tabicl, k_folds=3, device="cpu")

    assert set(cache.keys()) == {0}
    z = cache[0]["z_train"]
    assert z.shape == (3, 6)
    # Padding beyond each episode's true train length stays exactly zero.
    assert torch.equal(z[1, 4:], torch.zeros(2))
    assert torch.equal(z[2, 2:], torch.zeros(4))


def test_short_context_skipped_stays_zero() -> None:
    tabicl = FakeTabICL()
    batch = make_val_batch(B=1, P_max=5, n_train=[1])  # n_train < 2 -> skipped
    cache = _build_tabicl_val_z([batch], tabicl, k_folds=3, device="cpu")
    assert torch.equal(cache[0]["z_train"], torch.zeros(1, 5))


def test_deterministic_across_calls() -> None:
    """Two calls with the same model and batches give identical z_train."""
    tabicl = FakeTabICL()
    batch = make_val_batch(B=4, P_max=8, n_train=[8, 6, 3, 8])
    cache_1 = _build_tabicl_val_z([batch], tabicl, k_folds=4, device="cpu")
    cache_2 = _build_tabicl_val_z([batch], tabicl, k_folds=4, device="cpu")
    for key in ("z_train", "z_test", "log_pdf_test"):
        assert torch.equal(cache_1[0][key], cache_2[0][key])


class _FakeTabiclGroup:
    """Stand-in for cfg.tabicl with only .get."""

    def __init__(self, **kw: Any) -> None:
        self._d = kw

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)


class _FakeCfg:
    def __init__(self, **tabicl_kw: Any) -> None:
        self.tabicl = _FakeTabiclGroup(**tabicl_kw)


def test_resolve_pit_ckpt_pretrained_backbone_defaults_to_its_own_ckpt() -> None:
    """pretrained=true without pit_ckpt resolves to tabicl.ckpt."""
    cfg = _FakeCfg(pretrained=True, ckpt="tabicl-regressor-v2-20260212.ckpt")
    assert _resolve_pit_ckpt(cfg) == "tabicl-regressor-v2-20260212.ckpt"


def test_resolve_pit_ckpt_scratch_backbone_opts_in_via_pit_ckpt() -> None:
    """pretrained=false with pit_ckpt resolves to pit_ckpt."""
    cfg = _FakeCfg(pretrained=False, pit_ckpt="tabicl-regressor-v2-20260212.ckpt")
    assert _resolve_pit_ckpt(cfg) == "tabicl-regressor-v2-20260212.ckpt"


def test_resolve_pit_ckpt_scratch_backbone_without_override_disables_diagnostic() -> None:
    """pretrained=false without pit_ckpt resolves to None."""
    cfg = _FakeCfg(pretrained=False)
    assert _resolve_pit_ckpt(cfg) is None


def test_resolve_pit_ckpt_explicit_override_wins_over_backbone_ckpt() -> None:
    cfg = _FakeCfg(pretrained=True, ckpt="backbone.ckpt", pit_ckpt="other-marginal.ckpt")
    assert _resolve_pit_ckpt(cfg) == "other-marginal.ckpt"
