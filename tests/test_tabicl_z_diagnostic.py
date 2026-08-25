"""
test_tabicl_z_diagnostic.py — Sanity checks for the z_train sim-to-real
validation diagnostic (train.py::_build_tabicl_val_z).

This diagnostic re-runs the model on each val episode conditioned on
TabICL's own K-fold PIT z_train (src/pit.py::run_pit) instead of the exact
GP-LOO one, to check whether the correlation prediction holds up against the
same approximate PIT real (non-GP) deployment data would produce — not just
the closed-form oracle it's trained on almost everywhere else.

A FakeTabICL stands in for the real (network-downloaded) pretrained model:
it only needs to satisfy run_pit's interface (forward(X, y) -> logits,
.quantile_dist(logits) -> a distribution with .cdf/.log_prob), and being a
fixed-seed pure function of its input shapes, is fully deterministic —
which is exactly the property _build_tabicl_val_z relies on to justify
computing z_train_tabicl once instead of every validate() call.

Tests verify:
  1. _build_tabicl_val_z returns one (B, P_max) tensor per batch, zero-padded
     beyond each episode's true train length.
  2. Calling it twice (same frozen model, same batches) gives bit-identical
     output — the determinism the "compute once, cache" design relies on.
  3. Episodes with train_mask sum < 2 are left as all-zero (skipped, not
     just zero because P_max happens to equal n).
  4. _resolve_pit_ckpt (which checkpoint, if any, main() loads as the frozen
     marginal above) resolves correctly across the tabicl.pretrained/
     tabicl.ckpt/tabicl.pit_ckpt combinations copula_prod and copula_nano
     each rely on — see conf/model/{copula_prod,copula_nano}.yaml.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pit import resolve_pit_ckpt as _resolve_pit_ckpt
from train import _build_tabicl_val_z


class FakeTabICL(nn.Module):
    """Minimal stand-in satisfying pit.py::run_pit's interface: forward(X, y)
    -> logits (d, N, Q) for the N query rows after the P context rows in X;
    quantile_dist(logits) -> a distribution with .cdf/.log_prob. Seeded
    per-call so it's a deterministic pure function of (X, y)'s shapes/values,
    matching a real frozen, eval-mode model's determinism."""

    def __init__(self, q: int = 2):
        super().__init__()
        self.q = q

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d, T, _ = X.shape
        P = y.shape[1]
        n = T - P
        g = torch.Generator().manual_seed(int(X.sum().item() * 1000) % 2**31)
        return torch.randn(d, n, self.q, generator=g)

    def quantile_dist(self, logits_flat: torch.Tensor):
        loc = logits_flat[:, 0]
        scale = torch.nn.functional.softplus(logits_flat[:, 1]) + 1e-3
        return torch.distributions.Normal(loc, scale)


def make_val_batch(
    B: int, P_max: int, d_x: int = 2, n_train: "list[int] | None" = None,
    N_max: int = 4, n_test: "list[int] | None" = None, seed: int = 0,
):
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
        "x_train": x_train, "y_train": y_train, "train_mask": train_mask,
        "x_test": x_test, "y_test": y_test, "test_mask": test_mask,
    }


def test_cache_shape_and_padding():
    tabicl = FakeTabICL()
    batch = make_val_batch(B=3, P_max=6, n_train=[6, 4, 2])
    cache = _build_tabicl_val_z([batch], tabicl, k_folds=3, device="cpu")

    assert set(cache.keys()) == {0}
    z = cache[0]["z_train"]
    assert z.shape == (3, 6)
    # Padding beyond each episode's true train length stays exactly zero.
    assert torch.equal(z[1, 4:], torch.zeros(2))
    assert torch.equal(z[2, 2:], torch.zeros(4))


def test_short_context_skipped_stays_zero():
    tabicl = FakeTabICL()
    batch = make_val_batch(B=1, P_max=5, n_train=[1])  # n_train < 2 -> skipped
    cache = _build_tabicl_val_z([batch], tabicl, k_folds=3, device="cpu")
    assert torch.equal(cache[0]["z_train"], torch.zeros(1, 5))


def test_deterministic_across_calls():
    """The whole point of precomputing this once (see _build_tabicl_val_z's
    docstring): a frozen model on unchanged episodes must give the same
    z_train_tabicl every time, or caching it would silently go stale."""
    tabicl = FakeTabICL()
    batch = make_val_batch(B=4, P_max=8, n_train=[8, 6, 3, 8])
    cache_1 = _build_tabicl_val_z([batch], tabicl, k_folds=4, device="cpu")
    cache_2 = _build_tabicl_val_z([batch], tabicl, k_folds=4, device="cpu")
    for key in ("z_train", "z_test", "log_pdf_test"):
        assert torch.equal(cache_1[0][key], cache_2[0][key])


class _FakeTabiclGroup:
    """Minimal stand-in for the Hydra `cfg.tabicl` node: only needs `.get`,
    matching how _resolve_pit_ckpt reads it."""

    def __init__(self, **kw):
        self._d = kw

    def get(self, key, default=None):
        return self._d.get(key, default)


class _FakeCfg:
    def __init__(self, **tabicl_kw):
        self.tabicl = _FakeTabiclGroup(**tabicl_kw)


def test_resolve_pit_ckpt_pretrained_backbone_defaults_to_its_own_ckpt():
    """copula_prod.yaml's original behaviour: pretrained=true + ckpt set,
    no pit_ckpt override -> the diagnostic reuses the backbone's checkpoint."""
    cfg = _FakeCfg(pretrained=True, ckpt="tabicl-regressor-v2-20260212.ckpt")
    assert _resolve_pit_ckpt(cfg) == "tabicl-regressor-v2-20260212.ckpt"


def test_resolve_pit_ckpt_scratch_backbone_opts_in_via_pit_ckpt():
    """copula_nano.yaml's behaviour: pretrained=false (from-scratch backbone,
    no tabicl.ckpt at all) but pit_ckpt set explicitly -> the diagnostic still
    runs, using a checkpoint wholly separate from the backbone being trained."""
    cfg = _FakeCfg(pretrained=False, pit_ckpt="tabicl-regressor-v2-20260212.ckpt")
    assert _resolve_pit_ckpt(cfg) == "tabicl-regressor-v2-20260212.ckpt"


def test_resolve_pit_ckpt_scratch_backbone_without_override_disables_diagnostic():
    """A from-scratch backbone that does NOT set pit_ckpt gets no diagnostic
    (the pre-decoupling behaviour) rather than erroring on a missing ckpt."""
    cfg = _FakeCfg(pretrained=False)
    assert _resolve_pit_ckpt(cfg) is None


def test_resolve_pit_ckpt_explicit_override_wins_over_backbone_ckpt():
    cfg = _FakeCfg(pretrained=True, ckpt="backbone.ckpt", pit_ckpt="other-marginal.ckpt")
    assert _resolve_pit_ckpt(cfg) == "other-marginal.ckpt"
