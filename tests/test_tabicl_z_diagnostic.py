"""
test_tabicl_z_diagnostic.py — Sanity checks for the z_train sim-to-real
validation diagnostic (train.py::_build_tabicl_val_z, _corr_grid_fig's
"Pred (z_tabicl)" row).

This diagnostic re-runs the model on each plotted val episode conditioned on
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
  4. _corr_grid_fig draws a second "Pred (z_tabicl)" row when any episode
     carries an R_pred_tabicl key, and only the original single row when
     none do (unchanged behaviour for kernel_fit probe episodes, which never
     carry that key).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from train import _build_tabicl_val_z, _corr_grid_fig


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


def make_val_batch(B: int, P_max: int, d_x: int = 2, n_train: "list[int] | None" = None, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x_train = torch.randn(B, P_max, d_x, generator=g)
    y_train = torch.randn(B, P_max, generator=g)
    n_train = n_train or [P_max] * B
    train_mask = torch.zeros(B, P_max, dtype=torch.bool)
    for b, n in enumerate(n_train):
        train_mask[b, :n] = True
    return {"x_train": x_train, "y_train": y_train, "train_mask": train_mask}


def test_cache_shape_and_padding():
    tabicl = FakeTabICL()
    batch = make_val_batch(B=3, P_max=6, n_train=[6, 4, 2])
    cache = _build_tabicl_val_z([batch], tabicl, k_folds=3, device="cpu")

    assert set(cache.keys()) == {0}
    z = cache[0]
    assert z.shape == (3, 6)
    # Padding beyond each episode's true train length stays exactly zero.
    assert torch.equal(z[1, 4:], torch.zeros(2))
    assert torch.equal(z[2, 2:], torch.zeros(4))


def test_short_context_skipped_stays_zero():
    tabicl = FakeTabICL()
    batch = make_val_batch(B=1, P_max=5, n_train=[1])  # n_train < 2 -> skipped
    cache = _build_tabicl_val_z([batch], tabicl, k_folds=3, device="cpu")
    assert torch.equal(cache[0], torch.zeros(1, 5))


def test_deterministic_across_calls():
    """The whole point of precomputing this once (see _build_tabicl_val_z's
    docstring): a frozen model on unchanged episodes must give the same
    z_train_tabicl every time, or caching it would silently go stale."""
    tabicl = FakeTabICL()
    batch = make_val_batch(B=4, P_max=8, n_train=[8, 6, 3, 8])
    cache_1 = _build_tabicl_val_z([batch], tabicl, k_folds=4, device="cpu")
    cache_2 = _build_tabicl_val_z([batch], tabicl, k_folds=4, device="cpu")
    assert torch.equal(cache_1[0], cache_2[0])


def _make_episode(n: int, with_tabicl: bool, label: str) -> dict:
    """A plausible (bounded [-1, 1], unit-diagonal) correlation matrix, as
    _oracle_diagonal_order's 1 - |R_ora| seriation distance requires."""
    g = torch.Generator().manual_seed(hash(label) % 2**31)
    a = torch.randn(n, n, generator=g) * 0.3
    r_ora = ((a + a.T) / 2).clamp(-0.9, 0.9).numpy()
    np.fill_diagonal(r_ora, 1.0)
    ep = {"R_pred": r_ora, "R_ora": r_ora, "label": label}
    if with_tabicl:
        ep["R_pred_tabicl"] = r_ora
    return ep


def test_corr_grid_adds_tabicl_row_when_present():
    episodes = [_make_episode(4, with_tabicl=True, label="ep0"),
                _make_episode(4, with_tabicl=True, label="ep1")]
    fig = _corr_grid_fig(episodes, step=10)
    # 2 estimator rows x (oracle+pred) columns per episode, wrapped in bands.
    assert len(fig.axes) > 2 * 2 * 2  # generous lower bound; exact count includes colorbar
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_corr_grid_single_row_when_tabicl_absent():
    """kernel_fit probe episodes never carry R_pred_tabicl (see validate()) —
    the grid must fall back to exactly its original single-row layout."""
    episodes = [_make_episode(4, with_tabicl=False, label="ep0"),
                _make_episode(4, with_tabicl=False, label="ep1")]
    fig = _corr_grid_fig(episodes, step=10)
    import matplotlib.pyplot as plt
    n_ep = len(episodes)
    # n_est=1 row, 2 columns (oracle, pred) per episode, wrapped across
    # min(_CORR_GRID_N_WRAP, n_ep) bands, plus one shared colorbar axis.
    from train import _CORR_GRID_N_WRAP
    n_wrap = max(1, min(_CORR_GRID_N_WRAP, n_ep))
    import math
    per_line = math.ceil(n_ep / n_wrap)
    expected_grid_axes = n_wrap * 2 * per_line
    assert len(fig.axes) == expected_grid_axes + 1  # +1 colorbar
    plt.close(fig)
