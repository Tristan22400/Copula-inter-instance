"""
test_pit_batched.py — Tests for pit.py::run_pit_batched and its wiring into
data_gen.py::_generate_gp_batch_raw / generate_gp_batch (data.z_train_source
= "tabicl" in conf/data/gp_tasks.yaml).

Tests verify:
  1. run_pit_batched(B=1) matches the existing single-episode run_pit exactly.
  2. run_pit_batched(B>1) matches looping run_pit per episode -- the whole
     point of the batched version is to fold episodes into TabICL's own
     batch axis instead of a Python loop, so this must be bit-identical.
  3. Passing tabicl_model into _generate_gp_batch_raw overrides z_train only
     -- z_test/log_pdf_test and every other field stay exactly the oracle
     values, and the override actually changes z_train's values (not a
     silently inert no-op).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from data_gen import _generate_gp_batch_raw
from pit import run_pit, run_pit_batched


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
