"""EEG joint benchmark: task construction, copula/indep scores, the generic AR chain and the marginal check."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from eval.data import eeg_tasks
from eval.runners.eeg_joint_eval import copula_scores, episode_scores, logmeanexp, marginal_ks


def _write_trials(path: Path, T: int = 5, n_time: int = 40) -> np.ndarray:
    rng = np.random.default_rng(0)
    y = rng.normal(10.0, 3.0, size=(T, n_time, eeg_tasks.N_ELECTRODES))
    np.savez(
        path / "eeg_trials.npz",
        y=y,
        x=np.arange(n_time) / 256.0,
        electrodes=np.array(["FZ", "F1", "F2", "F3", "F4", "F5", "F6"]),
        train_mean=np.full(7, 10.0),
        train_std=np.full(7, 3.0),
        meta=np.array("{}"),
    )
    return y


def test_featurize_is_time_plus_one_hot() -> None:
    f = eeg_tasks.featurize(np.array([0.0, 0.5, 1.0]), np.array([0, 3, 6]))
    assert f.shape == (3, 8)
    np.testing.assert_array_equal(f[:, 0], [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(f[:, 1:].argmax(1), [0, 3, 6])
    np.testing.assert_array_equal(f[:, 1:].sum(1), 1.0)


@pytest.mark.parametrize("mode", ["int", "for"])
def test_hassan_batches_split_and_standardise(tmp_path: Path, mode: str) -> None:
    y = _write_trials(tmp_path)
    batches = list(eeg_tasks.hassan_batches(str(tmp_path), mode, 8, 4, batch_size=2, seed=3))
    assert [len(b.index) for b in batches] == [2, 2, 1]
    b = batches[0]
    assert b.x_train.shape == (2, 8 * 7, 8) and b.x_test.shape == (2, 4 * 7, 8)
    t_ctx = np.unique(b.x_train[0, :, 0])
    t_tgt = np.unique(b.x_test[0, :, 0])
    assert len(t_ctx) == 8 and len(t_tgt) == 4 and not set(t_ctx) & set(t_tgt)
    if mode == "for":
        assert t_tgt.min() > t_ctx.max() and t_ctx.min() == 0.0
    # Values are the trial's standardised measurements at those (time, electrode) rows.
    ti = np.rint(b.x_test[0, :, 0] * 256).astype(int)
    el = b.x_test[0, :, 1:].argmax(1)
    np.testing.assert_allclose(b.y_test[0], (y[0, ti, el] - 10.0) / 3.0)
    # The split of a trial does not depend on the batching.
    again = next(eeg_tasks.hassan_batches(str(tmp_path), mode, 8, 4, batch_size=5, seed=3))
    np.testing.assert_array_equal(again.x_test[1], batches[0].x_test[1])


def test_arcnp_batches_roundtrip(tmp_path: Path) -> None:
    B, nc, nt = 3, 5, 4
    # Any: np.savez types **kwds against its allow_pickle flag.
    arrays: dict[str, Any] = {
        "b0_xc": np.random.rand(B, nc, 1),
        "b0_oc": np.array([0, 0, 1, 2, 6]),
        "b0_yc": np.random.rand(B, nc),
        "b0_xt": np.random.rand(B, nt, 1),
        "b0_ot": np.array([1, 1, 3, 3]),
        "b0_yt": np.random.rand(B, nt),
    }
    np.savez(tmp_path / "eeg_arcnp_interpolation.npz", meta=np.array(json.dumps({"n_batches": 1})), **arrays)
    (b,) = list(eeg_tasks.arcnp_batches(str(tmp_path), "interpolation"))
    assert b.x_train.shape == (B, nc, 8) and b.y_test.shape == (B, nt)
    np.testing.assert_array_equal(b.x_test[0, :, 1:].argmax(1), [1, 1, 3, 3])
    np.testing.assert_array_equal(b.index, np.arange(B))


def test_copula_scores_match_dense_gaussian_copula() -> None:
    g = torch.Generator().manual_seed(0)
    B, N, r = 3, 6, 2
    W = torch.randn(B, N, r, generator=g, dtype=torch.float64)
    D_raw = torch.rand(B, N, generator=g, dtype=torch.float64) + 0.1
    c = (W * W).sum(-1) + D_raw
    U, D = W / c.sqrt().unsqueeze(-1), D_raw / c  # unit diagonal
    z = torch.randn(B, N, generator=g, dtype=torch.float64)
    got = copula_scores(U, D, z)
    for b in range(B):
        S = U[b] @ U[b].T + torch.diag(D[b])
        torch.testing.assert_close(torch.diagonal(S), torch.ones(N, dtype=torch.float64))
        want = 0.5 * (torch.logdet(S) + z[b] @ torch.linalg.solve(S, z[b]) - z[b] @ z[b])
        assert got[b] == pytest.approx(float(want), rel=1e-6)


def test_identity_copula_equals_independence() -> None:
    B, N = 2, 5
    U, D = torch.zeros(B, N, 1), torch.ones(B, N)
    z = torch.randn(B, N)
    log_pdf = np.random.default_rng(0).normal(size=(B, N))
    sc = episode_scores(log_pdf, copula_scores(U, D, z))
    np.testing.assert_allclose(sc["ll_copula"], sc["ll_indep"], atol=1e-6)
    np.testing.assert_allclose(sc["ll_indep"], log_pdf.mean(1))


def test_logmeanexp() -> None:
    a = np.array([[0.0, -1000.0], [np.log(3.0), -1000.0 + np.log(3.0)]])
    np.testing.assert_allclose(logmeanexp(a, axis=0), [np.log(2.0), -1000.0 + np.log(2.0)])


def test_copula_draws_keep_standard_normal_marginals() -> None:
    g = torch.Generator().manual_seed(1)
    N, r = 20, 3
    W = torch.randn(N, r, generator=g)
    D_raw = torch.full((N,), 0.05)
    c = (W * W).sum(-1) + D_raw
    ks_c, ks_i = marginal_ks(W / c.sqrt().unsqueeze(-1), D_raw / c, 4000, g)
    # Strong dependence, same marginals: both KS statistics are Monte Carlo noise (~0.87 / sqrt(S)).
    assert ks_c < 0.035 and ks_i < 0.035


def test_backend_chain_conditions_on_revealed_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake backend predicting N(mean of its context targets, 1): each step must see context + revealed values."""
    import copula_inter.marginal.finetune_adapters as fa
    from eval.baselines.autoregressive import backend_autoregressive_log_pdf

    def fake_pit(name: str, reg: Any, xc: list, yc: list, xq: list, yq: list) -> tuple[torch.Tensor, torch.Tensor]:
        mu = torch.tensor([float(np.mean(y)) for y in yc])[:, None]
        y = torch.tensor(np.stack(yq), dtype=torch.float64)
        dist = torch.distributions.Normal(mu.double(), 1.0)
        return dist.cdf(y), dist.log_prob(y)

    monkeypatch.setattr(fa, "backend_pit", fake_pit)
    B, P, N = 2, 4, 3
    g = torch.Generator().manual_seed(0)
    x_tr, x_te = torch.randn(B, P, 2, generator=g), torch.randn(B, N, 2, generator=g)
    y_tr, y_te = torch.randn(B, P, generator=g) * 2 + 1, torch.randn(B, N, generator=g) * 2 + 1
    out = backend_autoregressive_log_pdf("tabpfn35_fast", None, x_tr, y_tr, x_te, y_te, seed=5)
    for b in range(B):
        mean, std = y_tr[b].double().mean(), y_tr[b].double().std()
        ys_tr, ys_te = (y_tr[b].double() - mean) / std, (y_te[b].double() - mean) / std
        revealed: list[float] = []
        for i in range(N):
            j = int(out["order"][b, i])
            ctx = ys_tr.tolist() + revealed
            want = torch.distributions.Normal(float(np.mean(ctx)), 1.0).log_prob(ys_te[j]) - std.log()
            assert float(out["log_pdf"][b, j]) == pytest.approx(float(want), rel=1e-5)
            revealed.append(float(ys_te[j]))


def test_spread_is_even_and_covers_both_ends() -> None:
    np.testing.assert_array_equal(eeg_tasks.spread(256, 4), [0, 85, 170, 255])
    np.testing.assert_array_equal(eeg_tasks.spread(3, 10), [0, 1, 2])
    assert eeg_tasks.spread(10, 0).size == 0


def test_hassan_trial_subset_matches_full_pass(tmp_path: Path) -> None:
    _write_trials(tmp_path, T=6)
    full = {
        int(j): (b.x_test[i], b.y_test[i])
        for b in eeg_tasks.hassan_batches(str(tmp_path), "int", 8, 4, batch_size=4, seed=1)
        for i, j in enumerate(b.index)
    }
    (sub,) = list(eeg_tasks.hassan_batches(str(tmp_path), "int", 8, 4, batch_size=4, seed=1, trials=[1, 5]))
    np.testing.assert_array_equal(sub.index, [1, 5])
    for i, j in enumerate(sub.index):
        np.testing.assert_array_equal(sub.x_test[i], full[int(j)][0])
        np.testing.assert_array_equal(sub.y_test[i], full[int(j)][1])


def test_correlation_copula_nll_matches_low_rank_score() -> None:
    from eval.runners.eeg_gp_baselines import correlation_copula_nll

    g = torch.Generator().manual_seed(2)
    N, r = 6, 3
    W = torch.randn(N, r, generator=g, dtype=torch.float64)
    D_raw = torch.rand(N, generator=g, dtype=torch.float64) + 0.1
    c = (W * W).sum(-1) + D_raw
    U, D = W / c.sqrt().unsqueeze(-1), D_raw / c
    z = torch.randn(N, generator=g, dtype=torch.float64)
    dense = correlation_copula_nll(U @ U.T + torch.diag(D), z)
    assert dense == pytest.approx(float(copula_scores(U[None], D[None], z[None])[0]), rel=1e-8)
    assert correlation_copula_nll(torch.eye(N, dtype=torch.float64), z) == pytest.approx(0.0, abs=1e-12)
