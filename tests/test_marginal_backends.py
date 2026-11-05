"""The generic marginal backends (tabpfn35_fast, kumo_small) on their pretrained weights.

For each backend: the batched K-fold PIT equals the per-episode one and reads a
proper density with tails (the model's full predictive distribution), Phase A's
tier-0 patterns name real parameters and gradients reach them through the same
quantile function the PIT reads, and a Phase-A checkpoint round-trips through
make_regressor. TabPFN also checks bar_icdf against the bar distribution's own
icdf. Skipped without the backend's uv extra (and, for TabPFN, without
TABPFN_TOKEN with the TabPFN-3.5 licence accepted).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.pretrained

BACKENDS = [
    pytest.param("tabpfn35_fast", "tabpfn", id="tabpfn35_fast"),
    pytest.param("kumo_small", "sdm", id="kumo_small"),
]


def _require(backend: str, module: str) -> None:
    pytest.importorskip(module, reason=f"{module} not installed (uv extra)")
    if backend == "tabpfn35_fast" and not os.environ.get("TABPFN_TOKEN"):
        pytest.skip("TABPFN_TOKEN not set")


def _episodes(B: int, P: int, N: int, p_x: int, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(B, p_x)).astype(np.float32)
    X_train = rng.normal(size=(B, P, p_x)).astype(np.float32)
    X_test = rng.normal(size=(B, N, p_x)).astype(np.float32)
    Y_train = (np.sin(np.einsum("bpi,bi->bp", X_train, w)) + 0.2 * rng.normal(size=(B, P))).astype(np.float32)
    Y_test = (np.sin(np.einsum("bni,bi->bn", X_test, w)) + 0.2 * rng.normal(size=(B, N))).astype(np.float32)
    return X_train, Y_train, X_test, Y_test


@pytest.mark.parametrize("backend,module", BACKENDS)
def test_batched_pit_matches_per_episode(backend: str, module: str) -> None:
    """The batched K-fold PIT of B episodes equals the same driver run on each episode alone."""
    from copula_inter.backend_registry import batched_pit
    from copula_inter.marginal.backends import make_regressor

    _require(backend, module)
    regressor = make_regressor(backend, device="cpu")
    B, P, N, p_x, K = 3, 20, 6, 2, 4
    X_train, Y_train, X_test, Y_test = _episodes(B, P, N, p_x)
    seed = 12345
    run = batched_pit(backend)

    out = run(regressor, X_train, Y_train, X_test, Y_test, k_folds=K, seed=seed)
    for b in range(B):
        one = run(
            regressor,
            X_train[b : b + 1],
            Y_train[b : b + 1],
            X_test[b : b + 1],
            Y_test[b : b + 1],
            k_folds=K,
            seed=seed + b,
        )
        np.testing.assert_allclose(out["z_train"][b], one["z_train"][0], atol=1e-3)
        np.testing.assert_allclose(out["z_test"][b], one["z_test"][0], atol=1e-3)
        np.testing.assert_allclose(out["log_pdf_test"][b], one["log_pdf_test"][0], atol=1e-2)
    assert all(np.isfinite(v).all() for v in out.values())
    # Deterministic: a second pass gives the same residuals.
    again = run(regressor, X_train, Y_train, X_test, Y_test, k_folds=K, seed=seed)
    np.testing.assert_array_equal(again["z_test"], out["z_test"])


def _pit_on_grid(backend: str, regressor: Any, y_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(u, log_pdf) on y_grid for the first query row of one episode."""
    from copula_inter.marginal.finetune_adapters import backend_pit

    X_train, Y_train, X_test, _ = _episodes(1, 24, 1, 2, seed=5)
    u_parts, lp_parts = [], []
    for y in np.array_split(y_grid.astype(np.float32), max(1, len(y_grid) // 10_000)):
        xq = np.repeat(X_test[0], len(y), axis=0)
        with torch.no_grad():
            u, log_pdf = backend_pit(backend, regressor, [X_train[0]], [Y_train[0]], [xq], [y])
        u_parts.append(u[0].double().numpy())
        lp_parts.append(log_pdf[0].double().numpy())
    return np.concatenate(u_parts), np.concatenate(lp_parts)


@pytest.mark.parametrize("backend,module", BACKENDS)
def test_pit_density_is_proper_with_tails(backend: str, module: str) -> None:
    """The PIT reads a proper density: it integrates to 1, is the derivative of u, and u reaches the tails.

    The old 99-quantile estimator failed all three: u was clamped to [0.01, 0.99]
    (|z| <= 2.33) and the density stayed constant beyond the outer quantiles.
    """
    from copula_inter.marginal.backends import make_regressor

    _require(backend, module)
    regressor = make_regressor(backend, device="cpu")
    y = np.linspace(-30.0, 30.0, 24001)
    u, log_pdf = _pit_on_grid(backend, regressor, y)
    pdf = np.exp(log_pdf)
    assert np.isfinite(log_pdf).all()
    assert abs(np.trapezoid(pdf, y) - 1.0) < 2e-2, np.trapezoid(pdf, y)
    assert np.all(np.diff(u) >= -1e-6), "CDF not monotone"
    assert u[0] < 1e-4 and u[-1] > 1 - 1e-4, (u[0], u[-1])
    # F(y) - F(y_0) = integral of f: u and log_pdf describe one distribution. Checked on a
    # fine grid over the central 98%: these piecewise densities have steps (bar borders,
    # spline knots) and Kumo's narrowest quantile spacings are ~1e-5 wide.
    from scipy.integrate import cumulative_trapezoid

    lo, hi = y[np.searchsorted(u, 0.01)], y[np.searchsorted(u, 0.99)]
    y_fine = np.linspace(lo, hi, 200_001)
    u_fine, log_pdf_fine = _pit_on_grid(backend, regressor, y_fine)
    integral = u_fine[0] + cumulative_trapezoid(np.exp(log_pdf_fine), y_fine, initial=0.0)
    np.testing.assert_allclose(integral, u_fine, atol=5e-3)


@pytest.mark.parametrize("backend,module", BACKENDS)
def test_pit_inverts_the_quantile_function(backend: str, module: str) -> None:
    """u(q_p) = p: the PIT's CDF and the quantiles Phase A trains are the same distribution."""
    from copula_inter.marginal.backends import make_regressor
    from copula_inter.marginal.finetune_adapters import backend_pit, backend_quantiles

    _require(backend, module)
    regressor = make_regressor(backend, device="cpu")
    X_train, Y_train, X_test, _ = _episodes(1, 24, 4, 2, seed=8)
    probs = np.array([0.002, 0.05, 0.3, 0.5, 0.7, 0.95, 0.998])
    with torch.no_grad():
        q = backend_quantiles(backend, regressor, [X_train[0]], [Y_train[0]], [X_test[0]], probs)[0].numpy()
        for j, p in enumerate(probs):
            u, _ = backend_pit(backend, regressor, [X_train[0]], [Y_train[0]], [X_test[0]], [q[:, j]])
            np.testing.assert_allclose(u[0].numpy(), p, atol=2e-3)


@pytest.mark.parametrize("backend,module", BACKENDS)
def test_quantiles_are_affine_equivariant(backend: str, module: str) -> None:
    """Both backends standardize the context targets, so quantiles follow y -> a*y + c."""
    from copula_inter.marginal.backends import make_regressor, quantiles

    _require(backend, module)
    regressor = make_regressor(backend, device="cpu")
    X_train, Y_train, X_test, _ = _episodes(1, 24, 5, 2, seed=3)
    probs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    q = quantiles(backend, regressor, X_train[0], Y_train[0], X_test[0], probs)
    q_affine = quantiles(backend, regressor, X_train[0], 3.0 * Y_train[0] + 5.0, X_test[0], probs)
    np.testing.assert_allclose(q_affine, 3.0 * q + 5.0, atol=1e-3)
    assert (np.diff(q, axis=-1) >= -1e-6).all()


@pytest.mark.parametrize("backend,module", BACKENDS)
def test_phase_a_gradients_reach_tier0(backend: str, module: str) -> None:
    from copula_inter.marginal.finetune_adapters import load_backbone
    from copula_inter.marginal.tiers import apply_tier

    _require(backend, module)
    bb = load_backbone(backend, device="cpu")
    for p in bb.module.parameters():
        p.requires_grad_(True)
    report = apply_tier(bb.module, 0, lora_rank=4, backbone_name=backend, all_layers=True)
    assert 0 < report["n_trainable_params"] < report["n_total_params"]

    X_train, Y_train, X_test, Y_test = _episodes(2, 20, 5, 2)
    q = bb.quantile_forward(list(X_train), list(Y_train), list(X_test))
    assert q.shape == (2, 5, bb.native_quantile_count) and q.requires_grad
    loss = (q - torch.as_tensor(Y_test)[..., None]).abs().mean()
    loss.backward()
    grads = {n: p.grad for n, p in bb.module.named_parameters() if p.requires_grad and p.grad is not None}
    assert grads, "no trainable parameter received a gradient"
    assert any(float(g.abs().sum()) > 0 for g in grads.values())


@pytest.mark.parametrize("backend,module", BACKENDS)
def test_phase_a_checkpoint_round_trips(backend: str, module: str, tmp_path: Path) -> None:
    from copula_inter.marginal.backends import make_regressor, quantiles
    from copula_inter.marginal.finetune_adapters import load_backbone

    _require(backend, module)
    bb = load_backbone(backend, device="cpu")
    head = {
        "tabpfn35_fast": "heads.mlp_regression.mlp.0.weight",
        "kumo_small": "models.regression.icl_block.head.0.weight",
    }[backend]
    with torch.no_grad():
        dict(bb.module.named_parameters())[head].mul_(1.05)  # a stand-in for Phase-A training
    path = str(tmp_path / "phase_a.pt")
    bb.save(path, step=1)

    X_train, Y_train, X_test, _ = _episodes(1, 20, 5, 2, seed=5)
    probs = np.array([0.1, 0.5, 0.9])
    trained = quantiles(backend, bb.handle, X_train[0], Y_train[0], X_test[0], probs)
    reloaded = quantiles(
        backend, make_regressor(backend, device="cpu", ckpt=path), X_train[0], Y_train[0], X_test[0], probs
    )
    pretrained = quantiles(backend, make_regressor(backend, device="cpu"), X_train[0], Y_train[0], X_test[0], probs)
    np.testing.assert_allclose(reloaded, trained, atol=1e-5)
    assert np.abs(pretrained - trained).max() > 1e-4  # the edit is visible, and did not leak into a fresh regressor

    other = "kumo_small" if backend == "tabpfn35_fast" else "tabpfn35_fast"
    with pytest.raises(ValueError, match="was written for backbone"):
        from copula_inter.marginal.backends import _load_finetuned_weights

        _load_finetuned_weights(other, object(), path, "cpu")


def test_tabpfn_bar_icdf_matches_bar_distribution() -> None:
    _require("tabpfn35_fast", "tabpfn")
    from copula_inter.marginal.backends import make_regressor
    from copula_inter.marginal.tabpfn_marginal import bar_icdf

    marginal: Any = make_regressor("tabpfn35_fast", device="cpu")
    X_train, Y_train, X_test, _ = _episodes(1, 30, 7, 2, seed=1)
    marginal.fit(torch.as_tensor(X_train[0]), torch.as_tensor(Y_train[0]))
    bardist = marginal.regressor.znorm_space_bardist_
    with torch.no_grad():
        logits = marginal.logits(torch.as_tensor(X_test[0]))
    levels = [1e-4, 0.001, 0.1, 0.5, 0.9, 0.999, 1 - 1e-4]
    reference = torch.stack([bardist.icdf(logits, p) for p in levels], dim=-1)
    torch.testing.assert_close(bar_icdf(bardist, logits, torch.tensor(levels)), reference, atol=1e-5, rtol=1e-5)


def test_tabpfn_batched_fingerprints_match_the_library_step() -> None:
    """fingerprints() equals TorchAddFingerprintFeaturesStep bit for bit, incl. duplicate and all-NaN rows."""
    pytest.importorskip("tabpfn", reason="tabpfn not installed (uv extra)")
    from tabpfn.preprocessing.torch.steps import TorchAddFingerprintFeaturesStep

    from copula_inter.marginal.tabpfn_marginal import fingerprints

    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.normal(size=(40, 5, 3)), dtype=torch.float32)
    x[7] = x[3]  # duplicate training rows: collision re-salting
    x[12] = x[3]
    x[35] = x[3]  # a test row equal to a training row: not re-salted
    x[20, 1] = float("nan")  # all-NaN rows are exempt from the collision loop
    x[21, 1] = float("nan")
    x[5, 2, 0] = float("nan")
    n_train = 30
    ref = TorchAddFingerprintFeaturesStep().fit_transform(x, list(range(3)), num_train_rows=n_train).added_columns
    got = fingerprints(x, n_train, n_train * x.shape[-1])
    assert got.shape == ref.shape == (40, 5, 1)
    assert torch.equal(got, ref)


def test_tabpfn_bar_pit_matches_bar_distribution() -> None:
    """bar_pit equals FullSupportBarDistribution.cdf and its negative log-density, in the bins, on the borders and in both tails."""
    _require("tabpfn35_fast", "tabpfn")
    from copula_inter.marginal.backends import make_regressor
    from copula_inter.marginal.tabpfn_marginal import bar_pit

    marginal: Any = make_regressor("tabpfn35_fast", device="cpu")
    X_train, Y_train, X_test, _ = _episodes(1, 30, 6, 2, seed=2)
    marginal.fit(torch.as_tensor(X_train[0]), torch.as_tensor(Y_train[0]))
    bardist = marginal.regressor.znorm_space_bardist_
    borders = bardist.borders.float()
    with torch.no_grad():
        logits = marginal.logits(torch.as_tensor(X_test[0]))  # (6, n_bars)
    ys = torch.cat(
        [
            torch.linspace(float(borders[0]) - 3.0, float(borders[-1]) + 3.0, 401),
            borders[[0, 1, 2, len(borders) // 2, -3, -2, -1]],
        ]
    )
    L = logits.repeat_interleave(len(ys), 0)
    Y = ys.repeat(len(logits))
    u, log_pdf = bar_pit(bardist, L, Y)
    torch.testing.assert_close(u, bardist.cdf(L, Y[:, None]).squeeze(-1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(log_pdf, -bardist(L, Y.clone()), atol=1e-4, rtol=1e-4)


def test_tabpfn_feature_count_can_change() -> None:
    """One TabPFNMarginal serves episodes of different feature counts (GP episodes vary d)."""
    _require("tabpfn35_fast", "tabpfn")
    from copula_inter.marginal.backends import make_regressor, quantiles

    regressor = make_regressor("tabpfn35_fast", device="cpu")
    probs = np.array([0.25, 0.5, 0.75])
    for p_x in (2, 4, 2):
        X_train, Y_train, X_test, _ = _episodes(1, 20, 4, p_x, seed=p_x)
        q = quantiles("tabpfn35_fast", regressor, X_train[0], Y_train[0], X_test[0], probs)
        assert q.shape == (4, 3) and np.isfinite(q).all()


def _mixed_column_episodes() -> tuple[torch.Tensor, ...]:
    """Four episodes; episodes 2 and 3 each have a different feature constant over their context.

    They keep two columns each (but not the same two), so they share a forward
    after their kept columns are gathered; episodes 0 and 1 keep all three.
    """
    X_train, Y_train, X_test, _ = _episodes(4, 25, 7, 3, seed=5)
    X_train[2, :, 1] = 0.3
    X_train[3, :, 0] = -0.8
    X_test[0] *= 30.0  # queries far outside the context: exercises the soft clipping
    return torch.as_tensor(X_train), torch.as_tensor(Y_train), torch.as_tensor(X_test)


def test_kumo_batched_path_matches_the_library_call() -> None:
    """kumo_native_quantiles (batched torch recipe + one network forward) equals KumoTabular(..., recipe=kumo_recipe())."""
    _require("kumo_small", "sdm")
    from copula_inter.marginal.backends import make_regressor
    from copula_inter.marginal.kumo_marginal import _library_quantiles, kumo_native_quantiles

    model = make_regressor("kumo_small", device="cpu")
    xc, yc, xq = _mixed_column_episodes()
    with torch.no_grad():
        fast = kumo_native_quantiles(model, xc, yc, xq)
        library = _library_quantiles(model, xc, yc, xq)
    assert fast.shape == library.shape == (4, 7, 999)
    torch.testing.assert_close(fast, library, atol=1e-4, rtol=1e-4)


def test_tabpfn_batched_logits_match_the_per_episode_fit() -> None:
    """TabPFNMarginal.batched_logits equals fit + logits run on each episode alone (CPU: no autocast)."""
    _require("tabpfn35_fast", "tabpfn")
    from copula_inter.marginal.backends import make_regressor

    marginal: Any = make_regressor("tabpfn35_fast", device="cpu")
    xc, yc, xq = _mixed_column_episodes()
    with torch.no_grad():
        logits, mean, std = marginal.batched_logits(xc, yc, xq)
        for b in range(len(xc)):
            keep = (xc[b] != xc[b, :1]).any(dim=0)
            marginal.fit(xc[b][:, keep], yc[b])
            torch.testing.assert_close(logits[b], marginal.logits(xq[b][:, keep]), atol=1e-4, rtol=1e-4)
            assert mean[b].item() == pytest.approx(marginal.regressor.y_train_mean_, rel=1e-6)
            assert std[b].item() == pytest.approx(marginal.regressor.y_train_std_, rel=1e-6)


@pytest.mark.parametrize("backend,module", BACKENDS)
def test_constant_context_gets_the_feature_free_predictive(backend: str, module: str) -> None:
    """A context whose features are all constant (which both libraries refuse) falls back to N(mean, std) of its targets.

    Episodes in the same call that have a varying feature still go through the model.
    """
    from scipy.stats import norm

    from copula_inter.marginal.backends import make_regressor
    from copula_inter.marginal.finetune_adapters import backend_quantiles

    _require(backend, module)
    regressor = make_regressor(backend, device="cpu")
    X_train, Y_train, X_test, _ = _episodes(3, 20, 5, 2, seed=11)
    X_train[1] = 0.7  # every feature constant over episode 1's context
    probs = np.array([0.1, 0.5, 0.9])
    with torch.no_grad():
        q = backend_quantiles(backend, regressor, list(X_train), list(Y_train), list(X_test), probs).numpy()
        alone = backend_quantiles(backend, regressor, [X_train[0]], [Y_train[0]], [X_test[0]], probs).numpy()
    expected = Y_train[1].mean() + Y_train[1].std(ddof=1) * norm.ppf(probs)
    np.testing.assert_allclose(q[1], np.broadcast_to(expected, (5, 3)), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(q[0], alone[0], atol=1e-4)
    assert np.isfinite(q).all()

    from copula_inter.marginal.finetune_adapters import backend_pit

    y_query = np.resize(expected, 5).astype(np.float32)  # the 0.1/0.5/0.9 quantiles, then repeats
    with torch.no_grad():
        u, log_pdf = backend_pit(backend, regressor, list(X_train), list(Y_train), list(X_test), [y_query] * 3)
    gauss = norm(Y_train[1].mean(), Y_train[1].std(ddof=1))
    np.testing.assert_allclose(u[1, :3].numpy(), probs, atol=1e-5)
    np.testing.assert_allclose(log_pdf[1].numpy(), gauss.logpdf(y_query), atol=1e-4)


def test_tabpfn_bar_icdf_gradient_is_finite_on_bin_edges() -> None:
    """A level just inside a bin edge must not send NaN through the unselected tail branch.

    Phase A diverged on this: with cond = 1 - 2**-24, (1 + cond) / 2 rounds to 1.0
    in float32, ndtri(1) = inf, and torch.where back-propagates 0 * inf = NaN.
    """
    _require("tabpfn35_fast", "tabpfn")
    from copula_inter.marginal.backends import make_regressor
    from copula_inter.marginal.tabpfn_marginal import bar_icdf

    marginal: Any = make_regressor("tabpfn35_fast", device="cpu")
    X_train, Y_train, _, _ = _episodes(1, 20, 2, 2, seed=2)
    marginal.fit(torch.as_tensor(X_train[0]), torch.as_tensor(Y_train[0]))
    bardist = marginal.regressor.znorm_space_bardist_
    logits = torch.full((1, bardist.num_bars), -1e4)
    logits[0, 10] = logits[0, 20] = 0.0  # half the mass in each of two interior bins
    logits.requires_grad_(True)
    q = bar_icdf(bardist, logits, torch.tensor([0.25, 0.5 - 2.0**-25, 0.75]))  # cond = 1 - 2**-24 in bin 10
    q.sum().backward()
    assert torch.isfinite(q).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
