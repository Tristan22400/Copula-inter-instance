"""The generic marginal backends (tabpfn35_fast, kumo_small) on their pretrained weights.

For each backend: the batched K-fold PIT equals the per-episode one, Phase A's
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
    from copula_inter.backend_registry import batched_pit
    from copula_inter.marginal.backends import loo_pit, make_regressor, quantiles
    from copula_inter.marginal.kfold import compute_pit

    _require(backend, module)
    regressor = make_regressor(backend, device="cpu")
    B, P, N, p_x, K, probs_n = 3, 20, 6, 2, 4, 33
    X_train, Y_train, X_test, Y_test = _episodes(B, P, N, p_x)
    probs = np.linspace(1.0 / (probs_n + 1), probs_n / (probs_n + 1), probs_n)
    seed = 12345

    out = batched_pit(backend)(regressor, X_train, Y_train, X_test, Y_test, k_folds=K, probs_n=probs_n, seed=seed)
    for b in range(B):
        z_train = loo_pit(backend, regressor, X_train[b], Y_train[b], probs, k_folds=K, seed=seed + b)
        z_test, log_pdf = compute_pit(
            quantiles(backend, regressor, X_train[b], Y_train[b], X_test[b], probs), probs, Y_test[b]
        )
        np.testing.assert_allclose(out["z_train"][b], z_train, atol=1e-3)
        np.testing.assert_allclose(out["z_test"][b], z_test, atol=1e-3)
        np.testing.assert_allclose(out["log_pdf_test"][b], log_pdf, atol=1e-2)
    assert all(np.isfinite(v).all() for v in out.values())
    # Deterministic: a second pass gives the same residuals.
    again = batched_pit(backend)(regressor, X_train, Y_train, X_test, Y_test, k_folds=K, probs_n=probs_n, seed=seed)
    np.testing.assert_array_equal(again["z_test"], out["z_test"])


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
