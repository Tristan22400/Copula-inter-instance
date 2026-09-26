"""Phase-A support for non-TabICL marginal backbones: tier-0 patterns match, gradients reach the trunk, checkpoints round-trip.

tabpfn is not tested (licence-gated weights).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

BACKENDS = ["tabldm", "exaone"]


@pytest.fixture(scope="module", params=BACKENDS)
def backbone(request):
    pytest.importorskip(
        {"tabldm": "tabldm", "exaone": "exaonetabular"}[request.param],
        reason=f"{request.param} not installed",
    )
    from copula_inter.marginal_backbones import load_backbone

    return load_backbone(request.param, device="cpu")


def _episode(rng, n_ctx=12, n_qry=3, p_x=3):
    X = rng.normal(size=(n_ctx + n_qry, p_x)).astype(np.float32)
    y = (X[:, 0] * 1.4 + 0.3 * rng.normal(size=n_ctx + n_qry)).astype(np.float32)
    return X[:n_ctx], y[:n_ctx], X[n_ctx:]


def test_tier0_patterns_match_real_parameters(backbone):
    """Every tier-0 pattern matches at least one parameter."""
    from copula_inter.marginal_backbones import assert_patterns_match

    counts = assert_patterns_match(backbone.module, backbone.tier0_patterns)
    assert all(v > 0 for v in counts.values())
    n_tier0 = sum(
        p.numel()
        for n, p in backbone.module.named_parameters()
        if any(__import__("re").search(pat, n) for pat in backbone.tier0_patterns)
    )
    n_total = sum(p.numel() for p in backbone.module.parameters())
    # Tier 0 is a real but small fraction of the model.
    assert 0.0005 < n_tier0 / n_total < 0.25, f"tier-0 covers {n_tier0/n_total:.1%} of {backbone.name}"


def test_quantile_forward_is_differentiable_into_the_trunk(backbone):
    """Gradients from the quantiles reach trunk parameters, not only the head."""
    rng = np.random.default_rng(0)
    Xc, yc, Xq = _episode(rng)
    probs = np.linspace(0.1, 0.9, 9)

    for p in backbone.module.parameters():
        p.requires_grad_(True)
    backbone.module.zero_grad(set_to_none=True)

    q = backbone.quantile_forward([Xc], [yc], [Xq], probs)
    assert q.shape == (1, Xq.shape[0], len(probs))
    assert q.requires_grad, "quantile_forward returned a detached tensor"
    assert torch.isfinite(q).all()

    q.pow(2).mean().backward()
    with_grad = [n for n, p in backbone.module.named_parameters()
                 if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0]
    assert len(with_grad) > 10, f"only {len(with_grad)} parameters received gradient"
    # Something in the trunk gets a gradient.
    assert any(("tf_icl" in n) or ("transformer.layers" in n) for n in with_grad), (
        f"no trunk parameter received gradient for {backbone.name}"
    )


def test_native_grid_is_the_default_and_matches_the_model_head(backbone):
    """The default grid is the native 999 levels and matches the head."""
    rng = np.random.default_rng(3)
    Xc, yc, Xq = _episode(rng)

    assert backbone.native_quantile_count == 999
    q = backbone.quantile_forward([Xc], [yc], [Xq])          # probs=None
    assert q.shape == (1, Xq.shape[0], 999)
    assert q.requires_grad and torch.isfinite(q).all()

    # The head is built on the forward's levels.
    head = backbone.quantile_dist_module()
    dist = head(q.reshape(-1, q.shape[-1]))
    lp = dist.log_prob(torch.zeros(q.shape[0] * q.shape[1]))
    assert torch.isfinite(lp).all() and lp.requires_grad

    # An explicit grid still works, as the documented cost lever.
    q99 = backbone.quantile_forward([Xc], [yc], [Xq], np.linspace(1 / 100, 99 / 100, 99))
    assert q99.shape == (1, Xq.shape[0], 99)


def test_native_probs_follow_the_repo_grid_convention(backbone):
    """native_probs == linspace(1/(n+1), n/(n+1), n)."""
    n = backbone.native_quantile_count
    np.testing.assert_allclose(
        backbone.native_probs, np.linspace(1.0 / (n + 1), n / (n + 1), n), rtol=0, atol=1e-12
    )


def test_quantile_forward_is_monotone_in_probs(backbone):
    """Quantiles are nondecreasing in alpha."""
    rng = np.random.default_rng(1)
    Xc, yc, Xq = _episode(rng)
    probs = np.linspace(0.05, 0.95, 19)
    with torch.no_grad():
        q = backbone.quantile_forward([Xc], [yc], [Xq], probs)
    diffs = q.diff(dim=-1)
    assert (diffs >= -1e-4).all(), f"{backbone.name} quantiles decrease in alpha"


def test_checkpoint_round_trips(backbone, tmp_path):
    from copula_inter.marginal_backbones import load_backbone

    path = str(tmp_path / f"{backbone.name}_phase_a.pt")
    backbone.save(path, step=123)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["backbone"] == backbone.name
    assert payload["step"] == 123
    assert payload["state_dict"], "empty state_dict"

    restored = load_backbone(backbone.name, ckpt=path, device="cpu")
    a = backbone.module.state_dict()
    b = restored.module.state_dict()
    assert set(a) == set(b)
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)


def test_checkpoint_rejects_a_different_architecture(backbone, tmp_path):
    """Loading a checkpoint into a different architecture raises."""
    from copula_inter.marginal_backbones import load_backbone

    path = str(tmp_path / "mislabelled.pt")
    backbone.save(path, step=1)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["backbone"] = "some_other_model"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="was written for backbone"):
        load_backbone(backbone.name, ckpt=path, device="cpu")


@pytest.mark.parametrize(
    "name,tier,ok",
    [
        ("tabicl", 3, True), ("tabldm", 3, True),
        ("exaone", 0, True), ("tabpfn", 0, True),
        ("exaone", 1, False), ("tabpfn", 1, False),
    ],
)
def test_resolve_tier_gates_the_ladder(name, tier, ok):
    """Tier >= 1 raises where attention is not swappable."""
    from copula_inter.marginal_backbones import resolve_tier

    if ok:
        assert resolve_tier(name, tier) == tier
    else:
        with pytest.raises(ValueError, match="not available for backbone"):
            resolve_tier(name, tier)
