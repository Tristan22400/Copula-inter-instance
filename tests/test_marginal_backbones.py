"""test_marginal_backbones.py — Phase-A fine-tuning support for the
non-TabICL marginal backbones (src/marginal_backbones.py).

Three things have to hold for Phase A to mean anything on a new
architecture, and all three fail SILENTLY if they break:

  1. the tier-0 patterns match real parameters (a renamed parameter makes
     the run train a smaller set than its logs claim),
  2. a loss on the quantile output actually reaches the trunk (a numpy
     round-trip anywhere in the forward severs the graph, and the run just
     never improves), and
  3. a checkpoint round-trips (otherwise the fine-tune is unusable
     downstream, which is only discovered at the next copula run).

TabPFN is excluded by its own guard in load_backbone -- its weights are
licence-gated and have never been loaded here.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"),
           os.path.join(_REPO_ROOT, "tabicl_upstream", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BACKENDS = ["tabldm", "exaone"]


@pytest.fixture(scope="module", params=BACKENDS)
def backbone(request):
    pytest.importorskip(
        {"tabldm": "tabldm", "exaone": "exaonetabular"}[request.param],
        reason=f"{request.param} not installed",
    )
    from marginal_backbones import load_backbone

    return load_backbone(request.param, device="cpu")


def _episode(rng, n_ctx=12, n_qry=3, p_x=3):
    X = rng.normal(size=(n_ctx + n_qry, p_x)).astype(np.float32)
    y = (X[:, 0] * 1.4 + 0.3 * rng.normal(size=n_ctx + n_qry)).astype(np.float32)
    return X[:n_ctx], y[:n_ctx], X[n_ctx:]


def test_tier0_patterns_match_real_parameters(backbone):
    """Guards the silent-shrink failure: every tier-0 pattern must address
    at least one parameter of the actual loaded model."""
    from marginal_backbones import assert_patterns_match

    counts = assert_patterns_match(backbone.module, backbone.tier0_patterns)
    assert all(v > 0 for v in counts.values())
    n_tier0 = sum(
        p.numel()
        for n, p in backbone.module.named_parameters()
        if any(__import__("re").search(pat, n) for pat in backbone.tier0_patterns)
    )
    n_total = sum(p.numel() for p in backbone.module.parameters())
    # A sane recalibration head: a real fraction of the model, but nowhere
    # near all of it (which would mean the patterns are over-broad and the
    # "tier 0 cannot change context aggregation" claim is false).
    assert 0.0005 < n_tier0 / n_total < 0.25, f"tier-0 covers {n_tier0/n_total:.1%} of {backbone.name}"


def test_quantile_forward_is_differentiable_into_the_trunk(backbone):
    """The claim the whole module rests on: gradients survive the wrapper's
    preprocessing and reach parameters that are NOT just the output head."""
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
    # Not merely the head: something in the trunk must move too, or tier >= 1
    # (and even tier 0's norms) would be untrainable.
    assert any(("tf_icl" in n) or ("transformer.layers" in n) for n in with_grad), (
        f"no trunk parameter received gradient for {backbone.name}"
    )


def test_quantile_forward_is_monotone_in_probs(backbone):
    """A quantile function that isn't nondecreasing in alpha would make the
    downstream finite-difference density negative."""
    rng = np.random.default_rng(1)
    Xc, yc, Xq = _episode(rng)
    probs = np.linspace(0.05, 0.95, 19)
    with torch.no_grad():
        q = backbone.quantile_forward([Xc], [yc], [Xq], probs)
    diffs = q.diff(dim=-1)
    assert (diffs >= -1e-4).all(), f"{backbone.name} quantiles decrease in alpha"


def test_checkpoint_round_trips(backbone, tmp_path):
    from marginal_backbones import load_backbone

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
    """Loading a TabLDM Phase-A checkpoint into EXAONE (or vice versa) must
    fail loudly -- load_state_dict(strict=True) would too, but with a key
    diff rather than the actual reason."""
    from marginal_backbones import load_backbone

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
    """Tier >= 1 on an architecture with no swappable attention must raise,
    not silently fall back to tier 0 -- a run that logs tier=1 while training
    tier 0 would read as 'the ladder didn't help'."""
    from marginal_backbones import resolve_tier

    if ok:
        assert resolve_tier(name, tier) == tier
    else:
        with pytest.raises(ValueError, match="not available for backbone"):
            resolve_tier(name, tier)
