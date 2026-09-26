"""Tests for all-layer LoRA (apply_lora_all_layers) on every marginal backbone: identity at init, frozen base weights, shared rank, and checkpoint merging to stock names."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

RANK = 8
BACKENDS = ["tabldm", "exaone"]


def _load(name):
    pytest.importorskip(
        {"tabldm": "tabldm", "exaone": "exaonetabular"}[name], reason=f"{name} not installed"
    )
    from copula_inter.marginal_backbones import load_backbone

    return load_backbone(name, device="cpu")


def test_adapters_are_identity_at_initialisation():
    """Installing adapters leaves every weight unchanged (B = 0)."""
    from copula_inter.lora import apply_lora_all_layers

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))
    x = torch.randn(5, 6)
    before = model(x).detach().clone()
    n = apply_lora_all_layers(model, rank=RANK, alpha=16.0)
    assert n == 2, f"expected both Linear weights adapted, got {n}"
    torch.testing.assert_close(model(x), before, rtol=0, atol=0)


def test_frozen_base_is_never_unfrozen_by_an_allowlist_pattern():
    """A tier-0 pattern matching "...parametrizations.weight.original" does not unfreeze it."""
    from copula_inter.lora import apply_lora_all_layers

    model = nn.Sequential(nn.Linear(6, 8))
    apply_lora_all_layers(model, rank=RANK, alpha=16.0, also_trainable=(r"^0\.",))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not any(n.endswith(".original") for n in trainable), trainable
    # the bias (1-D, not adapted) is exactly what the allowlist should keep
    assert "0.bias" in trainable
    assert any(n.endswith((".A", ".B")) for n in trainable)


@pytest.mark.parametrize("name", BACKENDS)
def test_every_backbone_gets_the_same_rank_on_every_weight_matrix(name):
    from copula_inter.lora import apply_lora_all_layers

    bb = _load(name)
    n_matrices = sum(1 for _, p in bb.module.named_parameters() if p.dim() == 2)
    n_adapted = apply_lora_all_layers(bb.module, rank=RANK, alpha=16.0)
    assert n_adapted > 0

    ranks = set()
    for mod in bb.module.modules():
        for plist in getattr(mod, "parametrizations", {}).values():
            ranks.add(plist[0].A.shape[0])
            ranks.add(plist[0].B.shape[1])
    assert ranks == {RANK}, f"{name} has mixed LoRA ranks: {sorted(ranks)}"
    # Every former 2-D matrix is now behind an adapter.
    assert n_adapted == n_matrices, f"{name}: adapted {n_adapted} of {n_matrices} matrices"


@pytest.mark.parametrize("name", BACKENDS)
def test_only_adapters_train_and_gradients_reach_them(name):
    from copula_inter.lora import apply_lora_all_layers

    bb = _load(name)
    apply_lora_all_layers(bb.module, rank=RANK, alpha=16.0)
    trainable = [n for n, p in bb.module.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith((".A", ".B")) for n in trainable)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 3)).astype(np.float32)
    y = (X[:, 0] * 1.3 + 0.2 * rng.normal(size=15)).astype(np.float32)
    probs = np.linspace(0.1, 0.9, 9)
    bb.module.zero_grad(set_to_none=True)
    q = bb.quantile_forward([X[:12]], [y[:12]], [X[12:]], probs)
    assert torch.isfinite(q).all()
    q.pow(2).mean().backward()

    def _live(suffix):
        return [n for n, p in bb.module.named_parameters()
                if n.endswith(suffix) and p.grad is not None and p.grad.abs().sum() > 0]

    # At init B == 0, so only the B factors get gradients.
    n_b = sum(1 for n, _ in bb.module.named_parameters() if n.endswith(".B"))
    assert len(_live(".B")) > 0.9 * n_b, (
        f"{name}: only {len(_live('.B'))}/{n_b} B factors received gradient"
    )
    assert not _live(".A"), "A should have zero gradient while B is still zero"

    # After one step A gets gradients too.
    opt = torch.optim.SGD([p for p in bb.module.parameters() if p.requires_grad], lr=1e-2)
    opt.step()
    bb.module.zero_grad(set_to_none=True)
    q2 = bb.quantile_forward([X[:12]], [y[:12]], [X[12:]], probs)
    q2.pow(2).mean().backward()
    assert len(_live(".A")) > 0.5 * n_b, (
        f"{name}: A factors still dead after a step ({len(_live('.A'))}/{n_b})"
    )


@pytest.mark.parametrize("name", BACKENDS)
def test_checkpoint_merges_adapters_back_to_stock_parameter_names(name, tmp_path):
    """The checkpoint uses the original parameter names with deltas merged."""
    from copula_inter.lora import apply_lora_all_layers

    bb = _load(name)
    stock_keys = set(bb.module.state_dict().keys())
    apply_lora_all_layers(bb.module, rank=RANK, alpha=16.0)

    # a non-trivial delta, so "merged" is distinguishable from "base"
    with torch.no_grad():
        for mod in bb.module.modules():
            for plist in getattr(mod, "parametrizations", {}).values():
                plist[0].B.add_(0.01)

    path = str(tmp_path / f"{name}_all_layers.pt")
    bb.save(path, step=5)
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    assert set(sd.keys()) == stock_keys, (
        "merged state dict does not match the stock parameter names: "
        f"extra={sorted(set(sd) - stock_keys)[:3]} missing={sorted(stock_keys - set(sd))[:3]}"
    )
    assert not any(".parametrizations." in k for k in sd)

    from copula_inter.marginal_backbones import load_backbone

    fresh = load_backbone(name, device="cpu")
    base = fresh.module.state_dict()
    moved = [k for k in stock_keys
             if base[k].dim() == 2 and not torch.allclose(base[k].float(), sd[k].float())]
    assert moved, "no adapted weight changed; the delta was not merged in"
