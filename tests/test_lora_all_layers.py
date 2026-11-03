"""Tests for all-layer LoRA (apply_lora_all_layers): identity at init and frozen base weights."""

from __future__ import annotations

import torch
import torch.nn as nn

RANK = 8


def test_adapters_are_identity_at_initialisation() -> None:
    """Installing adapters leaves every weight unchanged (B = 0)."""
    from copula_inter.marginal.lora import apply_lora_all_layers

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))
    x = torch.randn(5, 6)
    before = model(x).detach().clone()
    n = apply_lora_all_layers(model, rank=RANK, alpha=16.0)
    assert n == 2, f"expected both Linear weights adapted, got {n}"
    torch.testing.assert_close(model(x), before, rtol=0, atol=0)


def test_frozen_base_is_never_unfrozen_by_an_allowlist_pattern() -> None:
    """A tier-0 pattern matching "...parametrizations.weight.original" does not unfreeze it."""
    from copula_inter.marginal.lora import apply_lora_all_layers

    model = nn.Sequential(nn.Linear(6, 8))
    apply_lora_all_layers(model, rank=RANK, alpha=16.0, also_trainable=(r"^0\.",))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not any(n.endswith(".original") for n in trainable), trainable
    # the bias (1-D, not adapted) is exactly what the allowlist should keep
    assert "0.bias" in trainable
    assert any(n.endswith((".A", ".B")) for n in trainable)
