"""Phase-A tier ladder: each marginal backbone accepts only the tiers it supports."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "name,tier,ok",
    [
        ("tabicl", 3, True),
        ("tabpfn35_fast", 0, True),
        ("tabpfn35_fast", 1, False),
        ("kumo_small", 0, True),
        ("kumo_small", 1, False),
    ],
)
def test_resolve_tier_gates_the_ladder(name: str, tier: int, ok: bool) -> None:
    """Tier >= 1 raises where attention is not swappable."""
    from copula_inter.marginal.finetune_adapters import resolve_tier

    if ok:
        assert resolve_tier(name, tier) == tier
    else:
        with pytest.raises(ValueError, match="not available for backbone"):
            resolve_tier(name, tier)
