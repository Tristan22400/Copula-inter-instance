"""test_lora_all_layers.py — universal LoRA: one shared rank on every 2-D
weight matrix, in every marginal backbone.

The stage/tier ladder adapts swappable attention MODULES, which covers ~91%
of TabLDM's parameters but ~2% of EXAONE's (its attention and feed-forward
weights are raw nn.Parameters inside custom modules with no submodule to
replace). ``apply_lora_all_layers`` uses torch parametrization instead, so
coverage no longer depends on which library used nn.Module children.

Each test here guards a failure that is silent rather than loud:
  - adapters that perturb a pretrained model at step 0,
  - the frozen base weight being unfrozen by a tier-0 pattern that still
    matches its post-rename name (i.e. quiet full fine-tuning),
  - rank drifting between architectures,
  - a checkpoint written with the wrong merger, which only fails at the next
    run's load_state_dict -- long after the training spend.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"),
           os.path.join(_REPO_ROOT, "tabicl_upstream", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RANK = 8
BACKENDS = ["tabldm", "exaone"]


def _load(name):
    pytest.importorskip(
        {"tabldm": "tabldm", "exaone": "exaonetabular"}[name], reason=f"{name} not installed"
    )
    from marginal_backbones import load_backbone

    return load_backbone(name, device="cpu")


def test_adapters_are_identity_at_initialisation():
    """B is zero-initialised, so installing adapters must not move a single
    weight -- otherwise every run starts from a perturbed pretrained model."""
    from lora import apply_lora_all_layers

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))
    x = torch.randn(5, 6)
    before = model(x).detach().clone()
    n = apply_lora_all_layers(model, rank=RANK, alpha=16.0)
    assert n == 2, f"expected both Linear weights adapted, got {n}"
    torch.testing.assert_close(model(x), before, rtol=0, atol=0)


def test_frozen_base_is_never_unfrozen_by_an_allowlist_pattern():
    """register_parametrization renames `0.weight` to
    `0.parametrizations.weight.original`, which still matches a prefix-style
    tier-0 pattern. Without the guard that silently full-fine-tunes the layer
    LoRA was just installed on."""
    from lora import apply_lora_all_layers

    model = nn.Sequential(nn.Linear(6, 8))
    apply_lora_all_layers(model, rank=RANK, alpha=16.0, also_trainable=(r"^0\.",))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not any(n.endswith(".original") for n in trainable), trainable
    # the bias (1-D, not adapted) is exactly what the allowlist should keep
    assert "0.bias" in trainable
    assert any(n.endswith((".A", ".B")) for n in trainable)


@pytest.mark.parametrize("name", BACKENDS)
def test_every_backbone_gets_the_same_rank_on_every_weight_matrix(name):
    from lora import apply_lora_all_layers

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
    # Essentially total coverage: every 2-D matrix that existed before is now
    # behind an adapter (the count shifts because `.original` replaces it).
    assert n_adapted == n_matrices, f"{name}: adapted {n_adapted} of {n_matrices} matrices"


@pytest.mark.parametrize("name", BACKENDS)
def test_only_adapters_train_and_gradients_reach_them(name):
    from lora import apply_lora_all_layers

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

    # At initialisation B == 0, so d(loss)/dA = B^T @ grad is exactly zero:
    # only the B factors move on the first step. This is standard LoRA
    # behaviour, not a broken graph -- asserting "all adapters get gradient"
    # here would be asserting something false.
    n_b = sum(1 for n, _ in bb.module.named_parameters() if n.endswith(".B"))
    assert len(_live(".B")) > 0.9 * n_b, (
        f"{name}: only {len(_live('.B'))}/{n_b} B factors received gradient"
    )
    assert not _live(".A"), "A should have zero gradient while B is still zero"

    # After one step B is nonzero, so A becomes trainable too -- this is what
    # proves the whole factorisation is live, not just half of it.
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
    """The written file must carry the ORIGINAL names with deltas baked in,
    or the next run's strict load_state_dict fails."""
    from lora import apply_lora_all_layers

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

    from marginal_backbones import load_backbone

    fresh = load_backbone(name, device="cpu")
    base = fresh.module.state_dict()
    moved = [k for k in stock_keys
             if base[k].dim() == 2 and not torch.allclose(base[k].float(), sd[k].float())]
    assert moved, "no adapted weight changed; the delta was not merged in"
