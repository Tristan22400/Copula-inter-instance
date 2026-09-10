"""test_lora_tabldm_compat.py — the assumption that lets Phase-A tier >= 1
(LoRA) work on a Xiaomi-TabLDM backbone as well as a TabICL one.

src/lora.py's LoRAMultiheadAttention is written against TabICL's
MultiheadAttention internals (in_proj_weight as one raw 3D x D Parameter,
tabicl's own multi_head_attention_forward, its rope/kv-cache types). TabLDM
forks that stack verbatim, so the adapter is a legitimate drop-in there --
but "verbatim" is an empirical fact about tabldm 0.1.0, not a guarantee.
These tests pin it down, so an upstream divergence fails HERE with an
obvious message rather than silently installing adapters whose forward no
longer matches the module they replaced.

_get_mha_class returns both classes as a tuple for exactly this reason: the
two are source-identical but distinct class OBJECTS, so the original
`isinstance(child, tabicl_MHA)` matched nothing on a TabLDM backbone and
apply_lora raised "found 0 MultiheadAttention modules" -- tier >= 1 was
unreachable for TabLDM for that reason alone.
"""

from __future__ import annotations

import inspect
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"),
           os.path.join(_REPO_ROOT, "tabicl_upstream", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("tabldm", reason="Xiaomi-TabLDM not installed")


@pytest.mark.parametrize(
    "module_path,symbol",
    [
        ("_model.layers", "MultiheadAttention"),
        ("_model.attention", "multi_head_attention_forward"),
        ("_model.rope", "RotaryEmbedding"),
        ("_model.kv_cache", "KVCacheEntry"),
    ],
)
def test_tabldm_attention_stack_is_source_identical_to_tabicl(module_path, symbol):
    """If any of these diverge, LoRAMultiheadAttention is no longer a valid
    drop-in for TabLDM and _get_mha_class must stop claiming it is."""
    import importlib

    tabicl_sym = getattr(importlib.import_module(f"tabicl.{module_path}"), symbol)
    tabldm_sym = getattr(importlib.import_module(f"tabldm.{module_path}"), symbol)
    assert inspect.getsource(tabicl_sym) == inspect.getsource(tabldm_sym), (
        f"tabldm.{module_path}.{symbol} has diverged from tabicl's. "
        "src/lora.py::_get_mha_class assumes they are identical -- either "
        "port the difference into LoRAMultiheadAttention or drop TabLDM from "
        "that tuple."
    )


def test_get_mha_class_includes_both():
    from lora import _get_mha_class

    from tabicl._model.layers import MultiheadAttention as TabICLMHA
    from tabldm._model.layers import MultiheadAttention as TabLDMMHA

    classes = _get_mha_class()
    assert isinstance(classes, tuple)
    assert TabICLMHA in classes
    assert TabLDMMHA in classes


def test_apply_lora_installs_adapters_on_a_tabldm_backbone():
    """The end the compatibility argument exists for: adapters actually get
    installed, and only on the requested stage."""
    import numpy as np
    from tabldm import TabLDMRegressor

    from lora import apply_lora

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3)).astype(np.float32)
    y = rng.normal(size=20).astype(np.float32)
    reg = TabLDMRegressor(n_estimators=1, device="cpu", random_state=0)
    reg.fit(X[:16], y[:16])
    backbone = reg.model_

    n_replaced = apply_lora(
        backbone=backbone, rank=4, alpha=8.0, target="qkvo",
        stages=["icl"], also_trainable=(),
    )
    assert n_replaced > 0

    lora_owners = {
        n.split(".lora_")[0] for n, _ in backbone.named_parameters() if ".lora_" in n
    }
    assert lora_owners, "no LoRA parameters registered"
    assert all(o.startswith("icl_predictor") for o in lora_owners), (
        f"stages=['icl'] leaked adapters outside icl_predictor: {sorted(lora_owners)[:5]}"
    )
