"""Tests that TabLDM's attention stack is source-identical to TabICL's, so LoRAMultiheadAttention applies to TabLDM backbones."""

from __future__ import annotations

import inspect

import pytest

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
def test_tabldm_attention_stack_is_source_identical_to_tabicl(module_path, symbol) -> None:
    """TabLDM's attention classes and functions have the same source as TabICL's."""
    import importlib

    tabicl_sym = getattr(importlib.import_module(f"tabicl.{module_path}"), symbol)
    tabldm_sym = getattr(importlib.import_module(f"tabldm.{module_path}"), symbol)
    assert inspect.getsource(tabicl_sym) == inspect.getsource(tabldm_sym), (
        f"tabldm.{module_path}.{symbol} has diverged from tabicl's. "
        "src/copula_inter/lora.py::_get_mha_class assumes they are identical -- either "
        "port the difference into LoRAMultiheadAttention or drop TabLDM from "
        "that tuple."
    )


def test_get_mha_class_includes_both() -> None:
    from tabldm._model.layers import MultiheadAttention as TabLDMMHA

    from copula_inter.lora import _get_mha_class
    from tabicl._model.layers import MultiheadAttention as TabICLMHA

    classes = _get_mha_class()
    assert isinstance(classes, tuple)
    assert TabICLMHA in classes
    assert TabLDMMHA in classes


def test_apply_lora_installs_adapters_on_a_tabldm_backbone() -> None:
    """apply_lora installs adapters on a TabLDM backbone, only on the requested stage."""
    import numpy as np
    from tabldm import TabLDMRegressor

    from copula_inter.lora import apply_lora

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3)).astype(np.float32)
    y = rng.normal(size=20).astype(np.float32)
    reg = TabLDMRegressor(n_estimators=1, device="cpu", random_state=0)
    reg.fit(X[:16], y[:16])
    backbone = reg.model_

    n_replaced = apply_lora(
        backbone=backbone,
        rank=4,
        alpha=8.0,
        target="qkvo",
        stages=["icl"],
        also_trainable=(),
    )
    assert n_replaced > 0

    lora_owners = {n.split(".lora_")[0] for n, _ in backbone.named_parameters() if ".lora_" in n}
    assert lora_owners, "no LoRA parameters registered"
    assert all(o.startswith("icl_predictor") for o in lora_owners), (
        f"stages=['icl'] leaked adapters outside icl_predictor: {sorted(lora_owners)[:5]}"
    )
