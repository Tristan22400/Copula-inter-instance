"""Tests for the copula backbone: strip_decoder, build_copula_transformer and its guardrail."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch
from conftest import make_batch

from copula_inter.config_path import merge_configs
from copula_inter.copula import backbones as copula_backbones
from copula_inter.copula.model import build_copula_transformer

if TYPE_CHECKING:
    from omegaconf import DictConfig


def test_tabicl_strip_decoder(small_model_cfg: DictConfig) -> None:
    base: Any = copula_backbones.load_raw_backbone("tabicl", small_model_cfg)
    in_features = copula_backbones.strip_decoder(base)
    assert in_features == 16 * 2  # embed_dim * row_num_cls from small_model_cfg
    assert isinstance(base.icl_predictor.decoder, torch.nn.Identity)


def test_build_copula_transformer_tabicl_default_backbone(small_model_cfg: DictConfig) -> None:
    """model.backbone defaults to "tabicl"."""
    assert "backbone" not in small_model_cfg.model
    model = build_copula_transformer(small_model_cfg)
    assert model.backbone_name == "tabicl"
    model.train()
    out = model(make_batch(B=2, P=6, N=3))
    assert out["W"].shape == (2, 3, small_model_cfg.model.rank)


def test_unknown_backbone_raises(small_model_cfg: DictConfig) -> None:
    cfg = merge_configs(small_model_cfg, {"model": {"backbone": "bogus"}})
    with pytest.raises(ValueError, match="Unknown cfg.model.backbone"):
        build_copula_transformer(cfg)
