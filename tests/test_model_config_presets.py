"""Tests composing the conf/model presets through Hydra.

1. copula_prod resolves to the pretrained backbone settings.
2. copula_nano resolves to the shrunk from-scratch backbone with a pit_ckpt.
3. copula_nano builds and runs a forward on CPU (per parametrization too).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import hydra
import pytest
import torch
from conftest import make_batch

from copula_inter.model import build_copula_transformer, build_sigma, low_rank_correlation
from copula_inter.pit import resolve_pit_ckpt as _resolve_pit_ckpt

if TYPE_CHECKING:
    from omegaconf import DictConfig

_CONF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conf")


def _compose(model_name: str) -> DictConfig:
    with hydra.initialize_config_dir(config_dir=_CONF_DIR, version_base=None):
        return hydra.compose(config_name="config", overrides=[f"model={model_name}"])


def test_copula_prod_resolves_pretrained_backbone() -> None:
    cfg = _compose("copula_prod")
    assert cfg.model.unfreeze_backbone is True
    assert cfg.tabicl.pretrained is True
    assert cfg.tabicl.ckpt  # non-empty HF checkpoint name; not downloaded here
    assert cfg.tabicl.pit_k_folds == 10
    assert cfg.model.correlation_parametrization == "covnorm"
    # No pit_ckpt: resolve_pit_ckpt uses tabicl.ckpt.
    assert _resolve_pit_ckpt(cfg) == cfg.tabicl.ckpt


def test_copula_nano_resolves_scratch_backbone() -> None:
    cfg = _compose("copula_nano")
    assert cfg.tabicl.pretrained is False
    # Shrunk width and depth.
    assert cfg.tabicl.arch.embed_dim == 32
    assert cfg.tabicl.arch.col_num_blocks == 1
    assert cfg.tabicl.arch.row_num_blocks == 1
    assert cfg.tabicl.arch.icl_num_blocks == 2
    # The scratch backbone still has a PIT marginal via pit_ckpt.
    assert cfg.tabicl.pit_ckpt
    assert _resolve_pit_ckpt(cfg) == cfg.tabicl.pit_ckpt
    assert cfg.model.correlation_parametrization == "covnorm"


def test_copula_nano_builds_and_runs_forward() -> None:
    """copula_nano builds via build_copula_transformer and runs a CPU forward."""
    cfg = _compose("copula_nano")
    torch.manual_seed(0)
    model = build_copula_transformer(cfg)
    model.train()  # eval() would route through TabICL's inference manager,
    # which auto-selects a CUDA execution device even for this CPU-only
    # scratch model whenever CUDA is available on the host (see test_model.py).

    batch = make_batch(B=2, P=10, N=5)
    with torch.no_grad():
        out = model(batch)

    rank = cfg.model.rank
    assert out["W"].shape == (2, 5, rank)
    assert out["s"].shape == (2, 5)

    Sigma = low_rank_correlation(out["W"], out["s"], batch["test_mask"])
    diag = Sigma.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-3), f"Diagonal not 1: {diag}"
    for b in range(Sigma.shape[0]):
        eigvals = torch.linalg.eigvalsh(Sigma[b])
        assert (eigvals >= -1e-4).all(), f"Batch {b}: negative eigenvalues: {eigvals[eigvals < 0]}"


def test_copula_head_accepts_half_precision_backbone_features() -> None:
    """Backbone inference may return half precision on CPU without autocast."""
    model = build_copula_transformer(_compose("copula_nano"))
    model.train()
    handle = model.feature_extractor.register_forward_hook(
        lambda _module, _inputs, output: output.to(dtype=torch.float16)
    )
    try:
        with torch.no_grad():
            out = model(make_batch(B=2, P=10, N=5))
    finally:
        handle.remove()

    assert out["W"].dtype == model.copula_head.weight.dtype
    assert torch.isfinite(out["W"]).all()


@pytest.mark.parametrize("parametrization", ["covnorm", "cossim", "tanhnorm", "sparse_covnorm"])
def test_copula_nano_builds_and_runs_forward_per_parametrization(parametrization: str) -> None:
    """copula_nano builds and runs for every correlation_parametrization."""
    cfg = _compose("copula_nano")
    cfg.model.correlation_parametrization = parametrization
    torch.manual_seed(0)
    model = build_copula_transformer(cfg)
    model.train()

    batch = make_batch(B=2, P=10, N=5)
    with torch.no_grad():
        out = model(batch)

    rank = cfg.model.rank
    assert out["W"].shape == (2, 5, rank)
    if parametrization == "tanhnorm":
        assert "s" not in out
    else:
        assert out["s"].shape == (2, 5)
    if parametrization == "sparse_covnorm":
        assert out["lam"].shape == (1,)

    with torch.no_grad():
        Sigma = build_sigma(out, cfg, test_mask=batch["test_mask"])
    diag = Sigma.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-3), f"Diagonal not 1: {diag}"
    for b in range(Sigma.shape[0]):
        eigvals = torch.linalg.eigvalsh(Sigma[b])
        assert (eigvals >= -1e-4).all(), f"Batch {b}: negative eigenvalues: {eigvals[eigvals < 0]}"
