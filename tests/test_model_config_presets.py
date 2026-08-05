"""test_model_config_presets.py — Verify the conf/model/*.yaml experiment presets.

conf/model/copula_prod.yaml and conf/model/copula_nano.yaml each use Hydra
`@package _global_` packaging to set both `model.*` (the copula head) and
`tabicl.*` (the backbone) from a single file (see conf/config.yaml's
defaults list). These tests compose the presets through real Hydra (not a
hand-rolled dict, unlike conftest.py's small_model_cfg) so a typo or renamed
key in either yaml file fails here instead of silently falling back to a
default deep inside model.py.

Tests verify:
  1. copula_prod resolves to the pretrained backbone's expected schema.
     Model construction itself is not exercised here — pretrained=true
     downloads a checkpoint from HuggingFace (see model.py's
     _load_pretrained_tabicl), which other tests avoid with a FakeTabICL
     stand-in (test_tabicl_z_diagnostic.py, test_reliability_diagram.py).
  2. copula_nano resolves to the from-scratch, width+depth-shrunk backbone,
     and still opts into the z_train diagnostic via its own pit_ckpt knob
     (see train.py::_resolve_pit_ckpt) despite training from scratch.
  3. copula_nano is small enough to actually build + forward on CPU, so it
     also exercises the real build_copula_transformer(cfg) codepath end to
     end and checks the same structural properties as test_model.py.
"""

from __future__ import annotations

import os

import hydra
import torch
from conftest import make_batch

from model import build_copula_transformer, low_rank_correlation
from train import _resolve_pit_ckpt

_CONF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conf")


def _compose(model_name: str):
    with hydra.initialize_config_dir(config_dir=_CONF_DIR, version_base=None):
        return hydra.compose(config_name="config", overrides=[f"model={model_name}"])


def test_copula_prod_resolves_pretrained_backbone():
    cfg = _compose("copula_prod")
    assert cfg.model.rank == 32
    assert cfg.model.unfreeze_backbone is True
    assert cfg.tabicl.pretrained is True
    assert cfg.tabicl.ckpt  # non-empty HF checkpoint name; not downloaded here
    assert cfg.tabicl.pit_k_folds == 10
    # z_train diagnostic (_resolve_pit_ckpt) falls back to the backbone's own
    # pretrained checkpoint here -- no pit_ckpt override needed.
    assert _resolve_pit_ckpt(cfg) == cfg.tabicl.ckpt


def test_copula_nano_resolves_scratch_backbone():
    cfg = _compose("copula_nano")
    assert cfg.model.rank == 8
    assert cfg.tabicl.pretrained is False
    # width+depth ablation vs. the pretrained checkpoint (128/3/3/12) --
    # catches accidentally reverting conf/model/copula_nano.yaml's shrink.
    assert cfg.tabicl.arch.embed_dim == 32
    assert cfg.tabicl.arch.col_num_blocks == 1
    assert cfg.tabicl.arch.row_num_blocks == 1
    assert cfg.tabicl.arch.icl_num_blocks == 2
    # Despite the from-scratch backbone, the z_train diagnostic still opts in
    # via its own pit_ckpt knob (see _resolve_pit_ckpt in train.py) rather
    # than being silently skipped because tabicl.pretrained is false.
    assert cfg.tabicl.pit_ckpt
    assert _resolve_pit_ckpt(cfg) == cfg.tabicl.pit_ckpt


def test_copula_nano_builds_and_runs_forward():
    """copula_nano is small enough to actually instantiate + forward on CPU --
    exercises the real build_copula_transformer(cfg) codepath (not the
    hand-rolled small_model_cfg fixture used by test_model.py), so it would
    catch config/model.py drift the resolution-only test above cannot.
    """
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
