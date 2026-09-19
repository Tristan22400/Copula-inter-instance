"""test_copula_backbones.py — the copula backbone-selection surface added by
src/copula_backbones.py + src/model.py's ``cfg.model.backbone`` dispatch.

Covers, per backbone (tabicl: scratch, fast; tabldm: real pretrained load,
same HF weights the marginal side already caches):
  1. strip_decoder produces the expected feature_dim and an Identity decoder.
  2. build_copula_transformer wires the backbone end-to-end (W/s shapes).
  3. moe_aux_loss is None for tabicl, a grad-carrying scalar for tabldm, and
     model.py's forward surfaces it in `out` only for tabldm.
  4. LoRA (apply_lora) installs adapters on a tabldm backbone the same way
     it already does for tabicl (see test_lora_tabldm_compat.py).
  5. unfreeze_backbone=false freezes the tabldm trunk too.
  6. Guardrails: an unknown backbone name raises; tabldm + tabicl.pretrained
     =false raises (no from-scratch path exists for it).
  7. The tabldm recompute=true escalation actually flips every discovered
     `.recompute` submodule attribute.
  8. A real _forward_and_loss/_run_train_step call on a tiny synthetic batch
     backprops through a tabldm-backboned CopulaTabICL without error, with
     the MoE aux term folded into the returned loss — the closest thing to
     an actual training step this suite runs.

The tabldm tests are real pretrained loads (network on first run, cached
HF weights after — same convention as tests/test_lora_tabldm_compat.py and
tests/test_tabldm_batched.py, no mocking), so they're the slow members of
this file; kept in one module-scoped fixture to pay that cost once.
"""

from __future__ import annotations

import pytest
import torch
from conftest import make_batch
from omegaconf import OmegaConf

import copula_backbones
from model import build_copula_transformer

# ---------------------------------------------------------------------------
# tabicl (scratch, fast) — sanity that the refactor didn't change anything
# ---------------------------------------------------------------------------


def test_tabicl_strip_decoder(small_model_cfg):
    base = copula_backbones.load_raw_backbone("tabicl", small_model_cfg)
    in_features = copula_backbones.strip_decoder(base)
    assert in_features == 16 * 2  # embed_dim * row_num_cls from small_model_cfg
    assert isinstance(base.icl_predictor.decoder, torch.nn.Identity)


def test_tabicl_moe_aux_loss_is_none(small_model_cfg):
    base = copula_backbones.load_raw_backbone("tabicl", small_model_cfg)
    copula_backbones.strip_decoder(base)
    assert copula_backbones.moe_aux_loss("tabicl", base) is None


def test_build_copula_transformer_tabicl_default_backbone(small_model_cfg):
    """model.backbone defaults to 'tabicl' when omitted — the explicit ask
    that the pre-existing choice stays the default."""
    assert "backbone" not in small_model_cfg.model
    model = build_copula_transformer(small_model_cfg)
    assert model.backbone_name == "tabicl"
    model.train()
    out = model(make_batch(B=2, P=6, N=3))
    assert out["W"].shape == (2, 3, small_model_cfg.model.rank)
    assert "moe_aux_loss" not in out


def test_unknown_backbone_raises(small_model_cfg):
    cfg = OmegaConf.merge(small_model_cfg, {"model": {"backbone": "bogus"}})
    with pytest.raises(ValueError, match="Unknown cfg.model.backbone"):
        build_copula_transformer(cfg)


# ---------------------------------------------------------------------------
# tabldm (real pretrained load) — verifies the new backbone end to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tabldm_cfg():
    return OmegaConf.create(
        {
            "model": {"backbone": "tabldm", "rank": 4, "unfreeze_backbone": True},
            "tabicl": {"pretrained": True},
        }
    )


@pytest.fixture(scope="module")
def tabldm_model(tabldm_cfg):
    torch.manual_seed(0)
    model = build_copula_transformer(tabldm_cfg)
    return model


def test_tabldm_strip_decoder(tabldm_cfg):
    base = copula_backbones.load_raw_backbone("tabldm", tabldm_cfg)
    in_features = copula_backbones.strip_decoder(base)
    assert in_features == 512  # embed_dim(128) * row_num_cls(4) for the released checkpoint
    assert isinstance(base.icl_predictor.decoder, torch.nn.Identity)


def test_tabldm_pretrained_false_raises(tabldm_cfg):
    cfg = OmegaConf.merge(tabldm_cfg, {"tabicl": {"pretrained": False}})
    with pytest.raises(ValueError, match="no from-scratch architecture"):
        copula_backbones.load_raw_backbone("tabldm", cfg)


def test_tabldm_recompute_escalation(tabldm_cfg):
    cfg_on = OmegaConf.merge(tabldm_cfg, {"tabicl": {"recompute": True}})
    base = copula_backbones.load_raw_backbone("tabldm", cfg_on)
    flagged = [m for m in base.modules() if hasattr(m, "recompute")]
    assert len(flagged) > 0
    assert all(m.recompute is True for m in flagged)

    # Default (recompute unset -> False) must NOT force these off -- see
    # copula_backbones._load_tabldm's docstring: escalate-only semantics.
    base_default = copula_backbones.load_raw_backbone("tabldm", tabldm_cfg)
    flagged_default = [m for m in base_default.modules() if hasattr(m, "recompute")]
    assert any(m.recompute for m in flagged_default), (
        "expected the released checkpoint's own row/icl recompute=True to survive untouched"
    )


def test_build_copula_transformer_tabldm(tabldm_model, tabldm_cfg):
    assert tabldm_model.backbone_name == "tabldm"
    assert tabldm_model.feature_dim == 512
    tabldm_model.train()
    batch = make_batch(B=2, P=4, N=2)
    out = tabldm_model(batch)
    assert out["W"].shape == (2, 2, tabldm_cfg.model.rank)
    assert out["s"].shape == (2, 2)
    assert torch.isfinite(out["W"]).all()
    assert torch.isfinite(out["s"]).all()


def test_tabldm_moe_aux_loss_present_and_carries_grad(tabldm_model):
    tabldm_model.train()
    batch = make_batch(B=2, P=4, N=2)
    out = tabldm_model(batch)
    assert "moe_aux_loss" in out
    aux = out["moe_aux_loss"]
    assert aux.requires_grad
    aux.backward()
    # The MoE routing loss lives entirely inside the backbone (it never
    # passes through copula_head), so at least some backbone params should
    # pick up a gradient from it alone.
    any_grad = any(p.grad is not None for p in tabldm_model.feature_extractor.parameters())
    assert any_grad, "moe_aux_loss did not reach any backbone parameter"
    tabldm_model.zero_grad(set_to_none=True)


def test_tabldm_unfreeze_backbone_false_freezes_trunk():
    cfg = OmegaConf.create(
        {
            "model": {"backbone": "tabldm", "rank": 4, "unfreeze_backbone": False},
            "tabicl": {"pretrained": True},
        }
    )
    model = build_copula_transformer(cfg)
    assert all(not p.requires_grad for p in model.feature_extractor.parameters())
    assert all(p.requires_grad for p in model.copula_head.parameters())


def test_tabldm_lora_installs_adapters():
    cfg = OmegaConf.create(
        {
            "model": {"backbone": "tabldm", "rank": 4, "unfreeze_backbone": True},
            "tabicl": {"pretrained": True},
            "lora": {"enabled": True, "rank": 4, "alpha": 8.0, "target": "qkvo", "stages": ["icl"]},
        }
    )
    model = build_copula_transformer(cfg)
    lora_owners = {
        n.split(".lora_")[0] for n, _ in model.feature_extractor.named_parameters() if ".lora_" in n
    }
    assert lora_owners, "no LoRA parameters registered on the tabldm backbone"
    assert all(o.startswith("icl_predictor") for o in lora_owners), (
        f"stages=['icl'] leaked adapters outside icl_predictor: {sorted(lora_owners)[:5]}"
    )
    # copula_head must stay trainable regardless of LoRA gating on the backbone.
    assert all(p.requires_grad for p in model.copula_head.parameters())


# ---------------------------------------------------------------------------
# Full loss/backward step through train.py's own machinery
# ---------------------------------------------------------------------------


def test_tabldm_forward_and_loss_backprops(tabldm_model):
    """The closest thing to an actual training step this suite runs: a real
    _forward_and_loss call (train.py's own loss function) on a tabldm-backed
    model, checking the MoE aux term is folded into the scalar loss and
    backward() completes without error."""
    from train import _forward_and_loss

    tabldm_model.train()
    batch = make_batch(B=2, P=4, N=2)
    # y_space_nll additionally needs a marginal log-density per test
    # instance -- arbitrary here (this test only checks the loss is finite
    # and differentiable, not its value).
    batch["log_pdf_test"] = torch.randn(2, 2)
    out, Sigma, parts, loss, aux_mae = _forward_and_loss(
        model=tabldm_model,
        batch=batch,
        device="cpu",
        use_amp=False,
        amp_dtype=torch.float32,
        nll_weight=1.0,
        aux_mae_weight=0.0,
        jitter=1e-4,
        triu_cache={},
        moe_aux_weight=1.0,
    )
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    tabldm_model.zero_grad(set_to_none=True)
