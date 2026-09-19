"""
test_z_train_source_validation.py — Regression test for
live_dataset.py::_validate_z_train_source.

Root-caused 2026-08-24: every `z_train_source in ("tabicl", "tabicl_split")`
gate in live_dataset.py (and train.py::_reserve_gpu_headroom_for_live_tabicl)
checks for the underscore spelling only. A data.z_train_source=tabicl-split
(hyphen) config typo silently fails that membership check, so
tabicl_live_enabled comes out False and the whole run trains as pure
analytic -- no TabICL model is ever loaded, and nothing warns or errors.
Confirmed live: two production runs launched with data.z_train_source=
analytic and data.z_train_source=tabicl-split produced bit-identical
train/loss_ema trajectories from step 0.

_validate_z_train_source closes this by raising immediately on any
data.z_train_source value outside {"analytic", "tabicl", "tabicl_split",
"exaone", "tabpfn", "tabldm"}, mirroring generate_pit_dataset.py's existing
validation for the on-disk pipeline. These tests pin down that behaviour directly,
without needing a GPU or a live-generation training run.

"exaone"/"tabpfn" added 2026-08-31, "tabldm" 2026-09-10, alongside
data_gen.py's generic marginal_backend override (see conf/data/gp_tasks.yaml's
z_train_source docstring) -- included in the parametrized "known values" cases
below, but not given their own dedicated integration test here since they
route through the same _validate_z_train_source/build_live_train_loader/
build_fixed_live_val_batches call sites already covered by the tabicl cases.
Each backend's own numerical correctness is covered by its
tests/test_*_batched.py equivalence test instead.

"y_train" added 2026-09-19 (a no-PIT control/ablation arm -- see
data_gen.py::_generate_gp_batch_raw's raw_y_override and live_dataset.py's
_RAW_Y_SOURCES): included in the parametrized "known values" cases, plus a
dedicated numerical test below (unlike exaone/tabpfn/tabldm, it has no
marginal model to cross-check against a per-episode fallback -- the only
thing to pin down is the raw-y-space arithmetic itself and that z_test/
log_pdf_test are left untouched), and a guard test confirming
generate_pit_dataset.py's on-disk path rejects it rather than silently
falling back to plain analytic generation.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from live_dataset import (
    _VALID_Z_TRAIN_SOURCES,
    _validate_z_train_source,
    build_fixed_live_val_batches,
    build_live_train_loader,
)
from train import _reserve_gpu_headroom_for_live_tabicl


@pytest.mark.parametrize(
    "value", ["analytic", "tabicl", "tabicl_split", "exaone", "tabpfn", "tabldm", "y_train"]
)
def test_validate_z_train_source_accepts_known_values(value):
    _validate_z_train_source(value)  # must not raise


@pytest.mark.parametrize(
    "value",
    [
        "tabicl-split",  # the actual typo that caused the silent no-op
        "Tabicl",
        "oracle",
        "",
        "tabicl_splitt",
    ],
)
def test_validate_z_train_source_rejects_unknown_values(value):
    with pytest.raises(ValueError, match="Unknown data.z_train_source"):
        _validate_z_train_source(value)


def test_valid_z_train_sources_matches_documented_set():
    # Guards against _VALID_Z_TRAIN_SOURCES silently drifting out of sync
    # with conf/data/gp_tasks.yaml's documented z_train_source values.
    assert set(_VALID_Z_TRAIN_SOURCES) == {
        "analytic", "tabicl", "tabicl_split", "exaone", "tabpfn", "tabldm", "y_train",
    }


# ---------------------------------------------------------------------------
# Integration-level: each real call site must raise on the typo BEFORE any
# of its other requirements (a resolvable TabICL checkpoint, device="cuda",
# ...) are even checked -- so this must reproduce with device="cpu" and no
# tabicl.* config at all, no GPU required.
# ---------------------------------------------------------------------------


def _cfg_with_bad_z_train_source():
    return OmegaConf.create({"data": {"z_train_source": "tabicl-split"}})


def test_build_live_train_loader_raises_on_typo():
    cfg = _cfg_with_bad_z_train_source()
    t = OmegaConf.create({"batch_size": 4})
    with pytest.raises(ValueError, match="Unknown data.z_train_source"):
        build_live_train_loader(cfg, t, device="cpu")


def test_build_fixed_live_val_batches_raises_on_typo():
    cfg = _cfg_with_bad_z_train_source()
    t = OmegaConf.create({"val_episodes": 4, "batch_size": 4})
    with pytest.raises(ValueError, match="Unknown data.z_train_source"):
        build_fixed_live_val_batches(cfg, t, device="cpu")


def test_reserve_gpu_headroom_raises_on_typo():
    cfg = _cfg_with_bad_z_train_source()
    t = OmegaConf.create({})
    with pytest.raises(ValueError, match="Unknown data.z_train_source"):
        _reserve_gpu_headroom_for_live_tabicl(cfg, t, device="cpu")


# ---------------------------------------------------------------------------
# "y_train" (raw_y_override): the ICL context becomes the raw, per-episode
# z-scored target instead of any PIT transform, while z_test/log_pdf_test
# stay the exact analytic oracle -- pin both halves of that contract.
# ---------------------------------------------------------------------------


def test_raw_y_override_z_train_matches_scaled_y_train(small_cfg):
    import torch
    from omegaconf import OmegaConf as OC

    from data_gen import generate_gp_batch

    cfg = OC.create(OC.to_container(small_cfg, resolve=True))
    cfg.data.P_min = cfg.data.P_max = 8
    cfg.data.N_min = cfg.data.N_max = 6
    cfg.data.kernel = "rbf"
    torch.manual_seed(0)
    episodes = generate_gp_batch(cfg, 6, "cpu", raw_y_override=True)
    for ep in episodes:
        y_train = ep["y_train"]
        expected = (y_train - y_train.mean()) / y_train.std().clamp(min=1e-8)
        assert torch.allclose(ep["z_train"], expected, atol=1e-5)


def test_raw_y_override_leaves_z_test_at_analytic_oracle(small_cfg):
    import torch
    from omegaconf import OmegaConf as OC

    from data_gen import generate_gp_batch

    cfg = OC.create(OC.to_container(small_cfg, resolve=True))
    cfg.data.P_min = cfg.data.P_max = 8
    cfg.data.N_min = cfg.data.N_max = 6
    cfg.data.kernel = "rbf"

    torch.manual_seed(0)
    raw_episodes = generate_gp_batch(cfg, 6, "cpu", raw_y_override=True)
    torch.manual_seed(0)
    analytic_episodes = generate_gp_batch(cfg, 6, "cpu", raw_y_override=False)

    for raw_ep, an_ep in zip(raw_episodes, analytic_episodes):
        assert torch.allclose(raw_ep["z_test"], an_ep["z_test"])
        assert torch.allclose(raw_ep["log_pdf_test"], an_ep["log_pdf_test"])
        # The whole point of the ablation: the context changes...
        assert not torch.allclose(raw_ep["z_train"], an_ep["z_train"])


def test_generate_pit_dataset_rejects_y_train_on_disk():
    from generate_pit_dataset import _reject_disk_unsupported_z_train_source

    with pytest.raises(ValueError, match="only supported under training.live_generation"):
        _reject_disk_unsupported_z_train_source("y_train")
    _reject_disk_unsupported_z_train_source("analytic")  # must not raise
