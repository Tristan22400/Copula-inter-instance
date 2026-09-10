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
"exaone", "tabpfn"}, mirroring generate_pit_dataset.py's existing validation
for the on-disk pipeline. These tests pin down that behaviour directly,
without needing a GPU or a live-generation training run.

"exaone"/"tabpfn" added 2026-08-31 alongside data_gen.py's generic
marginal_backend override (see conf/data/gp_tasks.yaml's z_train_source
docstring) -- included in the parametrized "known values" cases below, but
not given their own dedicated integration test here since they route through
the same _validate_z_train_source/build_live_train_loader/
build_fixed_live_val_batches call sites already covered by the tabicl cases.
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


@pytest.mark.parametrize("value", ["analytic", "tabicl", "tabicl_split", "exaone", "tabpfn"])
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
    assert set(_VALID_Z_TRAIN_SOURCES) == {"analytic", "tabicl", "tabicl_split", "exaone", "tabpfn"}


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
