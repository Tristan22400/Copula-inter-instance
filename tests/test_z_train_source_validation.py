"""Tests for data.z_train_source validation and the "y_train" source.

validate_z_train_source rejects unknown values (e.g. "tabicl-split") at
every call site before other requirements are checked. "y_train" uses the
z-scored target as z_train, leaves z_test/log_pdf_test analytic, and is
rejected by the on-disk generator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from omegaconf import OmegaConf

from copula_inter.backend_registry import Z_TRAIN_SOURCES, validate_z_train_source
from copula_inter.live_dataset import (
    build_fixed_live_val_batches,
    build_live_train_loader,
)
from copula_inter.train_setup import _reserve_gpu_headroom_for_live_tabicl

if TYPE_CHECKING:
    from omegaconf import DictConfig


@pytest.mark.parametrize("value", ["analytic", "tabicl", "tabicl_split", "exaone", "tabpfn", "tabldm", "y_train"])
def testvalidate_z_train_source_accepts_known_values(value: str) -> None:
    validate_z_train_source(value)  # must not raise


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
def testvalidate_z_train_source_rejects_unknown_values(value: str) -> None:
    with pytest.raises(ValueError, match="Unknown data.z_train_source"):
        validate_z_train_source(value)


def test_valid_z_train_sources_matches_documented_set() -> None:
    # Z_TRAIN_SOURCES matches the documented values.
    assert set(Z_TRAIN_SOURCES) == {
        "analytic",
        "tabicl",
        "tabicl_split",
        "exaone",
        "tabpfn",
        "tabldm",
        "y_train",
    }


# Each call site raises on an unknown value before other checks (CPU, no TabICL config).


def _cfg_with_bad_z_train_source() -> DictConfig:
    return OmegaConf.create({"data": {"z_train_source": "tabicl-split"}})


def test_build_live_train_loader_raises_on_typo() -> None:
    cfg = _cfg_with_bad_z_train_source()
    t = OmegaConf.create({"batch_size": 4})
    with pytest.raises(ValueError, match="Unknown data.z_train_source"):
        build_live_train_loader(cfg, t, device="cpu")


def test_build_fixed_live_val_batches_raises_on_typo() -> None:
    cfg = _cfg_with_bad_z_train_source()
    t = OmegaConf.create({"val_episodes": 4, "batch_size": 4})
    with pytest.raises(ValueError, match="Unknown data.z_train_source"):
        build_fixed_live_val_batches(cfg, t, device="cpu")


def test_reserve_gpu_headroom_raises_on_typo() -> None:
    cfg = _cfg_with_bad_z_train_source()
    t = OmegaConf.create({})
    with pytest.raises(ValueError, match="Unknown data.z_train_source"):
        _reserve_gpu_headroom_for_live_tabicl(cfg, t, device="cpu")


# "y_train": z_train is the z-scored target; z_test/log_pdf_test stay analytic.


def test_raw_y_override_z_train_matches_scaled_y_train(small_cfg: DictConfig) -> None:
    import torch
    from omegaconf import OmegaConf as OC

    from copula_inter.data_gen import generate_gp_batch

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


def test_raw_y_override_leaves_z_test_at_analytic_oracle(small_cfg: DictConfig) -> None:
    import torch
    from omegaconf import OmegaConf as OC

    from copula_inter.data_gen import generate_gp_batch

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


def test_generate_pit_dataset_rejects_y_train_on_disk() -> None:
    from copula_inter.generate_pit_dataset import _reject_disk_unsupported_z_train_source

    with pytest.raises(ValueError, match="only supported under training.live_generation"):
        _reject_disk_unsupported_z_train_source("y_train")
    _reject_disk_unsupported_z_train_source("analytic")  # must not raise


def test_missing_z_train_source_defaults_to_tabicl() -> None:
    from copula_inter.backend_registry import z_train_source

    assert z_train_source(OmegaConf.create({"data": {}})) == "tabicl"
    assert z_train_source(OmegaConf.create({})) == "tabicl"
    assert z_train_source(OmegaConf.create({"data": {"z_train_source": "analytic"}})) == "analytic"
