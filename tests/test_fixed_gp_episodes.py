"""generate_fixed_gp_batch: fixed evaluation episodes that do not depend on the machine or the marginal.

They are synthesized on CPU in float64 and returned in float32; only the
marginal PIT may run elsewhere (pit_device). Uses the shipped data config, so
the feature warps and the kernel-hidden transform run in float64 too.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import DictConfig

from copula_inter.config_path import compose_config, config_dir
from copula_inter.data.gp_episodes import generate_fixed_gp_batch

_XY = ("x_norm_train", "x_norm_test", "y_train", "y_test", "x_kernel_train", "x_kernel_test")


@pytest.fixture
def cfg() -> DictConfig:
    cfg = compose_config(config_dir(__file__), "config")
    cfg.data.P_min, cfg.data.P_max = 24, 40
    cfg.data.N_min, cfg.data.N_max = 16, 24
    cfg.seed = 20260723
    return cfg


def _assert_same_xy(a: list[dict], b: list[dict], tol: float = 0.0) -> None:
    assert len(a) == len(b)
    for ea, eb in zip(a, b, strict=True):
        for key in _XY:
            torch.testing.assert_close(ea[key], eb[key], rtol=tol, atol=tol)


def test_fields_are_float32_and_cached_factors_on_cpu(cfg: DictConfig) -> None:
    episodes = generate_fixed_gp_batch(cfg, 6, return_kernel_metadata=True)
    assert len(episodes) == 6
    assert torch.get_default_dtype() == torch.float32, "synthesis leaked its float64 default dtype"
    for ep in episodes:
        for key, value in ep.items():
            if torch.is_tensor(value) and value.is_floating_point():
                assert value.dtype == torch.float32, key
                assert value.device.type == "cpu", key
                assert torch.isfinite(value).all(), key


def test_same_seed_same_episodes(cfg: DictConfig) -> None:
    first = generate_fixed_gp_batch(cfg, 6, return_kernel_metadata=True)
    _assert_same_xy(first, generate_fixed_gp_batch(cfg, 6, return_kernel_metadata=True))


def test_episodes_do_not_depend_on_the_thread_count(cfg: DictConfig) -> None:
    """Float64 synthesis: a different BLAS blocking (thread count, as on another CPU) moves the float32 episodes by rounding at most."""
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        one = generate_fixed_gp_batch(cfg, 6, return_kernel_metadata=True)
        torch.set_num_threads(max(2, min(8, previous)))
        many = generate_fixed_gp_batch(cfg, 6, return_kernel_metadata=True)
    finally:
        torch.set_num_threads(previous)
    _assert_same_xy(one, many, tol=1e-6)


def test_episodes_do_not_depend_on_the_z_train_source(cfg: DictConfig) -> None:
    """The PIT replaces z only: the analytic source and the raw-y source see the same (x, y)."""
    analytic = generate_fixed_gp_batch(cfg, 6, return_kernel_metadata=True)
    raw_y = generate_fixed_gp_batch(cfg, 6, return_kernel_metadata=True, raw_y_override=True)
    _assert_same_xy(analytic, raw_y)
    assert not torch.allclose(analytic[0]["z_train"], raw_y[0]["z_train"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_episodes_do_not_depend_on_the_pit_device(cfg: DictConfig) -> None:
    on_cpu = generate_fixed_gp_batch(cfg, 6, return_kernel_metadata=True)
    on_gpu = generate_fixed_gp_batch(cfg, 6, pit_device="cuda", return_kernel_metadata=True)
    _assert_same_xy(on_cpu, on_gpu)
    for a, b in zip(on_cpu, on_gpu, strict=True):
        assert b["_L_ff"].device.type == "cpu"
        torch.testing.assert_close(a["z_test"], b["z_test"], rtol=1e-5, atol=1e-5)
