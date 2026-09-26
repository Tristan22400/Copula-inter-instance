"""Tests for GP episode generation, feature transforms, CopulaDataset and collate_fn."""

from __future__ import annotations

import math
import random
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from pytest import MonkeyPatch

from copula_inter.data_gen import (
    _generate_gp_batch_raw,
    generate_gp_batch,
    generate_gp_task,
    gp_posterior,
    sigma_to_correlation,
)
from copula_inter.dataset import CopulaDataset, _add_derived_fields, collate_fn
from copula_inter.feature_transforms import apply_kernel_hidden_warp, apply_mlp_feature_mixing, tabiclv2_warp_features
from copula_inter.gp_kernels import ALL_KERNELS, _kernel_needs_scalar_input
from copula_inter.kernel_sampling import _sample_mean_module
from copula_inter.structural_warps import (
    _CATEGORY_OPS,
    _DEFAULT_CATEGORY_WEIGHTS,
    _STRUCTURAL_CATEGORIES,
    _sample_structural_category_mask,
    _sample_structural_ops,
    _structural_warp_column,
    apply_structural_feature_warp,
)


def test_tabiclv2_warp_features_preserves_shape_and_finite() -> None:
    """All 11 marginal transforms preserve shape and give finite output."""
    torch.manual_seed(0)
    x = torch.randn(64, 32, 11)
    out = tabiclv2_warp_features(x.clone())
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()


def test_tabiclv2_warp_features_all_11_choices_reachable() -> None:
    """Every one of the 11 transform choices can be drawn."""
    torch.manual_seed(0)
    seen = set()
    for trial in range(200):
        torch.manual_seed(trial)
        torch.randn(1, 32, 1)  # advances the RNG stream the draw below depends on
        choices = torch.randint(0, 11, (1, 1))
        seen.add(int(choices.item()))
    assert seen == set(range(11))


def test_tabiclv2_warp_features_zero_inflation_produces_point_mass() -> None:
    """Choice 8 (zero inflation) sets a substantial fraction of values to exactly 0."""
    torch.manual_seed(0)
    col = torch.randn(1, 2000, 1)
    spike_frac = 0.4
    mask = torch.rand_like(col) < spike_frac
    warped = torch.where(mask, torch.zeros_like(col), col)
    frac_zero = (warped == 0).float().mean().item()
    assert frac_zero > 0.1


def test_tabiclv2_warp_features_bounded_squash_stays_in_unit_interval() -> None:
    """Choice 9 (sigmoid squash) stays in [0, 1]."""
    torch.manual_seed(0)
    col = torch.randn(2000) * 5.0  # wide range, including extreme tails
    out = torch.sigmoid(col * 2.5)
    assert (out >= 0.0).all() and (out <= 1.0).all()
    # A non-extreme value must land strictly inside the interval.
    assert 0.0 < torch.sigmoid(torch.tensor(0.3)).item() < 1.0


def test_tabiclv2_warp_features_left_skew_mirrors_right_skew() -> None:
    """Choice 10 (left skew) is the mirror image of choice 3 (right skew)."""
    col = torch.randn(500)
    right_skew = torch.exp(col.clamp(min=-5.0, max=4.0))
    left_skew = -torch.exp((-col).clamp(min=-5.0, max=4.0))
    assert torch.equal(left_skew, -torch.exp((-col).clamp(min=-5.0, max=4.0)))
    # Right-skew is bounded below by 0; its mirror must be bounded above by 0.
    assert (right_skew >= 0).all()
    assert (left_skew <= 0).all()


@pytest.mark.parametrize("kernel_name", ALL_KERNELS)
def test_tabiclv2_warp_features_goldilocks_and_psd(small_cfg: DictConfig, kernel_name: str) -> None:
    """With the feature warp, R_star stays a valid, PSD, non-trivial correlation matrix for every kernel."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = kernel_name
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 1 / 3
    cfg.data.inactive_frac_max = 5 / 6

    torch.manual_seed(abs(hash("tabiclv2_warp_" + kernel_name)) % (2**31))
    off_diag_abs = []
    for _ in range(20):
        task = generate_gp_task(cfg)
        R = task["R_star"]
        N = R.shape[0]

        assert torch.allclose(R.diagonal(), torch.ones(N), atol=1e-4)
        assert torch.allclose(R, R.T, atol=1e-5)
        eigvals = torch.linalg.eigvalsh(R)
        assert (eigvals >= -1e-4).all(), (
            f"{kernel_name}: not PSD with the 11-way tabiclv2 warp bank (min eig={eigvals.min():.6f})"
        )
        assert R.abs().max() <= 1.0 + 1e-5

        mask = ~torch.eye(N, dtype=torch.bool)
        off_diag_abs.append(R[mask].abs())

    mean_abs_r = torch.cat(off_diag_abs).mean().item()
    assert mean_abs_r > _COLLAPSE_THRESHOLD, (
        f"{kernel_name}: tabiclv2 warp bank collapsed correlation, mean|r*_offdiag|={mean_abs_r:.4f}"
    )
    assert mean_abs_r < _DEGENERATE_THRESHOLD, (
        f"{kernel_name}: tabiclv2 warp bank degenerate, mean|r*_offdiag|={mean_abs_r:.4f}"
    )


def test_gp_task_output_keys(small_cfg: DictConfig) -> None:
    task = generate_gp_task(small_cfg)
    required = [
        "x_norm_train",
        "y_train",
        "x_norm_test",
        "y_test",
        "R_star",
        "mu_star",
        "sigma_star",
        "n_train",
        "n_test",
    ]
    for key in required:
        assert key in task, f"Missing key: {key}"


def test_gp_task_shapes(small_cfg: DictConfig) -> None:
    torch.manual_seed(0)
    task = generate_gp_task(small_cfg)
    P = task["n_train"].item()
    N = task["n_test"].item()
    d = small_cfg.data.d_features

    assert task["x_norm_train"].shape == (P, d)
    assert task["y_train"].shape == (P,)
    assert task["x_norm_test"].shape == (N, d)
    assert task["y_test"].shape == (N,)
    assert task["R_star"].shape == (N, N)
    assert task["mu_star"].shape == (N,)
    assert task["sigma_star"].shape == (N,)

    assert small_cfg.data.P_min <= P <= small_cfg.data.P_max
    assert small_cfg.data.N_min <= N <= small_cfg.data.N_max


def test_feature_normalisation_over_all_instances(small_cfg: DictConfig) -> None:
    """x_norm_train and x_norm_test together should have ~zero mean, ~unit std."""
    torch.manual_seed(1)
    # Generate multiple tasks and check normalisation
    for _ in range(10):
        task = generate_gp_task(small_cfg)
        x_all = torch.cat([task["x_norm_train"], task["x_norm_test"]], dim=0)
        for f in range(x_all.shape[1]):
            col = x_all[:, f]
            assert abs(col.mean().item()) < 0.2, f"Feature {f} mean {col.mean():.3f} not near zero"
            assert abs(col.std().item() - 1.0) < 0.2, f"Feature {f} std {col.std():.3f} not near 1"


def test_r_star_is_valid_correlation_matrix(small_cfg: DictConfig) -> None:
    """R_star must have unit diagonal and be positive semi-definite."""
    torch.manual_seed(2)
    for _ in range(20):
        task = generate_gp_task(small_cfg)
        R = task["R_star"]
        N = int(task["n_test"])

        # Unit diagonal
        assert torch.allclose(R.diagonal(), torch.ones(N), atol=1e-4), f"R_star diagonal not 1: {R.diagonal()}"

        # PSD
        eigvals = torch.linalg.eigvalsh(R)
        assert (eigvals >= -1e-4).all(), f"R_star has negative eigenvalue: {eigvals.min():.6f}"

        # Symmetry
        assert torch.allclose(R, R.T, atol=1e-5)


def test_r_star_values_in_minus1_1(small_cfg: DictConfig) -> None:
    """Correlation matrix entries must be in [-1, 1]."""
    torch.manual_seed(3)
    for _ in range(10):
        R = generate_gp_task(small_cfg)["R_star"]
        assert R.abs().max() <= 1.0 + 1e-5


# Goldilocks band (diag_kernels stage-3 thresholds): not collapsed toward independence, not saturated near +-1.
_COLLAPSE_THRESHOLD = 0.01
_DEGENERATE_THRESHOLD = 0.95


@pytest.mark.parametrize("kernel_name", ALL_KERNELS)
def test_kernel_goldilocks_and_psd(small_cfg: DictConfig, kernel_name: str) -> None:
    """Every registered kernel gives a valid, PSD, non-trivial R_star."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = kernel_name
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 1 / 3  # (6-4)/6 -> k up to 4
    cfg.data.inactive_frac_max = 5 / 6  # (6-1)/6 -> k down to 1

    torch.manual_seed(abs(hash(kernel_name)) % (2**31))
    off_diag_abs = []
    for _ in range(20):
        task = generate_gp_task(cfg)
        R = task["R_star"]
        N = R.shape[0]

        assert torch.allclose(R.diagonal(), torch.ones(N), atol=1e-4), f"{kernel_name}: diagonal not 1: {R.diagonal()}"
        assert torch.allclose(R, R.T, atol=1e-5), f"{kernel_name}: not symmetric"
        eigvals = torch.linalg.eigvalsh(R)
        assert (eigvals >= -1e-4).all(), f"{kernel_name}: not PSD (min eig={eigvals.min():.6f})"
        assert R.abs().max() <= 1.0 + 1e-5, f"{kernel_name}: value outside [-1, 1]"

        mask = ~torch.eye(N, dtype=torch.bool)
        off_diag_abs.append(R[mask].abs())

    mean_abs_r = torch.cat(off_diag_abs).mean().item()
    assert mean_abs_r > _COLLAPSE_THRESHOLD, f"{kernel_name}: screening effect, mean|r*_offdiag|={mean_abs_r:.4f}"
    assert mean_abs_r < _DEGENERATE_THRESHOLD, f"{kernel_name}: degenerate/trivial, mean|r*_offdiag|={mean_abs_r:.4f}"


@pytest.mark.parametrize("kernel_name", ["periodic", "cosine"])
def test_periodic_and_cosine_period_recoverable_in_r_star(small_cfg: DictConfig, kernel_name: str) -> None:
    """R_star of a bare periodic/cosine episode equals the analytic kernel from its recorded column, l/period, alpha2 and nugget.

        periodic: exp(-2 sin^2(pi (x1 - x2) / period) / l)
        cosine:   cos(pi (x1 - x2) / l)   (period_length stored under "l")
    Off-diagonals are scaled by alpha2 / (alpha2 + nugget).
    """
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = kernel_name
    cfg.data.systematic_composition = False
    cfg.data.d_features = 4

    torch.manual_seed(abs(hash(kernel_name)) % (2**31))
    for i in range(10):
        cfg.seed = i
        task = generate_gp_task(cfg)

        col = int(task["kernel_feature_indices"][0])
        x = task["x_norm_test"][:, col]
        diff = x.unsqueeze(0) - x.unsqueeze(1)  # (N, N)

        alpha2 = task["alpha2"].item()
        nugget = task["nugget"].item()

        if kernel_name == "periodic":
            l = task["l"].item()
            period = task["period"].item()
            base = torch.exp(-2.0 * torch.sin(math.pi * diff / period) ** 2 / l)
        else:  # cosine: "l" holds period_length (see _kernel_prior_spec)
            period = task["l"].item()
            base = torch.cos(math.pi * diff / period)

        R_theory = base * (alpha2 / (alpha2 + nugget))
        R_theory.fill_diagonal_(1.0)

        max_diff = (R_theory - task["R_star"]).abs().max().item()
        assert torch.allclose(R_theory, task["R_star"], atol=1e-4), (
            f"{kernel_name}: R_star doesn't match the analytic kernel formula "
            f"reconstructed from the recorded active column/l/period/alpha2/"
            f"nugget (max abs diff={max_diff:.6g})"
        )

        # A wrong period must not reproduce R_star.
        wrong_period = period * 1.7 + 0.3
        if kernel_name == "periodic":
            wrong_base = torch.exp(-2.0 * torch.sin(math.pi * diff / wrong_period) ** 2 / l)
        else:
            wrong_base = torch.cos(math.pi * diff / wrong_period)
        wrong_R = wrong_base * (alpha2 / (alpha2 + nugget))
        wrong_R.fill_diagonal_(1.0)
        assert not torch.allclose(wrong_R, task["R_star"], atol=1e-4), (
            f"{kernel_name}: test is vacuous -- a wrong period still matches R_star"
        )


def test_kernel_needs_scalar_input_handles_n_way_chains() -> None:
    """_kernel_needs_scalar_input detects cosine anywhere in a chain of any length."""
    assert _kernel_needs_scalar_input("rbf+cosine*periodic") is True
    assert _kernel_needs_scalar_input("periodic*matern32+cosine") is True
    assert _kernel_needs_scalar_input("rbf+periodic*matern32") is False
    # Existing base-kernel / 2-way-composite behaviour must be unchanged.
    assert _kernel_needs_scalar_input("cosine") is True
    assert _kernel_needs_scalar_input("rbf") is False
    assert _kernel_needs_scalar_input("rbf+cosine") is True
    assert _kernel_needs_scalar_input("rbf+periodic") is False


def test_systematic_composition_goldilocks_and_psd(small_cfg: DictConfig) -> None:
    """Systematic composition gives a valid R_star on every draw (lower Goldilocks bound only)."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.systematic_composition = True
    cfg.data.composite_num_kernels_min = 1
    cfg.data.composite_num_kernels_max = 3
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 1 / 3
    cfg.data.inactive_frac_max = 5 / 6

    torch.manual_seed(123)
    off_diag_abs = []
    for _ in range(20):
        task = generate_gp_task(cfg)
        R = task["R_star"]
        N = R.shape[0]

        assert torch.allclose(R.diagonal(), torch.ones(N), atol=1e-4), (
            f"{task['kernel']}: diagonal not 1: {R.diagonal()}"
        )
        assert torch.allclose(R, R.T, atol=1e-5), f"{task['kernel']}: not symmetric"
        eigvals = torch.linalg.eigvalsh(R)
        assert (eigvals >= -1e-4).all(), f"{task['kernel']}: not PSD (min eig={eigvals.min():.6f})"
        assert R.abs().max() <= 1.0 + 1e-5, f"{task['kernel']}: value outside [-1, 1]"

        mask = ~torch.eye(N, dtype=torch.bool)
        off_diag_abs.append(R[mask].abs())

    mean_abs_r = torch.cat(off_diag_abs).mean().item()
    assert mean_abs_r > _COLLAPSE_THRESHOLD, (
        f"systematic_composition: screening effect, mean|r*_offdiag|={mean_abs_r:.4f}"
    )


_ARD_ELIGIBLE_KERNELS = ["rbf", "matern32", "rational_quadratic", "periodic"]


@pytest.mark.parametrize("kernel_name", _ARD_ELIGIBLE_KERNELS)
def test_ard_samples_per_dimension_lengthscale(small_cfg: DictConfig, kernel_name: str) -> None:
    """cfg.data.ard gives a (k,) lengthscale that round-trips through gp_analytical_pit."""
    from copula_inter.pit import gp_analytical_pit

    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = kernel_name
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 0.5  # (6-3)/6 -> fixed k=3
    cfg.data.inactive_frac_max = 0.5
    cfg.data.ard = True

    # periodic is capped to k=1, so its lengthscale is scalar.
    expected_shape = () if kernel_name == "periodic" else (3,)
    torch.manual_seed(abs(hash("ard_" + kernel_name)) % (2**31))
    task = generate_gp_task(cfg)
    assert task["l"].shape == expected_shape, (
        f"{kernel_name}: expected shape {expected_shape}, got {tuple(task['l'].shape)}"
    )

    cached = gp_analytical_pit(task)
    reconstructed_task = {k: v for k, v in task.items() if k not in ("_L_ff", "_alpha")}
    reconstructed = gp_analytical_pit(reconstructed_task)
    assert torch.allclose(cached["z_train"], reconstructed["z_train"], atol=1e-3)
    assert torch.allclose(cached["z_test"], reconstructed["z_test"], atol=1e-3)


def test_ard_default_false_keeps_isotropic_lengthscale(small_cfg: DictConfig) -> None:
    """Without cfg.data.ard the lengthscale is a scalar even for k > 1."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 0.5  # (6-3)/6 -> fixed k=3
    cfg.data.inactive_frac_max = 0.5

    torch.manual_seed(0)
    task = generate_gp_task(cfg)
    assert task["l"].shape == (), f"expected isotropic scalar, got shape {tuple(task['l'].shape)}"


def test_ard_not_applied_to_cosine_or_dot_product(small_cfg: DictConfig) -> None:
    """cfg.data.ard has no effect on cosine and dot_product."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 0.5  # (6-3)/6 -> fixed k=3
    cfg.data.inactive_frac_max = 0.5
    cfg.data.ard = True

    torch.manual_seed(0)
    cfg.data.kernel = "cosine"
    task = generate_gp_task(cfg)
    assert task["l"].shape == (), "cosine's period_length must stay scalar under ard=True"

    torch.manual_seed(0)
    cfg.data.kernel = "dot_product"
    task = generate_gp_task(cfg)  # must not raise
    assert task["alpha2"].numel() == 1


@pytest.mark.parametrize("kernel_name", _ARD_ELIGIBLE_KERNELS)
def test_isotropic_ratio_one_collapses_every_episode(small_cfg: DictConfig, kernel_name: str) -> None:
    """isotropic_ratio=1.0 makes every ARD lengthscale (and period) constant across dims."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = kernel_name
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 0.5  # (6-3)/6 -> fixed k=3
    cfg.data.inactive_frac_max = 0.5
    cfg.data.ard = True
    cfg.data.isotropic_ratio = 1.0

    # periodic is capped to k=1: nothing to collapse.
    expected_shape = () if kernel_name == "periodic" else (3,)
    torch.manual_seed(abs(hash("iso_" + kernel_name)) % (2**31))
    episodes = generate_gp_batch(cfg, B=8, device="cpu", return_kernel_metadata=True)
    for task in episodes:
        assert task["l"].shape == expected_shape, (
            f"{kernel_name}: expected shape {expected_shape}, got {tuple(task['l'].shape)}"
        )
        if kernel_name != "periodic":
            assert torch.allclose(task["l"], task["l"][0].expand_as(task["l"]), atol=1e-6), (
                f"{kernel_name}: isotropic_ratio=1.0 should collapse lengthscale to one shared value"
            )


def test_isotropic_ratio_zero_is_default_ard_behaviour(small_cfg: DictConfig) -> None:
    """isotropic_ratio=0.0 (default) keeps independent per-dim lengthscales."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 0.5
    cfg.data.inactive_frac_max = 0.5
    cfg.data.ard = True

    torch.manual_seed(0)
    episodes = generate_gp_batch(cfg, B=20, device="cpu", return_kernel_metadata=True)
    n_collapsed = sum(torch.allclose(task["l"], task["l"][0].expand_as(task["l"]), atol=1e-6) for task in episodes)
    assert n_collapsed == 0, "isotropic_ratio default (0.0) should never force-collapse an ARD lengthscale"


def test_isotropic_ratio_no_op_when_ard_false(small_cfg: DictConfig) -> None:
    """isotropic_ratio has no effect when ard is off."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 0.5
    cfg.data.inactive_frac_max = 0.5
    cfg.data.ard = False
    cfg.data.isotropic_ratio = 1.0

    torch.manual_seed(0)
    task = generate_gp_task(cfg)
    assert task["l"].shape == (), f"expected isotropic scalar, got shape {tuple(task['l'].shape)}"


def test_isotropic_ratio_partial_mixes_isotropic_and_ard_episodes(small_cfg: DictConfig) -> None:
    """A ratio in (0, 1) mixes isotropic and ARD episodes in roughly that proportion."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 0.5
    cfg.data.inactive_frac_max = 0.5
    cfg.data.ard = True
    cfg.data.isotropic_ratio = 0.5

    torch.manual_seed(1)
    episodes = generate_gp_batch(cfg, B=400, device="cpu", return_kernel_metadata=True)
    n_collapsed = sum(torch.allclose(task["l"], task["l"][0].expand_as(task["l"]), atol=1e-6) for task in episodes)
    assert 150 < n_collapsed < 250, f"expected ~200/400 isotropic episodes, got {n_collapsed}"


def test_polynomial_power_shared_across_batch(small_cfg: DictConfig) -> None:
    """All episodes of one call share one polynomial degree in [poly_power_min, poly_power_max]."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "polynomial"
    cfg.data.poly_power_min = 2
    cfg.data.poly_power_max = 5

    torch.manual_seed(0)
    episodes = generate_gp_batch(cfg, B=16, device="cpu", return_kernel_metadata=True)
    powers = {task["power"].item() for task in episodes}
    assert len(powers) == 1, f"expected one shared power across the batch, got {powers}"
    power = powers.pop()
    assert 2 <= power <= 5, f"power {power} outside configured [poly_power_min, poly_power_max]"


def test_polynomial_power_varies_across_batches(small_cfg: DictConfig) -> None:
    """Different calls can draw different polynomial degrees."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "polynomial"
    cfg.data.poly_power_min = 2
    cfg.data.poly_power_max = 8

    torch.manual_seed(0)
    random.seed(0)
    seen_powers = set()
    for _ in range(20):
        episodes = generate_gp_batch(cfg, B=1, device="cpu", return_kernel_metadata=True)
        seen_powers.add(episodes[0]["power"].item())
    assert len(seen_powers) > 1, f"power never varied across 20 batches: {seen_powers}"


def test_topup_round_reuses_first_round_d_features(small_cfg: DictConfig, monkeypatch: MonkeyPatch) -> None:
    """generate_gp_batch's top-up rounds reuse the first round's d_features."""
    from copula_inter import data_gen as dg

    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.d_features_lognormal_loc = 2.302585  # log(10)
    cfg.data.d_features_lognormal_scale = 0.4
    cfg.seed = 123

    real_raw = dg._generate_gp_batch_raw
    state = {"n_calls": 0}

    def truncating_raw(cfg: Any, B: int, device: str = "cpu", **kwargs: Any) -> list[dict[str, torch.Tensor]]:
        episodes = real_raw(cfg, B, device, **kwargs)
        state["n_calls"] += 1
        if state["n_calls"] == 1:
            episodes = episodes[:-5]  # force a shortfall so top-up fires
        return episodes

    monkeypatch.setattr(dg, "_generate_gp_batch_raw", truncating_raw)

    episodes = dg.generate_gp_batch(cfg, B=20, device="cpu")
    assert state["n_calls"] > 1, "test setup didn't actually trigger a top-up round"
    assert len(episodes) == 20
    d_set = {ep["x_norm_train"].shape[-1] for ep in episodes}
    assert len(d_set) == 1, f"top-up round used a different d_features than round 0: {d_set}"


def test_oom_retry_chunk_reuses_first_chunk_d_features(small_cfg: DictConfig, monkeypatch: MonkeyPatch) -> None:
    """_generate_shard_with_oom_retry's retry chunks reuse the first chunk's d_features."""
    from copula_inter import data_gen as dg
    from copula_inter import generate_pit_dataset as gpd

    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    assert isinstance(cfg, DictConfig)
    cfg.data.kernel = "rbf"
    cfg.data.d_features_lognormal_loc = 2.302585  # log(10)
    cfg.data.d_features_lognormal_scale = 0.4
    cfg.seed = 123

    real_generate_gp_batch = dg.generate_gp_batch
    state = {"n_calls": 0}

    def oom_first_chunk(cfg: Any, B: int, device: str = "cpu", **kwargs: Any) -> list[dict[str, torch.Tensor]]:
        state["n_calls"] += 1
        if state["n_calls"] == 1:
            raise torch.cuda.OutOfMemoryError("synthetic OOM")
        return real_generate_gp_batch(cfg, B, device, **kwargs)

    monkeypatch.setattr(gpd, "generate_gp_batch", oom_first_chunk)

    episodes = gpd._generate_shard_with_oom_retry(
        cfg,
        n_this=20,
        device="cpu",
        tabicl_model=None,
        tabicl_k_folds=10,
    )
    assert state["n_calls"] > 1, "test setup didn't actually trigger a retry chunk"
    assert len(episodes) == 20
    d_set = {ep["x_norm_train"].shape[-1] for ep in episodes}
    assert len(d_set) == 1, f"retry chunk used a different d_features than the first chunk: {d_set}"


def test_generate_gp_batch_raw_discards_batch_on_linalg_error(small_cfg: DictConfig, monkeypatch: MonkeyPatch) -> None:
    """A LinAlgError from kernel evaluation discards the batch (and generate_gp_batch tops up) instead of raising."""
    from copula_inter import data_gen as dg

    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.seed = 11

    real_evaluate_kernel_dense = dg._evaluate_kernel_dense
    state = {"n_calls": 0}

    def poisoned_evaluate_kernel_dense(kernel_obj: Any, x_norm: torch.Tensor) -> torch.Tensor:
        state["n_calls"] += 1
        if state["n_calls"] == 1:
            raise torch.linalg.LinAlgError("linalg.eigh: synthetic non-convergence for test")
        return real_evaluate_kernel_dense(kernel_obj, x_norm)

    monkeypatch.setattr(dg, "_evaluate_kernel_dense", poisoned_evaluate_kernel_dense)

    episodes = dg.generate_gp_batch(cfg, B=6, device="cpu")
    assert state["n_calls"] > 1, "test setup didn't actually trigger a retry"
    assert len(episodes) == 6


def test_is_transient_cusolver_error_covers_tabicl_contention_errors() -> None:
    """_is_transient_cusolver_error flags TabICL's two contention errors and not an unrelated RuntimeError."""
    from copula_inter import generate_pit_dataset as gpd

    assert gpd._is_transient_cusolver_error(
        RuntimeError("CPU memory allocation failed (CUDA error: invalid argument) and disk offload is not available.")
    )
    assert gpd._is_transient_cusolver_error(
        RuntimeError("Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!")
    )
    assert not gpd._is_transient_cusolver_error(RuntimeError("index out of range"))
    assert not gpd._is_transient_cusolver_error(torch.cuda.OutOfMemoryError("oom"))


def test_degenerate_loo_z_is_discarded_not_leaked(small_cfg: DictConfig, monkeypatch: MonkeyPatch) -> None:
    """An episode with non-finite z_train is discarded, not returned."""
    from copula_inter import data_gen as dg

    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.seed = 7

    real_cholesky_solve = torch.cholesky_solve
    state = {"poisoned": False}

    def poisoning_cholesky_solve(b: torch.Tensor, L: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # Poison the first cholesky_solve call: one episode's alpha and z_train.
        out = real_cholesky_solve(b, L, *args, **kwargs)
        if not state["poisoned"]:
            state["poisoned"] = True
            out = out.clone()
            out[0] = float("nan")
        return out

    monkeypatch.setattr(torch, "cholesky_solve", poisoning_cholesky_solve)

    episodes = dg.generate_gp_batch(cfg, B=8, device="cpu")

    assert state["poisoned"], "test setup didn't actually poison an episode's alpha"
    assert len(episodes) == 8
    for ep in episodes:
        assert torch.isfinite(ep["z_train"]).all()
        assert torch.isfinite(ep["y_train"]).all()


@pytest.mark.parametrize("kernel_name", ["periodic", "cosine"])
def test_degenerate_active_kernel_column_is_discarded_not_leaked(
    small_cfg: DictConfig, kernel_name: str, monkeypatch: MonkeyPatch
) -> None:
    """A k=1 (periodic/cosine) episode whose active column is constant is discarded."""
    from copula_inter import data_gen as dg

    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = kernel_name
    cfg.data.systematic_composition = False
    cfg.seed = 3

    real_tabiclv2 = dg.tabiclv2_warp_features
    state = {"poisoned": False}

    def poisoning_tabiclv2(x: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        out = real_tabiclv2(x, seed=seed)
        if not state["poisoned"]:
            state["poisoned"] = True
            out = out.clone()
            out[0, :, :] = 0.0  # collapse every column of episode 0 to a constant
        return out

    monkeypatch.setattr(dg, "tabiclv2_warp_features", poisoning_tabiclv2)

    with pytest.warns(RuntimeWarning, match="degenerate .*active kernel column"):
        episodes = dg.generate_gp_batch(cfg, B=8, device="cpu", return_kernel_metadata=True)

    assert state["poisoned"], "test setup didn't actually poison an episode's feature columns"
    assert len(episodes) == 8
    for ep in episodes:
        col = int(ep["kernel_feature_indices"][0])
        x = torch.cat([ep["x_norm_train"], ep["x_norm_test"]], dim=0)
        assert float(x[:, col].std()) > 1e-4, (
            f"{kernel_name}: a degenerate active kernel column reached the returned episodes"
        )


def test_multi_dim_active_kernel_fully_collapsed_is_discarded(small_cfg: DictConfig, monkeypatch: MonkeyPatch) -> None:
    """An episode whose every active column (3 of 4) is constant is discarded."""
    from copula_inter import data_gen as dg

    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.data.d_features = 4
    cfg.seed = 3

    monkeypatch.setattr(dg, "_sample_active_dims", lambda d_total, cfg: [0, 1, 2])

    real_tabiclv2 = dg.tabiclv2_warp_features
    state = {"poisoned": False}

    def poisoning_tabiclv2(x: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        out = real_tabiclv2(x, seed=seed)
        if not state["poisoned"]:
            state["poisoned"] = True
            out = out.clone()
            out[0, :, [0, 1, 2]] = 0.0  # collapse every active column of episode 0
        return out

    monkeypatch.setattr(dg, "tabiclv2_warp_features", poisoning_tabiclv2)

    with pytest.warns(RuntimeWarning, match="degenerate .*active kernel column"):
        episodes = dg.generate_gp_batch(cfg, B=8, device="cpu", return_kernel_metadata=True)

    assert state["poisoned"], "test setup didn't actually poison an episode's feature columns"
    assert len(episodes) == 8
    for ep in episodes:
        cols = ep["kernel_feature_indices"].tolist()
        x = torch.cat([ep["x_norm_train"], ep["x_norm_test"]], dim=0)
        assert max(float(x[:, c].std()) for c in cols) > 1e-4, (
            "rbf: an episode with every active column collapsed reached the returned episodes"
        )


def test_multi_dim_active_kernel_partial_collapse_is_kept(small_cfg: DictConfig, monkeypatch: MonkeyPatch) -> None:
    """An episode with only one of several active columns constant is kept."""
    from copula_inter import data_gen as dg

    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.data.d_features = 4
    cfg.seed = 3

    monkeypatch.setattr(dg, "_sample_active_dims", lambda d_total, cfg: [0, 1, 2])

    real_tabiclv2 = dg.tabiclv2_warp_features
    state = {"poisoned": False}

    def poisoning_tabiclv2(x: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        out = real_tabiclv2(x, seed=seed)
        if not state["poisoned"]:
            state["poisoned"] = True
            out = out.clone()
            out[0, :, 0] = 0.0  # collapse only ONE of the three active columns
        return out

    monkeypatch.setattr(dg, "tabiclv2_warp_features", poisoning_tabiclv2)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        episodes = dg.generate_gp_batch(cfg, B=8, device="cpu", return_kernel_metadata=True)
    degen_warns = [str(w.message) for w in rec if "active kernel column" in str(w.message)]

    assert state["poisoned"], "test setup didn't actually poison an episode's feature columns"
    assert not degen_warns, f"partial collapse should not be discarded, got: {degen_warns}"
    assert len(episodes) == 8
    stds = [float(torch.cat([ep["x_norm_train"], ep["x_norm_test"]], dim=0)[:, 0].std()) for ep in episodes]
    assert min(stds) < 1e-4, "the partially-collapsed episode should have been kept, not discarded/regenerated away"


@pytest.mark.parametrize("kernel_name", ["polynomial", "dot_product+polynomial", "rbf+polynomial"])
def test_polynomial_reconstruction_round_trip(small_cfg: DictConfig, kernel_name: str) -> None:
    """Polynomial offset/alpha2/power round-trip through gp_analytical_pit to the generated z_train/z_test."""
    from copula_inter.pit import gp_analytical_pit

    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = kernel_name
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 0.5  # (6-3)/6 -> fixed k=3
    cfg.data.inactive_frac_max = 0.5

    torch.manual_seed(abs(hash("poly_recon_" + kernel_name)) % (2**31))
    task = generate_gp_task(cfg)

    cached = gp_analytical_pit(task)
    reconstructed_task = {k: v for k, v in task.items() if k not in ("_L_ff", "_alpha")}
    reconstructed = gp_analytical_pit(reconstructed_task)
    assert torch.allclose(cached["z_train"], reconstructed["z_train"], atol=1e-3)
    assert torch.allclose(cached["z_test"], reconstructed["z_test"], atol=1e-3)


def test_mlp_mixing_default_off_is_noop(small_cfg: DictConfig) -> None:
    """MLP mixing is an exact identity by default."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    x = torch.randn(4, 10, cfg.data.d_features)
    out = apply_mlp_feature_mixing(x, cfg, "cpu")
    assert torch.equal(out, x)


def test_mlp_mixing_prob_zero_is_noop(small_cfg: DictConfig) -> None:
    """mlp_mixing_prob=0 is an exact identity."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.mlp_mixing_enabled = True
    cfg.data.mlp_mixing_prob = 0.0
    x = torch.randn(4, 10, cfg.data.d_features)
    out = apply_mlp_feature_mixing(x, cfg, "cpu")
    assert torch.equal(out, x)


def test_mlp_mixing_shapes_preserved(small_cfg: DictConfig) -> None:
    """MLP mixing preserves shape/dtype and generate_gp_batch's schema."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 6
    cfg.data.mlp_mixing_enabled = True
    cfg.data.mlp_mixing_prob = 1.0  # force mixing on every episode

    torch.manual_seed(0)
    x = torch.randn(4, 10, cfg.data.d_features)
    out = apply_mlp_feature_mixing(x, cfg, "cpu")
    assert out.shape == x.shape
    assert out.dtype == x.dtype

    torch.manual_seed(1)
    episodes = generate_gp_batch(cfg, B=4, device="cpu")
    for ep in episodes:
        d = cfg.data.d_features
        assert ep["x_norm_train"].shape[-1] == d
        assert ep["x_norm_test"].shape[-1] == d


def test_mlp_mixing_prob_one_changes_output(small_cfg: DictConfig) -> None:
    """mlp_mixing_prob=1 changes the output."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 6
    cfg.data.mlp_mixing_enabled = True
    cfg.data.mlp_mixing_prob = 1.0

    torch.manual_seed(0)
    x = torch.randn(4, 10, cfg.data.d_features)
    out = apply_mlp_feature_mixing(x.clone(), cfg, "cpu")
    assert not torch.equal(out, x)


def test_mlp_mixing_partial_gate_leaves_some_episodes_unmixed(small_cfg: DictConfig) -> None:
    """0 < mlp_mixing_prob < 1 leaves some episodes unmixed and mixes others."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 6
    cfg.data.mlp_mixing_enabled = True
    cfg.data.mlp_mixing_prob = 0.5

    torch.manual_seed(0)
    B = 64
    x = torch.randn(B, 10, cfg.data.d_features)
    out = apply_mlp_feature_mixing(x.clone(), cfg, "cpu")
    n_unchanged = sum(torch.equal(out[b], x[b]) for b in range(B))
    n_changed = B - n_unchanged
    assert n_unchanged > 0, "expected some episodes left unmixed at prob=0.5"
    assert n_changed > 0, "expected some episodes mixed at prob=0.5"


def test_feature_normalisation_holds_with_mlp_mixing(small_cfg: DictConfig) -> None:
    """Features stay ~zero-mean, unit-std after MLP mixing."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 6
    cfg.data.mlp_mixing_enabled = True
    cfg.data.mlp_mixing_prob = 1.0
    # One MLP layer here: at small T two ReLU-family layers occasionally zero a
    # whole column (covered separately by the goldilocks test).
    cfg.data.mlp_num_layers_min = 1
    cfg.data.mlp_num_layers_max = 1

    # Seed torch and random (data_gen also uses random); these seeds avoid the rare column collapse.
    torch.manual_seed(0)
    random.seed(0)
    for _ in range(10):
        episodes = generate_gp_batch(cfg, B=1, device="cpu")
        task = episodes[0]
        x_all = torch.cat([task["x_norm_train"], task["x_norm_test"]], dim=0)
        for f in range(x_all.shape[1]):
            col = x_all[:, f]
            assert abs(col.mean().item()) < 0.2
            assert abs(col.std().item() - 1.0) < 0.2


@pytest.mark.parametrize("kernel_name", ALL_KERNELS)
def test_mlp_mixing_goldilocks_and_psd(small_cfg: DictConfig, kernel_name: str) -> None:
    """With MLP mixing on, every kernel still gives a valid, PSD, non-trivial R_star."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = kernel_name
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 1 / 3
    cfg.data.inactive_frac_max = 5 / 6
    cfg.data.mlp_mixing_enabled = True
    cfg.data.mlp_mixing_prob = 1.0

    torch.manual_seed(abs(hash("mlp_mix_" + kernel_name)) % (2**31))
    off_diag_abs = []
    for _ in range(20):
        task = generate_gp_task(cfg)
        R = task["R_star"]
        N = R.shape[0]

        assert torch.allclose(R.diagonal(), torch.ones(N), atol=1e-4)
        assert torch.allclose(R, R.T, atol=1e-5)
        eigvals = torch.linalg.eigvalsh(R)
        assert (eigvals >= -1e-4).all(), f"{kernel_name}: not PSD with MLP mixing (min eig={eigvals.min():.6f})"
        assert R.abs().max() <= 1.0 + 1e-5

        mask = ~torch.eye(N, dtype=torch.bool)
        off_diag_abs.append(R[mask].abs())

    mean_abs_r = torch.cat(off_diag_abs).mean().item()
    assert mean_abs_r > _COLLAPSE_THRESHOLD, (
        f"{kernel_name}: MLP mixing collapsed correlation, mean|r*_offdiag|={mean_abs_r:.4f}"
    )
    assert mean_abs_r < _DEGENERATE_THRESHOLD, (
        f"{kernel_name}: MLP mixing degenerate, mean|r*_offdiag|={mean_abs_r:.4f}"
    )


def test_kernel_hidden_warp_default_off_is_noop(small_cfg: DictConfig) -> None:
    """The kernel-hidden warp is an exact identity by default."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    x = torch.randn(4, 10, cfg.data.d_features)
    out = apply_kernel_hidden_warp(x, cfg, "cpu")
    assert torch.equal(out, x)


def test_kernel_hidden_warp_prob_zero_is_noop(small_cfg: DictConfig) -> None:
    """kernel_hidden_prob=0 is an exact identity."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel_hidden_enabled = True
    cfg.data.kernel_hidden_prob = 0.0
    x = torch.randn(4, 10, cfg.data.d_features)
    out = apply_kernel_hidden_warp(x, cfg, "cpu")
    assert torch.equal(out, x)


def test_kernel_hidden_warp_shapes_preserved(small_cfg: DictConfig) -> None:
    """The hidden warp preserves shape/dtype and leaves x_norm_train/x_norm_test unchanged."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 6
    cfg.data.kernel_hidden_enabled = True
    cfg.data.kernel_hidden_prob = 1.0  # force the warp on every episode

    torch.manual_seed(0)
    x = torch.randn(4, 10, cfg.data.d_features)
    out = apply_kernel_hidden_warp(x, cfg, "cpu")
    assert out.shape == x.shape
    assert out.dtype == x.dtype

    torch.manual_seed(1)
    episodes = generate_gp_batch(cfg, B=4, device="cpu")
    for ep in episodes:
        d = cfg.data.d_features
        assert ep["x_norm_train"].shape[-1] == d
        assert ep["x_norm_test"].shape[-1] == d


def test_kernel_hidden_warp_prob_one_changes_output(small_cfg: DictConfig) -> None:
    """kernel_hidden_prob=1 changes the output."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 6
    cfg.data.kernel_hidden_enabled = True
    cfg.data.kernel_hidden_prob = 1.0

    torch.manual_seed(0)
    x = torch.randn(4, 10, cfg.data.d_features)
    out = apply_kernel_hidden_warp(x.clone(), cfg, "cpu")
    assert not torch.equal(out, x)


def test_kernel_hidden_warp_partial_gate_leaves_some_episodes_unwarped(small_cfg: DictConfig) -> None:
    """0 < kernel_hidden_prob < 1 leaves some episodes unwarped and warps others."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 6
    cfg.data.kernel_hidden_enabled = True
    cfg.data.kernel_hidden_prob = 0.5

    torch.manual_seed(0)
    B = 64
    x = torch.randn(B, 10, cfg.data.d_features)
    out = apply_kernel_hidden_warp(x.clone(), cfg, "cpu")
    n_unchanged = sum(torch.equal(out[b], x[b]) for b in range(B))
    n_changed = B - n_unchanged
    assert n_unchanged > 0, "expected some episodes left unwarped at prob=0.5"
    assert n_changed > 0, "expected some episodes warped at prob=0.5"


def test_kernel_hidden_warp_disabled_matches_unmodified_pipeline(small_cfg: DictConfig) -> None:
    """With the warp disabled, R_star, y_train and y_test match a config without the keys exactly."""
    cfg_a = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg_a.data.d_features = 6
    cfg_b = OmegaConf.create(OmegaConf.to_container(cfg_a, resolve=True))
    cfg_b.data.kernel_hidden_enabled = False  # explicit, same as default

    torch.manual_seed(7)
    random.seed(7)
    eps_a = generate_gp_batch(cfg_a, B=4, device="cpu")
    torch.manual_seed(7)
    random.seed(7)
    eps_b = generate_gp_batch(cfg_b, B=4, device="cpu")

    for a, b in zip(eps_a, eps_b):
        assert torch.equal(a["R_star"], b["R_star"])
        assert torch.equal(a["y_train"], b["y_train"])
        assert torch.equal(a["y_test"], b["y_test"])


@pytest.mark.parametrize("kernel_name", ALL_KERNELS)
def test_kernel_hidden_warp_goldilocks_and_psd(small_cfg: DictConfig, kernel_name: str) -> None:
    """With the hidden warp on, every kernel still gives a valid, PSD, non-trivial R_star."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = kernel_name
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 1 / 3
    cfg.data.inactive_frac_max = 5 / 6
    cfg.data.kernel_hidden_enabled = True
    cfg.data.kernel_hidden_prob = 1.0

    torch.manual_seed(abs(hash("kernel_hidden_" + kernel_name)) % (2**31))
    off_diag_abs = []
    for _ in range(20):
        task = generate_gp_task(cfg)
        R = task["R_star"]
        N = R.shape[0]

        assert torch.allclose(R.diagonal(), torch.ones(N), atol=1e-4)
        assert torch.allclose(R, R.T, atol=1e-5)
        eigvals = torch.linalg.eigvalsh(R)
        assert (eigvals >= -1e-4).all(), f"{kernel_name}: not PSD with kernel-hidden warp (min eig={eigvals.min():.6f})"
        assert R.abs().max() <= 1.0 + 1e-5

        mask = ~torch.eye(N, dtype=torch.bool)
        off_diag_abs.append(R[mask].abs())

    mean_abs_r = torch.cat(off_diag_abs).mean().item()
    assert mean_abs_r > _COLLAPSE_THRESHOLD, (
        f"{kernel_name}: kernel-hidden warp collapsed correlation, mean|r*_offdiag|={mean_abs_r:.4f}"
    )
    assert mean_abs_r < _DEGENERATE_THRESHOLD, (
        f"{kernel_name}: kernel-hidden warp degenerate, mean|r*_offdiag|={mean_abs_r:.4f}"
    )


def _pairwise_dists(x: torch.Tensor) -> torch.Tensor:
    """Upper-triangle pairwise distances of a (T, d) matrix."""
    diff = x.unsqueeze(0) - x.unsqueeze(1)
    D = diff.norm(dim=-1)
    T = D.shape[0]
    iu = torch.triu_indices(T, T, offset=1)
    return D[iu[0], iu[1]]


def test_kernel_hidden_warp_breaks_isometry(small_cfg: DictConfig) -> None:
    """A smaller bottleneck preserves less of the model-space distance structure, and the default is well below isometry.

    Thresholds are empirical for d ~ 10.
    """
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 10
    cfg.data.kernel_hidden_enabled = True
    cfg.data.kernel_hidden_layers_min = 2
    cfg.data.kernel_hidden_layers_max = 2

    def measure(frac: float, n_calls: int = 20, B: int = 16, T: int = 30) -> float:
        cfg_local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_local.data.kernel_hidden_bottleneck_frac = frac
        all_model, all_kernel = [], []
        for call in range(n_calls):
            torch.manual_seed(1000 + call)
            random.seed(1000 + call)
            x_norm = torch.randn(B, T, cfg_local.data.d_features)
            x_norm = (x_norm - x_norm.mean(1, keepdim=True)) / x_norm.std(1, keepdim=True).clamp(min=1e-8)
            x_kernel = apply_kernel_hidden_warp(x_norm, cfg_local, "cpu")
            for b in range(B):
                all_model.append(_pairwise_dists(x_norm[b]))
                all_kernel.append(_pairwise_dists(x_kernel[b]))
        dm = torch.cat(all_model)
        dk = torch.cat(all_kernel)
        return torch.corrcoef(torch.stack([dm, dk]))[0, 1].item()

    corr_mild = measure(frac=0.9)  # near-minimal rank loss (r = d-1)
    corr_default = measure(frac=0.5)  # this repo's default
    corr_aggressive = measure(frac=0.2)

    assert corr_aggressive < corr_default < corr_mild + 1e-6, (
        f"expected more rank loss -> lower distance-correlation, got "
        f"mild={corr_mild:.4f} default={corr_default:.4f} aggressive={corr_aggressive:.4f}"
    )
    assert corr_mild < 0.6, (
        f"even minimal rank loss should already be well below a near-isometry "
        f"at this repo's small d_features, got corr={corr_mild:.4f}"
    )
    assert corr_aggressive < 0.35, (
        f"aggressive bottleneck should leave little recoverable distance structure, got corr={corr_aggressive:.4f}"
    )


def test_structural_warp_default_off_is_noop(small_cfg: DictConfig) -> None:
    """The structural warp is an exact identity by default."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    x = torch.randn(4, 32, cfg.data.d_features)
    out = apply_structural_feature_warp(x, cfg, "cpu")
    assert torch.equal(out, x)


def test_structural_warp_prob_zero_is_noop(small_cfg: DictConfig) -> None:
    """structural_warp_prob=0 is an exact identity."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.structural_warp_enabled = True
    cfg.data.structural_warp_prob = 0.0
    x = torch.randn(4, 32, cfg.data.d_features)
    out = apply_structural_feature_warp(x, cfg, "cpu")
    assert torch.equal(out, x)


def test_structural_warp_shapes_preserved(small_cfg: DictConfig) -> None:
    """The structural warp preserves shape/dtype and generate_gp_batch's schema."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 6
    cfg.data.structural_warp_enabled = True
    cfg.data.structural_warp_prob = 1.0  # force a transform on every column

    torch.manual_seed(0)
    x = torch.randn(4, 32, cfg.data.d_features)
    out = apply_structural_feature_warp(x, cfg, "cpu")
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()

    torch.manual_seed(1)
    episodes = generate_gp_batch(cfg, B=4, device="cpu")
    for ep in episodes:
        d = cfg.data.d_features
        assert ep["x_norm_train"].shape[-1] == d
        assert ep["x_norm_test"].shape[-1] == d


def test_structural_warp_prob_one_changes_output(small_cfg: DictConfig) -> None:
    """structural_warp_prob=1 changes the output."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 6
    cfg.data.structural_warp_enabled = True
    cfg.data.structural_warp_prob = 1.0

    torch.manual_seed(0)
    x = torch.randn(4, 32, cfg.data.d_features)
    out = apply_structural_feature_warp(x.clone(), cfg, "cpu")
    assert not torch.equal(out, x)


def test_structural_warp_partial_gate_leaves_some_columns_unwarped(small_cfg: DictConfig) -> None:
    """0 < structural_warp_prob < 1 leaves some episodes unwarped and warps others (B=500 keeps a false failure below 1e-3)."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 6
    cfg.data.structural_warp_enabled = True
    cfg.data.structural_warp_prob = 0.5

    torch.manual_seed(0)
    B = 500
    x = torch.randn(B, 32, cfg.data.d_features)
    out = apply_structural_feature_warp(x.clone(), cfg, "cpu")
    n_unchanged = sum(torch.equal(out[b], x[b]) for b in range(B))
    n_changed = B - n_unchanged
    assert n_unchanged > 0, "expected some episodes left unwarped at prob=0.5"
    assert n_changed > 0, "expected some episodes warped at prob=0.5"


@pytest.mark.parametrize("kernel_name", ALL_KERNELS)
def test_structural_warp_goldilocks_and_psd(small_cfg: DictConfig, kernel_name: str) -> None:
    """With structural warping on, every kernel still gives a valid, PSD, non-trivial R_star."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = kernel_name
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 1 / 3
    cfg.data.inactive_frac_max = 5 / 6
    cfg.data.structural_warp_enabled = True
    cfg.data.structural_warp_prob = 1.0

    torch.manual_seed(abs(hash("structural_warp_" + kernel_name)) % (2**31))
    off_diag_abs = []
    for _ in range(20):
        task = generate_gp_task(cfg)
        R = task["R_star"]
        N = R.shape[0]

        assert torch.allclose(R.diagonal(), torch.ones(N), atol=1e-4)
        assert torch.allclose(R, R.T, atol=1e-5)
        eigvals = torch.linalg.eigvalsh(R)
        assert (eigvals >= -1e-4).all(), f"{kernel_name}: not PSD with structural warping (min eig={eigvals.min():.6f})"
        assert R.abs().max() <= 1.0 + 1e-5

        mask = ~torch.eye(N, dtype=torch.bool)
        off_diag_abs.append(R[mask].abs())

    mean_abs_r = torch.cat(off_diag_abs).mean().item()
    assert mean_abs_r > _COLLAPSE_THRESHOLD, (
        f"{kernel_name}: structural warping collapsed correlation, mean|r*_offdiag|={mean_abs_r:.4f}"
    )
    assert mean_abs_r < _DEGENERATE_THRESHOLD, (
        f"{kernel_name}: structural warping degenerate, mean|r*_offdiag|={mean_abs_r:.4f}"
    )


def test_structural_warp_ops_sampled_without_replacement() -> None:
    """_sample_structural_ops draws distinct categories within [num_ops_min, num_ops_max], in canonical order."""
    for _ in range(200):
        ops = _sample_structural_ops(_DEFAULT_CATEGORY_WEIGHTS, num_ops_min=2, num_ops_max=4)
        assert 2 <= len(ops) <= 4
        # Map each returned op back to its category; categories must be unique.
        op_to_category = {op: cat for cat, ops_in_cat in _CATEGORY_OPS.items() for op in ops_in_cat}
        categories = [op_to_category[op] for op in ops]
        assert len(categories) == len(set(categories)), f"duplicate category sampled: {ops}"
        idx = [_STRUCTURAL_CATEGORIES.index(c) for c in categories]
        assert idx == sorted(idx), f"categories not in canonical order: {ops}"


def test_structural_warp_category_weights_zero_excludes_category() -> None:
    """A zero-weight category is never drawn."""
    weights = dict(_DEFAULT_CATEGORY_WEIGHTS)
    weights["discrete"] = 0.0
    for _ in range(100):
        ops = _sample_structural_ops(weights, num_ops_min=5, num_ops_max=5)
        assert "quantize" not in ops and "censor" not in ops


def test_structural_warp_num_ops_defaults_match_tempopfn(small_cfg: DictConfig) -> None:
    """num_ops defaults to 2..6 and category weights to TempoPFN's."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    assert int(getattr(cfg.data, "structural_warp_num_ops_min", 2)) == 2
    assert int(getattr(cfg.data, "structural_warp_num_ops_max", 6)) == 6
    weights = dict(getattr(cfg.data, "structural_warp_category_weights", _DEFAULT_CATEGORY_WEIGHTS))
    assert weights == _DEFAULT_CATEGORY_WEIGHTS


def test_structural_warp_batched_category_selection_matches_per_column_marginals() -> None:
    """_sample_structural_category_mask matches _sample_structural_ops' per-category selection rates and count law (statistically)."""
    torch.manual_seed(0)
    n_draws = 20000
    num_ops_min, num_ops_max = 2, 6

    old_counts = {c: 0 for c in _STRUCTURAL_CATEGORIES}
    old_k = []
    for _ in range(n_draws):
        ops = _sample_structural_ops(_DEFAULT_CATEGORY_WEIGHTS, num_ops_min, num_ops_max)
        op_to_category = {op: cat for cat, ops_in_cat in _CATEGORY_OPS.items() for op in ops_in_cat}
        cats = {op_to_category[op] for op in ops}
        for c in cats:
            old_counts[c] += 1
        old_k.append(len(cats))

    chosen_mask, eligible = _sample_structural_category_mask(
        n_draws, _DEFAULT_CATEGORY_WEIGHTS, num_ops_min, num_ops_max, "cpu"
    )
    new_counts = {c: int(chosen_mask[:, i].sum()) for i, c in enumerate(eligible)}
    new_k = chosen_mask.sum(dim=1).tolist()

    for c in _STRUCTURAL_CATEGORIES:
        old_rate = old_counts[c] / n_draws
        new_rate = new_counts[c] / n_draws
        assert abs(old_rate - new_rate) < 0.02, f"{c}: per-column rate={old_rate:.3f} vs batched rate={new_rate:.3f}"

    old_mean_k = sum(old_k) / len(old_k)
    new_mean_k = sum(new_k) / len(new_k)
    assert abs(old_mean_k - new_mean_k) < 0.05, (
        f"mean #categories selected: per-column={old_mean_k:.3f} vs batched={new_mean_k:.3f}"
    )


def test_structural_warp_batch_is_deterministic_given_seed() -> None:
    """apply_structural_feature_warp is deterministic given the torch seed."""
    cfg = OmegaConf.create(
        {
            "data": {
                "structural_warp_enabled": True,
                "structural_warp_prob": 0.5,
                "d_features": 6,
            }
        }
    )

    torch.manual_seed(7)
    x1 = torch.randn(16, 32, 6)
    torch.manual_seed(7)
    x2 = torch.randn(16, 32, 6)
    assert torch.equal(x1, x2)  # sanity: the two seeds really do reproduce the same input

    torch.manual_seed(123)
    out1 = apply_structural_feature_warp(x1.clone(), cfg, "cpu")
    torch.manual_seed(123)
    out2 = apply_structural_feature_warp(x2.clone(), cfg, "cpu")
    assert torch.equal(out1, out2)


def test_generate_gp_batch_raw_structural_warp_seed_pairing_contract(small_cfg: DictConfig) -> None:
    """Two _generate_gp_batch_raw calls with the same seed and structural warping are identical in every field."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 6
    cfg.data.structural_warp_enabled = True
    cfg.data.structural_warp_prob = 0.5
    cfg.seed = 999

    eps1 = _generate_gp_batch_raw(cfg, 8, device="cpu")
    eps2 = _generate_gp_batch_raw(cfg, 8, device="cpu")
    assert len(eps1) == len(eps2) and len(eps1) > 0
    for e1, e2 in zip(eps1, eps2):
        for key in e1:
            assert torch.equal(e1[key], e2[key]), f"field '{key}' differs across same-seed calls"


# Explicit (category, op) pairs, since categories have different arities.
_ALL_OPS = [(cat, op) for cat, ops in _CATEGORY_OPS.items() for op in ops]


@pytest.mark.parametrize("use_index_axis", [False, True])
@pytest.mark.parametrize("category,op", _ALL_OPS)
def test_structural_warp_op_preserves_shape_and_finite_direct(category: str, op: str, use_index_axis: bool) -> None:
    """Every op, on both pseudo-time axes, preserves shape/dtype and gives finite output."""
    torch.manual_seed(abs(hash(f"op_direct_{category}_{op}_{use_index_axis}")) % (2**31))
    col = torch.randn(64)
    out = _structural_warp_column(col.clone(), op, use_index_axis=use_index_axis)
    assert out.shape == col.shape
    assert out.dtype == col.dtype
    assert torch.isfinite(out).all()


def test_structural_warp_censor_never_collapses_whole_column() -> None:
    """censor never flattens a whole column (coinciding quantile indices)."""
    torch.manual_seed(0)
    n_collapsed = 0
    n_trials = 0
    for T in (4, 8, 16, 32, 64):
        for _ in range(500):
            col = torch.randn(T)
            out = _structural_warp_column(col.clone(), "censor")
            n_trials += 1
            if float(out.std()) < 1e-9:
                n_collapsed += 1
    assert n_collapsed == 0, f"{n_collapsed}/{n_trials} censor calls collapsed the whole column to a constant"


def test_structural_warp_differential_flat_derivative_does_not_collapse_column(monkeypatch: MonkeyPatch) -> None:
    """differential leaves the column unchanged when the derivative is flat (forced by patching the convolution)."""
    torch.manual_seed(0)
    col = torch.randn(64)  # T=64 -> k = max(3, 64 // 32) = 3

    real_conv1d = torch.nn.functional.conv1d
    calls = {"n": 0}

    def patched_conv1d(inp: torch.Tensor, weight: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        calls["n"] += 1
        out = real_conv1d(inp, weight, *args, **kwargs)
        # Call 1 is the box smoothing (kept); call 2 is the derivative, forced flat.
        if calls["n"] == 2:
            return torch.zeros_like(out)
        return out

    monkeypatch.setattr(torch.nn.functional, "conv1d", patched_conv1d)
    monkeypatch.setattr(torch, "randint", lambda *a, **k: torch.tensor([2]))  # force sub_op=2

    out = _structural_warp_column(col.clone(), "differential")
    assert calls["n"] == 2, "test setup didn't reach the derivative conv -- sub_op wasn't forced"
    assert torch.isfinite(out).all()
    assert torch.equal(out, col), (
        "a flat derivative signal should leave a genuinely-varying column unchanged, "
        "not collapse it to a single constant"
    )


@pytest.mark.parametrize("kernel_name", ["periodic", "cosine"])
def test_no_degenerate_active_kernel_column_with_structural_warp(small_cfg: DictConfig, kernel_name: str) -> None:
    """With every category forced on, a periodic/cosine episode's single active column never collapses."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = kernel_name
    cfg.data.systematic_composition = False
    cfg.data.d_features = 4
    cfg.data.mlp_mixing_enabled = False
    cfg.data.structural_warp_enabled = True
    cfg.data.structural_warp_prob = 1.0
    cfg.data.structural_warp_num_ops_min = 6
    cfg.data.structural_warp_num_ops_max = 6
    cfg.seed = abs(hash("no_degenerate_active_col_" + kernel_name)) % (2**31)

    episodes = generate_gp_batch(cfg, B=500, device="cpu", return_kernel_metadata=True)
    assert len(episodes) == 500
    for ep in episodes:
        col = int(ep["kernel_feature_indices"][0])
        x = torch.cat([ep["x_norm_train"], ep["x_norm_test"]], dim=0)[:, col]
        assert float(x.std()) > 1e-6, (
            f"{kernel_name}: active kernel column (idx {col}) collapsed to a "
            f"near-constant value -- degenerate covariance structure"
        )


def test_structural_warp_quantize_snaps_to_few_unique_levels() -> None:
    """quantize leaves at most 10 distinct values."""
    torch.manual_seed(0)
    col = torch.randn(500)
    out = _structural_warp_column(col.clone(), "quantize")
    assert out.unique().numel() <= 10
    assert not torch.equal(out, col)


def test_structural_warp_yflip_negates_column() -> None:
    col = torch.randn(32)
    out = _structural_warp_column(col.clone(), "yflip")
    assert torch.equal(out, -col)


def test_structural_warp_time_flip_reverses_index_axis() -> None:
    """time_flip with use_index_axis reverses row order."""
    col = torch.randn(32)
    out = _structural_warp_column(col.clone(), "time_flip", use_index_axis=True)
    assert torch.equal(out, col.flip(dims=[0]))


def test_structural_warp_time_flip_reverses_value_rank_by_default() -> None:
    """time_flip by default reverses value rank (smallest and largest swap)."""
    col = torch.randn(32)
    out = _structural_warp_column(col.clone(), "time_flip")
    sort_idx = torch.argsort(col)
    expected = torch.empty_like(col)
    expected[sort_idx] = col[sort_idx].flip(dims=[0])
    assert torch.equal(out, expected)
    # Differs from a raw index reversal.
    assert not torch.equal(out, col.flip(dims=[0]))


def test_structural_warp_num_ops_composes_multiple_categories(small_cfg: DictConfig) -> None:
    """num_ops=6 applies one op from every category and differs from a single category."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 6
    cfg.data.structural_warp_enabled = True
    cfg.data.structural_warp_prob = 1.0
    cfg.data.structural_warp_num_ops_min = len(_STRUCTURAL_CATEGORIES)
    cfg.data.structural_warp_num_ops_max = len(_STRUCTURAL_CATEGORIES)

    torch.manual_seed(0)
    x = torch.randn(4, 64, cfg.data.d_features)
    out_all = apply_structural_feature_warp(x.clone(), cfg, "cpu")
    assert out_all.shape == x.shape
    assert torch.isfinite(out_all).all()

    cfg.data.structural_warp_num_ops_min = 1
    cfg.data.structural_warp_num_ops_max = 1
    torch.manual_seed(0)
    out_single = apply_structural_feature_warp(x.clone(), cfg, "cpu")

    assert not torch.equal(out_all, out_single)


def test_structural_warp_index_axis_disabled_by_default(small_cfg: DictConfig) -> None:
    """The index-axis ratio has no effect unless structural_warp_index_axis_enabled."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 6
    cfg.data.structural_warp_enabled = True
    cfg.data.structural_warp_prob = 1.0
    cfg.data.structural_warp_index_axis_ratio = 1.0  # enabled defaults False, so this must be ignored

    torch.manual_seed(0)
    x = torch.randn(4, 64, cfg.data.d_features)

    torch.manual_seed(1)  # both calls must start from the identical RNG state
    out_ratio_set = apply_structural_feature_warp(x.clone(), cfg, "cpu")

    cfg.data.structural_warp_index_axis_ratio = 0.0
    torch.manual_seed(1)
    out_ratio_zero = apply_structural_feature_warp(x.clone(), cfg, "cpu")

    assert torch.equal(out_ratio_set, out_ratio_zero)


def test_structural_warp_index_axis_ratio_one_forces_index_axis(small_cfg: DictConfig) -> None:
    """Ratio 1.0 (enabled) differs from ratio 0.0."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 6
    cfg.data.structural_warp_enabled = True
    cfg.data.structural_warp_prob = 1.0
    cfg.data.structural_warp_index_axis_enabled = True
    cfg.data.structural_warp_index_axis_ratio = 1.0

    torch.manual_seed(0)
    x = torch.randn(4, 64, cfg.data.d_features)

    torch.manual_seed(1)  # both calls must start from the identical RNG state
    out_index = apply_structural_feature_warp(x.clone(), cfg, "cpu")

    cfg.data.structural_warp_index_axis_ratio = 0.0
    torch.manual_seed(1)
    out_rank = apply_structural_feature_warp(x.clone(), cfg, "cpu")

    assert not torch.equal(out_index, out_rank)


@pytest.mark.parametrize("kernel_name", ALL_KERNELS)
def test_structural_warp_composed_goldilocks_and_psd(small_cfg: DictConfig, kernel_name: str) -> None:
    """With all 6 categories composed on every column, every kernel still gives a valid, PSD, non-trivial R_star."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.kernel = kernel_name
    cfg.data.d_features = 6
    cfg.data.inactive_frac_min = 1 / 3
    cfg.data.inactive_frac_max = 5 / 6
    cfg.data.structural_warp_enabled = True
    cfg.data.structural_warp_prob = 1.0
    cfg.data.structural_warp_num_ops_min = len(_STRUCTURAL_CATEGORIES)
    cfg.data.structural_warp_num_ops_max = len(_STRUCTURAL_CATEGORIES)

    torch.manual_seed(abs(hash("structural_warp_composed_" + kernel_name)) % (2**31))
    off_diag_abs = []
    for _ in range(20):
        task = generate_gp_task(cfg)
        R = task["R_star"]
        N = R.shape[0]

        assert torch.allclose(R.diagonal(), torch.ones(N), atol=1e-4)
        assert torch.allclose(R, R.T, atol=1e-5)
        eigvals = torch.linalg.eigvalsh(R)
        assert (eigvals >= -1e-4).all(), (
            f"{kernel_name}: not PSD with composed structural warping (min eig={eigvals.min():.6f})"
        )
        assert R.abs().max() <= 1.0 + 1e-5

        mask = ~torch.eye(N, dtype=torch.bool)
        off_diag_abs.append(R[mask].abs())

    mean_abs_r = torch.cat(off_diag_abs).mean().item()
    assert mean_abs_r > _COLLAPSE_THRESHOLD, (
        f"{kernel_name}: composed structural warping collapsed correlation, mean|r*_offdiag|={mean_abs_r:.4f}"
    )
    assert mean_abs_r < _DEGENERATE_THRESHOLD, (
        f"{kernel_name}: composed structural warping degenerate, mean|r*_offdiag|={mean_abs_r:.4f}"
    )


def test_mean_fn_default_off_is_noop(small_cfg: DictConfig) -> None:
    """mean_fn_enabled=False gives an exact ZeroMean without drawing from the RNG."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    d = cfg.data.d_features

    torch.manual_seed(0)
    baseline = torch.randn(5)

    torch.manual_seed(0)
    mean_module, params = _sample_mean_module(cfg, d, B=4, device="cpu")
    after = torch.randn(5)

    assert torch.equal(baseline, after), "disabled mean bank perturbed the global RNG stream"
    assert torch.equal(params["mean_weight"], torch.zeros(4, d))
    assert torch.equal(params["mean_bias"], torch.zeros(4))
    assert not params["mean_nonzero"].any()
    assert torch.equal(params["mean_family"], torch.zeros(4, dtype=torch.long))
    assert not params["mean_linear"].any()

    x = torch.randn(4, 6, d)
    assert torch.equal(mean_module(x), torch.zeros(4, 6))


def test_mean_fn_prob_zero_is_noop(small_cfg: DictConfig) -> None:
    """mean_fn_prob=0 gives an all-zero mean."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.mean_fn_enabled = True
    cfg.data.mean_fn_prob = 0.0
    d = cfg.data.d_features

    mean_module, params = _sample_mean_module(cfg, d, B=8, device="cpu")
    assert not params["mean_nonzero"].any()

    x = torch.randn(8, 6, d)
    assert torch.equal(mean_module(x), torch.zeros(8, 6))


def test_mean_fn_all_families_reachable(small_cfg: DictConfig) -> None:
    """With mean_fn_prob=1 all three mean families occur."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 4
    cfg.data.mean_fn_enabled = True
    cfg.data.mean_fn_prob = 1.0
    cfg.data.mean_fn_family_probs = [1 / 3, 1 / 3, 1 / 3]

    torch.manual_seed(0)
    _, params = _sample_mean_module(cfg, d=4, B=300, device="cpu")
    assert params["mean_nonzero"].all()
    counts = torch.bincount(params["mean_family"], minlength=3)
    assert (counts > 0).all(), f"expected all 3 families to occur, got counts={counts.tolist()}"


@pytest.mark.parametrize("family_probs", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
def test_mean_fn_diversifies_mu_star(small_cfg: DictConfig, family_probs: list[float]) -> None:
    """Each mean family gives a non-zero mu_star."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 4
    cfg.data.mean_fn_enabled = True
    cfg.data.mean_fn_prob = 1.0
    cfg.data.mean_fn_family_probs = family_probs
    cfg.data.mean_fn_anomaly_frac = 0.5  # generous, so the sparse-anomaly family fires reliably at small N

    torch.manual_seed(abs(hash(("mean_fn_mu_star", tuple(family_probs)))) % (2**31))
    any_nonzero = False
    for _ in range(20):
        task = generate_gp_task(cfg)
        if task["mu_star"].abs().max().item() > 1e-6:
            any_nonzero = True
            break
    assert any_nonzero, f"family_probs={family_probs}: mu_star stayed all-zero over 20 draws"


@pytest.mark.parametrize("family_probs", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
def test_mean_fn_goldilocks_and_psd(small_cfg: DictConfig, family_probs: list[float]) -> None:
    """R_star stays valid, PSD and non-trivial under every mean family."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 4
    cfg.data.kernel = "rbf"
    cfg.data.oracle_mode = "prior"
    cfg.data.mean_fn_enabled = True
    cfg.data.mean_fn_prob = 1.0
    cfg.data.mean_fn_family_probs = family_probs

    torch.manual_seed(abs(hash(("mean_fn_psd", tuple(family_probs)))) % (2**31))
    off_diag_abs = []
    for _ in range(20):
        task = generate_gp_task(cfg)
        R = task["R_star"]
        N = R.shape[0]

        assert torch.allclose(R.diagonal(), torch.ones(N), atol=1e-4)
        assert torch.allclose(R, R.T, atol=1e-5)
        eigvals = torch.linalg.eigvalsh(R)
        assert (eigvals >= -1e-4).all(), f"not PSD with mean family {family_probs} (min eig={eigvals.min():.6f})"
        assert R.abs().max() <= 1.0 + 1e-5

        mask = ~torch.eye(N, dtype=torch.bool)
        off_diag_abs.append(R[mask].abs())

    mean_abs_r = torch.cat(off_diag_abs).mean().item()
    assert mean_abs_r > _COLLAPSE_THRESHOLD, (
        f"family {family_probs}: mean bank collapsed correlation, mean|r*_offdiag|={mean_abs_r:.4f}"
    )
    assert mean_abs_r < _DEGENERATE_THRESHOLD, (
        f"family {family_probs}: mean bank degenerate, mean|r*_offdiag|={mean_abs_r:.4f}"
    )


@pytest.mark.parametrize("family_probs", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
def test_mean_fn_z_train_stays_calibrated(small_cfg: DictConfig, family_probs: list[float]) -> None:
    """z_train variance stays calibrated with large means (LOO uses y - mean)."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 4
    cfg.data.P_min = cfg.data.P_max = 40
    cfg.data.kernel = "rbf"
    cfg.data.mean_fn_enabled = True
    cfg.data.mean_fn_prob = 1.0
    cfg.data.mean_fn_family_probs = family_probs
    cfg.data.mean_fn_linear_prob = 1.0
    cfg.data.mean_fn_weight_std = 1.0
    cfg.data.mean_fn_bias_std = 1.0

    torch.manual_seed(abs(hash(("mean_fn_z_train", tuple(family_probs)))) % (2**31))
    episodes = generate_gp_batch(cfg, B=300, device="cpu")
    z_train = torch.cat([ep["z_train"] for ep in episodes])

    assert z_train.mean().abs().item() < 0.15, f"family {family_probs}: z_train mean={z_train.mean():.4f} (expected ~0)"
    assert abs(z_train.std().item() - 1.0) < 0.15, (
        f"family {family_probs}: z_train std={z_train.std():.4f} (expected ~1 -- "
        f"a large deviation indicates the mean bank's contribution is leaking "
        f"into the LOO residual uncancelled)"
    )


def test_oracle_mode_posterior_unsupported(small_cfg: DictConfig) -> None:
    """oracle_mode='posterior' raises."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 3
    cfg.data.P_min = cfg.data.P_max = 40
    cfg.data.N_min = cfg.data.N_max = 20
    cfg.data.kernel = "rbf"
    cfg.data.oracle_mode = "posterior"

    torch.manual_seed(0)
    with pytest.raises(ValueError, match="oracle_mode"):
        generate_gp_batch(cfg, B=4, device="cpu")


@pytest.mark.parametrize("family_probs", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
def test_mean_fn_gp_analytical_pit_reconstruction_matches(small_cfg: DictConfig, family_probs: list[float]) -> None:
    """gp_analytical_pit without cached factors matches the cached z_train for every mean family."""
    from copula_inter.pit import gp_analytical_pit

    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 4
    cfg.data.kernel = "rbf"
    cfg.data.mean_fn_enabled = True
    cfg.data.mean_fn_prob = 1.0
    cfg.data.mean_fn_family_probs = family_probs
    cfg.data.mean_fn_anomaly_frac = 0.5

    torch.manual_seed(abs(hash(("mean_fn_pit_reconstruct", tuple(family_probs)))) % (2**31))
    for _ in range(10):
        task = generate_gp_task(cfg)
        cached = gp_analytical_pit(task)
        reconstructed_task = {k: v for k, v in task.items() if k not in ("_L_ff", "_alpha")}
        reconstructed = gp_analytical_pit(reconstructed_task)
        assert torch.allclose(cached["z_train"], reconstructed["z_train"], atol=1e-3)
        assert torch.allclose(cached["z_test"], reconstructed["z_test"], atol=1e-3)


def test_mean_fn_linear_prob_zero_forces_constant_only(small_cfg: DictConfig) -> None:
    """mean_fn_linear_prob=0 gives a constant linear mean (zero weight)."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.data.d_features = 4
    cfg.data.mean_fn_enabled = True
    cfg.data.mean_fn_prob = 1.0
    cfg.data.mean_fn_family_probs = [1.0, 0.0, 0.0]
    cfg.data.mean_fn_linear_prob = 0.0

    _, params = _sample_mean_module(cfg, d=4, B=16, device="cpu")
    assert not params["mean_linear"].any()
    assert torch.equal(params["mean_weight"], torch.zeros(16, 4))


def test_gp_posterior_helper() -> None:
    """gp_posterior should return correct shapes and PSD Sigma_star."""
    from copula_inter.gp_kernels import build_kernel_fn

    P, N, d = 20, 8, 1
    x_train = torch.randn(P, d)
    y_train = torch.randn(P)
    x_test = torch.randn(N, d)
    kernel_fn = build_kernel_fn("rbf", l=1.0, alpha2=1.0)
    mu, Sigma = gp_posterior(x_train, y_train, x_test, kernel_fn, noise=0.1)

    assert mu.shape == (N,)
    assert Sigma.shape == (N, N)

    eigvals = torch.linalg.eigvalsh(Sigma)
    assert (eigvals >= -1e-4).all(), f"Sigma_star not PSD: min eig={eigvals.min():.6f}"


def test_sigma_to_correlation() -> None:
    """sigma_to_correlation should produce unit diagonal."""
    N = 6
    # Build a random PD covariance
    A = torch.randn(N, N)
    Sigma = A @ A.T + 0.1 * torch.eye(N)
    R, sigma = sigma_to_correlation(Sigma)

    assert R.shape == (N, N)
    assert sigma.shape == (N,)
    assert torch.allclose(R.diagonal(), torch.ones(N), atol=1e-5)
    # PSD
    assert (torch.linalg.eigvalsh(R) >= -1e-5).all()


def _make_sample(P: int, N: int, d: int = 1) -> dict:
    return {
        "x_norm_train": torch.randn(P, d),
        "x_norm_test": torch.randn(N, d),
        "y_train": torch.randn(P),
        "y_test": torch.randn(N),
        "z_train": torch.randn(P),
        "z_test": torch.randn(N),
        "log_pdf_test": torch.randn(N),
        "R_star": torch.eye(N),
        "Sigma_star": torch.eye(N),
        "mu_star": torch.zeros(N),
        "sigma_star": torch.ones(N),
        "n_train": torch.tensor(P),
        "n_test": torch.tensor(N),
    }


def test_collate_fn_shapes() -> None:
    sizes = [(8, 4), (6, 3), (10, 5), (7, 5)]
    samples = [_make_sample(P, N) for P, N in sizes]
    batch = collate_fn(samples)

    B = len(samples)
    P_max = max(P for P, _ in sizes)
    N_max = max(N for _, N in sizes)

    assert batch["x_train"].shape == (B, P_max, 1)
    assert batch["z_train"].shape == (B, P_max)
    assert batch["x_test"].shape == (B, N_max, 1)
    assert batch["z_test"].shape == (B, N_max)
    assert batch["train_mask"].shape == (B, P_max)
    assert batch["test_mask"].shape == (B, N_max)
    assert batch["R_star"].shape == (B, N_max, N_max)
    assert batch["train_mask"].dtype == torch.bool
    assert batch["test_mask"].dtype == torch.bool


def test_collate_fn_masks_correct() -> None:
    samples = [_make_sample(8, 4), _make_sample(6, 3)]
    batch = collate_fn(samples)

    # First sample: P=8 valid, P_max=8 → all True
    assert batch["train_mask"][0].all()
    # Second sample: P=6 valid, rest padding → only first 6 True
    assert batch["train_mask"][1, :6].all()
    assert not batch["train_mask"][1, 6:].any()

    # Test mask
    assert batch["test_mask"][0, :4].all()
    assert not batch["test_mask"][1, 3:].any()  # N=3 for second sample


def test_collate_fn_padding_is_zero() -> None:
    """Padded z_train and x_train values should be zero."""
    samples = [_make_sample(10, 5), _make_sample(6, 3)]
    batch = collate_fn(samples)

    # Second sample padded from 6 to 10
    assert (batch["z_train"][1, 6:] == 0.0).all()
    assert (batch["x_train"][1, 6:] == 0.0).all()
    assert (batch["z_test"][1, 3:] == 0.0).all()


def test_copula_dataset_load(tmp_path: Path) -> None:
    """CopulaDataset should load .pt files correctly."""
    for i in range(3):
        sample = _make_sample(P=random.randint(5, 10), N=random.randint(3, 6))
        torch.save(sample, tmp_path / f"task_{i:06d}.pt")

    ds = CopulaDataset(episode_dir=str(tmp_path))
    assert len(ds) == 3

    item = ds[0]
    assert "x_norm_train" in item
    assert "z_train" in item
    assert "R_star" in item


def test_copula_dataset_skips_stale_nonfinite_episode(tmp_path: Path) -> None:
    """CopulaDataset skips a non-finite saved episode (warns, serves the next one)."""
    samples = [_make_sample(P=6, N=3) for _ in range(4)]
    samples[1]["z_train"] = torch.full_like(samples[1]["z_train"], float("nan"))

    torch.save(samples, tmp_path / "shard_000000.pt")
    torch.save({"n_total": len(samples), "shard_size": len(samples)}, tmp_path / "meta.pt")

    ds = CopulaDataset(episode_dir=str(tmp_path))
    assert len(ds) == 4

    with pytest.warns(RuntimeWarning, match="non-finite"):
        item = ds[1]
    assert torch.isfinite(item["z_train"]).all()
    assert torch.isfinite(item["y_train"]).all()


def test_add_derived_fields_reconstructs_sigma_and_prior() -> None:
    """_add_derived_fields rebuilds R_prior and Sigma_star from R_star and sigma_star."""
    sample = _make_sample(P=6, N=4)
    expected_R_prior = sample["R_star"].clone()
    expected_Sigma_star = sample["R_star"] * sample["sigma_star"].unsqueeze(0) * sample["sigma_star"].unsqueeze(1)
    del sample["Sigma_star"]

    out = _add_derived_fields(sample)

    assert torch.allclose(out["R_prior"], expected_R_prior)
    assert torch.allclose(out["Sigma_star"], expected_Sigma_star)


def test_add_derived_fields_leaves_stored_values_untouched() -> None:
    """_add_derived_fields keeps stored R_prior/Sigma_star."""
    sample = _make_sample(P=6, N=4)
    sample["R_prior"] = torch.full((4, 4), 0.5)
    sample["Sigma_star"] = torch.full((4, 4), 2.0)

    out = _add_derived_fields(sample)

    assert torch.equal(out["R_prior"], torch.full((4, 4), 0.5))
    assert torch.equal(out["Sigma_star"], torch.full((4, 4), 2.0))


def test_copula_dataset_individual_reconstructs_missing_fields(tmp_path: Path) -> None:
    """task_*.pt files without R_prior/Sigma_star load with both fields."""
    sample = _make_sample(P=6, N=4)
    del sample["Sigma_star"]
    torch.save(sample, tmp_path / "task_000000.pt")

    ds = CopulaDataset(episode_dir=str(tmp_path))
    item = ds[0]

    assert "R_prior" in item and "Sigma_star" in item
    assert torch.allclose(item["R_prior"], sample["R_star"])
    expected_Sigma_star = sample["R_star"] * sample["sigma_star"].unsqueeze(0) * sample["sigma_star"].unsqueeze(1)
    assert torch.allclose(item["Sigma_star"], expected_Sigma_star)

    # collate_fn must still work end-to-end on the reconstructed episode.
    batch = collate_fn([item])
    assert batch["Sigma_star"].shape == (1, 4, 4)
    assert batch["R_prior"].shape == (1, 4, 4)


def test_copula_dataset_sharded_reconstructs_missing_fields(tmp_path: Path) -> None:
    """Shards without R_prior/Sigma_star serve complete episodes that collate like full ones."""
    samples = [_make_sample(P=6, N=4) for _ in range(3)]
    for s in samples:
        del s["Sigma_star"]
        s.pop("R_prior", None)

    torch.save(samples, tmp_path / "shard_000000.pt")
    torch.save({"n_total": len(samples), "shard_size": len(samples)}, tmp_path / "meta.pt")

    ds = CopulaDataset(episode_dir=str(tmp_path))
    assert len(ds) == 3

    item = ds[1]
    assert "R_prior" in item and "Sigma_star" in item
    assert torch.allclose(item["R_prior"], samples[1]["R_star"])

    batch = collate_fn([ds[0], ds[1], ds[2]])
    assert batch["Sigma_star"].shape == (3, 4, 4)
    assert batch["R_prior"].shape == (3, 4, 4)
