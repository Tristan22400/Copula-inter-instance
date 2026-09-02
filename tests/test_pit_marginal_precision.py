"""data_gen's batched posterior PIT and pit.gp_analytical_posterior must agree
about the SAME episode's exact GP posterior marginal NLL.

They are two implementations of one model-independent quantity, and validate()
reports their difference as oracle_diag/marginal_gap. That metric is only
readable if the two agree to float noise -- otherwise a nonzero value can mean
either "the routing broke" or "one path is simply less precise", and
oracle_diag/copula_gap (which subtracts the two Sklar splits) stops being a
like-for-like comparison.

Historically they did NOT agree. data_gen computed the Schur-complement
diagonal in float32 while pit.py used float64, leaving up to ~1e-4 nats/point
per episode; and train.py's process-global
torch.set_float32_matmul_precision("high") pushed that to a constant ~5e-4 on
GPU by putting TF32 into a cancellation (var_post = diag(K_ss) - sum(V_sf^2)).
Both are fixed by data_gen.full_precision_matmul plus the float64 upcast.

These run with matmul precision forced to "high", the same global setting a
real training run has. On CPU "high" is a no-op, so what actually gets guarded
here is the float32-vs-float64 half; the TF32 half is guarded by asserting the
precision in force *while the episode's linear algebra runs*.
"""

import math

import pytest
import torch
from omegaconf import OmegaConf

from data_gen import full_precision_matmul, generate_gp_batch
from pit import gp_analytical_pit, gp_analytical_posterior

# The reported metric is the MEAN over episodes, so that is what gets the
# tight bound. The per-episode tail is deliberately looser: data_gen's PIT
# reads the PSD-repaired K_all (= L_all @ L_all.mT, the matrix y_test was
# actually drawn from), while gp_analytical_posterior re-evaluates the kernel
# from x. On a well-conditioned episode those agree to float noise; on an
# ill-conditioned composite chain where psd_safe_cholesky had to jitter, they
# are genuinely slightly different matrices. Measured worst case ~3e-4 over
# 120 production-config episodes -- surfaced by
# oracle_diag/marginal_gap_max_abs, and still comfortably under
# train._MARGINAL_GAP_WARN.
TOL_MEAN = 1e-6
TOL_EPISODE = 1e-4


def _cfg(small_cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.seed = 1234
    cfg.data.d_features = 3
    cfg.data.P_min = cfg.data.P_max = 16
    cfg.data.N_min = cfg.data.N_max = 32
    cfg.data.kernel = "rbf"
    return cfg


def _marginal_from_log_pdf(log_pdf) -> float:
    n = log_pdf.shape[0]
    return -(log_pdf.double().sum().item()) / n


@pytest.fixture(autouse=True)
def _matmul_high():
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    yield
    torch.set_float32_matmul_precision(prev)


def test_batched_pit_marginal_matches_the_float64_ceiling(small_cfg):
    torch.manual_seed(0)
    episodes = generate_gp_batch(_cfg(small_cfg), 8, "cpu", return_kernel_metadata=True)
    assert episodes
    gaps = []
    for ep in episodes:
        post = gp_analytical_posterior(ep)
        n = ep["x_norm_test"].shape[0]
        gaps.append(_marginal_from_log_pdf(ep["log_pdf_test"]) - post["nll_post_marginal"] / n)
    mean_gap = sum(gaps) / len(gaps)
    worst = max(abs(g) for g in gaps)
    assert abs(mean_gap) < TOL_MEAN, f"mean marginal disagreement {mean_gap:+.3e} (all: {gaps})"
    assert worst < TOL_EPISODE, f"per-episode marginal disagreement {worst:.3e} (all: {gaps})"


def test_single_episode_pit_marginal_matches_the_float64_ceiling(small_cfg):
    """Same check for pit.gp_analytical_pit, the per-episode twin that
    generate_gp_task and the analytic val-z cache use."""
    torch.manual_seed(0)
    episodes = generate_gp_batch(_cfg(small_cfg), 8, "cpu", return_kernel_metadata=True)
    gaps = []
    for ep in episodes:
        pit_out = gp_analytical_pit(ep)
        post = gp_analytical_posterior(ep)
        n = ep["x_norm_test"].shape[0]
        gaps.append(_marginal_from_log_pdf(pit_out["log_pdf_test"]) - post["nll_post_marginal"] / n)
    # Tighter than the batched twin above on BOTH statistics: this path and
    # the ceiling re-evaluate the same kernel from x in the same way, so there
    # is no PSD-repaired-vs-raw K difference left to absorb -- only float64
    # rounding.
    mean_gap = sum(gaps) / len(gaps)
    worst = max(abs(g) for g in gaps)
    assert abs(mean_gap) < TOL_MEAN, f"mean marginal disagreement {mean_gap:+.3e}"
    assert worst < TOL_MEAN, f"per-episode marginal disagreement {worst:.3e} (all: {gaps})"


def test_precision_guard_forces_highest_and_restores_the_caller_setting():
    """A leaked "highest" would silently slow Muon down for the rest of the
    run, so the restore has to survive an exception too."""
    torch.set_float32_matmul_precision("high")
    with full_precision_matmul():
        assert torch.get_float32_matmul_precision() == "highest"
    assert torch.get_float32_matmul_precision() == "high"

    with pytest.raises(RuntimeError):
        with full_precision_matmul():
            raise RuntimeError("boom")
    assert torch.get_float32_matmul_precision() == "high"


def test_generation_actually_runs_under_the_guard(small_cfg, monkeypatch):
    """The guard has to be applied to the generation path, not merely
    available. On GPU that decoration IS the TF32 fix, and nothing else in the
    suite would notice it being removed."""
    torch.set_float32_matmul_precision("high")
    observed = []
    orig = torch.linalg.solve_triangular

    def _spy(*args, **kwargs):
        observed.append(torch.get_float32_matmul_precision())
        return orig(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "solve_triangular", _spy)
    generate_gp_batch(_cfg(small_cfg), 2, "cpu", return_kernel_metadata=True)

    assert observed, "generation never reached solve_triangular"
    assert set(observed) == {"highest"}, set(observed)
    assert torch.get_float32_matmul_precision() == "high"


def test_log_pdf_is_still_a_proper_gaussian_density(small_cfg):
    """The float64 upcast must not change the episode schema, and the emitted
    density must still satisfy log_pdf = -0.5 log 2pi - log sigma - 0.5 z^2 for
    a strictly positive sigma."""
    torch.manual_seed(0)
    episodes = generate_gp_batch(_cfg(small_cfg), 4, "cpu", return_kernel_metadata=True)
    for ep in episodes:
        assert ep["z_test"].dtype == torch.float32
        assert ep["log_pdf_test"].dtype == torch.float32
        z = ep["z_test"].double()
        log_sigma = -0.5 * math.log(2.0 * math.pi) - 0.5 * z ** 2 - ep["log_pdf_test"].double()
        assert torch.isfinite(log_sigma).all()


# ---------------------------------------------------------------------------
# The metric validate() actually reports
# ---------------------------------------------------------------------------


def _val_setup(small_cfg, small_model_cfg):
    """One small live-generated val batch plus its episode metadata, shaped the
    way live_dataset.build_fixed_live_val_batches hands them to validate()."""
    import copy

    from dataset import collate_fn
    from train import _attach_oracle_ceilings

    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.seed = 4321
    cfg.model = copy.deepcopy(small_model_cfg.model)
    cfg.tabicl = copy.deepcopy(small_model_cfg.tabicl)
    cfg.training = OmegaConf.create({"val_analytic_only": False, "live_source": "gp"})
    cfg.data.d_features = 3
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.data.z_train_source = "analytic"
    cfg.data.P_min = cfg.data.P_max = 16
    cfg.data.N_min = cfg.data.N_max = 24
    cfg.baselines = OmegaConf.create({"gp_baseline_val": False})

    torch.manual_seed(0)
    episodes = generate_gp_batch(cfg, 4, "cpu", return_kernel_metadata=True)
    batch = collate_fn(episodes)
    _attach_oracle_ceilings(episodes)
    return cfg, [batch], {0: episodes}


def test_validate_reports_marginal_gap_as_float_noise_on_the_analytic_basis(
    small_cfg, small_model_cfg
):
    """End-to-end version of the two checks above: what validate() actually
    logs, on the basis where the two marginals are the same quantity computed
    twice rather than two different marginals."""
    from model import build_copula_transformer
    from train import validate

    cfg, batches, meta = _val_setup(small_cfg, small_model_cfg)
    model = build_copula_transformer(cfg)
    metrics, _ = validate(
        model, batches, cfg, "cpu", step=0, do_plot=False, val_episodes_meta=meta,
    )

    assert metrics["oracle_diag/marginal_gap_exact_basis"] == 1.0
    assert metrics["oracle_diag/marginal_gap_n"] == 4.0
    assert abs(metrics["oracle_diag/marginal_gap"]) < TOL_MEAN, metrics["oracle_diag/marginal_gap"]
    assert metrics["oracle_diag/marginal_gap_max_abs"] < TOL_EPISODE


def test_marginal_gap_is_flagged_as_a_tabicl_measurement_not_a_defect(
    small_cfg, small_model_cfg
):
    """Under data.z_train_source=tabicl the batch marginal is TabICL's own PIT,
    so this difference is the frozen marginal's approximation error rather than
    a numerical disagreement -- it must still be reported, but not warned about
    as if the oracle_diag routing had broken."""
    from model import build_copula_transformer
    from train import validate

    cfg, batches, meta = _val_setup(small_cfg, small_model_cfg)
    cfg.data.z_train_source = "tabicl"
    model = build_copula_transformer(cfg)
    metrics, _ = validate(
        model, batches, cfg, "cpu", step=0, do_plot=False, val_episodes_meta=meta,
    )
    assert metrics["oracle_diag/marginal_gap_exact_basis"] == 0.0
    assert "oracle_diag/marginal_gap" in metrics


def test_caller_matmul_precision_hands_the_setting_back_inside_a_guarded_block():
    """The TabICL PIT forward runs inside _generate_gp_batch_raw's guarded
    region but should keep the caller's throughput setting -- it is an
    approximate marginal, and it runs on every live-generation batch in every
    DataLoader worker."""
    from data_gen import caller_matmul_precision

    torch.set_float32_matmul_precision("high")
    # No-op outside any guarded region.
    with caller_matmul_precision():
        assert torch.get_float32_matmul_precision() == "high"

    with full_precision_matmul():
        assert torch.get_float32_matmul_precision() == "highest"
        with caller_matmul_precision():
            assert torch.get_float32_matmul_precision() == "high"
        assert torch.get_float32_matmul_precision() == "highest"
    assert torch.get_float32_matmul_precision() == "high"


def test_caller_matmul_precision_reads_the_outermost_setting_when_nested():
    """gp_analytical_pit can be called from inside generation; the caller's
    setting is the process's, not an inner guard's."""
    from data_gen import caller_matmul_precision

    torch.set_float32_matmul_precision("high")
    with full_precision_matmul():
        with full_precision_matmul():
            with caller_matmul_precision():
                assert torch.get_float32_matmul_precision() == "high"
            assert torch.get_float32_matmul_precision() == "highest"
    assert torch.get_float32_matmul_precision() == "high"
