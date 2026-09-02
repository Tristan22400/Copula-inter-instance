"""val/gp_baseline/* — classical GP comparators on the validation episodes.

These are the middle of the three-way bracket the model's copula NLL is read
against:

    y_nll_oracle_posterior_copula  <=  gp_baseline/<k>/copula_nll  <=  0
      (Bayes-optimal, true kernel)      (misspecified classical GP)   (independence)

so what has to hold is (a) every method is scored through the SAME loss against
the SAME z, which is the only way two correlation structures can legitimately be
compared, and (b) the numbers are constant across validate() calls, since none
of them depend on the checkpoint.
"""

import copy

import pytest
import torch
from omegaconf import OmegaConf

from dataset import collate_fn
from data_gen import generate_gp_batch
from model import build_copula_transformer
from train import (
    _attach_oracle_ceilings,
    _build_gp_baseline_val_scores,
    validate,
)

KERNELS = ["matern32", "rational_quadratic"]


@pytest.fixture
def val_fixture(small_cfg, small_model_cfg):
    """Two small val batches plus their episode metadata, shaped the way
    live_dataset.build_fixed_live_val_batches hands them to validate()."""
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.seed = 11
    cfg.model = copy.deepcopy(small_model_cfg.model)
    cfg.tabicl = copy.deepcopy(small_model_cfg.tabicl)
    cfg.training = OmegaConf.create({"val_analytic_only": False, "live_source": "gp"})
    cfg.data.d_features = 3
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.data.P_min = cfg.data.P_max = 16
    cfg.data.N_min = cfg.data.N_max = 24
    batches, meta = [], {}
    for i in range(2):
        c = copy.deepcopy(cfg)
        c.seed = 11 + i * 104_729
        torch.manual_seed(c.seed)
        eps = generate_gp_batch(c, 3, "cpu", return_kernel_metadata=True)
        batches.append(collate_fn(eps))
        _attach_oracle_ceilings(eps)
        meta[i] = eps
    return cfg, batches, meta


def _cfg_with_baselines(cfg, **overrides):
    c = copy.deepcopy(cfg)
    c.baselines = OmegaConf.create(
        {
            "gp_baseline_val": True,
            "gp_baseline_val_kernels": KERNELS,
            "gp_baseline_val_episodes": None,
            "gp_baseline_val_n_steps_mle": 40,
            "gp_baseline_val_lr_mle": 0.05,
            "gp_baseline_val_n_restarts_mle": 1,
            "gp_baseline_val_ard": False,
            "gp_baseline_val_group_max": 64,
            **overrides,
        }
    )
    return c


def test_scores_every_episode_and_reports_both_conventions(val_fixture):
    cfg, batches, meta = val_fixture
    torch.manual_seed(0)
    out = _build_gp_baseline_val_scores(_cfg_with_baselines(cfg), batches, meta, None, "cpu")

    assert out is not None
    assert out["n_episodes"] == sum(len(v) for v in meta.values())
    for k in KERNELS:
        e = out["kernels"][k]
        # shared-marginal (comparable to oracle_diag/copula_nll) ...
        assert "copula_nll" in e and "total_nll" in e
        # ... and own-marginal end-to-end (comparable to y_nll_oracle_posterior)
        assert "own_total_nll" in e and "own_marginal_nll" in e and "own_copula_nll" in e
        assert e["failed"] == 0.0
        # The Sklar split of the own-marginal total is exact by construction.
        assert e["own_total_nll"] == pytest.approx(
            e["own_marginal_nll"] + e["own_copula_nll"], abs=1e-5
        )


def test_shared_marginal_total_minus_copula_is_the_episodes_own_marginal(val_fixture):
    """The load-bearing invariant of the shared-marginal convention: the
    baseline's total and the model's total differ ONLY in the copula term,
    because both are -sum(log_pdf_test)/n plus their own copula. If this drifts,
    gp_baseline/<k>/copula_nll and oracle_diag/copula_nll are no longer
    subtractable and the whole comparison is meaningless."""
    cfg, batches, meta = val_fixture
    torch.manual_seed(0)
    out = _build_gp_baseline_val_scores(_cfg_with_baselines(cfg), batches, meta, None, "cpu")

    marginals = []
    for b in batches:
        n = b["test_mask"].sum(-1).clamp(min=1).float()
        marginals.extend((-(b["log_pdf_test"].float() * b["test_mask"]).sum(-1) / n).tolist())
    expected = sum(marginals) / len(marginals)

    for k in KERNELS:
        e = out["kernels"][k]
        assert e["total_nll"] - e["copula_nll"] == pytest.approx(expected, abs=1e-4)


def test_baseline_is_bracketed_by_the_ceiling(val_fixture):
    """The oracle knows the true kernel AND its true hyperparameters, so a
    fitted misspecified kernel cannot beat it on average. A violation means the
    two are not being scored on the same basis."""
    cfg, batches, meta = val_fixture
    torch.manual_seed(0)
    out = _build_gp_baseline_val_scores(_cfg_with_baselines(cfg), batches, meta, None, "cpu")

    assert "copula_nll" in out["oracle"]
    for k in KERNELS:
        assert out["kernels"][k]["copula_nll"] >= out["oracle"]["copula_nll"] - 1e-6
        assert out["kernels"][k]["copula_gap"] == pytest.approx(
            out["kernels"][k]["copula_nll"] - out["oracle"]["copula_nll"], abs=1e-9
        )


def test_can_be_disabled_and_capped(val_fixture):
    cfg, batches, meta = val_fixture
    assert _build_gp_baseline_val_scores(
        _cfg_with_baselines(cfg, gp_baseline_val=False), batches, meta, None, "cpu"
    ) is None
    assert _build_gp_baseline_val_scores(
        _cfg_with_baselines(cfg, gp_baseline_val_kernels=[]), batches, meta, None, "cpu"
    ) is None
    # No episode metadata (disk-mode val_loader) -> nothing to fit, not a crash.
    assert _build_gp_baseline_val_scores(_cfg_with_baselines(cfg), batches, None, None, "cpu") is None

    torch.manual_seed(0)
    capped = _build_gp_baseline_val_scores(
        _cfg_with_baselines(cfg, gp_baseline_val_episodes=2), batches, meta, None, "cpu"
    )
    assert capped["n_episodes"] == 2


def test_metrics_reach_validate_and_do_not_move_with_the_model(val_fixture):
    cfg, batches, meta = val_fixture
    c = _cfg_with_baselines(cfg)

    torch.manual_seed(0)
    scores = _build_gp_baseline_val_scores(c, batches, meta, None, "cpu")
    model = build_copula_transformer(c)

    seen = []
    for _ in range(2):
        m, _figs = validate(
            model, batches, c, "cpu", step=0, do_plot=False,
            val_episodes_meta=meta, gp_baseline_scores=scores,
        )
        seen.append(m)
        # Perturb the model between calls: these baselines must not move.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)

    for k in KERNELS:
        key = f"gp_baseline/{k}/copula_nll"
        assert key in seen[0], sorted(x for x in seen[0] if x.startswith("gp_baseline/"))
        assert seen[0][key] == seen[1][key]
    assert seen[0]["gp_baseline/n_episodes"] == float(sum(len(v) for v in meta.values()))
    assert "gp_baseline/oracle_copula_nll" in seen[0]

