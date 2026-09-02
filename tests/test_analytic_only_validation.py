"""
test_analytic_only_validation.py — pins training.val_analytic_only, the
"perfect marginal access" validation mode.

The mode's whole claim is that NO approximate marginal is anywhere in the
validation loop: every metric is scored against the exact analytic GP marginal
(z_test standardized by the episode's exact POSTERIOR marginal -- see
pit.gp_analytical_pit and test_analytic_pit_posterior, which pins that
specifically -- and log_pdf_test its exact Gaussian log-density), so the copula
head's error is isolated from the frozen TabICL marginal's approximation error.
Four things have to hold for that claim to be true, and each is easy to break
silently:

  1. The config coupling guard (live_dataset.validate_analytic_only). Setting
     val_analytic_only=true next to a data config that reintroduces a PIT --
     data.z_train_source=tabicl* (the val episodes themselves would be
     TabICL-PIT'd) or data.z_train_corruption_enabled=true (z_train would be
     deliberately noised) -- must raise, not quietly produce metrics labelled
     "analytic" that aren't. Same fail-loud rationale as
     _validate_z_train_source next door (see test_z_train_source_validation).

  2. validate() must emit the oracle_diag/* family and NOT emit any
     TabICL-scored or real-data key, even when handed the caches a
     non-analytic run would have built. This is the executable form of
     "bypass the pretrained quantile head".

  3. oracle_diag/copula_gap must be >= 0, and equal to gap_nll up to the
     marginal residual. Both sides of that subtraction are now split at the
     SAME exact posterior marginal, so the marginal cancels and the copula
     terms become directly comparable -- which they were NOT while the analytic
     PIT emitted prior marginals. oracle_diag/marginal_gap is the residual and
     must be ~0.

  4. oracle_diag/gap_nll must be >= 0 up to numerical tolerance on a
     well-conditioned probe. It is the model's total Y-space NLL minus
     gp_analytical_posterior's exact Schur-complement posterior NLL, and the
     Bayes-optimality of the posterior predictive under log loss makes that a
     provable inequality in expectation, not a convention. A negative gap
     means the ceiling is wrong (e.g. the eigenvalue-floor repair fired), not
     that the model is superhuman -- so this is the guard that says whether
     any other number in the mode can be trusted.

No GPU, no network, and deliberately no TabICL checkpoint: the model here is a
tiny fake, and the point of case 2 is that nothing tries to load one.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from live_dataset import validate_analytic_only
from pit import episode_posterior_ceiling


def _cfg(**overrides):
    """Minimal cfg carrying only the keys validate_analytic_only reads."""
    base = OmegaConf.create({
        "data": {"z_train_source": "analytic", "z_train_corruption_enabled": False},
        "training": {"val_analytic_only": True, "live_source": "gp"},
    })
    return OmegaConf.merge(base, OmegaConf.create(overrides))


# ---------------------------------------------------------------------------
# 1. Config coupling guard
# ---------------------------------------------------------------------------


def test_analytic_only_accepts_a_consistent_config():
    assert validate_analytic_only(_cfg()) is True


def test_analytic_only_defaults_off_and_skips_every_check():
    """val_analytic_only unset/false must be a pure no-op -- it must NOT start
    rejecting the default recipe, which is data.z_train_source=tabicl."""
    cfg = _cfg(**{"training": {"val_analytic_only": False}, "data": {"z_train_source": "tabicl"}})
    assert validate_analytic_only(cfg) is False
    # Also with the key absent entirely (older configs / ad-hoc dicts).
    bare = OmegaConf.create({"data": {"z_train_source": "tabicl"}, "training": {}})
    assert validate_analytic_only(bare) is False


@pytest.mark.parametrize("z_train_source", ["tabicl", "tabicl_split"])
def test_analytic_only_rejects_tabicl_z_train_source(z_train_source):
    """build_fixed_live_val_batches builds the val episodes from
    data.z_train_source, so a TabICL setting here would put a PIT-estimated
    marginal inside a loop whose metrics claim to use the exact analytic one."""
    with pytest.raises(ValueError, match="z_train_source"):
        validate_analytic_only(_cfg(**{"data": {"z_train_source": z_train_source}}))


def test_analytic_only_rejects_z_train_corruption():
    """corrupt_z_train perturbs validation and kernel-probe batches too, not
    just training ones -- so the model would be conditioned on deliberately
    noised context while the metrics report an exact-oracle setting."""
    with pytest.raises(ValueError, match="z_train_corruption_enabled"):
        validate_analytic_only(_cfg(**{"data": {"z_train_corruption_enabled": True}}))


def test_analytic_only_rejects_era5_live_source():
    """Real ERA5 has no analytic GP marginal at all: era5_live_dataset.py
    builds both train and val z from TabICL PIT unconditionally. Analytic-only
    there isn't stricter validation, it's impossible."""
    with pytest.raises(ValueError, match="era5"):
        validate_analytic_only(_cfg(**{"training": {"live_source": "era5"}}))


# ---------------------------------------------------------------------------
# 2. validate() emits oracle_diag/* and nothing TabICL- or real-data-scored
# ---------------------------------------------------------------------------


class _FakeCopula(nn.Module):
    """Stand-in for CopulaTabICL: returns a low-rank (W, s) of the right shape.

    Deliberately not a real model -- these tests are about which metrics
    validate() emits and how they aggregate, not about their values.

    W is a deterministic function of each episode's OWN x_test, never of a
    per-call RNG. That matters for the chunking tests: a fake that reseeded a
    generator per forward would produce a different W for
    model(batch[0:2]) than for model(batch)[0:2], making chunked and
    un-chunked results differ for a reason that has nothing to do with the
    code under test. Any real model is slice-consistent in this sense (each
    episode's output depends only on that episode), so the fake must be too.
    """

    def __init__(self, rank: int = 2):
        super().__init__()
        self.rank = rank
        self.correlation_parametrization = "covnorm"
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, batch):
        x_test = batch["x_test"]                      # (B, N, d)
        b, n_test = x_test.shape[0], x_test.shape[1]
        # (B, N, 1) -> (B, N, rank) via a fixed per-column multiplier: bounded,
        # varies across points and episodes, and depends on nothing else.
        freqs = torch.arange(1, self.rank + 1, dtype=x_test.dtype, device=x_test.device)
        W = torch.tanh(x_test[..., :1] * freqs) * self.scale
        s = torch.ones(b, n_test, dtype=x_test.dtype, device=x_test.device) * self.scale
        return {"W": W, "s": s}


@pytest.fixture(scope="module")
def analytic_val_setup(small_cfg):
    """A tiny live-generated validation set plus the cfg validate() needs.

    Uses the same generate_gp_batch/collate_fn path build_fixed_live_val_batches
    uses (return_kernel_metadata=True, then _attach_oracle_ceilings), so what
    validate() sees here has the same shape as a real analytic-only run's val
    loader -- including the cached _oracle_ceiling this mode now relies on.

    Plain RBF, no composition: gp_analytical_posterior supports every kernel
    schema, but an elementary well-conditioned one keeps the >= 0 gap assertion
    below about the inequality itself rather than about PSD-repair edge cases.
    """
    from dataset import collate_fn
    from data_gen import generate_gp_batch
    from train import _attach_oracle_ceilings

    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.seed = 20260901
    cfg.model.sigma_jitter = 1e-4
    cfg.model.correlation_parametrization = "covnorm"
    cfg.data.kernel = "rbf"
    cfg.data.systematic_composition = False
    cfg.data.z_train_source = "analytic"
    cfg.data.z_train_corruption_enabled = False
    cfg.data.P_min = cfg.data.P_max = 24
    cfg.data.N_min = cfg.data.N_max = 12
    cfg.training = OmegaConf.create({"val_analytic_only": True, "live_source": "gp"})

    torch.manual_seed(0)
    episodes = generate_gp_batch(cfg, 4, device="cpu", return_kernel_metadata=True)
    batch = collate_fn(episodes)
    _attach_oracle_ceilings(episodes)
    return cfg, [batch], {0: episodes}


def test_validate_analytic_only_emits_oracle_diag_and_no_tabicl_keys(analytic_val_setup):
    from train import validate

    cfg, val_batches, episodes_meta = analytic_val_setup
    metrics, figs = validate(
        _FakeCopula(), val_batches, cfg, "cpu", step=1, do_plot=False,
        val_episodes_meta=episodes_meta, analytic_only=True,
    )

    # The analytic headline family is present...
    for key in (
        "oracle_diag/total_nll",
        "oracle_diag/copula_nll",
        "oracle_diag/gap_nll",
        "oracle_diag/corr_pearson",
        "oracle_diag/corr_mae",
        "oracle_diag/corr_rmse",
        "oracle_diag/corr_bias",
        "oracle_diag/copula_gap",
        "oracle_diag/copula_headroom",
        "oracle_diag/marginal_gap",
        "oracle_diag/corr_kl",
        "y_nll_oracle_posterior",
    ):
        assert key in metrics, f"analytic-only run is missing {key}"
        assert np.isfinite(metrics[key]), f"{key} is not finite"

    # ...and nothing that would have routed through TabICL's PIT or real data.
    forbidden = [
        k for k in metrics
        if k in ("y_nll_total", "y_nll_marginal", "y_nll_copula")
        or "tabicl" in k
        or k.startswith("era5_fit/")
    ]
    assert forbidden == [], f"analytic-only run emitted non-analytic metrics: {forbidden}"

    # The model's own Sklar decomposition is mirrored into val/ (these get a
    # "val/" prefix in the training loop's wandb.log), so the val panel group
    # isn't left empty in this mode. Suffixed to keep them off the same panel
    # as a default-recipe run's TabICL-marginal val/y_nll_*.
    for key in ("y_nll_total_analytic_z", "y_nll_copula_analytic_z", "y_nll_marginal_analytic_z"):
        assert key in metrics, f"analytic-only run is missing {key}"
        assert np.isfinite(metrics[key])
    # Same numbers as oracle_diag/*, and Sklar-consistent.
    assert metrics["y_nll_total_analytic_z"] == pytest.approx(metrics["oracle_diag/total_nll"])
    assert metrics["y_nll_total_analytic_z"] == pytest.approx(
        metrics["y_nll_copula_analytic_z"] + metrics["y_nll_marginal_analytic_z"], abs=1e-5,
    )

    # The marginal is the exact posterior marginal on BOTH sides -- the model
    # never predicts one -- so it must cancel. This is what makes copula_gap a
    # like-for-like comparison, and it is the guard that fails first if the
    # analytic PIT regresses to prior marginals.
    assert metrics["oracle_diag/marginal_gap"] == pytest.approx(0.0, abs=1e-4), (
        f"marginal_gap={metrics['oracle_diag/marginal_gap']:.6f} should be ~0: the "
        "model's marginal term and the oracle ceiling's marginal term are the same "
        "exact posterior marginal. A non-zero value means they are split at "
        "different marginals, which invalidates oracle_diag/copula_gap."
    )
    # ...and therefore the copula gap and the total gap are the same number.
    assert metrics["oracle_diag/copula_gap"] == pytest.approx(
        metrics["oracle_diag/gap_nll"], abs=1e-4,
    )
    # The headroom is -E[copula NLL at the optimum] = -0.5*log|R_post|/n >= 0.
    assert metrics["oracle_diag/copula_headroom"] >= -1e-9


def test_validate_analytic_only_ignores_stale_tabicl_caches(analytic_val_setup):
    """Passing TabICL/ERA5 caches to an analytic_only validate() must not
    reintroduce their metrics. main() already skips building them, but a stale
    caller shouldn't be able to silently un-do the mode's guarantee."""
    from train import validate

    cfg, val_batches, episodes_meta = analytic_val_setup
    metrics, _ = validate(
        _FakeCopula(), val_batches, cfg, "cpu", step=1, do_plot=False,
        val_episodes_meta=episodes_meta, analytic_only=True,
        tabicl_val_z={0: {"z_train": torch.zeros(4, 24), "z_test": torch.zeros(4, 12),
                          "log_pdf_test": torch.zeros(4, 12)}},
        era5_val_batches={"nowhere": {"batch": None}},
    )
    assert not any("tabicl" in k or k.startswith("era5_fit/") for k in metrics)


def test_validate_analytic_only_restores_correlation_figures(analytic_val_setup):
    """The two predicted-vs-oracle correlation plots are the diagnostic this
    mode exists to make valid again (an estimated marginal warps z-space and
    rules them out -- see feedback_no_raw_correlation_vs_oracle_comparison),
    so their absence would mean the mode lost its main qualitative output."""
    import matplotlib
    matplotlib.use("Agg")
    from train import validate

    cfg, val_batches, episodes_meta = analytic_val_setup
    _, figs = validate(
        _FakeCopula(), val_batches, cfg, "cpu", step=1, do_plot=True,
        val_episodes_meta=episodes_meta, analytic_only=True,
    )
    keys = {k for k, _ in figs}
    assert "val/corr_density_analytic_z" in keys
    assert "val/corr_grid_analytic_z" in keys
    for _, fig in figs:
        matplotlib.pyplot.close(fig)


# ---------------------------------------------------------------------------
# 3. The Bayes-optimality inequality
# ---------------------------------------------------------------------------


def test_gap_nll_is_non_negative_on_a_well_conditioned_probe(analytic_val_setup):
    """oracle_diag/gap_nll >= 0 is the mode's core correctness claim.

    The ceiling is the exact posterior predictive, which minimizes expected
    log loss, so no model's total NLL can beat it in expectation. A negative
    gap here would indicate the ceiling itself is broken (the classic cause
    being gp_analytical_posterior's eigenvalue-floor repair firing on an
    ill-conditioned episode and inflating the reported ceiling), which would
    invalidate every gap number the mode reports.

    Tolerance rather than a hard 0: this is a 4-episode sample and a single
    untrained fake model, so the inequality is checked as "not meaningfully
    negative", which is what a broken ceiling would violate by nats.
    """
    from train import validate

    cfg, val_batches, episodes_meta = analytic_val_setup
    metrics, _ = validate(
        _FakeCopula(), val_batches, cfg, "cpu", step=1, do_plot=False,
        val_episodes_meta=episodes_meta, analytic_only=True,
    )
    assert metrics["oracle_diag/gap_nll"] >= -1e-3, (
        f"gap_nll={metrics['oracle_diag/gap_nll']} is negative: the model beat the "
        "exact posterior predictive, which is impossible in expectation -- suspect "
        "the ceiling (gp_analytical_posterior), not the model."
    )
    # p90 is a tail statistic over the same per-episode gaps, so it can never
    # sit below the mean-based gap by more than sampling noise allows; mainly
    # this pins that it is populated at all.
    assert np.isfinite(metrics["oracle_diag/gap_nll_p90"])


def test_episode_posterior_ceiling_matches_gp_analytical_posterior(analytic_val_setup):
    """The cached ceiling must be exactly gp_analytical_posterior's own value,
    just per-point-normalized -- the caching is a performance change and must
    not alter a single reported number."""
    from pit import gp_analytical_posterior

    _, _, episodes_meta = analytic_val_setup
    ep = episodes_meta[0][0]
    n = int(ep["x_norm_test"].shape[0])
    cached = episode_posterior_ceiling(ep)
    direct = gp_analytical_posterior(ep)

    assert cached is not None
    assert cached["n_test"] == n
    assert cached["nll_post"] == pytest.approx(direct["nll_post"] / n, rel=1e-9)
    assert cached["nll_post_marginal"] == pytest.approx(direct["nll_post_marginal"] / n, rel=1e-9)
    assert cached["nll_post_copula"] == pytest.approx(direct["nll_post_copula"] / n, rel=1e-9)
    ri, ci = np.triu_indices(n, k=1)
    np.testing.assert_allclose(
        cached["off_R_post"], direct["R_post"].cpu().numpy()[ri, ci], rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# 4. Probe forward-pass chunking
# ---------------------------------------------------------------------------


def test_chunked_probe_nll_equals_unchunked(analytic_val_setup):
    """Chunking the probe forward pass must be a pure memory optimization.

    The kernel_fit/<family> and posterior_probe batches are collated whole, so
    a single model(batch) call made peak validation VRAM scale with
    baselines.synth_n_episodes -- which OOM'd a 24GB card at 256 episodes
    while passing on a 48GB one. _chunked_probe_nll reconstructs the exact
    same mean by weighting each chunk by its valid-episode count (y_space_nll
    averages over episodes with >= 1 test point), so every chunk size must
    agree with the un-chunked value to float precision. If this drifts, the
    curriculum signal drifts with it.
    """
    from train import _chunked_probe_nll
    from loss import y_space_nll
    from model import build_sigma

    cfg, val_batches, _ = analytic_val_setup
    model = _FakeCopula()
    batch = val_batches[0]
    n_episodes = batch["x_train"].shape[0]

    with torch.no_grad():
        out = model(batch)
        Sigma = build_sigma(out, cfg, jitter=1e-4, test_mask=batch["test_mask"])
        reference = y_space_nll(
            Sigma, batch["z_test"].float(), batch["log_pdf_test"].float(), batch["test_mask"]
        )

    # Includes a chunk size that doesn't divide the batch evenly (3 into 4),
    # the degenerate one-episode-at-a-time case, and one larger than the batch.
    for chunk in (1, 2, 3, n_episodes, n_episodes + 5):
        with torch.no_grad():
            got = _chunked_probe_nll(model, cfg, batch, jitter=1e-4, chunk_size=chunk)
        for key in ("total", "copula", "marginal"):
            assert got[key] == pytest.approx(float(reference[key]), abs=1e-5), (
                f"chunk_size={chunk} changed {key}: {got[key]} vs {float(reference[key])}"
            )


def test_probe_chunk_size_defaults_to_batch_size():
    """The default has to be training.batch_size, not a fixed constant: that
    is the one shape the training loop is already known to sustain on
    whatever GPU the job landed on, so the probe fits by construction."""
    from train import _resolve_probe_chunk_size

    cfg = OmegaConf.create({"training": {"batch_size": 24}, "baselines": {"probe_chunk_size": None}})
    assert _resolve_probe_chunk_size(cfg) == 24
    cfg.baselines.probe_chunk_size = 4
    assert _resolve_probe_chunk_size(cfg) == 4
    # Absent baselines block at all (ad-hoc cfgs in tests//scripts).
    assert _resolve_probe_chunk_size(OmegaConf.create({"training": {"batch_size": 8}})) == 8


def test_probe_forward_chunks_covers_every_episode_once(analytic_val_setup):
    """Off-by-one in the chunk offsets would silently pair a Sigma slice with
    the wrong episode's cached ceiling in the posterior_probe block, producing
    plausible-looking but meaningless correlation diagnostics."""
    from train import _probe_forward_chunks

    cfg, val_batches, _ = analytic_val_setup
    model = _FakeCopula()
    batch = val_batches[0]
    n_episodes = batch["x_train"].shape[0]

    for chunk in (1, 3, n_episodes + 5):
        seen = []
        with torch.no_grad():
            for start, sub, Sigma in _probe_forward_chunks(model, cfg, batch, 1e-4, chunk):
                assert Sigma.shape[0] == sub["x_train"].shape[0]
                seen.extend(range(start, start + sub["x_train"].shape[0]))
        assert seen == list(range(n_episodes)), f"chunk_size={chunk} gave indices {seen}"


def test_episode_posterior_ceiling_returns_none_without_kernel_metadata(analytic_val_setup):
    """Episodes loaded from an on-disk shard carry no kernel metadata. That
    must degrade to "ceiling unavailable for this episode", never take down a
    whole validation pass."""
    _, _, episodes_meta = analytic_val_setup
    stripped = {
        k: v for k, v in episodes_meta[0][0].items()
        if k not in ("kernel", "kernel_components", "kernel_ops", "kernel_component_params",
                     "_L_ff", "_alpha", "l", "alpha2")
    }
    assert episode_posterior_ceiling(stripped) is None
