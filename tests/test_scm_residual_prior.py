"""The SCM + GP residual prior (data.scm_residual): limits, conditional oracle, guard, contracts, determinism."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from copula_inter.config_path import merge_configs
from copula_inter.data.contracts import validate_episode
from copula_inter.data.gp_bayes import bayes_predictive
from copula_inter.data.gp_episodes import (
    _draw_gp,
    _ResidualMean,
    _sample_call_shape,
    _sample_episode_priors,
    _sample_features,
    generate_fixed_gp_batch,
    generate_gp_batch,
    sigma_to_correlation,
)
from copula_inter.data.manifest import generation_spec
from copula_inter.data.scm_prior import (
    MEAN_FAMILY_SCM,
    SCM_PRIOR_CODES,
    ScmResidualSettings,
    sample_amplitudes,
    sample_scm_tables,
    saved_mean_at,
)
from copula_inter.marginal.pit import (
    _kernel_fn_from_task,
    _mean_train_from_task,
    _unscaled_kernel_fn_from_task,
    gp_analytical_pit,
    gp_analytical_posterior,
    kernel_space_inputs,
)
from copula_inter.rng import seed_everything

REPO = Path(__file__).resolve().parents[1]


def _cfg(seed: int = 7, P: int = 12, N: int = 6, d: int = 4, **scm: Any) -> DictConfig:
    """The repository's data prior with a small fixed shape, one RBF kernel family and the SCM prior on."""
    data = OmegaConf.load(REPO / "conf" / "data" / "gp_tasks.yaml")
    assert isinstance(data, DictConfig)
    for key in ("d_features_lognormal_loc", "d_features_lognormal_scale"):
        data.pop(key)
    overrides = {
        "d_features": d,
        "P_min": P,
        "P_max": P,
        "N_min": N,
        "N_max": N,
        "systematic_composition": False,
        "kernel": "rbf",
        "z_train_source": "tabicl",
        "scm_residual": {"enabled": True, "prior_type": "graph_scm", **scm},
    }
    return OmegaConf.create({"seed": seed, "data": OmegaConf.merge(data, overrides)})


LAMBDA_ZERO: dict[str, Any] = {"lambda_p0": 1.0, "lambda_p1": 0.0}
LAMBDA_ONE: dict[str, Any] = {"lambda_p0": 0.0, "lambda_p1": 1.0}
# a_g^2 pinned to 1 (LogNormal scale 0): a mean and a residual in every episode.
PINNED_MEAN: dict[str, Any] = {"lambda_p0": 0.0, "lambda_p1": 0.0, "mean_amp_lognormal_scale": 0.0}


def test_lambda_zero_is_iid_given_the_scm() -> None:
    episodes = generate_gp_batch(_cfg(**LAMBDA_ZERO), 16, return_kernel_metadata=True)
    ratios = []
    for ep in episodes:
        N = int(ep["n_test"])
        assert torch.allclose(ep["R_star"], torch.eye(N), atol=1e-5)
        assert float(ep["lambda_resid"]) == 0.0
        resid = torch.cat([ep["y_train"] - ep["mean_train"], ep["y_test"] - ep["mean_test"]])
        ratios.append(float(resid.pow(2).mean()) / float(ep["nugget"]))
    # E[(y - m)^2] = sigma_n^2 per row; 16 episodes x 18 rows average this to within ~10%.
    assert abs(sum(ratios) / len(ratios) - 1.0) < 0.2


def test_lambda_one_matches_the_zero_mean_gp_path() -> None:
    """A zero mean with kernel_factor 1 draws exactly the existing GP path's episodes."""
    cfg = _cfg(**LAMBDA_ONE)
    seed_everything(3)
    shape = _sample_call_shape(cfg, 8, "cpu", None, None, 0.0)
    priors = _sample_episode_priors(cfg, shape, "cpu")
    feats = _sample_features(cfg, shape, "cpu", with_gates=False)
    zero = torch.zeros(shape.B, shape.T)
    scm = _draw_gp(cfg, shape, priors, feats, "cpu", _ResidualMean(mean=zero, kernel_factor=torch.ones(shape.B)))
    gp = _draw_gp(cfg, shape, priors, feats, "cpu")
    assert scm is not None and gp is not None
    assert torch.allclose(scm.R_star, gp.R_star, atol=1e-5)
    K = priors.kernel_obj(feats.x_kernel).to_dense()
    noise = priors.likelihood.noise.reshape(shape.B)
    P, N = shape.P, shape.N
    for b in range(shape.B):
        R_ref, _ = sigma_to_correlation(K[b, P : P + N, P : P + N] + noise[b] * torch.eye(N))
        assert torch.allclose(scm.R_star[b], R_ref, atol=1e-5)


def test_conditional_oracle_matches_monte_carlo() -> None:
    episodes = generate_gp_batch(_cfg(P=8, N=5, **PINNED_MEAN), 1, return_kernel_metadata=True)
    ep = episodes[0]
    assert int(ep["mean_family"]) == MEAN_FAMILY_SCM and 0.0 < float(ep["lambda_resid"]) < 1.0
    kernel_fn, nugget = _kernel_fn_from_task(ep)
    x_tr, x_te = kernel_space_inputs(ep)
    m_tr, m_te = _mean_train_from_task(ep, x_tr), _mean_train_from_task(ep, x_te)
    P, N = x_tr.shape[0], x_te.shape[0]
    # The rebuilt law is the generator's: prior covariance and mean at the queries.
    assert torch.allclose(kernel_fn(x_te, x_te) + nugget * torch.eye(N), ep["Sigma_star"], rtol=1e-4, atol=1e-5)
    assert torch.equal(m_te, ep["mu_star"])

    x = torch.cat([x_tr, x_te]).double()
    m = torch.cat([m_tr, m_te]).double()
    C = kernel_fn(x, x).double() + nugget * torch.eye(P + N, dtype=torch.float64)
    gen = torch.Generator().manual_seed(0)
    y = m + torch.randn(40_000, P + N, generator=gen, dtype=torch.float64) @ torch.linalg.cholesky(C).T
    post = gp_analytical_posterior(ep)
    gain = C[P:, :P] @ torch.linalg.inv(C[:P, :P])  # (N, P)
    mu_post = m[P:] + (y[:, :P] - m[:P]) @ gain.T  # (S, N)
    z = (y[:, P:] - mu_post) / post["Sigma_post"].double().diagonal().sqrt()
    assert z.mean(0).abs().max() < 0.03
    assert (z.var(0) - 1.0).abs().max() < 0.05
    R_emp = torch.corrcoef(z.T)
    assert torch.allclose(R_emp, post["R_post"].double(), atol=0.03)


DECOUPLED: dict[str, Any] = {"lambda_p0": 0.0, "lambda_p1": 0.0}


def test_decoupled_share_is_induced_by_the_two_amplitudes() -> None:
    episodes = generate_gp_batch(_cfg(**DECOUPLED), 16, return_kernel_metadata=True)
    shares = []
    for ep in episodes:
        validate_episode(ep)
        # The kernel keeps its own output scale; lambda = mean_i K_ii / (a_g^2 + mean_i K_ii).
        assert float(ep["kernel_scale"]) == 1.0
        k_raw, _ = _unscaled_kernel_fn_from_task(ep)
        x = torch.cat(kernel_space_inputs(ep))
        k_bar = float(k_raw(x, x).diagonal().mean())
        a_g2 = float(torch.cat([ep["mean_train"], ep["mean_test"]]).var())
        assert abs(float(ep["lambda_resid"]) - k_bar / (a_g2 + k_bar)) < 1e-3
        shares.append(float(ep["lambda_resid"]))
    # a_g^2 ~ LogNormal(0, 1.5^2) spreads the share over (0, 1).
    assert min(shares) < 0.3 and max(shares) > 0.7


def test_decoupled_endpoints() -> None:
    iid = generate_gp_batch(_cfg(**LAMBDA_ZERO), 4, return_kernel_metadata=True)
    for ep in iid:
        assert float(ep["lambda_resid"]) == 0.0 and float(ep["kernel_scale"]) == 0.0
        assert torch.allclose(ep["R_star"], torch.eye(int(ep["n_test"])), atol=1e-5)
    # lambda = 1 is the zero-mean GP prior with the kernel's own amplitude: R* = corr(K_QQ + sigma_n^2 I).
    gp = generate_gp_batch(_cfg(**LAMBDA_ONE), 4, return_kernel_metadata=True)
    for ep in gp:
        assert float(ep["lambda_resid"]) == 1.0 and not bool(ep["mean_test"].any())
        k_raw, nugget = _unscaled_kernel_fn_from_task(ep)
        _, x_te = kernel_space_inputs(ep)
        R_ref, _ = sigma_to_correlation(k_raw(x_te, x_te) + nugget * torch.eye(x_te.shape[0]))
        assert torch.allclose(ep["R_star"], R_ref, atol=1e-5)


@pytest.mark.parametrize("amps", [PINNED_MEAN, DECOUPLED])
def test_analytic_pit_matches_the_generator(amps: dict[str, Any]) -> None:
    for ep in generate_gp_batch(_cfg(**amps), 3, return_kernel_metadata=True):
        assert torch.allclose(gp_analytical_pit(ep)["z_test"], ep["z_test"], atol=1e-3)


def test_saved_mean_drives_the_analytic_pit() -> None:
    episodes = generate_gp_batch(_cfg(**PINNED_MEAN), 4, return_kernel_metadata=True)
    for ep in episodes:
        cached = gp_analytical_pit(ep)
        fresh = gp_analytical_pit({k: v for k, v in ep.items() if k not in ("_L_ff", "_alpha")})
        for out in (cached, fresh):
            assert torch.allclose(out["z_test"], ep["z_test"], atol=1e-3)
            assert torch.allclose(out["z_train"], ep["z_train"], atol=1e-3)
        # A fold subset is matched row by row; a foreign row is refused.
        x_tr, _ = kernel_space_inputs(ep)
        assert torch.equal(saved_mean_at(ep, x_tr[3:7]), ep["mean_train"][3:7])
        with pytest.raises(KeyError):
            saved_mean_at(ep, x_tr[:2] + 10.0)
        with pytest.raises(NotImplementedError):
            bayes_predictive(_cfg().data, ep, 8, torch.device("cpu"), seed=0)


def test_analytic_pit_is_refused_unless_for_diagnostics() -> None:
    cfg = _cfg()
    cfg.data.z_train_source = "analytic"
    with pytest.raises(ValueError, match="leaks"):
        generate_gp_batch(cfg, 2)
    cfg.data.scm_residual.allow_analytic_pit = True
    assert len(generate_gp_batch(cfg, 2)) == 2


def test_only_the_tabiclv2_graph_prior_is_accepted() -> None:
    with pytest.raises(ValueError, match="graph_scm"):
        generate_gp_batch(_cfg(prior_type="mix_scm"), 2)


def test_episodes_satisfy_the_contracts_and_align_rows() -> None:
    P, N, d = 10, 7, 5
    # The calibration rows of tabicl_split come after train and test; lambda=0 makes y - m pure noise,
    # so a misaligned mean vector would show up as a large residual.
    episodes = generate_gp_batch(
        _cfg(P=P, N=N, d=d, **LAMBDA_ZERO), 12, return_kernel_metadata=True, tabicl_split_calib_frac=0.5
    )
    assert len(episodes) == 12
    for ep in episodes:
        validate_episode(ep)
        # The feature count comes from TabICLv2's prior (one per call), not from data.d_features.
        d_ep = ep["x_norm_train"].shape[1]
        assert d_ep == episodes[0]["x_norm_train"].shape[1] and 1 <= d_ep <= 100
        assert ep["x_norm_test"].shape == (N, d_ep) and ep["x_norm_train"].shape[0] == P
        assert ep["mean_train"].shape == (P,) and ep["mean_test"].shape == (N,)
        noise_sd = float(ep["nugget"]) ** 0.5
        assert float((ep["y_train"] - ep["mean_train"]).abs().max()) < 6 * noise_sd
        assert float((ep["y_test"] - ep["mean_test"]).abs().max()) < 6 * noise_sd
        assert int(ep["scm_prior_type"]) == SCM_PRIOR_CODES["graph_scm"]
        assert not bool(ep["mean_weight"].any())
    with pytest.raises(ValueError, match="mean_train"):
        validate_episode(episodes[0] | {"mean_train": torch.zeros(P + 1)})


def test_gp_episodes_of_an_scm_run_share_the_schema() -> None:
    episodes = generate_gp_batch(_cfg(prob=0.0), 3, return_kernel_metadata=True)
    for ep in episodes:
        validate_episode(ep)
        assert float(ep["lambda_resid"]) == -1.0 and float(ep["kernel_scale"]) == 1.0
        assert int(ep["mean_family"]) != MEAN_FAMILY_SCM
        x_tr, _ = kernel_space_inputs(ep)
        assert torch.allclose(_mean_train_from_task(ep, x_tr), ep["mean_train"], atol=1e-5)
        k_scaled, _ = _kernel_fn_from_task(ep)
        k_raw, _ = _unscaled_kernel_fn_from_task(ep)
        assert torch.equal(k_scaled(x_tr, x_tr), k_raw(x_tr, x_tr))


def test_same_seed_same_episodes() -> None:
    runs = [generate_gp_batch(_cfg(seed=11), 6, return_kernel_metadata=True) for _ in range(2)]
    for a, b in zip(*runs):
        for key, value in a.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, b[key]), key
    seed_everything(5)
    first = sample_scm_tables(3, 20, 3, "graph_scm", device="cpu")
    seed_everything(5)
    second = sample_scm_tables(3, 20, 3, "graph_scm", device="cpu")
    assert torch.equal(first.X, second.X) and torch.equal(first.g, second.g)


def test_graph_tables_are_filtered_like_tabiclv2(monkeypatch: pytest.MonkeyPatch) -> None:
    """A table TabICLv2's should_filter rejects is redrawn, and an always-rejected episode is invalid."""
    import copula_inter.data.scm_prior as scm_prior

    dataset = scm_prior._prior_module("_dataset")
    calls: list[int] = []

    def reject_all(*args: Any, **kwargs: Any) -> bool:
        calls.append(1)
        return True

    monkeypatch.setattr(dataset, "should_filter", reject_all)
    seed_everything(0)
    draw = sample_scm_tables(2, 40, 3, "graph_scm", device="cpu")
    assert not bool(draw.valid.any())
    assert len(calls) <= 2 * scm_prior._MAX_TABLE_ATTEMPTS
    assert torch.isfinite(draw.X).all() and torch.isfinite(draw.g).all()


def test_fixed_sets_synthesize_scm_episodes_in_float64() -> None:
    """Validation probes synthesize under a float64 default dtype; the SCM tables must still draw."""
    episodes = generate_fixed_gp_batch(_cfg(**PINNED_MEAN), 3, return_kernel_metadata=True)
    assert len(episodes) == 3 and torch.get_default_dtype() == torch.float32
    for ep in episodes:
        validate_episode(ep)
        assert ep["y_train"].dtype == torch.float32


def test_scm_tables_have_the_requested_shape() -> None:
    seed_everything(0)
    draw = sample_scm_tables(4, 30, 6, "graph_scm", device="cpu")
    assert draw.X.shape == (4, 30, 6) and draw.g.shape == (4, 30)
    assert bool(draw.valid.all())
    assert torch.allclose(draw.g.mean(1), torch.zeros(4), atol=1e-5)
    assert torch.allclose(draw.g.std(1), torch.ones(4), atol=1e-4)


def test_amplitude_mixture_weights_and_location() -> None:
    seed_everything(0)
    settings = ScmResidualSettings(lambda_p0=0.2, lambda_p1=0.3, mean_amp_lognormal_loc=-1.0)
    amps = sample_amplitudes(settings, 20_000, "cpu")
    iid = amps.kernel_factor == 0
    gp_only = amps.mean_amp2 == 0
    assert abs(float(iid.float().mean()) - 0.2) < 0.015 and bool((amps.mean_amp2[iid] == 1).all())
    assert abs(float(gp_only.float().mean()) - 0.3) < 0.015
    inner = amps.mean_amp2[~iid & ~gp_only]
    assert abs(float(inner.log().median()) + 1.0) < 0.05  # LogNormal(-1, 1.5^2)


def test_scm_feature_count_follows_the_tabiclv2_prior() -> None:
    """d ~ round(Uniform(1, 100)) per call, whatever data.d_features says."""
    ds = {generate_gp_batch(_cfg(seed=s, d=4), 1)[0]["x_norm_train"].shape[1] for s in range(12)}
    assert len(ds) > 3 and all(1 <= d <= 100 for d in ds) and max(ds) > 20


def test_the_removed_coupled_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="coupled mode was removed"):
        generate_gp_batch(_cfg(amplitude_mode="coupled"), 2)


def test_manifest_identity_ignores_a_disabled_scm_prior() -> None:
    base = OmegaConf.create({"seed": 1, "data": {"P_min": 4, "z_train_source": "analytic"}})
    disabled = merge_configs(base, {"data": {"scm_residual": {"enabled": False, "prob": 0.3}}})
    enabled = merge_configs(base, {"data": {"scm_residual": {"enabled": True, "prob": 0.3}}})
    other = merge_configs(base, {"data": {"scm_residual": {"enabled": True, "prob": 0.5}}})
    assert generation_spec(base, None) == generation_spec(disabled, None)
    assert generation_spec(enabled, None) != generation_spec(base, None)
    assert generation_spec(enabled, None) != generation_spec(other, None)


def test_disabled_prior_draws_the_same_episodes_as_no_block() -> None:
    cfg = _cfg(seed=4)
    cfg.data.scm_residual.enabled = False
    without = OmegaConf.create(OmegaConf.to_container(cfg))
    without.data.pop("scm_residual")
    a = generate_gp_batch(cfg, 3, return_kernel_metadata=True)
    b = generate_gp_batch(without, 3, return_kernel_metadata=True)
    for ea, eb in zip(a, b):
        assert set(ea) == set(eb) and "lambda_resid" not in ea
        for key, value in ea.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, eb[key]), key
