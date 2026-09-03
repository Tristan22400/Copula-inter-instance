"""
test_marginal_finetune.py — Phase A (marginal fine-tuning) unit tests.

What these pin down, in the order the risk actually sits:

  1. **Tier routing** selects the parameters it claims to and nothing else —
     in particular the ICL stack's norms but NOT the identically-named norms in
     col_embedder/row_interactor, which is the whole reason the allowlist is
     regex rather than substrings.
  2. **The LoRA export round-trips.** A tier >= 1 checkpoint is written from a
     module whose attention has been structurally replaced; if
     merged_base_state_dict does not reproduce plain-TabICL key names AND the
     same forward output, the "drop-in tabicl.pit_ckpt" promise is silently
     false and only discovered a training run later.
  3. **The analytic target is the right quantity.** Two independent checks: it
     matches a brute-force MVN conditional built from the joint covariance, and
     it matches data_gen.gp_posterior(latent=False) — NOT latent=True. The
     latent/observable distinction is a silent, systematic over-sharpening of
     every target if it goes wrong, so it gets its own test.
  4. **The fold conditioning the target uses is the fold the model saw.**
     episode_fold_targets re-derives fold membership from ceil(P/K); this
     asserts it agrees with pit.py's own loop, so a change to one without the
     other fails here rather than quietly training against the wrong context.
  5. **The distillation term bottoms out at the analytic optimum** (a loss that
     is not zero at the thing it is distilling is not distilling it).
  6. **The grad path matches the no-grad path** numerically and actually
     produces finite gradients into the backbone.
  7. **The ERA5 train/val corpora do not overlap.** Cheap now; catches a future
     fetch_era5_global.py run landing overlapping months in both directories
     and silently leaking validation data into training.
"""

from __future__ import annotations

import glob
import math
import os
import re

import numpy as np
import pytest
import torch
import torch.nn as nn

from data_gen import build_kernel_fn, gp_posterior
from finetune_marginal import _generate_phase_a_gp_batch
from lora import merged_base_state_dict
from marginal_finetune import (
    TIER0_PATTERNS,
    AnchorPenalty,
    MarginalLossWeights,
    analytic_marginal_targets,
    apply_tier,
    episode_fold_targets,
    ks_uniform,
    marginal_objective,
    oracle_marginal_nll,
    phase_a_batch_loss,
    quantile_level_weights,
    rank_histogram,
)
from pit import _probit, run_pit_batched, run_pit_batched_grad

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def test_phase_a_generator_pins_shape_after_mixed_topup(monkeypatch):
    import finetune_marginal as entrypoint
    from omegaconf import OmegaConf

    calls = []

    def episode(P, N=3, d=2):
        return {"x_norm_train": torch.zeros(P, d), "x_norm_test": torch.zeros(N, d)}

    def fake_generate(cfg, B, device, **kwargs):
        calls.append((B, int(cfg.data.P_min), int(cfg.data.P_max), kwargs.get("d_override")))
        if len(calls) == 1:
            return [episode(4), episode(4), episode(7)]
        assert B == 1
        return [episode(4)]

    monkeypatch.setattr(entrypoint, "generate_gp_batch", fake_generate)
    cfg = OmegaConf.create({"seed": 9, "data": {"P_min": 2, "P_max": 8,
                                                  "N_min": 3, "N_max": 3}})
    got = _generate_phase_a_gp_batch(cfg, 3, "cpu")
    assert [ep["x_norm_train"].shape[0] for ep in got] == [4, 4, 4]
    assert calls[1][1:] == (4, 4, 2)


def _tiny_tabicl(num_quantiles: int = 33) -> nn.Module:
    """A structurally faithful but tiny TabICL (random init, no HF download).

    Every stage/module name this module's routing and export logic keys off is
    present — col_embedder.y_encoder, icl_predictor.{ln,y_encoder,decoder},
    icl_predictor.tf_icl.blocks.N.norm{1,2} — just at toy widths.
    """
    from tabicl._model.tabicl import TabICL  # type: ignore[import]

    return TabICL(
        max_classes=0,
        num_quantiles=num_quantiles,
        embed_dim=16,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        col_target_aware=True,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=2,
        icl_nhead=2,
        ff_factor=1,
        dropout=0.0,
    )


def _rbf_task(P: int = 12, N: int = 5, d: int = 2, ls: float = 0.7,
              alpha2: float = 1.3, nugget: float = 0.05, seed: int = 0) -> dict:
    """A minimal episode dict carrying exactly the keys _kernel_fn_from_task
    and _mean_train_from_task read, with a known RBF kernel and a genuine GP
    draw for y (so the analytic posterior really is the data-generating law)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(P + N, d, generator=g) * 2 - 1
    kfn = build_kernel_fn("rbf", ls, alpha2, active_dims=list(range(d)))
    K = kfn(x, x) + nugget * torch.eye(P + N)
    L = torch.linalg.cholesky(K.double())
    # A generated dataset is fixed supervision, not a differentiable function
    # of the temporary gpytorch kernel parameters. Detaching also permits the
    # same episode to be reused by the full-path overfit test below.
    y = (L @ torch.randn(P + N, 1, generator=g, dtype=torch.float64)).squeeze(-1).float().detach()

    zero = torch.tensor(0.0)
    return {
        "kernel": "rbf",
        "l": torch.tensor(ls),
        "alpha2": torch.tensor(alpha2),
        "nugget": torch.tensor(nugget),
        "period": zero, "rq_alpha": zero, "power": zero,
        "l_b": zero, "alpha2_b": zero, "period_b": zero,
        "rq_alpha_b": zero, "power_b": zero,
        "kernel_feature_indices": torch.arange(d),
        "mean_nonzero": torch.tensor(False),
        "x_norm_train": x[:P],
        "x_norm_test": x[P:],
        "y_train": y[:P],
        "y_test": y[P:],
    }


# ---------------------------------------------------------------------------
# 1. Tier routing
# ---------------------------------------------------------------------------


def test_tier0_selects_label_path_norms_and_decoder_only():
    m = _tiny_tabicl()
    report = apply_tier(m, 0)

    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable, "tier 0 left nothing trainable"
    assert report["lora_modules_replaced"] == 0, "tier 0 must install no adapters"

    pats = [re.compile(p) for p in TIER0_PATTERNS]
    for name in trainable:
        assert any(r.search(name) for r in pats), f"unexpected trainable param {name}"

    # The discriminating case: ICL-stack norms in, same-named norms elsewhere out.
    assert any(re.search(r"^icl_predictor\.tf_icl\.blocks\.\d+\.norm1\.", n) for n in trainable)
    for name in trainable:
        assert not name.startswith("row_interactor."), name
        assert not name.startswith("col_embedder.tf_col."), name

    # The decoder — the module that literally emits the marginal — must be in.
    assert any(n.startswith("icl_predictor.decoder.") for n in trainable)
    assert 0.0 < report["trainable_frac"] < 0.5


def test_tier1_adds_lora_on_icl_only_and_keeps_tier0():
    m = _tiny_tabicl()
    report = apply_tier(m, 1, lora_rank=4, lora_alpha=8.0)
    assert report["lora_modules_replaced"] > 0

    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    lora = {n for n in trainable if "lora_A_" in n or "lora_B_" in n}
    assert lora, "tier 1 installed no trainable adapters"
    assert all(n.startswith("icl_predictor.") for n in lora), sorted(lora)

    # Tier 0's selection survives the adapter install.
    assert any(n.startswith("icl_predictor.decoder.") for n in trainable)
    assert any(re.search(r"^icl_predictor\.tf_icl\.blocks\.\d+\.norm[12]\.", n) for n in trainable)
    assert report["n_trainable_params"] > apply_tier(_tiny_tabicl(), 0)["n_trainable_params"]


def test_tier3_adds_lora_to_every_backbone_stage():
    m = _tiny_tabicl()
    report = apply_tier(m, 3, lora_rank=2, lora_alpha=4.0)
    lora_names = {
        name for name, p in m.named_parameters()
        if p.requires_grad and ("lora_A_" in name or "lora_B_" in name)
    }
    assert report["lora_modules_replaced"] > 0
    assert any(name.startswith("col_embedder.") for name in lora_names)
    assert any(name.startswith("row_interactor.") for name in lora_names)
    assert any(name.startswith("icl_predictor.") for name in lora_names)


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        apply_tier(_tiny_tabicl(), 99)


# ---------------------------------------------------------------------------
# 2. LoRA export round-trip — the "drop-in pit_ckpt" promise
# ---------------------------------------------------------------------------


def test_merged_base_state_dict_loads_into_plain_tabicl_and_matches_forward():
    from tabicl._model.tabicl import TabICL  # type: ignore[import]

    torch.manual_seed(0)
    m = _tiny_tabicl()
    ref = _tiny_tabicl()
    ref.load_state_dict(m.state_dict())          # identical starting point

    apply_tier(m, 1, lora_rank=4, lora_alpha=8.0)
    # Perturb the adapters so the merge is a real merge, not a no-op on zeros.
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B_" in n:
                p.normal_(0.0, 0.05)

    sd = merged_base_state_dict(m)
    fresh = TabICL(**{
        "max_classes": 0, "num_quantiles": 33, "embed_dim": 16,
        "col_num_blocks": 1, "col_nhead": 2, "col_num_inds": 8,
        "col_target_aware": True, "row_num_blocks": 1, "row_nhead": 2,
        "row_num_cls": 2, "icl_num_blocks": 2, "icl_nhead": 2,
        "ff_factor": 1, "dropout": 0.0,
    })
    fresh.load_state_dict(sd)                    # strict — the actual assertion

    x = torch.randn(2, 14, 3)
    y = torch.randn(2, 10)
    m.train(); fresh.train()
    with torch.no_grad():
        a = m(x, y)
        b = fresh(x, y)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()

    # And the merge actually moved the weights off the pretrained base.
    with torch.no_grad():
        c = ref.train()(x, y)
    assert not torch.allclose(a, c, atol=1e-5), "adapter perturbation had no effect"


# ---------------------------------------------------------------------------
# 3. The analytic target is the right quantity
# ---------------------------------------------------------------------------


def test_analytic_marginal_matches_brute_force_mvn_conditional():
    """mu_i, sigma_i against the textbook Gaussian conditional built from the
    full joint covariance — an independent construction, not a rearrangement of
    the same Cholesky."""
    task = _rbf_task(P=10, N=4, seed=1)
    x_ctx, y_ctx = task["x_norm_train"], task["y_train"]
    x_qry = task["x_norm_test"]
    kfn = build_kernel_fn("rbf", 0.7, 1.3, active_dims=[0, 1])
    nug = float(task["nugget"])

    mu, sigma = analytic_marginal_targets(task, x_ctx, y_ctx, x_qry)

    x_all = torch.cat([x_ctx, x_qry]).double()
    K = kfn(x_all, x_all).double() + nug * torch.eye(x_all.shape[0], dtype=torch.float64)
    P = x_ctx.shape[0]
    K_ff, K_sf, K_ss = K[:P, :P], K[P:, :P], K[P:, P:]
    sol = torch.linalg.solve(K_ff, y_ctx.double())
    mu_bf = K_sf @ sol
    Sig_bf = K_ss - K_sf @ torch.linalg.solve(K_ff, K_sf.T)

    assert torch.allclose(mu, mu_bf.float(), atol=1e-4), (mu - mu_bf.float()).abs().max()
    assert torch.allclose(sigma, Sig_bf.diagonal().sqrt().float(), atol=1e-4)


def test_analytic_target_is_observable_y_not_latent_f():
    """The nugget must be IN the variance. gp_posterior defaults to latent=True
    (posterior over f*, noise excluded); using that would make every Phase-A
    target systematically over-sharp, which no downstream metric would flag as
    anything other than 'the model is underconfident'."""
    task = _rbf_task(P=10, N=4, seed=2)
    kfn = build_kernel_fn("rbf", 0.7, 1.3, active_dims=[0, 1])
    nug = float(task["nugget"])
    _, sigma = analytic_marginal_targets(task, task["x_norm_train"], task["y_train"],
                                         task["x_norm_test"])

    _, S_obs = gp_posterior(task["x_norm_train"], task["y_train"], task["x_norm_test"],
                            kfn, nug, latent=False)
    _, S_lat = gp_posterior(task["x_norm_train"], task["y_train"], task["x_norm_test"],
                            kfn, nug, latent=True)

    assert torch.allclose(sigma, S_obs.diagonal().clamp(min=0).sqrt(), atol=1e-4)
    assert not torch.allclose(sigma, S_lat.diagonal().clamp(min=0).sqrt(), atol=1e-3)
    # And the difference is exactly the nugget, on the variance scale.
    assert torch.allclose(sigma ** 2 - S_lat.diagonal(), torch.full_like(sigma, nug), atol=1e-4)


def test_analytic_target_contracts_with_more_context():
    """Posterior variance must shrink as context grows — the property Phase A
    exists to teach, so a target that did not have it would be teaching the
    wrong lesson."""
    task = _rbf_task(P=24, N=4, seed=3)
    x, y = task["x_norm_train"], task["y_train"]
    _, s_small = analytic_marginal_targets(task, x[:4], y[:4], task["x_norm_test"])
    _, s_big = analytic_marginal_targets(task, x, y, task["x_norm_test"])
    assert (s_big <= s_small + 1e-6).all(), (s_small, s_big)


def _with_cached_full_factors(task: dict) -> dict:
    task = dict(task)
    kfn = build_kernel_fn("rbf", 0.7, 1.3, active_dims=[0, 1])
    x, y = task["x_norm_train"], task["y_train"]
    K = kfn(x, x).double() + float(task["nugget"]) * torch.eye(len(x), dtype=torch.float64)
    L = torch.linalg.cholesky(K)
    task["_L_ff"] = L
    task["_alpha"] = torch.cholesky_solve(y.double().unsqueeze(-1), L).squeeze(-1)
    return task


def test_cached_full_context_target_matches_direct_recomputation():
    task = _with_cached_full_factors(_rbf_task(P=18, N=7, seed=31))
    args = (task, task["x_norm_train"], task["y_train"], task["x_norm_test"])
    direct = analytic_marginal_targets(*args)
    cached = analytic_marginal_targets(*args, use_cached_full_context=True)
    assert torch.allclose(cached[0], direct[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(cached[1], direct[1], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 4. Fold conditioning agreement between target and model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K", [3, 5, 10])
def test_episode_fold_targets_excludes_the_query_row_from_its_own_context(K):
    """The leakage guarantee, checked at the target side. A row's analytic
    target must be conditioned on a context that does not contain that row —
    otherwise the target is a memorization target and the model is rewarded for
    leaking."""
    P = 12
    task = _rbf_task(P=P, N=2, seed=4)
    idx = torch.arange(P)
    mu, sigma = episode_fold_targets(task, idx, K)

    fold_size = math.ceil(P / max(2, min(K, P)))
    x, y = task["x_norm_train"], task["y_train"]
    for i in range(P):
        k = i // fold_size
        ctx = torch.tensor([j for j in range(P) if not (k * fold_size <= j < min((k + 1) * fold_size, P))])
        mu_i, sig_i = analytic_marginal_targets(task, x[ctx], y[ctx], x[i:i + 1])
        assert torch.allclose(mu[i], mu_i[0], atol=1e-4), i
        assert torch.allclose(sigma[i], sig_i[0], atol=1e-4), i


@pytest.mark.parametrize("K", [3, 5, 10])
def test_cached_precision_fold_targets_match_direct_conditioning(K):
    task = _with_cached_full_factors(_rbf_task(P=17, N=2, seed=32 + K))
    idx = torch.tensor([0, 1, 5, 8, 12, 16])
    direct_task = {k: v for k, v in task.items() if k not in ("_L_ff", "_alpha")}
    direct = episode_fold_targets(direct_task, idx, K)
    cached = episode_fold_targets(task, idx, K)
    assert torch.allclose(cached[0], direct[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(cached[1], direct[1], atol=1e-5, rtol=1e-5)


def test_fold_subset_rows_match_a_full_pit_pass():
    """pit.run_pit_batched(fold_subset=...) must keep the fold GEOMETRY of a
    full pass — the scored rows' values have to be bit-identical to the same
    rows of a complete run, or Phase A trains under conditioning that does not
    exist at deployment."""
    from tests.test_pit_batched import RowIndependentFakeTabICL  # noqa: PLC0415

    torch.manual_seed(0)
    B, P, N, K = 2, 12, 3, 4
    tab = RowIndependentFakeTabICL(q=5)
    Xtr, Ytr = torch.randn(B, P, 2), torch.randn(B, P, 1)
    Xte, Yte = torch.randn(B, N, 2), torch.randn(B, N, 1)

    full = run_pit_batched(tab, Xtr, Ytr, Xte, Yte, k_folds=K)
    part = run_pit_batched(tab, Xtr, Ytr, Xte, Yte, k_folds=K, return_quantiles=True)
    assert torch.allclose(full["z_train"], part["z_train"])

    from pit import _run_pit_batched_impl

    sub = _run_pit_batched_impl(tab, Xtr, Ytr, Xte, Yte, K, 1e-6,
                                return_quantiles=True, fold_subset=[1, 3])
    rows = sub["train_query_idx"]
    assert rows.numel() > 0
    assert torch.allclose(sub["z_train"], full["z_train"][:, rows, :], atol=0)
    assert torch.allclose(sub["z_test"], full["z_test"], atol=0)


def test_fold_subset_empty_returns_test_only():
    from tests.test_pit_batched import RowIndependentFakeTabICL  # noqa: PLC0415
    from pit import _run_pit_batched_impl

    torch.manual_seed(0)
    tab = RowIndependentFakeTabICL(q=5)
    Xtr, Ytr = torch.randn(1, 8, 2), torch.randn(1, 8, 1)
    Xte, Yte = torch.randn(1, 3, 2), torch.randn(1, 3, 1)
    full = run_pit_batched(tab, Xtr, Ytr, Xte, Yte, k_folds=4)
    only = _run_pit_batched_impl(tab, Xtr, Ytr, Xte, Yte, 4, 1e-6,
                                 return_quantiles=True, fold_subset=[])
    assert "z_train" not in only
    assert torch.allclose(only["z_test"], full["z_test"], atol=0)
    assert only["q_test"].shape[:3] == (1, 3, 1)


def test_quantiles_only_fast_path_skips_pit_but_preserves_decoder_output():
    from tests.test_pit_batched import RowIndependentFakeTabICL  # noqa: PLC0415
    from pit import _run_pit_batched_impl

    torch.manual_seed(0)
    tab = RowIndependentFakeTabICL(q=7)
    Xtr, Ytr = torch.randn(2, 9, 2), torch.randn(2, 9, 1)
    Xte, Yte = torch.randn(2, 4, 2), torch.randn(2, 4, 1)
    full = _run_pit_batched_impl(
        tab, Xtr, Ytr, Xte, Yte, 3, 1e-6,
        return_quantiles=True, fold_subset=[0, 2],
    )
    fast = _run_pit_batched_impl(
        tab, Xtr, Ytr, Xte, Yte, 3, 1e-6,
        return_quantiles=True, fold_subset=[0, 2], compute_pit=False,
    )
    assert torch.equal(fast["train_query_idx"], full["train_query_idx"])
    assert torch.equal(fast["q_train"], full["q_train"])
    assert torch.equal(fast["q_test"], full["q_test"])
    assert not ({"z_train", "z_test", "log_pdf_test", "u_train", "u_test"} & fast.keys())


def test_fused_fold_forward_matches_separate_forwards():
    from tests.test_pit_batched import RowIndependentFakeTabICL  # noqa: PLC0415
    from pit import _run_pit_batched_impl

    torch.manual_seed(0)
    tab = RowIndependentFakeTabICL(q=7)
    Xtr, Ytr = torch.randn(2, 17, 2), torch.randn(2, 17, 1)
    Xte, Yte = torch.randn(2, 4, 2), torch.randn(2, 4, 1)
    kwargs = dict(return_quantiles=True, fold_subset=[0, 2, 4], compute_pit=False)
    separate = _run_pit_batched_impl(tab, Xtr, Ytr, Xte, Yte, 5, 1e-6, **kwargs)
    fused = _run_pit_batched_impl(
        tab, Xtr, Ytr, Xte, Yte, 5, 1e-6, fuse_folds=True, **kwargs
    )
    assert torch.equal(fused["train_query_idx"], separate["train_query_idx"])
    assert torch.equal(fused["q_train"], separate["q_train"])
    assert torch.equal(fused["q_test"], separate["q_test"])


# ---------------------------------------------------------------------------
# 5. The objective bottoms out where it should
# ---------------------------------------------------------------------------


def test_distillation_is_zero_at_the_analytic_optimum():
    m = _tiny_tabicl(num_quantiles=33)
    qd = m.quantile_dist
    alpha = qd.alpha_levels
    M = 7
    mu = torch.randn(M)
    sigma = torch.rand(M) + 0.5
    q_star = mu.unsqueeze(-1) + sigma.unsqueeze(-1) * _probit(alpha).unsqueeze(0)

    w = MarginalLossWeights(distill=1.0, nll=0.0, crps=0.0)
    out = marginal_objective(q_star, mu.clone(), qd, w, mu=mu, sigma=sigma)
    assert float(out["distill"]) < 1e-10, float(out["distill"])

    # And strictly positive once the prediction is wrong.
    worse = marginal_objective(q_star + 0.3, mu.clone(), qd, w, mu=mu, sigma=sigma)
    assert float(worse["distill"]) > 1e-4


def test_distillation_respects_target_mask():
    m = _tiny_tabicl(num_quantiles=33)
    qd = m.quantile_dist
    alpha = qd.alpha_levels
    M = 6
    mu = torch.randn(M)
    sigma = torch.rand(M) + 0.5
    q = mu.unsqueeze(-1) + sigma.unsqueeze(-1) * _probit(alpha).unsqueeze(0)
    q[:3] += 1.0                                # first three deliberately wrong

    w = MarginalLossWeights(distill=1.0, nll=0.0, crps=0.0)
    mask = torch.tensor([False] * 3 + [True] * 3)
    masked = marginal_objective(q, mu.clone(), qd, w, mu=mu, sigma=sigma, target_mask=mask)
    unmasked = marginal_objective(q, mu.clone(), qd, w, mu=mu, sigma=sigma)
    assert float(masked["distill"]) < 1e-10
    assert float(unmasked["distill"]) > 1e-4

    # An all-False mask degrades to "no distillation", not to a crash or a NaN.
    none = marginal_objective(q, mu.clone(), qd, w, mu=mu, sigma=sigma,
                              target_mask=torch.zeros(M, dtype=torch.bool))
    assert float(none["distill"]) == 0.0


def test_zero_nll_weight_excludes_nll_from_training_loss_but_still_reports_it():
    """The shipped objective monitors density NLL without backpropagating its
    sparse two-knot gradient into the raw 999-quantile decoder."""
    m = _tiny_tabicl(num_quantiles=33)
    qd = m.quantile_dist
    alpha = qd.alpha_levels
    mu = torch.tensor([0.2, -0.4])
    sigma = torch.tensor([0.7, 1.3])
    q = (mu[:, None] + sigma[:, None] * _probit(alpha)[None]).requires_grad_()
    y = torch.tensor([0.1, 0.3])

    weights = MarginalLossWeights(distill=1.0, nll=0.0, crps=0.0, pinball=1.0)
    out = marginal_objective(q, y, qd, weights, mu=mu, sigma=sigma)

    assert torch.isfinite(out["nll"]), "NLL must remain available for logging"
    assert torch.allclose(out["loss"], out["distill"] + out["pinball"])
    grad = torch.autograd.grad(out["loss"], q)[0]
    assert torch.isfinite(grad).all()
    assert (grad != 0).all(), "pinball/distillation should provide dense quantile gradients"


def test_pinball_uses_raw_quantile_identity_and_penalizes_permutation():
    m = _tiny_tabicl(num_quantiles=33)
    qd = m.quantile_dist
    alpha = qd.alpha_levels
    # Deterministic inverse-CDF samples make this a low-variance numerical
    # approximation to the population score under N(0, 1).
    y = _probit(torch.linspace(0.0005, 0.9995, 2001))
    q_star = _probit(alpha).expand(y.numel(), -1)
    q_reversed = q_star.flip(-1)
    weights = MarginalLossWeights(distill=0.0, nll=0.0, crps=0.0, pinball=1.0)

    good = marginal_objective(q_star, y, qd, weights)["pinball"]
    crossed = marginal_objective(q_reversed, y, qd, weights)["pinball"]
    assert crossed > good * 2


def test_exact_distillation_can_overfit_one_predictive_distribution():
    """A direct single-episode analogue at the decoder boundary: optimizing
    the diagnostic loss must recover its known analytic quantiles."""
    torch.manual_seed(7)
    m = _tiny_tabicl(num_quantiles=33)
    qd = m.quantile_dist
    alpha = qd.alpha_levels
    mu = torch.tensor([0.3, -0.7, 1.1])
    sigma = torch.tensor([0.4, 0.9, 1.5])
    target = mu[:, None] + sigma[:, None] * _probit(alpha)[None]
    q = (target + 0.5 * torch.randn_like(target)).requires_grad_()
    opt = torch.optim.Adam([q], lr=0.05)
    weights = MarginalLossWeights(distill=1.0, nll=0.0, crps=0.0)

    initial = marginal_objective(q, mu, qd, weights, mu=mu, sigma=sigma)["loss"].item()
    for _ in range(100):
        opt.zero_grad(set_to_none=True)
        loss = marginal_objective(q, mu, qd, weights, mu=mu, sigma=sigma)["loss"]
        loss.backward()
        opt.step()
    final = marginal_objective(q, mu, qd, weights, mu=mu, sigma=sigma)["loss"].item()

    assert final < initial * 1e-2, (initial, final)


def test_distillation_does_not_inverse_variance_weight_sharp_rows():
    """Equal quantile errors should have equal gradients regardless of the
    query's posterior sigma. The old z-standardized loss amplified the first
    row by 1 / 0.05 and let sharp GP points dominate clipped updates."""
    m = _tiny_tabicl(num_quantiles=33)
    qd = m.quantile_dist
    alpha = qd.alpha_levels
    mu = torch.zeros(2)
    sigma = torch.tensor([0.05, 2.0])
    target = mu[:, None] + sigma[:, None] * _probit(alpha)[None]
    q = (target + 0.01).requires_grad_()
    out = marginal_objective(
        q, mu, qd,
        MarginalLossWeights(distill=1.0, nll=0.0, crps=0.0, pinball=0.0),
        mu=mu, sigma=sigma,
    )
    grad = torch.autograd.grad(out["loss"], q)[0]
    assert torch.allclose(grad[0], grad[1], rtol=1e-5, atol=1e-8)


def test_full_phase_a_path_overfits_one_fixed_synthetic_episode():
    """Exercise model -> folded PIT -> analytic target -> loss -> optimizer,
    rather than only optimizing a free tensor at the decoder boundary."""
    torch.manual_seed(11)
    model = _tiny_tabicl(num_quantiles=33)
    episode = _rbf_task(P=8, N=4, d=2, seed=19)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=3e-3, weight_decay=0.0)
    weights = MarginalLossWeights(
        distill=1.0, nll=0.0, crps=0.0, pinball=0.0,
    )

    initial = phase_a_batch_loss(
        model, [episode], weights, k_folds=2, device="cpu"
    )["loss"].detach().item()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        result = phase_a_batch_loss(
            model, [episode], weights, k_folds=2, device="cpu"
        )
        result["loss"].backward()
        optimizer.step()
    final = phase_a_batch_loss(
        model, [episode], weights, k_folds=2, device="cpu"
    )["loss"].detach().item()

    assert final < initial * 0.2, (initial, final)


def test_quantile_level_weights_downweight_the_tails_and_average_to_one():
    a = torch.linspace(0.001, 0.999, 999)
    w = quantile_level_weights(a, tail_power=0.5)
    assert abs(float(w.mean()) - 1.0) < 1e-5
    assert float(w[0]) < float(w[len(w) // 2])
    assert float(w[-1]) < float(w[len(w) // 2])
    flat = quantile_level_weights(a, tail_power=0.0)
    assert torch.allclose(flat, torch.ones_like(flat))


def test_oracle_marginal_nll_matches_closed_form_gaussian():
    y = torch.tensor([0.3, -1.2])
    mu = torch.tensor([0.0, -1.0])
    sd = torch.tensor([1.0, 2.0])
    expect = float(
        torch.mean(0.5 * (torch.log(2 * math.pi * sd ** 2) + ((y - mu) / sd) ** 2))
    )
    assert abs(oracle_marginal_nll(y, mu, sd) - expect) < 1e-6


def test_anchor_penalty_is_zero_at_init_and_grows_with_drift():
    m = _tiny_tabicl()
    apply_tier(m, 0)
    anchor = AnchorPenalty(m)
    assert anchor(m).detach().item() == pytest.approx(0.0, abs=1e-12)
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.add_(0.01)
    assert anchor(m).detach().item() > 0.0


# ---------------------------------------------------------------------------
# 6. Grad path
# ---------------------------------------------------------------------------


def test_grad_pit_matches_nograd_pit_and_produces_finite_grads():
    torch.manual_seed(0)
    m = _tiny_tabicl(num_quantiles=33)
    apply_tier(m, 0)
    B, P, N = 2, 9, 3
    Xtr, Ytr = torch.randn(B, P, 3), torch.randn(B, P, 1)
    Xte, Yte = torch.randn(B, N, 3), torch.randn(B, N, 1)

    ref = run_pit_batched(m, Xtr, Ytr, Xte, Yte, k_folds=3)
    out = run_pit_batched_grad(m, Xtr, Ytr, Xte, Yte, k_folds=3)

    assert torch.allclose(ref["z_train"], out["z_train"], atol=1e-6)
    assert torch.allclose(ref["z_test"], out["z_test"], atol=1e-6)
    assert torch.allclose(ref["log_pdf_test"], out["log_pdf_test"], atol=1e-6)
    assert out["z_train"].requires_grad and not ref["z_train"].requires_grad

    loss = -out["log_pdf_test"].mean()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients reached any trainable parameter"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_grad_path_leaves_train_mode_as_it_found_it():
    m = _tiny_tabicl(num_quantiles=33)
    m.eval()
    Xtr, Ytr = torch.randn(1, 6, 2), torch.randn(1, 6, 1)
    Xte, Yte = torch.randn(1, 2, 2), torch.randn(1, 2, 1)
    run_pit_batched_grad(m, Xtr, Ytr, Xte, Yte, k_folds=3)
    assert not m.training, "_train_mode must restore the module's original mode"


# ---------------------------------------------------------------------------
# 7. Calibration diagnostics
# ---------------------------------------------------------------------------


def test_ks_uniform_small_for_uniform_large_for_shifted():
    rng = np.random.default_rng(0)
    u = rng.uniform(size=4000)
    assert ks_uniform(u) < 0.05
    assert ks_uniform(rng.beta(2.0, 5.0, size=4000)) > 0.15


def test_rank_histogram_flat_for_uniform():
    rng = np.random.default_rng(0)
    h = rank_histogram(rng.uniform(size=20000), n_bins=10)
    assert abs(h.sum() - 1.0) < 1e-9
    assert np.max(np.abs(h - 0.1)) < 0.02


# ---------------------------------------------------------------------------
# 8. ERA5 train/val corpus disjointness
# ---------------------------------------------------------------------------


def _corpus_months(dirname: str) -> set[str]:
    path = os.path.join(_REPO, "eval", "data", "cache", dirname)
    out = set()
    for f in glob.glob(os.path.join(path, "era5_global_t2m_*.nc")):
        m = re.search(r"era5_global_t2m_(\d{6})\.nc$", os.path.basename(f))
        if m:
            out.add(m.group(1))
    return out


def test_era5_train_and_val_corpora_are_disjoint():
    """Guard, not a split mechanism. The 2013-2022 / 2023 boundary already
    exists on disk and Phase A keeps it; this catches a future
    fetch_era5_global.py invocation landing overlapping months into both
    directories, which would leak validation data into training with no other
    visible symptom than a suspiciously good val curve."""
    train = _corpus_months("era5_global_train")
    val = _corpus_months("era5_global_val")
    if not train or not val:
        pytest.skip("ERA5 global corpora not fetched in this checkout")
    assert not (train & val), f"overlapping months: {sorted(train & val)}"
