"""eval.baselines.classical.fit_and_eval_gpytorch_batched.

The batched fitter exists so a classical GP baseline can be scored on the whole
validation set at training time (~2.5 s/episode/kernel sequentially is ~40 min
for 500 episodes x 2 kernels). It is only worth having if it is the SAME fit as
the per-episode function everything else in the repo already uses, so that is
what these check -- plus the one modelling decision behind the baseline: it is
fitted on y_train, never on z_train.
"""

import math

import pytest
import torch
from omegaconf import OmegaConf

from data_gen import generate_gp_batch
from eval.baselines.classical import (
    fit_and_eval_gpytorch,
    fit_and_eval_gpytorch_batched,
)
from pit import gp_analytical_pit, gp_analytical_posterior

KERNELS = ["matern32", "rational_quadratic"]
N_STEPS = 60
LR = 0.05


def _episodes(small_cfg, n, *, P=12, N=16, d=2, seed=7):
    cfg = OmegaConf.create(OmegaConf.to_container(small_cfg, resolve=True))
    cfg.seed = seed
    cfg.data.d_features = d
    cfg.data.P_min = cfg.data.P_max = P
    cfg.data.N_min = cfg.data.N_max = N
    cfg.data.kernel = "rbf"
    torch.manual_seed(seed)
    return generate_gp_batch(cfg, n, "cpu", return_kernel_metadata=True)


def _mean_abs_offdiag(R) -> float:
    n = R.shape[-1]
    iu = torch.triu_indices(n, n, offset=1)
    return float(R[..., iu[0], iu[1]].abs().mean())


@pytest.mark.parametrize("kernel_name", KERNELS)
def test_batched_e1_reproduces_the_per_episode_fit(small_cfg, kernel_name):
    """E=1 is the case where both paths draw the same number of random values
    in the same order (batched parameter shapes have the same numel), so with
    the same seed they are the same deterministic optimization and must agree
    numerically -- not merely land near the same optimum."""
    ep = _episodes(small_cfg, 1)[0]
    X_tr, y_tr, X_te = ep["x_norm_train"], ep["y_train"], ep["x_norm_test"]

    torch.manual_seed(123)
    single = fit_and_eval_gpytorch(
        X_tr, y_tr, X_te, kernel_name, n_steps=N_STEPS, lr=LR,
        oracle_mode="posterior", n_restarts=1,
    )
    torch.manual_seed(123)
    batched = fit_and_eval_gpytorch_batched(
        X_tr.unsqueeze(0), y_tr.unsqueeze(0), X_te.unsqueeze(0), kernel_name,
        n_steps=N_STEPS, lr=LR, oracle_mode="posterior", n_restarts=1,
    )

    assert bool(batched["ok"].all())
    torch.testing.assert_close(batched["R"][0], single["R"], atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(batched["mean"][0], single["mean"], atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(batched["Sigma"][0], single["Sigma"], atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("kernel_name", KERNELS)
def test_batched_output_is_a_valid_correlation_matrix_per_episode(small_cfg, kernel_name):
    eps = _episodes(small_cfg, 4)
    X_tr = torch.stack([e["x_norm_train"] for e in eps])
    y_tr = torch.stack([e["y_train"] for e in eps])
    X_te = torch.stack([e["x_norm_test"] for e in eps])

    out = fit_and_eval_gpytorch_batched(
        X_tr, y_tr, X_te, kernel_name, n_steps=N_STEPS, lr=LR,
        oracle_mode="posterior", n_restarts=1,
    )
    R, N = out["R"], X_te.shape[1]
    assert R.shape == (len(eps), N, N)
    assert bool(out["ok"].all())
    torch.testing.assert_close(R.diagonal(dim1=-2, dim2=-1), torch.ones(len(eps), N), atol=1e-5, rtol=0)
    torch.testing.assert_close(R, R.mT, atol=1e-6, rtol=0)
    # Positive definite after the jitter fit_and_eval_gpytorch_batched adds.
    torch.linalg.cholesky(R.double())
    assert torch.isfinite(out["loss"]).all()


def test_shape_contract_is_enforced(small_cfg):
    ep = _episodes(small_cfg, 1)[0]
    with pytest.raises(ValueError, match=r"expects \(E,P,d\)"):
        fit_and_eval_gpytorch_batched(
            ep["x_norm_train"], ep["y_train"], ep["x_norm_test"], "matern32",
            n_steps=2, lr=LR,
        )


def test_unsupported_kernel_is_refused_not_silently_substituted(small_cfg):
    eps = _episodes(small_cfg, 2)
    X_tr = torch.stack([e["x_norm_train"] for e in eps])
    y_tr = torch.stack([e["y_train"] for e in eps])
    X_te = torch.stack([e["x_norm_test"] for e in eps])
    with pytest.raises(RuntimeError):
        # every restart raises NotImplementedError inside, so the function
        # reports "all restarts failed" rather than returning a wrong kernel
        fit_and_eval_gpytorch_batched(X_tr, y_tr, X_te, "polynomial", n_steps=2, lr=LR)


def test_fitting_on_z_train_degenerates_which_is_why_the_baseline_uses_y_train(small_cfg):
    """Documents the modelling decision, so it is encoded in the suite rather
    than only in a docstring.

    z_train is a leave-one-out PIT residual: decorrelated by construction, with
    Cov(z_LOO) the partial-correlation matrix. Type-II ML on approximately
    white data pushes all variance into the nugget, so the fitted correlation
    collapses to the identity and the baseline silently becomes an expensive
    way to recompute the independence copula NLL.
    """
    # A realistic-ish shape on purpose: the collapse sharpens with context
    # size and feature width, which is the regime production actually runs in.
    # Measured mean |off-diagonal| here, matern32 / 200 steps:
    #   P=24 N=24 d=2 -> y 0.0555  z 0.0142  oracle 0.0722
    #   P=24 N=24 d=5 -> y 0.0733  z 0.0036  oracle 0.0964
    #   P=32 N=64 d=8 -> y 0.0340  z 0.0005  oracle 0.0587
    eps = _episodes(small_cfg, 4, P=32, N=64, d=8)
    X_tr = torch.stack([e["x_norm_train"] for e in eps])
    y_tr = torch.stack([e["y_train"] for e in eps])
    z_tr = torch.stack([gp_analytical_pit(e)["z_train"].float() for e in eps])
    X_te = torch.stack([e["x_norm_test"] for e in eps])

    torch.manual_seed(0)
    on_y = fit_and_eval_gpytorch_batched(
        X_tr, y_tr, X_te, "matern32", n_steps=200, lr=LR, oracle_mode="posterior",
    )
    torch.manual_seed(0)
    on_z = fit_and_eval_gpytorch_batched(
        X_tr, z_tr, X_te, "matern32", n_steps=200, lr=LR, oracle_mode="posterior",
    )

    offd_y = _mean_abs_offdiag(on_y["R"])
    offd_z = _mean_abs_offdiag(on_z["R"])
    offd_oracle = sum(
        _mean_abs_offdiag(gp_analytical_posterior(e)["R_post"]) for e in eps
    ) / len(eps)

    msg = f"y={offd_y:.4f} z={offd_z:.4f} oracle={offd_oracle:.4f}"
    assert offd_z < 0.2 * offd_oracle, f"expected the z_train fit to collapse toward I: {msg}"
    assert offd_y > 5 * offd_z, (
        f"the y_train fit should retain real correlation structure: {msg}"
    )


def test_posterior_mode_conditions_on_the_context(small_cfg):
    """oracle_mode='posterior' must actually condition: the predictive variance
    at the query points has to be no larger than the prior-mode one, which is
    the whole reason this baseline is a fair comparator for a model that also
    sees the context."""
    eps = _episodes(small_cfg, 3, P=24, N=12)
    X_tr = torch.stack([e["x_norm_train"] for e in eps])
    y_tr = torch.stack([e["y_train"] for e in eps])
    X_te = torch.stack([e["x_norm_test"] for e in eps])

    torch.manual_seed(5)
    post = fit_and_eval_gpytorch_batched(
        X_tr, y_tr, X_te, "matern32", n_steps=N_STEPS, lr=LR, oracle_mode="posterior",
    )
    torch.manual_seed(5)
    prior = fit_and_eval_gpytorch_batched(
        X_tr, y_tr, X_te, "matern32", n_steps=N_STEPS, lr=LR, oracle_mode="prior",
    )
    v_post = post["Sigma"].diagonal(dim1=-2, dim2=-1)
    v_prior = prior["Sigma"].diagonal(dim1=-2, dim2=-1)
    assert bool((v_post <= v_prior + 1e-5).all()), (
        f"conditioning increased the variance: max excess "
        f"{float((v_post - v_prior).max()):.3e}"
    )


def test_noise_stays_inside_the_constraint_interval(small_cfg):
    """The predictive variance can never fall below the fitted noise floor,
    which exp(-8) bounds -- a sanity check that the batched likelihood really
    carries the same Interval constraint the per-episode one does."""
    eps = _episodes(small_cfg, 3)
    X_tr = torch.stack([e["x_norm_train"] for e in eps])
    y_tr = torch.stack([e["y_train"] for e in eps])
    X_te = torch.stack([e["x_norm_test"] for e in eps])
    out = fit_and_eval_gpytorch_batched(
        X_tr, y_tr, X_te, "matern32", n_steps=N_STEPS, lr=LR, oracle_mode="posterior",
    )
    v = out["Sigma"].diagonal(dim1=-2, dim2=-1)
    assert bool((v > math.exp(-8.0) * 0.5).all())
