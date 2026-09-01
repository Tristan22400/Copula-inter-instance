"""
experiment_danp_gp_comparison.py — Dimension Agnostic Neural Processes
(Lee et al., ICLR 2025) Table 1 baseline comparison.

Replicates DANP's "From-scratch" n-dimensional GP-regression protocol
verbatim (Appendix C.3) and scores our TabICL-marginal + copula model
against it, END-TO-END with the model's OWN marginal density and OWN
correlation structure (Sklar's theorem decomposition, via
eval/metrics/joint_nll.py -> src/loss.py::y_space_nll) — no oracle leakage.
This is deliberately different from eval/runners/eval_checkpoint.py's
z-space copula-only NLL, which scores correlation structure against the
oracle marginal; DANP/TNP/etc. report a genuine end-to-end predictive
log-likelihood log p(y_test | context), so the comparison point here has to
be end-to-end too, or the numbers aren't in the same units.

DANP protocol (Appendix C.3, verbatim):
    x ~ Unif(-2, 2)^n, independently per dimension i in [n]
    RBF:        k(x,x') = s^2 exp(-||x-x'||^2 / (2 l^2))
    Matern-5/2: standard form (gpytorch's MaternKernel(nu=2.5), same formula)
    s ~ Unif(0.1, 1.0), l ~ Unif(0.1, 0.6)   -- isotropic, single scalar l/s
                                                 per episode, no ARD
    noiseless draw (no observation-noise term is added to y)
    |context| ~ Unif(n^2*5, n^2*50 - n^2*5)
    |target|  ~ Unif(n^2*5, n^2*50 - |context|)
    3 seeds, report mean +/- 1 sigma of the per-point TARGET log-likelihood
    (nats). DANP also reports a "context" log-likelihood column (the model
    scoring its own context points) -- every model in their Table 1 lands
    within ~1.34-1.38 there regardless of quality (near a shared ceiling,
    since it's an easy near-memorization task), so it's not a discriminating
    number and we don't compute an equivalent here.

Five numbers are reported per condition, forming a 2x2 of
(marginal source) x (correlation source) plus the Bayes-optimal floor:

  - "ours"          : TabICL marginal (own predicted density, via
                       inference.copula_inference.get_marginal_quantiles) +
                       copula correlation R_test (own prediction, via
                       get_test_correlation) -- the full model, and the only
                       row that is a real deployment number.
  - "independence"   : the SAME TabICL marginal, but R = I (diagonal
                       Gaussian-copula predictive density) -- an internal
                       ablation showing what the copula's correlation
                       structure buys beyond a factorized/diagonal
                       predictive density, which is what CANP/ANP/BANP/TNP
                       and DANP itself all reduce to per-point (they differ
                       from "ours" in how the diagonal mean/std is produced,
                       not in whether the predictive density factorizes
                       across target points).
  - "ours (oracle m)": the model's OWN correlation R_test, but scored against
                       the EXACT analytic GP posterior marginal instead of
                       TabICL's estimate -- the perfect-marginal-access
                       setting this branch's training.val_analytic_only mode
                       validates in. Not a deployment number (real data has
                       no analytic marginal); its value is that
                       "ours (oracle m)" - "ours" isolates how much of the
                       gap to DANP is the MARGINAL's fault, which no
                       end-to-end number alone can attribute.
  - "oracle diag"    : exact analytic posterior marginal with R = I. The
                       diagonal Bayes-optimal predictive density -- the best
                       any method that factorizes across target points can
                       do, DANP and TNP included.
  - "bayes-opt"      : exact analytic posterior marginal AND the true
                       posterior correlation R_post. A hard floor: no
                       predictive density can beat it in expectation
                       (Bayes-optimality of the posterior predictive under
                       log loss), so "bayes-opt" - "ours" is the total
                       headroom, and every published number should sit below
                       it. If one doesn't, suspect this script, not the paper.

Reading the decomposition: bayes-opt - oracle diag is what correlation
structure is worth on this task at all; ours (oracle m) - ours is the
marginal's contribution to our gap; bayes-opt - ours (oracle m) is what
remains for the copula head to close.

Important caveats (state these alongside any number this script prints):
  1. DANP/TNP etc. are purpose-trained on exactly this task distribution.
     Our TabICL backbone and copula model were NOT trained on Unif(-2,2)^n
     GP draws with these exact s/l ranges, or (crucially) on DANP's very
     small context sizes (as low as 5-20 points) -- this is an
     out-of-distribution generalization test for our model, not a
     like-for-like training setup.
  2. The "ours" number folds together TWO error sources (marginal
     miscalibration AND copula correlation error) exactly like DANP's own
     number does -- so it's a fair end-to-end comparison, but a lower
     "ours" number doesn't tell you which part is responsible; use the
     copula/marginal breakdown in the CSV for that.
  3. KNOWN, ARCHITECTURAL cause of a large chunk of the gap: both
     `kernel-sweep-classic-zcorrupt-noise-mild-bigN*` checkpoints are trained
     with cfg.data.oracle_mode=prior -- the loss target is the raw/
     unconditional kernel correlation among test points, never the
     context-conditioned posterior correlation (K_ss - K_st K_ff^-1 K_ts,
     which shrinks with denser/closer context). The model DOES take
     (X_train, z_train) as input, but under prior-mode training its only
     incentive to use that context is to infer WHICH kernel/hyperparameters
     govern the episode -- never to shrink correlation because context
     happens to be dense or nearby. DANP/TNP are trained end-to-end against
     the true posterior predictive log-likelihood, so they get
     context-shrinkage "for free"; scoring a prior-trained model on that
     same end-to-end objective (which is what this script does, by
     necessity -- see the module intro) makes it systematically
     overconfident wherever context is dense, especially at DANP's larger
     context counts (2D, up to ~180 points). This is NOT fixable post-hoc:
     model.py::low_rank_correlation's W is (B, N_test, r) -- the model
     never exposes train-test cross-correlation terms, so no
     Schur-complement-style correction can be bolted onto its output at
     inference time. Fixing this for real needs re-enabling
     oracle_mode=posterior dataset generation (disabled since 2026-07-29 for
     a real PSD bug -- see feedback_workflow memory) followed by a
     retrain/fine-tune, or an architecture change to expose train-test cross
     terms. Whether that's worth doing depends on whether posterior
     correlation is actually the right target for this repo's primary use
     case (see [[project_danp_gp_comparison]] memory's "when is prior more
     useful than posterior" discussion) -- prior-mode may be the
     deliberately correct choice for the ERA5 spatial-diagnostics use case
     even though it's a real handicap on DANP's specific benchmark protocol.
     See project memory `project_danp_gp_comparison.md` for the full
     writeup and the ranked list of causes (this one, N_max training-range,
     and a hypothesized lengthscale-domain mismatch).

Usage:
    python experiments/experiment_danp_gp_comparison.py \\
        [--copula-ckpt /path/to/step_XXXXXX.pt] \\
        [--tabicl-ckpt tabicl-regressor-v2-20260212.ckpt] \\
        [--conditions 1d_rbf,1d_matern52,2d_rbf,2d_matern52] \\
        [--n-episodes-per-seed 100] [--seeds 0,1,2] [--k-folds 10] \\
        [--out-csv ./results/danp_gp_comparison.csv] [--device auto]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gen import _safe_cholesky, build_kernel_fn  # noqa: E402
from loss import y_space_nll  # noqa: E402

from eval.metrics.joint_nll import compute_joint_nll  # noqa: E402
from inference.copula_inference import (  # noqa: E402
    get_marginal_quantiles,
    get_test_correlation,
    load_copula_model,
    load_tabicl_marginal,
    loo_pit,
    normalize_features,
)

DEFAULT_COPULA_CKPT = (
    "/srv/storage/thoth1@storage4.grenoble.grid5000.fr/trmartin/copula-inter/"
    "checkpoints/kernel-sweep-classic-zcorrupt-noise-mild-bigN/step_0285000.pt"
)
# Best-calibrated checkpoint per project memory (real-ERA5 + synthetic-kernel
# spatial-correlation sweeps across 4 checkpoint families) -- see
# eval/configs/checkpoints.py's "kernel-sweep-classic-zcorrupt-noise-mild-bigN"
# entry. Override with --copula-ckpt to test a different one.
DEFAULT_TABICL_CKPT = "tabicl-regressor-v2-20260212.ckpt"

CONDITIONS: dict[str, tuple[int, str]] = {
    "1d_rbf": (1, "rbf"),
    "1d_matern52": (1, "matern52"),
    "2d_rbf": (2, "rbf"),
    "2d_matern52": (2, "matern52"),
}

# DANP paper (Lee et al., ICLR 2025), Table 1, TARGET column, From-scratch
# scenario -- (mean, std) over their 3 seeds. TNP is their strongest
# non-DANP baseline (best of CANP/ANP/BANP/MPANP/TNP); DANP is their own
# model. Full 6-model table is in the paper.
PAPER_TARGET_LL = {
    "1d_rbf": {"TNP": (0.904, 0.003), "DANP": (0.921, 0.003)},
    "1d_matern52": {"TNP": (0.710, 0.001), "DANP": (0.723, 0.003)},
    "2d_rbf": {"TNP": (0.362, 0.001), "DANP": (0.373, 0.001)},
    "2d_matern52": {"TNP": (0.060, 0.002), "DANP": (0.068, 0.001)},
}

_JITTER = 1e-5  # matches experiments/_synthetic.py's proven-stable convention


def _danp_context_target_counts(n_dim: int, rng: np.random.Generator) -> tuple[int, int]:
    """|context| ~ Unif(n^2*5, n^2*50 - n^2*5), |target| ~ Unif(n^2*5, n^2*50 -
    |context|) -- verbatim from DANP Appendix C.3."""
    n2 = n_dim ** 2
    c_lo, c_hi = n2 * 5, n2 * 50 - n2 * 5
    n_context = int(rng.integers(c_lo, c_hi + 1))
    t_lo, t_hi = n2 * 5, n2 * 50 - n_context
    n_target = int(rng.integers(t_lo, max(t_lo, t_hi) + 1))
    return n_context, n_target


@torch.no_grad()
def sample_danp_episode(
    n_dim: int, kernel_name: str, rng_np: np.random.Generator, rng_torch: torch.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, int, int]:
    """One noiseless GP draw under DANP's exact From-scratch protocol.

    @torch.no_grad() matters here, not just for speed: build_kernel_fn's
    gpytorch kernels register lengthscale/outputscale as nn.Parameter
    (requires_grad=True by default), so without this K/L/y would carry a
    live autograd graph and the later `.numpy()` calls would raise "Can't
    call numpy() on Tensor that requires grad" — same reason
    experiments/_synthetic.py's caller (experiment_b's run_one_function) is
    itself @torch.no_grad()-decorated.
    """
    n_context, n_target = _danp_context_target_counts(n_dim, rng_np)
    n_total = n_context + n_target

    s = float(rng_np.uniform(0.1, 1.0))
    ell = float(rng_np.uniform(0.1, 0.6))

    X = torch.as_tensor(rng_np.uniform(-2.0, 2.0, size=(n_total, n_dim)), dtype=torch.float32)
    kernel_fn = build_kernel_fn(kernel_name, l=ell, alpha2=s ** 2)
    K = kernel_fn(X, X) + _JITTER * torch.eye(n_total)
    L = _safe_cholesky(K)
    z = torch.randn(n_total, generator=rng_torch)
    y = (L @ z.unsqueeze(-1)).squeeze(-1)

    X_train, X_test = X[:n_context].numpy(), X[n_context:].numpy()
    y_train, y_test = y[:n_context].numpy(), y[n_context:].numpy()
    return X_train, y_train, X_test, y_test, s, ell, n_context, n_target


@torch.no_grad()
def analytic_gp_posterior(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    kernel_name: str, s: float, ell: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact GP posterior at the target points for one DANP episode:
    (mu_post, sigma_post, R_post).

    The same Schur complement pit.gp_analytical_posterior computes, done
    inline here rather than by calling it: that function reads its kernel off
    a data_gen.py task dict's saved metadata schema (kernel/l/alpha2/nugget/
    period/rq_alpha/power/composite second-component sentinels), and these
    episodes are built by this script from DANP's own protocol, not by
    generate_gp_batch. Assembling a look-alike task dict would couple this
    script to every future change in that schema for six lines of algebra we
    already have the kernel for.

    Everything runs in float64. DANP's protocol is noiseless, so the only
    diagonal term is _JITTER -- with lengthscales up to 0.6 on a Unif(-2,2)
    domain and up to ~180 target points, K_ff is genuinely near-singular, and
    float32 here produces visibly wrong posteriors rather than merely
    imprecise ones.

    Returns numpy arrays: mu_post (N,), sigma_post (N,) = sqrt(diag), and the
    full posterior correlation R_post (N, N).
    """
    Xtr = torch.as_tensor(X_train, dtype=torch.float32)
    Xte = torch.as_tensor(X_test, dtype=torch.float32)
    ytr = torch.as_tensor(y_train, dtype=torch.float64)
    P, N = Xtr.shape[0], Xte.shape[0]

    kernel_fn = build_kernel_fn(kernel_name, l=ell, alpha2=s ** 2)
    K_ff = kernel_fn(Xtr, Xtr).double() + _JITTER * torch.eye(P, dtype=torch.float64)
    K_sf = kernel_fn(Xte, Xtr).double()
    K_ss = kernel_fn(Xte, Xte).double() + _JITTER * torch.eye(N, dtype=torch.float64)

    L_ff = _safe_cholesky(K_ff)
    # Zero prior mean: sample_danp_episode draws y = L @ z with no mean term.
    alpha = torch.cholesky_solve(ytr.unsqueeze(-1), L_ff).squeeze(-1)
    mu_post = (K_sf @ alpha)

    V = torch.linalg.solve_triangular(L_ff, K_sf.T, upper=False)
    Sigma_post = K_ss - V.T @ V
    Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)
    # Every eigenvalue of the true Sigma_post is >= _JITTER (it is
    # Cov(f_test|train) + jitter*I with the first term PSD), so this floor is
    # a hard bound, not a heuristic -- same argument as
    # gp_analytical_posterior's own repair.
    evals, evecs = torch.linalg.eigh(Sigma_post)
    if float(evals.min()) < _JITTER:
        Sigma_post = evecs @ torch.diag(evals.clamp(min=_JITTER)) @ evecs.T
        Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)

    sigma_post = Sigma_post.diagonal().clamp(min=_JITTER).sqrt()
    inv = (1.0 / sigma_post).unsqueeze(0)
    R_post = Sigma_post * inv * inv.T
    R_post = 0.5 * (R_post + R_post.T)
    R_post.fill_diagonal_(1.0)
    return mu_post.numpy(), sigma_post.numpy(), R_post.numpy()


def _gaussian_marginal_ll(
    y_test: np.ndarray, mu: np.ndarray, sigma: np.ndarray, R: np.ndarray,
) -> dict[str, float]:
    """Per-point log-likelihood of a Gaussian-marginal + Gaussian-copula
    predictive density, Sklar-decomposed.

    Scored through loss.y_space_nll -- the exact same function the training
    loss and every validation metric use -- so these rows sit in identical
    units to the TabICL-marginal rows above and to oracle_diag/total_nll in
    training logs. Deliberately NOT routed through
    eval/metrics/joint_nll.compute_joint_nll like the TabICL rows are: that
    path reconstructs a marginal from a discretized (quantile_grid, probs)
    representation and finite-differences it, which is the right thing for a
    quantile-head marginal but would add pointless discretization error to a
    marginal we know in closed form.

    Returns per-point log-likelihoods (higher = better), sign-flipped from
    y_space_nll's NLL convention to match DANP's Table 1.
    """
    z = torch.as_tensor((y_test - mu) / sigma, dtype=torch.float32).unsqueeze(0)
    log_pdf = torch.as_tensor(
        -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * ((y_test - mu) / sigma) ** 2,
        dtype=torch.float32,
    ).unsqueeze(0)
    Sigma = torch.as_tensor(R, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones_like(z, dtype=torch.bool)
    parts = y_space_nll(Sigma, z, log_pdf, mask)
    return {k: -float(v) for k, v in parts.items()}


def run_one_episode(
    seed: int, n_dim: int, kernel_name: str, tabicl_model, copula_model, k_folds: int,
) -> dict:
    rng_np = np.random.default_rng(seed)
    rng_torch = torch.Generator().manual_seed(seed)

    X_train, y_train, X_test, y_test, s, ell, n_context, n_target = sample_danp_episode(
        n_dim, kernel_name, rng_np, rng_torch
    )
    X_train_norm, X_test_norm = normalize_features(X_train, X_test)

    quantile_grid, probs = get_marginal_quantiles(tabicl_model, X_train_norm, y_train, X_test_norm)
    Z_train = loo_pit(tabicl_model, X_train_norm, y_train, k_folds=min(k_folds, len(X_train)))
    R_test = get_test_correlation(copula_model, X_train_norm, Z_train, X_test_norm)
    R_I = np.eye(n_target, dtype=np.float64)

    ours = compute_joint_nll(quantile_grid, probs, R_test, y_test)
    indep = compute_joint_nll(quantile_grid, probs, R_I, y_test)

    # Perfect-marginal-access rows. The model's correlation is conditioned on
    # the SAME context either way (R_test above is reused verbatim) -- only
    # the marginal changes -- so the difference between ours_oracle_m and ours
    # is attributable to the marginal alone.
    mu_post, sigma_post, R_post = analytic_gp_posterior(
        X_train, y_train, X_test, kernel_name, s, ell
    )
    ours_oracle_m = _gaussian_marginal_ll(y_test, mu_post, sigma_post, R_test.astype(np.float64))
    oracle_diag = _gaussian_marginal_ll(y_test, mu_post, sigma_post, R_I)
    bayes_opt = _gaussian_marginal_ll(y_test, mu_post, sigma_post, R_post)

    return {
        "seed": seed, "n_context": n_context, "n_target": n_target, "s": s, "ell": ell,
        "ll_ours_total": -ours["total"],
        "ll_ours_copula": -ours["copula"],
        "ll_ours_marginal": -ours["marginal"],
        "ll_independence_total": -indep["total"],
        "ll_ours_oracle_m_total": ours_oracle_m["total"],
        "ll_ours_oracle_m_copula": ours_oracle_m["copula"],
        "ll_oracle_diag_total": oracle_diag["total"],
        "ll_bayes_opt_total": bayes_opt["total"],
        "ll_bayes_opt_copula": bayes_opt["copula"],
        "ll_oracle_marginal": bayes_opt["marginal"],  # shared by all three oracle-m rows
    }


def run_condition(
    condition: str, seeds: list[int], n_episodes_per_seed: int,
    tabicl_model, copula_model, k_folds: int,
) -> tuple[list[dict], dict]:
    n_dim, kernel_name = CONDITIONS[condition]
    episodes: list[dict] = []
    n_failed = 0
    for seed in seeds:
        for i in range(n_episodes_per_seed):
            ep_seed = seed * 1_000_003 + i  # decorrelate seeds x episode index
            try:
                result = run_one_episode(ep_seed, n_dim, kernel_name, tabicl_model, copula_model, k_folds)
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                print(f"  [{condition} seed={seed} ep={i}] failed, skipping: {exc}")
                continue
            result["condition"] = condition
            result["outer_seed"] = seed
            episodes.append(result)

    # DANP's own convention: aggregate to a per-seed mean first, then report
    # mean +/- 1 sigma OVER THE SEED-LEVEL MEANS (3 seeds) -- not over raw
    # per-episode values, which would understate DANP's reported variance
    # (theirs already reflects seed-to-seed retraining variance, not just
    # sampling noise within one seed).
    seed_means = {
        k: [] for k in (
            "ll_ours_total", "ll_independence_total", "ll_ours_copula", "ll_ours_marginal",
            "ll_ours_oracle_m_total", "ll_oracle_diag_total", "ll_bayes_opt_total",
        )
    }
    for seed in seeds:
        seed_eps = [e for e in episodes if e["outer_seed"] == seed]
        if not seed_eps:
            continue
        for key in seed_means:
            seed_means[key].append(float(np.mean([e[key] for e in seed_eps])))

    summary = {
        "condition": condition, "n_dim": n_dim, "kernel": kernel_name,
        "n_episodes_ok": len(episodes), "n_failed": n_failed,
    }
    for key, vals in seed_means.items():
        summary[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
        summary[f"{key}_std"] = float(np.std(vals)) if vals else float("nan")
    return episodes, summary


def _print_report(summaries: list[dict]) -> None:
    def _pm(s: dict, key: str) -> str:
        return f"{s[f'{key}_mean']:.3f}±{s[f'{key}_std']:.3f}"

    print(f"\n{'=' * 118}")
    print("DANP (Lee et al., ICLR 2025) Table 1 comparison -- TARGET log-likelihood (nats/point, higher=better)")
    print(f"{'=' * 118}")
    header = (
        f"{'Condition':<15}{'ours':>14}{'independence':>14}{'ours(oracle m)':>16}"
        f"{'oracle diag':>14}{'bayes-opt':>14}{'TNP':>14}{'DANP':>14}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        cond = s["condition"]
        tnp_m, tnp_s = PAPER_TARGET_LL[cond]["TNP"]
        danp_m, danp_s = PAPER_TARGET_LL[cond]["DANP"]
        print(
            f"{cond:<15}{_pm(s, 'll_ours_total'):>14}{_pm(s, 'll_independence_total'):>14}"
            f"{_pm(s, 'll_ours_oracle_m_total'):>16}{_pm(s, 'll_oracle_diag_total'):>14}"
            f"{_pm(s, 'll_bayes_opt_total'):>14}"
            f"{f'{tnp_m:.3f}±{tnp_s:.3f}':>14}{f'{danp_m:.3f}±{danp_s:.3f}':>14}"
        )
    print("-" * len(header))
    # The decomposition the extra columns exist to expose, per condition.
    print("\nGap decomposition (nats/point):")
    print(f"{'Condition':<15}{'marginal cost':>16}{'copula headroom':>18}{'corr. is worth':>16}{'DANP vs bayes-opt':>20}")
    for s in summaries:
        marg_cost = s["ll_ours_oracle_m_total_mean"] - s["ll_ours_total_mean"]
        cop_head = s["ll_bayes_opt_total_mean"] - s["ll_ours_oracle_m_total_mean"]
        corr_worth = s["ll_bayes_opt_total_mean"] - s["ll_oracle_diag_total_mean"]
        danp_gap = s["ll_bayes_opt_total_mean"] - PAPER_TARGET_LL[s["condition"]]["DANP"][0]
        print(f"{s['condition']:<15}{marg_cost:>16.3f}{cop_head:>18.3f}{corr_worth:>16.3f}{danp_gap:>20.3f}")
    print(
        "  marginal cost      = ours(oracle m) - ours: how much of our gap is the TabICL marginal's fault\n"
        "  copula headroom    = bayes-opt - ours(oracle m): what's left for the correlation head to close\n"
        "  corr. is worth     = bayes-opt - oracle diag: the ceiling on what ANY correlation model can add\n"
        "                       here (DANP/TNP factorize across targets, so this is headroom they cannot reach)\n"
        "  DANP vs bayes-opt  = how far the published number sits below the analytic floor; must be >= 0,\n"
        "                       a negative value means this script's protocol or posterior is wrong, not DANP"
    )
    print("-" * len(header))
    n_ok = sum(s["n_episodes_ok"] for s in summaries)
    n_failed = sum(s["n_failed"] for s in summaries)
    print(f"Total episodes: {n_ok} ok, {n_failed} failed/skipped.")
    print(
        "Caveats: (1) our model was not trained on this exact task distribution "
        "(Unif(-2,2)^n inputs, these s/l ranges, or DANP's small context sizes) "
        "-- this is an OOD generalization test, not like-for-like training. "
        "(2) 'ours'/'independence' fold marginal + copula error together, same "
        "as DANP's own number -- that is what makes them comparable to it, and "
        "why the oracle-marginal columns (which are NOT comparable to DANP, "
        "having been handed the true marginal) exist to attribute the gap. "
        "(3) DANP's 'context' log-likelihood column is not computed here -- see "
        "module docstring."
    )
    print(f"{'=' * 118}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="DANP (ICLR 2025) Table 1 GP-regression comparison")
    parser.add_argument("--copula-ckpt", default=DEFAULT_COPULA_CKPT)
    parser.add_argument("--tabicl-ckpt", default=DEFAULT_TABICL_CKPT)
    parser.add_argument("--conditions", default=",".join(CONDITIONS.keys()))
    parser.add_argument("--n-episodes-per-seed", type=int, default=100)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--k-folds", type=int, default=10)
    parser.add_argument("--out-csv", default="./results/danp_gp_comparison.csv")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    assert all(c in CONDITIONS for c in conditions), f"Unknown condition(s) in {conditions}, must be subset of {list(CONDITIONS)}"
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        args.device if args.device != "auto" else "cpu"
    )
    print(f"Device: {device}")

    print(f"Loading TabICL marginal model: {args.tabicl_ckpt}")
    tabicl_model = load_tabicl_marginal(args.tabicl_ckpt, device)

    print(f"Loading copula model: {args.copula_ckpt}")
    copula_model, _copula_cfg = load_copula_model(args.copula_ckpt, device=device)

    all_episodes: list[dict] = []
    summaries: list[dict] = []
    for condition in conditions:
        print(f"\n--- {condition} ({args.n_episodes_per_seed} episodes x {len(seeds)} seeds) ---")
        episodes, summary = run_condition(
            condition, seeds, args.n_episodes_per_seed, tabicl_model, copula_model, args.k_folds
        )
        all_episodes.extend(episodes)
        summaries.append(summary)
        print(
            f"  ours: {summary['ll_ours_total_mean']:.3f} +/- {summary['ll_ours_total_std']:.3f}   "
            f"(paper DANP: {PAPER_TARGET_LL[condition]['DANP'][0]:.3f})"
        )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    if all_episodes:
        fieldnames = list(all_episodes[0].keys())
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_episodes)
        print(f"\nSaved per-episode results: {args.out_csv}")

    _print_report(summaries)


if __name__ == "__main__":
    main()
