"""analytic_copula_report.py — publication figures and tables for the
ANALYTIC-ONLY ("perfect marginal access") regime.

Scores a checkpoint trained under `+experiment=analytic_only` (or any
checkpoint, evaluated as if it had perfect marginal access) on synthetic GP
episodes whose exact analytic marginal is known, and emits the figure/table set
behind the claim: *given perfect marginals, an amortized copula head recovers
near-optimal dependence structure in one forward pass*.

Why this regime needs its own runner, rather than a flag on
eval_checkpoint.py: two things are true here that are false everywhere else in
this repo, and both change what may be plotted.

  1. THE MARGINAL IS A CONSTANT. `log_pdf_test` is the exact GP POSTERIOR
     predictive Gaussian density built in the generator (data_gen's
     _generate_gp_batch_raw / pit.gp_analytical_pit standardize y_test by
     (mu_post, sqrt(diag(Sigma_post))) -- mirroring the context-conditioned
     marginal TabICL supplies at deployment), and loss.y_space_nll's marginal
     term is just -log_pdf_test.sum(-1)/n -- never a model output. So
     `total = copula + per-episode constant`, and the COPULA NLL is the
     sufficient statistic for the model. Every headline number here is
     therefore a copula NLL, not a total.

  2. Z-SPACE IS UNDISTORTED, so predicted-vs-true correlation comparisons are
     valid again. This repo otherwise forbids them (see the
     feedback_no_raw_correlation_vs_oracle_comparison note): once TabICL's
     K-fold PIT is in the loop, Sigma lives in TabICL's own approximate
     z-space and comparing it against the GP's exact R conflates copula error
     with marginal-transform distortion, leaving the joint Y-space NLL as the
     only sound diagnostic. With exact analytic marginals there is no warp.
     That makes attribution -- *why* is the model wrong, not just *how much* --
     available only here, which is the whole scientific point of the mode.

WHAT THE MODEL IS ACTUALLY TRAINED TO DO (this drives the choice of ceiling).
The training loss is `nll_weight * y_space_nll(...)["total"]`
(src/train.py), and `aux_mae_weight` -- the only term that ever reads R_star --
defaults to 0.0 (conf/config.yaml). R_star therefore never enters the
gradient, so the Bayes-optimal solution of the training objective, given the
context, is the POSTERIOR structure. `oracle_mode="prior"` now names ONLY how
R_star/sigma_star -- that disabled auxiliary target -- are built; since the
z_test standardization was moved to the posterior marginals it no longer
touches the scored density at all, and it never made the prior correlation the
training target. The posterior is the target and the primary ceiling, and
z_test | context ~ N(0, R_post) exactly, so R_post is also attainable by the
model's unit-diagonal parametrization.

THE ONE CONVENTION TRAP. eval/metrics/joint_nll.py documents two incompatible
senses of "copula NLL". gp_analytical_posterior's `nll_post_copula` is in
POSTERIOR-standardized coordinates (own-marginal convention) and is NOT the
same quantity as the model's copula NLL, despite the name. So every reference
in the copula table below is recomputed in the SHARED-marginal convention --
corr_nll_single(R, z_test) against the one posterior-standardized z_test -- and
`nll_post_copula` appears only in the total-NLL table, where a different
marginal is legitimate because totals are comparable across marginals.

Four references bracket the model, all shared-marginal:

  - independence   : R = I. Exactly 0, by construction -- loss.py's copula term
                     0.5*(log|R| + z'R^-1 z - z'z)/N vanishes at R = I. Asserted
                     at runtime, since a non-zero value means a Sigma/masking
                     convention bug.
  - prior          : R = R_star. The CONTEXT-BLIND control. The model beating
                     this is precisely the evidence that the head uses the
                     context, rather than merely identifying which kernel
                     generated the episode.
  - posterior      : R = R_post (gp_analytical_posterior's Schur complement).
                     The headline ceiling.
  - achievable     : the correlation of E[z z' | context] (see
                     _achievable_optimum_R) -- the best a unit-diagonal Sigma
                     can do under whatever standardization the episode
                     actually used. It exists because the OLD prior
                     standardization pinned the model's predictive mean to
                     mu_prior, leaving a mean shift the head had no parameter
                     to absorb, so the posterior gap alone charged the head
                     for a marginal limitation. Posterior standardization
                     removes that constraint, so this now coincides with
                     `posterior` to numerical precision. It is kept, and
                     computed from the EMITTED marginal rather than assumed,
                     as a standing check: if the two rows ever separate again,
                     the standardization has regressed (see
                     tests/test_analytic_pit_posterior.py).

Usage:
    python eval/runners/analytic_copula_report.py --ckpt <path-or-dir> [...]

--ckpt accepts a checkpoint file OR a directory, in which case the highest
step_*.pt is used -- so this can be pointed at a run that is still training and
re-run as steps land. The GP-MLE/DKL baselines are cached on disk
(eval/baselines/classical.py's baseline_fingerprint machinery), so only the ICL
forward pass is recomputed between such re-runs.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import re
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import chi2, wasserstein_distance

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference.copula_inference import load_copula_model  # noqa: E402
from loss import y_space_nll  # noqa: E402
from model import low_rank_correlation  # noqa: E402
from pit import gp_analytical_posterior, mvn_nll  # noqa: E402

from eval.baselines.classical import (  # noqa: E402
    baseline_fingerprint,
    corr_nll_single,
    episode_cache_key,
    eval_baselines_episode,
    fit_and_eval_gpytorch,
    load_baseline_cache,
    save_baseline_cache,
)
from eval.metrics.energy_score import compute_energy_score  # noqa: E402
from eval.runners.eval_checkpoint import (  # noqa: E402
    _live_generate_alternating,
    _load_full_config,
    _select_best_baseline_cv,
)
from eval.spatial.diagnostics import (  # noqa: E402
    _exact_gp_loo_z_train,
    build_synthetic_grid_task,
    field_roughness,
    gearys_c,
    morans_i,
    semivariogram,
)
from eval.viz.correlation_plots import plot_synthetic_residual_grid  # noqa: E402

# Reference series, in the order they appear in every legend/table. Colors are
# fixed here rather than left to matplotlib's cycle so that the model keeps one
# identity across all six figures.
REFERENCE_STYLE = {
    "independence": ("Independence ($R=I$)", "#999999", ":"),
    "prior":        ("Prior $R_\\star$ (context-blind)", "#8172b2", "-."),
    "best_kernel":  ("Best fitted kernel", "#dd8452", "--"),
    "model":        ("Copula model (ours)", "#c44e52", "-"),
    "achievable":   ("Achievable optimum", "#4c72b0", (0, (3, 1, 1, 1))),
    "posterior":    ("Posterior $R_{post}$ (Bayes)", "#55a868", "-"),
}
SERIES_ORDER = ["independence", "prior", "best_kernel", "model", "achievable", "posterior"]

# Number of Monte-Carlo trajectories behind each episode's energy score. The
# energy score's sampling error falls as 1/sqrt(K) and it is only ever read as
# a mean over episodes, so a modest K per episode beats a large K on few.
_ENERGY_SAMPLES = 128


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_ckpt(path: str) -> str:
    """A checkpoint file, or the highest-step `step_*.pt` inside a directory.

    The directory form exists so this report can be aimed at a run that is
    still training and re-run as new steps land, without the caller having to
    look up the latest filename each time. Sorted by the parsed integer step,
    not lexicographically, so step_0100000 beats step_0090000 regardless of
    zero-padding width."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"--ckpt {path!r} is neither a file nor a directory")
    cands = glob.glob(os.path.join(path, "step_*.pt"))
    if not cands:
        raise FileNotFoundError(f"no step_*.pt checkpoints in {path!r}")

    def _step(p: str) -> int:
        m = re.search(r"step_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    best = max(cands, key=_step)
    print(f"[ckpt] {path} -> {os.path.basename(best)} (step {_step(best)})")
    return best


def ckpt_step(path: str) -> int:
    m = re.search(r"step_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------------------
# Model forward
# ---------------------------------------------------------------------------
def model_correlation(model, device, x_train, z_train, x_test) -> torch.Tensor:
    """(P,d),(P,),(N,d) -> the model's predicted (N,N) correlation matrix.

    Deliberately NOT eval/spatial/diagnostics.py::_forward_correlation, which
    calls low_rank_correlation(out["W"], out["s"]) with the default
    parametrization and no `lam`. That silently mis-decodes any checkpoint
    trained with a non-default correlation_parametrization. This mirrors
    eval_checkpoint.py::_eval_icl_episode's call exactly instead -- same
    parametrization lookup, same optional lam -- so a number produced here and
    one produced there are the same quantity for every checkpoint, not just
    the default-parametrization ones."""
    xtr = torch.as_tensor(x_train, dtype=torch.float32, device=device).unsqueeze(0)
    xte = torch.as_tensor(x_test, dtype=torch.float32, device=device).unsqueeze(0)
    ztr = torch.as_tensor(z_train, dtype=torch.float32, device=device).unsqueeze(0)
    batch = {
        "x_train": xtr,
        "x_test": xte,
        "z_train": ztr,
        "train_mask": torch.ones(1, xtr.shape[1], dtype=torch.bool, device=device),
    }
    with torch.no_grad():
        out = model(batch)
        Sigma = low_rank_correlation(
            out["W"], out.get("s"),
            parametrization=getattr(model, "correlation_parametrization", "covnorm"),
            lam=out.get("lam"),
        )
    return Sigma[0]


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
def _emitted_marginal(ep: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The (mu, sigma) the episode's own z_test/log_pdf_test were built from,
    recovered rather than assumed.

    Everything in this file that reconstructs a Y-space quantity from a
    correlation matrix needs the SAME marginal the copula NLL was scored
    against, or the Sklar split silently stops adding up. Reading mu_star /
    sigma_star is what that used to mean; it no longer is, because those stay
    PRIOR while z_test moved to the posterior predictive. Inverting the
    emitted log-density instead makes these references correct under either
    standardization, and makes a future regression show up as a moved number
    rather than a silently wrong one:

        log_pdf = -0.5*log(2pi) - log(sigma) - 0.5*z^2   =>   sigma, then
        mu = y_test - z * sigma

    float64 throughout: sigma comes back through an exp, and the downstream
    consumers divide by it."""
    z = ep["z_test"].double().cpu()
    log_pdf = ep["log_pdf_test"].double().cpu()
    sigma = torch.exp(-0.5 * float(np.log(2.0 * np.pi)) - 0.5 * z ** 2 - log_pdf)
    mu = ep["y_test"].double().cpu() - z * sigma
    return mu, sigma.clamp_min(1e-12)


def _achievable_optimum_R(ep: dict, post: dict) -> torch.Tensor:
    """Correlation of E[z z' | context], the best a unit-diagonal Sigma can do
    under whatever marginal the episode actually emitted.

    The model's scored density is N(mu_u, D Sigma D) with (mu_u, D) the fixed
    analytic marginal (_emitted_marginal); the model supplies only the
    correlation. The truth given the context is N(mu_post, Sigma_post), so the
    conditional second moment of the z the model is scored on is

        E[z z' | ctx] = D^-1 (Sigma_post + delta delta') D^-1,
        delta = mu_post - mu_u

    Normalizing that to a correlation matrix gives the achievable optimum. It
    depends only on (x_train, y_train), never on y_test, so it is a legitimate
    predictor and not oracle leakage.

    Under the CURRENT posterior standardization mu_u = mu_post and
    D^2 = diag(Sigma_post), so delta = 0, the diagonal is already 1, and this
    returns R_post -- the `achievable` and `posterior` rows coincide, which is
    the point: the head is no longer charged for a marginal limitation.

    It is deliberately not short-circuited to `post["R_post"]`. The general
    form is what makes the two rows separate again the moment the
    standardization regresses to the prior (where delta is a smooth non-zero
    field carrying most of the structure, and the diagonal is NOT 1), so this
    row doubles as a visible regression indicator.

    float64 throughout: delta delta' is an outer product of two O(1)
    quantities that can be much smaller than Sigma_post's own scale, and
    float32 cancellation there was already the documented failure mode in
    gp_analytical_posterior."""
    mu_u, sig = _emitted_marginal(ep)
    delta = post["mu_post"].double().cpu() - mu_u
    M = post["Sigma_post"].double().cpu() + torch.outer(delta, delta)
    M = M / sig.unsqueeze(1) / sig.unsqueeze(0)
    d = M.diagonal().clamp_min(1e-12).sqrt()
    R = M / d.unsqueeze(1) / d.unsqueeze(0)
    return (0.5 * (R + R.T)).float()


def _chi2_pit(R: torch.Tensor, z: torch.Tensor) -> float:
    """u = F_{chi2_N}(z' R^-1 z), the joint probability-integral transform.

    Under a correctly specified joint the quadratic form is exactly chi2 with N
    degrees of freedom, so u ~ Uniform(0,1). This is the one calibration test
    that looks at the WHOLE predictive covariance rather than its marginals:
    values piling up near 1 mean the model is overconfident (real residuals
    bigger than the predicted covariance allows), near 0 underconfident. It is
    exactly valid here because under the posterior standardization z_test is
    N(0, R_post) CONDITIONAL on the context -- a stronger statement than the
    prior standardization's unconditional N(0, R_prior), and one no PIT-based
    regime can claim."""
    n = z.shape[0]
    z64 = z.double()
    R64 = 0.5 * (R.double() + R.double().T)
    try:
        L = torch.linalg.cholesky(R64 + 1e-8 * torch.eye(n, dtype=torch.float64))
    except torch.linalg.LinAlgError:
        return float("nan")
    sol = torch.cholesky_solve(z64.unsqueeze(-1), L).squeeze(-1)
    q = float((z64 * sol).sum())
    return float(chi2.cdf(q, df=n))


def _energy_score(R: torch.Tensor, ep: dict, rng: np.random.Generator) -> float:
    """Energy score of the Y-space predictive implied by correlation R under
    the exact analytic marginal, against the realized y_test.

    A strictly proper scoring rule for multivariate distributions that is NOT
    the training loss, which is the point: an NLL-only result invites "you only
    win on the metric you optimized". Sampling is exact rather than via
    inference.copula_inference.sample_trajectories' quantile-grid inversion --
    the analytic marginal IS Gaussian here, so y = mu + sigma * (L @ eps) needs
    no quantile grid and introduces no interpolation error."""
    n = R.shape[0]
    R64 = 0.5 * (R.double() + R.double().T)
    try:
        L = torch.linalg.cholesky(R64 + 1e-8 * torch.eye(n, dtype=torch.float64)).numpy()
    except torch.linalg.LinAlgError:
        return float("nan")
    eps = rng.standard_normal((_ENERGY_SAMPLES, n))
    # The SAME marginal the copula NLL is scored against (posterior
    # predictive), recovered from the episode rather than read off mu_star /
    # sigma_star -- those are the prior, and sampling from them would score a
    # different predictive than every other number in this file.
    mu_t, sig_t = _emitted_marginal(ep)
    mu, sig = mu_t.numpy(), sig_t.numpy()
    samples = mu[None, :] + sig[None, :] * (eps @ L.T)
    return compute_energy_score(samples, ep["y_test"].double().cpu().numpy())


def _totals(R: torch.Tensor, ep: dict) -> dict:
    """Y-space total/copula/marginal NLL for correlation R under the exact
    analytic marginal, per point -- loss.y_space_nll on a batch of one.

    The marginal component is identical for every R (it is
    -log_pdf_test.mean(), a property of the episode alone). That is not
    redundancy: it is invariant 2 in the verification suite, and the sharpest
    available guard against accidentally mixing the own-marginal and
    shared-marginal conventions this module's docstring warns about."""
    n = R.shape[0]
    mask = torch.ones(1, n, dtype=torch.bool)
    parts = y_space_nll(
        R.float().cpu().unsqueeze(0),
        ep["z_test"].float().cpu().unsqueeze(0),
        ep["log_pdf_test"].float().cpu().unsqueeze(0),
        mask,
    )
    return {k: float(v) for k, v in parts.items()}


def _family_label(ep: dict) -> str:
    """Kernel-family bucket for the per-family breakdown.

    Composite/chain episodes collapse to one "composite" bucket rather than
    getting a bucket per realized chain: _live_generate_alternating forces only
    every OTHER episode to be non-composite, so the composite draws are spread
    thin across a combinatorial space of chains and per-chain buckets would
    each hold one or two episodes."""
    if ep.get("kernel_components") is not None and len(ep.get("kernel_components", [])) > 1:
        return "composite"
    k = ep.get("kernel")
    if isinstance(k, (list, tuple)):
        k = k[0] if len(k) == 1 else "composite"
    return str(k) if k is not None else "unknown"


def collect_episodes(model, device, cfg, args, prior_cfg, icl_rank) -> list[dict]:
    """One record per episode: every method's copula NLL (shared-marginal),
    every reference's total NLL, calibration and energy-score entries, and the
    raw off-diagonal/distance arrays the attribution figures pool.

    The GP-MLE/DKL/per-episode-transformer fits are read from (and written
    back to) the on-disk baseline cache keyed by generation+fitting settings,
    exactly as eval_checkpoint.py does. That is what makes re-running this
    report against successive checkpoints of a live training run cheap: the
    fitted baselines depend only on the episodes, so only the model forward
    pass is redone."""
    episodes = _live_generate_alternating(cfg, args.n_episodes, device, args.seed)

    fingerprint = baseline_fingerprint(
        cfg, True, None, args.seed, icl_rank, "prior",
        args.n_steps_mle, args.lr_mle, args.n_restarts_mle,
        args.n_steps_dkl, args.lr_dkl, args.n_steps_per_ep, args.patience_per_ep,
    )
    cache = load_baseline_cache(args.baseline_cache, fingerprint) if not args.no_baseline_cache else {}
    cache_dirty = False
    rng = np.random.default_rng(args.seed)
    records: list[dict] = []

    for i, ep in enumerate(episodes):
        n = int(ep["x_norm_test"].shape[0])
        if n < 2:
            continue
        z_test = ep["z_test"].float().cpu()

        # ---- exact analytic ceilings (posterior + prior, one call) ----
        try:
            post = gp_analytical_posterior(ep)
        except (KeyError, NotImplementedError) as exc:
            print(f"  [ep {i}] no analytic posterior ({type(exc).__name__}) — skipped")
            continue
        R_post = post["R_post"].float().cpu()
        R_prior = ep["R_star"].float().cpu()
        R_ach = _achievable_optimum_R(ep, post)
        R_indep = torch.eye(n)

        # ---- fitted baselines (cached) ----
        key = episode_cache_key(True, None, args.seed, i, i)
        cached = cache.get(key) if not args.refresh_baselines else None
        t_base0 = time.perf_counter()
        if cached is None:
            b_nlls, b_R, _ = eval_baselines_episode(
                ep=ep, icl_rank=icl_rank,
                n_steps_mle=args.n_steps_mle, lr_mle=args.lr_mle,
                n_steps_dkl=args.n_steps_dkl, lr_dkl=args.lr_dkl,
                n_steps_per_ep=args.n_steps_per_ep, patience_per_ep=args.patience_per_ep,
                device=torch.device(device), oracle_mode="prior", prior_cfg=prior_cfg,
                n_restarts_mle=args.n_restarts_mle,
            )
            cache[key] = {"nlls": b_nlls, "R_dict": {k: v.cpu() for k, v in b_R.items()}}
            cache_dirty = True
            baseline_sec = time.perf_counter() - t_base0
        else:
            b_nlls, b_R = cached["nlls"], cached["R_dict"]
            baseline_sec = float("nan")  # a cache hit times nothing meaningful

        # ---- model ----
        t0 = time.perf_counter()
        R_model = model_correlation(
            model, device,
            ep["x_norm_train"].cpu().numpy(), ep["z_train"].cpu().numpy(),
            ep["x_norm_test"].cpu().numpy(),
        ).float().cpu()
        model_sec = time.perf_counter() - t0

        # ---- best fitted baseline, chosen by nested CV (no winner's curse) ----
        fitted_R = {k: v for k, v in b_R.items() if k not in ("independence", "gp_prior_rbf")}
        best_nll, best_name, _ = _select_best_baseline_cv(
            {k: v.cpu() for k, v in fitted_R.items()}, z_test,
            args.n_folds, args.min_fold_size, args.seed + i,
        )
        R_best = fitted_R.get(best_name).cpu() if best_name is not None else None

        R_by = {
            "independence": R_indep, "prior": R_prior, "model": R_model,
            "achievable": R_ach, "posterior": R_post,
        }
        if R_best is not None:
            R_by["best_kernel"] = R_best

        rec: dict = {
            "idx": i, "family": _family_label(ep), "n_test": n,
            "n_train": int(ep["x_norm_train"].shape[0]),
            "model_sec": model_sec, "baseline_sec": baseline_sec,
            "best_kernel_name": best_name,
            "copula": {}, "total": {}, "marginal": {}, "chi2_pit": {}, "energy": {},
        }
        for name, R in R_by.items():
            rec["copula"][name] = corr_nll_single(R, z_test)
            parts = _totals(R, ep)
            rec["total"][name] = parts["total"]
            rec["marginal"][name] = parts["marginal"]
            rec["chi2_pit"][name] = _chi2_pit(R, z_test)
            rec["energy"][name] = _energy_score(R, ep, rng)
        # best_baseline's own CV-honest NLL differs from copula[best_kernel]
        # (which scores the winner on all N points): keep both, and use the CV
        # number wherever the two are compared to the model, since only it is
        # free of selection bias.
        rec["best_kernel_cv_nll"] = best_nll
        # Every individual fitted baseline, for the full table.
        rec["copula_all"] = dict(b_nlls)

        # Own-marginal oracle totals (gp_analytical_posterior's own Sklar
        # split, POSTERIOR-standardized -- see the module docstring's
        # convention warning). Only ever used in the total-NLL table.
        rec["oracle_prior_total"] = post["nll_prior"] / n
        rec["oracle_post_total"] = post["nll_post"] / n
        rec["oracle_prior_copula_own"] = post["nll_prior_copula"] / n
        rec["oracle_post_copula_own"] = post["nll_post_copula"] / n

        # Pooled arrays for the attribution figures.
        iu = np.triu_indices(n, k=1)
        rec["off_model"] = R_model.numpy()[iu]
        rec["off_post"] = R_post.numpy()[iu]
        rec["off_prior"] = R_prior.numpy()[iu]
        rec["off_best"] = R_best.numpy()[iu] if R_best is not None else None
        x_test = ep["x_norm_test"].cpu().numpy()
        rec["pair_dist"] = np.sqrt(((x_test[:, None, :] - x_test[None, :, :]) ** 2).sum(-1))[iu]
        records.append(rec)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [ep {i:4d}] N={n:4d} family={rec['family']:<22s} "
                  f"copula: model={rec['copula']['model']:+.4f} "
                  f"post={rec['copula']['posterior']:+.4f} "
                  f"prior={rec['copula']['prior']:+.4f} best={best_name}")

    if cache_dirty and not args.no_baseline_cache:
        save_baseline_cache(args.baseline_cache, fingerprint, cache)
    return records
