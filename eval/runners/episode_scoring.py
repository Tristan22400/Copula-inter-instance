"""Per-episode scoring for eval_checkpoint.

Baselines (prefit, cached or fitted), the marginal PIT, the autoregressive
chain, the ICL model and the TOTAL-table rows.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from eval.baselines.prefit import (
    _NON_FITTED_EXCLUDED,
    _count_physical_cores,
    _eval_zero_mean_gp_baselines,
    _prefit_baselines_parallel,
    _select_best_baseline_cv,
    _valid_cached_entry,
    episode_digest,
    episode_targets,
)
from eval.runners.eval_inputs import _PlannedEpisode
from eval.runners.eval_tables import (
    _GP_TOTAL_KEYS,
    _METHOD_LABELS,
    _kernel_composition_label,
)

if TYPE_CHECKING:
    from copula_inter.copula.model import CopulaTabICL
    from eval.runners.eval_args import EvalSpec

from copula_inter.copula.attainable import (
    any_copula_bound,
    attainable_correlation,
    expected_copula_nll,
    pit_moments,
)
from copula_inter.copula.loss import y_space_nll
from copula_inter.copula.model import low_rank_correlation
from copula_inter.data.gp_bayes import bayes_predictive
from copula_inter.marginal.pit import (
    TabICLLike,
    gp_analytical_posterior,
    normalize_targets,
    query_marginal,
    run_pit,
)
from eval.baselines.autoregressive import (
    ar_parts_from_log_pdf,
    autoregressive_log_pdf,
)
from eval.baselines.classical import (
    assert_shared_z_test,
    corr_nll_single,
    eval_baselines_episode,
    expected_baseline_keys,
    save_baseline_entry,
)
from eval.results import (
    NAN_PARTS as _NAN_PARTS,
)


@contextlib.contextmanager
def _snapshot_rng_and_threads(seed: int | None = None, cpu_threads: int | None = None) -> Iterator[None]:
    """Context manager: save and restore the torch RNG; optionally reseed and cap intra-op threads meanwhile."""
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    prev_threads = torch.get_num_threads() if cpu_threads is not None else None
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    if seed is not None:
        torch.manual_seed(seed)
    try:
        yield
    finally:
        if prev_threads is not None:
            torch.set_num_threads(prev_threads)
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)


def _eval_icl_episode(
    ep: dict,
    icl_model: nn.Module,
    device: torch.device,
    marginal_pit: dict[str, Tensor] | None = None,
) -> tuple[dict[str, float], dict[str, Tensor], Tensor | None, dict[str, dict[str, float]], dict[str, float]]:
    """Score the ICL model and the oracle on one episode.

    marginal_pit, when given, replaces the episode's z_train as the model input
    and supplies the marginal (z_test, log_pdf_test) for the model's Y-space NLL.

    The z-space oracle is R_post, the correlation of the known-theta GP posterior
    Sigma_post = K_QQ - K_QC K_CC^{-1} K_CQ: the shared z_test is the exact
    posterior PIT, so z_test ~ N(0, R_post) and R_post is the copula target. The
    generator's R_star (prior correlation among the queries, ignoring the context)
    is kept in R_dict["oracle_prior"] for the plot only.

    Returns:
        (nlls, R_dict, R_oracle, y_space_nlls, icl_y_parts): nlls holds the
        shared-z_test copula NLLs of icl and oracle; R_oracle is R_post (None on
        real data); y_space_nlls the analytic GP prior/posterior Y-space NLLs
        ({total, marginal, copula}, NaN when unavailable); icl_y_parts the
        model's own Y-space split (NaN without marginal_pit).
    """
    X_train = ep["x_norm_train"].to(device)  # (P, d_x)
    z_train = (
        marginal_pit["z_train"].to(device) if marginal_pit is not None else ep["z_train"].to(device)
    )  # (P,)  ICL's conditioning input — oracle LOO-PIT residual by default
    X_test = ep["x_norm_test"].to(device)  # (N, d_x)
    z_test = ep["z_test"].to(device)  # (N,)
    assert_shared_z_test(z_test, ep)
    # Real episodes have no generating kernel: no R_star, no oracle.
    R_prior = ep["R_star"].to(device) if "R_star" in ep else None  # (N, N) or None
    R_oracle: Tensor | None = None

    P, N = X_train.shape[0], X_test.shape[0]
    nlls: dict[str, float] = {}
    R_dict: dict[str, Tensor] = {}
    R_I = torch.eye(N, dtype=X_train.dtype, device=device)
    icl_y_parts = _NAN_PARTS.copy()

    try:
        train_mask = torch.ones(1, P, dtype=torch.bool, device=device)
        batch = {
            "x_train": X_train.unsqueeze(0),
            "x_test": X_test.unsqueeze(0),
            "z_train": z_train.unsqueeze(0),
            "train_mask": train_mask,
        }
        with torch.no_grad():
            out = icl_model(batch)
            Sigma_icl = low_rank_correlation(
                out["W"],
                out.get("s"),
                parametrization=getattr(icl_model, "correlation_parametrization", "covnorm"),
                lam=out.get("lam"),
            )  # (1, N, N)
        R_icl = Sigma_icl[0, :N, :N]
        nlls["icl"] = corr_nll_single(R_icl, z_test)
        R_dict["icl"] = R_icl
        if marginal_pit is not None:
            test_mask = torch.ones(1, N, dtype=torch.bool, device=device)
            icl_parts = y_space_nll(
                Sigma_icl[:, :N, :N],
                marginal_pit["z_test"].to(device).unsqueeze(0),
                marginal_pit["log_pdf_test"].to(device).unsqueeze(0),
                test_mask,
            )
            icl_y_parts = {k: v.item() for k, v in icl_parts.items()}
    except Exception as exc:
        print(f"  [icl] failed: {exc}")
        nlls["icl"] = float("nan")
        R_dict["icl"] = R_I.clone()

    y_space_nlls = {"prior": _NAN_PARTS.copy(), "posterior": _NAN_PARTS.copy()}
    try:
        if R_prior is None:
            raise NotImplementedError(
                "no generating kernel (real-data episode) -- no analytic GP prior/posterior reference exists"
            )
        post = gp_analytical_posterior(ep)
        R_oracle = post["R_post"].to(device=device, dtype=z_test.dtype)
        nlls["oracle"] = corr_nll_single(R_oracle, z_test)
        R_dict["oracle"] = R_oracle
        R_dict["oracle_prior"] = R_prior
        y_space_nlls = {
            "prior": {
                "total": post["nll_prior"],
                "marginal": post["nll_prior_marginal"],
                "copula": post["nll_prior_copula"],
            },
            "posterior": {
                "total": post["nll_post"],
                "marginal": post["nll_post_marginal"],
                "copula": post["nll_post_copula"],
            },
        }
        if post["repaired"]:
            print(f"    [oracle_posterior] PSD repair fired (min_eig={post['min_eig']:.2e})")
    except (KeyError, NotImplementedError) as exc:
        if R_prior is not None:
            nlls["oracle"] = float("nan")
        print(f"  [oracle_posterior] unavailable: {exc}")

    return nlls, R_dict, R_oracle, y_space_nlls, icl_y_parts


def _marginal_pit(
    ep: dict,
    tabicl_marginal: TabICLLike | None,
    k_folds: int,
    device: torch.device,
    marginal_backend: str | None = None,
    marginal_regressor: Any = None,
    seed: int = 0,
) -> dict[str, Tensor] | None:
    """K-fold PIT of one episode through a real marginal (TabICL or a batched backend).

    Returns {z_train, z_test, log_pdf_test} with log_pdf_test in raw nats, or None
    for fewer than 2 training points.
    """
    X_train = ep["x_norm_train"].to(device)  # (P, d_x)
    y_train = ep["y_train"].to(device)  # (P,)
    X_test = ep["x_norm_test"].to(device)  # (N, d_x)
    y_test = ep["y_test"].to(device)  # (N,)
    P = X_train.shape[0]
    if P < 2:
        return None
    y_train_scaled, y_test_scaled, _, std = normalize_targets(y_train, y_test)
    if marginal_backend is not None:
        # Batched backend PIT with a singleton episode axis.
        from copula_inter.backend_registry import batched_pit

        run_batched = batched_pit(marginal_backend)
        out = run_batched(
            marginal_regressor,
            X_train.unsqueeze(0).cpu().numpy(),
            y_train_scaled.unsqueeze(0).cpu().numpy(),
            X_test.unsqueeze(0).cpu().numpy(),
            y_test_scaled.unsqueeze(0).cpu().numpy(),
            k_folds=k_folds,
            seed=seed,
        )
        as_t = lambda a: torch.as_tensor(a[0], dtype=torch.float32, device=device)  # noqa: E731
        return {
            "z_train": as_t(out["z_train"]),
            "z_test": as_t(out["z_test"]),
            "log_pdf_test": as_t(out["log_pdf_test"]) - std.log(),
        }
    Y_train = y_train_scaled.unsqueeze(-1)  # (P, 1)
    Y_test = y_test_scaled.unsqueeze(-1)  # (N, 1)
    assert tabicl_marginal is not None
    pit_out = run_pit(
        tabicl_marginal,
        X_train,
        Y_train,
        X_test,
        Y_test,
        k_folds=k_folds,
        Y_train_raw=y_train.unsqueeze(-1),
    )
    return {
        "z_train": pit_out["z_train"].squeeze(-1),  # (P,)
        "z_test": pit_out["z_test"].squeeze(-1),  # (N,)
        "log_pdf_test": pit_out["log_pdf_test"].squeeze(-1) - std.log(),  # (N,) raw-nats
    }


def _baseline_worker_count(spec: EvalSpec, baseline_device: torch.device) -> int:
    """baselines.workers, defaulting to one per physical core (capped at 32); 1 on a GPU."""
    n_workers = spec.baselines.workers
    try:
        _aff = os.sched_getaffinity(0)
    except AttributeError:  # pragma: no cover - non-Linux
        _aff = set(range(os.cpu_count() or 1))
    n_physical = _count_physical_cores(_aff)
    if n_workers <= 0:
        n_workers = max(1, min(32, n_physical or len(_aff)))
    if n_physical and n_physical < len(_aff):
        print(
            f"  [prefit] note: the {len(_aff)} allocated logical CPUs are only "
            f"{n_physical} physical core(s) ({len(_aff) // n_physical} threads each) — "
            "expect scaling closer to the physical count; request more cores "
            "from the scheduler for a proportionally faster run"
        )
    if baseline_device.type != "cpu" and n_workers > 1:
        print(
            f"  [prefit] baselines.device={baseline_device.type}: forcing "
            "baselines.workers=1 (parallel processes would just contend for one GPU)"
        )
        n_workers = 1
    return n_workers


@dataclass
class _EvalContext:
    """Settings, models and cache identities shared by every scored episode."""

    spec: EvalSpec
    device: torch.device
    baseline_device: torch.device
    icl_model: CopulaTabICL
    tabicl_marginal: TabICLLike | None
    marginal_backend: str | None
    marginal_regressor: Any
    tabicl_pit_k_folds: int
    oracle_mode: str
    prior_cfg: dict
    fit_kwargs: dict[str, Any]
    # Baseline cache path, or None when caching is off.
    baseline_cache: str | None
    fingerprint: dict


def _prefit_baselines(ctx: _EvalContext, episode_plan: list[_PlannedEpisode], cache_entries: dict) -> dict[str, dict]:
    """Baselines available before scoring: valid cache entries, plus parallel fits of the rest."""
    spec = ctx.spec
    n_workers = _baseline_worker_count(spec, ctx.baseline_device)
    fitted: dict[str, dict] = {}
    if ctx.baseline_cache is not None and not spec.baselines.refresh:
        for planned in episode_plan:
            entry = _valid_cached_entry(
                cache_entries,
                planned.cache_key,
                planned.ep_i,
                expected_baseline_keys(spec.baselines.laplace),
                digest=episode_digest(planned.ep),
                ep=planned.ep,
            )
            if entry is not None:
                fitted[planned.cache_key] = entry

    pending = [(p.cache_key, p.fit_seed, p.ep) for p in episode_plan if p.cache_key not in fitted]
    print(
        f"\nBaselines: {len(fitted)} episode(s) reused from cache, "
        f"{len(pending)} to fit on {ctx.baseline_device.type}"
        + (f" across {n_workers} worker process(es)" if n_workers > 1 else " serially")
    )
    if pending and n_workers > 1:
        _prefit_baselines_parallel(
            pending,
            ctx.fit_kwargs,
            n_workers,
            ctx.baseline_cache,
            ctx.fingerprint,
            fitted,
        )
    return fitted


# (copula NLL per method, correlation matrix per method, Y-space parts per method)
_BaselineResults = tuple[dict[str, float], dict[str, Tensor], dict[str, dict[str, float]]]


def _episode_baselines(
    ctx: _EvalContext, entry: dict | None, ep: dict, cache_key: str, fit_seed: int
) -> _BaselineResults:
    """The episode's baseline results: from its prefit entry, or fitted here and cached."""
    device, baseline_device = ctx.device, ctx.baseline_device
    if entry is not None:
        return entry["nlls"], {k: v.to(device) for k, v in entry["R_dict"].items()}, entry["y_nlls"]
    # Serial fitting (one worker or a GPU baseline device): results moved to the
    # evaluation device; RNG saved/restored and one BLAS thread, matching the pool.
    with _snapshot_rng_and_threads(cpu_threads=1 if baseline_device.type == "cpu" else None):
        baseline_nlls, baseline_R, baseline_y_nlls = eval_baselines_episode(
            ep={k: (v.to(baseline_device) if isinstance(v, Tensor) else v) for k, v in ep.items()},
            device=baseline_device,
            fit_seed=fit_seed,
            **ctx.fit_kwargs,
        )
    baseline_R = {k: v.to(device) for k, v in baseline_R.items()}
    if ctx.baseline_cache is not None:
        save_baseline_entry(
            ctx.baseline_cache,
            ctx.fingerprint,
            cache_key,
            {
                "nlls": baseline_nlls,
                "R_dict": {k: v.cpu() for k, v in baseline_R.items()},
                "y_nlls": baseline_y_nlls,
                "episode_digest": episode_digest(ep),
                "episode_targets": episode_targets(ep),
            },
        )
    return baseline_nlls, baseline_R, baseline_y_nlls


def _episode_marginal_pit(ctx: _EvalContext, ep: dict, ep_i: int) -> dict[str, Tensor] | None:
    """The episode's own PIT (ERA5), else the loaded marginal's; None falls back to the oracle z_train."""
    marginal_pit = ep.get("marginal_pit")
    if marginal_pit is None and (ctx.tabicl_marginal is not None or ctx.marginal_regressor is not None):
        marginal_pit = _marginal_pit(
            ep=ep,
            tabicl_marginal=ctx.tabicl_marginal,
            k_folds=ctx.tabicl_pit_k_folds,
            device=ctx.device,
            marginal_backend=ctx.marginal_backend,
            marginal_regressor=ctx.marginal_regressor,
            seed=ep_i,
        )
        if marginal_pit is None:
            print(f"  [ep {ep_i}] fewer than 2 training points — falling back to oracle z_train for this episode")
    return marginal_pit


def _add_autoregressive_log_pdf(ctx: _EvalContext, ep: dict, local_i: int, ep_i: int) -> None:
    """Store the teacher-forced autoregressive chain's log-densities in ep (ERA5 precomputes them)."""
    spec = ctx.spec
    if not spec.autoregressive.enabled or "ar_log_pdf" in ep:
        return
    if spec.autoregressive.n_episodes is not None and local_i >= spec.autoregressive.n_episodes:
        return
    if ctx.tabicl_marginal is None:
        raise RuntimeError("autoregressive scoring needs the loaded TabICL marginal")
    device = ctx.device
    ep["ar_log_pdf"] = (
        autoregressive_log_pdf(
            ctx.tabicl_marginal,
            ep["x_norm_train"].to(device).unsqueeze(0),
            ep["y_train"].to(device).unsqueeze(0),
            ep["x_norm_test"].to(device).unsqueeze(0),
            ep["y_test"].to(device).unsqueeze(0),
            order=spec.autoregressive.order,
            conditioning=spec.autoregressive.conditioning,
            max_context=spec.autoregressive.max_context,
            seed=spec.seed,
            episode_indices=[ep_i],
        )["log_pdf"][0]
        .detach()
        .cpu()
    )


def _baselines_in_raw_nats(ep: dict, baselines: _BaselineResults) -> _BaselineResults:
    """Put the fitted baselines' Y-space rows in raw nats.

    They are fitted on ep["y_test"], which is standardized when the episode carries
    y_log_std = log sigma_y > 0; the change of variables adds log sigma_y per point to
    the marginal and total NLL (not the copula). Call this before merging rows that
    are already in raw nats (the zero-mean GP rows, scored on marginal_pit), or those
    rows get the shift a second time.
    """
    shift = float(ep.get("y_log_std", 0.0))
    if not shift:
        return baselines
    baseline_nlls, baseline_R, baseline_y_nlls = baselines
    shifted = {
        k: {"total": v["total"] + shift, "marginal": v["marginal"] + shift, "copula": v["copula"]}
        for k, v in baseline_y_nlls.items()
    }
    return baseline_nlls, baseline_R, shifted


def _with_zero_mean_gp(
    ctx: _EvalContext, ep: dict, marginal_pit: dict[str, Tensor], fit_seed: int, baselines: _BaselineResults
) -> _BaselineResults:
    """Add the zero-mean GP baselines on the marginal's z_train, fitted here (not cached).

    Seeded from fit_seed with the RNG restored and one thread.
    """
    spec = ctx.spec
    with _snapshot_rng_and_threads(
        seed=fit_seed + 1,
        cpu_threads=1 if ctx.baseline_device.type == "cpu" else None,
    ):
        zm_nlls, zm_R, zm_y_nlls = _eval_zero_mean_gp_baselines(
            ep=ep,
            marginal_pit=marginal_pit,
            device=ctx.baseline_device,
            n_steps=spec.baselines.n_steps_zeromean_gp,
            lr=spec.baselines.lr_zeromean_gp,
            n_restarts=spec.baselines.n_restarts_zeromean_gp,
            oracle_mode=ctx.oracle_mode,
            prior_cfg=ctx.prior_cfg,
        )
    baseline_nlls, baseline_R, baseline_y_nlls = baselines
    return (
        {**baseline_nlls, **zm_nlls},
        {**baseline_R, **{k: v.to(ctx.device) for k, v in zm_R.items()}},
        {**baseline_y_nlls, **zm_y_nlls},
    )


def _marginal_plugin_rows(
    marginal_pit: dict[str, Tensor] | None,
    baseline_nlls: dict[str, float],
    baseline_R: dict[str, Tensor],
    R_oracle: Tensor | None,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """The learned marginal with a plug-in Gaussian-copula correlation, in raw nats.

    marg_gp_evidence / marg_gp_laplace: R_post(theta_hat) of the BIC- / Laplace-selected
    GP-MLE, the matched competitors of the copula head (same marginal, GP-structured
    dependence).
    marg_oracle_corr: R_post(theta) of the known-theta GP, a reference (synthetic only).
    """
    rows = {
        "marg_gp_evidence": _NAN_PARTS.copy(),
        "marg_gp_laplace": _NAN_PARTS.copy(),
        "marg_oracle_corr": _NAN_PARTS.copy(),
    }
    if marginal_pit is None:
        return rows
    z = marginal_pit["z_test"].to(device).unsqueeze(0)
    log_pdf = marginal_pit["log_pdf_test"].to(device).unsqueeze(0)
    mask = torch.ones_like(z, dtype=torch.bool)
    # A failed selection has a nan shared-z NLL (and a placeholder identity R).
    sources: dict[str, Tensor | None] = {"marg_oracle_corr": R_oracle}
    for key, sel in (("marg_gp_evidence", "gp_evidence"), ("marg_gp_laplace", "gp_laplace")):
        sources[key] = baseline_R.get(sel) if np.isfinite(baseline_nlls.get(sel, float("nan"))) else None
    for key, R in sources.items():
        if R is None or not torch.isfinite(R).all():
            continue
        parts = y_space_nll(R.to(device=device, dtype=z.dtype).unsqueeze(0), z, log_pdf, mask)
        rows[key] = {k: v.item() for k, v in parts.items()}
    return rows


# Correlations whose expected copula loss the efficiency table compares with R_dagger:
# (key in the efficiency dict, source key in the scored R dicts).
_EFFICIENCY_SOURCES = (
    ("icl", "icl"),
    ("gp_evidence", "gp_evidence"),
    ("gp_laplace", "gp_laplace"),
    ("oracle_posterior", "oracle"),
)


def _corr_of(C: Tensor) -> Tensor:
    """The correlation matrix of a covariance C (float64), with a 1e-9 ridge so it is positive definite."""
    d = C.diagonal().clamp_min(1e-12).rsqrt()
    R = C * d[:, None] * d[None, :]
    return 0.5 * (R + R.T) + 1e-9 * torch.eye(C.shape[0], dtype=C.dtype, device=C.device)


def _attainable_rows(
    ctx: _EvalContext,
    ep: dict,
    ep_i: int,
    marginal_pit: dict[str, Tensor] | None,
    R_scored: dict[str, Tensor],
    nlls: dict[str, float],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """The attainable copulas under the frozen TabICL marginals, and the copula-efficiency losses.

    Draws y_Q ~ the known-theta GP posterior and maps each draw through the frozen marginals
    (query_marginal: the distributions behind marginal_pit's z_test). From the draws:
      M = E[z z'] gives the best Gaussian copula R_dagger over all correlations and in the
      head's covnorm family at each rank r (the head's rank plus attainable.ranks);
      any_copula_bound = E[-log p_U(U)] / N bounds every copula from below (equal for
      calibrated marginals), so the best copula of any family lies in [bound, l(R_dagger)].
    Rows (raw nats, the TOTAL table): marg_attainable (full rank) and marg_attainable_lowrank
    (the head's rank), each with the marginal. Efficiency (meta): l(R) = E[-log c_R(z)] / N
    for "attainable", "attainable_r<r>", "any_copula_bound" and every scored R in
    _EFFICIENCY_SOURCES (independence is 0); "m_diag" = mean(diag M), 1 for calibrated marginals.
    M = C + m m' with m = E[z], the marginals' bias at the queries: "mean_bias_sq" = mean(m^2) and
    "attainable_centered" = l(R_C) under M for R_C fitted to C alone, the best copula that only
    models the spread of z and not the shared bias the oracle knows, and "corr_z" = l(corr(C)),
    z's own correlation: the copula that models dependence only, neither the bias nor the
    over/under-dispersion of the marginals (diag C != 1) that R_C can still trade against.
    With attainable.bayes_draws > 0, the same from the Bayes predictive (theta marginalised given the
    context, gp_bayes.bayes_predictive): M_bayes from its draws gives R_bayes, a copula a context-only
    model with the kernel family and warps known could reach; "bayes_attainable" = l(R_bayes) under
    the TRUE M, "bayes_attainable_r<r>" its rank-r version, "bayes_attainable_centered" and "bayes_corr_z" the
    same from the Bayes covariance, "bayes_bias_cos" = cos(m_bayes, m), "bayes_attainable_self" under
    M_bayes, "bayes_ess" its importance ESS. Rows marg_bayes_attainable (the marginal + R_bayes)
    and oracle_bayes (the predictive's own NLL). "icl_offdiag_abs", "<name>_offdiag_abs" and
    "icl_<name>_offdiag_corr" compare the head's off-diagonal correlations with R_dagger's and R_bayes's.
    Empty/nan without a known-theta posterior (ERA5, real data), a TabICL marginal, or attainable.enabled.
    """
    rows = {
        key: _NAN_PARTS.copy()
        for key in ("marg_attainable", "marg_attainable_lowrank", "marg_bayes_attainable", "oracle_bayes")
    }
    att = ctx.spec.attainable
    if not att.enabled or marginal_pit is None or ctx.tabicl_marginal is None or ctx.marginal_backend is not None:
        return rows, {}
    try:
        post = gp_analytical_posterior(ep)
    except (KeyError, NotImplementedError):
        return rows, {}
    device = ctx.device
    y_train = ep["y_train"].to(device)
    _, _, y_mean, y_std = normalize_targets(y_train)
    marginal = query_marginal(
        ctx.tabicl_marginal, ep["x_norm_train"].to(device), (y_train - y_mean) / y_std, ep["x_norm_test"].to(device)
    )
    mu = post["mu_post"].to(device).double()
    Sigma = post["Sigma_post"].to(device).double()
    N = mu.shape[0]
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-9 * torch.eye(N, dtype=torch.float64, device=device)
    gen = torch.Generator(device="cpu").manual_seed((ctx.spec.seed * 1_000_003 + ep_i) % (2**31 - 1))
    noise = torch.randn(att.n_samples, N, generator=gen, dtype=torch.float64).to(device)
    y_draws = mu + noise @ torch.linalg.cholesky(Sigma).T  # (L, N) raw
    scaled = (y_draws - y_mean.double()) / y_std.double()  # on the marginal's scale
    chunk = 256
    with torch.no_grad():
        m_true, M = pit_moments(marginal.cdf, scaled, chunk=chunk)
        # sum_j log f_j(y_j) in raw nats (the scaled density's Jacobian is 1 / std per coordinate).
        log_marg = torch.cat(
            [marginal.log_prob(scaled[i : i + chunk]).double().sum(-1) for i in range(0, att.n_samples, chunk)]
        ) - N * torch.log(y_std.double())
        log_joint = torch.distributions.MultivariateNormal(mu, covariance_matrix=Sigma).log_prob(y_draws)

    head_rank = int(ctx.fit_kwargs["icl_rank"])
    parametrization = getattr(ctx.icl_model, "correlation_parametrization", "covnorm")
    R_full = attainable_correlation(M, None, max_iter=att.max_iter)
    C_true = M - torch.outer(m_true, m_true)
    R_centered = attainable_correlation(C_true, None, max_iter=att.max_iter)
    efficiency = {
        "m_diag": float(M.diagonal().mean()),
        "mean_bias_sq": float(m_true.pow(2).mean()),
        "any_copula_bound": any_copula_bound(log_joint, log_marg, N),
        "attainable": expected_copula_nll(R_full, M),
        "attainable_centered": expected_copula_nll(R_centered, M),
        "corr_z": expected_copula_nll(_corr_of(C_true), M),
    }
    ranks = sorted({head_rank, *att.ranks}) if parametrization == "covnorm" else []
    R_head = None
    for r in ranks:
        R_r = attainable_correlation(M, r, max_iter=att.max_iter, parametrization=parametrization)
        efficiency[f"attainable_r{r}"] = expected_copula_nll(R_r, M)
        if r == head_rank:
            R_head = R_r

    R_bayes = None
    if att.bayes_draws > 0:
        bayes = bayes_predictive(
            ctx.prior_cfg,
            ep,
            att.bayes_draws,
            device,
            seed=(ctx.spec.seed * 1_000_033 + ep_i) % (2**31 - 1),
            max_keep=att.bayes_keep,
        )
        with torch.no_grad():
            y_bayes = bayes.sample(att.n_samples, gen)
            m_bayes, M_bayes = pit_moments(marginal.cdf, (y_bayes - y_mean.double()) / y_std.double(), chunk=chunk)
            y_test = ep["y_test"].to(device).double().unsqueeze(0)
            total = -float(bayes.log_prob(y_test)) / N
            marg = -float(bayes.marginal_log_prob(y_test)) / N
        rows["oracle_bayes"] = {"total": total, "marginal": marg, "copula": total - marg}
        R_bayes = attainable_correlation(M_bayes, None, max_iter=att.max_iter)
        efficiency["bayes_attainable"] = expected_copula_nll(R_bayes, M)
        efficiency["bayes_attainable_self"] = expected_copula_nll(R_bayes, M_bayes)
        efficiency["bayes_ess"] = bayes.ess
        C_bayes = M_bayes - torch.outer(m_bayes, m_bayes)
        R_bayes_centered = attainable_correlation(C_bayes, None, max_iter=att.max_iter)
        efficiency["bayes_attainable_centered"] = expected_copula_nll(R_bayes_centered, M)
        efficiency["bayes_corr_z"] = expected_copula_nll(_corr_of(C_bayes), M)
        efficiency["bayes_bias_cos"] = float(m_bayes @ m_true / (m_bayes.norm() * m_true.norm()).clamp_min(1e-12))
        for r in ranks:
            R_r = attainable_correlation(M_bayes, r, max_iter=att.max_iter, parametrization=parametrization)
            efficiency[f"bayes_attainable_r{r}"] = expected_copula_nll(R_r, M)

    z = marginal_pit["z_test"].to(device).unsqueeze(0)
    log_pdf = marginal_pit["log_pdf_test"].to(device).unsqueeze(0)
    mask = torch.ones_like(z, dtype=torch.bool)
    for key, R in (
        ("marg_attainable", R_full),
        ("marg_attainable_lowrank", R_head),
        ("marg_bayes_attainable", R_bayes),
    ):
        if R is not None:
            parts = y_space_nll(R.to(device=device, dtype=z.dtype).unsqueeze(0), z, log_pdf, mask)
            rows[key] = {k: v.item() for k, v in parts.items()}
    for key, source in _EFFICIENCY_SOURCES:
        R = R_scored.get(source)
        # A failed fit or selection has a nan NLL and a placeholder identity R.
        if R is not None and np.isfinite(nlls.get(source, float("nan"))) and torch.isfinite(R).all():
            efficiency[key] = expected_copula_nll(R.to(device), M)
    R_icl = R_scored.get("icl")
    if "icl" in efficiency and R_icl is not None:
        # How the head's correlations relate to R_dagger's and R_bayes's: off-diagonal pattern
        # correlation and mean absolute off-diagonal (a shrunk but aligned head has corr ~1, smaller abs).
        iu = torch.triu_indices(N, N, offset=1, device=device)
        off_icl = R_icl.to(device).double()[iu[0], iu[1]]
        efficiency["icl_offdiag_abs"] = float(off_icl.abs().mean())
        for name, R in (("attainable", R_full), ("bayes", R_bayes)):
            if R is not None:
                off = R.double()[iu[0], iu[1]]
                efficiency[f"{name}_offdiag_abs"] = float(off.abs().mean())
                if float(off_icl.std()) > 0 and float(off.std()) > 0:
                    efficiency[f"icl_{name}_offdiag_corr"] = float(torch.corrcoef(torch.stack([off_icl, off]))[0, 1])
    efficiency["head_rank"] = float(head_rank)
    return rows, efficiency


def _total_nll_rows(
    ep: dict,
    baseline_y_nlls: dict[str, dict[str, float]],
    icl_y_parts: dict[str, float],
    y_space_nlls: dict[str, dict[str, float]],
    marginal_pit: dict[str, Tensor] | None,
) -> dict[str, dict[str, float]]:
    """Per-point total/marginal/copula Y-space NLL for every row of the TOTAL table.

    Every input row is already in raw nats (baseline_y_nlls via _baselines_in_raw_nats).
    """
    n_test = ep["z_test"].shape[0]
    # Autoregressive row (already in raw nats).
    ar_log_pdf = ep.get("ar_log_pdf")
    ar_parts = (
        ar_parts_from_log_pdf(ar_log_pdf, marginal_pit["log_pdf_test"].cpu())
        if ar_log_pdf is not None and marginal_pit is not None
        else _NAN_PARTS.copy()
    )
    marginal_only = {
        "total": icl_y_parts["marginal"],
        "marginal": icl_y_parts["marginal"],
        "copula": 0.0 if not np.isnan(icl_y_parts["marginal"]) else float("nan"),
    }
    total_nlls = {
        **baseline_y_nlls,
        "icl": icl_y_parts,
        "independence_marginal": marginal_only,
        "autoregressive": ar_parts,
        "oracle_prior": {k: v / n_test for k, v in y_space_nlls["prior"].items()},
        "oracle_posterior": {k: v / n_test for k, v in y_space_nlls["posterior"].items()},
    }
    gp_totals = [
        total_nlls.get(k, _NAN_PARTS)["total"]
        for k in _GP_TOTAL_KEYS
        if not np.isnan(total_nlls.get(k, _NAN_PARTS)["total"])
    ]
    total_nlls["best_gp_total"] = {
        "total": min(gp_totals) if gp_totals else float("nan"),
        "marginal": float("nan"),
        "copula": float("nan"),
    }
    return total_nlls


def _print_episode(
    ep: dict,
    ep_i: int,
    nlls: dict[str, float],
    total_nlls: dict[str, dict[str, float]],
    best_nll: float,
    fold_details: list[dict],
) -> None:
    """Per-episode summary: shared-z_test copula NLLs, the TOTAL rows, best baseline and top 5."""
    icl_nll = nlls.get("icl", float("nan"))
    ora_nll = nlls.get("oracle", float("nan"))
    ranked_baselines = sorted(
        ((k, v) for k, v in nlls.items() if k not in _NON_FITTED_EXCLUDED),
        key=lambda kv: kv[1],
    )
    top5 = ranked_baselines[:5]
    print(f"  ep {ep_i:04d}: kernel={_kernel_composition_label(ep)}")
    # Per-point numbers against the shared z_test.
    print(
        f"    icl(shared_marginal)={icl_nll:.4f}  "
        f"oracle(shared_marginal, R_post known theta)={ora_nll:.4f}  "
        f"GP-oracle-y-space-total(prior={total_nlls['oracle_prior']['total']:.4f}, "
        f"posterior={total_nlls['oracle_posterior']['total']:.4f})"
    )
    print(
        f"    total Y-space NLL, own marginal (total = marginal + copula): "
        f"icl=(cop={total_nlls['icl']['copula']:.4f}, "
        f"marg={total_nlls['icl']['marginal']:.4f}, "
        f"tot={total_nlls['icl']['total']:.4f})  "
        f"oracle_prior=(cop={total_nlls['oracle_prior']['copula']:.4f}, "
        f"marg={total_nlls['oracle_prior']['marginal']:.4f}, "
        f"tot={total_nlls['oracle_prior']['total']:.4f})  "
        f"oracle_posterior=(cop={total_nlls['oracle_posterior']['copula']:.4f}, "
        f"marg={total_nlls['oracle_posterior']['marginal']:.4f}, "
        f"tot={total_nlls['oracle_posterior']['total']:.4f})"
    )
    if fold_details:
        fold_summary = ", ".join(
            f"fold{fd['fold']}={_METHOD_LABELS.get(fd['selected'], fd['selected'])}" for fd in fold_details
        )
        print(f"    best_baseline (nested {len(fold_details)}-fold CV, pooled test NLL)={best_nll:.4f}")
        print(f"      per-fold picks: {fold_summary}")
    else:
        print("    best_baseline: unavailable (too few test points for nested CV)")
    print(
        "    top-5 baselines (ranked by shared-ground-truth-marginal copula "
        "NLL, diagnostic only — not the selection used for best_baseline "
        "above; 'own' columns are this baseline's OWN fitted marginal, "
        "NOT the same copula quantity as the ranking column — see "
        "eval_baselines_episode's docstring):"
    )
    for key, val in top5:
        own = total_nlls.get(key, _NAN_PARTS)
        print(
            f"      {_METHOD_LABELS.get(key, key):<28}"
            f"shared_copula={val:.4f}  "
            f"own(cop={own['copula']:.4f}, marg={own['marginal']:.4f}, "
            f"tot={own['total']:.4f})"
        )


@dataclass
class _EpisodeScore:
    nlls: dict[str, float]  # copula NLL per method (shared z_test), incl. best_baseline
    total_nlls: dict[str, dict[str, float]]  # TOTAL-table rows
    y_space_nlls: dict[str, dict[str, float]]  # oracle Y-space parts
    meta: dict
    R_dict: dict[str, Tensor]  # correlation per method
    R_oracle: Tensor | None
    best_key: str | None  # method nested CV picked (for the plot)


def _score_episode(
    ctx: _EvalContext,
    ep: dict,
    local_i: int,
    ep_i: int,
    fit_seed: int,
    baselines: _BaselineResults,
) -> _EpisodeScore:
    """Score one episode: the ICL model, the oracle and every baseline, then the nested-CV best baseline."""
    spec, device = ctx.spec, ctx.device
    marginal_pit = _episode_marginal_pit(ctx, ep, ep_i)
    _add_autoregressive_log_pdf(ctx, ep, local_i, ep_i)
    # Shift the standardized-y baseline rows to raw nats before adding the zero-mean GP
    # rows, which are scored on marginal_pit and already in raw nats.
    baselines = _baselines_in_raw_nats(ep, baselines)
    if spec.baselines.zeromean_gp and marginal_pit is not None:
        baselines = _with_zero_mean_gp(ctx, ep, marginal_pit, fit_seed, baselines)
    baseline_nlls, baseline_R, baseline_y_nlls = baselines

    icl_nlls, icl_R, R_oracle, y_space_nlls, icl_y_parts = _eval_icl_episode(
        ep=ep,
        icl_model=ctx.icl_model,
        device=device,
        marginal_pit=marginal_pit,
    )
    plugin_rows = _marginal_plugin_rows(marginal_pit, baseline_nlls, baseline_R, R_oracle, device)
    R_scored = {**baseline_R, **icl_R, **({"oracle": R_oracle} if R_oracle is not None else {})}
    attainable_rows, efficiency = _attainable_rows(ctx, ep, ep_i, marginal_pit, R_scored, {**baseline_nlls, **icl_nlls})
    total_nlls = _total_nll_rows(
        ep, {**baseline_y_nlls, **plugin_rows, **attainable_rows}, icl_y_parts, y_space_nlls, marginal_pit
    )
    meta = {
        "ep_i": ep_i,
        "n_test": ep["z_test"].shape[0],
        "kernel": _kernel_composition_label(ep),
    }
    if efficiency:
        # Expected copula NLL per point under M (see _attainable_rows); 0 = independence.
        meta["copula_efficiency"] = efficiency
    if "real_meta" in ep:
        # Identifies the episode (rows, split, group) for paired comparisons and cluster bootstraps.
        meta["episode"] = ep["real_meta"]
        # The marginal-calibration gate reads these.
        meta["z_test"] = [float(z) for z in ep["z_test"].tolist()]
    nlls = {**baseline_nlls, **icl_nlls}
    R_dict = {**baseline_R, **icl_R}

    # Best fitted baseline per episode, selected by nested CV over the test points.
    holdout_seed = (spec.seed * 1_000_003 + ep_i) % (2**31 - 1)
    best_nll, mode_key, fold_details = _select_best_baseline_cv(
        baseline_R,
        ep["z_test"].to(device),
        spec.selection.n_folds,
        spec.selection.min_fold_size,
        holdout_seed,
    )
    _print_episode(ep, ep_i, nlls, total_nlls, best_nll, fold_details)
    nlls["best_baseline"] = best_nll
    return _EpisodeScore(nlls, total_nlls, y_space_nlls, meta, R_dict, R_oracle, mode_key)
