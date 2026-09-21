"""eval_checkpoint.py — Evaluate a CopulaTabICL checkpoint against every
classical baseline (see eval/baselines/classical.py) plus the oracle
lower bound, on held-out PIT episodes.

This is the CLAUDE.md-documented daily workflow's evaluation entrypoint,
replacing src/evaluate_baselines.py.

Usage
-----
    python eval/runners/eval_checkpoint.py \\
        --config conf/config.yaml \\
        --ckpt   ./checkpoints/copula_transformer/step_XXXXXX_final.pt \\
        [--n_episodes 30]         # episodes to evaluate
        [--episode_idx 0]         # starting episode index (--dataset_dir only)
        [--episode_offset 0]      # first global episode index (--live_generate only;
                                  # shard one episode stream across array jobs)
        [--n_steps_mle 1000]      # Adam steps for GP MLE fitting (also used for ARD variants)
        [--lr_mle 0.05]           # learning rate for GP MLE
        [--n_restarts_mle 5]      # random restarts per GP-MLE kernel fit
        [--n_steps_dkl 5000]      # Adam steps for Deep Kernel Learning (MLP+GP) fitting
        [--lr_dkl 0.01]           # learning rate for DKL Adam
        [--n_restarts_dkl 2]      # random restarts per DKL kernel fit (fresh MLP each time)
        [--n_steps_per_ep 5000]   # training steps for PerEpisodeTransformer
        [--patience_per_ep 500]   # early stopping patience (steps without improvement)
        [--baseline_device cpu]   # where to fit baselines (cpu is FASTER here, see below)
        [--baseline_workers 0]    # 0=auto (min(8, allocated cores)); 1=serial
        [--z_train_source tabicl]  # (default) 'oracle', or exaone/tabpfn/tabldm:
                                   #   feed the ICL model the
                                   # exact GP-LOO z_train instead of TabICL's own
                                   # K-fold PIT estimate, to measure the sim-to-real
                                   # gap ('oracle' leaves the total-NLL table's icl
                                   # row NaN — no learned marginal to score against)
        [--tabicl_ckpt ...]        # TabICL checkpoint for --z_train_source=tabicl
                                   # (default: cfg.tabicl.ckpt from --config)
        [--tabicl_pit_k_folds 10]  # K-fold count for --z_train_source=tabicl
        [--zeromean_gp / --no-zeromean_gp]  # (default: on) fit "Marginal + Zero
                                   # Mean GP (RBF/Matern32)" directly on
                                   # marginal_pit['z_train'] -- see classical.
                                   # fit_zero_mean_gp_on_marginal's docstring.
                                   # No-op under --z_train_source=oracle.
        [--n_steps_zeromean_gp 500]   # Adam steps per Zero-Mean GP z-space fit
        [--lr_zeromean_gp 0.05]       # learning rate for the Zero-Mean GP Adam fits
        [--n_restarts_zeromean_gp 2]  # random restarts per Zero-Mean GP kernel fit
        [--plot_episode 0]        # local episode index to plot corr_grid for
        [--out_dir ./eval/results]  # directory to save corr_grid figure
        [--device auto]
        [--seed 42]
        [--baseline_cache ./baseline_cache.pt]  # cache fitted baseline results across runs
        [--no_baseline_cache]      # disable the cache entirely
        [--refresh_baselines]      # ignore cached entries, refit and overwrite them

Baseline caching is handled by eval/baselines/classical.py (see its module
docstring): GP-MLE/DKL/per_ep_transformer fitting dominates runtime and is
unaffected by which checkpoint is under test, so repeated runs against a new
checkpoint reuse the cached fits and only redo the cheap ICL forward pass +
oracle NLL. Each episode is written to the cache the moment it is fitted (one
small file per episode), so a run killed at its OAR walltime still leaves
behind every fit it paid for.

Runtime
-------
Baseline fitting is ~98% of this script's cost. Measured at the defaults
above (P=32 train / N=256 test / d_x=9): 649 s per episode, split GP-MLE
350 s (9 kernel+ARD labels x 5 restarts x 1000 steps) / polynomial 121 s
(3 degrees x 5 restarts) / DKL 168 s (at the OLD n_restarts_dkl=1 default;
see --n_restarts_dkl's help for why that was a bug, not a speed choice --
at the current default of 2 restarts DKL costs ~336 s instead, so 817 s/
episode total) / per_ep_transformer 10 s. That is ~85,000 Adam steps per
episode, every one of them a 32x32 Cholesky, so the work is dominated by
per-step launch latency rather than arithmetic. Two consequences, both
handled by the defaults:

  * --baseline_device defaults to **cpu**, which is ~2x faster than a GPU
    here for bit-comparable NLLs (measured: GP-MLE rbf 2.98 s vs 7.50 s per
    1000 steps, DKL rbf 20.23 s vs 38.54 s, nll 2.1604 either way).
  * --baseline_workers spreads episodes (perfectly independent) across
    processes, defaulting to the cores actually allocated to the job.

Measured end to end on a real 400-episode run: 78.6 s/episode against the
649 s/episode above, i.e. ~8x, on an allocation of 8 logical CPUs that were
only 4 physical cores. Speedup is ~2x from the device and the rest from
PHYSICAL core count, so ask the scheduler for real cores rather than threads
(see _count_physical_cores, which prints the distinction at startup). Note
what is
NOT a speed knob: --n_steps_mle materially changes the reported baseline
numbers rather than merely refining them (under oracle_mode="prior", longer
fits sharpen the fitted kernel's prior correlation at the test points and the
copula NLL rises monotonically — measured ~11 -> ~25 nats for ARD-RBF between
100 and 1000 steps), so it must be chosen on a convergence criterion over the
fitting objective and held fixed, never trimmed to fit a walltime.

"Marginal + Zero Mean GP (RBF/Matern32)" (--zeromean_gp, default on) adds a
SEPARATE, much cheaper fit on top of the above: 2 kernels x
--n_restarts_zeromean_gp (2) x --n_steps_zeromean_gp (500) restarts/steps on
(X_train, marginal_pit['z_train']) instead of raw y_train -- see
eval.baselines.classical.fit_zero_mean_gp_on_marginal's docstring for why
this is a distinct, legitimate baseline. Profiled on one real episode
(P=32/N=256/d_x=11, era5-12m fine-tuned TabICLv2 marginal): 1.8-1.9 ms/Adam-
step on 1 CPU thread vs. 2.8-2.9 ms/step on GPU (same ~1.5x CPU-is-faster
pattern as the y-space GP-MLE fits above, for the same reason -- P=32 means
every step is a small Cholesky dominated by launch latency, not arithmetic),
fit NLL plateauing by step 500 (500->1000 changes NLL by <0.0002) and by
restart 2 (a 3rd restart finds nothing further). At those defaults the whole
addition costs ~3.8 s/episode serially in the main loop (not the parallel
worker pool -- see _eval_zero_mean_gp_baselines' docstring for why), a small
fraction of the ~78.6 s/episode baseline-fitting cost measured above.

With --live_generate (the default), the episodes themselves come from
--config's own cfg.data — resolved through Hydra's defaults list, NOT the
checkpoint's own saved training cfg (see _load_full_config) — so the same
--config + --seed always produces the same episodes and the same baseline
cache fingerprint no matter which --ckpt you point at, even across
checkpoints trained under different cfg.data. The tradeoff: every checkpoint
is scored against one shared distribution (whatever --config currently
says) rather than its own training distribution. Edit --config or pass a
different file to change that distribution deliberately.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import multiprocessing as mp
import os
import random
import sys
import time
import zlib
from collections import Counter

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch import Tensor

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gen import _parse_composite, generate_gp_batch  # noqa: E402
from dataset import CopulaDataset  # noqa: E402
from inference.copula_inference import load_copula_model  # noqa: E402
from loss import y_space_nll  # noqa: E402
from model import low_rank_correlation  # noqa: E402
from pit import (  # noqa: E402
    DEFAULT_K_FOLDS,
    configure_tabicl_inference_amp,
    gp_analytical_posterior,
    load_tabicl,
    normalize_targets,
    run_pit,
)

from eval.baselines.classical import (  # noqa: E402
    EXPECTED_BASELINE_KEYS,
    GP_VAL_SELECT_MODES,
    assert_shared_z_test,
    baseline_fingerprint,
    corr_nll_single,
    episode_cache_key,
    eval_baselines_episode,
    fit_zero_mean_gp_on_marginal,
    load_baseline_cache,
    save_baseline_entry,
)
from eval.viz.correlation_plots import plot_corr_grid  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def _snapshot_rng_and_threads(seed: int | None = None, cpu_threads: int | None = None):
    """Snapshot the global torch RNG (CPU + CUDA) and restore it on exit;
    optionally reseed to `seed` and/or pin intra-op thread count to
    `cpu_threads` for the duration. Both the serial classical-baseline fit
    and the Marginal + Zero Mean GP fit below need this same pattern —
    reseed so one baseline's random draws don't shift what the next one in
    the loop sees, single-thread so a CPU fit's floating-point reduction
    order does not depend on how many cores the machine happened to offer
    (measured elsewhere in this file: up to 5.4e4 nats of difference
    between 1-thread and 8-thread runs on a near-singular R) — differing
    only in which half of the pattern each call site needs, hence one
    parameterised context manager instead of two hand-rolled copies.
    """
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


def _load_full_config(config_path: str) -> OmegaConf:
    """Resolve --config through Hydra's defaults list (model/data groups +
    _self_), the same composition train.py's @hydra.main gets, instead of a
    bare OmegaConf.load (which would leave cfg.data missing entirely — see
    conf/config.yaml's `defaults:` block).

    This is deliberately independent of any checkpoint: it's the fixed
    episode-generating distribution used for live generation and for the
    baseline cache's fingerprint (see main()), so switching --ckpt between
    checkpoints trained under different cfg.data no longer invalidates cached
    baseline fits — only editing --config itself, or passing a different
    one, does.
    """
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        return hydra.compose(config_name=config_name)


# ---------------------------------------------------------------------------
# ICL model + oracle evaluation (the cheap, per-checkpoint part)
# ---------------------------------------------------------------------------

# Shared all-nan {"total", "marginal", "copula"} placeholder — same shape as
# eval_baselines_episode's y_space_nlls entries (classical.py's _NAN_PARTS),
# duplicated here rather than imported since it's a plain literal, not
# shared state.
_NAN_PARTS: dict[str, float] = {"total": float("nan"), "marginal": float("nan"), "copula": float("nan")}


def _eval_icl_episode(
    ep: dict,
    icl_model: nn.Module,
    device: torch.device,
    marginal_pit: dict[str, Tensor] | None = None,
) -> tuple[dict[str, float], dict[str, Tensor], Tensor, dict[str, dict[str, float]], dict[str, float]]:
    """Evaluate just the ICL model + oracle lower bound on one episode — the
    cheap, per-checkpoint part of the comparison (no fitting/training loop),
    always recomputed even when the baseline results are served from cache.

    marginal_pit, when given (see --z_train_source=tabicl in main(), built by
    _tabicl_pit), replaces the episode's own exact GP-LOO z_train as the ICL
    model's conditioning input via its "z_train" key — everything else
    (z_test, R_oracle, the baselines) still scores/fits against the
    episode's true values, since only the model's *input* is meant to
    change, not what "correct" means for the copula-only `nlls` table.

    marginal_pit's "z_test"/"log_pdf_test" keys additionally provide a
    genuine (non-oracle) marginal for the ICL model itself, letting it be
    scored on total (marginal+copula) Y-space NLL the same way the
    GP-MLE/DKL baselines now are (see eval_baselines_episode) — this is
    only possible in --z_train_source=tabicl mode, since the default oracle
    z_test IS the ground truth (nothing to score a "marginal" against).

    Returns (nlls, R_dict, R_oracle, y_space_nlls, icl_y_parts) —
    y_space_nlls is a separate {"prior": {"total","marginal","copula"},
    "posterior": {...}} dict (see gp_analytical_posterior's docstring for
    why these two aren't folded into `nlls` alongside icl/oracle/baselines:
    they're a full multivariate-normal Y-space NLL, not the z-space
    copula-only NLL every other entry in `nlls` is, so they're not in the
    same units/comparable via the same table — all-nan dicts when
    unavailable). icl_y_parts is {"total","marginal","copula"}, all-nan
    whenever marginal_pit is None (oracle z_train mode) — same per-episode
    Sklar split now exposed for every baseline via eval_baselines_episode.
    """
    X_train = ep["x_norm_train"].to(device)   # (P, d_x)
    z_train = (
        marginal_pit["z_train"].to(device) if marginal_pit is not None
        else ep["z_train"].to(device)
    )                                            # (P,)  ICL's conditioning input — oracle LOO-PIT residual by default
    X_test  = ep["x_norm_test"].to(device)     # (N, d_x)
    z_test  = ep["z_test"].to(device)          # (N,)
    assert_shared_z_test(z_test, ep)
    R_oracle = ep["R_star"].to(device)         # (N, N)

    P, N = X_train.shape[0], X_test.shape[0]
    nlls: dict[str, float] = {}
    R_dict: dict[str, Tensor] = {}
    R_I = torch.eye(N, dtype=X_train.dtype, device=device)
    icl_y_parts = _NAN_PARTS.copy()

    try:
        train_mask = torch.ones(1, P, dtype=torch.bool, device=device)
        batch = {
            "x_train":   X_train.unsqueeze(0),
            "x_test":    X_test.unsqueeze(0),
            "z_train":   z_train.unsqueeze(0),
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

    nlls["oracle"] = corr_nll_single(R_oracle, z_test)
    R_dict["oracle"] = R_oracle

    # "oracle" above is the PRIOR reference (cfg.data.oracle_mode="prior" —
    # the only mode data_gen.py's training pipeline supports): R_star = raw
    # kernel correlation among test points, never conditioned on the
    # realized (x_train, y_train). It is NOT a true lower bound in the sense
    # of "the best achievable full predictive NLL" — the actual Bayes-optimal
    # reference additionally Schur-complement-conditions on context, computed
    # below via gp_analytical_posterior as a SEPARATE total Y-space NLL (not
    # folded into `nlls`/`R_dict`'s z-space-copula-only comparison — see that
    # function's docstring for why the two aren't in the same units). R_post
    # is still stashed into R_dict purely for the correlation-grid plot (a
    # descriptive visual of what conditioning does to the correlation
    # structure), never scored against z_test.
    y_space_nlls = {"prior": _NAN_PARTS.copy(), "posterior": _NAN_PARTS.copy()}
    try:
        post = gp_analytical_posterior(ep)
        R_dict["oracle_posterior"] = post["R_post"].to(device)
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
        R_dict["oracle_posterior"] = R_I.clone()
        print(f"  [oracle_posterior] unavailable: {exc}")

    return nlls, R_dict, R_oracle, y_space_nlls, icl_y_parts


def _marginal_pit(
    ep: dict,
    tabicl_marginal: nn.Module | None,
    k_folds: int,
    device: torch.device,
    marginal_backend: str | None = None,
    marginal_regressor=None,
    marginal_probs_n: int = 99,
    seed: int = 0,
) -> dict[str, Tensor] | None:
    """K-fold PIT from a real (non-oracle) marginal, in place of the
    episode's exact GP-LOO/posterior PIT — the same "does the model's
    correlation prediction hold up against an estimated marginal instead of
    the oracle one" check src/train.py's _build_tabicl_val_z runs during
    training, used here at eval time via --z_train_source.

    Two sources, one output contract. --z_train_source=tabicl uses the
    frozen TabICL marginal via pit.py::run_pit (`tabicl_marginal`); the
    other backends (exaone/tabpfn/tabldm, see
    eval/spatial/marginal_backends.py) go through their shared batched PIT
    module with a leading singleton episode axis (`marginal_backend` /
    `marginal_regressor`). Both return z_train/z_test/log_pdf_test with
    log_pdf_test in RAW nats, so nothing downstream needs to know which
    marginal produced them.

    Unlike this function's predecessor (_tabicl_z_train, which queried
    X_train[:1]/Y_train[:1] as a throwaway probe since it only needed
    z_train), this queries the episode's REAL X_test/Y_test, so run_pit's
    single test-side forward pass also returns a genuine TabICL marginal at
    the test points (z_test, log_pdf_test) — the missing ingredient for
    scoring the ICL model's own total (marginal+copula) Y-space NLL,
    instead of only the copula-only NLL scored against the oracle's
    ground-truth-standardized z_test.

    Returns None (caller falls back to the oracle z_train/z_test) when the
    episode has fewer than 2 training points, since run_pit's fold split
    needs at least that many.

    y_train/y_test are z-scored via pit.normalize_targets (y_test scaled
    with y_train's own mean/std, never its own — see that function's
    docstring) before reaching the raw TabICL module: run_pit does no
    target scaling of its own (unlike tabicl.TabICLRegressor.fit(), which
    fits a fresh StandardScaler before ever calling this same underlying
    model). Every other run_pit call site in the repo
    (inference/copula_inference.py::loo_pit, train.py::_build_tabicl_val_z)
    goes through the same helper, so this conditioning input is computed
    identically everywhere. Episode y's scale is not fixed — outputscale is
    drawn from a GammaPrior (data_gen.py's generative process) — so an
    unscaled call risks saturating the pretrained quantile head's CDF into
    its extreme tail for every point alike on high-outputscale episodes,
    collapsing the PIT residuals' spread instead of reflecting the true
    per-point rank.

    log_pdf_test comes back in that same normalize_targets-scaled space, so
    it is NOT directly comparable in raw nats to the GP-MLE/DKL baselines'
    mvn_nll (computed in raw y-units) or the oracle's nll_prior/nll_post
    (also raw-scale) — a Jacobian correction (log p_raw(y) = log
    p_scaled(y_scaled) - log(std), per normalize_targets' own docstring) is
    applied here before returning, so every caller of this function's
    log_pdf_test gets raw-nats units without needing to know about the
    internal scaling.
    """
    X_train = ep["x_norm_train"].to(device)   # (P, d_x)
    y_train = ep["y_train"].to(device)         # (P,)
    X_test  = ep["x_norm_test"].to(device)     # (N, d_x)
    y_test  = ep["y_test"].to(device)          # (N,)
    P = X_train.shape[0]
    if P < 2:
        return None
    y_train_scaled, y_test_scaled, _, std = normalize_targets(y_train, y_test)
    if marginal_backend is not None:
        # Same batched module the training pipelines use, B=1 -- so eval and
        # training score the identical PIT recipe per backend rather than
        # this file growing its own per-backend copy.
        from data_gen import _BATCHED_MARGINAL_BACKENDS

        run_batched = _BATCHED_MARGINAL_BACKENDS[marginal_backend]()
        out = run_batched(
            marginal_regressor,
            X_train.unsqueeze(0).cpu().numpy(), y_train_scaled.unsqueeze(0).cpu().numpy(),
            X_test.unsqueeze(0).cpu().numpy(), y_test_scaled.unsqueeze(0).cpu().numpy(),
            k_folds=k_folds, probs_n=marginal_probs_n, seed=seed,
        )
        as_t = lambda a: torch.as_tensor(a[0], dtype=torch.float32, device=device)  # noqa: E731
        return {
            "z_train": as_t(out["z_train"]),
            "z_test": as_t(out["z_test"]),
            "log_pdf_test": as_t(out["log_pdf_test"]) - std.log(),
        }
    Y_train = y_train_scaled.unsqueeze(-1)      # (P, 1)
    Y_test  = y_test_scaled.unsqueeze(-1)       # (N, 1)
    pit_out = run_pit(
        tabicl_marginal, X_train, Y_train, X_test, Y_test, k_folds=k_folds,
    )
    return {
        "z_train":      pit_out["z_train"].squeeze(-1),                    # (P,)
        "z_test":       pit_out["z_test"].squeeze(-1),                     # (N,)
        "log_pdf_test": pit_out["log_pdf_test"].squeeze(-1) - std.log(),   # (N,) raw-nats
    }


# The "few" kernels requested for the Zero-Mean GP baseline — RBF and
# Matern32 cover the two most common, well-behaved general-purpose kernel
# families (see classical.fit_zero_mean_gp_on_marginal's module docstring)
# without paying for the full GP-MLE kernel+ARD sweep a second time in
# z-space. Fixed (not CLI-configurable) so _METHOD_ORDER/_TOTAL_NLL_ORDER
# below can list both rows unconditionally instead of growing/shrinking
# columns at runtime.
_ZEROMEAN_GP_KERNELS = ("rbf", "matern32")


def _eval_zero_mean_gp_baselines(
    ep: dict,
    marginal_pit: dict[str, Tensor],
    device: torch.device,
    n_steps: int,
    lr: float,
    n_restarts: int,
    oracle_mode: str,
    prior_cfg: dict,
) -> tuple[dict[str, float], dict[str, Tensor], dict[str, dict[str, float]]]:
    """Zero-mean GP-MLE baselines ("Marginal + Zero Mean GP (...)") fit
    directly on marginal_pit["z_train"] — the SAME real, non-oracle marginal
    input the ICL model itself conditions on (see
    classical.fit_zero_mean_gp_on_marginal's module docstring for why this
    is legitimate, unlike fitting a baseline on the oracle z_train).

    nlls/R_dict follow the shared-ground-truth-marginal convention every
    other baseline uses (scored against ep["z_test"], not
    marginal_pit["z_test"]), so the caller can merge them straight into
    baseline_nlls/baseline_R and they participate in the best_baseline
    nested-CV ranking like any other fitted candidate. y_space_nlls instead
    scores each fit's OWN total (marginal + copula) Y-space NLL via
    marginal_pit's z_test/log_pdf_test — the same real marginal the
    correlation was just fit against — mirroring _eval_icl_episode's
    icl_y_parts exactly, so "does a classical GP beat the ICL copula head
    given an IDENTICAL real marginal" is an apples-to-apples comparison in
    both tables.

    Not routed through eval_baselines_episode/the baseline_cache worker
    pool: unlike every other classical baseline, this fit depends on which
    marginal produced z_train (--z_train_source / --tabicl_ckpt), which that
    checkpoint-independent, oracle-episode-only cache has no way to key on.
    """
    X_train = ep["x_norm_train"].to(device)   # (P, d_x)
    X_test = ep["x_norm_test"].to(device)     # (N, d_x)
    z_test = ep["z_test"].to(device)          # (N,) ground truth, shared across every baseline
    z_train_marg = marginal_pit["z_train"].to(device)   # (P,) real marginal's PIT residual
    N = X_test.shape[0]
    test_mask = torch.ones(1, N, dtype=torch.bool, device=device)
    R_I = torch.eye(N, dtype=X_train.dtype, device=device)

    nlls: dict[str, float] = {}
    R_dict: dict[str, Tensor] = {}
    y_space_nlls: dict[str, dict[str, float]] = {}
    for kname in _ZEROMEAN_GP_KERNELS:
        label = f"gp_zeromean_{kname}"
        try:
            fit = fit_zero_mean_gp_on_marginal(
                X_train, z_train_marg, X_test, kname,
                n_steps=n_steps, lr=lr, n_restarts=n_restarts,
                oracle_mode=oracle_mode, prior_cfg=prior_cfg,
            )
            nlls[label] = corr_nll_single(fit["R"], z_test)
            R_dict[label] = fit["R"]
            parts = y_space_nll(
                fit["R"].unsqueeze(0),
                marginal_pit["z_test"].to(device).unsqueeze(0),
                marginal_pit["log_pdf_test"].to(device).unsqueeze(0),
                test_mask,
            )
            y_space_nlls[label] = {k: v.item() for k, v in parts.items()}
        except Exception as exc:
            print(f"  [{label}] failed: {exc}")
            nlls[label] = float("nan")
            R_dict[label] = R_I.clone()
            y_space_nlls[label] = _NAN_PARTS.copy()
    return nlls, R_dict, y_space_nlls


def _make_folds(n: int, k: int, seed: int) -> list[Tensor]:
    """Deterministic, per-episode partition of the n test-point indices into
    k disjoint folds of near-equal size (sizes differ by at most 1) —
    independent of the global RNG (a fresh CPU-seeded Generator), so it
    doesn't perturb the GP-MLE/DKL restarts' own randomness elsewhere in the
    run.
    """
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    base, extra = divmod(n, k)
    folds: list[Tensor] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        folds.append(perm[start:start + size])
        start += size
    return folds


def _select_best_baseline_cv(
    baseline_R: dict[str, Tensor], z_test: Tensor, n_folds: int, min_fold_size: int, seed: int,
) -> tuple[float, str | None, list[dict]]:
    """Pick the per-episode best *fitted* baseline honestly via nested
    (leave-one-fold-out) cross-validation over the n test points, instead of
    argmin-ing directly over the same z_test the winner is then scored on
    (a selection-bias/winner's-curse leak — see Cawley & Talbot 2010, "On
    Over-fitting in Model Selection and Subsequent Selection Bias in
    Performance Evaluation") and instead of a single fixed val/test split
    (this function's predecessor, _select_best_baseline_holdout), which
    permanently sacrifices a fraction of the points to selection alone —
    wasteful and noisy at this repo's small N (N_min=8).

    For each of K folds: rank candidates by NLL on the other K-1 folds
    (val), then score the winner's NLL on the held-out fold (test) — every
    point plays val in K-1 folds and test in exactly 1, so no point is ever
    used to both select and score the same baseline.

    R_star being a valid (N, N) correlation matrix means any principal
    submatrix R[idx][:, idx] is too, so this needs no refit — the same NxN
    correlation each baseline already produced at fit time is just scored
    against different index subsets.

    K = min(n_folds, n // min_fold_size), so no fold — nor its (K-1)-fold
    val complement, which is always >= one fold's own size — ever scores a
    candidate on fewer than min_fold_size points. Returns (nan, None, [])
    when there are no fitted candidates or this leaves K < 2 (no CV
    possible, n_test too small): with min_fold_size=20 and this repo's
    N_min=8/N_max=128 (uniform), that's true for about a quarter of
    episodes (N < 40) — those simply contribute no best_baseline value
    rather than a noisy one (see _print_table's valid-count note).

    The pooled NLL returned is the size-weighted average of the K held-out
    fold NLLs (each already normalized by its own fold size in
    corr_nll_single) — i.e. the total unnormalized NLL summed across the K
    independent fold-blocks, divided by n. fold_details records each fold's
    selection/scores for diagnostics (console printing).
    """
    keys = [k for k in baseline_R if k not in _NON_FITTED_EXCLUDED]
    n = z_test.shape[0]
    if not keys or n // min_fold_size < 2:
        return float("nan"), None, []

    k_folds = min(n_folds, n // min_fold_size)
    folds = [f.to(z_test.device) for f in _make_folds(n, k_folds, seed)]

    def _sub_nll(key: str, idx: Tensor) -> float:
        R = baseline_R[key]
        R_sub = R.index_select(0, idx).index_select(1, idx)
        return corr_nll_single(R_sub, z_test.index_select(0, idx))

    fold_details: list[dict] = []
    weighted_sum = 0.0
    for i, test_idx in enumerate(folds):
        val_idx = torch.cat([f for j, f in enumerate(folds) if j != i])
        val_nll = {key: _sub_nll(key, val_idx) for key in keys}
        selected = min(val_nll, key=val_nll.get)
        test_nll = _sub_nll(selected, test_idx)
        weighted_sum += test_idx.numel() * test_nll
        fold_details.append({
            "fold": i, "size": test_idx.numel(), "selected": selected,
            "val_nll": val_nll[selected], "test_nll": test_nll,
        })

    pooled_nll = weighted_sum / n
    mode_key = Counter(fd["selected"] for fd in fold_details).most_common(1)[0][0]
    return pooled_nll, mode_key, fold_details


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

_METHOD_ORDER = [
    ("independence",        "Independence"),
    ("gp_prior_rbf",        "GP-Prior-RBF"),
    ("gp_mle_rbf",          "GP-MLE-RBF"),
    ("gp_mle_ard_rbf",      "GP-MLE-ARD-RBF"),
    ("gp_mle_matern32",     "GP-MLE-Matern32"),
    ("gp_mle_ard_matern32", "GP-MLE-ARD-Matern32"),
    ("gp_mle_periodic",     "GP-MLE-Periodic"),
    ("gp_mle_ard_periodic", "GP-MLE-ARD-Periodic"),
    ("gp_mle_rq",           "GP-MLE-RQ"),
    ("gp_mle_ard_rq",       "GP-MLE-ARD-RQ"),
    ("gp_mle_dot_product",  "GP-MLE-DotProduct"),
    ("gp_mle_polynomial",   "GP-MLE-Polynomial"),
    ("gp_zeromean_rbf",     "Marginal + Zero Mean GP (RBF)"),
    ("gp_zeromean_matern32", "Marginal + Zero Mean GP (Matern32)"),
    ("dkl_rbf",             "Deep Kernel Learning (RBF)"),
    ("dkl_matern32",        "Deep Kernel Learning (Matern32)"),
    ("dkl_rq",              "Deep Kernel Learning (RQ)"),
    ("dkl_dot_product",     "Deep Kernel Learning (DotProduct)"),
    ("per_ep_transformer",  "PerEp-Transformer"),
    ("best_baseline",       "Best-of-Baselines (per-episode)"),
    ("icl",                 "ICL (pretrained)"),
    ("oracle",              "Oracle (prior)"),
]

# Excluded from the "5 best baselines" ranking: independence/gp_prior_rbf
# are trivial, no-fit reference points rather than baselines, icl/oracle
# aren't baselines at all (icl is our model, oracle is a reference, not a
# fitted candidate), and best_baseline is itself derived from this same
# ranking (added after it's computed each episode — see main()'s loop).
_NON_FITTED_EXCLUDED = {"independence", "gp_prior_rbf", "icl", "oracle", "best_baseline"}

# oracle_posterior only ever appears in R_dict (the correlation-grid plot),
# never in `nlls`/_METHOD_ORDER's z-space table — see
# _eval_icl_episode's docstring for why its NLL isn't comparable in that
# table's units. Kept here only so the plot panel gets a readable title.
_METHOD_LABELS = dict(_METHOD_ORDER) | {"oracle_posterior": "Oracle (posterior)"}


def _kernel_composition_label(ep: dict) -> str:
    """Human-readable kernel-composition string for one episode (e.g.
    "rbf(ARD)+periodic, mlp-mixing"), built from the return_kernel_metadata
    fields generate_gp_batch attaches — absent entirely for episodes loaded
    from a pre-built dataset that didn't request that metadata (the common
    case for existing PIT datasets on disk today)."""
    if "kernel" not in ep:
        return "unavailable (pass --dataset_dir with pre-generated metadata, or use --live_generate)"

    if "kernel_components" in ep:
        parts = [ep["kernel_components"][0]]
        for op, comp_name in zip(ep["kernel_ops"], ep["kernel_components"][1:]):
            parts.append(op)
            parts.append(comp_name)
        label = " ".join(parts)
        ard_tags = [
            comp_name + "(ARD)"
            for comp_name, comp_params in zip(ep["kernel_components"], ep["kernel_component_params"])
            if torch.is_tensor(comp_params.get("l")) and comp_params["l"].numel() > 1
        ]
    else:
        label = ep["kernel"]
        composite = _parse_composite(ep["kernel"])
        ard_tags = []
        if composite is None:
            if torch.is_tensor(ep.get("l")) and ep["l"].numel() > 1:
                ard_tags.append(f"{ep['kernel']}(ARD)")
        else:
            name_a, _op, name_b = composite
            if torch.is_tensor(ep.get("l")) and ep["l"].numel() > 1:
                ard_tags.append(f"{name_a}(ARD)")
            if torch.is_tensor(ep.get("l_b")) and ep["l_b"].numel() > 1:
                ard_tags.append(f"{name_b}(ARD)")

    if ard_tags:
        label = f"{label}  [{', '.join(ard_tags)}]"
    if bool(ep.get("mlp_mixed", False)):
        label = f"{label}, mlp-mixing"
    return label


# ---------------------------------------------------------------------------
# Parallel baseline fitting
# ---------------------------------------------------------------------------
#
# Baseline fitting is ~98% of this script's runtime (measured at the argparse
# defaults, P=32/N=256/d_x=9: 649 s per episode, of which GP-MLE 350 s,
# polynomial 121 s, DKL 168 s, per_ep_transformer 10 s) and is perfectly
# independent across episodes — nothing in eval_baselines_episode reads any
# state shared with another episode. Two facts make that worth exploiting:
#
#   1. It is faster on ONE CPU core than on a GPU. Episodes carry P=32
#      training points, so every one of the ~85,000 Adam steps per episode is
#      a 32x32 Cholesky: nanoseconds of arithmetic behind a millisecond of
#      kernel-launch latency. Measured on a TITAN-RTX-class GPU vs. a single
#      Xeon E5-2623 v3 thread, same seeds, same steps: GP-MLE rbf 7.50 s ->
#      2.98 s, ARD matern32 6.41 s -> 3.53 s, RQ 7.37 s -> 4.37 s, DKL rbf
#      38.54 s -> 20.23 s, with identical NLLs (2.1604 vs 2.1604 for rbf).
#      Hence --baseline_device defaults to cpu.
#   2. Being CPU-bound and single-threaded, it then parallelises across cores
#      with no GPU contention and no memory pressure (an episode is a few MB).
#
# So the expensive part runs as a pool pre-pass over the uncached episodes,
# and the main evaluation loop below is left untouched: it finds every
# episode already in cache_entries and does only the cheap, genuinely
# checkpoint-dependent work (the ICL forward pass, the TabICL PIT, the oracle)
# on the GPU, in order, exactly as before.


def _fit_baselines_task(payload: tuple) -> tuple:
    """One episode's classical baselines, fit in a worker process.

    Module-level (not a closure) so it survives pickling under the "spawn"
    start method, which _prefit_baselines_parallel uses unconditionally: the
    parent has almost certainly initialised CUDA by this point (the ICL model
    and TabICL marginal are already resident), and a forked child inheriting
    a CUDA context crashes the moment it touches a tensor. Spawned workers
    re-import this module from scratch and never initialise CUDA at all.

    Tensors arrive and leave on the CPU; the caller moves R back to the
    evaluation device.
    """
    cache_key, ep, fit_seed, kwargs = payload
    import torch as _torch  # re-imported in the spawned interpreter

    # One thread per worker: these are 32x32 problems, so intra-op threading
    # buys nothing and merely oversubscribes the cores the pool is already
    # using for real parallelism (and on an OAR allocation, cores this job
    # was never given).
    _torch.set_num_threads(1)
    try:
        nlls, R_dict, y_nlls = eval_baselines_episode(
            ep=ep, device=_torch.device("cpu"), fit_seed=fit_seed, **kwargs
        )
    except Exception as exc:  # pragma: no cover - defensive
        import traceback

        return cache_key, None, f"{exc}\n{traceback.format_exc()}"
    return cache_key, {
        "nlls": nlls,
        "R_dict": {k: v.detach().cpu() for k, v in R_dict.items()},
        "y_nlls": y_nlls,
    }, None


def _results_fingerprint(baseline_fp: dict, args, tabicl_pit_k_folds: int) -> dict:
    """Everything that determines an episode's SCORED results, not just its
    fitted baselines.

    Strictly wider than baseline_fingerprint: baseline fits are deliberately
    checkpoint-independent (that is what lets one cache serve many --ckpt
    runs), but the numbers this runner reports are not — they include the ICL
    model's own NLL, the marginal PIT it conditions on, and the nested-CV
    best_baseline pick. Reusing a scored episode across a different --ckpt
    would silently report the OLD checkpoint's results, so the checkpoint and
    every marginal/CV setting belong in this key.
    """
    ckpt = os.path.abspath(args.ckpt) if args.ckpt else None
    try:
        ckpt_mtime = os.path.getmtime(ckpt) if ckpt and os.path.exists(ckpt) else None
    except OSError:
        ckpt_mtime = None
    return {
        "baseline": baseline_fp,
        "ckpt": ckpt,
        "ckpt_mtime": ckpt_mtime,
        "z_train_source": args.z_train_source,
        "tabicl_ckpt": (os.path.abspath(args.tabicl_ckpt)
                        if args.tabicl_ckpt else None),
        "tabicl_pit_k_folds": tabicl_pit_k_folds,
        "tabicl_amp": args.tabicl_amp,
        "marginal_probs_n": args.marginal_probs_n,
        "n_folds": args.n_folds,
        "min_fold_size": args.min_fold_size,
        "seed": args.seed,
        "zeromean_gp": args.zeromean_gp,
        "n_steps_zeromean_gp": args.n_steps_zeromean_gp,
        "lr_zeromean_gp": args.lr_zeromean_gp,
        "n_restarts_zeromean_gp": args.n_restarts_zeromean_gp,
    }


def _jsonable(obj):
    """Plain-Python copy of a nested result dict, for the JSON results cache.

    The per-episode values are floats almost everywhere, but a few come
    straight out of torch (gp_analytical_posterior's raw sums), and a 0-dim
    tensor would make json.dump raise mid-run — after the episode was already
    computed, which is exactly the work the cache exists to protect.
    """
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return float(obj)


def _load_results_cache(path: str, fingerprint: dict) -> dict[str, dict]:
    """Per-episode scored results from a previous, possibly interrupted run."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            blob = json.load(fh)
    except Exception as exc:
        print(f"  [results_cache] failed to read {path}: {exc} — starting fresh")
        return {}
    if blob.get("fingerprint") != fingerprint:
        print(f"  [results_cache] {path} was produced under different settings "
              "(checkpoint, marginal or episode config) — ignoring it")
        return {}
    entries = blob.get("episodes", {})
    print(f"  [results_cache] resuming with {len(entries)} already-scored episode(s) "
          f"from {path}")
    return entries


def _save_results_cache(path: str, fingerprint: dict, entries: dict[str, dict]) -> None:
    """Write scored results atomically.

    Called after EVERY episode rather than every N: unlike the baseline cache
    (whose entries carry N x N R matrices, ~4.3 MB each), these are a few
    dozen floats per episode, so rewriting the whole file each time is
    negligible and there is no reason to risk losing even one episode's ICL
    forward pass, marginal PIT and CV selection.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump({"fingerprint": fingerprint, "episodes": entries}, fh)
    os.replace(tmp, path)


def _count_physical_cores(cpus: set[int]) -> int:
    """How many distinct physical cores the given logical CPUs sit on.

    A scheduler allocation is reported in logical CPUs, which on a
    hyperthreaded node can be half as many real cores — and baseline fitting
    scales with the real ones. Returns 0 if the topology is unreadable, in
    which case the caller simply says nothing.
    """
    cores = set()
    for c in cpus:
        try:
            with open(
                f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
            ) as fh:
                cores.add(fh.read().strip())
        except OSError:
            return 0
    return len(cores)


def _baseline_fit_seed(seed: int, cache_key: str) -> int:
    """Deterministic per-episode seed for baseline fitting.

    Keyed off the episode's cache key (which encodes its global index), not
    its position in this run's loop, so an episode fits identically whether it
    was episode 3 of 400 in one job or episode 3 of a --episode_offset shard,
    and regardless of which worker process picked it up.

    crc32, not Python's hash(): str.__hash__ is salted by PYTHONHASHSEED and
    would make every run silently unreproducible.
    """
    return (zlib.crc32(cache_key.encode()) ^ (seed * 2_654_435_761)) % (2 ** 31 - 1)


def _valid_cached_entry(cache_entries: dict, cache_key: str, ep_i: int) -> dict | None:
    """The cached baseline entry for this episode, or None if it is missing or
    was written by an older version of eval_baselines_episode.

    A fingerprint match only guarantees the episode and the fitting
    hyperparameters agree; it says nothing about which baselines existed, or
    what shape their results had, when the entry was written. Each check below
    is a schema migration for one such change, and all of them mean the same
    thing: refit this episode rather than serve a result missing a key the
    tables downstream will index into.
    """
    cached = cache_entries.get(cache_key)
    if cached is None:
        return None
    if not EXPECTED_BASELINE_KEYS.issubset(cached["nlls"].keys()):
        # Predates a baseline added to eval_baselines_episode since (e.g.
        # gp_mle_polynomial).
        missing = EXPECTED_BASELINE_KEYS - cached["nlls"].keys()
        print(f"  [ep {ep_i}] cached baselines missing {sorted(missing)} — refitting")
        return None
    if "y_nlls" not in cached:
        # Predates the total Y-space NLL addition — refit rather than silently
        # leaving the total-NLL table's baseline rows as nan for this episode.
        print(f"  [ep {ep_i}] cached entry predates total-NLL tracking — refitting")
        return None
    if any(not isinstance(v, dict) for v in cached["y_nlls"].values()):
        # Predates the marginal/copula split of y_nlls (each value used to be
        # a bare total float) — refit rather than crashing on own["copula"].
        print(f"  [ep {ep_i}] cached y_nlls predates marginal/copula split — refitting")
        return None
    return cached


def _episode_to_cpu(ep: dict) -> dict:
    """CPU copy of an episode, for shipping to a worker process."""
    return {
        k: (v.detach().cpu() if isinstance(v, Tensor) else v) for k, v in ep.items()
    }


def _prefit_baselines_parallel(
    pending: list[tuple[str, int, dict]],
    fit_kwargs: dict,
    n_workers: int,
    cache_path: str,
    fingerprint: dict,
    fitted: dict,
    use_cache: bool,
) -> None:
    """Fit every episode in `pending` across a process pool, writing results
    into `fitted` (and, unless caching is off, straight to disk) as they
    complete.

    Each result is persisted the moment it arrives rather than at the end of
    the run, so a walltime kill keeps the fits already paid for. This matters
    more than it sounds: a full run is many GPU-hours, the whole point of the
    cache is that a *later* run against a different --ckpt reuses it, and
    before this change the single save at the end of main() meant any run
    that hit its walltime — the common case for large --n_episodes — wrote
    nothing at all and the next run started from zero.
    """
    total = len(pending)
    done = 0
    failures = 0
    t0 = time.time()

    ctx = mp.get_context("spawn")
    payloads = [
        (key, _episode_to_cpu(ep), fit_seed, fit_kwargs) for key, fit_seed, ep in pending
    ]

    with ctx.Pool(processes=n_workers) as pool:
        for cache_key, result, err in pool.imap_unordered(_fit_baselines_task, payloads):
            done += 1
            if err is not None:
                failures += 1
                print(f"  [prefit] {cache_key} FAILED:\n{err}", flush=True)
            else:
                fitted[cache_key] = result
                if use_cache:
                    # One small file per episode, written the moment it is
                    # fitted — see save_baseline_entry. Nothing completed is
                    # ever lost, and the cost does not grow with the cache.
                    save_baseline_entry(cache_path, fingerprint, cache_key, result)

            elapsed = time.time() - t0
            rate = elapsed / done
            eta = rate * (total - done)
            print(
                f"  [prefit {done}/{total}] {cache_key}  "
                f"({elapsed/60:.1f} min elapsed, {rate:.1f} s/ep wall, "
                f"ETA {eta/60:.1f} min)",
                flush=True,
            )

    print(
        f"  [prefit] fitted {done - failures}/{total} episode(s) on "
        f"{n_workers} worker(s) in {(time.time() - t0)/60:.1f} min"
        + (f" — {failures} FAILED" if failures else ""),
        flush=True,
    )


def _live_generate_alternating(
    gen_cfg, n_ep: int, device, seed: int, offset: int = 0,
) -> list[dict]:
    """Live-generate n_ep episodes, forcing every even global index (0, 2, 4,
    ...) to a single elementary kernel (no composition) so each consecutive
    pair of evaluated episodes includes one non-composite draw — otherwise
    non-composite episodes are rare under this repo's default composite
    kernel counts.

    generate_gp_batch samples its kernel structure once per call and shares
    it across the whole batch, so getting per-episode composition variety at
    all requires B=1 calls rather than a single batched B=n_ep call. Each
    call gets its own seed (seed + global_i): generate_gp_batch reseeds every
    RNG from cfg.seed at the start of each call, so reusing one seed across
    calls would otherwise resample the identical episode n_ep times.

    offset (--episode_offset) shifts the *global* episode index this call
    starts from, so a run can evaluate a contiguous slice of one shared
    episode stream instead of always restarting it at 0. Everything that
    identifies an episode — its generating seed, its even/odd non-composite
    parity, its baseline cache key, and its nested-CV holdout seed — is
    derived from global_i = offset + local_i rather than local_i, so episode
    k is bit-identical whether it was produced by a single --n_episodes 400
    run or by four --n_episodes 100 --episode_offset {0,100,200,300} shards.
    That is what makes sharding across OAR array jobs (each writing its own
    --baseline_cache, merged afterwards) a pure parallelisation of the same
    experiment rather than a different one. offset=0 reproduces the previous
    behaviour exactly (global_i == local_i).
    """
    episodes: list[dict] = []
    for local_i in range(n_ep):
        global_i = offset + local_i
        ep_cfg = copy.deepcopy(gen_cfg)
        ep_cfg.seed = seed + global_i
        if global_i % 2 == 0:
            # Force non-composite for both kernel-selection modes
            # _resolve_kernel_name / _sample_kernel_chain_structure support.
            if bool(getattr(ep_cfg.data, "systematic_composition", False)):
                ep_cfg.data.composite_num_kernels_min = 1
                ep_cfg.data.composite_num_kernels_max = 1
            else:
                fixed = getattr(ep_cfg.data, "kernel", None)
                if fixed:
                    composite = _parse_composite(str(fixed))
                    if composite is not None:
                        ep_cfg.data.kernel = composite[0]
                elif getattr(ep_cfg.data, "kernels", None):
                    pool = [k for k in ep_cfg.data.kernels if _parse_composite(k) is None]
                    if not pool:
                        raise ValueError(
                            f"cfg.data.kernels={list(ep_cfg.data.kernels)} contains only "
                            "composite kernels; cannot force a non-composite episode."
                        )
                    ep_cfg.data.kernels = pool
                # else: _resolve_kernel_name's own "rbf" default, already non-composite.
        episodes.extend(generate_gp_batch(ep_cfg, 1, device, return_kernel_metadata=True))
    return episodes


def _print_table(all_nlls: list[dict[str, float]], z_train_source: str = "tabicl") -> None:
    means = {k: float(np.nanmean([m.get(k, float("nan")) for m in all_nlls]))
             for k, _ in _METHOD_ORDER}
    stds  = {k: float(np.nanstd( [m.get(k, float("nan")) for m in all_nlls]))
             for k, _ in _METHOD_ORDER}

    col = max(22, max(len(label) for _, label in _METHOD_ORDER) + 2)
    total = col + 2 * 12
    print(f"\n{'─' * total}")
    print(f"Inter-instance copula NLL (z-space) — lower is better  [N={len(all_nlls)} episodes]")
    print(f"ICL z_train source: {z_train_source}"
          + ("  (exact GP-LOO PIT)" if z_train_source == "oracle"
             else f"  ({z_train_source} K-fold PIT estimate)"))
    print(f"{'─' * total}")
    print(f"{'Method':<{col}}{'Mean NLL':>12}{'Std NLL':>12}")
    print(f"{'─' * col}{'─' * 12}{'─' * 12}")
    for key, label in _METHOD_ORDER:
        m, s = means.get(key, float("nan")), stds.get(key, float("nan"))
        marker = ""
        if key == "best_baseline":
            n_valid = sum(1 for ep_m in all_nlls if not np.isnan(ep_m.get(key, float("nan"))))
            marker = (f"  ← per-episode best baseline (nested CV; "
                      f"valid for {n_valid}/{len(all_nlls)} episodes)")
        elif key == "icl":
            marker = "  ← our model"
        elif key == "oracle":
            marker = "  ← unconditional kernel corr. among test pts (NOT Bayes-optimal; see GP oracle Y-space NLL below)"
        print(f"{label:<{col}}{m:>12.4f}{s:>12.4f}{marker}")
    print(f"{'─' * total}\n")


def _print_y_space_oracle(y_space_nlls: list[dict[str, dict[str, float]]]) -> None:
    """Total (marginal + copula) Y-space multivariate-normal GP oracle NLL,
    prior vs. posterior — see gp_analytical_posterior's docstring for why
    this is a SEPARATE table from _print_table's z-space copula-only NLL
    (different units, not just a missing row there): this one directly
    answers "how much does conditioning on context help, in the units of a
    real predictive log-likelihood" and posterior <= prior is a real,
    provable guarantee here (Bayes-optimality of the posterior predictive
    under log-loss), unlike a same-z_test z-space copula comparison.

    Episodes where gp_analytical_posterior was unavailable (systematic-chain
    kernel with whole-chain outer sign modulation, or a --dataset_dir
    episode missing kernel metadata) contribute NaN to both columns and are
    excluded via nanmean/nanstd, same convention as _print_table.
    """
    prior_vals = [d["prior"]["total"] for d in y_space_nlls]
    post_vals  = [d["posterior"]["total"] for d in y_space_nlls]
    n_valid = sum(1 for d in y_space_nlls if not np.isnan(d["posterior"]["total"]))
    print(f"GP oracle total NLL (Y-space, marginal+copula) — lower is better, "
          f"posterior <= prior is a Bayes-optimality guarantee here "
          f"[valid for {n_valid}/{len(y_space_nlls)} episodes]")
    print(f"  prior (unconditioned):      mean={np.nanmean(prior_vals):.4f}  std={np.nanstd(prior_vals):.4f}")
    print(f"  posterior (Schur-conditioned): mean={np.nanmean(post_vals):.4f}  std={np.nanstd(post_vals):.4f}\n")


# Methods with a genuine (own) fit/marginal, i.e. real competitors in a
# ranking sense — independence/gp_prior_rbf are non-fit floor references and
# best_baseline/oracle are derived after the fact, so all four are excluded
# here (same reasons as _NON_FITTED_EXCLUDED, but icl stays IN since it's the
# model under evaluation). Shared by _print_total_nll_table's row order and
# both rank tables (_print_rank_table calls in main()).
_RANK_KEYS = [
    (k, label) for k, label in _METHOD_ORDER
    if k not in ("independence", "gp_prior_rbf", "best_baseline", "oracle")
]

# Row order for _print_total_nll_table: every method with a genuine (own)
# marginal, plus the two oracle rows (which have no z-space-only counterpart
# in _RANK_KEYS since they're Y-space-only quantities).
_TOTAL_NLL_ORDER = _RANK_KEYS + [
    ("oracle_prior", "Oracle (prior, unconditioned)"),
    ("oracle_posterior", "Oracle (posterior, Schur-conditioned)"),
]


def _print_total_nll_table(
    all_total_nlls: list[dict[str, dict[str, float]]], z_train_source: str,
) -> None:
    """Total (marginal + copula) Y-space NLL, EVERY method's own fitted/
    estimated marginal, all divided by that episode's own N (per-point,
    nats/point) — the genuinely cross-method-comparable counterpart to
    _print_table's shared-ground-truth-marginal copula-only table (see
    eval_baselines_episode's and _eval_icl_episode's docstrings for why
    each method supplying its own predictive density, scored at the same
    real y_test, is always a valid proper-scoring-rule comparison,
    regardless of how different the marginals are).

    Each `all_total_nlls[i][method]` is a {"total", "marginal", "copula"}
    dict (see eval_baselines_episode/_eval_icl_episode) — the Marginal/
    Copula columns below are each method's OWN split, not comparable to
    _print_table's shared-ground-truth-marginal copula NLL (see
    eval_baselines_episode's docstring, or the "NAMING TRAP" note in
    eval/metrics/joint_nll.py's module docstring, for why those are
    different quantities that happen to share a name).

    icl's row is nan whenever z_train_source == "oracle" (--z_train_source
    default): the oracle z_test the ICL model would otherwise be scored
    against IS the ground truth, so there is no learned marginal to score a
    total NLL against — --z_train_source=tabicl is required to populate it.

    oracle_prior/oracle_posterior here are gp_analytical_posterior's own
    nll_prior/nll_post (and their marginal/copula split), divided by N for
    this table only — the existing _print_y_space_oracle table's own
    numbers are NOT changed by this (kept unnormalized there for backward
    compatibility with any previously tracked output).
    """
    def _col(part: str) -> dict[str, float]:
        return {
            k: float(np.nanmean([m.get(k, _NAN_PARTS).get(part, float("nan")) for m in all_total_nlls]))
            for k, _ in _TOTAL_NLL_ORDER
        }

    means_total = _col("total")
    stds_total = {
        k: float(np.nanstd([m.get(k, _NAN_PARTS).get("total", float("nan")) for m in all_total_nlls]))
        for k, _ in _TOTAL_NLL_ORDER
    }
    means_marginal = _col("marginal")
    means_copula = _col("copula")

    col = max(22, max(len(label) for _, label in _TOTAL_NLL_ORDER) + 2)
    total = col + 4 * 12
    print(f"\n{'─' * total}")
    print(f"Total NLL (Y-space, marginal+copula, own marginal per method) — "
          f"lower is better  [N={len(all_total_nlls)} episodes]")
    print(f"ICL z_train source: {z_train_source}"
          + ("  (icl row n/a — oracle mode has no learned ICL marginal to score)"
             if z_train_source == "oracle" else f"  ({z_train_source} K-fold PIT estimate)"))
    print(f"{'─' * total}")
    print(f"{'Method':<{col}}{'Mean Total':>12}{'Std Total':>12}{'Mean Marg.':>12}{'Mean Cop.':>12}")
    print(f"{'─' * col}{'─' * 12}{'─' * 12}{'─' * 12}{'─' * 12}")
    for key, label in _TOTAL_NLL_ORDER:
        m, s = means_total.get(key, float("nan")), stds_total.get(key, float("nan"))
        mm, mc = means_marginal.get(key, float("nan")), means_copula.get(key, float("nan"))
        marker = "  ← our model" if key == "icl" else ""
        print(f"{label:<{col}}{m:>12.4f}{s:>12.4f}{mm:>12.4f}{mc:>12.4f}{marker}")
    print(f"{'─' * total}\n")


def _compute_ranks(values: list[dict[str, float]], keys: list[str]) -> dict[str, list[int]]:
    """Per-episode competition rank (1 = lowest/best NLL that episode) among
    `keys`. A method missing or NaN that episode (fit failure, oracle-mode
    no-op, too few training points) contributes no rank for it rather than a
    worst-case one — mirrors the nanmean/nanstd convention used everywhere
    else in this file, so a method's average rank reflects only the episodes
    it actually competed in (see the printed N column)."""
    ranks: dict[str, list[int]] = {k: [] for k in keys}
    for ep in values:
        scored = [(k, ep.get(k, float("nan"))) for k in keys]
        scored = [(k, v) for k, v in scored if not np.isnan(v)]
        scored.sort(key=lambda kv: kv[1])
        for r, (k, _) in enumerate(scored, start=1):
            ranks[k].append(r)
    return ranks


def _print_rank_table(
    values: list[dict[str, float]], order: list[tuple[str, str]], title: str,
    z_train_source: str,
) -> None:
    """Average and median per-episode rank for each method in `order`,
    computed by _compute_ranks over `values` (one {key: nll} dict per
    episode — either all_nlls' z-space NLLs or the "total" slice of
    all_total_nlls' Y-space NLLs, see the two call sites in main()). Mean
    rank rewards consistent placement, median is robust to the rare episode
    a normally-strong method fails or gets a pathological fit; reading both
    together separates "usually great, occasionally terrible" from
    "reliably mediocre". Sorted by mean rank ascending (best first).
    """
    keys = [k for k, _ in order]
    labels = dict(order)
    ranks = _compute_ranks(values, keys)
    rows = [
        (k, float(np.mean(rs)), float(np.median(rs)), len(rs))
        for k, rs in ranks.items() if rs
    ]
    rows.sort(key=lambda r: r[1])
    if not rows:
        return

    col = max(22, max(len(labels[k]) for k, *_ in rows) + 2)
    total = col + 3 * 12
    print(f"\n{'─' * total}")
    print(f"{title}  [1 = best of {len(keys)} candidates that episode; lower is better]")
    print(f"ICL z_train source: {z_train_source}")
    print(f"{'─' * total}")
    print(f"{'Method':<{col}}{'Mean Rank':>12}{'Median Rank':>12}{'N':>12}")
    print(f"{'─' * col}{'─' * 12}{'─' * 12}{'─' * 12}")
    for k, mean_r, med_r, n in rows:
        marker = "  ← our model" if k == "icl" else ""
        print(f"{labels[k]:<{col}}{mean_r:>12.2f}{med_r:>12.1f}{n:>12d}{marker}")
    print(f"{'─' * total}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ICL checkpoint vs baselines on inter-instance copula episodes"
    )
    parser.add_argument("--config",       default="conf/config.yaml",
                        help="Hydra config defining the eval-episode-generating "
                             "distribution (cfg.data) for --live_generate, "
                             "resolved through its own defaults list — "
                             "independent of --ckpt's saved training cfg. "
                             "Keeping this fixed is what lets the baseline "
                             "cache survive switching checkpoints.")
    parser.add_argument("--ckpt",         required=True)
    parser.add_argument("--dataset_dir",  default=None,
                        help="Episode directory to evaluate on (overrides "
                             "training.dataset_dir from --config). Passing "
                             "this disables --live_generate by default.")
    parser.add_argument("--live_generate", action=argparse.BooleanOptionalAction, default=None,
                        help="Generate evaluation episodes on the fly via "
                             "data_gen.generate_gp_batch(..., return_kernel_metadata=True) "
                             "instead of loading a pre-built PIT dataset directory. Default: "
                             "True unless --dataset_dir is given. --episode_idx is ignored "
                             "in this mode (episodes are freshly sampled, not indexed).")
    parser.add_argument("--n_episodes",   type=int,   default=30)
    parser.add_argument("--episode_idx",  type=int,   default=0)
    parser.add_argument("--n_steps_mle",  type=int,   default=1000,
                        help="Adam steps for GP kernel MLE fitting (also used for ARD variants)")
    parser.add_argument("--lr_mle",       type=float, default=0.05,
                        help="Learning rate for GP MLE Adam")
    parser.add_argument("--n_restarts_mle", type=int, default=5,
                        help="Independent random restarts per GP-MLE kernel fit (each "
                             "initialised by sampling from the same LogNormal/Gamma "
                             "hyperpriors data_gen.py's generative process uses); keeps "
                             "whichever restart reaches the best final training loss.")
    parser.add_argument("--zeromean_gp", action=argparse.BooleanOptionalAction, default=True,
                        help="Fit the 'Marginal + Zero Mean GP (RBF/Matern32)' baselines "
                             "directly on marginal_pit['z_train'] -- the same real "
                             "(non-oracle) marginal input the ICL model conditions on -- "
                             "instead of the raw y-space GP-MLE/DKL baselines above (see "
                             "eval.baselines.classical.fit_zero_mean_gp_on_marginal's "
                             "module docstring). Only 2 kernels (RBF, Matern32) by design, "
                             "to keep this cheap. Automatically a no-op for episodes with "
                             "no real marginal available (--z_train_source=oracle, or "
                             "fewer than 2 training points).")
    parser.add_argument("--n_steps_zeromean_gp", type=int, default=500,
                        help="Adam steps for each Zero-Mean GP z-space fit. Needs far "
                             "fewer than --n_steps_mle's 1000: profiled on a real episode "
                             "(P=32/N=256/d_x=11, RBF/Matern32, 1 CPU thread), the fit "
                             "NLL plateaus by step 500 (rbf -0.0252, matern32 -0.0386) and "
                             "barely moves by step 1000 (-0.0251/-0.0386) -- unlike raw "
                             "y-space GP-MLE, there is no separate mean/scale to also "
                             "discover here, just 2-3 kernel hyperparameters against an "
                             "already marginally-standardized target, so it converges much "
                             "faster. Chosen on that convergence plateau, not trimmed for "
                             "speed (see the module docstring's Runtime note on why "
                             "--n_steps_mle itself must never be tuned that way).")
    parser.add_argument("--lr_zeromean_gp", type=float, default=0.05,
                        help="Learning rate for the Zero-Mean GP Adam fits")
    parser.add_argument("--n_restarts_zeromean_gp", type=int, default=2,
                        help="Random restarts per Zero-Mean GP kernel fit -- fewer than "
                             "--n_restarts_mle's 5 by design. Profiled on the same episode: "
                             "restart 2 clearly beats restart 1 (rbf -0.0301 -> -0.0251, "
                             "matern32 -0.0442 -> -0.0386) but a 3rd restart finds nothing "
                             "further (identical NLL to 2), so 2 is the measured sweet spot "
                             "rather than a guess.")
    parser.add_argument("--n_steps_dkl",  type=int,   default=5000,
                        help="Adam steps for Deep Kernel Learning (MLP+GP) fitting")
    parser.add_argument("--lr_dkl",       type=float, default=0.01,
                        help="Learning rate for DKL Adam")
    parser.add_argument("--n_restarts_dkl", type=int, default=2,
                        help="Independent random restarts per DKL kernel fit -- each "
                             "restart gets a FRESH feature-extractor MLP (not the same "
                             "instance re-trained further), since a shared instance would "
                             "carry its already-updated weights from one restart into the "
                             "next and silently defeat the point of restarting (see "
                             "fit_and_eval_gpytorch's docstring). Was hardcoded to 1 "
                             "(no restart loop at all) until this option existed: DKL's "
                             "joint MLP+kernel landscape is harder and more init-sensitive "
                             "than plain GP-MLE's, so a single unlucky random MLP init had "
                             "no chance to recover, and diagnostics traced z-space copula "
                             "NLL swinging from ~5 to >100 nats/pt across episodes purely "
                             "on that one seed's luck. Measured on 4 held-out episodes "
                             "(dkl_rbf/dkl_dot_product, n_steps=2000): restart 2 roughly "
                             "halves the median z-space copula NLL vs. restart 1 (rbf "
                             "45.5 -> 18.9, dot_product 69.5 -> 55.9 nats/pt), a 3rd "
                             "restart found nothing further in every one of the 8 "
                             "(kernel, episode) combinations tried -- same restart-2-"
                             "suffices pattern as --n_restarts_zeromean_gp. DKL still ends "
                             "up well behind GP-MLE/Marginal+ZeroMean-GP even at restart 2: "
                             "unlike those, its kernel operates on a LEARNED feature space, "
                             "so the same rescale-to-defeat-the-lengthscale-prior "
                             "identifiability issue the module docstring already describes "
                             "for the held-out-NLL guard applies to the restarts here too "
                             "-- restarts pick a better basin, they do not fix the "
                             "underlying identifiability gap. (A LayerNorm on the MLP "
                             "output was tried as a fix and made things WORSE -- median "
                             "NLL 18.9 -> 57.0 for rbf, 55.9 -> 341.7 for dot_product on "
                             "the same episodes -- because per-sample normalisation erases "
                             "exactly the relative-magnitude information RBF/dot_product "
                             "need between different points; not applied.)")
    parser.add_argument("--n_steps_per_ep", type=int, default=5000,
                        help="Training steps for PerEpisodeTransformer")
    parser.add_argument("--patience_per_ep", type=int, default=500,
                        help="Early stopping patience for PerEpisodeTransformer")
    parser.add_argument("--z_train_source", default="tabicl",
                        choices=["oracle", "tabicl", "exaone", "tabpfn", "tabldm"],
                        help="What the ICL model conditions on for each episode's z_train. "
                             "'tabicl' (default): a K-fold cross-fitted PIT estimate from the "
                             "frozen TabICL marginal (pit.py::run_pit) — the same proxy "
                             "real-world deployment is stuck with, so this is the setting "
                             "that makes _print_total_nll_table's icl row (the genuinely "
                             "cross-method-comparable total marginal+copula NLL) populate; "
                             "adds the cost of one extra frozen TabICL forward pass per fold "
                             "per episode. 'oracle': the episode's exact GP-LOO PIT residual "
                             "(R&W Eq. 5.12) computed from the true generating kernel — "
                             "unavailable on real data, an idealized upper bound only; under "
                             "this mode the icl row of _print_total_nll_table is NaN (no "
                             "learned marginal to score a total NLL against). icl_nll under "
                             "either source is unaffected in every other respect: z_test, "
                             "R_oracle, and every baseline still score/fit against the "
                             "episode's true values — use both to measure the sim-to-real "
                             "gap. 'exaone'/'tabpfn'/'tabldm': the same non-oracle idea with "
                             "a different tabular foundation model supplying the marginal "
                             "(eval/spatial/marginal_backends.py), through the identical "
                             "batched PIT module the training pipelines use — so an eval "
                             "scores exactly the marginal a run trained against. These need "
                             "no --tabicl_ckpt; --tabicl_pit_k_folds still sets K, and "
                             "--marginal_probs_n sets their quantile-grid size.")
    parser.add_argument("--marginal_probs_n", type=int, default=99,
                        help="Quantile grid size for --z_train_source=exaone/tabpfn/tabldm "
                             "(ignored for oracle/tabicl, which use their own native grids). "
                             "Mirrors data.z_train_marginal_probs_n in conf/data/gp_tasks.yaml; "
                             "a proportional lever on those backends' per-episode cost.")
    parser.add_argument("--tabicl_ckpt",  default=None,
                        help="TabICL checkpoint filename for --z_train_source=tabicl. "
                             "Default: read from --config's cfg.tabicl.ckpt.")
    parser.add_argument("--tabicl_pit_k_folds", type=int, default=None,
                        help="K-fold count for --z_train_source=tabicl's run_pit call. "
                             f"Default: cfg.tabicl.pit_k_folds, falling back to "
                             f"pit.DEFAULT_K_FOLDS ({DEFAULT_K_FOLDS}).")
    parser.add_argument("--tabicl_amp", action=argparse.BooleanOptionalAction, default=True,
                        help="AMP (float16 autocast) for the frozen TabICL marginal's "
                             "forward passes under --z_train_source=tabicl (pit.py::"
                             "configure_tabicl_inference_amp) -- same knob training uses "
                             "via cfg.training.tabicl_inference_amp (default true there "
                             "too). eval_checkpoint.py never called this before, so every "
                             "past eval run got TabICL's own built-in default (AMP on) "
                             "regardless of this flag's default here. Pass --no-tabicl_amp "
                             "for float32 quantile-grid precision (conf/config.yaml's own "
                             "comment: matters more for eval's log_pdf_test/marginal-NLL "
                             "fidelity than for live-generation throughput) -- useful to "
                             "rule out AMP noise before attributing a small NLL gap (e.g. "
                             "float16 vs float32 backbone) to the checkpoints themselves.")
    parser.add_argument("--plot_episode", type=int,   default=0,
                        help="Local episode index to generate the corr_grid plot for")
    parser.add_argument("--out_dir",      default=os.path.join(_REPO_ROOT, "eval", "results"),
                        help="Directory for saved corr_grid figure")
    parser.add_argument("--dump_episodes", default=None,
                        help="Write per-episode NLLs (all_nlls + all_total_nlls, keyed by "
                             "ep_i, plus each episode's kernel label) to this JSON path. "
                             "Two runs (e.g. different --ckpt) sharing --config/--seed/"
                             "--live_generate see identical episodes (see "
                             "_live_generate_alternating), so their dumps can be joined on "
                             "ep_i for a PAIRED per-episode comparison — far lower variance "
                             "than comparing the two runs' printed Mean/Std NLL as independent "
                             "samples, since episode-to-episode difficulty (kernel, "
                             "lengthscale, N) cancels in the per-episode difference instead "
                             "of inflating each run's own across-episode std.")
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--n_folds",      type=int,   default=5,
                        help="Number of folds for nested (leave-one-fold-out) "
                             "cross-validation of the per-episode best_baseline pick: "
                             "each fold's held-out NLL is scored using a baseline "
                             "selected (argmin NLL) on the other K-1 folds only, and "
                             "the K fold NLLs are pooled — instead of argmin-ing "
                             "directly over z_test, which lets the winner peek at the "
                             "very data it's scored on (see _select_best_baseline_cv). "
                             "Actual K is capped at n_test // 3 per episode, so small-N "
                             "episodes automatically use fewer folds. The fold split is "
                             "deterministic per episode (seeded from --seed + episode "
                             "index) and never touches icl/oracle or any individual "
                             "baseline's own diagnostic row, which keep scoring against "
                             "the full test set as before.")
    parser.add_argument("--min_fold_size", type=int, default=20,
                        help="Minimum points required on both sides of every "
                             "leave-one-fold-out split (a fold's own size, and its "
                             "(K-1)-fold val complement) before a candidate baseline's "
                             "NLL there is trusted enough to base a selection on. Below "
                             "this, ranking ~12 candidate kernels by NLL on a handful of "
                             "points is closer to a coin flip than a real comparison, and "
                             "an occasional spurious pick of a blow-up-prone candidate "
                             "(e.g. dot_product/polynomial kernels, whose NLL is "
                             "unbounded above when misspecified) can drag the averaged "
                             "best_baseline mean above even a single steady fixed kernel "
                             "used for every episode. Effective K per episode is "
                             "min(--n_folds, n_test // min_fold_size); episodes where "
                             "that's < 2 report best_baseline=nan (excluded from its "
                             "mean/std, not silently zero) instead of forcing a low-"
                             "confidence selection.")
    parser.add_argument("--min_test_points", type=int, default=None,
                        help="Floor on eval episodes' test-point count N, so every episode "
                             "is valid for the nested-CV best_baseline selection (see "
                             "--min_fold_size: _select_best_baseline_cv needs n_test // "
                             "min_fold_size >= 2 folds, i.e. n_test >= 2 * min_fold_size, "
                             "or it reports best_baseline=nan for that episode — silently "
                             "excluding it from the summary table's best_baseline mean while "
                             "every other row's mean still includes it). Default: "
                             "2 * --min_fold_size (40 under the argparse defaults). For "
                             "--live_generate (the default), this raises cfg.data.N_min for "
                             "this run only — conf/data/gp_tasks.yaml's own N_min=8, and "
                             "hence training's distribution, is untouched. For --dataset_dir, "
                             "episode sizes are fixed by the pre-built dataset and can't be "
                             "regenerated, so episodes below this floor are skipped instead.")
    parser.add_argument("--oracle_mode",  default=None, choices=["prior", "posterior"],
                        help="How R_star was built for this dataset. Determines whether "
                             "GP-MLE/DKL score the fitted kernel's posterior (conditioned "
                             "on X_train) or its raw prior covariance at X_test. Default: "
                             "read from the checkpoint's own saved training config "
                             "(cfg.data.oracle_mode), falling back to 'prior' if absent.")
    parser.add_argument("--baseline_cache", default="./baseline_cache.pt",
                        help="Path to a cache file storing every classical baseline's fitted "
                             "NLL/correlation results, keyed per-episode. These are the "
                             "expensive, checkpoint-independent part of the comparison; the "
                             "ICL model + oracle are always recomputed fresh since they're "
                             "what actually changes between runs. A cache entry is only "
                             "reused when the episode-generating config and every baseline-"
                             "fitting hyperparameter below match exactly what produced it "
                             "(see eval.baselines.classical.baseline_fingerprint) — otherwise "
                             "it's recomputed and the cache updated in place.")
    parser.add_argument("--no_baseline_cache", action="store_true",
                        help="Disable baseline caching entirely: always recompute, never "
                             "read or write --baseline_cache.")
    parser.add_argument("--refresh_baselines", action="store_true",
                        help="Recompute every baseline even if a matching cache entry "
                             "exists, overwriting it (still writes --baseline_cache unless "
                             "--no_baseline_cache is also given).")
    parser.add_argument("--results_cache", default="./eval_results_partial.json",
                        help="Per-episode SCORED results (every table this script "
                             "prints), written after each episode and reused on a "
                             "restart. --baseline_cache only spares the classical "
                             "fitting; the ICL forward pass, the marginal PIT and the "
                             "nested-CV best_baseline pick lived only in memory until "
                             "the summary tables, so an interrupted run used to re-score "
                             "every episode. Unlike --baseline_cache this key INCLUDES "
                             "the checkpoint and every marginal/CV setting (results "
                             "depend on --ckpt; baseline fits deliberately do not), so "
                             "pointing two different checkpoints at one path makes the "
                             "second ignore the first's entries rather than report them "
                             "— give concurrent runs distinct paths.")
    parser.add_argument("--no_results_cache", action="store_true",
                        help="Disable the scored-results cache: always re-score every "
                             "episode and never read or write --results_cache.")
    parser.add_argument("--gp_val_select", choices=GP_VAL_SELECT_MODES, default="ard",
                        help="Which GP-MLE kernels pick their fit on a held-out 20%% "
                             "split of the episode's training points -- both which step "
                             "within a restart and which of --n_restarts_mle restarts -- "
                             "instead of running to --n_steps_mle and keeping the lowest "
                             "TRAINING loss. 'ard' (default) applies it to the ARD "
                             "kernels only, 'always' to every kernel, 'never' restores "
                             "the pre-v3 behaviour. ARD needs it: it fits one lengthscale "
                             "per input dimension (9 from P=32 points), the LogNormal "
                             "prior does not hold them, and they run away unbounded -- "
                             "ard_rbf ends up +6.06 nats/point worse at 1000 steps than "
                             "at its own optimum (10 steps). The non-ARD kernels do not: "
                             "with 2-3 hyperparameters the prior regularises them "
                             "adequately, so the split is pure data loss (dot_product, a "
                             "linear kernel with nothing to overfit, measures +0.36 "
                             "nats/point WORSE under 'always'). See "
                             "eval.baselines.classical._resolve_val_select for the "
                             "per-kernel numbers.")
    parser.add_argument("--baseline_device", default="cpu", choices=["cpu", "cuda", "auto"],
                        help="Device for fitting the classical baselines (GP-MLE/DKL/"
                             "per_ep_transformer) only — the ICL model, TabICL marginal "
                             "and oracle always run on --device. Defaults to cpu because "
                             "it is measurably FASTER here: episodes have P=32 training "
                             "points, so each of the ~85,000 Adam steps per episode is a "
                             "32x32 Cholesky whose launch latency dwarfs its arithmetic. "
                             "Measured same-seed, same-steps against a TITAN-RTX-class "
                             "GPU, one CPU thread runs GP-MLE rbf in 2.98 s vs 7.50 s and "
                             "DKL rbf in 20.23 s vs 38.54 s, for identical NLLs. 'auto' "
                             "follows --device; pass cuda to reproduce the old behaviour.")
    parser.add_argument("--baseline_workers", type=int, default=0,
                        help="Worker processes for fitting baselines, which are perfectly "
                             "independent across episodes. 0 (default) = auto: one per "
                             "PHYSICAL core this process is allowed to use "
                             "(os.sched_getaffinity, so an OAR allocation is respected; "
                             "hyperthread siblings are not counted twice because they add "
                             "well under a core here), capped at 32. 1 fits serially "
                             "in-process, as before. Only "
                             "used with --baseline_device=cpu: spreading GPU fits across "
                             "processes just contends for one device. Combined with the "
                             "cpu default this is the main speedup — ~2x from the device, "
                             "the rest from cores. Scaling tracks PHYSICAL cores: measured "
                             "78.6 s/episode vs 649 s on GPU (~8x) where the 8 allocated "
                             "logical CPUs were only 4 physical ones, so ask the scheduler "
                             "for real cores, not threads.")
    parser.add_argument("--episode_offset", type=int, default=0,
                        help="Global index of the first live-generated episode (ignored "
                             "unless --live_generate). Lets one episode stream be split "
                             "across OAR array jobs: --n_episodes 100 with "
                             "--episode_offset 0/100/200/300 evaluates the same 400 "
                             "episodes as a single --n_episodes 400 run, bit-identically "
                             "(generating seed, non-composite parity, cache key and "
                             "nested-CV holdout seed all key off the global index — see "
                             "_live_generate_alternating). Give each shard its own "
                             "--baseline_cache; the resulting files share a fingerprint "
                             "and can be merged by concatenating their 'entries' dicts.")
    args = parser.parse_args()

    _set_seed(args.seed)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    print(f"Device: {device}")

    # cfg is the eval-episode-generating config, resolved from --config's own
    # Hydra defaults (model/data groups + _self_) — deliberately NOT the
    # checkpoint's own saved training cfg. Keeping it fixed across --ckpt
    # values is what lets the baseline cache (see fingerprint below) survive
    # switching between checkpoints trained under different cfg.data: the
    # tradeoff is that all checkpoints are now scored against one shared
    # distribution rather than each against its own training distribution.
    # Point --config at a different file (or edit this one) to change it.
    cfg = _load_full_config(args.config)

    # ---- Load ICL model ----
    print(f"\nLoading ICL checkpoint: {args.ckpt}")
    icl_model, icl_cfg = load_copula_model(args.ckpt, config_path=args.config, device=str(device))
    icl_rank = int(icl_cfg.model.rank)
    n_params = sum(p.numel() for p in icl_model.parameters())
    print(f"ICL model parameters: {n_params:,}  rank={icl_rank}")

    # ---- Optionally load a second, frozen TabICL marginal purely to
    # K-fold-PIT each episode's z_train (see --z_train_source's help text) ----
    tabicl_marginal: nn.Module | None = None
    marginal_backend: str | None = (
        args.z_train_source if args.z_train_source not in ("oracle", "tabicl") else None
    )
    marginal_regressor = None
    tabicl_pit_k_folds = DEFAULT_K_FOLDS
    if marginal_backend is not None:
        from eval.spatial.marginal_backends import make_regressor

        tabicl_pit_k_folds = args.tabicl_pit_k_folds or int(
            OmegaConf.select(cfg, "tabicl.pit_k_folds", default=DEFAULT_K_FOLDS)
        )
        print(f"\nBuilding {marginal_backend} marginal for --z_train_source={marginal_backend} "
              f"(k_folds={tabicl_pit_k_folds}, probs_n={args.marginal_probs_n})")
        marginal_regressor = make_regressor(marginal_backend, device=str(device))
    elif args.z_train_source == "tabicl":
        tabicl_ckpt = args.tabicl_ckpt or OmegaConf.select(cfg, "tabicl.ckpt", default=None)
        if not tabicl_ckpt:
            raise ValueError(
                "--z_train_source=tabicl requires a TabICL checkpoint: pass --tabicl_ckpt "
                "or set cfg.tabicl.ckpt in --config."
            )
        tabicl_pit_k_folds = args.tabicl_pit_k_folds or int(
            OmegaConf.select(cfg, "tabicl.pit_k_folds", default=DEFAULT_K_FOLDS)
        )
        print(f"\nLoading frozen TabICL marginal for --z_train_source=tabicl: {tabicl_ckpt} "
              f"(k_folds={tabicl_pit_k_folds})")
        tabicl_marginal = load_tabicl(tabicl_ckpt, str(device))
        configure_tabicl_inference_amp(args.tabicl_amp)
        print(f"Frozen TabICL marginal inference AMP={'on' if args.tabicl_amp else 'off (float32)'}")
    print(f"z_train source (ICL conditioning input): {args.z_train_source}")

    # GP-MLE/DKL must score against the same convention used to build this
    # run's R_star ("prior" ignores training conditioning entirely,
    # "posterior" conditions on X_train) — see classical.fit_and_eval_gpytorch's
    # docstring. Read from cfg (the fixed eval-generating config above), the
    # actual generation config for these episodes; falls back to "prior"
    # (this repo's current datasets all use oracle_mode=prior, unlike
    # data_gen.py's own historical "posterior" default for dataset
    # *generation*).
    oracle_mode = args.oracle_mode or OmegaConf.select(cfg, "data.oracle_mode", default="prior")
    print(f"Oracle mode: {oracle_mode}")

    # GP-MLE/DKL hyperpriors: read the exact LogNormal/Gamma constants these
    # episodes are actually generated with (cfg, not the checkpoint's own
    # training cfg — see cfg's definition above), falling back to
    # classical._DEFAULT_PRIOR_CFG for any missing key.
    data_cfg = OmegaConf.select(cfg, "data", default=None)
    prior_cfg = OmegaConf.to_container(data_cfg) if data_cfg is not None else {}
    print(f"GP-MLE restarts: {args.n_restarts_mle}")
    print(f"DKL restarts: {args.n_restarts_dkl}")

    # Live-generate by default, unless the user points at a fixed dataset
    # with --dataset_dir (see --live_generate's help text).
    live_generate = args.live_generate if args.live_generate is not None else (args.dataset_dir is None)

    n_ep = args.n_episodes
    all_nlls: list[dict[str, float]] = []
    all_y_space_nlls: list[dict[str, dict[str, float]]] = []
    all_total_nlls: list[dict[str, dict[str, float]]] = []
    # Parallel to all_nlls/all_total_nlls (same append order, one entry per
    # evaluated episode) — only populated for --dump_episodes, so a paired
    # comparison across two --ckpt runs sharing --seed (identical episodes,
    # per _live_generate_alternating's determinism) can match rows up by
    # ep_i instead of assuming list order never drifts (e.g. a skipped
    # --dataset_dir episode).
    all_episode_meta: list[dict] = []
    plot_R_dict: dict[str, Tensor] | None = None
    plot_R_oracle: Tensor | None = None
    plot_best_key: str | None = None
    plot_best_R: Tensor | None = None

    # Eval-only floor on episodes' test-point count N (see --min_test_points'
    # help text) — 2 * --min_fold_size by default, the minimum n_test
    # _select_best_baseline_cv needs for >=2 CV folds.
    min_test_points = args.min_test_points if args.min_test_points is not None else 2 * args.min_fold_size

    if live_generate:
        # cfg (the fixed eval-generating config, not icl_cfg) drives live
        # generation — same source already used for prior_cfg above — so
        # every checkpoint evaluated against this --config gets identical
        # episodes for a given seed, regardless of what that checkpoint was
        # itself trained on.
        if cfg.data.N_min < min_test_points:
            print(f"Raising eval episode N_min {cfg.data.N_min} -> {min_test_points} "
                  f"(--min_test_points) for this run only — training's own "
                  "conf/data/gp_tasks.yaml N_min is untouched")
            cfg.data.N_min = min_test_points
            if cfg.data.N_max < cfg.data.N_min:
                cfg.data.N_max = cfg.data.N_min
        print(f"\nLive-generating {n_ep} episodes via generate_gp_batch "
              f"(return_kernel_metadata=True), seed={args.seed}, "
              f"global indices {args.episode_offset}..{args.episode_offset + n_ep - 1}, "
              "alternating every-other episode to a non-composite kernel")
        live_episodes = _live_generate_alternating(
            cfg, n_ep, device, args.seed, offset=args.episode_offset,
        )
    else:
        dataset_dir = args.dataset_dir or cfg.training.dataset_dir
        dataset = CopulaDataset(episode_dir=dataset_dir)
        n_available = len(dataset)
        print(f"\nEvaluating {n_ep} episodes from {dataset_dir} (start={args.episode_idx})")
        print(f"  Dataset size: {n_available} episodes")

    print(f"  GP MLE: {args.n_steps_mle} steps | DKL: {args.n_steps_dkl} steps | "
          f"PerEp: {args.n_steps_per_ep} steps (patience={args.patience_per_ep})")
    print("  [per-episode print legend] 'shared_marginal'/'shared_copula' values "
          "score every method's own correlation matrix against the SAME "
          "ground-truth-standardized z_test, so they rank correlation-structure "
          "quality alone. 'own(...)'/'own_marginal' values instead use each "
          "method's OWN fitted marginal, so its own copula/marginal split is "
          "NOT comparable across methods (a method's own marginal can score "
          "better OR worse than another's regardless of which has the better "
          "overall fit) — only 'total' is a proper scoring rule comparable "
          "across methods with different marginals.")

    # ---- Baseline cache: skip re-fitting GP-MLE/DKL/per_ep_transformer for
    # episodes already scored under an identical generation/fitting config ----
    use_cache = not args.no_baseline_cache
    fingerprint = baseline_fingerprint(
        cfg, live_generate, args.dataset_dir, args.seed, icl_rank, oracle_mode,
        args.n_steps_mle, args.lr_mle, args.n_restarts_mle,
        args.n_steps_dkl, args.lr_dkl, args.n_steps_per_ep, args.patience_per_ep,
        gp_val_select=args.gp_val_select, n_restarts_dkl=args.n_restarts_dkl,
    )
    cache_entries = load_baseline_cache(args.baseline_cache, fingerprint) if use_cache else {}

    # ---- Scored-results cache: the checkpoint-DEPENDENT half of a resume ----
    use_results_cache = not args.no_results_cache
    results_fp = _results_fingerprint(fingerprint, args, tabicl_pit_k_folds)
    results_entries = (
        _load_results_cache(args.results_cache, results_fp) if use_results_cache else {}
    )
    if use_results_cache and args.refresh_baselines:
        # Refit implies rescore: the results were produced from the very fits
        # being thrown away.
        results_entries = {}

    baseline_device = torch.device(
        str(device) if args.baseline_device == "auto" else args.baseline_device
    )
    fit_kwargs = dict(
        icl_rank=icl_rank,
        n_steps_mle=args.n_steps_mle,
        lr_mle=args.lr_mle,
        n_steps_dkl=args.n_steps_dkl,
        lr_dkl=args.lr_dkl,
        n_steps_per_ep=args.n_steps_per_ep,
        patience_per_ep=args.patience_per_ep,
        oracle_mode=oracle_mode,
        prior_cfg=prior_cfg,
        n_restarts_mle=args.n_restarts_mle,
        n_restarts_dkl=args.n_restarts_dkl,
        gp_val_select=args.gp_val_select,
    )

    # ---- Episode plan: every episode that will actually be evaluated, with
    # --dataset_dir's skip rules already applied, so the parallel pre-fit pass
    # below and the evaluation loop after it agree exactly on which episodes
    # exist and what each one's cache key is. ----
    episode_plan: list[tuple[int, int, str, dict, int]] = []
    for local_i in range(n_ep):
        if live_generate:
            # Global index (== local_i unless --episode_offset): what the
            # generating seed, the cache key and the nested-CV holdout seed
            # all key off, so shards of one episode stream agree.
            ep_i = args.episode_offset + local_i
            ep = live_episodes[local_i]
        else:
            ep_i = args.episode_idx + local_i
            if ep_i >= n_available:
                print(f"  [ep {ep_i}] index out of range ({n_available} available), skipping")
                continue
            ep = dataset[ep_i]
            n_test = ep["z_test"].shape[0]
            if n_test < min_test_points:
                print(f"  [ep {ep_i}] only {n_test} test points (< --min_test_points="
                      f"{min_test_points}), skipping — best_baseline needs enough for "
                      ">=2 nested-CV folds")
                continue
        cache_key = episode_cache_key(live_generate, args.dataset_dir, args.seed, ep_i)
        episode_plan.append(
            (local_i, ep_i, cache_key, ep, _baseline_fit_seed(args.seed, cache_key))
        )

    # ---- Parallel pre-fit of the expensive, checkpoint-independent half ----
    # Everything the evaluation loop needs that does NOT depend on --ckpt is
    # fitted here, across processes, so the loop itself only does the cheap
    # GPU work. See _prefit_baselines_parallel.
    n_workers = args.baseline_workers
    try:
        _aff = os.sched_getaffinity(0)
    except AttributeError:  # pragma: no cover - non-Linux
        _aff = set(range(os.cpu_count() or 1))
    n_physical = _count_physical_cores(_aff)
    # One worker per PHYSICAL core, not per logical CPU: these fits are
    # compute-bound enough that a hyperthread sibling adds well under a full
    # core, so the measured "~2x from the device x N from cores" speedup scales
    # with N = physical. Measured on an allocation of 8 logical CPUs that were
    # only 4 physical cores (Xeon E5-2623 v3): 78.6 s/episode against 649 s on
    # a GPU — ~8x, where 8 real cores would have given roughly twice that.
    # The cap only bounds memory and process churn on a very large allocation;
    # raise it with --baseline_workers if you have the cores (a fixed low cap
    # silently wasted most of a big node -- observed on a 24-physical-core
    # allocation where an earlier cap of 8 left two thirds of it idle).
    if n_workers <= 0:
        n_workers = max(1, min(32, n_physical or len(_aff)))
    if n_physical and n_physical < len(_aff):
        # Worth saying out loud, because "8 cores" from the scheduler looks
        # like 8 until the run comes in at half the expected rate.
        print(f"  [prefit] note: the {len(_aff)} allocated logical CPUs are only "
              f"{n_physical} physical core(s) ({len(_aff) // n_physical} threads each) — "
              "expect scaling closer to the physical count; request more cores "
              "from the scheduler for a proportionally faster run")
    if baseline_device.type != "cpu" and n_workers > 1:
        print(f"  [prefit] --baseline_device={baseline_device.type}: forcing "
              "--baseline_workers=1 (parallel processes would just contend for one GPU)")
        n_workers = 1

    # Every episode whose baselines this run already has: reusable cache
    # entries up front, plus whatever the pool fits below. The loop then has a
    # single question to ask per episode — is it in here? — instead of
    # re-deciding cache validity and separately remembering that a pooled fit
    # counts even when the on-disk cache is disabled.
    fitted: dict[str, dict] = {}
    if use_cache and not args.refresh_baselines:
        for _, ep_i, cache_key, _, _ in episode_plan:
            entry = _valid_cached_entry(cache_entries, cache_key, ep_i)
            if entry is not None:
                fitted[cache_key] = entry

    pending = [
        (cache_key, fit_seed, ep)
        for _, _, cache_key, ep, fit_seed in episode_plan
        if cache_key not in fitted
    ]
    print(f"\nBaselines: {len(fitted)} episode(s) reused from cache, "
          f"{len(pending)} to fit on {baseline_device.type}"
          + (f" across {n_workers} worker process(es)" if n_workers > 1 else " serially"))
    if pending and n_workers > 1:
        _prefit_baselines_parallel(
            pending, fit_kwargs, n_workers, args.baseline_cache, fingerprint,
            fitted, use_cache,
        )

    for local_i, ep_i, cache_key, ep, fit_seed in episode_plan:
        entry = fitted.get(cache_key)
        if entry is not None:
            baseline_nlls   = entry["nlls"]
            baseline_R      = {k: v.to(device) for k, v in entry["R_dict"].items()}
            baseline_y_nlls = entry["y_nlls"]
        else:
            # Serial path: --baseline_workers=1, or a GPU --baseline_device.
            # Fitting happens on baseline_device, but everything downstream
            # (the nested-CV selection, the plot) expects R on the evaluation
            # device, so move the results back.
            #
            # eval_baselines_episode reseeds the global RNG from fit_seed (see
            # its docstring). In a worker process that is harmless, but here it
            # would shift the stream every later episode's ICL-side work draws
            # from — so snapshot and restore it, leaving the rest of the loop
            # bit-identical to a run with no baseline fitting in it at all.
            #
            # One BLAS thread, matching _fit_baselines_task, so a fit does not
            # depend on how many cores the machine happened to offer. Thread
            # count changes floating-point reduction order, and these fits are
            # ill-conditioned enough to amplify that: measured on 4 episodes,
            # 8-thread and 1-thread runs agree to ~1e-5 relative on a typical
            # baseline but differ by 5.4e4 nats on gp_prior_rbf, whose NLL runs
            # to ~6e4 on a near-singular R. Pinned to 1, the serial and pooled
            # paths come out bit-identical across all 284 per-episode numbers.
            with _snapshot_rng_and_threads(
                cpu_threads=1 if baseline_device.type == "cpu" else None
            ):
                baseline_nlls, baseline_R, baseline_y_nlls = eval_baselines_episode(
                    ep={k: (v.to(baseline_device) if isinstance(v, Tensor) else v)
                        for k, v in ep.items()},
                    device=baseline_device,
                    fit_seed=fit_seed,
                    **fit_kwargs,
                )
            baseline_R = {k: v.to(device) for k, v in baseline_R.items()}
            if use_cache:
                save_baseline_entry(args.baseline_cache, fingerprint, cache_key, {
                    "nlls": baseline_nlls,
                    "R_dict": {k: v.cpu() for k, v in baseline_R.items()},
                    "y_nlls": baseline_y_nlls,
                })

        # ---- Already-scored episode? Reuse and skip the GPU work ----
        # The baseline cache above only spares the fitting; the ICL forward
        # pass, the marginal PIT and the nested-CV selection below used to be
        # redone on every restart because their results lived purely in the
        # in-memory accumulators until the summary tables at the end of main().
        # An interrupted run therefore resumed with correct baselines but
        # re-scored every episode from scratch.
        want_plot = (local_i == args.plot_episode)
        res_cached = results_entries.get(str(ep_i)) if use_results_cache else None
        if res_cached is not None and not want_plot:
            # want_plot is excluded because the corr_grid figure needs this
            # episode's live R matrices, which the results cache does not keep
            # (they are large, and only one episode is ever plotted).
            all_y_space_nlls.append(res_cached["y_space_nlls"])
            all_total_nlls.append(res_cached["total_nlls"])
            all_episode_meta.append(res_cached["meta"])
            all_nlls.append(res_cached["nlls"])
            print(f"  ep {ep_i:04d}: reusing scored results (--results_cache)")
            continue

        marginal_pit = None
        if tabicl_marginal is not None or marginal_regressor is not None:
            marginal_pit = _marginal_pit(
                ep=ep, tabicl_marginal=tabicl_marginal, k_folds=tabicl_pit_k_folds, device=device,
                marginal_backend=marginal_backend, marginal_regressor=marginal_regressor,
                marginal_probs_n=args.marginal_probs_n, seed=ep_i,
            )
            if marginal_pit is None:
                print(f"  [ep {ep_i}] fewer than 2 training points — "
                      "falling back to oracle z_train for this episode")

        # ---- Marginal + Zero Mean GP baselines: fit directly on the real
        # marginal's z_train (see _eval_zero_mean_gp_baselines' docstring).
        # Not part of the checkpoint-independent baseline_cache above (it
        # depends on --z_train_source/--tabicl_ckpt, which that cache has no
        # way to key on), so it is fit fresh here every run; only 2 kernels
        # at a reduced step/restart budget (--n_steps_zeromean_gp/
        # --n_restarts_zeromean_gp) keeps that cheap. A no-op whenever there
        # is no real marginal for this episode (--z_train_source=oracle, or
        # marginal_pit is None). Seeded off this episode's own fit_seed
        # (independent of --baseline_workers/cache-hit stream position), the
        # global RNG snapshot/restored around it, and pinned to one CPU
        # thread — all three matching the serial classical-baseline fit
        # above (same underlying fit_and_eval_gpytorch Cholesky machinery,
        # same thread-count-dependent floating-point reduction order), since
        # eval_baselines_episode's own fit_seed reseed makes every OTHER
        # baseline's reproducibility depend only on that reseed, not on what
        # ran before it here.
        if args.zeromean_gp and marginal_pit is not None:
            with _snapshot_rng_and_threads(
                seed=fit_seed + 1,
                cpu_threads=1 if baseline_device.type == "cpu" else None,
            ):
                zm_nlls, zm_R, zm_y_nlls = _eval_zero_mean_gp_baselines(
                    ep=ep, marginal_pit=marginal_pit, device=baseline_device,
                    n_steps=args.n_steps_zeromean_gp, lr=args.lr_zeromean_gp,
                    n_restarts=args.n_restarts_zeromean_gp,
                    oracle_mode=oracle_mode, prior_cfg=prior_cfg,
                )
            baseline_nlls = {**baseline_nlls, **zm_nlls}
            baseline_R = {**baseline_R, **{k: v.to(device) for k, v in zm_R.items()}}
            baseline_y_nlls = {**baseline_y_nlls, **zm_y_nlls}

        icl_nlls, icl_R, R_oracle, y_space_nlls, icl_y_parts = _eval_icl_episode(
            ep=ep, icl_model=icl_model, device=device, marginal_pit=marginal_pit,
        )
        all_y_space_nlls.append(y_space_nlls)

        n_test = ep["z_test"].shape[0]
        # oracle_prior/oracle_posterior: gp_analytical_posterior's raw-sum
        # {"total","marginal","copula"} dicts, divided by n_test here to put
        # them on the same per-point footing as every other row (baseline_y_nlls
        # and icl_y_parts are already per-point via gp_oracle_y_nll/y_space_nll)
        # — this table only, _print_y_space_oracle's own numbers stay raw.
        total_nlls = {
            **baseline_y_nlls,
            "icl": icl_y_parts,
            "oracle_prior": {k: v / n_test for k, v in y_space_nlls["prior"].items()},
            "oracle_posterior": {k: v / n_test for k, v in y_space_nlls["posterior"].items()},
        }
        all_total_nlls.append(total_nlls)
        all_episode_meta.append({
            "ep_i": ep_i,
            "n_test": n_test,
            "kernel": _kernel_composition_label(ep),
        })

        nlls   = {**baseline_nlls, **icl_nlls}
        R_dict = {**baseline_R, **icl_R}

        icl_nll = nlls.get("icl", float("nan"))
        ora_nll = nlls.get("oracle", float("nan"))
        ranked_baselines = sorted(
            ((k, v) for k, v in nlls.items() if k not in _NON_FITTED_EXCLUDED),
            key=lambda kv: kv[1],
        )
        top5 = ranked_baselines[:5]
        # Per-episode best fitted baseline's NLL, selected via nested
        # (leave-one-fold-out) CV over this episode's test points (see
        # _select_best_baseline_cv) — averaging this across episodes (see
        # _print_table's "Best-of-Baselines" row) is a tighter,
        # per-episode-optimal reference than any single baseline's own
        # average, so its gap to ICL's mean is the real "how much is ICL
        # leaving on the table vs. always picking the best baseline" number,
        # without the winner having been picked by peeking at the same
        # z_test it's scored on.
        holdout_seed = (args.seed * 1_000_003 + ep_i) % (2 ** 31 - 1)
        best_nll, mode_key, fold_details = _select_best_baseline_cv(
            baseline_R, ep["z_test"].to(device), args.n_folds, args.min_fold_size, holdout_seed,
        )
        nlls["best_baseline"] = best_nll
        all_nlls.append(nlls)

        # Persist this episode's scored results immediately. These are a few
        # dozen floats, so unlike the baseline cache there is no reason to
        # batch the writes: a run killed at any point resumes having lost at
        # most the episode currently in flight.
        if use_results_cache:
            results_entries[str(ep_i)] = _jsonable({
                "nlls": nlls,
                "total_nlls": total_nlls,
                "y_space_nlls": y_space_nlls,
                "meta": all_episode_meta[-1],
            })
            _save_results_cache(args.results_cache, results_fp, results_entries)

        if local_i == args.plot_episode:
            plot_R_dict   = R_dict
            plot_R_oracle = R_oracle
            if mode_key is not None:
                plot_best_key = mode_key
                plot_best_R   = R_dict[mode_key]

        print(f"  ep {ep_i:04d}: kernel={_kernel_composition_label(ep)}")
        # Every number on this line is per-point (nats/point) and scored
        # against the SHARED ground-truth marginal's z_test — icl/oracle_prior
        # here are the same z-space-copula-only quantity as nlls["icl"]/
        # nlls["oracle"]. GP-oracle-y-space-total pulls from total_nlls (not
        # y_space_nlls directly) so it's on that same per-point footing —
        # y_space_nlls itself is the raw (unnormalized) sum gp_analytical_posterior
        # returns, which _print_y_space_oracle's aggregate table prints as-is.
        print(f"    icl(shared_marginal)={icl_nll:.4f}  "
              f"oracle_prior(shared_marginal, z-space copula)={ora_nll:.4f}  "
              f"GP-oracle-y-space-total(prior={total_nlls['oracle_prior']['total']:.4f}, "
              f"posterior={total_nlls['oracle_posterior']['total']:.4f})")
        print(f"    total Y-space NLL, own marginal (total = marginal + copula): "
              f"icl=(cop={total_nlls['icl']['copula']:.4f}, "
              f"marg={total_nlls['icl']['marginal']:.4f}, "
              f"tot={total_nlls['icl']['total']:.4f})  "
              f"oracle_prior=(cop={total_nlls['oracle_prior']['copula']:.4f}, "
              f"marg={total_nlls['oracle_prior']['marginal']:.4f}, "
              f"tot={total_nlls['oracle_prior']['total']:.4f})  "
              f"oracle_posterior=(cop={total_nlls['oracle_posterior']['copula']:.4f}, "
              f"marg={total_nlls['oracle_posterior']['marginal']:.4f}, "
              f"tot={total_nlls['oracle_posterior']['total']:.4f})")
        if fold_details:
            fold_summary = ", ".join(
                f"fold{fd['fold']}={_METHOD_LABELS.get(fd['selected'], fd['selected'])}"
                for fd in fold_details
            )
            print(f"    best_baseline (nested {len(fold_details)}-fold CV, "
                  f"pooled test NLL)={best_nll:.4f}")
            print(f"      per-fold picks: {fold_summary}")
        else:
            print("    best_baseline: unavailable (too few test points for nested CV)")
        print("    top-5 baselines (ranked by shared-ground-truth-marginal copula "
              "NLL, diagnostic only — not the selection used for best_baseline "
              "above; 'own' columns are this baseline's OWN fitted marginal, "
              "NOT the same copula quantity as the ranking column — see "
              "eval_baselines_episode's docstring):")
        for key, val in top5:
            own = total_nlls.get(key, _NAN_PARTS)
            print(f"      {_METHOD_LABELS.get(key, key):<28}"
                  f"shared_copula={val:.4f}  "
                  f"own(cop={own['copula']:.4f}, marg={own['marginal']:.4f}, "
                  f"tot={own['total']:.4f})")

    if not all_nlls:
        print("No episodes evaluated successfully.")
        return

    _print_table(all_nlls, z_train_source=args.z_train_source)
    _print_y_space_oracle(all_y_space_nlls)
    _print_total_nll_table(all_total_nlls, z_train_source=args.z_train_source)
    _print_rank_table(
        all_nlls, _RANK_KEYS,
        title="Method rank — z-space copula NLL (shared ground-truth marginal)",
        z_train_source=args.z_train_source,
    )
    total_only = [
        {k: m.get(k, _NAN_PARTS).get("total", float("nan")) for k, _ in _TOTAL_NLL_ORDER}
        for m in all_total_nlls
    ]
    _print_rank_table(
        total_only, _TOTAL_NLL_ORDER,
        title="Method rank — total NLL, Y-space (own marginal per method)",
        z_train_source=args.z_train_source,
    )

    if args.dump_episodes:
        dump = {
            "ckpt": args.ckpt,
            "config": args.config,
            "seed": args.seed,
            "live_generate": live_generate,
            "dataset_dir": args.dataset_dir,
            "z_train_source": args.z_train_source,
            "episodes": [
                {**meta, "nlls": nlls, "total_nlls": total_nlls}
                for meta, nlls, total_nlls in zip(all_episode_meta, all_nlls, all_total_nlls)
            ],
        }
        os.makedirs(os.path.dirname(args.dump_episodes) or ".", exist_ok=True)
        with open(args.dump_episodes, "w") as f:
            json.dump(dump, f, indent=2)
        print(f"Dumped {len(dump['episodes'])} per-episode NLLs to: {args.dump_episodes}")

    # ---- Correlation heatmap ----
    if plot_R_dict is not None and plot_R_oracle is not None:
        import matplotlib
        matplotlib.use("Agg")

        os.makedirs(args.out_dir, exist_ok=True)
        # Exclude oracle from estimators dict (it's passed separately). Move
        # icl to the end and insert the best-performing fitted baseline for
        # this episode (lowest NLL, excluding icl/oracle/independence/
        # gp_prior_rbf — same ranking as the "top-5 baselines" console print
        # above) right before it, so oracle / best-baseline / icl sit next to
        # each other for a quick visual comparison instead of having to scan
        # all the individual baseline panels.
        estimators = {k: v for k, v in plot_R_dict.items() if k != "oracle"}
        icl_panel = estimators.pop("icl", None)
        if plot_best_key is not None:
            best_label = f"best_baseline ({_METHOD_LABELS.get(plot_best_key, plot_best_key)})"
            estimators[best_label] = plot_best_R
        if icl_panel is not None:
            estimators["icl"] = icl_panel
        fig = plot_corr_grid(
            estimators=estimators,
            oracle_R=plot_R_oracle,
            title=f"Correlation estimators — episode {args.episode_idx + args.plot_episode}",
        )
        out_path = os.path.join(args.out_dir, f"corr_grid_ep{args.plot_episode}.png")
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        print(f"Saved corr_grid to: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
