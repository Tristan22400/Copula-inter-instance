"""era5_episodes.py — real-ERA5 episodes in the exact dict shape
eval/runners/eval_checkpoint.py evaluates, so a checkpoint can be scored
against the same classical-baseline table on real worldwide 2m-temperature
data instead of synthetic GP draws.

This is the evaluation-side sibling of src/era5_live_dataset.py (which
serves the *training* loop): same corpus (eval/data/era5_global_corpus.py),
same frozen-TabICL K-fold PIT convention (src/pit.py::run_pit /
run_pit_batched via era5_live_dataset's `_pit_group`, including its
log-Jacobian correction back to raw nats), but materialized eagerly as
a finite, seed-reproducible LIST rather than an infinite IterableDataset —
eval_checkpoint.py needs to index episodes by a global index, hand them to a
worker pool, and key a disk cache on them.

What is and is not available on real data
-----------------------------------------
There is no generating kernel behind ERA5, so an episode built here
deliberately carries NO ``R_star``/``Sigma_star``/``mu_star``/``sigma_star``
and no kernel metadata. Everything in eval_checkpoint.py that reads those
(the ``oracle`` row of the z-space table, the analytic GP prior/posterior
Y-space rows) is therefore structurally unavailable and prints as n/a —
see eval_checkpoint.py's ``--era5`` help text.

What *is* available, and what makes the comparison valid, is ``z_test``:
every method in the shared-marginal copula table must be scored against one
literally-identical ``z_test`` (see classical.assert_shared_z_test). On
synthetic episodes that shared reference is the oracle's exact PIT residual;
here it is the frozen TabICL K-fold PIT, computed ONCE per episode in this
module and then shared by the ICL model, the GP-MLE/DKL fits and
per_ep_transformer alike. That keeps the table's cross-method ranking
meaningful — only the correlation matrix R differs between rows — while
changing what "the marginal" means from "ground truth" to "the frozen
TabICL marginal every method is held to". The Y-space total-NLL table is
unaffected by that choice: there every method supplies its own predictive
density and is scored at the same real ``y_test``, which is a proper
scoring rule on real data exactly as it is on synthetic.

Geometry
--------
Two draw modes, both centred area-uniformly over the whole globe:

  * fixed shape (default) — ``grid_size``/``n_context`` are exact, so every
    episode shares P = n_context and N = grid_size**2 - n_context. Region,
    day and box width still vary per episode. Homogeneous P/N is what lets
    the PIT run through ``run_pit_batched`` in groups (a ~40x cut in TabICL
    invocations, same trade LiveERA5Dataset's ``group_size`` makes), and it
    makes per-episode NLLs directly averageable without an N-weighting
    caveat.
  * varying geometry (``vary_geometry=True``) — grid_size/n_context_frac are
    drawn per episode from ranges, matching ``era5_live``'s training-time
    distribution. PIT then runs one episode at a time.

d_x is 6 either way: lon, lat, and the four
eval/data/fetch_era5_static.py::STATIC_VARS surface fields.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pit import normalize_targets

from eval.data.era5_global_corpus import GlobalERA5Corpus

__all__ = ["build_era5_eval_episodes", "era5_episode_fingerprint", "DEFAULT_CORPUS_DIR"]

# Held-out calendar year, disjoint from the multi-decade corpus
# era5_live.corpus_dir points training/finetuning at — the right default for
# *evaluating* a checkpoint, including one finetuned on ERA5.
DEFAULT_CORPUS_DIR = os.path.join(_HERE, "cache", "era5_global_val")


def _episode_rng(seed: int, ep_i: int) -> np.random.Generator:
    """Per-episode generator seeded from (seed, GLOBAL index), so episode k is
    bit-identical whether it came from one long run or from a shard starting
    at an offset — the same determinism contract eval_checkpoint.py's
    _live_generate_alternating gives synthetic episodes, and what lets the
    baseline cache be keyed on (seed, ep_i) alone."""
    return np.random.default_rng([int(seed), int(ep_i)])


def _draw_episode(
    corpus: GlobalERA5Corpus,
    rng: np.random.Generator,
    *,
    vary_geometry: bool,
    grid_size: int,
    n_context: int,
    box_deg_range: tuple[float, float],
    grid_size_range: tuple[int, int],
    n_context_frac_range: tuple[float, float],
    max_redraws: int = 200,
) -> dict:
    """One accepted geographic draw. `sample_episode_fixed_shape` returns None
    whenever the sampled box doesn't contain a full grid_size x grid_size
    block of native 0.25deg points (small boxes at any latitude, or any box
    near the poles along longitude) — redraw from the SAME generator rather
    than clipping, so every accepted episode really has the requested P/N."""
    for _ in range(max_redraws):
        if vary_geometry:
            ep = corpus.sample_episode(
                rng,
                grid_size_range=grid_size_range,
                box_deg_range=box_deg_range,
                n_context_frac_range=n_context_frac_range,
            )
        else:
            ep = corpus.sample_episode_fixed_shape(
                rng,
                grid_size=grid_size,
                box_deg_range=box_deg_range,
                n_context=n_context,
            )
        if ep is not None:
            return ep
    raise RuntimeError(
        f"No valid ERA5 episode after {max_redraws} redraws at grid_size={grid_size}, "
        f"box_deg_range={box_deg_range} — the box is too small to hold a "
        f"{grid_size}x{grid_size} block of 0.25deg points; raise --era5_box_deg_min "
        "or lower --era5_grid_size."
    )


def build_era5_eval_episodes(
    corpus_dir: str,
    n_episodes: int,
    *,
    seed: int,
    offset: int = 0,
    grid_size: int = 24,
    n_context: int = 30,
    box_deg_range: tuple[float, float] = (5.0, 25.0),
    vary_geometry: bool = False,
    grid_size_range: tuple[int, int] = (8, 28),
    n_context_frac_range: tuple[float, float] = (0.05, 0.4),
    tabicl_model=None,
    k_folds: int = 10,
    device: str | torch.device = "cpu",
    pit_group_size: int = 8,
    marginal_backend: Optional[str] = None,
    marginal_regressor=None,
    marginal_probs_n: int = 99,
    max_months: Optional[int] = None,
    lazy: Optional[bool] = None,
    standardize_y: bool = True,
    autoregressive: bool = False,
    ar_order: str = "random",
    ar_conditioning: str = "teacher_forcing",
    ar_max_context: Optional[int] = None,
    ar_n_episodes: Optional[int] = None,
    verbose: bool = True,
) -> list[dict]:
    """Materialize `n_episodes` real-ERA5 evaluation episodes, PIT'd and ready
    for eval_checkpoint.py's episode loop.

    Global episode indices run `offset .. offset + n_episodes - 1`; each one's
    geography is a pure function of (seed, its global index), so shards of one
    stream agree with a single long run (see _episode_rng).

    Returns a list of dicts carrying, per episode:
        x_norm_train (P, 6) / x_norm_test (N, 6)  — lon, lat + 4 static fields,
                                                    jointly normalized
        y_train (P,) / y_test (N,)                — 2m temperature, z-scored per
                                                    episode by y_train's own
                                                    mean/std unless
                                                    standardize_y=False (see
                                                    below)
        z_train (P,) / z_test (N,)                — frozen-TabICL K-fold PIT
        log_pdf_test (N,)                         — that marginal's log-density
                                                    at y_test, RAW nats
        ar_log_pdf (N,)                           — only under autoregressive=
                                                    True: the SAME marginal's
                                                    chain-rule log-density at
                                                    y_test, RAW nats (see
                                                    eval/baselines/
                                                    autoregressive.py)
        era5_meta                                 — {"ep_i", "grid_size", "P",
                                                    "N", "lat_bounds",
                                                    "lon_bounds"} for reporting
    and deliberately nothing else (no oracle keys — see the module docstring).

    autoregressive (default False) additionally runs
    eval.baselines.autoregressive.autoregressive_log_pdf over each group,
    giving every episode an ``ar_log_pdf`` alongside its one-shot
    ``log_pdf_test``. It runs HERE, next to the PIT, for two reasons: the
    marginal is loaded and on-device at exactly this point (eval_checkpoint.py
    releases it immediately afterwards, before the multi-hour CPU baseline
    pass), and the chain advances a whole group of episodes per forward the
    same way ``run_pit_batched`` does — at the --era5 defaults (P=30, N=546,
    group 8) that is a measured 13.4 ms per forward against 21.4 ms for a
    group of 8, i.e. ~4x.

    Absolute cost is GPU-bound and varies with the card by more than a factor
    of two: measured 1.5 s/episode on an RTX PRO 6000 Blackwell and 3.6
    s/episode on an RTX A5000 (~10 and ~24 min respectively for 400
    episodes). ar_order/ar_conditioning/ar_max_context are passed straight
    through; ar_n_episodes caps it to the first N episodes of the run (None =
    all), for when the chain is not worth its wall time on every episode.

    standardize_y (default True) z-scores y by the SAME per-episode statistics
    pit.normalize_targets uses (y_train's mean and std, applied to y_test too).
    This matters on real data and does not on synthetic: ERA5 targets are
    absolute Kelvin (~280), while eval/baselines/classical.py fits its GP-MLE
    and DKL baselines under LogNormal/Gamma hyperpriors calibrated for the
    O(1) targets data_gen.py draws. Left raw, those baselines are handicapped
    by a pure units mismatch — measured on a 2-episode probe, their own-marginal
    Y-space NLL sat near 100 nats/point — while the ICL model under test is NOT,
    because its TabICL PIT already normalizes internally. Standardizing puts
    every method in the same well-conditioned space; `y_log_std` in era5_meta
    is log(std), the per-point constant that converts any NLL computed in that
    space back to raw-Kelvin nats (log p_raw(y) = log p_scaled(y_scaled) -
    log(std)), which eval_checkpoint.py adds back to the baselines' marginal
    and total columns so the printed table is in raw Kelvin nats throughout.
    The copula column is scale-invariant and needs no correction.

    The PIT itself always runs on RAW y (normalize_targets does its own scaling
    and its log_pdf_test already comes back Jacobian-corrected to raw nats), so
    this flag changes nothing about z_train/z_test/log_pdf_test.
    """
    from era5_live_dataset import _pit_group

    from eval.baselines.autoregressive import autoregressive_log_pdf

    if tabicl_model is None and marginal_backend is None:
        raise ValueError(
            "ERA5 episodes need a real marginal to PIT with: pass tabicl_model "
            "(--z_train_source=tabicl) or a marginal_backend + marginal_regressor. "
            "There is no analytic/oracle PIT on real data."
        )
    if autoregressive and marginal_backend is not None:
        # The chain needs a module it can call one query at a time with a
        # growing context; the exaone/tabpfn/tabldm backends are reached only
        # through _BATCHED_MARGINAL_BACKENDS' fit-then-predict-a-whole-block
        # interface (see src/marginal_backbones.py), which has no such entry
        # point. Fail here rather than silently dropping the row.
        raise NotImplementedError(
            f"--autoregressive is implemented for the TabICL marginal only, not "
            f"for backend {marginal_backend!r}. Re-run with --no-autoregressive, "
            f"or with the default --z_train_source=tabicl."
        )

    if verbose:
        print(f"  [era5] loading corpus from {corpus_dir}")
    # `lazy` is forwarded rather than left to GlobalERA5Corpus's own
    # len(paths) > 60 heuristic, because max_months is applied to the file
    # list BEFORE that heuristic runs: capping a 396-month corpus to 60
    # flips it from memory-mapped to fully eager and *raises* resident RAM
    # (~7.5 GB for 60 global months) instead of lowering it.
    corpus = GlobalERA5Corpus(corpus_dir, max_months=max_months, lazy=lazy)
    if verbose:
        print(f"  [era5] corpus: {corpus.n_days_total} daily snapshots, "
              f"native grid {len(corpus.lat)}x{len(corpus.lon)}")

    # ---- Geographic/target draws (CPU, cheap) ----
    raw: list[dict] = []
    for local_i in range(n_episodes):
        ep_i = offset + local_i
        drawn = _draw_episode(
            corpus, _episode_rng(seed, ep_i),
            vary_geometry=vary_geometry, grid_size=grid_size, n_context=n_context,
            box_deg_range=box_deg_range, grid_size_range=grid_size_range,
            n_context_frac_range=n_context_frac_range,
        )
        raw.append({
            "ep_i": ep_i,
            "x_norm_train": torch.from_numpy(drawn["x_norm_train"]),
            "x_norm_test": torch.from_numpy(drawn["x_norm_test"]),
            "y_train": torch.from_numpy(drawn["y_train"]),
            "y_test": torch.from_numpy(drawn["y_test"]),
            # Both corpus samplers return these unconditionally (this
            # module's change to sample_episode_fixed_shape is what made that
            # true), so index rather than .get() -- a missing key is a bug in
            # the corpus, not a case to paper over with a fallback.
            "grid_size": int(drawn["grid_size"]),
            "lat_bounds": drawn["lat_bounds"],
            "lon_bounds": drawn["lon_bounds"],
        })

    # ---- PIT (GPU, the expensive half) ----
    # Fixed-shape episodes all share P/N, so they go through run_pit_batched in
    # groups; varying geometry falls back to one run_pit call per episode.
    dev = torch.device(device)
    group = 1 if vary_geometry else max(1, int(pit_group_size))
    episodes: list[dict] = []
    for chunk_i, start in enumerate(range(0, len(raw), group)):
        chunk = raw[start:start + group]
        # One code path for every chunk size: _pit_group handles B=1 (it
        # reduces with dim=-1/keepdim, and tests/test_pit_batched.py's
        # test_run_pit_batched_b1_matches_run_pit pins B=1 to run_pit within
        # 1e-5). Branching on len(chunk)==1 meant the vary_geometry path ran
        # code the default fixed-geometry run never exercised.
        out = _pit_group(
            torch.stack([r["x_norm_train"] for r in chunk]).to(dev),
            torch.stack([r["y_train"] for r in chunk]).to(dev),
            torch.stack([r["x_norm_test"] for r in chunk]).to(dev),
            torch.stack([r["y_test"] for r in chunk]).to(dev),
            tabicl_model, k_folds,
            marginal_backend=marginal_backend, marginal_regressor=marginal_regressor,
            marginal_probs_n=marginal_probs_n, seed=chunk[0]["ep_i"],
        )
        pits = [
            None if out is None else {k: out[k][b] for k in
                                      ("z_train", "z_test", "log_pdf_test")}
            for b in range(len(chunk))
        ]

        # ---- Chain-rule pass, same group, same marginal, same RAW y --------
        # Fed r["y_train"]/r["y_test"] (raw) and not the possibly-standardized
        # y stored on the episode below, exactly as the PIT above was: both do
        # their own normalize_targets-equivalent scaling and both undo it, so
        # both land in raw nats and ar_log_pdf is directly differenceable
        # against log_pdf_test regardless of `standardize_y`.
        ar_log_pdf: list[Optional[torch.Tensor]] = [None] * len(chunk)
        n_ar = (
            len(chunk) if ar_n_episodes is None
            else max(0, min(len(chunk), int(ar_n_episodes) - start))
        )
        if autoregressive and out is not None and n_ar > 0:
            sub = chunk[:n_ar]
            ar_out = autoregressive_log_pdf(
                tabicl_model,
                torch.stack([r["x_norm_train"] for r in sub]).to(dev),
                torch.stack([r["y_train"] for r in sub]).to(dev),
                torch.stack([r["x_norm_test"] for r in sub]).to(dev),
                torch.stack([r["y_test"] for r in sub]).to(dev),
                order=ar_order, conditioning=ar_conditioning,
                max_context=ar_max_context, seed=seed,
                episode_indices=[r["ep_i"] for r in sub],
            )
            for b in range(len(sub)):
                ar_log_pdf[b] = ar_out["log_pdf"][b]

        for r, pit, ar in zip(chunk, pits, ar_log_pdf):
            if pit is None:
                print(f"  [era5] ep {r['ep_i']}: PIT unavailable (too little context), skipping")
                continue
            P = int(r["x_norm_train"].shape[0])
            N = int(r["x_norm_test"].shape[0])
            # The SAME call the PIT above made internally -- not a
            # re-derivation of it. y_log_std is only the exact Jacobian for
            # the y the baselines are fitted on while both use one
            # convention, and sharing the function is what guarantees that;
            # recomputing the mean/std here would let the two drift apart the
            # next time normalize_targets changes its floor or its unbiased
            # flag.
            y_tr_scaled, y_te_scaled, _, y_std = normalize_targets(
                r["y_train"], r["y_test"]
            )
            if standardize_y:
                y_tr, y_te = y_tr_scaled, y_te_scaled
                y_log_std = float(y_std.log())
            else:
                y_tr, y_te = r["y_train"], r["y_test"]
                y_log_std = 0.0
            episode = {
                "x_norm_train": r["x_norm_train"],
                "x_norm_test": r["x_norm_test"],
                "y_train": y_tr,
                "y_test": y_te,
                "z_train": pit["z_train"].detach().float().cpu(),
                "z_test": pit["z_test"].detach().float().cpu(),
                "log_pdf_test": pit["log_pdf_test"].detach().float().cpu(),
                # Per-point nats to ADD to any NLL computed on this episode's
                # y to express it in raw units, 0.0 when the targets were left
                # raw. Deliberately TOP-LEVEL, not inside era5_meta: it is
                # numerics every consumer of an episode needs, not ERA5
                # reporting metadata, so a consumer reads ep.get("y_log_std",
                # 0.0) and never has to know which source built the episode.
                "y_log_std": y_log_std,
                # The PIT this episode already carries, so the runner reuses
                # it instead of branching on the episode source to decide
                # whether to recompute one (same duck-typing as "R_star" in ep).
                "marginal_pit": {
                    "z_train": pit["z_train"].detach().float().cpu(),
                    "z_test": pit["z_test"].detach().float().cpu(),
                    "log_pdf_test": pit["log_pdf_test"].detach().float().cpu(),
                },
                "era5_meta": {
                    "ep_i": r["ep_i"],
                    "grid_size": r["grid_size"],
                    "P": P,
                    "N": N,
                    "lat_bounds": r["lat_bounds"],
                    "lon_bounds": r["lon_bounds"],
                },
            }
            # Absent, not nan, when the chain did not run for this episode:
            # the runner reads ep.get("ar_log_pdf") and leaves the row's
            # entry all-nan, so a partial --ar_n_episodes run averages over
            # the episodes that have it instead of poisoning the mean.
            if ar is not None:
                episode["ar_log_pdf"] = ar.detach().float().cpu()
            episodes.append(episode)
        # Every chunk once the chain is on: it turns a ~1 s chunk into a ~12 s
        # one at the --era5 defaults, and a 10-chunk stride would leave a
        # 400-episode build silent for two minutes at a stretch.
        if verbose and (autoregressive or chunk_i % 10 == 0):
            print(f"  [era5] PIT {min(start + group, len(raw))}/{len(raw)} episodes", flush=True)

    if verbose and autoregressive:
        n_with = sum(1 for e in episodes if "ar_log_pdf" in e)
        print(f"  [era5] autoregressive chain: {n_with}/{len(episodes)} episodes, "
              f"order={ar_order}, conditioning={ar_conditioning}, "
              f"max_context={ar_max_context}")
    if verbose and episodes:
        Ps = [e["era5_meta"]["P"] for e in episodes]
        Ns = [e["era5_meta"]["N"] for e in episodes]
        print(f"  [era5] built {len(episodes)} episodes — "
              f"P {min(Ps)}..{max(Ps)}, N {min(Ns)}..{max(Ns)}, d_x="
              f"{episodes[0]['x_norm_train'].shape[-1]}")
    return episodes


def era5_episode_fingerprint(
    corpus_dir: str,
    *,
    grid_size: int,
    n_context: int,
    box_deg_range: tuple[float, float],
    vary_geometry: bool,
    grid_size_range: tuple[int, int],
    n_context_frac_range: tuple[float, float],
    k_folds: int,
    marginal: str,
    max_months: Optional[int],
    standardize_y: bool = True,
) -> dict:
    """Everything that changes what episode k IS on the ERA5 path, for the
    baseline cache's fingerprint. The PIT marginal is in here because z_test
    is what every cached baseline NLL was scored against — reusing a cache
    fitted under a different marginal would silently compare two different
    quantities (see the module docstring)."""
    return {
        "source": "era5",
        # realpath, not abspath: the corpus is commonly reached through a
        # symlink (e.g. a git worktree linking back to the main checkout's
        # eval/data/cache), and two spellings of the same directory must not
        # look like two different corpora -- that would silently discard a
        # whole run's worth of cached baseline fits.
        "corpus_dir": os.path.realpath(corpus_dir),
        # Symmetric with grid_size_range/n_context_frac_range below: each
        # pair is recorded only in the mode that actually reads it, so
        # changing an inert flag does not invalidate a multi-GB cache.
        "grid_size": None if vary_geometry else int(grid_size),
        "n_context": None if vary_geometry else int(n_context),
        "box_deg_range": [float(b) for b in box_deg_range],
        "vary_geometry": bool(vary_geometry),
        "grid_size_range": [int(g) for g in grid_size_range] if vary_geometry else None,
        "n_context_frac_range": (
            [float(f) for f in n_context_frac_range] if vary_geometry else None
        ),
        "pit_k_folds": int(k_folds),
        "pit_marginal": marginal,
        "max_months": max_months,
        # Changes the y the baselines are fitted on, hence every cached fit.
        "standardize_y": bool(standardize_y),
    }
