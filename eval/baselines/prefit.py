"""Fit the classical baselines for eval_checkpoint: zero-mean GP rows, nested-CV best-of-baselines, and the parallel, cached prefit pool."""

from __future__ import annotations

import multiprocessing as mp
import time
import zlib
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor



from copula_inter.loss import y_space_nll  # noqa: E402
from eval.baselines.classical import (  # noqa: E402
    EXPECTED_BASELINE_KEYS,
    corr_nll_single,
    eval_baselines_episode,
    fit_zero_mean_gp_on_marginal,
    save_baseline_entry,
)
from eval.results import (  # noqa: E402
    NAN_PARTS as _NAN_PARTS,
)

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


# Excluded from the "5 best baselines" ranking: independence/gp_prior_rbf
# are trivial, no-fit reference points rather than baselines, icl/oracle
# aren't baselines at all (icl is our model, oracle is a reference, not a
# fitted candidate), and best_baseline is itself derived from this same
# ranking (added after it's computed each episode — see main()'s loop).
_NON_FITTED_EXCLUDED = {"independence", "gp_prior_rbf", "icl", "oracle", "best_baseline"}


@dataclass(frozen=True)
class _PoolTensor:
    """Pickle-only representation of a CPU tensor sent to a pool worker."""

    value: np.ndarray


def _pool_encode_tensors(value):
    """Recursively replace every tensor in an episode with NumPy storage.

    Kernel metadata is nested (notably ``kernel_component_params`` is a
    list of dicts containing tensors), so converting only an episode's
    top-level values still leaves PyTorch's multiprocessing reducer active.
    That reducer creates shared-memory descriptors and was the direct cause
    of the ``RLIMIT_NOFILE`` failure in the 500-episode prefit.
    """
    if isinstance(value, Tensor):
        return _PoolTensor(value.detach().cpu().contiguous().numpy().copy())
    if isinstance(value, dict):
        return {key: _pool_encode_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_pool_encode_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pool_encode_tensors(item) for item in value)
    return value


def _pool_decode_tensors(value):
    """Inverse of :func:`_pool_encode_tensors`, executed in a worker."""
    if isinstance(value, _PoolTensor):
        return torch.from_numpy(value.value)
    if isinstance(value, dict):
        return {key: _pool_decode_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_pool_decode_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pool_decode_tensors(item) for item in value)
    return value


def _fit_baselines_task(payload: tuple) -> tuple:
    """One episode's classical baselines, fit in a worker process.

    Module-level (not a closure) so it survives pickling under the "spawn"
    start method, which _prefit_baselines_parallel uses unconditionally: the
    parent has almost certainly initialised CUDA by this point (the ICL model
    and TabICL marginal are already resident), and a forked child inheriting
    a CUDA context crashes the moment it touches a tensor. Spawned workers
    re-import this module from scratch and never initialise CUDA at all.

    Episode tensors arrive as NumPy arrays and correlation matrices leave as
    NumPy arrays.  This is intentional: passing ``torch.Tensor`` objects
    through a ``multiprocessing.Pool`` makes PyTorch create one shared-memory
    handle per storage.  With a large pending queue that exhausts the OAR
    job's low ``RLIMIT_NOFILE`` before workers can drain it.  Plain NumPy
    pickle transport is bounded by the pool pipe and has no shared-memory
    file-descriptor lifetime to manage.
    """
    cache_key, ep, fit_seed, kwargs = payload
    import torch as _torch  # re-imported in the spawned interpreter

    ep = _pool_decode_tensors(ep)

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
        # Copy so the returned ndarray owns its storage independently of any
        # temporary tensor created by a baseline implementation.
        "R_dict": {k: v.detach().cpu().contiguous().numpy().copy()
                   for k, v in R_dict.items()},
        "y_nlls": y_nlls,
    }, None


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


def _episode_to_pool_payload(ep: dict) -> dict:
    """Make an episode safe to send through a multiprocessing pool.

    Do not return CPU tensors here, including ones inside metadata. PyTorch's
    multiprocessing reducer turns them into shared-memory objects; a queue
    of hundreds of episodes then needs thousands of open descriptors/files.
    NumPy arrays use ordinary pickle transport instead.
    """
    return _pool_encode_tensors(ep)


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
    payloads = []
    for key, fit_seed, ep in pending:
        payloads.append((key, _episode_to_pool_payload(ep), fit_seed, fit_kwargs))

    with ctx.Pool(processes=n_workers) as pool:
        for cache_key, result, err in pool.imap_unordered(_fit_baselines_task, payloads):
            done += 1
            if err is not None:
                failures += 1
                print(f"  [prefit] {cache_key} FAILED:\n{err}", flush=True)
            else:
                # _fit_baselines_task deliberately returns ordinary ndarrays
                # to avoid PyTorch shared-memory transport.  Restore the
                # cache's established tensor schema before writing it.
                result["R_dict"] = {
                    k: torch.from_numpy(v) for k, v in result["R_dict"].items()
                }
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
