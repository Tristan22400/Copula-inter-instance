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

# Kernels for the zero-mean GP baseline (fixed, so the table rows are fixed).
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
    """Zero-mean GP-MLE baselines on marginal_pit["z_train"] (the model's own marginal input).

    nlls/R_dict use the shared ep["z_test"] like every baseline; y_space_nlls
    use marginal_pit's z_test/log_pdf_test like the ICL row. Not cached (depends
    on the marginal).
    """
    X_train = ep["x_norm_train"].to(device)  # (P, d_x)
    X_test = ep["x_norm_test"].to(device)  # (N, d_x)
    z_test = ep["z_test"].to(device)  # (N,) ground truth, shared across every baseline
    z_train_marg = marginal_pit["z_train"].to(device)  # (P,) real marginal's PIT residual
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
                X_train,
                z_train_marg,
                X_test,
                kname,
                n_steps=n_steps,
                lr=lr,
                n_restarts=n_restarts,
                oracle_mode=oracle_mode,
                prior_cfg=prior_cfg,
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
    """Randomly split range(n) into k near-equal folds using its own generator seeded with seed."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    base, extra = divmod(n, k)
    folds: list[Tensor] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        folds.append(perm[start : start + size])
        start += size
    return folds


def _select_best_baseline_cv(
    baseline_R: dict[str, Tensor],
    z_test: Tensor,
    n_folds: int,
    min_fold_size: int,
    seed: int,
) -> tuple[float, str | None, list[dict]]:
    """Best fitted baseline by nested leave-one-fold-out CV over the test points.

    For each fold, pick the candidate with the lowest NLL on the other folds and
    score it on the held-out fold (submatrices of each R, no refit).
    K = min(n_folds, n // min_fold_size); returns (nan, None, []) if K < 2 or no
    candidates.

    Returns:
        (pooled NLL: fold NLLs weighted by fold size, winner, fold details).
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
        fold_details.append(
            {
                "fold": i,
                "size": test_idx.numel(),
                "selected": selected,
                "val_nll": val_nll[selected],
                "test_nll": test_nll,
            }
        )

    pooled_nll = weighted_sum / n
    mode_key = Counter(fd["selected"] for fd in fold_details).most_common(1)[0][0]
    return pooled_nll, mode_key, fold_details


# Not candidates for best baseline: unfitted references, icl, oracle and best_baseline.
_NON_FITTED_EXCLUDED = {"independence", "gp_prior_rbf", "icl", "oracle", "best_baseline"}


@dataclass(frozen=True)
class _PoolTensor:
    """Pickle-only representation of a CPU tensor sent to a pool worker."""

    value: np.ndarray


def _pool_encode_tensors(value):
    """Recursively replace every tensor in an episode with NumPy storage (avoids torch's shared-memory transport)."""
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
    """Inverse of _pool_encode_tensors."""
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
    """Fit one episode's baselines in a worker (spawn start method; NumPy in and out, one thread)."""
    cache_key, ep, fit_seed, kwargs = payload
    import torch as _torch  # re-imported in the spawned interpreter

    ep = _pool_decode_tensors(ep)

    # One thread per worker.
    _torch.set_num_threads(1)
    try:
        nlls, R_dict, y_nlls = eval_baselines_episode(ep=ep, device=_torch.device("cpu"), fit_seed=fit_seed, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        import traceback

        return cache_key, None, f"{exc}\n{traceback.format_exc()}"
    return (
        cache_key,
        {
            "nlls": nlls,
            # Copy so the array owns its storage.
            "R_dict": {k: v.detach().cpu().contiguous().numpy().copy() for k, v in R_dict.items()},
            "y_nlls": y_nlls,
        },
        None,
    )


def _count_physical_cores(cpus: set[int]) -> int:
    """Number of distinct physical cores under the given logical CPUs (0 if unknown)."""
    cores = set()
    for c in cpus:
        try:
            with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as fh:
                cores.add(fh.read().strip())
        except OSError:
            return 0
    return len(cores)


def _baseline_fit_seed(seed: int, cache_key: str) -> int:
    """Per-episode fitting seed: zlib.crc32 of the cache key."""
    return (zlib.crc32(cache_key.encode()) ^ (seed * 2_654_435_761)) % (2**31 - 1)


def _valid_cached_entry(cache_entries: dict, cache_key: str, ep_i: int) -> dict | None:
    """The cached entry for this episode, or None if missing or written with an older result schema."""
    cached = cache_entries.get(cache_key)
    if cached is None:
        return None
    if not EXPECTED_BASELINE_KEYS.issubset(cached["nlls"].keys()):
        # Missing a baseline added later.
        missing = EXPECTED_BASELINE_KEYS - cached["nlls"].keys()
        print(f"  [ep {ep_i}] cached baselines missing {sorted(missing)} — refitting")
        return None
    if "y_nlls" not in cached:
        # Missing the Y-space NLLs.
        print(f"  [ep {ep_i}] cached entry predates total-NLL tracking — refitting")
        return None
    if any(not isinstance(v, dict) for v in cached["y_nlls"].values()):
        # Y-space NLLs without the marginal/copula split.
        print(f"  [ep {ep_i}] cached y_nlls predates marginal/copula split — refitting")
        return None
    return cached


def _episode_to_pool_payload(ep: dict) -> dict:
    """Convert an episode to NumPy for the process pool."""
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
    """Fit the pending episodes across a process pool, storing each result in fitted (and on disk, if caching) as it completes."""
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
                # Convert NumPy results back to tensors for the cache.
                result["R_dict"] = {k: torch.from_numpy(v) for k, v in result["R_dict"].items()}
                fitted[cache_key] = result
                if use_cache:
                    # Write this episode's shard now.
                    save_baseline_entry(cache_path, fingerprint, cache_key, result)

            elapsed = time.time() - t0
            rate = elapsed / done
            eta = rate * (total - done)
            print(
                f"  [prefit {done}/{total}] {cache_key}  "
                f"({elapsed / 60:.1f} min elapsed, {rate:.1f} s/ep wall, "
                f"ETA {eta / 60:.1f} min)",
                flush=True,
            )

    print(
        f"  [prefit] fitted {done - failures}/{total} episode(s) on "
        f"{n_workers} worker(s) in {(time.time() - t0) / 60:.1f} min" + (f" — {failures} FAILED" if failures else ""),
        flush=True,
    )
