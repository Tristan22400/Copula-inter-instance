"""Real-ERA5 evaluation episodes in the dict format eval_checkpoint.py scores.

Episodes are PIT'd once with a frozen marginal (as in era5_live_dataset);
that PIT's z_test is the shared z_test for every method. Episodes have no
oracle fields (R_star, Sigma_star, mu_star, sigma_star, kernel metadata).

Geometry, centred area-uniformly over the globe:
    fixed shape (default): exact grid_size and n_context, so all episodes
        share P and N and are PIT'd in batches.
    vary_geometry=True: grid_size and n_context_frac drawn per episode from
        ranges (era5_live's training distribution); PIT per episode.
d_x = 6: lon, lat and the four fetch_era5_static.STATIC_VARS fields.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from copula_inter.pit import TabICLLike

_HERE = os.path.dirname(os.path.abspath(__file__))

from copula_inter.pit import normalize_targets
from eval.data.era5_global_corpus import GlobalERA5Corpus

__all__ = ["build_era5_eval_episodes", "era5_episode_fingerprint", "DEFAULT_CORPUS_DIR"]

# Held-out year, disjoint from the training corpus.
DEFAULT_CORPUS_DIR = os.path.join(_HERE, "cache", "era5_global_val")


def _episode_rng(seed: int, ep_i: int) -> np.random.Generator:
    """Generator seeded from (seed, global episode index)."""
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
    """One geographic draw with a full grid_size x grid_size block, redrawing from the same generator until one is found."""
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
        f"{grid_size}x{grid_size} block of 0.25deg points; raise era5.box_deg_min "
        "or lower era5.grid_size."
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
    tabicl_model: TabICLLike | None = None,
    k_folds: int = 10,
    device: str | torch.device = "cpu",
    pit_group_size: int = 8,
    marginal_backend: Optional[str] = None,
    marginal_regressor: Any = None,
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
    """Build n_episodes PIT'd real-ERA5 evaluation episodes (global indices offset .. offset + n_episodes - 1).

    Each episode dict has x_norm_train (P, 6), x_norm_test (N, 6), y_train,
    y_test (z-scored by y_train's statistics when standardize_y), z_train,
    z_test, log_pdf_test (raw nats), marginal_pit, y_log_std (per-point nats to
    add to NLLs computed on standardized y; 0 when raw), era5_meta, and with
    autoregressive=True ar_log_pdf (N,), the teacher-forced chain log-density
    (raw nats) for the first ar_n_episodes episodes. The PIT and the chain run on
    raw y.
    """
    from copula_inter.era5_live_dataset import _pit_group
    from eval.baselines.autoregressive import autoregressive_log_pdf

    if tabicl_model is None and marginal_backend is None:
        raise ValueError(
            "ERA5 episodes need a real marginal to PIT with: pass tabicl_model "
            "(marginal.z_train_source=tabicl) or a marginal_backend + marginal_regressor. "
            "There is no analytic/oracle PIT on real data."
        )
    if autoregressive and marginal_backend is not None:
        # The chain needs TabICL; other backends have no incremental entry point.
        raise NotImplementedError(
            f"autoregressive.enabled is implemented for the TabICL marginal only, not "
            f"for backend {marginal_backend!r}. Re-run with autoregressive.enabled=false, "
            f"or with the default marginal.z_train_source=tabicl."
        )

    if verbose:
        print(f"  [era5] loading corpus from {corpus_dir}")
    # Pass lazy explicitly (max_months is applied before the corpus' own heuristic).
    corpus = GlobalERA5Corpus(corpus_dir, max_months=max_months, lazy=lazy)
    if verbose:
        print(
            f"  [era5] corpus: {corpus.n_days_total} daily snapshots, native grid {len(corpus.lat)}x{len(corpus.lon)}"
        )

    # ---- Geographic/target draws (CPU, cheap) ----
    raw: list[dict] = []
    for local_i in range(n_episodes):
        ep_i = offset + local_i
        drawn = _draw_episode(
            corpus,
            _episode_rng(seed, ep_i),
            vary_geometry=vary_geometry,
            grid_size=grid_size,
            n_context=n_context,
            box_deg_range=box_deg_range,
            grid_size_range=grid_size_range,
            n_context_frac_range=n_context_frac_range,
        )
        raw.append(
            {
                "ep_i": ep_i,
                "x_norm_train": torch.from_numpy(drawn["x_norm_train"]),
                "x_norm_test": torch.from_numpy(drawn["x_norm_test"]),
                "y_train": torch.from_numpy(drawn["y_train"]),
                "y_test": torch.from_numpy(drawn["y_test"]),
                # Both samplers always return these keys.
                "grid_size": int(drawn["grid_size"]),
                "lat_bounds": drawn["lat_bounds"],
                "lon_bounds": drawn["lon_bounds"],
            }
        )

    # PIT: batched for fixed shapes, per episode for varying geometry.
    dev = torch.device(device)
    group = 1 if vary_geometry else max(1, int(pit_group_size))
    episodes: list[dict] = []
    for chunk_i, start in enumerate(range(0, len(raw), group)):
        chunk = raw[start : start + group]
        # _pit_group also handles B=1.
        out = _pit_group(
            torch.stack([r["x_norm_train"] for r in chunk]).to(dev),
            torch.stack([r["y_train"] for r in chunk]).to(dev),
            torch.stack([r["x_norm_test"] for r in chunk]).to(dev),
            torch.stack([r["y_test"] for r in chunk]).to(dev),
            tabicl_model,
            k_folds,
            marginal_backend=marginal_backend,
            marginal_regressor=marginal_regressor,
            marginal_probs_n=marginal_probs_n,
            seed=chunk[0]["ep_i"],
        )
        pits = [
            None if out is None else {k: out[k][b] for k in ("z_train", "z_test", "log_pdf_test")}
            for b in range(len(chunk))
        ]

        # Autoregressive chain on the same group and marginal, raw y.
        ar_log_pdf: list[Optional[torch.Tensor]] = [None] * len(chunk)
        n_ar = len(chunk) if ar_n_episodes is None else max(0, min(len(chunk), int(ar_n_episodes) - start))
        if autoregressive and out is not None and n_ar > 0:
            sub = chunk[:n_ar]
            assert tabicl_model is not None  # autoregressive rejects every other backend above
            ar_out = autoregressive_log_pdf(
                tabicl_model,
                torch.stack([r["x_norm_train"] for r in sub]).to(dev),
                torch.stack([r["y_train"] for r in sub]).to(dev),
                torch.stack([r["x_norm_test"] for r in sub]).to(dev),
                torch.stack([r["y_test"] for r in sub]).to(dev),
                order=ar_order,
                conditioning=ar_conditioning,
                max_context=ar_max_context,
                seed=seed,
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
            # Same normalize_targets call as the PIT, so y_log_std is its exact Jacobian.
            y_tr_scaled, y_te_scaled, _, y_std = normalize_targets(r["y_train"], r["y_test"])
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
                # Per-point nats to add to NLLs computed on this episode's y (0 when raw).
                "y_log_std": y_log_std,
                # Reuse this PIT in the runner.
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
            # Absent (not NaN) when the chain did not run for this episode.
            if ar is not None:
                episode["ar_log_pdf"] = ar.detach().float().cpu()
            episodes.append(episode)
        # Report progress every chunk when the chain is on.
        if verbose and (autoregressive or chunk_i % 10 == 0):
            print(f"  [era5] PIT {min(start + group, len(raw))}/{len(raw)} episodes", flush=True)

    if verbose and autoregressive:
        n_with = sum(1 for e in episodes if "ar_log_pdf" in e)
        print(
            f"  [era5] autoregressive chain: {n_with}/{len(episodes)} episodes, "
            f"order={ar_order}, conditioning={ar_conditioning}, "
            f"max_context={ar_max_context}"
        )
    if verbose and episodes:
        Ps = [e["era5_meta"]["P"] for e in episodes]
        Ns = [e["era5_meta"]["N"] for e in episodes]
        print(
            f"  [era5] built {len(episodes)} episodes — "
            f"P {min(Ps)}..{max(Ps)}, N {min(Ns)}..{max(Ns)}, d_x="
            f"{episodes[0]['x_norm_train'].shape[-1]}"
        )
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
    """Digest of what defines episode k on the ERA5 path (corpus, geometry, marginal), for the baseline cache."""
    return {
        "source": "era5",
        # realpath so symlinked spellings of one corpus match.
        "corpus_dir": os.path.realpath(corpus_dir),
        # Record each range only in the mode that uses it.
        "grid_size": None if vary_geometry else int(grid_size),
        "n_context": None if vary_geometry else int(n_context),
        "box_deg_range": [float(b) for b in box_deg_range],
        "vary_geometry": bool(vary_geometry),
        "grid_size_range": [int(g) for g in grid_size_range] if vary_geometry else None,
        "n_context_frac_range": ([float(f) for f in n_context_frac_range] if vary_geometry else None),
        "pit_k_folds": int(k_folds),
        "pit_marginal": marginal,
        "max_months": max_months,
        # Changes the y the baselines are fitted on, hence every cached fit.
        "standardize_y": bool(standardize_y),
    }
