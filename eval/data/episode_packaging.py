"""Shared packaging of real-data episodes: the marginal PIT, the autoregressive chain and standardize_y.

Every real-data source (ERA5 and the eval/data/real datasets) draws raw
episodes on its own and hands them here. One frozen marginal PITs each episode
once; that PIT's z_test is the shared z_test every method is scored against.
tests/test_episode_packaging.py pins the ERA5 outputs bit-identically to the
loop this replaced.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from copula_inter.marginal.pit import normalize_targets

if TYPE_CHECKING:
    from copula_inter.marginal.pit import TabICLLike

__all__ = ["PitSettings", "RawEpisode", "package_pit_episodes"]


@dataclass
class RawEpisode:
    """One drawn episode before the PIT: normalized features, raw targets and source metadata."""

    ep_i: int
    x_norm_train: torch.Tensor  # (P, d_x)
    x_norm_test: torch.Tensor  # (N, d_x)
    y_train: torch.Tensor  # (P,) raw units
    y_test: torch.Tensor  # (N,) raw units
    meta: dict[str, Any]


@dataclass
class PitSettings:
    """The marginal and chain settings shared by every episode of one run."""

    tabicl_model: TabICLLike | None = None
    k_folds: int = 10
    device: str | torch.device = "cpu"
    marginal_backend: Optional[str] = None
    marginal_regressor: Any = None
    marginal_probs_n: int = 99
    standardize_y: bool = True
    autoregressive: bool = False
    ar_order: str = "random"
    ar_conditioning: str = "teacher_forcing"
    ar_max_context: Optional[int] = None
    ar_n_episodes: Optional[int] = None


def check_pit_settings(settings: PitSettings, source: str) -> None:
    """Raise when the settings cannot PIT real data (no marginal) or run the chain (non-TabICL backend)."""
    if settings.tabicl_model is None and settings.marginal_backend is None:
        raise ValueError(
            f"{source} episodes need a real marginal to PIT with: pass tabicl_model "
            "(marginal.z_train_source=tabicl) or a marginal_backend + marginal_regressor. "
            "There is no analytic/oracle PIT on real data."
        )
    if settings.autoregressive and settings.marginal_backend is not None:
        # The chain needs TabICL; other backends have no incremental entry point.
        raise NotImplementedError(
            f"autoregressive.enabled is implemented for the TabICL marginal only, not "
            f"for backend {settings.marginal_backend!r}. Re-run with autoregressive.enabled=false, "
            f"or with the default marginal.z_train_source=tabicl."
        )


def package_pit_episodes(
    raw: list[RawEpisode],
    settings: PitSettings,
    *,
    seed: int,
    group: int,
    meta_key: str,
    make_meta: Callable[[RawEpisode, int, int], dict[str, Any]],
    tag: str,
    verbose: bool = True,
) -> list[dict]:
    """PIT raw episodes in groups of `group` (which must share P and N) and package them for eval_checkpoint.

    Each episode dict has x_norm_train, x_norm_test, y_train, y_test (z-scored
    by y_train's statistics when standardize_y), z_train, z_test, log_pdf_test
    (raw nats), marginal_pit, y_log_std (per-point nats to add to NLLs computed
    on standardized y; 0 when raw), meta_key -> make_meta(raw, P, N), and with
    autoregressive ar_log_pdf (N,), the teacher-forced chain log-density (raw
    nats) for the first ar_n_episodes episodes. The PIT and the chain run on
    raw y. An episode whose PIT is unavailable (too little context) is skipped.
    """
    from copula_inter.data.era5_live_dataset import _pit_group
    from eval.baselines.autoregressive import autoregressive_log_pdf

    s = settings
    dev = torch.device(s.device)
    group = max(1, int(group))
    episodes: list[dict] = []
    for chunk_i, start in enumerate(range(0, len(raw), group)):
        chunk = raw[start : start + group]
        # _pit_group also handles B=1.
        out = _pit_group(
            torch.stack([r.x_norm_train for r in chunk]).to(dev),
            torch.stack([r.y_train for r in chunk]).to(dev),
            torch.stack([r.x_norm_test for r in chunk]).to(dev),
            torch.stack([r.y_test for r in chunk]).to(dev),
            s.tabicl_model,
            s.k_folds,
            marginal_backend=s.marginal_backend,
            marginal_regressor=s.marginal_regressor,
            marginal_probs_n=s.marginal_probs_n,
            seed=chunk[0].ep_i,
        )
        pits = [
            None if out is None else {k: out[k][b] for k in ("z_train", "z_test", "log_pdf_test")}
            for b in range(len(chunk))
        ]

        # Autoregressive chain on the same group and marginal, raw y.
        ar_log_pdf: list[Optional[torch.Tensor]] = [None] * len(chunk)
        n_ar = len(chunk) if s.ar_n_episodes is None else max(0, min(len(chunk), int(s.ar_n_episodes) - start))
        if s.autoregressive and out is not None and n_ar > 0:
            sub = chunk[:n_ar]
            assert s.tabicl_model is not None  # check_pit_settings rejects every other backend
            ar_out = autoregressive_log_pdf(
                s.tabicl_model,
                torch.stack([r.x_norm_train for r in sub]).to(dev),
                torch.stack([r.y_train for r in sub]).to(dev),
                torch.stack([r.x_norm_test for r in sub]).to(dev),
                torch.stack([r.y_test for r in sub]).to(dev),
                order=s.ar_order,
                conditioning=s.ar_conditioning,
                max_context=s.ar_max_context,
                seed=seed,
                episode_indices=[r.ep_i for r in sub],
            )
            for b in range(len(sub)):
                ar_log_pdf[b] = ar_out["log_pdf"][b]

        for r, pit, ar in zip(chunk, pits, ar_log_pdf):
            if pit is None:
                print(f"  [{tag}] ep {r.ep_i}: PIT unavailable (too little context), skipping")
                continue
            P = int(r.x_norm_train.shape[0])
            N = int(r.x_norm_test.shape[0])
            # Same normalize_targets call as the PIT, so y_log_std is its exact Jacobian.
            y_tr_scaled, y_te_scaled, _, y_std = normalize_targets(r.y_train, r.y_test)
            if s.standardize_y:
                y_tr, y_te = y_tr_scaled, y_te_scaled
                y_log_std = float(y_std.log())
            else:
                y_tr, y_te = r.y_train, r.y_test
                y_log_std = 0.0
            episode = {
                "x_norm_train": r.x_norm_train,
                "x_norm_test": r.x_norm_test,
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
                meta_key: make_meta(r, P, N),
            }
            # Absent (not NaN) when the chain did not run for this episode.
            if ar is not None:
                episode["ar_log_pdf"] = ar.detach().float().cpu()
            episodes.append(episode)
        # Report progress every chunk when the chain is on.
        if verbose and (s.autoregressive or chunk_i % 10 == 0):
            print(f"  [{tag}] PIT {min(start + group, len(raw))}/{len(raw)} episodes", flush=True)

    if verbose and s.autoregressive:
        n_with = sum(1 for e in episodes if "ar_log_pdf" in e)
        print(
            f"  [{tag}] autoregressive chain: {n_with}/{len(episodes)} episodes, "
            f"order={s.ar_order}, conditioning={s.ar_conditioning}, "
            f"max_context={s.ar_max_context}"
        )
    if verbose and episodes:
        Ps = [e[meta_key]["P"] for e in episodes]
        Ns = [e[meta_key]["N"] for e in episodes]
        print(
            f"  [{tag}] built {len(episodes)} episodes — "
            f"P {min(Ps)}..{max(Ps)}, N {min(Ns)}..{max(Ns)}, d_x="
            f"{episodes[0]['x_norm_train'].shape[-1]}"
        )
    return episodes
