"""The shared PIT/autoregressive/standardize_y packaging leaves ERA5 episodes bit-identical.

_legacy_package is a frozen, verbatim copy of the loop build_era5_eval_episodes
ran before it moved into eval/data/episode_packaging.py. The test drives the
real build_era5_eval_episodes on fixed tensors (the corpus and the draw are
patched) and requires torch.equal on every tensor it returns.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytest
import torch
import torch.nn as nn

import eval.data.era5_episodes as era5_episodes


class _FakeTabICL(nn.Module):
    """Deterministic stand-in with run_pit's interface (forward -> logits, quantile_dist -> Normal)."""

    def forward(self, X: torch.Tensor, y: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        d, T, _ = X.shape
        n = T - y.shape[1]
        g = torch.Generator().manual_seed(int(X.sum().item() * 1000) % 2**31)
        return torch.randn(d, n, 2, generator=g)

    def quantile_dist(self, logits_flat: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(logits_flat[:, 0], torch.nn.functional.softplus(logits_flat[:, 1]) + 1e-3)


class _FakeCorpus:
    n_days_total = 3
    lat = np.zeros(4)
    lon = np.zeros(4)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass


def _fake_draw(_corpus: object, rng: np.random.Generator, *, vary_geometry: bool, **_kw: object) -> dict:
    """Fixed-shape draws, or P/N varying with the generator under vary_geometry."""
    P, N = (int(rng.integers(5, 9)), int(rng.integers(3, 6))) if vary_geometry else (6, 4)
    return {
        "x_norm_train": rng.standard_normal((P, 3)).astype(np.float32),
        "x_norm_test": rng.standard_normal((N, 3)).astype(np.float32),
        "y_train": (280.0 + 5.0 * rng.standard_normal(P)).astype(np.float32),
        "y_test": (280.0 + 5.0 * rng.standard_normal(N)).astype(np.float32),
        "grid_size": 4,
        "lat_bounds": (1.0, 2.0),
        "lon_bounds": (3.0, 4.0),
    }


def _legacy_package(
    raw: list[dict],
    *,
    tabicl_model: Any,
    k_folds: int,
    device: str,
    group: int,
    seed: int,
    standardize_y: bool,
    autoregressive: bool,
    ar_order: str,
    ar_conditioning: str,
    ar_max_context: Optional[int],
    ar_n_episodes: Optional[int],
) -> list[dict]:
    """Verbatim copy of build_era5_eval_episodes' PIT loop before the refactor (TabICL marginal only)."""
    from copula_inter.data.era5_live_dataset import _pit_group
    from copula_inter.marginal.pit import normalize_targets
    from eval.baselines.autoregressive import autoregressive_log_pdf

    dev = torch.device(device)
    episodes: list[dict] = []
    for start in range(0, len(raw), group):
        chunk = raw[start : start + group]
        out = _pit_group(
            torch.stack([r["x_norm_train"] for r in chunk]).to(dev),
            torch.stack([r["y_train"] for r in chunk]).to(dev),
            torch.stack([r["x_norm_test"] for r in chunk]).to(dev),
            torch.stack([r["y_test"] for r in chunk]).to(dev),
            tabicl_model,
            k_folds,
            marginal_backend=None,
            marginal_regressor=None,
            marginal_probs_n=99,
            seed=chunk[0]["ep_i"],
        )
        pits = [
            None if out is None else {k: out[k][b] for k in ("z_train", "z_test", "log_pdf_test")}
            for b in range(len(chunk))
        ]
        ar_log_pdf: list[Optional[torch.Tensor]] = [None] * len(chunk)
        n_ar = len(chunk) if ar_n_episodes is None else max(0, min(len(chunk), int(ar_n_episodes) - start))
        if autoregressive and out is not None and n_ar > 0:
            sub = chunk[:n_ar]
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
                continue
            P = int(r["x_norm_train"].shape[0])
            N = int(r["x_norm_test"].shape[0])
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
                "y_log_std": y_log_std,
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
            if ar is not None:
                episode["ar_log_pdf"] = ar.detach().float().cpu()
            episodes.append(episode)
    return episodes


def _raw_draws(n: int, seed: int, offset: int, vary_geometry: bool) -> list[dict]:
    raw = []
    for ep_i in range(offset, offset + n):
        d = _fake_draw(None, era5_episodes._episode_rng(seed, ep_i), vary_geometry=vary_geometry)
        raw.append(
            {
                "ep_i": ep_i,
                **{k: torch.from_numpy(d[k]) for k in ("x_norm_train", "x_norm_test", "y_train", "y_test")},
                "grid_size": int(d["grid_size"]),
                "lat_bounds": d["lat_bounds"],
                "lon_bounds": d["lon_bounds"],
            }
        )
    return raw


def _assert_identical(new: list[dict], old: list[dict]) -> None:
    assert len(new) == len(old)
    for a, b in zip(new, old):
        assert set(a) == set(b)
        for key, va in a.items():
            vb = b[key]
            if isinstance(va, torch.Tensor):
                assert va.dtype == vb.dtype and torch.equal(va, vb), key
            elif isinstance(va, dict) and key == "marginal_pit":
                for k in va:
                    assert torch.equal(va[k], vb[k]), (key, k)
            else:
                assert va == vb, key


@pytest.mark.parametrize("vary_geometry", [False, True])
@pytest.mark.parametrize("standardize_y", [True, False])
@pytest.mark.parametrize("autoregressive", [True, False])
def test_era5_episodes_are_bit_identical_to_the_pre_refactor_loop(
    monkeypatch: pytest.MonkeyPatch, vary_geometry: bool, standardize_y: bool, autoregressive: bool
) -> None:
    monkeypatch.setattr(era5_episodes, "GlobalERA5Corpus", _FakeCorpus)
    monkeypatch.setattr(era5_episodes, "_draw_episode", _fake_draw)
    torch.manual_seed(0)
    model = _FakeTabICL()
    seed, offset, n = 7, 3, 7
    kwargs: dict[str, Any] = dict(
        tabicl_model=model,
        k_folds=3,
        device="cpu",
        seed=seed,
        standardize_y=standardize_y,
        autoregressive=autoregressive,
        ar_order="random",
        ar_conditioning="teacher_forcing",
        ar_max_context=None,
        ar_n_episodes=4,
    )
    new = era5_episodes.build_era5_eval_episodes(
        "unused",
        n,
        offset=offset,
        vary_geometry=vary_geometry,
        pit_group_size=3,
        verbose=False,
        **kwargs,
    )
    old = _legacy_package(_raw_draws(n, seed, offset, vary_geometry), group=1 if vary_geometry else 3, **kwargs)
    _assert_identical(new, old)
    if autoregressive:
        assert sum("ar_log_pdf" in e for e in new) == 4
