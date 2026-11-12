"""SDWPF real-data source: loader, cleaning, F1b geometry, the episode draw and the shared real-data plumbing.

Loader tests run on a tiny synthetic farm written in SDWPF's CSV format. Draw
tests build an in-memory SdwpfTable over the real 245-day calendar (so the
five 49-day folds apply) with 12 turbines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from eval.data.episode_packaging import PitSettings
from eval.data.real import build_real_eval_episodes, real_episode_fingerprint
from eval.data.real.base import block_split
from eval.data.real.sdwpf import (
    FOLD_DAYS,
    N_FOLDS,
    RAW_COLUMNS,
    STEPS_PER_DAY,
    SdwpfData,
    SdwpfDataError,
    SdwpfExpected,
    SdwpfTable,
    absolute_wind_direction,
    candidate_steps,
    circular_mean_deg,
    draw,
    farm_direction,
    find_raw_files,
    fingerprint,
    load_sdwpf,
    wind_aligned_axes,
)
from eval.metrics.cluster_bootstrap import cluster_bootstrap_ci
from eval.runners.eval_args import SdwpfSpec

N_TURB, N_DAYS = 12, 4


def _farm(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A 4 x 3 farm, 500 m spacing; Patv falls with the downwind coordinate of the true wind direction."""
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(4) * 500.0, np.arange(3) * 500.0)
    loc = pd.DataFrame({"TurbID": np.arange(1, N_TURB + 1), "x": gx.ravel(), "y": gy.ravel()})
    n_steps = N_DAYS * STEPS_PER_DAY
    phi = rng.uniform(0.0, 360.0, n_steps)
    base = rng.uniform(300.0, 1000.0, n_steps)
    u, _ = wind_aligned_axes(loc["x"].to_numpy()[None, :], loc["y"].to_numpy()[None, :], phi[:, None])
    patv = base[:, None] - 0.1 * u + rng.normal(0.0, 5.0, (n_steps, N_TURB))
    wdir = rng.uniform(-5.0, 5.0, (n_steps, N_TURB))
    # Cumulative yaw: some turbines a full turn behind, which the mod 360 must undo.
    ndir = phi[:, None] - wdir - 360.0 * rng.integers(0, 2, (n_steps, N_TURB))
    step = np.repeat(np.arange(n_steps), N_TURB)
    minutes = (step % STEPS_PER_DAY) * 10
    scada = pd.DataFrame(
        {
            "TurbID": np.tile(loc["TurbID"].to_numpy(), n_steps),
            "Day": step // STEPS_PER_DAY + 1,
            "Tmstamp": [f"{m // 60:02d}:{m % 60:02d}" for m in minutes],
            "Wspd": 8.0,
            "Wdir": wdir.ravel(),
            "Etmp": 20.0,
            "Itmp": 25.0,
            "Ndir": ndir.ravel(),
            "Pab1": 1.0,
            "Pab2": 1.0,
            "Pab3": 1.0,
            "Prtv": 0.0,
            "Patv": patv.ravel(),
        }
    )
    return scada[list(RAW_COLUMNS)], loc


def _write(root: Path, scada: pd.DataFrame, loc: pd.DataFrame) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    scada.to_csv(root / "wtbdata.csv", index=False)
    loc.to_csv(root / "turb_location.CSV", index=False)
    return root


EXPECTED = SdwpfExpected(n_turbines=N_TURB, n_days=N_DAYS, n_records=N_TURB * N_DAYS * STEPS_PER_DAY)


# ---------------------------------------------------------------------------
# Loader and cleaning (CSV round trip)
# ---------------------------------------------------------------------------


def test_load_grids_the_farm_and_reuses_its_cache(tmp_path: Path) -> None:
    scada, loc = _farm()
    root = _write(tmp_path / "sdwpf", scada, loc)
    table = load_sdwpf(root, expected=EXPECTED, verbose=False)
    assert table.patv.shape == (N_DAYS * STEPS_PER_DAY, N_TURB)
    assert table.valid.all() and table.n_raw == len(scada)
    # Record (TurbID 5, Day 2, 00:10) lands on step 145, column 4.
    rec = scada[(scada.TurbID == 5) & (scada.Day == 2) & (scada.Tmstamp == "00:10")]
    assert table.patv[STEPS_PER_DAY + 1, 4] == pytest.approx(float(rec.Patv.iloc[0]), rel=1e-6)
    caches = list(root.glob("sdwpf_clean_v*.npz"))
    assert len(caches) == 1
    again = load_sdwpf(root, expected=EXPECTED, verbose=False)
    np.testing.assert_array_equal(again.patv, table.patv)
    assert again.dropped == table.dropped and again.identity == table.identity
    # New raw bytes key a new cache.
    scada.loc[0, "Patv"] += 1.0
    _write(root, scada, loc)
    load_sdwpf(root, expected=EXPECTED, verbose=False)
    assert len(list(root.glob("sdwpf_clean_v*.npz"))) == 2


def test_each_cleaning_rule_drops_its_rows_and_negative_power_is_clipped(tmp_path: Path) -> None:
    scada, loc = _farm()
    cases = {
        0: {"Wspd": np.nan},  # missing
        1: {"Patv": 0.0, "Wspd": 3.0},  # unknown: no power above cut-in
        2: {"Pab2": 90.0},  # unknown: feathered blade
        3: {"Ndir": 721.0},  # abnormal yaw
        4: {"Wdir": -181.0},  # abnormal relative direction
    }
    for row, values in cases.items():
        for col, val in values.items():
            scada.loc[row, col] = val
    scada.loc[5, ["Patv", "Wspd"]] = [-7.0, 1.0]  # negative power below cut-in: kept, clipped
    scada.loc[6, ["Patv", "Wspd"]] = [0.0, 2.5]  # Wspd not above 2.5: kept
    table = load_sdwpf(_write(tmp_path, scada, loc), expected=EXPECTED, verbose=False)
    assert table.dropped == {"missing": 1, "unknown": 2, "abnormal": 2}
    assert table.n_clipped == 1
    # Rows 0..11 are step 0, turbines 1..12.
    assert np.isnan(table.patv[0, :5]).all()
    assert np.isnan(table.ndir[0, :5]).all()
    assert table.patv[0, 5] == 0.0 and table.patv[0, 6] == 0.0
    assert table.valid.sum() == len(scada) - 5


def test_verification_fails_loudly(tmp_path: Path) -> None:
    scada, loc = _farm()
    with pytest.raises(SdwpfDataError, match="no position for TurbID \\[12\\]"):
        load_sdwpf(_write(tmp_path / "a", scada, loc[loc.TurbID != 12]), expected=EXPECTED, verbose=False)
    with pytest.raises(SdwpfDataError, match="records"):
        load_sdwpf(_write(tmp_path / "b", scada.iloc[1:], loc), expected=EXPECTED, verbose=False)
    with pytest.raises(SdwpfDataError, match="turbines, expected 134"):
        load_sdwpf(_write(tmp_path / "c", scada, loc), verbose=False)
    with pytest.raises(SdwpfDataError, match="does not exist"):
        find_raw_files(tmp_path / "missing")
    root = _write(tmp_path / "d", scada, loc)
    loc.to_csv(root / "second_location.csv", index=False)
    with pytest.raises(SdwpfDataError, match="exactly one"):
        find_raw_files(root)


# ---------------------------------------------------------------------------
# F1b geometry
# ---------------------------------------------------------------------------


def test_circular_mean_wraps_through_north() -> None:
    mean, r = circular_mean_deg(np.array([350.0, 10.0]))
    assert abs(((float(mean) + 180.0) % 360.0) - 180.0) < 1e-9
    assert float(r) == pytest.approx(np.cos(np.deg2rad(10.0)))
    mean, _ = circular_mean_deg(np.array([[90.0, np.nan, 90.0]]), axis=1)
    assert mean[0] == pytest.approx(90.0)


def test_absolute_direction_undoes_cumulative_yaw() -> None:
    np.testing.assert_allclose(
        absolute_wind_direction(np.array([-350.0, 700.0]), np.array([-20.0, 30.0])), [350.0, 10.0]
    )


def test_wind_aligned_axes_are_a_rotation_with_downwind_larger_u() -> None:
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=20) * 1000, rng.normal(size=20) * 1000
    u, v = wind_aligned_axes(x, y, 37.0)
    d_xy = np.hypot(x[:, None] - x[None], y[:, None] - y[None])
    np.testing.assert_allclose(np.hypot(u[:, None] - u[None], v[:, None] - v[None]), d_xy, atol=1e-8)
    # Wind from the north blows south: the southern turbine is downwind.
    u, _ = wind_aligned_axes(np.array([0.0, 0.0]), np.array([1000.0, 0.0]), 0.0)
    assert u[1] > u[0]
    # Wind from the west blows east.
    u, _ = wind_aligned_axes(np.array([0.0, 1000.0]), np.array([0.0, 0.0]), 270.0)
    assert u[1] > u[0]


# ---------------------------------------------------------------------------
# Episode draw on an in-memory 245-day farm
# ---------------------------------------------------------------------------

_FULL_DAYS = 245
_RATED = 1500.0


def _table(seed: int = 0, n_turb: int = 12) -> SdwpfTable:
    """245 days x n_turb turbines; Patv mostly in partial load, some steps at 0 or rated, a few invalid records."""
    rng = np.random.default_rng(seed)
    n_steps = _FULL_DAYS * STEPS_PER_DAY
    patv = rng.uniform(100.0, 1400.0, (n_steps, n_turb)).astype(np.float32)
    patv[rng.random(n_steps) < 0.2, :3] = 0.0  # low-wind atoms on some turbines
    patv[rng.random(n_steps) < 0.1, 3:6] = _RATED  # rated-power atoms
    patv[rng.random((n_steps, n_turb)) < 0.02] = np.nan  # dropped records
    nan = np.isnan(patv)
    wdir = rng.uniform(-5, 5, (n_steps, n_turb)).astype(np.float32)
    ndir = rng.uniform(0, 360, (n_steps, 1)).astype(np.float32) - wdir
    wdir[nan] = np.nan
    ndir[nan] = np.nan
    gx, gy = np.meshgrid(np.arange(4) * 500.0, np.arange(3) * 500.0)
    return SdwpfTable(
        turb_ids=np.arange(1, n_turb + 1),
        x=gx.ravel()[:n_turb],
        y=gy.ravel()[:n_turb],
        patv=patv,
        wspd=np.full_like(patv, 8.0),
        wdir=wdir,
        ndir=ndir,
        n_raw=n_steps * n_turb,
        dropped={"missing": 0, "unknown": 0, "abnormal": 0},
        flagged={"missing": 0, "unknown": 0, "abnormal": 0},
        n_clipped=0,
        identity={"cleaning_version": 1, "test": seed},
    )


def _spec(**kw: Any) -> SdwpfSpec:
    base: dict[str, Any] = {"n_context": 3, "n_query": 4, "timestamp_tries": 8}
    base.update(kw)
    return SdwpfSpec(**base)


def _data(table: SdwpfTable, cfg: SdwpfSpec) -> SdwpfData:
    n = cfg.n_context + cfg.n_query
    return SdwpfData(table=table, rated=_RATED, candidates=tuple(candidate_steps(table, d, n) for d in FOLD_DAYS))


def _draw(data: SdwpfData, cfg: SdwpfSpec, ep_i: int, seed: int = 5) -> Any:
    return draw(data, np.random.default_rng([seed, ep_i]), cfg, seed=seed, ep_i=ep_i)


def test_folds_are_five_contiguous_49_day_blocks_and_each_episode_stays_in_its_fold() -> None:
    assert FOLD_DAYS[0] == (1, 49) and FOLD_DAYS[-1] == (197, 245) and N_FOLDS == 5
    table = _table()
    cfg = _spec()
    data = _data(table, cfg)
    for ep_i in range(40):
        d = _draw(data, cfg, ep_i)
        assert d is not None
        lo, hi = FOLD_DAYS[ep_i % 5]
        assert d.meta["fold"] == ep_i % 5 and lo <= d.meta["day"] <= hi
        assert d.meta["step"] % 6 == 0  # hourly grid
        assert d.meta["cluster"] == f"day{d.meta['day']}"


def test_episodes_own_distinct_timestamps_and_shards_reproduce_the_stream() -> None:
    table = _table()
    cfg = _spec()
    data = _data(table, cfg)
    full = [_draw(data, cfg, i) for i in range(60)]
    steps = [d.meta["step"] for d in full if d is not None]
    assert len(steps) == len(set(steps))
    # A shard (episode_offset) is the same function of (seed, ep_i): identical draws.
    for i in (17, 43, 59):
        again = _draw(data, cfg, i)
        assert again is not None and full[i] is not None
        assert again.meta == full[i].meta
        np.testing.assert_array_equal(again.y_test, full[i].y_test)
    assert _draw(data, cfg, 3, seed=6).meta != full[3].meta


def test_no_row_is_both_context_and_query_and_partial_load_holds_on_every_selected_turbine() -> None:
    table = _table()
    for mask in ("random", "block"):
        cfg = _spec(mask=mask)
        data = _data(table, cfg)
        for i in range(30):
            d = _draw(data, cfg, i)
            assert d is not None
            ctx, qry = d.meta["context_turbines"], d.meta["query_turbines"]
            assert len(ctx) == 3 and len(qry) == 4 and not set(ctx) & set(qry)
            y = np.concatenate([d.y_train, d.y_test])
            assert np.all((y > 0) & (y < 0.99 * _RATED))
            cols = np.asarray(ctx + qry) - 1
            assert table.valid[d.meta["step"], cols].all()


def test_regime_all_keeps_atoms() -> None:
    table = _table()
    cfg = _spec(regime="all")
    data = _data(table, cfg)
    ys = np.concatenate([np.concatenate([d.y_train, d.y_test]) for d in (_draw(data, cfg, i) for i in range(60)) if d])
    assert (ys == 0).any() and (ys >= 0.99 * _RATED).any()


def test_block_mask_queries_are_the_nearest_to_the_anchor() -> None:
    rng = np.random.default_rng(0)
    pool = np.arange(10, 30)
    coords = rng.normal(size=(20, 2))
    ctx, qry, anchor = block_split(rng, pool, coords, 5, 6)
    a = int(np.flatnonzero(pool == anchor)[0])
    d = ((coords - coords[a]) ** 2).sum(1)
    assert set(qry) == set(pool[np.argsort(d)[:6]]) and anchor in qry
    assert not set(ctx) & set(qry) and len(ctx) == 5
    # On the farm: every query turbine is at least as close to the anchor as every unselected-for-query one.
    table = _table()
    cfg = _spec(mask="block")
    dd = _draw(_data(table, cfg), cfg, 2)
    assert dd is not None
    anc = dd.meta["anchor_turbine"] - 1
    pool_cols = np.flatnonzero(table.valid[dd.meta["step"]])
    dist = np.hypot(table.x[pool_cols] - table.x[anc], table.y[pool_cols] - table.y[anc])
    q = np.asarray(dd.meta["query_turbines"]) - 1
    q_d = np.hypot(table.x[q] - table.x[anc], table.y[q] - table.y[anc])
    others = dist[~np.isin(pool_cols, q)]
    assert q_d.max() <= others.min() + 1e-9


def test_feature_sets_share_episodes_and_f1b_reads_context_rows_only() -> None:
    table = _table()
    f0, f1b = _spec(features="f0"), _spec(features="f1b")
    data = _data(table, f0)
    for i in range(10):
        a, b = _draw(data, f0, i), _draw(data, f1b, i)
        assert a is not None and b is not None
        assert a.meta == b.meta
        np.testing.assert_array_equal(a.y_train, b.y_train)
        np.testing.assert_array_equal(a.x_train, b.x_train[:, :2])
        assert b.x_train.shape[1] == 4
    # phi is the circular mean over the context turbines: scrambling every non-context direction leaves F1b unchanged.
    d = _draw(data, f1b, 4)
    assert d is not None
    step = d.meta["step"]
    ctx = np.asarray(d.meta["context_turbines"]) - 1
    leaked = _table()
    others = np.setdiff1d(np.arange(12), ctx)
    leaked.ndir[step, others] += 123.0
    leaked.wdir[step, others] -= 40.0
    d2 = _draw(_data(leaked, f1b), f1b, 4)
    assert d2 is not None and d2.meta == d.meta
    np.testing.assert_array_equal(d2.x_train, d.x_train)
    np.testing.assert_array_equal(d2.x_test, d.x_test)
    assert farm_direction(table, step, ctx, 0.0) == farm_direction(leaked, step, ctx, 0.0)


def test_query_targets_and_forbidden_columns_never_reach_features_or_context() -> None:
    table = _table()
    for features in ("f0", "f1b"):
        cfg = _spec(features=features, regime="all")
        d = _draw(_data(table, cfg), cfg, 7)
        assert d is not None
        step, q = d.meta["step"], np.asarray(d.meta["query_turbines"]) - 1
        changed = _table()
        changed.patv[step, q] = changed.patv[step, q] * 0.5 + 1.0  # query targets
        for f in ("wspd", "wdir", "ndir"):  # the query turbines' own SCADA
            getattr(changed, f)[step, q] += 17.0
        d2 = _draw(_data(changed, cfg), cfg, 7)
        assert d2 is not None and d2.meta == d.meta
        np.testing.assert_array_equal(d2.x_train, d.x_train)
        np.testing.assert_array_equal(d2.x_test, d.x_test)
        np.testing.assert_array_equal(d2.y_train, d.y_train)
        assert not np.array_equal(d2.y_test, d.y_test)


def test_fingerprint_changes_with_every_input() -> None:
    table = _table()
    cfg = _spec(features="f1b")
    data = _data(table, cfg)
    base = fingerprint(data, cfg)
    for kw in (
        {"n_context": 4},
        {"n_query": 5},
        {"mask": "block"},
        {"features": "f0"},
        {"regime": "all"},
        {"direction_offset_deg": 90},
        {"atom_hi_frac": 0.9},
        {"rated_quantile": 0.99},
        {"timestamp_tries": 4},
    ):
        assert fingerprint(data, _spec(**{"features": "f1b", **kw})) != base, kw
    assert fingerprint(_data(_table(seed=1), cfg), cfg) != base
    other = SdwpfData(table=table, rated=_RATED + 1, candidates=data.candidates)
    assert fingerprint(other, cfg) != base
    fp = real_episode_fingerprint("sdwpf", cfg, data, k_folds=5, marginal="tabicl", standardize_y=True)
    assert fp["source"] == "real-sdwpf"
    assert real_episode_fingerprint("sdwpf", cfg, data, k_folds=4, marginal="tabicl", standardize_y=True) != fp


def test_cluster_bootstrap_resamples_whole_clusters() -> None:
    days = np.repeat(np.arange(20), 5)
    mean, lo, hi = cluster_bootstrap_ci(np.ones(100), days)
    assert mean == lo == hi == 1.0
    # Values constant within a cluster: the interval is that of 20 draws, wider than 100 i.i.d. ones would give.
    vals = np.repeat(np.random.default_rng(0).normal(size=20), 5)
    mean, lo, hi = cluster_bootstrap_ci(vals, days, seed=3)
    assert lo < mean < hi
    assert hi - lo > 2 * 1.96 * vals.std() / np.sqrt(100)


# ---------------------------------------------------------------------------
# Builder (PIT packaging) and the paired tool
# ---------------------------------------------------------------------------


class _FakeTabICL(nn.Module):
    def forward(self, X: torch.Tensor, y: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        d, T, _ = X.shape
        g = torch.Generator().manual_seed(int(X.sum().item() * 1000) % 2**31)
        return torch.randn(d, T - y.shape[1], 2, generator=g)

    def quantile_dist(self, logits_flat: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(logits_flat[:, 0], torch.nn.functional.softplus(logits_flat[:, 1]) + 1e-3)


def test_builder_packages_real_episodes_with_meta_and_shards_identically() -> None:
    cfg = _spec(n_context=4, n_query=5)
    data = _data(_table(), cfg)
    settings = PitSettings(tabicl_model=_FakeTabICL(), k_folds=2, autoregressive=True)
    eps = build_real_eval_episodes("sdwpf", cfg, data, 6, settings, seed=3, pit_batch=4, verbose=False)
    assert len(eps) == 6
    e = eps[2]
    assert e["real_meta"]["ep_i"] == 2 and e["real_meta"]["dataset"] == "sdwpf"
    assert e["real_meta"]["P"] == 4 and e["real_meta"]["N"] == 5
    assert e["x_norm_train"].shape == (4, 2) and "ar_log_pdf" in e
    shard = build_real_eval_episodes("sdwpf", cfg, data, 2, settings, seed=3, offset=2, pit_batch=1, verbose=False)
    assert shard[0]["real_meta"] == e["real_meta"]
    torch.testing.assert_close(shard[0]["y_test"], e["y_test"], rtol=0, atol=0)


def _dump(path: Path, eps: list[dict[str, Any]]) -> str:
    path.write_text(json.dumps({"episodes": eps}))
    return str(path)


def _ep(i: int, icl: float, ind: float, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "ep_i": i,
        "episode": meta,
        "nlls": {"icl": icl, "best_baseline": icl + 0.1},
        "total_nlls": {
            "icl": {"total": icl, "marginal": 1.0, "copula": icl - 1.0},
            "independence_marginal": {"total": ind, "marginal": 1.0, "copula": ind - 1.0},
        },
    }


def test_paired_compare_matches_episodes_and_rejects_different_ones(tmp_path: Path) -> None:
    from eval.runners.paired_compare import PairedCompareSpec, paired_deltas, within_run_deltas

    metas = [{"fold": i % 5, "cluster": f"day{i // 2}", "step": i} for i in range(20)]
    a = _dump(tmp_path / "a.json", [_ep(i, 2.0, 2.5, m) for i, m in enumerate(metas)])
    b = _dump(tmp_path / "b.json", [_ep(i, 1.5, 2.5, m) for i, m in enumerate(metas)])
    res = paired_deltas(PairedCompareSpec(a=a, b=b, n_boot=200))
    s = res["icl"]["total"]
    assert s["mean"] == pytest.approx(-0.5) and s["win_rate"] == 1.0 and s["n_clusters"] == 10
    assert res["independence_marginal"]["total"]["win_rate"] == 0.0
    # Cross-fitting reads only the listed folds.
    sub = paired_deltas(PairedCompareSpec(a=a, b=b, n_boot=50, folds=[0, 1, 2, 3]))
    assert sub["icl"]["total"]["n"] == 16
    within = within_run_deltas(PairedCompareSpec(a=a, n_boot=50))
    assert within["copula - independence"]["total"]["mean"] == pytest.approx(-0.5)
    bad = [dict(m) for m in metas]
    bad[3]["step"] = 999
    c = _dump(tmp_path / "c.json", [_ep(i, 1.5, 2.5, m) for i, m in enumerate(bad)])
    with pytest.raises(ValueError, match="different episodes"):
        paired_deltas(PairedCompareSpec(a=a, b=c))


# ---------------------------------------------------------------------------
# Audit: per-fold direction checks and the cross-fitted decision
# ---------------------------------------------------------------------------


def _wake_table(seed: int = 0) -> SdwpfTable:
    """245 days x 12 turbines in partial load; Patv falls with the downwind coordinate of the true direction."""
    rng = np.random.default_rng(seed)
    n_steps = _FULL_DAYS * STEPS_PER_DAY
    gx, gy = np.meshgrid(np.arange(4) * 500.0, np.arange(3) * 500.0)
    x, y = gx.ravel(), gy.ravel()
    phi = rng.uniform(0.0, 360.0, n_steps)
    u, _ = wind_aligned_axes(x[None, :], y[None, :], phi[:, None])
    patv = rng.uniform(500.0, 900.0, n_steps)[:, None] - 0.2 * u + rng.normal(0, 5, (n_steps, 12))
    wdir = rng.uniform(-5, 5, (n_steps, 12))
    ndir = phi[:, None] - wdir
    return SdwpfTable(
        turb_ids=np.arange(1, 13),
        x=x,
        y=y,
        patv=patv.astype(np.float32),
        wspd=np.full((n_steps, 12), 8.0, dtype=np.float32),
        wdir=wdir.astype(np.float32),
        ndir=ndir.astype(np.float32),
        n_raw=n_steps * 12,
        dropped={"missing": 0, "unknown": 0, "abnormal": 0},
        flagged={"missing": 0, "unknown": 0, "abnormal": 0},
        n_clipped=0,
        identity={"test": "wake"},
    )


def test_audit_finds_the_wake_convention_per_fold() -> None:
    from eval.runners.sdwpf_audit import audit

    result, farm_phi = audit(_wake_table(), n_query=4, n_context=[4], n_boot=100, distance_bins=4)
    assert farm_phi.shape == (_FULL_DAYS * STEPS_PER_DAY // 6,)
    dr = result["direction"]
    for k in range(5):
        offs = dr["frame_alignment"]["partial_pooled"][f"fold{k}"]["offsets"]
        assert offs["+0"]["ci95"][1] < -0.5
        assert offs["+180"]["mean"] == pytest.approx(-offs["+0"]["mean"])
        dec = dr["cross_fitted_decisions"][f"fold{k}"]
        assert dec["chosen_offset_deg"] == 0 and dec["f1b_admissible"]
        assert f"fold{k}" not in dec["training_folds"]
    assert dr["spread"]["fold0"]["circular_std_deg"]["p95"] < 10.0
    assert result["eligibility"]["fold0"]["P4_N4"]["all"] == 49 * 24


def test_cross_fitted_decision_for_fold_k_never_reads_fold_k() -> None:
    from eval.runners.sdwpf_audit import audit

    base, _ = audit(_wake_table(), n_query=4, n_context=[4], n_boot=100, distance_bins=4)
    scrambled = _wake_table()
    lo, hi = FOLD_DAYS[2]
    rows = slice((lo - 1) * STEPS_PER_DAY, hi * STEPS_PER_DAY)
    # Fold 2's directions point the other way: its own statistics flip, its decision must not move.
    scrambled.ndir[rows] += 180.0
    other, _ = audit(scrambled, n_query=4, n_context=[4], n_boot=100, distance_bins=4)
    d0, d1 = base["direction"]["cross_fitted_decisions"], other["direction"]["cross_fitted_decisions"]
    assert d1["fold2"] == d0["fold2"]
    assert d1["fold0"]["offsets"] != d0["fold0"]["offsets"]
    f2 = other["direction"]["frame_alignment"]["partial_pooled"]["fold2"]["offsets"]
    assert f2["+180"]["mean"] < 0 < f2["+0"]["mean"]
