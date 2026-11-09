"""KAUST, EMNIST, California, Meuse and CTR23 real-data sources: loaders, cleaning, splits, folds, leakage, determinism.

Every test runs on a tiny synthetic file in the dataset's format (or an
in-memory table); fetch_openml / fetch_california_housing are monkeypatched, so
nothing touches the network or the real files.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eval.data.real import california, ctr23, emnist, kaust, meuse
from eval.data.real.base import RealDraw
from eval.runners.eval_args import CaliforniaSpec, Ctr23Spec, EmnistSpec, KaustSpec, MeuseSpec, compose_eval_spec


def _draw(mod: Any, data: Any, cfg: Any, ep_i: int, seed: int = 3) -> RealDraw:
    d = mod.draw(data, np.random.default_rng([seed, ep_i]), cfg, seed=seed, ep_i=ep_i)
    assert d is not None
    return d


def _same(a: RealDraw, b: RealDraw) -> bool:
    return (
        a.meta == b.meta
        and np.array_equal(a.x_train, b.x_train)
        and np.array_equal(a.x_test, b.x_test)
        and np.array_equal(a.y_train, b.y_train)
        and np.array_equal(a.y_test, b.y_test)
    )


# ---------------------------------------------------------------------------
# KAUST
# ---------------------------------------------------------------------------

_K_ROWS = 4000


def _kaust_file(root: Path, variant: str = "1a_2", n: int = _K_ROWS, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(size=(2, n))
    frame = pd.DataFrame({"id_train": np.arange(1, n + 1), "x": x, "y": y, "z": np.sin(6 * x) + rng.normal(0, 0.1, n)})
    frame.to_csv(root / f"{variant}_train.csv", index=False)
    return root


def _kaust_cfg(**kw: Any) -> KaustSpec:
    return dataclasses.replace(KaustSpec(n_context=40, n_query=20, window_cells=5), **kw)


def _kaust_data(tmp_path: Path) -> kaust.KaustData:
    return kaust.load_kaust(_kaust_file(tmp_path), "1a_2", expected_rows=_K_ROWS, verbose=False)


def test_kaust_loader_verifies_and_caches(tmp_path: Path) -> None:
    data = _kaust_data(tmp_path)
    assert data.n == _K_ROWS and np.all(np.diff(data.x) >= 0)
    assert len(list(tmp_path.glob("kaust_1a_2_clean_v1_*.npz"))) == 1
    again = kaust.load_kaust(tmp_path, "1a_2", expected_rows=_K_ROWS, verbose=False)
    assert np.array_equal(again.z, data.z)
    with pytest.raises(kaust.KaustDataError, match="rows, expected"):
        kaust.load_kaust(tmp_path, "1a_2", verbose=False)  # the real 90,000 count
    bad = pd.read_csv(tmp_path / "1a_2_train.csv")
    bad.loc[0, "x"] = 1.5
    bad.to_csv(tmp_path / "1a_1_train.csv", index=False)
    with pytest.raises(kaust.KaustDataError, match="outside"):
        kaust.load_kaust(tmp_path, "1a_1", expected_rows=_K_ROWS, verbose=False)


def test_kaust_windows_stay_in_their_strip_and_splits_are_disjoint(tmp_path: Path) -> None:
    data = _kaust_data(tmp_path)
    id_to_x = dict(zip(data.ids.tolist(), data.x.tolist()))
    for mask in kaust.MASKS:
        cfg = _kaust_cfg(mask=mask)
        for ep_i in range(15):
            d = _draw(kaust, data, cfg, ep_i)
            strip = ep_i % kaust.N_STRIPS
            ids = d.meta["context_ids"] + d.meta["query_ids"]
            assert not set(d.meta["context_ids"]) & set(d.meta["query_ids"])
            assert d.meta["fold"] == strip
            assert all(strip / 5 <= id_to_x[i] < (strip + 1) / 5 for i in ids)
            x0, y0, side = d.meta["window"]
            assert np.all((d.x_train[:, 0] >= x0 - 1e-9) & (d.x_train[:, 0] < x0 + side + 1e-9))
            assert d.meta["cluster"].startswith(f"cell{strip}_")  # window_cells=5: one cell column per strip


def test_kaust_block_queries_are_the_nearest_window_points(tmp_path: Path) -> None:
    data = _kaust_data(tmp_path)
    d = _draw(kaust, data, _kaust_cfg(mask="block"), 2)
    x0, y0, side = d.meta["window"]
    pool = data.window(x0, y0, side)
    anchor = int(np.flatnonzero(data.ids == d.meta["anchor_id"])[0])
    dist = np.hypot(data.x[pool] - data.x[anchor], data.y[pool] - data.y[anchor])
    radius = np.hypot(d.x_test[:, 0] - data.x[anchor], d.x_test[:, 1] - data.y[anchor]).max()
    assert (dist <= radius + 1e-12).sum() == len(d.y_test)


def test_kaust_query_targets_never_reach_context_or_features(tmp_path: Path) -> None:
    data = _kaust_data(tmp_path)
    cfg = _kaust_cfg()
    d = _draw(kaust, data, cfg, 4)
    z = data.z.copy()
    z[np.isin(data.ids, d.meta["query_ids"])] += 100.0
    d2 = _draw(kaust, dataclasses.replace(data, z=z), cfg, 4)
    assert np.array_equal(d.x_train, d2.x_train) and np.array_equal(d.x_test, d2.x_test)
    assert np.array_equal(d.y_train, d2.y_train) and d.meta == d2.meta
    assert np.allclose(d2.y_test, d.y_test + 100.0)


def test_kaust_is_deterministic_and_fingerprint_tracks_every_input(tmp_path: Path) -> None:
    data = _kaust_data(tmp_path)
    cfg = _kaust_cfg()
    assert _same(_draw(kaust, data, cfg, 7), _draw(kaust, data, cfg, 7))
    assert not _same(_draw(kaust, data, cfg, 7), _draw(kaust, data, cfg, 7, seed=4))
    base = kaust.fingerprint(data, cfg)
    for change in ({"mask": "block"}, {"n_context": 41}, {"n_query": 21}, {"window_cells": 10}, {"window_tries": 3}):
        assert kaust.fingerprint(data, dataclasses.replace(cfg, **change)) != base
    other = dataclasses.replace(data, identity={**data.identity, "file": "x"})
    assert kaust.fingerprint(other, cfg) != base


def test_kaust_skips_when_no_window_is_dense_enough(tmp_path: Path) -> None:
    data = _kaust_data(tmp_path)
    cfg = _kaust_cfg(n_context=900, window_tries=3)
    assert kaust.draw(data, np.random.default_rng([0, 0]), cfg, seed=0, ep_i=0) is None


# ---------------------------------------------------------------------------
# EMNIST
# ---------------------------------------------------------------------------

_E_N = 2 * emnist.N_CLASSES


def _idx(path: Path, magic: int, arr: np.ndarray) -> None:
    head = np.array([magic, *arr.shape], dtype=">u4").tobytes()
    with gzip.open(path, "wb") as fh:
        fh.write(head + arr.astype(np.uint8).tobytes())


def _emnist_files(root: Path, split: str = "test", seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    images = np.zeros((_E_N, 28, 28), dtype=np.uint8)
    images[:, 5:20, 10:14] = rng.integers(100, 256, size=(_E_N, 15, 4))
    # Stored transposed, as NIST distributes them.
    _idx(root / f"emnist-balanced-{split}-images-idx3-ubyte.gz", 2051, np.transpose(images, (0, 2, 1)))
    _idx(root / f"emnist-balanced-{split}-labels-idx1-ubyte.gz", 2049, np.tile(np.arange(emnist.N_CLASSES), 2))
    (root / "emnist-balanced-mapping.txt").write_text("".join(f"{i} {48 + i}\n" for i in range(emnist.N_CLASSES)))
    return root


def _emnist_data(tmp_path: Path) -> emnist.EmnistData:
    return emnist.load_emnist(_emnist_files(tmp_path), "test", expected_n=_E_N, verbose=False)


def test_emnist_loader_parses_idx_upright_and_rejects_bad_files(tmp_path: Path) -> None:
    data = _emnist_data(tmp_path)
    assert data.images.shape == (_E_N, 28, 28)
    # The vertical stroke at columns 10..13 comes back vertical (rows 5..19).
    assert data.images[0, 5:20, 10:14].min() >= 100 and data.images[0, :, 20:].max() == 0
    with pytest.raises(emnist.EmnistDataError, match="expected"):
        emnist.load_emnist(tmp_path, "test", verbose=False)
    _idx(tmp_path / "emnist-balanced-train-images-idx3-ubyte.gz", 9999, np.zeros((_E_N, 28, 28)))
    _idx(tmp_path / "emnist-balanced-train-labels-idx1-ubyte.gz", 2049, np.zeros(_E_N))
    with pytest.raises(emnist.EmnistDataError, match="magic"):
        emnist.load_emnist(tmp_path, "train", expected_n=_E_N, verbose=False)


def test_emnist_dequantized_targets_lie_in_the_unit_interval_and_bound_the_pixels(tmp_path: Path) -> None:
    data = _emnist_data(tmp_path)
    for mask in emnist.MASKS:
        cfg = EmnistSpec(n_context=50, n_query=64, mask=mask)
        for ep_i in range(10):
            d = _draw(emnist, data, cfg, ep_i)
            y = np.concatenate([d.y_train, d.y_test])
            assert np.all((y > 0) & (y < 1))
            pix = data.images[d.meta["image"]].reshape(-1)[d.meta["context_pixels"] + d.meta["query_pixels"]]
            assert np.array_equal(np.floor(y * 256).astype(int), pix)
            assert not set(d.meta["context_pixels"]) & set(d.meta["query_pixels"])
    assert emnist.bits_per_pixel(-np.log(256.0)) == pytest.approx(0.0)


def test_emnist_episodes_are_distinct_images_and_block_queries_are_nearest(tmp_path: Path) -> None:
    data = _emnist_data(tmp_path)
    cfg = EmnistSpec(n_context=50, n_query=64, mask="block")
    images = [_draw(emnist, data, cfg, i).meta["image"] for i in range(_E_N)]
    assert sorted(images) == list(range(_E_N))
    assert emnist.draw(data, np.random.default_rng([3, _E_N]), cfg, seed=3, ep_i=_E_N) is None
    d = _draw(emnist, data, cfg, 0)
    a = np.array(divmod(d.meta["anchor_pixel"], 28))
    radius = np.hypot(*(d.x_test - a).T).max()
    all_d = np.hypot(*(np.stack(np.divmod(np.arange(784), 28), 1) - a).T)
    assert (all_d < radius).sum() <= 64 <= (all_d <= radius).sum()


def test_emnist_query_pixels_never_reach_context_and_fingerprint_tracks_inputs(tmp_path: Path) -> None:
    data = _emnist_data(tmp_path)
    cfg = EmnistSpec(n_context=50, n_query=64)
    d = _draw(emnist, data, cfg, 1)
    imgs = data.images.copy().reshape(_E_N, -1)
    imgs[d.meta["image"], d.meta["query_pixels"]] = 255 - imgs[d.meta["image"], d.meta["query_pixels"]]
    d2 = _draw(emnist, dataclasses.replace(data, images=imgs.reshape(_E_N, 28, 28)), cfg, 1)
    assert np.array_equal(d.y_train, d2.y_train) and np.array_equal(d.x_test, d2.x_test) and d.meta == d2.meta
    assert _same(d, _draw(emnist, data, cfg, 1))
    base = emnist.fingerprint(data, cfg)
    for change in ({"mask": "block"}, {"n_context": 200}, {"n_query": 32}):
        assert emnist.fingerprint(data, dataclasses.replace(cfg, **change)) != base


# ---------------------------------------------------------------------------
# California
# ---------------------------------------------------------------------------


def _cal_bunch(n: int = 600, seed: int = 0) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    X[:, 6] = rng.uniform(32, 42, n)  # Latitude
    X[:, 7] = rng.uniform(-124, -114, n)  # Longitude
    y = rng.uniform(0.2, 4.9, n)
    y[:40] = 5.00001  # the top-code
    return SimpleNamespace(data=X, target=y, feature_names=list(california.FEATURE_NAMES))


def _cal_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> california.CaliforniaData:
    import sklearn.datasets

    monkeypatch.setattr(sklearn.datasets, "fetch_california_housing", lambda data_home: _cal_bunch())
    monkeypatch.setattr(california, "N_RAW", 600)
    return california.load(CaliforniaSpec(data_dir=str(tmp_path)), verbose=False)


def test_california_drops_exactly_the_censored_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _cal_data(tmp_path, monkeypatch)
    assert (data.n_raw, data.n_censored, len(data.y)) == (600, 40, 560)
    assert data.cap_value == pytest.approx(5.00001) and data.y.max() < 5.0
    X, y = np.zeros((4, 8)), np.array([1.0, 5.0, 5.00001, 5.00001])
    Xc, yc, n_cens, cap = california.clean(X, y, 5.0)
    assert (n_cens, cap) == (2, 5.00001) and yc.tolist() == [1.0, 5.0]  # $500,000 is kept
    with pytest.raises(california.CaliforniaDataError, match="no top-code"):
        california.clean(X, np.array([1.0, 2.0, 3.0, 4.0]), 5.0)


def test_california_feature_sets_share_episodes_and_block_uses_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _cal_data(tmp_path, monkeypatch)
    for mask in california.MASKS:
        f0 = _draw(california, data, CaliforniaSpec(n_context=128, n_query=64, mask=mask, features="f0"), 5)
        f1 = _draw(california, data, CaliforniaSpec(n_context=128, n_query=64, mask=mask, features="f1"), 5)
        assert f0.meta == f1.meta and np.array_equal(f0.y_test, f1.y_test)
        assert f0.x_train.shape[1] == 8 and f1.x_train.shape[1] == 6
        assert np.array_equal(f1.x_train, np.delete(f0.x_train, [6, 7], axis=1))
        assert not set(f0.meta["context_rows"]) & set(f0.meta["query_rows"])
    blk = _draw(california, data, CaliforniaSpec(mask="block"), 5)
    loc = data.X[:, 6:8]
    d = np.hypot(*(loc - loc[blk.meta["anchor_row"]]).T)
    assert set(np.argsort(d, kind="stable")[:64].tolist()) == set(blk.meta["query_rows"])
    assert blk.meta["cluster"] == f"anchor{blk.meta['anchor_row']}"


def test_california_query_targets_never_reach_context_and_fingerprint_tracks_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _cal_data(tmp_path, monkeypatch)
    cfg = CaliforniaSpec(mask="block")
    d = _draw(california, data, cfg, 2)
    y = data.y.copy()
    y[d.meta["query_rows"]] += 10.0
    d2 = _draw(california, dataclasses.replace(data, y=y), cfg, 2)
    assert np.array_equal(d.x_train, d2.x_train) and np.array_equal(d.y_train, d2.y_train) and d.meta == d2.meta
    base = california.fingerprint(data, cfg)
    for change in ({"mask": "random"}, {"features": "f1"}, {"n_context": 64}, {"n_query": 32}):
        assert california.fingerprint(data, dataclasses.replace(cfg, **change)) != base


# ---------------------------------------------------------------------------
# Meuse
# ---------------------------------------------------------------------------


def _meuse_data(tmp_path: Path) -> meuse.MeuseData:
    rng = np.random.default_rng(0)
    n = meuse.N_POINTS
    frame = pd.DataFrame(
        {
            "x": rng.uniform(178600, 181400, n),
            "y": rng.uniform(329700, 333600, n),
            "zinc": rng.uniform(113, 1839, n),
            "elev": rng.uniform(5, 10, n),
            "dist": rng.uniform(0, 0.9, n),
            "om": np.where(np.arange(n) < 2, np.nan, 5.0),
        }
    )
    frame.to_csv(tmp_path / "meuse.csv", index=False)
    return meuse.load(MeuseSpec(data_dir=str(tmp_path)), verbose=False)


def test_meuse_repeated_kfold_queries_every_point_equally(tmp_path: Path) -> None:
    data = _meuse_data(tmp_path)
    cfg = MeuseSpec(n_repeats=4)
    counts = np.zeros(meuse.N_POINTS, dtype=int)
    for ep_i in range(cfg.n_folds * cfg.n_repeats):
        d = _draw(meuse, data, cfg, ep_i)
        q = d.meta["query_points"]
        counts[q] += 1
        assert len(d.y_test) in (51, 52) and len(d.y_train) + len(d.y_test) == meuse.N_POINTS
        assert np.allclose(d.y_test, data.log_zinc[q])
    assert np.all(counts == cfg.n_repeats)
    assert meuse.draw(data, np.random.default_rng(0), cfg, seed=3, ep_i=12) is None
    assert meuse.episode_shape(cfg) == (103, 51)


def test_meuse_feature_sets_share_episodes_and_loader_rejects_bad_files(tmp_path: Path) -> None:
    data = _meuse_data(tmp_path)
    f0 = _draw(meuse, data, MeuseSpec(features="f0"), 4)
    f1 = _draw(meuse, data, MeuseSpec(features="f1"), 4)
    assert f0.meta == f1.meta and f1.x_train.shape[1] == 4 and np.array_equal(f1.x_train[:, :2], f0.x_train)
    frame = pd.read_csv(tmp_path / "meuse.csv").iloc[:-1]
    bad = tmp_path / "bad"
    bad.mkdir()
    frame.to_csv(bad / "meuse.csv", index=False)
    with pytest.raises(meuse.MeuseDataError, match="rows"):
        meuse.load(MeuseSpec(data_dir=str(bad)), verbose=False)
    base = meuse.fingerprint(data, MeuseSpec())
    for change in ({"features": "f1"}, {"n_folds": 5}, {"n_repeats": 10}):
        assert meuse.fingerprint(data, dataclasses.replace(MeuseSpec(), **change)) != base


# ---------------------------------------------------------------------------
# CTR23
# ---------------------------------------------------------------------------


def _ctr23_frame(n: int = 500, tied: bool = False, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 10, n).astype(float) if tied else rng.normal(size=n)
    y[3] = np.nan
    cat = pd.Categorical(rng.choice(["b", "a", "c"], n))
    num = rng.normal(size=n)
    num[5] = np.nan
    return pd.DataFrame({"f_num": num, "f_cat": cat, "f_bool": rng.integers(0, 2, n).astype(bool), "rings": y})


def _ctr23_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tied: bool = False) -> ctr23.Ctr23Data:
    import sklearn.datasets

    frame = _ctr23_frame(tied=tied)
    monkeypatch.setattr(
        sklearn.datasets,
        "fetch_openml",
        lambda **kw: SimpleNamespace(frame=frame, details={"name": "abalone", "version": "1"}),
    )
    return ctr23.load(Ctr23Spec(data_dir=str(tmp_path), data_id=44956), verbose=False)


def test_ctr23_preprocessing_matches_tabicl_and_the_atom_rule_excludes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _ctr23_load(tmp_path, monkeypatch)
    assert data.n_missing_target == 1 and len(data.y) == 499 and data.X.shape == (499, 3)
    assert set(np.unique(data.X[:, 1]).tolist()) == {0.0, 1.0, 2.0}  # ordinal, sorted categories
    assert np.isfinite(data.X).all()
    assert ctr23.tie_fraction(np.array([1.0, 1.0, 2.0, 3.0])) == pytest.approx(0.5)
    assert ctr23.atom_fraction(np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])) == pytest.approx(0.5)
    # Rounded but continuous: most rows repeat a value, yet no value holds more than 5 % of rows.
    rounded = np.round(np.random.default_rng(1).normal(size=2000), 2)
    assert ctr23.tie_fraction(rounded) > 0.5 and ctr23.atom_fraction(rounded) < 0.05
    with pytest.raises(ctr23.Ctr23Excluded, match="excluded"):
        _ctr23_load(tmp_path, monkeypatch, tied=True)


def test_ctr23_dyadic_knn_takes_the_neighbours_of_two_anchors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _ctr23_load(tmp_path, monkeypatch)
    cfg = Ctr23Spec(data_id=44956, n_context=128, n_query=64)
    d = _draw(ctr23, data, cfg, 0)
    q = np.array(d.meta["query_rows"])
    a1, a2 = d.meta["anchor_rows"]
    assert len(set(q.tolist())) == 64 and not set(q.tolist()) & set(d.meta["context_rows"])
    d1 = ((data.Xs - data.Xs[a1]) ** 2).sum(1)
    assert set(q[:32].tolist()) == set(np.argsort(d1, kind="stable")[:32].tolist())
    assert a2 not in q[:32]
    d2 = ((data.Xs - data.Xs[a2]) ** 2).sum(1)
    d2[q[:32]] = np.inf
    assert set(q[32:].tolist()) == set(np.argsort(d2, kind="stable")[:32].tolist())
    y = data.y.copy()
    y[q] += 5.0
    d3 = _draw(ctr23, dataclasses.replace(data, y=y), cfg, 0)
    assert np.array_equal(d.x_train, d3.x_train) and np.array_equal(d.y_train, d3.y_train) and d.meta == d3.meta
    assert _same(d, _draw(ctr23, data, cfg, 0))
    base = ctr23.fingerprint(data, cfg)
    for change in ({"mask": "random"}, {"n_context": 64}, {"max_atom_frac": 0.1}):
        assert ctr23.fingerprint(data, dataclasses.replace(cfg, **change)) != base


# ---------------------------------------------------------------------------
# Config, paired comparison, calibration gate
# ---------------------------------------------------------------------------


def test_every_real_dataset_composes_and_bad_fields_are_rejected() -> None:
    for name, extra in {
        "kaust": [],
        "emnist": [],
        "california": [],
        "meuse": [],
        "ctr23": ["real.ctr23.data_id=44956"],
    }.items():
        compose_eval_spec(["ckpt=x.pt", f"real.dataset={name}", *extra])
    bad = {
        "real.dataset=kaust real.kaust.variant=2a_1": "real.kaust.variant",
        "real.dataset=kaust real.kaust.window_cells=12": "window_cells",
        "real.dataset=kaust real.kaust.mask=dyadic_knn": "real.kaust.mask",
        "real.dataset=emnist real.emnist.split=val": "real.emnist.split",
        "real.dataset=emnist real.emnist.n_context=800": "784",
        "real.dataset=california real.california.features=f2": "real.california.features",
        "real.dataset=meuse real.meuse.features=f2": "real.meuse.features",
        "real.dataset=meuse real.meuse.n_folds=20": "min_test_points",
        "real.dataset=ctr23": "real.ctr23.data_id",
        "real.dataset=ctr23 real.ctr23.data_id=44956 real.ctr23.mask=block": "real.ctr23.mask",
        "real.dataset=ctr23 real.ctr23.data_id=44956 real.ctr23.n_query=8": "min_test_points",
    }
    for overrides, message in bad.items():
        with pytest.raises(ValueError, match=message):
            compose_eval_spec(["ckpt=x.pt", *overrides.split()])


def _dump(path: Path, dataset: str, eps: list[dict[str, Any]]) -> str:
    path.write_text(json.dumps({"real_dataset": dataset, "episodes": eps}))
    return str(path)


def _ep(i: int, icl: float, ind: float, name: str = "d") -> dict[str, Any]:
    rows = {
        "icl": {"total": icl, "marginal": 0.5, "copula": icl - 0.5},
        "independence_marginal": {"total": ind, "marginal": 0.5, "copula": ind - 0.5},
    }
    return {
        "ep_i": i,
        "episode": {"cluster": f"rep{i // 3}", "name": name},
        "nlls": {},
        "total_nlls": rows,
        "z_test": [0.1, -0.2],
    }


def test_meuse_comparisons_report_sd_without_ci(tmp_path: Path) -> None:
    from eval.runners.paired_compare import PairedCompareSpec, within_run_deltas

    a = _dump(tmp_path / "m.json", "meuse", [_ep(i, 1.0 - 0.01 * i, 1.0) for i in range(9)])
    out = within_run_deltas(PairedCompareSpec(a=a))["copula - independence"]["total"]
    assert np.isnan(out["lo"]) and np.isnan(out["hi"]) and out["sd"] > 0 and out["n_clusters"] == 0
    k = _dump(tmp_path / "k.json", "kaust", [_ep(i, 1.0 - 0.01 * i, 1.0) for i in range(9)])
    out_k = within_run_deltas(PairedCompareSpec(a=k))["copula - independence"]["total"]
    assert np.isfinite(out_k["lo"]) and out_k["n_clusters"] == 3


def test_across_datasets_runs_a_wilcoxon_on_per_dataset_means(tmp_path: Path) -> None:
    from eval.runners.paired_compare import PairedCompareSpec, across_datasets

    paths = [
        _dump(tmp_path / f"d{j}.json", "ctr23", [_ep(i, 1.0 - 0.01 * (j + 1), 1.0, f"ds{j}") for i in range(4)])
        for j in range(8)
    ]
    out = across_datasets(PairedCompareSpec(across=paths))["copula - independence / total"]
    assert out["n_datasets"] == 8 and out["n_better"] == 8 and out["p_value"] < 0.01


def test_calibration_gate_flags_a_miscalibrated_marginal() -> None:
    from eval.runners.real_calibration_gate import gate_stats

    rng = np.random.default_rng(0)
    good = gate_stats(rng.normal(size=20_000))
    assert not good["flag"] and good["quantile_ece"] < 0.01 and good["ks_stat"] < 0.02
    bad = gate_stats(1.6 * rng.normal(size=20_000))
    assert bad["flag"] and bad["frac_abs_gt3"] > 0.05 and bad["quantile_ece"] > 0.05


def test_paired_compare_merges_shards_and_rejects_duplicates(tmp_path: Path) -> None:
    from eval.runners.paired_compare import PairedCompareSpec, within_run_deltas

    _dump(tmp_path / "c_s0.json", "kaust", [_ep(i, 0.9, 1.0) for i in range(3)])
    _dump(tmp_path / "c_s1.json", "kaust", [_ep(i, 0.9, 1.0) for i in range(3, 6)])
    out = within_run_deltas(PairedCompareSpec(a=str(tmp_path / "c_s*.json")))["copula - independence"]["total"]
    assert out["n"] == 6 and out["mean"] == pytest.approx(-0.1)
    _dump(tmp_path / "c_s2.json", "kaust", [_ep(0, 0.9, 1.0)])
    with pytest.raises(ValueError, match="two shards"):
        within_run_deltas(PairedCompareSpec(a=str(tmp_path / "c_s*.json")))
