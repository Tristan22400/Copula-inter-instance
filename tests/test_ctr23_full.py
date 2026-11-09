"""The full-dataset CTR23 protocol: official split mapping, query partitions, chunk copula NLL and marginal metrics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from eval.data.real import ctr23
from eval.runners.ctr23_full_eval import chunk_copula_nll, quantile_metrics
from eval.runners.eval_args import Ctr23Spec


def _load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int = 200) -> ctr23.Ctr23Data:
    import sklearn.datasets

    rng = np.random.default_rng(0)
    y = rng.normal(size=n)
    y[3] = np.nan  # dropped: absent from the kept rows and from both sides of every fold
    frame = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "rings": y})
    monkeypatch.setattr(
        sklearn.datasets, "fetch_openml", lambda **kw: SimpleNamespace(frame=frame, details={"name": "abalone"})
    )
    return ctr23.load(Ctr23Spec(data_dir=str(tmp_path), data_id=44956), verbose=False)


def _write_arff(tmp_path: Path, folds: list[np.ndarray], n: int, extra: str = "") -> None:
    lines = ["@relation x_splits", "", "@attribute type {TRAIN,TEST}", "@attribute rowid numeric"]
    lines += ["@attribute repeat numeric", "@attribute fold numeric", "", "@data"]
    for k, test in enumerate(folds):
        for r in range(n):
            lines.append(f"{'TEST' if r in set(test.tolist()) else 'TRAIN'},{r},0,{k}")
        lines.append(f"TRAIN,0,1,{k}")  # repeat 1 is ignored
    (tmp_path / "ctr23_44956_task361234_splits.arff").write_text("\n".join(lines) + extra + "\n")


def test_openml_folds_map_row_ids_to_kept_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _load(tmp_path, monkeypatch)
    assert len(data.y) == 199 and 3 not in data.row_ids.tolist()
    rows = np.random.default_rng(1).permutation(200)
    _write_arff(tmp_path, np.array_split(rows, 5), 200)
    folds = ctr23.openml_folds(data, str(tmp_path))
    assert len(folds) == 5
    covered = np.concatenate([te for _, te in folds])
    assert sorted(covered.tolist()) == list(range(199))
    for (tr, te), raw in zip(folds, np.array_split(rows, 5)):
        assert np.intersect1d(tr, te).size == 0 and len(tr) + len(te) == 199
        # Kept index i holds OpenML row row_ids[i].
        assert sorted(data.row_ids[te].tolist()) == sorted(r for r in raw.tolist() if r != 3)


def test_openml_folds_reject_a_test_fold_that_misses_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _load(tmp_path, monkeypatch)
    rows = np.arange(200)
    _write_arff(tmp_path, [rows[:50], rows[60:200]], 200)  # rows 50..59 are never tested
    with pytest.raises(ctr23.Ctr23DataError, match="exactly once"):
        ctr23.openml_folds(data, str(tmp_path))


@pytest.mark.parametrize("n", [1, 63, 64, 65, 300])
def test_partitions_cover_every_row_once(n: int) -> None:
    rng = np.random.default_rng(0)
    Xs = rng.normal(size=(n, 3))
    rows = np.arange(n)
    for chunks in (ctr23.random_partition(rng, rows, 64), ctr23.dyadic_knn_partition(rng, Xs, rows, 64)):
        assert sorted(np.concatenate(chunks).tolist()) == rows.tolist()
        assert all(1 <= len(c) <= 64 for c in chunks)
    sizes = [len(c) for c in ctr23.random_partition(rng, rows, 64)]
    assert max(sizes) - min(sizes) <= 1


def test_dyadic_partition_chunks_are_two_tight_clusters() -> None:
    rng = np.random.default_rng(0)
    Xs = rng.normal(size=(640, 2))
    rows = np.arange(640)

    def spread(chunks: list[np.ndarray]) -> float:
        # Mean distance of each half-chunk's rows to the half's centroid.
        halves = [h for c in chunks[:-1] for h in (c[:32], c[32:])]
        return float(np.mean([np.linalg.norm(Xs[h] - Xs[h].mean(0), axis=1).mean() for h in halves]))

    assert spread(ctr23.dyadic_knn_partition(rng, Xs, rows, 64)) < 0.5 * spread(ctr23.random_partition(rng, rows, 64))


def test_chunk_copula_nll_matches_the_dense_gaussian_copula() -> None:
    g = torch.Generator().manual_seed(0)
    N, r = 40, 3
    W = torch.randn(N, r, generator=g, dtype=torch.float64)
    Draw = torch.rand(N, generator=g, dtype=torch.float64) + 0.2
    c = (W * W).sum(-1) + Draw
    U, D = W / c.sqrt()[:, None], Draw / c  # unit diagonal
    z = torch.randn(N, generator=g, dtype=torch.float64)
    S = U @ U.T + torch.diag(D)
    dense = 0.5 * (torch.logdet(S) + z @ torch.linalg.solve(S, z) - z @ z)
    assert chunk_copula_nll(U, D, z) == pytest.approx(float(dense), rel=1e-9)
    # A chunk's NLL is the copula's joint over that chunk: Sigma's sub-block.
    idx = torch.tensor([1, 5, 7, 30])
    Ss = S[idx][:, idx]
    zs = z[idx]
    sub = 0.5 * (torch.logdet(Ss) + zs @ torch.linalg.solve(Ss, zs) - zs @ zs)
    assert chunk_copula_nll(U[idx], D[idx], zs) == pytest.approx(float(sub), rel=1e-9)
    # Independence: zero.
    assert chunk_copula_nll(torch.zeros(N, r, dtype=torch.float64), torch.ones(N, dtype=torch.float64), z) == (
        pytest.approx(0.0, abs=1e-12)
    )


def test_quantile_metrics_of_a_standard_normal_predictive() -> None:
    from scipy.stats import norm

    levels = np.arange(1, 1000) / 1000.0
    y = np.array([-1.5, 0.0, 0.3, 2.0])
    q = np.tile(norm.ppf(levels), (len(y), 1))
    m = quantile_metrics(q, levels, y)
    crps = y * (2 * norm.cdf(y) - 1) + 2 * norm.pdf(y) - 1 / np.sqrt(np.pi)
    assert m["crps"] == pytest.approx(float(crps.mean()), abs=5e-3)
    assert m["rmse"] == pytest.approx(float(np.sqrt((y**2).mean())), abs=1e-3)
    assert m["mae_median"] == pytest.approx(float(np.abs(y).mean()), abs=1e-3)


def _blob(data_id: int, deltas: list[float], n_folds: int = 3) -> dict:
    folds = {
        str(k): {
            "marginal_nll": 1.0,
            "rmse": 0.5,
            "r2": 0.9,
            "mae_median": 0.4,
            "crps": 0.3,
            "z_test_std": 1.0,
            "mean_abs_offdiag_corr": 0.1,
            **{f"copula_{t}": v for t in ("whole", "random", "dyadic_knn")},
        }
        for k, v in enumerate(deltas)
    }
    return {"dataset": {"data_id": data_id, "name": f"d{data_id}", "n_rows": 100, "n_folds": n_folds}, "folds": folds}


def test_compare_with_independence_ranks_and_tests_across_datasets() -> None:
    from eval.runners.ctr23_full_eval import compare_with_independence

    blobs = [_blob(i, [-0.1 * (i + 1), -0.05, 0.01]) for i in range(5)] + [_blob(9, [0.2, 0.2, 0.2])]
    blobs.append(_blob(10, [-1.0], n_folds=3))  # incomplete: left out by default
    out = compare_with_independence(blobs)
    assert [r["data_id"] for r in out["per_dataset"]] == [0, 1, 2, 3, 4, 9]
    s = out["summary"]["whole"]
    assert (s["n_datasets"], s["copula_wins"], s["independence_wins"]) == (6, 5, 1)
    assert s["mean_rank_copula"] == pytest.approx((5 * 1 + 2) / 6)
    assert s["median_rank_copula"] == 1.0 and s["median_rank_independence"] == 2.0
    assert out["per_dataset"][0]["fold_win_rate_random"] == pytest.approx(2 / 3)
    assert 0 < s["p_value"] < 1
    assert len(compare_with_independence(blobs, include_partial=True)["per_dataset"]) == 7
