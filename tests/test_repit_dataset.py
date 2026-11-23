"""PIT overlays: copula_inter.repit_dataset writes them, CopulaDataset reads them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

import copula_inter.repit_dataset as repit
from copula_inter.data.dataset import PIT_OVERLAY_META, CopulaDataset


def _episode(P: int, N: int, d: int, offset: float) -> dict[str, Tensor]:
    return {
        "x_norm_train": torch.randn(P, d),
        "x_norm_test": torch.randn(N, d),
        "y_train": torch.arange(P, dtype=torch.float32) + offset,
        "y_test": torch.arange(N, dtype=torch.float32) - offset,
        "z_train": torch.zeros(P),
        "z_test": torch.zeros(N),
        "log_pdf_test": torch.zeros(N),
        "mu_star": torch.zeros(N),
        "sigma_star": torch.ones(N),
        "n_train": torch.tensor(P),
        "n_test": torch.tensor(N),
    }


def _fake_pit(
    x_norm_train: Tensor, y_train: Tensor, x_norm_test: Tensor, y_test: Tensor, device: str, **_: Any
) -> tuple[Tensor, Tensor, Tensor]:
    """A marginal whose PIT is the target itself, so each output can be traced to its episode."""
    return y_train * 2, y_test * 3, y_test * 0 - 1


def test_repit_shard_keeps_episode_order_across_shapes_and_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repit, "marginal_pit_tables", _fake_pit)
    episodes = [_episode(5, 3, 2, 0.0), _episode(4, 2, 2, 10.0), _episode(5, 3, 2, 20.0), _episode(5, 3, 2, 30.0)]
    out = repit.repit_shard(episodes, "cpu", seed=0, k_folds=2, marginal_backend="stub", max_chunk=2)
    assert len(out) == len(episodes)
    for ep, entry in zip(episodes, out):
        assert torch.equal(entry["z_train"], ep["y_train"] * 2)
        assert torch.equal(entry["z_test"], ep["y_test"] * 3)
        assert torch.equal(entry["log_pdf_test"], torch.full_like(ep["y_test"], -1.0))


def _write_dataset(root: Path, n_shards: int, shard_size: int, digest: str) -> Path:
    pit = root / "pit"
    pit.mkdir(parents=True)
    for s in range(n_shards):
        torch.save([_episode(6, 4, 3, 100.0 * s + i) for i in range(shard_size)], pit / f"shard_{s:06d}.pt")
    torch.save({"n_total": n_shards * shard_size, "shard_size": shard_size}, pit / "meta.pt")
    (pit / "manifest.json").write_text(json.dumps({"digest": digest, "spec": {}}))
    return pit


def _write_overlay(root: Path, source: Path, n_shards: int, shard_size: int, digest: str) -> Path:
    out = root / "overlay" / "pit"
    out.mkdir(parents=True)
    for s in range(n_shards):
        episodes = torch.load(source / f"shard_{s:06d}.pt", weights_only=False)
        torch.save(repit.repit_shard(episodes, "cpu", seed=s, k_folds=2), out / f"shard_{s:06d}.pt")
    (out / PIT_OVERLAY_META).write_text(json.dumps({"source_digest": digest, "shard_size": shard_size}))
    return out


def test_overlay_replaces_the_pit_and_keeps_the_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repit, "marginal_pit_tables", _fake_pit)
    source = _write_dataset(tmp_path, n_shards=2, shard_size=3, digest="abc")
    overlay = _write_overlay(tmp_path, source, n_shards=2, shard_size=3, digest="abc")
    plain = CopulaDataset(episode_dir=str(source))
    over = CopulaDataset(episode_dir=str(source), pit_overlay_dir=str(overlay))
    assert len(over) == len(plain) == 6
    for i in range(6):
        a, b = plain[i], over[i]
        assert torch.equal(a["y_train"], b["y_train"]) and torch.equal(a["x_norm_test"], b["x_norm_test"])
        assert torch.equal(b["z_train"], a["y_train"] * 2)
        assert torch.equal(b["z_test"], a["y_test"] * 3)


def test_overlay_of_another_dataset_or_incomplete_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repit, "marginal_pit_tables", _fake_pit)
    source = _write_dataset(tmp_path, n_shards=2, shard_size=3, digest="abc")
    overlay = _write_overlay(tmp_path, source, n_shards=2, shard_size=3, digest="other")
    with pytest.raises(ValueError, match="another dataset"):
        CopulaDataset(episode_dir=str(source), pit_overlay_dir=str(overlay))
    (overlay / PIT_OVERLAY_META).write_text(json.dumps({"source_digest": "abc", "shard_size": 3}))
    (overlay / "shard_000001.pt").unlink()
    with pytest.raises(ValueError, match="lacks 1 of 2 shards"):
        CopulaDataset(episode_dir=str(source), pit_overlay_dir=str(overlay))


def test_overlay_meta_refuses_a_second_marginal(tmp_path: Path) -> None:
    spec = {"source_digest": "abc", "z_train_source": "kumo_small"}
    repit._ensure_overlay_meta(str(tmp_path), spec)
    repit._ensure_overlay_meta(str(tmp_path), dict(spec))
    with pytest.raises(ValueError, match="another re-PIT"):
        repit._ensure_overlay_meta(str(tmp_path), {**spec, "z_train_source": "tabpfn35_fast"})


def test_manifest_identity_ignores_the_repit_source() -> None:
    from omegaconf import OmegaConf

    from copula_inter.data.manifest import generation_spec

    base = OmegaConf.create({"seed": 1, "data": {"P_min": 4, "z_train_source": "analytic"}})
    with_source = OmegaConf.create(
        {"seed": 1, "data": {"P_min": 4, "z_train_source": "analytic", "repit_source_dir": "x"}}
    )
    assert generation_spec(base, None) == generation_spec(with_source, None)


def test_max_shards_keeps_the_first_shards_and_their_overlay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repit, "marginal_pit_tables", _fake_pit)
    source = _write_dataset(tmp_path, n_shards=3, shard_size=2, digest="abc")
    overlay = _write_overlay(tmp_path, source, n_shards=3, shard_size=2, digest="abc")
    (overlay / "shard_000002.pt").unlink()  # an overlay of the first 2 shards only
    first = CopulaDataset(episode_dir=str(source), pit_overlay_dir=str(overlay), max_shards=2)
    assert len(first) == 4
    assert torch.equal(first[3]["z_train"], first[3]["y_train"] * 2)
    with pytest.raises(ValueError, match="lacks 1 of 3 shards"):
        CopulaDataset(episode_dir=str(source), pit_overlay_dir=str(overlay))


def test_float32_overlay_identity_has_no_precision_key(tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    source = _write_dataset(tmp_path, n_shards=1, shard_size=1, digest="abc")
    (source / "manifest.json").write_text(
        json.dumps({"digest": "abc", "spec": {"data": {"n_tasks": 1, "shard_size": 1}}})
    )
    base = {"data": {"z_train_source": "tabicl", "z_train_tabicl_k_folds": 5}, "training": {}}
    fp32 = repit.overlay_spec(OmegaConf.create(base), str(source), "ckpt")
    amp = repit.overlay_spec(
        OmegaConf.create({**base, "training": {"tabicl_inference_amp": True}}), str(source), "ckpt"
    )
    assert "tabicl_inference_amp" not in fp32
    assert amp == {**fp32, "tabicl_inference_amp": True}
