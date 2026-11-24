"""Synthetic grid scoring and paired query-size comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from eval.runners import grid_score
from eval.runners.grid_score import GridScoreSpec, grid_cells, oracle_nll
from eval.runners.hydra_cli import compose_spec


def test_grid_oracle_nll_is_the_gaussian_nll_per_row() -> None:
    torch.manual_seed(0)
    N = 6
    A = torch.randn(N, N, dtype=torch.float64)
    C = A @ A.T + N * torch.eye(N, dtype=torch.float64)
    sigma = C.diagonal().sqrt()
    R = C / (sigma[:, None] * sigma[None, :])
    mu, y = torch.randn(N, dtype=torch.float64), torch.randn(N, dtype=torch.float64)
    expected = -torch.distributions.MultivariateNormal(mu, C).log_prob(y) / N
    assert torch.allclose(oracle_nll(y[None], mu[None], sigma[None], R[None])[0], expected)


def test_grid_discovers_fixed_and_variable_cells(tmp_path: Path) -> None:
    for cell in ("axis/P100", "axis/P32", "axis/variable"):
        (tmp_path / cell / "pit").mkdir(parents=True)
    assert grid_cells(str(tmp_path), []) == ["axis/P32", "axis/P100", "axis/variable"]
    assert grid_cells(str(tmp_path), ["axis/variable"]) == ["axis/variable"]


def test_grid_reads_the_exact_episode_on_disk(tmp_path: Path) -> None:
    vanished = tmp_path / "task_000000.pt"
    torch.save({"z_test": torch.ones(1)}, vanished)
    torch.save({"z_test": torch.ones(1)}, tmp_path / "task_000001.pt")
    dataset = grid_score.CopulaDataset(episode_dir=str(tmp_path))
    vanished.unlink()
    assert dataset[0]["z_test"].item() == 1
    with pytest.raises(FileNotFoundError):
        dataset.get_exact(0)


class _IndependentModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.queries: dict[tuple[int, int], torch.Tensor] = {}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for x, y in zip(batch["x_test"], batch["y_test"], strict=True):
            self.queries[(int(y[0]), len(x))] = x[:, 0]
        return {"W": torch.zeros_like(batch["z_test"][..., None]), "s": torch.zeros_like(batch["z_test"])}


def test_query_sweep_keeps_episodes_paired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cell = tmp_path / "random/variable/pit"
    cell.mkdir(parents=True)
    # Splitting by P gives non-contiguous batch groups. The two models use different batch sizes below.
    for i, (p, n) in enumerate(((2, 5), (3, 4), (2, 5), (2, 2))):
        torch.save(
            {
                "n_train": p,
                "n_test": n,
                "x_norm_train": torch.zeros(p, 1),
                "x_norm_test": torch.arange(n).unsqueeze(-1).float(),
                "y_train": torch.zeros(p),
                "y_test": torch.full((n,), float(i)),
                "z_train": torch.zeros(p),
                "z_test": torch.zeros(n),
                "log_pdf_test": torch.full((n,), -float(i)),
                "mu_star": torch.zeros(n),
                "sigma_star": torch.ones(n),
                "mu_post": torch.zeros(n),
                "sigma_post": torch.ones(n),
                "R_post": torch.eye(n),
            },
            cell / f"task_{i:06d}.pt",
        )
    dataset = grid_score.CopulaDataset(episode_dir=str(cell))
    model = _IndependentModel()
    cfg = OmegaConf.create({"model": {}})
    first = grid_score.score_cell(model, cfg, dataset, "cpu", 3, [0, 1, 2], query_size=4)
    second = grid_score.score_cell(model, cfg, dataset, "cpu", 1, [0, 1, 2], query_size=4)
    assert first["total"].tolist() == pytest.approx([0.0, 1.0, 2.0], abs=1e-6)
    assert second["total"].tolist() == pytest.approx(first["total"].tolist(), abs=1e-6)

    output = tmp_path / "scores.json"
    monkeypatch.setattr(grid_score, "load_copula_model", lambda *args, **kwargs: (model, cfg))
    spec = compose_spec(
        "grid_score",
        GridScoreSpec,
        [
            f"grid_dir={tmp_path}",
            "ckpts=[a.pt,b.pt]",
            "pit_overlay_roots=[null,null]",
            "query_sizes=[2,4]",
            f"out_json={output}",
            "device=cpu",
        ],
    )
    grid_score.run(spec)
    result = json.loads(output.read_text())
    assert result["pit_overlay_roots"] == {"a.pt": None, "b.pt": None}
    for n in (2, 4):
        row = result["cells"][f"random/variable/N{n}"]
        assert row["a.pt"]["n"] == row["b.pt"]["n"] == 3
        assert row["b.pt"]["total_minus_ref"] == 0
    assert all(set(model.queries[i, 2].tolist()) <= set(model.queries[i, 4].tolist()) for i in range(3))
