"""Fixed-pool versions of the four Cakmak et al. BoRisk experiments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from copula_inter.artifacts import artifact_identity, atomic_write, canonical_digest
from copula_inter.paths import REPO_ROOT
from eval.risk_bo import WEIGHTS, RiskPool, branin_williams, scenario_inputs

PROBLEMS = ("branin-williams", "f6", "portfolio", "covid-testing")
UPSTREAM_REVISION = "88f87bc4640a4d8127ad3e170e094e13a78b7fca"


@dataclass(frozen=True)
class Benchmark:
    inputs: Tensor
    weights: Tensor
    alpha: float
    measure: Literal["var", "cvar"]
    problem: str


def _sobol(n: int, dim: int, seed: int) -> Tensor:
    return torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed).draw(n, dtype=torch.float64)


def _cross(designs: Tensor, scenarios: Tensor) -> Tensor:
    return torch.cat(
        (designs[:, None, :].expand(-1, len(scenarios), -1), scenarios[None].expand(len(designs), -1, -1)), dim=-1
    )


def benchmark_design(problem: str, pool_size: int, pool_seed: int, scenario_seed: int) -> Benchmark:
    """Return the fixed design/scenario grid in the evaluator's input units."""
    if problem == "branin-williams":
        x = scenario_inputs(_sobol(pool_size, 2, pool_seed))
        return Benchmark(x, x.new_tensor(WEIGHTS), 0.7, "cvar", problem)
    if problem == "f6":
        x = _cross(10 * _sobol(pool_size, 4, pool_seed) - 5, 4 * _sobol(8, 3, scenario_seed) - 2)
        return Benchmark(x, x.new_full((8,), 1 / 8), 0.75, "cvar", problem)
    if problem == "portfolio":
        # The released portfolio oracle takes unit-cube coordinates, with the
        # last two dimensions mapping to the published transaction cost ranges.
        x = _cross(_sobol(pool_size, 3, pool_seed), _sobol(10, 2, scenario_seed))
        return Benchmark(x, x.new_full((10,), 1 / 10), 0.8, "var", problem)
    if problem == "covid-testing":
        x = _sobol(pool_size, 2, pool_seed).sort(dim=-1).values
        # Two successive spacings give a uniform design on the allocation simplex.
        x = torch.stack((x[:, 0], x[:, 1] - x[:, 0]), dim=-1)
        indices = list(product(range(3), repeat=3))
        levels = ((0.001, 0.0025, 0.004), (0.002, 0.004, 0.006), (0.002, 0.005, 0.008))
        scenarios = x.new_tensor([[levels[j][i] for j, i in enumerate(row)] for row in indices])
        # The authors' runner uses 2**(number of middle levels) / 64.
        weights = x.new_tensor([2 ** sum(i == 1 for i in row) / 64 for row in indices])
        return Benchmark(_cross(x, scenarios), weights, 0.9, "cvar", problem)
    raise ValueError(f"Unknown risk BO problem {problem!r}")


def marzat_f6(inputs: Tensor) -> Tensor:
    """Evaluate the released BoRisk implementation of Marzat's f6."""
    a, b, c, d, u, v, w = inputs.unbind(-1)
    return (
        u * (a.square() - b + c - d + 2)
        + v * (-a + 2 * b.square() - c.square() + 2 * d + 1)
        + w * (2 * a - b + 2 * c - d.square() + 5)
        + 5 * a.square()
        + 4 * b.square()
        + 3 * c.square()
        + 2 * d.square()
        - u.square()
        - v.square()
        - w.square()
    )


def _require_files(paths: list[Path]) -> None:
    missing = next((path for path in paths if not path.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(f"Missing {missing}. Run scripts/fetch_borisk_assets.sh and set asset_dir.")


def _portfolio_files(root: Path) -> list[Path]:
    return [root / "portfolio_surrogate_state_dict.pt"] + [
        root / "port_evals" / f"port_n=100_seed={i}" for i in range(1, 31)
    ]


@torch.no_grad()
def _portfolio_losses(inputs: Tensor, paths: list[Path]) -> Tensor:
    from gpytorch.kernels import MaternKernel, ScaleKernel

    # Older SingleTaskGP releases used a Matérn-5/2 ARD kernel with softplus
    # transforms. The released Gaussian likelihood used raw_noise directly.
    state = torch.load(paths[0], weights_only=True, map_location="cpu")
    parts = [torch.load(path, weights_only=True, map_location="cpu") for path in paths[1:]]
    train = torch.cat([p["X"] for p in parts]).reshape(-1, 5).float()
    y = torch.cat([p["Y"] for p in parts]).flatten().float()
    kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=5))
    kernel.raw_outputscale.copy_(state["covar_module.raw_outputscale"])
    kernel.base_kernel.raw_lengthscale.copy_(state["covar_module.base_kernel.raw_lengthscale"])
    mean = state["mean_module.constant"].reshape(())
    y_mean = state["outcome_transform.means"].reshape(())
    y_std = state["outcome_transform.stdvs"].reshape(())
    covariance = kernel(train).to_dense()
    covariance.diagonal().add_(state["likelihood.noise_covar.raw_noise"].item())
    chol = torch.linalg.cholesky(covariance)
    coeff = torch.cholesky_solve(((y - y_mean) / y_std - mean)[:, None], chol)
    posterior = [
        (kernel(query, train).to_dense() @ coeff).flatten() * y_std + mean * y_std + y_mean
        for query in inputs.flatten(0, 1).float().split(512)
    ]
    # Upstream port_eval.py records the base simulator's negative-return loss.
    return torch.cat(posterior).reshape(inputs.shape[:2]).to(inputs)


def _covid_losses(inputs: Tensor, assets: Path, python: str | None, cache_dir: Path) -> tuple[Tensor, object]:
    root = assets / "upstream"
    files = sorted(root.rglob("*.py"))
    _require_files([root / "BoRisk/test_functions/covid_exp_class.py"])
    key = canonical_digest(
        {
            "inputs": inputs.tolist(),
            "seeds": list(range(1, 11)),
            "revision": UPSTREAM_REVISION,
            "code": {str(p.relative_to(root)): artifact_identity(p) for p in files},
        }
    )
    path = cache_dir / f"covid-testing-{key}.json"
    if not path.is_file():
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(root), str(assets / "python-deps"), str(REPO_ROOT), env.get("PYTHONPATH")])
        )
        env.setdefault("MPLBACKEND", "Agg")
        result = subprocess.run(
            [python or sys.executable, "-m", "eval.runners.risk_bo_covid"],
            input=json.dumps(inputs.tolist()),
            text=True,
            stdout=subprocess.PIPE,
            env=env,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                "Testing-allocation evaluator failed. See its stderr and check covid_python dependencies."
            )
        values = json.loads(result.stdout)
        array = inputs.new_tensor(values)
        if array.shape != inputs.shape[:2] or not torch.isfinite(array).all():
            raise ValueError("Invalid simulator response for testing allocation")

        def write(tmp: str) -> None:
            Path(tmp).write_text(json.dumps(values, allow_nan=False))

        atomic_write(path, write)
    return inputs.new_tensor(json.loads(path.read_text())), {"source": key, "table": artifact_identity(path)}


def prepare_problem(
    benchmark: Benchmark, assets: Path, covid_python: str | None, cache_dir: Path, device: str
) -> tuple[RiskPool, object]:
    """Prepare pool outcomes once. BO only receives labels for its selected rows."""
    x = benchmark.inputs
    identity: object = UPSTREAM_REVISION
    if benchmark.problem == "branin-williams":
        # Preserve the established device-specific baseline numerics.
        x = x.to(device)
        losses = branin_williams(x)
    elif benchmark.problem == "f6":
        losses = marzat_f6(x)
    elif benchmark.problem == "portfolio":
        paths = _portfolio_files(assets)
        _require_files(paths)
        losses = _portfolio_losses(x, paths)
        identity = {str(p.relative_to(assets)): artifact_identity(p) for p in paths}
    elif benchmark.problem == "covid-testing":
        losses, identity = _covid_losses(x, assets, covid_python, cache_dir)
    else:
        raise ValueError(f"Unknown risk BO problem {benchmark.problem!r}")
    if losses.shape != x.shape[:2] or not torch.isfinite(losses).all():
        raise ValueError(f"{benchmark.problem} produced invalid pool outcomes")
    return RiskPool(
        x.to(device), losses.to(device), benchmark.weights.to(device), benchmark.alpha, benchmark.measure
    ), identity
