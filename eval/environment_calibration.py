"""The 12-reading calibration benchmark from the released BOCF experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor

from eval.risk_bo import BundlePool
from eval.risk_bo_models import GPRiskSurrogate

PROBLEM = "environment-calibration"
BOUNDS = ((7.0, 0.02, 0.01, 30.01), (13.0, 0.12, 3.0, 30.295))
TRUE_PARAMETERS = (10.0, 0.07, 1.505, 30.1525)
POSITIONS = (0.0, 1.0, 2.5)
TIMES = (15.0, 30.0, 45.0, 60.0)
READINGS = len(POSITIONS) * len(TIMES)


def concentration(designs: Tensor, grid: Tensor) -> Tensor:
    """Evaluate concentrations for each design at the ordered position–time pairs."""
    mass, diffusion, location, start = designs[..., None, :].unbind(-1)
    position, time = grid.unbind(-1)
    first_scale = 4 * diffusion * time
    first = mass * torch.exp(-position.square() / first_scale) / (math.pi * first_scale).sqrt()
    elapsed = time - start
    # The inactive branch uses a safe positive elapsed time before masking.
    second_scale = 4 * diffusion * torch.where(elapsed > 0, elapsed, 1)
    second = mass * torch.exp(-(position - location).square() / second_scale) / (math.pi * second_scale).sqrt()
    return first + torch.where(elapsed > 0, second, 0)


def squared_mismatch(samples: Tensor, reference: Tensor) -> Tensor:
    return (samples - reference).square().sum(-1)


def calibration_pool(pool_size: int, seed: int, device: str) -> tuple[BundlePool, dict[str, object]]:
    unit = torch.quasirandom.SobolEngine(4, scramble=True, seed=seed).draw(pool_size, dtype=torch.float64)
    lower, upper = unit.new_tensor(BOUNDS)
    designs = (lower + (upper - lower) * unit).to(device)
    # The released test_4a.py uses position-major ordering for the 3 by 4 grid.
    grid = torch.cartesian_prod(designs.new_tensor(POSITIONS), designs.new_tensor(TIMES))
    reference = concentration(designs.new_tensor(TRUE_PARAMETERS), grid)
    inputs = torch.cat((designs[:, None, :].expand(-1, READINGS, -1), grid.expand(pool_size, -1, -1)), dim=-1)
    outcomes = concentration(designs, grid)
    benchmark = BundlePool(inputs, outcomes, partial(squared_mismatch, reference=reference), 1, 0.0)
    identity: dict[str, object] = {
        "reference_implementation": "BOCF/test_4a.py",
        "bounds": BOUNDS,
        "reference_parameters": TRUE_PARAMETERS,
        "measurement_grid": grid.tolist(),
        "reference": reference.tolist(),
    }
    return benchmark, identity


@dataclass
class IndependentOutputGP:
    """One Matérn-5/2 GP over designs per measurement position and time."""

    n_steps: int

    def draw(self, x_train: Tensor, y_train: Tensor, x_query: Tensor, base: Tensor, seed: int) -> tuple[Tensor, None]:
        gp = GPRiskSurrogate(log_output=False, n_steps=self.n_steps)
        train_designs = x_train[::READINGS, :4]
        query_designs = x_query[::READINGS, :4]
        paths = [
            gp.draw(train_designs, y_train[i::READINGS], query_designs, base[:, i::READINGS], seed + i)[0]
            for i in range(READINGS)
        ]
        return torch.stack(paths, dim=-1).flatten(1), None
