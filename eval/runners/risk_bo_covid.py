"""JSON subprocess bridge to the released testing-allocation evaluator."""

from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import torch
from torch import Tensor

_evaluator: Callable[[Tensor], Tensor] | None = None


def _append(frame: pd.DataFrame, row: pd.DataFrame, *, ignore_index: bool) -> pd.DataFrame:
    return pd.concat((frame, row), ignore_index=ignore_index)


def _setup_worker() -> None:
    # The released simulator resets global RNG state per seed. Give each
    # worker its own process and keep it separate from the BO driver.
    global _evaluator
    with contextlib.redirect_stdout(sys.stderr):
        from BoRisk.test_functions.covid_exp_class import CovidEval

        class ReleasedEvaluator(CovidEval):
            continuous_inds = list(range(5))

            # BoTorch now requires this hook. The released forward() handles
            # the seed averaging and calls the original simulator.
            def _evaluate_true(self, x: Tensor) -> Tensor:
                return self.forward(x)

        # The archived simulator calls DataFrame.append, removed in pandas 2.
        if not hasattr(pd.DataFrame, "append"):
            setattr(pd.DataFrame, "append", _append)
        torch.set_num_threads(1)
        _evaluator = ReleasedEvaluator()


def _evaluate(point: list[float]) -> float:
    assert _evaluator is not None
    # Single rows avoid the upstream evaluator's unbounded worker pool.
    with contextlib.redirect_stdout(sys.stderr), torch.no_grad():
        return float(_evaluator(torch.tensor(point, dtype=torch.float64)).item())


def _collect(values: Iterator[float], n_scenarios: int, n_rows: int) -> list[float]:
    result = []
    for value in values:
        result.append(value)
        if len(result) % n_scenarios == 0:
            print(f"testing allocation: {len(result)}/{n_rows} scenario means", file=sys.stderr, flush=True)
    return result


def main() -> None:
    points = torch.as_tensor(json.load(sys.stdin), dtype=torch.float64)
    n_designs, n_scenarios, _ = points.shape
    rows = points.flatten(0, 1).tolist()
    requested = int(os.environ.get("BORISK_COVID_WORKERS", "4"))
    if requested < 1:
        raise ValueError("BORISK_COVID_WORKERS must be positive")
    cpu_limit = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    workers = min(requested, cpu_limit, len(rows))
    if workers == 1:
        _setup_worker()
        values = _collect(map(_evaluate, rows), n_scenarios, len(rows))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_setup_worker
        ) as pool:
            values = _collect(pool.map(_evaluate, rows), n_scenarios, len(rows))
    json.dump(
        torch.tensor(values, dtype=torch.float64).reshape(n_designs, n_scenarios).tolist(), sys.stdout, allow_nan=False
    )


if __name__ == "__main__":
    main()
