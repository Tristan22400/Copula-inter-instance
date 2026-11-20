"""Export the EEG evaluation data of Bruinsma et al. (2023), "Autoregressive Conditional Neural Processes".

Runs in the generator venv (scripts/npbench/install_np_venv.sh), not the main one. It uses the
paper's own code only: `neuralprocesses.EEGGenerator` (subject split with the fixed seed 99, the
evaluation generators of experiment/data/eeg.py with their default seed 0, batch 16, 2^12 tasks)
and wbml's EEG parser. Scoring runs in the main venv (eval/runners/npbench_eval.py).

Writes into --out-dir:

eeg_arcnp_<mode>.npz, mode in {interpolation, reconstruction, forecasting}: the paper's evaluation
    batches, iterated as train.py::eval does. A batch's 16 tasks share their context/target split,
    so each batch is stored as rectangular arrays with the 7 electrodes stacked as rows:
        b{i}_xc (B, Nc, 1)  b{i}_oc (Nc,)  b{i}_yc (B, Nc)    context rows (time in s, electrode, value)
        b{i}_xt (B, Nt, 1)  b{i}_ot (Nt,)  b{i}_yt (B, Nt)    target rows
    The score of a task is log q(yt | ctx) / Nt (nps.num_data: target scalars over all electrodes).

eeg_trials.npz: the raw trials of the 10 evaluation subjects, in EEGGenerator's order:
    y (T, 256, 7) in microvolts, x (256,) in seconds, electrodes (7,),
    and the per-electrode mean/std over every training-subject trial (train_mean, train_std, (7,)),
    used to standardise the Hassan et al. (2026) EEG tasks built from these trials.

Usage (from the checkout root, on a node):
    .venv-npbench/bin/python -I scripts/npbench/export_eeg.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Iterable

import numpy as np

BATCH_SIZE = 16
NUM_TASKS_EVAL = 2**12
MODES = ("interpolation", "reconstruction", "forecasting")


def _np(x: Any) -> np.ndarray:
    import lab as B

    return np.asarray(B.to_numpy(x))


def _targets(xt: Any, yt: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(xt_i (B, 1, n_i), yt_i (B, 1, n_i))] per electrode, from an aggregated or plain target set."""
    import neuralprocesses.torch as nps

    if isinstance(xt, nps.AggregateInput):
        return [(_np(xi), _np(yi)) for (xi, _), yi in zip(xt, yt)]
    return [(_np(xt), _np(yt))]


def _stack(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows of every electrode stacked: x (B, n, 1), o (n,), y (B, n)."""
    xs = [np.swapaxes(x, 1, 2) for x, _ in parts]
    ys = [y[:, 0, :] for _, y in parts]
    o = np.concatenate([np.full(x.shape[1], i, dtype=np.int64) for i, x in enumerate(xs)])
    return np.concatenate(xs, axis=1), o, np.concatenate(ys, axis=1)


def _write_batches(path: str, batches: Iterable[dict], meta: dict) -> None:
    # Any: np.savez_compressed types **kwds against its allow_pickle flag.
    arrays: dict[str, Any] = {}
    n = 0
    t0 = time.time()
    for i, batch in enumerate(batches):
        xc, oc, yc = _stack([(_np(x), _np(y)) for x, y in batch["contexts"]])
        xt, ot, yt = _stack(_targets(batch["xt"], batch["yt"]))
        arrays |= {f"b{i}_xc": xc, f"b{i}_oc": oc, f"b{i}_yc": yc, f"b{i}_xt": xt, f"b{i}_ot": ot, f"b{i}_yt": yt}
        n = i + 1
    meta = meta | {"n_batches": n, "seconds": round(time.time() - t0, 1)}
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)
    print(f"wrote {path}: {n} batches ({meta['seconds']} s)", flush=True)


def export_batches(out_dir: str, num_tasks: int) -> None:
    import neuralprocesses.torch as nps
    import torch

    for mode in MODES:
        path = os.path.join(out_dir, f"eeg_arcnp_{mode}.npz")
        if os.path.isfile(path):
            continue
        # experiment/data/eeg.py::gens_eval: no seed argument, so EEGGenerator's default (0).
        gen = nps.EEGGenerator(
            torch.float32, batch_size=BATCH_SIZE, num_tasks=num_tasks, mode=mode, subset="eval", device="cpu"
        )
        meta = {
            "benchmark": "arcnp",
            "task": "eeg",
            "mode": mode,
            "seed": 0,
            "batch_size": BATCH_SIZE,
            "num_tasks": num_tasks,
            "torch": torch.__version__,
        }
        _write_batches(path, gen.epoch(), meta)


def export_trials(out_dir: str) -> None:
    from neuralprocesses.data.eeg import EEGGenerator

    path = os.path.join(out_dir, "eeg_trials.npz")
    if os.path.isfile(path):
        return
    # EEGGenerator._load_trials applies the paper's subject split (seed 99) and electrode selection.
    eval_trials = EEGGenerator._load_trials("eval")
    train_trials = EEGGenerator._load_trials("train")
    electrodes = list(eval_trials[0].columns)
    y = np.stack([np.asarray(t[electrodes], dtype=np.float64) for t in eval_trials])
    x = np.asarray(eval_trials[0].index, dtype=np.float64)
    train = np.concatenate([np.asarray(t[electrodes], dtype=np.float64) for t in train_trials])
    meta = {"n_eval_trials": len(eval_trials), "n_train_trials": len(train_trials), "electrodes": electrodes}
    np.savez_compressed(
        path,
        y=y,
        x=x,
        electrodes=np.array(electrodes),
        train_mean=train.mean(axis=0),
        train_std=train.std(axis=0),
        meta=np.array(json.dumps(meta)),
    )
    print(f"wrote {path}: {meta}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-tasks", type=int, default=NUM_TASKS_EVAL)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    export_trials(args.out_dir)
    export_batches(args.out_dir, args.num_tasks)


if __name__ == "__main__":
    main()
