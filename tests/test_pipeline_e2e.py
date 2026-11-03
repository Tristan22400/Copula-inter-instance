"""The documented workflow runs end to end on CPU: generate PIT episodes -> train -> eval_checkpoint.

Every other test exercises one stage in isolation; this one chains the three real CLI entry
points as subprocesses (so Hydra parsing, on-disk dataset/manifest format, checkpoint save/load
and the eval runner's checkpoint loader all have to agree). Kept CI-sized: 4 analytic-PIT
episodes, the from-scratch copula_nano preset for 2 steps, 2 oracle-marginal eval episodes with
a handful of baseline-fit steps. No GPU, no network, no pretrained weights.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
TIMEOUT_S = 900


def _run(args: list[str], cwd: Path) -> None:
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "WANDB_MODE": "disabled",
        "PYTHONPATH": os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]),
    }
    proc = subprocess.run(
        [sys.executable, "-m", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert proc.returncode == 0, f"{args[0]} failed ({proc.returncode}):\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"


def test_generate_train_eval_pipeline(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    ckpt_dir = tmp_path / "ckpt"

    _run(
        [
            "copula_inter.generate_pit_dataset",
            "data.n_tasks=4",
            f"data.dataset_dir={data_dir}",
            "data.z_train_source=analytic",
            "data.P_min=8",
            "data.P_max=8",
            "data.N_min=8",
            "data.N_max=8",
        ],
        cwd=tmp_path,
    )
    assert (data_dir / "pit" / "manifest.json").is_file()

    _run(
        [
            "copula_inter.train",
            "model=copula_nano",
            "training.live_generation=false",
            f"training.dataset_dir={data_dir / 'pit'}",
            f"training.ckpt_dir={ckpt_dir}",
            "training.device=cpu",
            "training.steps=2",
            "training.warmup_steps=1",
            "training.batch_size=2",
            "training.val_episodes=2",
            "training.log_every=1",
            "training.val_every=1000",
            "training.save_every=1000",
            "training.plot_val_every=0",
            "training.startup_probes=false",
            "training.adaptive_kernel_sampling=false",
            "baselines.enabled=false",
            "baselines.era5_enabled=false",
            # copula_nano points the validation-only frozen marginal at a local Phase A checkpoint.
            "tabicl.pit_ckpt=null",
        ],
        cwd=tmp_path,
    )
    ckpts = sorted(ckpt_dir.glob("*.pt"))
    assert ckpts, f"train wrote no checkpoint to {ckpt_dir}"
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    assert isinstance(state, dict) and state, "checkpoint is empty"

    results = tmp_path / "results.json"
    _run(
        [
            "eval.runners.eval_checkpoint",
            f"ckpt={ckpts[-1]}",
            f"config={REPO / 'conf' / 'config.yaml'}",
            "n_episodes=2",
            "device=cpu",
            "marginal.z_train_source=oracle",
            "autoregressive.enabled=false",
            "baselines.n_steps_mle=5",
            "baselines.n_restarts_mle=1",
            "baselines.n_steps_zeromean_gp=5",
            "baselines.n_restarts_zeromean_gp=1",
            "baselines.n_steps_dkl=5",
            "baselines.n_restarts_dkl=1",
            "baselines.n_steps_per_ep=5",
            "baselines.patience_per_ep=5",
            f"baselines.cache={tmp_path / 'baseline_cache.pt'}",
            f"output.results_cache={results}",
            f"output.out_dir={tmp_path / 'out'}",
        ],
        cwd=tmp_path,
    )
    episodes = json.loads(results.read_text())["episodes"]
    assert len(episodes) == 2
    for ep in episodes.values():
        # "icl" is the trained copula head's z-space NLL; under the oracle marginal the
        # Y-space total is nan by design, so the copula score is what must be finite.
        for key in ("icl", "oracle", "independence", "best_baseline"):
            assert math.isfinite(ep["nlls"][key]), f"nlls[{key!r}] = {ep['nlls'][key]}"
