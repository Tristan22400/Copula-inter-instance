#!/usr/bin/env python3
"""run_debug.py — single entry point for the z_train_source=tabicl plateau
debug pipeline (see debug/README.md for what each stage answers).

Each stage is a standalone script under debug/stages/ with its own CLI
(run it directly for full control over its flags); this dispatcher just
forwards everything after the stage name to that script's own main(), so
`--help` on either path shows the same flags. Its only added value is
`all`, which runs every stage that needs no stage-specific flag (S0-S3
unconditionally; S5/S6 too if --ckpt is given) under ONE shared --run-id,
so their JSON output lands in the same debug/results/<run_id>/ directory
and report.py can aggregate it in one pass.

S4 (overfit, needs --kernel/--target), S7a (backbone diagnostic, needs
--ckpt and per-backend setup), S7b (backend training comparison, needs
--backends and is itself a short training run), and S8 (single-kernel
train_fast.py launch, needs --kernel and is a training run, not a
diagnostic) are deliberately NOT part of `all` -- invoke them directly.

Usage:
    python debug/run_debug.py all --n-episodes 200
    python debug/run_debug.py all --ckpt kernel-sweep-all-tabicl-retrain-15k
    python debug/run_debug.py s1 --n-episodes 64 --ranks 8,16,32,64
    python debug/run_debug.py s4 --kernel rbf --target posterior
    python debug/run_debug.py s7b --backends tabicl,tabpfn --steps 200
    python debug/run_debug.py s8 --kernel matern52
    python debug/run_debug.py report --run-id <run_id>
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_STAGES_DIR = os.path.join(_HERE, "stages")

STAGE_SCRIPTS = {
    "s0": "s0_signal.py",
    "s1": "s1_rank_ceiling.py",
    "s2": "s2_uspace.py",
    "s3": "s3_pit_floor.py",
    "s4": "s4_overfit.py",
    "s5": "s5_kfold.py",
    "s6": "s6_guards.py",
    "s7": "s7_backbone.py",
    "s7b": "s7b_backend_train.py",
    "s8": "s8_single_kernel.py",
}

ALL_STAGES_NO_CKPT = ["s0", "s1", "s2", "s3"]
ALL_STAGES_NEEDS_CKPT = ["s5", "s6"]


def _git_sha() -> str:
    from config import _git_sha as gs

    return gs(os.path.dirname(_HERE))


def _run_stage(stage: str, extra_args: "list[str]") -> int:
    script = os.path.join(_STAGES_DIR, STAGE_SCRIPTS[stage])
    cmd = [sys.executable, script, *extra_args]
    print(f"\n{'=' * 70}\n[{stage}] {' '.join(cmd)}\n{'=' * 70}")
    return subprocess.call(cmd, cwd=os.path.dirname(_HERE))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=[*STAGE_SCRIPTS, "all", "report"])
    args, extra = p.parse_known_args()

    if args.stage == "report":
        from report import main as report_main

        sys.argv = ["report.py", *extra]
        report_main()
        return

    if args.stage != "all":
        raise SystemExit(_run_stage(args.stage, extra))

    # "all": one shared run_id across every stage so their JSON lands in
    # the same debug/results/<run_id>/ directory. Handles both the
    # one-token (--run-id=foo) and two-token (--run-id foo) argparse forms
    # -- getting this wrong would silently re-append a second, different
    # --run-id and let each stage's own argparse pick whichever wins by
    # last-occurrence, defeating the whole point of a SHARED id.
    has_ckpt = any(a.startswith("--ckpt") for a in extra)
    run_id = None
    for i, a in enumerate(extra):
        if a.startswith("--run-id="):
            run_id = a.split("=", 1)[1]
        elif a == "--run-id" and i + 1 < len(extra):
            run_id = extra[i + 1]
    if run_id is None:
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{_git_sha()}_all"
        extra = [*extra, f"--run-id={run_id}"]

    stages = list(ALL_STAGES_NO_CKPT)
    if has_ckpt:
        stages += ALL_STAGES_NEEDS_CKPT
    else:
        print(
            "[run_debug] No --ckpt given -- skipping S5 (K-fold probe) and S6 "
            "(guard audit), both frozen-checkpoint stages. Pass --ckpt <name> "
            "to include them."
        )

    failures = []
    for stage in stages:
        code = _run_stage(stage, extra)
        if code != 0:
            failures.append(stage)

    print(f"\n{'=' * 70}\nrun_id={run_id}  stages_run={stages}  failures={failures or 'none'}\n{'=' * 70}")
    if not failures:
        from report import build_report

        out_path = build_report(run_id)
        print(f"Report written -> {out_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.path.insert(0, _HERE)
    main()
