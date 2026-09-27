#!/bin/bash
#OAR -n CopulaEval
#OAR -l gpu=1,walltime=48:00:00
#OAR -O logs/eval_%jobid%.out
#OAR -E logs/eval_%jobid%.err
#OAR -q p1
#
# Evaluate an ICL checkpoint against classical baselines (eval/runners/eval_checkpoint.py).
# Arguments are Hydra overrides on eval/runners/eval_args.py::EvalSpec.
#
# Submit with:
#     mkdir -p logs
#     oarsub -S "./scripts/eval_checkpoint.sh ckpt=./checkpoints/<run>/step_XXXXXXX.pt"
#
# Pass any eval_checkpoint override through, e.g.:
#     oarsub -S "./scripts/eval_checkpoint.sh ckpt=./checkpoints/test_temp/step_0005000.pt live_generate=true n_episodes=200"

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"


echo "[$(date +%H:%M:%S)] OAR job ${OAR_JOB_ID:-local} — host: $(hostname)"
echo "[$(date +%H:%M:%S)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[$(date +%H:%M:%S)] CPU cores available to this job: $(python -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null || nproc)"
echo "[$(date +%H:%M:%S)] Evaluating checkpoint..."
echo "    args: $*"

# -u: unbuffered. Python block-buffers stdout when it is a file, so without
# this an OAR job's .out held nothing but the bash echoes above until the
# process exited — and a run killed at its walltime therefore showed no
# progress at all for the whole reservation.
python -u -m eval.runners.eval_checkpoint "$@"

echo "[$(date +%H:%M:%S)] Evaluation complete."
