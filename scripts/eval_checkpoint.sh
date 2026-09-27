#!/bin/bash
#OAR -n CopulaEval
#OAR -l gpu=1,walltime=24:00:00
#OAR -O logs/eval_%jobid%.out
#OAR -E logs/eval_%jobid%.err
#OAR -q p1
#
# Evaluate an ICL checkpoint against classical baselines (eval/runners/eval_checkpoint.py).
#
# Submit with (--ckpt optional -- it defaults to
# eval/configs/checkpoints.py's DEFAULT_CHECKPOINT_FAMILY):
#     mkdir -p logs
#     oarsub -S ./scripts/eval_checkpoint.sh
#
# Pass any eval_checkpoint.py flag through, including another checkpoint by
# path or by registry name:
#     oarsub -S "./scripts/eval_checkpoint.sh --ckpt kernel-sweep-all-tabicl-retrain-15k --n_episodes 200"

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# ----- Env -----
source ~/thoth_storage/miniconda3/etc/profile.d/conda.sh
conda activate multivariate-icl
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date +%H:%M:%S)] OAR job ${OAR_JOB_ID:-local} — host: $(hostname)"
echo "[$(date +%H:%M:%S)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[$(date +%H:%M:%S)] CPU cores available to this job: $(python -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null || nproc)"
echo "[$(date +%H:%M:%S)] Evaluating checkpoint..."
echo "    args: $*"

# -u: unbuffered. Python block-buffers stdout when it is a file, so without
# this an OAR job's .out held nothing but the bash echoes above until the
# process exited — and a run killed at its walltime therefore showed no
# progress at all for the whole reservation.
python -u eval/runners/eval_checkpoint.py "$@"

echo "[$(date +%H:%M:%S)] Evaluation complete."
