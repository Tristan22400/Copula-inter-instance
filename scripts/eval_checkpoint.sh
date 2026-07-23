#!/bin/bash
#OAR -n CopulaEval
#OAR -l gpu=1,walltime=2:00:00
#OAR -O logs/eval_%jobid%.out
#OAR -E logs/eval_%jobid%.err
#
# Evaluate an ICL checkpoint against classical baselines (eval/runners/eval_checkpoint.py).
#
# Submit with:
#     mkdir -p logs
#     oarsub -S "./scripts/eval_checkpoint.sh --ckpt ./checkpoints/<run>/step_XXXXXXX.pt"
#
# Pass any eval_checkpoint.py flag through, e.g.:
#     oarsub -S "./scripts/eval_checkpoint.sh --ckpt ./checkpoints/test_temp/step_0005000.pt --live_generate --n_episodes 200"

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
echo "[$(date +%H:%M:%S)] Evaluating checkpoint..."
echo "    args: $*"

python eval/runners/eval_checkpoint.py "$@"

echo "[$(date +%H:%M:%S)] Evaluation complete."
