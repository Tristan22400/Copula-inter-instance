#!/bin/bash
#OAR -n CopulaPIT_Generate
#OAR -l gpu=1,walltime=24:00:00
#OAR -O logs/generate_%jobid%.out
#OAR -E logs/generate_%jobid%.err
#
# Generate the PIT episode dataset for the Copula Transformer.
#
# Submit with:
#     mkdir -p logs
#     oarsub -S ./scripts/generate_dataset.sh
#
# Override config values via extra args, e.g.:
#     oarsub -S "./scripts/generate_dataset.sh data.n_tasks=5000 data.kernel=cosine"

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# ----- Env -----
source ~/thoth_storage/miniconda3/etc/profile.d/conda.sh
conda activate multivariate-icl
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date +%H:%M:%S)] OAR job $OAR_JOB_ID — host: $(hostname)"
echo "[$(date +%H:%M:%S)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[$(date +%H:%M:%S)] Generating PIT dataset..."
echo "    overrides: $*"

python src/generate_pit_dataset.py "$@"

echo "[$(date +%H:%M:%S)] Generation complete."
