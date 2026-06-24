#!/bin/bash
#OAR -n CopulaPIT_Generate
#OAR -l gpu=1,walltime=16:00:00
#OAR -O logs/generate_%jobid%.out
#OAR -E logs/generate_%jobid%.err
#
# Generate the PIT episode dataset for the Copula Transformer.
#
# Submit with:
#     mkdir -p logs
#     oarsub -S ./scripts/generate_dataset.sh
#
# Override defaults via extra args, e.g.:
#     oarsub -S "./scripts/generate_dataset.sh data.n_tasks=5000 tabicl.k_folds=10"

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# ----- Env -----
source ~/thoth_storage/miniconda3/etc/profile.d/conda.sh
conda activate multivariate-icl
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ----- Defaults (override via CLI args passed to this script) -----
# The episode schema changed (now carries y_test, log_pdf_test, Sigma_star),
# so old episodes under ./data/pit_episodes are not compatible. Use a fresh
# directory for the new dataset by default.
DEFAULTS=(
    "data.n_tasks=50000"
    "data.raw_dir=./data/gp_raw_tasks_v2"
    "data.latent_dir=./data/pit_episodes_v2"
    "tabicl.k_folds=10"
)

echo "[$(date +%H:%M:%S)] OAR job $OAR_JOB_ID — host: $(hostname)"
echo "[$(date +%H:%M:%S)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[$(date +%H:%M:%S)] Generating PIT dataset..."
echo "    defaults : ${DEFAULTS[*]}"
echo "    overrides: $*"

python src/generate_pit_dataset.py "${DEFAULTS[@]}" "$@"

echo "[$(date +%H:%M:%S)] Generation complete."
