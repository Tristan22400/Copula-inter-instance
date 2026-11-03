#!/bin/bash
#OAR -n CopulaTestsFull
#OAR -l gpu=1,walltime=2:00:00
#OAR -O logs/test_full_%jobid%.out
#OAR -E logs/test_full_%jobid%.err
#OAR -q p1
#
# Full test suite on a GPU node, including the slow / gpu / pretrained tiers the
# CPU CI job skips. Run it before merging and after touching a backend.
#
#     mkdir -p logs
#     oarsub -S ./scripts/test_full.sh
#     oarsub -S "./scripts/test_full.sh -m 'slow or gpu or pretrained'"   # only the tiers CI skips
#
# Extra arguments are passed to pytest.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

echo "Running the full test suite (Job ID: ${OAR_JOB_ID:-local})"
python -m pytest -q --durations=20 "$@"
