#!/bin/bash
#OAR -n eeg-gp
#OAR -l gpu=1,walltime=12:00:00
#OAR -O logs/eeg_gp_%jobid%.out
#OAR -E logs/eeg_gp_%jobid%.err
#
# GP baselines on the EEG joint tasks (eval/runners/eeg_gp_baselines.py). Hydra overrides, e.g.
#     python3 scripts/g5k_gpus.py submit --walltime 8 --name eeg-gp-arcnp -- \
#         scripts/npbench/eeg_gp.sh 'tasks=[arcnp_interpolation,arcnp_reconstruction]' out_dir=results/npbench/eeg

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
export REAL_DATA_DIR="${REAL_DATA_DIR:-$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/eval/data/cache}"
echo "[$(date +%H:%M:%S)] OAR job ${OAR_JOB_ID:-local} on $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
python -u -m eval.runners.eeg_gp_baselines "$@"
