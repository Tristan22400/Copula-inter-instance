#!/bin/bash
#OAR -n eeg-joint
#OAR -l gpu=1,walltime=24:00:00
#OAR -O logs/eeg_joint_%jobid%.out
#OAR -E logs/eeg_joint_%jobid%.err
#
# EEG joint log-likelihood benchmark (eval/runners/eeg_joint_eval.py): one marginal per job.
# Arguments are Hydra overrides on EegJointEvalSpec, e.g.
#     python3 scripts/g5k_gpus.py submit --walltime 24 --name eeg-joint-tabicl -- \
#         scripts/npbench/eeg_joint.sh marginals=[tabicl] out_dir=results/npbench/eeg
# The data must be exported first (scripts/npbench/export_eeg.py, see docs/npbench/README.md).

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_env.sh"
export REAL_DATA_DIR="${REAL_DATA_DIR:-$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/eval/data/cache}"
echo "[$(date +%H:%M:%S)] OAR job ${OAR_JOB_ID:-local} on $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "    REAL_DATA_DIR=$REAL_DATA_DIR args: $*"
python -u -m eval.runners.eeg_joint_eval "$@"
