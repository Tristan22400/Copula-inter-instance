#!/bin/bash
#OAR -n TabICL_Train
#OAR -l gpu=1,walltime=96:00:00
#OAR -p gpu_model != 'TITAN RTX' AND gpu_model != 'TitanRTX' AND gpu_model != 'Quadro RTX 8000' AND gpu_model != 'L4' AND gpu_model != 'NVIDIA L4'
#OAR -O logs/train_%jobid%.out
#OAR -E logs/train_%jobid%.err
#OAR -q p1

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

configure_cuda_devices train.sh

# The frozen-TabICL-marginal load (z_train sim-to-real diagnostic) does a HEAD
# request to huggingface.co to check for updates even though the checkpoint is
# already fully cached locally; that request has been taking 60-100s+ on this
# cluster's network. Skip it and read straight from cache. Unset/override
# HF_HUB_OFFLINE=0 before calling this script if a checkpoint isn't cached yet.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# data.z_train_source=tabpfn35_fast reads TABPFN_TOKEN, which _env.sh loads.

echo "Starting Training... (Job ID: ${OAR_JOB_ID:-local})"
if [[ "${TRAIN_SH_DRY_RUN:-0}" == "1" ]]; then
    echo "[train.sh] Dry run; command would be: python -m copula_inter.train $*"
    exit 0
fi
python -m copula_inter.train "$@"
echo "Training complete."
