#!/bin/bash
#OAR -n CopulaEra5Finetune
#OAR -l gpu=1,walltime=24:00:00
#OAR -O logs/finetune_era5_%jobid%.out
#OAR -E logs/finetune_era5_%jobid%.err
#OAR -q p1
#
# Finetune an existing copula-model checkpoint on real, worldwide ARCO-ERA5
# data: copula_inter.train with the conf/experiment/finetune_era5.yaml preset.
# Arguments are Hydra overrides for copula_inter.train.
#
# Prerequisite: a local ERA5 corpus (one-time, ~125MB/month; run on a
# frontend or its own OAR job -- needs network, not GPU):
#     python eval/data/fetch_era5_global.py --start 2022-01 --n-months 24
#
# Submit with:
#     mkdir -p logs
#     oarsub -S "./scripts/finetune_era5.sh training.resume_ckpt=kernel-sweep-all-tabicl-retrain-15k model.rank=32"
#
# Pass any other override through, e.g.:
#     oarsub -S "./scripts/finetune_era5.sh training.resume_ckpt=./checkpoints/<run>/step_XXXXXXX.pt training.steps=20000 era5_live.corpus_dir=./eval/data/cache/era5_global"

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

# See scripts/train.sh's comment: the frozen-TabICL-marginal load does a slow
# HF Hub HEAD check even when fully cached locally. Skip it once cached.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p logs

echo "[$(date +%H:%M:%S)] OAR job ${OAR_JOB_ID:-local} — host: $(hostname)"
echo "[$(date +%H:%M:%S)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[$(date +%H:%M:%S)] Finetuning on real ERA5 data..."
echo "    args: $*"

python -m copula_inter.train experiment=finetune_era5 "$@"

echo "[$(date +%H:%M:%S)] Finetuning complete."
