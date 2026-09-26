#!/bin/bash
#OAR -n CopulaEra5Finetune
#OAR -l gpu=1,walltime=24:00:00
#OAR -O logs/finetune_era5_%jobid%.out
#OAR -E logs/finetune_era5_%jobid%.err
#OAR -q p1
#
# Finetune an existing copula-model checkpoint on real, worldwide ARCO-ERA5
# data (src/copula_inter/finetune_era5.py -> src/copula_inter/train.py training.live_source=era5).
#
# Prerequisite: a local ERA5 corpus (one-time, ~125MB/month; run on a
# frontend or its own OAR job -- needs network, not GPU):
#     python eval/data/fetch_era5_global.py --start 2022-01 --n-months 24
#
# Submit with:
#     mkdir -p logs
#     oarsub -S "./scripts/finetune_era5.sh --ckpt ./checkpoints/kernel-sweep-all-tabicl-retrain/step_0015000.pt"
#
# Pass any finetune_era5.py flag through, e.g.:
#     oarsub -S "./scripts/finetune_era5.sh --ckpt ./checkpoints/<run>/step_XXXXXXX.pt --steps 20000 --corpus-dir ./eval/data/cache/era5_global"

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

python -m copula_inter.finetune_era5 "$@"

echo "[$(date +%H:%M:%S)] Finetuning complete."
