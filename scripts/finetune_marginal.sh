#!/bin/bash
#OAR -n MarginalFinetune
#OAR -l gpu=1,walltime=24:00:00
#OAR -O logs/finetune_marginal_%jobid%.out
#OAR -E logs/finetune_marginal_%jobid%.err
#OAR -q p1
#
# Phase A — fine-tune a standalone TabICL so its MARGINAL posterior predictive
# is correct for the GP prior the copula is trained on
# (src/finetune_marginal.py). This trains nothing jointly with the copula; the
# output is a drop-in tabicl.pit_ckpt for a later src/train.py run.
#
# Prerequisites:
#   * the ERA5 mixture corpus (one-time, needs network not GPU -- run on a
#     frontend). Phase A reads the TRAIN half; the 2023 val half stays held out:
#         python eval/data/fetch_era5_global.py --start 2013-01 --n-months 120
#     (already present as eval/data/cache/era5_global_train/ + era5_global_val/)
#   * a baseline measurement to compare against, so the run has a before:
#         python eval/runners/marginal_calibration_eval.py --ckpt pretrained
#
# Submit with:
#     mkdir -p logs
#     oarsub -S ./scripts/finetune_marginal.sh
#
# Any Hydra override passes straight through, e.g.:
#     oarsub -S "./scripts/finetune_marginal.sh marginal.tier=1 training.lr=2e-5"
#     oarsub -S "./scripts/finetune_marginal.sh marginal.era5.mix_frac=0.3 training.steps=40000"
# ERA5-only control (validation remains the held-out 2023 ERA5 split):
#     oarsub -S "./scripts/finetune_marginal.sh marginal.era5.mix_frac=1.0"

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# ----- Env (mirrors scripts/finetune_era5.sh) -----
CONDA_BASE="$HOME/thoth_storage/miniconda3"
CONDA_ENV="$CONDA_BASE/envs/multivariate-icl"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    source "$CONDA_BASE/bin/activate" "$CONDA_ENV"
fi
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# See scripts/train.sh: loading the TabICL marginal does a slow HF Hub HEAD
# check even when fully cached. Skip it once cached.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Reuse one large ERA5 cache across linked worktrees instead of requiring a
# second 15 GB copy (or silently fetching validation files again). OAR_O_WORKDIR
# can itself be another worktree, so a directory existing there is not enough:
# accept a candidate only when it contains actual global-corpus files, then
# fall back to the primary checkout next to Git's common .git directory.
EXTRA_OVERRIDES=()
has_global_corpus() {
    compgen -G "$1/era5_global_train/era5_global_t2m_*.nc" >/dev/null
}

if [[ -z "${ERA5_CACHE_DIR:-}" ]]; then
    GIT_COMMON_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
    PRIMARY_CHECKOUT="$(dirname "$GIT_COMMON_DIR")"
    CACHE_CANDIDATES=(
        "${OAR_O_WORKDIR:-}/eval/data/cache"
        "$PRIMARY_CHECKOUT/eval/data/cache"
    )
    for candidate in "${CACHE_CANDIDATES[@]}"; do
        if [[ -n "$candidate" ]] && has_global_corpus "$candidate"; then
            export ERA5_CACHE_DIR="$candidate"
            break
        fi
    done
fi

if [[ -n "${ERA5_CACHE_DIR:-}" ]] && has_global_corpus "$ERA5_CACHE_DIR"; then
    EXTRA_OVERRIDES+=("marginal.era5.corpus_dir=$ERA5_CACHE_DIR/era5_global_train")
fi

mkdir -p logs

echo "[$(date +%H:%M:%S)] OAR job ${OAR_JOB_ID:-local} — host: $(hostname)"
echo "[$(date +%H:%M:%S)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[$(date +%H:%M:%S)] ERA5 cache: ${ERA5_CACHE_DIR:-not found}"
echo "[$(date +%H:%M:%S)] Phase A: marginal fine-tuning..."
echo "    overrides: $*"

python src/finetune_marginal.py "${EXTRA_OVERRIDES[@]}" "$@"

echo "[$(date +%H:%M:%S)] Phase A complete."
echo "Next:"
echo "  1) re-measure:  python eval/runners/marginal_calibration_eval.py --ckpt <the _final.pt>"
echo "  2) forgetting gate:  python eval/runners/run_benchmarks.py"
echo "  3) Phase B:     python src/train.py tabicl.pit_ckpt=<the _final.pt>"
