#!/bin/bash
# One CTR23 dataset on its official 10-fold CV with the full training folds as context
# (eval/runners/ctr23_full_eval.py; docs/workflows.md, "OpenML-CTR23 on the full datasets").
#
# Usage (one OAR job per dataset, one large-memory GPU):
#     oarsub -q p1 -p "gpu_mem >= 40000" -l gpu=1,walltime=12:00:00 "./scripts/ctr23_full.sh <data_id>"
# writes results/real/ctr23_full/<data_id>_<tag>.json (+ .npz); a rerun resumes at the first missing fold.
# The copula checkpoint and its marginal come from REAL_CKPT / REAL_MARGINAL (scripts/real_grid.sh's pair).

set -euo pipefail

CK="${REAL_CKPT:-/home/trmartin/thoth_storage/copula-inter/checkpoints/copula_nano/copula-finetune-marginal-float32/step_0630000.pt}"
MG="${REAL_MARGINAL:-/home/trmartin/thoth_storage/copula-inter/checkpoints/marginal/ablations/marginal_finetune_era5_run1/step_0177600_final.pt}"
TAG="${REAL_CKPT_TAG:-nano}"

DATA_ID="$1"
shift
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/_env.sh
# Raw files and the split ARFFs live under the MAIN checkout's cache (as in eval_checkpoint_real.sh).
if [ -z "${REAL_DATA_DIR:-}" ]; then
    REAL_DATA_DIR="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)/eval/data/cache"
fi
export REAL_DATA_DIR
mkdir -p results/real/ctr23_full
python -m eval.runners.ctr23_full_eval ckpt="$CK" tabicl_ckpt="$MG" data_id="$DATA_ID" \
    out_json="results/real/ctr23_full/${DATA_ID}_${TAG}.json" "$@"
