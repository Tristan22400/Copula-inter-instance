#!/bin/bash
# One CTR23 dataset on its official 10-fold CV with the full training folds as context
# (eval/runners/ctr23_full_eval.py; docs/workflows.md, "OpenML-CTR23 on the full datasets").
#
# Usage (one OAR job per dataset and method, one GPU):
#     oarsub -p "gpu_mem >= 24000" -l gpu=1,walltime=12:00:00 "./scripts/ctr23_full.sh <data_id> [<method>]"
# methods:
#     nano (default, REAL_CKPT_TAG)  the copula (REAL_CKPT) on the TabICL marginal it was trained with (REAL_MARGINAL);
#                                    its marginal alone is the fine-tuned TabICL independence row
#     tabicl_base                    TabICL v2 pretrained, marginal only (independence)
#     tabpfn35_fast, kumo_small      that backend, pretrained, marginal only; needs a venv with the
#                                    tabpfn / kumo extras (COPULA_VENV) and TABPFN_TOKEN for TabPFN
#     kumo_ft_nano, tabpfn_ft_nano   the nano copula trained on that backend's ERA5 fine-tune (KUMO_FT_CKPT /
#                                    TABPFN_FT_CKPT), with that fine-tuned marginal (KUMO_FT_MARGINAL /
#                                    TABPFN_FT_MARGINAL); its marginal alone is the fine-tuned independence row
#     pre_tabicl_nano, pre_tabpfn_nano, pre_kumo_nano
#                                    the nano copula trained on that backend's PRETRAINED marginal
#                                    (PRE_nano_PN_*; PRE_TABICL_CKPT / PRE_TABPFN_CKPT / PRE_KUMO_CKPT), with
#                                    that pretrained marginal; its marginal alone matches tabicl_base /
#                                    tabpfn35_fast / kumo_small
# writes results/real/ctr23_full/<data_id>_<method>.json (+ .npz); a rerun resumes at the first missing fold.
# Further key=value arguments are passed to the runner.

set -euo pipefail

CK="${REAL_CKPT:-/home/trmartin/thoth_storage/copula-inter/checkpoints/copula_nano/copula-finetune-marginal-float32/step_0630000.pt}"
MG="${REAL_MARGINAL:-/home/trmartin/thoth_storage/copula-inter/checkpoints/marginal/ablations/marginal_finetune_era5_run1/step_0177600_final.pt}"
TAG="${REAL_CKPT_TAG:-nano}"
ROOT=/home/trmartin/thoth_storage/copula-inter
MBC="$ROOT/.claude/worktrees/marginal-backend-choice/checkpoints/marginal_backend_choice"
KUMO_FT_CKPT="${KUMO_FT_CKPT:-$MBC/ft_nano/FT_nano_kumo/step_0100000.pt}"
KUMO_FT_MARGINAL="${KUMO_FT_MARGINAL:-$ROOT/checkpoints/marginal/ablations/marginal_finetune_era5_kumo_small/step_0114500_final.pt}"
TABPFN_FT_CKPT="${TABPFN_FT_CKPT:-$MBC/ft_nano/FT_nano_tabpfn/step_0100000.pt}"
TABPFN_FT_MARGINAL="${TABPFN_FT_MARGINAL:-$ROOT/checkpoints/marginal/ablations/marginal_finetune_era5_tabpfn35_fast/step_0027500_final.pt}"
SIZE_GRID="$MBC/size_grid"
PRE_TABICL_CKPT="${PRE_TABICL_CKPT:-$ROOT/checkpoints/copula_nano/pretrained-marginal-tabicl/step_0100000.pt}"
PRE_TABPFN_CKPT="${PRE_TABPFN_CKPT:-$SIZE_GRID/PRE_nano_PN_tabpfn/step_0100000.pt}"
# PRE_nano_PN_kumo was launched from the size-grid-metrics worktree, so it saves there.
PRE_KUMO_CKPT="${PRE_KUMO_CKPT:-$ROOT/.claude/worktrees/size-grid-metrics/checkpoints/marginal_backend_choice/size_grid/PRE_nano_PN_kumo/step_0100000.pt}"

DATA_ID="$1"
shift
METHOD="$TAG"
if [ $# -gt 0 ] && [[ "$1" != *=* ]]; then METHOD="$1"; shift; fi
case "$METHOD" in
"$TAG") ARGS=(ckpt="$CK" tabicl_ckpt="$MG") ;;
tabicl_base) ARGS=(tabicl_ckpt=pretrained) ;;
tabpfn35_fast | kumo_small) ARGS=(marginal="$METHOD") ;;
kumo_ft_nano) ARGS=(ckpt="$KUMO_FT_CKPT" marginal=kumo_small marginal_ckpt="$KUMO_FT_MARGINAL") ;;
tabpfn_ft_nano) ARGS=(ckpt="$TABPFN_FT_CKPT" marginal=tabpfn35_fast marginal_ckpt="$TABPFN_FT_MARGINAL") ;;
pre_tabicl_nano) ARGS=(ckpt="$PRE_TABICL_CKPT" tabicl_ckpt=pretrained) ;;
pre_tabpfn_nano) ARGS=(ckpt="$PRE_TABPFN_CKPT" marginal=tabpfn35_fast) ;;
pre_kumo_nano) ARGS=(ckpt="$PRE_KUMO_CKPT" marginal=kumo_small) ;;
*) echo "unknown method $METHOD" >&2; exit 2 ;;
esac
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source scripts/_env.sh
# Raw files and the split ARFFs live under the MAIN checkout's cache (as in eval_checkpoint_real.sh).
if [ -z "${REAL_DATA_DIR:-}" ]; then
    REAL_DATA_DIR="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)/eval/data/cache"
fi
export REAL_DATA_DIR
mkdir -p results/real/ctr23_full
python -m eval.runners.ctr23_full_eval "${ARGS[@]}" data_id="$DATA_ID" \
    out_json="results/real/ctr23_full/${DATA_ID}_${METHOD}.json" "$@"
