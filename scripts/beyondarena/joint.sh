#!/bin/bash
# BeyondArena joint track (eval/runners/beyondarena_joint_eval.py) for one dataset and one model.
#
#     oarsub -l gpu=1,walltime=12:00:00 "./scripts/beyondarena/joint.sh <dataset> <model> [key=value ...]"
#
# models (docs/beyondarena/preregistration.md): each copula runs with the marginal it was trained on.
#     tabicl-ft  tabpfn-ft  kumo-ft      size-robust-* copulas on the ERA5-fine-tuned marginals
#     tabicl-pre tabpfn-pre kumo-pre     pretrained-marginal-* copulas on the pretrained marginals
# Every run also scores indep and exch on the same marginal and blocks (and lmm, see LMM_DIR).
# Results: results/beyondarena/joint/<TAG:-main>/<model>/; a rerun resumes at the first missing split.
# Extra key=value arguments go to the runner (ablations: cross_fit=random drop_group=true rank=4 p_cop=256);
# set TAG to keep an ablation's results apart from the main run.
#
#OAR -n beyondarena-joint
#OAR -l gpu=1,walltime=12:00:00
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
DATASET="$1"
MODEL="$2"
shift 2
C=checkpoints/copula_nano
M=checkpoints/marginal/ablations
case "$MODEL" in
tabicl-ft) ARGS=(ckpt=$C/size-robust-tabicl/step_0100000.pt tabicl_ckpt=$M/marginal_finetune_era5_run1/step_0177600_final.pt) ;;
tabpfn-ft) ARGS=(max_features=500 ckpt=$C/size-robust-tabpfn35_fast/step_0100000.pt marginal=tabpfn35_fast marginal_ckpt=$M/marginal_finetune_era5_tabpfn35_fast/step_0027500_final.pt) ;;
kumo-ft) ARGS=(ckpt=$C/size-robust-kumo_small/step_0100000.pt marginal=kumo_small marginal_ckpt=$M/marginal_finetune_era5_kumo_small/step_0114500_final.pt) ;;
tabicl-pre) ARGS=(ckpt=$C/pretrained-marginal-tabicl/step_0100000.pt tabicl_ckpt=pretrained) ;;
tabpfn-pre) ARGS=(max_features=500 ckpt=$C/pretrained-marginal-tabpfn35_fast/step_0100000.pt marginal=tabpfn35_fast) ;;
kumo-pre) ARGS=(ckpt=$C/pretrained-marginal-kumo_small/step_0100000.pt marginal=kumo_small) ;;
*) echo "unknown model $MODEL" >&2; exit 2 ;;
esac
# shellcheck disable=SC1091
source scripts/_env.sh
configure_cuda_devices beyondarena-joint
OUT="results/beyondarena/joint/${TAG:-main}/$MODEL"
LMM_DIR="${LMM_DIR:-results/beyondarena/lmm}"
exec python -m eval.runners.beyondarena_joint_eval dataset="$DATASET" method="$MODEL" "${ARGS[@]}" \
    lmm_dir="$LMM_DIR" out_dir="$OUT" "$@"
