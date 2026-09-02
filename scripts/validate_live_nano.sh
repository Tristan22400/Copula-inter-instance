#!/bin/bash
#OAR -n CopulaLiveNano_Validate
#OAR -l gpu=1,walltime=1:00:00
#OAR -O logs/validate_live_nano_%jobid%.out
#OAR -E logs/validate_live_nano_%jobid%.err
#OAR -q p1
#
# Smoke-test training.live_generation=true against the nano model preset
# (conf/model/copula_nano.yaml) across every model.correlation_parametrization
# option in src/correlation_factory.py (covnorm, cossim, tanhnorm,
# sparse_covnorm). Each variant runs a handful of steps with a small episode
# size, so this validates that on-the-fly generation + training + the
# live-mode validation batch (build_fixed_live_val_batches) all run end to
# end for that parametrization -- it is not a convergence/quality check.
#
# Submit with:
#     mkdir -p logs
#     oarsub -S ./scripts/validate_live_nano.sh
#
# Override the parametrization list or step count, e.g.:
#     oarsub -S "./scripts/validate_live_nano.sh covnorm tanhnorm"
#     VALIDATE_STEPS=100 oarsub -S ./scripts/validate_live_nano.sh

set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# ----- Env (mirrors scripts/train.sh) -----
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
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Keep the smoke test out of the real wandb project history.
export WANDB_MODE="${WANDB_MODE:-offline}"

PARAMETRIZATIONS=("$@")
if [ "${#PARAMETRIZATIONS[@]}" -eq 0 ]; then
    PARAMETRIZATIONS=(covnorm cossim tanhnorm sparse_covnorm)
fi

VALIDATE_STEPS="${VALIDATE_STEPS:-40}"
VALIDATE_VAL_EVERY="${VALIDATE_VAL_EVERY:-20}"
JOB_TAG="${OAR_JOB_ID:-$$}"

mkdir -p logs

echo "[$(date +%H:%M:%S)] OAR job $OAR_JOB_ID — host: $(hostname)"
echo "[$(date +%H:%M:%S)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[$(date +%H:%M:%S)] Validating live_generation for model=copula_nano across: ${PARAMETRIZATIONS[*]}"

declare -A RESULTS
for p in "${PARAMETRIZATIONS[@]}"; do
    out="logs/validate_live_nano_${JOB_TAG}_${p}.out"
    err="logs/validate_live_nano_${JOB_TAG}_${p}.err"
    echo "[$(date +%H:%M:%S)] --- ${p}: starting (log: $out) ---"
    if python src/train.py model=copula_nano \
        training.live_generation=true \
        model.correlation_parametrization="$p" \
        data.P_min=8 data.P_max=64 data.N_min=4 data.N_max=32 \
        training.steps="$VALIDATE_STEPS" \
        training.warmup_steps=5 \
        training.log_every=10 \
        training.val_every="$VALIDATE_VAL_EVERY" \
        training.val_episodes=32 \
        training.save_every=1000000 \
        training.live_num_workers=4 \
        training.ckpt_dir="./checkpoints/validate_live_nano/${JOB_TAG}/${p}" \
        wandb.project=copula-inter-smoke \
        > "$out" 2> "$err"; then
        RESULTS["$p"]="PASS"
        echo "[$(date +%H:%M:%S)] --- ${p}: PASS ---"
    else
        RESULTS["$p"]="FAIL"
        echo "[$(date +%H:%M:%S)] --- ${p}: FAIL (see $err) ---"
    fi
done

echo ""
echo "===== live_generation / copula_nano parametrization validation summary ====="
status=0
for p in "${PARAMETRIZATIONS[@]}"; do
    printf '%-16s %s\n' "$p" "${RESULTS[$p]}"
    [ "${RESULTS[$p]}" = "PASS" ] || status=1
done
exit "$status"
