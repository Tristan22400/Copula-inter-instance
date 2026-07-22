#!/bin/bash
#OAR -n TabICL_Train
#OAR -l gpu=1,walltime=36:00:00
#OAR -p gpu_model != 'TITAN RTX' AND gpu_model != 'TitanRTX'


set -euo pipefail

# Navigate to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

FORBIDDEN_GPU_REGEX="${FORBIDDEN_GPU_REGEX:-TITAN[[:space:]]*RTX|TitanRTX}"

configure_cuda_devices() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[train.sh] nvidia-smi not found; relying on scheduler GPU constraints."
        return
    fi

    local gpu_rows
    gpu_rows="$(nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader 2>/dev/null || true)"
    if [[ -z "$gpu_rows" ]]; then
        echo "[train.sh] No GPUs reported by nvidia-smi."
        return
    fi

    local selected=()
    local rejected=()
    local idx name uuid entry row match

    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a selected <<< "$CUDA_VISIBLE_DEVICES"
        for entry in "${selected[@]}"; do
            entry="${entry//[[:space:]]/}"
            match=""
            while IFS=',' read -r idx name uuid; do
                idx="${idx//[[:space:]]/}"
                name="${name#"${name%%[![:space:]]*}"}"
                name="${name%"${name##*[![:space:]]}"}"
                uuid="${uuid//[[:space:]]/}"
                if [[ "$entry" == "$idx" || "$entry" == "$uuid" ]]; then
                    match="$name"
                    break
                fi
            done <<< "$gpu_rows"
            if [[ -n "$match" && "$match" =~ $FORBIDDEN_GPU_REGEX ]]; then
                echo "[train.sh] Refusing to run: CUDA_VISIBLE_DEVICES includes forbidden GPU '$match' (entry $entry)." >&2
                exit 1
            fi
        done
        echo "[train.sh] Using pre-set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        return
    fi

    while IFS=',' read -r idx name uuid; do
        idx="${idx//[[:space:]]/}"
        name="${name#"${name%%[![:space:]]*}"}"
        name="${name%"${name##*[![:space:]]}"}"
        if [[ "$name" =~ $FORBIDDEN_GPU_REGEX ]]; then
            rejected+=("$idx:$name")
        else
            selected+=("$idx")
        fi
    done <<< "$gpu_rows"

    if (( ${#selected[@]} == 0 )); then
        echo "[train.sh] Refusing to run: only forbidden GPU models are visible (${rejected[*]})." >&2
        exit 1
    fi

    local IFS=,
    export CUDA_VISIBLE_DEVICES="${selected[*]}"
    echo "[train.sh] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (excluded: ${rejected[*]:-none})"
}

configure_cuda_devices

# Setup environment
CONDA_BASE="$HOME/thoth_storage/miniconda3"
CONDA_ENV="$CONDA_BASE/envs/multivariate-icl"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    source "$CONDA_BASE/bin/activate" "$CONDA_ENV"
fi
export PYTHONNOUSERSITE=1
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$PYTHONPATH:$(pwd)"
else
    export PYTHONPATH="$(pwd)"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting Training... (Job ID: ${OAR_JOB_ID:-local})"
if [[ "${TRAIN_SH_DRY_RUN:-0}" == "1" ]]; then
    echo "[train.sh] Dry run; command would be: python src/train.py $*"
    exit 0
fi
python src/train.py "$@"
echo "Training complete."
