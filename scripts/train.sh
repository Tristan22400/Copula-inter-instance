#!/bin/bash
#OAR -n TabICL_Train
#OAR -l gpu=1,walltime=36:00:00
#OAR -p gpu_model != 'TITAN RTX' AND gpu_model != 'TitanRTX' AND gpu_model != 'Quadro RTX 8000' AND gpu_model != 'L4' AND gpu_model != 'NVIDIA L4'
#OAR -q p1


set -euo pipefail

# Navigate to project root
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

FORBIDDEN_GPU_REGEX="${FORBIDDEN_GPU_REGEX:-TITAN[[:space:]]*RTX|TitanRTX|Quadro[[:space:]]*RTX[[:space:]]*8000|(^|[^0-9A-Za-z])L4($|[^0-9A-Za-z])}"

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

# The frozen-TabICL-marginal load (z_train sim-to-real diagnostic) does a HEAD
# request to huggingface.co to check for updates even though the checkpoint is
# already fully cached locally; that request has been taking 60-100s+ on this
# cluster's network. Skip it and read straight from cache. Unset/override
# HF_HUB_OFFLINE=0 before calling this script if a checkpoint isn't cached yet.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# data.z_train_source=tabpfn needs a one-time PriorLabs license acceptance +
# API key (https://ux.priorlabs.ai) -- never pass it as a CLI arg (oarstat/ps
# show job command lines to other users on this shared cluster) or commit it.
# Same convention as debug/launch_full_debug.sh: read from a private,
# untracked file outside the repo if TABPFN_TOKEN isn't already set.
TABPFN_TOKEN_FILE="${TABPFN_TOKEN_FILE:-$HOME/.config/tabpfn_token}"
if [[ -z "${TABPFN_TOKEN:-}" && -f "$TABPFN_TOKEN_FILE" ]]; then
    export TABPFN_TOKEN
    TABPFN_TOKEN="$(cat "$TABPFN_TOKEN_FILE")"
fi

echo "Starting Training... (Job ID: ${OAR_JOB_ID:-local})"
if [[ "${TRAIN_SH_DRY_RUN:-0}" == "1" ]]; then
    echo "[train.sh] Dry run; command would be: python -m copula_inter.train $*"
    exit 0
fi
python -m copula_inter.train "$@"
echo "Training complete."
