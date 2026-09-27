# Shared job setup: cd to this checkout's root, activate the conda env, and put
# this checkout's packages first on PYTHONPATH. Source it from scripts/*.sh.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_BASE="${CONDA_BASE:-$HOME/thoth_storage/miniconda3}"
CONDA_ENV="${CONDA_ENV:-multivariate-icl}"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    source "$CONDA_BASE/bin/activate" "$CONDA_ENV"
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:$REPO_ROOT/tabicl_upstream/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Restrict CUDA_VISIBLE_DEVICES to GPUs not matching FORBIDDEN_GPU_REGEX (or refuse
# to run if a pre-set CUDA_VISIBLE_DEVICES includes one). Call it explicitly:
#     configure_cuda_devices <log tag>
FORBIDDEN_GPU_REGEX="${FORBIDDEN_GPU_REGEX:-TITAN[[:space:]]*RTX|TitanRTX|Quadro[[:space:]]*RTX[[:space:]]*8000|(^|[^0-9A-Za-z])L4($|[^0-9A-Za-z])}"

configure_cuda_devices() {
    local tag="${1:-job}"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[$tag] nvidia-smi not found; relying on scheduler GPU constraints."
        return
    fi

    local gpu_rows
    gpu_rows="$(nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader 2>/dev/null || true)"
    if [[ -z "$gpu_rows" ]]; then
        echo "[$tag] No GPUs reported by nvidia-smi."
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
                echo "[$tag] Refusing to run: CUDA_VISIBLE_DEVICES includes forbidden GPU '$match' (entry $entry)." >&2
                exit 1
            fi
        done
        echo "[$tag] Using pre-set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
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
        echo "[$tag] Refusing to run: only forbidden GPU models are visible (${rejected[*]})." >&2
        exit 1
    fi

    local IFS=,
    export CUDA_VISIBLE_DEVICES="${selected[*]}"
    echo "[$tag] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (excluded: ${rejected[*]:-none})"
}

