# Shared job setup: cd to this checkout's root, activate its uv virtualenv, and
# put this checkout's packages first on PYTHONPATH. Source it from scripts/*.sh.
#
# The venv is created once per machine, from the checkout root:
#     uv sync --locked --extra dev --extra gpu      # --extra cpu on a machine without CUDA
# A git worktree without its own .venv uses the main checkout's; COPULA_VENV
# overrides both.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${COPULA_VENV:-}" ]]; then
    COPULA_VENV="$REPO_ROOT/.venv"
    if [[ ! -d "$COPULA_VENV" ]]; then
        git_common_dir="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
        [[ -n "$git_common_dir" ]] && COPULA_VENV="$(dirname "$git_common_dir")/.venv"
    fi
fi
if [[ ! -f "$COPULA_VENV/bin/activate" ]]; then
    echo "_env.sh: no virtualenv at $COPULA_VENV; run 'uv sync --locked --extra dev --extra gpu' in $REPO_ROOT" >&2
    exit 1
fi
source "$COPULA_VENV/bin/activate"

# The tabpfn35_fast marginal needs a PriorLabs API key whose account accepted the
# TabPFN-3.5 licence (https://ux.priorlabs.ai). Never pass it as a CLI arg
# (oarstat/ps show job command lines to other users on this shared cluster) or
# commit it: it is read from a private, untracked file outside the repo when
# TABPFN_TOKEN isn't already set.
TABPFN_TOKEN_FILE="${TABPFN_TOKEN_FILE:-$HOME/.config/tabpfn_token}"
if [[ -z "${TABPFN_TOKEN:-}" && -f "$TABPFN_TOKEN_FILE" ]]; then
    export TABPFN_TOKEN
    TABPFN_TOKEN="$(cat "$TABPFN_TOKEN_FILE")"
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

