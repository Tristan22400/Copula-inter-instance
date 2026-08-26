#!/bin/bash
#OAR -n CopulaDebugPipeline
#OAR -l gpu=1,walltime=6:00:00
#OAR -O logs/debug_pipeline_%jobid%.out
#OAR -E logs/debug_pipeline_%jobid%.err
#OAR -q p1
#
# Full debug pipeline (see debug/README.md): S0-S3 signal/rank/PIT
# diagnostics, S5-S6 frozen-checkpoint probes, S4 overfit sanity checks,
# S7b real training comparison across marginal backends, S8 single-kernel
# probe -- one OAR job, one shared --run-id, ending in an aggregated
# debug/results/<run_id>/report.md (see debug/report.py).
#
# Prerequisite for the tabpfn backend (S7b): one-time license acceptance at
# https://ux.priorlabs.ai, then save the API key to a PRIVATE file (never
# commit it, never pass it as a CLI arg -- oarstat/ps show job command
# lines to other users on this shared cluster):
#     umask 077 && echo '<your-api-key>' > ~/.config/tabpfn_token
# If that file is absent, S7b falls back to --backends tabicl (tabicl-only
# comparison) -- everything else in this script still runs.
#
# Submit with:
#     mkdir -p logs
#     oarsub -S "./debug/launch_full_debug.sh"
#     oarsub -S "./debug/launch_full_debug.sh --ckpt ./checkpoints/<run>/step_XXXXXXX.pt"
#
# Override knobs via env vars (same convention as scripts/generate_dataset.sh's
# GEN_WORKERS, scripts/validate_live_nano.sh's VALIDATE_STEPS):
#     N_EPISODES=200 S7B_STEPS=80 S8_STEPS=1500 RUN_ID=my_run \
#         oarsub -S "./debug/launch_full_debug.sh --ckpt ./checkpoints/<run>/step_XXXXXXX.pt"

set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# ----- Env (mirrors scripts/train.sh / scripts/finetune_era5.sh) -----
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

TABPFN_TOKEN_FILE="${TABPFN_TOKEN_FILE:-$HOME/.config/tabpfn_token}"
if [[ -z "${TABPFN_TOKEN:-}" && -f "$TABPFN_TOKEN_FILE" ]]; then
    export TABPFN_TOKEN
    TABPFN_TOKEN="$(cat "$TABPFN_TOKEN_FILE")"
fi
BACKENDS="tabicl"
if [[ -n "${TABPFN_TOKEN:-}" ]]; then
    BACKENDS="tabicl,tabpfn"
fi

N_EPISODES="${N_EPISODES:-200}"
S7B_STEPS="${S7B_STEPS:-80}"
S8_STEPS="${S8_STEPS:-1500}"
CKPT_DEFAULT="./checkpoints/kernel-sweep-all-tabicl-retrain/step_0015000.pt"

# --ckpt is required by S5/S6; default to the family this project's own
# findings flag as the reference point (see debug/README.md's evidence
# table). Anything else on the command line is forwarded to run_debug.py.
# Parsed BEFORE RUN_ID's default below, so submitting several jobs that
# only differ by --ckpt (e.g. a step_0015000/0030000/0060000 sweep) get
# distinct default run_ids automatically -- no reliance on OAR propagating
# the submitting shell's env vars into the job (it may not).
CKPT="$CKPT_DEFAULT"
EXTRA_ARGS=()
skip_next=0
for arg in "$@"; do
    if [[ $skip_next -eq 1 ]]; then
        CKPT="$arg"; skip_next=0; continue
    fi
    if [[ "$arg" == "--ckpt" ]]; then
        skip_next=1; continue
    fi
    EXTRA_ARGS+=("$arg")
done

CKPT_TAG="$(basename "$(dirname "$CKPT")")_$(basename "$CKPT" .pt)"
RUN_ID="${RUN_ID:-job_full_debug_${CKPT_TAG}}"

mkdir -p logs "debug/results/$RUN_ID"

echo "[$(date +%H:%M:%S)] OAR job ${OAR_JOB_ID:-local} — host: $(hostname)"
echo "[$(date +%H:%M:%S)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[$(date +%H:%M:%S)] run_id=$RUN_ID ckpt=$CKPT n_episodes=$N_EPISODES s7b_backends=$BACKENDS extra_args: ${EXTRA_ARGS[*]:-}"

echo "=========================================================="
echo "[1/5] run_debug.py all  (S0,S1,S2,S3,S5,S6)"
echo "=========================================================="
python debug/run_debug.py all --n-episodes "$N_EPISODES" --ckpt "$CKPT" --device cuda --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
echo "[$(date +%H:%M:%S)] stage 1 exit=$?"

echo "=========================================================="
echo "[2/5] S4 overfit -- target=prior (baseline sanity check)"
echo "=========================================================="
python debug/stages/s4_overfit.py --kernel rbf --k-realizations 200 --steps 800 --batch-size 16 \
    --log-every 100 --plot "debug/results/$RUN_ID/s4_prior.png"
echo "[$(date +%H:%M:%S)] stage 2 exit=$?"

echo "=========================================================="
echo "[3/5] S4 overfit -- target=posterior, z-source=tabicl"
echo "=========================================================="
python debug/stages/s4_overfit.py --kernel rbf --target posterior --z-source tabicl --k-realizations 200 \
    --steps 800 --batch-size 16 --log-every 100 --plot "debug/results/$RUN_ID/s4_posterior_tabicl.png"
echo "[$(date +%H:%M:%S)] stage 3 exit=$?"

echo "=========================================================="
echo "[4/5] S7b -- real training comparison across backends=$BACKENDS"
echo "=========================================================="
python debug/stages/s7b_backend_train.py --backends "$BACKENDS" --steps "$S7B_STEPS" --batch-size 4 \
    --eval-every 10 --n-eval 8 --k-folds 5 --probs-n 49 --n-episodes "$N_EPISODES" --device cuda --run-id "$RUN_ID"
echo "[$(date +%H:%M:%S)] stage 4 exit=$?"

echo "=========================================================="
echo "[5/5] S8 -- single kernel family probe (rbf, $S8_STEPS steps)"
echo "=========================================================="
python debug/stages/s8_single_kernel.py --kernel rbf -- training.steps="$S8_STEPS" training.batch_size=16 \
    training.ckpt_dir=./checkpoints/_debug_s8_rbf > "debug/results/$RUN_ID/s8_rbf_log.txt" 2>&1
echo "[$(date +%H:%M:%S)] stage 5 exit=$? (full log: debug/results/$RUN_ID/s8_rbf_log.txt)"

echo "=========================================================="
echo "Regenerating aggregated report (now includes s7b)"
echo "=========================================================="
python debug/run_debug.py report --run-id "$RUN_ID"

echo "[$(date +%H:%M:%S)] ALL_STAGES_DONE run_id=$RUN_ID"
