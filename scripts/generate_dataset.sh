#!/bin/bash
#OAR -n CopulaPIT_Generate
#OAR -l gpu=1,walltime=24:00:00
#OAR -O logs/generate_%jobid%.out
#OAR -E logs/generate_%jobid%.err
#
# Generate the PIT episode dataset for the Copula Transformer.
#
# Submit with:
#     mkdir -p logs
#     oarsub -S ./scripts/generate_dataset.sh
#
# Override config values via extra args, e.g.:
#     oarsub -S "./scripts/generate_dataset.sh data.n_tasks=5000 data.kernel=cosine"
#
# GEN_WORKERS: how many generate_pit_dataset.py processes to run concurrently
# against the single allocated GPU (data.py's src/data_gen.py::generate_gp_batch
# spends much of its time on host-side Python/kernel-construction work between
# GPU calls, so one process alone leaves the GPU idle a large fraction of the
# time -- measured 0-45% utilization, <2GB of 24GB VRAM used, from a single
# worker on an A5000). Running several workers overlaps one worker's host-side
# work with another's GPU kernels. Empirically (this A5000, nproc=8, analytic
# z_train_source) aggregate throughput peaked around 4 workers (~2x a single
# worker's episodes/sec, GPU utilization mostly 90-100%) and *fell* at 8
# workers (CPU-side contention outweighing the extra GPU overlap) -- default
# below follows that shape (nproc/2, capped) rather than "more is better".
# Override per-submission if you've benchmarked a better value for your node:
#     GEN_WORKERS=6 oarsub -S "./scripts/generate_dataset.sh data.n_tasks=5000"
# GEN_WORKERS=1 reproduces the original single-process script exactly.
#
# Concurrent CUDA context creation across workers occasionally races inside
# cuSOLVER (cusolverDnCreate returning CUSOLVER_STATUS_INTERNAL_ERROR, no OOM
# involved) -- generate_pit_dataset.py retries that specific error internally
# (_is_transient_cusolver_error), and this script staggers worker startup and
# restarts a worker that still exits non-zero (data.resume=true, so a restart
# only redoes the one shard that was in flight, never already-written work).

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# ----- Env -----
source ~/thoth_storage/miniconda3/etc/profile.d/conda.sh
conda activate multivariate-icl
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DEFAULT_WORKERS=$(( $(nproc) / 2 ))
[ "$DEFAULT_WORKERS" -lt 1 ] && DEFAULT_WORKERS=1
[ "$DEFAULT_WORKERS" -gt 6 ] && DEFAULT_WORKERS=6
GEN_WORKERS="${GEN_WORKERS:-$DEFAULT_WORKERS}"
JOB_TAG="${OAR_JOB_ID:-$$}"
MAX_WORKER_RETRIES=5

mkdir -p logs

echo "[$(date +%H:%M:%S)] OAR job $OAR_JOB_ID — host: $(hostname)"
echo "[$(date +%H:%M:%S)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[$(date +%H:%M:%S)] Generating PIT dataset with GEN_WORKERS=$GEN_WORKERS (nproc=$(nproc))..."
echo "    overrides: $*"

if [ "$GEN_WORKERS" -le 1 ]; then
    python src/generate_pit_dataset.py "$@"
    echo "[$(date +%H:%M:%S)] Generation complete."
    exit 0
fi

# One supervised worker: retries on any non-zero exit (transient CUDA context
# races that outlast generate_pit_dataset.py's own internal retry budget,
# transient NFS/storage hiccups on the shard write, etc.) up to
# MAX_WORKER_RETRIES times before giving up on this worker_id for good. Forces
# data.resume=true on every attempt after the first so a restart only redoes
# the shard that was in flight when the previous attempt died -- placed after
# "$@" so it overrides any data.resume the caller passed for the first attempt.
run_worker() {
    local worker_id="$1"; shift
    local out="logs/generate_${JOB_TAG}_w${worker_id}.out"
    local err="logs/generate_${JOB_TAG}_w${worker_id}.err"
    local attempt=0
    while true; do
        if [ "$attempt" -eq 0 ]; then
            python src/generate_pit_dataset.py "$@" worker_id="$worker_id" num_workers="$GEN_WORKERS" \
                >> "$out" 2>> "$err" && return 0
        else
            python src/generate_pit_dataset.py "$@" worker_id="$worker_id" num_workers="$GEN_WORKERS" data.resume=true \
                >> "$out" 2>> "$err" && return 0
        fi
        attempt=$((attempt + 1))
        if [ "$attempt" -ge "$MAX_WORKER_RETRIES" ]; then
            echo "[$(date +%H:%M:%S)] worker $worker_id: gave up after $attempt attempts, see $err" >&2
            return 1
        fi
        echo "[$(date +%H:%M:%S)] worker $worker_id: attempt $attempt failed, retrying in $((5 * attempt))s (see $err)" >&2
        sleep "$((5 * attempt))"
    done
}

pids=()
for w in $(seq 0 $((GEN_WORKERS - 1))); do
    run_worker "$w" "$@" &
    pids+=("$!")
    # Stagger startup: concurrent CUDA context creation across freshly
    # launched processes is the main trigger for the cusolver race above --
    # spreading launches out reduces (though, per generate_pit_dataset.py's
    # own retry, doesn't need to eliminate) how often it fires.
    sleep 2
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

if [ "$status" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] Generation complete ($GEN_WORKERS workers)."
else
    echo "[$(date +%H:%M:%S)] Generation finished with at least one worker failure — check logs/generate_${JOB_TAG}_w*.err" >&2
fi
exit "$status"
