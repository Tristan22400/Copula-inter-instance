#!/bin/bash
#OAR -n CopulaEvalERA5
#OAR -l gpu=1,walltime=48:00:00
#OAR -O logs/eval_era5_%jobid%.out
#OAR -E logs/eval_era5_%jobid%.err
#OAR -q p1
#
# Evaluate an ICL checkpoint against every classical baseline on REAL
# ARCO-ERA5 2m-temperature episodes — the real-data counterpart of
# scripts/eval_checkpoint.sh, running the exact same comparison method
# (eval/runners/eval_checkpoint.py --era5, see eval/data/era5_episodes.py).
#
# What carries over from the synthetic script, unchanged: the classical
# baselines (10 GP-MLE variants, 4 DKL variants, per-episode transformer,
# independence, GP-prior-RBF), the nested-CV best-of-baselines selection,
# both summary tables, and the resumable per-episode baseline/results caches.
#
# What CANNOT carry over: real ERA5 has no generating kernel, so the
# "Oracle (prior)" row and the analytic GP prior/posterior Y-space rows are
# nan, and the shared z_test every row is scored against is the frozen-TabICL
# K-fold PIT rather than a ground-truth marginal. The Y-space total-NLL
# table (each method's own predictive density, scored at the same real
# y_test) is the one to read on real data — it is a proper scoring rule
# regardless. --z_train_source=oracle is rejected.
#
# Requires the global ERA5 corpus to be cached first (once):
#     python eval/data/fetch_era5_global.py --start 2023-01 --n-months 12 \
#         --cache-dir ./eval/data/cache/era5_global_val
#
# Submit with:
#     mkdir -p logs
#     oarsub -S "./scripts/eval_checkpoint_era5.sh --ckpt ./checkpoints/<run>/step_XXXXXXX.pt"
#
# Any eval_checkpoint.py flag passes through, e.g. a different geometry:
#     oarsub -S "./scripts/eval_checkpoint_era5.sh --ckpt <...> --era5_grid_size 16 --n_episodes 800"
#
# Disk: the baseline cache holds 16 N x N correlation matrices per episode, so
# its per-episode size is quadratic in N. At this script's default geometry
# (grid 24 => N=546) that is ~20 MB/episode, i.e. ~7.8 GB for 400 episodes --
# about 4.5x the ~4.3 MB/episode the synthetic default (N=256) costs. Halving
# --era5_grid_size roughly quarters it.
#
# Measured runtime at the defaults on 32 physical cores: ~8.4 s/episode wall
# for the baseline fit pass (~56 min for 400), plus a short scoring pass.
# Episodes here are cheaper than the synthetic ones (P=30, d_x=6 vs P=32,
# d_x=9), so do not size a reservation off eval_checkpoint.py's synthetic
# 78.6 s/episode figure.
#
# Sharding across an OAR array: every episode is a pure function of
# (--seed, its GLOBAL index), so --n_episodes 100 with --episode_offset
# 0/100/200/300 covers exactly the same 400 episodes as one --n_episodes 400
# run. Give each shard its own --results_cache; they can SHARE one
# --baseline_cache (per-episode shard files, written as each fit completes).

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# ----- Env -----
source ~/thoth_storage/miniconda3/etc/profile.d/conda.sh
conda activate multivariate-icl
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="${ERA5_EVAL_OUT_DIR:-./eval/results/era5}"
mkdir -p "$OUT_DIR"

echo "[$(date +%H:%M:%S)] OAR job ${OAR_JOB_ID:-local} — host: $(hostname)"
echo "[$(date +%H:%M:%S)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[$(date +%H:%M:%S)] CPU cores available to this job: $(python -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null || nproc)"
echo "[$(date +%H:%M:%S)] Evaluating checkpoint on REAL ERA5 episodes..."
echo "    args: $*"

# Defaults chosen here rather than in argparse so a bare invocation is a
# complete, several-hundred-episode run; every one is overridable because
# "$@" comes last and argparse takes the LAST occurrence of a flag.
#
# --baseline_cache is deliberately NOT per-checkpoint: baseline fitting is
# ~98% of the runtime and is checkpoint-independent, so a second --ckpt over
# the same episodes reuses every fit and only redoes the cheap ICL forward
# pass. --results_cache IS checkpoint-dependent; override it per checkpoint.
# -u: unbuffered, so a run killed at its walltime still shows its progress.
python -u eval/runners/eval_checkpoint.py \
    --era5 \
    --n_episodes 400 \
    --era5_grid_size 24 \
    --era5_n_context 30 \
    --baseline_cache "$OUT_DIR/era5_g24_baseline_cache.pt" \
    --results_cache "$OUT_DIR/era5_g24_results_partial.json" \
    --out_dir "$OUT_DIR" \
    "$@"

echo "[$(date +%H:%M:%S)] Evaluation complete."
