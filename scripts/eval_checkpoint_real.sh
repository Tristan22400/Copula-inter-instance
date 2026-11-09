#!/bin/bash
#OAR -n CopulaEvalReal
#OAR -l host=1,walltime=24:00:00
#OAR -p gpu_count > 0
#OAR -O logs/eval_real_%jobid%.out
#OAR -E logs/eval_real_%jobid%.err
#OAR -q p1
#
# Evaluate an ICL checkpoint on one real-data benchmark cell
# (eval_checkpoint real.dataset=<name>, eval/data/real/<name>.py): the same
# baselines, nested-CV best_baseline, autoregressive row and tables as ERA5.
# Read the total Y-space NLL table; the copula-only table needs a shared
# z_test and real data has no oracle.
#
# host=1 reserves the whole node: baseline fitting is ~98 % of the runtime and
# CPU-bound across baselines.workers processes (see eval_checkpoint_era5.sh).
#
# Required overrides: ckpt=<copula ckpt>, marginal.tabicl_ckpt=<the marginal it was
# trained with: tabicl.pit_ckpt in its config>, real.dataset=<name>, and
# output.dump_episodes=<cell dump> (paired_compare and real_calibration_gate read it).
#     mkdir -p logs
#     oarsub -q p1 -p "gpu_count > 0" -l host=1,walltime=24:00:00 \
#         "REAL_CELL=kaust_1a_2_random ./scripts/eval_checkpoint_real.sh ckpt=<ckpt> marginal.tabicl_ckpt=<marginal> \
#         real.dataset=kaust real.kaust.variant=1a_2 real.kaust.mask=random \
#         output.dump_episodes=results/real/kaust/1a_2_random.json"
# (OAR does not forward the submitting shell's environment, so REAL_CELL goes inside the command.)
#
# Raw files are read from REAL_DATA_DIR (default: the MAIN checkout's
# eval/data/cache, also from a git worktree, so the data is placed once).
# REAL_CELL names the cell (dataset, mask, features, P): its baselines.cache is
# REAL_OUT_DIR/<cell>_baseline_cache.pt, shared across checkpoints (fits do not
# depend on the checkpoint) but never across cells (the cache key holds the
# dataset and ep_i, not the mask or feature set). output.results_cache is per
# checkpoint: REAL_CKPT_TAG (default "ckpt") distinguishes them.
#
# Sharding: episode ep_i is a pure function of (seed, ep_i), so
# n_episodes=100 episode_offset=0/100/200/300 covers the same 400 episodes as one
# run; give each shard its own output.results_cache and output.dump_episodes.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

if [ -z "${REAL_DATA_DIR:-}" ]; then
    REAL_DATA_DIR="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)/eval/data/cache"
fi
export REAL_DATA_DIR
OUT_DIR="${REAL_OUT_DIR:-./results/real}"
CELL="${REAL_CELL:?set REAL_CELL to the cell name, e.g. kaust_1a_2_random}"
TAG="${REAL_CKPT_TAG:-ckpt}"
mkdir -p "$OUT_DIR"

echo "[$(date +%H:%M:%S)] OAR job ${OAR_JOB_ID:-local} — host: $(hostname)"
echo "[$(date +%H:%M:%S)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "[$(date +%H:%M:%S)] CPU cores: $(python -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null || nproc)"
echo "[$(date +%H:%M:%S)] REAL_DATA_DIR=$REAL_DATA_DIR"
echo "    args: $*"

# -u: unbuffered, so a run killed at its walltime still shows its progress.
python -u -m eval.runners.eval_checkpoint \
    n_episodes=400 \
    "baselines.cache=$OUT_DIR/${CELL}_baseline_cache.pt" \
    "output.results_cache=$OUT_DIR/${CELL}_${TAG}_results_partial.json" \
    "output.out_dir=$OUT_DIR" \
    "$@"

echo "[$(date +%H:%M:%S)] Evaluation complete."
