#!/bin/bash
# One shard of one cell of the real-data grid (docs/workflows.md, "Real-data grid protocol").
#
# Usage (one OAR job per shard, whole GPU node):
#     oarsub -q p1 -p "gpu_count > 0" -l host=1,walltime=6:00:00 "./scripts/real_grid.sh <cell> <offset> <n>"
# runs episodes offset .. offset + n - 1 of <cell> at the reported seed (42) and writes
# results/real/<dataset>/<cell>_<tag>_o<offset>.json (paired_compare a='results/real/<dataset>/<cell>_<tag>_o*.json'
# merges the shards). The copula checkpoint and the marginal it was trained with come from
# REAL_CKPT and REAL_MARGINAL (default: the ERA5-report pair); REAL_CKPT_TAG names its results caches.
# `./scripts/real_grid.sh list` prints every cell.

set -euo pipefail

CK="${REAL_CKPT:-/home/trmartin/thoth_storage/copula-inter/checkpoints/copula_nano/copula-finetune-marginal-float32/step_0630000.pt}"
MG="${REAL_MARGINAL:-/home/trmartin/thoth_storage/copula-inter/checkpoints/marginal/ablations/marginal_finetune_era5_run1/step_0177600_final.pt}"
TAG="${REAL_CKPT_TAG:-nano}"

# The 24 CTR23 datasets the atom rule keeps (results/real/audit/ctr23_suite.json, 2026-11-05).
CTR23_KEPT="44957 44958 44959 44963 44964 44969 44973 44974 44975 44976 44977 44979 44980 44981 44983 44984 44989 44990 44992 41021 44960 44970 44994 45402"

cells() {
    echo sdwpf_f0_random sdwpf_f0_block sdwpf_f0_random_p16 sdwpf_f0_random_p64
    for v in 1a_2 1b_2 1a_1 1b_1; do echo kaust_${v}_random kaust_${v}_block; done
    echo kaust_1a_2_random_p64 kaust_1a_2_random_p256
    echo emnist_p50_random emnist_p50_block emnist_p200_random emnist_p200_block
    echo california_f0_random california_f0_block california_f1_random california_f1_block
    echo california_f0_random_p64 california_f0_random_p256
    echo meuse_f0 meuse_f1
    for d in $CTR23_KEPT; do echo ctr23_${d}_dyadic_knn ctr23_${d}_random; done
}

overrides() {
    case "$1" in
    sdwpf_f0_random) echo real.dataset=sdwpf real.sdwpf.features=f0 real.sdwpf.mask=random ;;
    sdwpf_f0_block) echo real.dataset=sdwpf real.sdwpf.features=f0 real.sdwpf.mask=block ;;
    sdwpf_f0_random_p16) echo real.dataset=sdwpf real.sdwpf.mask=random real.sdwpf.n_context=16 ;;
    sdwpf_f0_random_p64) echo real.dataset=sdwpf real.sdwpf.mask=random real.sdwpf.n_context=64 real.sdwpf.timestamp_tries=7 ;;
    kaust_*_random_p64) echo real.dataset=kaust real.kaust.variant=1a_2 real.kaust.mask=random real.kaust.n_context=64 ;;
    kaust_*_random_p256) echo real.dataset=kaust real.kaust.variant=1a_2 real.kaust.mask=random real.kaust.n_context=256 ;;
    kaust_*) v=${1#kaust_}; echo real.dataset=kaust real.kaust.variant=${v%_*} real.kaust.mask=${v##*_} ;;
    emnist_p*) p=${1#emnist_p}; echo real.dataset=emnist real.emnist.split=test real.emnist.n_context=${p%_*} real.emnist.mask=${p#*_} ;;
    california_f0_random_p64) echo real.dataset=california real.california.mask=random real.california.n_context=64 ;;
    california_f0_random_p256) echo real.dataset=california real.california.mask=random real.california.n_context=256 ;;
    california_*) f=${1#california_}; echo real.dataset=california real.california.features=${f%_*} real.california.mask=${f#*_} ;;
    meuse_*) echo real.dataset=meuse real.meuse.features=${1#meuse_} ;;
    ctr23_*) r=${1#ctr23_}; echo real.dataset=ctr23 real.ctr23.data_id=${r%%_*} real.ctr23.mask=${r#*_} ;;
    *) echo "unknown cell $1" >&2; return 2 ;;
    esac
}

if [ "${1:-}" = list ]; then cells; exit 0; fi
CELL="$1"; OFFSET="$2"; N="$3"
ARGS="$(overrides "$CELL")"
DS="${CELL%%_*}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="results/real/$DS"
mkdir -p "$OUT"
# The baseline cache is per cell and per checkpoint tag: its fingerprint holds the PIT marginal
# (and the copula rank), so two checkpoints with different marginals cannot share fits.
# shellcheck disable=SC2086
REAL_CELL="${CELL}_${TAG}" REAL_OUT_DIR="$OUT" REAL_CKPT_TAG="o${OFFSET}" ./scripts/eval_checkpoint_real.sh \
    ckpt="$CK" marginal.tabicl_ckpt="$MG" $ARGS \
    n_episodes="$N" episode_offset="$OFFSET" \
    output.dump_episodes="$OUT/${CELL}_${TAG}_o${OFFSET}.json"
