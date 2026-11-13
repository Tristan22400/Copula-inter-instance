#!/bin/bash
# Submit the thesis evaluation tables again with posterior-mode GP baselines (2026-11-13).
#
# Before 2026-11-13 the synthetic tables scored every GP-MLE/DKL row with its fitted
# PRIOR at the queries (data.oracle_mode=prior), which ignores the context. This reruns
# the same cells -- same checkpoint, marginal, episode distribution (conf/eval_<cell>.yaml,
# pinned from the old runs' cache fingerprints), seed and launch flags -- with the defaults
# baselines.gp_predict=posterior baselines.gp_mean=constant, into fresh caches.
#
# Run from the checkout on the Grid5000 frontend:
#     ./scripts/rerun_gp_posterior_tables.sh smoke era5 table2_systematic   # 12 episodes each
#     ./scripts/rerun_gp_posterior_tables.sh full all                       # 500 episodes each
# Cells: table2_{rbf,matern32,rational_quadratic,systematic},
#        varshape_{rbf,matern32,rational_quadratic,systematic}, era5; "all" = every cell.
# Outputs: eval/results/gp_posterior_<mode>/<cell>/ (tables in logs/rerun_<cell>_<mode>_<jobid>.out,
# per-episode scores in episodes.json for paired_compare).

set -euo pipefail

mode=${1:?usage: $0 smoke|full <cell>... | all}
shift
case "$mode" in
  smoke) n_episodes=12; walltime=2:00:00 ;;
  full) n_episodes=500; walltime=16:00:00 ;;
  *) echo "mode must be smoke or full" >&2; exit 1 ;;
esac

synthetic=(table2_rbf table2_matern32 table2_rational_quadratic table2_systematic
           varshape_rbf varshape_matern32 varshape_rational_quadratic varshape_systematic)
cells=("$@")
[[ "${cells[0]:-}" == all ]] && cells=("${synthetic[@]}" era5)

CKPT=checkpoints/copula_nano/copula-finetune-marginal-float32/step_0630000.pt
MARGINAL=checkpoints/marginal/ablations/marginal_finetune_era5_run1/step_0177600_final.pt
mkdir -p logs

for cell in "${cells[@]}"; do
  out="eval/results/gp_posterior_${mode}/${cell}"
  mkdir -p "$out"
  common=(
    "ckpt=$CKPT" "marginal.tabicl_ckpt=$MARGINAL" "n_episodes=$n_episodes"
    "baselines.cache=$out/baseline_cache.pt" "output.results_cache=$out/results_partial.json"
    "output.out_dir=$out" "output.dump_episodes=$out/episodes.json"
  )
  case "$cell" in
    era5)
      cmd=(./scripts/eval_checkpoint_era5.sh "${common[@]}") ;;
    table2_systematic)
      cmd=(./scripts/eval_checkpoint.sh "config=conf/eval_${cell}.yaml" alternate_noncomposite=false "${common[@]}") ;;
    table2_*)
      cmd=(./scripts/eval_checkpoint.sh "config=conf/eval_${cell}.yaml" "${common[@]}") ;;
    varshape_systematic)
      cmd=(./scripts/eval_checkpoint.sh "config=conf/eval_${cell}.yaml" alternate_noncomposite=false
           selection.min_test_points=16 "${common[@]}") ;;
    varshape_*)
      cmd=(./scripts/eval_checkpoint.sh "config=conf/eval_${cell}.yaml" selection.min_test_points=16 "${common[@]}") ;;
    *) echo "unknown cell: $cell" >&2; exit 1 ;;
  esac
  # One GPU (never an L4) and its share of the node's cores: baseline fitting runs one process
  # per allocated core, and a single GPU starts far sooner than a whole free node.
  oarsub -q p1 -n "rerun-${cell}-${mode}" -l "gpu=1,walltime=$walltime" \
    -p "gpu_count > 0 AND gpu_model NOT LIKE '%L4%'" \
    -O "logs/rerun_${cell}_${mode}_%jobid%.out" -E "logs/rerun_${cell}_${mode}_%jobid%.err" \
    "ERA5_EVAL_OUT_DIR=$out ${cmd[*]}"
done
