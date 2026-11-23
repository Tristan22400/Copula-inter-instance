#!/bin/bash
# Submit the variable-size synthetic and ERA5 evaluations with posterior-mode GP baselines.
#
# Each variable-size config pins the episode distribution for paired comparisons.
# Use --gp-mean linear to fit an affine mean for GP-MLE and DKL in a separate cache.
#
# Run from the checkout on the Grid5000 frontend:
#     ./scripts/rerun_gp_posterior_tables.sh smoke era5 varshape_systematic   # 12 episodes each
#     ./scripts/rerun_gp_posterior_tables.sh full all                        # 500 episodes each
#     ./scripts/rerun_gp_posterior_tables.sh --gp-mean linear full synthetic
# Cells: varshape_{rbf,matern32,rational_quadratic,systematic}, era5.
# Use "synthetic" for the four variable-size cells, or "all" to include ERA5 too.
# Each shorthand takes the place of the entire cell list.
# Outputs: eval/results/gp_posterior_<mode>[_linear]/<cell>/, with per-episode
# scores in episodes.json for paired_compare and tables in logs/g5k/.

set -euo pipefail

gp_mean=constant
if [[ ${1:-} == --gp-mean ]]; then
  gp_mean=${2:?expected constant or linear after --gp-mean}
  shift 2
fi
case "$gp_mean" in
  constant) mean_suffix= ;;
  linear) mean_suffix=_linear ;;
  *) echo "GP mean must be constant or linear" >&2; exit 2 ;;
esac

mode=${1:?usage: $0 [--gp-mean constant|linear] smoke|full <cell>... | synthetic | all}
shift
case "$mode" in
  smoke) n_episodes=12; walltime=2:00:00 ;;
  full) n_episodes=500; walltime=24:00:00 ;;
  *) echo "mode must be smoke or full" >&2; exit 1 ;;
esac

synthetic=(varshape_rbf varshape_matern32 varshape_rational_quadratic varshape_systematic)
valid_cells=("${synthetic[@]}" era5)
cells=("$@")
case "${cells[0]:-}" in
  all|synthetic)
    if (( ${#cells[@]} != 1 )); then
      printf '%s must be used on its own\n' "${cells[0]}" >&2
      exit 2
    fi
    if [[ $1 == all ]]; then
      cells=("${valid_cells[@]}")
    else
      cells=("${synthetic[@]}")
    fi ;;
  '') echo "supply at least one cell, synthetic, or all" >&2; exit 2 ;;
esac

# Reject an invalid cell before any part of the request is submitted.
for cell in "${cells[@]}"; do
  for valid_cell in "${valid_cells[@]}"; do
    if [[ $cell == "$valid_cell" ]]; then
      continue 2
    fi
  done
  printf 'unknown cell: %s\nValid cells: %s\n' "$cell" "${valid_cells[*]}" >&2
  exit 1
done

CKPT=checkpoints/copula_nano/copula-finetune-marginal-float32/step_0630000.pt
MARGINAL=checkpoints/marginal/ablations/marginal_finetune_era5_run1/step_0177600_final.pt
failed=0

for cell in "${cells[@]}"; do
  out="eval/results/gp_posterior_${mode}${mean_suffix}/${cell}"
  common=(
    "ckpt=$CKPT" "marginal.tabicl_ckpt=$MARGINAL" "n_episodes=$n_episodes" "seed=42"
    "baselines.gp_predict=posterior" "baselines.gp_mean=$gp_mean"
    "baselines.cache=$out/baseline_cache.pt" "output.results_cache=$out/results_partial.json"
    "output.out_dir=$out" "output.dump_episodes=$out/episodes.json"
  )
  case "$cell" in
    era5)
      cmd=(./scripts/eval_checkpoint_era5.sh "${common[@]}") ;;
    varshape_systematic)
      cmd=(./scripts/eval_checkpoint.sh "config=conf/eval_${cell}.yaml" alternate_noncomposite=false
           selection.min_test_points=16 "${common[@]}") ;;
    *)
      cmd=(./scripts/eval_checkpoint.sh "config=conf/eval_${cell}.yaml" selection.min_test_points=16 "${common[@]}") ;;
  esac
  mkdir -p "$out"
  if ! /usr/bin/python3 -u scripts/g5k_gpus.py submit --walltime "$walltime" \
    --name "rerun-${cell}-${mode}${mean_suffix}" --env "ERA5_EVAL_OUT_DIR=$out" -- "${cmd[@]}"; then
    echo "Submission failed for $cell" >&2
    failed=1
  fi
done

exit "$failed"
