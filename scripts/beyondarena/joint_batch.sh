#!/bin/bash
# Run scripts/beyondarena/joint.sh for several datasets in one job (one model, one GPU).
#
#     oarsub -l gpu=1,walltime=24:00:00 "./scripts/beyondarena/joint_batch.sh <model> <shard> [key=value ...]"
#
# <shard> names a file under scripts/beyondarena/shards/ with one dataset per line. A dataset that
# fails is logged and skipped (the next run resumes it); the job exits non-zero if any failed.
#OAR -n beyondarena-joint-batch
#OAR -l gpu=1,walltime=24:00:00
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MODEL="$1"
SHARD="$2"
shift 2
failed=()
while read -r ds; do
    [[ -z "$ds" || "$ds" == \#* ]] && continue
    echo "=== $ds ($MODEL $*)"
    ./scripts/beyondarena/joint.sh "$ds" "$MODEL" "$@" || failed+=("$ds")
done < "scripts/beyondarena/shards/$SHARD"
if (( ${#failed[@]} )); then
    echo "FAILED: ${failed[*]}" >&2
    exit 1
fi
