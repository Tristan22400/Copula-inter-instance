#!/bin/bash
#OAR -n risk-bo
#OAR -l gpu=1,walltime=6:00:00

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
configure_cuda_devices risk-bo
main_root="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
exec python -m eval.runners.risk_bo checkpoint_dir="$main_root/checkpoints" "$@"
