#!/bin/bash
# One-time: the joint-track venv = the project's locked environment (torch, tabpfn, kumo as in uv.lock)
# + TabArena core (+ preprocessing, data-foundry), so a TabArena model can call copula_inter's marginals
# and copulas. Separate from the project .venv and from the tabarena-env venv; uv.lock is not modified.
#     oarsub -l host=1/core=8,walltime=1:30:00 "./scripts/beyondarena/install_joint_venv.sh [--dry-run]"
set -euxo pipefail
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
VENV="${BEYONDARENA_JOINT_VENV:-$HOME/thoth_storage/beyondarena-joint-venv}"
TA="${TABARENA_SRC:-$HOME/thoth_storage/tabarena-env/tabarena/packages/tabarena}"
UV_PROJECT_ENVIRONMENT="$VENV" uv sync --locked --extra dev --extra gpu --extra tabpfn --extra kumo --extra era5
VIRTUAL_ENV="$VENV" uv pip install --prerelease=allow ${1:-} -e "$TA[preprocessing,data-foundry]" statsmodels
VIRTUAL_ENV="$VENV" uv pip freeze > "$VENV/freeze.txt"
