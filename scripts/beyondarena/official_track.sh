#!/bin/bash
# BeyondArena official track (scripts/beyondarena/official_track.py) in the separate tabarena venv.
#
#     oarsub -l gpu=1,walltime=24:00:00 -O logs/%jobid%.out -E logs/%jobid%.err \
#         "./scripts/beyondarena/official_track.sh method=ft datasets='[<name>,...]'"
#
# The tabarena venv (TABARENA_VENV, default ~/thoth_storage/tabarena-env/.venv) is built once by
# scripts/beyondarena/install_tabarena.sh. It is NOT the project venv: the project's PYTHONPATH
# (vendored tabicl_upstream first) is dropped so that TabArena's own tabicl 2.2.0 is imported.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# shellcheck disable=SC1091
source scripts/_env.sh  # only for configure_cuda_devices (excludes the GPUs known not to work)
configure_cuda_devices beyondarena-official
deactivate
TABARENA_VENV="${TABARENA_VENV:-$HOME/thoth_storage/tabarena-env/.venv}"
# shellcheck disable=SC1091
source "$TABARENA_VENV/bin/activate"
export PYTHONPATH=""
export PYTHONNOUSERSITE=1
python -c "import importlib.metadata as m, tabicl; print('tabicl', m.version('tabicl'), tabicl.__file__)"
exec python scripts/beyondarena/official_track.py "$@"
