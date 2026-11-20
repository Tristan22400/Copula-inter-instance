#!/bin/bash
# BeyondArena joint track inside the framework (scripts/beyondarena/joint_track.py), beyondarena-joint venv.
#
#     oarsub -l gpu=1,walltime=18:00:00 "./scripts/beyondarena/joint_track.sh marginals='[tabicl-ft]' \"datasets=[...]\""
#
# The venv (BEYONDARENA_JOINT_VENV, default ~/thoth_storage/beyondarena-joint-venv) is the project's locked
# environment plus TabArena core; build it once with scripts/beyondarena/install_joint_venv.sh.
#OAR -n beyondarena-joint-ta
#OAR -l gpu=1,walltime=18:00:00
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# shellcheck disable=SC1091
COPULA_VENV="${BEYONDARENA_JOINT_VENV:-$HOME/thoth_storage/beyondarena-joint-venv}" source scripts/_env.sh
configure_cuda_devices beyondarena-joint-ta
exec python scripts/beyondarena/joint_track.py "$@"
