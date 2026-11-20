#!/bin/bash
# Separate venv for the neural-process benchmark generators (Bruinsma et al. 2023, `neuralprocesses`).
# Only scripts/npbench/export_*.py run in it; scoring runs in the main venv (scripts/_env.sh).
# Usage (on a node): bash scripts/npbench/install_np_venv.sh [venv_dir]
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
venv="${1:-$here/.venv-npbench}"
base_python="${COPULA_VENV:-$here/.venv}/bin/python"
"$base_python" -c 'import sys; print(sys.base_prefix)' >/dev/null
real_python="$("$base_python" -c 'import sys, os; print(os.path.join(sys.base_prefix, "bin", "python3"))')"
"$real_python" -m venv "$venv"
"$venv/bin/python" -m pip install -q --upgrade pip
"$venv/bin/python" -m pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
"$venv/bin/python" -m pip install -q neuralprocesses==0.2.7
"$venv/bin/python" -c "import neuralprocesses.torch as nps, torch; print('neuralprocesses ok', torch.__version__)"
"$venv/bin/python" -m pip freeze > "$here/scripts/npbench/np_venv_freeze.txt"
