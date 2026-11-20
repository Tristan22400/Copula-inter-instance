#!/bin/bash
# One-off: generator venv + EEG raw data (runs detached on a node; log in logs/npbench/setup.log).
set -uo pipefail
cd "$(dirname "$0")/../.."
if ! .venv-npbench/bin/python -I -c "import neuralprocesses.torch" 2>/dev/null; then
  rm -rf .venv-npbench
  bash scripts/npbench/install_np_venv.sh || { echo "INSTALL FAILED"; exit 1; }
fi
bash scripts/npbench/fetch_eeg.sh "${NPBENCH_RAW:?}/eeg_raw" && echo "SETUP DONE" || echo "FETCH FAILED"
