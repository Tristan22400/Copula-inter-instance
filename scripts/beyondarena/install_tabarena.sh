#!/bin/bash
# One-time: build the separate BeyondArena (tabarena) venv on cluster storage. Run as a CPU OAR job:
#     oarsub -l host=1/core=8,walltime=1:30:00 ./scripts/beyondarena/install_tabarena.sh
# It never touches the project's .venv or uv.lock (tabarena pins its own torch/tabpfn/autogluon).
set -euxo pipefail
export PATH=$HOME/.local/bin:$PATH
DEST="${TABARENA_ENV_DIR:-$HOME/thoth_storage/tabarena-env}"
REV="${TABARENA_REV:-cc507b84d1f32d2ca5b5704326bf4f086ee6519a}"  # the commit the study ran with
mkdir -p "$DEST" && cd "$DEST"
[ -d tabarena ] || git clone https://github.com/autogluon/tabarena.git
git -C tabarena fetch --depth 1 origin "$REV" 2>/dev/null || true
git -C tabarena checkout "$REV"
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --prerelease=allow -e "./tabarena/packages/tabarena[benchmark]"
uv pip install statsmodels
uv pip freeze > freeze.txt
