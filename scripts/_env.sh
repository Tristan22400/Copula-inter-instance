# Shared job setup: cd to this checkout's root, activate the conda env, and put
# this checkout's packages first on PYTHONPATH. Source it from scripts/*.sh.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_BASE="${CONDA_BASE:-$HOME/thoth_storage/miniconda3}"
CONDA_ENV="${CONDA_ENV:-multivariate-icl}"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    source "$CONDA_BASE/bin/activate" "$CONDA_ENV"
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:$REPO_ROOT/tabicl_upstream/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
