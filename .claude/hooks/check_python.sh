#!/usr/bin/env bash
# PostToolUse hook for Edit|Write: format the edited .py file with ruff, then
# report (exit 2 -> shown to the agent) its lint errors and every mypy error in
# the project. The project type-checks clean, so any mypy error is new --
# including callers in other files broken by a signature change.
set -uo pipefail

f=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty')
[[ "$f" == *.py && -f "$f" ]] || exit 0

root=$(git -C "$(dirname "$f")" rev-parse --show-toplevel 2>/dev/null) || exit 0
case "$f" in
  "$root"/tabicl_upstream/* | "$root"/*_upstream/*) exit 0 ;;
esac

env_bin=${COPULA_ENV_BIN:-/srv/storage/thoth1@storage4.grenoble.grid5000.fr/trmartin/miniconda3/envs/multivariate-icl/bin}
tool() {
  if [[ -x "$root/.venv/bin/$1" ]]; then echo "$root/.venv/bin/$1"; return; fi
  command -v "$1" 2>/dev/null || { [[ -x "$env_bin/$1" ]] && echo "$env_bin/$1"; }
}
ruff=$(tool ruff)
mypy=$(tool mypy)

problems=""
if [[ -n "$ruff" ]]; then
  "$ruff" format --quiet "$f"
  "$ruff" check --quiet --fix --select I "$f" >/dev/null 2>&1
  lint=$("$ruff" check --quiet --output-format concise "$f" 2>&1) || problems+="$lint"$'\n'
else
  echo "check_python.sh: ruff not found (pip install ruff)" >&2
fi

if [[ -n "$mypy" ]]; then
  # Local-disk cache (the checkout may sit on network storage): ~2 s warm.
  cache="${TMPDIR:-/tmp}/mypy-cache-$(printf '%s' "$root" | md5sum | cut -c1-8)"
  out=$(cd "$root" && timeout 150 "$mypy" --cache-dir "$cache" --no-error-summary 2>&1)
  rc=$?
  if (( rc == 124 )); then
    # Cold cache: finish it in the background instead of blocking this edit.
    (cd "$root" && setsid nohup "$mypy" --cache-dir "$cache" >/dev/null 2>&1 &)
    echo "check_python.sh: mypy cache is cold; warming it in the background, types unchecked for this edit" >&2
  elif (( rc != 0 )); then
    errors=$(grep -E ": (error|note):" <<<"$out")
    n=$(grep -c ": error:" <<<"$errors")
    problems+="mypy: $n error(s)"$'\n'"$(head -n 40 <<<"$errors")"$'\n'
  fi
else
  echo "check_python.sh: mypy not found (uv sync --extra dev)" >&2
fi

if [[ -n "$problems" ]]; then
  printf '%s' "$problems" >&2
  exit 2
fi
exit 0
