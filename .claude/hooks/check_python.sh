#!/usr/bin/env bash
# PostToolUse hook for Edit|Write: format the edited .py file with ruff, then
# report (exit 2 -> shown to the agent) lint errors in the file and mypy errors
# on the lines that differ from HEAD. Pre-existing type errors elsewhere in the
# file are not reported.
set -uo pipefail

f=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty')
[[ "$f" == *.py && -f "$f" ]] || exit 0

root=$(git -C "$(dirname "$f")" rev-parse --show-toplevel 2>/dev/null) || exit 0
case "$f" in
  "$root"/tabicl_upstream/* | "$root"/*_upstream/*) exit 0 ;;
esac

env_bin=${COPULA_ENV_BIN:-/srv/storage/thoth1@storage4.grenoble.grid5000.fr/trmartin/miniconda3/envs/multivariate-icl/bin}
tool() { command -v "$1" 2>/dev/null || { [[ -x "$env_bin/$1" ]] && echo "$env_bin/$1"; }; }
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

if [[ -n "$mypy" && "$f" == "$root"/@(src|eval|inference)/* ]]; then
  changed=$(git -C "$root" diff -U0 HEAD -- "$f" 2>/dev/null |
    sed -n 's/^@@ .*+\([0-9]*\)\(,\([0-9]*\)\)\? @@.*/\1 \3/p')
  if ! git -C "$root" ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    changed="1 1000000"
  fi
  if [[ -n "$changed" ]]; then
    rel=${f#"$root"/}
    out=$(cd "$root" && timeout 60 "$mypy" --no-error-summary "$rel" 2>/dev/null | grep ": error:")
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      ln=$(cut -d: -f2 <<<"$line")
      while read -r start count; do
        count=${count:-1}
        if (( ln >= start && ln < start + count )); then problems+="$line"$'\n'; break; fi
      done <<<"$changed"
    done <<<"$out"
  fi
fi

if [[ -n "$problems" ]]; then
  printf '%s' "$problems" >&2
  exit 2
fi
exit 0
