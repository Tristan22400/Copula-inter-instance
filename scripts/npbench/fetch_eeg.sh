#!/bin/bash
# Fetch the UCI EEG Database (Begleiter) for the neuralprocesses EEG generator.
# wbml downloads https://kdd.ics.uci.edu/databases/eeg/eeg_full.tar, which now returns 403. The
# current UCI archive (dataset 121) ships that tar's contents unpacked: eeg_full/<subject>.tar.gz
# (its eeg_full.tar entry is a dangling symlink). We rebuild eeg_full.tar from those files (at the
# tar's root, as wbml's extraction expects) at the path wbml's loader checks first
# (wbml.data.data_path("eeg", "eeg_full.tar")), so wbml's own extraction and parser run unchanged.
# Usage: bash scripts/npbench/fetch_eeg.sh <raw_dir> [venv_dir]
set -euo pipefail
raw="$1"; venv="${2:-.venv-npbench}"
mkdir -p "$raw"
[ -f "$raw/eeg+database.zip" ] || curl -sSfL -o "$raw/eeg+database.zip" https://archive.ics.uci.edu/static/public/121/eeg+database.zip
sha256sum "$raw/eeg+database.zip"
target="$("$venv/bin/python" -I -W ignore -c 'from wbml.data.data import data_path; print(data_path("eeg", "eeg_full.tar"))')"
mkdir -p "$(dirname "$target")"
[ -L "$target" ] && [ ! -e "$target" ] && rm "$target"   # the zip's dangling symlink
if [ ! -f "$target" ]; then
  rm -rf "$raw/eeg_full"
  unzip -q -o "$raw/eeg+database.zip" 'eeg_full/*.tar.gz' -d "$raw"
  n=$(ls "$raw/eeg_full" | grep -c 'tar.gz$')
  echo "subjects in eeg_full/: $n"
  (cd "$raw/eeg_full" && tar cf "$target.tmp" ./*.tar.gz) && mv "$target.tmp" "$target"
fi
ls -l "$target"
