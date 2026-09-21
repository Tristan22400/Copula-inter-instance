"""summarize_results_cache.py — reprint eval_checkpoint.py's summary tables
from a --results_cache file, without re-running anything.

eval_checkpoint.py only prints its tables after its episode loop finishes, so
a run killed at its OAR walltime leaves behind every episode it scored (the
--results_cache is written after each one) and no table to read them from.
This prints exactly the same tables from that file, over however many
episodes actually completed.

It is also the cheapest way to compare two checkpoints that were scored over
the same episodes: point it at each one's --results_cache in turn.

Usage
-----
    python eval/runners/summarize_results_cache.py <results_cache.json> [...]

        [--era5]   # label the tables as real-ERA5 (see eval_checkpoint.py's
                   # --era5: the oracle rows are structurally nan and the
                   # shared z_test is the frozen-TabICL PIT, not ground truth).
                   # Inferred automatically when the cached fingerprint says so.
        [--max_episodes N]   # summarize only the first N by episode index

The tables' own row semantics are documented on _print_table and
_print_total_nll_table in eval_checkpoint.py — this module deliberately calls
those rather than reimplementing them, so a change to either shows up here
too.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.runners.eval_checkpoint import (  # noqa: E402
    _ar_note,
    _print_table,
    _print_total_nll_table,
)


def summarize(path: str, era5: bool | None = None, max_episodes: int | None = None) -> None:
    with open(path) as fh:
        blob = json.load(fh)

    fp = blob.get("fingerprint", {}) or {}
    entries = blob.get("episodes", {}) or {}
    if not entries:
        print(f"{path}: no scored episodes in this cache.")
        return

    # The fingerprint carries the whole run config, so the labels below are
    # read off the file rather than guessed from the filename.
    if era5 is None:
        # The stored fingerprint is eval_checkpoint.py's _results_fingerprint,
        # which nests the baseline fingerprint under "baseline" -- and the
        # era5 sub-dict is set on that baseline fingerprint. There is no
        # top-level "era5" key to fall back to.
        era5 = bool((fp.get("baseline") or {}).get("era5"))
    z_src = fp.get("z_train_source") or "tabicl"
    ckpt = fp.get("ckpt")

    # Sort by episode index, not by dict order: a resumed run appends its new
    # episodes after the reused ones, so insertion order is not episode order.
    keys = sorted(entries, key=lambda k: int(k))
    if max_episodes is not None:
        keys = keys[:max_episodes]

    all_nlls = [entries[k]["nlls"] for k in keys]
    all_total = [entries[k]["total_nlls"] for k in keys]

    print(f"\n=== {path} ===")
    if ckpt:
        print(f"checkpoint: {ckpt}")
    print(f"episodes scored: {len(keys)} (indices {keys[0]}..{keys[-1]})")

    _print_table(all_nlls, z_train_source=z_src, era5=era5)
    # The chain's settings are in the fingerprint too, so the autoregressive
    # row keeps its footnote (including the --ar_conditioning=sample warning)
    # when the table is reprinted from a cache instead of from a live run.
    _print_total_nll_table(
        all_total, z_train_source=z_src, era5=era5,
        ar_note=_ar_note(all_total, fp.get("ar_order") or "random",
                         fp.get("ar_conditioning") or "teacher_forcing",
                         fp.get("ar_max_context")),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caches", nargs="+", help="one or more --results_cache JSON files")
    ap.add_argument("--era5", action=argparse.BooleanOptionalAction, default=None,
                    help="force the real-ERA5 table labelling on/off (default: read "
                         "it from each file's stored fingerprint)")
    ap.add_argument("--max_episodes", type=int, default=None)
    args = ap.parse_args()
    for path in args.caches:
        summarize(path, era5=args.era5, max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()
