"""Reprint eval_checkpoint.py's summary tables from one or more --results_cache files.

Usage:
    python eval/runners/summarize_results_cache.py <results_cache.json> [...] \
        [--era5] [--max_episodes N]

--era5 labels the tables as real ERA5 (inferred from the cached fingerprint).
--max_episodes keeps the first N by episode index.
"""

from __future__ import annotations

import argparse
import json


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

    # Labels come from the cached fingerprint.
    if era5 is None:
        # The era5 settings live under the fingerprint's "baseline" entry.
        era5 = bool((fp.get("baseline") or {}).get("era5"))
    z_src = fp.get("z_train_source") or "tabicl"
    ckpt = fp.get("ckpt")

    # Sort by episode index.
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
    # Autoregressive footnote settings come from the fingerprint.
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
