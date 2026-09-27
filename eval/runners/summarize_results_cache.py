"""Reprint eval_checkpoint's summary tables from one or more output.results_cache files.

Usage:
    python -m eval.runners.summarize_results_cache caches=[<results_cache.json>,...] \
        [era5=true] [max_episodes=N]

era5 labels the tables as real ERA5 (default: read from the cached fingerprint);
max_episodes keeps the first N by episode index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from omegaconf import MISSING

from eval.runners.eval_tables import _ar_note, _print_table, _print_total_nll_table
from eval.runners.hydra_cli import hydra_entry


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
    # New caches key the checkpoint by content (ckpt_identity); pre-refactor ones stored its path.
    identity = fp.get("ckpt_identity") or {}
    ckpt = fp.get("ckpt") or (
        f"sha256:{identity['sha256'][:12]}" if "sha256" in identity else identity.get("reference")
    )

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
        all_total,
        z_train_source=z_src,
        era5=era5,
        ar_note=_ar_note(
            all_total,
            fp.get("ar_order") or "random",
            fp.get("ar_conditioning") or "teacher_forcing",
            fp.get("ar_max_context"),
        ),
    )


@dataclass
class SummarizeSpec:
    """Reprint eval_checkpoint's summary tables from results caches."""

    # One or more output.results_cache JSON files.
    caches: list[str] = MISSING
    # Force the real-ERA5 table labelling on/off; null reads it from each file's fingerprint.
    era5: bool | None = None
    # Keep the first N episodes by index.
    max_episodes: int | None = None


def run(args: SummarizeSpec) -> None:
    for path in args.caches:
        summarize(path, era5=args.era5, max_episodes=args.max_episodes)


main = hydra_entry("summarize_results_cache", SummarizeSpec, run)

if __name__ == "__main__":
    main()
