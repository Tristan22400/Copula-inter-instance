"""Fine-tune a copula checkpoint on real ARCO-ERA5 data by running copula_inter.train with ERA5 overrides.

Runs python -m copula_inter.train training.resume_ckpt=...
training.live_generation=true training.live_source=era5 .... Needs a local
corpus (python eval/data/fetch_era5_global.py --start 2022-01 --n-months 24).

Usage:
    python -m copula_inter.finetune_era5 --ckpt kernel-sweep-all-tabicl-retrain-15k
    python -m copula_inter.finetune_era5 --ckpt <path> --dry-run
    python -m copula_inter.finetune_era5 --ckpt <path> --marginal tabldm
    python -m copula_inter.finetune_era5 --ckpt <path> -- era5_live.grid_size_max=32 wandb.entity=me
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from copula_inter.backend_registry import MARGINAL_BACKENDS

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--ckpt", required=True, help="Checkpoint to finetune: a path, or a CHECKPOINT_FAMILIES name[:step]."
    )
    p.add_argument(
        "--corpus-dir",
        default="./eval/data/cache/era5_global",
        help="Local global-ERA5 corpus dir (see eval/data/fetch_era5_global.py).",
    )
    p.add_argument(
        "--val-corpus-dir",
        default=None,
        help="Separate corpus dir for the fixed validation set (e.g. a held-out year fetched into its own "
        "cache dir), disjoint from --corpus-dir's training years. Default: falls back to --corpus-dir "
        "(validates on a different random slice of the SAME date range training draws from).",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Total finetune steps (default 10000 -- short, unlike a from-scratch pretraining run).",
    )
    p.add_argument(
        "--warmup-steps", type=int, default=200, help="LR warmup steps for the finetune schedule (default 200)."
    )
    p.add_argument(
        "--muon-lr",
        type=float,
        default=4.0e-5,
        help="Peak Muon LR (default 4e-5 -- a finetune-scale fraction of a typical pretraining peak).",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grid-size-min", type=int, default=None)
    p.add_argument("--grid-size-max", type=int, default=None)
    p.add_argument("--box-deg-min", type=float, default=None)
    p.add_argument("--box-deg-max", type=float, default=None)
    p.add_argument(
        "--ckpt-dir",
        default=None,
        help="Output checkpoint dir (default: checkpoints/era5-finetune-<basename(ckpt)>-<timestamp>).",
    )
    p.add_argument(
        "--keep-schedule",
        action="store_true",
        help="Continue the source checkpoint's own cosine LR schedule (training.resume_reset_schedule=false) "
        "instead of a fresh warmup/decay over --steps (default: fresh schedule -- this is a deliberate warm "
        "start into a new data regime, not a continuation of the original run).",
    )
    p.add_argument(
        "--marginal",
        default=None,
        choices=MARGINAL_BACKENDS,
        help="Marginal backend for the ERA5 PIT (default: leave data.z_train_source "
        "as configured, i.e. the frozen TabICL). Shorthand for the "
        "data.z_train_source=<v> Hydra override -- see "
        "eval/spatial/marginal_backends.py for what each backend is, and "
        "era5_live_dataset.py::_resolve_marginal for how it is read here.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the resulting train.py command without running it.")
    p.add_argument("overrides", nargs=argparse.REMAINDER, help="Extra raw Hydra overrides, e.g. -- wandb.entity=me")
    args = p.parse_args()

    from eval.configs.checkpoints import resolve_checkpoint

    args.ckpt = os.path.abspath(resolve_checkpoint(args.ckpt))
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"--ckpt not found: {args.ckpt}")
    if not os.path.isdir(args.corpus_dir) or not os.listdir(args.corpus_dir):
        raise FileNotFoundError(
            f"--corpus-dir {args.corpus_dir!r} is empty or missing -- fetch a corpus first:\n"
            "  python eval/data/fetch_era5_global.py --start 2022-01 --n-months 24"
        )
    if args.val_corpus_dir is not None and (
        not os.path.isdir(args.val_corpus_dir) or not os.listdir(args.val_corpus_dir)
    ):
        raise FileNotFoundError(
            f"--val-corpus-dir {args.val_corpus_dir!r} is empty or missing -- fetch a held-out-year corpus first, "
            "e.g.:\n  python eval/data/fetch_era5_global.py --start 2023-01 --n-months 12 --cache-dir <val dir>"
        )

    ckpt_dir = args.ckpt_dir or os.path.join(
        "checkpoints",
        f"era5-finetune-{os.path.splitext(os.path.basename(args.ckpt))[0]}-{int(time.time())}",
    )

    overrides = [
        f"training.resume_ckpt={args.ckpt}",
        f"training.resume_reset_schedule={'false' if args.keep_schedule else 'true'}",
        "training.live_generation=true",
        "training.live_source=era5",
        f"training.steps={args.steps}",
        f"training.warmup_steps={args.warmup_steps}",
        f"training.muon_lr={args.muon_lr}",
        f"training.batch_size={args.batch_size}",
        "training.aux_mae_weight=0.0",
        f"training.ckpt_dir={ckpt_dir}",
        f"era5_live.corpus_dir={args.corpus_dir}",
    ]
    if args.marginal is not None:
        overrides.append(f"data.z_train_source={args.marginal}")
    if args.val_corpus_dir is not None:
        overrides.append(f"era5_live.val_corpus_dir={args.val_corpus_dir}")
    if args.grid_size_min is not None:
        overrides.append(f"era5_live.grid_size_min={args.grid_size_min}")
    if args.grid_size_max is not None:
        overrides.append(f"era5_live.grid_size_max={args.grid_size_max}")
    if args.box_deg_min is not None:
        overrides.append(f"era5_live.box_deg_min={args.box_deg_min}")
    if args.box_deg_max is not None:
        overrides.append(f"era5_live.box_deg_max={args.box_deg_max}")

    extra = [o for o in args.overrides if o != "--"]
    overrides.extend(extra)

    cmd = [sys.executable, "-m", "copula_inter.train", *overrides]
    print("[finetune_era5] " + " ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
