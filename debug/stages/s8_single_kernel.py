"""s8_single_kernel.py — force the model to a single kernel family.

If S1 (rank ceiling) showed capacity is the binding constraint, a
single-family run is expected to plateau too and this stage mostly serves
as a negative control; it's most informative once S1/S4 have ruled rank
out, or to check whether one specific family (e.g. periodic, known
unrecoverable per project_checkpoint_family_spatial_comparison) is
dragging down the mixed-kernel average.

Thin wrapper over scripts/train_fast.py: forces
data.systematic_composition=false data.kernel=<KERNEL>
training.adaptive_kernel_sampling=false (that flag silently goes inert on
the fixed-kernel branch anyway, see data_gen.py::_resolve_kernel_name --
disabling it here just makes the run's intent explicit in its config/logs)
and forwards everything else verbatim, so this is not a separate
implementation of the debug loop, just a documented invocation of the
existing one.

Usage:
    python debug/run_debug.py s8 --kernel rbf
    python debug/stages/s8_single_kernel.py --kernel matern52 -- training.steps=2000 data.P_min=32 data.P_max=32
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

AVAILABLE_KERNELS = [
    "rbf", "matern12", "matern32", "matern52", "cosine", "periodic",
    "rational_quadratic", "dot_product", "polynomial",
]


def build_overrides(kernel: str, extra: "list[str]") -> "list[str]":
    if kernel not in AVAILABLE_KERNELS:
        raise ValueError(f"Unknown kernel {kernel!r}, choose from {AVAILABLE_KERNELS}")
    return [
        "data.systematic_composition=false",
        f"data.kernel={kernel}",
        "training.adaptive_kernel_sampling=false",
        *extra,
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kernel", default="rbf", choices=AVAILABLE_KERNELS)
    p.add_argument("extra", nargs=argparse.REMAINDER, help="Extra Hydra overrides forwarded to train_fast.py verbatim (put a lone -- before them if any start with --)")
    args = p.parse_args()

    extra = [a for a in args.extra if a != "--"]
    overrides = build_overrides(args.kernel, extra)
    cmd = [sys.executable, os.path.join(_REPO_ROOT, "scripts", "train_fast.py"), *overrides]
    print(f"[s8_single_kernel] {' '.join(cmd)}")
    raise SystemExit(subprocess.call(cmd, cwd=_REPO_ROOT))


if __name__ == "__main__":
    main()
