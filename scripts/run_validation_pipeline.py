import re
import subprocess
import argparse
import shutil
import sys
import os
import tempfile

# Kernels whose prior can be negative (oscillatory) get the wider-tolerance
# dataset test (test_dataset_corr_uniform.py); every other kernel has a
# non-negative prior and gets test_dataset_corr_nonneg.py. See the docstrings
# of those two files for the statistical reasoning. A composite ("A+B"/"A*B")
# inherits this if either component is oscillatory: rbf*cosine's prior can
# still go negative directly (rbf>0 times a negative cosine), so the tight
# negative-fraction bound in test_dataset_corr_nonneg.py doesn't apply.
OSCILLATORY_KERNELS = {"cosine"}


def _is_oscillatory(kernel: str) -> bool:
    return any(part in OSCILLATORY_KERNELS for part in re.split(r"[+*]", kernel))


failures = []


def run_command(command: list, description: str, env: dict | None = None):
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {description}")
    print(f"💻 Command: {' '.join(command)}")
    print(f"{'='*60}\n")

    run_env = {**os.environ, **env} if env else None
    try:
        # Stream output directly to the console
        result = subprocess.run(command, check=True, env=run_env)
        print(f"\n✅ SUCCESS: {description}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {description} (Exit Code: {e.returncode})\n")
        failures.append((description, e.returncode))
        return False

def main():
    parser = argparse.ArgumentParser(description="Automated Kernel Validation Pipeline")
    parser.add_argument("--kernel", type=str, default="rbf", help="Kernel to validate (e.g., rbf)")
    parser.add_argument("--skip-dataset-validation", action="store_true",
                         help="Skip generating a dataset and checking its R_star correlation structure")
    parser.add_argument("--n-episodes", type=int, default=500,
                         help="Episodes to generate for the dataset correlation check (default: 500)")
    parser.add_argument("--keep-dataset", action="store_true",
                         help="Keep the generated validation dataset instead of deleting it afterwards")
    args, extra_overrides = parser.parse_known_args()
    if extra_overrides:
        print(f"📎 Extra Hydra overrides for Step 2a (dataset generation): {extra_overrides}")

    kernel = args.kernel
    print(f"🧪 Starting Validation Pipeline for Kernel: {kernel}\n")

    # Step 1: Mathematical Stability & Goldilocks Bound (Pytest)
    # test_kernel_goldilocks_and_psd is parametrized over data_gen.ALL_KERNELS
    # (tests/test_data.py), so the node id carries the kernel name in brackets
    # rather than in the function name.
    test_name = f"tests/test_data.py::test_kernel_goldilocks_and_psd[{kernel}]"
    run_command(
        ["pytest", test_name, "-v"],
        description="Step 1: Math Stability (PSD) & Goldilocks Correlation Bounds"
    )

    # Step 2: Generate a small dataset and verify the R_star correlation
    # structure across real episodes (dataset-level checks that a single
    # episode can't catch: sign balance, saturation, conditioning).
    if not args.skip_dataset_validation:
        dataset_dir = tempfile.mkdtemp(prefix=f"validation_{kernel}_")
        pit_dir = os.path.join(dataset_dir, "pit")
        try:
            run_command(
                [
                    "python", "src/generate_pit_dataset.py",
                    f"data.kernel={kernel}",
                    f"data.n_tasks={args.n_episodes}",
                    f"data.dataset_dir={dataset_dir}",
                    *extra_overrides,
                ],
                description=f"Step 2a: Generate {args.n_episodes} episodes for dataset validation"
            )

            dataset_test_file = (
                "tests/test_dataset_corr_uniform.py" if _is_oscillatory(kernel)
                else "tests/test_dataset_corr_nonneg.py"
            )
            run_command(
                ["pytest", dataset_test_file, "-v"],
                description=f"Step 2b: Dataset Correlation Structure ({dataset_test_file})",
                env={"DATASET_DIR": pit_dir},
            )
        finally:
            if args.keep_dataset:
                print(f"📁 Kept validation dataset at: {dataset_dir}")
            else:
                shutil.rmtree(dataset_dir, ignore_errors=True)
    else:
        print("⏭️ Skipping Step 2 (Dataset Correlation Validation) as requested.")

    # Step 3: Structural Diversity Visualization
    run_command(
        ["python", "scripts/visualize_kernel.py", "--kernel", kernel],
        description="Step 3: Structural Diversity Visualization (Headless)"
    )

    if failures:
        print(f"\n⚠️  PIPELINE FINISHED FOR '{kernel}' WITH {len(failures)} FAILURE(S):")
        for description, returncode in failures:
            print(f"   ❌ {description} (Exit Code: {returncode})")
        sys.exit(1)
    else:
        print(f"\n🎉 ALL PIPELINE STEPS COMPLETED SUCCESSFULLY FOR '{kernel}'!")

if __name__ == "__main__":
    main()
