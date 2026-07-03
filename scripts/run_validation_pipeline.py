import subprocess
import argparse
import sys
import os

def run_command(command: list, description: str, fail_fast: bool = True):
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {description}")
    print(f"💻 Command: {' '.join(command)}")
    print(f"{'='*60}\n")

    try:
        # Stream output directly to the console
        result = subprocess.run(command, check=True)
        print(f"\n✅ SUCCESS: {description}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {description} (Exit Code: {e.returncode})\n")
        if fail_fast:
            print("🛑 Pipeline aborted due to failure.")
            sys.exit(1)
        return False

def main():
    parser = argparse.ArgumentParser(description="Automated Kernel Validation Pipeline")
    parser.add_argument("--kernel", type=str, default="lsh_forest", help="Kernel to validate (e.g., lsh_forest)")
    parser.add_argument("--skip-overfit", action="store_true", help="Skip the overfitting step (which can be slow)")
    args = parser.parse_args()

    kernel = args.kernel
    print(f"🧪 Starting Validation Pipeline for Kernel: {kernel}\n")

    # Step 2 & 4: Mathematical Stability & Goldilocks Bound (Pytest)
    # test_kernel_goldilocks_and_psd is parametrized over data_gen.ALL_KERNELS
    # (tests/test_data.py), so the node id carries the kernel name in brackets
    # rather than in the function name.
    test_name = f"tests/test_data.py::test_kernel_goldilocks_and_psd[{kernel}]"
    run_command(
        ["pytest", test_name, "-v"],
        description="Steps 2 & 4: Math Stability (PSD) & Goldilocks Correlation Bounds"
    )

    # Step 3: Structural Diversity Visualization
    run_command(
        ["python", "scripts/visualize_kernel.py", "--kernel", kernel],
        description="Step 3: Structural Diversity Visualization (Headless)"
    )

    # Step 5: Model Capacity Check (Overfit Single Batch)
    if not args.skip_overfit:
        run_command(
            ["python", "src/overfit_single.py", "--kernel", kernel],
            description="Step 5: Model Capacity Check (Single-Batch Overfit)"
        )
    else:
        print("⏭️ Skipping Step 5 (Overfit Check) as requested.")

    print(f"\n🎉 ALL PIPELINE STEPS COMPLETED SUCCESSFULLY FOR '{kernel}'!")

if __name__ == "__main__":
    main()
