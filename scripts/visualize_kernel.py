"""
visualize_kernel.py — Step 3: Structural Diversity Visualization (headless).

Draws N_SAMPLES independent covariance matrices K(X, X) for a given kernel
(printing a one-line summary for each, as a quick multi-draw sanity check),
then hierarchically clusters the last draw and saves a raw-vs-sorted plot to
disk. Never opens a GUI window — safe to run over SSH on a host with no
DISPLAY (matplotlib's Agg backend is forced before pyplot is imported).

Usage:
    python scripts/visualize_kernel.py --kernel lsh_forest
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless: must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
import torch

# Ensure src is in path to import kernels
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_gen import KERNEL_REGISTRY  # noqa: E402

N, D = 150, 5
N_SAMPLES = 8  # print at least 8 generated covariance matrices along the way
LSH_NUM_TREES, LSH_DEPTH = 20, 4  # match data_gen.py's defaults for lsh_forest


def _extra_kernel_kwargs(kernel_name: str) -> dict:
    """Hyperparameters some kernels require beyond the common (l, alpha2).

    lsh_forest has no default for lsh_W/lsh_b (they must be sampled once per
    draw, same realization for every kernel_fn call within that draw) — see
    lsh_forest_kernel's docstring in src/data_gen.py.
    """
    if kernel_name == "lsh_forest":
        return {
            "lsh_W": torch.randn(D, LSH_NUM_TREES, LSH_DEPTH),
            "lsh_b": torch.randn(LSH_NUM_TREES, LSH_DEPTH),
        }
    return {}


def visualize(kernel_name: str):
    print(f"[*] Generating visualization for {kernel_name}...")
    if kernel_name not in KERNEL_REGISTRY:
        print(f"[!] Kernel {kernel_name} not found in KERNEL_REGISTRY. "
              f"Available: {sorted(KERNEL_REGISTRY)}")
        sys.exit(1)

    kernel_fn = KERNEL_REGISTRY[kernel_name]

    K = None
    for i in range(N_SAMPLES):
        X = torch.randn(N, D)
        K_i = kernel_fn(X, X, alpha2=1.0, l=1.0, **_extra_kernel_kwargs(kernel_name))
        mask = ~torch.eye(N, dtype=torch.bool)
        print(
            f"    [{i + 1}/{N_SAMPLES}] covariance draw: shape={tuple(K_i.shape)}  "
            f"range=[{K_i.min().item():+.3f}, {K_i.max().item():+.3f}]  "
            f"mean|off-diag|={K_i[mask].abs().mean().item():.3f}"
        )
        K = K_i.numpy()  # keep the last draw for the plot below

    # Apply Hierarchical Clustering
    distance_matrix = 1.0 - K
    distance_matrix = np.clip(0.5 * (distance_matrix + distance_matrix.T), 0, 1)
    np.fill_diagonal(distance_matrix, 0.0)

    linkage = sch.linkage(sch.distance.squareform(distance_matrix), method='average')
    dendro = sch.dendrogram(linkage, no_plot=True)
    idx = dendro['leaves']

    # Sort the Kernel matrix
    K_sorted = K[idx, :][:, idx]

    # Plot and Save
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(K, cmap='viridis', interpolation='nearest')
    axes[0].set_title(f"Raw {kernel_name} Covariance")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(K_sorted, cmap='viridis', interpolation='nearest')
    axes[1].set_title(f"Sorted {kernel_name} (Clustered)")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()

    os.makedirs("outputs/visualizations", exist_ok=True)
    save_path = f"outputs/visualizations/{kernel_name}_cov.png"
    plt.savefig(save_path, dpi=150)
    print(f"[*] Saved visualization to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=str, required=True, help="Name of the kernel to visualize")
    args = parser.parse_args()
    visualize(args.kernel)
