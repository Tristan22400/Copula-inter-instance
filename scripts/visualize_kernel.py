"""Headless visualization of a kernel's R_star (the prior test correlation) across episodes.

Prints a one-line summary per episode for N_SAMPLES generate_gp_batch draws and
saves a grid of N_PLOT clustered matrices (raw and sorted). Uses the Agg
backend.

Usage:
    python scripts/visualize_kernel.py --kernel rbf
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless: must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from omegaconf import OmegaConf

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from copula_inter.data_gen import KERNEL_REGISTRY, generate_gp_batch  # noqa: E402

N_SAMPLES = 8  # print at least 8 generated posterior draws along the way
N_PLOT = 4     # number of those draws to actually plot (raw + sorted each)


def _load_cfg(kernel_name: str):
    """The project config (same YAML files as training) fixed to one kernel."""
    base_cfg = OmegaConf.load(os.path.join(_ROOT, "conf", "config.yaml"))
    data_cfg = OmegaConf.load(os.path.join(_ROOT, "conf", "data", "gp_tasks.yaml"))
    OmegaConf.set_struct(base_cfg, False)
    cfg = OmegaConf.merge(base_cfg, OmegaConf.create({"data": data_cfg}))
    cfg.data.kernel = kernel_name
    cfg.data.kernels = []
    return cfg


def visualize(kernel_name: str):
    print(f"[*] Generating posterior R* visualization for {kernel_name}...")
    if kernel_name not in KERNEL_REGISTRY:
        print(f"[!] Kernel {kernel_name} not found in KERNEL_REGISTRY. "
              f"Available: {sorted(KERNEL_REGISTRY)}")
        sys.exit(1)

    cfg = _load_cfg(kernel_name)
    episodes = generate_gp_batch(cfg, N_SAMPLES, device="cpu")

    all_R = []
    for i, ep in enumerate(episodes):
        R_i = ep["R_star"].numpy()
        mask = ~np.eye(R_i.shape[0], dtype=bool)
        print(
            f"    [{i + 1}/{N_SAMPLES}] posterior R* draw: shape={R_i.shape}  "
            f"range=[{R_i.min():+.3f}, {R_i.max():+.3f}]  "
            f"mean|off-diag|={np.abs(R_i[mask]).mean():.3f}"
        )
        all_R.append(R_i)

    # Plot several draws, raw and sorted.
    n_plot = min(N_PLOT, len(all_R))
    fig, axes = plt.subplots(2, n_plot, figsize=(5 * n_plot, 10), squeeze=False)

    for col in range(n_plot):
        R = all_R[col]

        # Hierarchical clustering, per draw (block structure differs per episode)
        distance_matrix = 1.0 - R
        distance_matrix = np.clip(0.5 * (distance_matrix + distance_matrix.T), 0, 1)
        np.fill_diagonal(distance_matrix, 0.0)

        linkage = sch.linkage(sch.distance.squareform(distance_matrix), method='average')
        dendro = sch.dendrogram(linkage, no_plot=True)
        idx = dendro['leaves']
        R_sorted = R[idx, :][:, idx]

        im1 = axes[0][col].imshow(R, cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
        axes[0][col].set_title(f"Posterior R* ({kernel_name}) — draw {col + 1}")
        plt.colorbar(im1, ax=axes[0][col])

        im2 = axes[1][col].imshow(R_sorted, cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
        axes[1][col].set_title(f"Sorted R* — draw {col + 1}")
        plt.colorbar(im2, ax=axes[1][col])

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
