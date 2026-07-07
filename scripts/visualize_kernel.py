"""
visualize_kernel.py — Step 3: Structural Diversity Visualization (headless).

Draws N_SAMPLES independent episodes for a given kernel via generate_gp_batch
(the same code path generate_pit_dataset.py uses), printing a one-line summary
of each episode's R_star — the GP *posterior* correlation matrix at the test
points, i.e. what the model is actually trained to predict — as a quick
multi-draw sanity check. It then hierarchically clusters N_PLOT of those
draws (raw + sorted) and saves a grid plot to disk — one draw's matrix can
look degenerate by chance, so seeing several side by side is what actually
shows whether the kernel has healthy structural diversity.

Deliberately does NOT plot the raw prior kernel K(X,X): the prior ignores
conditioning on training data (K_st K_ff^-1 K_ts), so it can look structured
even when the posterior the model must learn has been shrunk toward
independence — checking the prior alone would validate the wrong quantity.

Never opens a GUI window — safe to run over SSH on a host with no DISPLAY
(matplotlib's Agg backend is forced before pyplot is imported).

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
sys.path.append(os.path.join(_ROOT, 'src'))
from data_gen import generate_gp_batch, KERNEL_REGISTRY  # noqa: E402

N_SAMPLES = 8  # print at least 8 generated posterior draws along the way
N_PLOT = 4     # number of those draws to actually plot (raw + sorted each)


def _load_cfg(kernel_name: str):
    """Build the real project config (same yaml files as training/generation), fixed to one kernel."""
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

    # Plot N_PLOT separate draws (raw + sorted each) so one lucky/unlucky
    # draw doesn't stand in for the whole kernel's behaviour.
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
