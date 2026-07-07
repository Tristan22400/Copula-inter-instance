"""
Visualize the collapse of the GP posterior correlation matrix R_star toward
the identity as the number of context (training) points P grows.

Uses the actual data-generation code in src/data_gen.py (rbf_kernel,
gp_posterior, sigma_to_correlation) rather than a reimplementation, so this
reflects exactly what generate_gp_task/generate_gp_batch compute and save as
R_star. A fixed set of test points and a growing, nested pool of context
points (first P of one fixed random draw, so increasing P only *adds*
points) isolates the one effect this script demonstrates: more/denser
training context explains away more of the prior correlation among test
points, so off-diagonal entries of R_star shrink toward 0 and R_star -> I.

Usage:
    python plots/posterior_covariance_collapse.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PLOTS_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_gen import _safe_cholesky, gp_posterior, rbf_kernel, sigma_to_correlation  # noqa: E402

SEED = 42
N_TEST = 30           # fixed test points; R_star is (N_TEST, N_TEST)
CONTEXT_SIZES = [2, 8, 32, 256]  # increasing number of context points P
L_LENGTHSCALE = 1.0   # fixed RBF hyperparameters (mid-range of conf/data/gp_tasks.yaml's l_min/l_max)
ALPHA2 = 1.0
NUGGET = 0.15         # mid-range of gp_tasks.yaml's nugget_min/nugget_max


def kernel_fn(x1, x2):
    return rbf_kernel(x1, x2, l=L_LENGTHSCALE, alpha2=ALPHA2)


def build_points(seed=SEED):
    """Fixed test points, and a fixed nested pool of context points.

    Both are drawn once (1D inputs, N(0, 1)) so growing P only appends new
    context points instead of resampling the whole context set — otherwise
    the collapse would be confounded with the context set itself changing.
    """
    g = torch.Generator().manual_seed(seed)
    x_test = torch.randn(N_TEST, 1, generator=g)
    x_test, _ = x_test.sort(dim=0)  # sort for a visually banded matrix

    max_p = max(CONTEXT_SIZES)
    x_train_pool = torch.randn(max_p, 1, generator=g)

    # One real joint GP draw over the full pool + test points, so y_train is
    # an actual sample consistent with kernel_fn/NUGGET (needed by
    # gp_posterior to compute alpha, even though Sigma_star doesn't depend on
    # y_train at all).
    x_all = torch.cat([x_train_pool, x_test], dim=0)
    K_all = kernel_fn(x_all, x_all) + NUGGET * torch.eye(x_all.shape[0])
    y_all = _safe_cholesky(K_all) @ torch.randn(x_all.shape[0], generator=g)
    y_train_pool = y_all[:max_p]

    return x_test, x_train_pool, y_train_pool


def collapse_stats(R_star: torch.Tensor, sigma_star: torch.Tensor) -> dict:
    """Summary statistics quantifying how close R_star is to the identity."""
    n = R_star.shape[0]
    off_diag_mask = ~torch.eye(n, dtype=torch.bool)
    off_diag = R_star[off_diag_mask]
    identity_dist = torch.linalg.matrix_norm(R_star - torch.eye(n), ord="fro")
    return {
        "mean |off-diag|": off_diag.abs().mean().item(),
        "max |off-diag|": off_diag.abs().max().item(),
        "||R* - I||_F": identity_dist.item(),
        "mean posterior std": sigma_star.mean().item(),
    }


def main():
    x_test, x_train_pool, y_train_pool = build_points()

    fig, axes = plt.subplots(1, len(CONTEXT_SIZES), figsize=(4.2 * len(CONTEXT_SIZES), 5.4))
    im = None
    for ax, P in zip(axes, CONTEXT_SIZES):
        x_train = x_train_pool[:P]
        y_train = y_train_pool[:P]

        _, Sigma_star = gp_posterior(x_train, y_train, x_test, kernel_fn, NUGGET)
        R_star, sigma_star = sigma_to_correlation(Sigma_star)

        stats = collapse_stats(R_star, sigma_star)
        print(f"P={P:>4d} context points: " + ", ".join(f"{k}={v:.4f}" for k, v in stats.items()))

        im = ax.imshow(R_star.numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"P = {P} context points")
        ax.set_xlabel("test index $j$")
        ax.set_xticks([])
        ax.set_yticks([])

        stats_text = "\n".join(f"{k}: {v:.3f}" for k, v in stats.items())
        ax.text(
            0.5, -0.14, stats_text,
            transform=ax.transAxes, ha="center", va="top", fontsize=8, family="monospace",
        )
    axes[0].set_ylabel("test index $i$")
    axes[0].set_yticks([])

    fig.suptitle(
        "Posterior correlation matrix $R^*$ collapses to the identity as context grows\n"
        "(RBF kernel, l=%.1f, alpha2=%.1f, nugget=%.2f)" % (L_LENGTHSCALE, ALPHA2, NUGGET)
    )
    plt.subplots_adjust(bottom=0.28, top=0.82, right=0.9)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.85, pad=0.02, fraction=0.03, label="Correlation")

    out_path = os.path.join(PLOTS_DIR, "posterior_covariance_collapse.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
