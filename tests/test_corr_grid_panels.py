"""val/corr_grid_analytic_z's renderer must give every panel the same size.

Regression test for a reported "the oracle and the predicted matrix aren't the
same size" in the logged figure. The matrices always ARE the same size (see
train.py::_corr_grid_fig, which slices Sigma to the episode's own n and pairs it
with gp_analytical_posterior's n x n R_post from that same episode) -- what
differed was the rendering: the colorbar used to be drawn inside the LAST
heatmap call, taking its width out of that one axes, and square=True then shrank
its height to match. Measured 3.56in vs 2.92in for two identical 64x64 inputs.
"""

import matplotlib

matplotlib.use("Agg")

import torch

from eval.viz.correlation_plots import plot_corr_grid


def _panel_sizes(fig):
    """Width/height in inches of every heatmap axes (the colorbar axes has no
    title, which is what distinguishes it here)."""
    w_in, h_in = fig.get_size_inches()
    return [
        (ax.get_position().width * w_in, ax.get_position().height * h_in)
        for ax in fig.axes
        if ax.get_title()
    ]


def test_every_panel_is_the_same_size():
    n = 64
    for n_estimators in (1, 2, 3):
        estimators = {f"pred_{i}": torch.eye(n) for i in range(n_estimators)}
        fig = plot_corr_grid(estimators, torch.eye(n), title="t")
        fig.canvas.draw()
        sizes = _panel_sizes(fig)
        assert len(sizes) == n_estimators + 1, sizes
        w0, h0 = sizes[0]
        for w, h in sizes[1:]:
            assert abs(w - w0) < 1e-6, sizes
            assert abs(h - h0) < 1e-6, sizes
        matplotlib.pyplot.close(fig)


def test_colorbar_is_still_drawn():
    fig = plot_corr_grid({"pred": torch.eye(8)}, torch.eye(8))
    fig.canvas.draw()
    # 2 heatmaps + 1 colorbar
    assert len(fig.axes) == 3, [ax.get_title() for ax in fig.axes]
    matplotlib.pyplot.close(fig)


def test_subsamples_large_episodes_consistently():
    """Both panels must be subsampled to the same max_show grid."""
    n = 128
    fig = plot_corr_grid({"pred": torch.eye(n)}, torch.eye(n), max_show=40)
    fig.canvas.draw()
    meshes = [ax.collections[0] for ax in fig.axes if ax.get_title()]
    shapes = {tuple(m.get_array().shape) for m in meshes}
    assert len(shapes) == 1, shapes
    matplotlib.pyplot.close(fig)
