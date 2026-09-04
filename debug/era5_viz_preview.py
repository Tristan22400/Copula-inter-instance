"""era5_viz_preview.py — render validate()'s val/era5_predictions figure for
a checkpoint WITHOUT launching training, and print the Moran's I table behind
it.

Same code path validate()'s do_plot block takes (_build_era5_viz_batch ->
_era5_viz_fig), so what this writes to disk is what lands in wandb -- the
point being that you can iterate on the figure, or eyeball a finished
checkpoint, without a training run in the loop.

The Moran's I table is the numeric version of the figure's central
comparison. Rows 2-4 (exact GP posterior / copula model / independent) are
posterior SAMPLES on the same context and the same latent noise vector; row
1 is a fully-observed realization. So the model row should be read against
the GP row, never against the ground-truth row -- a model that emits the
prior correlation rather than the posterior renders smoother than either and
scores far worse on held-out NLL. Measured on western_europe at 5% context,
the fitted-Matern32 GP posterior sample comes out SMOOTHER than the ground
truth (I ~ 0.86-0.95 vs 0.82-0.92), so a copula sample that is visibly
grainier than the GP row is a real deficit and not an artifact of sampling.

Usage:
    python debug/era5_viz_preview.py checkpoints/<run>/step_XXXXXXX.pt
    python debug/era5_viz_preview.py <ckpt> --n-days 4 --gp-kernel rational_quadratic
    python debug/era5_viz_preview.py <ckpt> --no-gp        # old 3-row figure
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from eval.spatial.diagnostics import morans_i  # noqa: E402
from eval.spatial.sweep_core import get_model  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", help="path to a copula checkpoint (.pt)")
    p.add_argument("--out", default=os.path.join(_REPO_ROOT, "results", "era5_viz_gp_row",
                                                 "era5_predictions_gp_row.png"))
    p.add_argument("--n-days", type=int, default=3)
    p.add_argument("--region", default=None, help="override baselines.era5_viz_region")
    p.add_argument("--grid-size", type=int, default=None)
    p.add_argument("--gp-kernel", default="matern32")
    p.add_argument("--no-gp", action="store_true", help="drop the exact-GP row")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    import train as T

    model, cfg, device, marginal = get_model(args.ckpt, args.device)
    OmegaConf.set_struct(cfg, False)
    cfg.baselines.era5_viz_n_days = args.n_days
    cfg.baselines.era5_viz_gp = not args.no_gp
    cfg.baselines.era5_viz_gp_kernel = args.gp_kernel
    if args.region is not None:
        cfg.baselines.era5_viz_region = args.region
    if args.grid_size is not None:
        cfg.baselines.era5_viz_grid_size = args.grid_size

    vb = T._build_era5_viz_batch(cfg, marginal, device)
    if vb is None:
        raise SystemExit("era5 viz probe unavailable (check baselines.era5_regions)")
    print(f"region={vb['region']} D={vb['D']} n_context={vb['n_context']} "
          f"({100.0 * vb['n_context'] / vb['D']:.1f}% of grid) days={vb['days']}")

    jitter = float(cfg.model.get("sigma_jitter", 1e-4))
    with torch.no_grad():
        fig = T._era5_viz_fig(model, cfg, vb, jitter, device)
    if fig is None:
        raise SystemExit("figure unavailable (degenerate probe grid)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")

    # Moran's I per row, recomputed on the same shared-z draws the figure used.
    gp_post = vb.get("gp_post_per_day") or [None] * len(vb["days"])
    shape = vb["grid_shape"]
    rng = np.random.default_rng(vb["seed"])
    R_indep = np.eye(vb["D"])
    x_tr = torch.as_tensor(vb["x_train_norm"], dtype=torch.float32, device=device).unsqueeze(0)
    x_te = torch.as_tensor(vb["x_test_norm"], dtype=torch.float32, device=device).unsqueeze(0)
    print(f"\nMoran's I (spatial autocorrelation; rows 2-4 are posterior samples)")
    print(f"{'day':>6} {'truth':>9} {'GP post':>9} {'model':>9} {'indep':>9}")
    with torch.no_grad():
        for i, d in enumerate(vb["days"]):
            z_tr = torch.as_tensor(vb["z_train_per_day"][i], dtype=torch.float32,
                                   device=device).unsqueeze(0)
            out_v = model({"x_train": x_tr, "z_train": z_tr, "x_test": x_te})
            Sigma = T.build_sigma(out_v, cfg, jitter=jitter)[0].float().cpu().numpy()
            z = rng.standard_normal(vb["D"])
            di, ym, ys = vb["dists_per_day"][i], vb["y_mean_per_day"][i], vb["y_std_per_day"][i]
            m_gp = (morans_i(T._era5_viz_gp_field(gp_post[i], z).reshape(shape))
                    if gp_post[i] is not None else float("nan"))
            print(f"{d:>6} {morans_i(vb['true_fields'][i]):>9.3f} {m_gp:>9.3f} "
                  f"{morans_i(T._era5_viz_field(Sigma, di, ym, ys, z, device).reshape(shape)):>9.3f} "
                  f"{morans_i(T._era5_viz_field(R_indep, di, ym, ys, z, device).reshape(shape)):>9.3f}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
