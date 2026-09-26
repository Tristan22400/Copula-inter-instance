"""Debug stage S4: overfit the model on K realizations of one episode's correlation target.

Target: R_star (--target prior) or the exact posterior R_post (--target
posterior, needs --kernel). Realizations: z_test ~ N(0, R) (--z-source oracle)
or K posterior y_test draws PIT'd through TabICL (--z-source tabicl, via
s3_pit_floor.sample_and_pit). Reports ||R_hat - R||_F and the copula NLL
against the full-rank oracle and the rank-r ceiling (s1_rank_ceiling).

Usage:
    python debug/stages/s4_overfit.py --episode data/pit_episodes/shard_000000.pt [--task-idx 3]
    python debug/stages/s4_overfit.py --episode <shard> --k-realizations 500 --steps 5000 --lr 1e-3
    python debug/stages/s4_overfit.py --episode <shard> --freeze-backbone
    python debug/stages/s4_overfit.py --kernel rbf [--target posterior] [--z-source tabicl]
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import OmegaConf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_ROOT = _REPO_ROOT  # kept for the --model/config path joins below

from copula_inter.data_gen import generate_gp_batch
from copula_inter.dataset import collate_fn
from copula_inter.loss import _safe_cholesky, oracle_copula_nll, y_space_nll
from copula_inter.model import build_copula_transformer, build_sigma
from debug import common  # noqa: E402 -- debug/common.py, added to sys.path above
from debug.stages.s1_rank_ceiling import fit_rank_ceiling  # noqa: E402
from debug.stages.s3_pit_floor import sample_and_pit  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--episode",
        default=None,
        help="Path to .pt episode file: either a single-episode task_*.pt, or a "
        "sharded shard_*.pt (a list of episodes — one is picked via --task-idx). "
        "Required unless --kernel is given.",
    )
    p.add_argument(
        "--task-idx",
        type=int,
        default=0,
        help="Which episode to select when --episode points at a shard_*.pt file "
        "(list of episodes). Ignored for single-episode files. Default: 0.",
    )
    p.add_argument(
        "--kernel",
        default=None,
        help="Generate a single fresh episode for this kernel on the fly "
        "(conf/data/gp_tasks.yaml hyperparameter ranges) instead of loading "
        "--episode from disk. Takes precedence over --episode.",
    )
    p.add_argument(
        "--k-realizations",
        type=int,
        default=200,
        help="Number of synthetic z_test samples drawn from N(0, R*) (default: 200)",
    )
    p.add_argument("--steps", type=int, default=3000, help="Gradient steps")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--plot", default="overfit_correlation.png", help="Output plot path")
    p.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze TabICL backbone (faster but may converge slower)",
    )
    p.add_argument(
        "--model",
        default="copula_prod",
        help="conf/model/<name>.yaml preset to load (default: copula_prod)",
    )
    p.add_argument(
        "--parametrization",
        default=None,
        help="Override cfg.model.correlation_parametrization (default: whatever "
        "the --model preset sets, normally 'covnorm'). One of covnorm, cossim, "
        "tanhnorm, sparse_covnorm — see src/copula_inter/correlation_factory.py.",
    )
    p.add_argument(
        "--target", choices=["prior", "posterior"], default="prior",
        help="Correlation to overfit to: R* (prior, unconditioned kernel "
        "correlation -- original behaviour) or R_post (exact GP posterior "
        "conditioned on the context, pit.py::gp_analytical_posterior). "
        "--target posterior requires --kernel (fresh episode with kernel "
        "metadata attached), not --episode.",
    )
    p.add_argument(
        "--z-source", choices=["oracle", "tabicl"], default="oracle",
        help="How the K realizations' z_test are drawn. 'oracle': exact "
        "z ~ N(0, R) synthetic draws (original behaviour). 'tabicl': draw "
        "y_test ~ N(mu_post, Sigma_post) and PIT each through a frozen "
        "TabICL (debug.stages.s3_pit_floor.sample_and_pit) -- bakes in the "
        "same PIT distortion S3 measures. Requires --target posterior.",
    )
    return p.parse_args()


def plot_correlation_comparison(
    R_star: torch.Tensor,
    R_hat: torch.Tensor,
    frob: float,
    gap: float,
    out_path: str,
) -> None:
    """Save a 3-panel figure: R*, R̂, and |R̂ - R*|."""
    R_s = R_star.cpu().numpy()
    R_h = R_hat.cpu().detach().numpy()
    diff = abs(R_h - R_s)

    vmax = max(abs(R_s).max(), abs(R_h).max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, mat, title in zip(
        axes,
        [R_s, R_h, diff],
        [r"Oracle $R^*$", r"Predicted $\hat{R}$", r"$|\hat{R} - R^*|$"],
    ):
        if "diff" in title or "|" in title:
            im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=diff.max())
        else:
            im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("test instance")
        ax.set_ylabel("test instance")

    fig.suptitle(
        rf"Overfit sanity check — $\|R^* - \hat{{R}}\|_F = {frob:.4f}$,  copula gap $= {gap:.4f}$",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out_path}")


def build_synthetic_dataset(episode: dict, K: int) -> list[dict]:
    """Generate K synthetic episodes from the same task by sampling z_test ~ N(0, R*)."""
    n_test = int(episode["n_test"].item())
    R_star = episode["R_star"][:n_test, :n_test].float()
    return build_synthetic_dataset_from_R(episode, R_star, K)


def build_synthetic_dataset_from_R(episode: dict, R: torch.Tensor, K: int) -> list[dict]:
    """K synthetic episodes with z_test ~ N(0, R) for a unit-diagonal R."""
    n_test = R.shape[0]
    L = _safe_cholesky(R)
    eps = torch.randn(n_test, K, device=R.device, dtype=L.dtype)  # (n_test, K)
    z_samples = (L @ eps).T.cpu()         # (K, n_test) -- realizations are stored/cloned as CPU episode dicts

    realizations = []
    for k in range(K):
        ep = {key: (val.clone() if torch.is_tensor(val) else val) for key, val in episode.items()}
        ep["z_test"] = z_samples[k]
        # Marginal log_pdf_test is set to 0 — we only care about the copula term.
        ep["log_pdf_test"] = torch.zeros(n_test)
        realizations.append(ep)
    return realizations


def build_tabicl_dataset(episode: dict, post: dict, K: int, tabicl_model, device: str) -> list[dict]:
    """K episodes whose z_test are posterior y_test draws PIT'd through TabICL (s3_pit_floor.sample_and_pit)."""
    n_test = int(episode["n_test"].item())
    z_samples = sample_and_pit(tabicl_model, episode, post, K, device).cpu()  # (n_test, K)

    realizations = []
    for k in range(K):
        ep = {key: (val.clone() if torch.is_tensor(val) else val) for key, val in episode.items()}
        ep["z_test"] = z_samples[:, k]
        ep["log_pdf_test"] = torch.zeros(n_test)
        realizations.append(ep)
    return realizations


def main() -> None:
    args = parse_args()

    # Build config from yaml files without Hydra.
    base_cfg = OmegaConf.load(os.path.join(_ROOT, "conf", "config.yaml"))
    model_cfg = OmegaConf.load(
        os.path.join(_ROOT, "conf", "model", f"{args.model}.yaml")
    )
    data_cfg = OmegaConf.load(os.path.join(_ROOT, "conf", "data", "gp_tasks.yaml"))
    OmegaConf.set_struct(base_cfg, False)
    # The model presets are `# @package _global_`: merge at the top level.
    cfg = OmegaConf.merge(base_cfg, model_cfg, OmegaConf.create({"data": data_cfg}))
    if args.freeze_backbone:
        cfg.model.unfreeze_backbone = False
    if args.parametrization is not None:
        cfg.model.correlation_parametrization = args.parametrization

    if args.z_source == "tabicl" and args.target != "posterior":
        raise SystemExit("--z-source tabicl requires --target posterior.")
    if args.target == "posterior" and args.episode is not None:
        raise SystemExit(
            "--target posterior requires --kernel (a fresh episode carries the "
            "kernel metadata gp_analytical_posterior needs); an on-disk "
            "--episode file is not guaranteed to have it."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Episode: a fresh draw (--kernel) or a saved task/shard file.
    if args.kernel is not None:
        cfg.data.kernel = args.kernel
        cfg.data.kernels = []
        episode = generate_gp_batch(cfg, 1, device, return_kernel_metadata=(args.target == "posterior"))[0]
        print(f"Episode : fresh draw ({args.kernel})")
    elif args.episode is not None:
        loaded = torch.load(args.episode, map_location="cpu", weights_only=True)
        if isinstance(loaded, list):
            # Shard: a list of episode dicts.
            task_idx = min(args.task_idx, len(loaded) - 1)
            episode = loaded[task_idx]
            print(f"Episode : {os.path.basename(args.episode)}  (shard of {len(loaded)}, task_idx={task_idx})")
        else:
            episode = loaded
            print(f"Episode : {os.path.basename(args.episode)}")
    else:
        raise SystemExit("Provide either --kernel (fresh episode) or --episode (load from disk).")

    n_train = int(episode["n_train"].item())
    n_test = int(episode["n_test"].item())
    print(f"  n_train={n_train}  n_test={n_test}  d_x={episode['x_norm_train'].shape[-1]}")
    print(f"  target={args.target}  z-source={args.z_source}  K realizations: {args.k_realizations}")

    # Pick the correlation target and build the K realizations from it.
    post = None
    if args.target == "prior":
        R_target = episode["R_star"][:n_test, :n_test].float()
        realizations = build_synthetic_dataset_from_R(episode, R_target, args.k_realizations)
    else:
        post = common.posterior_oracle(episode)
        if post is None:
            raise SystemExit(
                "gp_analytical_posterior could not score this episode's kernel "
                "(rare unsupported schema -- whole-chain outer sign modulation). "
                "Try a different --kernel or re-run (a fresh draw resamples hyperparameters)."
            )
        R_target = post["R_post"].float()
        if args.z_source == "oracle":
            realizations = build_synthetic_dataset_from_R(episode, R_target, args.k_realizations)
        else:
            tabicl_model = common.load_frozen_tabicl(
                common.DebugConfig(cfg=cfg, device=device)
            )
            realizations = build_tabicl_dataset(episode, post, args.k_realizations, tabicl_model, device)
    R_target = R_target.to(device)

    # Build model.
    model = build_copula_transformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")
    print(f"Backbone frozen : {not bool(cfg.model.get('unfreeze_backbone', False))}")

    jitter = float(cfg.model.get("sigma_jitter", 1e-4))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.0,
    )

    # Full-rank oracle NLL under the actual target R.
    oracle_nll_vals = []
    for i in range(0, args.k_realizations, args.batch_size):
        chunk = realizations[i : i + args.batch_size]
        b = collate_fn(chunk)
        b = {k: v.to(device) for k, v in b.items()}
        with torch.no_grad():
            R_batch = R_target.unsqueeze(0).expand(b["z_test"].shape[0], -1, -1)
            oracle_nll_vals.append(
                oracle_copula_nll(R_batch, b["z_test"].float(), b["test_mask"]).item()
            )
    oracle_nll = sum(oracle_nll_vals) / len(oracle_nll_vals)

    # Rank-r ceiling on the same target.
    rank = int(cfg.model.get("rank", 32))
    ceiling_per_ep, _ = fit_rank_ceiling(
        R_target.unsqueeze(0).float(), min(rank, n_test - 1), jitter=jitter, device=device,
    )
    rank_ceiling_nll = float(ceiling_per_ep.item())

    R_star = R_target  # kept for the unchanged plotting/printing code below

    print(f"\nFull-rank oracle copula NLL ({args.target}) = {oracle_nll:.6f}")
    print(f"Rank-{rank} ceiling on the same target       = {rank_ceiling_nll:.6f}  (S1's exact fit, see debug/stages/s1_rank_ceiling.py)")
    if args.z_source == "tabicl":
        print("(z-source=tabicl: run debug/stages/s3_pit_floor.py on the same episode for the PIT-distorted attainable floor)")
    header = (
        f"{'step':>6}  {'cop_nll':>10}  {'oracle':>10}  {'gap':>10}"
        f"  {'||R-R*||_F':>12}  {'||R-R*||_F/N²':>15}"
    )
    print(header)
    print("-" * len(header))

    model.train()
    step = 0
    while step <= args.steps:
        # Shuffle realizations each pass.
        indices = torch.randperm(len(realizations)).tolist()
        for i in range(0, len(realizations), args.batch_size):
            if step > args.steps:
                break
            chunk = [realizations[j] for j in indices[i : i + args.batch_size]]
            batch = collate_fn(chunk)
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(batch)
            Sigma = build_sigma(out, cfg, jitter=jitter, test_mask=batch["test_mask"])
            parts = y_space_nll(
                Sigma,
                batch["z_test"].float(),
                batch["log_pdf_test"].float(),
                batch["test_mask"],
            )
            loss = parts["copula"]  # ignore marginal (set to 0 in synthetic data)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            if step % args.log_every == 0:
                with torch.no_grad():
                    # Evaluate R̂ on the first batch (fixed input → same R̂ always).
                    eval_batch = collate_fn([realizations[0]])
                    eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                    out_eval = model(eval_batch)
                    Sigma_eval = build_sigma(out_eval, cfg, jitter=jitter, test_mask=eval_batch["test_mask"])
                    R_hat = Sigma_eval[0, :n_test, :n_test]
                    frob = (R_hat - R_star).norm().item()
                    frob_per_n2 = frob / (n_test ** 2)
                print(
                    f"{step:>6}  {parts['copula'].item():>10.4f}  {oracle_nll:>10.4f}"
                    f"  {parts['copula'].item() - oracle_nll:>10.4f}"
                    f"  {frob:>12.4f}  {frob_per_n2:>15.6f}"
                )
            step += 1

    # Final visual: compare R̂ vs R*.
    with torch.no_grad():
        eval_batch = collate_fn([realizations[0]])
        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
        out_final = model(eval_batch)
        Sigma_final = build_sigma(out_final, cfg, jitter=jitter, test_mask=eval_batch["test_mask"])
        R_hat_final = Sigma_final[0, :n_test, :n_test]

    show = min(n_test, 5)
    print(f"\nR* (top-left {show}×{show}):")
    print(R_star[:show, :show].cpu().numpy().round(3))
    print(f"\nR̂  (top-left {show}×{show}):")
    print(R_hat_final[:show, :show].cpu().detach().numpy().round(3))

    final_frob = (R_hat_final - R_star).norm().item()
    final_gap = parts["copula"].item() - oracle_nll
    final_gap_vs_ceiling = parts["copula"].item() - rank_ceiling_nll
    print(f"\nFinal ||R̂ - R||_F              = {final_frob:.4f}")
    print(f"Final copula gap (vs full-rank) = {final_gap:.4f}")
    print(f"Final copula gap (vs rank-{rank})  = {final_gap_vs_ceiling:.4f}  (0 here means rank isn't limiting convergence)")

    plot_correlation_comparison(R_star, R_hat_final, final_frob, final_gap, args.plot)


if __name__ == "__main__":
    main()
