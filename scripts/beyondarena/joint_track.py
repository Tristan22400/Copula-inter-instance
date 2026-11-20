"""BeyondArena joint track inside the framework (beyondarena-joint venv; scripts/beyondarena/joint_track.sh).

Usage:
    python joint_track.py marginals='[tabicl-ft]' "datasets=[<name>,...]"      # one shard
    python joint_track.py lmm=true subset='[core2k,regression,grouped]'

Each marginal (TabICLv2 / TabPFN-3.5 Fast / Kumo small, pretrained or ERA5-fine-tuned) runs as one
TabArena method, eval.beyondarena.joint_model.CopulaJointModel with its own copula, under the official
BeyondArena protocol: core2k regression splits, the arena's validation protocol and bagging
(refit_folds, as TabArena's foundation models), TabArena preprocessing. The runner
(eval.beyondarena.joint_runner.JointExperimentRunner) adds the joint scores of the refit model on the
whole test fold for the variants indep (the marginal alone, i.e. without the copula), exch and copula,
which share the fit. lmm=true runs the random-intercept mixed model instead (grouped tasks).
Results: <out_root>/<run>/data/<method>/<dataset>/<split>/results.pkl; a rerun resumes. A failed split is
skipped (raise_on_failure=false, TabArena's default for benchmarking) and the job goes on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
C = REPO / "checkpoints/copula_nano"
M = REPO / "checkpoints/marginal/ablations"
MARGINALS = {
    "tabicl-pre": {
        "marginal": "tabicl",
        "tabicl_ckpt": "pretrained",
        "copula_ckpt": str(C / "pretrained-marginal-tabicl/step_0100000.pt"),
    },
    "tabicl-ft": {
        "marginal": "tabicl",
        "tabicl_ckpt": str(M / "marginal_finetune_era5_run1/step_0177600_final.pt"),
        "copula_ckpt": str(C / "size-robust-tabicl/step_0100000.pt"),
    },
    "tabpfn-pre": {
        "marginal": "tabpfn35_fast",
        "max_features": 500,
        "copula_ckpt": str(C / "pretrained-marginal-tabpfn35_fast/step_0100000.pt"),
    },
    "tabpfn-ft": {
        "marginal": "tabpfn35_fast",
        "max_features": 500,
        "marginal_ckpt": str(M / "marginal_finetune_era5_tabpfn35_fast/step_0027500_final.pt"),
        "copula_ckpt": str(C / "size-robust-tabpfn35_fast/step_0100000.pt"),
    },
    "kumo-pre": {"marginal": "kumo_small", "copula_ckpt": str(C / "pretrained-marginal-kumo_small/step_0100000.pt")},
    "kumo-ft": {
        "marginal": "kumo_small",
        "marginal_ckpt": str(M / "marginal_finetune_era5_kumo_small/step_0114500_final.pt"),
        "copula_ckpt": str(C / "size-robust-kumo_small/step_0100000.pt"),
    },
}
DEFAULTS = {
    "marginals": list(MARGINALS),
    "lmm": False,
    "subset": ["core2k", "regression"],
    "datasets": [],
    "splits": None,
    "run": "main",
    # False (TabArena's benchmarking behaviour): a failed split (time limit, OOM) is skipped, not cached, and
    # TabArena imputes it at compare time; the job continues with the next split. True stops the job.
    "raise_on_failure": False,
    "out_root": str(REPO / "results/beyondarena/joint_ta"),
}


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULTS), OmegaConf.from_cli())
    sys.path.insert(0, str(REPO))
    from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
    from tabarena.contexts import BeyondArenaContext
    from tabarena.utils.config_utils import ConfigGenerator

    from eval.beyondarena.joint_model import MODEL_CLASSES, MixedLMJointModel
    from eval.beyondarena.joint_runner import JointExperimentRunner

    if cfg.lmm:
        names = ["lmm"]
        models = [(ConfigGenerator(model_cls=MixedLMJointModel, search_space={}, manual_configs=[{}]), 0)]
    else:
        names = list(cfg.marginals)
        models = [
            (
                ConfigGenerator(
                    model_cls=MODEL_CLASSES[n], search_space={}, manual_configs=[MARGINALS[n] | {"tag": n}]
                ),
                0,
            )
            for n in names
        ]
    experiments = BeyondArenaExperimentBundle(models=models).build_experiments()
    for e in experiments:
        e.experiment_cls = JointExperimentRunner
    out_dir = Path(cfg.out_root) / str(cfg.run)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Method name -> marginal tag (one model class, hence one TabArena method name, per marginal).
    tags = {e.name: names[i] for i, e in enumerate(experiments)} if len(experiments) == len(names) else {}
    if len(set(tags)) != len(names):
        raise RuntimeError(f"method names are not unique per marginal: {tags}")
    (out_dir / f"configs_{'lmm' if cfg.lmm else '-'.join(names)}.json").write_text(
        json.dumps({"tags": tags, "cfg": OmegaConf.to_container(cfg)}, indent=1)
    )
    build_kwargs: dict = {}
    if cfg.datasets:
        build_kwargs["dataset_names"] = list(cfg.datasets)
    if cfg.splits:
        build_kwargs["split_indices"] = cfg.splits
    BeyondArenaContext().build_and_run_jobs(
        experiments,
        expname=str(out_dir),
        subset=list(cfg.subset),
        build_kwargs=build_kwargs,
        new_result_prefix="[New] ",
        debug_mode=True,
        raise_on_failure=bool(cfg.raise_on_failure),
        cache_mode="default",
    )


if __name__ == "__main__":
    main()
