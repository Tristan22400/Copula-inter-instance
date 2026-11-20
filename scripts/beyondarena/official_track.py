"""BeyondArena official track: run a TabICLv2 marginal under the official protocol (tabarena venv).

Usage (from scripts/beyondarena/official_track.sh, which activates the tabarena venv):
    python official_track.py method=ft datasets='[airfoil_self_noise-6278f0ada559]'
    python official_track.py method=repro splits=lite

Overrides are key=value (OmegaConf.from_cli; this venv has no Hydra):
    method    ft    -> TA-TabICLv2-ERA5FT (TabArena's TA-TabICLv2 with model_path=ckpt)
              repro -> TA-TabICLv2 default config rerun here, to check that this hardware and
                       environment reproduce the cached baseline (a control, not a result)
    ckpt      the fine-tuned regression checkpoint ({"config", "state_dict"}, tabicl 2.2.0 schema)
    subset    BeyondArena subset predicates (default: core2k regression, the official protocol)
    datasets  dataset names to restrict to (empty = every dataset of the subset); one OAR job per shard
    splits    null = the subset's splits; "lite" = the first split of each dataset
    out_root  experiment directory root; results land in <out_root>/<method>

A rerun skips every finished (dataset, split) job (cache_mode=default).
"""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULTS = {
    "method": "ft",
    "ckpt": str(REPO / "checkpoints/marginal/ablations/marginal_finetune_era5_run1/step_0177600_final.pt"),
    "subset": ["core2k", "regression"],
    "datasets": [],
    "splits": None,
    "out_root": str(REPO / "results/beyondarena/official"),
}


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULTS), OmegaConf.from_cli())
    sys.path.insert(0, str(HERE))  # ta_models must be importable by name (pickled with the results)

    from ta_models import TabICLv2FTModel
    from tabarena.benchmark.experiment import BeyondArenaExperimentBundle
    from tabarena.contexts import BeyondArenaContext
    from tabarena.models.tabicl.model import TabICLv2Model
    from tabarena.utils.config_utils import ConfigGenerator

    if cfg.method == "ft":
        if not Path(cfg.ckpt).is_file():
            raise FileNotFoundError(cfg.ckpt)
        gen = ConfigGenerator(model_cls=TabICLv2FTModel, search_space={}, manual_configs=[{"model_path": cfg.ckpt}])
    elif cfg.method == "repro":
        gen = ConfigGenerator(model_cls=TabICLv2Model, search_space={}, manual_configs=[{}])
    else:
        raise ValueError(f"method={cfg.method!r}: expected ft or repro")

    experiments = BeyondArenaExperimentBundle(models=[(gen, 0)]).build_experiments()
    build_kwargs: dict = {}
    if cfg.datasets:
        build_kwargs["dataset_names"] = list(cfg.datasets)
    if cfg.splits:
        build_kwargs["split_indices"] = cfg.splits
    out_dir = Path(cfg.out_root) / str(cfg.method)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    ctx = BeyondArenaContext()
    ctx.build_and_run_jobs(
        experiments,
        expname=str(out_dir),
        subset=list(cfg.subset),
        build_kwargs=build_kwargs,
        new_result_prefix="[New] ",
        debug_mode=True,  # in-process, one GPU, sequential jobs
        cache_mode="default",  # resume: finished jobs are skipped
    )


if __name__ == "__main__":
    main()
