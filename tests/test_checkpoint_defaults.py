"""test_checkpoint_defaults.py — pins what eval_checkpoint.py scores when it
is given no --ckpt and no --tabicl_ckpt.

The two defaults are a pair: the nano copula checkpoint was fine-tuned ON the
ERA5-run1 marginal (conf/model/copula_nano.yaml's tabicl.pit_ckpt), so scoring
it against a different marginal is not the configuration it was trained as.
Both are spelled in exactly one place each -- DEFAULT_CHECKPOINT_FAMILY and
DEFAULT_MARGINAL_FAMILY -- and this pins the files those names resolve to, so
a registry edit that moves either one has to move this test with it rather
than silently re-pointing every default run.

Paths are checked by their tail, not by existence: checkpoints/ is gitignored
and absent on any machine that has not trained or fetched them.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.configs.checkpoints import (  # noqa: E402
    CHECKPOINT_FAMILIES,
    DEFAULT_CHECKPOINT_FAMILY,
    DEFAULT_MARGINAL_FAMILY,
    MARGINAL_FAMILIES,
    resolve_checkpoint,
    resolve_marginal_checkpoint,
)


def test_default_copula_checkpoint_is_the_nano_marginal_finetune():
    assert DEFAULT_CHECKPOINT_FAMILY in CHECKPOINT_FAMILIES
    resolved = resolve_checkpoint(DEFAULT_CHECKPOINT_FAMILY)
    assert resolved.endswith(
        os.path.join(
            "checkpoints", "copula_nano", "copula-finetune-marginal-float32",
            "step_0630000.pt",
        )
    )


def test_default_marginal_is_the_era5_run1_finetune():
    assert DEFAULT_MARGINAL_FAMILY in MARGINAL_FAMILIES
    resolved = resolve_marginal_checkpoint(DEFAULT_MARGINAL_FAMILY)
    assert resolved.endswith(
        os.path.join(
            "checkpoints", "marginal", "ablations",
            "marginal_finetune_era5_run1", "step_0177600_final.pt",
        )
    )


def test_a_repo_root_relative_path_resolves_from_any_cwd(tmp_path, monkeypatch):
    """The configs spell their checkpoints "./checkpoints/...". Resolved from
    a different cwd that spelling used to fall through to the HF hub and fail
    with a 404 instead of a path error -- so a worktree or an OAR script that
    cd's elsewhere silently lost the configured marginal."""
    rel = os.path.join(
        "checkpoints", "marginal", "ablations", "marginal_finetune_era5_run1",
        "step_0177600_final.pt",
    )
    target = os.path.join(_ROOT, rel)
    if not os.path.exists(target):
        # Nothing to resolve against on a machine without the checkpoints;
        # the two tests above already pin the names.
        return
    monkeypatch.chdir(tmp_path)
    assert os.path.isabs(resolve_marginal_checkpoint(rel))
    assert os.path.exists(resolve_marginal_checkpoint(rel))
