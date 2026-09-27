"""This checkout reproduces pre-refactor main's numerics (tests/refactor_golden.py has the what and how).

The golden file was computed on main @ 42c0402 with the pre-refactor source tree; on the same
machine the refactor reproduces it bit-for-bit. The tolerance only absorbs cross-machine BLAS /
CPU-kernel differences -- a real behavioural change (different RNG consumption, a changed
formula, a reordered parameter) moves these values by orders of magnitude more.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from refactor_golden import GOLDEN_REF, SMALL_MODEL_CFG, compute, load_modules

REPO = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "data" / "refactor_golden.pt"
RTOL, ATOL = 1e-4, 1e-5


@pytest.fixture(scope="module")
def golden() -> dict:
    return torch.load(GOLDEN, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def current() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return compute(load_modules("branch", REPO))


def _mismatches(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> list[str]:
    bad = [f"{k}: missing" for k in sorted(set(expected) - set(actual))]
    bad += [f"{k}: not in golden (regenerate?)" for k in sorted(set(actual) - set(expected))]
    for k in sorted(set(expected) & set(actual)):
        e, a = expected[k], actual[k]
        if e.shape != a.shape or e.dtype != a.dtype:
            bad.append(f"{k}: {tuple(e.shape)} {e.dtype} -> {tuple(a.shape)} {a.dtype}")
        elif not e.is_floating_point():
            if not torch.equal(e, a):
                bad.append(f"{k}: integer/bool values differ")
        elif not torch.allclose(a.double(), e.double(), rtol=RTOL, atol=ATOL, equal_nan=True):
            diff = (a.double() - e.double()).abs().nan_to_num(float("inf")).max().item()
            bad.append(f"{k}: max |diff| {diff:.3g}")
    return bad


def test_golden_is_from_pre_refactor_main(golden: dict) -> None:
    assert golden["ref"] == GOLDEN_REF


def test_numerics_match_pre_refactor_main(golden: dict, current: tuple) -> None:
    values, _ = current
    bad = _mismatches(golden["values"], values)
    assert not bad, f"{len(bad)} value(s) drifted from main @ {GOLDEN_REF}:\n" + "\n".join(bad)


def test_seeded_model_init_matches_pre_refactor_main(golden: dict, current: tuple) -> None:
    _, init_state = current
    bad = _mismatches(golden["init_state"], init_state)
    assert not bad, f"{len(bad)} parameter(s) drifted from main @ {GOLDEN_REF}:\n" + "\n".join(bad)


def test_pre_refactor_state_dict_loads_strictly() -> None:
    """A state_dict written by pre-refactor code loads into today's model with no key/shape drift."""
    from omegaconf import OmegaConf

    from copula_inter.model import build_copula_transformer

    state = torch.load(GOLDEN, map_location="cpu", weights_only=False)["init_state"]
    model = build_copula_transformer(OmegaConf.create(SMALL_MODEL_CFG))
    model.load_state_dict(state, strict=True)
