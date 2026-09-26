"""The vendored tabicl_upstream is the TabICL every entrypoint imports."""

from pathlib import Path

import tabicl
from tabicl.__about__ import __version__
from tabicl._model import attention

VENDORED = Path(__file__).resolve().parents[1] / "tabicl_upstream" / "src" / "tabicl"


def test_tabicl_resolves_to_vendored_copy() -> None:
    assert Path(tabicl.__file__).resolve().parent == VENDORED
    assert __version__ == "2.2.0"


def test_sdpa_grid_limit_patch_present() -> None:
    assert attention._MAX_SDPA_FLAT_BATCH == 65535
