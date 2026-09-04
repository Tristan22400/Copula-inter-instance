"""marginal_finetune.py — backward-compatibility shim.

All Phase-A implementation and entrypoints have been merged into
src/finetune_marginal.py. This file re-exports everything so existing imports
and tests continue to work without modification.
"""

from finetune_marginal import *  # noqa: F401, F403
