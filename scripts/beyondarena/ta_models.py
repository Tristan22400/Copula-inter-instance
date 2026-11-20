"""TabArena model classes for the BeyondArena official track (imported inside the tabarena venv only).

The fine-tuned marginal runs as TabArena's own TA-TabICLv2 wrapper, unchanged except
for its method name and the regression checkpoint, which the experiment passes as the
``model_path`` hyperparameter. Preprocessing, the validation protocol, bagging and
the n_estimators / batch-size overrides are TabArena's, so a difference from the
cached TA-TabICLv2 baseline comes from the weights alone.
"""

from __future__ import annotations

from tabarena.models.tabicl.model import TabICLv2Model


class TabICLv2FTModel(TabICLv2Model):
    """TA-TabICLv2 with the ERA5-fine-tuned regression weights (``model_path``)."""

    ag_key = "TA-TABICLv2-ERA5FT"
    ag_name = "TA-TabICLv2-ERA5FT"
