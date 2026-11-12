"""Pure ML: gradient boosting on geometry and weather forecasts only (no physics, no measurements).

See `tabular` for features, targets and fitting. Configs must not have a `physics` section.
"""

from __future__ import annotations

import pandas as pd

from pv_forecast_arena.evaluation.splits import Split
from pv_forecast_arena.models import tabular


def predict(frame: pd.DataFrame, split: Split, cfg: dict) -> pd.Series:
    if "physics" in cfg["model"] or cfg["model"].get("use_past", False):
        raise ValueError("pure ML uses no physics and no past measurements; use type 'hybrid'")
    return tabular.fit_predict(frame, split, cfg, "gbm")
