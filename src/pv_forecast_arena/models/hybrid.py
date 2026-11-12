"""Hybrids of the physics chain and gradient boosting.

`model.target` picks the hybrid (see `tabular` for details):
- `ac`: physics forecast as a feature, target AC;
- `residual`: ML on the residual of the physics forecast;
- `csi`: target = AC / physics clear-sky power.
`model.physics` holds the physics settings (decomposition, transposition, calibration);
`model.use_past` adds rolling k and the smart-persistence ratio as of the issue time.
"""

from __future__ import annotations

import pandas as pd

from pv_forecast_arena.evaluation.splits import Split
from pv_forecast_arena.models import tabular


def predict(frame: pd.DataFrame, split: Split, cfg: dict) -> pd.Series:
    if "physics" not in cfg["model"]:
        raise ValueError("a hybrid needs a `physics` section")
    return tabular.fit_predict(frame, split, cfg, "hybrid")
