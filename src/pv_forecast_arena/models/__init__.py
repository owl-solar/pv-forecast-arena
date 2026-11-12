"""Model dispatch: `model.type` of an experiment config -> a function returning the forecast.

Every model function has the signature `predict(frame, split, cfg) -> pd.Series` and returns forecasts
for `split.test_index`, using only what is known at the issue time of each hour. Models that learn
parameters fit them on `split.train_index` only.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from pv_forecast_arena.evaluation.splits import Split
from pv_forecast_arena.models import baselines

Predict = Callable[[pd.DataFrame, Split, dict], pd.Series]


def _smart_persistence(frame: pd.DataFrame, split: Split, cfg: dict) -> pd.Series:
    return baselines.smart_persistence(frame, cfg).reindex(split.test_index)


MODELS: dict[str, Predict] = {"smart_persistence": _smart_persistence}


def get_model(model_type: str) -> Predict:
    try:
        return MODELS[model_type]
    except KeyError:
        raise ValueError(f"unknown model type {model_type!r}; known: {sorted(MODELS)}") from None
