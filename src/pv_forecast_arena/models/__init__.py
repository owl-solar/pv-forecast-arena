"""Model dispatch: `model.type` of an experiment config -> a function returning the forecast.

Every model function has the signature `predict(frame, split, cfg) -> pd.Series` and returns forecasts
for `split.test_index`, using only what is known at the issue time of each hour. Models that learn
parameters fit them on `split.train_index` only. A model may put per-fold details (a flat dict) in
`forecast.attrs["info"]`; the runner saves them to runs/<run_id>/folds.csv.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from pv_forecast_arena.evaluation.splits import Split
from pv_forecast_arena.models import (
    baselines,
    gbm,
    hybrid,
    intraday_baselines,
    intraday_gbm,
    intraday_physics,
    physics,
    quantile,
)

Predict = Callable[[pd.DataFrame, Split, dict], pd.Series]


def _smart_persistence(frame: pd.DataFrame, split: Split, cfg: dict) -> pd.Series:
    return baselines.smart_persistence(frame, cfg).reindex(split.test_index)


def _persistence(frame: pd.DataFrame, split: Split, cfg: dict) -> pd.Series:
    return baselines.persistence(frame, cfg).reindex(split.test_index)


def _lazy_dl(name: str) -> Callable:
    """The deep-learning models import torch only when they run (the `dl` extra is optional)."""

    def predict(frame, split, cfg):
        from pv_forecast_arena.models import dl

        return getattr(dl, name)(frame, split, cfg)

    return predict


MODELS: dict[str, Predict] = {
    "persistence": _persistence,
    "smart_persistence": _smart_persistence,
    "physics": physics.predict,
    "physics_measured_poa": physics.predict_measured_poa,
    "gbm": gbm.predict,
    "hybrid": hybrid.predict,
    "dl_lstm": _lazy_dl("predict_lstm"),
}


QuantilePredict = Callable[[pd.DataFrame, Split, dict], pd.DataFrame]

# Day-ahead quantile models: `predict(frame, split, cfg)` -> DataFrame, one column per level.
QUANTILE_MODELS: dict[str, QuantilePredict] = {
    "quantile_empirical": quantile.predict_empirical,
    "quantile_lgbm": quantile.predict_lgbm,
    "dl_tft_quantile": _lazy_dl("predict_tft_quantiles"),
}


def get_quantile_model(model_type: str) -> QuantilePredict:
    try:
        return QUANTILE_MODELS[model_type]
    except KeyError:
        raise ValueError(
            f"unknown quantile model type {model_type!r}; known: {sorted(QUANTILE_MODELS)}"
        ) from None


IntradayPredict = Callable[[pd.DataFrame, Split, dict, int], pd.Series]

INTRADAY_MODELS: dict[str, IntradayPredict] = {
    "intraday_persistence": intraday_baselines.persistence,
    "intraday_smart_persistence": intraday_baselines.smart_persistence,
    "intraday_physics": intraday_physics.predict,
    "intraday_physics_nowcast": intraday_physics.predict_nowcast,
}


IntradayFoldPredict = Callable[[pd.DataFrame, Split, dict], dict[int, pd.Series]]

# Models that train once per fold for all horizons together (they return {horizon: forecast}).
INTRADAY_FOLD_MODELS: dict[str, IntradayFoldPredict] = {"intraday_gbm": intraday_gbm.predict_all}


def get_intraday_fold_model(model_type: str) -> IntradayFoldPredict:
    """Fold-level view of every intraday model: `predict(frame, split, cfg) -> {horizon: forecast}`."""
    if model_type in INTRADAY_FOLD_MODELS:
        return INTRADAY_FOLD_MODELS[model_type]
    per_horizon = get_intraday_model(model_type)

    def predict(frame: pd.DataFrame, split: Split, cfg: dict) -> dict[int, pd.Series]:
        return {h: per_horizon(frame, split, cfg, h) for h in cfg["intraday"]["horizons"]}

    return predict


def get_intraday_model(model_type: str) -> IntradayPredict:
    """Intraday models take the horizon (hours ahead) as a fourth argument."""
    try:
        return INTRADAY_MODELS[model_type]
    except KeyError:
        raise ValueError(
            f"unknown intraday model type {model_type!r}; known: {sorted(INTRADAY_MODELS)}"
        ) from None


def get_model(model_type: str) -> Predict:
    try:
        return MODELS[model_type]
    except KeyError:
        raise ValueError(f"unknown model type {model_type!r}; known: {sorted(MODELS)}") from None
