"""Physics chain for intraday horizons, fed with the lead-1 forecast of the target hour.

`intraday_physics`: the Stage 2 chain (static scale k, limit = largest AC in the training window)
on `fc_lead1_*` instead of the day-ahead forecast. The lead-1 run starts 24 h before the target
hour, so it is fresher than the 10:00 D-1 issue and known at any intraday issue time.

`intraday_physics_nowcast`: the same, multiplied by the plant's recent performance ratio
r = sum(AC) / sum(model AC) over the last `window_hours` known hours (hours of the target's own
lead-1 forecast, so r measures plant state and the lead-1 forecast error together). The correction
fades with the horizon, r_eff = 1 + (r - 1) * exp(-h / decay_hours); both parameters are fixed in the
config and were not tuned. Without a known daytime hour r is undefined and the forecast is the plain
physics one, so this model covers every daytime hour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pv_forecast_arena import features
from pv_forecast_arena.features import select_past
from pv_forecast_arena.models import physics

MIN_WINDOW_MODEL_KW = 1.0  # sum of modelled power below this says nothing about the ratio
RATIO_RANGE = (0.2, 2.0)


def components(frame: pd.DataFrame, split, cfg: dict, horizon: int) -> dict:
    weather = features.lead1_forecast(frame, physics.WEATHER, cfg, horizon)
    model_cfg = {**cfg["model"], "calibration": "static"}
    return physics.components(frame, split, cfg, model_cfg, weather=weather)


def predict(frame: pd.DataFrame, split, cfg: dict, horizon: int) -> pd.Series:
    forecast = components(frame, split, cfg, horizon)["forecast"]
    return forecast.reindex(split.test_index).rename("intraday_physics")


def recent_ratio(
    frame: pd.DataFrame, model: pd.Series, limit: float, window_hours: int
) -> pd.Series:
    """Backward-looking r at every stamp: sum(AC) / sum(model) over its last `window_hours` stamps.

    Clipped hours (near the inverter limit) and non-daytime hours are left out, as in the day-ahead
    calibration. Uses stamps <= the stamp itself only.
    """
    ac = select_past(frame, ["ac_power_kw"])["ac_power_kw"]
    ok = (
        frame["is_daytime"] & ac.notna() & model.notna() & (ac < physics.UNCLIPPED_FRACTION * limit)
    )
    window = f"{window_hours}h"
    num = ac.where(ok, 0.0).rolling(window).sum()
    den = model.where(ok, 0.0).rolling(window).sum()
    return (num / den).where(den >= MIN_WINDOW_MODEL_KW).clip(*RATIO_RANGE)


def predict_nowcast(frame: pd.DataFrame, split, cfg: dict, horizon: int) -> pd.Series:
    m = cfg["model"]
    parts = components(frame, split, cfg, horizon)
    scaled = parts["k"] * parts["unlimited"]
    ratio = recent_ratio(frame, scaled, parts["limit"], int(m.get("window_hours", 1)))
    ratio = features.as_of(ratio, horizon, features.delay_hours(cfg))
    fade = np.exp(-horizon / float(m.get("decay_hours", 3.0)))
    effective = (1.0 + (ratio - 1.0) * fade).fillna(1.0)
    forecast = (effective * scaled).clip(lower=0.0, upper=parts["limit"])
    return forecast.reindex(split.test_index).rename("intraday_physics_nowcast")
