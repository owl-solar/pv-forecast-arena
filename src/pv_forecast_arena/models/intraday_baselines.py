"""Intraday benchmarks (horizons 1-6 h). Signature: `predict(frame, split, cfg, horizon)`.

The forecast for target hour tau is issued at tau - horizon and sees measurements stamped at most
`horizon + data_delay_hours` before tau (see `features.as_of`). Missing inputs give NaN, never a fill.
"""

from __future__ import annotations

import pandas as pd

from pv_forecast_arena import features
from pv_forecast_arena.features import select_past

MIN_CLEARSKY_SUM = 20.0  # W/m2 summed over the window; below this the ratio is noise


def clearsky_ratio_window(frame: pd.DataFrame, window_hours: int) -> pd.Series:
    """k = sum(power) / sum(clear-sky POA) over the daytime hours of the last `window_hours` stamps.

    Backward-looking: the value at stamp s uses stamps <= s only. NaN when the window has no
    observed daytime hour (night, or all missing).
    """
    ac = select_past(frame, ["ac_power_kw"])["ac_power_kw"]
    observed = frame["is_daytime"] & ac.notna()
    window = f"{window_hours}h"
    power = ac.where(observed, 0.0).rolling(window).sum()
    clearsky = frame["clearsky_poa"].where(observed, 0.0).rolling(window).sum()
    return (power / clearsky).where(clearsky >= MIN_CLEARSKY_SUM)


def persistence(frame: pd.DataFrame, split, cfg: dict, horizon: int) -> pd.Series:
    """Newest known measurement of AC, carried forward unchanged."""
    ac = select_past(frame, ["ac_power_kw"])["ac_power_kw"]
    out = features.as_of(ac, horizon, features.delay_hours(cfg))
    return out.reindex(split.test_index).rename("persistence_h")


def smart_persistence(frame: pd.DataFrame, split, cfg: dict, horizon: int) -> pd.Series:
    """Clear-sky persistence: newest known k times the clear-sky POA of the target hour.

    `model.window_hours` hours of measurements form k (1 = the newest hour alone).
    """
    window = int(cfg["model"].get("window_hours", 1))
    k = features.as_of(clearsky_ratio_window(frame, window), horizon, features.delay_hours(cfg))
    forecast = (k * frame["clearsky_poa"]).clip(0.0, cfg["site"]["ac_capacity_kw"])
    return forecast.reindex(split.test_index).rename("smart_persistence_h")
