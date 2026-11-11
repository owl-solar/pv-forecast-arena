"""Benchmarks. Smart persistence is the reference for every skill score."""

from __future__ import annotations

import pandas as pd

from pv_forecast_arena.evaluation.splits import issue_time

MIN_OBSERVED_HOURS = 3


def smart_persistence(frame: pd.DataFrame, cfg: dict) -> pd.Series:
    """Day-ahead smart persistence: clear-sky power ratio of the last 24 h known at issue time.

    For each issue time, k = sum(power) / sum(clear-sky POA) over daytime hours in (issue - 24 h, issue];
    a 24 h window spans one full daily cycle. The forecast is k * clear-sky POA of the target hour,
    limited to [0, capacity]. Uses only past measurements and deterministic geometry.
    Issues with fewer than `MIN_OBSERVED_HOURS` observed daytime hours give NaN.
    """
    observed = frame["is_daytime"] & frame["ac_power_kw"].notna()
    power = frame["ac_power_kw"].where(observed, 0.0).rolling("24h").sum()
    clearsky = frame["clearsky_poa"].where(observed, 0.0).rolling("24h").sum()
    hours = observed.astype(float).rolling("24h").sum()
    ratio = (power / clearsky).where(hours >= MIN_OBSERVED_HOURS)

    issued = issue_time(
        frame.index, cfg["site"]["local_timezone"], cfg["weather"]["forecast_issue_hour_local"]
    )
    forecast = ratio.reindex(issued).to_numpy() * frame["clearsky_poa"].to_numpy()
    return pd.Series(forecast, index=frame.index, name="smart_persistence").clip(
        0.0, cfg["site"]["ac_capacity_kw"]
    )
