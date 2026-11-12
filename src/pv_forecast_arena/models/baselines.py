"""Benchmarks. Smart persistence is the reference for every skill score."""

from __future__ import annotations

import pandas as pd

from pv_forecast_arena.evaluation.splits import issue_time

MIN_OBSERVED_HOURS = 3


def persistence(frame: pd.DataFrame, cfg: dict) -> pd.Series:
    """Day-ahead persistence: the measurement of the same hour on the latest day known at issue time.

    The forecast for a target hour is the power 24 h earlier if that hour was fully observed at the
    issue time (10:00 local D-1), which holds for target hours up to 10:00 local. Later target hours
    would see a value that is still in the future at issue time, so they use the value 48 h earlier.
    Missing measurements give NaN: no gap filling.
    """
    issued = issue_time(
        frame.index, cfg["site"]["local_timezone"], cfg["weather"]["forecast_issue_hour_local"]
    )
    source = frame.index - pd.Timedelta("24h")
    source = source.where(source <= issued, frame.index - pd.Timedelta("48h"))
    # Cannot trigger on a regular hourly index; keeps "nothing after issue time" explicit.
    source = source.where(source <= issued)
    values = frame["ac_power_kw"].reindex(source).to_numpy()
    return pd.Series(values, index=frame.index, name="persistence")


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
