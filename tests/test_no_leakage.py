"""Leakage checks shared by every model that reads past measurements.

For a sample of target hours: destroy all data stamped after the target's issue time and the forecast
for that hour must not change. A model that peeks past the issue time fails this.
"""

import numpy as np
import pandas as pd
import pytest

from pv_forecast_arena.evaluation.splits import issue_time
from pv_forecast_arena.models import baselines

TZ = "America/Los_Angeles"
CFG = {
    "site": {"ac_capacity_kw": 700.0, "local_timezone": TZ},
    "weather": {"variables": ["shortwave_radiation"], "forecast_issue_hour_local": 10},
}
MODELS = {
    "persistence": baselines.persistence,
    "smart_persistence": baselines.smart_persistence,
}


def make_frame(days=12):
    index = pd.date_range("2025-06-01 01:00", periods=24 * days, freq="1h", tz="UTC")
    rng = np.random.default_rng(3)
    local_hour = index.tz_convert(TZ).hour
    daytime = (local_hour >= 7) & (local_hour <= 18)
    clearsky = np.where(daytime, 800.0, 0.0)
    return pd.DataFrame(
        {
            "ac_power_kw": clearsky * (0.3 + 0.4 * rng.random(len(index))),
            "clearsky_poa": clearsky,
            "is_daytime": daytime,
        },
        index=index,
    )


@pytest.mark.parametrize("name", MODELS)
def test_forecast_ignores_everything_after_issue_time(name):
    frame = make_frame()
    model = MODELS[name]
    base = model(frame, CFG)
    targets = frame.index[72::5]
    issued = issue_time(targets, TZ, 10)
    checked = 0
    for target, when in zip(targets, issued):
        if np.isnan(base[target]):
            continue
        broken = frame.copy()
        future = broken.index > when
        broken.loc[future, "ac_power_kw"] = np.random.default_rng(0).random(future.sum()) * 1e4
        assert model(broken, CFG)[target] == base[target], f"{name} leaks at {target}"
        checked += 1
    assert checked > 10


def test_persistence_takes_24h_if_known_at_issue_else_48h():
    frame = make_frame()
    frame["ac_power_kw"] = np.arange(len(frame), dtype=float)
    forecast = baselines.persistence(frame, CFG)
    # 09:00 local: yesterday's 09:00 was observed before the 10:00 issue time.
    early = pd.Timestamp("2025-06-08 16:00", tz="UTC")
    # 15:00 local: yesterday's 15:00 is after the issue time, so the day before is used.
    late = pd.Timestamp("2025-06-08 22:00", tz="UTC")
    assert forecast[early] == frame.loc[early - pd.Timedelta("24h"), "ac_power_kw"]
    assert forecast[late] == frame.loc[late - pd.Timedelta("48h"), "ac_power_kw"]


def test_persistence_does_not_fill_gaps_and_starts_with_nan():
    frame = make_frame()
    hole = pd.Timestamp("2025-06-05 20:00", tz="UTC")
    frame.loc[hole, "ac_power_kw"] = np.nan
    forecast = baselines.persistence(frame, CFG)
    assert np.isnan(forecast[hole + pd.Timedelta("48h")])
    assert forecast.iloc[:24].isna().all()
