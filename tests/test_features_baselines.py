import numpy as np
import pandas as pd
import pytest

from pv_forecast_arena.evaluation.splits import issue_time
from pv_forecast_arena.features import allowed_features, select_features
from pv_forecast_arena.models.baselines import smart_persistence

TZ = "America/Los_Angeles"
CFG = {
    "site": {"ac_capacity_kw": 700.0, "local_timezone": TZ},
    "weather": {
        "variables": ["shortwave_radiation", "cloud_cover"],
        "forecast_issue_hour_local": 10,
    },
}


@pytest.mark.parametrize(
    "column",
    [
        "fc_lead1_shortwave_radiation",
        "oracle_shortwave_radiation",
        "ac_power_kw",
        "obs_poa_wm2",
        "flag_bad",
        "flag_outage",
        "brand_new_column",
    ],
)
def test_leaky_columns_are_rejected(column):
    frame = pd.DataFrame({column: [1.0]})
    with pytest.raises(ValueError, match=column):
        select_features(frame, [column], CFG)


def test_main_run_allows_only_main_lead_forecasts():
    allowed = allowed_features(CFG)
    assert "fc_shortwave_radiation" in allowed
    assert not [c for c in allowed if c.startswith(("oracle_", "fc_lead1_"))]


def test_oracle_columns_need_explicit_oracle_run():
    frame = pd.DataFrame({"oracle_cloud_cover": [1.0]})
    assert list(select_features(frame, ["oracle_cloud_cover"], CFG, oracle=True).columns)


def make_frame():
    index = pd.date_range(
        "2025-06-01 01:00", "2025-06-12 00:00", freq="1h", tz="UTC", name="valid_time"
    )
    local_hour = index.tz_convert(TZ).hour
    daytime = (local_hour >= 7) & (local_hour <= 18)
    clearsky = np.where(daytime, 800.0, 0.0)
    return pd.DataFrame(
        {"ac_power_kw": 0.5 * clearsky, "clearsky_poa": clearsky, "is_daytime": daytime},
        index=index,
    )


def test_recovers_constant_ratio():
    frame = make_frame()
    forecast = smart_persistence(frame, CFG).dropna()
    assert len(forecast) > 150
    assert np.allclose(forecast, frame.loc[forecast.index, "ac_power_kw"])


def test_uses_no_data_after_issue_time():
    frame = make_frame()
    base = smart_persistence(frame, CFG)
    target = pd.Timestamp("2025-06-08 20:00", tz="UTC")  # 13:00 local
    issued = issue_time(pd.DatetimeIndex([target]), TZ, 10)[0]

    changed = frame.copy()
    changed.loc[changed.index > issued, "ac_power_kw"] *= 0.1
    assert smart_persistence(changed, CFG)[target] == base[target]

    changed = frame.copy()
    changed.loc[changed.index <= issued, "ac_power_kw"] *= 0.1
    assert smart_persistence(changed, CFG)[target] != base[target]


def test_missing_history_gives_nan():
    frame = make_frame()
    frame["ac_power_kw"] = np.nan
    assert smart_persistence(frame, CFG).isna().all()
