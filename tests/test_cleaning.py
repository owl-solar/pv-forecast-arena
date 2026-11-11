import numpy as np
import pandas as pd

from pv_forecast_arena.data import cleaning, pvdaq

CFG = {
    "site": {"ac_capacity_kw": 700.0},
    "cleaning": {"daytime_elevation_deg": 5, "stuck_window_hours": 4, "clipping_quantile": 0.995},
}


def make(n=48):
    index = pd.date_range("2025-06-01 01:00", periods=n, freq="1h", tz="UTC")
    hour = (index.hour + index.minute / 60).to_numpy()
    # simple geometry: day = 14..24 UTC (about 7:00-17:00 local), night otherwise
    day = (hour >= 15) | (hour <= 2)
    elevation = pd.Series(np.where(day, 40.0, -20.0), index=index)
    geometry = pd.DataFrame({"sun_elevation": elevation, "clearsky_ghi": 0.0})
    poa = pd.Series(np.where(day, 700.0, 0.0), index=index)
    ac = pd.Series(np.where(day, 300.0 + 3.0 * np.arange(n) % 17, 0.0), index=index)
    return ac, poa, geometry


def test_night_values_are_zeroed_and_counted():
    ac, poa, geometry = make()
    night_idx = ac.index[geometry["sun_elevation"] < 0][3]
    ac[night_idx] = 12.0
    out = cleaning.clean_target(ac, poa, geometry, CFG)
    assert out.loc[night_idx, "ac_power_kw"] == 0.0
    assert out.loc[night_idx, "night_zeroed"]
    assert out["night_zeroed"].sum() == 1
    assert len(out) == len(ac)


def test_outage_flagged_when_sun_but_no_power():
    ac, poa, geometry = make()
    hour = ac.index[geometry["sun_elevation"] > 5][4]
    ac[hour] = 0.0
    out = cleaning.clean_target(ac, poa, geometry, CFG)
    assert out.loc[hour, "flag_outage"]
    assert out["flag_outage"].sum() == 1


def test_stuck_values_flagged_in_daytime_only():
    ac, poa, geometry = make()
    days = ac.index[geometry["sun_elevation"] > 5]
    ac[days[2:8]] = 345.5
    out = cleaning.clean_target(ac, poa, geometry, CFG)
    assert out["flag_stuck"].sum() >= 4
    assert not out.loc[geometry["sun_elevation"] < 0, "flag_stuck"].any()


def test_missing_is_flagged_not_dropped():
    ac, poa, geometry = make()
    hour = ac.index[geometry["sun_elevation"] > 5][5]
    ac[hour] = np.nan
    out = cleaning.clean_target(ac, poa, geometry, CFG)
    assert out.loc[hour, "flag_missing"] and out["flag_missing"].sum() == 1 and len(out) == len(ac)


def test_to_utc_local_clock_and_dst():
    idx = pd.to_datetime(["2025-01-15 12:00", "2025-07-15 12:00"])
    out = pvdaq.to_utc(pd.Series([1.0, 2.0], index=idx), "local", "America/Los_Angeles")
    assert list(out.index.hour) == [20, 19]
    out_utc = pvdaq.to_utc(pd.Series([1.0, 2.0], index=idx), "utc", "America/Los_Angeles")
    assert list(out_utc.index.hour) == [12, 12]
