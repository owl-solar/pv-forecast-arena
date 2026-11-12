"""Leakage checks shared by every model that reads past measurements.

For a sample of target hours: destroy all data stamped after the target's issue time and the forecast
for that hour must not change. A model that peeks past the issue time fails this.
"""

import numpy as np
import pandas as pd
import pytest

from pv_forecast_arena.evaluation.splits import Split, issue_time
from pv_forecast_arena.models import baselines, gbm, hybrid, physics, tabular

TZ = "America/Los_Angeles"
CFG = {
    "site": {
        "ac_capacity_kw": 700.0,
        "dc_capacity_kw": 893.0,
        "latitude": 38.996,
        "longitude": -122.134,
        "tilt_deg": 25,
        "azimuth_deg": 180,
        "local_timezone": TZ,
    },
    "weather": {
        "variables": [*physics.WEATHER, "cloud_cover"],
        "forecast_issue_hour_local": 10,
    },
    "cleaning": {"daytime_elevation_deg": 5},
    "seed": 42,
}
PAST_ONLY = ("ac_power_kw", "obs_poa_wm2")


def _physics(calibration):
    def model(frame, cfg):
        cfg = {
            **cfg,
            "model": {
                "decomposition": "erbs",
                "transposition": "perez",
                "calibration": calibration,
                "rolling_days": 3,
            },
        }
        # Train on the first day only, so corrupted values never reach the fitted limit or static k.
        train = frame.index[:24]
        return physics.predict(frame, Split(train, frame.index), cfg)

    return model


def _tabular(predict, model_cfg):
    def model(frame, cfg):
        cfg = {
            **cfg,
            "model": {
                **model_cfg,
                "min_train_hours": 5,
                "params": {"min_data_in_leaf": 2, "min_child_weight": 2},
            },
        }
        # Labels of the first day only: every later value may be corrupted by the test.
        return predict(frame, Split(frame.index[:24], frame.index), cfg)

    return model


PHYSICS_CFG = {"decomposition": "erbs", "transposition": "perez", "calibration": "static"}
MODELS = {
    "persistence": baselines.persistence,
    "smart_persistence": baselines.smart_persistence,
    "physics_rolling_poa": _physics("rolling_poa"),
    "physics_rolling_forecast": _physics("rolling_forecast"),
    "gbm": _tabular(gbm.predict, {"target": "ac"}),
    "hybrid_residual_past": _tabular(
        hybrid.predict,
        {"target": "residual", "use_past": True, "rolling_days": 3, "physics": PHYSICS_CFG},
    ),
    "hybrid_csi_past_xgb": _tabular(
        hybrid.predict,
        {
            "target": "csi",
            "learner": "xgboost",
            "use_past": True,
            "rolling_days": 3,
            "physics": PHYSICS_CFG,
        },
    ),
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
            "clearsky_ghi": clearsky * 0.95,
            "sun_elevation": np.where(daytime, 40.0, -10.0),
            "is_daytime": daytime,
            "flag_bad": False,
            "obs_poa_wm2": clearsky * (0.5 + 0.5 * rng.random(len(index))),
            "fc_shortwave_radiation": clearsky * 0.9,
            "fc_direct_radiation": clearsky * 0.6,
            "fc_temperature_2m": 20.0,
            "fc_wind_speed_10m": 8.0,
            "fc_cloud_cover": 100.0 * rng.random(len(index)),
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
        for column in PAST_ONLY:
            broken.loc[future, column] = np.random.default_rng(0).random(future.sum()) * 1e4
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


@pytest.mark.parametrize("target", ["ac", "residual", "csi"])
def test_tree_model_features_ignore_everything_after_issue_time(target):
    # Trees map an out-of-range corrupted value to the same leaf as a large true value, so the
    # forecast-level test above is weak for them: check the feature row of the target hour directly.
    frame = make_frame()
    model_cfg = {"target": target, "use_past": True, "rolling_days": 3, "physics": PHYSICS_CFG}
    cfg = {**CFG, "model": model_cfg}
    split = Split(frame.index[:24], frame.index)
    base, base_line, base_extra = tabular.design(frame, split, cfg)
    targets = frame.index[72::5]
    for target_hour, when in zip(targets, issue_time(targets, TZ, 10)):
        broken = frame.copy()
        future = broken.index > when
        for column in PAST_ONLY:
            broken.loc[future, column] = np.random.default_rng(0).normal(0, 1e4, future.sum())
        x, line, extra = tabular.design(broken, split, cfg)
        pd.testing.assert_series_equal(x.loc[target_hour], base.loc[target_hour])
        assert line[target_hour] == base_line[target_hour]
        assert extra["denominator"][target_hour] == base_extra["denominator"][target_hour]
