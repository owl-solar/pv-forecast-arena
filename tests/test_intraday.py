"""Intraday: what a forecast issued at tau - h may see, and that the benchmarks follow it."""

import numpy as np
import pandas as pd
import pytest
from test_no_leakage import CFG, make_frame

from pv_forecast_arena import features
from pv_forecast_arena.evaluation import intraday as scoring
from pv_forecast_arena.evaluation.splits import Split
from pv_forecast_arena.models import intraday_baselines

ICFG = {**CFG, "intraday": {"horizons": [1, 2, 3, 4, 5, 6], "data_delay_hours": 1}}
PAST_ONLY = ("ac_power_kw", "obs_poa_wm2")


def _model(name, window=1):
    predict = {
        "persistence": intraday_baselines.persistence,
        "smart": intraday_baselines.smart_persistence,
    }[name]

    def run(frame, horizon, delay=1):
        cfg = {
            **ICFG,
            "intraday": {**ICFG["intraday"], "data_delay_hours": delay},
            "model": {"window_hours": window},
        }
        return predict(frame, Split(frame.index[:24], frame.index), cfg, horizon)

    return run


MODELS = {
    "persistence": _model("persistence"),
    "smart_1h": _model("smart", 1),
    "smart_3h": _model("smart", 3),
}


def test_as_of_gives_the_newest_stamp_known_at_issue_time():
    index = pd.date_range("2025-06-01", periods=48, freq="1h", tz="UTC")
    series = pd.Series(np.arange(48.0), index=index)
    target = index[30]
    # issued at target - 3 h, a measured hour is known 1 h after it ends: newest stamp is target - 4 h
    assert features.as_of(series, 3, 1)[target] == series[target - pd.Timedelta("4h")]
    assert features.as_of(series, 3, 0)[target] == series[target - pd.Timedelta("3h")]
    assert features.as_of(series, 1, 1).iloc[:2].isna().all()


@pytest.mark.parametrize("horizon", [0, 7, -1])
def test_horizon_outside_intraday_range_is_refused(horizon):
    frame = make_frame()
    with pytest.raises(ValueError, match="horizon"):
        features.as_of(frame["ac_power_kw"], horizon, 1)
    with pytest.raises(ValueError, match="horizon"):
        features.lead1_forecast(frame, ["shortwave_radiation"], ICFG, horizon)


def test_lead1_is_only_a_forecast_variable_and_unprefixed():
    frame = make_frame().assign(fc_lead1_shortwave_radiation=1.0, fc_lead1_cloud_cover=2.0)
    out = features.lead1_forecast(frame, ["shortwave_radiation", "cloud_cover"], ICFG, 6)
    assert list(out.columns) == ["shortwave_radiation", "cloud_cover"]
    with pytest.raises(ValueError, match="forecast variable"):
        features.lead1_forecast(frame, ["ac_power_kw"], ICFG, 1)


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_forecast_ignores_everything_after_issue_time(name, horizon):
    frame = make_frame()
    model = MODELS[name]
    base = model(frame, horizon)
    targets = frame.index[72::5]
    checked = 0
    for target in targets:
        if np.isnan(base[target]):
            continue
        # Known at issue time: stamps <= target - horizon - delay (delay = 1).
        last_known = target - pd.Timedelta(hours=horizon + 1)
        broken = frame.copy()
        future = broken.index > last_known
        for column in PAST_ONLY:
            broken.loc[future, column] = np.random.default_rng(0).random(future.sum()) * 1e4
        assert model(broken, horizon)[target] == base[target], f"{name} h={horizon} leaks"
        checked += 1
    assert checked > 10


@pytest.mark.parametrize("name", MODELS)
def test_the_leakage_test_can_fail(name):
    # Mutation check: the newest stamp the forecaster is allowed to see does change the forecast.
    frame = make_frame()
    model = MODELS[name]
    base = model(frame, 2)
    changed = 0
    for target in frame.index[72::5]:
        if np.isnan(base[target]):
            continue
        broken = frame.copy()
        broken.loc[target - pd.Timedelta("3h"), "ac_power_kw"] = 1e4  # = target - h - delay
        changed += model(broken, 2)[target] != base[target]
    assert changed > 0


def test_persistence_carries_the_newest_known_measurement():
    frame = make_frame()
    frame["ac_power_kw"] = np.arange(len(frame), dtype=float)
    target = frame.index[100]
    assert (
        MODELS["persistence"](frame, 4)[target] == frame["ac_power_kw"][target - pd.Timedelta("5h")]
    )


def test_smart_persistence_scales_clearsky_by_the_last_hours_ratio():
    frame = make_frame()
    hour = pd.Timestamp("2025-06-05 20:00", tz="UTC")  # local 13:00, daytime
    frame.loc[hour, "ac_power_kw"] = 0.5 * frame.loc[hour, "clearsky_poa"]
    target = hour + pd.Timedelta("3h")  # horizon 2 + delay 1
    expected = 0.5 * frame.loc[target, "clearsky_poa"]
    assert MODELS["smart_1h"](frame, 2)[target] == pytest.approx(expected)


def test_smart_persistence_has_no_forecast_when_the_newest_hours_are_night():
    frame = make_frame()
    morning = pd.Timestamp("2025-06-05 15:00", tz="UTC")  # local 08:00; known stamps are at night
    assert frame.loc[morning, "is_daytime"]
    assert np.isnan(MODELS["smart_1h"](frame, 6)[morning])


def test_scoring_reports_every_horizon_pooled_row_and_coverage():
    frame = make_frame()
    truth = frame["ac_power_kw"]
    forecasts = {h: truth * (1 + 0.1 * h) for h in (1, 2)}
    reference = {h: truth * 1.5 for h in (1, 2)}
    reference[2] = reference[2].where(frame.index.hour != 20)  # hole in the second horizon
    rows = scoring.score_horizons(frame, forecasts, reference, ICFG, "all")
    assert [r["lead_h"] for r in rows] == [1, 2, scoring.POOLED]
    assert rows[0]["coverage"] == pytest.approx(1.0)
    assert rows[1]["coverage"] < 1.0
    assert rows[2]["n_hours"] == rows[0]["n_hours"] + rows[1]["n_hours"]
    assert rows[0]["nrmse"] < rows[1]["nrmse"]
