"""Intraday: what a forecast issued at tau - h may see, and that the benchmarks follow it."""

import numpy as np
import pandas as pd
import pytest
from test_no_leakage import CFG, make_frame

from pv_forecast_arena import features
from pv_forecast_arena.evaluation import intraday as scoring
from pv_forecast_arena.evaluation.splits import Split
from pv_forecast_arena.models import intraday_baselines, intraday_gbm, intraday_physics

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


def _physics_model(predict, **model_cfg):
    def run(frame, horizon, delay=1):
        frame = _with_lead1(frame)
        cfg = {
            **ICFG,
            "model": {"decomposition": "erbs", "transposition": "perez", **model_cfg},
        }
        # Static scale and limit come from the first day only: later values may be corrupted.
        return predict(frame, Split(frame.index[:24], frame.index), cfg, horizon)

    return run


def _with_lead1(frame):
    """Lead-1 forecasts are archived weather, not measurements: the leakage test leaves them alone."""
    out = frame.copy()
    for variable in ICFG["weather"]["variables"]:
        out[f"fc_lead1_{variable}"] = frame[f"fc_{variable}"] * 1.05
    return out


MODELS = {
    "persistence": _model("persistence"),
    "physics_lead1": _physics_model(intraday_physics.predict),
    "physics_nowcast": _physics_model(
        intraday_physics.predict_nowcast, window_hours=2, decay_hours=3
    ),
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


@pytest.mark.parametrize("name", [n for n in MODELS if n != "physics_lead1"])  # reads no live data
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


GBM_VARIANTS = {
    "residual": {"target": "residual"},
    "residual_nowcast": {"target": "residual", "baseline": "nowcast"},
    "csi": {"target": "csi"},
    "nonwp": {"target": "csi", "use_nwp": False},
}


def _gbm_cfg(**model):
    base = {"decomposition": "erbs", "transposition": "perez", "min_train_hours": 5}
    base["params"] = {"min_data_in_leaf": 2}
    return {**ICFG, "model": {**base, **model}}


@pytest.mark.parametrize("variant", GBM_VARIANTS)
@pytest.mark.parametrize("horizon", [1, 4, 6])
def test_gbm_feature_row_ignores_everything_after_issue_time(variant, horizon):
    # Trees hide leakage in the forecast (a corrupted value lands in the same leaf), so check the rows.
    frame = _with_lead1(make_frame())
    cfg = _gbm_cfg(**GBM_VARIANTS[variant])
    split = Split(frame.index[:24], frame.index)  # limit and static scale from day one only

    def rows(f):
        static, parts = intraday_gbm.static_design(f, split, cfg)
        return intraday_gbm.horizon_design(f, static, parts, cfg, horizon)

    base_x, base_line, base_den = rows(frame)
    targets = frame.index[72::5]
    for target in targets:
        broken = frame.copy()
        future = broken.index > target - pd.Timedelta(hours=horizon + 1)
        for column in PAST_ONLY:
            broken.loc[future, column] = np.random.default_rng(0).normal(0, 1e4, future.sum())
        x, line, den = rows(broken)
        pd.testing.assert_series_equal(x.loc[target], base_x.loc[target])
        assert line[target] == base_line[target] or (
            np.isnan(line[target]) and np.isnan(base_line[target])
        )
        assert den[target] == base_den[target]

    # Mutation check: the newest allowed stamp is read (so the comparison above can fail).
    changed = 0
    for target in targets:
        broken = frame.copy()
        broken.loc[target - pd.Timedelta(hours=horizon + 1), "ac_power_kw"] = 1e4
        x, _, _ = rows(broken)
        changed += not x.loc[target].equals(base_x.loc[target])
    assert changed > 0


@pytest.mark.parametrize("per_horizon", [False, True])
@pytest.mark.parametrize("variant", GBM_VARIANTS)
def test_gbm_forecasts_every_horizon_bounded_and_zero_at_night(variant, per_horizon):
    frame = _with_lead1(make_frame())
    cfg = _gbm_cfg(**GBM_VARIANTS[variant], per_horizon=per_horizon)
    split = Split(frame.index[: 24 * 8], frame.index[24 * 8 :])
    out = intraday_gbm.predict_all(frame, split, cfg)
    assert sorted(out) == [1, 2, 3, 4, 5, 6]
    for forecast in out.values():
        assert forecast.index.equals(split.test_index)
        assert forecast.notna().all() and (forecast >= 0).all()
        assert (forecast[~frame.loc[forecast.index, "is_daytime"]] == 0).all()
        assert "rounds" in forecast.attrs["info"]
