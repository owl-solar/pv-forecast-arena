import numpy as np
import pandas as pd
import pytest
from test_no_leakage import CFG, PHYSICS_CFG, TZ, make_frame

from pv_forecast_arena.evaluation.splits import Split
from pv_forecast_arena.models import gbm, hybrid, tabular

SMALL = {"min_train_hours": 5, "params": {"min_data_in_leaf": 2, "min_child_weight": 2}}


def _cfg(**model):
    return {**CFG, "model": {**SMALL, **model}}


def _split(frame, train_days=6):
    return Split(frame.index[: 24 * train_days], frame.index[24 * train_days :])


def test_features_use_no_measurements_flags_or_lead1():
    frame = make_frame()
    frame["fc_lead1_shortwave_radiation"] = 1.0
    x = tabular.weather_features(frame, CFG)
    before = x.copy()
    for column in ["ac_power_kw", "obs_poa_wm2", "fc_lead1_shortwave_radiation"]:
        frame[column] = -1.0
    frame["flag_bad"] = True
    pd.testing.assert_frame_equal(tabular.weather_features(frame, CFG), before)


def test_oracle_features_ignore_the_forecast():
    frame = make_frame()
    for column in [c for c in frame if c.startswith("fc_")]:
        frame["oracle_" + column.removeprefix("fc_")] = frame[column] * 1.1
    cfg = {**CFG, "oracle": True}
    before = tabular.weather_features(frame, cfg)
    frame[[c for c in frame if c.startswith("fc_")]] = -1.0
    pd.testing.assert_frame_equal(tabular.weather_features(frame, cfg), before)


def test_daily_aggregates_stay_inside_the_target_local_day():
    frame = make_frame()
    x = tabular.weather_features(frame, CFG)
    day = tabular._local_day(frame.index, CFG)
    target_day = day.unique()[4]
    later = day > target_day
    frame.loc[later, ["fc_shortwave_radiation", "fc_cloud_cover"]] = 0.0
    y = tabular.weather_features(frame, CFG)
    rows = day == target_day
    for column in ["day_kt", "day_cloud_cover"]:
        pd.testing.assert_series_equal(x.loc[rows, column], y.loc[rows, column])


def test_daily_aggregates_use_local_days_not_utc_days():
    frame = make_frame()
    day = tabular._local_day(frame.index, CFG)
    # 01:00 UTC stamp covers 00:00-01:00 UTC = 17:00-18:00 local of the previous local day (PDT).
    stamp = pd.Timestamp("2025-06-05 01:00", tz="UTC")
    assert day[frame.index.get_loc(stamp)] == pd.Timestamp("2025-06-04")
    assert stamp.tz_convert(TZ).day == 4


@pytest.mark.parametrize(
    ("predict", "model"),
    [
        (gbm.predict, {"target": "ac"}),
        (gbm.predict, {"target": "csi", "learner": "xgboost"}),
        (hybrid.predict, {"target": "ac", "physics": PHYSICS_CFG}),
        (hybrid.predict, {"target": "residual", "use_past": True, "physics": PHYSICS_CFG}),
        (hybrid.predict, {"target": "csi", "physics": PHYSICS_CFG, "day_of_year": True}),
    ],
)
def test_forecast_is_reproducible_bounded_and_zero_at_night(predict, model):
    frame = make_frame()
    split = _split(frame)
    cfg = _cfg(rolling_days=3, **model)
    first = predict(frame, split, cfg)
    second = predict(frame, split, cfg)
    pd.testing.assert_series_equal(first, second)
    limit = frame.loc[split.train_index, "ac_power_kw"].max()
    assert first.index.equals(split.test_index)
    assert first.between(0.0, limit).all()
    assert (first[~frame.loc[split.test_index, "is_daytime"]] == 0.0).all()
    info = first.attrs["info"]
    assert info["rounds"] > 0 and info["train_hours"] > 0
    assert sum(v for k, v in info.items() if k.startswith("imp_")) == pytest.approx(1.0)


def test_targets_round_trip_to_power():
    frame = make_frame()
    ac = frame["ac_power_kw"]
    baseline = ac * 0.7
    denominator = pd.Series(500.0, index=frame.index)
    for target in tabular.TARGETS:
        cfg = _cfg(target=target)
        y = tabular.make_target(ac, baseline, denominator, cfg)
        back = tabular.to_power(y.to_numpy(), baseline, denominator, cfg)
        np.testing.assert_allclose(back, ac)


def test_flagged_hours_are_not_training_labels():
    frame = make_frame()
    split = _split(frame)
    cfg = _cfg(target="ac")
    base = gbm.predict(frame, split, cfg)
    frame.loc[split.train_index[::3], "flag_bad"] = True
    assert (
        gbm.predict(frame, split, cfg).attrs["info"]["train_hours"]
        < base.attrs["info"]["train_hours"]
    )


def test_config_errors():
    frame = make_frame()
    split = _split(frame)
    with pytest.raises(ValueError, match="pure ML"):
        gbm.predict(frame, split, _cfg(physics=PHYSICS_CFG))
    with pytest.raises(ValueError, match="needs a `physics`"):
        hybrid.predict(frame, split, _cfg(target="residual"))
    with pytest.raises(ValueError, match="unknown target"):
        hybrid.predict(frame, split, _cfg(target="other", physics=PHYSICS_CFG))
    with pytest.raises(ValueError, match="unknown learner"):
        hybrid.predict(frame, split, _cfg(learner="catboost", physics=PHYSICS_CFG))
