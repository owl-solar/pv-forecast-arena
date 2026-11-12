import numpy as np
import pandas as pd
import pytest
from test_no_leakage import CFG, PAST_ONLY, TZ, make_frame

pytest.importorskip("darts")

from pv_forecast_arena.evaluation.splits import Split, issue_time
from pv_forecast_arena.models import dl

MODEL = {
    "input_hours": 24,
    "hidden": 4,
    "heads": 1,
    "epochs": 1,
    "batch_size": 64,
    "val_days": 3,
    "patience": 1,
}


def _cfg(**extra):
    return {**CFG, "model": {**MODEL, **extra}, "quantile": {"levels": [0.1, 0.5, 0.9]}}


def _split(frame, train_days=12):
    # Two days of embargo between the training window and the test window, as in the real splits.
    return Split(frame.index[: 24 * train_days], frame.index[24 * (train_days + 2) :])


def test_history_up_to_the_issue_time_ignores_everything_after_it():
    # The network sees the target history up to the issue time and the covariates, nothing else.
    frame = make_frame(days=20)
    split = _split(frame)
    base = dl.build_inputs(frame, split, _cfg())
    targets = frame.index[24 * 14 :: 7]
    for target, when in zip(targets, issue_time(targets, TZ, 10)):
        broken = frame.copy()
        future = broken.index > when
        for column in PAST_ONLY:
            broken.loc[future, column] = np.random.default_rng(0).normal(0, 1e4, future.sum())
        inputs = dl.build_inputs(broken, split, _cfg())
        pd.testing.assert_series_equal(inputs.target.loc[:when], base.target.loc[:when])
        pd.testing.assert_frame_equal(inputs.covariates, base.covariates)
        assert inputs.limit == base.limit


def test_the_history_test_can_fail():
    frame = make_frame(days=20)
    split = _split(frame)
    base = dl.build_inputs(frame, split, _cfg())
    when = frame.index[24 * 15]
    broken = frame.copy()
    broken.loc[when, "ac_power_kw"] += 300.0
    inputs = dl.build_inputs(broken, split, _cfg())
    assert not inputs.target.loc[:when].equals(base.target.loc[:when])


def test_missing_hours_are_filled_from_the_past_and_night_is_zero():
    frame = make_frame(days=20)
    hole = frame.index[24 * 8 + 12]
    frame.loc[hole, "ac_power_kw"] = np.nan
    inputs = dl.build_inputs(frame, _split(frame), _cfg())
    assert inputs.target.notna().all()
    assert inputs.target[hole] == inputs.target[hole - pd.Timedelta("1h")]
    assert (inputs.target[~frame["is_daytime"]] == 0.0).all()


@pytest.mark.parametrize("kind", ["lstm", "tft"])
def test_smoke_forecasts_are_bounded_and_zero_at_night(kind):
    frame = make_frame(days=20)
    split = _split(frame)
    if kind == "lstm":
        out = dl.predict_lstm(frame, split, _cfg())
        values = out.to_frame()
    else:
        values = dl.predict_tft_quantiles(frame, split, _cfg())
        assert list(values.columns) == ["q10", "q50", "q90"]
        assert (values.diff(axis=1).iloc[:, 1:] >= 0).all().all()
    assert values.notna().all().all()
    assert (values >= 0).all().all()
    night = ~frame.loc[split.test_index, "is_daytime"].to_numpy()
    assert (values.to_numpy()[night] == 0).all()
