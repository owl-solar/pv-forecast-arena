import numpy as np
import pandas as pd
import pytest

from pv_forecast_arena.data.loading import load_dataset
from pv_forecast_arena.evaluation import evaluate, registry

CFG = {
    "site": {"ac_capacity_kw": 100.0, "local_timezone": "America/Los_Angeles"},
    "evaluation": {"holdout_start": "2025-01-01"},
}


def make_frame(days=20, seed=0):
    index = pd.date_range("2024-06-01 01:00", periods=24 * days, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    hour = index.tz_convert(CFG["site"]["local_timezone"]).hour
    daytime = (hour >= 8) & (hour <= 17)
    power = np.where(daytime, 60 + 20 * rng.random(len(index)), 0.0)
    return pd.DataFrame(
        {
            "ac_power_kw": power,
            "is_daytime": daytime,
            "flag_bad": False,
        },
        index=index,
    )


def test_holdout_rows_are_not_loaded_by_default(tmp_path):
    index = pd.date_range("2024-12-31 22:00", periods=6, freq="1h", tz="UTC")
    pd.DataFrame({"x": range(6)}, index=index).to_parquet(tmp_path / "dataset.parquet")
    cfg = {**CFG, "data": {"processed_dir": str(tmp_path)}}
    assert load_dataset(cfg).index.max() < pd.Timestamp("2025-01-01", tz="UTC")
    assert len(load_dataset(cfg, holdout=True)) == 6


def test_mask_is_common_to_model_and_reference_and_daytime_only():
    frame = make_frame()
    forecast = frame["ac_power_kw"].copy()
    reference = frame["ac_power_kw"].copy()
    gap_a = frame.index[frame["is_daytime"]][3]
    gap_b = frame.index[frame["is_daytime"]][7]
    forecast[gap_a] = np.nan
    reference[gap_b] = np.nan
    mask = evaluate.evaluation_mask(frame, forecast, reference, "all")
    assert not mask[gap_a] and not mask[gap_b]
    assert mask.sum() == frame["is_daytime"].sum() - 2
    assert not mask[~frame["is_daytime"]].any()


def test_clean_variant_drops_flagged_hours():
    frame = make_frame()
    bad = frame.index[frame["is_daytime"]][5]
    frame.loc[bad, "flag_bad"] = True
    forecast = frame["ac_power_kw"]
    assert evaluate.evaluation_mask(frame, forecast, forecast, "all")[bad]
    assert not evaluate.evaluation_mask(frame, forecast, forecast, "clean")[bad]
    with pytest.raises(ValueError):
        evaluate.evaluation_mask(frame, forecast, forecast, "other")


def test_perfect_forecast_has_skill_one_and_zero_errors():
    frame = make_frame()
    reference = frame["ac_power_kw"] * 0.8
    result = evaluate.score(frame, frame["ac_power_kw"], reference, CFG)
    assert result["nrmse"] == 0.0 and result["mbe"] == 0.0
    assert result["skill"] == pytest.approx(1.0)
    assert result["n_hours"] == frame["is_daytime"].sum()


def test_reference_against_itself_has_zero_skill_and_zero_width_interval():
    frame = make_frame()
    reference = frame["ac_power_kw"] * 0.8
    result = evaluate.score(frame, reference, reference, CFG)
    assert result["skill"] == 0.0
    assert result["skill_lo"] == 0.0 and result["skill_hi"] == 0.0


def test_skill_interval_contains_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    days = np.repeat(np.arange(40), 8)
    err_ref = rng.normal(0, 10, len(days))
    err_model = err_ref * 0.6 + rng.normal(0, 1, len(days))
    low, high = evaluate.skill_interval(err_model, err_ref, days)
    point = 1 - np.sqrt((err_model**2).mean()) / np.sqrt((err_ref**2).mean())
    assert low < point < high and high - low > 0
    assert (low, high) == evaluate.skill_interval(err_model, err_ref, days)


def test_registry_appends_and_keeps_all_columns(tmp_path):
    path = tmp_path / "experiments.csv"
    registry.append([{"run_id": "a", "nrmse": 0.1}], path)
    table = registry.append([{"run_id": "b", "nrmse": 0.2, "extra": 1}], path)
    assert list(table["run_id"]) == ["a", "b"]
    assert set(table.columns) == {"run_id", "nrmse", "extra"}
    assert registry.config_hash({"a": 1}) == registry.config_hash({"a": 1})
    assert registry.config_hash({"a": 1}) != registry.config_hash({"a": 2})
