import numpy as np
import pandas as pd
import pytest

from pv_forecast_arena.evaluation.splits import Split
from pv_forecast_arena.models import physics

TZ = "America/Los_Angeles"
CFG = {
    "site": {
        "latitude": 38.996,
        "longitude": -122.134,
        "tilt_deg": 25,
        "azimuth_deg": 180,
        "dc_capacity_kw": 893.0,
        "ac_capacity_kw": 708.0,
        "local_timezone": TZ,
    },
    "weather": {"variables": list(physics.WEATHER), "forecast_issue_hour_local": 10},
}


def make_frame(days=40, seed=1):
    """Synthetic hours with clear-sky-like forecasts and a plant at 90% of the PVWatts output."""
    index = pd.date_range("2024-06-01 01:00", periods=24 * days, freq="1h", tz="UTC")
    geo = physics._geometry(index, CFG["site"])
    cos_z = np.cos(np.radians(geo["apparent_zenith"])).clip(lower=0.0).to_numpy()
    rng = np.random.default_rng(seed)
    ghi = 1000.0 * cos_z * (0.6 + 0.4 * rng.random(len(index)))
    frame = pd.DataFrame(
        {
            "fc_shortwave_radiation": ghi,
            "fc_direct_radiation": 0.8 * ghi,
            "fc_temperature_2m": 25.0,
            "fc_wind_speed_10m": 10.0,
            "is_daytime": geo["apparent_elevation"].to_numpy() > 5,
        },
        index=index,
    )
    weather = frame.rename(columns=lambda c: c.removeprefix("fc_"))
    model = physics.unlimited_ac(weather, CFG, "erbs", "perez")
    frame["ac_power_kw"] = (0.9 * model).clip(upper=650.0)
    frame["obs_poa_wm2"] = 0.95 * ghi
    return frame


def split_of(frame, train_days=10):
    cut = frame.index[24 * train_days]
    return Split(
        frame.index[frame.index < cut], frame.index[frame.index >= cut + pd.Timedelta("2D")]
    )


def cfg_with(**model):
    return {**CFG, "model": {"decomposition": "erbs", "transposition": "perez", **model}}


@pytest.mark.parametrize("decomposition", physics.DECOMPOSITIONS)
@pytest.mark.parametrize("transposition", physics.TRANSPOSITIONS)
def test_chain_is_zero_at_night_and_positive_at_noon(decomposition, transposition):
    frame = make_frame(days=3)
    weather = frame.rename(columns=lambda c: c.removeprefix("fc_"))
    ac = physics.unlimited_ac(weather, CFG, decomposition, transposition)
    night = ~frame["is_daytime"] & (frame["fc_shortwave_radiation"] == 0)
    assert (ac[night] == 0).all()
    assert ac[frame["is_daytime"]].max() > 300
    assert ac.notna().all()


def test_static_calibration_recovers_plant_scale():
    frame = make_frame()
    split = split_of(frame)
    raw = physics.predict(frame, split, cfg_with(calibration="raw"))
    static = physics.predict(frame, split, cfg_with(calibration="static"))
    limit = frame["ac_power_kw"].reindex(split.train_index).max()
    day = frame["is_daytime"].reindex(split.test_index) & (raw < limit)
    assert np.allclose(static[day], 0.9 * raw[day], rtol=1e-6)


@pytest.mark.parametrize("calibration", physics.CALIBRATIONS)
def test_forecast_never_exceeds_training_maximum(calibration):
    frame = make_frame()
    split = split_of(frame)
    forecast = physics.predict(frame, split, cfg_with(calibration=calibration, rolling_days=14))
    assert forecast.max() <= frame["ac_power_kw"].reindex(split.train_index).max() + 1e-9
    assert (forecast >= 0).all()


def test_unknown_options_are_rejected():
    frame = make_frame(days=12)
    with pytest.raises(ValueError, match="calibration"):
        physics.predict(frame, split_of(frame), cfg_with(calibration="magic"))
    with pytest.raises(ValueError, match="transposition"):
        physics.predict(frame, split_of(frame), cfg_with(calibration="raw", transposition="x"))


def test_oracle_run_reads_only_oracle_columns():
    frame = make_frame(days=12)
    oracle = frame.rename(columns=lambda c: c.replace("fc_", "oracle_"))
    cfg = {**cfg_with(calibration="raw"), "oracle": True}
    a = physics.predict(oracle, split_of(oracle), cfg)
    b = physics.predict(frame, split_of(frame), cfg_with(calibration="raw"))
    assert np.allclose(a, b)
