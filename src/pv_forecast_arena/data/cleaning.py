"""Clean the hourly target, flag bad hours and build data/processed/dataset.parquet.

Flags are columns, rows are never dropped, so each experiment can decide what to exclude:
- flag_missing:  no valid target value for the hour.
- flag_stuck:    daytime output repeated unchanged for `stuck_window_hours` hours (pvanalytics).
- flag_outage:   the on-site POA sensor sees sun but the plant delivers (almost) nothing.
- flag_clipping: power at the plateau of the daily profile (pvanalytics); kept in the data, informative only.
- night_zeroed:  night hours with non-zero output that were set to 0 (counted, not a quality flag).
`flag_bad` = missing | stuck | outage; metrics on "clean" hours use ~flag_bad.
Columns prefixed `obs_` are on-site measurements and `oracle_` columns are observed weather: both are
for QA, persistence baselines built from the past, and oracle runs only, never future inputs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pvlib
from pvanalytics.features import clipping
from pvanalytics.quality import gaps

from pv_forecast_arena.data import pvdaq, weather

OUTAGE_POA_WM2 = 150  # sensor sees real sun ...
OUTAGE_POWER_FRACTION = 0.02  # ... but output is below 2% of capacity


def solar_geometry(index: pd.DatetimeIndex, latitude: float, longitude: float) -> pd.DataFrame:
    """Sun position and clear-sky GHI at the middle of each hour-ending interval."""
    mid = index - pd.Timedelta("30min")
    location = pvlib.location.Location(latitude, longitude, tz="UTC")
    position = location.get_solarposition(mid)
    clearsky = location.get_clearsky(mid, model="simplified_solis")
    return pd.DataFrame(
        {
            "sun_elevation": position["apparent_elevation"].to_numpy(),
            "clearsky_ghi": clearsky["ghi"].to_numpy(),
        },
        index=index,
    )


def clean_target(
    ac_kw: pd.Series, poa: pd.Series, geometry: pd.DataFrame, cfg: dict
) -> pd.DataFrame:
    """Return the cleaned target and its flags (all aligned to `ac_kw.index`)."""
    capacity = cfg["site"]["ac_capacity_kw"]
    c = cfg["cleaning"]
    elevation = geometry["sun_elevation"]
    daytime = elevation > c["daytime_elevation_deg"]

    ac = ac_kw.copy()
    negative = ac < 0
    ac[negative] = 0.0
    night = elevation < 0
    night_zeroed = night & (ac.abs() > 0)
    ac[night] = 0.0

    missing = ac.isna()
    stale = gaps.stale_values_round(ac, window=c["stuck_window_hours"], decimals=1, mark="all")
    stuck = stale & daytime & (ac > 0)
    outage = daytime & (poa > OUTAGE_POA_WM2) & (ac < OUTAGE_POWER_FRACTION * capacity)

    valid = ac.where(~missing)
    clip = clipping.threshold(valid.fillna(0.0), freq="1h", power_quantile=c["clipping_quantile"])
    return pd.DataFrame(
        {
            "ac_power_kw": ac,
            "flag_missing": missing,
            "flag_stuck": stuck,
            "flag_outage": outage,
            "flag_clipping": clip & ~missing,
            "night_zeroed": night_zeroed,
            "negative_fixed": negative,
        }
    )


def build_dataset(cfg: dict) -> pd.DataFrame:
    """Join target, flags, solar geometry, archived forecasts and oracle weather on one hourly UTC index."""
    meter = pvdaq.load_hourly(cfg, "meter")
    poa = pvdaq.load_hourly(cfg, "irradiance")
    start = pd.Timestamp(cfg["data"]["start"], tz="UTC")
    end = pd.Timestamp(cfg["data"]["end"], tz="UTC") + pd.Timedelta("1D")
    index = pd.date_range(start + pd.Timedelta("1h"), end, freq="1h", name="valid_time")
    meter, poa = meter.reindex(index), poa.reindex(index)

    geometry = solar_geometry(index, cfg["site"]["latitude"], cfg["site"]["longitude"])
    frame = clean_target(meter, poa, geometry, cfg)
    frame["obs_poa_wm2"] = poa
    frame = frame.join(geometry)
    frame["is_daytime"] = frame["sun_elevation"] > cfg["cleaning"]["daytime_elevation_deg"]
    frame["flag_bad"] = frame[["flag_missing", "flag_stuck", "flag_outage"]].any(axis=1)

    frame = frame.join(weather.load_forecasts(cfg).add_prefix("fc_"))
    for lead in cfg["weather"].get("extra_lead_days", []):
        frame = frame.join(weather.load_forecasts(cfg, lead).add_prefix(f"fc_lead{lead}_"))
    return frame.join(weather.load_oracle(cfg))


def save_dataset(frame: pd.DataFrame, cfg: dict) -> Path:
    path = Path(cfg["data"]["processed_dir"]) / "dataset.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return path


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Row counts and share of flagged hours, for all rows and for daytime rows."""
    flags = [c for c in frame.columns if c.startswith("flag_")] + ["night_zeroed", "negative_fixed"]
    rows = {"all hours": frame, "daytime hours": frame[frame["is_daytime"]]}
    return pd.DataFrame(
        {
            name: {"rows": len(f), **{c: round(100 * f[c].mean(), 2) for c in flags}}
            for name, f in rows.items()
        }
    )
