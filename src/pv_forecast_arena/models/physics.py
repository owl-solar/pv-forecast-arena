"""Physical chain with pvlib: forecast GHI -> DNI/DHI -> POA -> IAM -> cell temperature -> PVWatts.

Inputs are archived forecasts (`fc_*`, or `oracle_*` in oracle runs) read through the feature allow-list,
plus deterministic sun geometry at the middle of each hour-ending interval.

Steps:
1. Decomposition: `erbs` splits GHI into DNI/DHI; `gfs_direct` uses the forecast's own direct
   (horizontal) radiation, DNI = direct / cos(zenith).
2. Transposition onto the 25 deg / 180 deg plane: `isotropic`, `haydavies` or `perez`.
3. Incidence-angle losses on the beam component (physical IAM model).
4. Cell temperature: SAPM, open rack glass/polymer; forecast wind is km/h and converted to m/s.
5. PVWatts DC at the nameplate DC capacity, a fixed system loss and inverter efficiency, then the
   inverter limit = the largest AC measured in the training window.

Calibration of a single plant-state scale factor k (power = min(k * unlimited AC, limit)):
- `raw`: k = 1.
- `static`: least squares on the training window.
- `rolling_poa`: k = sum(measured AC) / sum(model AC driven by measured POA) over the `rolling_days`
  before each issue time. It tracks plant state (soiling, availability, seasonal PR drift) separately
  from weather-forecast error. Uses past measurements only; flags are not used (flag_stuck looks ahead).
- `rolling_forecast`: the same ratio against the forecast-driven model, so it also absorbs recent
  forecast bias.
Rolling issues with too few usable hours fall back to the static k of the fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from pv_forecast_arena.evaluation.splits import Split, issue_time
from pv_forecast_arena.features import select_diagnostic, select_features, select_past

DECOMPOSITIONS = ("erbs", "gfs_direct")
TRANSPOSITIONS = ("isotropic", "haydavies", "perez")
CALIBRATIONS = ("raw", "static", "rolling_poa", "rolling_forecast")

GAMMA_PDC = -0.0037  # 1/K, typical crystalline silicon
SYSTEM_LOSS = 0.14  # PVWatts default DC losses (soiling, wiring, mismatch, ...)
ETA_INVERTER = 0.96
ALBEDO = 0.2
MAX_ZENITH_BEAM = 87.0  # beyond this cos(zenith) blows up direct/cos(zenith)
UNCLIPPED_FRACTION = 0.97  # hours above this share of the limit say nothing about k
MIN_ROLLING_HOURS = 20
WEATHER = ("shortwave_radiation", "direct_radiation", "temperature_2m", "wind_speed_10m")


def _geometry(index: pd.DatetimeIndex, site: dict) -> pd.DataFrame:
    mid = index - pd.Timedelta("30min")
    location = pvlib.location.Location(site["latitude"], site["longitude"], tz="UTC")
    position = location.get_solarposition(mid)
    position["dni_extra"] = pvlib.irradiance.get_extra_radiation(mid).to_numpy()
    position["airmass"] = pvlib.atmosphere.get_relative_airmass(
        position["apparent_zenith"]
    ).to_numpy()
    position.index = index
    return position


def _dc_from_poa(
    poa_effective: pd.Series, poa_global: pd.Series, temp_air: pd.Series, wind_kmh: pd.Series, cfg
):
    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"]
    t_cell = pvlib.temperature.sapm_cell(poa_global, temp_air, wind_kmh / 3.6, **params)
    dc = pvlib.pvsystem.pvwatts_dc(poa_effective, t_cell, cfg["site"]["dc_capacity_kw"], GAMMA_PDC)
    return (dc * (1.0 - SYSTEM_LOSS)).clip(lower=0.0)


def unlimited_ac(
    weather: pd.DataFrame, cfg: dict, decomposition: str, transposition: str
) -> pd.Series:
    """AC power (kW) before the inverter limit, from columns named like `WEATHER`."""
    if decomposition not in DECOMPOSITIONS:
        raise ValueError(f"unknown decomposition {decomposition!r}; known: {DECOMPOSITIONS}")
    if transposition not in TRANSPOSITIONS:
        raise ValueError(f"unknown transposition {transposition!r}; known: {TRANSPOSITIONS}")
    site = cfg["site"]
    geo = _geometry(weather.index, site)
    zenith = geo["apparent_zenith"]
    ghi = weather["shortwave_radiation"].clip(lower=0.0)
    if decomposition == "erbs":
        doy = (weather.index - pd.Timedelta("30min")).day_of_year.to_numpy()
        parts = pvlib.irradiance.erbs(ghi, zenith, doy)
        dni, dhi = parts["dni"], parts["dhi"]
    else:
        direct = weather["direct_radiation"].clip(lower=0.0).clip(upper=ghi)
        cos_z = np.cos(np.radians(zenith))
        dni = (direct / cos_z).where(zenith < MAX_ZENITH_BEAM, 0.0)
        dhi = ghi - direct.where(zenith < MAX_ZENITH_BEAM, 0.0)

    poa = pvlib.irradiance.get_total_irradiance(
        site["tilt_deg"],
        site["azimuth_deg"],
        zenith,
        geo["azimuth"],
        dni,
        ghi,
        dhi,
        dni_extra=geo["dni_extra"],
        airmass=geo["airmass"],
        albedo=ALBEDO,
        model=transposition,
    )
    aoi = pvlib.irradiance.aoi(site["tilt_deg"], site["azimuth_deg"], zenith, geo["azimuth"])
    iam = pvlib.iam.physical(aoi)
    beam = poa["poa_direct"].fillna(0.0)
    diffuse = poa["poa_diffuse"].fillna(0.0)
    effective = beam * iam.fillna(0.0) + diffuse
    dc = _dc_from_poa(
        effective, beam + diffuse, weather["temperature_2m"], weather["wind_speed_10m"], cfg
    )
    ac = (dc * ETA_INVERTER).where(ghi.notna())
    return ac.rename("unlimited_ac")


def unlimited_ac_from_poa(poa: pd.Series, weather: pd.DataFrame, cfg: dict) -> pd.Series:
    """AC power before the inverter limit driven by a measured POA series (no IAM: sensor sees it)."""
    poa = poa.clip(lower=0.0)
    dc = _dc_from_poa(poa, poa, weather["temperature_2m"], weather["wind_speed_10m"], cfg)
    return (dc * ETA_INVERTER).where(poa.notna()).rename("unlimited_ac_poa")


def fit_scale(ac: pd.Series, model: pd.Series, limit: float) -> float:
    """Least-squares k in ac ~ k * model over hours where both exist and the plant is not clipping."""
    ok = ac.notna() & model.notna() & (model > 0) & (ac < UNCLIPPED_FRACTION * limit)
    den = float((model[ok] ** 2).sum())
    return float((ac[ok] * model[ok]).sum() / den) if den > 0 else 1.0


def rolling_scale(ac: pd.Series, model: pd.Series, limit: float, cfg: dict, days: int) -> pd.Series:
    """k at the issue time of every hour: sum(ac) / sum(model) over (issue - days, issue].

    Only hours stamped at or before the issue time enter the sums, so the value is known when the
    forecast is issued.
    """
    ok = ac.notna() & model.notna() & (model > 0) & (ac < UNCLIPPED_FRACTION * limit)
    window = f"{days}D"
    num = ac.where(ok, 0.0).rolling(window).sum()
    den = model.where(ok, 0.0).rolling(window).sum()
    hours = ok.astype(float).rolling(window).sum()
    ratio = (num / den).where(hours >= MIN_ROLLING_HOURS)
    issued = issue_time(
        ac.index, cfg["site"]["local_timezone"], cfg["weather"]["forecast_issue_hour_local"]
    )
    return pd.Series(ratio.reindex(issued).to_numpy(), index=ac.index, name="k")


def predict(frame: pd.DataFrame, split: Split, cfg: dict) -> pd.Series:
    """Physics forecast for `split.test_index`; scale and limit are fitted on `split.train_index`."""
    m = cfg["model"]
    calibration = m.get("calibration", "raw")
    if calibration not in CALIBRATIONS:
        raise ValueError(f"unknown calibration {calibration!r}; known: {CALIBRATIONS}")
    oracle = bool(cfg.get("oracle", False))
    prefix = "oracle_" if oracle else "fc_"
    weather = select_features(frame, [prefix + v for v in WEATHER], cfg, oracle=oracle)
    weather = weather.rename(columns=lambda c: c.removeprefix(prefix))
    model = unlimited_ac(weather, cfg, m["decomposition"], m["transposition"])

    train = split.train_index
    past = select_past(frame, ["ac_power_kw", "obs_poa_wm2"])
    ac = past["ac_power_kw"]
    limit = float(ac.reindex(train).max())
    daytime = frame["is_daytime"]
    k_static = fit_scale(ac.reindex(train).where(daytime), model.reindex(train), limit)

    if calibration == "raw":
        k = pd.Series(1.0, index=frame.index)
    elif calibration == "static":
        k = pd.Series(k_static, index=frame.index)
    else:
        if calibration == "rolling_poa":
            # Measured POA is a past-only input: it enters solely through sums up to issue time.
            reference = unlimited_ac_from_poa(past["obs_poa_wm2"], weather, cfg)
        else:
            reference = model
        k = rolling_scale(ac.where(daytime), reference, limit, cfg, int(m["rolling_days"]))
        k = k.fillna(k_static)

    forecast = (k * model).clip(lower=0.0, upper=limit)
    return forecast.reindex(split.test_index).rename("physics")


def predict_measured_poa(frame: pd.DataFrame, split: Split, cfg: dict) -> pd.Series:
    """Diagnostic, not a forecast: the measured POA of the target hour drives PVWatts.

    Isolates the plant model from the weather and transposition error. Air temperature and wind
    still come from the forecast. Scale (static) and limit are fitted on `split.train_index`.
    """
    poa = select_diagnostic(frame, ["obs_poa_wm2"], cfg)["obs_poa_wm2"]
    weather = select_features(frame, ["fc_temperature_2m", "fc_wind_speed_10m"], cfg)
    weather = weather.rename(columns=lambda c: c.removeprefix("fc_"))
    model = unlimited_ac_from_poa(poa, weather, cfg)
    train = split.train_index
    ac = select_past(frame, ["ac_power_kw"])["ac_power_kw"]
    limit = float(ac.reindex(train).max())
    k = 1.0
    if cfg["model"].get("calibration", "raw") == "static":
        k = fit_scale(ac.reindex(train).where(frame["is_daytime"]), model.reindex(train), limit)
    forecast = (k * model).clip(lower=0.0, upper=limit)
    return forecast.reindex(split.test_index).rename("physics_measured_poa")
