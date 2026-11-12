"""Gradient boosting for horizons 1-6 h, one fold at a time: `predict_all(frame, split, cfg)`.

A row is a target hour tau forecast from the issue time tau - h. Features, all known at that time:
- geometry at tau (sun elevation/azimuth, angle of incidence, clear-sky GHI and POA);
- NWP (`model.use_nwp`, default true): the lead-1 forecast of tau and of its neighbouring hours
  (`features.lead1_forecast`), the physics chain on it, and the day-ahead (lead 2) clear-sky index of
  tau and of its local day, which was issued before the day started;
- the newest measurements (`features.as_of`, i.e. stamped at most `h + data_delay_hours` before tau):
  clear-sky ratio over the last 1 h and 3 h, the last AC and its change, and, with NWP, the plant's
  ratio to the physics chain over the last 1 h and 3 h and the faded nowcast of `intraday_physics`;
- the horizon h (single model; dropped when there is one model per horizon).
Hours without a known daytime measurement (mornings) keep NaN in the measurement features, which
trees handle natively, so every daytime hour gets a forecast.

Targets (scale-free, as in `tabular`): `residual` = (AC - baseline) / capacity, forecast = baseline +
prediction, baseline = physics on lead-1 (`model.baseline: physics`, default) or the nowcast;
`csi` = AC / (capacity * clear-sky POA / 1000). Fixed hyper-parameters, number of trees by early
stopping on the last `val_days` of the training window (2-day embargo), then refit: see `tabular`.
Training rows are daytime hours with a measured, non-flagged target (flags select labels only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pv_forecast_arena import features
from pv_forecast_arena.evaluation.splits import Split
from pv_forecast_arena.features import select_past
from pv_forecast_arena.models import intraday_baselines, intraday_physics, tabular

TARGETS = ("residual", "csi")
GEOMETRY = ["sun_elevation", "sun_azimuth", "aoi", "clearsky_ghi", "clearsky_poa"]


def static_design(frame: pd.DataFrame, split: Split, cfg: dict) -> tuple[pd.DataFrame, dict | None]:
    """Features that do not depend on the horizon, and the physics pieces (None without NWP)."""
    m = cfg["model"]
    day_ahead = tabular.weather_features(
        frame, cfg
    )  # lead-2 forecast (known before the day starts)
    x = day_ahead[GEOMETRY].copy()
    if not m.get("use_nwp", True):
        return x, None

    cap = cfg["site"]["ac_capacity_kw"]
    x["kt_ahead"] = day_ahead["kt"]
    x["day_kt_ahead"] = day_ahead["day_kt"]
    x["day_cloud_cover_ahead"] = day_ahead["day_cloud_cover"]
    # The lead-1 forecast of a target hour is the same at every horizon <= 6: check the strictest.
    lead1 = features.lead1_forecast(
        frame, cfg["weather"]["variables"], cfg, features.MAX_INTRADAY_HORIZON
    )
    ghi = lead1["shortwave_radiation"]
    clear_ghi = day_ahead["clearsky_ghi"]
    x["kt"] = (ghi / clear_ghi).where(clear_ghi > 20.0).clip(0.0, 1.5)
    x["direct_fraction"] = (lead1["direct_radiation"] / ghi).where(ghi > 10.0).clip(0.0, 1.0)
    x["cloud_cover"] = lead1["cloud_cover"]
    x["temperature_2m"] = lead1["temperature_2m"]
    x["wind_speed_10m"] = lead1["wind_speed_10m"]
    x["ghi_prev"] = ghi.shift(1, freq="1h").reindex(frame.index)
    x["ghi_next"] = ghi.shift(-1, freq="1h").reindex(frame.index)

    parts = intraday_physics.components(frame, split, cfg, features.MAX_INTRADAY_HORIZON)
    x["phys_unlimited"] = parts["k"] * parts["unlimited"] / cap
    x["phys_ac"] = parts["forecast"] / cap
    return x, parts


def horizon_design(
    frame: pd.DataFrame, static: pd.DataFrame, parts: dict | None, cfg: dict, horizon: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Feature matrix of horizon h over the whole frame, with the target baseline and denominator."""
    m = cfg["model"]
    target = m.get("target", "residual")
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; known: {TARGETS}")
    if target == "residual" and parts is None:
        raise ValueError("target 'residual' needs NWP (physics baseline)")
    cap = cfg["site"]["ac_capacity_kw"]
    delay = features.delay_hours(cfg)
    ac = select_past(frame, ["ac_power_kw"])["ac_power_kw"]
    daytime = frame["is_daytime"]

    x = static.copy()
    x["csi_1h"] = features.as_of(intraday_baselines.clearsky_ratio_window(frame, 1), horizon, delay)
    x["csi_3h"] = features.as_of(intraday_baselines.clearsky_ratio_window(frame, 3), horizon, delay)
    last = ac.where(daytime)
    x["ac_last"] = features.as_of(last, horizon, delay) / cap
    x["ac_change"] = (
        features.as_of(last - last.shift(1, freq="1h").reindex(ac.index), horizon, delay) / cap
    )

    baseline = pd.Series(0.0, index=frame.index)
    if parts is not None:
        scaled = parts["k"] * parts["unlimited"]
        for window in (1, 3):
            ratio = intraday_physics.recent_ratio(frame, scaled, parts["limit"], window)
            x[f"ratio_{window}h"] = features.as_of(ratio, horizon, delay)
        nowcast = intraday_physics.nowcast_forecast(
            frame,
            parts,
            cfg,
            horizon,
            int(m.get("window_hours", 1)),
            float(m.get("decay_hours", 3.0)),
        )
        x["phys_nowcast"] = nowcast / cap
        baseline = nowcast if m.get("baseline", "physics") == "nowcast" else parts["forecast"]
    x["h"] = horizon
    denominator = cap * static["clearsky_poa"] / 1000.0
    return x, baseline, denominator


def _target(ac: pd.Series, baseline: pd.Series, denominator: pd.Series, cfg: dict) -> pd.Series:
    cap = cfg["site"]["ac_capacity_kw"]
    if cfg["model"].get("target", "residual") == "residual":
        return (ac - baseline) / cap
    return (ac / denominator).where(denominator > tabular.MIN_CLEARSKY_FRACTION * cap)


def _power(pred: np.ndarray, baseline: pd.Series, denominator: pd.Series, cfg: dict) -> pd.Series:
    if cfg["model"].get("target", "residual") == "residual":
        return baseline + pred * cfg["site"]["ac_capacity_kw"]
    return pd.Series(pred, index=baseline.index) * denominator


def predict_all(frame: pd.DataFrame, split: Split, cfg: dict) -> dict[int, pd.Series]:
    """Forecasts of `split.test_index` for every horizon; fold info goes to each series' `.attrs`."""
    m = cfg["model"]
    horizons = list(cfg["intraday"]["horizons"])
    static, parts = static_design(frame, split, cfg)
    ac = select_past(frame, ["ac_power_kw"])["ac_power_kw"]
    train = split.train_index
    limit = float(ac.reindex(train).max())
    designs = {h: horizon_design(frame, static, parts, cfg, h) for h in horizons}

    usable = frame["is_daytime"] & ~frame["flag_bad"]
    rows, targets = {}, {}
    for h, (_, baseline, denominator) in designs.items():
        y = _target(ac, baseline, denominator, cfg)
        ok = usable & y.notna() & np.isfinite(y)
        rows[h] = train[ok.reindex(train).to_numpy()]
        targets[h] = y

    per_horizon = bool(m.get("per_horizon", False))
    groups = [[h] for h in horizons] if per_horizon else [horizons]
    out: dict[int, pd.Series] = {}
    for group in groups:
        x_parts = [designs[h][0].loc[rows[h]] for h in group]
        x_fit_all = pd.concat(x_parts)
        if per_horizon:
            x_fit_all = x_fit_all.drop(columns="h")
        y_all = pd.concat([targets[h][rows[h]] for h in group]).to_numpy()
        times = x_fit_all.index
        if len(times) < int(m.get("min_train_hours", tabular.MIN_ROWS)):
            raise ValueError(f"only {len(times)} usable training rows")

        val_start = times.max() - pd.Timedelta(days=int(m.get("val_days", tabular.VAL_DAYS)))
        fit = np.asarray(times < val_start - tabular.INNER_EMBARGO)
        val = np.asarray(times >= val_start)
        booster = tabular._Booster(
            m.get("learner", "lightgbm"), m.get("params", {}), int(cfg.get("seed", 42))
        )
        if fit.sum() >= tabular.MIN_ROWS and val.sum() >= tabular.MIN_ROWS:
            rounds = booster.best_rounds(x_fit_all[fit], y_all[fit], x_fit_all[val], y_all[val])
        else:
            rounds = tabular.FALLBACK_ROUNDS
        booster.fit(x_fit_all, y_all, rounds)
        importance = booster.importance(list(x_fit_all.columns))

        for h in group:
            x, baseline, denominator = designs[h]
            test = split.test_index
            x_test = x.loc[test].drop(columns="h") if per_horizon else x.loc[test]
            pred = _power(booster.predict(x_test), baseline[test], denominator[test], cfg)
            pred = pred.clip(lower=0.0, upper=limit).where(frame.loc[test, "is_daytime"], 0.0)
            pred = pred.rename(f"intraday_gbm_h{h}")
            pred.attrs["info"] = {
                "test_start": str(split.test_start),
                "train_rows": len(times),
                "rounds": rounds,
                **{f"imp_{c}": v for c, v in importance.items()},
            }
            out[h] = pred
    return out
