"""Shared machinery of the gradient-boosting models: features, targets, fitting.

Used by `gbm` (pure ML) and `hybrid` (physics + ML). Every feature is known at the issue time
(10:00 local D-1):
- geometry: sun elevation and azimuth, angle of incidence on the plane, clear-sky GHI and POA;
- weather: the main-lead `fc_*` forecasts (`oracle_*` in oracle runs) and values derived from them
  only: clear-sky index of the forecast, direct fraction, GHI of the neighbouring hours, daily
  clear-sky index and mean daytime cloud cover of the target local day (all of day D is issued at
  lead 2 before the issue time);
- optional day of year (sin/cos);
- physics (hybrids): the physics forecast with and without the inverter limit;
- past (hybrids): rolling k of the physics model against the forecast and the smart-persistence
  clear-sky ratio, both as of the issue time.

Targets, all scale-free so that trees trained in spring are not asked to extrapolate to summer peaks:
- `ac`: AC / capacity;
- `residual`: (AC - physics) / capacity, forecast = physics + prediction (needs physics);
- `csi`: AC / clear-sky power. With physics the clear-sky power is the physics chain driven by
  clear-sky GHI; without it, capacity * clear-sky POA / 1000.

Hyper-parameters are fixed (not tuned on the development folds). The number of trees comes from early
stopping on an inner validation window: the last `val_days` of the training window, separated from
the rest by an embargo. The model is then refit on the whole training window with that many trees.
Training uses daytime hours with a measured target that are not flagged bad; flags are never inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from pv_forecast_arena.evaluation.splits import Split
from pv_forecast_arena.features import select_features, select_past
from pv_forecast_arena.models import baselines, physics

LEARNERS = ("lightgbm", "xgboost")
TARGETS = ("ac", "residual", "csi")

LGBM_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "verbose": -1,
}
# The same model in XGBoost terms: leaf-wise growth, min_child_weight = rows per leaf for L2 loss.
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eta": 0.03,
    "max_leaves": 15,
    "grow_policy": "lossguide",
    "tree_method": "hist",
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "nthread": 4,
}
MAX_ROUNDS = 2000
EARLY_STOPPING = 100
VAL_DAYS = 30
INNER_EMBARGO = pd.Timedelta("2D")
FALLBACK_ROUNDS = 200  # when the training window is too short for an inner validation window
MIN_ROWS = 20
MIN_CLEARSKY_FRACTION = 0.02  # csi target only on hours with clear-sky power above this share


def _local_day(index: pd.DatetimeIndex, cfg: dict) -> pd.DatetimeIndex:
    """Local calendar day of the start of each hour-ending stamp (the day the issue time targets)."""
    start = (index - pd.Timedelta("1h")).tz_convert(cfg["site"]["local_timezone"])
    return start.tz_localize(None).normalize()


def weather_features(frame: pd.DataFrame, cfg: dict, day_of_year: bool = False) -> pd.DataFrame:
    """Geometry and forecast features over the whole frame (no measurements of any kind)."""
    oracle = bool(cfg.get("oracle", False))
    prefix = "oracle_" if oracle else "fc_"
    geo_cols = ["sun_elevation", "clearsky_ghi", "clearsky_poa"]
    weather_cols = [prefix + v for v in cfg["weather"]["variables"]]
    base = select_features(frame, geo_cols + weather_cols, cfg, oracle=oracle)
    w = base.rename(columns=lambda c: c.removeprefix(prefix))

    site = cfg["site"]
    geo = physics.geometry(frame.index, site)
    aoi = pvlib.irradiance.aoi(
        site["tilt_deg"], site["azimuth_deg"], geo["apparent_zenith"], geo["azimuth"]
    )
    out = pd.DataFrame(index=frame.index)
    out["sun_elevation"] = w["sun_elevation"]
    out["sun_azimuth"] = geo["azimuth"]
    out["aoi"] = aoi
    out["clearsky_ghi"] = w["clearsky_ghi"]
    out["clearsky_poa"] = w["clearsky_poa"]
    for v in cfg["weather"]["variables"]:
        out[v] = w[v]

    ghi = w["shortwave_radiation"]
    out["kt"] = (ghi / w["clearsky_ghi"]).where(w["clearsky_ghi"] > 20.0).clip(0.0, 1.5)
    out["direct_fraction"] = (w["direct_radiation"] / ghi).where(ghi > 10.0).clip(0.0, 1.0)
    out["ghi_prev"] = ghi.shift(1, freq="1h").reindex(frame.index)
    out["ghi_next"] = ghi.shift(-1, freq="1h").reindex(frame.index)

    day = _local_day(frame.index, cfg)
    daytime = w["sun_elevation"] > cfg["cleaning"]["daytime_elevation_deg"]
    day_ghi = ghi.groupby(day).transform("sum")
    day_cs = w["clearsky_ghi"].groupby(day).transform("sum")
    out["day_kt"] = (day_ghi / day_cs).where(day_cs > 0)
    out["day_cloud_cover"] = w["cloud_cover"].where(daytime).groupby(day).transform("mean")

    if day_of_year:
        doy = day.day_of_year.to_numpy()
        out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def physics_clearsky(frame: pd.DataFrame, cfg: dict, model_cfg: dict) -> pd.Series:
    """Unlimited AC of the physics chain driven by clear-sky GHI (Erbs; air temp and wind as forecast)."""
    weather = physics.weather_inputs(frame, cfg)
    clear = weather.assign(
        shortwave_radiation=select_features(frame, ["clearsky_ghi"], cfg)["clearsky_ghi"]
    )
    return physics.unlimited_ac(clear, cfg, "erbs", model_cfg["transposition"])


def design(frame: pd.DataFrame, split: Split, cfg: dict) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Feature matrix over the whole frame, the clear-sky/physics pieces and the inverter limit.

    Returns (features, baseline, extras): `baseline` is what the target is relative to (0 for `ac`
    and `csi`, the physics forecast for `residual`); `extras` holds `denominator` (csi) and `limit`.
    """
    m = cfg["model"]
    target = m.get("target", "ac")
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; known: {TARGETS}")
    use_physics = "physics" in m
    if target == "residual" and not use_physics:
        raise ValueError("target 'residual' needs a `physics` section")
    if m.get("use_past", False) and not use_physics:
        raise ValueError("past features (rolling k) need a `physics` section")

    capacity = cfg["site"]["ac_capacity_kw"]
    x = weather_features(frame, cfg, day_of_year=bool(m.get("day_of_year", False)))
    ac = select_past(frame, ["ac_power_kw"])["ac_power_kw"]
    limit = float(ac.reindex(split.train_index).max())
    baseline = pd.Series(0.0, index=frame.index)
    denominator = capacity * x["clearsky_poa"] / 1000.0

    if use_physics:
        parts = physics.components(frame, split, cfg, m["physics"])
        x["phys_unlimited"] = parts["k"] * parts["unlimited"] / capacity
        x["phys_ac"] = parts["forecast"] / capacity
        if target == "residual":
            baseline = parts["forecast"]
        if target == "csi":
            denominator = parts["k_static"] * physics_clearsky(frame, cfg, m["physics"])
        if m.get("use_past", False):
            daytime = frame["is_daytime"]
            k_roll = physics.rolling_scale(
                ac.where(daytime), parts["unlimited"], limit, cfg, int(m.get("rolling_days", 30))
            )
            x["k_rolling"] = k_roll / parts["k_static"]
            x["clearsky_ratio"] = baselines.clearsky_ratio(frame, cfg)
    return x, baseline, {"denominator": denominator, "limit": limit}


def make_target(ac: pd.Series, baseline: pd.Series, denominator: pd.Series, cfg: dict) -> pd.Series:
    capacity = cfg["site"]["ac_capacity_kw"]
    target = cfg["model"].get("target", "ac")
    if target == "ac":
        return ac / capacity
    if target == "residual":
        return (ac - baseline) / capacity
    return (ac / denominator).where(denominator > MIN_CLEARSKY_FRACTION * capacity)


def to_power(pred: np.ndarray, baseline: pd.Series, denominator: pd.Series, cfg: dict) -> pd.Series:
    capacity = cfg["site"]["ac_capacity_kw"]
    target = cfg["model"].get("target", "ac")
    if target == "ac":
        return pd.Series(pred * capacity, index=baseline.index)
    if target == "residual":
        return baseline + pred * capacity
    return pd.Series(pred, index=baseline.index) * denominator


class _Booster:
    """Thin wrapper so that LightGBM and XGBoost look the same to `fit_predict`."""

    def __init__(self, learner: str, params: dict, seed: int):
        if learner not in LEARNERS:
            raise ValueError(f"unknown learner {learner!r}; known: {LEARNERS}")
        self.learner = learner
        base = LGBM_PARAMS if learner == "lightgbm" else XGB_PARAMS
        self.params = {**base, **params, "seed": seed}

    def best_rounds(self, x_fit, y_fit, x_val, y_val) -> int:
        if self.learner == "lightgbm":
            import lightgbm as lgb

            fit = lgb.Dataset(x_fit, y_fit)
            booster = lgb.train(
                self.params,
                fit,
                MAX_ROUNDS,
                valid_sets=[lgb.Dataset(x_val, y_val, reference=fit)],
                callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
            )
            return int(booster.best_iteration)
        import xgboost as xgb

        booster = xgb.train(
            self.params,
            xgb.DMatrix(x_fit, y_fit),
            MAX_ROUNDS,
            evals=[(xgb.DMatrix(x_val, y_val), "val")],
            early_stopping_rounds=EARLY_STOPPING,
            verbose_eval=False,
        )
        return int(booster.best_iteration) + 1

    def fit(self, x, y, rounds: int) -> None:
        if self.learner == "lightgbm":
            import lightgbm as lgb

            self.model = lgb.train(self.params, lgb.Dataset(x, y), rounds)
        else:
            import xgboost as xgb

            self.model = xgb.train(self.params, xgb.DMatrix(x, y), rounds)

    def predict(self, x) -> np.ndarray:
        if self.learner == "lightgbm":
            return self.model.predict(x)
        import xgboost as xgb

        return self.model.predict(xgb.DMatrix(x))

    def importance(self, columns) -> dict:
        if self.learner == "lightgbm":
            gain = dict(zip(columns, self.model.feature_importance("gain")))
        else:
            gain = self.model.get_score(importance_type="total_gain")
        total = sum(gain.values()) or 1.0
        return {c: float(gain.get(c, 0.0)) / total for c in columns}


def fit_predict(frame: pd.DataFrame, split: Split, cfg: dict, name: str) -> pd.Series:
    """Forecast for `split.test_index`; fold info (trees, feature importance) goes to `.attrs`."""
    m = cfg["model"]
    x, baseline, extras = design(frame, split, cfg)
    ac = select_past(frame, ["ac_power_kw"])["ac_power_kw"]
    y = make_target(ac, baseline, extras["denominator"], cfg)

    train = split.train_index
    # Flags only select clean training labels (all in the past of the test window); never inputs.
    usable = frame["is_daytime"] & ~frame["flag_bad"] & y.notna() & np.isfinite(y)
    rows = train[usable.reindex(train).to_numpy()]
    if len(rows) < int(m.get("min_train_hours", MIN_ROWS)):
        raise ValueError(f"only {len(rows)} usable training hours")

    val_start = rows.max() - pd.Timedelta(days=int(m.get("val_days", VAL_DAYS)))
    fit_rows = rows[rows < val_start - INNER_EMBARGO]
    val_rows = rows[rows >= val_start]
    booster = _Booster(m.get("learner", "lightgbm"), m.get("params", {}), int(cfg.get("seed", 42)))
    if len(fit_rows) >= MIN_ROWS and len(val_rows) >= MIN_ROWS:
        rounds = booster.best_rounds(x.loc[fit_rows], y[fit_rows], x.loc[val_rows], y[val_rows])
    else:
        rounds = FALLBACK_ROUNDS
    booster.fit(x.loc[rows], y[rows], rounds)

    test = split.test_index
    pred = to_power(booster.predict(x.loc[test]), baseline[test], extras["denominator"][test], cfg)
    pred = pred.clip(lower=0.0, upper=extras["limit"]).where(frame.loc[test, "is_daytime"], 0.0)
    pred = pred.rename(name)
    pred.attrs["info"] = {
        "test_start": str(split.test_start),
        "train_hours": len(rows),
        "rounds": rounds,
        **{f"imp_{c}": v for c, v in booster.importance(list(x.columns)).items()},
    }
    return pred
