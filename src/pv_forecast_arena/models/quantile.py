"""Day-ahead quantile forecasts: `predict(frame, split, cfg) -> DataFrame` with one column per level.

Both models forecast the residual of the physics chain, as the Stage 3 winner does, so that their
point forecast (the median) is comparable with `hybrid_lgbm_residual`.

- `quantile_empirical`: physics forecast + empirical quantiles of the training residuals (a constant
  band in units of capacity). No features, no learning: the reference every learned band must beat.
- `quantile_lgbm`: one LightGBM per level (objective `quantile`) on the same features as the hybrid
  (`tabular.design`), fixed hyper-parameters, number of trees by early stopping on the last
  `val_days` of the training window with an embargo, then refit. Training rows exclude `flag_bad`
  hours (labels only, never inputs). `model.recalibrate: true` shifts every level by the quantile of
  the validation residuals (the model is then fit on the data before the validation window only).

Levels come from `quantile.levels` of the config. Quantiles are sorted row by row afterwards, so they
never cross, and clipped to [0, inverter limit]; night hours are 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pv_forecast_arena.evaluation.splits import Split
from pv_forecast_arena.features import select_past
from pv_forecast_arena.models import physics, tabular

DEFAULT_LEVELS = (0.05, 0.1, 0.25, 1 / 3, 0.5, 2 / 3, 0.75, 0.9, 0.95)


def levels(cfg: dict) -> list[float]:
    return [float(q) for q in cfg.get("quantile", {}).get("levels", DEFAULT_LEVELS)]


def column(q: float) -> str:
    """Column name of a level, e.g. 0.1 -> 'q10', 1/3 -> 'q33.33'."""
    return f"q{round(q * 100, 2):g}"


def _finish(
    raw: dict[float, pd.Series], limit: float, daytime: pd.Series, name: str
) -> pd.DataFrame:
    out = pd.DataFrame({column(q): s for q, s in sorted(raw.items())})
    out = pd.DataFrame(np.sort(out.to_numpy(), axis=1), index=out.index, columns=out.columns)
    out = out.clip(lower=0.0, upper=limit)
    out = out.mul(daytime.reindex(out.index).to_numpy().astype(float), axis=0)
    out.attrs["name"] = name
    return out


def predict_empirical(frame: pd.DataFrame, split: Split, cfg: dict) -> pd.DataFrame:
    m = cfg["model"]
    cap = cfg["site"]["ac_capacity_kw"]
    parts = physics.components(frame, split, cfg, m["physics"])
    ac = select_past(frame, ["ac_power_kw"])["ac_power_kw"]
    resid = (ac - parts["forecast"]) / cap
    usable = frame["is_daytime"] & ~frame["flag_bad"] & resid.notna()
    rows = split.train_index[usable.reindex(split.train_index).to_numpy()]
    if len(rows) < int(m.get("min_train_hours", tabular.MIN_ROWS)):
        raise ValueError(f"only {len(rows)} usable training hours")
    test = split.test_index
    base = parts["forecast"].reindex(test)
    raw = {q: base + float(np.quantile(resid[rows], q)) * cap for q in levels(cfg)}
    limit = float(ac.reindex(split.train_index).max())
    out = _finish(raw, limit, frame.loc[test, "is_daytime"], "quantile_empirical")
    out.attrs["info"] = {"test_start": str(split.test_start), "train_hours": len(rows)}
    return out


def predict_lgbm(frame: pd.DataFrame, split: Split, cfg: dict) -> pd.DataFrame:
    m = cfg["model"]
    if m.get("learner", "lightgbm") != "lightgbm":
        raise ValueError("quantile_lgbm supports the lightgbm learner only")
    if m.get("target", "residual") != "residual":
        raise ValueError("quantile_lgbm forecasts the residual of the physics chain")
    x, baseline, extras = tabular.design(frame, split, cfg)
    ac = select_past(frame, ["ac_power_kw"])["ac_power_kw"]
    y = tabular.make_target(ac, baseline, extras["denominator"], cfg)

    train = split.train_index
    usable = frame["is_daytime"] & ~frame["flag_bad"] & y.notna() & np.isfinite(y)
    rows = train[usable.reindex(train).to_numpy()]
    if len(rows) < int(m.get("min_train_hours", tabular.MIN_ROWS)):
        raise ValueError(f"only {len(rows)} usable training hours")
    val_start = rows.max() - pd.Timedelta(days=int(m.get("val_days", tabular.VAL_DAYS)))
    fit_rows = rows[rows < val_start - tabular.INNER_EMBARGO]
    val_rows = rows[rows >= val_start]
    split_ok = len(fit_rows) >= tabular.MIN_ROWS and len(val_rows) >= tabular.MIN_ROWS

    test = split.test_index
    recalibrate = bool(m.get("recalibrate", False))
    raw, info = {}, {"test_start": str(split.test_start), "train_hours": len(rows)}
    for q in levels(cfg):
        params = {**m.get("params", {}), "objective": "quantile", "alpha": q}
        booster = tabular._Booster("lightgbm", params, int(cfg.get("seed", 42)))
        if split_ok:
            rounds = booster.best_rounds(x.loc[fit_rows], y[fit_rows], x.loc[val_rows], y[val_rows])
        else:
            rounds = tabular.FALLBACK_ROUNDS
        if recalibrate and split_ok:
            # Fit before the validation window only, then shift the level so that the share of
            # validation hours below the forecast equals q (conformalised quantile regression, one
            # constant per level). The model never saw the validation window.
            booster.fit(x.loc[fit_rows], y[fit_rows], rounds)
            shift = float(np.quantile(y[val_rows] - booster.predict(x.loc[val_rows]), q))
            pred = booster.predict(x.loc[test]) + shift
            info[f"shift_{column(q)}"] = shift
        else:
            booster.fit(x.loc[rows], y[rows], rounds)
            pred = booster.predict(x.loc[test])
        raw[q] = tabular.to_power(pred, baseline[test], None, cfg)
        info[f"rounds_{column(q)}"] = rounds
    out = _finish(raw, extras["limit"], frame.loc[test, "is_daytime"], "quantile_lgbm")
    out.attrs["info"] = info
    return out
