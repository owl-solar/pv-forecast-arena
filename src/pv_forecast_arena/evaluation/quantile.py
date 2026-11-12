"""Scoring of quantile forecasts and the cost of imbalance.

Everything is computed on one set of rows: daytime hours with a measured target and a forecast of
every level (variant `clean` also drops hours flagged bad). Pinball loss is normalised by capacity.
Intervals are day-block bootstraps of a ratio of summed losses (whole local days are resampled).

Imbalance cost of a bid b against the measured power y (kW, per hour):
    cost = c_short * max(b - y, 0) + c_surplus * max(y - b, 0)
`c_short` is paid per kWh promised and not delivered, `c_surplus` per kWh delivered above the bid.
The bid minimising the expected cost is the quantile q* = c_surplus / (c_short + c_surplus).
Prices are parameters of the scenario, not market data. The cost is reported per kWh of measured
energy, so that 0.05 means "5% of the delivered energy at unit price".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pv_forecast_arena.evaluation import evaluate, metrics
from pv_forecast_arena.models import quantile as models


def optimal_level(c_short: float, c_surplus: float) -> float:
    return c_surplus / (c_short + c_surplus)


def imbalance_cost(truth, bid, c_short: float, c_surplus: float) -> np.ndarray:
    """Cost of each hour (kW times unit price; one hourly stamp = one kWh per kW)."""
    diff = np.asarray(bid, dtype=float) - np.asarray(truth, dtype=float)
    return c_short * np.maximum(diff, 0.0) + c_surplus * np.maximum(-diff, 0.0)


def pinball_rows(truth, bid, q: float) -> np.ndarray:
    d = np.asarray(truth, dtype=float) - np.asarray(bid, dtype=float)
    return np.maximum(q * d, (q - 1.0) * d)


def mask(frame: pd.DataFrame, forecast: pd.DataFrame, variant: str, *others) -> pd.Series:
    """Rows where every level (and every series in `others`) exists; see `evaluate`."""
    first = forecast.iloc[:, 0]
    ok = forecast.notna().all(axis=1)
    return evaluate.evaluation_mask(frame, first, first, variant, others) & ok


def ratio_interval(
    loss_model: np.ndarray,
    loss_ref: np.ndarray,
    days: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    level: float = 0.90,
) -> tuple[float, float]:
    """Interval of 1 - sum(loss_model) / sum(loss_ref) with whole days resampled."""
    codes, uniques = pd.factorize(days)
    n = len(uniques)
    a = np.bincount(codes, weights=loss_model, minlength=n)
    b = np.bincount(codes, weights=loss_ref, minlength=n)
    picks = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    ratio = a[picks].sum(axis=1) / b[picks].sum(axis=1)
    low, high = np.quantile(1.0 - ratio, [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)


def _days(frame: pd.DataFrame, index: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    tz = cfg["site"]["local_timezone"]
    return index.tz_convert(tz).tz_localize(None).normalize().to_numpy()


def score(
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    reference: pd.DataFrame,
    cfg: dict,
    variant: str,
) -> dict:
    """Summary of `forecast` (columns = levels) against `reference` (same columns).

    `skill_pinball` = 1 - mean pinball of the model / of the reference, averaged over all levels.
    """
    cap = cfg["site"]["ac_capacity_kw"]
    cols = list(forecast.columns)
    levels = models.levels(cfg)
    ok = mask(frame, forecast, variant) & mask(frame, reference, variant).reindex(forecast.index)
    index = forecast.index[ok.to_numpy()]
    if len(index) == 0:
        raise ValueError("no hours left to evaluate")
    truth = frame.loc[index, "ac_power_kw"].to_numpy()
    days = _days(frame, index, cfg)

    loss = sum(pinball_rows(truth, forecast.loc[index, c], q) for c, q in zip(cols, levels))
    ref_loss = sum(pinball_rows(truth, reference.loc[index, c], q) for c, q in zip(cols, levels))
    lo, hi = ratio_interval(loss, ref_loss, days)
    out = {
        "variant": variant,
        "n_hours": len(index),
        "pinball": float(loss.sum() / len(levels) / len(index) / cap),
        "ref_pinball": float(ref_loss.sum() / len(levels) / len(index) / cap),
        "skill_pinball": float(1.0 - loss.sum() / ref_loss.sum()),
        "skill_pinball_lo": lo,
        "skill_pinball_hi": hi,
    }
    median = models.column(0.5)
    if median in cols:
        out["nrmse_median"] = metrics.nrmse(truth, forecast.loc[index, median], cap)
        out["mbe_median"] = metrics.mbe(truth, forecast.loc[index, median], cap)
    for low_q, high_q in ((0.1, 0.9), (0.05, 0.95)):
        a, b = models.column(low_q), models.column(high_q)
        if a in cols and b in cols:
            tag = f"{round((high_q - low_q) * 100)}"
            out[f"coverage_{tag}"] = metrics.coverage(
                truth, forecast.loc[index, a], forecast.loc[index, b]
            )
            out[f"width_{tag}"] = float(
                np.mean(forecast.loc[index, b].to_numpy() - forecast.loc[index, a].to_numpy()) / cap
            )
    return out


def per_level(
    frame: pd.DataFrame, forecast: pd.DataFrame, cfg: dict, variant: str = "all"
) -> pd.DataFrame:
    """Pinball loss and empirical frequency of y <= forecast per level (calibration table)."""
    cap = cfg["site"]["ac_capacity_kw"]
    ok = mask(frame, forecast, variant)
    index = forecast.index[ok.to_numpy()]
    truth = frame.loc[index, "ac_power_kw"].to_numpy()
    rows = []
    for column, q in zip(forecast.columns, models.levels(cfg)):
        bid = forecast.loc[index, column].to_numpy()
        rows.append(
            {
                "level": q,
                "pinball": float(pinball_rows(truth, bid, q).mean() / cap),
                "frequency": float((truth <= bid).mean()),
            }
        )
    return pd.DataFrame(rows)


def cost_table(
    frame: pd.DataFrame,
    strategies: dict[str, pd.Series],
    scenarios: dict[str, dict],
    cfg: dict,
    variant: str = "all",
    reference: str = "median",
) -> pd.DataFrame:
    """Imbalance cost per kWh of energy of every bid series under every price scenario.

    `strategies` maps a name to a bid series; rows are the hours where all of them exist. The cost
    skill and its interval are against the strategy named `reference` on the same hours.
    """
    cap_free = frame["ac_power_kw"]
    frame_index = None
    for s in strategies.values():
        frame_index = s.index if frame_index is None else frame_index.intersection(s.index)
    first = pd.Series(0.0, index=frame_index)
    ok = evaluate.evaluation_mask(
        frame, first, first, variant, tuple(s.reindex(frame_index) for s in strategies.values())
    )
    index = frame_index[ok.to_numpy()]
    truth = cap_free[index].to_numpy()
    days = _days(frame, index, cfg)
    rows = []
    for scenario, price in scenarios.items():
        c_short, c_surplus = float(price["c_short"]), float(price["c_surplus"])
        costs = {
            name: imbalance_cost(truth, s.loc[index], c_short, c_surplus)
            for name, s in strategies.items()
        }
        for name, cost in costs.items():
            lo, hi = ratio_interval(cost, costs[reference], days)
            rows.append(
                {
                    "scenario": scenario,
                    "c_short": c_short,
                    "c_surplus": c_surplus,
                    "strategy": name,
                    "variant": variant,
                    "n_hours": len(index),
                    "cost_per_kwh": float(cost.sum() / truth.sum()),
                    "saving_vs_" + reference: float(1.0 - cost.sum() / costs[reference].sum()),
                    "saving_lo": lo,
                    "saving_hi": hi,
                }
            )
    return pd.DataFrame(rows)
