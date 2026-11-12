"""Which dataset columns a model may see.

This is an allow-list on purpose: the dataset also holds the target, its quality flags (computed from
the target), on-site measurements and observed weather, and any new column is forbidden until it is
listed here. `fc_lead1_*` starts with `fc_` but is not leakage-free for hours after the issue time,
so forecast columns are named exactly rather than matched by prefix.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

DETERMINISTIC_FEATURES = ("sun_elevation", "clearsky_ghi", "clearsky_poa")


def allowed_features(cfg: dict, oracle: bool = False) -> list[str]:
    """Columns a model may use: geometry plus main-lead forecasts, or oracle weather if `oracle`."""
    prefix = "oracle_" if oracle else "fc_"
    return [*DETERMINISTIC_FEATURES, *(f"{prefix}{v}" for v in cfg["weather"]["variables"])]


def select_features(
    frame: pd.DataFrame, columns: Iterable[str], cfg: dict, oracle: bool = False
) -> pd.DataFrame:
    """Return `frame[columns]`, raising if any column is outside the allow-list for this run."""
    columns = list(columns)
    forbidden = sorted(set(columns) - set(allowed_features(cfg, oracle)))
    if forbidden:
        raise ValueError(f"Columns not allowed as model inputs (oracle={oracle}): {forbidden}")
    return frame[columns]


# Measurements of the target hour itself. A model may read them only to aggregate hours stamped at or
# before the issue time (rolling calibration, persistence); tests/test_no_leakage.py corrupts every
# later value of these columns and requires the forecast to stay unchanged.
PAST_ONLY_COLUMNS = ("ac_power_kw", "obs_poa_wm2")


def select_past(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return past-only measurement columns, raising for anything else."""
    columns = list(columns)
    forbidden = sorted(set(columns) - set(PAST_ONLY_COLUMNS))
    if forbidden:
        raise ValueError(f"Not a past-only measurement column: {forbidden}")
    return frame[columns]


MAX_INTRADAY_HORIZON = 6
# `fc_lead1_*` of target hour tau is the run started at tau - 24 h; assume it is published at most 6 h
# after it starts. It is known at issue time tau - h only if h <= 24 - 6.
LEAD1_PUBLICATION_DELAY_H = 6
MAX_LEAD1_HORIZON = 24 - LEAD1_PUBLICATION_DELAY_H


def delay_hours(cfg: dict) -> int:
    """Hours between the end of a measured hour and its availability to the forecaster."""
    return int(cfg.get("intraday", {}).get("data_delay_hours", 1))


def _check_horizon(horizon: int) -> None:
    if not 1 <= horizon <= MAX_INTRADAY_HORIZON:
        raise ValueError(f"intraday horizon must be 1..{MAX_INTRADAY_HORIZON}, got {horizon}")


def as_of(series: pd.Series, horizon: int, delay: int) -> pd.Series:
    """Align a backward-looking series to the target hour of an intraday forecast.

    The forecast for target hour tau is issued at tau - horizon, and a measurement stamped s is
    known from s + delay on. So the value returned at tau is the one stamped tau - horizon - delay:
    the newest the forecaster may see. `series` must itself use only stamps <= its own stamp
    (a raw measurement or a backward rolling statistic of it); tests/test_intraday.py corrupts
    every later value and requires the forecast to stay unchanged.
    """
    _check_horizon(horizon)
    if delay < 0:
        raise ValueError("delay must be >= 0")
    return series.shift(horizon + delay, freq="1h").reindex(series.index)


def lead1_forecast(
    frame: pd.DataFrame, columns: Iterable[str], cfg: dict, horizon: int
) -> pd.DataFrame:
    """Day-1 archived forecasts of the target hour, valid only for intraday horizons.

    Columns come back without the `fc_lead1_` prefix. For day-ahead use this would leak (the run is
    24 h before the target hour, after the 10:00 issue time); for horizon <= MAX_LEAD1_HORIZON it
    was published before the intraday issue time.
    """
    _check_horizon(horizon)
    if horizon > MAX_LEAD1_HORIZON:
        raise ValueError(f"lead-1 forecasts are not known {horizon} h ahead of the target")
    columns = list(columns)
    forbidden = sorted(set(columns) - set(cfg["weather"]["variables"]))
    if forbidden:
        raise ValueError(f"Not a forecast variable: {forbidden}")
    return frame[[f"fc_lead1_{c}" for c in columns]].rename(
        columns=lambda c: c.removeprefix("fc_lead1_")
    )


def select_diagnostic(frame: pd.DataFrame, columns: Iterable[str], cfg: dict) -> pd.DataFrame:
    """On-site measurements of the target hour itself, for runs marked `diagnostic: true` only.

    Such runs say how much error comes from the weather input (they are not forecasts) and must never
    be compared with real forecasts without that label.
    """
    if not cfg.get("diagnostic", False):
        raise ValueError("on-site measurements of the target hour need `diagnostic: true`")
    columns = list(columns)
    forbidden = sorted(set(columns) - {"obs_poa_wm2"})
    if forbidden:
        raise ValueError(f"Not allowed even in diagnostic runs: {forbidden}")
    return frame[columns]
