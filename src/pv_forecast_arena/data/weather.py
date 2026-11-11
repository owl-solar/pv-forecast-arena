"""Open-Meteo weather, downloaded once and cached to data/raw as parquet (UTC, hour-ending labels).

Two kinds of data, kept in separate files so they cannot be mixed by accident:
- forecasts: Previous Runs API, value issued exactly `lead` days before the valid hour. Main input.
- oracle: Archive API (observed / reanalysis weather). Only for explicitly marked oracle runs.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
USER_AGENT = "pv-forecast-arena"
RETRY_STATUS = {429, 500, 502, 503, 504}


def _get_json(url: str, params: dict, attempts: int = 10) -> dict:
    """GET with retries; Open-Meteo answers 503 'overloaded' and resets connections from time to time."""
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                url, params=params, headers={"User-Agent": USER_AGENT}, timeout=120
            )
            if response.status_code not in RETRY_STATUS:
                response.raise_for_status()
                return response.json()
        except (requests.ConnectionError, requests.Timeout):
            if attempt == attempts:
                raise
        if attempt == attempts:
            raise RuntimeError(
                f"Open-Meteo kept failing: HTTP {response.status_code} {response.text[:200]}"
            )
        time.sleep(min(10 * attempt, 60))
    raise AssertionError("unreachable")


def _to_frame(payload: dict) -> pd.DataFrame:
    hourly = payload["hourly"]
    frame = pd.DataFrame(hourly).set_index("time")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index), name="valid_time").tz_localize(
        "UTC"
    )
    return frame.astype("float64")


def cache_path(raw_dir: str | Path, kind: str, model: str, cfg: dict) -> Path:
    site = cfg["site"]
    stem = f"{kind}_{model}_{site['latitude']:.3f}_{site['longitude']:.3f}_{cfg['data']['start']}_{cfg['data']['end']}"
    return Path(raw_dir) / "openmeteo" / f"{stem}.parquet"


def _store(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_parquet(temporary)
    temporary.replace(path)
    return path


def _base_params(cfg: dict) -> dict:
    site = cfg["site"]
    return {
        "latitude": site["latitude"],
        "longitude": site["longitude"],
        "start_date": cfg["data"]["start"],
        "end_date": cfg["data"]["end"],
        "timezone": "UTC",
    }


def _fetch_chunked(url: str, params: dict, months: int = 3, pause: float = 3.0) -> pd.DataFrame:
    """Request the period in chunks of `months`: small requests are far less likely to hit rate limits."""
    start, end = pd.Timestamp(params["start_date"]), pd.Timestamp(params["end_date"])
    parts = []
    while start <= end:
        stop = min(start + pd.DateOffset(months=months) - pd.Timedelta("1D"), end)
        chunk = {
            **params,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": stop.strftime("%Y-%m-%d"),
        }
        parts.append(_to_frame(_get_json(url, chunk)))
        start = stop + pd.Timedelta("1D")
        time.sleep(pause)
    return pd.concat(parts)


def download_forecasts(cfg: dict) -> Path:
    """Archived forecasts for every configured lead, columns like `shortwave_radiation_previous_day2`."""
    w = cfg["weather"]
    path = cache_path(cfg["data"]["raw_dir"], "forecast", w["previous_runs_model"], cfg)
    if path.exists():
        return path
    leads = [w["lead_days"], *w.get("extra_lead_days", [])]
    names = [f"{v}_previous_day{lead}" for lead in leads for v in w["variables"]]
    params = {**_base_params(cfg), "models": w["previous_runs_model"], "hourly": ",".join(names)}
    return _store(_fetch_chunked(PREVIOUS_RUNS_URL, params), path)


def download_oracle(cfg: dict) -> Path:
    """Observed/reanalysis weather. ORACLE ONLY: never feed this to a leakage-free forecast."""
    w = cfg["weather"]
    path = cache_path(cfg["data"]["raw_dir"], "oracle", w["oracle_model"], cfg)
    if path.exists():
        return path
    params = {**_base_params(cfg), "models": w["oracle_model"], "hourly": ",".join(w["variables"])}
    return _store(_fetch_chunked(ARCHIVE_URL, params), path)


def load_forecasts(cfg: dict, lead_days: int | None = None) -> pd.DataFrame:
    """Cached forecasts for one lead (default: the main `weather.lead_days`), columns without the suffix."""
    w = cfg["weather"]
    lead = w["lead_days"] if lead_days is None else lead_days
    frame = pd.read_parquet(
        cache_path(cfg["data"]["raw_dir"], "forecast", w["previous_runs_model"], cfg)
    )
    suffix = f"_previous_day{lead}"
    columns = [c for c in frame.columns if c.endswith(suffix)]
    return frame[columns].rename(columns=lambda c: c.removesuffix(suffix))


def load_oracle(cfg: dict) -> pd.DataFrame:
    """Cached observed weather; columns prefixed `oracle_` so they stand out in any feature table."""
    frame = pd.read_parquet(
        cache_path(cfg["data"]["raw_dir"], "oracle", cfg["weather"]["oracle_model"], cfg)
    )
    return frame.add_prefix("oracle_")
