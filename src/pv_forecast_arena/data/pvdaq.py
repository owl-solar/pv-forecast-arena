"""Download NREL PVDAQ Data Prize files once and cache them to data/raw as parquet.

Raw parquet files keep the source columns and naive timestamps exactly as published: the files
of one system use different clocks, so conversion to UTC happens in a later, explicit step.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

BUCKET_URL = "https://oedi-data-lake.s3.amazonaws.com"
TIME_COLUMNS = ("measured_on", "utc_measured_on")
USER_AGENT = "pv-forecast-arena"


def raw_path(raw_dir: str | Path, system_id: int, filename: str) -> Path:
    """Cache location of one source CSV."""
    return Path(raw_dir) / "pvdaq" / str(system_id) / f"{Path(filename).stem}.parquet"


def parse_csv(content: bytes) -> pd.DataFrame:
    """Parse a PVDAQ CSV and index it by its timestamp column, left naive."""
    data = pd.read_csv(io.BytesIO(content))
    time_column = next((c for c in TIME_COLUMNS if c in data.columns), None)
    if time_column is None:
        raise ValueError(f"No timestamp column among {list(data.columns)[:5]}")
    data[time_column] = pd.to_datetime(data[time_column])
    return data.set_index(time_column).sort_index()


def _fetch(url: str, attempts: int = 4) -> bytes:
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=300)
            response.raise_for_status()
            return response.content
        except (requests.ConnectionError, requests.Timeout):
            if attempt == attempts:
                raise
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def download_file(system_id: int, prefix: str, filename: str, raw_dir: str | Path) -> Path:
    """Download one CSV unless its parquet cache exists; return the cache path."""
    path = raw_path(raw_dir, system_id, filename)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = parse_csv(_fetch(f"{BUCKET_URL}/{prefix}/{filename}"))
    temporary = path.with_suffix(".tmp")
    data.to_parquet(temporary)
    temporary.replace(path)
    return path


def download_all(cfg: dict) -> list[Path]:
    """Download every file listed under `pvdaq.files` in the config."""
    system_id = cfg["site"]["pvdaq_system_id"]
    prefix = cfg["pvdaq"]["prefix"]
    raw_dir = cfg["data"]["raw_dir"]
    return [
        download_file(system_id, prefix, filename, raw_dir)
        for group in cfg["pvdaq"]["files"].values()
        for filename in group
    ]


def to_utc(series: pd.Series, clock: str, local_timezone: str) -> pd.Series:
    """Convert a naive-indexed series to a UTC index.

    `clock="local"`: wall-clock time with DST (the fall-back hour occurs once in these files, so
    ambiguous times are taken as the first occurrence; nonexistent spring-forward times are dropped).
    `clock="utc"`: already UTC.
    """
    series = series[~series.index.duplicated()].sort_index()
    if clock == "utc":
        series.index = series.index.tz_localize("UTC")
        return series
    if clock != "local":
        raise ValueError(f"Unknown clock {clock!r}")
    index = series.index.tz_localize(local_timezone, ambiguous=True, nonexistent="NaT")
    series = series[~index.isna()]
    series.index = index[~index.isna()].tz_convert("UTC")
    return series[~series.index.duplicated()].sort_index()


def load_hourly(cfg: dict, kind: str) -> pd.Series:
    """Cached series `kind` ("meter" or "irradiance") as hourly means, UTC, hour-ending labels.

    The label at T covers [T-1h, T], the same convention as Open-Meteo's `shortwave_radiation`.
    Meter stamps mark the start of a 15-min interval, so stamps in [T-1h, T) belong to T;
    irradiance stamps are taken as the end of their averaging interval, so (T-1h, T] belongs to T.
    Hours with fewer than three quarters of the expected samples are set to NaN.
    """
    spec = cfg["pvdaq"]["use"][kind]
    path = raw_path(cfg["data"]["raw_dir"], cfg["site"]["pvdaq_system_id"], spec["file"])
    raw = pd.read_parquet(path).iloc[:, 0].astype(float)
    series = to_utc(raw, spec["clock"], cfg["site"]["local_timezone"])
    closed = "left" if spec["label"] == "start" else "right"
    # Coverage of an hour = total time represented by its samples (spacing to the previous sample,
    # capped at 15 min), so mixed 5/10/15-min stamping is handled and gaps are not mistaken for data.
    spacing = (
        series.index.to_series().diff().clip(upper=pd.Timedelta("15min")).dt.total_seconds() / 3600
    )
    hourly = series.resample("1h", closed=closed, label="right").mean()
    coverage = spacing.resample("1h", closed=closed, label="right").sum().reindex(hourly.index)
    hourly[coverage < 0.75] = float("nan")
    return hourly.rename(kind)
