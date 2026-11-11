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
