"""Download PVDAQ data once and cache it to data/raw."""

import argparse

import pandas as pd

from pv_forecast_arena.config import load_config
from pv_forecast_arena.data import pvdaq

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    cfg = load_config(ap.parse_args().config)
    for path in pvdaq.download_all(cfg):
        data = pd.read_parquet(path)
        print(f"{path.name}: {len(data)} rows, {data.index.min()} ... {data.index.max()}")
