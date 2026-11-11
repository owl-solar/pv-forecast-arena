"""Download Open-Meteo archived forecasts (and optionally the oracle weather) once into data/raw."""

import argparse

import pandas as pd

from pv_forecast_arena.config import load_config
from pv_forecast_arena.data import weather

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument(
        "--oracle", action="store_true", help="also download observed weather (oracle runs only)"
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = [weather.download_forecasts(cfg)]
    if args.oracle:
        paths.append(weather.download_oracle(cfg))
    for path in paths:
        data = pd.read_parquet(path)
        print(f"{path.name}: {len(data)} rows, {data.index.min()} ... {data.index.max()}")
        print((data.notna().mean() * 100).round(1).to_string())
