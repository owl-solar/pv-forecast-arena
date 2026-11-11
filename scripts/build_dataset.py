"""Build data/processed/dataset.parquet from cached raw data and print the QA summary."""

import argparse

from pv_forecast_arena.config import load_config
from pv_forecast_arena.data import cleaning

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    cfg = load_config(ap.parse_args().config)
    frame = cleaning.build_dataset(cfg)
    path = cleaning.save_dataset(frame, cfg)
    print(f"saved {path}: {frame.shape[0]} rows x {frame.shape[1]} columns")
    print(f"coverage: {frame.index.min()} ... {frame.index.max()}")
    print("flagged share, % of rows:")
    print(cleaning.summarize(frame).to_string())
    print("forecast / oracle non-null share, %:")
    cols = [c for c in frame.columns if c.startswith(("fc_", "oracle_"))]
    print((frame[cols].notna().mean() * 100).round(1).to_string())
