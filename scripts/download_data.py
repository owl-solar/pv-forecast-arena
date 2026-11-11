"""Download PVDAQ + Open-Meteo data once and cache to data/raw. TODO."""

import argparse

from pv_forecast_arena.config import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    cfg = load_config(ap.parse_args().config)
    raise NotImplementedError("Implement in stage 1")
