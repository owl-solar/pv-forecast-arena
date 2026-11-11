"""Run one experiment from a YAML config and append results to the registry. TODO."""

import argparse

from pv_forecast_arena.config import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    raise NotImplementedError("Implement in stage 2")
