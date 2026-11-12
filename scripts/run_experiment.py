"""Run one experiment from a YAML config: walk-forward forecasts, metrics, registry row.

Development period by default. `--holdout` evaluates the hold-out period instead and is refused on
a dirty git tree, so the code that produced a hold-out result is always a commit.
"""

import argparse
from pathlib import Path

import pandas as pd

from pv_forecast_arena.config import load_config
from pv_forecast_arena.data.loading import load_dataset
from pv_forecast_arena.evaluation import evaluate, registry, splits
from pv_forecast_arena.models import baselines, get_model


def run(cfg: dict, holdout: bool = False) -> list[dict]:
    frame = load_dataset(cfg, holdout=holdout)
    ev = cfg["evaluation"]
    make_splits = splits.holdout_splits if holdout else splits.development_splits
    predict = get_model(cfg["model"]["type"])

    parts = [predict(frame, s, cfg) for s in make_splits(frame.index, ev)]
    forecast = pd.concat(parts).rename("forecast")
    reference = baselines.smart_persistence(frame, cfg)

    run_id = registry.new_run_id(cfg["name"])
    commit, dirty = registry.git_state()
    rows = []
    for variant in evaluate.VARIANTS:
        rows.append(
            {
                "run_id": run_id,
                "name": cfg["name"],
                "model": cfg["model"]["type"],
                "horizon": cfg["horizon"],
                "period": "holdout" if holdout else "development",
                "oracle": bool(cfg.get("oracle", False)),
                "diagnostic": bool(cfg.get("diagnostic", False)),
                **evaluate.score(frame, forecast, reference, cfg, variant),
                "commit": commit,
                "dirty": dirty,
                "config_hash": registry.config_hash(cfg),
            }
        )
    out = Path("runs") / run_id
    out.mkdir(parents=True, exist_ok=True)
    forecast.to_frame().to_parquet(out / "forecast.parquet")
    evaluate.monthly(frame, forecast, reference, cfg).to_csv(out / "monthly.csv")
    registry.append(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--holdout", action="store_true", help="evaluate the hold-out (final comparison)"
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.holdout and registry.git_state()[1]:
        raise SystemExit("refusing to touch the hold-out with uncommitted changes: commit first")
    for row in run(cfg, holdout=args.holdout):
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
