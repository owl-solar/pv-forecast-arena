"""Run one day-ahead quantile experiment from a YAML config.

Walk-forward like the point models (same splits, embargo, issue time). Writes the quantile forecasts
to runs/<run_id>/quantiles.parquet and one registry row per variant with the pinball skill against
the reference band (`quantile.reference`, empirical residual quantiles), coverage and width.
Costs of imbalance are computed afterwards by scripts/compare_quantiles.py from the saved bids.

Development period by default; `--holdout` is refused on a dirty git tree.
"""

import argparse
from pathlib import Path

import pandas as pd

from pv_forecast_arena.config import load_config
from pv_forecast_arena.data.loading import load_dataset
from pv_forecast_arena.evaluation import quantile as scoring
from pv_forecast_arena.evaluation import registry, splits
from pv_forecast_arena.models import get_quantile_model


def walk_forward(frame: pd.DataFrame, cfg: dict, holdout: bool) -> tuple[pd.DataFrame, list[dict]]:
    make_splits = splits.holdout_splits if holdout else splits.development_splits
    predict = get_quantile_model(cfg["model"]["type"])
    parts = [predict(frame, s, cfg) for s in make_splits(frame.index, cfg["evaluation"])]
    folds = [p.attrs.pop("info") for p in parts if "info" in p.attrs]
    return pd.concat(parts), folds


def run(cfg: dict, holdout: bool = False) -> list[dict]:
    frame = load_dataset(cfg, holdout=holdout)
    forecast, folds = walk_forward(frame, cfg, holdout)
    ref = cfg["quantile"]["reference"]
    reference, _ = walk_forward(frame, {**cfg, "name": ref["name"], "model": ref["model"]}, holdout)

    run_id = registry.new_run_id(cfg["name"])
    commit, dirty = registry.git_state()
    rows = []
    for variant in ("all", "clean"):
        rows.append(
            {
                "run_id": run_id,
                "name": cfg["name"],
                "model": cfg["model"]["type"],
                "horizon": cfg["horizon"],
                "period": "holdout" if holdout else "development",
                "oracle": False,
                "diagnostic": False,
                **scoring.score(frame, forecast, reference, cfg, variant),
                "commit": commit,
                "dirty": dirty,
                "config_hash": registry.config_hash(cfg),
            }
        )
    out = Path("runs") / run_id
    out.mkdir(parents=True, exist_ok=True)
    if folds:
        pd.DataFrame(folds).to_csv(out / "folds.csv", index=False)
    forecast.to_parquet(out / "quantiles.parquet")
    registry.append(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--holdout", action="store_true", help="evaluate the hold-out (final)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.holdout and registry.git_state()[1]:
        raise SystemExit("refusing to touch the hold-out with uncommitted changes: commit first")
    for row in run(cfg, holdout=args.holdout):
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
