"""Run one intraday experiment (horizons 1-6 h) from a YAML config.

Walk-forward over target hours with the day-ahead splits and embargo. Every horizon is scored on its
own common mask against two references: the intraday reference (`evaluation.intraday_reference`,
smart persistence on the newest known hours; `skill`) and the day-ahead model
(`evaluation.dayahead_reference`; `skill_dayahead`). One registry row per run, variant and horizon,
plus a `pooled` row; forecasts go to runs/<run_id>/forecast_h<h>.parquet.

Development period by default; `--holdout` is refused on a dirty git tree.
"""

import argparse
from pathlib import Path

import pandas as pd
import run_experiment

from pv_forecast_arena.config import load_config
from pv_forecast_arena.data.loading import load_dataset
from pv_forecast_arena.evaluation import intraday as scoring
from pv_forecast_arena.evaluation import registry, splits
from pv_forecast_arena.models import get_intraday_model


def walk_forward(
    frame: pd.DataFrame, cfg: dict, holdout: bool
) -> tuple[dict[int, pd.Series], list[dict]]:
    """Concatenated test-window forecasts of `cfg["model"]` for every horizon, and per-fold info."""
    make_splits = splits.holdout_splits if holdout else splits.development_splits
    predict = get_intraday_model(cfg["model"]["type"])
    folds, out = [], {}
    for h in cfg["intraday"]["horizons"]:
        parts = [predict(frame, s, cfg, h) for s in make_splits(frame.index, cfg["evaluation"])]
        folds += [{"lead_h": h, **p.attrs.pop("info")} for p in parts if "info" in p.attrs]
        out[h] = pd.concat(parts).rename("forecast")
    return out, folds


def intraday_reference(frame: pd.DataFrame, cfg: dict, holdout: bool) -> dict[int, pd.Series]:
    ref = cfg["evaluation"]["intraday_reference"]
    ref_cfg = {**cfg, "name": ref["name"], "model": ref["model"]}
    return walk_forward(frame, ref_cfg, holdout)[0]


def dayahead_reference(frame: pd.DataFrame, cfg: dict, holdout: bool) -> pd.Series:
    """Day-ahead forecast of every target hour (the same series for every horizon)."""
    ref = cfg["evaluation"]["dayahead_reference"]
    ref_cfg = {**cfg, "name": ref["name"], "model": ref["model"], "oracle": False}
    return run_experiment.walk_forward(frame, ref_cfg, holdout)[0]


def run(cfg: dict, holdout: bool = False) -> list[dict]:
    frame = load_dataset(cfg, holdout=holdout)
    forecasts, folds = walk_forward(frame, cfg, holdout)
    reference = intraday_reference(frame, cfg, holdout)
    dayahead = dayahead_reference(frame, cfg, holdout)
    dayahead_h = {h: dayahead for h in forecasts}

    run_id = registry.new_run_id(cfg["name"])
    commit, dirty = registry.git_state()
    rows = []
    meta = {
        "run_id": run_id,
        "name": cfg["name"],
        "model": cfg["model"]["type"],
        "horizon": "intraday",
        "period": "holdout" if holdout else "development",
        "oracle": bool(cfg.get("oracle", False)),
        "diagnostic": bool(cfg.get("diagnostic", False)),
    }
    tail = {"commit": commit, "dirty": dirty, "config_hash": registry.config_hash(cfg)}
    for variant in ("all", "clean"):
        # common: hours where the intraday reference exists too (it has none in the morning, so
        # long horizons shrink to the afternoon). Skill is against it; skill_dayahead on the same hours.
        vs_main = scoring.score_horizons(
            frame, forecasts, reference, cfg, variant, {h: (dayahead,) for h in forecasts}
        )
        vs_day = scoring.score_horizons(
            frame, forecasts, dayahead_h, cfg, variant, {h: (reference[h],) for h in forecasts}
        )
        for main, day in zip(vs_main, vs_day, strict=True):
            rows.append(
                {
                    **meta,
                    "scope": "common",
                    **main,
                    "dayahead_nrmse": day["ref_nrmse"],
                    "skill_dayahead": day["skill"],
                    "skill_dayahead_lo": day["skill_lo"],
                    "skill_dayahead_hi": day["skill_hi"],
                    **tail,
                }
            )
        # fullday: every daytime hour the model forecasts, against the day-ahead model only.
        # This is the fair view for models that, unlike smart persistence, work in the morning.
        for day in scoring.score_horizons(frame, forecasts, dayahead_h, cfg, variant):
            rows.append(
                {
                    **meta,
                    "scope": "fullday",
                    **{k: v for k, v in day.items() if k in ("lead_h", "variant", "n_hours")},
                    "nrmse": day["nrmse"],
                    "nmae": day["nmae"],
                    "mbe": day["mbe"],
                    "coverage": day["coverage"],
                    "dayahead_nrmse": day["ref_nrmse"],
                    "skill_dayahead": day["skill"],
                    "skill_dayahead_lo": day["skill_lo"],
                    "skill_dayahead_hi": day["skill_hi"],
                    **tail,
                }
            )
    out = Path("runs") / run_id
    out.mkdir(parents=True, exist_ok=True)
    if folds:
        pd.DataFrame(folds).to_csv(out / "folds.csv", index=False)
    for h, forecast in forecasts.items():
        forecast.to_frame().to_parquet(out / f"forecast_h{h}.parquet")
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
        if row["variant"] == "all" and row["lead_h"] == "pooled":
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
