"""Score intraday forecasts per horizon and pooled over horizons.

A forecast for horizon h is a series indexed by target hour, so the day-ahead machinery applies
unchanged: for every horizon the model and all references are compared on one common set of hours
(daytime, measured target, every forecast present). Models that need recent daytime measurements have
no forecast for the first hours of the day; `coverage` says which share of the daytime target hours
survives, so that an apparent gain is not just a different set of easy hours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pv_forecast_arena.evaluation import evaluate

POOLED = "pooled"


def _daytime_target_hours(frame: pd.DataFrame, index: pd.DatetimeIndex, variant: str) -> int:
    rows = frame.loc[index]
    ok = rows["is_daytime"] & rows["ac_power_kw"].notna()
    if variant == "clean":
        ok &= ~rows["flag_bad"]
    return int(ok.sum())


def score_horizons(
    frame: pd.DataFrame,
    forecasts: dict[int, pd.Series],
    reference: dict[int, pd.Series],
    cfg: dict,
    variant: str,
    others: dict[int, tuple[pd.Series, ...]] | None = None,
) -> list[dict]:
    """One row per horizon plus a `pooled` row; skill is against `reference` of the same horizon.

    `others` (per horizon) only narrow the mask, as in `evaluate.score`.
    """
    others = others or {}
    rows, parts = [], []
    for h, forecast in sorted(forecasts.items()):
        extra = others.get(h, ())
        mask = evaluate.evaluation_mask(frame, forecast, reference[h], variant, extra)
        if not mask.any():
            raise ValueError(f"no hours left to evaluate at horizon {h}")
        arrays = evaluate.masked_arrays(frame, forecast, reference[h], mask, cfg)
        total = _daytime_target_hours(frame, forecast.index, variant)
        rows.append(
            {
                "lead_h": h,
                **evaluate.summarise(*arrays, cfg, variant),
                "coverage": int(mask.sum()) / total,
            }
        )
        parts.append((arrays, total))
    pooled = tuple(np.concatenate([p[0][i] for p in parts]) for i in range(4))
    total = sum(p[1] for p in parts)
    rows.append(
        {
            "lead_h": POOLED,
            **evaluate.summarise(*pooled, cfg, variant),
            "coverage": len(pooled[0]) / total,
        }
    )
    return rows
