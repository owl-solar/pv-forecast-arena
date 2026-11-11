"""Which dataset columns a model may see.

This is an allow-list on purpose: the dataset also holds the target, its quality flags (computed from
the target), on-site measurements and observed weather, and any new column is forbidden until it is
listed here. `fc_lead1_*` starts with `fc_` but is not leakage-free for hours after the issue time,
so forecast columns are named exactly rather than matched by prefix.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

DETERMINISTIC_FEATURES = ("sun_elevation", "clearsky_ghi", "clearsky_poa")


def allowed_features(cfg: dict, oracle: bool = False) -> list[str]:
    """Columns a model may use: geometry plus main-lead forecasts, or oracle weather if `oracle`."""
    prefix = "oracle_" if oracle else "fc_"
    return [*DETERMINISTIC_FEATURES, *(f"{prefix}{v}" for v in cfg["weather"]["variables"])]


def select_features(
    frame: pd.DataFrame, columns: Iterable[str], cfg: dict, oracle: bool = False
) -> pd.DataFrame:
    """Return `frame[columns]`, raising if any column is outside the allow-list for this run."""
    columns = list(columns)
    forbidden = sorted(set(columns) - set(allowed_features(cfg, oracle)))
    if forbidden:
        raise ValueError(f"Columns not allowed as model inputs (oracle={oracle}): {forbidden}")
    return frame[columns]
