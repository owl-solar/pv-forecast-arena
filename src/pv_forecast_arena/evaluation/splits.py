"""Time-based walk-forward splits and the hold-out period. No shuffling, no random folds.

Development period = [train_start, holdout_start): hyper-parameters are tuned only here.
Hold-out period    = [holdout_start, end]: touched once for the final comparison. Inside the
hold-out, models are still refit every `refit_every_months` on all data that was known before each
test window (expanding window), which is what an operational system would do.

A test window starts at `test_start`. Training rows end `embargo` before it: a day-ahead forecast for
the first test day is issued the day before, so the newest targets are not yet known.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex

    @property
    def test_start(self) -> pd.Timestamp:
        return self.test_index.min()


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def walk_forward_splits(
    index: pd.DatetimeIndex,
    train_start: str | pd.Timestamp,
    first_test_start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    refit_every_months: int = 1,
    embargo: str | pd.Timedelta = "1D",
) -> Iterator[Split]:
    """Yield expanding-window splits: train = [train_start, test_start - embargo), test = one window.

    `end` is exclusive. Windows with no test rows are skipped.
    """
    if not index.is_monotonic_increasing or index.tz is None:
        raise ValueError("index must be sorted and timezone-aware (UTC)")
    train_start, test_start = _utc(train_start), _utc(first_test_start)
    end = _utc(end) if end is not None else index.max() + pd.Timedelta("1ns")
    embargo = pd.Timedelta(embargo)
    step = pd.DateOffset(months=refit_every_months)
    while test_start < end:
        test_end = min(test_start + step, end)
        train = index[(index >= train_start) & (index < test_start - embargo)]
        test = index[(index >= test_start) & (index < test_end)]
        if len(train) and len(test):
            yield Split(train, test)
        test_start = test_start + step


def development_splits(index: pd.DatetimeIndex, eval_cfg: dict) -> Iterator[Split]:
    """Walk-forward splits strictly before the hold-out, for hyper-parameter selection."""
    first_test = _utc(eval_cfg["train_start"]) + pd.DateOffset(months=eval_cfg["min_train_months"])
    return walk_forward_splits(
        index,
        eval_cfg["train_start"],
        first_test,
        end=eval_cfg["holdout_start"],
        refit_every_months=eval_cfg["refit_every_months"],
    )


def holdout_splits(index: pd.DatetimeIndex, eval_cfg: dict) -> Iterator[Split]:
    """Walk-forward splits over the hold-out period, refitting on everything known earlier."""
    return walk_forward_splits(
        index,
        eval_cfg["train_start"],
        eval_cfg["holdout_start"],
        refit_every_months=eval_cfg["refit_every_months"],
    )
