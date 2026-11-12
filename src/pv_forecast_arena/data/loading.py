"""Load the processed dataset with the hold-out period cut off unless explicitly requested."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(cfg: dict, holdout: bool = False) -> pd.DataFrame:
    """Read dataset.parquet; without `holdout` every row from `holdout_start` on is dropped.

    The cut happens right after reading, so development code never holds hold-out rows in memory.
    """
    frame = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "dataset.parquet")
    if not holdout:
        frame = frame[frame.index < pd.Timestamp(cfg["evaluation"]["holdout_start"], tz="UTC")]
    return frame
