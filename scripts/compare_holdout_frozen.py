"""Hold-out 2025 report of the frozen Stage 5 and Stage 6 finalists (run once, nothing tuned).

Reads the hold-out rows of results/experiments.csv written by `run_intraday.py --holdout` and
`run_quantiles.py --holdout` on a clean tree, and the quantile forecasts of runs/<run_id>/.

Outputs results/holdout_frozen_{intraday,quantiles,calibration}.csv.
"""

from pathlib import Path

import pandas as pd

from pv_forecast_arena.config import load_config
from pv_forecast_arena.data.loading import load_dataset
from pv_forecast_arena.evaluation import quantile as scoring
from pv_forecast_arena.evaluation import registry

INTRADAY = "intraday_lgbm_perh"
QUANTILES = ("quantile_empirical", "quantile_lgbm_residual")


def latest(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = rows[rows["name"] == name]
    return rows[rows["run_id"] == rows["run_id"].max()]


def main(config: str = "configs/base.yaml") -> None:
    cfg = load_config(config)
    rows = pd.read_csv(registry.REGISTRY_PATH)
    rows = rows[rows["period"] == "holdout"]
    results = Path("results")

    cols = ["name", "scope", "variant", "lead_h", "n_hours", "nrmse", "nmae", "mbe", "skill"]
    cols += ["skill_lo", "skill_hi", "dayahead_nrmse", "skill_dayahead", "skill_dayahead_lo"]
    cols += ["skill_dayahead_hi"]
    intraday = latest(rows, INTRADAY)
    intraday[cols].to_csv(results / "holdout_frozen_intraday.csv", index=False, float_format="%.4f")

    quant = pd.concat([latest(rows, n) for n in QUANTILES])
    qcols = ["name", "variant", "n_hours", "pinball", "ref_pinball", "skill_pinball"]
    qcols += ["skill_pinball_lo", "skill_pinball_hi", "nrmse_median", "mbe_median"]
    qcols += ["coverage_80", "width_80", "coverage_90", "width_90"]
    quant[qcols].to_csv(results / "holdout_frozen_quantiles.csv", index=False, float_format="%.4f")

    frame = load_dataset(cfg, holdout=True)
    calib = []
    for name in QUANTILES:
        run_id = latest(quant, name)["run_id"].iloc[0]
        band = pd.read_parquet(Path("runs") / run_id / "quantiles.parquet")
        calib.append(scoring.per_level(frame, band, cfg).assign(name=name))
    pd.concat(calib).to_csv(
        results / "holdout_frozen_calibration.csv", index=False, float_format="%.4f"
    )


if __name__ == "__main__":
    main()
