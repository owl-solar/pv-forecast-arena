"""Final comparison on the hold-out (2025), after the models were frozen on the development period.

Reads the latest hold-out run of each finalist from the registry (written by
`run_experiment.py --holdout` on a clean git tree). Nothing is tuned here.

Outputs results/holdout_{comparison,pairs,monthly,sky}.csv and results/figures/holdout_*.png.
"""

from pathlib import Path

import compare_models as cm
import pandas as pd

from pv_forecast_arena.config import load_config
from pv_forecast_arena.data.loading import load_dataset
from pv_forecast_arena.evaluation import evaluate, registry

PHYS, ML, HYB = cm.PHYSICS, "ml_lgbm_csi", "hybrid_lgbm_residual"
NAMES = [cm.REFERENCE, PHYS, ML, HYB]
PAIRS = [(HYB, PHYS), (ML, PHYS), (HYB, ML)]
FOCUS = {
    cm.REFERENCE: ("smart persistence", cm.MUTED),
    PHYS: ("physics, static k", cm.BLUE),
    ML: ("pure ML, clear-sky index", cm.YELLOW),
    HYB: ("hybrid: physics + ML residual", cm.ORANGE),
}


def holdout_runs(path: Path = registry.REGISTRY_PATH) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows = rows[(rows["period"] == "holdout") & rows["name"].isin(NAMES)]
    last = rows.groupby("name")["run_id"].max()
    return rows[rows["run_id"].isin(last)]


def main(config: str = "configs/base.yaml") -> None:
    cfg = load_config(config)
    frame = load_dataset(cfg, holdout=True)
    frame = frame[frame.index >= pd.Timestamp(cfg["evaluation"]["holdout_start"], tz="UTC")]
    runs = holdout_runs()
    forecasts = cm.load_forecasts(runs).reindex(frame.index)
    results, figures = Path("results"), Path("results/figures")

    cols = ["name", "variant", "n_hours", "nrmse", "nmae", "mbe", *cm.STAGES[3]["skill_cols"]]
    table = runs[cols].sort_values(["variant", "nrmse"]).reset_index(drop=True)
    table.to_csv(results / "holdout_comparison.csv", index=False, float_format="%.4f")

    rows = []
    for a, b in PAIRS:
        for variant in evaluate.VARIANTS:
            s = evaluate.score(frame, forecasts[a], forecasts[b], cfg, variant)
            rows.append({"model": a, "against": b, **s})
    pd.DataFrame(rows).to_csv(results / "holdout_pairs.csv", index=False, float_format="%.4f")

    monthly = cm.monthly_table(frame, forecasts, cfg, FOCUS)
    monthly.to_csv(results / "holdout_monthly.csv", index=False, float_format="%.4f")
    sky = cm.sky_table(frame, forecasts, cfg, FOCUS)
    sky.to_csv(results / "holdout_sky.csv", index=False, float_format="%.4f")
    cm.plot_monthly(monthly, figures / "holdout_monthly.png", FOCUS)
    cm.plot_sky(sky, figures / "holdout_sky.png", FOCUS)
    t = table.assign(oracle=False, diagnostic=False)
    cm.plot_skill(t, figures / "holdout_skill.png", 3)


if __name__ == "__main__":
    main()
