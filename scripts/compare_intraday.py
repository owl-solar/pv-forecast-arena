"""Stage 5 comparison of the intraday runs on the development period: tables, paired tests, figures.

Reads the latest development run of every intraday experiment from the registry and its forecasts in
runs/<run_id>/forecast_h<h>.parquet. The day-ahead hybrid (Stage 3 winner) is added from its own run.
Never touches the hold-out.

Outputs:
- results/stage5_comparison.csv   per model, scope, variant and horizon: metrics, coverage, skills
- results/stage5_pairs.csv        paired skill of model A against model B (B is the reference), per
                                  horizon and pooled, on the full-day mask of the two models
- results/stage5_sky.csv          nRMSE by sky condition of the day (pooled over horizons)
- results/stage5_hour.csv         nRMSE by local hour of the target (pooled over horizons)
- results/figures/stage5_*.png

`common` scope = hours where the intraday smart persistence also has a forecast (its mornings are
missing); `fullday` = every daytime hour the model forecasts. Sky condition and hour of day stratify
the evaluation after the fact and are never model inputs.
"""

from pathlib import Path

import compare_models as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pv_forecast_arena.config import load_config
from pv_forecast_arena.data.loading import load_dataset
from pv_forecast_arena.evaluation import evaluate, registry
from pv_forecast_arena.evaluation import intraday as scoring

DAYAHEAD = "hybrid_lgbm_residual"
SMART = "intraday_smartpers_1h"
SMART_3H = "intraday_smartpers_3h"
PHYS = "intraday_physics_lead1"
NOWCAST = "intraday_physics_nowcast"
RESIDUAL = "intraday_lgbm_residual"
PAIRS = [
    (PHYS, DAYAHEAD),  # fresher NWP alone
    (NOWCAST, PHYS),  # recent measurements, physics only
    (RESIDUAL, PHYS),  # ML on top of the same physics
    (RESIDUAL, NOWCAST),
    (RESIDUAL, DAYAHEAD),  # what intraday adds to the day-ahead model
    ("intraday_lgbm_perh", RESIDUAL),  # one model per horizon
    ("intraday_lgbm_csi", RESIDUAL),  # target
    ("intraday_lgbm_residual_nowcast", RESIDUAL),  # baseline of the residual
    ("intraday_lgbm_nonwp", RESIDUAL),  # value of NWP
    (RESIDUAL, "intraday_lgbm_nonwp"),
    ("intraday_lgbm_perh", "intraday_lgbm_residual_nowcast"),  # the two winning choices, one each
    # Added after the results (not pre-registered): both choices together.
    ("intraday_lgbm_perh_nowcast", "intraday_lgbm_perh"),
    ("intraday_lgbm_perh_nowcast", "intraday_lgbm_residual_nowcast"),
    ("intraday_lgbm_perh_nowcast", DAYAHEAD),
]
FOCUS = {
    DAYAHEAD: ("day-ahead hybrid", cm.MUTED),
    PHYS: ("physics, lead-1 NWP", cm.BLUE),
    NOWCAST: ("physics + nowcast", cm.AQUA),
    RESIDUAL: ("hybrid, intraday", cm.ORANGE),
    "intraday_lgbm_perh": ("hybrid, per horizon", cm.INK),
    "intraday_lgbm_nonwp": ("ML without NWP", cm.YELLOW),
}


def latest_intraday_runs(path: Path = registry.REGISTRY_PATH) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows = rows[(rows["period"] == "development") & (rows["horizon"] == "intraday")]
    rows = rows[rows["scope"].notna()]
    last = rows.groupby("name")["run_id"].max()
    return rows[rows["run_id"].isin(last)]


def latest_dayahead_run(name: str) -> str:
    rows = pd.read_csv(registry.REGISTRY_PATH)
    rows = rows[(rows["period"] == "development") & (rows["name"] == name)]
    return rows["run_id"].max()


def load_forecasts(runs: pd.DataFrame, horizons: list[int]) -> dict[str, dict[int, pd.Series]]:
    out = {}
    for name, run_id in runs.groupby("name")["run_id"].first().items():
        out[name] = {
            h: pd.read_parquet(Path("runs") / run_id / f"forecast_h{h}.parquet")["forecast"]
            for h in horizons
        }
    day = pd.read_parquet(Path("runs") / latest_dayahead_run(DAYAHEAD) / "forecast.parquet")
    out[DAYAHEAD] = {h: day["forecast"] for h in horizons}
    return out


def comparison_table(runs: pd.DataFrame) -> pd.DataFrame:
    cols = ["name", "scope", "variant", "lead_h", "n_hours", "coverage", "nrmse", "nmae", "mbe"]
    cols += ["skill", "skill_lo", "skill_hi", "dayahead_nrmse"]
    cols += ["skill_dayahead", "skill_dayahead_lo", "skill_dayahead_hi"]
    return runs[cols].reset_index(drop=True)


def pair_table(frame, forecasts, cfg) -> pd.DataFrame:
    rows = []
    for a, b in PAIRS:
        if a not in forecasts or b not in forecasts:
            print(f"skipping pair {a} vs {b}: no development run yet")
            continue
        for variant in evaluate.VARIANTS:
            for row in scoring.score_horizons(frame, forecasts[a], forecasts[b], cfg, variant):
                rows.append({"model": a, "against": b, **row})
    return pd.DataFrame(rows)


def _stratified(frame, forecasts, cfg, key: pd.Series, label: str) -> pd.DataFrame:
    """nRMSE by `key` over all horizons, on the hours where every focus model has a forecast."""
    cap = cfg["site"]["ac_capacity_kw"]
    horizons = sorted(next(iter(forecasts.values())))
    rows = []
    for h in horizons:
        mask = frame["is_daytime"] & frame["ac_power_kw"].notna()
        for name in FOCUS:
            mask &= forecasts[name][h].reindex(frame.index).notna()
        for name in FOCUS:
            err = (forecasts[name][h].reindex(frame.index) - frame["ac_power_kw"])[mask]
            for k, e in err.groupby(key[mask], observed=True):
                rows.append({"name": name, label: k, "lead_h": h, "n": len(e), "sq": (e**2).sum()})
    t = pd.DataFrame(rows).groupby(["name", label], observed=True)[["n", "sq"]].sum().reset_index()
    t["nrmse"] = np.sqrt(t["sq"] / t["n"]) / cap
    return t.drop(columns="sq")


def plot_horizon(table: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, scope in zip(axes, ["fullday", "common"]):
        t = table[(table["variant"] == "all") & (table["scope"] == scope)]
        t = t[t["lead_h"] != scoring.POOLED]
        for name, (label, color) in FOCUS.items():
            s = t[t["name"] == name]
            if len(s):
                ax.plot(
                    s["lead_h"].astype(int), s["nrmse"], color=color, lw=2, marker="o", label=label
                )
        # The day-ahead hybrid on the hours of the per-horizon run (the same mask as its own line).
        day = t[t["name"] == "intraday_lgbm_perh"]
        if len(day):
            ax.plot(
                day["lead_h"].astype(int),
                day["dayahead_nrmse"],
                color=cm.MUTED,
                lw=2,
                marker="o",
                label=FOCUS[DAYAHEAD][0],
            )
        for name, color, label in [(SMART, cm.INK, "smart persistence, 1 h")]:
            s = t[t["name"] == name]
            if len(s):
                ax.plot(
                    s["lead_h"].astype(int), s["nrmse"], color=color, lw=1.5, ls="--", label=label
                )
        ax.set_xlabel("horizon, h")
        ax.set_title(
            "all daytime hours the model covers"
            if scope == "fullday"
            else "hours with a persistence forecast",
            loc="left",
            fontsize=10,
        )
    axes[0].set_ylabel("nRMSE")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Error against horizon, development period (test Jun-Dec 2024)",
        x=0.02,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_skill_dayahead(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    t = table[(table["variant"] == "all") & (table["scope"] == "fullday")]
    t = t[t["lead_h"] != scoring.POOLED]
    names = [n for n in FOCUS if n != DAYAHEAD]
    for i, name in enumerate(names):
        label, color = FOCUS[name]
        s = t[t["name"] == name]
        x = s["lead_h"].astype(int) + (i - (len(names) - 1) / 2) * 0.1
        ax.errorbar(
            x,
            s["skill_dayahead"],
            yerr=[
                s["skill_dayahead"] - s["skill_dayahead_lo"],
                s["skill_dayahead_hi"] - s["skill_dayahead"],
            ],
            color=color,
            lw=1.5,
            marker="o",
            ms=4,
            label=label,
            capsize=2,
        )
    ax.axhline(0, color=cm.MUTED, lw=1)
    ax.set_xlabel("horizon, h")
    ax.set_ylabel("skill vs day-ahead hybrid, 90% interval")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(
        "What intraday adds to the day-ahead forecast (all daytime hours)", loc="left", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_hour(hour: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, (label, color) in FOCUS.items():
        s = hour[hour["name"] == name]
        ax.plot(s["local_hour"], s["nrmse"], color=color, lw=2, marker="o", ms=4, label=label)
    ax.set_xlabel("local hour of the target (start of the hour)")
    ax.set_ylabel("nRMSE")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Error by hour of day, pooled over horizons 1-6 h", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(config: str = "configs/base.yaml") -> None:
    cfg = load_config(config)
    frame = load_dataset(cfg)
    runs = latest_intraday_runs()
    horizons = list(cfg["intraday"]["horizons"])
    forecasts = load_forecasts(runs, horizons)
    results, figures = Path("results"), Path("results/figures")
    figures.mkdir(parents=True, exist_ok=True)

    table = comparison_table(runs)
    table.to_csv(results / "stage5_comparison.csv", index=False, float_format="%.4f")
    pair_table(frame, forecasts, cfg).to_csv(
        results / "stage5_pairs.csv", index=False, float_format="%.4f"
    )
    sky = _stratified(frame, forecasts, cfg, cm.sky_class(frame, cfg), "sky")
    sky.to_csv(results / "stage5_sky.csv", index=False, float_format="%.4f")
    local_hour = pd.Series(
        (frame.index - pd.Timedelta("1h")).tz_convert(cfg["site"]["local_timezone"]).hour,
        index=frame.index,
    )
    hour = _stratified(frame, forecasts, cfg, local_hour, "local_hour")
    hour.to_csv(results / "stage5_hour.csv", index=False, float_format="%.4f")

    plot_horizon(table, figures / "stage5_horizon.png")
    plot_skill_dayahead(table, figures / "stage5_skill_dayahead.png")
    plot_hour(hour, figures / "stage5_hour.png")


if __name__ == "__main__":
    main()
