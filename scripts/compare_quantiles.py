"""Stage 6 comparison of the day-ahead quantile forecasts on the development period.

Reads the latest development run of every quantile experiment (runs/<run_id>/quantiles.parquet) and
the day-ahead hybrid's point forecast (its mean forecast, the Stage 3 winner). Never touches the
hold-out.

Outputs:
- results/stage6_comparison.csv   per run and variant: pinball, skill vs the empirical band, coverage
- results/stage6_calibration.csv  per model and level: pinball and share of hours with y <= forecast
- results/stage6_sky.csv          coverage of the 10-90% band by sky condition (descriptive only)
- results/stage6_cost.csv         imbalance cost of bidding mean / median / optimal quantile
- results/figures/stage6_{calibration,cost,fan}.png

Price scenarios are parameters of `quantile.scenarios` in the config (relative, not market data).
"""

from pathlib import Path

import compare_models as cm
import matplotlib.pyplot as plt
import pandas as pd

from pv_forecast_arena.config import load_config
from pv_forecast_arena.data.loading import load_dataset
from pv_forecast_arena.evaluation import quantile as scoring
from pv_forecast_arena.evaluation import registry
from pv_forecast_arena.models import quantile as models

DAYAHEAD = "hybrid_lgbm_residual"
EMPIRICAL = "quantile_empirical"
LGBM = "quantile_lgbm_residual"
POSTHOC = "quantile_lgbm_residual_conformal"  # added after the results, listed only if present
NAMES = (EMPIRICAL, LGBM, POSTHOC)
LABELS = {
    EMPIRICAL: "empirical band",
    LGBM: "LightGBM quantiles",
    POSTHOC: "LightGBM + recalibration",
}
COLORS = {EMPIRICAL: cm.BLUE, LGBM: cm.ORANGE, POSTHOC: cm.AQUA}
MEAN = "mean (point hybrid)"


def latest_runs(path: Path = registry.REGISTRY_PATH) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows = rows[(rows["period"] == "development") & (rows["horizon"] == "day_ahead_quantile")]
    return rows[rows["run_id"].isin(rows.groupby("name")["run_id"].max())]


def load_bids(runs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ids = runs.groupby("name")["run_id"].first()
    out = {n: pd.read_parquet(Path("runs") / r / "quantiles.parquet") for n, r in ids.items()}
    rows = pd.read_csv(registry.REGISTRY_PATH)
    day = rows[(rows["period"] == "development") & (rows["name"] == DAYAHEAD)]["run_id"].max()
    out[DAYAHEAD] = pd.read_parquet(Path("runs") / day / "forecast.parquet")
    return out


def strategies(bids: dict, level: float) -> dict[str, pd.Series]:
    """Bid series of every strategy for one optimal level."""
    opt, med = models.column(level), models.column(0.5)
    out = {MEAN: bids[DAYAHEAD]["forecast"]}
    for name in NAMES:
        if name in bids:
            out[f"{LABELS[name]}: median"] = bids[name][med]
            out[f"{LABELS[name]}: q*"] = bids[name][opt]
    return out


def cost_tables(frame, bids, cfg, variant: str) -> pd.DataFrame:
    parts = []
    for scenario, price in cfg["quantile"]["scenarios"].items():
        q_star = scoring.optimal_level(price["c_short"], price["c_surplus"])
        table = scoring.cost_table(
            frame, strategies(bids, q_star), {scenario: price}, cfg, variant, reference=MEAN
        )
        table["q_star"] = q_star
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def coverage_by_sky(frame, bids, cfg) -> pd.DataFrame:
    sky = cm.sky_class(frame, cfg)
    lo, hi = models.column(0.1), models.column(0.9)
    rows = []
    for name in NAMES:
        if name not in bids:
            continue
        q = bids[name]
        ok = scoring.mask(frame, q, "all")
        for label in cm.SKY_LABELS:
            m = (ok & (sky.reindex(q.index) == label)).to_numpy()
            index = q.index[m]
            y = frame.loc[index, "ac_power_kw"]
            band = q.loc[index]
            rows.append(
                {
                    "name": name,
                    "sky": label,
                    "n": len(y),
                    "coverage_80": float(((y >= band[lo]) & (y <= band[hi])).mean()),
                    "width_80": float((band[hi] - band[lo]).mean() / cfg["site"]["ac_capacity_kw"]),
                }
            )
    return pd.DataFrame(rows)


def plot_calibration(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color=cm.MUTED, lw=1, ls="--", label="perfect")
    for name, s in table.groupby("name"):
        label = LABELS.get(name, name)
        ax.plot(s["level"], s["frequency"], marker="o", lw=2, color=COLORS[name], label=label)
    ax.set_xlabel("nominal level")
    ax.set_ylabel("share of hours with measured <= forecast")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Calibration of quantiles, development (daytime)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_cost(cost: pd.DataFrame, out: Path) -> None:
    t = cost[cost["variant"] == "all"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for ax, (scenario, s) in zip(axes, t.groupby("scenario", sort=False)):
        ax.barh(s["strategy"], s["cost_per_kwh"], color=cm.BLUE)
        ax.invert_yaxis()
        ax.set_title(f"{scenario} (q*={s['q_star'].iloc[0]:.2f})", loc="left", fontsize=10)
        ax.set_xlabel("imbalance cost per kWh of energy")
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_fan(frame, bids, cfg, out: Path) -> None:
    """Three consecutive days of the LightGBM band, a clear and an overcast stretch (descriptive)."""
    q = bids[LGBM]
    local = q.index.tz_convert(cfg["site"]["local_timezone"]).tz_localize(None).normalize()
    sky = pd.Series(cm.sky_class(frame, cfg).reindex(q.index).to_numpy(), index=local)
    days = sky.groupby(level=0).first()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, label in zip(axes, [cm.SKY_LABELS[2], cm.SKY_LABELS[0]]):
        hit = set(days[days == label].index)
        runs = sorted(d for d in hit if all(d + pd.Timedelta(days=k) in hit for k in (1, 2)))
        pool = runs or sorted(hit)
        start = pool[len(pool) // 2]
        sel = (local >= start) & (local < start + pd.Timedelta(days=3))
        part = q[sel]
        t = part.index
        for lo, hi, a in [("q5", "q95", 0.15), ("q10", "q90", 0.25), ("q25", "q75", 0.4)]:
            ax.fill_between(t, part[lo], part[hi], color=cm.ORANGE, alpha=a, lw=0)
        ax.plot(t, part["q50"], color=cm.ORANGE, lw=1.5, label="median")
        ax.plot(t, frame.loc[t, "ac_power_kw"], color=cm.INK, lw=1.2, label="measured")
        ax.set_title(f"{label}: from {start:%Y-%m-%d}, bands 50/80/90%", loc="left", fontsize=10)
        ax.set_ylabel("AC power, kW")
    axes[0].legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(config: str = "configs/base.yaml") -> None:
    cfg = load_config(config)
    frame = load_dataset(cfg)
    runs = latest_runs()
    bids = load_bids(runs)
    results, figures = Path("results"), Path("results/figures")
    figures.mkdir(parents=True, exist_ok=True)

    cols = ["name", "variant", "n_hours", "pinball", "ref_pinball", "skill_pinball"]
    cols += ["skill_pinball_lo", "skill_pinball_hi", "nrmse_median", "mbe_median"]
    cols += ["coverage_80", "width_80", "coverage_90", "width_90"]
    runs[cols].to_csv(results / "stage6_comparison.csv", index=False, float_format="%.4f")

    calib = pd.concat(
        [scoring.per_level(frame, bids[n], cfg).assign(name=n) for n in NAMES if n in bids],
        ignore_index=True,
    )
    calib.to_csv(results / "stage6_calibration.csv", index=False, float_format="%.4f")
    coverage_by_sky(frame, bids, cfg).to_csv(
        results / "stage6_sky.csv", index=False, float_format="%.4f"
    )
    cost = pd.concat([cost_tables(frame, bids, cfg, v) for v in ("all", "clean")])
    cost.to_csv(results / "stage6_cost.csv", index=False, float_format="%.4f")

    plot_calibration(calib, figures / "stage6_calibration.png")
    plot_cost(cost, figures / "stage6_cost.png")
    plot_fan(frame, bids, cfg, figures / "stage6_fan.png")


if __name__ == "__main__":
    main()
