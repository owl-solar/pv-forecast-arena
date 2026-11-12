"""Stage 7 comparison of the deep-learning forecasters with the Stage 3 and Stage 6 winners.

Development period only. The point forecast of the TFT is its median (`q50`). Pairs are scored with
the day-block bootstrap of the skill score (model A with model B as the reference) on the common
hours of both; the quantile forecasts are compared with the LightGBM quantiles on the levels both
have (0.1 / 0.5 / 0.9 and the others in common).

Outputs:
- results/stage7_comparison.csv  per model and variant: nRMSE, nMAE, MBE, skills against the references
- results/stage7_pairs.csv       paired skill of the DL models against the hybrid and each other
- results/stage7_quantiles.csv   pinball, coverage and width of the TFT bands against LightGBM's
- results/stage7_monthly.csv     nRMSE and MBE per month
- results/figures/stage7_{month,calibration}.png
"""

from pathlib import Path

import compare_models as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pv_forecast_arena.config import load_config
from pv_forecast_arena.data.loading import load_dataset
from pv_forecast_arena.evaluation import evaluate, registry
from pv_forecast_arena.evaluation import quantile as qscoring
from pv_forecast_arena.models import baselines
from pv_forecast_arena.models import quantile as qmodels

HYBRID = "hybrid_lgbm_residual"
PHYSICS = "physics_static_erbs_perez"
LSTM = "dl_lstm"
TFT = "dl_tft_quantile"
QLGBM = "quantile_lgbm_residual"
LABELS = {
    HYBRID: "hybrid LightGBM",
    PHYSICS: "physics",
    LSTM: "LSTM",
    TFT: "TFT (median)",
}
COLORS = {HYBRID: cm.INK, PHYSICS: cm.MUTED, LSTM: cm.BLUE, TFT: cm.ORANGE}


def latest(name: str) -> str:
    rows = pd.read_csv(registry.REGISTRY_PATH)
    rows = rows[(rows["period"] == "development") & (rows["name"] == name)]
    return rows["run_id"].max()


def point_forecasts() -> dict[str, pd.Series]:
    out = {}
    for name in (HYBRID, PHYSICS, LSTM):
        out[name] = pd.read_parquet(Path("runs") / latest(name) / "forecast.parquet")["forecast"]
    tft = pd.read_parquet(Path("runs") / latest(TFT) / "quantiles.parquet")
    out[TFT] = tft["q50"].rename("forecast")
    return out


def comparison_table(frame, forecasts, cfg) -> pd.DataFrame:
    smart = baselines.smart_persistence(frame, cfg)
    rows = []
    for variant in evaluate.VARIANTS:
        for name, forecast in forecasts.items():
            others = tuple(f for n, f in forecasts.items() if n != name) + (smart,)
            row = evaluate.score(frame, forecast, smart, cfg, variant, others)
            vs_hybrid = evaluate.score(frame, forecast, forecasts[HYBRID], cfg, variant, others)
            rows.append(
                {
                    "name": name,
                    "variant": variant,
                    **{k: row[k] for k in ("n_hours", "nrmse", "nmae", "mbe")},
                    "skill_smart": row["skill"],
                    "skill_smart_lo": row["skill_lo"],
                    "skill_smart_hi": row["skill_hi"],
                    "skill_hybrid": vs_hybrid["skill"],
                    "skill_hybrid_lo": vs_hybrid["skill_lo"],
                    "skill_hybrid_hi": vs_hybrid["skill_hi"],
                }
            )
    return pd.DataFrame(rows)


def pair_table(frame, forecasts, cfg) -> pd.DataFrame:
    pairs = [(LSTM, HYBRID), (TFT, HYBRID), (LSTM, PHYSICS), (TFT, PHYSICS), (TFT, LSTM)]
    rows = []
    for a, b in pairs:
        for variant in evaluate.VARIANTS:
            s = evaluate.score(frame, forecasts[a], forecasts[b], cfg, variant)
            rows.append({"model": a, "against": b, **s})
    return pd.DataFrame(rows)


def quantile_table(frame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bands of the TFT and of LightGBM restricted to the shared levels, on their common hours."""
    tft = pd.read_parquet(Path("runs") / latest(TFT) / "quantiles.parquet")
    lgbm = pd.read_parquet(Path("runs") / latest(QLGBM) / "quantiles.parquet")
    tft_levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    shared = [qmodels.column(q) for q in tft_levels]
    lgbm = lgbm[shared]
    sub = {**cfg, "quantile": {"levels": tft_levels}}
    rows, calibration = [], []
    for variant in evaluate.VARIANTS:
        both = qscoring.mask(frame, tft, variant) & qscoring.mask(frame, lgbm, variant)
        for name, band in (("TFT", tft), ("LightGBM", lgbm)):
            ref = lgbm if name == "TFT" else tft  # restrict both to the same hours
            ok = both.to_numpy()
            index = tft.index[ok]
            sc = qscoring.score(frame, band.loc[index], ref.loc[index], sub, variant)
            rows.append({"name": name, **sc})
    for name, band in (("TFT", tft), ("LightGBM", lgbm)):
        calibration.append(qscoring.per_level(frame, band, sub).assign(name=name))
    return pd.DataFrame(rows), pd.concat(calibration, ignore_index=True)


def monthly_table(frame, forecasts, cfg) -> pd.DataFrame:
    cap = cfg["site"]["ac_capacity_kw"]
    smart = baselines.smart_persistence(frame, cfg)
    mask = evaluate.evaluation_mask(
        frame, smart, smart, "all", tuple(f for f in forecasts.values())
    )
    index = smart.index[mask.to_numpy()]
    rows = []
    for name, f in forecasts.items():
        err = f.loc[index] - frame.loc[index, "ac_power_kw"]
        for month, e in err.groupby(index.tz_convert(None).to_period("M")):
            rows.append(
                {
                    "name": name,
                    "month": str(month),
                    "n": len(e),
                    "nrmse": float(np.sqrt((e**2).mean()) / cap),
                    "mbe": float(e.mean() / cap),
                }
            )
    return pd.DataFrame(rows)


def plot_month(monthly: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, column, label in zip(axes, ["nrmse", "mbe"], ["nRMSE", "MBE (forecast - measured)"]):
        for name, color in COLORS.items():
            s = monthly[monthly["name"] == name]
            ax.plot(s["month"], s[column], marker="o", lw=2, color=color, label=LABELS[name])
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=45)
    axes[1].axhline(0, color=cm.MUTED, lw=1)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Day-ahead error by month, development test", x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_calibration(calibration: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color=cm.MUTED, lw=1, ls="--", label="perfect")
    for name, color in (("TFT", cm.ORANGE), ("LightGBM", cm.BLUE)):
        s = calibration[calibration["name"] == name]
        ax.plot(s["level"], s["frequency"], marker="o", lw=2, color=color, label=name)
    ax.set_xlabel("nominal level")
    ax.set_ylabel("share of hours with measured <= forecast")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Calibration of the quantile bands", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(config: str = "configs/base.yaml") -> None:
    cfg = load_config(config)
    frame = load_dataset(cfg)
    forecasts = point_forecasts()
    results, figures = Path("results"), Path("results/figures")
    figures.mkdir(parents=True, exist_ok=True)

    comparison_table(frame, forecasts, cfg).to_csv(
        results / "stage7_comparison.csv", index=False, float_format="%.4f"
    )
    pair_table(frame, forecasts, cfg).to_csv(
        results / "stage7_pairs.csv", index=False, float_format="%.4f"
    )
    bands, calibration = quantile_table(frame, cfg)
    bands.to_csv(results / "stage7_quantiles.csv", index=False, float_format="%.4f")
    monthly = monthly_table(frame, forecasts, cfg)
    monthly.to_csv(results / "stage7_monthly.csv", index=False, float_format="%.4f")
    plot_month(monthly, figures / "stage7_month.png")
    plot_calibration(calibration, figures / "stage7_calibration.png")


if __name__ == "__main__":
    main()
