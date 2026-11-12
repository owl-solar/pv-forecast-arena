"""Stage 1 verification figures: power vs forecast GHI, one clear and one cloudy week.

Built from the development period only (before `evaluation.holdout_start`).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pv_forecast_arena.config import load_config

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.size": 10,
    }
)


def scatter(d, cfg, out):
    day = d[d.is_daytime & ~d.flag_bad]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        day.fc_shortwave_radiation, day.ac_power_kw, s=6, color=BLUE, alpha=0.25, linewidths=0
    )
    ax.set_xlabel("Forecast GHI, GFS 2 days ahead (W/m²)")
    ax.set_ylabel("Measured AC power (kW)")
    r = day.ac_power_kw.corr(day.fc_shortwave_radiation)
    ax.set_title(
        f"AC power vs forecast GHI, daytime hours (n={len(day)}, r={r:.2f})",
        loc="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def week(d, cfg, start, title, out):
    w = d.loc[start : start + pd.Timedelta("7D") - pd.Timedelta("1h")]
    cap = cfg["site"]["ac_capacity_kw"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(w.index, w.ac_power_kw / cap, color=BLUE, lw=2, label="Measured AC power / capacity")
    ax.plot(
        w.index,
        w.fc_shortwave_radiation / 1000,
        color=ORANGE,
        lw=2,
        label="Forecast GHI / 1000 W/m²",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of reference")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    cfg = load_config(ap.parse_args().config)
    d = pd.read_parquet(Path(cfg["data"]["processed_dir"]) / "dataset.parquet")
    # Development period only: the hold-out must not be looked at before the final comparison.
    d = d[d.index < pd.Timestamp(cfg["evaluation"]["holdout_start"], tz="UTC")]
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    scatter(d, cfg, out / "power_vs_forecast_ghi.png")

    # Weeks are picked from on-site clear-sky index (not from the forecast), full weeks without bad hours.
    # Measured POA is compared with clear-sky POA: against clear-sky GHI the tilted plane looks
    # "clearer" in winter, which biased the pick towards December.
    daily = d.resample("1D").agg({"obs_poa_wm2": "sum", "clearsky_poa": "sum", "flag_bad": "sum"})
    daily["csi"] = daily.obs_poa_wm2 / daily.clearsky_poa
    weeks = daily.resample("7D").agg({"csi": "mean", "flag_bad": "sum"})
    weeks = weeks[(weeks.flag_bad == 0) & (weeks.index >= "2024-03-01")].iloc[:-1]
    clear, cloudy = weeks.csi.idxmax(), weeks.csi.idxmin()
    week(
        d,
        cfg,
        clear,
        f"Clear week from {clear:%Y-%m-%d} (UTC): forecast vs measured",
        out / "week_clear.png",
    )
    week(
        d,
        cfg,
        cloudy,
        f"Cloudy week from {cloudy:%Y-%m-%d} (UTC): forecast vs measured",
        out / "week_cloudy.png",
    )
    print(
        "clear",
        clear.date(),
        round(weeks.csi[clear], 2),
        "cloudy",
        cloudy.date(),
        round(weeks.csi[cloudy], 2),
    )
