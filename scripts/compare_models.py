"""Stage 2 comparison on the development period: tables, paired tests and figures.

Reads the latest development run of each experiment from the registry and its saved forecast in
runs/<run_id>/. Never touches the hold-out.

Outputs:
- results/stage2_comparison.csv  skill vs smart persistence with 90% day-block intervals, both variants
- results/stage2_pairs.csv       paired skill of model A against model B (B is the reference)
- results/stage2_sky.csv         nRMSE by sky condition of the day
- results/figures/stage2_*.png

Sky condition is the daily clear-sky index of the on-site POA sensor. It is used to stratify the
evaluation after the fact, never as a model input.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pv_forecast_arena.config import load_config
from pv_forecast_arena.data.loading import load_dataset
from pv_forecast_arena.evaluation import evaluate, registry

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
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

REFERENCE = "baseline_smart_persistence"
# Pairs (A, B): skill of A with B as the reference. Positive = A better on the same days.
PAIRS = [
    ("physics_raw_erbs_perez", REFERENCE),
    ("physics_static_erbs_perez", "physics_raw_erbs_perez"),
    ("physics_static_gfsdirect_perez", "physics_static_erbs_perez"),
    ("physics_static_erbs_isotropic", "physics_static_erbs_perez"),
    ("physics_rollfc30_erbs_perez", "physics_static_erbs_perez"),
    ("physics_rollfc14_erbs_perez", "physics_static_erbs_perez"),
    ("physics_rollfc30_erbs_perez", "physics_rollfc14_erbs_perez"),
    ("physics_rollpoa14_erbs_perez", "physics_raw_erbs_perez"),
    ("oracle_physics_rollfc30_erbs_perez", "physics_rollfc30_erbs_perez"),
]
# Models drawn in the monthly and sky figures, with fixed colors.
FOCUS = {
    REFERENCE: ("smart persistence", MUTED),
    "baseline_persistence": ("persistence", YELLOW),
    "physics_static_erbs_perez": ("physics, static k", BLUE),
    "physics_rollfc30_erbs_perez": ("physics, rolling k 30 d", ORANGE),
}
SKY_BINS = [0.0, 0.4, 0.8, np.inf]
SKY_LABELS = ["overcast (<0.4)", "partly cloudy", "clear (>=0.8)"]


def latest_runs(path: Path = registry.REGISTRY_PATH) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows = rows[rows["period"] == "development"]
    last = rows.groupby("name")["run_id"].max()
    return rows[rows["run_id"].isin(last)]


def load_forecasts(runs: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for name, run_id in runs.groupby("name")["run_id"].first().items():
        out[name] = pd.read_parquet(Path("runs") / run_id / "forecast.parquet")["forecast"]
    return pd.DataFrame(out)


def sky_class(frame: pd.DataFrame, cfg: dict) -> pd.Series:
    """Daily clear-sky index of measured POA, mapped back to every hour of the local day."""
    day = frame.index.tz_convert(cfg["site"]["local_timezone"]).tz_localize(None).normalize()
    use = frame["is_daytime"] & frame["obs_poa_wm2"].notna()
    obs = frame["obs_poa_wm2"].where(use).groupby(day).sum(min_count=1)
    clear = frame["clearsky_poa"].where(use).groupby(day).sum(min_count=1)
    index = (obs / clear).reindex(day).to_numpy()
    return pd.Series(pd.cut(index, SKY_BINS, labels=SKY_LABELS, right=False), index=frame.index)


def comparison_table(runs: pd.DataFrame) -> pd.DataFrame:
    cols = ["name", "variant", "oracle", "diagnostic", "n_hours", "nrmse", "nmae", "mbe"]
    cols += ["skill", "skill_lo", "skill_hi"]
    return runs[cols].sort_values(["variant", "nrmse"]).reset_index(drop=True)


def pair_table(frame, forecasts, cfg) -> pd.DataFrame:
    rows = []
    for a, b in PAIRS:
        for variant in evaluate.VARIANTS:
            s = evaluate.score(frame, forecasts[a], forecasts[b], cfg, variant)
            rows.append({"model": a, "against": b, **s})
    return pd.DataFrame(rows)


def common_mask(frame, forecasts) -> pd.Series:
    mask = frame["is_daytime"] & frame["ac_power_kw"].notna()
    for name in FOCUS:
        mask &= forecasts[name].reindex(frame.index).notna()
    return mask


def _nrmse(err: pd.Series, capacity: float) -> float:
    return float(np.sqrt((err**2).mean()) / capacity)


def monthly_table(frame, forecasts, cfg) -> pd.DataFrame:
    mask = common_mask(frame, forecasts)
    cap = cfg["site"]["ac_capacity_kw"]
    # UTC months, the same windows the walk-forward refits use.
    month = frame.index.tz_localize(None).to_period("M")
    rows = []
    for name in FOCUS:
        err = (forecasts[name].reindex(frame.index) - frame["ac_power_kw"])[mask]
        for m, e in err.groupby(month[mask]):
            rows.append(
                {"name": name, "month": str(m), "nrmse": _nrmse(e, cap), "mbe": e.mean() / cap}
            )
    return pd.DataFrame(rows)


def sky_table(frame, forecasts, cfg) -> pd.DataFrame:
    mask = common_mask(frame, forecasts)
    sky = sky_class(frame, cfg)
    cap = cfg["site"]["ac_capacity_kw"]
    rows = []
    for name in FOCUS:
        err = (forecasts[name].reindex(frame.index) - frame["ac_power_kw"])[mask]
        for label, e in err.groupby(sky[mask], observed=True):
            rows.append(
                {
                    "name": name,
                    "sky": label,
                    "n_hours": len(e),
                    "nrmse": _nrmse(e, cap),
                    "mbe": e.mean() / cap,
                }
            )
    return pd.DataFrame(rows)


def plot_skill(table: pd.DataFrame, out: Path) -> None:
    t = table[(table["variant"] == "all") & (table["name"] != REFERENCE)]
    t = t.sort_values("skill")
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.2))
    y = np.arange(len(t))
    special = t["oracle"] | t["diagnostic"]
    line = np.where(special, MUTED, np.where(t["skill"] < 0, YELLOW, BLUE))
    fill = np.where(special, GRID, line)
    ax.hlines(y, t["skill_lo"], t["skill_hi"], color=line, lw=2)
    ax.scatter(t["skill"], y, s=40, color=fill, edgecolor=line, zorder=3)
    labels = [
        n + ("  [oracle]" if o else "") + ("  [diagnostic]" if d else "")
        for n, o, d in zip(t["name"], t["oracle"], t["diagnostic"])
    ]
    ax.set_yticks(y, labels)
    ax.axvline(0, color=MUTED, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("skill vs smart persistence, 90% day-block bootstrap interval")
    fig.suptitle(
        "Skill on the development period (2024-06..12, all daytime hours)\n"
        "grey = oracle / diagnostic runs, not forecasts",
        x=0.02,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_monthly(monthly: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for name, (label, color) in FOCUS.items():
        m = monthly[monthly["name"] == name]
        for ax, col in zip(axes, ["nrmse", "mbe"]):
            ax.plot(m["month"], m[col], color=color, lw=2, marker="o", ms=5, label=label)
    axes[0].set_ylabel("nRMSE")
    axes[1].set_ylabel("MBE (fraction of AC capacity)")
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    axes[0].set_title("Monthly error on common daytime hours", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_sky(sky: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(FOCUS)
    x = np.arange(len(SKY_LABELS))
    for i, (name, (label, color)) in enumerate(FOCUS.items()):
        s = sky[sky["name"] == name].set_index("sky").reindex(SKY_LABELS)
        ax.bar(x + (i - 1.5) * width, s["nrmse"], width * 0.92, color=color, label=label)
    hours = sky[sky["name"] == REFERENCE].set_index("sky").reindex(SKY_LABELS)["n_hours"]
    ax.set_xticks(x, [f"{lab}\n{int(n)} h" for lab, n in zip(SKY_LABELS, hours)])
    ax.set_ylabel("nRMSE")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(
        "Error by sky condition of the day (daily clear-sky index of measured POA)",
        loc="left",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(config: str = "configs/base.yaml") -> None:
    cfg = load_config(config)
    frame = load_dataset(cfg)
    runs = latest_runs()
    forecasts = load_forecasts(runs).reindex(frame.index)
    results, figures = Path("results"), Path("results/figures")
    figures.mkdir(parents=True, exist_ok=True)

    table = comparison_table(runs)
    table.to_csv(results / "stage2_comparison.csv", index=False, float_format="%.4f")
    pair_table(frame, forecasts, cfg).to_csv(
        results / "stage2_pairs.csv", index=False, float_format="%.4f"
    )
    monthly = monthly_table(frame, forecasts, cfg)
    sky = sky_table(frame, forecasts, cfg)
    sky.to_csv(results / "stage2_sky.csv", index=False, float_format="%.4f")

    plot_skill(table, figures / "stage2_skill.png")
    plot_monthly(monthly, figures / "stage2_monthly.png")
    plot_sky(sky, figures / "stage2_sky.png")


if __name__ == "__main__":
    main()
