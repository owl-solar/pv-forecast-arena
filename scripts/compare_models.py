"""Stage comparison on the development period: tables, paired tests and figures.

Usage: compare_models.py --stage 2|3 (default 3).

Reads the latest development run of each experiment from the registry and its saved forecast in
runs/<run_id>/. Never touches the hold-out.

Outputs:
- results/stage<N>_comparison.csv  skill vs smart persistence (and, from Stage 3, vs the Stage 2
                                  physics winner) with 90% day-block intervals, both variants
- results/stage<N>_pairs.csv       paired skill of model A against model B (B is the reference)
- results/stage<N>_sky.csv         nRMSE by sky condition of the day
- results/figures/stage<N>_*.png   (Stage 3 adds feature importance of the tree models)

Sky condition is the daily clear-sky index of the on-site POA sensor. It is used to stratify the
evaluation after the fact, never as a model input.
"""

import argparse
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
PHYSICS = "physics_static_erbs_perez"
STAGE2_PAIRS = [
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
STAGE3_MODELS = [
    "ml_lgbm_ac",
    "ml_xgb_ac",
    "ml_lgbm_csi",
    "hybrid_lgbm_feature",
    "hybrid_lgbm_residual",
    "hybrid_lgbm_csi",
    "hybrid_lgbm_residual_nopast",
    "hybrid_lgbm_residual_doy",
]
STAGE3_PAIRS = [
    *((m, PHYSICS) for m in STAGE3_MODELS),
    ("ml_xgb_ac", "ml_lgbm_ac"),  # learner
    ("ml_lgbm_csi", "ml_lgbm_ac"),  # scale-free target, pure ML
    ("hybrid_lgbm_residual", "ml_lgbm_csi"),  # what physics adds beyond a scale-free target
    ("hybrid_lgbm_residual", "hybrid_lgbm_feature"),  # hybrid type
    ("hybrid_lgbm_residual", "hybrid_lgbm_csi"),
    ("hybrid_lgbm_residual", "hybrid_lgbm_residual_nopast"),  # past measurements
    ("hybrid_lgbm_residual_doy", "hybrid_lgbm_residual"),  # day of year
    ("oracle_hybrid_lgbm_residual", "hybrid_lgbm_residual"),  # weather error of the hybrid
    # ML gain with true weather:
    ("oracle_hybrid_lgbm_residual", "oracle_physics_static_erbs_perez"),
]
# Per stage: which registry names belong to it, the pairs, and the models drawn in the monthly and
# sky figures with fixed colors.
STAGES = {
    2: {
        "names": lambda n: n.startswith(("baseline_", "physics_", "oracle_physics", "diagnostic_")),
        "pairs": STAGE2_PAIRS,
        "focus": {
            REFERENCE: ("smart persistence", MUTED),
            "baseline_persistence": ("persistence", YELLOW),
            PHYSICS: ("physics, static k", BLUE),
            "physics_rollfc30_erbs_perez": ("physics, rolling k 30 d", ORANGE),
        },
        "skill_cols": ["skill", "skill_lo", "skill_hi"],
    },
    3: {
        "names": lambda n: (
            n in (REFERENCE, PHYSICS, "oracle_physics_static_erbs_perez")
            or n.startswith(("ml_", "hybrid_", "oracle_hybrid_", "oracle_ml_"))
        ),
        "pairs": STAGE3_PAIRS,
        "focus": {
            REFERENCE: ("smart persistence", MUTED),
            PHYSICS: ("physics, static k", BLUE),
            "ml_lgbm_csi": ("pure ML, clear-sky index", YELLOW),
            "hybrid_lgbm_residual": ("hybrid: physics + ML residual", ORANGE),
        },
        "skill_cols": [
            "skill",
            "skill_lo",
            "skill_hi",
            "phys_nrmse",
            "skill_phys",
            "skill_phys_lo",
            "skill_phys_hi",
        ],
    },
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


def comparison_table(runs: pd.DataFrame, stage: dict) -> pd.DataFrame:
    cols = ["name", "variant", "oracle", "diagnostic", "n_hours", "nrmse", "nmae", "mbe"]
    cols += stage["skill_cols"]
    runs = runs[runs["name"].map(stage["names"])]
    return runs[cols].sort_values(["variant", "nrmse"]).reset_index(drop=True)


def pair_table(frame, forecasts, cfg, stage: dict) -> pd.DataFrame:
    rows = []
    for a, b in stage["pairs"]:
        if a not in forecasts or b not in forecasts:
            print(f"skipping pair {a} vs {b}: no development run yet")
            continue
        for variant in evaluate.VARIANTS:
            s = evaluate.score(frame, forecasts[a], forecasts[b], cfg, variant)
            rows.append({"model": a, "against": b, **s})
    return pd.DataFrame(rows)


def common_mask(frame, forecasts, focus) -> pd.Series:
    mask = frame["is_daytime"] & frame["ac_power_kw"].notna()
    for name in focus:
        mask &= forecasts[name].reindex(frame.index).notna()
    return mask


def _nrmse(err: pd.Series, capacity: float) -> float:
    return float(np.sqrt((err**2).mean()) / capacity)


def monthly_table(frame, forecasts, cfg, focus) -> pd.DataFrame:
    mask = common_mask(frame, forecasts, focus)
    cap = cfg["site"]["ac_capacity_kw"]
    # UTC months, the same windows the walk-forward refits use.
    month = frame.index.tz_localize(None).to_period("M")
    rows = []
    for name in focus:
        err = (forecasts[name].reindex(frame.index) - frame["ac_power_kw"])[mask]
        for m, e in err.groupby(month[mask]):
            rows.append(
                {"name": name, "month": str(m), "nrmse": _nrmse(e, cap), "mbe": e.mean() / cap}
            )
    return pd.DataFrame(rows)


def sky_table(frame, forecasts, cfg, focus) -> pd.DataFrame:
    mask = common_mask(frame, forecasts, focus)
    sky = sky_class(frame, cfg)
    cap = cfg["site"]["ac_capacity_kw"]
    rows = []
    for name in focus:
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


def _skill_panel(ax, t: pd.DataFrame, prefix: str, reference_label: str) -> None:
    y = np.arange(len(t))
    special = t["oracle"] | t["diagnostic"]
    line = np.where(special, MUTED, np.where(t[prefix] < 0, YELLOW, BLUE))
    fill = np.where(special, GRID, line)
    ax.hlines(y, t[f"{prefix}_lo"], t[f"{prefix}_hi"], color=line, lw=2)
    ax.scatter(t[prefix], y, s=40, color=fill, edgecolor=line, zorder=3)
    ax.axvline(0, color=MUTED, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"skill vs {reference_label}, 90% interval")


def plot_skill(table: pd.DataFrame, out: Path, stage_no: int) -> None:
    t = table[(table["variant"] == "all") & (table["name"] != REFERENCE)]
    t = t.sort_values("skill").reset_index(drop=True)
    panels = [("skill", "smart persistence")]
    if stage_no >= 3:
        panels.append(("skill_phys", "physics static k"))
    width = 8 if len(panels) == 1 else 11
    fig, axes = plt.subplots(
        1, len(panels), figsize=(width, 0.42 * len(t) + 1.4), sharey=True, squeeze=False
    )
    for ax, (prefix, label) in zip(axes[0], panels):
        _skill_panel(ax, t, prefix, label)
    labels = [
        n + ("  [oracle]" if o else "") + ("  [diagnostic]" if d else "")
        for n, o, d in zip(t["name"], t["oracle"], t["diagnostic"])
    ]
    axes[0][0].set_yticks(np.arange(len(t)), labels)
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


def plot_monthly(monthly: pd.DataFrame, out: Path, focus: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for name, (label, color) in focus.items():
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


def plot_sky(sky: pd.DataFrame, out: Path, focus: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(focus)
    x = np.arange(len(SKY_LABELS))
    for i, (name, (label, color)) in enumerate(focus.items()):
        s = sky[sky["name"] == name].set_index("sky").reindex(SKY_LABELS)
        offset = (i - (len(focus) - 1) / 2) * width
        ax.bar(x + offset, s["nrmse"], width * 0.92, color=color, label=label)
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


def importance_table(runs: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Mean share of gain per feature over the walk-forward folds of each tree model."""
    out = {}
    for name in names:
        run_id = runs.loc[runs["name"] == name, "run_id"]
        if not len(run_id):
            continue
        path = Path("runs") / str(run_id.iloc[0]) / "folds.csv"
        if path.exists():
            folds = pd.read_csv(path).filter(like="imp_")
            out[name] = folds.mean().rename(lambda c: c.removeprefix("imp_"))
    return pd.DataFrame(out)


def plot_importance(imp: pd.DataFrame, out: Path, focus: dict, top: int = 14) -> None:
    order = imp.max(axis=1).sort_values().index[-top:]
    imp = imp.reindex(order)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(imp) + 1.4))
    y = np.arange(len(imp))
    height = 0.8 / len(imp.columns)
    for i, name in enumerate(imp.columns):
        label, color = focus.get(name, (name, MUTED))
        offset = (i - (len(imp.columns) - 1) / 2) * height
        ax.barh(y + offset, imp[name].fillna(0.0), height * 0.92, color=color, label=label)
    ax.set_yticks(y, imp.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of total gain, mean over folds")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Feature importance of the tree models", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(stage_no: int = 3, config: str = "configs/base.yaml") -> None:
    stage = STAGES[stage_no]
    focus = stage["focus"]
    cfg = load_config(config)
    frame = load_dataset(cfg)
    runs = latest_runs()
    forecasts = load_forecasts(runs).reindex(frame.index)
    results, figures = Path("results"), Path("results/figures")
    figures.mkdir(parents=True, exist_ok=True)
    tag = f"stage{stage_no}"

    table = comparison_table(runs, stage)
    table.to_csv(results / f"{tag}_comparison.csv", index=False, float_format="%.4f")
    pair_table(frame, forecasts, cfg, stage).to_csv(
        results / f"{tag}_pairs.csv", index=False, float_format="%.4f"
    )
    monthly = monthly_table(frame, forecasts, cfg, focus)
    monthly.to_csv(results / f"{tag}_monthly.csv", index=False, float_format="%.4f")
    sky = sky_table(frame, forecasts, cfg, focus)
    sky.to_csv(results / f"{tag}_sky.csv", index=False, float_format="%.4f")

    plot_skill(table, figures / f"{tag}_skill.png", stage_no)
    plot_monthly(monthly, figures / f"{tag}_monthly.png", focus)
    plot_sky(sky, figures / f"{tag}_sky.png", focus)
    if stage_no >= 3:
        imp = importance_table(runs, ["ml_lgbm_csi", "hybrid_lgbm_residual"])
        imp.to_csv(results / f"{tag}_importance.csv", float_format="%.4f")
        plot_importance(imp, figures / f"{tag}_importance.png", focus)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=3, choices=sorted(STAGES))
    main(ap.parse_args().stage)
