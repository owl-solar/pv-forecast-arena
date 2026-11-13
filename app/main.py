"""Streamlit viewer of the benchmark results. Reads results/*.csv and figures only; computes nothing.

Run: streamlit run app/main.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = RESULTS / "figures"

st.set_page_config(page_title="PV Forecast Arena", layout="wide")


@st.cache_data
def table(name: str) -> pd.DataFrame | None:
    path = RESULTS / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def figure(name: str, caption: str = "") -> None:
    path = FIGURES / f"{name}.png"
    if path.exists():
        st.image(str(path), caption=caption)


def skill_bars(df: pd.DataFrame, label: str, lo: str, hi: str, value: str) -> None:
    df = df.copy()
    fig = px.bar(df, x=label, y=value, error_y=df[hi] - df[value], error_y_minus=df[value] - df[lo])
    fig.update_layout(yaxis_title="skill, 90% day-block bootstrap interval", xaxis_title="")
    st.plotly_chart(fig, width="stretch")


st.title("PV Forecast Arena")
st.caption(
    "PVDAQ 2107 (Arbuckle, CA) + archived GFS forecasts. All metrics are normalised by AC "
    "capacity and computed on daytime hours. Development period = 2024-02 .. 2024-12 tests; "
    "the 2025 hold-out is not independent any more."
)
variant = st.sidebar.radio("Hours", ["all", "clean"], help="clean drops hours flagged bad")
st.sidebar.markdown("Selection rule: if the skill interval includes 0, the simpler model wins.")

tabs = st.tabs(
    ["Day-ahead", "Intraday", "Quantiles & cost", "Deep learning", "Hold-out 2025", "Caveats"]
)

with tabs[0]:
    parts = [t for t in (table("stage2_comparison"), table("stage3_comparison")) if t is not None]
    if parts:
        df = pd.concat(parts, ignore_index=True)
        df = df[(df["variant"] == variant) & ~df.get("oracle", False).astype(bool)]
        df = df[~df.get("diagnostic", False).astype(bool)].drop_duplicates("name", keep="last")
        df = df.sort_values("nrmse")
        st.subheader("Models vs smart persistence (development)")
        cols = ["name", "n_hours", "nrmse", "nmae", "mbe", "skill", "skill_lo", "skill_hi"]
        st.dataframe(df[cols], hide_index=True, width="stretch")
        skill_bars(df.head(12), "name", "skill_lo", "skill_hi", "skill")
    c1, c2 = st.columns(2)
    with c1:
        figure("stage3_monthly", "error by month")
    with c2:
        figure("stage3_sky", "error by sky condition")
    figure("stage4_window_curve", "training-window length")

with tabs[1]:
    df = table("stage5_comparison")
    if df is not None:
        df = df[(df["variant"] == variant) & (df["scope"] == "fullday")]
        st.subheader("nRMSE by horizon (full day, development)")
        all_names = sorted(df["name"].unique())
        default = [
            n
            for n in (
                "intraday_smartpers_1h",
                "intraday_physics_lead1",
                "intraday_lgbm_nonwp",
                "intraday_lgbm_perh",
            )
            if n in all_names
        ]
        shown = st.multiselect(
            "Models (plain persistence, nRMSE 0.3-0.56, is off by default: it flattens the rest)",
            all_names,
            default,
            key="nrmse_models",
        )
        fig = px.line(df[df["name"].isin(shown)], x="lead_h", y="nrmse", color="name", markers=True)
        st.plotly_chart(fig, width="stretch")
        names = st.multiselect(
            "Skill against the day-ahead hybrid",
            sorted(df["name"].unique()),
            ["intraday_lgbm_perh"],
        )
        sub = df[df["name"].isin(names)]
        if len(sub):
            fig = px.line(sub, x="lead_h", y="skill_dayahead", color="name", markers=True)
            st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        figure("stage5_horizon")
    with c2:
        figure("stage5_hour")

with tabs[2]:
    cmp_ = table("stage6_comparison")
    if cmp_ is not None:
        st.subheader("Quantile forecasts (development)")
        st.dataframe(cmp_[cmp_["variant"] == variant], hide_index=True, width="stretch")
    cost = table("stage6_cost")
    if cost is not None:
        st.subheader("Imbalance cost of a bid")
        st.caption(
            "Prices are scenario parameters, not market data. q* = c_surplus / (c_short + c_surplus)."
        )
        scenario = st.selectbox("Scenario", cost["scenario"].unique())
        sub = cost[(cost["scenario"] == scenario) & (cost["variant"] == variant)]
        st.dataframe(sub.drop(columns=["scenario", "variant"]), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        figure("stage6_calibration", "calibration: the bands under-cover")
    with c2:
        figure("stage6_fan")

with tabs[3]:
    df = table("stage7_comparison")
    if df is not None:
        st.subheader("LSTM / TFT vs the hybrid (development)")
        st.dataframe(df[df["variant"] == variant], hide_index=True, width="stretch")
    pairs = table("stage7_pairs")
    if pairs is not None:
        st.dataframe(pairs[pairs["variant"] == variant], hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        figure("stage7_month")
    with c2:
        figure("stage7_calibration")

with tabs[4]:
    st.warning(
        "The 2025 hold-out was used once for the frozen finalists. It is a report, not a basis "
        "for further decisions."
    )
    df = table("holdout_comparison")
    if df is not None:
        st.dataframe(df[df["variant"] == variant], hide_index=True, width="stretch")
    for name in ("holdout_frozen_intraday", "holdout_frozen_quantiles"):
        extra = table(name)
        if extra is not None:
            st.subheader(name.replace("holdout_frozen_", "").title())
            st.dataframe(extra[extra["variant"] == variant], hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        figure("holdout_monthly", "error by month: the end of 2025 stands out")
    with c2:
        figure("holdout_sky")

with tabs[5]:
    st.markdown(
        """
- **Station drift at the end of 2025.** Output per unit of measured POA fell ~8-13% year over
  year (performance ratio down); every model has a positive bias (MBE) in Nov-Dec 2025. Soiling,
  degradation, sensor drift or unmodelled losses cannot be told apart from the data.
- **Quantile bands under-cover.** The nominal 80% band of the LightGBM quantiles covers ~63%.
- **Wide intervals.** About 200 independent days; intervals are day-block bootstraps.
- **Hyper-parameters are fixed**, not tuned, to avoid overfitting the development period.
- **Imbalance prices are scenarios**, not market data.
- **One station, 20 months**: conclusions about deep learning are specific to this data volume.
        """
    )
