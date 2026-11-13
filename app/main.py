"""Streamlit viewer of the benchmark results. Reads results/*.csv and figures only; computes nothing.

English and Russian interface (sidebar). Table columns and the plots drawn in the pipeline scripts
stay in English.

Run: streamlit run app/main.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
DIAGRAMS = ROOT / "docs" / "img"

st.set_page_config(page_title="PV Forecast Arena", layout="wide")

TEXT = {
    "en": {
        "language": "Language",
        "caption": (
            "PVDAQ 2107 (Arbuckle, CA) + archived GFS forecasts. All metrics are normalised by AC "
            "capacity and computed on daytime hours. Development period = 2024-02 .. 2024-12 tests; "
            "the 2025 hold-out is not independent any more."
        ),
        "hours": "Hours",
        "hours_help": "clean drops hours flagged bad",
        "rule": "Selection rule: if the skill interval includes 0, the simpler model wins.",
        "tabs": [
            "About",
            "Day-ahead",
            "Intraday",
            "Quantiles & cost",
            "Deep learning",
            "Hold-out 2025",
            "Caveats",
        ],
        "about": """
**Question.** How well can the output of a solar plant be forecast for tomorrow (issued 10:00 on the
previous day) and 1-6 hours ahead, and which method is worth using? Very different methods are run
under identical, leak-free conditions on one real plant (20 months).

**Main findings**

1. **Physics + ML wins.** A physical model (pvlib) whose residual is learned by LightGBM forecasts
   tomorrow with an error of 14.8% of capacity on the 2025 hold-out: 28% better than smart
   persistence, 11% better than physics alone.
2. **Most of the remaining error is the weather input** (about 80% of the squared error), not the
   model.
3. **For 1-6 h a fresher weather forecast matters most**: -28% error at 1 h, -10% at 6 h against
   the day-ahead model.
4. **LSTM and Transformer did not beat it** on this amount of data.
5. **Quantile forecasts save money when a shortfall is expensive** (-9% imbalance cost in a
   scenario analysis), but the bands are too narrow (nominal 80% covers 63%).
6. **The plant drifted at the end of 2025** (output per unit of sunlight -8..13%): all models
   over-forecast in Nov-Dec 2025.
        """,
        "diagrams": "How it works",
        "dayahead_title": "Models vs smart persistence (development)",
        "monthly": "error by month",
        "sky": "error by sky condition",
        "window": "training-window length",
        "skill_axis": "skill, 90% day-block bootstrap interval",
        "nrmse_title": "nRMSE by horizon (full day, development)",
        "models_pick": "Models (plain persistence, nRMSE 0.3-0.56, is off by default: it flattens the rest)",
        "skill_pick": "Skill against the day-ahead hybrid",
        "quantile_title": "Quantile forecasts (development)",
        "cost_title": "Imbalance cost of a bid",
        "cost_caption": "Prices are scenario parameters, not market data. q* = c_surplus / (c_short + c_surplus).",
        "scenario": "Scenario",
        "calibration": "calibration: the bands under-cover",
        "dl_title": "LSTM / TFT vs the hybrid (development)",
        "holdout_warn": (
            "The 2025 hold-out was used once for the frozen finalists. It is a report, not a basis "
            "for further decisions."
        ),
        "intraday": "Intraday",
        "quantiles": "Quantiles",
        "holdout_month": "error by month: the end of 2025 stands out",
        "caveats": """
- **Station drift at the end of 2025.** Output per unit of measured POA fell ~8-13% year over
  year (performance ratio down); every model has a positive bias (MBE) in Nov-Dec 2025. Soiling,
  degradation, sensor drift or unmodelled losses cannot be told apart from the data.
- **Quantile bands under-cover.** The nominal 80% band of the LightGBM quantiles covers ~63%.
- **Wide intervals.** About 200 independent days; intervals are day-block bootstraps.
- **Hyper-parameters are fixed**, not tuned, to avoid overfitting the development period.
- **Imbalance prices are scenarios**, not market data.
- **One station, 20 months**: conclusions about deep learning are specific to this data volume.
        """,
    },
    "ru": {
        "language": "Язык",
        "caption": (
            "PVDAQ 2107 (Arbuckle, Калифорния) + архивные прогнозы GFS. Все метрики нормированы на "
            "установленную AC-мощность и считаются по дневным часам. Период разработки: тесты "
            "2024-02 .. 2024-12; hold-out 2025 больше не независим."
        ),
        "hours": "Часы",
        "hours_help": "clean — без часов, помеченных как плохие",
        "rule": "Правило отбора: если интервал skill включает 0, выигрывает более простая модель.",
        "tabs": [
            "О проекте",
            "Day-ahead",
            "Intraday",
            "Квантили и стоимость",
            "Глубокое обучение",
            "Hold-out 2025",
            "Ограничения",
        ],
        "about": """
**Вопрос.** Насколько точно можно прогнозировать выработку солнечной станции на завтра (выпуск в
10:00 предыдущего дня) и на 1-6 часов вперёд, и какой метод стоит применять? Очень разные методы
запускаются в одинаковых условиях без утечки будущего на одной реальной станции (20 месяцев).

**Главные выводы**

1. **Физика + ML выигрывает.** Физическая модель (pvlib), остаток которой учит LightGBM,
   прогнозирует завтра с ошибкой 14,8% мощности на hold-out 2025: на 28% лучше smart persistence,
   на 11% лучше одной физики.
2. **Основная оставшаяся ошибка — во входной погоде** (около 80% квадрата ошибки), а не в модели.
3. **На 1-6 ч решает более свежий прогноз погоды**: ошибка меньше на 28% при 1 ч и на 10% при 6 ч
   относительно day-ahead модели.
4. **LSTM и Transformer не обошли** гибрид на таком объёме данных.
5. **Квантильные прогнозы экономят деньги при дорогой недопоставке** (-9% затрат на дисбаланс в
   сценарном анализе), но полосы слишком узкие (номинальные 80% покрывают 63%).
6. **Станция «поплыла» в конце 2025** (выработка на единицу солнечной энергии -8..13%): все
   модели завышают прогноз в ноябре-декабре 2025.
        """,
        "diagrams": "Как это устроено",
        "dayahead_title": "Модели против smart persistence (разработка)",
        "monthly": "ошибка по месяцам",
        "sky": "ошибка по типу неба",
        "window": "длина окна обучения",
        "skill_axis": "skill, 90% бутстреп-интервал по суткам",
        "nrmse_title": "nRMSE по горизонту (весь день, разработка)",
        "models_pick": "Модели (обычный persistence, nRMSE 0,3-0,56, по умолчанию выключен: сплющивает остальные)",
        "skill_pick": "Skill против day-ahead гибрида",
        "quantile_title": "Квантильные прогнозы (разработка)",
        "cost_title": "Стоимость дисбаланса заявки",
        "cost_caption": "Цены — параметры сценария, а не рыночные данные. q* = c_surplus / (c_short + c_surplus).",
        "scenario": "Сценарий",
        "calibration": "калибровка: полосы недопокрывают",
        "dl_title": "LSTM / TFT против гибрида (разработка)",
        "holdout_warn": (
            "Hold-out 2025 использован один раз для замороженных финалистов. Это отчёт, а не "
            "основа для дальнейших решений."
        ),
        "intraday": "Intraday",
        "quantiles": "Квантили",
        "holdout_month": "ошибка по месяцам: выделяется конец 2025",
        "caveats": """
- **Дрейф станции в конце 2025.** Выработка на единицу измеренного POA упала примерно на 8-13%
  год к году (performance ratio ниже); у всех моделей положительное смещение (MBE) в ноябре-декабре
  2025. Загрязнение, деградация, дрейф датчика или неучтённые потери по данным не различить.
- **Квантильные полосы недопокрывают.** Номинальная 80%-я полоса квантилей LightGBM покрывает ~63%.
- **Широкие интервалы.** Около 200 независимых суток; интервалы — бутстреп по суткам.
- **Гиперпараметры фиксированные**, не подбирались, чтобы не переобучиться на период разработки.
- **Цены дисбаланса — сценарии**, а не рыночные данные.
- **Одна станция, 20 месяцев**: выводы про глубокое обучение относятся к этому объёму данных.
        """,
    },
}

lang = st.sidebar.radio(
    "Language / Язык", ["en", "ru"], format_func=lambda c: {"en": "English", "ru": "Русский"}[c]
)
T = TEXT[lang]


@st.cache_data
def table(name: str) -> pd.DataFrame | None:
    path = RESULTS / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def figure(name: str, caption: str = "") -> None:
    path = FIGURES / f"{name}.png"
    if path.exists():
        st.image(str(path), caption=caption)


def diagram(name: str) -> None:
    path = DIAGRAMS / f"{name}.png"
    if path.exists():
        st.image(str(path))


def skill_bars(df: pd.DataFrame, label: str, lo: str, hi: str, value: str) -> None:
    df = df.copy()
    fig = px.bar(df, x=label, y=value, error_y=df[hi] - df[value], error_y_minus=df[value] - df[lo])
    fig.update_layout(yaxis_title=T["skill_axis"], xaxis_title="")
    st.plotly_chart(fig, width="stretch")


st.title("PV Forecast Arena")
st.caption(T["caption"])
variant = st.sidebar.radio(T["hours"], ["all", "clean"], help=T["hours_help"])
st.sidebar.markdown(T["rule"])

tabs = st.tabs(T["tabs"])

with tabs[0]:
    st.markdown(T["about"])
    st.subheader(T["diagrams"])
    for name in (
        "diagram_pipeline",
        "diagram_hybrid",
        "diagram_timeline",
        "diagram_walkforward",
        "diagram_error_budget",
    ):
        diagram(name)

with tabs[1]:
    parts = [t for t in (table("stage2_comparison"), table("stage3_comparison")) if t is not None]
    if parts:
        df = pd.concat(parts, ignore_index=True)
        df = df[(df["variant"] == variant) & ~df.get("oracle", False).astype(bool)]
        df = df[~df.get("diagnostic", False).astype(bool)].drop_duplicates("name", keep="last")
        df = df.sort_values("nrmse")
        st.subheader(T["dayahead_title"])
        cols = ["name", "n_hours", "nrmse", "nmae", "mbe", "skill", "skill_lo", "skill_hi"]
        st.dataframe(df[cols], hide_index=True, width="stretch")
        skill_bars(df.head(12), "name", "skill_lo", "skill_hi", "skill")
    c1, c2 = st.columns(2)
    with c1:
        figure("stage3_monthly", T["monthly"])
    with c2:
        figure("stage3_sky", T["sky"])
    figure("stage4_window_curve", T["window"])

with tabs[2]:
    df = table("stage5_comparison")
    if df is not None:
        df = df[(df["variant"] == variant) & (df["scope"] == "fullday")]
        st.subheader(T["nrmse_title"])
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
        shown = st.multiselect(T["models_pick"], all_names, default, key="nrmse_models")
        fig = px.line(df[df["name"].isin(shown)], x="lead_h", y="nrmse", color="name", markers=True)
        st.plotly_chart(fig, width="stretch")
        names = st.multiselect(T["skill_pick"], all_names, ["intraday_lgbm_perh"])
        sub = df[df["name"].isin(names)]
        if len(sub):
            fig = px.line(sub, x="lead_h", y="skill_dayahead", color="name", markers=True)
            st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        figure("stage5_horizon")
    with c2:
        figure("stage5_hour")

with tabs[3]:
    cmp_ = table("stage6_comparison")
    if cmp_ is not None:
        st.subheader(T["quantile_title"])
        st.dataframe(cmp_[cmp_["variant"] == variant], hide_index=True, width="stretch")
    cost = table("stage6_cost")
    if cost is not None:
        st.subheader(T["cost_title"])
        st.caption(T["cost_caption"])
        scenario = st.selectbox(T["scenario"], cost["scenario"].unique())
        sub = cost[(cost["scenario"] == scenario) & (cost["variant"] == variant)]
        st.dataframe(sub.drop(columns=["scenario", "variant"]), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        figure("stage6_calibration", T["calibration"])
    with c2:
        figure("stage6_fan")

with tabs[4]:
    df = table("stage7_comparison")
    if df is not None:
        st.subheader(T["dl_title"])
        st.dataframe(df[df["variant"] == variant], hide_index=True, width="stretch")
    pairs = table("stage7_pairs")
    if pairs is not None:
        st.dataframe(pairs[pairs["variant"] == variant], hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        figure("stage7_month")
    with c2:
        figure("stage7_calibration")

with tabs[5]:
    st.warning(T["holdout_warn"])
    df = table("holdout_comparison")
    if df is not None:
        st.dataframe(df[df["variant"] == variant], hide_index=True, width="stretch")
    for name, key in (
        ("holdout_frozen_intraday", "intraday"),
        ("holdout_frozen_quantiles", "quantiles"),
    ):
        extra = table(name)
        if extra is not None:
            st.subheader(T[key])
            st.dataframe(extra[extra["variant"] == variant], hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        figure("holdout_monthly", T["holdout_month"])
    with c2:
        figure("holdout_sky")

with tabs[6]:
    st.markdown(T["caveats"])
