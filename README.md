# PV Forecast Arena

A benchmark of solar PV generation forecasting approaches on one real plant, built the way a forecasting model should be evaluated before it goes near production: the same data, the same walk-forward splits, the same metrics, a strict no-look-ahead rule, and statistical intervals on every claimed improvement.

**Live results viewer:** https://pv-forecast-arena-ddg9bmsxem37hkcw29ye3w.streamlit.app/ (free hosting; it sleeps without visitors and wakes up in about half a minute).

**In one paragraph.** On a 893 kW(DC) plant in California, a hybrid model (a physical pvlib chain whose *residual* is learned by LightGBM) forecasts the next day's hourly output with an nRMSE of 14.8% of capacity on a year of held-out data, 28% better than a smart-persistence baseline and 11% better than the physics alone. For 1–6 h ahead, adding the newest weather forecast and the latest measurements cuts the error further (by 28% at 1 h, 10% at 6 h against the day-ahead model). Deep learning (LSTM, TFT) did **not** beat the hybrid with 20 months of data. Quantile forecasts are useful for bidding when shortfalls are expensive (−9% imbalance cost in a scenario analysis), but their bands are **too narrow**, and the plant itself drifted at the end of 2025. All of this, including what failed, is documented below.

## What this project demonstrates

| Skill | Where |
|---|---|
| Time-series forecasting without leakage | one feature allow-list, issue-time logic, tests that corrupt the future and check that the forecast does not move (with a mutation check that the test can fail) |
| Domain knowledge (PV physics) | pvlib chain: Erbs decomposition → Perez transposition → IAM → SAPM cell temperature → PVWatts DC/AC with an inverter limit |
| Honest model comparison | walk-forward CV, 2-day embargo, day-block bootstrap intervals, a pre-declared selection rule, a hold-out used and labelled |
| ML engineering | YAML-driven experiments, a registry (`results/experiments.csv`: commit, config hash, dirty flag), reproducible scripts, 147 tests, ruff |
| Probabilistic forecasting and decision value | quantile regression, pinball loss, calibration, imbalance-cost optimisation |
| Communicating limitations | see "What did not work" and "Limitations" below |

## Data and setup

- **Plant:** NREL PVDAQ system 2107 (Arbuckle, CA; 893 kW DC, fixed 25° tilt, south-facing). The AC capacity (708 kW) is the observed maximum, used only to normalise metrics.
- **Measurements:** 15-min AC power and plane-of-array (POA) irradiance, resampled to hour-ending UTC; hours with less than 75% coverage are invalid. Data quality flags (missing, stuck, outage, clipping) via `pvanalytics` and own rules; flagged hours are dropped from training labels and kept visible in the `all` vs `clean` evaluation variants.
- **Weather:** Open-Meteo *Previous Runs* archive of the **GFS** model, `previous_day2`: the forecast that was actually published about two days before the valid hour. This matches a day-ahead issue at 10:00 local time on D-1. The `previous_day1` run is exactly 24 h before the target hour, which for hours after 10:00 would be a leak; it is used only for intraday horizons, where it is legitimate, and for an explicit leak comparison. Observed ("oracle") weather is used only in runs labelled as oracle, to measure how much error comes from the weather forecast.
- **Span:** 2024-02 … 2025-12. Development: 2024 (first test after 4 months of training). Hold-out: 2025.
- **Tasks:** day-ahead hourly forecast (issued 10:00 local D-1) and intraday 1–6 h ahead (issued at the end of an hour, measurements available one hour later).

## Evaluation protocol

- Walk-forward (rolling-origin) splits, monthly retraining with an expanding window, **2-day embargo** between train and test. No shuffling.
- Daytime hours only; **nRMSE / nMAE / MBE as a share of AC capacity**, in two variants: `all` and `clean`.
- **Skill score** = 1 − error(model) / error(reference), with a **90% day-block bootstrap interval** (whole local days are resampled, because hours within a day are not independent).
- **Selection rule, fixed in advance:** if the interval of an improvement includes 0, the simpler model wins.
- **Hyper-parameters are fixed**, not tuned, to avoid overfitting a short history. The list of runs per stage was fixed before looking at results; extra runs done afterwards are marked "after results".
- **Hold-out:** 2025 was used once for the day-ahead finalists and once more, frozen, for the intraday and quantile finalists. It is **not independent any more**: new experiments are judged on the development period only.

## Methods and tools

| Stage | Approach | Tools |
|---|---|---|
| Baselines | persistence (same hour yesterday); **smart persistence**: clear-sky POA × a scale factor measured over the last 24 h before issue | pvlib clear-sky, pandas |
| Physics | Erbs (GHI → direct/diffuse) or GFS direct → Perez/Hay-Davies/isotropic transposition → physical IAM → SAPM cell temperature → PVWatts, scale calibrated on the training window; variants: static, rolling | pvlib |
| Tabular ML | LightGBM / XGBoost on geometry and NWP features, target = AC power, clear-sky index, or **residual of the physics chain** | LightGBM, XGBoost, scikit-learn |
| Hybrid | physics output as a feature, or as the base of the residual | `tabular.py`, `hybrid.py` |
| Intraday | one model per horizon on the residual of physics driven by the newest NWP run, plus lagged measurements through a single audited path (`features.as_of`) | LightGBM |
| Quantiles | LightGBM with the quantile objective per level (5…95%) on the physics residual; empirical band of training residuals as the reference | LightGBM, own pinball/cost scoring |
| Deep learning | LSTM and Temporal Fusion Transformer (quantile loss) on 48 h of clear-sky-index history plus future NWP covariates | darts, PyTorch (CPU), PyTorch Lightning |
| Evaluation | day-block bootstrap, calibration, imbalance cost | numpy, pandas, matplotlib |
| Reporting | Streamlit viewer of the result tables | Streamlit, Plotly |

LightGBM settings are fixed (15 leaves, min 50 samples per leaf, learning rate 0.03, feature and bagging fraction 0.8); the number of trees comes from early stopping on the last 30 days of the training window (with an inner embargo), followed by a refit on all of it.

## Results

### Day-ahead (nRMSE as a share of AC capacity; skill vs smart persistence, 90% interval)

| Model | Development 2024 | Hold-out 2025 |
|---|---|---|
| smart persistence | 0.167 | 0.206 |
| pvlib chain (static calibration) | 0.129 (+0.22) | 0.166 (+0.19 [0.14; 0.25]) |
| LightGBM on AC power, no physics | 0.134 | — |
| LightGBM on clear-sky index | 0.126 | 0.154 (+0.25) |
| **hybrid: LightGBM on the residual of the physics chain** | **0.125 (+0.25)** | **0.148 (+0.28 [0.23; 0.33])** |
| LSTM / TFT | 0.132 / 0.138 | not run (not adopted) |

How to read this:

- **Physics alone is already strong**: calibrating the physical chain to the plant improves it by +0.075 [0.060; 0.092] over the uncalibrated chain, the largest single gain in the project.
- **The hybrid adds a smaller but real step.** Against the physics: +0.033 [+0.006; +0.062] on the development period (borderline, 8 comparisons were made) and +0.108 [+0.087; +0.130] on the hold-out, so it was not a fluke.
- **Where the error comes from.** With exact (oracle) weather the hybrid's nRMSE falls to 0.110 and its advantage over oracle physics disappears (+0.020, interval includes 0). A diagnostic run on *measured* POA reaches 0.055. In squared-error terms (physics chain, 0.128 → 0.108 with oracle weather → 0.055 with measured POA), roughly 30% of the error is the GFS forecast error, about half is the gap between the reanalysis irradiance and what the plant sensor sees, and the remaining ~20% is plant-model error. So about 80% of the error sits in the irradiance input. ML mostly corrects the **systematic error of the weather forecast**, not the plant model.
- **Training strategy:** windows of 2–6 months versus expanding, and monthly versus one-off training, are statistically indistinguishable on the development period, so the simple expanding window with monthly retraining stays. One-off training shows a larger bias (drift).
- **By sky:** on clear days the hybrid is clearly better than physics; on overcast days the gain is small or negative (trees pull towards "usually clear").

![skill](results/figures/holdout_skill.png)

### Intraday, 1–6 h

Per-horizon LightGBM on the residual of physics driven by the newest (lead-1) NWP run, plus recent measurements.

| h | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| nRMSE, development | 0.104 | 0.113 | 0.117 | 0.119 | 0.121 | 0.121 |
| skill vs day-ahead hybrid, hold-out [90%] | +0.277 [0.237; 0.315] | | | | | +0.095 [0.058; 0.129] |

Pooled over horizons: +0.073 [+0.036; +0.113] (development), +0.152 [+0.114; +0.187] (hold-out). Findings:

- The fresher weather run alone (physics on lead-1 NWP, no ML) is as good as the day-ahead hybrid, i.e. most of the intraday gain is **better NWP, not nowcasting**.
- Measurements of the last hours help noticeably only for h ≤ 3. Without NWP, models using only measurements are worse than day-ahead at every horizon.
- A per-horizon model beats one model with the horizon as a feature (+0.028 [+0.015; +0.041]), almost entirely at h ≤ 3.
- On the development period the gain over day-ahead at h = 6 was zero; on the hold-out it is +0.095. Treat the long horizons as "somewhat better", not as a precise number.

![horizon](results/figures/stage5_horizon.png)

### Quantile forecasts and the value of a bid

LightGBM quantile regression on the physics residual, against an empirical band of training residuals. Pinball skill +0.143 [+0.113; +0.174] (development), +0.167 [+0.147; +0.188] (hold-out).

The imbalance cost of a bid *b* against the delivered power *y* is `c_short·max(b−y,0) + c_surplus·max(y−b,0)`; the cost-optimal bid is the quantile `q* = c_surplus / (c_short + c_surplus)`. **Prices are scenario parameters, not market data.** Cost against bidding the point hybrid (development, saving with 90% interval):

| Scenario | q* | Saving of the LightGBM quantile at q* |
|---|---|---|
| symmetric 1 : 1 | 0.50 | −0.4% [−2.8; +1.7], no difference |
| shortfall twice as expensive | 0.33 | **+9.2% [6.9; 11.9]** |
| surplus twice as expensive | 0.67 | +4.5% [0.1; 8.4], borderline |

The point is that the *shape* of the cost matters: with symmetric prices, a better median does not reduce cost; with asymmetric prices, shifting the bid to the right quantile does. Bidding the median in an asymmetric scenario was worse than bidding the mean.

![cost](results/figures/stage6_cost.png)

### Deep learning

LSTM and TFT (darts, CPU, fixed hyper-parameters) lose to the hybrid on this data: skill against the hybrid −0.057 [−0.105; −0.018] and −0.103 [−0.153; −0.066]. The TFT's quantiles are also worse than LightGBM's (pinball skill −0.124 [−0.187; −0.069]). Training four times longer made the LSTM worse (0.147), so this is overfitting, not under-training.

## What did not work

Reported on purpose; these are results too.

- **Deep learning did not beat physics + trees.** ~5–8 thousand training hours are too few for sequence models; the physical model injects knowledge the networks have to learn from scratch.
- **ML without physics is worse than physics** (LightGBM on AC power: −0.034 against the physics, because trees cannot extrapolate the scale of the plant). Predicting the clear-sky index fixes that; predicting the residual of the physics is best but not significantly better than the clear-sky index (+0.007 [−0.020; +0.030]).
- **Quantile bands under-cover.** The nominal 80% band of the LightGBM quantiles covers only 63% of the hours (development) and 57% (hold-out); the empirical band covers 78% and 64%. The forecasts lean upward. A post-hoc conformal shift (last 30 days) raised coverage to 70% but made the pinball loss and the costs worse, so it was rejected.
- **Rolling recalibration of the physics** (30-day scale from forecasts) was not significantly better than a static one (+0.014 [−0.007; +0.032]).
- **Longer or shorter training windows** brought nothing measurable, and the day-of-year feature did not help (+0.004 [−0.012; +0.022]).
- **Past measurements as features in day-ahead** are not significantly useful (+0.012 [−0.009; +0.030]).
- **Nowcast correction** of the physics (decay of the last hour's error) helps only at h = 1–2 and not significantly.
- **ECMWF NWP** was planned as a comparison but its archive covers only part of the period.

## Limitations

1. **Station drift at the end of 2025.** Output per unit of measured POA fell about 8–13% year over year at the same sky type (performance ratio), and all models have a clearly positive bias (hybrid MBE +0.11 / +0.14 of capacity in Nov / Dec 2025). December 2025 was also much cloudier than December 2024, which is the weak spot of every model; in December smart persistence beat all models. Soiling, degradation, sensor drift or unmodelled losses cannot be told apart with the available data. The hold-out numbers include this period.
2. **Wide intervals.** About 200 independent days per period. A borderline lower bound (e.g. +0.006 for hybrid vs physics in development) is borderline.
3. **Fixed hyper-parameters** by design. Tuning, especially of the networks, might improve them somewhat.
4. **One station, 20 months.** Conclusions about deep learning and training-window length are specific to this data volume. There is only one full seasonal cycle plus most of a second.
5. **Hold-out no longer independent** (see the protocol).
6. **Imbalance prices are scenarios**, not market data; the value in money depends on the real tariff.
7. **Capacity is an estimate** (observed maximum), so absolute nRMSE values shift slightly with it; comparisons are unaffected.

## What could be expected with different input data

These are expectations from the error decomposition above, **not tested results**.

- **Better weather input is the main lever.** The error budget says about 80% of the squared day-ahead error sits in the irradiance input (forecast error plus its mismatch with the on-site sensor). A local high-resolution model, an NWP ensemble (also giving honest uncertainty), or a blend of GFS with ECMWF/ICON would help more than any change of learner. Archived forecasts with several years of history would also allow proper multi-model comparisons.
- **Satellite cloud motion / all-sky imagers for h = 1–3.** The measurements-only intraday models are weak because plant history cannot see approaching clouds; an imager or satellite nowcast is the usual fix at these horizons.
- **Several years of history.** Would allow multi-year seasonality, tuning of hyper-parameters on a separate period, and a fair test for sequence models. I would expect deep learning to become competitive (not necessarily better) in the multi-year, multi-plant regime.
- **Several plants.** A shared model across plants, with plant-level corrections, typically helps the data-poor plants and would let the station drift be separated from a regional effect.
- **Inverter-level or string-level data and a soiling/maintenance log.** Would show whether the 2025 drop is soiling, a failing string, or a sensor problem, and make the physical model's loss factors identifiable. With a known cause, the drift can be corrected in the physics rather than absorbed by the trees.
- **Real market prices.** Would turn the scenario analysis into an actual value estimate, including the revenue effect of the narrow bands.
- **Calibrated probabilistic inputs.** NWP ensembles or a calibration window that spans several seasons are the likely fix for the under-covering bands; a constant shift on the last month did not work.

## Streamlit viewer

```bash
pip install -e ".[app]"
streamlit run app/main.py
```

The app reads `results/*.csv` and `results/figures` only; nothing is recomputed.

Hosted version: https://pv-forecast-arena-ddg9bmsxem37hkcw29ye3w.streamlit.app/

## Project structure

```
app/            Streamlit results viewer
configs/        base config + one YAML per experiment (configs/experiments/)
results/        comparison tables (stage*.csv, holdout*.csv), experiment registry, figures
scripts/        download, run_experiment / run_intraday / run_quantiles, compare_* reports
src/pv_forecast_arena/
    data/       PVDAQ + Open-Meteo loaders, cleaning, flags
    features.py allow-list of features; the only path to past measurements and lead-1 forecasts
    models/     baselines, physics (pvlib), tabular/gbm/hybrid, intraday, quantile, dl
    evaluation/ splits, metrics, bootstrap skill, quantile and cost scoring, registry
tests/          unit tests and leakage tests (corrupt the future, the forecast must not move)
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"       # add ,dl for the deep-learning models
python scripts/download_data.py --config configs/base.yaml
python scripts/download_weather.py --config configs/base.yaml
python scripts/build_dataset.py --config configs/base.yaml
python scripts/run_experiment.py --config configs/experiments/hybrid_lgbm_residual.yaml
python scripts/run_intraday.py --config configs/experiments/intraday_lgbm_perh.yaml
python scripts/run_quantiles.py --config configs/experiments/quantile_lgbm_residual.yaml
pytest
```

Every run appends rows to `results/experiments.csv` (commit, config hash, dirty flag). `--holdout` is refused on a dirty git tree.

## Author

Marharyta Cherkashyna — solar PV design engineer moving into PV performance & ML.
