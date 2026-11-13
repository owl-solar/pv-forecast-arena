# PV Forecast Arena

Benchmark of solar PV generation forecasting approaches on one real plant: naive persistence, a physical chain (pvlib), gradient boosting, hybrid physics + ML, deep learning (LSTM, TFT), quantile forecasts and their value as an imbalance cost. Every approach runs on the same data, the same walk-forward splits and the same metrics, with a strict no-look-ahead rule.

- **Plant:** NREL PVDAQ system 2107 (Arbuckle, CA; 893 kW DC, fixed 25° / 180°; observed AC maximum 708 kW, used as the capacity for normalisation).
- **Weather:** Open-Meteo *Previous Runs* archive of GFS forecasts (`previous_day2`), i.e. what was published before the forecast was issued, not observed weather.
- **Horizons:** day-ahead (issued 10:00 local on D-1, hourly) and intraday 1–6 h.
- **Data span:** 2024-02 … 2025-12. Development period: tests in 2024 (walk-forward, monthly refit, 2-day embargo). 2025 is the hold-out.

## Evaluation protocol

- Walk-forward (rolling-origin) splits only; no shuffling; all timestamps in UTC (hour-ending).
- Daytime hours; nRMSE / nMAE / MBE normalised by AC capacity; two variants: `all` and `clean` (hours flagged bad dropped).
- **Skill score** against smart persistence (clear-sky persistence scaled by the last 24 h) with a **90% day-block bootstrap interval**.
- **Selection rule:** if the interval of a skill difference includes 0, the simpler model wins. Hyper-parameters are fixed (no tuning on the development folds).
- **Leakage tests:** features go through one allow-list; every model that uses past measurements is tested by corrupting everything after the issue time and checking that the forecast does not move (with a mutation check that the test can fail). 147 tests.
- **Honesty about the hold-out:** the 2025 hold-out was used once for the day-ahead finalists, and once more, frozen, for the intraday and quantile finalists. It is **not independent any more**: new experiments are judged on the development period.

## Results

### Day-ahead (nRMSE, share of AC capacity; skill vs smart persistence [90% interval])

| Model | Development 2024 | Hold-out 2025 |
|---|---|---|
| smart persistence | 0.167 | 0.206 |
| pvlib chain (static calibration) | 0.129 (+0.22) | 0.166 (+0.19 [0.14; 0.25]) |
| LightGBM on AC, no physics | 0.134 | — |
| LightGBM on clear-sky index | 0.126 | 0.154 (+0.25) |
| **hybrid: LightGBM on the residual of the physics chain** | **0.125 (+0.25)** | **0.148 (+0.28 [0.23; 0.33])** |
| LSTM / TFT (darts) | 0.132 / 0.138 | not run (not adopted) |

The hybrid beats the physics chain by +0.033 [+0.006; +0.062] on the development period and by +0.108 [+0.087; +0.130] on the hold-out. With exact (oracle) weather the gain over physics almost vanishes (+0.020, interval includes 0): ML mostly corrects the systematic error of the GFS forecast, not the plant model. Training-window length (2–6 months vs expanding) and monthly vs one-off training are statistically indistinguishable on the development period; the simpler expanding window with monthly refit stays.

![skill](results/figures/holdout_skill.png)

### Deep learning

LSTM and TFT (darts, CPU) read 48 h of clear-sky-index history plus the same future NWP features. On this amount of data (20 months, one plant, 5–8 thousand training hours) **they lose to the hybrid**: LSTM −0.057 [−0.105; −0.018], TFT −0.103 [−0.153; −0.066] in skill against the hybrid; training longer made it worse (overfitting, not undertraining). Not adopted.

### Intraday, 1–6 h

Per-horizon LightGBM on the residual of the physics chain driven by the newest (lead-1) forecast, plus the last known measurements (available one hour after the end of the hour).

| h | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| nRMSE, development | 0.104 | 0.113 | 0.117 | 0.119 | 0.121 | 0.121 |
| skill vs day-ahead hybrid, hold-out [90%] | +0.277 [0.237; 0.315] | | | | | +0.095 [0.058; 0.129] |

Pooled over horizons: +0.073 [+0.036; +0.113] on the development period and +0.152 [+0.114; +0.187] on the hold-out. The gain comes from the fresher NWP run and from recent measurements at h ≤ 3; on the development period the gain over day-ahead at h = 6 is not distinguishable from 0. Without NWP the intraday models are worse than day-ahead.

![horizon](results/figures/stage5_horizon.png)

### Quantiles and the value of a bid

LightGBM quantile regression on the physics residual (levels 5 … 95%), against an empirical band of the training residuals. Pinball skill +0.143 [+0.113; +0.174] on the development period and +0.167 [+0.147; +0.188] on the hold-out.

The **imbalance cost** of a bid is `c_short·max(b−y,0) + c_surplus·max(y−b,0)`; the optimal bid is the quantile `q* = c_surplus / (c_short + c_surplus)`. Prices are scenario parameters, not market data. Cost against bidding the point hybrid (development, saving [90%]):

| Scenario | q* | Saving of the LightGBM quantile at q* |
|---|---|---|
| symmetric 1 : 1 | 0.50 | −0.4% [−2.8; +1.7] (no difference) |
| shortfall twice as expensive | 0.33 | **+9.2% [6.9; 11.9]** |
| surplus twice as expensive | 0.67 | +4.5% [0.1; 8.4] (borderline) |

![cost](results/figures/stage6_cost.png)

## Limitations — please read

1. **Station drift at the end of 2025.** Output per unit of measured plane-of-array irradiance fell about 8–13% year over year (performance ratio, same sky type), and all models have a clearly positive bias (MBE) in November–December 2025 (hybrid +0.11 / +0.14 of capacity). December 2025 was also much cloudier than December 2024, which is the weak spot of every model. Soiling, degradation, sensor drift or unmodelled losses cannot be told apart from the data. In December 2025 smart persistence beat all models. The hold-out numbers above include this period.
2. **The quantile bands under-cover.** The nominal 80% band of the LightGBM quantiles covers 63% of the hours on the development period and 57% on the hold-out (the empirical band: 78% and 64%); the forecasts lean upward (the share of hours at or below the median is 0.71 and 0.55). Do not call the bands "80% intervals". Post-hoc recalibration by a constant shift did not help the pinball loss or the cost.
3. **Wide intervals.** About 200 independent days per period; day-block bootstrap intervals are wide. A borderline lower bound (e.g. +0.006 for hybrid vs physics) is borderline.
4. **Fixed hyper-parameters** by design; a tuned deep-learning model might do a little better, but is unlikely to beat physics + trees on this volume.
5. **One station, 20 months.** Conclusions (especially about deep learning and training-window length) are specific to this data volume.
6. **Prices are scenarios**, not market data.
7. **ECMWF ensemble** was left out: its archive covers only part of the period.

## Streamlit viewer

```bash
pip install -e ".[app]"
streamlit run app/main.py
```

The app reads `results/*.csv` and `results/figures` only; nothing is recomputed.

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
tests/          unit tests and leakage tests (corrupt the future, forecast must not move)
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
