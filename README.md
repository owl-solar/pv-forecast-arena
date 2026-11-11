# PV Forecast Arena

Benchmark of solar PV generation forecasting approaches — from naive persistence and physical models (pvlib) to gradient boosting, hybrid physics + ML and deep learning — evaluated on the same data, splits and metrics.

**Horizons:** day-ahead (hourly) and intraday (1–6 h ahead).
**Data:** NREL PVDAQ (plant measurements) + Open-Meteo (archived weather *forecasts*, not observed weather — no look-ahead leakage).

## What is compared

| Axis | Variants |
|---|---|
| Model type | persistence, smart persistence, pvlib physical chain, LightGBM/XGBoost, hybrid (pvlib feature / residual learning / clear-sky index target), LSTM/TFT, quantile models |
| Training strategy | target definition, history length (3–24 months), single vs rolling retraining, single NWP model vs ensemble |

## Evaluation

- Walk-forward (rolling-origin) cross-validation; last year held out
- Daytime hours only; nRMSE, nMAE (normalised by capacity), MBE, skill score vs smart persistence
- Breakdown by season, sky condition and horizon
- Probabilistic: pinball loss, interval coverage
- Value: imbalance cost of forecast error

## Results

_Work in progress._

## Project structure

```
configs/        experiment configs (YAML)
data/           raw / interim / processed (not tracked)
notebooks/      exploration
results/        comparison table and figures
scripts/        entry points (download, run experiment)
src/pv_forecast_arena/
    data/       PVDAQ + Open-Meteo loaders, cleaning
    models/     baselines, physics, gbm, hybrid, quantile
    evaluation/ splits, metrics, experiment registry
tests/
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
python scripts/download_data.py --config configs/base.yaml
python scripts/run_experiment.py --config configs/experiments/baseline_persistence.yaml
```

## Author

Marharyta Cherkashyna — solar PV design engineer moving into PV performance & ML.
