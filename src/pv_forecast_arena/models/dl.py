"""Deep-learning forecasters (darts): an LSTM and a Temporal Fusion Transformer, day-ahead.

Both read the same inputs, all known at the issue time (10:00 local D-1):
- target history: the clear-sky index of the plant, AC / (capacity * clear-sky POA / 1000), hourly,
  up to the issue time (`features.select_past`; gaps are filled forward from the past only, and the
  index is zero at night);
- future covariates over the forecast horizon: geometry and the lead-2 NWP features of
  `tabular.weather_features` (clear-sky index of the forecast, direct fraction, cloud cover, air
  temperature, wind, daily clear-sky index), standardised with the training rows.
The forecast of a target hour is clear-sky index x clear-sky power, clipped to [0, inverter limit]
and zero at night, as for the trees.

Training uses the training window of the fold only (it ends `embargo` before the test window): the
last `val_days` of it are held out for early stopping, with the same inner embargo as `tabular`.
Hyper-parameters are fixed in the config. Flagged-bad hours keep the forward-filled value in the
history (the network cannot take gaps); that is the only place flags touch a model.

`predict_lstm` returns a point forecast; `predict_tft_quantiles` the quantiles of a TFT trained with a
quantile loss (`quantile.levels`), one column per level, for the run_quantiles runner.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pv_forecast_arena.evaluation.splits import Split, issue_time
from pv_forecast_arena.features import select_past
from pv_forecast_arena.models import quantile as quantile_models
from pv_forecast_arena.models import tabular

COVARIATES = [
    "sun_elevation",
    "aoi",
    "clearsky_poa",
    "kt",
    "direct_fraction",
    "cloud_cover",
    "temperature_2m",
    "wind_speed_10m",
    "day_kt",
]
MAX_CSI = 1.5
OUTPUT_HOURS = (
    40  # issue 10:00 local D-1 -> last hour of local day D is 38 h ahead (39 on a 25 h day)
)


@dataclass
class Inputs:
    target: pd.Series  # clear-sky index, every hour, causally filled
    covariates: pd.DataFrame  # standardised, every hour
    denominator: pd.Series  # clear-sky power in kW
    limit: float


def build_inputs(frame: pd.DataFrame, split: Split, cfg: dict) -> Inputs:
    cap = cfg["site"]["ac_capacity_kw"]
    x = tabular.weather_features(frame, cfg)
    train = split.train_index
    cov = x[COVARIATES].copy()
    mean, std = cov.loc[train].mean(), cov.loc[train].std().replace(0.0, 1.0)
    cov = ((cov - mean) / std).fillna(0.0)
    # The last issue times look OUTPUT_HOURS past the end of the frame: pad with zeros (those hours
    # are never target hours of any forecast that is kept).
    pad = pd.date_range(cov.index[-1], periods=OUTPUT_HOURS + 1, freq="1h")[1:]
    cov = pd.concat([cov, pd.DataFrame(0.0, index=pad, columns=cov.columns)])

    denominator = cap * x["clearsky_poa"] / 1000.0
    ac = select_past(frame, ["ac_power_kw"])["ac_power_kw"]
    day = denominator > tabular.MIN_CLEARSKY_FRACTION * cap
    csi = (ac / denominator).where(day).clip(0.0, MAX_CSI)
    # Night is a known zero; a missing daytime hour repeats the last known one (causal fill).
    csi = csi.where(day, 0.0).ffill().fillna(0.0)
    limit = float(ac.reindex(train).max())
    return Inputs(csi, cov, denominator, limit)


def _series(values, index: pd.DatetimeIndex):
    from darts import TimeSeries

    frame = pd.DataFrame(
        np.asarray(values, dtype=np.float32), index=index.tz_convert("UTC").tz_localize(None)
    )
    frame.index.freq = "1h"
    return TimeSeries.from_dataframe(frame, freq="1h")


def _seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)


def _quiet() -> None:
    warnings.filterwarnings("ignore")
    for name in ("pytorch_lightning", "lightning", "lightning.pytorch", "darts"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _fit(model, inputs: Inputs, split: Split, cfg: dict, covariates: bool) -> int:
    """Train on the fold's training window with early stopping on its last `val_days`."""
    m = cfg["model"]
    train = split.train_index
    val_start = train.max() - pd.Timedelta(days=int(m.get("val_days", tabular.VAL_DAYS)))
    fit_idx = train[train < val_start - tabular.INNER_EMBARGO]
    lookback = int(m["input_hours"])
    # Validation windows may start inside the fit window (the history is input only, never a target).
    val_idx = train[train >= val_start - pd.Timedelta(hours=lookback)]
    kwargs = {}
    if covariates:
        kwargs = {
            "future_covariates": _series(inputs.covariates.loc[fit_idx], fit_idx),
            "val_future_covariates": _series(inputs.covariates.loc[val_idx], val_idx),
        }
    model.fit(
        _series(inputs.target.loc[fit_idx], fit_idx),
        val_series=_series(inputs.target.loc[val_idx], val_idx),
        verbose=False,
        **kwargs,
    )
    return int(getattr(model.trainer, "current_epoch", 0))


def _trainer_kwargs(cfg: dict) -> dict:
    from pytorch_lightning.callbacks import EarlyStopping

    return {
        "accelerator": "cpu",
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
        "enable_checkpointing": False,
        "callbacks": [EarlyStopping("val_loss", patience=int(cfg["model"].get("patience", 3)))],
    }


def _issue_series(inputs: Inputs, test: pd.DatetimeIndex, cfg: dict):
    """Issue times of the test hours and, for each, the history up to (and including) the stamp."""
    tz = cfg["site"]["local_timezone"]
    hour = int(cfg["weather"]["forecast_issue_hour_local"])
    issues = issue_time(test, tz, hour)
    return pd.Series(issues, index=test), sorted(set(issues))


def _forecast_issues(model, inputs: Inputs, issues: list, cfg: dict, quantiles: bool, covariates):
    """Predict every issue time in one batch; returns {issue: DataFrame indexed by UTC stamp}."""
    lookback = int(cfg["model"]["input_hours"])
    series = [
        _series(
            inputs.target.loc[t - pd.Timedelta(hours=lookback - 1) : t],
            inputs.target.loc[t - pd.Timedelta(hours=lookback - 1) : t].index,
        )
        for t in issues
    ]
    kwargs = {}
    if covariates:
        cov = _series(inputs.covariates, inputs.covariates.index)
        kwargs["future_covariates"] = [cov] * len(issues)
    if quantiles:
        kwargs["predict_likelihood_parameters"] = True
    out = model.predict(OUTPUT_HOURS, series=series, verbose=False, **kwargs)
    return {
        t: o.to_dataframe().set_axis(o.time_index.tz_localize("UTC")) for t, o in zip(issues, out)
    }


def _to_power(
    forecast: dict, issue_of: pd.Series, columns: list[str], inputs: Inputs, frame, test
) -> pd.DataFrame:
    rows = {}
    for t, f in forecast.items():
        hours = issue_of.index[issue_of == t]
        rows[t] = f.reindex(hours)[columns]
    csi = pd.concat(rows.values()).reindex(test)
    power = csi.mul(inputs.denominator.reindex(test), axis=0)
    power = power.clip(lower=0.0, upper=inputs.limit)
    return power.mul(frame.loc[test, "is_daytime"].to_numpy().astype(float), axis=0)


def _build(kind: str, cfg: dict, levels: list[float] | None):
    from darts.models import RNNModel, TFTModel
    from darts.utils.likelihood_models import QuantileRegression

    m = cfg["model"]
    common = {
        "input_chunk_length": int(m["input_hours"]),
        "n_epochs": int(m.get("epochs", 30)),
        "batch_size": int(m.get("batch_size", 128)),
        "optimizer_kwargs": {"lr": float(m.get("lr", 1e-3))},
        "random_state": int(cfg.get("seed", 42)),
        "pl_trainer_kwargs": _trainer_kwargs(cfg),
        "force_reset": True,
        "save_checkpoints": False,
    }
    if kind == "lstm":
        return RNNModel(
            model="LSTM",
            hidden_dim=int(m.get("hidden", 32)),
            n_rnn_layers=int(m.get("layers", 1)),
            dropout=float(m.get("dropout", 0.0)),
            training_length=int(m["input_hours"]) + OUTPUT_HOURS,
            **common,
        )
    return TFTModel(
        output_chunk_length=OUTPUT_HOURS,
        hidden_size=int(m.get("hidden", 16)),
        lstm_layers=1,
        num_attention_heads=int(m.get("heads", 2)),
        dropout=float(m.get("dropout", 0.1)),
        add_relative_index=False,
        likelihood=QuantileRegression(quantiles=levels),
        **common,
    )


def _run(frame: pd.DataFrame, split: Split, cfg: dict, kind: str, levels: list[float] | None):
    _quiet()
    _seed(int(cfg.get("seed", 42)))
    inputs = build_inputs(frame, split, cfg)
    model = _build(kind, cfg, levels)
    epochs = _fit(model, inputs, split, cfg, covariates=True)
    test = split.test_index
    issue_of, issues = _issue_series(inputs, test, cfg)
    forecast = _forecast_issues(
        model, inputs, issues, cfg, quantiles=kind == "tft", covariates=True
    )
    return inputs, forecast, issue_of, epochs


def predict_lstm(frame: pd.DataFrame, split: Split, cfg: dict) -> pd.Series:
    inputs, forecast, issue_of, epochs = _run(frame, split, cfg, "lstm", None)
    column = next(iter(forecast.values())).columns[0]
    power = _to_power(forecast, issue_of, [column], inputs, frame, split.test_index)
    out = power[column].rename("dl_lstm")
    out.attrs["info"] = {"test_start": str(split.test_start), "epochs": epochs}
    return out


def predict_tft_quantiles(frame: pd.DataFrame, split: Split, cfg: dict) -> pd.DataFrame:
    levels = quantile_models.levels(cfg)
    inputs, forecast, issue_of, epochs = _run(frame, split, cfg, "tft", levels)
    first = next(iter(forecast.values())).columns
    # darts names quantile components "<target>_q<level>", in the order of `levels`.
    columns = sorted(first, key=lambda c: float(c.rsplit("_q", 1)[1]))
    power = _to_power(forecast, issue_of, columns, inputs, frame, split.test_index)
    power.columns = [quantile_models.column(q) for q in sorted(levels)]
    out = pd.DataFrame(np.sort(power.to_numpy(), axis=1), index=power.index, columns=power.columns)
    out.attrs["info"] = {"test_start": str(split.test_start), "epochs": epochs}
    return out
