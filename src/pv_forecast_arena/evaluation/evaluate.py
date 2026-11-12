"""Score a forecast against smart persistence on one common set of hours.

Every model is scored on the same rows: daytime hours with a measured target and a forecast from both
the model and the reference. Two variants are reported: `all` (every such hour) and `clean` (without
hours flagged bad: missing, stuck, outage). The skill score comes with a day-block bootstrap
interval, because neighbouring hours of one day share their errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pv_forecast_arena.evaluation import metrics

VARIANTS = ("all", "clean")


def evaluation_mask(
    frame: pd.DataFrame,
    forecast: pd.Series,
    reference: pd.Series,
    variant: str,
    others: tuple[pd.Series, ...] = (),
) -> pd.Series:
    """Boolean mask of the hours that enter the metrics (index-aligned to `forecast`).

    `others` are further forecasts that must exist too, so that scores against several references
    are computed on the same hours.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    rows = frame.loc[forecast.index]
    mask = rows["is_daytime"] & rows["ac_power_kw"].notna() & forecast.notna()
    for other in (reference, *others):
        mask &= other.reindex(forecast.index).notna()
    if variant == "clean":
        mask &= ~rows["flag_bad"]
    return mask


def skill_interval(
    err_model: np.ndarray,
    err_ref: np.ndarray,
    days: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    level: float = 0.90,
) -> tuple[float, float]:
    """Day-block bootstrap interval of the RMSE skill score (resample whole days with replacement)."""
    codes, uniques = pd.factorize(days)
    n_days = len(uniques)
    sq_model = np.bincount(codes, weights=err_model**2, minlength=n_days)
    sq_ref = np.bincount(codes, weights=err_ref**2, minlength=n_days)
    count = np.bincount(codes, minlength=n_days).astype(float)
    picks = np.random.default_rng(seed).integers(0, n_days, size=(n_boot, n_days))
    rmse_model = np.sqrt(sq_model[picks].sum(axis=1) / count[picks].sum(axis=1))
    rmse_ref = np.sqrt(sq_ref[picks].sum(axis=1) / count[picks].sum(axis=1))
    low, high = np.quantile(1.0 - rmse_model / rmse_ref, [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)


def score(
    frame: pd.DataFrame,
    forecast: pd.Series,
    reference: pd.Series,
    cfg: dict,
    variant: str = "all",
    others: tuple[pd.Series, ...] = (),
) -> dict:
    """Metrics of `forecast` on the common mask; skill is against `reference` (smart persistence).

    `others` only narrow the mask (see `evaluation_mask`).
    """
    mask = evaluation_mask(frame, forecast, reference, variant, others)
    if not mask.any():
        raise ValueError("no hours left to evaluate")
    return summarise(*masked_arrays(frame, forecast, reference, mask, cfg), cfg, variant)


def masked_arrays(
    frame: pd.DataFrame, forecast: pd.Series, reference: pd.Series, mask: pd.Series, cfg: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(truth, forecast, reference, local day) of the masked hours."""
    index = forecast.index[mask]
    truth = frame.loc[index, "ac_power_kw"].to_numpy()
    # Local days: in summer the evening of one local day falls after midnight UTC.
    local_days = index.tz_convert(cfg["site"]["local_timezone"]).tz_localize(None).normalize()
    return truth, forecast[index].to_numpy(), reference[index].to_numpy(), local_days.to_numpy()


def summarise(
    truth: np.ndarray,
    pred: np.ndarray,
    ref: np.ndarray,
    days: np.ndarray,
    cfg: dict,
    variant: str,
) -> dict:
    """Metrics and skill (with bootstrap interval) of already masked arrays."""
    capacity = cfg["site"]["ac_capacity_kw"]
    rmse_model = metrics.nrmse(truth, pred, capacity)
    rmse_ref = metrics.nrmse(truth, ref, capacity)
    low, high = skill_interval(pred - truth, ref - truth, days)
    return {
        "variant": variant,
        "n_hours": len(truth),
        "nrmse": rmse_model,
        "nmae": metrics.nmae(truth, pred, capacity),
        "mbe": metrics.mbe(truth, pred, capacity),
        "ref_nrmse": rmse_ref,
        "skill": metrics.skill_score(rmse_model, rmse_ref),
        "skill_lo": low,
        "skill_hi": high,
    }


def monthly(
    frame: pd.DataFrame, forecast: pd.Series, reference: pd.Series, cfg: dict
) -> pd.DataFrame:
    """nRMSE and MBE per calendar month (UTC) of the `all` variant, for seasonal diagnostics."""
    capacity = cfg["site"]["ac_capacity_kw"]
    mask = evaluation_mask(frame, forecast, reference, "all")
    index = forecast.index[mask]
    err = pd.DataFrame(
        {
            "model": forecast[index] - frame.loc[index, "ac_power_kw"],
            "ref": reference[index] - frame.loc[index, "ac_power_kw"],
        }
    )
    grouped = err.groupby(index.tz_convert(None).to_period("M"))
    return pd.DataFrame(
        {
            "n_hours": grouped.size(),
            "nrmse": grouped["model"].apply(lambda e: np.sqrt((e**2).mean()) / capacity),
            "ref_nrmse": grouped["ref"].apply(lambda e: np.sqrt((e**2).mean()) / capacity),
            "mbe": grouped["model"].mean() / capacity,
        }
    )
