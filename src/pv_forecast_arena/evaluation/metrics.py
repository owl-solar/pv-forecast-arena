"""Forecast metrics. Errors normalised by installed AC capacity; evaluate daytime hours only."""

import numpy as np


def _arr(x):
    return np.asarray(x, dtype=float)


def nrmse(y_true, y_pred, capacity: float) -> float:
    e = _arr(y_pred) - _arr(y_true)
    return float(np.sqrt(np.nanmean(e**2)) / capacity)


def nmae(y_true, y_pred, capacity: float) -> float:
    e = _arr(y_pred) - _arr(y_true)
    return float(np.nanmean(np.abs(e)) / capacity)


def mbe(y_true, y_pred, capacity: float) -> float:
    e = _arr(y_pred) - _arr(y_true)
    return float(np.nanmean(e) / capacity)


def skill_score(rmse_model: float, rmse_reference: float) -> float:
    """1 - RMSE_model / RMSE_reference (reference = smart persistence). >0 means better."""
    return 1.0 - rmse_model / rmse_reference


def pinball_loss(y_true, y_q, q: float) -> float:
    d = _arr(y_true) - _arr(y_q)
    return float(np.nanmean(np.maximum(q * d, (q - 1) * d)))


def coverage(y_true, lower, upper) -> float:
    y = _arr(y_true)
    return float(np.nanmean((y >= _arr(lower)) & (y <= _arr(upper))))
