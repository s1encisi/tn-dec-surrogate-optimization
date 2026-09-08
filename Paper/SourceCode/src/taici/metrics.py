from __future__ import annotations

import numpy as np


def naive_scale(training_actual: np.ndarray) -> float:
    values = np.asarray(training_actual, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("At least two one-dimensional training observations are required.")
    scale = float(np.mean(np.abs(np.diff(values))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid naive scale: {scale}")
    return scale


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    training_scale: float | None = None,
) -> dict[str, float | int]:
    y = np.asarray(actual, dtype=float)
    yhat = np.asarray(predicted, dtype=float)
    if y.ndim != 1 or yhat.ndim != 1 or len(y) != len(yhat) or not len(y):
        raise ValueError("Actual and prediction must be non-empty, equally sized 1-D arrays.")
    if not np.isfinite(y).all() or not np.isfinite(yhat).all():
        raise ValueError("Metrics require finite actual and predicted values.")
    residual = y - yhat
    absolute = np.abs(residual)
    squared = residual**2
    denominator = float(np.sum((y - y.mean()) ** 2))
    nonzero = y != 0
    smape_denominator = np.abs(y) + np.abs(yhat)
    smape_valid = smape_denominator > 0
    result: dict[str, float | int] = {
        "n": int(len(y)),
        "R2": float(1.0 - np.sum(squared) / denominator) if denominator > 0 else np.nan,
        "RMSE": float(np.sqrt(np.mean(squared))),
        "MAE": float(np.mean(absolute)),
        "MAPE_pct": float(np.mean(absolute[nonzero] / np.abs(y[nonzero])) * 100.0)
        if nonzero.any()
        else np.nan,
        "n_MAPE": int(nonzero.sum()),
        "sMAPE_pct": float(
            np.mean(2.0 * absolute[smape_valid] / smape_denominator[smape_valid]) * 100.0
        )
        if smape_valid.any()
        else np.nan,
        "WAPE_pct": float(np.sum(absolute) / np.sum(np.abs(y)) * 100.0)
        if np.sum(np.abs(y)) > 0
        else np.nan,
        "PBIAS_pct": float(np.sum(yhat - y) / np.sum(y) * 100.0)
        if np.sum(y) != 0
        else np.nan,
        "Q90_abs_error": float(np.quantile(absolute, 0.90)),
    }
    if training_scale is not None:
        if not np.isfinite(training_scale) or training_scale <= 0:
            raise ValueError(f"Invalid training scale: {training_scale}")
        result["MASE"] = float(np.mean(absolute) / training_scale)
        result["Q90_scaled_abs_error"] = float(np.quantile(absolute / training_scale, 0.90))
    return result
