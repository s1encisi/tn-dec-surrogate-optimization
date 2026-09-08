from __future__ import annotations

import numpy as np

from taici.metrics import naive_scale, regression_metrics


def test_perfect_prediction_metrics() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    metrics = regression_metrics(actual, actual.copy(), training_scale=1.0)
    assert metrics["R2"] == 1.0
    assert metrics["RMSE"] == 0.0
    assert metrics["MASE"] == 0.0


def test_mape_excludes_zero_and_reports_count() -> None:
    metrics = regression_metrics(
        np.array([0.0, 2.0]), np.array([1.0, 1.0]), training_scale=1.0
    )
    assert metrics["n_MAPE"] == 1
    assert metrics["MAPE_pct"] == 50.0


def test_naive_scale_uses_training_first_difference() -> None:
    assert naive_scale(np.array([1.0, 3.0, 4.0])) == 1.5

